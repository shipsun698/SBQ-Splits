"""Check unlocked entry points without executing models, data, jobs, or CUDA."""

import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import List


ROOT = Path(__file__).resolve().parents[1]
MESSAGE = "Release Gate disabled"
SENTINEL_MESSAGE = "MODEL_ENTRY_REACHED_WITHOUT_EXECUTION"
SENTINEL_EXIT = 73
ENTRYPOINTS = ("release_guard.py", "main.py", "run_with_submitit.py", "tools/benchmark.py")
FUNCTIONS = {
    "main.py": ("build_model_main", "main"),
    "run_with_submitit.py": (
        "main", "get_shared_folder", "get_init_file", "Trainer.__call__",
        "Trainer.checkpoint", "Trainer._setup_gpu_args",
    ),
    "engine.py": ("train_one_epoch", "evaluate", "test"),
    "models/__init__.py": ("build_model",),
    "models/dino/dino.py": ("build_dino", "DINO.forward"),
    "tools/benchmark.py": ("benchmark",),
}


def parsed(relative: str) -> ast.Module:
    """Read source without importing model modules."""
    return ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))


def find_function(tree: ast.Module, qualified: str) -> ast.FunctionDef:
    """Find a top-level function or one direct class method."""
    scope = tree.body
    for name in qualified.split("."):
        node = next(item for item in scope if getattr(item, "name", None) == name)
        scope = node.body
    return node


def load_guard() -> object:
    """Import only the stdlib-only release compatibility module."""
    spec = importlib.util.spec_from_file_location("preview_release_guard", ROOT / "release_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def isolated_command(entrypoint: str, arguments: List[str], module: bool = False) -> str:
    """Intercept the first model import before any dependency or model executes."""
    return "\n".join((
        "import argparse, builtins, importlib.util, io, os, pathlib, runpy, sys, typing",
        "root = pathlib.Path(" + repr(str(ROOT)) + ").resolve()",
        "entry = root / " + repr(entrypoint),
        "allowed_sources = {entry, root / 'release_guard.py', root / 'tools' / '__init__.py'}",
        "allowed_cache = {pathlib.Path(importlib.util.cache_from_source(str(p))) for p in allowed_sources}",
        "stdlib_root = pathlib.Path(os.__file__).resolve().parent",
        ("sys.path.insert(0, str(root))" if module
         else "sys.path.insert(0, str(entry.parent))"),
        "sys.argv = [str(entry)] + " + repr(arguments),
        "blocked = {'main', 'torch', 'torchvision', 'timm', 'submitit', 'numpy', 'fvcore'}",
        "original_import = builtins.__import__",
        "def checked_import(name, *args, **kwargs):",
        "    if name.split('.')[0] in blocked:",
        "        sys.stdout.write(" + repr(SENTINEL_MESSAGE + "\n") + ")",
        "        raise SystemExit(" + str(SENTINEL_EXIT) + ")",
        "    return original_import(name, *args, **kwargs)",
        "builtins.__import__ = checked_import",
        "def audit(event, args):",
        "    if event in {'subprocess.Popen', 'os.system', 'os.fork', 'os.posix_spawn'}:",
        "        raise AssertionError('FORBIDDEN SPAWN: ' + event)",
        "    if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename'}:",
        "        raise AssertionError('FORBIDDEN FILE MUTATION: ' + event)",
        "    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):",
        "        path = pathlib.Path(os.fsdecode(args[0])).resolve()",
        "        if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):",
        "            raise AssertionError('FORBIDDEN FILE WRITE: ' + str(path))",
        "        if path.suffix.lower() in {'.json', '.yaml', '.yml', '.pth', '.pt', '.jpg', '.png'}:",
        "            raise AssertionError('FORBIDDEN DATA READ: ' + str(path))",
        "        if root in path.parents and path not in allowed_sources | allowed_cache:",
        "            raise AssertionError('FORBIDDEN PREVIEW READ: ' + str(path))",
        "        if stdlib_root not in path.parents and path not in allowed_sources | allowed_cache:",
        "            raise AssertionError('FORBIDDEN EXTERNAL READ: ' + str(path))",
        "sys.addaudithook(audit)",
        ("runpy.run_module('tools.benchmark', run_name='__main__')" if module
         else "runpy.run_path(str(entry), run_name='__main__')"),
    ))


def isolated_result(entrypoint: str, arguments: List[str], directory: str,
                    module: bool = False) -> subprocess.CompletedProcess:
    """Run an import-intercepted CLI in an unrelated working directory."""
    return subprocess.run(
        [sys.executable, "-B", "-c", isolated_command(entrypoint, arguments, module)],
        cwd=directory, text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=15,
    )


class ReleaseBoundaryTests(unittest.TestCase):
    """Verify restrictions are removed without validating actual model execution."""

    def test_public_functions_have_no_release_guard(self) -> None:
        for relative, qualified_names in FUNCTIONS.items():
            tree = parsed(relative)
            for qualified in qualified_names:
                with self.subTest(file=relative, function=qualified):
                    node = find_function(tree, qualified)
                    calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)
                             and isinstance(item.func, ast.Name)
                             and item.func.id == "require_model_release"]
                    self.assertEqual(calls, [], "obsolete model release restriction remains")

    def test_legacy_guard_api_is_nonblocking(self) -> None:
        guard = load_guard()
        self.assertIsNone(guard.require_model_release())
        for operation in ("training", "evaluation", "inference", "submission"):
            with self.subTest(operation=operation):
                self.assertIsNone(guard.require_model_release(operation))

    def test_cli_execution_arguments_fall_through(self) -> None:
        guard = load_guard()
        for arguments in (
            [], ["--eval"], ["--train"], ["--inference"], ["--help", "--eval"],
            ["--release-status", "--train"], ["--config_file", "never-read.py"],
        ):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertIsNone(guard.cli(arguments))

    def test_cli_info_is_lightweight_from_external_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="preview-unlocked-info-") as directory:
            for relative in ENTRYPOINTS:
                for arguments in (["--help"], ["-h"], ["--release-status"]):
                    with self.subTest(entrypoint=relative, arguments=arguments):
                        result = isolated_result(relative, arguments, directory)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn(MESSAGE, result.stdout)
                        self.assertNotIn(SENTINEL_MESSAGE, result.stdout)

    def test_model_cli_reaches_original_imports_without_model_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="preview-unlocked-entry-") as directory:
            for relative in ENTRYPOINTS[1:]:
                for arguments in (
                    [], ["--eval"], ["--train"], ["--inference"],
                    ["--help", "--eval"], ["--release-status", "--train"],
                    ["--config_file", "never-read.py", "--resume", "never-load.pth"],
                ):
                    with self.subTest(entrypoint=relative, arguments=arguments):
                        result = isolated_result(relative, arguments, directory)
                        self.assertEqual(result.returncode, SENTINEL_EXIT, result.stdout + result.stderr)
                        self.assertIn(SENTINEL_MESSAGE, result.stdout)

    def test_benchmark_module_info_and_fallthrough(self) -> None:
        with tempfile.TemporaryDirectory(prefix="preview-unlocked-module-") as directory:
            requests = ((["--help"], 0), (["--release-status"], 0), ([], SENTINEL_EXIT))
            for arguments, expected in requests:
                with self.subTest(arguments=arguments):
                    result = isolated_result("tools/benchmark.py", arguments, directory, module=True)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    self.assertIn(MESSAGE if expected == 0 else SENTINEL_MESSAGE, result.stdout)

    def test_real_script_info_from_external_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="preview-unlocked-direct-") as directory:
            for relative in ENTRYPOINTS:
                with self.subTest(entrypoint=relative):
                    result = subprocess.run(
                        [sys.executable, "-B", str(ROOT / relative), "--release-status"],
                        cwd=directory, text=True, capture_output=True, check=False, timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(MESSAGE, result.stdout)

    def test_shell_launchers_have_no_release_exit(self) -> None:
        scripts = sorted((ROOT / "scripts").glob("*.sh"))
        self.assertEqual(len(scripts), 12)
        for script in scripts:
            with self.subTest(script=script.name):
                source = script.read_text(encoding="utf-8-sig")
                self.assertNotIn("SOURCE PREVIEW RELEASE GATE", source)
                self.assertNotIn("release_guard.py", source)
                self.assertNotIn('exit "$release_status"', source)
                self.assertIn("python ", source)

    def test_shell_launchers_have_valid_bash_syntax_when_available(self) -> None:
        bash = shutil.which("bash")
        if bash is None and os.name == "nt":
            git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
            if git_bash.is_file():
                bash = str(git_bash)
        if bash is None:
            self.skipTest("bash unavailable; launcher contents checked statically")
        for script in sorted((ROOT / "scripts").glob("*.sh")):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [bash, "-n"], input=script.read_text(encoding="utf-8-sig"),
                    text=True, capture_output=True, check=False, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
