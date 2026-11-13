"""Standalone utility tests using synthetic fixtures, never benchmark results."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def load_tool(name: str) -> Any:
    """Load an allocated utility without importing the upstream model package."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogToolsTests(unittest.TestCase):
    """All data in this class is fabricated solely as a test fixture."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_log(self, rows: List[Dict[str, Any]]) -> Path:
        path = self.root / "synthetic_test_log.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return path

    def test_missing_zero_raw_values_and_threshold_keys_stay_distinct(self) -> None:
        tool = load_tool("export_training_metrics")
        source = self.write_log([
            {"epoch": 0, "train_loss": 0, "test_coco_eval_bbox": [0, 0.2, 0.1],
             "test_Precision": 0.4, "test_Precision@0.3": 0.8},
            {"epoch": 1, "test_loss": 2, "test_Recall": 0.5, "test_F1": 0.6},
        ])
        output = self.root / "metrics.csv"
        tool.export_metrics(source, output)
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["train_loss"], "0")
        self.assertEqual(rows[1]["train_loss"], "")
        self.assertEqual(rows[0]["mAP"], "0")
        self.assertEqual(rows[1]["mAP"], "")
        self.assertEqual(rows[0]["test_Precision"], "0.4")
        self.assertEqual(rows[0]["test_Precision@0.3"], "0.8")
        self.assertEqual(json.loads(rows[0]["test_coco_eval_bbox"]), [0, 0.2, 0.1])
        self.assertNotIn("maxPrecision", rows[0])

    def test_malformed_line_is_reported_and_no_output_written(self) -> None:
        tool = load_tool("export_training_metrics")
        source = self.root / "broken.jsonl"
        source.write_text('{"epoch": 0}\nnot-json\n', encoding="utf-8")
        output = self.root / "metrics.csv"
        with self.assertRaisesRegex(ValueError, "line 2"):
            tool.export_metrics(source, output)
        self.assertFalse(output.exists())

    def test_nonfinite_and_wrong_metric_types_fail(self) -> None:
        tool = load_tool("export_training_metrics")
        for row in ({"epoch": 0, "train_loss": "bad"},
                    {"epoch": 0, "test_coco_eval_bbox": [1, "bad", 0]},
                    {"epoch": 0, "test_F1": float("nan")},
                    {"epoch": True, "test_loss": 1},
                    {"epoch": 0, "train_loss": True}):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    tool.read_records(self.write_log([row]))

    def test_summary_uses_actual_maximum_and_first_input_tie(self) -> None:
        tool = load_tool("summarize_training_run")
        source = self.write_log([
            {"epoch": 7, "test_coco_eval_bbox": [0.6, 0.8, 0.4]},
            {"epoch": 2},
            {"epoch": 3, "test_coco_eval_bbox": [0.7, 0.9, 0.5]},
            {"epoch": 4, "test_coco_eval_bbox": [0.7, 1, 0.5]},
        ])
        summary = tool.summarize_run(source)
        self.assertEqual(summary["best_epoch"], 3)
        self.assertEqual(summary["best_value"], 0.7)
        self.assertEqual(summary["present_count"], 3)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["source_key"], "test_coco_eval_bbox[0]")
        with self.assertRaisesRegex(ValueError, "no numeric values"):
            tool.summarize_run(source, "test_F1")

    def test_raw_named_metric_source_and_summary_output_safety(self) -> None:
        tool = load_tool("summarize_training_run")
        source = self.write_log([{"epoch": 0, "mAP": 0.4}])
        self.assertEqual(tool.summarize_run(source)["source_key"], "mAP")
        output = self.root / "summary.json"
        tool.write_summary(source, output)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["best_value"], 0.4)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            tool.write_summary(source, output)
        with self.assertRaises(ValueError):
            tool.write_summary(source, source, overwrite=True)
        with self.assertRaises(ValueError):
            tool.write_summary(source, output, "missing", overwrite=True)
        self.assertEqual(output.read_bytes(), before)

    def test_duplicate_json_keys_are_rejected_instead_of_losing_raw_values(self) -> None:
        tool = load_tool("export_training_metrics")
        source = self.root / "duplicate.jsonl"
        source.write_text('{"epoch":0,"train_loss":1,"train_loss":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 1.*duplicate"):
            tool.read_records(source)

    def test_blank_log_lines_are_not_silently_skipped(self) -> None:
        tool = load_tool("export_training_metrics")
        source = self.root / "blank.jsonl"
        source.write_text('{"epoch":0}\n\n{"epoch":1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            tool.read_records(source)

    def test_unknown_raw_metadata_is_preserved_without_scalar_coercion(self) -> None:
        tool = load_tool("export_training_metrics")
        source = self.write_log([{"epoch": 0, "train_metadata": {"schedule": [1, 2]},
                                 "test_metadata": "user-supplied", "test_coco_eval_masks": [0, 1, 2]}])
        output = self.root / "metadata.csv"
        tool.export_metrics(source, output)
        with output.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(json.loads(row["train_metadata"]), {"schedule": [1, 2]})
        self.assertEqual(row["test_metadata"], "user-supplied")
        self.assertEqual(json.loads(row["test_coco_eval_masks"]), [0, 1, 2])

    def test_output_collision_and_input_overwrite_refused(self) -> None:
        tool = load_tool("export_training_metrics")
        source = self.write_log([{"epoch": 0, "train_loss": 1}])
        original = source.read_bytes()
        output = self.root / "metrics.csv"
        output.write_text("sentinel", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            tool.export_metrics(source, output)
        self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")
        tool.export_metrics(source, output, overwrite=True)
        for overwrite in (False, True):
            with self.assertRaises(ValueError):
                tool.export_metrics(source, source, overwrite=overwrite)
        self.assertEqual(source.read_bytes(), original)
        with self.assertRaises(FileNotFoundError):
            tool.ensure_output_path(self.root / "absent" / "out.csv", [source])

    def test_sparse_curve_data_keeps_gaps_and_does_not_fill_zero(self) -> None:
        tool = load_tool("plot_training_curves")
        source = self.root / "sparse.csv"
        source.write_text("epoch,mAP,train_loss,test_loss\n0,0,2,\n1,,1,\n2,0.5,,\n",
                          encoding="utf-8")
        epochs, curves = tool.read_curves(source)
        self.assertEqual(epochs, [0, 1, 2])
        self.assertEqual(curves["mAP"], [0, None, 0.5])
        self.assertEqual(curves["train_loss"], [2, 1, None])
        self.assertNotIn("test_loss", curves)
        with self.assertRaises(ValueError):
            tool.read_curves(source, ["unavailable"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional matplotlib")
    def test_actual_headless_plot_preserves_input(self) -> None:
        tool = load_tool("plot_training_curves")
        source = self.root / "sparse.csv"
        source.write_text("epoch,mAP\n0,0\n1,\n2,0.5\n", encoding="utf-8")
        original = source.read_bytes()
        output = self.root / "curve.png"
        tool.plot_curves(source, output)
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(source.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            tool.plot_curves(source, output)
        with self.assertRaises(ValueError):
            tool.plot_curves(source, source, overwrite=True)

    def test_detection_coordinate_and_type_validation(self) -> None:
        tool = load_tool("render_detections")
        valid = [{"bbox": [0, 0, 6, 5], "label": "synthetic", "score": 0.4}]
        self.assertEqual(tool.validate_predictions(valid, 8, 6), valid)
        invalid = ([{"bbox": [-1, 0, 4, 3]}], [{"bbox": [1, 1, 1, 3]}],
                   [{"bbox": [0, 0, 8, 5]}], [{"bbox": [0, 0, 4, float("inf")]}],
                   [{"bbox": [0, 0, 4, 3], "score": "0.5"}],
                   [{"bbox": [0, 0, 4, 3], "label": 1}])
        for predictions in invalid:
            with self.subTest(predictions=predictions):
                with self.assertRaises(ValueError):
                    tool.validate_predictions(predictions, 8, 6)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "optional Pillow")
    def test_actual_render_preserves_input_and_dimensions(self) -> None:
        from PIL import Image
        tool = load_tool("render_detections")
        source = self.root / "synthetic_test_image.png"
        Image.new("RGB", (8, 6), "white").save(source)
        predictions = self.root / "synthetic_test_boxes.json"
        predictions.write_text('[{"bbox":[0,0,6,5],"score":0.001}]', encoding="utf-8")
        before = (source.read_bytes(), predictions.read_bytes())
        output = self.root / "boxes.png"
        tool.render_detections(source, predictions, output)
        with Image.open(output) as rendered:
            self.assertEqual(rendered.size, (8, 6))
            self.assertNotEqual(rendered.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual((source.read_bytes(), predictions.read_bytes()), before)
        with self.assertRaises(FileExistsError):
            tool.render_detections(source, predictions, output)
        with self.assertRaises(ValueError):
            tool.render_detections(source, predictions, source, overwrite=True)
        with self.assertRaises(ValueError):
            tool.render_detections(source, predictions, predictions, overwrite=True)

    def test_help_and_imports_do_not_load_heavy_dependencies(self) -> None:
        for name in ("export_training_metrics", "plot_training_curves",
                     "summarize_training_run", "render_detections"):
            with self.subTest(name=name):
                help_result = subprocess.run(
                    [sys.executable, "-B", str(TOOLS / (name + ".py")), "--help"],
                    capture_output=True, text=True, check=False)
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertIn("--output", help_result.stdout)
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "import export_training_metrics, plot_training_curves, "
                "summarize_training_run, render_detections; "
                "assert not any(k.split('.')[0] in {'torch','PIL','matplotlib','numpy'} "
                "for k in sys.modules)")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(TOOLS)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
