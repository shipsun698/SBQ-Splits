"""Lightweight release information without restricting model execution.

This helper does not launch a model or supply unreleased SBQ-Net components.
"""

import sys
from typing import Optional, Sequence


RELEASE_STATUS_MESSAGE = (
    "Release Gate disabled; upstream model entry points are no longer blocked. "
    "Unreleased SBQ-Net innovation modules are not included"
)


class ReleaseUnavailableError(RuntimeError):
    """Legacy exception retained for compatibility; no longer raised here."""


def require_model_release(operation: str = "model execution") -> None:
    """Allow legacy callers to continue without an availability restriction."""
    return None


def cli(argv: Optional[Sequence[str]] = None) -> Optional[int]:
    """Handle standalone information requests; otherwise allow normal execution."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["--help"], ["-h"], ["--release-status"]):
        sys.stdout.write(f"Source preview release status: {RELEASE_STATUS_MESSAGE}.\n")
        if arguments != ["--release-status"]:
            sys.stdout.write(
                "Usage: python -B release_guard.py [--help | --release-status]\n"
                "Use a model entry point for training, evaluation, or inference.\n"
                "Other model arguments continue to that entry point unchanged.\n"
            )
        return 0
    return None


if __name__ == "__main__":
    raise SystemExit(cli())
