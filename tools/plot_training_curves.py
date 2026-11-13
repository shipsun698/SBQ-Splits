"""Plot raw exported metrics with gaps for missing cells; no interpolation.

Default columns are mAP, train_loss, and test_loss, when they contain values.
Each column has its own axis and original scale. Rendering needs matplotlib;
importing this module and requesting --help need only the standard library.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__:
    from .export_training_metrics import ensure_output_path
else:
    from export_training_metrics import ensure_output_path


def parse_cell(cell: Optional[str], column: str, line: int) -> Optional[float]:
    """Keep an absent numeric observation as a gap."""
    if cell is None or not cell.strip():
        return None
    try:
        value = float(cell)
    except ValueError as error:
        raise ValueError("line {}: {} is not numeric".format(line, column)) from error
    if not math.isfinite(value):
        raise ValueError("line {}: {} must be finite".format(line, column))
    return value


def read_curves(csv_path: Path, metrics: Optional[Sequence[str]] = None
                ) -> Tuple[List[Optional[float]], Dict[str, List[Optional[float]]]]:
    """Load requested CSV columns, retaining row order and every missing gap."""
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if "epoch" not in headers or len(set(headers)) != len(headers):
            raise ValueError("CSV requires unique column names including epoch")
        selected = list(metrics) if metrics is not None else [
            key for key in ("mAP", "train_loss", "test_loss") if key in headers]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("choose at least one distinct metric column")
        if any(key not in headers for key in selected):
            raise ValueError("requested metric column is absent")
        epochs: List[Optional[float]] = []
        curves: Dict[str, List[Optional[float]]] = {key: [] for key in selected}
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError("line {}: CSV row width does not match header".format(line))
            epoch = parse_cell(row["epoch"], "epoch", line)
            if epoch is not None and (epoch < 0 or not epoch.is_integer()):
                raise ValueError("line {}: epoch must be a nonnegative integer".format(line))
            epochs.append(epoch)
            for key in selected:
                curves[key].append(parse_cell(row[key], key, line))
    curves = {key: values for key, values in curves.items()
              if any(x is not None and y is not None for x, y in zip(epochs, values))}
    if not curves:
        raise ValueError("CSV contains no plottable numeric observations")
    return epochs, curves


def plot_curves(csv_path: Path, output_path: Path,
                metrics: Optional[Sequence[str]] = None, overwrite: bool = False) -> None:
    """Render independent raw metric axes with the headless Agg backend."""
    output = ensure_output_path(output_path, [csv_path], overwrite)
    if output.suffix.lower() != ".png":
        raise ValueError("plot output must have a .png extension")
    epochs, curves = read_curves(csv_path, metrics)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise ValueError("rendering requires optional dependency matplotlib") from error
    figure, axes = plt.subplots(len(curves), 1, figsize=(8, 3 * len(curves)), squeeze=False)
    try:
        for axis, (key, values) in zip(axes[:, 0], curves.items()):
            axis.plot(epochs, values, marker="o", linewidth=1)
            axis.set_xlabel("epoch")
            axis.set_ylabel(key)
            axis.grid(alpha=0.25)
        figure.tight_layout()
        with output.open("wb" if overwrite else "xb") as handle:
            figure.savefig(handle, format="png", dpi=160)
    finally:
        plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the explicit-input curve renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="CSV exported from a supplied log")
    parser.add_argument("--output", required=True, type=Path, help="PNG output; parent must exist")
    parser.add_argument("--metrics", help="comma-separated exact column names")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)
    metrics = [item.strip() for item in args.metrics.split(",")] if args.metrics is not None else None
    try:
        plot_curves(args.csv, args.output, metrics, args.overwrite)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
