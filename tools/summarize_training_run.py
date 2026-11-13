"""Select the maximum exact metric in a user-supplied DINO JSONL log.

The first input record wins a tie. No default values are supplied for missing
metrics. This is a log summary, not an evaluation or a checkpoint measurement.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

if __package__:
    from .export_training_metrics import ensure_output_path, is_number, metric_source, read_records
else:
    from export_training_metrics import ensure_output_path, is_number, metric_source, read_records


def summarize_run(log_path: Path, metric: str = "mAP") -> Dict[str, Any]:
    """Report observed counts and an actual maximum without rescaling values."""
    records = read_records(log_path)
    present = [record for record in records if record.get(metric) is not None]
    if not present:
        raise ValueError("no numeric values for metric: " + metric)
    if any(not is_number(record[metric]) for record in present):
        raise ValueError("selected metric is not a finite numeric scalar: " + metric)
    best = max(present, key=lambda record: record[metric])
    keys = sorted(set().union(*(record.keys() for record in records)))
    counts = {key: sum(is_number(record.get(key)) for record in records)
              for key in keys if any(is_number(record.get(key)) for record in records)}
    return {
        "metric": metric,
        "source_key": metric_source(metric, best),
        "selection": "maximum; first input record wins ties",
        "record_count": len(records),
        "present_count": len(present),
        "missing_count": len(records) - len(present),
        "present_counts": counts,
        "best_epoch": best.get("epoch"),
        "best_value": best[metric],
    }


def write_summary(log_path: Path, output_path: Path, metric: str = "mAP",
                  overwrite: bool = False) -> None:
    """Write only after input validation and collision checks succeed."""
    output = ensure_output_path(output_path, [log_path], overwrite)
    summary = summarize_run(log_path, metric)
    with output.open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the explicit-input maximum selector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="DINO JSONL log")
    parser.add_argument("--output", required=True, type=Path, help="JSON output; parent must exist")
    parser.add_argument("--metric", default="mAP", help="exact column name; always maximized")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)
    try:
        write_summary(args.log, args.output, args.metric, args.overwrite)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
