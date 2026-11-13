"""Export user-supplied DINO JSONL records without inventing missing metrics.

The only aliases are mAP = test_coco_eval_bbox[0], AP50 = [1], and AP75 = [2].
All original keys and the complete original array remain in the CSV. Missing
values are empty cells; threshold-specific precision keys remain separate.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


METRIC_ALIASES = {"mAP": 0, "AP50": 1, "AP75": 2}
BASE_COLUMNS = ("epoch", "train_loss", "test_loss", "mAP", "AP50", "AP75")
SCALAR_PREFIXES = ("train_loss", "test_loss", "test_Precision", "test_Recall", "test_F1")
SCALAR_KEYS = ("train_lr", "train_class_error", "test_class_error", "train_grad_norm")


def is_number(value: Any) -> bool:
    """Accept finite JSON numbers, excluding booleans."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value)))


def reject_constant(value: str) -> None:
    """Reject nonstandard JSON constants accepted by Python's default decoder."""
    raise ValueError("nonfinite JSON number: " + value)


def reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Reject ambiguous objects instead of discarding an earlier raw value."""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def validate_finite(value: Any, location: str) -> None:
    """Check nested raw values as well as selected metrics."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(location + " must be finite")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite(item, "{}[{}]".format(location, index))
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, location + "." + key)


def validate_record(record: Dict[str, Any]) -> None:
    """Validate the known DINO metric schema while preserving raw fields."""
    validate_finite(record, "record")
    epoch = record.get("epoch")
    if epoch is not None and (type(epoch) is not int or epoch < 0):
        raise ValueError("epoch must be a nonnegative integer or null")
    for key, value in record.items():
        if value is None:
            continue
        if key == "test_coco_eval_bbox":
            if (not isinstance(value, list) or len(value) < 3
                    or not all(is_number(item) for item in value)):
                raise ValueError("test_coco_eval_bbox must contain at least 3 finite numbers")
        elif key.startswith(SCALAR_PREFIXES) or key in SCALAR_KEYS or key in METRIC_ALIASES:
            if not is_number(value):
                raise ValueError(key + " must be a finite number or null")


def read_records(log_path: Path) -> List[Dict[str, Any]]:
    """Read every JSONL line strictly and attach source line numbers to errors."""
    records: List[Dict[str, Any]] = []
    with Path(log_path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line, parse_constant=reject_constant,
                                    object_pairs_hook=reject_duplicate_keys)
                if not isinstance(record, dict):
                    raise ValueError("each JSONL record must be an object")
                validate_record(record)
                records.append(expand_record(record))
            except (ValueError, TypeError) as error:
                raise ValueError("line {}: {}".format(line_number, error)) from error
    if not records:
        raise ValueError("log contains no JSONL records")
    return records


def expand_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Add documented COCO aliases without mutating the original record."""
    result = dict(record)
    values = record.get("test_coco_eval_bbox")
    if values is not None:
        for alias, index in METRIC_ALIASES.items():
            if alias in record and record[alias] != values[index]:
                raise ValueError("conflicting raw and mapped metric: " + alias)
            result[alias] = values[index]
    return result


def metric_source(metric: str, record: Dict[str, Any]) -> str:
    """Return an explicit definition for a selected column."""
    if metric in METRIC_ALIASES and record.get("test_coco_eval_bbox") is not None:
        return "test_coco_eval_bbox[{}]".format(METRIC_ALIASES[metric])
    return metric


def ensure_output_path(output_path: Path, inputs: Sequence[Path],
                       overwrite: bool = False) -> Path:
    """Resolve aliases, protect every input, and require an existing parent."""
    output = Path(output_path).resolve()
    for input_path in inputs:
        source = Path(input_path).resolve()
        if output == source or (output.exists() and source.exists()
                                and output.samefile(source)):
            raise ValueError("output must not overwrite an input file")
    if not output.parent.is_dir():
        raise FileNotFoundError("output parent directory does not exist: " + str(output.parent))
    if output.exists():
        if not output.is_file():
            raise ValueError("output must be a file")
        if not overwrite:
            raise FileExistsError("output exists; use --overwrite: " + str(output))
    return output


def csv_value(value: Any) -> Any:
    """Keep complex raw JSON values intact and missing cells empty."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return value


def export_metrics(log_path: Path, output_path: Path, overwrite: bool = False) -> None:
    """Export validated records, with stable base columns and raw field names."""
    output = ensure_output_path(output_path, [log_path], overwrite)
    records = read_records(log_path)
    extra = sorted(set().union(*(record.keys() for record in records)) - set(BASE_COLUMNS))
    columns = list(BASE_COLUMNS) + extra
    with output.open("w" if overwrite else "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: csv_value(record.get(key)) for key in columns}
                         for record in records)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the explicit-input CSV exporter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="DINO JSONL log")
    parser.add_argument("--output", required=True, type=Path, help="CSV output; parent must exist")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)
    try:
        export_metrics(args.log, args.output, args.overwrite)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
