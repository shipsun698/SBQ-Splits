"""Draw user-provided boxes on a supplied image; this tool performs no inference.

Predictions are a JSON list of objects with bbox=[x1,y1,x2,y2], optional string
label, and optional finite numeric score. Coordinates are image pixel positions:
0 <= x1 < x2 <= width-1 and 0 <= y1 < y2 <= height-1. No boxes are filtered,
clipped, rescaled, or recalibrated. Rendering requires the optional Pillow package.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__:
    from .export_training_metrics import (
        ensure_output_path, is_number, reject_constant, reject_duplicate_keys,
    )
else:
    from export_training_metrics import (
        ensure_output_path, is_number, reject_constant, reject_duplicate_keys,
    )


def validate_predictions(predictions: Any, width: int, height: int) -> List[Dict[str, Any]]:
    """Reject invalid coordinates rather than silently modifying supplied boxes."""
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a JSON list")
    for index, prediction in enumerate(predictions):
        prefix = "prediction {}: ".format(index)
        if not isinstance(prediction, dict):
            raise ValueError(prefix + "must be an object")
        bbox = prediction.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(is_number(value) for value in bbox)):
            raise ValueError(prefix + "bbox must contain exactly 4 finite numbers")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width - 1 and 0 <= y1 < y2 <= height - 1):
            raise ValueError(prefix + "bbox must be nondegenerate and inside pixel coordinates")
        if "label" in prediction and not isinstance(prediction["label"], str):
            raise ValueError(prefix + "label must be a string")
        if "score" in prediction and not is_number(prediction["score"]):
            raise ValueError(prefix + "score must be a finite number")
    return predictions


def render_detections(image_path: Path, predictions_path: Path, output_path: Path,
                      overwrite: bool = False) -> None:
    """Draw every validated supplied box on a separate image at original size."""
    output = ensure_output_path(output_path, [image_path, predictions_path], overwrite)
    formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP",
               ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP"}
    if output.suffix.lower() not in formats:
        raise ValueError("unsupported output image extension")
    with Path(predictions_path).open("r", encoding="utf-8-sig") as handle:
        predictions = json.load(handle, parse_constant=reject_constant,
                                object_pairs_hook=reject_duplicate_keys)
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ValueError("rendering requires optional dependency Pillow") from error
    with Image.open(image_path) as source:
        validate_predictions(predictions, source.width, source.height)
        image = source.convert("RGB")
    try:
        drawing = ImageDraw.Draw(image)
        for prediction in predictions:
            drawing.rectangle(prediction["bbox"], outline="red", width=2)
            text = prediction.get("label", "")
            if "score" in prediction:
                text = (text + " " + repr(prediction["score"])).strip()
            if text:
                drawing.text(tuple(prediction["bbox"][:2]), text, fill="red")
        with output.open("wb" if overwrite else "xb") as handle:
            image.save(handle, format=formats[output.suffix.lower()])
    finally:
        image.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the explicit-input box visualization utility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="existing image")
    parser.add_argument("--predictions", required=True, type=Path, help="existing JSON boxes")
    parser.add_argument("--output", required=True, type=Path, help="image output; parent must exist")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)
    try:
        render_detections(args.image, args.predictions, args.output, args.overwrite)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
