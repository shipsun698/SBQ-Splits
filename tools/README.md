# Analysis and visualization tools

These tools accompany the SBQ-Net code. They read existing logs or saved predictions and write metric tables, run summaries, plots, or annotated images.

`benchmark.py` contains detector benchmarking code. Running it requires model files and dependencies outside the current upload.

Run commands from the package root with your own input paths and an existing output parent directory. No experimental logs, predictions, or sample detection images are supplied. Each tool offers `--help`, refuses an existing output unless `--overwrite` is given, and never allows its output to overwrite an input file.

| Tool | Input | Output | Dependencies |
| --- | --- | --- | --- |
| [export_training_metrics.py](export_training_metrics.py) | Existing JSON-lines log | CSV of recorded metrics | Python standard library |
| [summarize_training_run.py](summarize_training_run.py) | Existing JSON-lines log and a metric name | JSON summary | Python standard library |
| [plot_training_curves.py](plot_training_curves.py) | Existing metric CSV | Curve image | Optional `matplotlib` |
| [render_detections.py](render_detections.py) | Existing image and prediction JSON | Image with supplied boxes | Optional `Pillow` |

## Export recorded metrics

```sh
python tools/export_training_metrics.py --log run.log --output metrics.csv
```

Every log line must contain one JSON object. Blank or invalid JSON lines fail with an error. The tool preserves original fields, including the complete `test_coco_eval_bbox` array, and adds only the fixed aliases `mAP = test_coco_eval_bbox[0]`, `AP50 = [1]`, and `AP75 = [2]` when that array is present. Conflicting raw and aliased values are errors. Missing values remain empty CSV cells. Separate precision fields retain their names and values instead of being merged into a best precision value.

## Summarize a run

```sh
python tools/summarize_training_run.py --log run.log --output summary.json --metric mAP
```

Select the metric explicitly when a different recorded field is needed. The default is `mAP`, including the fixed COCO-array alias described above. The summary reports the maximum of that exact metric across records, its epoch, and observed/missing counts. The first input record wins ties. Every selected field is maximized, including a loss field, so this is not a general checkpoint-selection policy. It does not combine precision fields across thresholds or evaluate a model.

## Plot curves

```sh
python tools/plot_training_curves.py --csv metrics.csv --output curves.png --metrics mAP,AP50
```

Use names present in the CSV. Without `--metrics`, the tool selects available numeric observations from `mAP`, `train_loss`, and `test_loss`. Each metric uses a separate axis at its original scale. Missing measurements remain gaps. The tool does not replace them with zero or normalize them. The output must be a PNG file. `matplotlib` is imported only when plotting is requested, so help remains available without it.

## Draw existing predictions

```sh
python tools/render_detections.py --image image.png --predictions predictions.json --output detections.png
```

The prediction file must be a JSON list. Each entry contains `bbox` as `[x1, y1, x2, y2]` in image pixel coordinates, with `0 <= x1 < x2 <= width - 1` and `0 <= y1 < y2 <= height - 1`. `label` is an optional string and `score` is an optional finite number:

```json
[
  {"bbox": [24, 30, 90, 72], "label": "ship", "score": 0.91}
]
```

This synthetic format example is not a research prediction. The tool draws every supplied box and optional text at the original image size without filtering, clipping, rescaling, or changing scores. It performs no inference and calculates no detection metric. `Pillow` is imported only when drawing is requested.

## Output safety and tests

Add `--overwrite` only when replacing an existing output is intended. This option never permits overwriting any input.

The package's synthetic tests can be run from the package root:

```sh
python -B -m unittest tests.test_log_tools -v
```

This command checks the standalone tools with synthetic inputs. It does not run a detector or reproduce the paper's experiments.

See the [project overview](../README.md) for the published directory structure and current release status.
