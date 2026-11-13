# SBQ-Net: SAR Ship Detection

**Code for our paper on SBQ-Net.**

SBQ-Net studies ship detection in synthetic aperture radar (SAR) imagery, with attention to small targets, weak returns, and uncertain boundaries in coastal scenes where clutter can resemble ship responses.

[Project structure](#project-structure) · [Analysis tools](#analysis-tools) · [Dependencies](#dependencies) · [Release status](#release-status)

## Method overview

Small ships can occupy only a few pixels. When the visible return covers part of a hull or blends with nearby clutter, selecting a reliable detection candidate and locating its boundary become closely related problems.

| Component | Research focus |
| --- | --- |
| CA-SGM | Feature refinement in challenging SAR backgrounds. |
| SBA | Scale- and boundary-aware refinement, with attention to target extent and uncertain localization. |
| MSQ | Quality-aware query prediction, with attention to the reliability of detection candidates. |

These descriptions outline the roles studied in the paper. The release status below describes the implementation files available in this repository.

## Project structure

The current snapshot contains **54 files**, grouped by their role in the project. The [file inventory](UPLOAD_FILES.txt) lists every published path.

| Path | Files | Contents |
| --- | ---: | --- |
| [`models/`](https://github.com/shipsun698/SBQ-Splits/tree/main/models) | 14 | Position encoding, segmentation, transformer modules, model registration, and attention operator sources. |
| [`scripts/`](https://github.com/shipsun698/SBQ-Splits/tree/main/scripts) | 12 | Training and evaluation launch scripts, including distributed, multi-scale, ConvNeXt, and Swin variants. |
| [`tests/`](https://github.com/shipsun698/SBQ-Splits/tree/main/tests) | 3 | Tests for log tools and execution entry points. |
| [`tools/`](https://github.com/shipsun698/SBQ-Splits/tree/main/tools) | 6 | Metric export, curve plotting, run summaries, prediction rendering, and benchmarking source. |
| [`util/`](https://github.com/shipsun698/SBQ-Splits/tree/main/util) | 14 | Box operations and losses, configuration handling, logging, timing, and visualization utilities. |
| Root files | 5 | This README, the file inventory, dependencies, release compatibility code, and job submission source. |

Script names describe their task. Examples include [`train.sh`](scripts/train.sh), [`eval.sh`](scripts/eval.sh), and [`train_dist.sh`](scripts/train_dist.sh).

## Analysis tools

The standalone tools work with existing logs and saved predictions. Run them from the repository root, replacing the example paths with your own files.

```sh
python tools/export_training_metrics.py --log run.log --output metrics.csv
python tools/summarize_training_run.py --log run.log --output summary.json --metric mAP
python tools/plot_training_curves.py --csv metrics.csv --output curves.png --metrics mAP,AP50
python tools/render_detections.py --image image.png --predictions predictions.json --output detections.png
```

Metric export and run summaries use the Python standard library. Plotting requires `matplotlib`, and image rendering requires `Pillow`.

The [tool guide](tools/README.md) describes input formats, metric aliases, and output handling. Each tool accepts `--help`.

## Dependencies

[`requirements.txt`](requirements.txt) records dependencies for the code package. Additional model, configuration, and dataset files are required to run the detector.

## Release status

This is a partial source release. The files in [`UPLOAD_FILES.txt`](UPLOAD_FILES.txt) are available for inspection, and the standalone analysis tools can process user-supplied logs or predictions.

Complete detector training, evaluation, and inference require source modules and configurations that are outside the current upload. Model weights and experimental logs are also absent. Further code uploads and usage instructions will be recorded here.
