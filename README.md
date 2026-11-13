# SBQ-Net: SAR Ship Detection

This repository accompanies our paper on **SBQ-Net**, a ship detection method for synthetic aperture radar (SAR) imagery. It hosts the documentation and file inventory for the paper's code.

## Research focus

In SAR imagery, small ships and weak returns can be difficult to separate from coastal clutter, especially when target boundaries are unclear and ship sizes vary within the same scene. Localization also becomes harder when the visible ship response covers only part of the target.

These cases motivate SBQ-Net.

| Component | Research focus |
| --- | --- |
| CA-SGM | Feature refinement in challenging SAR backgrounds. |
| SBA | Scale- and boundary-aware refinement, with attention to target extent and uncertain localization. |
| MSQ | Quality-aware query prediction, with attention to the reliability of detection candidates. |

The project studies how these components can help detect difficult ship targets. The descriptions above outline their intended roles. Implementation details and experimental results will be documented with the corresponding release.

## Repository contents

Two files are currently published.

| File | Contents |
| --- | --- |
| `README.md` | An overview of SBQ-Net and the current code release status. |
| [`UPLOAD_FILES.txt`](UPLOAD_FILES.txt) | The file inventory for the paper's code package. |

The inventory lists paths for the planned package, including model code, configurations, dataset utilities, documentation, and analysis tools. Listing a path in `UPLOAD_FILES.txt` does not mean that the file has already been uploaded.

## Code organization

The planned package places detector definitions and related operators in `models/`, with configuration files in `config/` and dataset loading, transformation, and evaluation code in `datasets/`. Shared utilities belong in `util/`.

Execution entry points include `main.py` and `engine.py`. Launch scripts are grouped under `scripts/`.

The tools listed in the inventory cover metric export from existing JSON-lines logs, training-curve plots, run summaries, and drawing detection boxes from saved predictions on existing images. Those drawing tools use saved results. They perform no inference.

Documentation paths under `docs/` cover the project structure, environment setup, code availability, training workflow, metrics, and common usage questions. The inventory also includes release-status information and a source manifest.

## Code availability

SBQ-Net model components, training recipes, checkpoints, and experimental results have not been released. Detector training, evaluation, and inference are unavailable in the current snapshot.

Further uploads and usage instructions will be recorded in this README. A model release date has not been set.
