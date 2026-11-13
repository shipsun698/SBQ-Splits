# SBQ-Net: SAR Ship Detection

SBQ-Net is a research project focused on ship detection in synthetic aperture radar (SAR) imagery. It studies how scale, boundary uncertainty, prediction quality, and feature refinement can inform the detection of difficult ship targets.

> **Source preview.** This snapshot includes real upstream DINO source for inspection, project documentation, and standalone tools for existing logs and predictions. SBQ-Net components, training recipes, checkpoints, and experimental results are not released. All detector training, evaluation, and inference entry points are explicitly unavailable, including the DINO baseline paths. There is no fixed model release date.

## Motivation

Ships can be difficult to distinguish from their surroundings in SAR imagery. Small or weak targets may provide limited visual evidence, while nearshore clutter can resemble ship responses. Uncertain boundaries make localization harder, and differences in apparent target scale complicate a shared detection task.

SBQ-Net explores these challenges through three high-level research directions:

| Research focus | Intended role |
| --- | --- |
| CA-SGM | Feature refinement in challenging SAR backgrounds. |
| SBA | Scale- and boundary-aware refinement, with attention to target extent and uncertain localization. |
| MSQ | Quality-aware query prediction, with attention to the reliability of detection candidates. |

These descriptions summarize research intent. They do not disclose the unreleased implementation or establish a measured performance improvement.

## Conceptual overview

![Conceptual flow from an input SAR image through a detector to detection results.](assets/overview.svg)

The figure illustrates the task: **input SAR image -> detector -> detection results**. It is not measured model output and does not depict the internal architecture. The separate images in `figs/` belong to the historical upstream DINO documentation; they are not SBQ-Net measurements.

## Potential application contexts

The research is relevant to maritime observation tasks that require locating ships in SAR images, including:

- Reviewing ship candidates across broad maritime scenes.
- Examining ports and coastal waters with complex surroundings.
- Supporting analyst review of weak, small, or ambiguous targets.

These are motivating contexts for the research. This source preview does not provide a deployable detection service or demonstrate operational readiness.

## What is available

| Material | Current use |
| --- | --- |
| Upstream DINO model, utility, operator, and generic dataset source | Read and inspect the retained implementation and its provenance. Detector execution is gated. |
| Upstream reference configurations and launch scripts | Inspect historical DINO organization; these are not SBQ-Net training recipes or active execution instructions. |
| Log export, run summary, and curve plotting tools | Process existing JSON-lines training logs without constructing a detector. |
| Detection drawing tool | Draw boxes from an existing prediction file on an existing image; it performs no inference. |
| Research overview and metric documentation | Understand the project scope and reporting terminology. |

Start with the [tool guide](tools/README.md) for the standalone tools. Model availability can be checked without loading model dependencies:

```sh
python main.py --release-status
```

## Explore the documentation

| Document | What it covers |
| --- | --- |
| [Release status](RELEASE_STATUS.md) | Available material and unavailable model operations. |
| [Project structure](docs/PROJECT_STRUCTURE.md) | The source tree and its intended uses. |
| [Environment](docs/ENVIRONMENT.md) | Dependency boundaries for tools and historical upstream sources. |
| [Code availability](docs/CODE_AVAILABILITY.md) | Execution guards and the unreleased SBQ-Net scope. |
| [Tool guide](tools/README.md) | Commands for existing logs, curves, summaries, and predictions. |
| [Training overview](docs/TRAINING_OVERVIEW.md) | The retained source flow and requirements for a future model release. |
| [Metric guide](docs/METRICS.md) | Detection and efficiency terminology and tool output semantics. |
| [FAQ](docs/FAQ.md) | Common questions about use and reproduction. |
| [Upstream provenance](docs/UPSTREAM.md) | Source origin and local changes. |
| [Upload file list](UPLOAD_FILES.txt) | The exact package inventory. |
| [Source manifest](SOURCE_MANIFEST.json) | File provenance and hashes. |
| [Notice](NOTICE.md) | Attribution and licensing context. |
| [License](LICENSE) | The retained license terms. |

## Release outlook

A future model release would need supported dependencies, reviewed implementation and configurations, checkpoints, usage instructions, and reproducibility evidence. Its scope and timing have not been fixed. Updates belong in [Release status](RELEASE_STATUS.md).

The research builds on [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO). The retained [DINO README](DINO_README.md) is historical upstream reference material: its commands, model-zoo links, figures, and reported results describe DINO and do not establish local execution support or SBQ-Net performance.
