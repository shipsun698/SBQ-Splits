# SBQ-Net

**Code for the paper:** *SBQ-Net: A Coordinate-Scale-Quality Synergistic Transformer for SAR Ship Detection*

SBQ-Net is an end-to-end Transformer framework for ship detection in synthetic aperture radar (SAR) imagery. It refines object queries from coarse to fine, addressing coastal clutter and speckle-induced boundary ambiguity while calibrating detection confidence against localization quality.

---

## 🏗️ Architecture

SBQ-Net follows an encoder-decoder-head refinement pipeline. Each stage handles a different part of the detection task.

| Module | Full name | Role in SBQ-Net |
| --- | --- | --- |
| **CA-SGM** | Coordinate-Aware Scale-Guided Memory | Calibrates encoder memory using directional coordinate encoding and scale-guided modulation, driven by a layer-wise query-distribution prior. |
| **SBA-Decoder** | Scale- and Boundary-Aware Decoder | Couples dynamic reference boxes with Sobel boundary cues extracted from deep features to refine queries across target scales. |
| **MSQ-Head** | Multi-Scale Quality-Aware Detection Head | Combines localization quality, scale, and boundary evidence to calibrate classification logits and rerank detection candidates. |

CA-SGM prepares the encoder memory for decoder cross-attention. SBA-Decoder then uses box guidance and local boundary evidence to refine ship localization, while MSQ-Head processes the final regular queries to produce calibrated scores and refined boxes. Denoising queries are used during training and removed before MSQ-Head.

---

## 🚀 Quick Start

### 1. 📦 Dependencies

The package dependencies are listed in [`requirements.txt`](requirements.txt).

```sh
pip install -r requirements.txt
```

For plotting curves and drawing saved detections, install the optional image tools:

```sh
pip install matplotlib Pillow
```

Metric export and run summaries use the Python standard library. The paper's experiments used PyTorch 2.10 with a Swin-T backbone on an NVIDIA GeForce RTX 5060 Ti GPU with 16 GB of memory.

### 2. ⚙️ Experimental Configuration

The following settings come from the SBQ-Net paper.

| Setting | HRSID | SSDD |
| --- | --- | --- |
| Backbone | Swin-T | Swin-T |
| Input size | 800 × 800 | 512 × 512 |
| Object queries | 200 | 250 |
| Encoder / decoder layers | 6 / 6 | 6 / 6 |
| Feature levels | 3 | 3 |
| Batch size | 4 | 4 |
| Training epochs | 200 | 200 |

Both experiments use AdamW with an initial learning rate and weight decay of `1e-4`. The learning rate drops by a factor of 0.1 at epochs 150 and 180. The single-run seed is 42.

### 3. ▶️ Run the Analysis Tools

Run these commands from the repository root with your own logs, images, and saved predictions.

**Export training metrics:**

```sh
python tools/export_training_metrics.py --log run.log --output metrics.csv
```

**Summarize a training run:**

```sh
python tools/summarize_training_run.py --log run.log --output summary.json --metric mAP
```

**Plot recorded curves:**

```sh
python tools/plot_training_curves.py --csv metrics.csv --output curves.png --metrics mAP,AP50
```

**Draw saved detections:**

```sh
python tools/render_detections.py --image image.png --predictions predictions.json --output detections.png
```

These commands operate on existing files. The rendering tool draws supplied boxes without running detector inference.

### 4. 📄 Data Format

Training logs use JSON Lines, with one JSON object per line. Metric export preserves the recorded fields and reads COCO bounding-box metrics from `test_coco_eval_bbox` when present.

The detection-rendering tool accepts a JSON list of prediction records:

```json
[
  {
    "bbox": [24, 30, 90, 72],
    "label": "ship",
    "score": 0.91
  }
]
```

Each `bbox` contains `[x1, y1, x2, y2]` in image pixel coordinates, with ordered corners inside the image bounds. The `label` and `score` fields are optional.

---

## 📊 Evaluation

The paper evaluates SBQ-Net on **HRSID** and **SSDD** using source-scene-disjoint partitions. Images derived from the same source SAR scene remain in one partition, and the test sets are held out during model selection.

The following results are reported in the paper. Values are percentages.

| Dataset | mAP@[0.5:0.95] | AP50 | AP75 | AP-small |
| --- | ---: | ---: | ---: | ---: |
| HRSID | **66.4** | **90.5** | **79.1** | **68.1** |
| SSDD | **69.5** | **96.7** | **83.9** | **66.2** |

AP-small measures detection performance for objects with an area below 32 × 32 pixels under the COCO evaluation protocol. The paper also examines nearshore and offshore scenes, candidate score-IoU consistency, and feature responses in difficult SAR backgrounds.

---

## 📌 Notes

This repository contains a partial source release and standalone analysis tools for the SBQ-Net paper. Complete detector training, evaluation, and inference require additional model modules, dataset configurations, and weights that are outside the current upload.

See the [tool guide](tools/README.md) for command options, input validation, and output-file handling.
