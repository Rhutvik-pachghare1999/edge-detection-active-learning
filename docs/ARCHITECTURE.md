# AECS-SDC Architecture

This document describes the current offline benchmark architecture. The original
live hardware path is preserved under `supervisor/` and labeled as **legacy**.

## System boundary

The benchmark is a self-contained, hardware-free pipeline. It does not require:

- A Raspberry Pi
- A live camera
- ROS2
- Isaac Sim
- A drone / ultrasonic sensors
- A web dashboard

Those components belong to the legacy live path and are not exercised by
`benchmark.py`.

## Components

### 1. Dataset layer (`src/aecs_sdc/dataset.py`)

- Loads the committed 50-clip subset from `configs/benchmark_subset.txt`.
- Extracts a single frame near the 5-second trigger offset from each MP4.
- Reads YOLO-format labels from matching `.txt` files.
- Validates and optionally remaps label class IDs through `LabelMap`.

### 2. Student model (`src/aecs_sdc/student.py`)

- Wraps `models/yolov8n.onnx` via ONNX Runtime.
- Letterbox-resizes frames to 640x640, normalizes to [0, 1], and runs inference.
- Parses the `(1, 84, 8400)` output into normalized `cx, cy, bw, bh` boxes and
  COCO class scores.
- Maps output boxes back to original frame coordinates by undoing padding and
  the letterbox scale.
- Allows an explicit provider list (e.g. CPU-only benchmarking).
- Provides a `mock()` factory for unit tests that do not need the real model.

### 3. Teacher model (`src/aecs_sdc/teacher.py`)

- Wraps `PekingU/rtdetr_r50vd` from Hugging Face Transformers.
- Runs on CUDA if available, otherwise CPU.
- Returns normalized detections with class names.

### 4. Disagreement (`src/aecs_sdc/disagreement.py`)

Computes three per-frame signals:

- **Margin**: absolute confidence gap between teacher and student.
- **Entropy**: normalized entropy over the student's class distribution.
- **Class mismatch**: 1.0 if the top-1 classes differ, else 0.0.

The **combined** score is a weighted sum configured in `configs/benchmark.yaml`.

Also includes a label-matching helper that computes student-vs-teacher-label
precision/recall using IoU and class-id matching.

### 5. Harvest policy (`src/aecs_sdc/harvest.py`)

Three policies are implemented:

- **Threshold**: harvest when the combined disagreement exceeds a configurable
  `tau`. Optionally adapts `tau` with a simple rate controller.
- **Budget**: harvest the top-K frames by disagreement score.
- **Diversity**: harvest frames to cover under-represented teacher classes.

The active policy is selected in `configs/benchmark.yaml`.

### 6. Evaluator (`src/aecs_sdc/evaluator.py`)

- Collects per-frame records.
- Computes mean/std/percentile distributions for each disagreement signal.
- Computes mean precision/recall against the YOLO labels.
- Produces disagreement histograms and explainable harvest samples.
- Writes:
  - `results/benchmark_summary.json` (compact summary)
  - `results/benchmark_details.json` (summary + per-frame records)
  - `results/benchmark_summary.txt` (human-readable)
  - `results/disagreement_distribution.json` (histograms and top/bottom frames)

### 7. Label map (`src/aecs_sdc/label_map.py`)

- Validates that dataset label class IDs fall inside the COCO 0-79 range.
- Supports explicit remapping if a future dataset uses a custom taxonomy.
- Loaded from `configs/label_map.yaml` via the benchmark config.

### 8. Configuration (`src/aecs_sdc/config.py`)

- Loads `configs/benchmark.yaml`.
- Resolves relative paths against the project root.

### 9. Logging (`src/aecs_sdc/logging_config.py`)

- Uses Loguru.
- Writes to `logs/benchmark.log` and the console.
- Logs config, subset size, per-clip progress, and results paths.

### 10. Retraining proof-of-concept (`scripts/retrain_experiment.py`)

- Creates a temporary YOLO-format dataset from the benchmark clips.
- Splits clips deterministically into train/val.
- Runs the harvest policy on the training split.
- Fine-tunes a `yolov8n.pt` baseline, a random-subset model, and a harvested-subset
  model for a small number of epochs.
- Evaluates all three on the same held-out validation set.
- Writes `experiments/retrain/retrain_experiment.json`.

### 11. Edge benchmark (`scripts/benchmark_edge.py`)

- Loads the committed ONNX student with `CPUExecutionProvider`.
- Runs inference on a synthetic 640x480 frame with a configurable number of runs.
- Reports latency percentiles and throughput.
- Writes `results/edge_performance.json`.

## Data flow

```
configs/benchmark.yaml
         |
         v
benchmark.py
   |-- load subset (dataset.py)
   |-- load models (student.py, teacher.py)
   |-- for each clip:
   |       extract frame
   |       teacher.infer(frame)
   |       student.infer(frame)
   |       disagreement.combine(...)
   |       harvest_policy.decide(...)
   |       evaluator.add(record)
   |-- evaluator.finish()
            |-- results/benchmark_summary.json
            |-- results/benchmark_details.json
            |-- results/benchmark_summary.txt
            |-- results/disagreement_distribution.json

scripts/retrain_experiment.py  (optional)
   |-- experiments/retrain/yolo_dataset_*/
   |-- experiments/retrain/runs/*
   |-- experiments/retrain/retrain_experiment.json

scripts/benchmark_edge.py  (optional)
   |-- results/edge_performance.json
```


## Legacy hardware path (`supervisor/`)

The original system was designed to run live:

- `supervisor/main.py` — FastAPI server that receives frames from a Raspberry Pi.
- `supervisor/teacher.py` — RT-DETR wrapper used live.
- `supervisor/discrepancy.py` — original confidence-gap disagreement logic.
- `supervisor/database.py` — SQLite event logging.
- `supervisor/config.py` — `.env` loader.

This path is **not currently reproducible** because it depends on:

- Raspberry Pi hardware and network configuration
- Live camera / ROS2 topic
- Drone sensors whose logs show authentication failures and all-zero readings
- A web dashboard that is not part of the benchmark

It is kept for reference and for a future live demo.
