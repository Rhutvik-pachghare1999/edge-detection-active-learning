# AECS-SDC

A dataset-driven benchmark for a teacher-student active-learning disagreement
system. The project is aimed at edge robotics / perception pipelines: a large
teacher detector (RT-DETR) labels frames, a small student detector (YOLOv8n)
runs on edge hardware, and a disagreement metric decides which frames are worth
keeping to retrain the student.

This repository now includes a **Tier-1 COCO-2017 active-learning benchmark**
that measures the student only against held-out **human COCO labels**, avoiding
the circular teacher-as-ground-truth trap of the original 50-clip experiment.

## What this project does

### Legacy 50-clip benchmark

- Provides a reproducible, hardware-free benchmark entry point (`benchmark.py`).
- Runs the committed student model (`models/yolov8n.onnx`) and the RT-DETR
  teacher on 50 local MP4 clips with YOLO-format labels.
- Computes three disagreement signals:
  1. Confidence margin between teacher and student.
  2. Normalized prediction entropy of the student.
  3. Top-1 class mismatch.
- Applies a configurable harvest policy (threshold, budget, or diversity).
- Runs a tiny held-out retraining experiment that compares the harvested subset
  to a random subset of the same size (`scripts/retrain_experiment.py`).
- Measures CPU-only ONNX edge latency and throughput (`scripts/benchmark_edge.py`).

### Tier-1 COCO honest active-learning benchmark

- Downloads a **fixed 1,500-image subset** of COCO-2017 validation
  (`scripts/fetch_dataset.py`).
- Splits it deterministically into **1,200 TRAIN_POOL** images and **300 TEST**
  images; the TEST set is held out and never used for selection.
- Uses RT-DETR pseudo-labels **only** to select training data and to fine-tune
  the student; final accuracy is computed **only** against human COCO TEST labels.
- Compares three acquisition arms at budgets K = {100, 250, 500}:
  1. **HARVESTED** — top-K teacher-student disagreement.
  2. **RANDOM** — seeded uniform sampling baseline.
  3. **ENTROPY** — top-K student prediction entropy.
- Evaluates each arm over 3 seeds and plots mAP50 vs budget with error bars
  (`scripts/run_tier1_experiment.py`).
- Includes unit tests for dataset integrity, split leakage, acquisition logic,
  and YOLO dataset preparation.


## What it does NOT do

- It does not run on live drone / Raspberry Pi hardware. The original hardware
path remains in `supervisor/` as a **legacy path** and is not exercised by the
benchmark.
- It does not validate the historical numbers in `PAPER_FINAL.txt`. The surviving
  database (`data/events.db`) contains 603 events, which does not match the paper's
  claims.
- It is not "production-ready," "real-time validated," or "deployed."
- The retraining experiment is a tiny proof-of-concept; it is not a full active
  learning loop over a large or stratified dataset.


## Quick start

### Install

A clean Python 3.10+ virtual environment is required:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

### Legacy 50-clip benchmark

```bash
python benchmark.py --config configs/benchmark.yaml
```

To run a smaller smoke test:

```bash
python benchmark.py --subset-size 5
```

### Tier-1 COCO benchmark

On a machine with internet, prepare the fixed subset:

```bash
python scripts/fetch_dataset.py --data-root data/coco2017
```

Then run the reduced-first-pass experiment locally (CPU is supported, but a GPU
is strongly recommended):

```bash
python scripts/run_tier1_experiment.py \
    --data-root data/coco2017 \
    --output-dir results/tier1 \
    --config configs/tier1_reduced.yaml \
    --base-model yolov8n.pt \
    --student-onnx models/yolov8n.onnx
```

To run the same experiment on ASU SOL A100, see the untracked SOL helper scripts
in `sol/` (copied by rsync but gitignored) and `sol/COMMANDS.md`.

## Repository layout

```
configs/
  benchmark.yaml           # legacy 50-clip benchmark configuration
  benchmark_subset.txt     # committed 50-clip subset
  tier1_reduced.yaml       # Tier-1 reduced first-pass config
  tier1_full.yaml          # Tier-1 full budget-sweep config
src/aecs_sdc/              # benchmark package
  benchmark.py             # legacy CLI entry point
  config.py                # config loader
  dataset.py               # clip/label loading
  student.py               # YOLOv8n ONNX wrapper
  teacher.py               # RT-DETR wrapper
  disagreement.py          # disagreement metrics
  harvest.py               # harvest policies
  evaluator.py             # result aggregation
  label_map.py             # label/class-id mapping
  logging_config.py        # loguru setup
  acquisition.py           # active-learning acquisition functions
  tier1.py                 # Tier-1 dataset / aggregation helpers
scripts/
  retrain_experiment.py    # tiny held-out retraining proof
  benchmark_edge.py        # CPU-only latency benchmark
  fetch_dataset.py         # download COCO-2017 fixed subset
  run_tier1_experiment.py  # end-to-end Tier-1 benchmark
sol/                       # UNTRACKED SOL/ASU helper scripts
  setup_sol.sh             # create venv and install pinned deps
  run_tier1.sbatch         # SLURM job submission
  COMMANDS.md              # exact rsync / setup / submit / fetch steps
supervisor/                # legacy live hardware path (see ARCHITECTURE.md)
data/
  events.db                # surviving historical evidence (read-only)
clips/                     # 363 MP4 clips + 343 YOLO labels
models/
  yolov8n.onnx             # committed student model
```

## Tier-1 COCO results

The Tier-1 experiment is designed to run on a single A100 GPU node. It has not
yet been executed at scale, so no final mAP numbers are reported here.

Once the SLURM job in `sol/run_tier1.sbatch` completes, results are written to
`results/tier1/`:

- `tier1_reduced_map50.png` — mAP50 vs training budget K.
- `tier1_results_aggregated.json` — mean ± std per arm and K.
- `tier1_summary.json` — run metadata and the full aggregated table.

See `sol/COMMANDS.md` for the exact commands to stage data, submit the job, and
pull results back from SOL.

## Latest legacy 50-clip benchmark results

Latest run (2026-09-03T23:21:03Z) with the committed 50-clip subset:

- Total frames: 50
- Harvested: 23 (46.0%)
- Mean combined disagreement: 0.506
- Mean student-vs-teacher-label precision: 0.099
- Mean student-vs-teacher-label recall: 0.365

Full numbers are written to `results/benchmark_summary.json`,
`results/benchmark_summary.txt`, and `results/disagreement_distribution.json`
after each run.

### Retraining proof-of-concept

Held-out validation on the same 50-clip subset (15 clips for validation, 23
clips selected for each training condition, 3 epochs):

| Model | mAP50 | mAP50-95 | Precision | Recall |
|-------|------:|---------:|----------:|-------:|
| Pretrained YOLOv8n | 0.3717 | 0.2349 | 0.6418 | 0.3083 |
| Fine-tuned on 23 random clips | 0.3615 | 0.2282 | 0.7306 | 0.3112 |
| Fine-tuned on 23 harvested clips | 0.3621 | 0.2273 | 0.7332 | 0.3111 |

The harvested subset is marginally better than the random subset of the same
size, but both underfit relative to the pretrained baseline because the dataset
is extremely small. See `docs/HISTORICAL_RESULTS.md` for the honest context.

### Edge performance

CPU-only ONNX Runtime inference on a 640x640 input (RTX 3050 Ti CPU fall-back,
mean over 50 runs):

| Stat | Value |
|------|------:|
| Mean latency | 42.7 ms |
| p50 latency | 41.8 ms |
| p95 latency | 53.1 ms |
| Throughput | 23.4 FPS |

Full numbers are in `results/edge_performance.json`.

## Known limitations

### Legacy 50-clip benchmark

- The local labels were generated by the teacher at one point in time, so the
student-vs-label metric is a pseudo-ground-truth metric, not independent
annotation.
- The 50-clip subset is the alphabetically first labeled clips, not a stratified
  sample.
- The harvest threshold policy currently harvests 46% of frames with the default
settings; this is higher than the 5% target because the disagreement score is
dominated by the entropy component, which is near 1.0 for most student
predictions.
- The legacy `run_experiment.sh` deletes `data/events.db` and must not be run.

### Tier-1 COCO benchmark

- The student is fine-tuned on RT-DETR pseudo-labels, so any improvement over the
  random baseline is conditional on the teacher being a useful oracle. Final
  accuracy is still measured against human COCO labels on the held-out TEST set.
- A GPU is required for a full run in reasonable time; CPU training is possible
  but slow.
- Results from the SOL A100 run are pending; this README will be updated once
  the reduced-first-pass and (if triggered) the full sweep complete.

## Security note

`aecs-sdc/.env` was replaced with the placeholder content from `.env.example`.
The original real password is no longer in the repository, but it may have been
exposed previously. **You must still rotate that password at the provider** if
there is any chance it was shared or logged.

## Future work

1. Run the Tier-1 reduced-first-pass on SOL A100 and report the resulting
   honest mAP50 numbers against the human-labeled COCO TEST set.
2. If the reduced pass shows HARVESTED beating RANDOM beyond noise, run the full
   K = {100, 250, 500} × 3-arm sweep and report mean ± std with plots.
3. If the reduced pass does not show a clear signal, document the null/negative
   result rather than running the more expensive full sweep.
4. Tune the disagreement weights and harvest threshold so the legacy harvest
   rate aligns with the 5% target instead of 46%.
5. Reconcile or retract the claims in `PAPER_FINAL.txt` based on the surviving
   evidence.
6. Add a lightweight CI workflow that runs `pytest` on every commit.

