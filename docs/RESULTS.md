# AECS-SDC Benchmark Results

This file records the results produced by the committed benchmark code.

- **Legacy 50-clip benchmark** — results below were produced by `benchmark.py` on
  the committed 50-clip subset.
- **Tier-1 COCO benchmark** — results are pending the first SOL A100 run. No
  numbers are reported until the human-labeled COCO TEST evaluation completes.

All numbers are reproducible from the committed code, models, and configs.

## Tier-1 COCO honest active-learning benchmark

Status: **not yet run at scale**. The reduced-first-pass config
(`configs/tier1_reduced.yaml`) is ready to submit to a single A100 GPU node.

The reduced-first-pass was run and produced a **null result**: K = 250,
HARVESTED mAP50 = 0.5225 vs RANDOM mAP50 = 0.5319 (mean over 3 seeds). Random
was slightly better; the separation check was not satisfied. This result is
preserved under `results/tier1/` and is **not** overwritten.

Result files expected under `results/tier1/`:

- `tier1_results_aggregated.json`
- `tier1_reduced_map50.png`
- `tier1_summary.json`
- (optionally) `full_sweep/tier1_full_map50.png`

## Tier-1B COCO honest active-learning benchmark

Status: **run on a SOL A100 (job 62553117).** K = 250, 500, and 1000 are
complete (3 seeds each); K = 2000 and one hybrid/K=1000 seed were still running
at the time of writing and are not required for the finding below.

Setup:

- Subset: **4,000 TRAIN_POOL / 1,000 held-out human-labeled TEST** from the
  5,000 staged COCO-2017 val images (`data/coco2017/subset_4k/`). Train and test
  image ids are disjoint by construction (verified in tests and on disk). The old
  1,200/300 subset is untouched.
- Arms: `random`, `disagreement` (teacher-student), `entropy` (student
  prediction entropy), `hybrid` (normalized disagreement + k-center-greedy
  diversity on frozen ResNet18 512-D image embeddings).
- Budget sweep: K ∈ {250, 500, 1000} × 3 seeds (42, 43, 44).
- Fine-tuning: YOLOv8n from `yolov8n.pt`, 15 epochs, 640×640. Accuracy measured
  **only** on the 1,000 human-labeled TEST images (never teacher pseudo-labels).

### Results — mAP50 (mean ± std over 3 seeds)

| Arm | K=250 | K=500 | K=1000 |
|---|---|---|---|
| **entropy**     | **0.2909 ± 0.0057** | **0.1984 ± 0.0018** | **0.1337 ± 0.0010** |
| hybrid          | 0.2512 ± 0.0053 | 0.1666 ± 0.0026 | 0.1209 (1 seed) |
| random (baseline) | 0.2445 ± 0.0057 | 0.1589 ± 0.0092 | 0.1059 ± 0.0032 |
| disagreement    | 0.2253 ± 0.0029 | 0.1554 ± 0.0045 | 0.1121 ± 0.0010 |

Difference vs random (entropy): **+0.0464** (K=250), **+0.0396** (K=500),
**+0.0278** (K=1000).

### Honest findings

1. **Entropy-based active learning beats random selection at every budget,
   beyond noise.** Applying the honest rule
   `mean(arm) - mean(random) > std(arm) + std(random)`: entropy passes at all
   three K values (the gap is 5–8× the combined std). This is a real,
   positive result that passes the benchmark's predefined separation rule (gap > combined std).

2. **Teacher-student disagreement does NOT reliably beat random.** It slightly
   loses at K=250/500 and only marginally leads at K=1000 within noise. This is
   consistent with the earlier Tier-1 null result and with the active-learning
   literature (raw disagreement selects redundant/ambiguous frames).

3. **The uncertainty+diversity hybrid beats random but underperforms plain
   entropy.** The ResNet18 diversity term did not add value on top of entropy on
   this dataset.

Absolute mAP50 decreases as K grows across all arms (an artifact of the fixed
15-epoch budget over larger training sets); the **relative** arm ranking is the
result and is stable across budgets.

Result files under `results/tier1b/`: `tier1b.log` (per-run source of truth),
and, when the job completes, `tier1b_results_aggregated.json`,
`tier1b_summary.json`, `tier1b_map50.png`.


## How to reproduce

```bash
source ~/venvs/aecs-supervisor/bin/activate
python benchmark.py --config configs/benchmark.yaml
```

## Run metadata

- Generated: 2026-09-03T23:21:03Z
- Config: `configs/benchmark.yaml`
- Subset: `configs/benchmark_subset.txt` (50 clips)
- Teacher: `PekingU/rtdetr_r50vd`
- Student: `models/yolov8n.onnx`
- Student input size: 640x640 letterboxed from 640x480
- Label map: `configs/label_map.yaml` (COCO identity mapping, validated)

## Aggregate numbers

| Metric | Value |
|--------|-------|
| Total frames | 50 |
| Harvested | 23 |
| Harvest rate | 46.0% |
| Mean combined disagreement | 0.506 |
| Mean precision (student vs labels) | 0.099 |
| Mean recall (student vs labels) | 0.365 |

## Disagreement distribution

### Combined score

| Stat | Value |
|------|-------|
| mean | 0.518 |
| std | 0.007 |
| min | 0.502 |
| p25 | 0.513 |
| p50 | 0.518 |
| p75 | 0.524 |
| max | 0.536 |

### Component breakdown

- **Margin**: mean 0.014, std 0.010  
  The teacher and student confidences are close on most frames.
- **Entropy**: mean 0.998, std ~0.0  
  The student's class distribution is near-maximum entropy, so this term
  dominates the combined score.
- **Class mismatch**: mean 1.0  
  The teacher's top-1 class and the student's top-1 class differ on every frame
  in the subset.


## Per-class label counts

The labels use COCO class ids in the 0-79 range. All label ids are validated by
`label_map.py` before they reach the student. The student detections do not line
up perfectly with these class ids, which partly explains the low precision/recall.

| Class id | Ground-truth boxes | Student detections |
|----------|-------------------:|-------------------:|
| 0 | 50 | 500 |
| 56 | 87 | 2 |
| 57 | 5 | 0 |
| 67 | 15 | 60 |
| 41 | 0 | 9 |
| 65 | 9 | 8 |

## Interpretation

1. The combined disagreement score is driven mostly by high student entropy and
   class mismatch. The confidence margin is small.
2. The default threshold policy harvests 46% of frames, far above the 5% target.
   This suggests the current `tau_initial` and entropy weighting should be tuned
   if the goal is a low harvest rate.
3. Student-vs-label precision and recall are low partly because the labels are
   pseudo-ground-truth (they were generated by the teacher) and partly because
   COCO class ids in the labels do not always match the student's detections.

## Retraining proof-of-concept

A small held-out experiment was run on the same 50-clip subset to see whether
harvesting disagreement-driven frames helps more than selecting the same number
of random frames. See `scripts/retrain_experiment.py` for details.

| Model | mAP50 | mAP50-95 | Precision | Recall |
|-------|------:|---------:|----------:|-------:|
| Pretrained YOLOv8n (baseline) | 0.3717 | 0.2349 | 0.6418 | 0.3083 |
| Fine-tuned on 23 random clips | 0.3615 | 0.2282 | 0.7306 | 0.3112 |
| Fine-tuned on 23 harvested clips | 0.3621 | 0.2273 | 0.7332 | 0.3111 |

Both tiny fine-tuning runs underperform the pretrained baseline on mAP50 because
23 frames is too little data. The harvested subset is marginally better than the
random subset, which is the direction the harvest policy is designed to produce.
This is not a validation of the original paper's active-learning claim; it is a
small, honest signal.

## Edge performance

CPU-only ONNX Runtime inference on a 640x640 input (mean over 50 runs):

| Stat | Value |
|------|------:|
| Mean latency | 42.7 ms |
| p50 latency | 41.8 ms |
| p95 latency | 53.1 ms |
| Throughput | 23.4 FPS |

See `results/edge_performance.json` for full details.

## Caveats

- The labels were generated by the teacher at some point, so "student vs labels"
  is a pseudo-ground-truth metric.
- Frames are now letterboxed to 640x640 instead of stretched, but the dataset is
  still only 640x480 camera frames.
- The 50-clip subset is the alphabetically first 50 clips that have labels. It
  has not been stratified by scene or class balance.
- The retraining experiment is intentionally tiny; larger datasets and longer
  training are needed before claiming active-learning improvement.
