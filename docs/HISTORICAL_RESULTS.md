# AECS-SDC Historical Results Reconciliation

This document tracks how the benchmark numbers have changed as the methodology
improved. It exists so the project can be honest about what changed and why.

## Original claim (`PAPER_FINAL.txt`)

The original paper draft claimed:

- 1,670 frames collected
- 91 harvest events
- 5.45% harvest rate
- Evidence of active-learning improvement

## Surviving evidence

The only surviving historical evidence is `data/events.db`. It contains:

- 603 events total
- 32 `harvest` rows
- All `sonar_m` and `accel_mag` columns are `0.0`
- `logs/supervisor.log` shows hardware authentication failures

This means the original 1,670-frame / 91-harvest numbers cannot be reproduced
from the data that remains in the repository.

## Current reproducible benchmark

The new offline benchmark uses the committed 50-clip subset, the committed ONNX
student, and a deterministic RT-DETR teacher. The results below can be reproduced
by running the commands in `docs/DEMO_RUNBOOK.md`.

### Latest run (2026-09-03T23:21:03Z)

| Metric | Value |
|--------|-------|
| Clips evaluated | 50 |
| Frames harvested | 23 |
| Harvest rate | 46.0% |
| Mean combined disagreement | 0.506 |
| Mean precision (student vs labels) | 0.099 |
| Mean recall (student vs labels) | 0.365 |

The harvest rate is much higher than the 5.45% target because the combined
disagreement score is dominated by near-maximum student entropy and persistent
top-1 class mismatch, not by confidence margin alone.

### Retraining proof-of-concept

A tiny held-out validation experiment was run on the same 50-clip subset:

| Model | mAP50 | mAP50-95 | Precision | Recall |
|-------|------:|---------:|----------:|-------:|
| Pretrained YOLOv8n (baseline) | 0.3717 | 0.2349 | 0.6418 | 0.3083 |
| Fine-tuned on 23 random clips | 0.3615 | 0.2282 | 0.7306 | 0.3112 |
| Fine-tuned on 23 harvested clips | 0.3621 | 0.2273 | 0.7332 | 0.3111 |

The retrained models slightly underperform the pretrained baseline on mAP50
because 23 frames is far too little data. The harvested subset marginally
outperforms the random subset of the same size, which is the direction the
harvest policy is designed to produce. This is not a validation of the original
paper's claim; it is a small, honest signal that disagreement-based harvesting is
at least no worse than random selection in this constrained setting.

### Edge latency

CPU-only ONNX Runtime inference on a 640x640 input:

| Stat | ms |
|------|---:|
| Mean | 42.7 |
| Std | 5.5 |
| Min | 33.9 |
| p50 | 41.8 |
| p95 | 53.1 |
| FPS | 23.4 |

## What changed and why

1. **Letterbox preprocessing**: the original code stretched frames to 640x640,
   distorting aspect ratio. The benchmark now letterboxes 640x480 frames and maps
   boxes back to original coordinates.
2. **Label mapping**: the original code did not explicitly map dataset class IDs
   to the student's COCO IDs. The benchmark now validates that label IDs fall in
   the COCO 0-79 range and applies an identity mapping by default.
3. **Disagreement components**: the original used a single confidence gap. The
   benchmark now combines margin, entropy, and class mismatch.
4. **Held-out evaluation**: the original paper reported aggregate harvest counts.
   The benchmark now splits clips into train/val, trains tiny models, and reports
   mAP on the held-out set.
5. **Honest reporting**: numbers that cannot be reproduced are explicitly flagged
   as `false` or `unverified` in `docs/CLAIM_EVIDENCE_AUDIT.md`.

## Outstanding gaps

- The 50-clip subset is not stratified by scene or class balance.
- Precision/recall against the local labels is a pseudo-ground-truth metric
  because those labels were themselves produced by the teacher.
- A full active-learning loop with incremental retraining and evaluation over
  many more clips has not been run.
