# Phase 0 — Dataset Choice and Experiment Plan

This document records the dataset choice, subset size, splits, class handling, and
expected compute cost **before** any data is downloaded. Approval must be obtained
before proceeding to download or run the benchmark.

## Dataset choice

**COCO-2017 validation subset** (`instances_val2017.json` + `val2017` images).

Reasons:
- Public, human-annotated object-detection benchmark.
- Same 80-class COCO taxonomy as the committed YOLOv8n student, so **no class
  remapping is required**.
- Same class space as the existing local webcam clips (person, cell phone,
  keyboard, remote, etc.), so the experiment is a fair extension rather than a
  taxonomy mismatch.
- Standard COCO evaluation is well understood and reproducible.

## Subset size and splits

- **Total fixed subset**: 1,500 images.
- **TRAIN_POOL**: 1,200 images. These are treated as unlabeled to the student.
  The RT-DETR teacher pseudo-labels them and computes disagreement/uncertainty.
  The human COCO annotations are **not** used during training or harvesting.
- **TEST**: 300 images. Human COCO annotations are held out and used **only**
  for final mAP evaluation. No TEST image is ever pseudo-labeled, harvested, or
  shown to the student during training.
- **Selection method**: all `instances_val2017.json` images are sorted by COCO
  `id`; the first 1,500 are selected; the first 1,200 become TRAIN_POOL and the
  remaining 300 become TEST. This is deterministic and committed in the fetch
  script/manifest.

## Budget sweep and seeds

- **Training-set sizes**: K ∈ {100, 250, 500} images drawn from TRAIN_POOL.
- **Arms**:
  1. **HARVESTED**: top-K TRAIN_POOL images by the corrected disagreement score
     (1 - mean_IoU + margin).
  2. **RANDOM**: K images sampled uniformly from TRAIN_POOL (seeded).
  3. **ENTROPY** (optional Phase-2 acquisition): top-K TRAIN_POOL images by
     student prediction entropy. Evaluated at least at K = 250 to compare
     acquisition functions honestly; additional K values are run if time allows.
- **Seeds**: 3 seeds (42, 43, 44) for the RANDOM arm and for any stochastic
  training behavior. HARVESTED/ENTROPY selection is deterministic given the
  teacher/student outputs, so the same 3 training seeds are used for fine-tuning.
- **Total training runs (upper bound)**: 3 seeds × 3 K values × 2 arms
  (HARVESTED + RANDOM) = 18 fine-tuning runs. Adding ENTROPY across all K values
  would be 9 additional runs; we will run at least 3 for K=250 and report exactly
  what was executed.

## Class handling

COCO category IDs are mapped to YOLOv8n's contiguous class indices 0-79 using the
standard COCO 80-class order. The mapping is committed in `scripts/fetch_dataset.py`
and validated by tests. Because both COCO and YOLOv8n share the same 80 classes,
this is an identity mapping in label space, but it is explicit and tested.

## Compute estimate (NVIDIA RTX 3050 Ti Laptop GPU)

| Step | Estimate |
|------|----------|
| Download COCO val2017 + annotations | 5-15 min (~1.2 GB) |
| Extract archives | 1-2 min |
| Teacher (RT-DETR) pseudo-label TRAIN_POOL (1,200 images) | 20-40 min |
| Student eval on TEST (300 images, per model) | ~5 min |
| YOLOv8n fine-tuning, 500-image subset, 10 epochs | ~8-15 min |
| Full experiment (18 training runs + extras) | ~3-6 hours |

The full run is expected to exceed the 30-minute stop threshold, so this proposal
is submitted for approval before any download begins.

## Honesty constraints re-stated

- TEST-set human labels are the **only** accuracy oracle. The teacher labels on
  TRAIN_POOL are pseudo-labels and are used only for selection/training, never
  for accuracy claims.
- No result will be fabricated. If harvesting does not beat random beyond the
  combined standard deviation, the final report will state that plainly.
- `data/events.db`, `archive/pre_fix_invalid/`, `clips/`, and `PAPER_FINAL.txt`
  will not be modified or deleted.
