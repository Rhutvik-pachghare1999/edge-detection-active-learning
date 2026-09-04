# AECS-SDC Claim / Evidence Audit

This document maps historical claims from the repository to the evidence that
exists today. It is used to decide what the project can honestly say.

## Scope

- Evidence comes from `data/events.db`, `logs/supervisor.log`, the local
  `clips/` directory, `models/yolov8n.onnx`, and `PAPER_FINAL.txt`.
- Claims come from `PAPER_FINAL.txt`, file names, and variable names.
- Status values: `verified`, `partial`, `false`, `unverified`, `planned`.

## Audit table

| # | Claim | Source | Evidence | Status | Notes |
|---|-------|--------|----------|--------|-------|
| 1 | The system uses a teacher model (RT-DETR) and a student model (YOLOv8n). | `PAPER_FINAL.txt`, `supervisor/`, `models/yolov8n.onnx` | Teacher loads from Hugging Face; committed ONNX student exists. | verified | Both models are present and loadable. |
| 2 | The project has local video clips and labels. | `clips/` directory | 363 MP4 clips, 343 `.txt` YOLO-format labels. | verified | Counted on disk. |
| 3 | The benchmark can run without special hardware. | New `benchmark.py` | Runs on local clips with committed models. | verified | Smoke test and full 50-clip run pass. |
| 4 | The system collects 1,670 frames / 91 harvest events. | `PAPER_FINAL.txt` | `data/events.db` contains 603 events; only 32 are `harvest` rows. | false | The surviving database does not support the claim. |
| 5 | Drone-mounted ultrasonic / IMU sensor fusion is validated. | `PAPER_FINAL.txt`, `supervisor/` code | `logs/supervisor.log` shows hardware auth failures; DB `sonar_m` and `accel_mag` columns are all `0.0`. | false | No validated sensor fusion evidence exists. |
| 6 | The system runs on a Raspberry Pi 5 with real-time performance. | `PAPER_FINAL.txt`, `supervisor/main.py` | The live path is not exercised in the benchmark; no Raspberry Pi latency measurements exist. | unverified | CPU-only ONNX latency on an RTX 3050 Ti laptop is ~43 ms / 23 FPS (`results/edge_performance.json`). This is not Raspberry Pi evidence. |
| 7 | The student model is trained / improved by active learning. | `PAPER_FINAL.txt` | A tiny held-out experiment exists (`scripts/retrain_experiment.py`). | partial | 23 harvested frames marginally outperformed 23 random frames, but both underperformed the pretrained baseline. This is not a validation of the original active-learning claim. |
| 8 | The project uses ROS2 and Isaac Sim. | File names, `supervisor/main.py`, `.env` | ROS2 topic is referenced, but no launch files, sim recordings, or ROS bag files are present. | unverified | Marked as legacy / aspirational. |
| 9 | Label class IDs are aligned with the student's COCO taxonomy. | New `label_map.py`, `configs/label_map.yaml` | All label IDs in the 50-clip subset are inside the COCO 0-79 range and mapped with identity. | verified | The benchmark validates every label. |
| 10 | The student uses letterbox preprocessing. | `src/aecs_sdc/student.py` | Frames are letterboxed to 640x640 and boxes are mapped back to original coordinates. | verified | Verified by code inspection and benchmark run. |

## Consequences

1. The README and any public-facing text must not repeat the 1,670-frame / 91-harvest claim without independent evidence.
2. The drone sensor fusion claim must be removed or qualified as not validated.
3. The live hardware path must be clearly labeled as legacy and unverified in the
   current benchmark.
4. The active-learning improvement claim must be qualified: the current evidence
   is a tiny held-out experiment where the harvested subset is only marginally
   better than a random subset of the same size, and both underperform the
   pretrained baseline.
5. A new **Tier-1 COCO benchmark** is planned / in progress to test the
   active-learning signal on a real human-labeled dataset. Until it completes,
   no active-learning improvement claim can be made.
6. If the Tier-1 reduced pass does **not** show HARVESTED beating RANDOM beyond
   noise, the active-learning improvement claim must be retracted or reframed as
   "not observed at K ≤ 500 on COCO-2017 val."

## Tier-1 COCO benchmark audit plan

| # | Claim | Source | Evidence needed | Status | Notes |
|---|-------|--------|-----------------|--------|-------|
| 11 | The COCO-2017 subset contains 1,500 images with 1,200 TRAIN_POOL and 300 TEST. | `docs/PHASE0_DATASET_PROPOSAL.md` | Run `scripts/fetch_dataset.py` and inspect `subset/manifest.json`. | planned | Deterministic split; no human labels used during selection. |
| 12 | TRAIN_POOL and TEST are disjoint. | `scripts/fetch_dataset.py` | Verify `train_test_are_disjoint` and the manifest split field. | planned | Enforced by sorted-id slicing. |
| 13 | Final accuracy is computed only against human COCO labels. | `scripts/run_tier1_experiment.py` | TEST labels come from COCO annotations; evaluation uses Ultralytics `val` on TEST only. | planned | Teacher pseudo-labels are never used for mAP. |
| 14 | HARVESTED selection outperforms RANDOM at K = 250. | Hypothesis under test | Reduced-first-pass results: mean ± std of mAP50 over 3 seeds. | planned | Continue to full sweep only if `mean_h - mean_r > std_h + std_r`. |
| 15 | The full budget sweep compares HARVESTED, RANDOM, and ENTROPY at K = {100, 250, 500}. | `configs/tier1_full.yaml` | Full-sweep results under `results/tier1/full_sweep/`. | planned | Only executed if reduced pass shows separation. |
