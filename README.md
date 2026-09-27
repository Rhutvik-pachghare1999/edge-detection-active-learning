# EdgeAL: Active Learning for Edge Object Detection

[![Tests](https://img.shields.io/badge/tests-77%20passing-brightgreen)](https://github.com/Rhutvik-pachghare1999/edge-detection-active-learning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

---

## What this is

A rigorous, leakage-free benchmark for **active learning in object detection** on COCO-2017. Given a large unlabeled image pool and a small annotation budget, which images should you label to improve a detector the most?

This repo compares multiple acquisition strategies against **held-out human COCO labels** — never against teacher pseudo-labels. The benchmark implements two tracks:

- **Track A (Human-Label AL)**: COCO labels hidden during acquisition, revealed only for selected images
- **Track B (Pseudo-Label Engine)**: Teacher (RT-DETR) pseudo-labels used for training; human labels only for final evaluation

Both tracks evaluate on the same held-out TEST set with human annotations.

---

## Key results (Track A, 2 seeds × 2 budgets)

| Method | mAP@50:95 (2%) | mAP@50:95 (4%) | mAP@50 (2%) | mAP@50 (4%) |
|--------|----------------|----------------|-------------|-------------|
| Random | **0.2821 ± 0.0021** | **0.3026 ± 0.0008** | **0.4137 ± 0.0023** | **0.4359 ± 0.0022** |
| Max Entropy | 0.2839 ± 0.0045 | — | 0.4127 ± 0.0057 | — |
| Mean Entropy | 0.2630 ± 0.0026 | 0.2984 ± 0.0021 | 0.3850 ± 0.0035 | 0.4263 ± 0.0031 |
| Least Confidence | 0.2593 ± 0.0006 | — | 0.3891 ± 0.0021 | — |
| Hybrid (Entropy + Div) | 0.2592 ± 0.0029 | — | 0.3790 ± 0.0025 | — |
| Margin | 0.2450 ± 0.0025 | — | 0.3672 ± 0.0030 | — |

**Takeaway**: The random baseline is competitive with or better than all tested acquisition methods. Max Entropy (top-1 instance) slightly edges out random at 2% on mAP@50:95, but the gap is within noise. Mean Entropy, Least Confidence, Margin, and Hybrid all underperform random.

This aligns with active learning literature: uncertainty sampling helps most when the model is poorly calibrated or the pool has high diversity. On COCO with a strong RT-DETR teacher, the signal-to-noise ratio may not favor these heuristics.

---

## Benchmark design

### Data splits (verified disjoint)
- **TRAIN_POOL**: 4,000 COCO val2017 images (first 4,000 by COCO ID)
- **TEST**: 1,000 held-out images (next 1,000 by COCO ID)
- Labels: Human COCO annotations converted to YOLO format

### Tracks

| Track | Selection signal | Training labels | Eval labels |
|-------|------------------|-----------------|-------------|
| A (Human-Label AL) | Student uncertainty only | **Human labels revealed post-selection** | Human TEST |
| B (Pseudo-Label Engine) | Teacher-student disagreement | Teacher pseudo-labels (conf ≥ 0.5) | Human TEST |

### Acquisition arms (Track A)
| Arm | Signal |
|-----|--------|
| Random | Uniform baseline |
| Mean Entropy | Image-level mean binary entropy over detections |
| Max Entropy | Image-level max binary entropy over detections |
| Least Confidence | Image-level mean (1 - max class prob) |
| Margin | Image-level mean (1 - (top1 - top2)) |
| Hybrid | Mean Entropy + k-center-greedy diversity (ResNet18 embeddings) |

### Protocol fixes (critical for fairness)
1. **K-scaled epochs**: `epochs = round(base_epochs × ref_K / K)` — keeps optimizer steps constant across budgets
2. **Diversity-aware selection**: k-center-greedy on ResNet18 embeddings prevents redundant frames
3. **Pseudo-label confidence floor**: Drop teacher boxes with conf < 0.5 (~45% of boxes)
4. **Leak-free evaluation**: Use `last.pt` (fixed schedule), never `best.pt` (selected on val=TEST)
5. **Human labels hidden during acquisition** (Track A only revealed post-selection)

---

## Reproduce

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run tests (77 tests: integrity, leakage, acquisition, dataset prep)
python -m pytest tests/ -q

# 3. Stage COCO-2017 subset (internet needed once, ~1 GB)
python scripts/fetch_dataset.py --data-root data/coco2017 \
    --subset-name subset_4k --subset-size 5000 --train-size 4000

# 4. Run Track A (Human-Label AL) — GPU recommended
python scripts/benchmarks/run_active_learning_benchmark.py \
    --data-root data/coco2017 --output-dir results/al_benchmark \
    --config configs/al_benchmark.yaml \
    --base-model yolov8n.pt --student-onnx models/yolov8n.onnx \
    --device cuda --track A --budgets 2 4 6 8 10 --seeds 42 43 44 45 46

# 5. Run Track B (Pseudo-Label Engine)
python scripts/benchmarks/run_active_learning_benchmark.py \
    --data-root data/coco2017 --output-dir results/al_benchmark \
    --config configs/al_benchmark.yaml \
    --base-model yolov8n.pt --student-onnx models/yolov8n.onnx \
    --device cuda --track B --budgets 2 4 6 8 10 --seeds 42 43 44 45 46

# 6. Generate figures
python scripts/visualize_results.py --results-dir results/al_benchmark/trackA \
    --output-dir docs/figures --metric mAP50_95
```

Outputs: `results/al_benchmark/trackA/al_benchmark_aggregated.json`, `al_benchmark_summary.json`, and figures in `docs/figures/`.

---

## Repository structure

```
edge-detection-active-learning/
├── src/edgeal/                 # Main package (pip install -e .)
│   ├── acquisition.py          # Acquisition functions (entropy, LC, margin, hybrid, etc.)
│   ├── coco_label_map.py       # COCO ↔ YOLO class mapping
│   ├── config.py               # Config dataclasses
│   ├── dataset.py              # Clip/label loading utilities
│   ├── disagreement.py         # Teacher-student disagreement metrics
│   ├── embeddings.py           # ResNet18 image embeddings for diversity
│   ├── evaluator.py            # Benchmark evaluation & reporting
│   ├── harvest.py              # Harvest policies (threshold/budget/diversity)
│   ├── logging_config.py       # Loguru setup
│   ├── retrain.py              # YOLO fine-tuning helpers
│   ├── student.py              # YOLOv8n ONNX wrapper (fixed entropy)
│   ├── teacher.py              # RT-DETR teacher wrapper
│   └── tier1.py                # Tier-1 experiment helpers
├── scripts/
│   ├── benchmarks/
│   │   └── run_active_learning_benchmark.py  # Main benchmark (Track A + B)
│   ├── fetch_dataset.py        # COCO subset staging
│   ├── retrain_experiment.py   # Legacy retrain script
│   ├── benchmark_edge.py       # ONNX CPU latency benchmark
│   └── visualize_results.py    # Publication figures generator
├── configs/
│   ├── al_benchmark.yaml       # Main benchmark config
│   └── tier1b_v2.yaml          # Legacy tier1 config
├── tests/                      # 77 unit/integration tests
├── docs/figures/               # Generated figures (PNG + PDF)
├── legacy/                     # Moved legacy code (supervisor, isaac_bridge, etc.)
├── data/                       # Staged COCO subsets (gitignored)
├── models/                     # Committed models (yolov8n.onnx, .pt)
├── results/                    # Benchmark outputs (gitignored)
├── pyproject.toml              # Package metadata (name: edgeal)
├── requirements.txt            # Pinned dependencies
└── LICENSE                     # MIT
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EDGEAL PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐   │
│  │ Unlabeled    │     │ Student      │     │ Acquisition      │   │
│  │ Pool         │────▶│ (YOLOv8n     │────▶│ Function         │   │
│  │ (TRAIN_POOL) │     │  ONNX)       │     │ (Uncertainty/    │   │
│  └──────────────┘     │ Uncertainty  │     │  Diversity)      │   │
│                       └──────────────┘     └────────┬─────────┘   │
│                                                      │             │
│                       ┌──────────────┐               ▼             │
│                       │ Teacher      │     ┌──────────────────┐   │
│                       │ (RT-DETR)    │     │ Select Top-K     │   │
│                       │ Pseudo-Labels│     │ Images           │   │
│                       └──────────────┘     └────────┬─────────┘   │
│                                                      │             │
│                     ┌──────────────┐               ▼             │
│                     │ Track A      │     ┌──────────────────┐   │
│                     │ Reveal Human │     │ Prepare YOLO     │   │
│                     │ Labels Only  │────▶│ Dataset          │   │
│                     │ For Selected │     │ (Human Labels)   │   │
│                     └──────────────┘     └────────┬─────────┘   │
│                                                  │             │
│                                                  ▼             │
│                                    ┌────────────────────────┐  │
│                                    │ Fine-tune YOLOv8n      │  │
│                                    │ (K-scaled epochs,      │  │
│                                    │  last.pt checkpoint)   │  │
│                                    └────────────┬───────────┘  │
│                                                 │              │
│                                                 ▼              │
│                                    ┌────────────────────────┐  │
│                                    │ Evaluate on Held-out   │  │
│                                    │ Human TEST (COCO)      │  │
│                                    │ mAP@50:95, mAP@50,     │  │
│                                    │ AP_small/medium/large  │  │
│                                    └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Generated figures

| Figure | Description |
|--------|-------------|
| `map_vs_budget_mAP50_95.png` | mAP@50:95 vs annotation budget with error bars |
| `map_vs_budget_mAP50.png` | mAP@50 vs annotation budget |
| `track_comparison_mAP50_95.png` | Track A vs Track B comparison (when both run) |
| `uncertainty_radar_mAP50_95_10pct.png` | Uncertainty signal comparison radar chart |
| `al_pipeline.png` | Pipeline architecture diagram |
| `results_table_mAP50_95.md/csv/tex` | Publication-ready results tables |

---

## Citation

```bibtex
@misc{edgeal2026,
  title = {EdgeAL: Active Learning for Edge Object Detection},
  author = {Pachghare, Rhutvik},
  year = {2026},
  note = {Benchmark suite for label-efficient object detection on COCO}
}
```

---

## License

MIT License — see `LICENSE` for details.

---

## Security note

Rotate any credential that was ever committed to `.env`; use `.env.example` as the template and keep real secrets out of the repo.