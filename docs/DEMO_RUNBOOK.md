# AECS-SDC Demo Runbook

This runbook shows how to run the offline benchmark and inspect results.

## Prerequisites

- Python 3.10 or later
- A virtual environment with dependencies installed
- The `aecs-sdc` package installed in editable mode

## One-time setup

```bash
cd /home/rhutvik/aecs-sdc
python3 -m venv ~/venvs/aecs-supervisor
source ~/venvs/aecs-supervisor/bin/activate
pip install -e .
```

If the environment already exists, just activate it.

## Run the benchmark

```bash
source ~/venvs/aecs-supervisor/bin/activate
cd /home/rhutvik/aecs-sdc
python benchmark.py --config configs/benchmark.yaml
```

Expected output:

- Console logs showing each clip name.
- Final summary lines with total frames, harvested count, and mean disagreement.
- JSON results in `results/benchmark_summary.json` and
  `results/benchmark_details.json`.
- A human-readable summary in `results/benchmark_summary.txt`.

## Run a quick smoke test

```bash
python benchmark.py --subset-size 5
```

This runs on only 5 clips and is useful for verifying the environment.

## Run the tests

```bash
python -m pytest tests/ -q
```

All tests are unit tests and do not require real model weights or hardware.

## Inspect results

```bash
# Compact summary
cat results/benchmark_summary.json | python -m json.tool

# Per-frame details
cat results/benchmark_details.json | python -m json.tool

# Disagreement distribution and harvest samples
cat results/disagreement_distribution.json | python -m json.tool

# Human-readable summary
cat results/benchmark_summary.txt

# Retraining proof-of-concept
cat experiments/retrain/retrain_experiment.json | python -m json.tool

# CPU edge-latency benchmark
cat results/edge_performance.json | python -m json.tool
```

## Run the retraining proof-of-concept

```bash
python scripts/retrain_experiment.py --config configs/benchmark.yaml --epochs 3 --val-ratio 0.3
```

This creates a temporary YOLO dataset under `experiments/retrain/`, fine-tunes
`yolov8n.pt` on the harvested training subset and on a random subset of the same
size, and evaluates both on a held-out validation split. Results are written to
`experiments/retrain/retrain_experiment.json`.

The default settings are intentionally small (3 epochs, 50-clip subset) so the
script completes quickly. The result is a marginal improvement signal, not a
validation of active learning.

## Run the CPU edge-latency benchmark

```bash
python scripts/benchmark_edge.py --config configs/benchmark.yaml --runs 100
```

This loads the committed ONNX student with `CPUExecutionProvider`, runs 100
inferences on a synthetic 640x480 frame (plus warmup), and writes latency
percentiles and throughput to `results/edge_performance.json`.

## What not to do

- **Do not run `run_experiment.sh`**. It deletes `data/events.db`.
- **Do not reuse the old `.env` password**. `.env` now contains a placeholder, but
  if that password was ever shared or logged it must still be rotated at the
  provider.
- **Do not treat `PAPER_FINAL.txt` numbers as verified**. They do not match the
  surviving database.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'aecs_sdc'` | Package not installed | `pip install -e .` |
| Teacher model downloads repeatedly | No local cache | First run downloads weights; subsequent runs use cache |
| ONNX dimension error | Wrong input size in config | Keep `image_size: [640, 640]` |
| All tests fail | Missing pytest | `pip install pytest` |

## Legacy hardware path

The original live demo used:

```bash
supervisor/main.py
```

This path is not part of the current benchmark. It requires a Raspberry Pi,
camera, and ROS2 topic. See `docs/ARCHITECTURE.md` for details.
