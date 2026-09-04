#!/usr/bin/env python3
"""CPU-only edge latency benchmark for the committed ONNX student.

Usage:
    source ~/venvs/aecs-supervisor/bin/activate
    python scripts/benchmark_edge.py --config configs/benchmark.yaml --runs 100

This script does NOT use the GPU and does NOT modify data/events.db. It loads
models/yolov8n.onnx with CPUExecutionProvider, runs inference on a synthetic
640x480 frame, and reports end-to-end latency and throughput.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aecs_sdc.config import Config
from aecs_sdc.logging_config import configure_logging
from aecs_sdc.student import StudentModel


def benchmark(model_path: str, input_size: tuple, runs: int = 100, warmup: int = 10):
    student = StudentModel(
        model_path=model_path,
        conf_threshold=0.25,
        input_size=input_size,
        providers=["CPUExecutionProvider"],
    )
    # Use a synthetic BGR frame that looks like a real camera frame.
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for _ in range(warmup):
        student.infer(frame)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        student.infer(frame)
        times.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(times)
    return {
        "device": "cpu",
        "provider": "CPUExecutionProvider",
        "model_path": model_path,
        "input_size": list(input_size),
        "runs": runs,
        "warmup": warmup,
        "mean_ms": round(float(arr.mean()), 3),
        "std_ms": round(float(arr.std()), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "fps": round(1000.0 / float(arr.mean()), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="CPU edge benchmark for ONNX student")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/edge_performance.json")
    args = parser.parse_args()

    cfg_path = args.config if Path(args.config).is_absolute() else str(PROJECT_ROOT / args.config)
    cfg = Config.from_yaml(cfg_path)
    cfg = cfg.resolve_paths(PROJECT_ROOT)
    configure_logging(cfg.logging.level, cfg.logging.file, cfg.logging.format)

    logger.info("Starting CPU-only edge benchmark")
    result = benchmark(
        cfg.models.student,
        tuple(cfg.models.image_size),
        runs=args.runs,
        warmup=args.warmup,
    )

    out_path = Path(args.output) if Path(args.output).is_absolute() else PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Edge benchmark result: {result}")
    logger.info(f"Wrote edge performance to {out_path}")


if __name__ == "__main__":
    main()
