#!/usr/bin/env python3
"""Harvest-policy retraining experiment.

Usage:
    source ~/venvs/aecs-supervisor/bin/activate
    python scripts/retrain_experiment.py --config configs/benchmark.yaml

This script does NOT modify data/events.db or run_experiment.sh. It creates a
temporary YOLO-format dataset from the benchmark clips, runs the benchmark
harvester on a train split, then compares:

1. Baseline (pretrained yolov8n.pt) on the held-out validation set.
2. Retrained student on the harvested train frames.

Results are saved to results/retrain_experiment.json.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aecs_sdc.config import Config
from aecs_sdc.dataset import iter_samples, load_subset
from aecs_sdc.disagreement import compute_disagreement
from aecs_sdc.harvest import build_policy
from aecs_sdc.label_map import LabelMap
from aecs_sdc.logging_config import configure_logging
from aecs_sdc.retrain import (
    COCO_NAMES,
    create_yolo_split,
    evaluate_yolo_model,
    split_train_val,
    train_yolo_model,
    write_data_yaml,
)
from aecs_sdc.student import StudentModel
from aecs_sdc.teacher import TeacherModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(cfg: Config, experiment_dir: Path, epochs: int, val_ratio: float, seed: int):
    cfg = cfg.resolve_paths(PROJECT_ROOT)
    set_seed(seed)

    label_map = LabelMap.from_yaml(cfg.benchmark.label_map)
    logger.info(f"Label map: {label_map.summary()}")

    subset = load_subset(cfg.benchmark.subset_file)
    subset = subset[: cfg.benchmark.subset_size]
    logger.info(f"Loaded subset: {len(subset)} clips")

    train_names, val_names = split_train_val(subset, val_ratio, seed)
    logger.info(f"Train: {len(train_names)} clips, Val: {len(val_names)} clips")

    # Create temporary YOLO dataset.
    data_root = experiment_dir / "yolo_dataset"
    if data_root.exists():
        shutil.rmtree(data_root)
    create_yolo_split(
        train_names,
        cfg.benchmark.clips_dir,
        cfg.benchmark.labels_dir,
        data_root,
        "train",
        cfg.benchmark.trigger_offset_seconds,
    )
    create_yolo_split(
        val_names,
        cfg.benchmark.clips_dir,
        cfg.benchmark.labels_dir,
        data_root,
        "val",
        cfg.benchmark.trigger_offset_seconds,
    )
    data_yaml = write_data_yaml(data_root, nc=80, names=COCO_NAMES)
    logger.info(f"Wrote YOLO dataset to {data_root}")

    # Run the disagreement harvester on the train split to pick curated frames.
    teacher = TeacherModel(
        model_name=cfg.models.teacher,
        device=cfg.models.teacher_device,
        conf_threshold=cfg.models.teacher_conf_threshold,
    )
    student = StudentModel(
        model_path=cfg.models.student,
        conf_threshold=cfg.models.student_conf_threshold,
        input_size=tuple(cfg.models.image_size),
    )
    harvest_policy = build_policy(cfg.harvest)

    harvested_names = []
    train_records = []
    for sample in iter_samples(
        cfg.benchmark.clips_dir,
        cfg.benchmark.labels_dir,
        train_names,
        cfg.benchmark.trigger_offset_seconds,
        label_map,
    ):
        teacher_result = teacher.infer(sample.frame)
        student_result = student.infer(sample.frame)
        disagreement = compute_disagreement(
            teacher_result,
            student_result,
            cfg.disagreement.weights,
        )
        decision = harvest_policy.decide(disagreement.combined)
        record = {
            "clip_name": sample.clip_name,
            "frame_idx": sample.frame_idx,
            "combined_disagreement": disagreement.combined,
            "margin": disagreement.margin,
            "entropy": disagreement.entropy,
            "class_mismatch": disagreement.class_mismatch,
            "iou": disagreement.iou,
            "harvested": decision.harvested,
        }
        train_records.append(record)
        if decision.harvested:
            harvested_names.append(sample.clip_name)

    logger.info(f"Harvested {len(harvested_names)} / {len(train_names)} train frames")

    # Build a random subset of the same size as the harvested set.
    harvest_size = len(harvested_names)
    random_names = random.sample(train_names, harvest_size) if harvest_size <= len(train_names) else train_names[:]

    # Write split-specific data.yaml files pointing at the same images but with
    # selected labels. Because YOLO uses the labels directory, we create
    # filtered label directories that include only the selected clip names.
    def write_filtered_split(selected_names, split_name):
        split_root = experiment_dir / f"yolo_dataset_{split_name}"
        if split_root.exists():
            shutil.rmtree(split_root)
        (split_root / "images" / "train").mkdir(parents=True)
        (split_root / "images" / "val").mkdir(parents=True)
        (split_root / "labels" / "train").mkdir(parents=True)
        (split_root / "labels" / "val").mkdir(parents=True)

        # Link/copy images/labels for the selected train clips and all val clips.
        for src_split, dst_split in [("train", "train"), ("val", "val")]:
            src_img = data_root / "images" / src_split
            src_lbl = data_root / "labels" / src_split
            dst_img = split_root / "images" / dst_split
            dst_lbl = split_root / "labels" / dst_split
            selected = selected_names if src_split == "train" else val_names
            for name in selected:
                base = Path(name).stem
                src_img_file = src_img / f"{base}.png"
                src_lbl_file = src_lbl / f"{base}.txt"
                if src_img_file.exists():
                    shutil.copy2(src_img_file, dst_img / f"{base}.png")
                if src_lbl_file.exists():
                    shutil.copy2(src_lbl_file, dst_lbl / f"{base}.txt")

        return write_data_yaml(split_root, nc=80, names=COCO_NAMES)

    random_yaml = write_filtered_split(random_names, "random")
    harvested_yaml = write_filtered_split(harvested_names, "harvested")

    device = "0" if torch.cuda.is_available() else "cpu"
    batch = 4 if device == "cpu" else 8

    results = {
        "meta": {
            "seed": seed,
            "epochs": epochs,
            "val_ratio": val_ratio,
            "train_count": len(train_names),
            "val_count": len(val_names),
            "harvested_count": harvest_size,
            "random_count": len(random_names),
            "device": device,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "baseline": None,
        "random_subset": None,
        "harvested_subset": None,
        "train_records": train_records,
    }

    # Evaluate the pretrained YOLOv8n baseline on the validation set using the
    # same data.yaml so metrics are comparable.
    logger.info("Evaluating baseline pretrained yolov8n.pt")
    baseline_metrics = evaluate_yolo_model(
        "yolov8n.pt",
        harvested_yaml,
        device=device,
        project=str(experiment_dir / "val_runs"),
        name="baseline",
    )
    if baseline_metrics:
        results["baseline"] = baseline_metrics

    # Train on the random subset.
    if harvest_size > 0:
        logger.info(f"Training on random subset ({harvest_size} clips)")
        random_best = train_yolo_model(
            random_yaml,
            experiment_dir / "runs",
            "random_subset",
            epochs=epochs,
            imgsz=cfg.models.image_size[0],
            batch=batch,
            device=device,
        )
        if random_best:
            results["random_subset"] = {
                "model_path": str(random_best),
                "metrics": evaluate_yolo_model(str(random_best), random_yaml, device=device),
            }

        # Train on the harvested subset.
        logger.info(f"Training on harvested subset ({harvest_size} clips)")
        harvested_best = train_yolo_model(
            harvested_yaml,
            experiment_dir / "runs",
            "harvested_subset",
            epochs=epochs,
            imgsz=cfg.models.image_size[0],
            batch=batch,
            device=device,
        )
        if harvested_best:
            results["harvested_subset"] = {
                "model_path": str(harvested_best),
                "metrics": evaluate_yolo_model(str(harvested_best), harvested_yaml, device=device),
            }

    results["meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return results


def main():
    parser = argparse.ArgumentParser(description="AECS-SDC harvest retraining experiment")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--experiment-dir", type=str, default="experiments/retrain")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for each subset")
    parser.add_argument("--val-ratio", type=float, default=0.3, help="Fraction of clips used for validation")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if not Path(args.config).is_absolute():
        cfg = Config.from_yaml(str(PROJECT_ROOT / args.config))
    else:
        cfg = Config.from_yaml(args.config)

    configure_logging(cfg.logging.level, cfg.logging.file, cfg.logging.format)

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = PROJECT_ROOT / experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(cfg, experiment_dir, args.epochs, args.val_ratio, args.seed)
    out_file = experiment_dir / "retrain_experiment.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote retrain experiment results to {out_file}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
