#!/usr/bin/env python3
"""AECS-SDC dataset-driven benchmark entry point.

Usage:
    source ~/venvs/aecs-supervisor/bin/activate
    python benchmark.py --config configs/benchmark.yaml

This script does NOT require a Raspberry Pi, ROS2, Isaac Sim, or live camera.
It runs the committed teacher (RT-DETR) and student (YOLOv8n ONNX) on the
committed local clips and YOLO labels, computes disagreement, applies a harvest
policy, and writes reproducible JSON/text results.

Rules observed:
* Never touches data/events.db.
* Never runs run_experiment.sh.
* Uses only local clips/labels.
* Seeds RNG for deterministic subset selection.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Ensure src/ is importable regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aecs_sdc.config import Config
from aecs_sdc.dataset import build_subset, iter_samples, load_subset
from aecs_sdc.disagreement import compute_disagreement, compute_student_teacher_agreement
from aecs_sdc.evaluator import Evaluator
from aecs_sdc.harvest import build_policy
from aecs_sdc.label_map import LabelMap
from aecs_sdc.logging_config import configure_logging
from aecs_sdc.student import StudentModel
from aecs_sdc.teacher import TeacherModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms are best-effort; not all ops support it.
    torch.use_deterministic_algorithms(False, warn_only=True)


def make_frame_record(sample, teacher_result, student_result, disagreement, decision, agreement_metrics):
    """Serialize a single frame's results."""
    return {
        "clip_name": sample.clip_name,
        "frame_idx": sample.frame_idx,
        "teacher_top_confidence": teacher_result["top_confidence"],
        "teacher_top_class_id": teacher_result["top_class_id"],
        "teacher_top_class_name": teacher_result["top_class_name"],
        "student_top_confidence": student_result["top_confidence"],
        "student_top_class_id": student_result["top_class_id"],
        "student_top_class_name": student_result["top_class_name"],
        "disagreement": {
            "margin": disagreement.margin,
            "entropy": disagreement.entropy,
            "class_mismatch": disagreement.class_mismatch,
            "iou": disagreement.iou,
            "combined": disagreement.combined,
        },
        "harvested": decision.harvested,
        "harvest_reason": decision.reason,
        "harvest_score": decision.score,
        "tau_at_decision": decision.tau,
        "student_vs_teacher_agreement": agreement_metrics,
        "labels": [
            {"class_id": lb.class_id, "cx": lb.cx, "cy": lb.cy, "bw": lb.bw, "bh": lb.bh}
            for lb in sample.labels
        ],
        "teacher_detections": teacher_result["detections"],
        "student_detections": student_result["detections"],
    }


def run(cfg: Config) -> dict:
    cfg = cfg.resolve_paths(PROJECT_ROOT)
    log_cfg = cfg.logging
    configure_logging(log_cfg.level, log_cfg.file, log_cfg.format)

    set_seed(cfg.benchmark.seed)
    logger.info("Starting AECS-SDC benchmark")
    logger.info(f"Project root: {PROJECT_ROOT}")
    config_log = json.dumps(
        {
            'benchmark': cfg.benchmark.__dict__,
            'models': cfg.models.__dict__,
            'disagreement': cfg.disagreement.__dict__,
            'harvest': cfg.harvest.__dict__,
        },
        indent=2,
        default=str,
    )
    logger.info(f"Config: {config_log}")

    # Load subset: committed file first, then truncate to the configured size.
    if Path(cfg.benchmark.subset_file).exists():
        subset = load_subset(cfg.benchmark.subset_file)
        logger.info(f"Loaded committed subset: {len(subset)} clips")
    else:
        logger.warning("Committed subset file missing; building deterministic subset")
        subset = build_subset(
            cfg.benchmark.clips_dir,
            cfg.benchmark.labels_dir,
            cfg.benchmark.subset_size,
            cfg.benchmark.seed,
        )

    if not subset:
        raise RuntimeError("No clips found for benchmark")

    requested = cfg.benchmark.subset_size
    if requested and 0 < requested < len(subset):
        subset = subset[:requested]

    logger.info(f"Benchmark subset: {len(subset)} clips")

    # Load and validate label mapping.
    label_map = LabelMap.from_yaml(cfg.benchmark.label_map)
    logger.info(f"Label map: {label_map.summary()}")
    all_label_ids = set()
    for name in subset:
        label_path = Path(cfg.benchmark.labels_dir) / Path(name).with_suffix(".txt").name
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        all_label_ids.add(int(parts[0]))
    validation = label_map.validate(all_label_ids)
    logger.info(f"Label validation: {validation}")
    if validation["unknown"] > 0:
        logger.warning("Unknown label class IDs found; see logs above")

    # Load models.
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
    budget_decisions = None
    is_budget_mode = cfg.harvest.mode == "budget"

    evaluator = Evaluator(cfg, label_map)
    evaluator.start()

    for sample in iter_samples(
        cfg.benchmark.clips_dir,
        cfg.benchmark.labels_dir,
        subset,
        cfg.benchmark.trigger_offset_seconds,
        label_map,
    ):
        logger.info(f"Processing {sample.clip_name}")
        teacher_result = teacher.infer(sample.frame)
        student_result = student.infer(sample.frame)

        disagreement = compute_disagreement(
            teacher_result,
            student_result,
            cfg.disagreement.weights,
        )

        # Harvest decision.
        if cfg.harvest.mode == "diversity":
            teacher_class_ids = [int(d["label"]) for d in teacher_result["detections"]]
            decision = harvest_policy.decide(sample.clip_name, disagreement.combined, teacher_class_ids)
        else:
            decision = harvest_policy.decide(disagreement.combined)

        agreement_metrics = compute_student_teacher_agreement(
            student_result["detections"],
            [{"class_id": lb.class_id, "cx": lb.cx, "cy": lb.cy, "bw": lb.bw, "bh": lb.bh}
             for lb in sample.labels],
        )

        record = make_frame_record(
            sample, teacher_result, student_result, disagreement, decision, agreement_metrics
        )
        evaluator.add(record)

    # Finalize budget policy if used.
    if is_budget_mode:
        budget_decisions = harvest_policy.finalize()
        for record in evaluator._records:
            decision = budget_decisions[record["clip_name"]]
            record["harvested"] = decision.harvested
            record["harvest_reason"] = decision.reason
            record["harvest_score"] = decision.score
            record["tau_at_decision"] = decision.tau

    summary = evaluator.finish()
    logger.info("Benchmark complete")
    logger.info(f"Total frames: {summary['meta']['total_frames']}")
    logger.info(f"Harvested: {summary['meta']['harvested_count']} ({summary['meta']['harvest_rate_pct']}%)")
    logger.info(f"Mean combined disagreement: {summary['disagreement']['combined']['mean']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="AECS-SDC offline benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to benchmark config file (YAML).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Override the number of clips to benchmark.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    cfg = Config.from_yaml(str(config_path))
    if args.subset_size is not None:
        cfg.benchmark.subset_size = args.subset_size

    summary = run(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
