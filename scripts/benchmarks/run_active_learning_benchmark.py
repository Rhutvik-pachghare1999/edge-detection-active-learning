#!/usr/bin/env python3
"""
EdgeAL Active Learning Benchmark for Object Detection

This script implements a rigorous active learning benchmark following best practices
from recent literature (CVPR/ICCV 2022-2026):

Track A: Human-Label Active Learning (the gold standard)
  - COCO human labels HIDDEN during acquisition
  - Select K images using uncertainty/diversity signals
  - REVEAL human labels only for selected images
  - Train on revealed human labels
  - Evaluate on held-out TEST set

Track B: Pseudo-Label Data Engine (practical alternative)
  - Teacher (RT-DETR) pseudo-labels TRAIN_POOL
  - Select K images using teacher-student disagreement
  - Train on teacher pseudo-labels (with confidence floor)
  - Evaluate on held-out human TEST set

Metrics: COCO AP50:95 (primary), AP50, AP75, AP_small, AP_medium, AP_large
Budgets: 2%, 4%, 6%, 8%, 10% of TRAIN_POOL
Seeds: 5 (42, 43, 44, 45, 46)
Baselines: Random, Entropy, Least Confidence, Margin, Disagreement, Core-Set, Hybrid

Usage:
    python scripts/benchmarks/run_active_learning_benchmark.py \
        --data-root data/coco2017 \
        --output-dir results/al_benchmark \
        --track A \
        --budgets 2 4 6 8 10 \
        --seeds 42 43 44 45 46
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edgeal.acquisition import (
    check_separation,
    select_random_k,
    select_subset,
)
from edgeal.embeddings import ImageFeatureExtractor, ensure_embeddings_cache
from edgeal.logging_config import configure_logging
from edgeal.student import StudentModel
from edgeal.teacher import TeacherModel
from edgeal.tier1 import (
    COCO_NAMES,
    aggregate_results,
    load_manifest,
    manifest_id_to_info,
    manifest_split_ids,
    plot_map50,
    prepare_yolo_dataset,
    save_results,
    train_test_are_disjoint,
)
from edgeal.coco_label_map import remap_class_id, NUM_COCO_CLASSES


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BUDGETS_PCT = [2, 4, 6, 8, 10]  # Percentage of TRAIN_POOL
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_EPOCHS = 15
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16

# Track A arms (human labels revealed after selection) - ONLY student uncertainty signals
TRACK_A_ARMS = [
    "random",
    "entropy",           # image-level mean entropy
    "max_entropy",       # image-level max entropy
    "least_confidence",  # image-level mean least confidence
    "margin",            # image-level mean margin (inverted)
    "hybrid",            # entropy + diversity
]

# Track B arms (pseudo-labels from teacher) - includes teacher-student disagreement
TRACK_B_ARMS = [
    "random",
    "entropy",
    "disagreement",
    "disagreement_div",  # disagreement + diversity
    "entropy_div",       # entropy + diversity
]


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)


def resolve_device(device: Optional[str]) -> str:
    if device is None or device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device


def epochs_for_k(base_epochs: int, k: int, ref_k: int = 250, max_epochs: int = 120) -> int:
    """Scale epochs so total gradient steps stay ~constant across budgets."""
    if k <= 0:
        return base_epochs
    scaled = round(base_epochs * ref_k / k)
    lower = max(1, base_epochs // 3)
    return int(np.clip(scaled, lower, max_epochs))


def remap_teacher_detections(result: dict) -> dict:
    """Convert COCO category IDs from RT-DETR into YOLO 0..79 indices."""
    mapped = dict(result)
    mapped["top_class_id"] = remap_class_id(result.get("top_class_id", -1))
    mapped["top_class_name"] = (
        COCO_NAMES[mapped["top_class_id"]]
        if 0 <= mapped["top_class_id"] < NUM_COCO_CLASSES
        else "unknown"
    )
    mapped_detections = []
    for det in result.get("detections", []):
        md = dict(det)
        md["label"] = remap_class_id(det.get("label", -1))
        mapped_detections.append(md)
    mapped["detections"] = mapped_detections
    return mapped


# ──────────────────────────────────────────────────────────────────────────────
# Caching: Teacher & Student Predictions
# ──────────────────────────────────────────────────────────────────────────────

def ensure_teacher_cache(
    data_root: Path,
    output_dir: Path,
    train_ids: List[str],
    teacher_cfg: dict,
    device: str,
    subset_name: str = "subset",
) -> Dict[str, dict]:
    """Run RT-DETR on TRAIN_POOL and cache pseudo-labels as JSON."""
    cache_path = output_dir / "teacher_pseudo_labels.json"
    if cache_path.exists():
        logger.info(f"Loading cached teacher pseudo-labels from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    logger.info("Running RT-DETR teacher on TRAIN_POOL")
    teacher = TeacherModel(
        model_name=teacher_cfg.get("model", "PekingU/rtdetr_r50vd"),
        device=device,
        conf_threshold=float(teacher_cfg.get("conf_threshold", 0.30)),
    )
    subset_root = data_root / subset_name
    lookup = manifest_id_to_info(load_manifest(subset_root / "manifest.json"))

    results: Dict[str, dict] = {}
    for idx, im_id in enumerate(train_ids, 1):
        info = lookup[im_id]
        img_path = subset_root / "images" / "train" / info["file_name"]
        frame = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        results[im_id] = remap_teacher_detections(teacher.infer(frame))
        if idx % 50 == 0:
            logger.info(f"Teacher labeled {idx}/{len(train_ids)} train images")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Cached teacher pseudo-labels to {cache_path}")
    return results


def ensure_student_cache(
    data_root: Path,
    output_dir: Path,
    train_ids: List[str],
    student_onnx: str,
    student_conf: float,
    image_size: List[int],
    subset_name: str = "subset",
) -> Dict[str, dict]:
    """Run YOLOv8n ONNX student on TRAIN_POOL and cache predictions as JSON."""
    cache_path = output_dir / "student_predictions.json"
    if cache_path.exists():
        logger.info(f"Loading cached student predictions from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    logger.info("Running YOLOv8n ONNX student on TRAIN_POOL")
    student = StudentModel(
        model_path=student_onnx,
        conf_threshold=float(student_conf),
        input_size=tuple(image_size),
    )
    subset_root = data_root / subset_name
    lookup = manifest_id_to_info(load_manifest(subset_root / "manifest.json"))

    results: Dict[str, dict] = {}
    for idx, im_id in enumerate(train_ids, 1):
        info = lookup[im_id]
        img_path = subset_root / "images" / "train" / info["file_name"]
        frame = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
        results[im_id] = student.infer(frame)
        if idx % 50 == 0:
            logger.info(f"Student predicted {idx}/{len(train_ids)} train images")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Cached student predictions to {cache_path}")
    return results


def ensure_pool_embeddings(
    data_root: Path,
    output_dir: Path,
    train_ids: List[str],
    manifest: dict,
    device: str,
    subset_name: str = "subset",
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Compute and cache ResNet18 image embeddings for the TRAIN_POOL."""
    cache_path = output_dir / "pool_embeddings.npy"
    if cache_path.exists():
        logger.info(f"Loading cached pool embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        loaded = data.item() if isinstance(data, np.ndarray) and data.dtype == object else dict(data)
        if set(train_ids).issubset(set(loaded.keys())):
            return {im_id: loaded[im_id] for im_id in train_ids}
        # Convert path-keyed cache to id-keyed
        lookup = manifest_id_to_info(manifest)
        subset_root = data_root / subset_name
        id_to_path = {
            im_id: str((subset_root / "images" / "train" / lookup[im_id]["file_name"]).resolve())
            for im_id in train_ids
        }
        id_embeddings = {}
        for im_id, path in id_to_path.items():
            if path in loaded:
                id_embeddings[im_id] = loaded[path]
        if len(id_embeddings) == len(train_ids):
            return id_embeddings
        logger.warning("Cached embeddings incomplete; recomputing")

    logger.info("Computing ResNet18 embeddings for TRAIN_POOL")
    extractor = ImageFeatureExtractor(device=device)
    lookup = manifest_id_to_info(manifest)
    subset_root = data_root / subset_name
    image_paths = [
        str((subset_root / "images" / "train" / lookup[im_id]["file_name"]).resolve())
        for im_id in train_ids
    ]
    path_embeddings = extractor.extract_batch(image_paths, batch_size=batch_size)
    id_embeddings = {
        im_id: path_embeddings[str(path)]
        for im_id, path in zip(train_ids, image_paths)
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, id_embeddings)
    logger.info(f"Cached {len(id_embeddings)} pool embeddings to {cache_path}")
    return id_embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Track A: Human-Label Active Learning
# ──────────────────────────────────────────────────────────────────────────────

def run_track_a_human_label(
    cfg: dict,
    data_root: Path,
    output_dir: Path,
    base_model: Path,
    student_onnx: Path,
    device: str,
    budgets_pct: List[int],
    seeds: List[int],
) -> List[dict]:
    """
    Track A: True Active Learning with Human Labels
    
    Protocol:
    1. TRAIN_POOL has HIDDEN human labels (COCO annotations)
    2. Student runs on TRAIN_POOL to get uncertainty signals
    3. Select K images using uncertainty/diversity
    4. REVEAL human labels ONLY for selected K images
    5. Train on revealed human labels
    6. Evaluate on held-out TEST (human labels)
    """
    exp_cfg = cfg.get("experiment", {})
    teacher_cfg = cfg.get("teacher", {})
    student_cfg = cfg.get("student", {})
    hybrid_cfg = cfg.get("hybrid", {})
    
    epochs = int(exp_cfg.get("epochs", DEFAULT_EPOCHS))
    imgsz = int(exp_cfg.get("imgsz", DEFAULT_IMGSZ))
    batch = int(exp_cfg.get("batch", DEFAULT_BATCH))
    subset_name = str(exp_cfg.get("subset_name", "subset"))
    
    manifest_path = data_root / subset_name / "manifest.json"
    manifest = load_manifest(manifest_path)
    train_ids, test_ids = manifest_split_ids(manifest)
    logger.info(f"Track A: TRAIN_POOL={len(train_ids)}, TEST={len(test_ids)}")
    
    if not train_test_are_disjoint(manifest):
        raise ValueError("TRAIN_POOL and TEST are not disjoint!")
    
    # Student cache (for uncertainty signals)
    student_results = ensure_student_cache(
        data_root, output_dir, train_ids,
        str(student_onnx),
        float(student_cfg.get("conf_threshold", 0.25)),
        list(student_cfg.get("image_size", [640, 640])),
        subset_name,
    )
    
    # Embeddings for diversity
    pool_embeddings = ensure_pool_embeddings(
        data_root, output_dir, train_ids, manifest, device, subset_name
    )
    
    # Load human labels for TRAIN_POOL (HIDDEN during selection, REVEALED after)
    subset_root = data_root / subset_name
    lookup = manifest_id_to_info(manifest)
    human_labels = {}
    for im_id in train_ids:
        info = lookup[im_id]
        label_path = subset_root / info["label_file"]
        if label_path.exists():
            with open(label_path) as f:
                human_labels[im_id] = [
                    line.strip() for line in f if line.strip()
                ]
        else:
            human_labels[im_id] = []
    
    results = []
    
    for pct in budgets_pct:
        k = max(1, int(len(train_ids) * pct / 100))
        logger.info(f"Budget: {pct}% = K={k}")
        
        for arm in TRACK_A_ARMS:
            for seed in seeds:
                set_seed(seed)
                
                # Select K images using uncertainty (human labels HIDDEN)
                if arm == "random":
                    selected = select_subset(
                        mode="random",
                        train_ids=train_ids,
                        teacher_results={},  # Not used for random
                        student_results=student_results,
                        k=k,
                        seed=seed,
                    )
                elif arm == "hybrid":
                    selected = select_subset(
                        mode="hybrid",
                        train_ids=train_ids,
                        teacher_results={},
                        student_results=student_results,
                        k=k,
                        seed=seed,
                        embeddings=pool_embeddings,
                        hybrid_uncertainty_mode="entropy",
                        diversity_weight=float(hybrid_cfg.get("diversity_weight", 0.5)),
                    )
                else:
                    # Map arm names to acquisition modes (Track A only uses student uncertainty)
                    mode_map = {
                        "entropy": "image_entropy",
                        "max_entropy": "image_max_entropy",
                        "least_confidence": "least_confidence",
                        "margin": "margin",
                    }
                    mode = mode_map.get(arm, arm)
                    selected = select_subset(
                        mode=mode,
                        train_ids=train_ids,
                        teacher_results={},  # Not used for Track A uncertainty modes
                        student_results=student_results,
                        k=k,
                        seed=seed,
                        embeddings=pool_embeddings if arm == "hybrid" else None,
                    )
                
                if len(selected) != k:
                    logger.warning(f"Selection returned {len(selected)} for K={k}")
                
                # REVEAL human labels for selected images ONLY
                run_name = f"trackA_{arm}_k{k}_pct{pct}_seed{seed}"
                ds_dir = output_dir / "datasets" / run_name
                
                # Prepare YOLO dataset with REVEALED human labels
                train_pseudo_labels = {}
                for im_id in selected:
                    # Use HUMAN labels (converted to YOLO format)
                    lines = human_labels.get(im_id, [])
                    train_pseudo_labels[im_id] = [
                        {"label": int(l.split()[0]), "cx": float(l.split()[1]), 
                         "cy": float(l.split()[2]), "bw": float(l.split()[3]), "bh": float(l.split()[4])}
                        for l in lines
                    ]
                
                data_yaml = prepare_yolo_dataset(
                    data_root=data_root,
                    output_dir=ds_dir,
                    selected_train_ids=selected,
                    test_ids=test_ids,
                    train_pseudo_labels=train_pseudo_labels,  # Human labels!
                    manifest=manifest,
                    subset_name=subset_name,
                    run_name=run_name,
                )
                
                # Train and evaluate
                metrics = train_and_evaluate(
                    data_yaml=data_yaml,
                    base_model=base_model,
                    output_dir=ds_dir,
                    run_name="train",
                    device=device,
                    epochs=epochs_for_k(epochs, k),
                    imgsz=imgsz,
                    batch=batch,
                    seed=seed,
                )
                
                if metrics is None:
                    logger.error(f"Run {run_name} produced no metrics")
                    continue
                
                record = {
                    "track": "A",
                    "arm": arm,
                    "k": k,
                    "budget_pct": pct,
                    "seed": seed,
                    "selected": selected,
                    "run_dir": str(ds_dir),
                    **metrics,
                }
                results.append(record)
                logger.info(f"Track A {run_name}: mAP50={metrics['mAP50']:.4f} mAP50-95={metrics['mAP50_95']:.4f}")
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Track B: Pseudo-Label Data Engine
# ──────────────────────────────────────────────────────────────────────────────

def run_track_b_pseudo_label(
    cfg: dict,
    data_root: Path,
    output_dir: Path,
    base_model: Path,
    student_onnx: Path,
    device: str,
    budgets_pct: List[int],
    seeds: List[int],
) -> List[dict]:
    """
    Track B: Pseudo-Label Data Engine
    
    Protocol:
    1. Teacher (RT-DETR) pseudo-labels entire TRAIN_POOL
    2. Student runs on TRAIN_POOL for disagreement signals
    3. Select K images using disagreement/entropy
    4. Train on teacher pseudo-labels (with confidence floor)
    5. Evaluate on held-out human TEST
    """
    exp_cfg = cfg.get("experiment", {})
    teacher_cfg = cfg.get("teacher", {})
    student_cfg = cfg.get("student", {})
    hybrid_cfg = cfg.get("hybrid", {})
    
    epochs = int(exp_cfg.get("epochs", DEFAULT_EPOCHS))
    imgsz = int(exp_cfg.get("imgsz", DEFAULT_IMGSZ))
    batch = int(exp_cfg.get("batch", DEFAULT_BATCH))
    subset_name = str(exp_cfg.get("subset_name", "subset"))
    
    manifest_path = data_root / subset_name / "manifest.json"
    manifest = load_manifest(manifest_path)
    train_ids, test_ids = manifest_split_ids(manifest)
    logger.info(f"Track B: TRAIN_POOL={len(train_ids)}, TEST={len(test_ids)}")
    
    # Teacher and student caches
    teacher_results = ensure_teacher_cache(
        data_root, output_dir, train_ids, teacher_cfg, device, subset_name
    )
    student_results = ensure_student_cache(
        data_root, output_dir, train_ids,
        str(student_onnx),
        float(student_cfg.get("conf_threshold", 0.25)),
        list(student_cfg.get("image_size", [640, 640])),
        subset_name,
    )
    
    # Embeddings for diversity
    pool_embeddings = ensure_pool_embeddings(
        data_root, output_dir, train_ids, manifest, device, subset_name
    )
    
    # Confidence floor for teacher pseudo-labels
    label_floor = float(teacher_cfg.get("train_label_conf", 0.5))
    
    results = []
    
    for pct in budgets_pct:
        k = max(1, int(len(train_ids) * pct / 100))
        logger.info(f"Budget: {pct}% = K={k}")
        
        for arm in TRACK_B_ARMS:
            for seed in seeds:
                set_seed(seed)
                
                # Select K images
                if arm == "random":
                    selected = select_subset(
                        mode="random",
                        train_ids=train_ids,
                        teacher_results=teacher_results,
                        student_results=student_results,
                        k=k,
                        seed=seed,
                    )
                elif arm in ("disagreement_div", "entropy_div"):
                    uncertainty_mode = "disagreement" if arm == "disagreement_div" else "entropy"
                    selected = select_subset(
                        mode="hybrid",
                        train_ids=train_ids,
                        teacher_results=teacher_results,
                        student_results=student_results,
                        k=k,
                        seed=seed,
                        embeddings=pool_embeddings,
                        hybrid_uncertainty_mode=uncertainty_mode,
                        diversity_weight=float(hybrid_cfg.get("diversity_weight", 0.5)),
                    )
                else:
                    mode_map = {
                        "entropy": "entropy",
                        "disagreement": "disagreement",
                    }
                    selected = select_subset(
                        mode=mode_map.get(arm, arm),
                        train_ids=train_ids,
                        teacher_results=teacher_results,
                        student_results=student_results,
                        k=k,
                        seed=seed,
                    )
                
                if len(selected) != k:
                    logger.warning(f"Selection returned {len(selected)} for K={k}")
                
                run_name = f"trackB_{arm}_k{k}_pct{pct}_seed{seed}"
                ds_dir = output_dir / "datasets" / run_name
                
                # Filter teacher pseudo-labels by confidence floor
                pseudo_labels = {}
                for im_id in selected:
                    dets = teacher_results.get(im_id, {}).get("detections", [])
                    if label_floor > 0.0:
                        dets = [
                            d for d in dets
                            if float(d.get("score", d.get("confidence", 1.0))) >= label_floor
                        ]
                    pseudo_labels[im_id] = dets
                
                data_yaml = prepare_yolo_dataset(
                    data_root=data_root,
                    output_dir=ds_dir,
                    selected_train_ids=selected,
                    test_ids=test_ids,
                    train_pseudo_labels=pseudo_labels,
                    manifest=manifest,
                    subset_name=subset_name,
                    run_name=run_name,
                )
                
                metrics = train_and_evaluate(
                    data_yaml=data_yaml,
                    base_model=base_model,
                    output_dir=ds_dir,
                    run_name="train",
                    device=device,
                    epochs=epochs_for_k(epochs, k),
                    imgsz=imgsz,
                    batch=batch,
                    seed=seed,
                )
                
                if metrics is None:
                    logger.error(f"Run {run_name} produced no metrics")
                    continue
                
                record = {
                    "track": "B",
                    "arm": arm,
                    "k": k,
                    "budget_pct": pct,
                    "seed": seed,
                    "selected": selected,
                    "run_dir": str(ds_dir),
                    **metrics,
                }
                results.append(record)
                logger.info(f"Track B {run_name}: mAP50={metrics['mAP50']:.4f} mAP50-95={metrics['mAP50_95']:.4f}")
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    data_yaml: Path,
    base_model: Path,
    output_dir: Path,
    run_name: str,
    device: str,
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int,
) -> Optional[dict]:
    """Fine-tune YOLOv8n and evaluate on val split (human TEST labels)."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        logger.error(f"Ultralytics not available: {e}")
        return None

    set_seed(seed)

    logger.info(f"Training {run_name} on {data_yaml} for {epochs} epochs")
    run_dir = output_dir / "runs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    try:
        model = YOLO(str(base_model))
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(output_dir / "runs"),
            name=run_name,
            exist_ok=True,
            verbose=False,
            plots=False,
            save=True,
            workers=0,
            deterministic=True,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"Training failed for {run_name}: {e}")
        return None

    # Use last.pt for leak-free evaluation
    last = output_dir / "runs" / run_name / "weights" / "last.pt"
    if not last.exists():
        logger.error(f"No trained weights (last.pt) found for {run_name}")
        return None

    logger.info(f"Evaluating {run_name} on held-out human TEST")
    try:
        val_model = YOLO(str(last))
        metrics = val_model.val(
            data=str(data_yaml),
            split="val",
            device=device,
            verbose=False,
            workers=0,
            project=str(output_dir / "val_runs"),
            name=run_name,
            exist_ok=True,
        )
        return {
            "mAP50": round(float(metrics.box.map50), 6),
            "mAP50_95": round(float(metrics.box.map), 6),
            "mAP75": round(float(metrics.box.map75), 6) if hasattr(metrics.box, 'map75') else 0.0,
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
        }
    except Exception as e:
        logger.error(f"Evaluation failed for {run_name}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {config_path}")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="EdgeAL Active Learning Benchmark")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to staged COCO-2017 subset (contains subset/manifest.json)")
    parser.add_argument("--output-dir", type=str, default="results/al_benchmark",
                        help="Directory for results, caches, and plots")
    parser.add_argument("--config", type=str, default="configs/al_benchmark.yaml",
                        help="Experiment config YAML")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt",
                        help="Base YOLOv8n checkpoint for fine-tuning")
    parser.add_argument("--student-onnx", type=str, default="models/yolov8n.onnx",
                        help="ONNX student model for uncertainty signals")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for teacher and YOLO training")
    parser.add_argument("--track", type=str, choices=["A", "B", "both"], default="both",
                        help="Which track to run: A=human-label, B=pseudo-label, both")
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS_PCT,
                        help="Budget percentages (e.g., 2 4 6 8 10)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds for reproducibility")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Ignore existing caches and recompute")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    base_model = Path(args.base_model)
    if not base_model.is_absolute():
        base_model = PROJECT_ROOT / base_model
    student_onnx = Path(args.student_onnx)
    if not student_onnx.is_absolute():
        student_onnx = PROJECT_ROOT / student_onnx

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    log_cfg = cfg.get("logging", {})
    configure_logging(
        log_cfg.get("level", "INFO"),
        str(output_dir / log_cfg.get("file", "al_benchmark.log")),
        log_cfg.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"),
    )

    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Track: {args.track}")
    logger.info(f"Budgets: {args.budgets}%")
    logger.info(f"Seeds: {args.seeds}")

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not base_model.exists():
        raise FileNotFoundError(f"Base YOLO model not found: {base_model}")
    if not student_onnx.exists():
        raise FileNotFoundError(f"ONNX student not found: {student_onnx}")

    if args.skip_cache:
        for cache in [
            output_dir / "teacher_pseudo_labels.json",
            output_dir / "student_predictions.json",
            output_dir / "pool_embeddings.npy",
        ]:
            if cache.exists():
                cache.unlink()

    all_results = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if args.track in ("A", "both"):
        logger.info("=" * 60)
        logger.info("RUNNING TRACK A: Human-Label Active Learning")
        logger.info("=" * 60)
        track_a_results = run_track_a_human_label(
            cfg, data_root, output_dir / "trackA",
            base_model, student_onnx, device,
            args.budgets, args.seeds
        )
        all_results.extend(track_a_results)

    if args.track in ("B", "both"):
        logger.info("=" * 60)
        logger.info("RUNNING TRACK B: Pseudo-Label Data Engine")
        logger.info("=" * 60)
        track_b_results = run_track_b_pseudo_label(
            cfg, data_root, output_dir / "trackB",
            base_model, student_onnx, device,
            args.budgets, args.seeds
        )
        all_results.extend(track_b_results)

    # Aggregate and save
    aggregated = aggregate_results(all_results)
    exp_name = cfg.get("experiment", {}).get("name", "al_benchmark")
    save_results(all_results, aggregated, output_dir, name=exp_name)

    # Plot mAP50 vs budget for each track
    for track in ["A", "B"]:
        track_results = [r for r in all_results if r.get("track") == track]
        if not track_results:
            continue
        track_agg = aggregate_results(track_results)
        arms = TRACK_A_ARMS if track == "A" else TRACK_B_ARMS
        plot_map50(
            track_agg,
            output_path=output_dir / f"{exp_name}_track{track}_map50.png",
            arms=arms,
            metric="mAP50",
        )
        # Also plot mAP50-95
        plot_map50(
            track_agg,
            output_path=output_dir / f"{exp_name}_track{track}_map50_95.png",
            arms=arms,
            metric="mAP50_95",
        )

    # Summary
    summary = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": str(config_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "device": device,
        "track": args.track,
        "budgets_pct": args.budgets,
        "seeds": args.seeds,
        "results_count": len(all_results),
        "aggregated": aggregated,
    }
    with open(output_dir / f"{exp_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 80)
    print("EDGEAL ACTIVE LEARNING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Track: {args.track} | Budgets: {args.budgets}% | Seeds: {len(args.seeds)}")
    print("-" * 80)
    
    for track in ["A", "B"]:
        track_results = [r for r in all_results if r.get("track") == track]
        if not track_results:
            continue
        track_agg = aggregate_results(track_results)
        print(f"\nTrack {track} ({'Human-Label AL' if track == 'A' else 'Pseudo-Label Engine'}):")
        print(f"{'Arm':<25} {'Budget%':>8} {'mAP50':>10} {'mAP50-95':>10} {'mAP75':>10}")
        print("-" * 65)
        arms = TRACK_A_ARMS if track == "A" else TRACK_B_ARMS
        for arm in arms:
            for pct in args.budgets:
                key = f"{arm}_k{max(1, int(1000 * pct / 100))}"  # approximate
                # Find exact key
                for k, v in track_agg.items():
                    if v["arm"] == arm and v["k"] == max(1, int(len(track_results[0].get("selected", [])) * pct / 100)):
                        m = v["metrics"]
                        print(f"{arm:<25} {pct:>7}% {m.get('mAP50', {}).get('mean', 0):>10.4f} "
                              f"{m.get('mAP50_95', {}).get('mean', 0):>10.4f} "
                              f"{m.get('mAP75', {}).get('mean', 0):>10.4f}")
    
    logger.info("EdgeAL benchmark complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()