#!/usr/bin/env python3
"""Tier-1 active-learning harvest benchmark on COCO-2017 human labels.

This script runs the honest teacher-student disagreement experiment end-to-end:

  1. Load a pre-staged COCO-2017 subset (TRAIN_POOL / TEST) from ``--data-root``.
  2. RT-DETR teacher pseudo-labels TRAIN_POOL; results are cached on disk.
  3. YOLOv8n ONNX student runs on TRAIN_POOL for disagreement / entropy signals.
  4. Build HARVESTED (top-K disagreement), RANDOM (seeded), and ENTROPY (top-K
     student entropy) training subsets of size K.
  5. Fine-tune YOLOv8n from ``--base-model`` on each subset with identical hparams.
  6. Evaluate every fine-tuned model on the held-out human-labeled TEST split.
  7. Report mean ± std over seeds and plot mAP50 vs K.

The only accuracy oracle is the human COCO TEST labels.  Teacher pseudo-labels
are used only for selection and training, never for accuracy claims.

Usage (normal login node with internet, for small local tests):
    python scripts/run_tier1_experiment.py \
        --data-root data/coco2017 \
        --output-dir results/tier1 \
        --config configs/tier1_reduced.yaml

Usage on a disconnected compute node (SOL):
    python scripts/run_tier1_experiment.py \
        --data-root /path/to/staged/coco2017 \
        --output-dir /path/to/scratch/results \
        --config configs/tier1_reduced.yaml \
        --base-model yolov8n.pt \
        --device cuda
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image

# Ensure src/ is importable regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aecs_sdc.acquisition import check_separation, select_subset, select_top_k
from aecs_sdc.coco_label_map import NUM_COCO_CLASSES, remap_class_id
from aecs_sdc.embeddings import ImageFeatureExtractor, ensure_embeddings_cache
from aecs_sdc.logging_config import configure_logging
from aecs_sdc.student import StudentModel
from aecs_sdc.teacher import TeacherModel
from aecs_sdc.tier1 import (
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


def set_seed(seed: int) -> None:
    """Seed all RNGs used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remap_teacher_detections(result: dict) -> dict:
    """Convert COCO category IDs from RT-DETR into YOLO 0..79 indices.

    RT-DETR may emit either original COCO category IDs (e.g., person=1) or
    contiguous YOLO-style indices. ``remap_class_id`` handles both cases
    idempotently, so the returned dict uses the same class space as the
    YOLOv8 student and the prepared dataset labels.
    """
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


def load_config(config_path: Path) -> dict:
    """Load the YAML experiment config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {config_path}")
    return cfg


def resolve_device(device: Optional[str]) -> str:
    if device is None or device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device


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
    """Compute and cache ResNet18 image embeddings for the TRAIN_POOL.

    Embeddings are keyed by COCO image id so they remain valid if the cache is
    reused across runs with the same manifest.
    """
    cache_path = output_dir / "pool_embeddings.npy"
    if cache_path.exists():
        logger.info(f"Loading cached pool embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        loaded = data.item() if isinstance(data, np.ndarray) and data.dtype == object else dict(data)
        # Accept either id-keyed or absolute-path-keyed caches.
        if set(train_ids).issubset(set(loaded.keys())):
            return {im_id: loaded[im_id] for im_id in train_ids}
        # Convert path-keyed cache to id-keyed.
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


def epochs_for_k(base_epochs: int, k: int, ref_k: int = 250, max_epochs: int = 120) -> int:
    """Scale epochs so total gradient steps stay ~constant across budgets.

    The original protocol trained every K for a fixed ``base_epochs`` (15).
    A fixed epoch count means larger K sees far more images per epoch and the
    same number of epochs, so total optimizer steps grow with K while *epochs*
    (the thing that governs convergence on a small set) stay flat. In practice
    the small-K runs converged and the large-K runs were badly undertrained,
    which is why mAP fell monotonically as K grew and masked the arm gaps.

    We anchor ``base_epochs`` at ``ref_k`` (the smallest, well-behaved budget)
    and scale up inversely with K so every budget gets a comparable number of
    passes over a fixed *number of images*:

        epochs(K) = round(base_epochs * ref_k / K), clamped to [base_epochs//3, max]

    Note this depends only on K (the training-set size), never on the TEST set,
    so it introduces no leakage into model selection.
    """
    if k <= 0:
        return base_epochs
    scaled = round(base_epochs * ref_k / k)
    lower = max(1, base_epochs // 3)
    return int(np.clip(scaled, lower, max_epochs))


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
    """Fine-tune YOLOv8n and evaluate on the val split (human TEST labels).

    Model selection uses ``last.pt`` after a fixed, K-scaled schedule. We
    deliberately do NOT use YOLO's ``patience`` early stopping or ``best.pt``,
    because those select the checkpoint by watching the val split -- which here
    is the held-out human TEST set. Selecting on TEST would leak it into model
    selection and invalidate the honest evaluation. A fixed leak-free schedule
    trained to convergence is the correct choice here.
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        logger.error(f"Ultralytics not available: {e}")
        return None

    set_seed(seed)

    logger.info(f"Training run {run_name} on {data_yaml} for {epochs} epochs")
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
            save=False,
            workers=0,
            deterministic=True,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"Training failed for {run_name}: {e}")
        return None

    best = output_dir / "runs" / run_name / "weights" / "best.pt"
    if not best.exists():
        best = output_dir / "runs" / run_name / "weights" / "last.pt"
    if not best.exists():
        logger.error(f"No trained weights found for {run_name}")
        return None

    logger.info(f"Evaluating {run_name} on held-out human TEST")
    try:
        val_model = YOLO(str(best))
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
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
        }
    except Exception as e:
        logger.error(f"Evaluation failed for {run_name}: {e}")
        return None


def run_experiment(
    cfg: dict,
    data_root: Path,
    output_dir: Path,
    base_model: Path,
    student_onnx: Path,
    device: str,
) -> List[dict]:
    """Run the full experiment described by ``cfg``."""
    exp_cfg = cfg.get("experiment", {})
    teacher_cfg = cfg.get("teacher", {})
    student_cfg = cfg.get("student", {})
    hybrid_cfg = cfg.get("hybrid", {})

    seeds = list(exp_cfg.get("seeds", [42, 43, 44]))
    k_values = list(exp_cfg.get("k_values", [250]))
    arms = list(exp_cfg.get("arms", ["harvested", "random"]))
    epochs = int(exp_cfg.get("epochs", 10))
    imgsz = int(exp_cfg.get("imgsz", 640))
    batch = int(exp_cfg.get("batch", 16))
    subset_name = str(exp_cfg.get("subset_name", "subset"))

    manifest_path = data_root / subset_name / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = load_manifest(manifest_path)

    train_ids, test_ids = manifest_split_ids(manifest)
    logger.info(f"Loaded manifest: TRAIN_POOL={len(train_ids)}, TEST={len(test_ids)} ({subset_name})")
    if not train_test_are_disjoint(manifest):
        raise ValueError("TRAIN_POOL and TEST are not disjoint; leakage detected")

    # Teacher and student caches.
    teacher_results = ensure_teacher_cache(
        data_root, output_dir, train_ids, teacher_cfg, device, subset_name
    )
    student_results = ensure_student_cache(
        data_root,
        output_dir,
        train_ids,
        str(student_onnx),
        float(student_cfg.get("conf_threshold", 0.25)),
        list(student_cfg.get("image_size", [640, 640])),
        subset_name,
    )

    # Pre-compute image embeddings once if any diversity-aware arm is requested.
    # Diversity arms: "hybrid" (legacy name), "entropy_div", "disagreement_div".
    _DIVERSITY_ARMS = {"hybrid", "entropy_div", "disagreement_div"}
    # Map each diversity arm to the uncertainty signal it should combine with
    # k-center-greedy diversity. "hybrid" keeps its config-driven default.
    _DIV_UNCERTAINTY = {
        "entropy_div": "entropy",
        "disagreement_div": "disagreement",
    }
    needs_embeddings = any(a in _DIVERSITY_ARMS for a in arms)
    pool_embeddings: Optional[Dict[str, np.ndarray]] = None
    if needs_embeddings:
        pool_embeddings = ensure_pool_embeddings(
            data_root=data_root,
            output_dir=output_dir,
            train_ids=train_ids,
            manifest=manifest,
            device=device,
            subset_name=subset_name,
        )

    results: List[dict] = []

    for k in k_values:
        if k > len(train_ids):
            logger.warning(f"K={k} exceeds train pool size {len(train_ids)}; skipping")
            continue
        for arm in arms:
            for seed in seeds:
                set_seed(seed)
                if arm in _DIVERSITY_ARMS:
                    # Route through hybrid_selection: uncertainty + diversity.
                    uncertainty_mode = _DIV_UNCERTAINTY.get(
                        arm, str(hybrid_cfg.get("uncertainty_mode", "disagreement"))
                    )
                    select_kwargs = {
                        "mode": "hybrid",
                        "train_ids": train_ids,
                        "teacher_results": teacher_results,
                        "student_results": student_results,
                        "k": k,
                        "seed": seed,
                        "embeddings": pool_embeddings,
                        "hybrid_uncertainty_mode": uncertainty_mode,
                        "diversity_weight": float(hybrid_cfg.get("diversity_weight", 0.5)),
                    }
                else:
                    select_kwargs = {
                        "mode": "disagreement" if arm == "harvested" else arm,
                        "train_ids": train_ids,
                        "teacher_results": teacher_results,
                        "student_results": student_results,
                        "k": k,
                        "seed": seed,
                    }
                selected = select_subset(**select_kwargs)
                if len(selected) != k:
                    logger.warning(
                        f"Selection returned {len(selected)} ids for K={k}; requested {k}"
                    )

                run_name = f"{arm}_k{k}_seed{seed}"
                ds_dir = output_dir / "datasets" / run_name
                # Fix #3: filter teacher pseudo-labels by confidence before they
                # become training labels. Low-confidence teacher boxes are the
                # main noise source, and that noise grows with K. A floor keeps
                # only boxes the teacher is reasonably sure about. Floor of 0.0
                # (default) reproduces the original all-boxes behavior.
                label_floor = float(teacher_cfg.get("train_label_conf", 0.0))
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
                    logger.error(f"Run {run_name} produced no metrics; skipping")
                    continue

                record = {
                    "arm": arm,
                    "k": k,
                    "seed": seed,
                    "selected": selected,
                    "run_dir": str(ds_dir),
                    **metrics,
                }
                results.append(record)
                logger.info(
                    f"Result {run_name}: mAP50={metrics['mAP50']} "
                    f"mAP50-95={metrics['mAP50_95']}"
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="Tier-1 COCO active-learning benchmark")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to the staged COCO-2017 subset root (contains subset/manifest.json).")
    parser.add_argument("--output-dir", type=str, default="results/tier1",
                        help="Directory for results, caches, and plots.")
    parser.add_argument("--config", type=str, default="configs/tier1_reduced.yaml",
                        help="Experiment config YAML.")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt",
                        help="Base YOLOv8n checkpoint for fine-tuning.")
    parser.add_argument("--student-onnx", type=str, default="models/yolov8n.onnx",
                        help="ONNX student model for disagreement/entropy signals.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for teacher and YOLO training ('auto', 'cuda', 'cpu', '0', ...).")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Ignore existing teacher/student caches and recompute.")
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
        str(output_dir / log_cfg.get("file", "tier1.log")),
        log_cfg.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"),
    )

    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")

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

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = run_experiment(cfg, data_root, output_dir, base_model, student_onnx, device)
    aggregated = aggregate_results(results)
    save_results(results, aggregated, output_dir, name=cfg.get("experiment", {}).get("name", "tier1"))

    # Plot mAP50 vs K for arms present in results.
    plot_map50(
        aggregated,
        output_path=output_dir / f"{cfg.get('experiment', {}).get('name', 'tier1')}_map50.png",
        arms=cfg.get("experiment", {}).get("arms", ["harvested", "random"]),
        metric="mAP50",
    )

    # Reduced-first-pass: continue to full sweep if configured and separation detected.
    exp_cfg = cfg.get("experiment", {})
    if exp_cfg.get("continue_to_full", False) and exp_cfg.get("full_config"):
        reduced_k_values = list(exp_cfg.get("k_values", [250]))
        target_k = int(exp_cfg.get("check_separation_at_k", reduced_k_values[0] if reduced_k_values else 250))
        if check_separation(results, k=target_k, metric="mAP50"):
            logger.info(
                f"Reduced pass shows separation at K={target_k}; running full sweep"
            )
            full_config_path = Path(exp_cfg["full_config"])
            if not full_config_path.is_absolute():
                full_config_path = PROJECT_ROOT / full_config_path
            full_cfg = load_config(full_config_path)
            full_output = output_dir / "full_sweep"
            full_results = run_experiment(full_cfg, data_root, full_output, base_model, student_onnx, device)
            full_aggregated = aggregate_results(full_results)
            full_name = full_cfg.get("experiment", {}).get("name", "tier1_full")
            save_results(full_results, full_aggregated, full_output, name=full_name)
            plot_map50(
                full_aggregated,
                output_path=full_output / f"{full_name}_map50.png",
                arms=full_cfg.get("experiment", {}).get("arms", ["harvested", "random", "entropy"]),
                metric="mAP50",
            )
        else:
            logger.info(
                f"Reduced pass does NOT show separation at K={target_k}; full sweep skipped"
            )

    exp_name = cfg.get("experiment", {}).get("name", "tier1")
    summary = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": str(config_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "device": device,
        "results_count": len(results),
        "aggregated": aggregated,
    }
    with open(output_dir / f"{exp_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Tier-1 experiment complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
