"""Minimal honest retraining experiment for the AECS-SDC harvest policy.

This module creates a temporary YOLO-format dataset from the benchmark clips,
trains a YOLOv8n model on either the harvested subset or a random subset of the
same size, and evaluates on a held-out validation set. The goal is to produce
measurable before/after evidence that harvesting disagreement-driven frames
helps the student model.

Because the benchmark only samples one frame per clip, the dataset is tiny. The
experiment is therefore intentionally small and reports the result honestly,
even if the improvement is marginal or negative.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml
from loguru import logger


# COCO 80 class names in the order the YOLOv8 student expects.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def extract_frame(clip_path: str, offset_seconds: float = 5.0) -> Tuple[Optional[object], int]:
    """Extract a single BGR frame near the trigger time."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None, -1
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(int(offset_seconds * fps), max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return (frame, idx) if ok else (None, -1)


def read_yolo_labels(label_path: str) -> List[List[float]]:
    """Return normalized YOLO boxes [[class, cx, cy, bw, bh], ...]."""
    boxes = []
    if not Path(label_path).exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def create_yolo_split(
    clip_names: List[str],
    clips_dir: str,
    labels_dir: str,
    out_dir: Path,
    split_name: str,
    offset_seconds: float = 5.0,
) -> int:
    """Copy frames and labels for one split into a temporary YOLO dataset.

    Returns the number of images created.
    """
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for name in clip_names:
        clip_path = Path(clips_dir) / name
        label_path = Path(labels_dir) / Path(name).with_suffix(".txt").name
        frame, _ = extract_frame(str(clip_path), offset_seconds)
        if frame is None:
            logger.warning(f"Could not extract frame from {name}; skipping split {split_name}")
            continue
        base = Path(name).stem
        img_file = img_dir / f"{base}.png"
        cv2.imwrite(str(img_file), frame)

        boxes = read_yolo_labels(str(label_path))
        if boxes:
            with open(lbl_dir / f"{base}.txt", "w") as f:
                for box in boxes:
                    f.write(" ".join(str(x) for x in box) + "\n")
        else:
            # YOLO expects an empty file for background images.
            (lbl_dir / f"{base}.txt").write_text("")
        count += 1
    return count


def write_data_yaml(out_dir: Path, nc: int = 80, names: Optional[List[str]] = None) -> str:
    """Write a YOLO data.yaml file and return its path."""
    if names is None:
        names = COCO_NAMES
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names,
    }
    path = out_dir / "data.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path)


def split_train_val(items: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Deterministic train/val split."""
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[n_val:], shuffled[:n_val]


def train_yolo_model(
    data_yaml: str,
    output_dir: Path,
    run_name: str,
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 8,
    device: Optional[str] = None,
    base_model: str = "yolov8n.pt",
) -> Optional[Path]:
    """Train a YOLOv8 model and return the path to best.pt."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        logger.error(f"Ultralytics not available: {e}")
        return None

    if device is None:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    run_dir = output_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    logger.info(f"Training {base_model} on {data_yaml} for {epochs} epochs (device={device})")
    try:
        model = YOLO(base_model)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
            verbose=False,
            plots=False,
            save=False,
            workers=0,
            deterministic=True,
        )
        best = output_dir / run_name / "weights" / "best.pt"
        if not best.exists():
            # Fallback to last.pt if best.pt is missing.
            best = output_dir / run_name / "weights" / "last.pt"
        return best if best.exists() else None
    except Exception as e:
        logger.error(f"Training failed for {run_name}: {e}")
        return None


def evaluate_yolo_model(
    model_path: str,
    data_yaml: str,
    device: Optional[str] = None,
    project: Optional[str] = None,
    name: str = "val",
) -> Optional[Dict]:
    """Evaluate a trained YOLO model on the val split and return metrics."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        logger.error(f"Ultralytics not available: {e}")
        return None

    if device is None:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Evaluating {model_path} on validation split (device={device})")
    try:
        model = YOLO(model_path)
        project = project or str(Path(model_path).parent)
        metrics = model.val(
            data=data_yaml,
            split="val",
            device=device,
            verbose=False,
            workers=0,
            project=project,
            name=name,
            exist_ok=True,
        )
        return {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None
