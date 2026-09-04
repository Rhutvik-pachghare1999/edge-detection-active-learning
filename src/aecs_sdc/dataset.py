"""Dataset utilities for the AECS-SDC offline benchmark.

Loads a deterministic subset of MP4 clips and their YOLO-format label files,
extracts a representative frame from each clip, and returns structured sample
dicts for the pipeline.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


@dataclass
class Label:
    class_id: int
    cx: float
    cy: float
    bw: float
    bh: float


@dataclass
class Sample:
    clip_name: str
    clip_path: str
    label_path: str
    frame: np.ndarray
    frame_idx: int
    labels: List[Label] = field(default_factory=list)


def read_yolo_labels(label_path: str, label_map=None) -> List[Label]:
    """Read a YOLO-format label file and optionally remap class IDs."""
    from aecs_sdc.label_map import LabelMap

    labels = []
    path = Path(label_path)
    if not path.exists():
        logger.warning(f"Missing label file: {label_path}")
        return labels
    if label_map is None:
        label_map = LabelMap()
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    logger.warning(f"Malformed label line in {label_path}: {line}")
                    continue
                raw_class_id = int(parts[0])
                class_id = label_map.remap(raw_class_id)
                if class_id == label_map.unknown_id:
                    logger.warning(
                        f"Unknown class id {raw_class_id} in {label_path}; skipping box"
                    )
                    continue
                labels.append(Label(
                    class_id=class_id,
                    cx=float(parts[1]),
                    cy=float(parts[2]),
                    bw=float(parts[3]),
                    bh=float(parts[4]),
                ))
    except Exception as e:
        logger.error(f"Failed to read labels {label_path}: {e}")
    return labels


def load_subset(subset_file: str) -> List[str]:
    """Load the committed benchmark subset list."""
    path = Path(subset_file)
    if not path.exists():
        raise FileNotFoundError(f"Subset file not found: {path}")
    with open(path, "r") as f:
        names = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return names


def build_subset(clips_dir: str, labels_dir: str, subset_size: int, seed: int) -> List[str]:
    """Build a deterministic subset of clips that have matching label files.

    The subset is the alphabetically first ``subset_size`` clips with labels.
    ``seed`` is reserved for reproducibility in downstream processing (torch,
    numpy) and is intentionally not used to shuffle the subset here, so the
    committed ``benchmark_subset.txt`` is stable and reviewable.
    """
    _ = seed  # reserved for downstream RNG seeding
    clips = sorted(Path(clips_dir).glob("*.mp4"))
    paired = []
    for clip in clips:
        label = Path(labels_dir) / clip.with_suffix(".txt").name
        if label.exists():
            paired.append(clip.name)
    return paired[:subset_size]


def extract_frame(clip_path: str, offset_seconds: float = 5.0) -> Tuple[Optional[np.ndarray], int]:
    """Extract a single frame near the trigger time from an MP4 clip.

    The AECS-SDC clips are recorded with a 5-second pre-trigger buffer, so the
    trigger frame is approximately at offset_seconds from the start. If the clip
    is shorter, the last frame is used.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logger.error(f"Could not open clip: {clip_path}")
        return None, -1

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_idx = min(int(offset_seconds * fps), max(total_frames - 1, 0))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        logger.error(f"Could not read frame {target_idx} from {clip_path}")
        return None, -1

    return frame, target_idx


def iter_samples(
    clips_dir: str,
    labels_dir: str,
    subset: List[str],
    offset_seconds: float = 5.0,
    label_map=None,
) -> Iterator[Sample]:
    """Yield benchmark samples for each clip in the subset."""
    from aecs_sdc.label_map import LabelMap

    if label_map is None:
        label_map = LabelMap()
    for name in subset:
        clip_path = Path(clips_dir) / name
        label_path = Path(labels_dir) / Path(name).with_suffix(".txt").name
        frame, frame_idx = extract_frame(str(clip_path), offset_seconds)
        if frame is None:
            logger.warning(f"Skipping {name}: frame extraction failed")
            continue
        labels = read_yolo_labels(str(label_path), label_map)
        yield Sample(
            clip_name=name,
            clip_path=str(clip_path),
            label_path=str(label_path),
            frame=frame,
            frame_idx=frame_idx,
            labels=labels,
        )
