"""Acquisition functions for active-learning frame selection.

All scoring is done against the TRAIN_POOL only.  The held-out TEST set is never
used for selection.
"""

import random
from typing import Dict, List

import numpy as np

from aecs_sdc.disagreement import _box_disagreement


def disagreement_score(teacher_result: dict, student_result: dict) -> float:
    """Corrected disagreement acquisition: (1 - mean best IoU) + confidence margin.

    High score means the student fails to spatially reproduce the teacher's
    boxes and/or disagrees on the top confidence.
    """
    iou_disagree = _box_disagreement(
        teacher_result.get("detections", []),
        student_result.get("detections", []),
    )
    margin = abs(
        float(teacher_result.get("top_confidence", 0.0))
        - float(student_result.get("top_confidence", 0.0))
    )
    return float(np.clip(iou_disagree + margin, 0.0, 1.0))


def entropy_score(student_result: dict) -> float:
    """Student prediction-entropy acquisition signal (already normalized)."""
    return float(student_result.get("entropy", 0.0))


def score_frames(
    teacher_results: Dict[str, dict],
    student_results: Dict[str, dict],
    mode: str,
) -> Dict[str, float]:
    """Return ``{frame_id: acquisition_score}`` for a given mode.

    Modes:
        * ``disagreement``  -- corrected teacher-student disagreement
        * ``entropy``       -- student prediction entropy
        * ``random``        -- uniform (score is ignored by ``select_random_k``)
    """
    scores: Dict[str, float] = {}
    common_ids = set(teacher_results.keys()) & set(student_results.keys())
    for frame_id in common_ids:
        if mode == "disagreement":
            scores[frame_id] = disagreement_score(
                teacher_results[frame_id], student_results[frame_id]
            )
        elif mode == "entropy":
            scores[frame_id] = entropy_score(student_results[frame_id])
        elif mode == "random":
            scores[frame_id] = 0.0
        else:
            raise ValueError(f"Unknown acquisition mode: {mode}")
    return scores


def select_top_k(scores: Dict[str, float], k: int) -> List[str]:
    """Return frame IDs with the top ``k`` scores.

    Ties are broken by sorting on frame_id so the result is deterministic.
    If ``k`` exceeds the number of frames, all frames are returned.
    """
    if k <= 0:
        return []
    sorted_items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [frame_id for frame_id, _ in sorted_items[:k]]


def select_random_k(frame_ids: List[str], k: int, seed: int) -> List[str]:
    """Return ``k`` uniformly sampled frame IDs; deterministic by ``seed``.

    If ``k`` exceeds the number of frames, all frames are returned.
    """
    if k <= 0:
        return []
    rng = random.Random(seed)
    ids = frame_ids[:]
    rng.shuffle(ids)
    return ids[: min(k, len(ids))]


def select_subset(
    mode: str,
    train_ids: List[str],
    teacher_results: Dict[str, dict],
    student_results: Dict[str, dict],
    k: int,
    seed: int,
) -> List[str]:
    """Select a training subset of size ``k`` using ``mode``.

    For ``random`` the seed controls sampling; for ``disagreement`` and
    ``entropy`` the seed has no effect because selection is deterministic given
    the teacher/student outputs.
    """
    if mode == "random":
        return select_random_k(train_ids, k, seed)
    scores = score_frames(teacher_results, student_results, mode)
    # Ensure we only select from the provided train IDs.
    scores = {fid: s for fid, s in scores.items() if fid in train_ids}
    return select_top_k(scores, k)


def check_separation(
    results: List[dict],
    k: int,
    metric: str = "mAP50",
    baseline_arm: str = "random",
    candidate_arm: str = "harvested",
) -> bool:
    """Return True if ``candidate_arm`` beats ``baseline_arm`` beyond noise at K.

    "Beyond noise" means ``mean(candidate) - mean(baseline) > std(candidate) +
    std(baseline)``.
    """
    cand = [r[metric] for r in results if r.get("k") == k and r.get("arm") == candidate_arm]
    base = [r[metric] for r in results if r.get("k") == k and r.get("arm") == baseline_arm]
    if len(cand) < 2 or len(base) < 2:
        return False
    mean_gap = float(np.mean(cand)) - float(np.mean(base))
    combined_std = float(np.std(cand, ddof=1)) + float(np.std(base, ddof=1))
    return mean_gap > combined_std
