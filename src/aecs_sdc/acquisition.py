"""Acquisition functions for active-learning frame selection.

All scoring is done against the TRAIN_POOL only.  The held-out TEST set is never
used for selection.
"""

import random
from typing import Dict, List, Optional

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
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    hybrid_uncertainty_mode: str = "disagreement",
    diversity_weight: float = 0.5,
) -> List[str]:
    """Select a training subset of size ``k`` using ``mode``.

    Modes:
        * ``random``        -- uniform seed-controlled sampling
        * ``disagreement``  -- top-K teacher-student disagreement
        * ``entropy``       -- top-K student entropy
        * ``hybrid``        -- normalized disagreement/entropy + k-center-greedy
                               diversity on image embeddings
    """
    if mode == "random":
        return select_random_k(train_ids, k, seed)
    if mode == "hybrid":
        if embeddings is None:
            raise ValueError("hybrid mode requires embeddings")
        uncertainty = score_frames(
            teacher_results,
            student_results,
            mode=hybrid_uncertainty_mode,
        )
        uncertainty = {fid: s for fid, s in uncertainty.items() if fid in train_ids}
        embeddings = {fid: v for fid, v in embeddings.items() if fid in train_ids}
        return hybrid_selection(
            frame_ids=train_ids,
            uncertainty_scores=uncertainty,
            embeddings=embeddings,
            k=k,
            diversity_weight=diversity_weight,
        )
    scores = score_frames(teacher_results, student_results, mode)
    # Ensure we only select from the provided train IDs.
    scores = {fid: s for fid, s in scores.items() if fid in train_ids}
    return select_top_k(scores, k)


def hybrid_selection(
    frame_ids: List[str],
    uncertainty_scores: Dict[str, float],
    embeddings: Dict[str, np.ndarray],
    k: int,
    diversity_weight: float = 0.5,
) -> List[str]:
    """Select ``k`` frames maximizing normalized uncertainty + diversity.

    Diversity is enforced by weighted k-center-greedy on the provided image
    embeddings.  Both the uncertainty scores and the min-distance-to-selected
    values are min-max normalized to [0, 1] per iteration so the weighted sum is
    scale-free.  Ties are broken first by distance (prefer spread), then by
    deterministic frame_id order.
    """
    if k <= 0:
        return []
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError(f"diversity_weight must be in [0, 1], got {diversity_weight}")

    missing = [
        fid
        for fid in frame_ids
        if fid not in uncertainty_scores or fid not in embeddings
    ]
    if missing:
        raise KeyError(f"Missing uncertainty/embedding for {len(missing)} frame(s): {missing[:5]}")

    available = frame_ids[:]
    if k >= len(available):
        return available

    u_vals = np.array([uncertainty_scores[fid] for fid in available], dtype=float)
    u_min, u_max = u_vals.min(), u_vals.max()
    u_range = u_max - u_min
    u_norm = (
        (u_vals - u_min) / u_range
        if u_range > 1e-12
        else np.zeros_like(u_vals)
    )

    # Map frame id -> index in available[] for fast vectorized distance compute.
    id_to_index = {fid: i for i, fid in enumerate(available)}
    emb_matrix = np.stack([embeddings[fid] for fid in available])

    selected: List[str] = []
    remaining = set(available)

    # Seed with the highest-uncertainty frame.
    seed_idx = int(np.argmax(u_norm))
    seed_id = available[seed_idx]
    selected.append(seed_id)
    remaining.remove(seed_id)

    while remaining and len(selected) < k:
        # Deterministic iteration order: sorted frame ids.
        cand_ids = sorted(remaining)
        cand_indices = np.array([id_to_index[fid] for fid in cand_ids])
        cand_embs = emb_matrix[cand_indices]
        sel_indices = np.array([id_to_index[fid] for fid in selected])
        sel_embs = emb_matrix[sel_indices]
        # Euclidean distance on L2-normalized vectors is monotonic with cosine distance.
        dists = np.linalg.norm(cand_embs[:, None, :] - sel_embs[None, :, :], axis=2)
        min_dists = dists.min(axis=1)

        d_min, d_max = min_dists.min(), min_dists.max()
        d_range = d_max - d_min
        d_norm = (
            (min_dists - d_min) / d_range
            if d_range > 1e-12
            else np.zeros_like(min_dists)
        )

        cand_u_norm = u_norm[cand_indices]
        scores = (1.0 - diversity_weight) * cand_u_norm + diversity_weight * d_norm

        # Tie-break: prefer larger diversity distance, then deterministic id order.
        best_score = float(scores.max())
        top_mask = scores == best_score
        top_indices = np.where(top_mask)[0]
        if len(top_indices) > 1:
            best_dist = float(d_norm[top_indices].max())
            top_indices = top_indices[d_norm[top_indices] == best_dist]
        best_local_idx = int(top_indices[0])
        best_id = cand_ids[best_local_idx]
        selected.append(best_id)
        remaining.remove(best_id)

    return selected


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
