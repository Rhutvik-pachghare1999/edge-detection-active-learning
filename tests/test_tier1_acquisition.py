"""Tests for acquisition / active-learning subset selection."""

import numpy as np
import pytest

from aecs_sdc.acquisition import (
    check_separation,
    disagreement_score,
    entropy_score,
    hybrid_selection,
    select_random_k,
    select_subset,
    select_top_k,
)


def _det(label, cx=0.5, cy=0.5, bw=0.2, bh=0.2, score=0.9):
    return {"label": label, "cx": cx, "cy": cy, "bw": bw, "bh": bh, "score": score}


def _teacher_result(detections, top_confidence=0.9, top_class_id=0):
    return {
        "detections": detections,
        "top_confidence": top_confidence,
        "top_class_id": top_class_id,
    }


def _student_result(detections, top_confidence=0.9, top_class_id=0, entropy=0.0):
    return {
        "detections": detections,
        "top_confidence": top_confidence,
        "top_class_id": top_class_id,
        "entropy": entropy,
    }


def test_select_top_k_returns_exactly_k_and_sorted():
    scores = {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.9, "e": 0.2}
    selected = select_top_k(scores, k=3)
    assert len(selected) == 3
    assert selected == ["d", "b", "c"]


def test_select_random_k_is_deterministic_with_seed():
    ids = list(map(str, range(100)))
    a = select_random_k(ids, k=10, seed=42)
    b = select_random_k(ids, k=10, seed=42)
    c = select_random_k(ids, k=10, seed=43)
    assert a == b
    assert len(a) == 10
    assert set(a).issubset(set(ids))
    assert a != c


def test_select_subset_harvested_uses_disagreement():
    train_ids = ["1", "2", "3", "4"]
    teacher = {
        "1": _teacher_result([_det(0, cx=0.5)]),
        "2": _teacher_result([_det(0, cx=0.9)]),
        "3": _teacher_result([_det(1, cx=0.5)]),
        "4": _teacher_result([_det(0, cx=0.5)]),
    }
    student = {
        "1": _student_result([_det(0, cx=0.5)], entropy=0.1),
        "2": _student_result([_det(0, cx=0.1)], entropy=0.5),
        "3": _student_result([_det(0, cx=0.5)], entropy=0.2),
        "4": _student_result([_det(0, cx=0.5)], entropy=0.3),
    }
    selected = select_subset(
        mode="disagreement",
        train_ids=train_ids,
        teacher_results=teacher,
        student_results=student,
        k=2,
        seed=0,
    )
    assert len(selected) == 2
    # Image 2 has high IoU disagreement; image 3 has class mismatch; image 4 has entropy.
    # With default weights, disagreement dominates and picks 2 and 3.
    assert selected[0] == "2"
    assert "3" in selected


def test_select_subset_random_ignores_scores():
    train_ids = [str(i) for i in range(20)]
    teacher = {im: _teacher_result([]) for im in train_ids}
    student = {im: _student_result([], entropy=float(i) / 20) for i, im in enumerate(train_ids)}
    selected_a = select_subset(
        mode="random",
        train_ids=train_ids,
        teacher_results=teacher,
        student_results=student,
        k=5,
        seed=42,
    )
    selected_b = select_subset(
        mode="random",
        train_ids=train_ids,
        teacher_results=teacher,
        student_results=student,
        k=5,
        seed=42,
    )
    assert len(selected_a) == 5
    assert selected_a == selected_b


def test_select_subset_entropy_uses_student_entropy():
    train_ids = ["1", "2", "3", "4"]
    teacher = {im: _teacher_result([]) for im in train_ids}
    student = {
        "1": _student_result([], entropy=0.1),
        "2": _student_result([], entropy=0.4),
        "3": _student_result([], entropy=0.2),
        "4": _student_result([], entropy=0.9),
    }
    selected = select_subset(
        mode="entropy",
        train_ids=train_ids,
        teacher_results=teacher,
        student_results=student,
        k=2,
        seed=0,
    )
    assert selected == ["4", "2"]


def test_select_subset_k_larger_than_pool_returns_all():
    ids = ["1", "2", "3"]
    teacher = {im: _teacher_result([]) for im in ids}
    student = {im: _student_result([], entropy=0.0) for im in ids}
    selected = select_subset(
        mode="random",
        train_ids=ids,
        teacher_results=teacher,
        student_results=student,
        k=10,
        seed=0,
    )
    assert len(selected) == 3


def test_check_separation_detects_clear_signal():
    results = [
        {"arm": "harvested", "k": 250, "seed": 42, "mAP50": 0.35},
        {"arm": "harvested", "k": 250, "seed": 43, "mAP50": 0.36},
        {"arm": "harvested", "k": 250, "seed": 44, "mAP50": 0.34},
        {"arm": "random", "k": 250, "seed": 42, "mAP50": 0.25},
        {"arm": "random", "k": 250, "seed": 43, "mAP50": 0.26},
        {"arm": "random", "k": 250, "seed": 44, "mAP50": 0.25},
    ]
    assert check_separation(results, k=250, metric="mAP50") is True


def test_check_separation_rejects_noise():
    results = [
        {"arm": "harvested", "k": 250, "seed": 42, "mAP50": 0.30},
        {"arm": "harvested", "k": 250, "seed": 43, "mAP50": 0.28},
        {"arm": "harvested", "k": 250, "seed": 44, "mAP50": 0.32},
        {"arm": "random", "k": 250, "seed": 42, "mAP50": 0.29},
        {"arm": "random", "k": 250, "seed": 43, "mAP50": 0.31},
        {"arm": "random", "k": 250, "seed": 44, "mAP50": 0.27},
    ]
    assert check_separation(results, k=250, metric="mAP50") is False


def test_disagreement_score_is_non_negative():
    teacher = _teacher_result([_det(0)])
    student = _student_result([_det(0)])
    score = disagreement_score(teacher, student)
    assert score >= 0.0


def test_entropy_score_matches_top_confidence():
    low_conf = _student_result([_det(0, score=0.55)], top_confidence=0.55, entropy=0.5)
    high_conf = _student_result([_det(0, score=0.95)], top_confidence=0.95, entropy=0.1)
    assert entropy_score(low_conf) > entropy_score(high_conf)


def test_hybrid_selection_spreads_over_clusters():
    """When three identical high-uncertainty clones exist, hybrid must spread."""
    frame_ids = ["a", "b", "c", "d"]
    # a, b, c are identical high-uncertainty points; d is far away and lower uncertainty.
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([1.0, 0.0]),
        "c": np.array([1.0, 0.0]),
        "d": np.array([0.0, 1.0]),
    }
    uncertainty = {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.4}
    selected = hybrid_selection(frame_ids, uncertainty, embeddings, k=2, diversity_weight=0.5)
    # First pick is the highest-uncertainty clone (a, then by tie-break id).
    assert selected[0] == "a"
    # Second pick must be the distant point d, not another clone.
    assert selected[1] == "d"


def test_hybrid_selection_returns_all_when_k_too_large():
    frame_ids = ["x", "y"]
    embeddings = {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0])}
    uncertainty = {"x": 0.2, "y": 0.8}
    selected = hybrid_selection(frame_ids, uncertainty, embeddings, k=10)
    assert len(selected) == 2
    assert set(selected) == {"x", "y"}


def test_hybrid_selection_rejects_missing_embeddings():
    frame_ids = ["a", "b"]
    uncertainty = {"a": 0.5, "b": 0.5}
    embeddings = {"a": np.array([1.0, 0.0])}
    with pytest.raises(KeyError):
        hybrid_selection(frame_ids, uncertainty, embeddings, k=2)


def test_select_subset_hybrid_requires_embeddings():
    train_ids = ["1", "2", "3"]
    teacher = {im: _teacher_result([]) for im in train_ids}
    student = {im: _student_result([], entropy=0.5) for im in train_ids}
    with pytest.raises(ValueError, match="hybrid mode requires embeddings"):
        select_subset(
            mode="hybrid",
            train_ids=train_ids,
            teacher_results=teacher,
            student_results=student,
            k=2,
            seed=0,
        )


def test_select_subset_hybrid_with_embeddings_runs():
    train_ids = ["1", "2", "3"]
    teacher = {im: _teacher_result([]) for im in train_ids}
    student = {im: _student_result([], entropy=0.5) for im in train_ids}
    embeddings = {
        "1": np.array([1.0, 0.0]),
        "2": np.array([0.0, 1.0]),
        "3": np.array([0.0, 0.0]),
    }
    selected = select_subset(
        mode="hybrid",
        train_ids=train_ids,
        teacher_results=teacher,
        student_results=student,
        k=2,
        seed=0,
        embeddings=embeddings,
        hybrid_uncertainty_mode="entropy",
        diversity_weight=0.5,
    )
    assert len(selected) == 2
    assert set(selected).issubset(set(train_ids))
