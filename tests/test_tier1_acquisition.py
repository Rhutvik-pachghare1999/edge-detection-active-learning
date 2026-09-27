"""Tests for acquisition / active-learning subset selection."""

import numpy as np
import pytest

from edgeal.acquisition import (
    check_separation,
    disagreement_score,
    entropy_score,
    image_entropy_score,
    image_max_entropy_score,
    image_least_confidence_score,
    image_max_least_confidence_score,
    image_margin_score,
    image_min_margin_score,
    hybrid_selection,
    select_random_k,
    select_subset,
    select_top_k,
)
from edgeal.student import _instance_uncertainties


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


def _student_result_with_image_unc(detections, top_confidence=0.9, top_class_id=0, entropy=0.0,
                                   mean_entropy=0.5, max_entropy=0.8,
                                   mean_least_confidence=0.4, max_least_confidence=0.7,
                                   mean_margin=0.2, min_margin=0.1, detection_count=2):
    """Student result with full image_uncertainty dict for testing new modes."""
    base = _student_result(detections, top_confidence, top_class_id, entropy)
    base["image_uncertainty"] = {
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "mean_least_confidence": mean_least_confidence,
        "max_least_confidence": max_least_confidence,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
        "detection_count": detection_count,
    }
    return base


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


# New tests for image-level uncertainty acquisition functions

def test_instance_uncertainties_high_entropy():
    """Test instance uncertainty for uniform class distribution (high entropy)."""
    # Uniform distribution: all classes have p=0.5 (sigmoid output)
    class_scores = np.full(80, 0.5, dtype=np.float32)
    unc = _instance_uncertainties(class_scores)
    # Max binary entropy per class is log(2) ~ 0.693, so max total = 80 * log(2)
    max_entropy = 80 * np.log(2)
    assert unc["entropy"] == pytest.approx(1.0, abs=1e-6)  # normalized
    assert unc["least_confidence"] == pytest.approx(0.5, abs=1e-6)
    assert unc["margin"] == pytest.approx(0.0, abs=1e-6)


def test_instance_uncertainties_low_entropy():
    """Test instance uncertainty for confident prediction (low entropy)."""
    # One class has high confidence, rest near 0
    class_scores = np.full(80, 0.01, dtype=np.float32)
    class_scores[0] = 0.95
    unc = _instance_uncertainties(class_scores)
    assert unc["entropy"] < 0.1  # very low entropy
    assert unc["least_confidence"] == pytest.approx(0.05, abs=1e-2)
    assert unc["margin"] > 0.9  # large margin


def test_image_entropy_score_uses_image_uncertainty():
    """Test image_entropy_score reads from image_uncertainty dict."""
    student = _student_result_with_image_unc([], mean_entropy=0.6, max_entropy=0.9)
    assert image_entropy_score(student) == pytest.approx(0.6)
    assert image_max_entropy_score(student) == pytest.approx(0.9)


def test_image_least_confidence_score():
    """Test least_confidence acquisition modes."""
    student = _student_result_with_image_unc(
        [], mean_least_confidence=0.4, max_least_confidence=0.8
    )
    assert image_least_confidence_score(student) == pytest.approx(0.4)
    assert image_max_least_confidence_score(student) == pytest.approx(0.8)


def test_image_margin_score():
    """Test margin acquisition modes (inverted: lower margin = higher uncertainty)."""
    student = _student_result_with_image_unc(
        [], mean_margin=0.2, min_margin=0.05
    )
    # margin score = 1 - mean_margin = 0.8 (high uncertainty)
    assert image_margin_score(student) == pytest.approx(0.8)
    # min_margin score = 1 - min_margin = 0.95 (very high uncertainty)
    assert image_min_margin_score(student) == pytest.approx(0.95)


def test_score_frames_new_modes():
    """Test all new acquisition modes in score_frames."""
    train_ids = ["1", "2"]
    teacher = {im: _teacher_result([]) for im in train_ids}
    student = {
        "1": _student_result_with_image_unc(
            [], mean_entropy=0.3, max_entropy=0.5,
            mean_least_confidence=0.2, max_least_confidence=0.6,
            mean_margin=0.1, min_margin=0.05
        ),
        "2": _student_result_with_image_unc(
            [], mean_entropy=0.7, max_entropy=0.9,
            mean_least_confidence=0.5, max_least_confidence=0.8,
            mean_margin=0.3, min_margin=0.1
        ),
    }
    
    # Test all new modes
    # Note: margin score = 1 - mean_margin, so LOWER mean_margin = HIGHER score = MORE uncertain
    # Image "1": mean_margin=0.1 -> score=0.9 (high uncertainty)
    # Image "2": mean_margin=0.3 -> score=0.7 (lower uncertainty)
    # So "1" should be selected first for margin mode
    modes_and_expected = [
        ("image_entropy", ["2", "1"]),
        ("image_max_entropy", ["2", "1"]),
        ("least_confidence", ["2", "1"]),
        ("max_least_confidence", ["2", "1"]),
        ("margin", ["1", "2"]),  # 1 - mean_margin: "1" has 0.9, "2" has 0.7
        ("min_margin", ["1", "2"]),  # 1 - min_margin: "1" has 0.95, "2" has 0.9
    ]
    
    for mode, expected in modes_and_expected:
        scores = select_subset(
            mode=mode,
            train_ids=train_ids,
            teacher_results=teacher,
            student_results=student,
            k=2,
            seed=0,
        )
        assert scores == expected, f"Mode {mode} failed: got {scores}"


def test_select_subset_new_modes():
    """Test select_subset with all new uncertainty modes."""
    train_ids = ["1", "2", "3", "4"]
    teacher = {im: _teacher_result([]) for im in train_ids}
    student = {
        "1": _student_result_with_image_unc([], mean_entropy=0.1),
        "2": _student_result_with_image_unc([], mean_entropy=0.4),
        "3": _student_result_with_image_unc([], mean_entropy=0.2),
        "4": _student_result_with_image_unc([], mean_entropy=0.9),
    }
    
    # image_entropy should pick highest mean_entropy
    selected = select_subset(
        mode="image_entropy",
        train_ids=train_ids,
        teacher_results=teacher,
        student_results=student,
        k=2,
        seed=0,
    )
    assert selected == ["4", "2"]
    
    # least_confidence should pick highest mean_least_confidence
    student_lc = {
        "1": _student_result_with_image_unc([], mean_least_confidence=0.1),
        "2": _student_result_with_image_unc([], mean_least_confidence=0.5),
        "3": _student_result_with_image_unc([], mean_least_confidence=0.2),
        "4": _student_result_with_image_unc([], mean_least_confidence=0.8),
    }
    selected = select_subset(
        mode="least_confidence",
        train_ids=train_ids,
        teacher_results=teacher,
        student_results=student_lc,
        k=2,
        seed=0,
    )
    assert selected == ["4", "2"]
