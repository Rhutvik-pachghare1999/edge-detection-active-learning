"""Unit tests for disagreement metrics."""

import pytest

from aecs_sdc.disagreement import compute_disagreement, compute_student_teacher_agreement, _iou


def _box(label, cx, cy, bw, bh, score=0.9):
    return {"label": label, "cx": cx, "cy": cy, "bw": bw, "bh": bh, "score": score}


def test_iou_identical_boxes():
    box = _box(0, 0.5, 0.5, 0.2, 0.2)
    assert _iou(box, box) == pytest.approx(1.0, 1e-6)


def test_iou_non_overlapping_boxes():
    a = _box(0, 0.1, 0.1, 0.1, 0.1)
    b = _box(0, 0.9, 0.9, 0.1, 0.1)
    assert _iou(a, b) == pytest.approx(0.0, 1e-6)


def test_perfect_agreement():
    teacher = {"top_confidence": 0.9, "top_class_id": 0, "detections": [_box(0, 0.5, 0.5, 0.2, 0.2)]}
    student = {"top_confidence": 0.9, "top_class_id": 0, "entropy": 0.0,
               "detections": [_box(0, 0.5, 0.5, 0.2, 0.2)]}
    weights = {"margin": 0.2, "entropy": 0.2, "class_mismatch": 0.1, "iou": 0.5}
    d = compute_disagreement(teacher, student, weights)
    assert d.margin == 0.0
    assert d.class_mismatch == 0.0
    assert d.iou == pytest.approx(0.0, 1e-5)
    assert d.combined == pytest.approx(0.0, 1e-5)


def test_class_mismatch_only():
    teacher = {"top_confidence": 0.9, "top_class_id": 0, "detections": []}
    student = {"top_confidence": 0.9, "top_class_id": 1, "entropy": 0.0, "detections": []}
    weights = {"margin": 0.2, "entropy": 0.2, "class_mismatch": 0.1, "iou": 0.5}
    d = compute_disagreement(teacher, student, weights)
    assert d.margin == 0.0
    assert d.class_mismatch == 1.0
    assert d.iou == 0.0  # no teacher detections -> no IoU disagreement
    assert d.combined == pytest.approx(0.1, 0.01)


def test_margin_only():
    teacher = {"top_confidence": 0.9, "top_class_id": 0, "detections": []}
    student = {"top_confidence": 0.6, "top_class_id": 0, "entropy": 0.0, "detections": []}
    weights = {"margin": 1.0, "entropy": 0.0, "class_mismatch": 0.0, "iou": 0.0}
    d = compute_disagreement(teacher, student, weights)
    assert d.margin == 0.3
    assert d.combined == 0.3


def test_iou_disagreement_high():
    teacher = {"top_confidence": 0.9, "top_class_id": 0,
               "detections": [_box(0, 0.5, 0.5, 0.2, 0.2)]}
    student = {"top_confidence": 0.9, "top_class_id": 0, "entropy": 0.0,
               "detections": [_box(0, 0.9, 0.9, 0.1, 0.1)]}
    weights = {"margin": 0.0, "entropy": 0.0, "class_mismatch": 0.0, "iou": 1.0}
    d = compute_disagreement(teacher, student, weights)
    assert d.iou > 0.9
    assert d.combined > 0.9


def test_student_teacher_agreement_perfect_match():
    """Agreement treats teacher boxes as pseudo-ground-truth; not human truth."""
    teacher = [_box(0, 0.5, 0.5, 0.2, 0.2)]
    student = [_box(0, 0.5, 0.5, 0.2, 0.2)]
    m = compute_student_teacher_agreement(student, teacher, iou_threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["true_positives"] == 1


def test_student_teacher_agreement_empty_inputs():
    m = compute_student_teacher_agreement([], [])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
