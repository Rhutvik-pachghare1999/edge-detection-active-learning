"""Disagreement metrics for teacher-student detection pairs.

This module replaces the single confidence-gap delta with a configurable set
of disagreement signals:

* margin        — |c_teacher - c_student|
* entropy       — normalized entropy of the student's class distribution
* class_mismatch— 1 if the top-1 predicted classes differ, else 0
* combined      — weighted combination of the above

All scores are normalized to [0, 1].
"""

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class DisagreementScores:
    margin: float
    entropy: float
    class_mismatch: float
    iou: float
    combined: float


def _iou(box_a: Dict, box_b: Dict) -> float:
    """Compute IoU between two YOLO-format boxes."""
    ax1 = box_a["cx"] - box_a["bw"] / 2
    ay1 = box_a["cy"] - box_a["bh"] / 2
    ax2 = box_a["cx"] + box_a["bw"] / 2
    ay2 = box_a["cy"] + box_a["bh"] / 2

    bx1 = box_b["cx"] - box_b["bw"] / 2
    by1 = box_b["cy"] - box_b["bh"] / 2
    bx2 = box_b["cx"] + box_b["bw"] / 2
    by2 = box_b["cy"] + box_b["bh"] / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = box_a["bw"] * box_a["bh"]
    area_b = box_b["bw"] * box_b["bh"]
    union = area_a + area_b - inter_area
    return inter_area / (union + 1e-10)


def _box_disagreement(teacher_detections: List[Dict], student_detections: List[Dict]) -> float:
    """Box-level disagreement: 1 - mean best IoU for each teacher detection.

    For every teacher box we look for the best same-class student box using IoU.
    The disagreement is high when the student fails to reproduce the teacher's
    spatial predictions. If the teacher reports no detections there is nothing
    to disagree with, so disagreement is 0.0.
    """
    if not teacher_detections:
        return 0.0

    best_ious = []
    for gt in teacher_detections:
        gt_label = int(gt.get("label", -1))
        best_iou = 0.0
        for det in student_detections:
            if int(det.get("label", -1)) != gt_label:
                continue
            iou = _iou(det, gt)
            if iou > best_iou:
                best_iou = iou
        best_ious.append(best_iou)

    mean_iou = float(np.mean(best_ious)) if best_ious else 0.0
    return float(np.clip(1.0 - mean_iou, 0.0, 1.0))


def compute_disagreement(
    teacher_result: Dict,
    student_result: Dict,
    weights: Dict[str, float],
) -> DisagreementScores:
    """Compute disagreement scores between teacher and student results.

    Args:
        teacher_result: dict from TeacherModel.infer()
        student_result: dict from StudentModel.infer()
        weights: mapping of component names to weights (must sum to ~1.0)

    Returns:
        DisagreementScores dataclass.
    """
    c_teacher = float(teacher_result.get("top_confidence", 0.0))
    c_student = float(student_result.get("top_confidence", 0.0))
    margin = abs(c_teacher - c_student)

    entropy = float(student_result.get("entropy", 0.0))

    teacher_id = int(teacher_result.get("top_class_id", -1))
    student_id = int(student_result.get("top_class_id", -1))
    class_mismatch = 1.0 if (teacher_id >= 0 and student_id >= 0 and teacher_id != student_id) else 0.0

    # Box-level IoU disagreement between the full detection lists.
    iou = _box_disagreement(
        teacher_result.get("detections", []),
        student_result.get("detections", []),
    )

    # Weighted combination of all signals. Weights should sum to ~1.0.
    combined = (
        weights.get("margin", 0.0) * margin
        + weights.get("entropy", 0.0) * entropy
        + weights.get("class_mismatch", 0.0) * class_mismatch
        + weights.get("iou", 0.0) * iou
    )
    combined = float(np.clip(combined, 0.0, 1.0))

    return DisagreementScores(
        margin=round(margin, 6),
        entropy=round(entropy, 6),
        class_mismatch=round(class_mismatch, 6),
        iou=round(iou, 6),
        combined=round(combined, 6),
    )


def compute_student_teacher_agreement(
    student_detections: List[Dict],
    teacher_labels: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict:
    """Compute per-frame student-vs-teacher pseudo-label agreement.

    This treats the teacher-generated YOLO labels as pseudo-ground-truth and
    measures how many the student recalls and how many of the student's boxes
    are correct. It is NOT accuracy against human ground truth; no human labels
    exist in this repository.
    """
    matched = set()
    true_positives = 0
    for gt in teacher_labels:
        best_iou = 0.0
        best_idx = -1
        for idx, det in enumerate(student_detections):
            if idx in matched:
                continue
            if int(det.get("label", -1)) != int(gt.get("class_id", gt.get("label", -1))):
                continue
            iou = _iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold:
            matched.add(best_idx)
            true_positives += 1

    gt_count = len(teacher_labels)
    det_count = len(student_detections)
    recall = true_positives / gt_count if gt_count > 0 else 0.0
    precision = true_positives / det_count if det_count > 0 else 0.0
    return {
        "true_positives": true_positives,
        "ground_truth_count": gt_count,
        "detection_count": det_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
