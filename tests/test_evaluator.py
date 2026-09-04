"""Unit tests for the evaluator."""

import json
import tempfile
from pathlib import Path

import pytest

from aecs_sdc.config import BenchmarkConfig, Config
from aecs_sdc.evaluator import Evaluator


def _record(combined, harvested, agreement=None, **kwargs):
    rec = {
        "clip_name": kwargs.get("clip_name", "test.mp4"),
        "frame_idx": kwargs.get("frame_idx", 0),
        "disagreement": {
            "combined": combined,
            "margin": 0.1,
            "entropy": 0.2,
            "class_mismatch": 0.0,
            "iou": kwargs.get("iou", 0.1),
        },
        "harvested": harvested,
        "harvest_reason": kwargs.get("harvest_reason", "threshold" if harvested else "none"),
        "labels": [],
        "student_detections": [],
        "teacher_detections": [],
    }
    if agreement is not None:
        rec["student_vs_teacher_agreement"] = agreement
    return rec


def test_evaluator_writes_summary_and_details():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            benchmark=BenchmarkConfig(
                results_dir=tmp,
                summary_file=str(Path(tmp) / "summary.json"),
                details_file=str(Path(tmp) / "details.json"),
            )
        )
        ev = Evaluator(cfg)
        ev.start()
        ev.add(_record(0.5, True, agreement={
            "true_positives": 1,
            "ground_truth_count": 2,
            "detection_count": 3,
            "precision": 0.33,
            "recall": 0.5,
        }))
        summary = ev.finish()

        assert summary["meta"]["total_frames"] == 1
        assert summary["meta"]["harvested_count"] == 1
        assert Path(cfg.benchmark.summary_file).exists()
        assert Path(cfg.benchmark.details_file).exists()
        with open(cfg.benchmark.summary_file) as f:
            compact = json.load(f)
        assert "records" not in compact
        assert "disagreement" in compact
        assert "distribution" in compact["disagreement"]
        assert "student_vs_teacher_agreement" in compact


def test_evaluator_disagreement_distribution():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            benchmark=BenchmarkConfig(
                results_dir=tmp,
                summary_file=str(Path(tmp) / "summary.json"),
                details_file=str(Path(tmp) / "details.json"),
            )
        )
        ev = Evaluator(cfg)
        ev.start()
        for i in range(5):
            ev.add(_record(0.1 * i, i >= 2, agreement={
                "true_positives": 0,
                "ground_truth_count": 1,
                "detection_count": 1,
                "precision": 0.0,
                "recall": 0.0,
            }))
        summary = ev.finish()
        dist = summary["disagreement"]["distribution"]
        assert len(dist["top_harvested_frames"]) == 3
        assert dist["top_harvested_frames"][0]["combined"] == pytest.approx(0.4, 0.01)
        assert len(dist["bottom_non_harvested_frames"]) == 2
        assert dist["bottom_non_harvested_frames"][0]["combined"] == pytest.approx(0.0, 0.01)


def test_evaluator_missing_optional_fields_is_robust():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            benchmark=BenchmarkConfig(
                results_dir=tmp,
                summary_file=str(Path(tmp) / "summary.json"),
                details_file=str(Path(tmp) / "details.json"),
            )
        )
        ev = Evaluator(cfg)
        ev.start()
        ev.add(_record(0.5, True, agreement={
            "true_positives": 0,
            "ground_truth_count": 0,
            "detection_count": 0,
            "precision": 0.0,
            "recall": 0.0,
        }))
        summary = ev.finish()
        top = summary["disagreement"]["distribution"]["top_harvested_frames"][0]
        assert top["frame_idx"] == 0
        assert top["harvest_reason"] == "threshold"


def test_evaluator_empty_agreement_is_robust():
    """Bug 9: overall agreement must not crash when no records carry agreement metrics."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            benchmark=BenchmarkConfig(
                results_dir=tmp,
                summary_file=str(Path(tmp) / "summary.json"),
                details_file=str(Path(tmp) / "details.json"),
            )
        )
        ev = Evaluator(cfg)
        ev.start()
        ev.add(_record(0.5, True))
        summary = ev.finish()
        assert summary["student_vs_teacher_agreement"]["precision"] is None
        assert summary["student_vs_teacher_agreement"]["recall"] is None
        assert "No human ground truth" in summary["student_vs_teacher_agreement"]["note"]
