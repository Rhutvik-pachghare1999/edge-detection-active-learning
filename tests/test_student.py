"""Unit tests for the student model wrapper using a mock."""

import numpy as np
import pytest

from aecs_sdc.student import StudentModel


def test_mock_student_returns_valid_shape():
    student = StudentModel.mock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = student.infer(frame)
    assert "top_confidence" in result
    assert 0.0 <= result["top_confidence"] <= 1.0
    assert result["n_detections"] >= 0
    assert isinstance(result["detections"], list)


def test_real_student_requires_existing_model():
    with pytest.raises(FileNotFoundError):
        StudentModel(model_path="nonexistent.onnx")


def test_real_student_accepts_cpu_providers_only():
    with pytest.raises(FileNotFoundError):
        StudentModel(model_path="nonexistent.onnx", providers=["CPUExecutionProvider"])
