"""Unit tests for the student model wrapper using a mock."""

import numpy as np
import pytest

from edgeal.student import StudentModel, _instance_uncertainties, _binary_entropy


def test_mock_student_returns_valid_shape():
    student = StudentModel.mock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = student.infer(frame)
    assert "top_confidence" in result
    assert 0.0 <= result["top_confidence"] <= 1.0
    assert result["n_detections"] >= 0
    assert isinstance(result["detections"], list)
    # New fields
    assert "image_uncertainty" in result
    assert "instance_uncertainties" in result


def test_real_student_requires_existing_model():
    with pytest.raises(FileNotFoundError):
        StudentModel(model_path="nonexistent.onnx")


def test_real_student_accepts_cpu_providers_only():
    with pytest.raises(FileNotFoundError):
        StudentModel(model_path="nonexistent.onnx", providers=["CPUExecutionProvider"])


def test_binary_entropy():
    """Test binary entropy function."""
    # p=0.5 -> max entropy = log(2)
    assert _binary_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2), rel=1e-6)
    # p=0 or 1 -> entropy = 0
    assert _binary_entropy(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)
    assert _binary_entropy(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-6)
    # p=0.1 and p=0.9 should give same entropy
    e1 = _binary_entropy(np.array([0.1]))[0]
    e2 = _binary_entropy(np.array([0.9]))[0]
    assert e1 == pytest.approx(e2, rel=1e-6)


def test_instance_uncertainties_uniform():
    """Test instance uncertainty for uniform distribution (max entropy)."""
    class_scores = np.full(80, 0.5, dtype=np.float32)
    unc = _instance_uncertainties(class_scores)
    assert unc["entropy"] == pytest.approx(1.0, abs=1e-6)  # normalized to [0,1]
    assert unc["least_confidence"] == pytest.approx(0.5, abs=1e-6)
    assert unc["margin"] == pytest.approx(0.0, abs=1e-6)
    assert unc["max_confidence"] == pytest.approx(0.5, abs=1e-6)


def test_instance_uncertainties_confident():
    """Test instance uncertainty for confident prediction (low entropy)."""
    class_scores = np.full(80, 0.01, dtype=np.float32)
    class_scores[0] = 0.95
    unc = _instance_uncertainties(class_scores)
    assert unc["entropy"] < 0.1  # very low normalized entropy
    assert unc["least_confidence"] == pytest.approx(0.05, abs=1e-2)
    assert unc["margin"] > 0.9  # large margin between top-1 and top-2
    assert unc["max_confidence"] == pytest.approx(0.95, abs=1e-6)


def test_instance_uncertainties_two_class():
    """Test instance uncertainty for two-class competition (medium entropy)."""
    class_scores = np.full(80, 0.01, dtype=np.float32)
    class_scores[0] = 0.6
    class_scores[1] = 0.4
    unc = _instance_uncertainties(class_scores)
    # Should have moderate entropy
    assert 0.1 < unc["entropy"] < 0.8
    assert unc["least_confidence"] == pytest.approx(0.4, abs=1e-2)
    assert unc["margin"] == pytest.approx(0.2, abs=1e-2)
