"""Unit tests for benchmark configuration loading."""

import pytest
from pathlib import Path

from aecs_sdc.config import Config


def test_config_loads_default_yaml():
    root = Path(__file__).resolve().parent.parent
    cfg = Config.from_yaml(root / "configs" / "benchmark.yaml")
    assert cfg.benchmark.seed == 42
    assert cfg.models.student.endswith("yolov8n.onnx")
    assert cfg.disagreement.mode == "combined"


def test_config_resolve_paths():
    root = Path(__file__).resolve().parent.parent
    cfg = Config.from_yaml(root / "configs" / "benchmark.yaml")
    cfg.resolve_paths(root)
    assert str(Path(cfg.benchmark.clips_dir).resolve()) == str((root / "clips").resolve())
    assert Path(cfg.models.student).is_absolute()
