"""Unit tests for error handling and edge cases."""

from pathlib import Path

import pytest

from aecs_sdc.config import Config
from aecs_sdc.dataset import load_subset


def test_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("/this/does/not/exist.yaml")


def test_load_subset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_subset("/this/does/not/exist.txt")


def test_config_from_dict_defaults():
    cfg = Config.from_dict({})
    assert cfg.benchmark.seed == 42
    assert cfg.disagreement.mode == "combined"
    assert cfg.harvest.mode == "threshold"


def test_config_validates_label_map_path():
    root = Path(__file__).resolve().parent.parent
    cfg = Config.from_yaml(root / "configs" / "benchmark.yaml")
    assert cfg.benchmark.label_map.endswith("label_map.yaml")
    assert Path(cfg.benchmark.label_map).name == "label_map.yaml"


def test_config_resolve_paths_makes_results_absolute():
    root = Path(__file__).resolve().parent.parent
    cfg = Config.from_yaml(root / "configs" / "benchmark.yaml")
    cfg.resolve_paths(root)
    assert Path(cfg.benchmark.results_dir).is_absolute()
    assert Path(cfg.benchmark.summary_file).is_absolute()
    assert Path(cfg.benchmark.details_file).is_absolute()
