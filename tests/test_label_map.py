"""Unit tests for label-to-model class mapping."""

from pathlib import Path

import pytest

from aecs_sdc.label_map import LabelMap


def test_identity_mapping():
    lm = LabelMap.from_yaml("/nonexistent/label_map.yaml")
    assert lm.is_identity()
    assert lm.remap(0) == 0
    assert lm.remap(79) == 79


def test_unknown_id_outside_range():
    lm = LabelMap()
    assert lm.remap(80) == -1
    assert lm.remap(-1) == -1


def test_explicit_mapping_overrides_identity():
    lm = LabelMap(mapping={0: 5, 5: 0}, unknown_id=-1, valid_range=[0, 79])
    assert lm.remap(0) == 5
    assert lm.remap(5) == 0
    assert lm.remap(10) == 10
    assert not lm.is_identity()


def test_validate_class_ids():
    lm = LabelMap()
    summary = lm.validate({0, 5, 79, 80, -1})
    assert summary["total_unique"] == 5
    assert summary["unknown"] == 2
    assert summary["identity_mapped"] == 3
    # Class id 80 is outside the valid range, so it maps to unknown_id (-1),
    # which is counted as remapped because the destination differs from source.
    assert summary["remapped"] == 1
    assert summary["conflicts"] == 2


def test_label_map_from_committed_yaml():
    root = Path(__file__).resolve().parent.parent
    lm = LabelMap.from_yaml(root / "configs" / "label_map.yaml")
    assert lm.is_identity()
    assert lm.valid_range == [0, 79]
    assert lm.unknown_id == -1


def test_label_map_summary():
    lm = LabelMap(mapping={1: 2})
    summary = lm.summary()
    assert summary["is_identity"] is False
    assert summary["explicit_mappings"] == 1
