"""Unit tests for dataset utilities."""

from pathlib import Path

from aecs_sdc.dataset import build_subset, load_subset, read_yolo_labels


def test_committed_subset_exists_and_is_non_empty():
    root = Path(__file__).resolve().parent.parent
    subset_file = root / "configs" / "benchmark_subset.txt"
    subset = load_subset(subset_file)
    assert len(subset) == 50
    assert all(name.endswith(".mp4") for name in subset)


def test_build_subset_matches_committed():
    """If the committed subset is regenerated deterministically it must match."""
    root = Path(__file__).resolve().parent.parent
    built = build_subset(str(root / "clips"), str(root / "clips"), 50, 42)
    committed = load_subset(root / "configs" / "benchmark_subset.txt")
    assert built == committed


def test_read_yolo_labels_missing_file_returns_empty():
    labels = read_yolo_labels("/this/does/not/exist.txt")
    assert labels == []


def test_read_yolo_labels_remaps_class_ids():
    from aecs_sdc.label_map import LabelMap
    lm = LabelMap(mapping={0: 1}, unknown_id=-1, valid_range=[0, 79])
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
        f.write("99 0.1 0.1 0.1 0.1\n")
        path = f.name
    try:
        labels = read_yolo_labels(path, lm)
        assert len(labels) == 1
        assert labels[0].class_id == 1
    finally:
        import os
        os.remove(path)
