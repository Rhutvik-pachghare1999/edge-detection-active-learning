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
    """If the committed subset is regenerated deterministically it must match.

    Depends on the raw `clips/` data, which is not committed to the public repo
    (large legacy-prototype media). Skips when the clips are absent — e.g. in CI
    on a fresh clone — so it validates determinism locally without failing CI.
    """
    import pytest
    root = Path(__file__).resolve().parent.parent
    clips_dir = root / "clips"
    clip_files = list(clips_dir.glob("*.mp4")) if clips_dir.is_dir() else []
    if not clip_files:
        pytest.skip("clips/ data not present (not committed to public repo)")
    built = build_subset(str(clips_dir), str(clips_dir), 50, 42)
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
