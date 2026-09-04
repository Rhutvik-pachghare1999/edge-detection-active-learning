"""Tests for the COCO-2017 subset preparation and integrity."""

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image


def _load_fetch_module():
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "fetch_dataset.py"
    spec = importlib.util.spec_from_file_location("fetch_dataset", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_synthetic_coco(tmp_path: Path, n_images: int = 10):
    """Write a tiny COCO-style annotation and one-pixel images."""
    coco_root = tmp_path / "source"
    images_dir = coco_root / "val2017"
    ann_dir = coco_root / "annotations"
    images_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    images = []
    annotations = []
    for i in range(1, n_images + 1):
        file_name = f"{i:012d}.jpg"
        img_path = images_dir / file_name
        Image.new("RGB", (10, 10), color=(i % 256, 0, 0)).save(img_path)
        images.append({
            "id": i,
            "file_name": file_name,
            "width": 10,
            "height": 10,
        })
        # Add two boxes per image with COCO ids 1 (person -> yolo 0) and 3 (car -> yolo 2)
        for cat_id, x in [(1, 1), (3, 5)]:
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": i,
                "category_id": cat_id,
                "bbox": [float(x), 1.0, 3.0, 3.0],
                "area": 9.0,
                "iscrowd": 0,
            })

    ann_path = ann_dir / "instances_val2017.json"
    ann_path.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
        ],
    }))
    return coco_root


def test_build_subset_sizes_and_disjointness():
    """TRAIN_POOL + TEST must equal subset_size and be disjoint."""
    fetch = _load_fetch_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        coco_root = _make_synthetic_coco(tmp_path, n_images=10)
        out_root = tmp_path / "subset"
        manifest = fetch.build_subset(coco_root, out_root, subset_size=10, train_size=7)

    assert manifest["subset_size"] == 10
    assert manifest["train_size"] == 7
    assert manifest["test_size"] == 3

    train_ids = {im["id"] for im in manifest["images"] if im["split"] == "train"}
    test_ids = {im["id"] for im in manifest["images"] if im["split"] == "test"}
    assert len(train_ids) == 7
    assert len(test_ids) == 3
    assert train_ids.isdisjoint(test_ids)


def test_build_subset_sorted_by_coco_id():
    """Selection must be sorted by COCO id, first N."""
    fetch = _load_fetch_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        coco_root = _make_synthetic_coco(tmp_path, n_images=10)
        out_root = tmp_path / "subset"
        manifest = fetch.build_subset(coco_root, out_root, subset_size=8, train_size=5)

    ids = [im["id"] for im in manifest["images"]]
    assert ids == sorted(ids)
    assert ids == list(range(1, 9))


def test_build_subset_yolo_class_mapping():
    """COCO category ids must map to YOLO contiguous 0-79 indices."""
    fetch = _load_fetch_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        coco_root = _make_synthetic_coco(tmp_path, n_images=10)
        out_root = tmp_path / "subset"
        fetch.build_subset(coco_root, out_root, subset_size=10, train_size=5)

        label_file = out_root / "labels" / "train" / "000000000001.txt"
        lines = label_file.read_text().strip().splitlines()
    assert len(lines) == 2
    classes = {int(line.split()[0]) for line in lines}
    assert classes == {0, 2}  # person -> 0, car -> 2


def test_build_subset_manifest_has_checksums_and_relative_paths():
    """Manifest records must contain sha256 and relative label_file paths."""
    fetch = _load_fetch_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        coco_root = _make_synthetic_coco(tmp_path, n_images=10)
        out_root = tmp_path / "subset"
        manifest = fetch.build_subset(coco_root, out_root, subset_size=10, train_size=5)

    for im in manifest["images"]:
        assert "sha256" in im
        assert len(im["sha256"]) == 64
        assert im["label_file"].startswith("labels/")


def test_build_subset_rejects_missing_annotation():
    fetch = _load_fetch_module()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            fetch.build_subset(Path(tmp), Path(tmp) / "out", subset_size=10, train_size=5)
