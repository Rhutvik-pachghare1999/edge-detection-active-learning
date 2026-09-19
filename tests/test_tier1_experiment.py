"""Tests for the Tier-1 experiment orchestration helpers."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aecs_sdc import tier1


def test_load_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "source": "COCO-2017 validation",
            "images": [
                {"id": 1, "file_name": "000000000001.jpg", "split": "train"},
                {"id": 2, "file_name": "000000000002.jpg", "split": "test"},
            ],
        }
        manifest_path.write_text(json.dumps(manifest))
        loaded = tier1.load_manifest(manifest_path)
    assert loaded["source"] == "COCO-2017 validation"
    assert len(loaded["images"]) == 2


def test_manifest_split_ids_and_disjointness():
    manifest = {
        "images": [
            {"id": 1, "file_name": "1.jpg", "split": "train"},
            {"id": 2, "file_name": "2.jpg", "split": "train"},
            {"id": 3, "file_name": "3.jpg", "split": "test"},
        ]
    }
    train_ids, test_ids = tier1.manifest_split_ids(manifest)
    assert train_ids == ["1", "2"]
    assert test_ids == ["3"]
    assert tier1.train_test_are_disjoint(manifest)


def test_train_test_disjointness_fails_on_overlap():
    manifest = {
        "images": [
            {"id": 1, "file_name": "1.jpg", "split": "train"},
            {"id": 1, "file_name": "1.jpg", "split": "test"},
        ]
    }
    assert not tier1.train_test_are_disjoint(manifest)


def test_detections_to_yolo_lines():
    dets = [
        {"label": 0, "cx": 0.5, "cy": 0.5, "bw": 0.2, "bh": 0.2},
        {"label": 79, "cx": 0.1, "cy": 0.2, "bw": 0.05, "bh": 0.05},
    ]
    lines = tier1.detections_to_yolo_lines(dets)
    assert len(lines) == 2
    assert lines[0].startswith("0 0.500000 0.500000 0.200000 0.200000")
    assert lines[1].startswith("79 0.100000 0.200000 0.050000 0.050000")


def test_prepare_yolo_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_root = tmp_path / "data"
        subset_root = data_root / "subset"
        for split in ["train", "val"]:
            (subset_root / "images" / split).mkdir(parents=True)
            (subset_root / "labels" / split).mkdir(parents=True)

        # Create 3 train + 2 test tiny images and labels.
        # The TEST split is stored under the 'val' directory to match Ultralytics.
        for im_id, split in [(1, "train"), (2, "train"), (3, "train"), (4, "val"), (5, "val")]:
            fname = f"{im_id:012d}.jpg"
            Image.new("RGB", (10, 10)).save(subset_root / "images" / split / fname)
            (subset_root / "labels" / split / f"{im_id:012d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        manifest = {
            "images": [
                {"id": 1, "file_name": "000000000001.jpg", "split": "train",
                 "label_file": "labels/train/000000000001.txt"},
                {"id": 2, "file_name": "000000000002.jpg", "split": "train",
                 "label_file": "labels/train/000000000002.txt"},
                {"id": 3, "file_name": "000000000003.jpg", "split": "train",
                 "label_file": "labels/train/000000000003.txt"},
                    {"id": 4, "file_name": "000000000004.jpg", "split": "test",
                     "label_file": "labels/val/000000000004.txt"},
                    {"id": 5, "file_name": "000000000005.jpg", "split": "test",
                     "label_file": "labels/val/000000000005.txt"},
            ]
        }
        (subset_root / "manifest.json").write_text(json.dumps(manifest))

        pseudo = {
            "1": [{"label": 0, "cx": 0.4, "cy": 0.4, "bw": 0.1, "bh": 0.1}],
            "2": [],
            "3": [{"label": 1, "cx": 0.3, "cy": 0.3, "bw": 0.2, "bh": 0.2}],
        }
        data_yaml = tier1.prepare_yolo_dataset(
            data_root=data_root,
            output_dir=tmp_path / "out",
            selected_train_ids=["1", "3"],
            test_ids=["4", "5"],
            train_pseudo_labels=pseudo,
            manifest=manifest,
            subset_name="subset",
            run_name="testrun",
        )

        assert data_yaml.exists()
        content = data_yaml.read_text()
        assert "names:" in content
        assert "nc: 80" in content
        out_root = data_yaml.parent
        assert (out_root / "images" / "train" / "000000000001.jpg").exists()
        assert (out_root / "labels" / "train" / "000000000001.txt").exists()
        assert (out_root / "images" / "val" / "000000000004.jpg").exists()
        assert (out_root / "labels" / "val" / "000000000004.txt").exists()


def test_aggregate_results():
    results = [
        {"arm": "harvested", "k": 250, "seed": 42, "mAP50": 0.31, "mAP50_95": 0.15},
        {"arm": "harvested", "k": 250, "seed": 43, "mAP50": 0.33, "mAP50_95": 0.17},
        {"arm": "random", "k": 250, "seed": 42, "mAP50": 0.29, "mAP50_95": 0.14},
        {"arm": "random", "k": 250, "seed": 43, "mAP50": 0.30, "mAP50_95": 0.16},
    ]
    agg = tier1.aggregate_results(results)
    assert "harvested_k250" in agg
    assert "random_k250" in agg
    assert agg["harvested_k250"]["metrics"]["mAP50"]["mean"] == pytest.approx(0.32, abs=0.01)
    assert agg["random_k250"]["metrics"]["mAP50"]["mean"] == pytest.approx(0.295, abs=0.01)


def test_plot_map50_creates_png():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        aggregated = {
            "harvested_k250": {
                "arm": "harvested",
                "k": 250,
                "metrics": {
                    "mAP50": {"mean": 0.32, "std": 0.01, "n": 3, "values": [0.31, 0.32, 0.33]},
                },
            },
            "random_k250": {
                "arm": "random",
                "k": 250,
                "metrics": {
                    "mAP50": {"mean": 0.29, "std": 0.01, "n": 3, "values": [0.28, 0.29, 0.30]},
                },
            },
        }
        plot_path = tmp_path / "map50.png"
        tier1.plot_map50(aggregated, plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
