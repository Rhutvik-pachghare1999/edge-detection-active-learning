"""Helpers for the Tier-1 COCO active-learning experiment.

This module contains the pure logic used by ``scripts/run_tier1_experiment.py``:
manifest parsing, temporary YOLO dataset creation, result aggregation, plotting,
and separation checks.  It stays independent of SOL/ASU paths so it can be
committed and run anywhere.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from loguru import logger


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def load_manifest(manifest_path: Path) -> dict:
    """Load the COCO subset manifest written by ``fetch_dataset.py``."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def manifest_split_ids(manifest: dict) -> Tuple[List[str], List[str]]:
    """Return (train_image_ids, test_image_ids) as ordered lists of COCO ids."""
    train = [str(im["id"]) for im in manifest["images"] if im["split"] == "train"]
    test = [str(im["id"]) for im in manifest["images"] if im["split"] == "test"]
    return train, test


def manifest_id_to_info(manifest: dict) -> Dict[str, dict]:
    """Return a lookup ``{image_id: manifest image record}``."""
    return {str(im["id"]): im for im in manifest["images"]}


def train_test_are_disjoint(manifest: dict) -> bool:
    """Return True if no image id appears in both splits."""
    train_ids = {str(im["id"]) for im in manifest["images"] if im["split"] == "train"}
    test_ids = {str(im["id"]) for im in manifest["images"] if im["split"] == "test"}
    return len(train_ids & test_ids) == 0


def detections_to_yolo_lines(detections: List[dict]) -> List[str]:
    """Convert detection dicts to YOLO-format lines."""
    lines = []
    for det in detections:
        lines.append(
            f"{int(det['label'])} {float(det['cx']):.6f} "
            f"{float(det['cy']):.6f} {float(det['bw']):.6f} {float(det['bh']):.6f}"
        )
    return lines


def prepare_yolo_dataset(
    data_root: Path,
    output_dir: Path,
    selected_train_ids: List[str],
    test_ids: List[str],
    train_pseudo_labels: Dict[str, List[dict]],
    manifest: dict,
    subset_name: str = "subset",
    run_name: str = "tier1",
) -> Path:
    """Create a temporary YOLO dataset for one arm/seed/K run.

    ``train_pseudo_labels`` maps image id -> list of teacher detection dicts.
    Test labels are the human COCO labels already produced by ``fetch_dataset.py``.

    Returns the path to the generated ``data.yaml``.
    """
    ds_root = output_dir / f"yolo_{run_name}"
    if ds_root.exists():
        shutil.rmtree(ds_root)
    (ds_root / "images" / "train").mkdir(parents=True)
    (ds_root / "images" / "val").mkdir(parents=True)
    (ds_root / "labels" / "train").mkdir(parents=True)
    (ds_root / "labels" / "val").mkdir(parents=True)

    lookup = manifest_id_to_info(manifest)
    subset_root = data_root / subset_name

    def _copy_split(ids: List[str], split: str, labels_source: Optional[Dict[str, List[dict]]] = None):
        for im_id in ids:
            info = lookup.get(im_id)
            if info is None:
                logger.warning(f"Image id {im_id} not in manifest; skipping")
                continue
            file_name = info["file_name"]
            stem = Path(file_name).stem
            src_img = subset_root / "images" / split / file_name
            dst_img = ds_root / "images" / split / file_name
            if not src_img.exists():
                logger.warning(f"Missing image {src_img}; skipping")
                continue
            shutil.copy2(src_img, dst_img)

            if labels_source is not None:
                # Train split: pseudo-labels from teacher cache.
                lines = detections_to_yolo_lines(labels_source.get(im_id, []))
                (ds_root / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
            else:
                # Test split: human COCO labels from the prepared subset.
                src_lbl = subset_root / info["label_file"]
                if src_lbl.exists():
                    shutil.copy2(src_lbl, ds_root / "labels" / split / f"{stem}.txt")
                else:
                    (ds_root / "labels" / split / f"{stem}.txt").write_text("")

    _copy_split(selected_train_ids, "train", train_pseudo_labels)
    _copy_split(test_ids, "val", None)

    data_yaml = ds_root / "data.yaml"
    yaml_content = {
        "path": str(ds_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 80,
        "names": COCO_NAMES,
    }
    with open(data_yaml, "w") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)

    logger.info(f"Prepared YOLO dataset at {ds_root}: train={len(selected_train_ids)}, val={len(test_ids)}")
    return data_yaml


def aggregate_results(results: List[dict]) -> Dict[str, dict]:
    """Aggregate a list of per-run metrics into mean±std per arm and K.

    ``results`` items are dicts with keys: ``arm``, ``k``, ``seed``, ``mAP50``,
    ``mAP50_95``, ``precision``, ``recall``.
    """
    import numpy as np

    grouped: Dict[Tuple[str, int], List[dict]] = {}
    for r in results:
        key = (r.get("arm"), r.get("k"))
        grouped.setdefault(key, []).append(r)

    aggregated = {}
    for (arm, k), runs in grouped.items():
        metrics = {}
        for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
            vals = [r[metric] for r in runs if metric in r]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            metrics[metric] = {
                "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else 0.0,
                "n": len(vals),
                "values": [round(float(v), 6) for v in vals],
            }
        aggregated[f"{arm}_k{k}"] = {
            "arm": arm,
            "k": k,
            "metrics": metrics,
        }
    return aggregated


def save_results(
    results: List[dict],
    aggregated: dict,
    output_dir: Path,
    name: str = "tier1_results",
):
    """Write raw and aggregated results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{name}.json"
    agg_path = output_dir / f"{name}_aggregated.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Wrote results to {raw_path} and {agg_path}")


def plot_map50(
    aggregated: dict,
    output_path: Path,
    arms: Optional[List[str]] = None,
    metric: str = "mAP50",
):
    """Plot ``metric`` vs budget K with error bars for each arm."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib not available; skipping plot: {e}")
        return

    # Re-group by arm then k.
    by_arm: Dict[str, Dict[int, dict]] = {}
    for _, summary in aggregated.items():
        arm = summary["arm"]
        k = summary["k"]
        by_arm.setdefault(arm, {})[k] = summary["metrics"].get(metric, {})

    if arms is None:
        arms = list(by_arm.keys())

    plt.figure(figsize=(8, 5))
    for arm in arms:
        ks = sorted(by_arm[arm].keys())
        means = [by_arm[arm][k].get("mean") for k in ks if "mean" in by_arm[arm][k]]
        stds = [by_arm[arm][k].get("std", 0.0) for k in ks if "mean" in by_arm[arm][k]]
        valid_ks = [k for k in ks if "mean" in by_arm[arm][k]]
        if not means:
            continue
        plt.errorbar(valid_ks, means, yerr=stds, marker="o", capsize=4, label=arm)

    plt.xlabel("Training-set size K")
    plt.ylabel(metric)
    plt.title(f"{metric} vs training budget on held-out human TEST")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Wrote plot to {output_path}")
