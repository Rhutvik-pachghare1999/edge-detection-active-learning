#!/usr/bin/env python3
"""Fetch and prepare the COCO-2017 validation subset for AECS-SDC.

Downloads COCO val2017 images + annotations, selects a fixed 1,500-image subset
(1,200 TRAIN_POOL, 300 TEST), converts annotations to YOLO format, and writes a
manifest with image IDs, filenames, split, and SHA256 checksums.

The manifest is the reproducibility anchor: re-running this script on a fresh
machine produces the same file list, split, and YOLO labels (image pixel
checksums depend on the upstream zip, which is versioned by COCO 2017).

Usage:
    python scripts/fetch_dataset.py --data-root data/coco2017
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


# Standard COCO 80 category IDs in the contiguous order expected by YOLOv8.
# YOLO class index = position in this list.
COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
CAT_ID_TO_YOLO_IDX = {cat_id: idx for idx, cat_id in enumerate(COCO_CAT_IDS)}
YOLO_IDX_TO_NAME = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
    52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

DATASET_URLS = {
    "images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

DEFAULT_SUBSET_SIZE = 1500
DEFAULT_TRAIN_SIZE = 1200


def sha256_file(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, chunk_size: int = 8192 * 1024) -> None:
    """Download ``url`` to ``dest`` with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {url} -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB ({pct:.1f}%)",
                      end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a zip archive if ``dest`` does not already contain its content."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if marker.exists():
        return
    print(f"Extracting {zip_path} -> {dest}", file=sys.stderr)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    marker.write_text(str(zip_path.resolve()))


def load_coco_annotation(ann_path: Path) -> dict:
    """Load a COCO-format JSON annotation file."""
    with open(ann_path, "r") as f:
        return json.load(f)


def convert_bbox_coco_to_yolo(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert COCO (x, y, width, height) in pixels to normalized YOLO (cx, cy, bw, bh)."""
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h
    return cx, cy, bw, bh


def build_subset(
    coco_root: Path,
    out_root: Path,
    subset_size: int,
    train_size: int,
) -> Dict:
    """Create the fixed TRAIN_POOL/TEST split and YOLO labels.

    Returns the manifest dictionary.
    """
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    coco = load_coco_annotation(ann_file)
    images = sorted(coco["images"], key=lambda im: im["id"])

    if len(images) < subset_size:
        raise ValueError(
            f"COCO val2017 has only {len(images)} images, requested {subset_size}"
        )

    selected = images[:subset_size]
    train_ids = {im["id"] for im in selected[:train_size]}
    test_ids = {im["id"] for im in selected[train_size:]}

    # Build annotation lookup.
    anns_by_image: Dict[int, List[dict]] = {im["id"]: [] for im in selected}
    skipped_cats = set()
    for ann in coco["annotations"]:
        im_id = ann["image_id"]
        if im_id not in anns_by_image:
            continue
        cat_id = ann["category_id"]
        if cat_id not in CAT_ID_TO_YOLO_IDX:
            skipped_cats.add(cat_id)
            continue
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[im_id].append(ann)

    if skipped_cats:
        print(f"Note: skipped {len(skipped_cats)} non-COCO-80 categories "
              f"in the annotation file.", file=sys.stderr)

    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "COCO-2017 validation",
        "subset_size": subset_size,
        "train_size": train_size,
        "test_size": subset_size - train_size,
        "selection": "images sorted by COCO id, first 1500",
        "mapping": "COCO category id -> YOLOv8 contiguous index 0-79",
        "images": [],
    }

    for im in selected:
        im_id = im["id"]
        split = "train" if im_id in train_ids else "test"
        # Ultralytics expects validation images in images/val and labels in labels/val,
        # so store the TEST split under the 'val' directory while keeping 'split=test'
        # in the manifest.
        split_dir = "train" if split == "train" else "val"
        src_img = coco_root / "val2017" / im["file_name"]
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")

        dst_img = img_dir / split_dir / im["file_name"]
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        checksum = sha256_file(dst_img)

        # Write YOLO-format labels.
        labels = []
        for ann in anns_by_image.get(im_id, []):
            x, y, w, h = ann["bbox"]
            img_w, img_h = im["width"], im["height"]
            cx, cy, bw, bh = convert_bbox_coco_to_yolo(x, y, w, h, img_w, img_h)
            yolo_class = CAT_ID_TO_YOLO_IDX[ann["category_id"]]
            labels.append(f"{yolo_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_name = Path(im["file_name"]).stem + ".txt"
        dst_lbl = lbl_dir / split_dir / label_name
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(labels) + ("\n" if labels else ""))

        manifest["images"].append({
            "id": im_id,
            "file_name": im["file_name"],
            "split": split,
            "sha256": checksum,
            "width": im["width"],
            "height": im["height"],
            "label_file": str(dst_lbl.relative_to(out_root)),
        })

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Fetch COCO-2017 val subset")
    parser.add_argument("--data-root", type=str, default="data/coco2017",
                        help="Root directory to store downloaded COCO data and subset.")
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE,
                        help="Total number of images in the fixed subset.")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE,
                        help="Number of TRAIN_POOL images; remainder is TEST.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Assume archives are already in data-root; skip download.")
    args = parser.parse_args()

    if args.train_size >= args.subset_size:
        raise ValueError("train-size must be smaller than subset-size")

    data_root = Path(args.data_root).resolve()
    archives = data_root / "archives"
    coco_root = data_root / "source"
    subset_root = data_root / "subset"

    # 1. Download archives.
    if not args.skip_download:
        img_zip = archives / "val2017.zip"
        ann_zip = archives / "annotations_trainval2017.zip"
        download_file(DATASET_URLS["images"], img_zip)
        download_file(DATASET_URLS["annotations"], ann_zip)
    else:
        img_zip = archives / "val2017.zip"
        ann_zip = archives / "annotations_trainval2017.zip"
        if not img_zip.exists() or not ann_zip.exists():
            raise FileNotFoundError(f"--skip-download set but archives missing in {archives}")

    # 2. Extract.
    extract_zip(img_zip, coco_root)
    extract_zip(ann_zip, coco_root)

    # 3. Build fixed subset + YOLO labels.
    manifest = build_subset(
        coco_root,
        subset_root,
        args.subset_size,
        args.train_size,
    )

    # 4. Write manifest.
    manifest_path = subset_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # 5. Write a small human-readable summary.
    summary_path = subset_root / "SUBSET_SUMMARY.txt"
    summary_lines = [
        "COCO-2017 Validation Subset",
        "=" * 40,
        f"Total images: {manifest['subset_size']}",
        f"TRAIN_POOL:   {manifest['train_size']}",
        f"TEST:         {manifest['test_size']}",
        f"Selection:    {manifest['selection']}",
        f"Mapping:      {manifest['mapping']}",
        "",
        "Splits are disjoint. TEST labels are held out and used only for final mAP evaluation.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Subset ready at {subset_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
