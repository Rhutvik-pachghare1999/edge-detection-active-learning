"""Canonical COCO-80 → YOLO contiguous class mapping.

COCO category IDs are not contiguous (they contain gaps). YOLO expects class
indices in the range 0..79. This module defines the standard COCO-80 order used
by both the prepared dataset and the YOLOv8 ONNX student, and provides helpers
to remap teacher predictions that may arrive as either COCO category IDs or
already-contiguous YOLO indices.
"""

from typing import Dict, List

# Standard COCO-80 category IDs in the contiguous YOLOv8 order.
COCO_CAT_IDS: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

CAT_ID_TO_YOLO_IDX: Dict[int, int] = {cat_id: idx for idx, cat_id in enumerate(COCO_CAT_IDS)}
YOLO_IDX_TO_NAME: Dict[int, str] = {
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

NUM_COCO_CLASSES = len(COCO_CAT_IDS)


def remap_class_id(class_id: int) -> int:
    """Convert a COCO category ID or a YOLO index to a YOLO 0..79 index.

    The function is idempotent for already-contiguous YOLO indices and maps
    original COCO category IDs (e.g., person=1) to their YOLO positions.
    Unknown IDs are mapped to -1.
    """
    cid = int(class_id)
    if cid in CAT_ID_TO_YOLO_IDX:
        return CAT_ID_TO_YOLO_IDX[cid]
    if 0 <= cid < NUM_COCO_CLASSES:
        return cid
    return -1
