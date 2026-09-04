"""YOLOv8n ONNX student model wrapper.

The committed `models/yolov8n.onnx` is the same student used by the original
supervisor. This wrapper runs inference on a single BGR frame and returns the
top confidence score plus detections above a configurable threshold.

The ONNX output tensor has shape (1, 84, 8400): 4 box coordinates followed by
80 COCO class scores for each of 8400 anchor points.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
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


class StudentModel:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.input_size = input_size  # (width, height)
        self.providers = providers
        self.session = None
        self._load()

    def _load(self) -> None:
        import onnxruntime as ort
        if not cv2.os.path.exists(self.model_path):
            raise FileNotFoundError(f"Student model not found: {self.model_path}")
        logger.info(f"Loading student model from {self.model_path}")
        providers = self.providers if self.providers is not None else ort.get_available_providers()
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Student loaded; providers={self.session.get_providers()}")

    def _preprocess(self, frame: np.ndarray):
        """Letterbox resize, normalize to [0,1], convert BGR→RGB, and add batch dim.

        Returns the preprocessed input tensor plus the scale and padding needed
        to map model-output box coordinates back to the original frame.
        """
        target_h, target_w = self.input_size[1], self.input_size[0]
        orig_h, orig_w = frame.shape[:2]
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        # ONNX model expects (B, C, H, W)
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return tensor, scale, pad_left, pad_top, orig_w, orig_h

    def infer(self, frame: np.ndarray) -> Dict:
        """Run inference and return scores, detections, and entropy."""
        if self.session is None:
            raise RuntimeError("Student model not loaded")

        input_tensor, scale, pad_left, pad_top, orig_w, orig_h = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = outputs[0][0]  # shape (84, 8400)

        # Split into box coordinates and class logits/scores.
        # YOLOv8n ONNX exports (1, 84, 8400): 4 box values (cx, cy, bw, bh in
        # input pixels) followed by 80 COCO class scores.
        boxes = predictions[:4, :]           # (4, 8400) in input-pixel cxcywh
        class_scores = predictions[4:, :]    # (80, 8400)
        input_w, input_h = float(self.input_size[0]), float(self.input_size[1])

        # Per-anchor max class score and class id
        anchor_scores = np.max(class_scores, axis=0)
        anchor_classes = np.argmax(class_scores, axis=0)

        top_anchor_idx = int(np.argmax(anchor_scores)) if anchor_scores.size > 0 else -1
        top_confidence = float(anchor_scores[top_anchor_idx]) if top_anchor_idx >= 0 else 0.0
        top_class_id = int(anchor_classes[top_anchor_idx]) if top_anchor_idx >= 0 else -1
        top_class_name = COCO_NAMES[top_class_id] if 0 <= top_class_id < len(COCO_NAMES) else "unknown"

        # Compute prediction entropy over the top anchor's class distribution
        # as a simple uncertainty proxy.
        top_dist = class_scores[:, top_anchor_idx] if top_anchor_idx >= 0 else np.zeros(class_scores.shape[0])
        # Softmax for entropy
        exp_scores = np.exp(top_dist - np.max(top_dist))
        probs = exp_scores / (exp_scores.sum() + 1e-10)
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = np.log(len(COCO_NAMES))
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        detections = []
        valid = anchor_scores >= self.conf_threshold
        for idx in np.where(valid)[0]:
            cx_px, cy_px, bw_px, bh_px = boxes[:, idx]
            # Map model-input-pixel coordinates back to the original frame
            # after removing letterbox padding and undoing the resize scale.
            cx_orig = (float(cx_px) - pad_left) / scale / orig_w
            cy_orig = (float(cy_px) - pad_top) / scale / orig_h
            bw_orig = float(bw_px) / scale / orig_w
            bh_orig = float(bh_px) / scale / orig_h
            detections.append({
                "label": int(anchor_classes[idx]),
                "name": COCO_NAMES[int(anchor_classes[idx])] if 0 <= int(anchor_classes[idx]) < len(COCO_NAMES) else "unknown",
                "score": round(float(anchor_scores[idx]), 4),
                "cx": float(np.clip(cx_orig, 0.0, 1.0)),
                "cy": float(np.clip(cy_orig, 0.0, 1.0)),
                "bw": float(np.clip(bw_orig, 0.0, 1.0)),
                "bh": float(np.clip(bh_orig, 0.0, 1.0)),
            })

        return {
            "top_confidence": top_confidence,
            "top_class_id": top_class_id,
            "top_class_name": top_class_name,
            "entropy": normalized_entropy,
            "n_detections": len(detections),
            "detections": detections,
        }

    @staticmethod
    def mock():
        """Return a tiny mock student for unit tests that does not load ONNX."""
        class MockStudent:
            conf_threshold = 0.25
            input_size = (640, 480)
            def infer(self, frame):
                return {
                    "top_confidence": 0.42,
                    "top_class_id": 0,
                    "top_class_name": "person",
                    "entropy": 0.35,
                    "n_detections": 1,
                    "detections": [{"label": 0, "name": "person", "score": 0.42,
                                    "cx": 0.5, "cy": 0.5, "bw": 0.3, "bh": 0.5}],
                }
        return MockStudent()
