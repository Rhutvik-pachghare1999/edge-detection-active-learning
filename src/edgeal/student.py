"""YOLOv8n ONNX student model wrapper with proper uncertainty estimation.

The committed `models/yolov8n.onnx` is the same student used by the original
supervisor. This wrapper runs inference on a single BGR frame and returns
detections above a configurable threshold plus multiple uncertainty signals
for active learning.

The ONNX output tensor has shape (1, 84, 8400): 4 box coordinates (cx, cy, bw, bh)
followed by 80 COCO class *confidence scores* (sigmoid-activated, not logits)
for each of 8400 anchor points.

Uncertainty signals provided (following active learning literature for object detection):
- Instance-level: per-detection entropy, least-confidence, margin
- Image-level: mean/max instance entropy, mean/max least-confidence, mean/max margin,
  detection count, detection density
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

NUM_CLASSES = len(COCO_NAMES)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Compute binary entropy for independent Bernoulli variables.
    
    H = -sum(p * log(p) + (1-p) * log(1-p)) for each class.
    This is correct for multi-label classification (sigmoid outputs).
    """
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _instance_uncertainties(class_scores: np.ndarray) -> Dict[str, float]:
    """Compute uncertainty measures for a single detection instance.
    
    Args:
        class_scores: (80,) array of sigmoid-activated class confidences
        
    Returns:
        Dict with entropy, least_confidence, margin
    """
    # Entropy over independent class probabilities (binary entropy)
    entropy = float(np.sum(_binary_entropy(class_scores)))
    max_entropy = float(NUM_CLASSES * np.log(2))  # max when all p=0.5
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Least confidence: 1 - max class probability
    max_conf = float(np.max(class_scores))
    least_confidence = 1.0 - max_conf
    
    # Margin: difference between top-1 and top-2
    sorted_scores = np.sort(class_scores)[::-1]
    margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 1.0
    
    return {
        "entropy": normalized_entropy,
        "least_confidence": least_confidence,
        "margin": margin,
        "max_confidence": max_conf,
    }


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
        """Run inference and return detections with multiple uncertainty signals."""
        if self.session is None:
            raise RuntimeError("Student model not loaded")

        input_tensor, scale, pad_left, pad_top, orig_w, orig_h = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = outputs[0][0]  # shape (84, 8400)

        # Split into box coordinates and class confidence scores (sigmoid-activated).
        # YOLOv8n ONNX exports (1, 84, 8400): 4 box values (cx, cy, bw, bh in
        # input pixels) followed by 80 COCO class confidence scores.
        boxes = predictions[:4, :]           # (4, 8400) in input-pixel cxcywh
        class_scores = predictions[4:, :]    # (80, 8400) -- already sigmoid
        input_w, input_h = float(self.input_size[0]), float(self.input_size[1])

        # Per-anchor max class score and class id
        anchor_scores = np.max(class_scores, axis=0)
        anchor_classes = np.argmax(class_scores, axis=0)

        # Top anchor overall (for backward compatibility)
        top_anchor_idx = int(np.argmax(anchor_scores)) if anchor_scores.size > 0 else -1
        top_confidence = float(anchor_scores[top_anchor_idx]) if top_anchor_idx >= 0 else 0.0
        top_class_id = int(anchor_classes[top_anchor_idx]) if top_anchor_idx >= 0 else -1
        top_class_name = COCO_NAMES[top_class_id] if 0 <= top_class_id < NUM_CLASSES else "unknown"

        # Compute uncertainty for all anchors above threshold
        valid_mask = anchor_scores >= self.conf_threshold
        valid_indices = np.where(valid_mask)[0]
        
        detections = []
        instance_entropies = []
        instance_least_conf = []
        instance_margins = []
        
        for idx in valid_indices:
            cx_px, cy_px, bw_px, bh_px = boxes[:, idx]
            # Map model-input-pixel coordinates back to the original frame
            cx_orig = (float(cx_px) - pad_left) / scale / orig_w
            cy_orig = (float(cy_px) - pad_top) / scale / orig_h
            bw_orig = float(bw_px) / scale / orig_w
            bh_orig = float(bh_px) / scale / orig_h
            
            class_id = int(anchor_classes[idx])
            score = float(anchor_scores[idx])
            
            # Per-instance uncertainty
            unc = _instance_uncertainties(class_scores[:, idx])
            instance_entropies.append(unc["entropy"])
            instance_least_conf.append(unc["least_confidence"])
            instance_margins.append(unc["margin"])
            
            detections.append({
                "label": class_id,
                "name": COCO_NAMES[class_id] if 0 <= class_id < NUM_CLASSES else "unknown",
                "score": round(score, 4),
                "cx": float(np.clip(cx_orig, 0.0, 1.0)),
                "cy": float(np.clip(cy_orig, 0.0, 1.0)),
                "bw": float(np.clip(bw_orig, 0.0, 1.0)),
                "bh": float(np.clip(bh_orig, 0.0, 1.0)),
                # Per-instance uncertainty
                "entropy": round(unc["entropy"], 6),
                "least_confidence": round(unc["least_confidence"], 6),
                "margin": round(unc["margin"], 6),
            })

        # Image-level aggregation (following active learning literature)
        if instance_entropies:
            mean_entropy = float(np.mean(instance_entropies))
            max_entropy = float(np.max(instance_entropies))
            mean_least_conf = float(np.mean(instance_least_conf))
            max_least_conf = float(np.max(instance_least_conf))
            mean_margin = float(np.mean(instance_margins))
            min_margin = float(np.min(instance_margins))
        else:
            mean_entropy = max_entropy = 0.0
            mean_least_conf = max_least_conf = 0.0
            mean_margin = min_margin = 0.0

        # Also compute top-anchor entropy (for backward compat, but using correct method)
        top_entropy = 0.0
        if top_anchor_idx >= 0:
            top_unc = _instance_uncertainties(class_scores[:, top_anchor_idx])
            top_entropy = top_unc["entropy"]

        return {
            # Backward-compatible fields
            "top_confidence": top_confidence,
            "top_class_id": top_class_id,
            "top_class_name": top_class_name,
            "entropy": round(top_entropy, 6),  # top-anchor normalized entropy
            "n_detections": len(detections),
            "detections": detections,
            # New: image-level uncertainty aggregations (proper for active learning)
            "image_uncertainty": {
                "mean_entropy": round(mean_entropy, 6),
                "max_entropy": round(max_entropy, 6),
                "mean_least_confidence": round(mean_least_conf, 6),
                "max_least_confidence": round(max_least_conf, 6),
                "mean_margin": round(mean_margin, 6),
                "min_margin": round(min_margin, 6),
                "detection_count": len(detections),
            },
            # Raw per-instance uncertainties for advanced aggregation
            "instance_uncertainties": {
                "entropies": [round(e, 6) for e in instance_entropies],
                "least_confidences": [round(e, 6) for e in instance_least_conf],
                "margins": [round(e, 6) for e in instance_margins],
            },
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
                                    "cx": 0.5, "cy": 0.5, "bw": 0.3, "bh": 0.5,
                                    "entropy": 0.35, "least_confidence": 0.58, "margin": 0.1}],
                    "image_uncertainty": {
                        "mean_entropy": 0.35,
                        "max_entropy": 0.35,
                        "mean_least_confidence": 0.58,
                        "max_least_confidence": 0.58,
                        "mean_margin": 0.1,
                        "min_margin": 0.1,
                        "detection_count": 1,
                    },
                    "instance_uncertainties": {
                        "entropies": [0.35],
                        "least_confidences": [0.58],
                        "margins": [0.1],
                    },
                }
        return MockStudent()
