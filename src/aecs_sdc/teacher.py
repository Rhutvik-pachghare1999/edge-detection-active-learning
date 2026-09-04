"""RT-DETR teacher model wrapper for offline benchmarking.

Loads the HuggingFace RT-DETR-r50vd model once and runs inference on individual
frames. This is the same teacher used by the live supervisor, but isolated from
the WebSocket / FastAPI lifecycle.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger


class TeacherModel:
    def __init__(self, model_name: str = "PekingU/rtdetr_r50vd",
                 device: Optional[str] = "auto",
                 conf_threshold: float = 0.30):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.conf_threshold = conf_threshold
        self.processor = None
        self.model = None
        self.id2label = {}
        self._load()

    def _resolve_device(self, device: Optional[str]) -> str:
        if device == "auto" or device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load(self) -> None:
        logger.info(f"Loading teacher model {self.model_name} on {self.device}")
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        self.processor = RTDetrImageProcessor.from_pretrained(self.model_name)
        self.model = RTDetrForObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label
        # Warm-up inference on a blank frame so the first real frame is not slow.
        self._run(Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)))
        logger.info("Teacher model ready")

    def _run(self, image: Image.Image) -> Dict:
        w, h = image.size
        inputs = {
            k: v.to(self.device)
            for k, v in self.processor(images=image, return_tensors="pt").items()
        }
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),
            threshold=self.conf_threshold,
        )[0]

        scores = results["scores"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()  # [N, 4] xyxy absolute pixels
        labels = results["labels"].cpu().tolist()

        detections = []
        for score, label, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            name = self.id2label.get(label, str(label))
            detections.append({
                "label": int(label),
                "name": name,
                "score": round(float(score), 4),
                "cx": round(float(cx), 6),
                "cy": round(float(cy), 6),
                "bw": round(float(bw), 6),
                "bh": round(float(bh), 6),
            })

        return {
            "top_confidence": float(scores.max()) if len(scores) > 0 else 0.0,
            "top_class_id": int(labels[np.argmax(scores)]) if len(scores) > 0 else -1,
            "top_class_name": (self.id2label.get(int(labels[np.argmax(scores)]), "unknown")
                               if len(scores) > 0 else "none"),
            "n_detections": int(len(scores)),
            "detections": detections,
        }

    def infer(self, frame_bgr: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._run(Image.fromarray(rgb))


# Lazy import to avoid a hard dependency on cv2 at module load time for tests.
import cv2  # noqa: E402
