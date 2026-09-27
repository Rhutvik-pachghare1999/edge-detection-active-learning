# Teacher model — RT-DETR-r50vd running on the laptop (GPU if available).
# This is the high-accuracy reference detector.  Its confidence scores are
# compared against the student's scores on the Pi to decide what to harvest.

import numpy as np
import torch
from PIL import Image
from loguru import logger
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from config import TEACHER_MODEL


class TeacherModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {TEACHER_MODEL} on {self.device}")

        self.processor = RTDetrImageProcessor.from_pretrained(TEACHER_MODEL)
        self.model     = RTDetrForObjectDetection.from_pretrained(TEACHER_MODEL)
        self.model.to(self.device).eval()

        # Run one blank frame through so the first real inference isn't slow
        self._run(Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)))
        logger.info("Teacher model ready")

    def _run(self, image: Image.Image) -> dict:
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
            threshold=0.3,
        )[0]

        scores = results["scores"].cpu().numpy()
        boxes  = results["boxes"].cpu().numpy()   # [N, 4] xyxy absolute pixels
        labels = results["labels"].cpu().tolist()

        # Normalise boxes to YOLO format (cx, cy, bw, bh) in [0,1]
        detections = []
        for score, label, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            name = self.model.config.id2label.get(label, str(label))
            detections.append({"label": int(label), "name": name,
                                "score": round(float(score), 4),
                                "cx": round(float(cx), 6), "cy": round(float(cy), 6),
                                "bw": round(float(bw), 6), "bh": round(float(bh), 6)})
        return {
            "top_confidence": float(scores.max()) if len(scores) > 0 else 0.0,
            "n_detections":   int(len(scores)),
            "detections":     detections,
        }

    def infer(self, frame_rgb: np.ndarray) -> dict:
        return self._run(Image.fromarray(frame_rgb))
