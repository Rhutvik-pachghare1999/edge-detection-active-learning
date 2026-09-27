"""Real image-feature embeddings for active-learning diversity.

Uses a small, pretrained torchvision image encoder so the hybrid acquisition
function gets feature-space spread instead of a coarse class-count histogram.
The model is treated as a frozen feature extractor; it is never fine-tuned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


class ImageFeatureExtractor:
    """ResNet18 pooled-feature extractor (512-D) with ImageNet normalization."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, device: Optional[Union[str, torch.device]] = "auto"):
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.model = torchvision.models.resnet18(weights=weights)
        # Replace classification head with identity so forward emits avg-pooled features.
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

    @torch.inference_mode()
    def extract(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Return a 512-D, L2-normalized feature vector for one image.

        Accepts a file path or an H×W×C numpy array (RGB or BGR). BGR arrays
        are converted to RGB.
        """
        if isinstance(image, (str, Path)):
            pil = Image.open(str(image)).convert("RGB")
        else:
            # Accept numpy array (BGR from OpenCV or RGB from PIL).
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[2] == 3:  # assume BGR if uint8 and channel-last
                arr = arr[:, :, ::-1]
            pil = Image.fromarray(arr.astype(np.uint8))

        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        vec = features.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @torch.inference_mode()
    def extract_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:
        """Return ``{path: vector}`` for a list of image paths."""
        embeddings: Dict[str, np.ndarray] = {}
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            tensors = []
            for p in batch_paths:
                pil = Image.open(str(p)).convert("RGB")
                tensors.append(self.transform(pil))
            batch = torch.stack(tensors).to(self.device)
            feats = self.model(batch).cpu().numpy().astype(np.float32)
            for p, vec in zip(batch_paths, feats):
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings[str(p)] = vec
        return embeddings


def ensure_embeddings_cache(
    image_paths: List[Union[str, Path]],
    cache_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = "auto",
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Load cached embeddings or compute and save them.

    The cache stores a dict keyed by absolute image path. This is stable across
    re-runs as long as the pool image paths do not move.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            return data.item()
        return dict(data)

    extractor = ImageFeatureExtractor(device=device)
    embeddings = extractor.extract_batch(image_paths, batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings
