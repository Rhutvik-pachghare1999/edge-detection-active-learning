"""Label-to-model class mapping for the AECS-SDC benchmark.

The local YOLO-format label files use integer class IDs. The student model
(YOLOv8n ONNX) outputs COCO 80 class scores. When the two taxonomies differ,
this layer maps label IDs to model IDs so evaluation compares like with like.

For the current dataset the label IDs are already COCO 80 IDs, so the default
mapping is the identity. Any class ID outside the valid COCO range [0, 79] is
treated as unknown and logged.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import yaml
from loguru import logger


COCO_RANGE = range(0, 80)


@dataclass
class LabelMap:
    name: str = "coco_identity"
    source: str = "dataset_label_id"
    target: str = "coco_80"
    mapping: Dict[int, int] = field(default_factory=dict)
    unknown_id: int = -1
    valid_range: List[int] = field(default_factory=lambda: [0, 79])

    @classmethod
    def from_yaml(cls, path: str) -> "LabelMap":
        path = Path(path)
        if not path.exists():
            logger.warning(f"Label map file not found: {path}; using identity COCO mapping")
            return cls()
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "LabelMap":
        mapping = {}
        for k, v in (raw.get("mapping") or {}).items():
            mapping[int(k)] = int(v)
        return cls(
            name=raw.get("name", "coco_identity"),
            source=raw.get("source", "dataset_label_id"),
            target=raw.get("target", "coco_80"),
            mapping=mapping,
            unknown_id=raw.get("unknown_id", -1),
            valid_range=raw.get("valid_range", [0, 79]),
        )

    def remap(self, class_id: int) -> int:
        """Return the model class ID for a label class ID."""
        if class_id in self.mapping:
            return self.mapping[class_id]
        lo, hi = self.valid_range
        if lo <= class_id <= hi:
            return class_id
        return self.unknown_id

    def validate(self, class_ids: Set[int]) -> Dict[str, int]:
        """Validate a set of label class IDs and return a summary."""
        lo, hi = self.valid_range
        unknown = {cid for cid in class_ids if not (lo <= cid <= hi) and cid not in self.mapping}
        remapped = {cid: self.remap(cid) for cid in class_ids}
        conflicts = {cid: dst for cid, dst in remapped.items() if dst == self.unknown_id}
        return {
            "total_unique": len(class_ids),
            "unknown": len(unknown),
            "conflicts": len(conflicts),
            "identity_mapped": sum(1 for cid, dst in remapped.items() if cid == dst and cid not in unknown),
            "remapped": sum(1 for cid, dst in remapped.items() if cid != dst),
        }

    def is_identity(self) -> bool:
        """True when the mapping does not translate any known IDs."""
        return not self.mapping

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "is_identity": self.is_identity(),
            "unknown_id": self.unknown_id,
            "valid_range": self.valid_range,
            "explicit_mappings": len(self.mapping),
        }
