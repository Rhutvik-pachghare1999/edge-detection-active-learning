"""Configuration loader for the AECS-SDC benchmark."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    seed: int = 42
    clips_dir: str = "clips"
    labels_dir: str = "clips"
    subset_file: str = "configs/benchmark_subset.txt"
    subset_size: int = 50
    trigger_offset_seconds: float = 5.0
    frames_per_clip: int = 1
    results_dir: str = "results"
    summary_file: str = "results/benchmark_summary.json"
    details_file: str = "results/benchmark_details.json"
    label_map: str = "configs/label_map.yaml"


@dataclass
class ModelConfig:
    student: str = "models/yolov8n.onnx"
    teacher: str = "PekingU/rtdetr_r50vd"
    teacher_device: str = "auto"
    student_conf_threshold: float = 0.25
    teacher_conf_threshold: float = 0.30
    image_size: List[int] = field(default_factory=lambda: [640, 480])


@dataclass
class DisagreementConfig:
    mode: str = "combined"
    entropy_tau: float = 0.85
    weights: dict = field(default_factory=lambda: {
        "margin": 0.50,
        "entropy": 0.30,
        "class_mismatch": 0.20,
    })


@dataclass
class HarvestConfig:
    mode: str = "threshold"
    budget: Optional[int] = None
    target_rate: float = 0.05
    min_interval_seconds: float = 0.0
    tau_initial: float = 0.15
    tau_min: float = 0.05
    tau_max: float = 0.80


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/benchmark.log"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"


@dataclass
class Config:
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    disagreement: DisagreementConfig = field(default_factory=DisagreementConfig)
    harvest: HarvestConfig = field(default_factory=HarvestConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        return cls(
            benchmark=BenchmarkConfig(**raw.get("benchmark", {})),
            models=ModelConfig(**raw.get("models", {})),
            disagreement=DisagreementConfig(**raw.get("disagreement", {})),
            harvest=HarvestConfig(**raw.get("harvest", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )

    def resolve_paths(self, root_dir: str = ".") -> "Config":
        """Resolve relative paths against the project root."""
        root = Path(root_dir).resolve()
        self.benchmark.clips_dir = str(root / self.benchmark.clips_dir)
        self.benchmark.labels_dir = str(root / self.benchmark.labels_dir)
        self.benchmark.subset_file = str(root / self.benchmark.subset_file)
        self.benchmark.results_dir = str(root / self.benchmark.results_dir)
        self.benchmark.summary_file = str(root / self.benchmark.summary_file)
        self.benchmark.details_file = str(root / self.benchmark.details_file)
        self.benchmark.label_map = str(root / self.benchmark.label_map)
        self.models.student = str(root / self.models.student)
        self.logging.file = str(root / self.logging.file)
        return self
