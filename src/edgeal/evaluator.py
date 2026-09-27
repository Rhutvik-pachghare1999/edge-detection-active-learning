"""Evaluation and reporting for the AECS-SDC benchmark.

Aggregates per-frame results, computes distributions, per-class statistics,
and writes JSON + human-readable summaries.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger


class Evaluator:
    def __init__(self, cfg, label_map=None):
        self.cfg = cfg
        self.label_map = label_map
        self.results_dir = Path(cfg.benchmark.results_dir)
        self.summary_file = Path(cfg.benchmark.summary_file)
        self.details_file = Path(cfg.benchmark.details_file)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict] = []
        self._start_time = None
        self._end_time = None

    def start(self) -> None:
        self._start_time = time.time()

    def add(self, record: Dict) -> None:
        """Add one frame's results."""
        self._records.append(record)

    def finish(self) -> Dict:
        self._end_time = time.time()
        summary = self._compute_summary()
        self._write_json(summary)
        self._write_summary_text(summary)
        self._write_disagreement_distribution(summary)
        return summary

    def _compute_summary(self) -> Dict:
        total = len(self._records)
        harvested = sum(1 for r in self._records if r["harvested"])
        scores = [r["disagreement"]["combined"] for r in self._records]
        margins = [r["disagreement"]["margin"] for r in self._records]
        entropies = [r["disagreement"]["entropy"] for r in self._records]
        class_mismatches = [r["disagreement"]["class_mismatch"] for r in self._records]
        ious = [r["disagreement"]["iou"] for r in self._records]

        def _dist(values):
            arr = np.array(values)
            return {
                "mean": round(float(arr.mean()), 6) if len(arr) else 0.0,
                "std": round(float(arr.std()), 6) if len(arr) else 0.0,
                "min": round(float(arr.min()), 6) if len(arr) else 0.0,
                "max": round(float(arr.max()), 6) if len(arr) else 0.0,
                "p25": round(float(np.percentile(arr, 25)), 6) if len(arr) else 0.0,
                "p50": round(float(np.percentile(arr, 50)), 6) if len(arr) else 0.0,
                "p75": round(float(np.percentile(arr, 75)), 6) if len(arr) else 0.0,
            }

        def _histogram(values, bins=10):
            arr = np.array(values)
            if len(arr) == 0:
                return []
            counts, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
            return [
                {"bin_start": round(float(edges[i]), 3), "bin_end": round(float(edges[i + 1]), 3), "count": int(counts[i])}
                for i in range(len(counts))
            ]

        # Per-class pseudo-ground-truth metrics.
        class_stats = defaultdict(lambda: {"gt": 0, "det": 0})
        for r in self._records:
            for label in r["labels"]:
                cid = int(label["class_id"])
                class_stats[cid]["gt"] += 1
            for det in r["student_detections"]:
                cid = int(det["label"])
                class_stats[cid]["det"] += 1

        per_class = {}
        for cid, stats in class_stats.items():
            per_class[str(cid)] = {
                "ground_truth_count": stats["gt"],
                "detection_count": stats["det"],
            }

        # These metrics compare the student detections against the teacher
        # pseudo-labels. There is no human ground truth in this repo, so this
        # is agreement, not accuracy.
        precisions = [
            r["student_vs_teacher_agreement"]["precision"]
            for r in self._records
            if "student_vs_teacher_agreement" in r
            and "precision" in r["student_vs_teacher_agreement"]
        ]
        recalls = [
            r["student_vs_teacher_agreement"]["recall"]
            for r in self._records
            if "student_vs_teacher_agreement" in r
            and "recall" in r["student_vs_teacher_agreement"]
        ]
        overall_metrics = {
            "precision": round(float(np.mean(precisions)), 4) if precisions else None,
            "recall": round(float(np.mean(recalls)), 4) if recalls else None,
            "note": "Student-vs-teacher pseudo-label agreement. No human ground truth exists.",
        }

        # Disagreement distribution and explainable harvest samples.
        disagreement_distribution = {
            "combined_histogram": _histogram(scores),
            "harvested_count_by_bin": _histogram([r["disagreement"]["combined"] for r in self._records if r["harvested"]]),
            "top_harvested_frames": [
                {
                    "clip_name": r.get("clip_name", ""),
                    "frame_idx": r.get("frame_idx", -1),
                    "combined": r["disagreement"]["combined"],
                    "margin": r["disagreement"]["margin"],
                    "entropy": r["disagreement"]["entropy"],
                    "class_mismatch": r["disagreement"]["class_mismatch"],
                    "iou": r["disagreement"]["iou"],
                    "harvest_reason": r.get("harvest_reason", "unknown"),
                }
                for r in sorted(
                    [r for r in self._records if r["harvested"]],
                    key=lambda x: x["disagreement"]["combined"],
                    reverse=True,
                )[:5]
            ],
            "bottom_non_harvested_frames": [
                {
                    "clip_name": r.get("clip_name", ""),
                    "frame_idx": r.get("frame_idx", -1),
                    "combined": r["disagreement"]["combined"],
                    "margin": r["disagreement"]["margin"],
                    "entropy": r["disagreement"]["entropy"],
                    "class_mismatch": r["disagreement"]["class_mismatch"],
                    "iou": r["disagreement"]["iou"],
                }
                for r in sorted(
                    [r for r in self._records if not r["harvested"]],
                    key=lambda x: x["disagreement"]["combined"],
                )[:5]
            ],
        }

        label_map_summary = self.label_map.summary() if self.label_map else {"name": "none"}

        summary = {
            "meta": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config_file": str(self.cfg.benchmark.subset_file),
                "label_map": label_map_summary,
                "total_frames": total,
                "harvested_count": harvested,
                "harvest_rate_pct": round(100.0 * harvested / total, 2) if total else 0.0,
                "elapsed_seconds": round(self._end_time - self._start_time, 3) if self._end_time and self._start_time else 0.0,
            },
            "disagreement": {
                "combined": _dist(scores),
                "margin": _dist(margins),
                "entropy": _dist(entropies),
                "class_mismatch": _dist(class_mismatches),
                "iou": _dist(ious),
                "distribution": disagreement_distribution,
            },
            "student_vs_teacher_agreement": overall_metrics,
            "per_class": per_class,
            "records": self._records,
        }
        return summary

    def _write_json(self, summary: Dict) -> None:
        # Detailed records go into the details file; summary is the small file.
        details = {k: v for k, v in summary.items() if k != "records"}
        details["records"] = summary["records"]
        with open(self.details_file, "w") as f:
            json.dump(details, f, indent=2)
        logger.info(f"Wrote detailed results to {self.details_file}")

        compact = {k: v for k, v in summary.items() if k != "records"}
        with open(self.summary_file, "w") as f:
            json.dump(compact, f, indent=2)
        logger.info(f"Wrote summary results to {self.summary_file}")

    def _write_disagreement_distribution(self, summary: Dict) -> None:
        path = self.results_dir / "disagreement_distribution.json"
        distribution = summary["disagreement"].get("distribution", {})
        with open(path, "w") as f:
            json.dump(distribution, f, indent=2)
        logger.info(f"Wrote disagreement distribution to {path}")

    def _write_summary_text(self, summary: Dict) -> None:
        path = self.results_dir / "benchmark_summary.txt"
        meta = summary["meta"]
        label_map = meta.get("label_map", {})
        dist = summary["disagreement"].get("distribution", {})
        lines = [
            "AECS-SDC Benchmark Summary",
            "=" * 40,
            f"Generated:       {meta['generated_at']}",
            f"Total frames:    {meta['total_frames']}",
            f"Harvested:       {meta['harvested_count']}",
            f"Harvest rate:    {meta['harvest_rate_pct']}%",
            f"Elapsed seconds: {meta['elapsed_seconds']}",
            "",
            "Label mapping",
            "-" * 40,
            f"  Name:          {label_map.get('name', 'none')}",
            f"  Identity map:  {label_map.get('is_identity', False)}",
            f"  Valid range:   {label_map.get('valid_range', [])}",
            "",
            "Disagreement distribution (combined score)",
            "-" * 40,
        ]
        for k, v in summary["disagreement"]["combined"].items():
            lines.append(f"  {k:6}: {v}")
        lines.extend([
            "",
            "Why frames were harvested (top drivers)",
            "-" * 40,
        ])
        for r in dist.get("top_harvested_frames", [])[:5]:
            lines.append(
                f"  {r['clip_name']} frame {r['frame_idx']}: "
                f"combined={r['combined']:.4f} margin={r['margin']:.4f} "
                f"entropy={r['entropy']:.4f} mismatch={r['class_mismatch']} "
                f"reason={r['harvest_reason']}"
            )
        lines.extend([
            "",
            "Student vs teacher pseudo-label agreement (not human ground truth)",
            "-" * 40,
            f"  Mean precision: {summary['student_vs_teacher_agreement']['precision']}",
            f"  Mean recall:    {summary['student_vs_teacher_agreement']['recall']}",
            f"  Note:           {summary['student_vs_teacher_agreement'].get('note', '')}",
            "",
            "Top per-class counts",
            "-" * 40,
        ])
        for cid, stats in sorted(summary["per_class"].items(), key=lambda x: x[1]["ground_truth_count"], reverse=True)[:10]:
            lines.append(f"  class {cid}: gt={stats['ground_truth_count']} det={stats['detection_count']}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"Wrote human-readable summary to {path}")
