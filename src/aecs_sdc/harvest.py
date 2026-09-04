"""Harvest decision policies.

Policies decide which frames are kept as training data. The benchmark supports:

* threshold  — keep frames whose disagreement score exceeds tau
* budget     — keep the top-K frames by disagreement
* diversity  — keep frames to balance classes seen in teacher detections
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class HarvestDecision:
    harvested: bool
    reason: str
    score: float
    tau: float


class ThresholdPolicy:
    def __init__(self, tau_initial: float, tau_min: float, tau_max: float,
                 target_rate: Optional[float] = None, window_size: int = 100):
        self.tau = float(tau_initial)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.target_rate = target_rate
        self.window_size = window_size
        self._history: List[bool] = []

    def decide(self, score: float) -> HarvestDecision:
        harvested = score > self.tau
        reason = f"score {score:.4f} > tau {self.tau:.4f}"
        if not harvested:
            reason = f"score {score:.4f} <= tau {self.tau:.4f}"

        self._history.append(harvested)
        if len(self._history) > self.window_size:
            self._history.pop(0)

        if self.target_rate is not None and len(self._history) >= 20:
            # Simple proportional controller: if rate is above target, raise tau.
            rate = sum(self._history) / len(self._history)
            error = rate - self.target_rate
            adjustment = 0.10 * error  # small gain to avoid oscillation
            self.tau = float(max(self.tau_min, min(self.tau_max, self.tau + adjustment)))
            logger.debug(f"PID-like update: rate={rate:.3f}, target={self.target_rate:.3f}, tau={self.tau:.4f}")

        return HarvestDecision(harvested=harvested, reason=reason, score=score, tau=self.tau)


class BudgetPolicy:
    def __init__(self, budget: int):
        self.budget = int(budget)
        self._buffer: List[Dict] = []
        self._finalized = False
        self._selected_ids = set()

    def decide(self, frame_id: str, score: float) -> HarvestDecision:
        self._buffer.append({"frame_id": frame_id, "score": score})
        # Defer final decision until finalize() is called.
        return HarvestDecision(
            harvested=False,
            reason=f"budget policy: candidate score {score:.4f}",
            score=score,
            tau=0.0,
        )

    def finalize(self) -> Dict[str, HarvestDecision]:
        """Select top-K by score and return decisions for all buffered frames."""
        sorted_by_score = sorted(self._buffer, key=lambda x: x["score"], reverse=True)
        top_ids = {item["frame_id"] for item in sorted_by_score[:self.budget]}
        decisions = {}
        for item in self._buffer:
            harvested = item["frame_id"] in top_ids
            decisions[item["frame_id"]] = HarvestDecision(
                harvested=harvested,
                reason=(
                    f"top-{self.budget} by disagreement score"
                    if harvested else "below budget cutoff"
                ),
                score=item["score"],
                tau=0.0,
            )
        self._finalized = True
        self._selected_ids = top_ids
        return decisions


class DiversityPolicy:
    def __init__(self, budget_per_class: int = 2, tau: float = 0.15):
        self.budget_per_class = budget_per_class
        self.tau = tau
        self._selected_per_class: Counter = Counter()

    def decide(self, frame_id: str, score: float, teacher_class_ids: List[int]) -> HarvestDecision:
        if score <= self.tau:
            return HarvestDecision(
                harvested=False,
                reason=f"score {score:.4f} <= tau {self.tau:.4f}",
                score=score,
                tau=self.tau,
            )
        # Pick the class with fewest selections among the teacher's top classes.
        candidate_class = min(
            teacher_class_ids,
            key=lambda c: self._selected_per_class[c],
            default=-1,
        )
        if candidate_class >= 0 and self._selected_per_class[candidate_class] < self.budget_per_class:
            self._selected_per_class[candidate_class] += 1
            return HarvestDecision(
                harvested=True,
                reason=f"diversity quota for class {candidate_class}",
                score=score,
                tau=self.tau,
            )
        return HarvestDecision(
            harvested=False,
            reason="diversity quota reached for all classes in frame",
            score=score,
            tau=self.tau,
        )


def build_policy(cfg) -> object:
    """Factory that builds a harvest policy from the config object."""
    if cfg.mode == "threshold":
        return ThresholdPolicy(
            tau_initial=cfg.tau_initial,
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
            target_rate=cfg.target_rate,
        )
    if cfg.mode == "budget":
        if cfg.budget is None or cfg.budget <= 0:
            raise ValueError("budget mode requires a positive harvest.budget value")
        return BudgetPolicy(budget=cfg.budget)
    if cfg.mode == "diversity":
        return DiversityPolicy(
            budget_per_class=cfg.budget or 2,
            tau=cfg.tau_initial,
        )
    raise ValueError(f"Unknown harvest mode: {cfg.mode}")
