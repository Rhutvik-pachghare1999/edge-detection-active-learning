"""Unit tests for harvest decision policies."""

import pytest

from aecs_sdc.config import HarvestConfig
from aecs_sdc.harvest import (
    BudgetPolicy,
    DiversityPolicy,
    ThresholdPolicy,
    build_policy,
)


def test_threshold_policy_harvests_when_score_exceeds_tau():
    policy = ThresholdPolicy(tau_initial=0.5, tau_min=0.1, tau_max=0.9, target_rate=None)
    decision = policy.decide(0.6)
    assert decision.harvested is True
    assert decision.score == pytest.approx(0.6)


def test_threshold_policy_skips_when_score_below_tau():
    policy = ThresholdPolicy(tau_initial=0.5, tau_min=0.1, tau_max=0.9, target_rate=None)
    decision = policy.decide(0.4)
    assert decision.harvested is False


def test_budget_policy_selects_top_k():
    policy = BudgetPolicy(budget=2)
    for i in range(5):
        policy.decide(f"frame_{i}", float(i) * 0.1)
    decisions = policy.finalize()
    assert decisions["frame_4"].harvested is True
    assert decisions["frame_3"].harvested is True
    assert decisions["frame_0"].harvested is False


def test_diversity_policy_respects_class_quota():
    policy = DiversityPolicy(budget_per_class=1, tau=0.1)
    # First frame fills the least-seen class (0), class 1 remains available.
    assert policy.decide("a", 0.2, [0, 1]).harvested is True
    assert policy.decide("b", 0.2, [0]).harvested is False  # class 0 budget exhausted
    assert policy.decide("c", 0.2, [1]).harvested is True   # class 1 still available
    assert policy.decide("d", 0.05, [1]).harvested is False  # below tau
    assert policy.decide("e", 0.2, [1]).harvested is False  # class 1 budget now exhausted


def test_build_policy_threshold():
    cfg = HarvestConfig(mode="threshold", target_rate=0.1)
    policy = build_policy(cfg)
    assert isinstance(policy, ThresholdPolicy)


def test_build_policy_budget_requires_positive_budget():
    cfg = HarvestConfig(mode="budget", budget=None)
    with pytest.raises(ValueError):
        build_policy(cfg)
