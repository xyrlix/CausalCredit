"""Pricing package — rate optimizer and related modules.

@requirement REQ-BIZ-003
"""
from .rate_optimizer import (
    DEFAULT_LGD,
    DEFAULT_RATE_GRID,
    SEGMENT_NEUTRAL,
    SEGMENT_RATE_SENSITIVE,
    SEGMENT_SLEEPING_DOG,
    RateOptimization,
    RateOptimizer,
    annualized_rate,
    classify_segment,
    compute_elasticity,
    compute_pd_grid,
    expected_profit,
    pick_recommended_rate,
)

__all__ = [
    "DEFAULT_LGD",
    "DEFAULT_RATE_GRID",
    "SEGMENT_NEUTRAL",
    "SEGMENT_RATE_SENSITIVE",
    "SEGMENT_SLEEPING_DOG",
    "RateOptimization",
    "RateOptimizer",
    "annualized_rate",
    "classify_segment",
    "compute_elasticity",
    "compute_pd_grid",
    "expected_profit",
    "pick_recommended_rate",
]
