"""Interest-rate optimizer: sleeping-dogs / rate-sensitive segmentation.

@requirement REQ-BIZ-003
@design docs/plans/2026-06-05-causalcredit-architecture-design.md §D7

For each applicant we run a counterfactual sweep across a grid of
"effective rates" — encoded as ``(AMT_ANNUITY / AMT_CREDIT) * 12`` —
and observe how the model's predicted P(default) responds.

Three segments (industry-standard retail-banking taxonomy):

* **sleeping_dog**   — currently low risk (P<5%) but a small rate CUT
                      would meaningfully reduce risk AND the segment is
                      under-priced. These are the highest-LTV
                      opportunities: a 50bps rate cut grows the book
                      without inflating loss.
* **rate_sensitive** — P(default) moves > 1 percentage point per 1pp
                      rate change. Standard price-elasticity band;
                      price carefully or risk adverse selection.
* **neutral**        — small response; rate decisions are dominated by
                      margin considerations.

The optimizer also recommends a rate that maximizes expected profit
``E[profit] = (1 - p) * revenue - p * LGD - cost_of_funds * principal``,
sweeping a discrete rate grid and picking the argmax.

NOTE: this module is a *demonstration* of the segmentation. The
"rate" is derived from AMT_ANNUITY / AMT_CREDIT — Home Credit has no
explicit APR column. For production, swap the simulation for a
rate-elasticity model trained on historical repricing data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("causalcredit.pricing")

# Segment labels
SEGMENT_SLEEPING_DOG = "sleeping_dog"
SEGMENT_RATE_SENSITIVE = "rate_sensitive"
SEGMENT_NEUTRAL = "neutral"

# Default grids
DEFAULT_RATE_GRID: tuple = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15)
DEFAULT_LGD = 0.45  # loss given default (Home Credit–typical)
DEFAULT_COST_OF_FUNDS = 0.025  # 2.5% — roughly HKMA HIBOR


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RateOptimization:
    """Per-applicant rate-optimization result."""
    applicant_id: Optional[str]
    base_rate: float
    base_pd: float
    rate_grid: List[float]
    pd_grid: List[float]
    elasticity: float  # ΔP per 1pp rate change around the base
    segment: str  # sleeping_dog / rate_sensitive / neutral
    segment_reasons: List[str] = field(default_factory=list)
    recommended_rate: float = 0.0
    expected_profit_at_recommended: float = 0.0
    expected_profit_at_base: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "applicant_id": self.applicant_id,
            "base_rate": self.base_rate,
            "base_pd": self.base_pd,
            "rate_grid": list(self.rate_grid),
            "pd_grid": list(self.pd_grid),
            "elasticity": self.elasticity,
            "segment": self.segment,
            "segment_reasons": list(self.segment_reasons),
            "recommended_rate": self.recommended_rate,
            "expected_profit_at_recommended": self.expected_profit_at_recommended,
            "expected_profit_at_base": self.expected_profit_at_base,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def annualized_rate(amt_annuity: float, amt_credit: float) -> float:
    """Effective annual rate from annuity/credit ratio.

    ``rate = (annuity / credit) * 12``  — matches the standard loan
    formula when annuity is the yearly payment.
    """
    if amt_credit <= 0:
        return 0.0
    return float(amt_annuity) / float(amt_credit) * 12.0


def compute_pd_grid(
    *,
    model,
    features: Dict[str, float],
    base_annuity_key: str,
    credit_key: str,
    rate_grid: Sequence[float],
    registry=None,
    feature_cols: Optional[Sequence[str]] = None,
) -> List[float]:
    """For each target rate, set ``AMT_ANNUITY`` = rate*credit/12 and
    predict P(default). Returns a list of probabilities aligned with
    ``rate_grid``.

    Uses :func:`registry.transform_features` if available, else builds a
    DataFrame restricted to ``feature_cols`` (so the sklearn
    feature-name check is satisfied).
    """
    credit = float(features[credit_key])
    pd_out: List[float] = []
    for r in rate_grid:
        annuity = credit * r / 12.0
        new_features = dict(features)
        new_features[base_annuity_key] = annuity
        if registry is not None:
            X = registry.transform_features(new_features)
        elif feature_cols is not None:
            # Restrict to the columns the model was trained on
            X = pd.DataFrame([{k: new_features[k] for k in feature_cols}])
        else:
            X = pd.DataFrame([new_features])
        p = float(model.predict_proba(X)[:, 1][0])
        pd_out.append(p)
    return pd_out


def compute_elasticity(rate_grid: Sequence[float], pd_grid: Sequence[float]) -> float:
    """First-order elasticity: slope of P(default) vs rate, around the
    centre of the grid. Units: percentage-point change in P per 1pp
    rate change.
    """
    r = np.asarray(rate_grid, dtype=float)
    p = np.asarray(pd_grid, dtype=float)
    if len(r) < 2 or np.std(r) < 1e-9:
        return 0.0
    # Linear fit; slope * 1pp (0.01)
    slope, _ = np.polyfit(r, p, 1)
    return float(slope * 0.01)


def expected_profit(
    *,
    rate: float,
    pd_value: float,
    credit: float,
    lgd: float = DEFAULT_LGD,
    cost_of_funds: float = DEFAULT_COST_OF_FUNDS,
) -> float:
    """Expected profit per dollar of credit over a 1-year horizon.

    ``E[profit] = (1 - p) * (rate - cost_of_funds) - p * LGD``
    per unit principal, scaled by ``credit``.
    """
    p = float(pd_value)
    revenue = (1.0 - p) * max(0.0, rate - cost_of_funds)
    loss = p * lgd
    return float((revenue - loss) * credit)


def classify_segment(
    *,
    base_pd: float,
    elasticity: float,
    base_rate: float,
    pd_grid: Sequence[float],
    rate_grid: Sequence[float],
    low_pd_threshold: float = 0.05,
    high_elasticity_threshold: float = 0.005,  # 0.5pp per 1pp rate
) -> tuple:
    """Classify into sleeping_dog / rate_sensitive / neutral.

    Returns ``(segment, reasons)``.
    """
    reasons: List[str] = []
    p = float(base_pd)
    e = float(elasticity)
    abs_e = abs(e)

    # Rate-sensitivity dominates first
    if abs_e >= high_elasticity_threshold:
        reasons.append(
            f"|elasticity|={abs_e:.4f} ≥ {high_elasticity_threshold} — "
            f"P(default) moves substantially with rate"
        )
        return SEGMENT_RATE_SENSITIVE, reasons

    # Sleeping-dog: low risk + we could lower rate and P drops further
    if p < low_pd_threshold:
        # Check if a rate cut would meaningfully reduce P
        if e > 0:
            # Higher rate → higher P. So lowering rate would lower P.
            # Is the drop material?
            r = np.asarray(rate_grid, dtype=float)
            pg = np.asarray(pd_grid, dtype=float)
            if base_rate > r[0]:
                base_idx = int(np.argmin(np.abs(r - base_rate)))
                low_idx = int(np.argmin(r))
                pd_drop = pg[base_idx] - pg[low_idx]
                if pd_drop > 0.002:  # 0.2pp drop material
                    reasons.append(
                        f"P(default)={p:.4f} < {low_pd_threshold} (low risk) and "
                        f"a rate cut from {base_rate:.3f} → {r[low_idx]:.3f} reduces "
                        f"P(default) by {pd_drop:.4f} — under-priced opportunity"
                    )
                    return SEGMENT_SLEEPING_DOG, reasons

    reasons.append(
        f"|elasticity|={abs_e:.4f} < {high_elasticity_threshold} and "
        f"P(default)={p:.4f} — small response to rate"
    )
    return SEGMENT_NEUTRAL, reasons


def pick_recommended_rate(
    *,
    rate_grid: Sequence[float],
    pd_grid: Sequence[float],
    credit: float,
    lgd: float = DEFAULT_LGD,
    cost_of_funds: float = DEFAULT_COST_OF_FUNDS,
) -> tuple:
    """Pick the rate that maximizes expected profit on the grid.

    Returns ``(recommended_rate, expected_profit)``.
    """
    best_rate = float(rate_grid[0])
    best_profit = -np.inf
    for r, p in zip(rate_grid, pd_grid):
        prof = expected_profit(
            rate=float(r), pd_value=float(p), credit=credit,
            lgd=lgd, cost_of_funds=cost_of_funds,
        )
        if prof > best_profit:
            best_profit = prof
            best_rate = float(r)
    return best_rate, float(best_profit)


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------
class RateOptimizer:
    """End-to-end rate optimization for a single applicant."""

    def __init__(
        self,
        model,
        *,
        feature_cols: Optional[List[str]] = None,
        registry=None,
        rate_grid: Sequence[float] = DEFAULT_RATE_GRID,
        lgd: float = DEFAULT_LGD,
        cost_of_funds: float = DEFAULT_COST_OF_FUNDS,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols or []
        self.registry = registry
        self.rate_grid = tuple(rate_grid)
        self.lgd = lgd
        self.cost_of_funds = cost_of_funds

    def score_applicant(
        self,
        features: Dict[str, float],
        *,
        applicant_id: Optional[str] = None,
        credit_key: str = "AMT_CREDIT",
        annuity_key: str = "AMT_ANNUITY",
    ) -> RateOptimization:
        credit = float(features[credit_key])
        base_annuity = float(features[annuity_key])
        base_rate = annualized_rate(base_annuity, credit)

        pd_grid = compute_pd_grid(
            model=self.model,
            features=features,
            base_annuity_key=annuity_key,
            credit_key=credit_key,
            rate_grid=self.rate_grid,
            registry=self.registry,
            feature_cols=self.feature_cols or None,
        )
        base_pd = float(self._predict_single(features))

        elasticity = compute_elasticity(self.rate_grid, pd_grid)
        segment, reasons = classify_segment(
            base_pd=base_pd,
            elasticity=elasticity,
            base_rate=base_rate,
            rate_grid=self.rate_grid,
            pd_grid=pd_grid,
        )

        # Recommended rate = grid argmax of expected profit
        rec_rate, rec_profit = pick_recommended_rate(
            rate_grid=self.rate_grid,
            pd_grid=pd_grid,
            credit=credit,
            lgd=self.lgd,
            cost_of_funds=self.cost_of_funds,
        )
        base_profit = expected_profit(
            rate=base_rate, pd_value=base_pd, credit=credit,
            lgd=self.lgd, cost_of_funds=self.cost_of_funds,
        )

        return RateOptimization(
            applicant_id=applicant_id,
            base_rate=float(base_rate),
            base_pd=float(base_pd),
            rate_grid=list(self.rate_grid),
            pd_grid=list(pd_grid),
            elasticity=float(elasticity),
            segment=segment,
            segment_reasons=reasons,
            recommended_rate=rec_rate,
            expected_profit_at_recommended=rec_profit,
            expected_profit_at_base=base_profit,
        )

    def _predict_single(self, features: Dict[str, float]) -> float:
        if self.registry is not None:
            X = self.registry.transform_features(features)
        else:
            X = pd.DataFrame([features])
        return float(self.model.predict_proba(X)[:, 1][0])

    # Convenience: batch over a DataFrame
    def score_batch(self, df: pd.DataFrame, **kwargs) -> List[RateOptimization]:
        return [
            self.score_applicant(row.to_dict(), applicant_id=str(idx), **kwargs)
            for idx, row in df.iterrows()
        ]
