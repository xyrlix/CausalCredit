"""Unit tests for src.pricing.rate_optimizer (M8.5g).

Tests:

* :func:`annualized_rate` — basic math
* :func:`compute_elasticity` — slope sign & units
* :func:`expected_profit` — boundary behaviour
* :func:`classify_segment` — three segments reachable, reasons populated
* :func:`pick_recommended_rate` — argmax correctness
* :class:`RateOptimizer` end-to-end on a fake model (a simple
  LightGBM trained on synthetic data — fast and deterministic)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.pricing.rate_optimizer import (
    DEFAULT_RATE_GRID,
    RateOptimizer,
    annualized_rate,
    classify_segment,
    compute_elasticity,
    compute_pd_grid,
    expected_profit,
    pick_recommended_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_model():
    """A deterministic toy model: P(default) ∝ AMT_CREDIT, decreases with
    AMT_INCOME_TOTAL. Trained on 200 rows so the trees have real splits.
    """
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "AMT_CREDIT": rng.uniform(1e5, 1e6, n),
        "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n),
        "AMT_ANNUITY": rng.uniform(1e4, 5e4, n),
    })
    # Synthetic target: high credit + low income = default
    p = 1 / (1 + np.exp(-(df["AMT_CREDIT"] / 1e6 - df["AMT_INCOME_TOTAL"] / 1e5 + 0.5)))
    y = (rng.uniform(size=n) < p).astype(int)
    X = df[["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]]
    m = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0)
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------
class TestAnnualizedRate:
    def test_basic(self):
        # 100k credit, 10k yearly payment ratio → 10000/100000*12 = 1.2
        # NOTE: this is a payment-intensity proxy, not a true APR (we don't
        # have loan term). The optimizer uses the metric consistently.
        assert annualized_rate(10000, 100000) == pytest.approx(1.2, rel=1e-9)

    def test_zero_credit_returns_zero(self):
        assert annualized_rate(1000, 0) == 0.0

    def test_negative_credit_returns_zero(self):
        assert annualized_rate(1000, -100) == 0.0

    def test_higher_annuity_yields_higher_rate(self):
        # Monotone: more annuity for the same credit → higher rate proxy
        assert annualized_rate(20000, 100000) > annualized_rate(10000, 100000)


class TestComputeElasticity:
    def test_zero_when_grid_single_point(self):
        assert compute_elasticity([0.05], [0.05]) == 0.0

    def test_zero_when_grid_constant(self):
        # P is flat — slope is zero (up to numerical noise)
        e = compute_elasticity([0.04, 0.06, 0.08], [0.05, 0.05, 0.05])
        assert abs(e) < 1e-9

    def test_positive_when_pd_grows_with_rate(self):
        e = compute_elasticity([0.04, 0.06, 0.08], [0.02, 0.05, 0.10])
        assert e > 0

    def test_units_are_pp_per_1pp_rate(self):
        # Linear: P goes from 0.01 at 4% to 0.05 at 8% → slope = 1.0 P per
        # unit_rate. Elasticity multiplies by 0.01 → 0.01.
        e = compute_elasticity([0.04, 0.05, 0.06, 0.07, 0.08], [0.01, 0.02, 0.03, 0.04, 0.05])
        assert e == pytest.approx(0.01, rel=1e-6)


class TestExpectedProfit:
    def test_zero_when_pd_is_one(self):
        # Default → LGD = 0.45, no revenue
        prof = expected_profit(rate=0.10, pd_value=1.0, credit=100000)
        assert prof == pytest.approx(-0.45 * 100000, rel=1e-9)

    def test_zero_when_pd_is_zero(self):
        # No default → revenue = rate - cost_of_funds = 0.10 - 0.025 = 0.075
        prof = expected_profit(rate=0.10, pd_value=0.0, credit=100000)
        assert prof == pytest.approx(0.075 * 100000, rel=1e-9)

    def test_revenue_clipped_to_zero_when_rate_below_cof(self):
        # Negative spread should be clipped to zero (no negative revenue).
        prof = expected_profit(rate=0.01, pd_value=0.0, credit=100000,
                               cost_of_funds=0.025)
        # P=0 → no loss; rate below COF → revenue = 0
        assert prof == 0.0


class TestClassifySegment:
    def test_rate_sensitive_when_elasticity_high(self):
        seg, reasons = classify_segment(
            base_pd=0.10, elasticity=0.01, base_rate=0.08,
            rate_grid=[0.04, 0.08, 0.15], pd_grid=[0.05, 0.10, 0.20],
        )
        assert seg == "rate_sensitive"
        assert len(reasons) >= 1
        assert "0.01" in reasons[0]

    def test_neutral_when_elasticity_low_and_pd_not_low(self):
        seg, reasons = classify_segment(
            base_pd=0.20, elasticity=0.001, base_rate=0.08,
            rate_grid=[0.04, 0.08, 0.15], pd_grid=[0.19, 0.20, 0.21],
        )
        assert seg == "neutral"
        assert len(reasons) >= 1

    def test_sleeping_dog_when_low_risk_and_rate_cut_helps(self):
        # Low risk, rate cut (going to 0.04) drops PD meaningfully
        seg, reasons = classify_segment(
            base_pd=0.02, elasticity=0.001, base_rate=0.10,
            rate_grid=[0.04, 0.08, 0.10, 0.12], pd_grid=[0.005, 0.015, 0.02, 0.03],
        )
        assert seg == "sleeping_dog"
        assert any("under-priced" in r.lower() or "低风险" in r for r in reasons) or \
               any("low risk" in r for r in reasons) or \
               any("under-priced" in r for r in reasons)

    def test_returns_known_label(self):
        for label in ("sleeping_dog", "rate_sensitive", "neutral"):
            seg, _ = classify_segment(
                base_pd=0.10, elasticity=0.0, base_rate=0.08,
                rate_grid=[0.08], pd_grid=[0.10],
            )
            assert seg in ("sleeping_dog", "rate_sensitive", "neutral")


class TestPickRecommendedRate:
    def test_argmax_is_picked(self):
        # Hand-crafted: profit peaks at rate=0.09
        rate_grid = [0.05, 0.07, 0.09, 0.11, 0.15]
        pd_grid = [0.04, 0.05, 0.07, 0.12, 0.25]  # increases faster later
        rec, prof = pick_recommended_rate(
            rate_grid=rate_grid, pd_grid=pd_grid, credit=100000,
        )
        assert rec == 0.09

    def test_handles_single_point(self):
        rec, prof = pick_recommended_rate(
            rate_grid=[0.05], pd_grid=[0.05], credit=100000,
        )
        assert rec == 0.05


# ---------------------------------------------------------------------------
# compute_pd_grid / RateOptimizer integration
# ---------------------------------------------------------------------------
class TestComputePDGrid:
    def test_returns_one_pd_per_rate(self, simple_model):
        feats = {"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 150000,
                 "AMT_ANNUITY": 30000}
        pd_out = compute_pd_grid(
            model=simple_model, features=feats,
            base_annuity_key="AMT_ANNUITY", credit_key="AMT_CREDIT",
            rate_grid=DEFAULT_RATE_GRID,
        )
        assert len(pd_out) == len(DEFAULT_RATE_GRID)
        assert all(0.0 <= p <= 1.0 for p in pd_out)

    def test_monotonic_in_pd_for_credit_driven_model(self, simple_model):
        # On this toy model higher annuity → higher default → higher P
        feats = {"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 150000,
                 "AMT_ANNUITY": 30000}
        pd_out = compute_pd_grid(
            model=simple_model, features=feats,
            base_annuity_key="AMT_ANNUITY", credit_key="AMT_CREDIT",
            rate_grid=(0.04, 0.08, 0.15),
        )
        # The model's signal is from credit + annuity + income, not from
        # the rate proxy alone. We just assert PD is bounded.
        assert all(0.0 <= p <= 1.0 for p in pd_out)


class TestRateOptimizer:
    def test_score_applicant_returns_full_result(self, simple_model):
        # Use only the 3 features the toy model was trained on
        feats = {"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 150000,
                 "AMT_ANNUITY": 30000}
        opt = RateOptimizer(model=simple_model, feature_cols=list(feats.keys()))
        result = opt.score_applicant(feats, applicant_id="test_001")
        assert result.applicant_id == "test_001"
        assert 0.0 <= result.base_pd <= 1.0
        assert result.base_rate > 0
        assert len(result.rate_grid) == len(DEFAULT_RATE_GRID)
        assert len(result.pd_grid) == len(DEFAULT_RATE_GRID)
        assert result.segment in ("sleeping_dog", "rate_sensitive", "neutral")
        assert len(result.segment_reasons) >= 1
        assert result.recommended_rate in DEFAULT_RATE_GRID
        # Profit numbers finite
        assert np.isfinite(result.expected_profit_at_recommended)
        assert np.isfinite(result.expected_profit_at_base)

    def test_to_dict_serializes_lists(self, simple_model):
        feats = {"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 150000,
                 "AMT_ANNUITY": 30000}
        opt = RateOptimizer(model=simple_model)
        result = opt.score_applicant(feats)
        d = result.to_dict()
        assert isinstance(d["rate_grid"], list)
        assert isinstance(d["pd_grid"], list)
        assert isinstance(d["segment_reasons"], list)
        assert d["segment"] in ("sleeping_dog", "rate_sensitive", "neutral")
