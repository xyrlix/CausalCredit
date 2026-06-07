"""Unit tests for src.fairness.oaxaca (M8.5f).

Three test groups:

* Sanity — known artificial data with a clear gap recovers it
* Numerical — total = explained + unexplained, shares sum to 1
* Robustness — UNKNOWN exclusion, group_a / group_b override, single-group
  raises, insufficient data raises
* Visualization — chart file is produced
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness.oaxaca import (
    OaxacaBlinderResult,
    oaxaca_blinder_decomposition,
    plot_oaxaca_decomposition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_groups():
    """A controlled example: 200 rows, two groups with KNOWN feature means.

    Group A: higher income, higher credit, lower default.
    Group B: lower income, lower credit, higher default.
    Outcome: y = 0.6 - 8e-6·income + 5e-7·credit - 5e-5·days_employed + noise.
    The "real" gap (after noise averaging) is dominated by income — group A's
    higher income drives a substantially lower default probability.
    """
    rng = np.random.default_rng(7)
    n_each = 200
    feats_a = pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.normal(200_000, 30_000, n_each),
        "AMT_CREDIT": rng.normal(500_000, 50_000, n_each),
        "DAYS_EMPLOYED": rng.normal(-3000, 500, n_each),
    })
    feats_b = pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.normal(120_000, 25_000, n_each),
        "AMT_CREDIT": rng.normal(450_000, 50_000, n_each),
        "DAYS_EMPLOYED": rng.normal(-2000, 500, n_each),
    })
    X = pd.concat([feats_a, feats_b], ignore_index=True)
    groups = np.array(["A"] * n_each + ["B"] * n_each)
    beta = np.array([-0.000_008, 0.000_000_5, -0.000_05])
    intercept = 0.6
    noise = rng.normal(0, 0.05, 2 * n_each)
    y_score = intercept + X.values @ beta + noise
    y_prob = 1 / (1 + np.exp(-y_score))
    y = (rng.uniform(size=2 * n_each) < y_prob).astype(float)
    return y, X, groups


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------
class TestOaxacaBlinderSanity:
    def test_returns_result_with_required_fields(self, synthetic_groups):
        y, X, groups = synthetic_groups
        res = oaxaca_blinder_decomposition(y, X, groups)
        assert isinstance(res, OaxacaBlinderResult)
        assert res.group_a == "A"
        assert res.group_b == "B"
        assert res.n_a == 200
        assert res.n_b == 200
        assert isinstance(res.total_gap, float)
        assert isinstance(res.explained_gap, float)
        assert isinstance(res.unexplained_gap, float)
        assert isinstance(res.feature_contributions, pd.DataFrame)
        assert len(res.feature_contributions) == 3

    def test_total_equals_explained_plus_unexplained(self, synthetic_groups):
        y, X, groups = synthetic_groups
        res = oaxaca_blinder_decomposition(y, X, groups)
        recomb = res.explained_gap + res.unexplained_gap
        assert recomb == pytest.approx(res.total_gap, abs=1e-6)

    def test_discrimination_index_in_unit_interval(self, synthetic_groups):
        y, X, groups = synthetic_groups
        res = oaxaca_blinder_decomposition(y, X, groups)
        assert 0.0 <= res.discrimination_index <= 1.0

    def test_explained_unexplained_shares_sum_to_one(self, synthetic_groups):
        y, X, groups = synthetic_groups
        res = oaxaca_blinder_decomposition(y, X, groups)
        if abs(res.total_gap) > 1e-9:
            assert res.explained_share + res.unexplained_share == pytest.approx(1.0, abs=1e-6)

    def test_feature_contributions_are_numeric(self, synthetic_groups):
        y, X, groups = synthetic_groups
        res = oaxaca_blinder_decomposition(y, X, groups)
        for col in ("explained", "unexplained", "abs_total"):
            assert pd.api.types.is_numeric_dtype(res.feature_contributions[col])
        # Sorted descending by abs_total
        abs_totals = res.feature_contributions["abs_total"].values
        assert all(abs_totals[i] >= abs_totals[i + 1] for i in range(len(abs_totals) - 1))


# ---------------------------------------------------------------------------
# Numerical — close to known gap
# ---------------------------------------------------------------------------
class TestOaxacaBlinderNumerical:
    def test_gap_sign_matches_group_means(self, synthetic_groups):
        y, X, groups = synthetic_groups
        res = oaxaca_blinder_decomposition(y, X, groups)
        # Group A is higher-income → lower default → mean_y_a < mean_y_b
        # So total_gap should be NEGATIVE (B - A is positive)
        assert res.total_gap < 0
        assert res.mean_y_a < res.mean_y_b

    def test_explained_dominates_for_synthetic(self, synthetic_groups):
        y, X, groups = synthetic_groups
        res = oaxaca_blinder_decomposition(y, X, groups)
        # Most of the gap is "endowments" (different feature distributions)
        # — the coefficients are roughly the same. So |explained| >>
        # |unexplained|.
        # (This is the desired pattern: a fair world with group
        #  differences in features, but no coefficient bias.)
        assert abs(res.explained_gap) > abs(res.unexplained_gap)

    def test_pooled_reference_gives_same_identity(self, synthetic_groups):
        y, X, groups = synthetic_groups
        res_b = oaxaca_blinder_decomposition(y, X, groups, reference="B")
        res_pooled = oaxaca_blinder_decomposition(y, X, groups, reference="pooled")
        # Both satisfy total = explained + unexplained
        for r in (res_b, res_pooled):
            recomb = r.explained_gap + r.unexplained_gap
            assert recomb == pytest.approx(r.total_gap, abs=1e-6)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
class TestOaxacaBlinderRobustness:
    def test_unknown_excluded(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2])
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]})
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "UNKNOWN", "UNKNOWN"])
        res = oaxaca_blinder_decomposition(y, X, groups)
        # Should run with 5 + 5 = 10 observations (2 UNKNOWN dropped)
        assert res.n_a == 5
        assert res.n_b == 5

    def test_nan_groups_excluded(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2])
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]})
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", None, None])
        res = oaxaca_blinder_decomposition(y, X, groups)
        assert res.n_a == 5 and res.n_b == 5

    def test_explicit_group_a_b(self):
        y = np.array([0.1] * 10 + [0.2] * 10)
        X = pd.DataFrame({"f": [1.0] * 10 + [2.0] * 10})
        groups = np.array(["X"] * 10 + ["Y"] * 10)
        res = oaxaca_blinder_decomposition(y, X, groups, group_a="Y", group_b="X")
        assert res.group_a == "Y"
        assert res.group_b == "X"
        assert res.mean_y_a > res.mean_y_b  # Y has higher y

    def test_raises_on_too_few_observations(self):
        y = np.array([0.1, 0.2, 0.3, 0.4])
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        groups = np.array(["A", "A", "B", "B"])
        with pytest.raises(ValueError, match="at least 5"):
            oaxaca_blinder_decomposition(y, X, groups)

    def test_raises_on_single_group(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        groups = np.array(["A", "A", "A", "A", "A", "A"])
        with pytest.raises(ValueError, match="two distinct"):
            oaxaca_blinder_decomposition(y, X, groups)

    def test_raises_on_bad_reference(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5])
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        with pytest.raises(ValueError, match="reference"):
            oaxaca_blinder_decomposition(y, X, groups, reference="bogus")

    def test_length_mismatch_raises(self):
        y = np.array([0.1, 0.2, 0.3, 0.4])
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0]})  # 3 rows
        groups = np.array(["A", "A", "B", "B"])
        with pytest.raises(ValueError, match="same length"):
            oaxaca_blinder_decomposition(y, X, groups)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
class TestOaxacaVisualization:
    def test_plot_writes_png(self, synthetic_groups, tmp_path):
        y, X, groups = synthetic_groups
        res = oaxaca_blinder_decomposition(y, X, groups)
        out = tmp_path / "oaxaca.png"
        plot_oaxaca_decomposition(res, str(out))
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG
