"""Unit tests for src.causal.stability (M8.6c).

Coverage:

* Sanity — both tiers run on a controlled DGP
* Tier1 — 30× split-half bootstrap returns a sensible mean Spearman
* Tier2 — 10× hyperparameter grid returns a sensible min pairwise
* Robustness — invalid args raise, to_dict roundtrips, small data raises
* Visualization — chart file is produced
"""

from __future__ import annotations

import numpy as np
import pytest

from src.causal.stability import CATEStabilityTester, StabilityResult, plot_stability


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dgp_strong():
    """A DGP with strong heterogeneous signal — Tier1 ρ should be high."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 4))
    T = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(X[:, 0] + X[:, 1])))).astype(float)
    y = (
        1.0
        + 0.5 * T
        + 0.3 * X[:, 0]
        + 0.3 * X[:, 0] * T
        + rng.normal(scale=0.3, size=n)
    )
    return y, T, X


@pytest.fixture
def dgp_continuous():
    """A continuous-treatment DGP — T is real-valued."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 4))
    T = X[:, 0] + X[:, 1] + rng.normal(scale=0.3, size=n)
    y = 1.0 + 0.5 * T + 0.3 * X[:, 0] * T + rng.normal(scale=0.3, size=n)
    return y, T, X


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------
class TestStabilitySanity:
    def test_returns_stability_result(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=3, random_state=0,
        )
        res = tester.run(y, T, X)
        assert isinstance(res, StabilityResult)
        assert res.method == "LinearDML"
        assert isinstance(res.summary, str)
        assert "STABLE" in res.summary or "UNSTABLE" in res.summary

    def test_summary_contains_method(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=3, random_state=0,
        )
        res = tester.run(y, T, X)
        assert "LinearDML" in res.summary
        assert "Tier1" in res.summary
        assert "Tier2" in res.summary

    def test_to_dict_roundtrip(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=3, random_state=0,
        )
        res = tester.run(y, T, X)
        d = res.to_dict()
        assert "tier1" in d
        assert "tier2" in d
        assert "overall_pass" in d
        assert "summary" in d
        assert "mean_spearman" in d["tier1"]
        assert "min_pairwise_spearman" in d["tier2"]


# ---------------------------------------------------------------------------
# Tier 1 — split-half bootstrap
# ---------------------------------------------------------------------------
class TestTier1:
    def test_returns_dict(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=5, n_configs=3, random_state=0,
        )
        out = tester.tier1_split_half(y, T, X)
        assert "mean_spearman" in out
        assert "per_iter" in out
        assert "pass" in out
        assert len(out["per_iter"]) == 5
        assert -1.0 <= out["mean_spearman"] <= 1.0

    def test_strong_signal_yields_positive_correlation(self, dgp_continuous):
        y, T, X = dgp_continuous
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=10, n_configs=3, random_state=0,
        )
        out = tester.tier1_split_half(y, T, X)
        # With a strong signal and a stable method, the mean should
        # be strictly positive.  (Conservative — we don't require > 0.8.)
        assert out["mean_spearman"] > 0.0

    def test_pass_field_is_boolean(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=3, random_state=0,
        )
        out = tester.tier1_split_half(y, T, X)
        assert isinstance(out["pass"], bool)


# ---------------------------------------------------------------------------
# Tier 2 — hyperparameter sensitivity
# ---------------------------------------------------------------------------
class TestTier2:
    def test_returns_dict(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=5, random_state=0,
        )
        out = tester.tier2_hyperparameter_sensitivity(y, T, X)
        assert "min_pairwise_spearman" in out
        assert "pairwise" in out
        assert "pass" in out
        # 5 configs -> C(5, 2) = 10 pairs
        assert len(out["pairwise"]) == 10
        assert -1.0 <= out["min_pairwise_spearman"] <= 1.0

    def test_pairwise_records_include_indices(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=4, random_state=0,
        )
        out = tester.tier2_hyperparameter_sensitivity(y, T, X)
        for p in out["pairwise"]:
            assert "i" in p
            assert "j" in p
            assert "spearman" in p
            assert "config_i" in p
            assert "config_j" in p

    def test_strong_signal_yields_high_pairwise(self, dgp_continuous):
        y, T, X = dgp_continuous
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=5, random_state=0,
        )
        out = tester.tier2_hyperparameter_sensitivity(y, T, X)
        # All pairwise Spearman should be positive for a strong signal
        for p in out["pairwise"]:
            assert p["spearman"] > 0.0


# ---------------------------------------------------------------------------
# Method variation
# ---------------------------------------------------------------------------
class TestMethods:
    def test_linear_dml_runs(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=3, random_state=0,
        )
        res = tester.run(y, T, X)
        assert res.method == "LinearDML"

    def test_causal_forest_runs(self, dgp_strong):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="CausalForestDML", n_bootstrap=3, n_configs=3, random_state=0,
        )
        res = tester.run(y, T, X)
        assert res.method == "CausalForestDML"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            CATEStabilityTester(method="NotAMethod")


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_raises_on_length_mismatch(self):
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=3,
        )
        with pytest.raises(ValueError, match="same length"):
            tester.run(np.zeros(10), np.zeros(8), np.zeros((10, 2)))

    def test_raises_on_too_few_observations(self):
        y = np.zeros(50)
        T = np.zeros(50)
        X = np.zeros((50, 2))
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=3,
        )
        with pytest.raises(ValueError, match="at least 100"):
            tester.run(y, T, X)

    def test_raises_on_invalid_n_bootstrap(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            CATEStabilityTester(method="LinearDML", n_bootstrap=1)

    def test_raises_on_invalid_n_configs(self):
        with pytest.raises(ValueError, match="n_configs"):
            CATEStabilityTester(method="LinearDML", n_configs=1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
class TestVisualization:
    def test_plot_writes_png(self, dgp_strong, tmp_path):
        y, T, X = dgp_strong
        tester = CATEStabilityTester(
            method="LinearDML", n_bootstrap=3, n_configs=3, random_state=0,
        )
        res = tester.run(y, T, X)
        out = tmp_path / "stability.png"
        plot_stability(res, str(out))
        assert out.exists()
        assert out.stat().st_size > 1000
