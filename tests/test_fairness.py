"""Unit tests for src.fairness.metrics and src.fairness.slicing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth():
    """Synthesize a 1000-row test set with a clear gender bias."""
    rng = np.random.default_rng(0)
    n = 1000
    gender = rng.choice(["M", "F", "UNKNOWN"], n, p=[0.45, 0.45, 0.10])
    y_true = rng.choice([0, 1], n, p=[0.92, 0.08])
    # Inject bias: F gets a +0.15 boost in positive rate
    p_pred = 0.08 + 0.15 * (gender == "F")
    y_pred = (rng.uniform(size=n) < p_pred).astype(int)
    y_score = p_pred + rng.normal(scale=0.05, size=n)
    return y_true, y_pred, y_score, gender


# ----------------------------------------------------------------- metrics


def test_demographic_parity_gap_zero_when_perfect_parity(synth):
    from src.fairness.metrics import demographic_parity_gap
    y_t, y_p, y_s, g = synth
    # Same selection rate for M and F
    y_p_const = np.where(g == "F", 0, 1)
    y_p_const = np.where(g == "UNKNOWN", 0, y_p_const)
    # Set both to sel rate 0.1
    y_p_const = np.full_like(y_p, 0)
    y_p_const[g == "F"] = 1
    y_p_const[g == "M"] = 1
    gap = demographic_parity_gap(y_p_const, g)
    assert gap < 0.01


def test_demographic_parity_gap_detects_bias(synth):
    from src.fairness.metrics import demographic_parity_gap
    y_t, y_p, y_s, g = synth
    gap = demographic_parity_gap(y_p, g)
    # We injected F = +0.15 boost
    assert gap > 0.10


def test_equal_opportunity_gap(synth):
    from src.fairness.metrics import equal_opportunity_gap
    y_t, y_p, y_s, g = synth
    gap = equal_opportunity_gap(y_t, y_p, g)
    assert 0.0 <= gap <= 1.0


def test_disparate_impact_ratio(synth):
    from src.fairness.metrics import disparate_impact_ratio
    y_t, y_p, y_s, g = synth
    di = disparate_impact_ratio(y_p, g)
    assert 0.0 <= di <= 1.0
    # With injected F bias, DI should be < 1
    assert di < 1.0


def test_group_rates_returns_dataframe(synth):
    from src.fairness.metrics import group_rates
    y_t, y_p, y_s, g = synth
    df = group_rates(y_t, y_p, y_s, g)
    assert isinstance(df, pd.DataFrame)
    assert set(df.index) >= {"M", "F"}
    assert "selection_rate" in df.columns
    assert "tpr" in df.columns
    assert "fpr" in df.columns
    assert "auc" in df.columns


def test_summarize_fairness_returns_fairnesssummary(synth):
    from src.fairness.metrics import summarize_fairness
    y_t, y_p, y_s, g = synth
    summary = summarize_fairness("test", y_t, y_p, y_s, g)
    assert summary.status in ("FAIR", "WARNING", "UNFAIR")
    assert 0.0 <= summary.dp_gap <= 1.0
    assert 0.0 <= summary.eo_gap <= 1.0
    assert 0.0 <= summary.di_ratio <= 1.0


# ----------------------------------------------------------------- slicing


def test_slice_dataset_gender(synth):
    from src.fairness.slicing import slice_dataset
    y_t, y_p, y_s, g = synth
    X = pd.DataFrame({"CODE_GENDER": g})
    out = slice_dataset(X, {"name": "gender", "column": "CODE_GENDER"})
    assert set(out) <= {"M", "F", "UNKNOWN"}


def test_slice_dataset_age_buckets():
    from src.fairness.slicing import slice_dataset
    X = pd.DataFrame({"DAYS_BIRTH": [-10000, -20000, -30000, -5000]})
    # 10000 days ~ 27 yrs (young)
    # 20000 days ~ 55 yrs (mid)
    # 30000 days ~ 82 yrs (old)
    # 5000 days ~ 14 yrs (below 18, clipped to "young")
    out = slice_dataset(X, {"name": "age_group", "column": "DAYS_BIRTH"})
    assert out[0] == "young"
    assert out[1] == "mid"
    assert out[2] == "old"


def test_slice_dataset_income_tertiles():
    from src.fairness.slicing import slice_dataset
    X = pd.DataFrame({"AMT_INCOME_TOTAL": [50_000, 100_000, 200_000, 500_000]})
    out = slice_dataset(X, {"name": "income_group", "column": "AMT_INCOME_TOTAL"})
    assert set(out) <= {"low", "mid", "high"}


def test_slice_dataset_missing_column_returns_unknown():
    from src.fairness.slicing import slice_dataset
    X = pd.DataFrame({"OTHER": [1, 2, 3]})
    out = slice_dataset(X, {"name": "gender", "column": "CODE_GENDER"})
    assert (out == "UNKNOWN").all()


def test_build_default_slices(synth):
    from src.fairness.slicing import build_default_slices
    y_t, y_p, y_s, g = synth
    X = pd.DataFrame({
        "CODE_GENDER": g,
        "DAYS_BIRTH": -np.random.default_rng(1).integers(5000, 30000, len(g)),
        "AMT_INCOME_TOTAL": np.random.default_rng(2).lognormal(11, 0.5, len(g)),
        "NAME_EDUCATION_TYPE": np.random.default_rng(3).choice(
            ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"],
            len(g),
        ),
    })
    slices = build_default_slices(X)
    assert set(slices.keys()) == {"gender", "age_group", "income_group", "education_group"}
    for k, v in slices.items():
        assert len(v) == len(g)


# ---------------------------------------------------------------------------
# Small-group filter
# ---------------------------------------------------------------------------
class TestMinGroupSize:
    def test_small_group_filter_excludes_tiny_groups(self):
        """Groups with < min_group_size samples must be excluded from DI/DP/EO."""
        from src.fairness.metrics import (
            demographic_parity_gap,
            disparate_impact_ratio,
            summarize_fairness,
        )
        # 3 groups: 200 F, 200 M, 5 OTHER (tiny)
        rng = np.random.default_rng(7)
        n_f, n_m, n_o = 200, 200, 5
        # Deterministic selection rates: F=20/200=0.10, M=10/200=0.05, OTHER=0
        y_pred = np.array(
            [1] * 20 + [0] * (n_f - 20)
            + [1] * 10 + [0] * (n_m - 10)
            + [0] * n_o
        )
        groups = np.array(["F"] * n_f + ["M"] * n_m + ["OTHER"] * n_o)
        y_true = np.zeros_like(y_pred)
        y_score = y_pred.astype(float)
        # With min_group_size=100, OTHER excluded → DP/EO/DI from F vs M only.
        # F=0.10, M=0.05 → DP=0.05, DI=0.5 (real fairness concern).
        s = summarize_fairness(
            "gender", y_true, y_pred, y_score, groups, min_group_size=100,
        )
        assert "OTHER" in s.groups_filtered
        assert s.dp_gap == pytest.approx(0.05, abs=1e-9)
        assert s.di_ratio == pytest.approx(0.5, abs=1e-9)

    def test_min_group_size_default_zero_keeps_all(self):
        """min_group_size=0 (default) keeps tiny groups — backward compatible."""
        from src.fairness.metrics import disparate_impact_ratio
        rng = np.random.default_rng(7)
        n_f, n_m, n_o = 200, 200, 5
        y_pred = np.array(
            [1] * 20 + [0] * (n_f - 20)
            + [1] * 10 + [0] * (n_m - 10)
            + [0] * n_o
        )
        groups = np.array(["F"] * n_f + ["M"] * n_m + ["OTHER"] * n_o)
        di = disparate_impact_ratio(y_pred, groups)
        # With min_group_size=0, OTHER (sel=0) is included → DI = 0
        assert di == pytest.approx(0.0, abs=1e-9)

    def test_demographic_parity_gap_with_filter(self):
        from src.fairness.metrics import demographic_parity_gap
        # 300 each (F/M/X), selection rates F=0.10, M=0.05, X=0
        y_pred = np.array(
            [1] * 30 + [0] * 270      # F: 30/300 = 0.10
            + [1] * 15 + [0] * 285    # M: 15/300 = 0.05
            + [0] * 300               # X: 0/300 = 0
        )
        groups = np.array(["F"] * 300 + ["M"] * 300 + ["X"] * 300)
        # min_group_size=50 keeps all → DP = 0.10 − 0 = 0.10
        assert demographic_parity_gap(y_pred, groups, min_group_size=50) == pytest.approx(0.10, abs=1e-9)
        # min_group_size=400 drops all (each 300 < 400) → 0 groups comparable
        assert demographic_parity_gap(y_pred, groups, min_group_size=400) == 0.0
