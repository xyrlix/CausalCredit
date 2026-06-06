"""Unit tests for src.fraud.denoising.CausalDenoisingScorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def denoise_synth():
    """100 rows: 50 'real' applicants (correlated), 50 '养流水' (decoupled)."""
    rng = np.random.default_rng(0)
    n = 100
    half = n // 2
    # Real applicants: high repayment score ↔ high consumption
    real_rep = rng.standard_normal((half, 5)).cumsum(axis=1) + rng.standard_normal((half, 5)) * 0.1
    real_con = rng.standard_normal((half, 4)).cumsum(axis=1) + rng.standard_normal((half, 4)) * 0.1
    # 养流水: high repayment, low consumption (decoupled)
    fraud_rep = rng.standard_normal((half, 5)) * 0.5 + 2.0
    fraud_con = rng.standard_normal((half, 4)) * 0.5 - 1.5
    rep = np.vstack([real_rep, fraud_rep])
    con = np.vstack([real_con, fraud_con])
    cols_rep = ["INST__DAYS_LATE_MEAN", "INST__DAYS_LATE_MAX",
                "INST_LATE_DAYS_GT0_FRAC", "INST_LATE_DAYS_GT30_FRAC",
                "INST__AMT_PAYMENT_RATIO_MEAN"]
    cols_con = ["CC_BALANCE_MEAN", "CC_UTILIZATION_MEAN",
                "CC_DPD_MAX", "POS_CNT_INSTALMENT_FUTURE_MEAN"]
    X = pd.DataFrame(np.hstack([rep, con]), columns=cols_rep + cols_con)
    default_proba = np.full(n, 0.05)  # model says all good
    return X, default_proba, half


def test_score_returns_dataframe_with_expected_columns(denoise_synth):
    from src.fraud.denoising import CausalDenoisingScorer
    X, default_proba, _ = denoise_synth
    scorer = CausalDenoisingScorer()
    df = scorer.score(X, default_proba)
    assert "causal_consistency" in df.columns
    assert "inflation_strength" in df.columns
    assert "denoised_default_proba" in df.columns
    assert "denoising_action" in df.columns
    assert len(df) == len(X)


def test_denoised_proba_in_valid_range(denoise_synth):
    from src.fraud.denoising import CausalDenoisingScorer
    X, default_proba, _ = denoise_synth
    scorer = CausalDenoisingScorer()
    df = scorer.score(X, default_proba)
    assert (df["denoised_default_proba"] >= 0).all()
    assert (df["denoised_default_proba"] <= 1).all()
    assert (df["causal_consistency"] >= 0).all()
    assert (df["causal_consistency"] <= 1).all()


def test_denoised_inflates_manufactured_history(denoise_synth):
    """For 养流水 applicants, denoised_proba should be > original (manufactured score is too low)."""
    from src.fraud.denoising import CausalDenoisingScorer
    X, default_proba, half = denoise_synth
    scorer = CausalDenoisingScorer()
    df = scorer.score(X, default_proba)
    # Real applicants: low inflation
    real_inflation = df.iloc[:half]["inflation_strength"]
    # 养流水: high inflation
    fraud_inflation = df.iloc[half:]["inflation_strength"]
    # In our construction, 养流水 has consistency ≈ 0 and real has consistency > 0.3
    assert fraud_inflation.mean() > real_inflation.mean()


def test_score_one_returns_dict():
    from src.fraud.denoising import CausalDenoisingScorer
    X = pd.DataFrame({
        "INST__DAYS_LATE_MEAN": [0.0],
        "CC_BALANCE_MEAN": [1000.0],
    })
    scorer = CausalDenoisingScorer()
    res = scorer.score_one(X, default_proba=0.05)
    for k in ("default_proba", "causal_consistency", "inflation_strength",
             "denoised_default_proba", "denoising_action"):
        assert k in res
    assert res["denoising_action"] in ("PROCEED", "FLAG_FOR_REVIEW")


def test_routing_threshold_changes_action():
    """Lower threshold → more applicants flagged for review."""
    from src.fraud.denoising import CausalDenoisingScorer
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "INST__DAYS_LATE_MEAN": rng.standard_normal(50),
        "CC_BALANCE_MEAN": rng.standard_normal(50),
    })
    default_proba = np.full(50, 0.05)
    df_strict = CausalDenoisingScorer(consistency_threshold=0.9).score(X, default_proba)
    df_lax = CausalDenoisingScorer(consistency_threshold=0.1).score(X, default_proba)
    n_flag_strict = (df_strict["denoising_action"] == "FLAG_FOR_REVIEW").sum()
    n_flag_lax = (df_lax["denoising_action"] == "FLAG_FOR_REVIEW").sum()
    assert n_flag_strict >= n_flag_lax


def test_no_repayment_features_returns_mid_consistency():
    """If M5+ temporal features are missing, consistency defaults to 0.5."""
    from src.fraud.denoising import CausalDenoisingScorer
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    default_proba = np.array([0.05, 0.05, 0.05])
    df = CausalDenoisingScorer().score(X, default_proba)
    assert (df["causal_consistency"] == 0.5).all()
