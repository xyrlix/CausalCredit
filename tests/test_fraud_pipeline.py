"""Unit tests for src.fraud.pipeline.FraudGuard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def guard_synth():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.lognormal(mean=11, sigma=0.5, size=n),
        "AMT_GOODS_PRICE": rng.lognormal(mean=10.5, sigma=0.5, size=n),
        "AMT_CREDIT": rng.lognormal(mean=10.7, sigma=0.5, size=n),
        "AMT_ANNUITY": rng.lognormal(mean=8.5, sigma=0.5, size=n),
        "EXT_SOURCE_2": rng.uniform(0, 1, size=n),
        "DAYS_BIRTH": -rng.integers(5000, 30000, size=n).astype(float),
        "DAYS_EMPLOYED": -rng.integers(0, 10000, size=n).astype(float),
        "EXT_SOURCE_1": rng.uniform(0, 1, size=n),
        "ORGANIZATION_TYPE": rng.choice(
            ["Business Entity Type 3", "Industry: mining", "Trade: type 7",
             "Construction", "Self-employed", "Bank", "Other"],
            size=n,
        ),
        "INST__DAYS_LATE_MEAN": rng.standard_normal(n),
        "INST__DAYS_LATE_MAX": rng.choice([0, 5, 30, 60], size=n),
        "INST_LATE_DAYS_GT0_FRAC": rng.uniform(0, 1, size=n),
        "INST_LATE_DAYS_GT30_FRAC": rng.uniform(0, 0.2, size=n),
        "INST__AMT_PAYMENT_RATIO_MEAN": rng.uniform(0.5, 1.5, size=n),
        "CC_BALANCE_MEAN": rng.lognormal(mean=8, sigma=0.5, size=n),
        "CC_UTILIZATION_MEAN": rng.uniform(0, 1, size=n),
        "CC_DPD_MAX": rng.choice([0, 0, 0, 30], size=n),
        "POS_CNT_INSTALMENT_FUTURE_MEAN": rng.integers(0, 30, size=n),
    })
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.92, 0.08]))
    return df, y


@pytest.fixture
def fq_for_synth():
    feature_names = [
        "AMT_INCOME_TOTAL", "EXT_SOURCE_2", "DAYS_BIRTH",
        "DAYS_EMPLOYED", "CC_BALANCE_MEAN", "INST__DAYS_LATE_MEAN",
        "AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY", "EXT_SOURCE_1",
    ]
    return {
        "per_feature": pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.linspace(0.1, 0.01, len(feature_names)),
            "abs_causal_proxy": np.linspace(0.005, 0.001, len(feature_names)),
            "quadrant": (
                ["TRUSTED"] * 4 + ["UNTRUSTED"] * 3 + ["MASKED"] * 2 + ["NEGLIGIBLE"]
            ),
        }),
        "counts": pd.Series({"TRUSTED": 4, "UNTRUSTED": 3, "MASKED": 2, "NEGLIGIBLE": 1}),
        "thresholds": (0.05, 0.003),
        "causal_features": ["AMT_INCOME_TOTAL", "EXT_SOURCE_2"],
    }


def test_fit_then_score_one(guard_synth, fq_for_synth):
    from src.fraud.pipeline import FraudGuard
    df, y = guard_synth
    guard = FraudGuard(classifier_params={"n_estimators": 30, "verbosity": -1, "n_jobs": 1})
    guard.fit(df, y, fq_for_synth)
    r = guard.score_one(df, default_proba=0.10, four_quadrant=fq_for_synth, applicant_idx=0)
    assert "fraud_score" in r
    assert "packaging_score" in r
    assert "denoised_default_proba" in r
    assert "routing" in r
    assert 0.0 <= r["fraud_score"] <= 0.10
    assert 0.0 <= r["packaging_score"] <= 1.0
    assert 0.0 <= r["denoised_default_proba"] <= 1.0
    assert r["routing"] in (
        "REJECT_FRAUD", "REJECT_PACKAGING", "REVIEW_DENOISED",
        "REVIEW_BORDERLINE", "PROCEED",
    )


def test_score_batch(guard_synth, fq_for_synth):
    from src.fraud.pipeline import FraudGuard
    df, y = guard_synth
    guard = FraudGuard(classifier_params={"n_estimators": 30, "verbosity": -1, "n_jobs": 1})
    guard.fit(df, y, fq_for_synth)
    out = guard.score_batch(
        df.iloc[:20],
        default_proba=np.full(20, 0.05),
        four_quadrant=fq_for_synth,
    )
    assert len(out) == 20
    assert "routing" in out.columns
    assert "fraud_score" in out.columns
    # All routing values must be valid
    valid = {"REJECT_FRAUD", "REJECT_PACKAGING", "REVIEW_DENOISED",
             "REVIEW_BORDERLINE", "PROCEED"}
    assert set(out["routing"].unique()) <= valid


def test_routing_decision_high_fraud(guard_synth, fq_for_synth):
    """If default_proba is 1.0 and applicant is a defaulter, fraud_score > 0.10 → REJECT_FRAUD."""
    from src.fraud.pipeline import FraudGuard
    df, y = guard_synth
    # Force a defaulter
    df = df.copy()
    y2 = y.copy()
    df.iloc[0, df.columns.get_loc("INST__DAYS_LATE_MAX")] = 90
    df.iloc[0, df.columns.get_loc("AMT_INCOME_TOTAL")] = 1_000_000
    df.iloc[0, df.columns.get_loc("DAYS_EMPLOYED")] = -30
    y2.iloc[0] = 1
    guard = FraudGuard(classifier_params={"n_estimators": 30, "verbosity": -1, "n_jobs": 1})
    guard.fit(df, y2, fq_for_synth)
    r = guard.score_one(df, default_proba=1.0, four_quadrant=fq_for_synth, applicant_idx=0)
    assert r["fraud_score"] >= 0.05
    # The high default_proba = 1.0 forces fraud_score high
    assert r["routing"] in ("REJECT_FRAUD", "REVIEW_BORDERLINE")


def test_routing_reasons_contain_useful_strings(guard_synth, fq_for_synth):
    from src.fraud.pipeline import FraudGuard
    df, y = guard_synth
    guard = FraudGuard(classifier_params={"n_estimators": 30, "verbosity": -1, "n_jobs": 1})
    guard.fit(df, y, fq_for_synth)
    r = guard.score_one(df, default_proba=0.50, four_quadrant=fq_for_synth, applicant_idx=0)
    # reasons is a list of strings
    assert isinstance(r["routing_reasons"], list)
    for s in r["routing_reasons"]:
        assert isinstance(s, str)


def test_score_one_without_fq_works(guard_synth):
    """Packaging requires four_quadrant; verify it gracefully degrades."""
    from src.fraud.pipeline import FraudGuard
    df, y = guard_synth
    guard = FraudGuard(classifier_params={"n_estimators": 30, "verbosity": -1, "n_jobs": 1})
    guard.fit(df, y, four_quadrant=None)
    r = guard.score_one(df, default_proba=0.05, four_quadrant=None, applicant_idx=0)
    assert r["packaging_score"] == 0.0
    assert r["routing"] in (
        "REJECT_FRAUD", "REJECT_PACKAGING", "REVIEW_DENOISED",
        "REVIEW_BORDERLINE", "PROCEED",
    )
