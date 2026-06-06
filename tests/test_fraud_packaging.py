"""Unit tests for src.fraud.packaging.PackagingDetector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fq_synth():
    """Mimic a four-quadrant result with controllable UNTRUSTED fraction."""
    feature_names = [
        "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "AMT_CREDIT",
        "AMT_ANNUITY", "EXT_SOURCE_2", "DAYS_BIRTH",
        "DAYS_EMPLOYED", "EXT_SOURCE_3", "CODE_GENDER", "FLAG_OWN_CAR",
    ]
    # 4 trusted, 4 untrusted, 1 masked, 1 negligible
    quadrant_labels = (
        ["TRUSTED"] * 4 + ["UNTRUSTED"] * 4 + ["MASKED"] + ["NEGLIGIBLE"]
    )
    return {
        "per_feature": pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.linspace(0.1, 0.01, len(feature_names)),
            "abs_causal_proxy": np.linspace(0.005, 0.001, len(feature_names)),
            "quadrant": quadrant_labels,
        }),
        "counts": pd.Series({
            "TRUSTED": 4, "UNTRUSTED": 4, "MASKED": 1, "NEGLIGIBLE": 1,
        }),
        "thresholds": (0.05, 0.003),
        "causal_features": ["AMT_INCOME_TOTAL", "EXT_SOURCE_2"],
    }


@pytest.fixture
def applicant_synth():
    return pd.DataFrame({
        "AMT_INCOME_TOTAL": [200_000.0],
        "AMT_GOODS_PRICE": [180_000.0],
        "AMT_CREDIT": [180_000.0],
        "AMT_ANNUITY": [10_000.0],
        "EXT_SOURCE_2": [0.6],
        "DAYS_BIRTH": [-15000.0],
        "DAYS_EMPLOYED": [-3000.0],
    })


def test_calibrate_sets_feature_credibility(fq_synth, applicant_synth):
    from src.fraud.packaging import PackagingDetector
    det = PackagingDetector()
    det.calibrate(applicant_synth, fq_synth)
    assert len(det.feature_credibility_) == 10
    assert all(0.0 <= v <= 1.0 for v in det.feature_credibility_.values())
    assert 0.0 <= det.global_path_integrity_ <= 1.0


def test_score_returns_expected_keys(fq_synth, applicant_synth):
    from src.fraud.packaging import PackagingDetector
    det = PackagingDetector()
    det.calibrate(applicant_synth, fq_synth)
    res = det.score(applicant_synth, fq_synth, applicant_idx=0)
    for k in ("packaging_score", "path_integrity", "feature_credibility", "routing"):
        assert k in res
    assert 0.0 <= res["packaging_score"] <= 1.0
    assert 0.0 <= res["path_integrity"] <= 1.0
    assert res["routing"] in ("PROCEED", "MANUAL_REVIEW", "REJECT_PACKAGING_SUSPECTED")


def test_high_untrusted_yields_high_packaging(fq_synth, applicant_synth):
    """If most features are UNTRUSTED, packaging_score should be high.

    Pass per-applicant SHAP that forces 8/10 features to be UNTRUSTED:
    high |SHAP| but low causal proxy for those 8.
    """
    from src.fraud.packaging import PackagingDetector
    n_feat = len(fq_synth["per_feature"])
    th_shap, th_causal = fq_synth["thresholds"]
    # First 8 features: high SHAP (>= th_shap) and low causal proxy (< th_causal) → UNTRUSTED
    # Last 2 features: low SHAP and low causal proxy → NEGLIGIBLE
    row_shap = np.zeros(n_feat)
    row_shap[:8] = 1.0  # high SHAP
    # Make causal_proxy low for first 8 (use existing low values)
    fq_synth["per_feature"].loc[:7, "abs_causal_proxy"] = 0.0001
    fq_synth["per_feature"].loc[8:, "abs_causal_proxy"] = 0.0001
    fq_synth["per_feature"].loc[8:, "mean_abs_shap"] = 0.001  # below th_shap
    det = PackagingDetector()
    res = det.score(applicant_synth, fq_synth, applicant_idx=0, row_shap=row_shap)
    # n_credible (TRUSTED+MASKED) = 0, total = 10 → packaging_score = 1.0
    assert res["packaging_score"] >= 0.9
    assert res["routing"] == "REJECT_PACKAGING_SUSPECTED"


def test_low_untrusted_yields_low_packaging(fq_synth, applicant_synth):
    """If most features are TRUSTED, packaging_score should be low.

    Pass per-applicant SHAP that forces 8/10 features to be TRUSTED:
    high |SHAP| and high causal proxy for those 8.
    """
    from src.fraud.packaging import PackagingDetector
    n_feat = len(fq_synth["per_feature"])
    th_shap, th_causal = fq_synth["thresholds"]
    # First 8 features: high SHAP and high causal proxy → TRUSTED
    # Last 1: high SHAP and high causal → TRUSTED too (or MASKED if low SHAP)
    # Last 2: low SHAP, low causal → NEGLIGIBLE
    row_shap = np.zeros(n_feat)
    row_shap[:8] = 1.0
    fq_synth["per_feature"].loc[:7, "abs_causal_proxy"] = 1.0
    fq_synth["per_feature"].loc[8:, "abs_causal_proxy"] = 0.0001
    fq_synth["per_feature"].loc[8:, "mean_abs_shap"] = 0.001
    det = PackagingDetector()
    res = det.score(applicant_synth, fq_synth, applicant_idx=0, row_shap=row_shap)
    # n_credible = 8 (TRUSTED) + 0 (MASKED) = 8, total = 10 → packaging_score = 0.2
    assert res["packaging_score"] <= 0.3
    assert res["routing"] == "PROCEED"


def test_score_batch_returns_dataframe(fq_synth, applicant_synth):
    from src.fraud.packaging import PackagingDetector
    X = pd.concat([applicant_synth] * 5, ignore_index=True)
    det = PackagingDetector()
    df = det.score_batch(X, fq_synth)
    assert len(df) == 5
    assert "packaging_score" in df.columns
    assert "routing" in df.columns
    # All same features → same packaging_score
    assert df["packaging_score"].nunique() == 1


def test_path_integrity_for_broken_chain():
    """If income is huge but goods_price is tiny, the chain is broken."""
    from src.fraud.packaging import _compute_path_integrity
    X_broken = pd.DataFrame({
        "AMT_INCOME_TOTAL": [10_000_000.0],  # mega rich
        "AMT_GOODS_PRICE": [50_000.0],  # but tiny purchase
        "AMT_CREDIT": [50_000.0],
        "AMT_ANNUITY": [3000.0],
    })
    pi = _compute_path_integrity(X_broken)
    assert 0.0 <= pi < 1.0  # at least the income→goods path is broken


def test_path_integrity_for_coherent_chain():
    """If income and goods_price are roughly proportional, paths are intact."""
    from src.fraud.packaging import _compute_path_integrity
    X_ok = pd.DataFrame({
        "AMT_INCOME_TOTAL": [200_000.0],
        "AMT_GOODS_PRICE": [180_000.0],
        "AMT_CREDIT": [180_000.0],
        "AMT_ANNUITY": [10_000.0],
        "EXT_SOURCE_2": [0.5],
        "DAYS_BIRTH": [-15000.0],
        "DAYS_EMPLOYED": [-3000.0],
    })
    pi = _compute_path_integrity(X_ok)
    assert pi == 1.0
