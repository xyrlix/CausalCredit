"""Tests for the fairness block embedded in decision reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_build_fairness_block_returns_full_structure():
    from src.explain.decision import DecisionAdvisor
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "CODE_GENDER": rng.choice(["M", "F", "XNA"], n, p=[0.45, 0.45, 0.10]),
        "DAYS_BIRTH": -rng.integers(5000, 25000, n),
        "AMT_INCOME_TOTAL": rng.lognormal(11, 0.5, n),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"], n
        ),
    })
    y_true = rng.choice([0, 1], n, p=[0.92, 0.08])
    p_pred = 0.08 + 0.10 * (df["CODE_GENDER"] == "F")
    y_pred = (rng.uniform(size=n) < p_pred).astype(int)
    y_score = p_pred + rng.normal(scale=0.05, size=n)

    advisor = DecisionAdvisor()
    features = {
        "CODE_GENDER": "F",
        "DAYS_BIRTH": -15000,
        "AMT_INCOME_TOTAL": 100000.0,
        "NAME_EDUCATION_TYPE": "Higher education",
    }
    block = advisor.build_fairness_block(features, df, y_true, y_pred, y_score, n_test=500)

    # Top-level shape
    assert set(block.keys()) == {
        "applicant_groups", "slice_summaries", "verdict", "violated_slices", "regulatory_note"
    }
    assert block["verdict"] in ("FAIR", "WARNING", "UNFAIR")
    # applicant_groups must list 4 slices
    assert set(block["applicant_groups"].keys()) == {"gender", "age_group", "income_group", "education_group"}
    # per-slice summary
    assert len(block["slice_summaries"]) == 4
    for s in block["slice_summaries"]:
        assert s["status"] in ("FAIR", "WARNING", "UNFAIR")
        assert 0.0 <= s["dp_gap"] <= 1.0
        assert 0.0 <= s["di_ratio"] <= 1.0


def test_build_fairness_block_detects_injected_bias():
    from src.explain.decision import DecisionAdvisor
    rng = np.random.default_rng(1)
    n = 800
    # Heavy bias: F gets 0.50, M gets 0.05
    g = rng.choice(["M", "F"], n, p=[0.5, 0.5])
    y_true = rng.choice([0, 1], n, p=[0.92, 0.08])
    p_pred = np.where(g == "F", 0.50, 0.05)
    y_pred = (rng.uniform(size=n) < p_pred).astype(int)
    y_score = p_pred + rng.normal(scale=0.05, size=n)
    X = pd.DataFrame({"CODE_GENDER": g})

    advisor = DecisionAdvisor()
    block = advisor.build_fairness_block(
        {"CODE_GENDER": "F"}, X, y_true, y_pred, y_score
    )
    gender_slice = next(s for s in block["slice_summaries"] if s["slice"] == "gender")
    assert gender_slice["status"] in ("WARNING", "UNFAIR")
    assert gender_slice["dp_gap"] > 0.30
    assert gender_slice["di_ratio"] < 0.50
    assert "gender" in block["violated_slices"]


def test_decision_report_embeds_fairness_block():
    from src.explain.decision import DecisionAdvisor
    advisor = DecisionAdvisor()
    features = {"CODE_GENDER": "M", "DAYS_BIRTH": -15000, "AMT_INCOME_TOTAL": 100000.0}
    fairness_block = {
        "applicant_groups": {"gender": "M"},
        "slice_summaries": [],
        "verdict": "FAIR",
        "violated_slices": [],
        "regulatory_note": "ok",
    }
    report = advisor.generate_decision_report(
        features=features, default_probability=0.10, fairness_block=fairness_block
    )
    assert "fairness" in report
    assert report["fairness"]["verdict"] == "FAIR"
