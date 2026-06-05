"""Unit tests for src.explain.decision (scoring math + DecisionAdvisor)."""

from __future__ import annotations

import pytest

from src.explain.decision import DecisionAdvisor


# ---------------------------------------------------------------------------
# Pure scoring math (no dependencies on a trained model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,expected_range", [
    (0.00, (840, 850)),   # near-zero P → near 850
    (0.01, (800, 850)),
    (0.05, (730, 820)),
    (0.10, (660, 800)),
    (0.50, (430, 530)),
    (0.90, (300, 400)),
    (1.00, (300, 305)),   # P=1 → 300
])
def test_compute_score_monotonic_in_p(p, expected_range):
    s = DecisionAdvisor.compute_score(p)
    lo, hi = expected_range
    assert lo <= s <= hi, f"p={p}: expected {lo}-{hi}, got {s}"


def test_compute_score_bounded_300_850():
    for p in [-0.5, 0.0, 0.5, 1.0, 1.5]:
        s = DecisionAdvisor.compute_score(p)
        assert 300 <= s <= 850


def test_compute_score_strictly_decreasing_in_p():
    scores = [DecisionAdvisor.compute_score(p) for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


@pytest.mark.parametrize("score,grade", [
    (850, "A"),
    (800, "A"),
    (750, "A"),
    (749, "B"),
    (700, "B"),
    (650, "B"),
    (649, "C"),
    (600, "C"),
    (550, "C"),
    (549, "D"),
    (500, "D"),
    (450, "D"),
    (449, "E"),
    (400, "E"),
    (300, "E"),
])
def test_compute_grade_boundaries(score, grade):
    assert DecisionAdvisor.compute_grade(score) == grade


def test_compute_grade_all_letters_reachable():
    grades = {DecisionAdvisor.compute_grade(s) for s in range(300, 851, 25)}
    assert grades == {"A", "B", "C", "D", "E"}


def test_compute_recommendation_for_each_grade():
    for grade in ["A", "B", "C", "D", "E"]:
        rec = DecisionAdvisor.compute_recommendation(grade, 0.1)
        assert isinstance(rec, str) and len(rec) > 0


def test_compute_recommendation_a_is_approve():
    assert DecisionAdvisor.compute_recommendation("A", 0.02).upper().startswith("APPROVE")


def test_compute_recommendation_e_is_decline():
    assert DecisionAdvisor.compute_recommendation("E", 0.85).upper().startswith("DECLINE")


# ---------------------------------------------------------------------------
# generate_decision_report — minimal mocks
# ---------------------------------------------------------------------------

class _StubCF:
    def generate_counterfactuals(self, *args, **kwargs):
        return {"baseline_proba": 0.1, "cfs": []}


class _StubSHAP:
    pass


def test_generate_decision_report_minimal_inputs():
    advisor = DecisionAdvisor(counterfactual_reasoner=_StubCF(), shap_explainer=_StubSHAP())
    report = advisor.generate_decision_report(
        features={"AMT_CREDIT": 500000.0},
        applicant_id="UT_001",
        default_probability=0.05,
    )
    assert report["applicant_id"] == "UT_001"
    assert "score" in report
    assert "risk_grade" in report
    assert 300 <= report["score"] <= 850
    assert report["risk_grade"] in {"A", "B", "C", "D", "E"}
    assert "decision_suggestion" in report


def test_generate_decision_report_with_all_optionals():
    import numpy as np
    import pandas as pd
    advisor = DecisionAdvisor(counterfactual_reasoner=_StubCF(), shap_explainer=_StubSHAP())
    report = advisor.generate_decision_report(
        features={"AMT_CREDIT": 500000.0, "DAYS_BIRTH": -12000.0},
        applicant_id="UT_002",
        default_probability=0.30,
        shap_values=np.array([[0.5, -0.3]]),
        X_for_shap=pd.DataFrame([{"AMT_CREDIT": 500000.0, "DAYS_BIRTH": -12000.0}]),
        cate_value=0.05,
        four_quadrant={"counts": {"TRUSTED": 2, "UNTRUSTED": 0, "NEGLIGIBLE": 0, "MASKED": 0}},
        causal_effect_summary={"ate": 0.04, "robustness_score": 0.75},
    )
    assert "top_risk_factors" in report
    assert "cate_insights" in report
    assert any("CATE" in i for i in report["cate_insights"])


def test_generate_suggestion_text_zh_and_en():
    advisor = DecisionAdvisor(counterfactual_reasoner=_StubCF(), shap_explainer=_StubSHAP())
    report = advisor.generate_decision_report(
        features={"AMT_CREDIT": 500000.0}, applicant_id="UT_003",
        default_probability=0.05,
    )
    en = advisor.generate_suggestion_text(report, language="en")
    zh = advisor.generate_suggestion_text(report, language="zh")
    assert isinstance(en, str) and len(en) > 0
    assert isinstance(zh, str) and len(zh) > 0
