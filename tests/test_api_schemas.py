"""Unit tests for src.api (schemas + service helpers)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    CausalEffectRequest,
    CausalEffectResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    CreditRequest,
    CreditResponse,
    ExplainRequest,
    ExplainResponse,
)
from src.api.services import CreditScoringService


# ---------------------------------------------------------------------------
# Schemas — validation
# ---------------------------------------------------------------------------

def test_credit_request_minimal():
    req = CreditRequest(features={"AMT_CREDIT": 500000})
    assert req.features["AMT_CREDIT"] == 500000
    assert req.include_counterfactual is True
    assert req.include_explanation is True


def test_credit_request_rejects_missing_features():
    with pytest.raises(ValidationError):
        CreditRequest()  # type: ignore[call-arg]


def test_credit_response_score_in_range():
    resp = CreditResponse(score=720, default_probability=0.08, risk_grade="A")
    assert resp.score == 720
    assert resp.default_probability == 0.08
    assert resp.risk_grade == "A"


def test_counterfactual_request_needs_interventions():
    with pytest.raises(ValidationError):
        CounterfactualRequest(features={"AMT_CREDIT": 500000})  # type: ignore[call-arg]


def test_explain_request_defaults_top_k():
    req = ExplainRequest(features={"AMT_CREDIT": 500000})
    assert req.top_k == 5


def test_causal_effect_request_minimal():
    req = CausalEffectRequest(treatment="AMT_CREDIT")
    assert req.treatment == "AMT_CREDIT"
    assert req.outcome == "default"  # schema default


# ---------------------------------------------------------------------------
# Service helpers — pure functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,expected_score_range", [
    (0.01, (800, 850)),
    (0.10, (660, 800)),
    (0.50, (430, 530)),
    (0.90, (300, 400)),
])
def test_service_compute_score_matches_decision_advisor(p, expected_score_range):
    s = CreditScoringService._compute_score(p)
    lo, hi = expected_score_range
    assert lo <= s <= hi


def test_service_compute_grade_buckets():
    assert CreditScoringService._compute_grade(800) == "A"
    assert CreditScoringService._compute_grade(700) == "B"
    assert CreditScoringService._compute_grade(600) == "C"
    assert CreditScoringService._compute_grade(500) == "D"
    assert CreditScoringService._compute_grade(400) == "E"


def test_service_compute_suggestion_text_for_each_grade():
    for grade in ["A", "B", "C", "D", "E"]:
        s = CreditScoringService._compute_suggestion(grade, 0.1)
        assert isinstance(s, str) and len(s) > 0
    assert CreditScoringService._compute_suggestion("A", 0.02).startswith("APPROVE")
    assert CreditScoringService._compute_suggestion("E", 0.85).startswith("DECLINE")
