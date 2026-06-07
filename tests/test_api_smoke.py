"""End-to-end API smoke tests using FastAPI TestClient.

Exercises all 5 endpoints (health, score, counterfactual, explain,
causal-effect) in-process. Requires the ModelRegistry pickle cache at
``output/models/registry_v1.pkl`` to exist (built by the API's
``lifespan`` on first run, or by ``dependencies._train_from_scratch``
which takes ~30s on first cold start).

This is *not* a unit test (it loads the full LightGBM model) — keep it
in a slow-tests group; CI may skip it via ``-k 'not api_smoke'``.

Run directly:
    pytest tests/test_api_smoke.py -v --tb=short
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_model_registry


SAMPLE_FEATURES = {
    "AMT_CREDIT": 500000,
    "AMT_INCOME_TOTAL": 150000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2000,
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
    "AMT_ANNUITY": 25000,
    "CODE_GENDER": "M",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "OCCUPATION_TYPE": "Laborers",
    "REGION_RATING_CLIENT": 2,
    "CNT_CHILDREN": 0,
    "AMT_GOODS_PRICE": 450000,
    "NAME_HOUSING_TYPE": "House / apartment",
}


@pytest.fixture(scope="module")
def client():
    """Yield a TestClient with the registry already loaded.

    The ``with`` block drives the FastAPI lifespan, which calls
    ``registry.load()`` — this is what we want to verify.
    """
    # Pre-warm the registry so failures are visible in the test session
    reg = get_model_registry()
    if not reg.is_loaded():
        reg.load()
    assert reg.is_loaded(), "ModelRegistry failed to load (no cache + cold start failed)"

    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] >= 5
    assert body["has_calibrator"] is True
    assert body["has_shap"] is True
    assert body["has_counterfactual"] is True
    assert body["has_ate_summary"] is True


def test_root_documents_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "endpoints" in body
    # Endpoint keys are valid Python identifiers (score / counterfactual / etc.)
    for ep in ("score", "counterfactual", "explain", "causal_effect", "health"):
        assert ep in body["endpoints"]


def test_score_returns_full_decision(client):
    r = client.post("/api/v1/score", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 200
    body = r.json()
    # Required fields
    assert 300 <= body["score"] <= 850
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_grade"] in {"A", "B", "C", "D", "E"}
    # Optional but should be present (include_* defaults to True)
    assert body["causal_effect"] is not None
    assert "ate" in body["causal_effect"]
    assert isinstance(body["counterfactual"], list)
    assert len(body["counterfactual"]) >= 1
    assert body["explanation"] is not None
    assert len(body["explanation"]["top_features"]) >= 1
    assert "decision_suggestion" in body


def test_score_with_minimal_payload(client):
    """Score with only the most critical features (others get imputed)."""
    r = client.post("/api/v1/score", json={
        "features": {"AMT_CREDIT": 200000, "EXT_SOURCE_2": 0.3},
    })
    assert r.status_code == 200
    body = r.json()
    assert 300 <= body["score"] <= 850
    assert body["risk_grade"] in {"A", "B", "C", "D", "E"}


def test_score_without_counterfactual(client):
    r = client.post("/api/v1/score", json={
        "features": SAMPLE_FEATURES,
        "include_counterfactual": False,
        "include_explanation": False,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["counterfactual"] is None
    assert body["explanation"] is None


def test_explain_returns_evidence_chain(client):
    r = client.post("/api/v1/explain", json={
        "features": SAMPLE_FEATURES, "top_k": 5,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_features"]) == 5
    assert len(body["evidence_chain"]) == 5
    for f in body["top_features"]:
        assert "feature" in f and "shap" in f and "direction" in f
    # Every evidence entry should have a human-readable narrative
    for e in body["evidence_chain"]:
        assert "narrative" in e
        assert "SHAP" in e["narrative"]


def test_counterfactual_with_intervention(client):
    r = client.post("/api/v1/counterfactual", json={
        "features": SAMPLE_FEATURES,
        "interventions": {"AMT_CREDIT": 350000},
    })
    assert r.status_code == 200
    body = r.json()
    assert "baseline_probability" in body
    assert "counterfactual_probability" in body
    assert "probability_change" in body
    assert -0.5 <= body["probability_change"] <= 0.5
    assert 0.0 <= body["confidence"] <= 1.0


def test_causal_effect_returns_ate(client):
    r = client.post("/api/v1/causal-effect", json={
        "treatment": "AMT_CREDIT", "outcome": "default",
    })
    assert r.status_code == 200
    body = r.json()
    assert "ate" in body
    assert isinstance(body["ate"], (int, float))
    assert "ate_ci" in body
    assert len(body["ate_ci"]) == 2
    assert body["ate_ci"][0] <= body["ate"] <= body["ate_ci"][1]


def test_score_rejects_missing_features(client):
    r = client.post("/api/v1/score", json={})
    assert r.status_code == 422  # Pydantic validation


def test_counterfactual_rejects_missing_interventions(client):
    r = client.post("/api/v1/counterfactual", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 422


def test_causal_effect_rejects_missing_treatment(client):
    r = client.post("/api/v1/causal-effect", json={"outcome": "default"})
    assert r.status_code == 422
