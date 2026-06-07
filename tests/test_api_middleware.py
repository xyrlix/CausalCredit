"""Unit + integration tests for the API middleware trio (M8.5c).

Three middlewares, three test groups:

* :class:`PrivacyFilter` — pure-function tests of the suppressors
* :class:`AuthMiddleware`  — TestClient tests with disabled/enabled keys
* :class:`RateLimitMiddleware` — TestClient test that exhausts the window

The TestClient tests do **not** load the model registry — they target
``/api/v1/health`` (which the registry reports as "loading" but returns
200) and the public root, so they're fast and CI-friendly.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import AuthMiddleware, PrivacyFilter, PrivacyFilterMiddleware, RateLimitMiddleware

# ---------------------------------------------------------------------------
# PrivacyFilter — pure-function tests
# ---------------------------------------------------------------------------


class TestPrivacyFilterDirectIdentifiers:
    def test_redacts_named_fields(self):
        f = PrivacyFilter(enabled=True)
        out = f.suppress({"name": "Alice", "score": 0.8})
        assert out["name"] == "[REDACTED]"
        assert out["score"] == 0.8

    def test_redacts_recursively_in_features(self):
        f = PrivacyFilter(enabled=True)
        out = f.suppress({
            "applicant_id": "A001",
            "features": {
                "AMT_CREDIT": 200_000,
                "email": "alice@example.com",
                "ssn": "123-45-6789",
                "phone": "+86-138-0000-0000",
            },
        })
        assert out["features"]["email"] == "[REDACTED]"
        assert out["features"]["ssn"] == "[REDACTED]"
        assert out["features"]["phone"] == "[REDACTED]"
        assert out["features"]["AMT_CREDIT"] == 200_000
        # applicant_id is not in DIRECT_IDENTIFIERS so it survives (it's a row key, not PII)
        assert out["applicant_id"] == "A001"

    def test_disabled_filter_is_passthrough(self):
        f = PrivacyFilter(enabled=False)
        out = f.suppress({"name": "Alice", "email": "a@b.c"})
        assert out["name"] == "Alice"
        assert out["email"] == "a@b.c"


class TestPrivacyFilterQuasiIdentifiers:
    @pytest.mark.parametrize("age,expected", [
        (10, "18-24"),
        (24, "18-24"),
        (25, "25-34"),
        (34, "25-34"),
        (45, "45-54"),
        (64, "55-64"),
        (65, "65+"),
        (90, "65+"),
    ])
    def test_age_bands(self, age, expected):
        f = PrivacyFilter(enabled=True)
        out = f.suppress({"features": {"age": age}})
        assert out["features"]["age"] == expected

    @pytest.mark.parametrize("income,expected", [
        (10_000, "<25K"),
        (24_999, "<25K"),
        (25_000, "25K-50K"),
        (49_999, "25K-50K"),
        (50_000, "50K-100K"),
        (99_999, "50K-100K"),
        (100_000, "100K-250K"),
        (249_999, "100K-250K"),
        (250_000, ">250K"),
        (1_000_000, ">250K"),
    ])
    def test_income_bands(self, income, expected):
        f = PrivacyFilter(enabled=True)
        out = f.suppress({"features": {"income": income}})
        assert out["features"]["income"] == expected

    def test_dependents_count_bands(self):
        f = PrivacyFilter(enabled=True)
        assert f.suppress({"features": {"dependents_count": 0}})["features"]["dependents_count"] == "0"
        assert f.suppress({"features": {"dependents_count": 1}})["features"]["dependents_count"] == "1-2"
        assert f.suppress({"features": {"dependents_count": 3}})["features"]["dependents_count"] == "3-4"
        assert f.suppress({"features": {"dependents_count": 7}})["features"]["dependents_count"] == "5+"

    def test_region_truncation(self):
        f = PrivacyFilter(enabled=True)
        out = f.suppress({"features": {"region": "Beijing"}})
        assert out["features"]["region"] == "Be***"

    def test_handles_non_dict_root(self):
        f = PrivacyFilter(enabled=True)
        assert f.suppress([1, 2, 3]) == [1, 2, 3]
        assert f.suppress("string") == "string"
        assert f.suppress(42) == 42


# ---------------------------------------------------------------------------
# AuthMiddleware — TestClient tests (minimal app, no model loading)
# ---------------------------------------------------------------------------


def _build_test_app(api_key: str = "", admin_key: str = "") -> FastAPI:
    """Build a minimal FastAPI app with the middleware trio wired up."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"service": "test"}

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/v1/score")
    async def score():
        return {"score": 750}

    @app.get("/api/v1/admin/metrics")
    async def admin_metrics():
        return {"metrics": "secret"}

    # Outermost first → add in reverse order. Final order in request
    # path is: CORS (none here) → PrivacyFilter → Auth → RateLimit → app.
    app.add_middleware(AuthMiddleware, api_key=api_key, admin_api_key=admin_key)
    app.add_middleware(PrivacyFilterMiddleware, enabled=False)  # disabled for auth tests
    return app


class TestAuthMiddleware:
    def test_open_access_when_no_keys_configured(self):
        client = TestClient(_build_test_app())
        r = client.get("/")
        assert r.status_code == 200
        r = client.post("/api/v1/score", json={})
        assert r.status_code == 200

    def test_health_is_public(self):
        client = TestClient(_build_test_app(api_key="secret"))
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_inference_requires_api_key_when_set(self):
        client = TestClient(_build_test_app(api_key="secret"))
        r = client.post("/api/v1/score", json={})
        assert r.status_code == 401
        assert r.json()["error_code"] == "UNAUTHORIZED"

    def test_inference_accepts_correct_api_key(self):
        client = TestClient(_build_test_app(api_key="secret"))
        r = client.post("/api/v1/score", json={}, headers={"X-API-Key": "secret"})
        assert r.status_code == 200

    def test_inference_rejects_wrong_api_key(self):
        client = TestClient(_build_test_app(api_key="secret"))
        r = client.post("/api/v1/score", json={}, headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_admin_rejects_when_no_admin_key_configured(self):
        client = TestClient(_build_test_app(api_key="user-key", admin_key=""))
        r = client.get("/api/v1/admin/metrics", headers={"X-API-Key": "user-key"})
        assert r.status_code == 401

    def test_admin_requires_admin_api_key(self):
        client = TestClient(_build_test_app(api_key="user-key", admin_key="admin-key"))
        # Inference key alone should not unlock admin paths
        r = client.get("/api/v1/admin/metrics", headers={"X-API-Key": "user-key"})
        assert r.status_code == 401
        r = client.get("/api/v1/admin/metrics", headers={"X-Admin-API-Key": "admin-key"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# RateLimitMiddleware — TestClient test that exhausts the window
# ---------------------------------------------------------------------------


def _build_rate_limited_app(max_per_sec: int) -> FastAPI:
    app = FastAPI()

    @app.post("/api/v1/score")
    async def score():
        return {"score": 750}

    app.add_middleware(RateLimitMiddleware, max_per_second=max_per_sec)
    return app


class TestRateLimitMiddleware:
    def test_disabled_by_default(self):
        client = TestClient(_build_rate_limited_app(0))
        for _ in range(20):
            assert client.post("/api/v1/score", json={}).status_code == 200

    def test_blocks_after_window_full(self):
        client = TestClient(_build_rate_limited_app(3))
        # 3 quick requests all pass
        for _ in range(3):
            assert client.post("/api/v1/score", json={}).status_code == 200
        # 4th hits the limit
        r = client.post("/api/v1/score", json={})
        assert r.status_code == 429
        assert r.json()["error_code"] == "RATE_LIMITED"
        assert r.headers.get("Retry-After") == "1"

    def test_window_slides(self):
        client = TestClient(_build_rate_limited_app(2))
        # First burst
        assert client.post("/api/v1/score", json={}).status_code == 200
        assert client.post("/api/v1/score", json={}).status_code == 200
        assert client.post("/api/v1/score", json={}).status_code == 429
        # Wait for the window to slide
        time.sleep(1.1)
        assert client.post("/api/v1/score", json={}).status_code == 200


# ---------------------------------------------------------------------------
# End-to-end: privacy filter wired into a real response
# ---------------------------------------------------------------------------


def _build_privacy_app() -> FastAPI:
    app = FastAPI()

    @app.post("/api/v1/score")
    async def score():
        return {
            "score": 750,
            "default_probability": 0.05,
            "features": {
                "AMT_CREDIT": 200_000,
                "age": 32,
                "income": 75_000,
                "dependents_count": 2,
                "region": "Shanghai",
                "email": "leak@example.com",
                "name": "leaky",
            },
        }

    @app.get("/api/v1/admin/raw")
    async def admin_raw():
        return {"features": {"email": "admin-sees-this@example.com"}}

    app.add_middleware(PrivacyFilterMiddleware, enabled=True)
    return app


class TestPrivacyFilterMiddlewareEndToEnd:
    def test_masks_pii_in_json_responses(self):
        client = TestClient(_build_privacy_app())
        r = client.post("/api/v1/score", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["features"]["email"] == "[REDACTED]"
        assert body["features"]["name"] == "[REDACTED]"
        # Quasi-identifiers coarsened
        assert body["features"]["age"] == "25-34"
        assert body["features"]["income"] == "50K-100K"
        assert body["features"]["dependents_count"] == "1-2"
        assert body["features"]["region"] == "Sh***"
        # Non-PII passes through
        assert body["features"]["AMT_CREDIT"] == 200_000
        # Non-features fields untouched
        assert body["score"] == 750

    def test_admin_paths_skip_filter(self):
        client = TestClient(_build_privacy_app())
        r = client.get("/api/v1/admin/raw")
        assert r.status_code == 200
        assert r.json()["features"]["email"] == "admin-sees-this@example.com"

    def test_disabled_middleware_passes_through(self):
        app = FastAPI()

        @app.post("/api/v1/score")
        async def score():
            return {"features": {"email": "raw@example.com"}}

        app.add_middleware(PrivacyFilterMiddleware, enabled=False)
        client = TestClient(app)
        r = client.post("/api/v1/score", json={})
        assert r.json()["features"]["email"] == "raw@example.com"
