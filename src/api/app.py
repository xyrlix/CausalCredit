"""FastAPI entry point for the CausalCredit API.

Start with:
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Middleware stack (innermost first; last added is outermost):

  1. :class:`fastapi.middleware.cors.CORSMiddleware` — permissive CORS for the
     Streamlit front-end.
  2. :class:`PrivacyFilterMiddleware` — masks direct identifiers (``name`` /
     ``email`` / …) and coarsens quasi-identifiers (``age`` / ``income``) in
     JSON responses (NFR-007). Enabled by default; disable with
     ``middleware.pii_filter: false`` in ``configs/config.yaml``.
  3. :class:`AuthMiddleware` — gates ``/api/v1/admin/*`` behind
     ``X-Admin-API-Key`` and the rest behind ``X-API-Key``. Empty keys (the
     default) leave the corresponding role open so dev / tests need no setup.
  4. :class:`RateLimitMiddleware` — sliding-window per-IP throttle. Off when
     ``rate_limit_per_sec <= 0`` (the default).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_model_registry
from .middleware import AuthMiddleware, PrivacyFilterMiddleware, RateLimitMiddleware
from .routes import router

logger = logging.getLogger("causalcredit.app")


def _resolve_rate_limit() -> int:
    """Read rate-limit from env or YAML; default 0 (disabled)."""
    env_val = os.environ.get("CAUSALCREDIT_RATE_LIMIT_PER_SEC")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    try:
        from src.utils.config import load_config

        cfg = load_config()
        return int(cfg.get("middleware", {}).get("rate_limit_per_sec", 0))
    except Exception:
        return 0


def _resolve_pii_default() -> Optional[bool]:
    env_val = os.environ.get("CAUSALCREDIT_PII_FILTER")
    if env_val is not None:
        return env_val.lower() in {"1", "true", "yes", "on"}
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train (or load cached) artefacts at startup."""
    registry = get_model_registry()
    print("[lifespan] starting up CausalCredit API…")
    try:
        registry.load()
        print("[lifespan] registry loaded; service is ready.")
    except Exception as exc:
        print(f"[lifespan] registry load FAILED: {exc}")
    yield
    print("[lifespan] shutting down.")


app = FastAPI(
    title="CausalCredit API",
    description=(
        "Causal-inference-enhanced credit scoring. Provides scoring, "
        "SHAP explanations, DiCE counterfactuals, and DoWhy causal effects."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# 1. CORS — added first so it sits innermost (last request, first response).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Privacy filter — enabled by default; admin paths skip the rewriter.
app.add_middleware(PrivacyFilterMiddleware, enabled=_resolve_pii_default())

# 3. Auth — X-API-Key / X-Admin-API-Key. Empty keys = no enforcement.
app.add_middleware(AuthMiddleware)

# 4. Rate limit — sliding window per-IP. 0 / unset = disabled.
app.add_middleware(RateLimitMiddleware, max_per_second=_resolve_rate_limit())

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "CausalCredit API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "score": "POST /api/v1/score",
            "counterfactual": "POST /api/v1/counterfactual",
            "explain": "POST /api/v1/explain",
            "causal_effect": "POST /api/v1/causal-effect",
            "health": "GET /api/v1/health",
        },
    }
