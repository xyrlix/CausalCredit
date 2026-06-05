"""FastAPI entry point for the CausalCredit API.

Start with:
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_model_registry
from .routes import router


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
