"""API routes for CausalCredit.

Endpoints (all under /api/v1):
- POST /score             — full pipeline (LightGBM + SHAP + CF + decision)
- POST /counterfactual    — single-intervention what-if
- POST /explain           — SHAP top-k + evidence chain
- POST /causal-effect     — pre-computed ATE summary
- GET  /health            — registry liveness
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import ModelRegistry, get_model_registry
from .schemas import (
    CausalEffectRequest,
    CausalEffectResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    CreditRequest,
    CreditResponse,
    ExplainRequest,
    ExplainResponse,
)
from .services import CreditScoringService

router = APIRouter(prefix="/api/v1", tags=["CausalCredit"])


def _service(registry: ModelRegistry = Depends(get_model_registry)) -> CreditScoringService:
    if not registry.is_loaded():
        raise HTTPException(status_code=503, detail="Model registry not loaded yet")
    return CreditScoringService(registry)


@router.post("/score", response_model=CreditResponse)
async def score_credit(
    request: CreditRequest,
    svc: CreditScoringService = Depends(_service),
):
    try:
        return svc.score(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"score failed: {exc}") from exc


@router.post("/counterfactual", response_model=CounterfactualResponse)
async def counterfactual_analysis(
    request: CounterfactualRequest,
    svc: CreditScoringService = Depends(_service),
):
    try:
        return svc.counterfactual(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"counterfactual failed: {exc}") from exc


@router.post("/explain", response_model=ExplainResponse)
async def explain_score(
    request: ExplainRequest,
    svc: CreditScoringService = Depends(_service),
):
    try:
        return svc.explain(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"explain failed: {exc}") from exc


@router.post("/causal-effect", response_model=CausalEffectResponse)
async def query_causal_effect(
    request: CausalEffectRequest,
    svc: CreditScoringService = Depends(_service),
):
    try:
        return svc.causal_effect(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"causal_effect failed: {exc}") from exc


@router.get("/health")
async def health_check(registry: ModelRegistry = Depends(get_model_registry)):
    return {
        "status": "ok" if registry.is_loaded() else "loading",
        "n_features": len(registry.feature_cols) if registry.is_loaded() else 0,
        "has_calibrator": registry.calibrator is not None,
        "has_shap": registry.shap_explainer is not None,
        "has_counterfactual": registry.counterfactual_reasoner is not None,
        "has_ate_summary": bool(registry.ate_summary),
        # M8.5e: model provenance for ops & audit
        "active_version": registry.active_version or None,
        "model_hash": (registry.model_hash[:16] + "…") if registry.model_hash else None,
    }
