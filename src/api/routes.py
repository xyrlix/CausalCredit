"""API route definitions for CausalCredit.

Endpoints:
- POST /api/v1/score        Credit scoring + causal analysis
- POST /api/v1/counterfactual  Counterfactual scenario simulation
- POST /api/v1/explain       SHAP explainability analysis
- POST /api/v1/causal-effect Causal effect query
- GET  /api/v1/health        Health check
"""

from fastapi import APIRouter, Depends

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


@router.post("/score", response_model=CreditResponse)
async def score_credit(request: CreditRequest):
    """Run full credit scoring pipeline with causal analysis."""
    ...


@router.post("/counterfactual", response_model=CounterfactualResponse)
async def counterfactual_analysis(request: CounterfactualRequest):
    """Run counterfactual scenario simulation."""
    ...


@router.post("/explain", response_model=ExplainResponse)
async def explain_score(request: ExplainRequest):
    """Run SHAP explainability analysis."""
    ...


@router.post("/causal-effect", response_model=CausalEffectResponse)
async def query_causal_effect(request: CausalEffectRequest):
    """Query causal effect for a treatment-outcome pair."""
    ...


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
