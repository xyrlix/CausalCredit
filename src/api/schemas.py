"""Pydantic V2 request/response models for CausalCredit API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreditRequest(BaseModel):
    """Credit scoring request."""
    applicant_id: Optional[str] = Field(None, description="Applicant ID")
    features: Dict[str, Any] = Field(..., description="Application features")
    include_counterfactual: bool = Field(True, description="Include counterfactual analysis")
    include_explanation: bool = Field(True, description="Include SHAP explanation")


class CounterfactualRequest(BaseModel):
    """Counterfactual scenario request."""
    applicant_id: Optional[str] = Field(None)
    features: Dict[str, Any] = Field(..., description="Baseline features")
    interventions: Dict[str, Any] = Field(..., description="Intervention plan")


class ExplainRequest(BaseModel):
    """SHAP explanation request."""
    applicant_id: Optional[str] = Field(None)
    features: Dict[str, Any] = Field(..., description="Application features")
    top_k: int = Field(5, description="Top-K features to explain")


class CausalEffectRequest(BaseModel):
    """Causal effect query request."""
    treatment: str = Field(..., description="Treatment variable name")
    outcome: str = Field(default="default", description="Outcome variable name")
    subgroup: Optional[Dict[str, Any]] = Field(None, description="Subgroup filter")


class CreditResponse(BaseModel):
    """Credit scoring response."""
    score: int = Field(..., description="Credit score 300-850")
    default_probability: float = Field(..., description="Default probability 0-1")
    risk_grade: str = Field(..., description="Risk grade A/B/C/D/E")
    causal_effect: Optional[Dict] = Field(None, description="Causal effect summary")
    counterfactual: Optional[List[Dict]] = Field(None, description="Counterfactual scenarios")
    explanation: Optional[Dict] = Field(None, description="SHAP explanation")
    decision_suggestion: Optional[str] = Field(None, description="Decision suggestion")


class CounterfactualResponse(BaseModel):
    """Counterfactual scenario response."""
    baseline_probability: float
    counterfactual_probability: float
    probability_change: float
    intervention_details: Dict
    confidence: float


class ExplainResponse(BaseModel):
    """SHAP explanation response."""
    top_features: List[Dict]
    evidence_chain: List[Dict]


class CausalEffectResponse(BaseModel):
    """Causal effect response."""
    ate: float
    ate_ci: tuple
    cate_subgroup: Optional[Dict]
    refutation_results: Optional[Dict]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
