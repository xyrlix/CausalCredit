"""Business logic service layer for CausalCredit API."""

from .dependencies import ModelRegistry
from .schemas import (
    CausalEffectResponse,
    CounterfactualResponse,
    CreditRequest,
    CreditResponse,
    ExplainRequest,
    ExplainResponse,
)


class CreditScoringService:
    """Credit scoring service orchestrator."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def score(self, request: CreditRequest) -> CreditResponse:
        """Execute full scoring pipeline."""
        ...

    def counterfactual(self, request) -> CounterfactualResponse:
        """Execute counterfactual analysis."""
        ...

    def explain(self, request: ExplainRequest) -> ExplainResponse:
        """Execute SHAP explanation."""
        ...

    def causal_effect(self, request) -> CausalEffectResponse:
        """Query causal effects."""
        ...
