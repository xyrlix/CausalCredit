"""Dependency injection and model loading for FastAPI."""

from functools import lru_cache


class ModelRegistry:
    """Singleton model registry for lazy loading."""

    def __init__(self):
        self.lgbm_model = None
        self.cate_model = None
        self.calibrator = None
        self.shap_explainer = None
        self.feature_encoder = None
        self.causal_graph = None
        self.counterfactual_reasoner = None
        self.decision_advisor = None


@lru_cache()
def get_model_registry() -> ModelRegistry:
    """Get or create the model registry singleton."""
    return ModelRegistry()
