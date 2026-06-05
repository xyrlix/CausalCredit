"""SHAP-based model explainability analysis."""

from typing import Dict, List

import numpy as np
import pandas as pd


class SHAPExplainer:
    """SHAP explainer using TreeExplainer for LightGBM models."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values."""
        ...

    def global_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Global feature importance ranked by mean |SHAP|."""
        ...

    def dependence_plot(self, shap_values: np.ndarray, X: pd.DataFrame,
                        feature: str, interaction_feature: str = None, output_path: str = None):
        """SHAP dependence plot."""
        ...

    def local_explanation(self, shap_values: np.ndarray, X: pd.DataFrame,
                          idx: int, output_path: str = None):
        """Local waterfall plot for a single sample."""
        ...

    def causal_vs_noncausal_contribution(self, shap_values: np.ndarray,
                                          causal_features: List[str]) -> Dict:
        """Compare SHAP contributions of causal vs non-causal features."""
        ...

    def subgroup_shap_comparison(self, shap_values: np.ndarray, X: pd.DataFrame,
                                  subgroup_col: str) -> pd.DataFrame:
        """Compare SHAP patterns across subgroups."""
        ...

    def generate_evidence_chain(self, shap_values: np.ndarray, X: pd.DataFrame,
                                 idx: int, top_k: int = 5) -> List[Dict]:
        """Generate evidence chain for a single prediction."""
        ...
