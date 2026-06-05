"""Model and data drift detection.

Monitors feature drift (PSI), prediction drift, and concept drift
for CausalCredit's scoring pipeline.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class DriftDetector:
    """Drift detection for model monitoring.

    Tracks:
    - Feature drift (Population Stability Index)
    - Prediction drift (score distribution shift)
    - Concept drift (AUC/KS degradation over time)
    """

    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.reference_stats: Dict[str, Any] = {}

    def compute_psi(self, feature: str, current: pd.Series, bins: int = 10) -> float:
        """Compute Population Stability Index (PSI) for a single feature.

        PSI < 0.1: no drift
        PSI 0.1-0.2: moderate drift
        PSI > 0.2: significant drift (alert)
        """
        ...

    def detect_feature_drift(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Compute PSI for all features and flag those above threshold."""
        ...

    def detect_prediction_drift(self, current_scores: pd.Series, bins: int = 10) -> Dict[str, float]:
        """Detect drift in score distribution."""
        ...

    def detect_concept_drift(self, current_auc: float, current_ks: float,
                              baseline_auc: float, baseline_ks: float) -> Dict[str, Any]:
        """Check if model performance has degraded significantly."""
        ...

    def compute_feature_statistics(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Compare basic statistics (mean, std, quantiles) between reference and current."""
        ...

    def generate_drift_report(self, current_data: pd.DataFrame,
                               current_scores: pd.Series | None = None,
                               current_targets: pd.Series | None = None) -> str:
        """Generate a comprehensive drift report in Markdown."""
        ...
