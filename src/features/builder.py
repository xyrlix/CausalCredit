"""Main feature building pipeline entry point.

Orchestrates feature construction from raw data, including encoding,
causal feature generation, and final feature assembly.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.data.preprocessing.encoder import DataEncoder
from src.features.causal_features import CausalFeatureBuilder


class FeatureBuilder:
    """Master feature building orchestrator."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.encoder = DataEncoder()
        self.causal_builder = CausalFeatureBuilder()
        self._feature_names: List[str] = []
        self._fitted: bool = False

    def build(self, X: pd.DataFrame, categorical_cols: List[str],
              numerical_cols: List[str],
              fit: bool = True) -> pd.DataFrame:
        """Execute the full feature engineering pipeline.

        Steps:
        1. Encode categorical features with LabelEncoder
        2. Scale numerical features with StandardScaler
        3. Build causal features
        4. Concatenate all features
        """
        if fit:
            encoded = self.encoder.fit_transform(X, categorical_cols, numerical_cols)
        else:
            encoded = self.encoder.transform(X)

        causal_df = self.causal_builder.build_all(X)
        causal_df = causal_df.fillna(0)

        feature_df = pd.concat([encoded.reset_index(drop=True),
                                causal_df.reset_index(drop=True)], axis=1)

        self._feature_names = list(feature_df.columns)
        self._fitted = True
        return feature_df

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of output feature names."""
        return self._feature_names
