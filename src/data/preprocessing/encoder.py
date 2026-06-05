"""Categorical and numerical feature encoding."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataEncoder:
    """Encoder for mixed-type features using LabelEncoder and StandardScaler."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame, categorical_cols: List[str],
            numerical_cols: List[str]) -> "DataEncoder":
        """Fit all encoders on training data."""
        self.categorical_cols = [c for c in categorical_cols if c in df.columns]
        self.numerical_cols = [c for c in numerical_cols if c in df.columns]

        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le

        if self.numerical_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.numerical_cols].astype(float))

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders."""
        if not self._fitted:
            raise RuntimeError("DataEncoder must be fitted before transform")
        df = df.copy()

        for col in self.categorical_cols:
            if col in df.columns:
                le = self.label_encoders[col]
                known = set(le.classes_)
                series = df[col].astype(str)
                series = series.apply(lambda x: x if x in known else "unknown")
                if "unknown" not in known:
                    le.classes_ = np.append(le.classes_, "unknown")
                df[col] = le.transform(series)

        if self.numerical_cols and self.scaler is not None:
            available_num = [c for c in self.numerical_cols if c in df.columns]
            if available_num:
                df[available_num] = self.scaler.transform(df[available_num].astype(float))

        return df

    def fit_transform(self, df: pd.DataFrame, categorical_cols: List[str],
                      numerical_cols: List[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, categorical_cols, numerical_cols)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of output feature names."""
        return self.categorical_cols + self.numerical_cols
