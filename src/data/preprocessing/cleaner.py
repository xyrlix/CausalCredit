"""Data preprocessing pipeline for German Credit dataset."""

from typing import Dict, List, Optional, Tuple

import pandas as pd


class DataCleaner:
    """Data cleaning and preprocessing pipeline for German Credit data."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.missing_threshold = self.config.get("missing_threshold", 0.70)

    def drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with missing rate above threshold."""
        missing_rate = df.isnull().mean()
        cols_to_drop = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        if cols_to_drop:
            return df.drop(columns=cols_to_drop)
        return df

    def winsorize_outliers(self, df: pd.DataFrame, columns: List[str],
                           limits: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
        """Winsorize extreme values in specified columns."""
        df = df.copy()
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                lower = df[col].quantile(limits[0])
                upper = df[col].quantile(limits[1])
                df[col] = df[col].clip(lower, upper)
        return df

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values: median for numeric, mode for categorical."""
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                else:
                    df[col] = df[col].fillna("missing")
        return df

    def clean(self, df: pd.DataFrame, numerical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Execute full cleaning pipeline."""
        df = self.drop_high_missing(df)
        df = self.impute_missing(df)
        if numerical_cols:
            df = self.winsorize_outliers(df, numerical_cols)
        return df
