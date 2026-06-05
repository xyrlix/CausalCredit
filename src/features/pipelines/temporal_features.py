"""Temporal feature extraction pipeline.

Extracts time-series features from bureau_balance, installments_payments,
credit_card_balance, and POS_CASH_balance.
"""

import numpy as np
import pandas as pd


class TemporalFeatureExtractor:
    """Time-series feature extractor for credit behavior patterns."""

    def extract_bureau_dpd_trend(self, bureau_bal: pd.DataFrame) -> pd.DataFrame:
        """Extract Days Past Due trend over last 6 months."""
        ...

    def extract_bureau_status_entropy(self, bureau_bal: pd.DataFrame) -> pd.DataFrame:
        """Compute behavioral entropy from bureau status transitions."""
        ...

    def extract_installment_late_trend(self, inst: pd.DataFrame) -> pd.DataFrame:
        """Extract late payment days trend from installments."""
        ...

    def extract_credit_utilization_trend(self, cc: pd.DataFrame) -> pd.DataFrame:
        """Extract credit utilization ratio trend over 6 months."""
        ...

    def extract_repayment_volatility(self, inst: pd.DataFrame) -> pd.DataFrame:
        """Extract repayment amount volatility (coefficient of variation)."""
        ...

    def extract_behavioral_embedding(self, sequences: list[np.ndarray], dim: int = 32) -> np.ndarray:
        """Extract latent behavioral patterns via LSTM encoder."""
        ...

    def extract_all(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract all temporal features from multi-table data."""
        ...
