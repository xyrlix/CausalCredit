"""Causal feature builder.

Generates causal inference features:
- Debt-to-income ratio
- Credit utilization
- Payment discipline score
- Approximate interest rate
- Thin credit flag
"""

import pandas as pd


class CausalFeatureBuilder:
    """Builder for causal inference features."""

    def build_debt_to_income(self, df: pd.DataFrame) -> pd.Series:
        """Build debt-to-income ratio = AMT_ANNUITY / AMT_INCOME_TOTAL."""
        ...

    def build_credit_utilization(self, df: pd.DataFrame, cc_agg: pd.DataFrame) -> pd.Series:
        """Build credit utilization = credit_card_balance / credit_limit."""
        ...

    def build_payment_discipline(self, inst_agg: pd.DataFrame) -> pd.Series:
        """Build payment discipline score."""
        ...

    def build_approx_interest_rate(self, df: pd.DataFrame) -> pd.Series:
        """Build approximate interest rate from loan terms."""
        ...

    def build_thin_credit_flag(self, bureau_agg: pd.DataFrame, prev_agg: pd.DataFrame) -> pd.Series:
        """Build thin credit flag."""
        ...

    def build_all_causal_features(self, df: pd.DataFrame, agg_features: pd.DataFrame) -> pd.DataFrame:
        """Build all causal features."""
        ...
