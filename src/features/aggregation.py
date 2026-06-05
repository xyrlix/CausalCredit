"""Multi-table aggregation feature builder."""

from typing import Dict

import pandas as pd


class MultiTableAggregator:
    """Aggregate features from multiple related tables."""

    def aggregate_bureau(self, bureau_df: pd.DataFrame, bureau_bal_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate credit bureau features."""
        ...

    def aggregate_previous_app(self, prev_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate previous application features."""
        ...

    def aggregate_pos_cash(self, pos_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate POS/cash loan features."""
        ...

    def aggregate_installments(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate installment payment features."""
        ...

    def aggregate_credit_card(self, cc_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate credit card features."""
        ...

    def aggregate_all(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate all tables and return feature DataFrame indexed by SK_ID_CURR."""
        ...
