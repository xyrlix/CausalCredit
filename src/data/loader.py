"""German Credit dataset loader via sklearn fetch_openml."""

from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.datasets import fetch_openml

GERMAN_CREDIT_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]

CATEGORICAL_COLUMNS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

NUMERICAL_COLUMNS = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]


class GermanCreditLoader:
    """Loader for the German Credit dataset via sklearn fetch_openml."""

    def __init__(self):
        self._raw: Optional[pd.DataFrame] = None

    def fetch(self) -> pd.DataFrame:
        """Download and return the German Credit dataset as a DataFrame."""
        if self._raw is not None:
            return self._raw.copy()

        raw = fetch_openml(name="credit-g", version=1, as_frame=True)
        self._raw = raw.frame.copy()
        return self._raw.copy()

    def get_feature_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return features (X) and binary target (y) where 1=bad credit."""
        df = self.fetch()
        X = df.drop(columns=["class"])
        y = (df["class"] == "bad").astype(int)
        return X, y

    def get_metadata(self) -> Dict:
        """Return metadata about the dataset."""
        df = self.fetch()
        return {
            "n_samples": len(df),
            "n_features": len(df.columns) - 1,
            "target_distribution": df["class"].value_counts().to_dict(),
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numerical_columns": NUMERICAL_COLUMNS,
            "column_names": GERMAN_CREDIT_COLUMNS,
        }
