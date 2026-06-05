"""Causal feature builder for German Credit dataset.

Generates causal inference features:
- debt_to_income_ratio: credit_amount relative to a proxy income estimated from installment_commitment
- loan_burden: monthly payment relative to disposable income (installment_commitment)
- age_credit_interaction: interaction between age and credit amount
- duration_credit_ratio: duration relative to credit amount
"""

from typing import Optional

import numpy as np
import pandas as pd


def build_debt_to_income_ratio(df: pd.DataFrame,
                                credit_col: str = "credit_amount",
                                installment_col: str = "installment_commitment") -> pd.Series:
    """Build debt-to-income ratio.

    Uses installment_commitment as a proxy for income level since it represents
    the installment rate as a percentage of disposable income. We estimate
    income proxy = credit_amount / (installment_rate * duration / 12).

    Returns a continuous ratio; higher values indicate greater debt burden.
    """
    credit = df[credit_col].astype(float)
    installment_rate = df[installment_col].astype(float).clip(lower=1)
    duration = df.get("duration", pd.Series([12] * len(df))).astype(float).clip(lower=1)

    monthly_payment_approx = (installment_rate / 100.0) * (credit / duration.clip(lower=1))
    income_proxy = (monthly_payment_approx / (installment_rate / 100.0)).clip(lower=1)

    dti = credit / income_proxy
    dti = dti.clip(lower=dti.quantile(0.01), upper=dti.quantile(0.99))
    dti.name = "debt_to_income_ratio"
    return dti


def build_loan_burden(df: pd.DataFrame,
                       installment_col: str = "installment_commitment",
                       duration_col: str = "duration") -> pd.Series:
    """Build loan burden score.

    Represents the overall installment burden weighted by loan duration.
    Higher values indicate heavier financial burden.
    """
    installment = df[installment_col].astype(float)
    duration = df[duration_col].astype(float)
    burden = installment * np.log1p(duration)
    burden = burden.clip(lower=burden.quantile(0.01), upper=burden.quantile(0.99))
    burden.name = "loan_burden"
    return burden


def build_age_credit_interaction(df: pd.DataFrame,
                                  age_col: str = "age",
                                  credit_col: str = "credit_amount") -> pd.Series:
    """Build age and credit amount interaction feature."""
    age = df[age_col].astype(float)
    credit = df[credit_col].astype(float)
    interaction = age * np.log1p(credit)
    interaction.name = "age_credit_interaction"
    return interaction


def build_credit_per_year(df: pd.DataFrame,
                           credit_col: str = "credit_amount",
                           duration_col: str = "duration") -> pd.Series:
    """Build credit amount per year of loan duration."""
    credit = df[credit_col].astype(float)
    duration = df[duration_col].astype(float).clip(lower=1)
    credit_per_year = credit / (duration / 12.0).clip(lower=1)
    credit_per_year.name = "credit_per_year"
    return credit_per_year


def build_existing_credit_ratio(df: pd.DataFrame,
                                 existing_credits_col: str = "existing_credits",
                                 credit_col: str = "credit_amount") -> pd.Series:
    """Build ratio of existing credits to current credit amount."""
    existing = df[existing_credits_col].astype(float).clip(lower=0)
    credit = df[credit_col].astype(float).clip(lower=1)
    ratio = existing / np.log1p(credit)
    ratio.name = "existing_credit_ratio"
    return ratio


class CausalFeatureBuilder:
    """Builder for causal inference features on German Credit data."""

    def __init__(self):
        self._feature_names: list[str] = []

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all causal features and return as new DataFrame columns."""
        result = pd.DataFrame(index=df.index)

        result["debt_to_income_ratio"] = build_debt_to_income_ratio(df).astype(float)
        result["loan_burden"] = build_loan_burden(df).astype(float)
        result["age_credit_interaction"] = build_age_credit_interaction(df).astype(float)
        result["credit_per_year"] = build_credit_per_year(df).astype(float)
        result["existing_credit_ratio"] = build_existing_credit_ratio(df).astype(float)

        self._feature_names = list(result.columns)
        return result

    def get_feature_names(self) -> list[str]:
        """Return the names of features built."""
        return self._feature_names
