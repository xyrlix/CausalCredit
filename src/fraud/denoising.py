"""Causal denoising scorer.

Implements "因果去噪评分引擎" from
``docs/CausalCredit_反欺诈能力覆盖分析.md`` §4.1.4.

The core idea: a fraudster can "养流水" (manufacture transaction
history) to make themselves look like a model citizen.  Such
applicants show up with a low observed default probability, but the
underlying causal signal is weaker.  The denoising scorer estimates
``P(真实评分 | do(去除养流水效应))`` by:

1. Computing a **causal-consistency** score between repayment history
   and consumption history. 养流水 users have repayment history that
   is *disconnected* from their consumption pattern (the repayments
   were "manufactured" — the money came from outside, not from earned
   income being spent).

2. Estimating a **denoised default probability** as
   ``P(default | X_real)`` where ``X_real`` is the observed X with
   the inflated (manufactured) component removed:
   ``X_real = X_observed - inflation_strength * (1 - consistency)``

The output is a per-applicant tuple:
    - ``denoised_default_proba``: adjusted P(default) after denoising
    - ``causal_consistency``:     repayment↔consumption correlation [0,1]
    - ``inflation_strength``:     estimated "养流水" inflation [0,1]
    - ``denoising_action``:       PROCEED / FLAG_FOR_REVIEW
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Repayment-history features (from installments_payments aggregation in M5+)
REPAYMENT_FEATURES = [
    "INST__DAYS_LATE_MEAN",
    "INST__DAYS_LATE_MAX",
    "INST_LATE_DAYS_GT0_FRAC",
    "INST_LATE_DAYS_GT30_FRAC",
    "INST__AMT_PAYMENT_RATIO_MEAN",  # payment / installment
]

# Consumption features (from credit_card_balance + POS in M5+)
CONSUMPTION_FEATURES = [
    "CC_BALANCE_MEAN",
    "CC_UTILIZATION_MEAN",
    "CC_DPD_MAX",
    "POS_CNT_INSTALMENT_FUTURE_MEAN",
]


class CausalDenoisingScorer:
    """Score applicants for "养流水" (manufactured history) risk.

    Stateless — fits a quick per-applicant consistency score from
    repayment + consumption features, then denoises P(default) by
    adding back a fraction of the inflation that the model was
    missing.
    """

    def __init__(
        self,
        consistency_threshold: float = 0.5,
        inflation_strength_max: float = 0.15,
    ):
        self.consistency_threshold = consistency_threshold
        self.inflation_strength_max = inflation_strength_max

    def _to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = X.copy()
        for c in Xn.columns:
            if str(Xn[c].dtype) in ("object", "category"):
                Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
        return Xn.fillna(0.0)

    def _causal_consistency(self, X: pd.DataFrame) -> np.ndarray:
        """Per-applicant causal consistency between repayment and consumption.

        We compare a one-dimensional summary of repayment features
        against a one-dimensional summary of consumption features.
        A real applicant has a positive relationship (high repayment
        score ↔ high consumption).  A 养流水 user has decoupled values.

        To compare 1-D summaries we take the per-row mean of each side
        (after per-column z-scoring across rows) and use the resulting
        sign agreement.
        """
        rep = [c for c in REPAYMENT_FEATURES if c in X.columns]
        con = [c for c in CONSUMPTION_FEATURES if c in X.columns]
        if not rep or not con:
            return np.full(len(X), 0.5)  # can't compute; assume mid
        rep_mat = X[rep].values
        con_mat = X[con].values
        rep_z = (rep_mat - rep_mat.mean(axis=0)) / (rep_mat.std(axis=0) + 1e-9)
        con_z = (con_mat - con_mat.mean(axis=0)) / (con_mat.std(axis=0) + 1e-9)
        # Per-row 1-D summary = mean of standardized columns
        rep_score = rep_z.mean(axis=1)
        con_score = con_z.mean(axis=1)
        # Sign agreement: +1 if both positive or both negative, -1 if opposite
        sign = np.sign(rep_score) * np.sign(con_score)
        # Map [-1, 1] to [0, 1]
        return ((sign + 1.0) / 2.0).clip(0.0, 1.0)

    def score(
        self,
        X: pd.DataFrame,
        default_proba: np.ndarray,
    ) -> pd.DataFrame:
        """Compute denoised scores for a batch of applicants.

        Args:
            X: feature matrix (rows = applicants).  Must contain at
                least some REPAYMENT_FEATURES and CONSUMPTION_FEATURES
                for the consistency check; otherwise consistency
                defaults to 0.5.
            default_proba: (n,) array of P(default) from the binary
                model.
        """
        default_proba = np.asarray(default_proba).flatten()
        Xn = self._to_numeric(X)
        consistency = self._causal_consistency(Xn)
        # Inflation strength: high when consistency is low
        # (decoupled repayment / consumption = manufactured history)
        inflation = np.clip(
            (1.0 - consistency) * self.inflation_strength_max * 5.0,  # scale so that consistency=0 → max inflation
            0.0, self.inflation_strength_max,
        )
        # Denoised probability: the model's P(default) is biased *low*
        # for manufactured-history applicants.  We add the inflation
        # back to get a more honest P(default).
        denoised = np.clip(default_proba + inflation, 0.0, 1.0)

        # Routing
        action = np.where(
            consistency < self.consistency_threshold,
            "FLAG_FOR_REVIEW",
            "PROCEED",
        )

        return pd.DataFrame({
            "default_proba": default_proba,
            "causal_consistency": consistency,
            "inflation_strength": inflation,
            "denoised_default_proba": denoised,
            "denoising_action": action,
        })

    def score_one(
        self,
        X_one: pd.DataFrame,
        default_proba: float,
    ) -> Dict:
        """Convenience wrapper for a single applicant (X_one is one row)."""
        df = self.score(X_one, np.array([default_proba]))
        row = df.iloc[0].to_dict()
        return {
            "default_proba": float(row["default_proba"]),
            "causal_consistency": float(row["causal_consistency"]),
            "inflation_strength": float(row["inflation_strength"]),
            "denoised_default_proba": float(row["denoised_default_proba"]),
            "denoising_action": str(row["denoising_action"]),
        }
