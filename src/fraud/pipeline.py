"""FraudGuard — orchestrator for the three anti-fraud modules.

Combines:
* ``ThreeClassFraudClassifier`` → fraud_score (P(fraudulent))
* ``PackagingDetector``        → packaging_score (path integrity)
* ``CausalDenoisingScorer``    → denoised_default_proba

into a single per-applicant output suitable for embedding in the
existing decision report.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.fraud.three_class import ThreeClassFraudClassifier, ALL_LABELS, DEFRAUDER_CLASSES
from src.fraud.packaging import PackagingDetector
from src.fraud.denoising import CausalDenoisingScorer


class FraudGuard:
    """End-to-end anti-fraud guard.

    Usage:
        guard = FraudGuard()
        guard.fit(X, y, four_quadrant)
        report = guard.score_one(X, default_proba, four_quadrant, applicant_idx=0)
    """

    def __init__(
        self,
        classifier_params: Optional[Dict] = None,
        packaging_kwargs: Optional[Dict] = None,
        denoising_kwargs: Optional[Dict] = None,
    ):
        self.classifier = ThreeClassFraudClassifier(params=classifier_params)
        self.packaging = PackagingDetector(**(packaging_kwargs or {}))
        self.denoising = CausalDenoisingScorer(**(denoising_kwargs or {}))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        four_quadrant: Optional[Dict] = None,
    ) -> "FraudGuard":
        """Train the 3-class sub-classifier and calibrate packaging detector."""
        # 3-class sub-classifier
        labels = self.classifier.fit_pseudo_labels(X, y)
        self.classifier.fit(X, y, labels=labels)
        # Calibrate packaging detector on a reference population
        if four_quadrant is not None:
            self.packaging.calibrate(X, four_quadrant)
        return self

    def score_one(
        self,
        X: pd.DataFrame,
        default_proba: float,
        four_quadrant: Optional[Dict] = None,
        applicant_idx: int = 0,
        row_shap: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute the three fraud scores for a single applicant."""
        # 1) fraud_score
        fraud_score = float(
            self.classifier.fraud_score(X.iloc[[applicant_idx]], np.array([default_proba]))[0]
        )
        p_sub = self.classifier.predict_proba(X.iloc[[applicant_idx]])[0]
        sub_proba = {
            "fraudulent": float(p_sub[0]),
            "non_malicious": float(p_sub[1]),
            "systemic": float(p_sub[2]),
        }
        # 2) packaging
        if four_quadrant is None:
            packaging = {
                "packaging_score": 0.0,
                "path_integrity": 0.0,
                "routing": "UNKNOWN",
            }
        else:
            packaging = self.packaging.score(
                X, four_quadrant,
                applicant_idx=applicant_idx,
                row_shap=row_shap,
            )
        # 3) denoising
        denoise = self.denoising.score_one(
            X.iloc[[applicant_idx]], default_proba=default_proba
        )
        # Final routing decision
        routing = _fraud_routing(
            fraud_score=fraud_score,
            packaging_score=packaging["packaging_score"],
            denoising_action=denoise["denoising_action"],
        )
        return {
            "fraud_score": fraud_score,
            "defaulter_sub_proba": sub_proba,
            "packaging_score": packaging["packaging_score"],
            "path_integrity": packaging["path_integrity"],
            "denoised_default_proba": denoise["denoised_default_proba"],
            "causal_consistency": denoise["causal_consistency"],
            "inflation_strength": denoise["inflation_strength"],
            "routing": routing,
            "routing_reasons": _routing_reasons(
                fraud_score=fraud_score,
                packaging_score=packaging["packaging_score"],
                consistency=denoise["causal_consistency"],
            ),
        }

    def score_batch(
        self,
        X: pd.DataFrame,
        default_proba: np.ndarray,
        four_quadrant: Optional[Dict] = None,
        shap_values: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Score many applicants at once; returns one DataFrame row per applicant.

        Args:
            shap_values: (n, n_features) per-applicant SHAP values used
                by the packaging detector for per-applicant four-quadrant
                classification.  If None, packaging falls back to the
                global quadrant labels.
        """
        rows = []
        for i in range(len(X)):
            row_sv = None if shap_values is None else shap_values[i]
            r = self.score_one(
                X, default_proba[i], four_quadrant,
                applicant_idx=i, row_shap=row_sv,
            )
            r["applicant_idx"] = i
            r["default_proba"] = float(default_proba[i])
            rows.append(r)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------- routing

def _fraud_routing(
    fraud_score: float,
    packaging_score: float,
    denoising_action: str,
) -> str:
    """Combine three sub-signals into a single routing decision.

    Priority (highest first):
    - REJECT_FRAUD: fraud_score >= 0.10
    - REJECT_PACKAGING: packaging_score >= 0.50
    - REVIEW_DENOISED: denoising_action == "FLAG_FOR_REVIEW"
    - REVIEW_BORDERLINE: any signal in [0.3, threshold)
    - PROCEED: clean
    """
    if fraud_score >= 0.10:
        return "REJECT_FRAUD"
    if packaging_score >= 0.50:
        return "REJECT_PACKAGING"
    if denoising_action == "FLAG_FOR_REVIEW":
        return "REVIEW_DENOISED"
    if fraud_score >= 0.05 or packaging_score >= 0.30:
        return "REVIEW_BORDERLINE"
    return "PROCEED"


def _routing_reasons(
    fraud_score: float,
    packaging_score: float,
    consistency: float,
) -> List[str]:
    reasons: List[str] = []
    if fraud_score >= 0.10:
        reasons.append(f"fraud_score={fraud_score:.3f} ≥ 0.10 (P(fraud) high)")
    elif fraud_score >= 0.05:
        reasons.append(f"fraud_score={fraud_score:.3f} in [0.05, 0.10) borderline")
    if packaging_score >= 0.50:
        reasons.append(f"packaging_score={packaging_score:.3f} ≥ 0.50 (UNTRUSTED features dominate)")
    elif packaging_score >= 0.30:
        reasons.append(f"packaging_score={packaging_score:.3f} in [0.30, 0.50) borderline")
    if consistency < 0.5:
        reasons.append(f"causal_consistency={consistency:.2f} < 0.50 (repayment↔consumption decoupled)")
    return reasons
