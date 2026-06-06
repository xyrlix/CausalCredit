"""FraudGuard — orchestrator for the three anti-fraud modules.

Combines:
* ``ThreeClassFraudClassifier`` → fraud_score (P(fraudulent))
* ``PackagingDetector``        → packaging_score (path integrity)
* ``CausalDenoisingScorer``    → denoised_default_proba

into a single per-applicant output suitable for embedding in the
existing decision report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.fraud.three_class import ThreeClassFraudClassifier, ALL_LABELS, DEFRAUDER_CLASSES
from src.fraud.packaging import PackagingDetector
from src.fraud.denoising import CausalDenoisingScorer


@dataclass
class FraudGuardConfig:
    """Routing thresholds for the anti-fraud guard.

    All thresholds live here so they can be tuned per-deployment
    (e.g. tighten ``fraud_reject_threshold`` from 0.10 to 0.05 for
    a high-risk product, loosen ``packaging_reject_threshold``
    from 0.50 to 0.60 for a thin-file product) without touching
    the code.  Loaded from ``configs/config.yaml``::

        fraud_guard:
          fraud_reject_threshold: 0.10
          fraud_borderline_threshold: 0.05
          packaging_reject_threshold: 0.50
          packaging_borderline_threshold: 0.30
          consistency_flag_threshold: 0.50
    """
    fraud_reject_threshold: float = 0.10
    fraud_borderline_threshold: float = 0.05
    packaging_reject_threshold: float = 0.50
    packaging_borderline_threshold: float = 0.30
    consistency_flag_threshold: float = 0.50

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "FraudGuardConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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
        config: Optional[FraudGuardConfig] = None,
    ):
        self.classifier = ThreeClassFraudClassifier(params=classifier_params)
        self.packaging = PackagingDetector(**(packaging_kwargs or {}))
        self.denoising = CausalDenoisingScorer(**(denoising_kwargs or {}))
        self.config = config or FraudGuardConfig()

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
            config=self.config,
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
                config=self.config,
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
    config: Optional[FraudGuardConfig] = None,
) -> str:
    """Combine three sub-signals into a single routing decision.

    Priority (highest first):
    - REJECT_FRAUD:     fraud_score >= config.fraud_reject_threshold
    - REJECT_PACKAGING: packaging_score >= config.packaging_reject_threshold
    - REVIEW_DENOISED:  denoising_action == "FLAG_FOR_REVIEW"
    - REVIEW_BORDERLINE: fraud_score >= config.fraud_borderline_threshold OR
                         packaging_score >= config.packaging_borderline_threshold
    - PROCEED: clean

    Falls back to the dataclass defaults if ``config`` is None.
    """
    cfg = config or FraudGuardConfig()
    if fraud_score >= cfg.fraud_reject_threshold:
        return "REJECT_FRAUD"
    if packaging_score >= cfg.packaging_reject_threshold:
        return "REJECT_PACKAGING"
    if denoising_action == "FLAG_FOR_REVIEW":
        return "REVIEW_DENOISED"
    if fraud_score >= cfg.fraud_borderline_threshold or packaging_score >= cfg.packaging_borderline_threshold:
        return "REVIEW_BORDERLINE"
    return "PROCEED"


def _routing_reasons(
    fraud_score: float,
    packaging_score: float,
    consistency: float,
    config: Optional[FraudGuardConfig] = None,
) -> List[str]:
    """Build a human-readable list of why an applicant was routed where they were."""
    cfg = config or FraudGuardConfig()
    reasons: List[str] = []
    if fraud_score >= cfg.fraud_reject_threshold:
        reasons.append(f"fraud_score={fraud_score:.3f} ≥ {cfg.fraud_reject_threshold:.2f} (P(fraud) high)")
    elif fraud_score >= cfg.fraud_borderline_threshold:
        reasons.append(
            f"fraud_score={fraud_score:.3f} in [{cfg.fraud_borderline_threshold:.2f}, {cfg.fraud_reject_threshold:.2f}) borderline"
        )
    if packaging_score >= cfg.packaging_reject_threshold:
        reasons.append(
            f"packaging_score={packaging_score:.3f} ≥ {cfg.packaging_reject_threshold:.2f} (UNTRUSTED features dominate)"
        )
    elif packaging_score >= cfg.packaging_borderline_threshold:
        reasons.append(
            f"packaging_score={packaging_score:.3f} in [{cfg.packaging_borderline_threshold:.2f}, {cfg.packaging_reject_threshold:.2f}) borderline"
        )
    if consistency < cfg.consistency_flag_threshold:
        reasons.append(
            f"causal_consistency={consistency:.2f} < {cfg.consistency_flag_threshold:.2f} (repayment↔consumption decoupled)"
        )
    return reasons
