"""Business logic for the CausalCredit API.

Each method maps an API request to one or more pieces of the trained
artefacts in `ModelRegistry`. The services are thin orchestration —
all heavy lifting (model.predict, SHAP, DiCE, DoWhy) lives in
`src/models`, `src/explain`, and `src/causal`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .dependencies import ModelRegistry
from .schemas import (
    CausalEffectRequest,
    CausalEffectResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    CreditRequest,
    CreditResponse,
    ExplainRequest,
    ExplainResponse,
)


class CreditScoringService:
    """Orchestrates scoring + explanation + causal queries."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.r = registry

    # ------------------------------------------------------------------
    # /score
    # ------------------------------------------------------------------
    def score(self, request: CreditRequest) -> CreditResponse:
        X1 = self.r.transform_features(request.features)
        p_raw = float(self.r.lgbm_model.predict_proba(X1)[:, 1][0])
        p_cal = float(self.r.calibrator.transform(np.array([p_raw]))[0]) if self.r.calibrator else p_raw

        score = self._compute_score(p_cal)
        grade = self._compute_grade(score)
        suggestion = self._compute_suggestion(grade, p_cal)

        explanation: Optional[Dict] = None
        if request.include_explanation:
            sv = self.r.shap_explainer.compute_shap_values(X1)
            top = self._top_features_from_shap(sv, X1, top_k=5)
            explanation = {"top_features": top, "method": "TreeSHAP"}

        counterfactual: Optional[List[Dict]] = None
        if request.include_counterfactual:
            try:
                cf_res = self.r.counterfactual_reasoner.generate_counterfactuals(
                    {c: float(X1.iloc[0][c]) for c in self.r.feature_cols},
                    total_cfs=3, desired_class=0,
                )
                counterfactual = [
                    {
                        "cf_index": c["cf_index"],
                        "counterfactual_proba": c["counterfactual_proba"],
                        "delta_proba": c["delta_proba"],
                        "causal_plausibility": c["causal_plausibility"],
                        "deltas": c["deltas"],
                    }
                    for c in cf_res.get("cfs", [])
                ]
            except Exception as exc:
                counterfactual = [{"error": str(exc)}]

        return CreditResponse(
            score=score,
            default_probability=p_cal,
            risk_grade=grade,
            causal_effect=self.r.ate_summary or None,
            counterfactual=counterfactual,
            explanation=explanation,
            decision_suggestion=suggestion,
        )

    # ------------------------------------------------------------------
    # /counterfactual
    # ------------------------------------------------------------------
    def counterfactual(self, request: CounterfactualRequest) -> CounterfactualResponse:
        X0 = self.r.transform_features(request.features)
        p0 = float(self.r.lgbm_model.predict_proba(X0)[:, 1][0])

        # Apply interventions
        intervened = dict(request.features)
        intervened.update(request.interventions)
        X1 = self.r.transform_features(intervened)
        p1 = float(self.r.lgbm_model.predict_proba(X1)[:, 1][0])

        # Plausibility: ratio of intervention magnitude to historical ±2σ
        plausibility = 1.0
        for k, v in request.interventions.items():
            base = float(request.features.get(k, 0.0))
            if abs(v - base) > 0:
                hist_std = float(self.r.training_data[k].std()) if k in self.r.training_data.columns else 1.0
                plausibility = min(plausibility, 1.0 - min(abs(v - base) / (2 * hist_std + 1e-6), 1.0))

        return CounterfactualResponse(
            baseline_probability=p0,
            counterfactual_probability=p1,
            probability_change=p1 - p0,
            intervention_details=request.interventions,
            confidence=plausibility,
        )

    # ------------------------------------------------------------------
    # /explain
    # ------------------------------------------------------------------
    def explain(self, request: ExplainRequest) -> ExplainResponse:
        X1 = self.r.transform_features(request.features)
        sv = self.r.shap_explainer.compute_shap_values(X1)
        top = self._top_features_from_shap(sv, X1, top_k=request.top_k)
        evidence = self.r.evidence_generator.generate_risk_evidence(sv, X1, top_k=request.top_k)
        return ExplainResponse(top_features=top, evidence_chain=evidence)

    # ------------------------------------------------------------------
    # /causal-effect
    # ------------------------------------------------------------------
    def causal_effect(self, request: CausalEffectRequest) -> CausalEffectResponse:
        if not self.r.ate_summary:
            return CausalEffectResponse(
                ate=0.0,
                ate_ci=(0.0, 0.0),
                cate_subgroup=None,
                refutation_results={"note": "ATE pre-compute unavailable"},
            )
        s = self.r.ate_summary
        return CausalEffectResponse(
            ate=s["ate"],
            ate_ci=(s["ci_lower"], s["ci_upper"]),
            cate_subgroup=None,
            refutation_results={"method": s.get("method", "DoWhy")},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_score(p: float) -> int:
        p = max(0.0, min(1.0, p))
        return int(round(300 + 550 * (1 - p) ** 1.5))

    @staticmethod
    def _compute_grade(score: int) -> str:
        if score >= 750:
            return "A"
        if score >= 650:
            return "B"
        if score >= 550:
            return "C"
        if score >= 450:
            return "D"
        return "E"

    @staticmethod
    def _compute_suggestion(grade: str, p: float) -> str:
        if grade == "A":
            return "APPROVE — low expected loss"
        if grade == "B":
            return "APPROVE with standard terms"
        if grade == "C":
            return "REVIEW — request additional documentation"
        if grade == "D":
            return "REVIEW — secured product or higher rate"
        return "DECLINE — high risk; recommend rejection or sub-prime product"

    def _top_features_from_shap(self, sv: np.ndarray, X1: pd.DataFrame, top_k: int) -> List[Dict]:
        if sv.ndim == 1:
            row = sv
        else:
            row = sv[0]
        order = np.argsort(-np.abs(row))[:top_k]
        return [
            {
                "feature": self.r.feature_cols[i],
                "value": float(X1.iloc[0][self.r.feature_cols[i]]),
                "shap": float(row[i]),
                "direction": "increases_default" if row[i] > 0 else "decreases_default",
            }
            for i in order
        ]
