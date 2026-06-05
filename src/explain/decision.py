"""Decision advisory engine.

Generates personalized loan decision recommendations based on causal
analysis and counterfactual reasoning. The output shape matches the
API `CreditResponse` schema in `src/api/schemas.py`:
    - `score`         int [300, 850]
    - `risk_grade`    str   A | B | C | D | E
    - `default_probability`  float [0, 1]
    - `causal_effect`, `counterfactual`, `explanation`, `decision_suggestion`  optional dicts/lists/str

Mapping (per docs and schema):
    score = round(300 + 550 * (1 - p_default)**1.5)
    grade A: score >= 750
    grade B: score >= 650
    grade C: score >= 550
    grade D: score >= 450
    grade E: else
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ===========================================================================
# Helpers
# ===========================================================================

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _score_to_grade(score: int) -> str:
    if score >= 750:
        return "A"
    if score >= 650:
        return "B"
    if score >= 550:
        return "C"
    if score >= 450:
        return "D"
    return "E"


def _grade_recommendation(grade: str, p_default: float) -> str:
    if grade in ("A", "B"):
        return "APPROVE — low expected loss"
    if grade == "C":
        return "REVIEW — moderate risk; consider income verification"
    if grade == "D":
        return "REFER — elevated risk; require collateral or co-signer"
    return "DECLINE — high risk; recommend rejection or sub-prime product"


# ===========================================================================
# DecisionAdvisor
# ===========================================================================

class DecisionAdvisor:
    """Credit decision advisory engine.

    Args:
        counterfactual_reasoner: optional `CounterfactualReasoner` instance.
        shap_explainer: optional `SHAPExplainer` instance.
        cate_estimator: optional `CATEEstimator` instance.
    """

    def __init__(
        self,
        counterfactual_reasoner: Any = None,
        shap_explainer: Any = None,
        cate_estimator: Any = None,
    ):
        self.counterfactual_reasoner = counterfactual_reasoner
        self.shap_explainer = shap_explainer
        self.cate_estimator = cate_estimator

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def compute_score(p_default: float) -> int:
        """Convert default probability to a 300-850 credit score."""
        p = _clamp(p_default, 0.0, 1.0)
        return int(round(300.0 + 550.0 * (1.0 - p) ** 1.5))

    @staticmethod
    def compute_grade(score: int) -> str:
        return _score_to_grade(score)

    @staticmethod
    def compute_recommendation(grade: str, p_default: float) -> str:
        return _grade_recommendation(grade, p_default)

    # ------------------------------------------------------------------ report
    def generate_decision_report(
        self,
        features: Dict[str, float],
        applicant_id: Optional[str] = None,
        default_probability: Optional[float] = None,
        shap_values: Optional[np.ndarray] = None,
        X_for_shap: Optional[pd.DataFrame] = None,
        cate_value: Optional[float] = None,
        cf_results: Optional[Dict] = None,
        causal_effect_summary: Optional[Dict] = None,
        four_quadrant: Optional[Dict] = None,
    ) -> Dict:
        """Generate a complete decision report dict.

        Args:
            features: raw applicant features (used for context / narrative).
            applicant_id: optional external ID.
            default_probability: model P(default). If None, falls back to
                the counterfactual_reasoner model.
            shap_values: (n, d) SHAP values for `X_for_shap`. The report
                uses row 0 of these values.
            X_for_shap: DataFrame aligned with `shap_values` (row 0 = this
                applicant).
            cate_value: optional |CATE| for the relevant treatment.
            cf_results: dict from `CounterfactualReasoner.generate_counterfactuals`.
            causal_effect_summary: free-form dict with ATE / robustness info.
            four_quadrant: dict from `SHAPExplainer.causal_vs_noncausal_contribution`.
        """
        # Probability & score
        if default_probability is None:
            if self.counterfactual_reasoner is None:
                raise ValueError("default_probability required when no counterfactual_reasoner is wired")
            default_probability = self.counterfactual_reasoner._baseline_proba(features)
        p = float(_clamp(default_probability, 0.0, 1.0))
        score = self.compute_score(p)
        grade = self.compute_grade(score)
        recommendation = self.compute_recommendation(grade, p)

        # Top risk factors (with quadrant status if available)
        top_risk_factors = self._build_risk_factors(
            features, shap_values, X_for_shap, four_quadrant, top_k=5
        )

        # CATE insights
        cate_insights = self._build_cate_insights(cate_value, four_quadrant, causal_effect_summary)

        # Counterfactual recommendations
        cf_recommendations = self._build_cf_recommendations(features, cf_results)

        # Causal narrative
        narrative = self._build_narrative(
            features, p, score, grade, top_risk_factors,
            cate_value=cate_value, cf_results=cf_results,
        )

        return {
            "applicant_id": applicant_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "default_probability": round(p, 4),
            "score": score,
            "risk_grade": grade,
            "decision_suggestion": recommendation,
            "top_risk_factors": top_risk_factors,
            "cate_insights": cate_insights,
            "counterfactual_recommendations": cf_recommendations,
            "causal_narrative": narrative,
        }

    # ------------------------------------------------------------------ suggestions
    def generate_suggestion_text(self, report: Dict, language: str = "zh") -> str:
        """Render the report as a human-readable suggestion in zh / en."""
        if language == "zh":
            lines = [
                f"申请人ID: {report.get('applicant_id', 'N/A')}",
                f"违约概率: {report['default_probability']:.2%}",
                f"信用评分: {report['score']} (等级 {report['risk_grade']})",
                f"决策建议: {report['decision_suggestion']}",
                "",
                "主要风险因素:",
            ]
            for r in report.get("top_risk_factors", []):
                lines.append(f"  - {r['feature']} (SHAP={r['shap']:+.4f}, 四象限={r['quadrant']})")
            if report.get("counterfactual_recommendations"):
                lines.append("")
                lines.append("反事实建议:")
                for c in report["counterfactual_recommendations"]:
                    lines.append(f"  - {c['description']}")
            if report.get("causal_narrative"):
                lines.append("")
                lines.append("因果解读:")
                lines.append(f"  {report['causal_narrative']}")
        else:
            lines = [
                f"Applicant ID: {report.get('applicant_id', 'N/A')}",
                f"Default probability: {report['default_probability']:.2%}",
                f"Credit score: {report['score']} (grade {report['risk_grade']})",
                f"Recommendation: {report['decision_suggestion']}",
                "",
                "Top risk factors:",
            ]
            for r in report.get("top_risk_factors", []):
                lines.append(f"  - {r['feature']} (SHAP={r['shap']:+.4f}, quadrant={r['quadrant']})")
            if report.get("counterfactual_recommendations"):
                lines.append("")
                lines.append("Counterfactual recommendations:")
                for c in report["counterfactual_recommendations"]:
                    lines.append(f"  - {c['description']}")
            if report.get("causal_narrative"):
                lines.append("")
                lines.append("Causal narrative:")
                lines.append(f"  {report['causal_narrative']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ subgroup compare
    def compare_subgroup_effect(
        self,
        features: Dict[str, float],
        subgroup_col: str = "thin_credit_flag",
        cate_for_subgroup: Optional[Dict] = None,
    ) -> Dict:
        """Compare this applicant's CATE vs a subgroup (e.g. thin credit).

        Args:
            features: applicant features.
            subgroup_col: name of the subgroup flag (kept for API parity).
            cate_for_subgroup: optional dict {subgroup_name: CATE_value}.
        """
        if cate_for_subgroup is None:
            cate_for_subgroup = {}
        return {
            "applicant_features": features,
            "subgroup_col": subgroup_col,
            "subgroup_cates": cate_for_subgroup,
            "narrative": (
                f"Applicant CATE vs subgroup {subgroup_col}: "
                f"{cate_for_subgroup.get('applicant', 'N/A')} vs {cate_for_subgroup.get('subgroup_mean', 'N/A')}"
            ),
        }

    # ------------------------------------------------------------------ helpers
    def _build_risk_factors(
        self,
        features: Dict[str, float],
        shap_values: Optional[np.ndarray],
        X_for_shap: Optional[pd.DataFrame],
        four_quadrant: Optional[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Build the top-K risk-factor list with quadrant status."""
        if shap_values is None or X_for_shap is None:
            return []
        row_sv = shap_values[0]
        # quadrant lookup
        quad_map = {}
        if four_quadrant and "per_feature" in four_quadrant:
            for _, r in four_quadrant["per_feature"].iterrows():
                quad_map[r["feature"]] = r["quadrant"]
        order = np.argsort(-np.abs(row_sv))[:top_k]
        factors = []
        for j in order:
            f = X_for_shap.columns[j]
            factors.append({
                "feature": str(f),
                "value": float(X_for_shap.iloc[0][f]),
                "shap": float(row_sv[j]),
                "quadrant": quad_map.get(f, "UNKNOWN"),
            })
        return factors

    def _build_cate_insights(
        self,
        cate_value: Optional[float],
        four_quadrant: Optional[Dict],
        causal_effect_summary: Optional[Dict],
    ) -> List[str]:
        insights: List[str] = []
        if cate_value is not None:
            insights.append(f"Heterogeneous treatment effect (CATE) for this applicant: {cate_value:.4f}")
        if four_quadrant is not None and "counts" in four_quadrant:
            counts = four_quadrant["counts"].to_dict() if hasattr(four_quadrant["counts"], "to_dict") else four_quadrant["counts"]
            insights.append(
                "Four-quadrant distribution: "
                f"TRUSTED={counts.get('TRUSTED', 0)}, "
                f"UNTRUSTED={counts.get('UNTRUSTED', 0)}, "
                f"NEGLIGIBLE={counts.get('NEGLIGIBLE', 0)}, "
                f"MASKED={counts.get('MASKED', 0)}"
            )
        if causal_effect_summary is not None:
            if "ate" in causal_effect_summary:
                insights.append(f"ATE estimate: {causal_effect_summary['ate']:.4f}")
            if "robustness_score" in causal_effect_summary:
                insights.append(f"Refutation robustness score: {causal_effect_summary['robustness_score']:.2f}")
        return insights

    def _build_cf_recommendations(
        self,
        features: Dict[str, float],
        cf_results: Optional[Dict],
    ) -> List[Dict]:
        if cf_results is None or "cfs" not in cf_results:
            return []
        recs = []
        for c in cf_results["cfs"][:3]:
            changed = sorted(c["deltas"].items(), key=lambda x: -abs(x[1]))[:3]
            descr = (
                f"Change P(default) from {cf_results['baseline_proba']:.2%} "
                f"to {c['counterfactual_proba']:.2%} (Δ={c['delta_proba']:+.4f}) by "
                f"adjusting: " + ", ".join(f"{k}={v:+.2f}" for k, v in changed)
            )
            recs.append({
                "cf_index": c["cf_index"],
                "description": descr,
                "delta_proba": float(c["delta_proba"]),
                "causal_plausibility": float(c["causal_plausibility"]),
            })
        return recs

    def _build_narrative(
        self,
        features: Dict[str, float],
        p: float,
        score: int,
        grade: str,
        top_risk_factors: List[Dict],
        cate_value: Optional[float] = None,
        cf_results: Optional[Dict] = None,
    ) -> str:
        parts = [f"Default risk is {p:.2%} (score={score}, grade {grade})."]
        if top_risk_factors:
            top = top_risk_factors[0]
            parts.append(
                f"Primary driver: {top['feature']} (SHAP={top['shap']:+.4f}, "
                f"quadrant={top['quadrant']})."
            )
        if cate_value is not None:
            parts.append(f"Heterogeneous effect estimate: |CATE|={abs(cate_value):.4f}.")
        if cf_results is not None and cf_results.get("n_cfs", 0) > 0:
            n = cf_results["n_cfs"]
            plaus = cf_results.get("mean_causal_plausibility", 0.0)
            parts.append(
                f"DiCE found {n} counterfactual scenario(s) for the applicant; "
                f"mean plausibility={plaus:.2f}."
            )
        return " ".join(parts)
