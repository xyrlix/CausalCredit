"""Evidence chain generator.

Produces structured, traceable evidence for each credit decision,
suitable for inclusion in a decision-report PDF or audit log.

The four generators are:
  1. `generate_risk_evidence`      — top-K SHAP features as risk bullets
  2. `generate_causal_evidence`    — ATE / CATE / robustness facts
  3. `generate_counterfactual_evidence` — DiCE scenarios as text
  4. `generate_full_evidence_report` — markdown combining all of the above
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class EvidenceChainGenerator:
    """Generates evidence chains for credit decisions."""

    # ------------------------------------------------------------------ 1. risk
    def generate_risk_evidence(
        self,
        shap_values: np.ndarray,
        features: Dict[str, float],
        top_k: int = 5,
    ) -> List[Dict]:
        """Top-K SHAP features as structured risk bullets.

        Each bullet is a dict with feature, value, shap, direction, and
        a one-sentence `narrative` describing the contribution.
        """
        if isinstance(features, pd.DataFrame):
            row = features.iloc[0]
            feature_names = list(features.columns)
        else:
            row = pd.Series(features)
            feature_names = list(features.keys())
        # shap_values[0] aligns with the query
        sv = np.asarray(shap_values[0])
        order = np.argsort(-np.abs(sv))[:top_k]
        out: List[Dict] = []
        for j in order:
            f = feature_names[j]
            v = float(row[f]) if f in row.index else None
            s = float(sv[j])
            direction = "increases_default" if s > 0 else "decreases_default"
            narrative = (
                f"{f} = {v:.3g} {direction} by SHAP {s:+.4f}"
                if v is not None else f"{f} {direction} by SHAP {s:+.4f}"
            )
            out.append({
                "feature": f,
                "value": v,
                "shap": s,
                "direction": direction,
                "narrative": narrative,
            })
        return out

    # ------------------------------------------------------------------ 2. causal
    def generate_causal_evidence(
        self,
        ate_results: Dict,
        cate_value: Optional[float] = None,
        subgroup: Optional[str] = None,
    ) -> List[Dict]:
        """Causal evidence chain from ATE / refutation results.

        Each fact is a dict with `claim`, `value`, and `source`.
        """
        facts: List[Dict] = []
        if "ate" in ate_results:
            facts.append({
                "claim": "Average treatment effect on the outcome",
                "value": float(ate_results["ate"]),
                "source": "DoWhy / propensity matching",
            })
        if "ci_lower" in ate_results and "ci_upper" in ate_results:
            facts.append({
                "claim": "95% confidence interval for ATE",
                "value": f"[{ate_results['ci_lower']:.4f}, {ate_results['ci_upper']:.4f}]",
                "source": "bootstrap CI",
            })
        if cate_value is not None:
            facts.append({
                "claim": f"Heterogeneous treatment effect{' for ' + subgroup if subgroup else ''}",
                "value": float(cate_value),
                "source": "EconML CausalForestDML",
            })
        if "robustness_score" in ate_results:
            facts.append({
                "claim": "Refutation robustness (0-1; 1=passes all refuters)",
                "value": float(ate_results["robustness_score"]),
                "source": "DoWhy refuters + E-value",
            })
        if "refutation_results" in ate_results and isinstance(ate_results["refutation_results"], dict):
            for method, r in ate_results["refutation_results"].items():
                passed = bool(r.get("passed", False))
                facts.append({
                    "claim": f"Refuter {method}",
                    "value": "PASS" if passed else "FAIL",
                    "source": f"DoWhy / E-value ({method})",
                })
        return facts

    # ------------------------------------------------------------------ 3. counterfactual
    def generate_counterfactual_evidence(self, cf_result: Dict) -> str:
        """Render DiCE results as a paragraph."""
        n = cf_result.get("n_cfs", 0)
        if n == 0:
            return "No counterfactual scenarios found."
        base = cf_result.get("baseline_proba", float("nan"))
        plaus = cf_result.get("mean_causal_plausibility", 0.0)
        cfs = cf_result.get("cfs", [])
        sentences = [
            f"Baseline P(default) = {base:.2%}; DiCE produced {n} counterfactual scenarios "
            f"with mean plausibility {plaus:.2f}."
        ]
        for c in cfs[:3]:
            changed = sorted(c["deltas"].items(), key=lambda x: -abs(x[1]))[:3]
            ch_str = ", ".join(f"{k}={v:+.2f}" for k, v in changed)
            sentences.append(
                f"  - CF{c['cf_index']}: P={c['counterfactual_proba']:.2%} "
                f"(Δ={c['delta_proba']:+.4f}, plausibility={c['causal_plausibility']:.2f}); "
                f"key changes: {ch_str}"
            )
        return "\n".join(sentences)

    # ------------------------------------------------------------------ 4. full report
    def generate_full_evidence_report(
        self,
        risk_evidence: List[Dict],
        causal_evidence: List[Dict],
        cf_evidence: str,
        decision_summary: Optional[Dict] = None,
    ) -> str:
        """Combine all evidence into a markdown report."""
        lines: List[str] = []
        if decision_summary is not None:
            lines.append("# Decision Evidence Report")
            lines.append("")
            lines.append(f"- Applicant: {decision_summary.get('applicant_id', 'N/A')}")
            lines.append(f"- Timestamp: {decision_summary.get('timestamp', 'N/A')}")
            lines.append(f"- Default probability: {decision_summary.get('default_probability', 0):.2%}")
            lines.append(f"- Credit score: {decision_summary.get('score', 'N/A')}")
            lines.append(f"- Risk grade: {decision_summary.get('risk_grade', 'N/A')}")
            lines.append(f"- Recommendation: {decision_summary.get('decision_suggestion', 'N/A')}")
            lines.append("")

        lines.append("## 1. Risk Factors (SHAP)")
        if not risk_evidence:
            lines.append("_No SHAP evidence available._")
        else:
            lines.append("| # | Feature | Value | SHAP | Direction | Narrative |")
            lines.append("|---|---------|-------|------|-----------|-----------|")
            for i, r in enumerate(risk_evidence, 1):
                v = r.get("value")
                v_str = f"{v:.3g}" if v is not None else "N/A"
                lines.append(
                    f"| {i} | {r['feature']} | {v_str} | {r['shap']:+.4f} | "
                    f"{r['direction']} | {r['narrative']} |"
                )
        lines.append("")

        lines.append("## 2. Causal Evidence")
        if not causal_evidence:
            lines.append("_No causal evidence available._")
        else:
            lines.append("| Claim | Value | Source |")
            lines.append("|-------|-------|--------|")
            for r in causal_evidence:
                v = r['value']
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"| {r['claim']} | {v_str} | {r['source']} |")
        lines.append("")

        lines.append("## 3. Counterfactual Scenarios")
        lines.append(cf_evidence)
        lines.append("")

        return "\n".join(lines)
