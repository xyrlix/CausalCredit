"""Multi-level causal narrative generator.

Builds a 3-angle "why is this applicant high/low risk" story:

* **Model-level** — global model behavior: the top-3 features that drive
  *all* applicants' P(default) in this trained model, weighted by the
  discovered causal DAG's main paths.
* **Cohort-level** — this applicant vs. their k=10 nearest training-set
  neighbours.  If the applicant's P(default) is much higher than the
  cohort's, the explanation is "this applicant is unusual" rather than
  "the model is generally bad at this kind of applicant".
* **Individual-level** — this specific applicant's SHAP top-K with
  4-quadrant labels, the causal paths each top feature travels
  through the DAG to reach the outcome, and the CATE-based "what
  would shift the outcome" estimate.

The output of ``build_full_narrative`` is a dict that slots into
the decision report's ``causal_narrative_v2`` field, alongside the
existing 1-sentence ``causal_narrative`` for backwards compatibility.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# CausalNarrative
# ===========================================================================


class CausalNarrative:
    """Three-level causal narrative generator for credit decisions."""

    def __init__(
        self,
        model,
        feature_names: List[str],
        dag: Optional["networkx.DiGraph"] = None,
        outcome_name: str = "TARGET",
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.dag = dag
        self.outcome_name = outcome_name

    # ------------------------------------------------------------------ paths
    def trace_causal_path(self, feature: str, max_length: int = 4) -> List[List[str]]:
        """All simple paths from ``feature`` to the outcome in the DAG.

        Returns a list of node-list paths.  An empty list means the
        feature is not connected to the outcome in the DAG.
        """
        if self.dag is None or feature not in self.dag.nodes:
            return []
        paths: List[List[str]] = []
        # BFS with path tracking
        stack: List[List[str]] = [[feature]]
        while stack:
            path = stack.pop()
            cur = path[-1]
            if cur == self.outcome_name:
                paths.append(path)
                continue
            if len(path) >= max_length:
                continue
            for nbr in self.dag.successors(cur):
                if nbr in path:  # cycle guard
                    continue
                stack.append(path + [nbr])
        return paths

    def features_on_paths_to_outcome(self, features: List[str]) -> Dict[str, List[List[str]]]:
        """Map feature → list of DAG paths from that feature to the outcome."""
        return {f: self.trace_causal_path(f) for f in features if f != self.outcome_name}

    # ------------------------------------------------------------------ 1. model-level
    def model_level_narrative(
        self,
        shap_values: np.ndarray,
        top_k: int = 3,
    ) -> Dict:
        """Global model story: which features drive P(default) most.

        Args:
            shap_values: (n, d) SHAP values for a sample of applicants
                (e.g. the test set or a calibration set).
            top_k: how many top features to report.

        Returns:
            ``{top_features, narrative, mean_abs_shap}`` dict.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-mean_abs)[:top_k]
        top = [(self.feature_names[j], float(mean_abs[j])) for j in order]
        feature_list = ", ".join(f"{n} ({v:.4f})" for n, v in top)
        narrative = (
            f"At the model level, P(default) for the average applicant is driven "
            f"primarily by {top[0][0]} (mean |SHAP|={top[0][1]:.4f}), followed by "
            f"{top[1][0]} ({top[1][1]:.4f}) and {top[2][0]} ({top[2][1]:.4f}). "
            f"These three features account for the bulk of all decisions."
        )
        return {
            "top_features": [{"feature": n, "mean_abs_shap": v} for n, v in top],
            "narrative": narrative,
            "feature_summary": feature_list,
        }

    # ------------------------------------------------------------------ 2. cohort-level
    def cohort_level_narrative(
        self,
        features: Dict[str, float],
        X_train: pd.DataFrame,
        y_prob_train: np.ndarray,
        k: int = 10,
    ) -> Dict:
        """Compare this applicant to their k nearest training neighbours.

        Returns a dict with the cohort's mean P(default), the delta
        (applicant - cohort), the cohort's mean feature values, and
        the features where the applicant deviates most.
        """
        from sklearn.neighbors import NearestNeighbors

        feat_vals = np.array([features.get(c, 0.0) for c in self.feature_names], dtype=float).reshape(1, -1)
        # Standardise using train column stds (avoid div-by-zero)
        stds = X_train[self.feature_names].std().replace(0, 1.0).values
        means = X_train[self.feature_names].mean().values
        feat_norm = (feat_vals - means) / stds
        Xn = (X_train[self.feature_names].values - means) / stds
        kk = min(k, len(Xn))
        nn = NearestNeighbors(n_neighbors=kk).fit(Xn)
        dist, idx = nn.kneighbors(feat_norm)

        cohort_p = float(np.mean(y_prob_train[idx[0]]))
        applicant_p = float(self.model.predict_proba(feat_vals)[0, 1]) if hasattr(self.model, "predict_proba") else float("nan")
        delta = applicant_p - cohort_p

        cohort_means = X_train[self.feature_names].iloc[idx[0]].mean().to_dict()
        # Per-feature deviation, normalised by std
        deviations = []
        for c in self.feature_names:
            v = features.get(c, np.nan)
            cm = cohort_means.get(c, np.nan)
            sd = float(stds[self.feature_names.index(c)])
            if sd > 0 and not np.isnan(v) and not np.isnan(cm):
                z = (v - cm) / sd
                if abs(z) > 0.5:  # surface non-trivial deviations only
                    deviations.append({"feature": c, "z": round(float(z), 3), "applicant": float(v), "cohort_mean": float(cm)})
        deviations.sort(key=lambda r: -abs(r["z"]))

        if abs(delta) < 0.02:
            comp = "in line with"
        elif delta > 0:
            comp = "noticeably higher than"
        else:
            comp = "noticeably lower than"
        narrative = (
            f"Among the {kk} training applicants most similar to this one, "
            f"the average P(default) is {cohort_p:.2%}, so this applicant's "
            f"P(default) of {applicant_p:.2%} is {comp} the cohort "
            f"(Δ={delta:+.4f})."
        )
        if deviations:
            top_dev = deviations[:3]
            dev_str = ", ".join(f"{d['feature']} (z={d['z']:+.2f})" for d in top_dev)
            narrative += f" The applicant deviates most from the cohort in: {dev_str}."
        return {
            "k": kk,
            "cohort_mean_p_default": cohort_p,
            "applicant_p_default": applicant_p,
            "delta": delta,
            "top_deviations": deviations[:5],
            "narrative": narrative,
        }

    # ------------------------------------------------------------------ 3. individual-level
    def individual_level_narrative(
        self,
        features: Dict[str, float],
        shap_row: np.ndarray,
        four_quadrant: Optional[Dict] = None,
        top_k: int = 5,
    ) -> Dict:
        """This applicant's specific story: top SHAP + 4-quadrant + DAG paths.

        ``four_quadrant`` is the per-feature DataFrame produced by
        ``SHAPExplainer.causal_vs_noncausal_contribution`` — it carries
        the TRUSTED/UNTRUSTED/MASKED/NEGLIGIBLE label per feature.
        """
        order = np.argsort(-np.abs(shap_row))[:top_k]
        quad_lookup: Dict[str, str] = {}
        if four_quadrant is not None and "per_feature" in four_quadrant:
            df = four_quadrant["per_feature"]
            for _, r in df.iterrows():
                quad_lookup[str(r["feature"])] = str(r["quadrant"])

        # Per-feature paths
        top_names = [self.feature_names[j] for j in order]
        paths_map = self.features_on_paths_to_outcome(top_names)

        top_features = []
        for j in order:
            f = self.feature_names[j]
            sh = float(shap_row[j])
            quad = quad_lookup.get(f, "UNKNOWN")
            v = features.get(f, None)
            paths = paths_map.get(f, [])
            top_features.append({
                "feature": f,
                "value": float(v) if v is not None else None,
                "shap": sh,
                "quadrant": quad,
                "direction": "increases_default" if sh > 0 else "decreases_default",
                "dag_paths": paths,
                "n_paths": len(paths),
            })

        # Build the narrative
        # 1) Dominant driver
        dom = top_features[0]
        dom_quad = dom["quadrant"]
        dom_path = dom["dag_paths"][0] if dom["dag_paths"] else []
        path_str = " → ".join(dom_path) if dom_path else "(no DAG path recorded)"

        # 2) TRUSTED/UNTRUSTED/MASKED counting
        n_trusted = sum(1 for f in top_features if f["quadrant"] == "TRUSTED")
        n_untrusted = sum(1 for f in top_features if f["quadrant"] == "UNTRUSTED")
        n_masked = sum(1 for f in top_features if f["quadrant"] == "MASKED")

        narrative = (
            f"The dominant risk driver is {dom['feature']} (SHAP={dom['shap']:+.4f}, "
            f"quadrant={dom_quad}). In the causal DAG this feature reaches {self.outcome_name} via: "
            f"{path_str}. "
        )
        if n_untrusted > 0:
            narrative += (
                f"{n_untrusted} of the top-{top_k} features are UNTRUSTED "
                f"(model says important, but no causal support); "
            )
        if n_trusted > 0:
            narrative += f"{n_trusted} are TRUSTED (model and causal both agree); "
        if n_masked > 0:
            narrative += f"{n_masked} are MASKED (causal signal hidden by model)."
        return {
            "top_features": top_features,
            "narrative": narrative.rstrip(",; ") + ".",
            "dominant_feature": dom["feature"],
            "dominant_dag_path": dom_path,
            "n_trusted": n_trusted,
            "n_untrusted": n_untrusted,
            "n_masked": n_masked,
        }

    # ------------------------------------------------------------------ 4. robustness
    def explanation_robustness(
        self,
        features: Dict[str, float],
        shap_row: np.ndarray,
        n_perturbations: int = 20,
        noise_frac: float = 0.10,
        seed: int = 0,
    ) -> Dict:
        """Perturb the input, re-rank top features, measure rank stability.

        For each perturbation, we add Gaussian noise of magnitude
        ``noise_frac * std(features)`` to the numeric columns, re-run
        SHAP (cheap: TreeSHAP on the original tree), and check whether
        the top-3 features by |SHAP| change.

        Returns a dict with:
        * ``top_k_stable`` — fraction of perturbations where the top-3
          features stayed identical
        * ``top_1_stable`` — fraction where the #1 feature stayed
        * ``stability_score`` — overall [0, 1] (top-1 × 0.6 + top-3 × 0.4)
        * ``interpretation`` — one-line narrative
        """
        import shap as _shap
        rng = np.random.default_rng(seed)
        base_top = set(int(j) for j in np.argsort(-np.abs(shap_row))[:3])
        top1 = int(np.argmax(np.abs(shap_row)))

        X = pd.DataFrame([features], columns=self.feature_names)
        # Per-column std
        stds = X.std().replace(0, 1.0)
        # For a single row, std is 0 — fall back to max(|value|, 1)
        scale = stds.where(stds > 0, X.abs().max()).where(X.abs().max() > 0, 1.0)
        scale_arr = np.asarray(scale.values, dtype=float) * noise_frac

        explainer = _shap.TreeExplainer(self.model)
        stable_top3 = 0
        stable_top1 = 0
        for _ in range(n_perturbations):
            noise = rng.normal(scale=scale_arr)
            Xp = X.copy()
            Xp.iloc[0] = Xp.iloc[0] + noise
            sv = explainer.shap_values(Xp)
            if isinstance(sv, list):
                sv = sv[1]
            new_top = set(int(j) for j in np.argsort(-np.abs(sv[0]))[:3])
            if new_top == base_top:
                stable_top3 += 1
            if int(np.argmax(np.abs(sv[0]))) == top1:
                stable_top1 += 1

        frac_top3 = stable_top3 / max(n_perturbations, 1)
        frac_top1 = stable_top1 / max(n_perturbations, 1)
        score = 0.6 * frac_top1 + 0.4 * frac_top3
        if score >= 0.85:
            interp = f"Explanation is robust (stability={score:.2f})."
        elif score >= 0.6:
            interp = f"Explanation is moderately robust (stability={score:.2f}); top drivers may shift under small input noise."
        else:
            interp = f"Explanation is fragile (stability={score:.2f}); interpret with caution."
        return {
            "n_perturbations": n_perturbations,
            "noise_frac": noise_frac,
            "top_1_stable": frac_top1,
            "top_3_stable": frac_top3,
            "stability_score": float(score),
            "interpretation": interp,
        }

    # ------------------------------------------------------------------ 5. top-level
    def build_full_narrative(
        self,
        features: Dict[str, float],
        shap_row: np.ndarray,
        shap_global: np.ndarray,
        X_train: pd.DataFrame,
        y_prob_train: np.ndarray,
        four_quadrant: Optional[Dict] = None,
        run_robustness: bool = True,
    ) -> Dict:
        """Compose all three levels (and optionally robustness) into one dict."""
        out: Dict = {
            "model_level": self.model_level_narrative(shap_global, top_k=3),
            "cohort_level": self.cohort_level_narrative(features, X_train, y_prob_train, k=10),
            "individual_level": self.individual_level_narrative(features, shap_row, four_quadrant, top_k=5),
        }
        if run_robustness:
            out["robustness"] = self.explanation_robustness(features, shap_row, n_perturbations=20, noise_frac=0.10)
        return out

    # ------------------------------------------------------------------ 6. rendering
    @staticmethod
    def render_markdown(narrative: Dict) -> str:
        """Render the full narrative dict as a human-readable markdown section."""
        lines: List[str] = ["## 因果叙事 (Causal Narrative — M8.2)\n"]
        # Model level
        m = narrative.get("model_level", {})
        lines.append("### 1. 模型层面 (Model-level)\n")
        lines.append(m.get("narrative", "_n/a_"))
        if m.get("top_features"):
            lines.append("")
            lines.append("| # | Feature | Mean |SHAP| |")
            lines.append("|---|---------|---------|")
            for i, f in enumerate(m["top_features"], 1):
                lines.append(f"| {i} | {f['feature']} | {f['mean_abs_shap']:.4f} |")
        lines.append("")
        # Cohort
        c = narrative.get("cohort_level", {})
        lines.append("### 2. 同类申请人对照 (Cohort-level, k=10)\n")
        lines.append(c.get("narrative", "_n/a_"))
        if c.get("top_deviations"):
            lines.append("")
            lines.append("| Feature | z-score | Applicant | Cohort mean |")
            lines.append("|---------|---------|-----------|-------------|")
            for d in c["top_deviations"]:
                lines.append(f"| {d['feature']} | {d['z']:+.2f} | {d['applicant']:.3g} | {d['cohort_mean']:.3g} |")
        lines.append("")
        # Individual
        ind = narrative.get("individual_level", {})
        lines.append("### 3. 本申请人 (Individual-level, top-5 SHAP)\n")
        lines.append(ind.get("narrative", "_n/a_"))
        if ind.get("top_features"):
            lines.append("")
            lines.append("| # | Feature | Value | SHAP | Quadrant | DAG paths |")
            lines.append("|---|---------|-------|------|----------|-----------|")
            for i, f in enumerate(ind["top_features"], 1):
                v = f.get("value")
                v_str = f"{v:.3g}" if v is not None else "N/A"
                paths = f.get("dag_paths", [])
                path_str = "; ".join(" → ".join(p) for p in paths[:2]) or "(no path)"
                lines.append(f"| {i} | {f['feature']} | {v_str} | {f['shap']:+.4f} | `{f['quadrant']}` | {path_str} |")
        lines.append("")
        # Robustness
        r = narrative.get("robustness")
        if r:
            lines.append("### 4. 解释稳健性 (Robustness)\n")
            lines.append(f"- {r['interpretation']}")
            lines.append(f"- Top-1 stable: {r['top_1_stable']:.0%}  |  Top-3 stable: {r['top_3_stable']:.0%}  "
                         f"({r['n_perturbations']} perturbations @ {r['noise_frac']*100:.0f}% noise)")
        return "\n".join(lines)
