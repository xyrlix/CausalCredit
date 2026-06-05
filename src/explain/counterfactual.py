"""Counterfactual reasoning module.

Answers: "If we change X, how would Y change?"

Built on top of `dice_ml` (DiCE) with the genetic / NSGA-II backend.
Three classes of features per the docs:

  * **IMMUTABLE** — physically cannot be changed by the applicant
    (age, gender, education level, ID publication date).
  * **SEMI_MUTABLE** — can change but with bounded range (income,
    employment days, credit / annuity). Constraint enforced via
    `permitted_range` so DiCE doesn't push them to extreme values.
  * **MUTABLE** — fully free (any other feature).

A causal-propagation step handles the Home Credit DAG's deterministic
edges (e.g. `AMT_CREDIT -> AMT_GOODS_PRICE`, `AMT_CREDIT -> AMT_ANNUITY`):
when the CF changes `AMT_CREDIT`, we propagate a proportional change
to `AMT_ANNUITY` / `AMT_GOODS_PRICE` before re-evaluating the model so
the resulting probability is causally consistent.

causal_plausibility = 1 - clip(|delta_proba| / |cate|, 0, 1)
A large change in P(default) for a tiny CATE is implausible (and vice
versa).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# Feature-class rules (Home Credit DAG)
# ===========================================================================

# Features that cannot change at all under any intervention.
IMMUTABLE_FEATURES: List[str] = [
    "DAYS_BIRTH", "CODE_GENDER", "DAYS_ID_PUBLISH", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "CNT_CHILDREN",  # past, locked in
]

# Features that can change, but only within a bounded fraction of the
# baseline value (e.g. income could grow but not 10x overnight).
SEMI_MUTABLE_FEATURES: List[str] = [
    "AMT_INCOME_TOTAL",
    "DAYS_EMPLOYED",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_2",
]

# Per-feature max change as a fraction of baseline. 0.5 = ±50%.
SEMI_MUTABLE_MAX_FRAC: Dict[str, float] = {
    "AMT_INCOME_TOTAL": 0.5,
    "DAYS_EMPLOYED": 0.3,
    "AMT_CREDIT": 0.5,
    "AMT_ANNUITY": 0.5,
    "AMT_GOODS_PRICE": 0.5,
    "EXT_SOURCE_2": 0.2,
}


# ===========================================================================
# CounterfactualReasoner
# ===========================================================================

class CounterfactualReasoner:
    """Counterfactual reasoner for credit scoring interventions.

    Wraps a fitted binary classifier (sklearn-style `.predict_proba`) and
    a DiCE explainer built lazily on the first call. The explainer is
    cached so subsequent CF queries are fast.
    """

    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        feature_names: List[str],
        outcome_name: str = "TARGET",
        immutables: Optional[List[str]] = None,
        semi_mutables: Optional[List[str]] = None,
        method: str = "genetic",
        continuous_features: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.model = model
        self.training_data = training_data
        self.feature_names = list(feature_names)
        self.outcome_name = outcome_name
        self.immutables = list(immutables or IMMUTABLE_FEATURES)
        self.semi_mutables = list(semi_mutables or SEMI_MUTABLE_FEATURES)
        self.method = method
        self.random_state = random_state
        # Auto-detect continuous features: numeric columns in the training data
        # (intersected with feature_names) are continuous; everything else is
        # categorical for DiCE's purposes.
        numeric_cols = set(training_data.select_dtypes(include=[np.number]).columns)
        if continuous_features is None:
            self.continuous_features = [c for c in feature_names if c in numeric_cols]
        else:
            self.continuous_features = list(continuous_features)
        self._exp: Optional[Any] = None

    # ------------------------------------------------------------------ helpers
    def _build_explainer(self) -> Any:
        import dice_ml
        from dice_ml import Data, Model

        # Drop rows with NaN in feature columns before building DiCE data
        # interface; DiCE does not handle NaNs in the data dict.
        cols = self.feature_names + [self.outcome_name]
        df_clean = self.training_data[cols].dropna()
        d = Data(
            dataframe=df_clean,
            continuous_features=self.continuous_features,
            outcome_name=self.outcome_name,
        )
        m = Model(model=self.model, backend="sklearn", model_type="classifier")
        return dice_ml.Dice(d, m, method=self.method)

    def _get_explainer(self) -> Any:
        if self._exp is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._exp = self._build_explainer()
        return self._exp

    def _to_query_frame(self, features: Dict[str, float]) -> pd.DataFrame:
        """Align user-provided feature dict to a 1-row DataFrame in feature order."""
        return pd.DataFrame([features])[self.feature_names]

    def _predict_proba_row(self, row: pd.DataFrame) -> float:
        return float(self.model.predict_proba(row)[:, 1][0])

    def _baseline_proba(self, features: Dict[str, float]) -> float:
        return self._predict_proba_row(self._to_query_frame(features))

    def _mutables(self) -> List[str]:
        """All feature names minus immutables."""
        return [c for c in self.feature_names if c not in self.immutables]

    def _permitted_range(self, baseline: Dict[str, float]) -> Dict[str, List[float]]:
        """Compute per-feature permitted_range from SEMI_MUTABLE rules."""
        rng: Dict[str, List[float]] = {}
        for c in self.semi_mutables:
            if c in baseline and c in self.feature_names:
                v = float(baseline[c])
                frac = SEMI_MUTABLE_MAX_FRAC.get(c, 0.5)
                rng[c] = [v * (1 - frac), v * (1 + frac)]
        return rng

    def _apply_causal_propagation(
        self, baseline: Dict[str, float], modified: Dict[str, float]
    ) -> Dict[str, float]:
        """Propagate changes through the hand-coded DAG's deterministic edges.

        Home Credit rules (from the DAG):
          - AMT_CREDIT -> AMT_ANNUITY: annuity scales ~linearly with credit
            (annuity ~ credit / term). We propagate via the same ratio.
          - AMT_CREDIT -> AMT_GOODS_PRICE: goods price scales ~linearly
            with credit for the same product. Same ratio.
        """
        out = dict(modified)
        if "AMT_CREDIT" in baseline and "AMT_CREDIT" in modified:
            b_credit = baseline["AMT_CREDIT"]
            m_credit = modified["AMT_CREDIT"]
            if b_credit > 0 and abs(m_credit - b_credit) > 0:
                ratio = m_credit / b_credit
                for downstream in ("AMT_ANNUITY", "AMT_GOODS_PRICE"):
                    if downstream in baseline and downstream in out:
                        out[downstream] = float(baseline[downstream] * ratio)
        return out

    def _causal_plausibility(self, p_base: float, p_cf: float, cate_abs: float) -> float:
        """Score in [0, 1] — 1 = highly plausible, 0 = wildly off."""
        delta = abs(p_cf - p_base)
        if cate_abs < 1e-6:
            # No measurable CATE — any delta is suspicious; penalize proportionally.
            return max(0.0, 1.0 - 10.0 * delta)
        return float(np.clip(1.0 - delta / (10.0 * cate_abs + 1e-6), 0.0, 1.0))

    # ------------------------------------------------------------------ public API
    def predict_counterfactual(
        self,
        features: Dict[str, float],
        interventions: Dict[str, float],
        enforce_propagation: bool = True,
    ) -> Dict:
        """Predict counterfactual default probability under explicit interventions.

        The function applies the user-supplied `interventions` to `features`,
        optionally runs causal propagation to downstream variables, then
        re-evaluates the model. No DiCE search is involved — this is a
        fast path for known interventions.

        Args:
            features: baseline feature dict.
            interventions: subset of feature -> new value.
            enforce_propagation: if True, propagate AMT_CREDIT changes
                to AMT_ANNUITY / AMT_GOODS_PRICE per the DAG.
        """
        # Sanity: drop interventions on immutables and warn
        for k in list(interventions.keys()):
            if k in self.immutables:
                warnings.warn(f"Intervention on immutable feature {k!r} dropped.")
                interventions.pop(k, None)

        modified = dict(features)
        modified.update(interventions)
        if enforce_propagation:
            modified = self._apply_causal_propagation(features, modified)

        p_base = self._baseline_proba(features)
        p_new = self._predict_proba_row(self._to_query_frame(modified))
        return {
            "baseline_proba": p_base,
            "counterfactual_proba": p_new,
            "delta_proba": p_new - p_base,
            "interventions": interventions,
            "modified_features": {k: v for k, v in modified.items() if k in self.feature_names},
        }

    def generate_counterfactuals(
        self,
        features: Dict[str, float],
        total_cfs: int = 3,
        desired_class: int = 0,
        cate_for_plausibility: Optional[float] = None,
    ) -> Dict:
        """DiCE-driven CF search for a single query instance.

        Args:
            features: baseline feature dict.
            total_cfs: number of CFs to generate.
            desired_class: 0 = "no default", 1 = "default".
            cate_for_plausibility: optional |CATE| for the relevant treatment,
                used in `causal_plausibility` scoring. If None, falls back
                to |delta_proba| itself (DiCE-only plausibility).

        Returns:
            Dict with keys: baseline_proba, cfs (list of dict), n_cfs,
            mean_causal_plausibility, raw_cf_dataframe.
        """
        exp = self._get_explainer()
        q = self._to_query_frame(features)
        perm_range = self._permitted_range(features)
        features_to_vary = self._mutables()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cf = exp.generate_counterfactuals(
                    q,
                    total_CFs=total_cfs,
                    desired_class=int(desired_class),
                    features_to_vary=features_to_vary,
                    permitted_range=perm_range if perm_range else None,
                )
        except Exception as e:
            return {
                "baseline_proba": self._baseline_proba(features),
                "cfs": [],
                "n_cfs": 0,
                "mean_causal_plausibility": 0.0,
                "raw_cf_dataframe": None,
                "error": f"{type(e).__name__}: {e}",
            }

        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            return {
                "baseline_proba": self._baseline_proba(features),
                "cfs": [],
                "n_cfs": 0,
                "mean_causal_plausibility": 0.0,
                "raw_cf_dataframe": None,
            }

        p_base = self._baseline_proba(features)
        cfs: List[Dict] = []
        for i in range(len(cf_df)):
            cf_row = cf_df.iloc[i]
            mod = {c: float(cf_row[c]) for c in self.feature_names if c in cf_row.index}
            mod = self._apply_causal_propagation(features, mod)
            p_cf = self._predict_proba_row(self._to_query_frame(mod))
            cat = cate_for_plausibility if cate_for_plausibility is not None else abs(p_cf - p_base)
            plaus = self._causal_plausibility(p_base, p_cf, cat)
            deltas = {c: float(cf_row[c]) - float(features[c]) for c in self.feature_names if c in features and c in cf_row.index}
            cfs.append({
                "cf_index": i,
                "counterfactual_proba": p_cf,
                "delta_proba": p_cf - p_base,
                "causal_plausibility": plaus,
                "deltas": deltas,
            })
        plausibilities = [c["causal_plausibility"] for c in cfs]
        return {
            "baseline_proba": p_base,
            "cfs": cfs,
            "n_cfs": len(cfs),
            "mean_causal_plausibility": float(np.mean(plausibilities)) if plausibilities else 0.0,
            "raw_cf_dataframe": cf_df,
        }

    def predict_multiple_scenarios(
        self,
        features: Dict[str, float],
        scenarios: List[Dict[str, float]],
    ) -> List[Dict]:
        """Run `predict_counterfactual` for each scenario dict."""
        return [self.predict_counterfactual(features, s) for s in scenarios]

    def generate_standard_scenarios(self, features: Dict[str, float]) -> List[Dict[str, float]]:
        """Three canonical "what-if" scenarios for credit-risk applicants.

        Returns a list of intervention dicts ready to pass to
        `predict_counterfactual`:
          1. AMT_CREDIT  -30%
          2. AMT_ANNUITY -30%
          3. EXT_SOURCE_2 + 0.1 (better external risk score)
        """
        return [
            {"AMT_CREDIT": float(features.get("AMT_CREDIT", 0)) * 0.7},
            {"AMT_ANNUITY": float(features.get("AMT_ANNUITY", 0)) * 0.7},
            {"EXT_SOURCE_2": min(1.0, float(features.get("EXT_SOURCE_2", 0.5)) + 0.1)},
        ]

    def find_optimal_intervention(
        self,
        features: Dict[str, float],
        target_probability: float = 0.10,
        budget_constraint: Optional[Dict] = None,
        max_iter: int = 50,
    ) -> Dict:
        """Greedy search for an intervention that drives P(default) to target.

        Args:
            features: baseline feature dict.
            target_probability: desired P(default) (lower is safer for an
                applicant; higher would be in the bank's interest for some
                products — we default to a "safer" target).
            budget_constraint: optional per-feature cap on relative change,
                e.g. `{"AMT_CREDIT": 0.3}`. Falls back to
                `SEMI_MUTABLE_MAX_FRAC` rules.

        Returns:
            Dict with `best_intervention`, `achieved_proba`, `gap`,
            and `iterations_used`.
        """
        p_base = self._baseline_proba(features)
        direction = -1 if target_probability < p_base else 1

        budget = dict(SEMI_MUTABLE_MAX_FRAC)
        if budget_constraint:
            budget.update(budget_constraint)

        # Try each semi-mutable feature at its cap; pick the one that
        # moves P(default) most in the right direction.
        best: Dict = {
            "best_intervention": {},
            "achieved_proba": p_base,
            "gap": abs(p_base - target_probability),
            "iterations_used": 0,
        }
        for k in self.semi_mutables:
            if k not in features:
                continue
            v0 = float(features[k])
            v_new = v0 * (1 + direction * budget.get(k, 0.5))
            if v_new < 0:
                continue
            r = self.predict_counterfactual(features, {k: v_new})
            gap = abs(r["counterfactual_proba"] - target_probability)
            if gap < best["gap"]:
                best = {
                    "best_intervention": {k: v_new},
                    "achieved_proba": r["counterfactual_proba"],
                    "gap": gap,
                    "iterations_used": 1,
                }
        return best

    # ------------------------------------------------------------------ visualization
    def visualize_counterfactuals(
        self,
        cf_result: Dict,
        output_path: str = "output/demo_m1/counterfactual_examples.png",
        top_n: int = 5,
    ) -> str:
        """Render a 1x2 figure: top-N CF deltas + plausibility bar chart."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cfs = cf_result.get("cfs", [])[:top_n]
        if not cfs:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No counterfactuals", ha="center", va="center", transform=ax.transAxes)
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            return output_path

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: top-3 features by |delta| across the top-N CFs
        ax = axes[0]
        feat_delta_sum: Dict[str, float] = {}
        for cf in cfs:
            for f, d in cf["deltas"].items():
                feat_delta_sum[f] = feat_delta_sum.get(f, 0.0) + abs(d)
        top_feats = sorted(feat_delta_sum.items(), key=lambda x: -x[1])[:6]
        feat_names = [t[0] for t in top_feats]
        widths = np.linspace(0.4, 0.8, len(cfs))
        for i, cf in enumerate(cfs):
            vals = [abs(cf["deltas"].get(f, 0.0)) for f in feat_names]
            ax.barh(np.arange(len(feat_names)) + i * 0.15, vals, height=0.15,
                    left=-i * 0.0, label=f"CF{i}", alpha=0.8)
        ax.set_yticks(np.arange(len(feat_names)) + 0.15 * (len(cfs) - 1) / 2)
        ax.set_yticklabels(feat_names, fontsize=8)
        ax.set_xlabel("|delta|")
        ax.set_title("Top-6 feature deltas per CF")
        ax.legend(fontsize=7)

        # Panel B: causal_plausibility per CF
        ax = axes[1]
        plaus = [cf["causal_plausibility"] for cf in cfs]
        ax.bar(range(len(cfs)), plaus, color="#1f77b4", alpha=0.85)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, label="threshold 0.5")
        ax.set_xticks(range(len(cfs)))
        ax.set_xticklabels([f"CF{i}" for i in range(len(cfs))])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("causal_plausibility")
        ax.set_title("Per-CF plausibility")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"Counterfactuals (baseline P={cf_result['baseline_proba']:.3f}, "
            f"mean plausibility={cf_result.get('mean_causal_plausibility', 0):.2f})",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path
