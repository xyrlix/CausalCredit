"""SHAP-based model explainability analysis.

Combines TreeSHAP with a causal-proxy score (per-X linear effect of
treatment on the model's predicted probability) to label each feature
into one of four quadrants per docs section 4.4:

    |                     high |SHAP|
    |  TRUSTED  |  UNTRUSTED  |
    | ----------+-------------|  high
    |           |             |  |causal_proxy|
    | NEGLIGIBLE|  MASKED     |
    | ----------+-------------|  low
    low |SHAP|    high |SHAP|

* **TRUSTED**       — model & causal proxy both point the same way
* **UNTRUSTED**     — high model signal, but no causal support
* **NEGLIGIBLE**    — both low, safe to ignore
* **MASKED**        — model says nothing but the variable has a real
                       causal effect (often a hidden driver)

causal_proxy is computed by perturbing each X column ±1σ and
re-evaluating the model — a cheap local sensitivity that approximates
the partial derivative of P(default) w.r.t. the feature.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# SHAPExplainer
# ===========================================================================

class SHAPExplainer:
    """SHAP explainer using TreeExplainer (works for LightGBM / sklearn GBT / XGBoost)."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = list(feature_names)
        self._explainer = None

    def _get_explainer(self):
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a DataFrame. Returns (n, d) array."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        explainer = self._get_explainer()
        sv = explainer.shap_values(X)
        # For binary classifiers some shap versions return a list of two
        # (n, d) arrays; we want the positive-class SHAP.
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        return np.asarray(sv)

    def global_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Global feature importance ranked by mean |SHAP|."""
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.asarray(shap_values)
        mean_abs = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        })
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def dependence_plot(
        self, shap_values: np.ndarray, X: pd.DataFrame,
        feature: str, output_path: Optional[str] = None,
    ) -> Optional[str]:
        """SHAP dependence plot for a single feature (matplotlib backend)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        idx = self.feature_names.index(feature)
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            idx, shap_values, X.values,
            feature_names=self.feature_names, show=False,
        )
        plt.tight_layout()
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close()
            return output_path
        plt.close()
        return None

    def local_explanation(
        self, shap_values: np.ndarray, X: pd.DataFrame,
        idx: int, output_path: Optional[str] = None,
    ) -> Dict:
        """Return a dict with per-feature SHAP contributions at row `idx`.

        Optionally render a waterfall-style matplotlib bar chart.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        row_sv = shap_values[idx]
        row_x = X.iloc[idx]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "value": row_x.values,
            "shap": row_sv,
        })
        df["abs_shap"] = np.abs(df["shap"])
        df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

        if output_path:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            top = df.head(12)
            plt.figure(figsize=(8, 5))
            colors = ["#d62728" if v < 0 else "#1f77b4" for v in top["shap"]]
            plt.barh(top["feature"][::-1], top["shap"][::-1], color=colors[::-1])
            plt.xlabel("SHAP value")
            plt.title(f"Local explanation (row {idx})")
            plt.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close()
        return {"row_idx": idx, "contributions": df.to_dict(orient="records")}

    # ------------------------------------------------------------------ four-quadrant
    def causal_proxy(self, X: pd.DataFrame, std_frac: float = 0.1) -> np.ndarray:
        """Local sensitivity of P(default) to each X column (mean over rows).

        For each X column j we perturb every row by ±std_frac * sigma_j
        and measure the mean change in the predicted probability. This
        is a cheap surrogate for the partial derivative ∂P/∂X_j that
        does not require a separate model.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        # Align columns to feature_names; fillna with column median
        X_use = X[self.feature_names].copy()
        for c in self.feature_names:
            if X_use[c].isna().any():
                X_use[c] = X_use[c].fillna(X_use[c].median())
        base_proba = self._safe_predict_proba(X_use.values)
        stds = X_use.std(axis=0).values + 1e-9
        d = len(self.feature_names)
        proxy = np.zeros(d)
        for j in range(d):
            X_plus = X_use.values.copy()
            X_plus[:, j] = X_plus[:, j] + std_frac * stds[j]
            X_minus = X_use.values.copy()
            X_minus[:, j] = X_minus[:, j] - std_frac * stds[j]
            p_plus = self._safe_predict_proba(X_plus)
            p_minus = self._safe_predict_proba(X_minus)
            proxy[j] = float(np.mean(np.abs(p_plus - p_minus)) / (2.0 * std_frac * stds[j] + 1e-12))
        return proxy

    def _safe_predict_proba(self, X_arr: np.ndarray) -> np.ndarray:
        try:
            proba = self.model.predict_proba(X_arr)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba.flatten()
        except Exception as e:  # pragma: no cover
            warnings.warn(f"predict_proba failed: {e}")
            return np.zeros(len(X_arr))

    def causal_vs_noncausal_contribution(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        causal_features: List[str],
        threshold_shap: Optional[float] = None,
        threshold_causal: Optional[float] = None,
    ) -> Dict:
        """Label each feature into one of four causal-vs-SHAP quadrants.

        Returns:
            Dict with:
                - "per_feature": pd.DataFrame with columns
                    [feature, mean_abs_shap, abs_causal_proxy, quadrant]
                - "counts": pd.Series of quadrant counts
                - "thresholds": (th_shap, th_causal)
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        # Use a small subsample for the proxy to keep cost reasonable
        n = len(X)
        if n > 2000:
            X_sub = X.sample(n=2000, random_state=0)
        else:
            X_sub = X

        global_imp = self.global_importance(shap_values)
        proxy = self.causal_proxy(X_sub)
        proxy_map = dict(zip(self.feature_names, proxy))

        rows = []
        mean_abs = dict(zip(global_imp["feature"], global_imp["mean_abs_shap"]))
        th_shap = threshold_shap if threshold_shap is not None else float(np.median(list(mean_abs.values())))
        th_causal = threshold_causal if threshold_causal is not None else float(np.median(proxy))

        for f in self.feature_names:
            ms = float(mean_abs.get(f, 0.0))
            cp = float(proxy_map.get(f, 0.0))
            high_shap = ms >= th_shap
            high_causal = cp >= th_causal
            if high_shap and high_causal:
                q = "TRUSTED"
            elif high_shap and not high_causal:
                q = "UNTRUSTED"
            elif (not high_shap) and (not high_causal):
                q = "NEGLIGIBLE"
            else:
                q = "MASKED"
            rows.append({"feature": f, "mean_abs_shap": ms, "abs_causal_proxy": cp, "quadrant": q})

        df = pd.DataFrame(rows)
        counts = df["quadrant"].value_counts().reindex(
            ["TRUSTED", "UNTRUSTED", "NEGLIGIBLE", "MASKED"]
        ).fillna(0).astype(int)
        return {
            "per_feature": df,
            "counts": counts,
            "thresholds": (th_shap, th_causal),
            "causal_features": list(causal_features),
        }

    def visualize_four_quadrant(
        self,
        quadrant_result: Dict,
        output_path: str = "output/demo_m1/four_quadrant.png",
        top_n: int = 30,
    ) -> str:
        """Render a 2-panel figure: scatter |SHAP| × |causal_proxy| + quadrant counts."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = quadrant_result["per_feature"]
        # Keep top_n by combined importance for readability
        df = df.copy()
        df["combined"] = df["mean_abs_shap"] + df["abs_causal_proxy"]
        df = df.sort_values("combined", ascending=False).head(top_n)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel A: scatter, colored by quadrant
        ax = axes[0]
        colors = {
            "TRUSTED": "#2ca02c",       # green
            "UNTRUSTED": "#d62728",     # red
            "NEGLIGIBLE": "#7f7f7f",    # gray
            "MASKED": "#ff7f0e",        # orange
        }
        for q in ["TRUSTED", "UNTRUSTED", "NEGLIGIBLE", "MASKED"]:
            sub = df[df["quadrant"] == q]
            ax.scatter(
                sub["mean_abs_shap"], sub["abs_causal_proxy"],
                s=80, alpha=0.8, label=f"{q} (n={len(sub)})", color=colors[q],
            )
            for _, row in sub.iterrows():
                ax.annotate(row["feature"], (row["mean_abs_shap"], row["abs_causal_proxy"]),
                            fontsize=7, alpha=0.75, xytext=(3, 3), textcoords="offset points")

        th_shap, th_causal = quadrant_result["thresholds"]
        ax.axvline(th_shap, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(th_causal, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("|SHAP| (mean abs)")
        ax.set_ylabel("|causal proxy| (∂P/∂X)")
        ax.set_title("Four-Quadrant: SHAP × Causal Proxy")
        ax.legend(fontsize=8, loc="best")
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.set_yscale("symlog", linthresh=1e-5)

        # Panel B: quadrant counts
        ax = axes[1]
        counts = quadrant_result["counts"]
        bars = ax.bar(
            counts.index, counts.values,
            color=[colors[q] for q in counts.index], alpha=0.85,
        )
        for b, c in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                    str(c), ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("# features")
        ax.set_title(f"Quadrant distribution (top {len(df)} features)")

        fig.suptitle("SHAP vs Causal Proxy — consistency check", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def subgroup_shap_comparison(
        self, shap_values: np.ndarray, X: pd.DataFrame, subgroup_col: str
    ) -> pd.DataFrame:
        """Compare per-subgroup mean |SHAP| for the top features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        if subgroup_col not in X.columns:
            raise ValueError(f"subgroup_col {subgroup_col!r} not in X")
        global_imp = self.global_importance(shap_values)
        top_features = list(global_imp.head(10)["feature"])
        rows = []
        for grp, idx in X.groupby(subgroup_col).groups.items():
            sub_sv = shap_values[idx]
            row = {"subgroup": str(grp), "n": len(idx)}
            for f in top_features:
                j = self.feature_names.index(f)
                row[f] = float(np.mean(np.abs(sub_sv[:, j])))
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_evidence_chain(
        self, shap_values: np.ndarray, X: pd.DataFrame, idx: int, top_k: int = 5,
    ) -> List[Dict]:
        """Top-K SHAP contributors at row idx, packaged as evidence."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        row_sv = shap_values[idx]
        row_x = X.iloc[idx]
        order = np.argsort(-np.abs(row_sv))[:top_k]
        out = []
        for j in order:
            f = self.feature_names[j]
            out.append({
                "feature": f,
                "value": float(row_x[f]),
                "shap": float(row_sv[j]),
                "direction": "increases_default" if row_sv[j] > 0 else "decreases_default",
            })
        return out
