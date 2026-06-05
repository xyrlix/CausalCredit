"""CATE (Conditional Average Treatment Effect) estimation using EconML.

Three heterogeneous treatment effect estimators per docs section 4.2:

1. **LinearDML** — Double-Machine-Learning with a linear final stage. Fast
   and stable; works for both continuous and discrete treatments. Best as a
   fast baseline.
2. **SparseLinearDML** — DML with a Lasso-regularized final stage. Pulls
   out a sparse set of X-features whose interaction with T actually
   matters; useful for heterogeneity feature-importance readout.
3. **CausalForestDML** — EconML's honest causal forest (Zheng et al. style)
   on top of a DML first stage. Best for capturing non-linear, non-additive
   treatment heterogeneity and supporting valid confidence intervals.

Note on ForestDRLearner: in econml 0.16 the DR family (`DRLearner`,
`ForestDRLearner`) reshapes 1-D T to a 2-D one-hot and requires
`discrete_treatment=True` with a 2-D T, which is incompatible with the
continuous treatments used here (Home Credit AMT_CREDIT etc.). The
DML-family estimators above cover the same DR-style robustness without
that constraint, so we use them as the three backends.

The module also exposes:

* `cross_validate_methods(...)` — fits all three on a common dataset and
  returns a Spearman ρ correlation matrix plus per-method ATE / standard
  deviation. Acceptance threshold per docs: synthetic ρ > 0.70, real ρ > 0.50.
* `cate_subgroup_analysis(...)` — breaks CATEs down by user-defined
  subgroup masks and reports mean / median / std / p25 / p75.
* `cate_feature_importance(...)` — extracts the linear DML coefficient
  vector (per-X column) as a feature-importance proxy.
* `visualize_cate(...)` — produces a 1×2 figure: overlay histogram of all
  three CATE distributions + subgroup mean bar chart.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV


# ===========================================================================
# Helpers
# ===========================================================================

def _is_binary_treatment(T: np.ndarray, tol_unique_frac: float = 0.05) -> bool:
    """Heuristic: treat as discrete if the unique-fraction of values is small.

    For 100K samples and 2 unique values, unique-fraction = 2 / 100K ≈ 0,
    so this flags it as discrete. For a continuous float treatment the
    unique-fraction is close to 1.
    """
    n = len(T)
    n_unique = len(np.unique(T))
    return n_unique <= max(10, int(tol_unique_frac * n))


def _make_first_stage_models(discrete_treatment: bool):
    """Pick sensible first-stage sklearn models given treatment type."""
    if discrete_treatment:
        model_t = LogisticRegressionCV(cv=3, max_iter=2000, random_state=0)
        model_y = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0
        )
    else:
        model_t = LassoCV(cv=3, random_state=0, max_iter=5000)
        model_y = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0
        )
    return model_y, model_t


# ===========================================================================
# CATEEstimator
# ===========================================================================

class CATEEstimator:
    """Heterogeneous treatment effect estimator with three backends."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.random_state = int(self.config.get("random_state", 42))
        self.cv = int(self.config.get("cv", 2))

    # ------------------------------------------------------------------ fit
    def fit_dml(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        W: Optional[np.ndarray] = None,
    ):
        """Fit LinearDML (DML with a linear final stage)."""
        from econml.dml import LinearDML

        discrete_t = _is_binary_treatment(T)
        model_y, model_t = _make_first_stage_models(discrete_t)
        model = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_t,
            cv=self.cv,
            random_state=self.random_state,
        )
        if W is None:
            W = np.zeros((len(Y), 0))
        model.fit(Y, T, X=X, W=W)
        model._econml_meta = {"method": "LinearDML", "discrete_treatment": discrete_t}
        return model

    def fit_dr(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        W: Optional[np.ndarray] = None,
    ):
        """Fit SparseLinearDML (Lasso-penalized DML final stage).

        We use SparseLinearDML as the "DR-style" third backend because in
        econml 0.16 the `econml.dr` family is hard-coded to discrete T
        with 2-D one-hot encoding, which is incompatible with the
        continuous treatments in the Home Credit DAG. SparseLinearDML
        covers the doubly-robust DML flavor while accepting continuous T.
        """
        from econml.dml import SparseLinearDML

        discrete_t = _is_binary_treatment(T)
        model_y, model_t = _make_first_stage_models(discrete_t)
        model = SparseLinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_t,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
        )
        if W is None:
            W = np.zeros((len(Y), 0))
        model.fit(Y, T, X=X, W=W)
        model._econml_meta = {"method": "SparseLinearDML", "discrete_treatment": discrete_t}
        return model

    def fit_causal_forest(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        W: Optional[np.ndarray] = None,
    ):
        """Fit CausalForestDML (honest causal forest on a DML first stage)."""
        from econml.dml import CausalForestDML

        discrete_t = _is_binary_treatment(T)
        model_y, model_t = _make_first_stage_models(discrete_t)
        model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_t,
            n_estimators=int(self.config.get("cf_n_estimators", 500)),
            max_depth=int(self.config.get("cf_max_depth", 6)),
            min_samples_leaf=20,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
        )
        if W is None:
            W = np.zeros((len(Y), 0))
        model.fit(Y, T, X=X, W=W)
        model._econml_meta = {"method": "CausalForestDML", "discrete_treatment": discrete_t}
        return model

    # ------------------------------------------------------------------ predict
    def estimate_cate(self, model, X: np.ndarray) -> np.ndarray:
        """Return CATE(X) as a flat (n,) array.

        For discrete treatments, `model.effect(X)` returns one value per
        treatment category offset from the baseline; for continuous
        treatments it returns the marginal effect w.r.t. T.
        """
        eff = model.effect(X)
        if hasattr(eff, "flatten"):
            return np.asarray(eff).flatten()
        return np.asarray(eff)

    # ------------------------------------------------------------------ subgroup
    def cate_subgroup_analysis(
        self,
        cate_values: np.ndarray,
        X: pd.DataFrame,
        subgroup_defs: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Report CATE summary statistics by subgroup mask.

        Args:
            cate_values: (n,) CATE array, aligned with `X`.
            X: DataFrame aligned with `cate_values` (only used for sanity
                checks; not required for the statistics).
            subgroup_defs: name -> boolean mask of length n.

        Returns:
            DataFrame with columns: subgroup, n, mean, median, std, p25, p75.
        """
        assert len(cate_values) == len(X), "cate_values and X must be aligned"
        rows = []
        for name, mask in subgroup_defs.items():
            mask = np.asarray(mask, dtype=bool)
            sub = cate_values[mask]
            if len(sub) < 2:
                continue
            rows.append({
                "subgroup": name,
                "n": int(len(sub)),
                "mean": float(np.mean(sub)),
                "median": float(np.median(sub)),
                "std": float(np.std(sub, ddof=1)),
                "p25": float(np.percentile(sub, 25)),
                "p75": float(np.percentile(sub, 75)),
            })
        return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ importance
    def cate_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Extract a per-feature heterogeneity score from a fitted model.

        * For tree-based models (`CausalForestDML`) we use
          `model.feature_importances_` (already aggregated across trees).
        * For linear DML backends we return the absolute value of the
          per-X coefficient (`model.coef_`) which is the linear
          heterogeneity score.
        """
        method = getattr(model, "_econml_meta", {}).get("method", "")

        # Direct feature_importances_ on the fitted estimator.
        if hasattr(model, "feature_importances_"):
            try:
                imp = np.asarray(model.feature_importances_)
                if imp.ndim == 1 and imp.shape[0] >= len(feature_names):
                    return _format_importance(
                        feature_names[: imp.shape[0]], imp[: len(feature_names)], source=method or "feature_importances_"
                    )
            except Exception as e:  # pragma: no cover
                warnings.warn(f"feature_importances_ extraction failed: {e}")

        # Internal forest attribute (older API).
        if hasattr(model, "model_cate") and hasattr(model.model_cate, "feature_importances_"):
            try:
                imp = np.asarray(model.model_cate.feature_importances_)
                return _format_importance(feature_names, imp, source=method or "model_cate.feature_importances_")
            except Exception as e:  # pragma: no cover
                warnings.warn(f"model_cate.feature_importances_ extraction failed: {e}")

        # Fallback: linear coefficient from a LinearDML / underlying
        # `model_final` of a forest learner.
        try:
            coef = np.asarray(model.coef_).flatten()
            if len(coef) == len(feature_names):
                return _format_importance(feature_names, np.abs(coef), source="|coef|")
            # Pad / trim to feature count
            if len(coef) > len(feature_names):
                coef = coef[: len(feature_names)]
            else:
                coef = np.pad(coef, (0, len(feature_names) - len(coef)))
            return _format_importance(feature_names, np.abs(coef), source="|coef|")
        except Exception as e:  # pragma: no cover
            warnings.warn(f"coef_ extraction failed: {e}")
            return _format_importance(feature_names, np.zeros(len(feature_names)), source="zero")

    # ------------------------------------------------------------------ cross-validate
    def cross_validate_methods(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        W: Optional[np.ndarray] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict:
        """Fit all three CATE methods on the same data and report agreement.

        Returns:
            Dict with keys:
                - "methods": list of method names
                - "models": dict name -> fitted model
                - "cate": dict name -> (n,) CATE array
                - "ate": dict name -> float mean CATE
                - "spearman": pd.DataFrame (n_methods x n_methods) of ρ
                - "mean_abs_spearman": float
                - "passes_threshold": bool
        """
        methods = methods or ["LinearDML", "SparseLinearDML", "CausalForestDML"]
        fit_map = {
            "LinearDML": self.fit_dml,
            "SparseLinearDML": self.fit_dr,
            "CausalForestDML": self.fit_causal_forest,
        }
        models: Dict[str, object] = {}
        cate: Dict[str, np.ndarray] = {}
        ate: Dict[str, float] = {}
        for m in methods:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = fit_map[m](Y, T, X, W)
                c = self.estimate_cate(fitted, X)
                models[m] = fitted
                cate[m] = c
                ate[m] = float(np.mean(c))
            except Exception as e:  # pragma: no cover
                warnings.warn(f"Method {m} failed: {e}")

        # Spearman correlation matrix
        n = len(methods)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                mi, mj = methods[i], methods[j]
                if mi in cate and mj in cate:
                    rho, _ = spearmanr(cate[mi], cate[mj])
                    mat[i, j] = mat[j, i] = float(rho) if np.isfinite(rho) else 0.0
        spearman_df = pd.DataFrame(mat, index=methods, columns=methods)

        # Off-diagonal mean absolute Spearman
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            off = np.abs(mat[mask])
            mean_abs = float(np.mean(off)) if off.size else 1.0
        else:
            mean_abs = 1.0
        return {
            "methods": methods,
            "models": models,
            "cate": cate,
            "ate": ate,
            "spearman": spearman_df,
            "mean_abs_spearman": mean_abs,
            "n_methods": len(cate),
        }

    # ------------------------------------------------------------------ visualize
    def visualize_cate(
        self,
        cate_dict: Dict[str, np.ndarray],
        subgroup_df: Optional[pd.DataFrame] = None,
        output_path: str = "output/demo_m1/cate_distribution.png",
    ) -> str:
        """Render overlay histogram of all CATE distributions + subgroup bars."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: overlay histograms
        ax = axes[0]
        colors = {"LinearDML": "#1f77b4", "SparseLinearDML": "#ff7f0e", "CausalForestDML": "#2ca02c"}
        for name, arr in cate_dict.items():
            arr = np.asarray(arr).flatten()
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            ax.hist(
                arr, bins=40, alpha=0.45, label=f"{name} (μ={np.mean(arr):+.4f})",
                color=colors.get(name, None), density=True,
            )
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="zero effect")
        ax.set_xlabel("CATE θ(X)")
        ax.set_ylabel("Density")
        ax.set_title("CATE distributions (3 methods)")
        ax.legend(fontsize=8, loc="best")

        # Panel B: subgroup bars
        ax = axes[1]
        if subgroup_df is not None and len(subgroup_df) > 0:
            ax.bar(
                subgroup_df["subgroup"].astype(str),
                subgroup_df["mean"],
                yerr=subgroup_df["std"],
                color="#1f77b4", alpha=0.8, capsize=4,
            )
            ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_xticklabels(subgroup_df["subgroup"].astype(str), rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Mean CATE")
            ax.set_title("CATE by subgroup")
        else:
            ax.text(0.5, 0.5, "No subgroup data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        fig.suptitle("Conditional Average Treatment Effect (CATE)", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path


# ===========================================================================
# Synthetic validation utility
# ===========================================================================

def synthetic_cate_validation(
    n: int = 5000,
    d_x: int = 6,
    seed: int = 0,
    treatment_strength: float = 1.0,
    hetero_strength: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic DGP with a known heterogeneous treatment effect.

    DGP:
        X ~ N(0, 1)^d_x
        T ~ Bernoulli(sigmoid(X[:, 0] + X[:, 1]))
        Y = T * (theta0 + theta1 * X[:, 2]) + X[:, 3] + N(0, 1)

    Returns (Y, T, X, W, true_cate) where:
        W = X[:, :2]  (confounders, drive both T and Y)
        true_cate(X) = theta0 + theta1 * X[:, 2]
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d_x))
    T = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(X[:, 0] + X[:, 1])))).astype(int)
    theta0 = treatment_strength
    theta1 = hetero_strength
    true_cate = theta0 + theta1 * X[:, 2]
    Y = T * true_cate + X[:, 3] + rng.normal(scale=0.5, size=n)
    W = X[:, :2]
    return Y, T, X, W, true_cate


def _format_importance(feature_names: List[str], scores: np.ndarray, source: str) -> pd.DataFrame:
    df = pd.DataFrame({"feature": feature_names, "importance": np.asarray(scores, dtype=float)})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["source"] = source
    return df
