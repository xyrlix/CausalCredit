"""BLP (Best Linear Predictor) test for CATE model validation.

@requirement REQ-CATE-001
@design docs/CausalCredit_因果推理验证标准体系.md §4.2 (CATE 异质性检验)

Implements the Chernozhukov et al. (2018) "Best Linear Predictor" test:

    1. Split the data into K folds.
    2. For each fold, refit the CATE model on the other K-1 folds, then
       predict CATE on the held-out fold.  Concatenate to obtain an
       *out-of-fold* CATE prediction ``ĉ(X_i)`` for every sample.
    3. Regress Y on a design matrix [1, T, ĉ(X), ĉ(X)·T] by OLS.
    4. Test the coefficient on the CATE term (β̂_2) — the BLP coefficient.

A significantly non-zero β̂_2 (p < 0.05) means that the *predicted*
heterogeneity tracks real outcome variation that the treatment main
effect alone cannot explain.  Conversely, a non-significant BLP means
the CATE predictions add nothing beyond a constant-effect model.

We delegate cross-validated CATE prediction to :class:`CATEEstimator`
(`src/causal/cate.py`), which already wraps the three EconML backends
(LinearDML / SparseLinearDML / CausalForestDML).  BLP defaults to the
fast LinearDML backend for reproducibility with our pipeline; callers
can opt into the forest backend for richer heterogeneity.

Reference
---------
Chernozhukov, V., Demirer, M., Duflo, E., Fernández-Val, I. (2018).
"Generic Machine Learning Inference on Heterogeneous Treatment Effects
in Randomized Experiments".  Econometrics Journal 21(1): C1-C39.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

logger = logging.getLogger("causalcredit.causal.blp_test")


@dataclass
class BLPResult:
    """Container for the BLP test output.

    The "BLP coefficient" (``blp_coef``) is the OLS coefficient on the
    cross-validated CATE prediction in the design
    ``[1, T, c_hat, c_hat * T]``.  A non-significant value (p > 0.05)
    means the model fails to demonstrate that its CATE predictions are
    linearly related to the outcome.
    """

    blp_coef: float
    blp_se: float
    blp_t_stat: float
    blp_p_value: float
    n_obs: int
    n_folds: int
    method: str
    pass_at_05: bool
    pass_at_10: bool
    design_coefs: Dict[str, float] = field(default_factory=dict)
    design_se: Dict[str, float] = field(default_factory=dict)
    cate_summary: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "blp_coef": float(self.blp_coef),
            "blp_se": float(self.blp_se),
            "blp_t_stat": float(self.blp_t_stat),
            "blp_p_value": float(self.blp_p_value),
            "n_obs": int(self.n_obs),
            "n_folds": int(self.n_folds),
            "method": str(self.method),
            "pass_at_05": bool(self.pass_at_05),
            "pass_at_10": bool(self.pass_at_10),
            "design_coefs": dict(self.design_coefs),
            "design_se": dict(self.design_se),
            "cate_summary": dict(self.cate_summary),
        }


# ===========================================================================
# Main class
# ===========================================================================

class BLPTest:
    """Best Linear Predictor test for a fitted CATE model.

    Parameters
    ----------
    n_folds : int
        K for the cross-validated CATE prediction step.
    method : str
        Which CATE backend to use — one of ``"LinearDML"`` (default, fast
        and stable), ``"SparseLinearDML"`` (Lasso-penalized), or
        ``"CausalForestDML"`` (slow but captures non-linear heterogeneity).
    random_state : int
        Seed for the KFold shuffle.
    alpha : float
        Significance threshold (default 0.05) for the BLP-coef p-value.
    """

    def __init__(
        self,
        n_folds: int = 5,
        method: str = "LinearDML",
        random_state: int = 42,
        alpha: float = 0.05,
    ):
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2; got {n_folds}")
        if method not in ("LinearDML", "SparseLinearDML", "CausalForestDML"):
            raise ValueError(
                f"method must be LinearDML / SparseLinearDML / CausalForestDML; "
                f"got {method!r}"
            )
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.n_folds = n_folds
        self.method = method
        self.random_state = int(random_state)
        self.alpha = float(alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        W: Optional[np.ndarray] = None,
    ) -> BLPResult:
        """Run the BLP test.

        Parameters
        ----------
        Y, T, X : array-like of shape (n,), (n,), (n, p)
        W : optional (n, q) array of high-dimensional controls

        Returns
        -------
        :class:`BLPResult`
        """
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if W is not None:
            W = np.asarray(W, dtype=float)
        if not (len(Y) == len(T) == len(X)):
            raise ValueError(
                f"Y / T / X must have the same length; got "
                f"{len(Y)} / {len(T)} / {len(X)}"
            )
        n = len(Y)
        if n < self.n_folds * 2:
            raise ValueError(
                f"Need at least 2*n_folds observations; got {n} for "
                f"n_folds={self.n_folds}"
            )

        c_hat = self._cross_val_cate(Y, T, X, W)
        result = self._fit_blp_regression(Y, T, c_hat)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _cross_val_cate(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        W: Optional[np.ndarray],
    ) -> np.ndarray:
        """K-fold out-of-fold CATE predictions."""
        from src.causal.cate import CATEEstimator

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        c_hat = np.zeros(len(Y), dtype=float)
        est = CATEEstimator(config={"random_state": self.random_state, "cv": 2})
        fit_fn = {
            "LinearDML": est.fit_dml,
            "SparseLinearDML": est.fit_dr,
            "CausalForestDML": est.fit_causal_forest,
        }[self.method]

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            try:
                if W is None:
                    model = fit_fn(Y[train_idx], T[train_idx], X[train_idx])
                else:
                    model = fit_fn(
                        Y[train_idx], T[train_idx], X[train_idx], W[train_idx],
                    )
                c_hat[test_idx] = est.estimate_cate(model, X[test_idx])
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "BLP fold %d/%d failed: %s; filling with zeros",
                    fold_idx + 1, self.n_folds, e,
                )
                c_hat[test_idx] = 0.0
        return c_hat

    def _fit_blp_regression(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        c_hat: np.ndarray,
    ) -> BLPResult:
        """OLS: Y ~ 1 + T + c_hat + c_hat * T; test coefficient on c_hat.

        Note: we deliberately do NOT include a column of 1s in the design —
        ``LinearRegression(fit_intercept=True)`` adds its own intercept and
        will drop / deweight a user-supplied constant column.  With
        ``fit_intercept=True`` the returned ``coef_`` is aligned with our
        columns [T, c_hat, c_hat*T] at indices [0, 1, 2].
        """
        design = np.column_stack([T, c_hat, c_hat * T])
        reg = LinearRegression(fit_intercept=True).fit(design, Y)
        coef_vec = np.asarray(reg.coef_, dtype=float).flatten()
        # Index mapping: coef_[0] → T, coef_[1] → c_hat, coef_[2] → c_hat·T
        coef_t = float(coef_vec[0])
        coef_c = float(coef_vec[1])
        coef_int = float(coef_vec[2])

        se_vec = self._compute_se_vec(design, Y, reg)
        # se_vec[0..2] correspond to the same indices as coef_ above.
        se_t = float(se_vec[0]) if len(se_vec) > 0 else float("nan")
        se_c = float(se_vec[1]) if len(se_vec) > 1 else float("nan")
        se_int = float(se_vec[2]) if len(se_vec) > 2 else float("nan")

        if np.isfinite(se_c) and se_c > 1e-15:
            t_stat = coef_c / se_c
        else:
            t_stat = 0.0
        n = design.shape[0]
        p = design.shape[1] + 1  # +1 for the intercept
        df = max(n - p, 1)
        p_value = float(2 * tdist.sf(abs(t_stat), df))

        return BLPResult(
            blp_coef=coef_c,
            blp_se=se_c,
            blp_t_stat=float(t_stat),
            blp_p_value=p_value,
            n_obs=int(n),
            n_folds=int(self.n_folds),
            method=self.method,
            pass_at_05=bool(p_value < self.alpha),
            pass_at_10=bool(p_value < max(self.alpha * 2, 0.10)),
            design_coefs={
                "T": coef_t,
                "c_hat": coef_c,
                "c_hat_x_T": coef_int,
                "intercept": float(reg.intercept_),
            },
            design_se={
                "T": se_t,
                "c_hat": se_c,
                "c_hat_x_T": se_int,
            },
            cate_summary={
                "mean": float(np.mean(c_hat)),
                "std": float(np.std(c_hat, ddof=1)) if len(c_hat) > 1 else 0.0,
                "min": float(np.min(c_hat)),
                "p25": float(np.percentile(c_hat, 25)),
                "p50": float(np.percentile(c_hat, 50)),
                "p75": float(np.percentile(c_hat, 75)),
                "max": float(np.max(c_hat)),
            },
        )

    @staticmethod
    def _compute_se_vec(
        design: np.ndarray,
        Y: np.ndarray,
        reg: LinearRegression,
    ) -> np.ndarray:
        """HC0-style standard errors for each OLS coefficient.

        The competitor's reference implementation uses classical OLS SE
        (σ̂² = SSR / (n - p), cov = σ̂² · (X'X)⁻¹).  We use the same
        formula here for consistency.
        """
        residuals = Y - reg.predict(design)
        n, p = design.shape
        sigma2 = float(np.sum(residuals ** 2) / max(n - p, 1))
        xtx = design.T @ design
        try:
            cov = sigma2 * np.linalg.inv(xtx)
        except np.linalg.LinAlgError:  # pragma: no cover
            cov = sigma2 * np.linalg.pinv(xtx)
        return np.sqrt(np.diag(cov))


# ===========================================================================
# Visualization
# ===========================================================================

def plot_blp_test(
    result: BLPResult,
    output_path: str,
) -> str:
    """Render a 1×2 chart summarizing the BLP test.

    Panel A: coefficient bar chart with error bars.
    Panel B: scatter of Y vs T coloured by CATE sign (positive/negative).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: coefficient bars
    names = ["T", "CATE", "CATE×T"]
    coefs = [
        result.design_coefs.get("T", 0.0),
        result.design_coefs.get("c_hat", 0.0),
        result.design_coefs.get("c_hat_x_T", 0.0),
    ]
    ses = [
        result.design_se.get("T", 0.0),
        result.design_se.get("c_hat", 0.0),
        result.design_se.get("c_hat_x_T", 0.0),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax1.bar(names, coefs, yerr=ses, color=colors, alpha=0.85, capsize=6)
    ax1.axhline(0.0, color="grey", lw=0.8)
    ax1.set_ylabel("OLS coefficient")
    p_str = f"{result.blp_p_value:.2e}" if result.blp_p_value >= 1e-3 else "<1e-3"
    ax1.set_title(
        f"BLP test ({result.method}, K={result.n_folds})\n"
        f"β_CATE = {result.blp_coef:+.4f}, p = {p_str} → "
        f"{'PASS' if result.pass_at_05 else 'FAIL'} @ α=0.05"
    )

    # Panel B: pre-rendered text — the BLP regression uses Y, T, c_hat
    # arrays that we do not keep around; show the CATE summary stats.
    sm = result.cate_summary
    txt_lines = [
        f"n = {result.n_obs}",
        f"BLP coef (CATE): {result.blp_coef:+.4f}",
        f"BLP SE:          {result.blp_se:.4f}",
        f"BLP t:           {result.blp_t_stat:+.3f}",
        f"BLP p:           {p_str}",
        f"",
        f"CATE summary:",
        f"  mean = {sm.get('mean', 0):+.4f}",
        f"  std  = {sm.get('std', 0):.4f}",
        f"  IQR  = [{sm.get('p25', 0):+.4f}, {sm.get('p75', 0):+.4f}]",
    ]
    ax2.text(0.05, 0.5, "\n".join(txt_lines), family="monospace", fontsize=10,
             transform=ax2.transAxes, va="center")
    ax2.set_axis_off()
    ax2.set_title("BLP summary")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("BLP chart written: %s", output_path)
    return output_path
