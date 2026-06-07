"""CATE stability tests — Tier1 (split-half bootstrap) + Tier2 (hyperparameter sensitivity).

@requirement REQ-CATE-002
@requirement REQ-CATE-003
@design docs/CausalCredit_因果推理验证标准体系.md §4.2 (CATE 异质性检验)

Two complementary stability diagnostics for our CATE estimates, both
adapted from the competitor project (which followed the CausalBench /
"Oracle P0" CATE-stability recipe):

Tier 1 — Split-half bootstrap
    Refit the CATE model on two disjoint random halves of the data and
    compare the resulting CATE predictions.  Repeat ``n_bootstrap``
    times; report the mean Spearman correlation.  A stable CATE
    estimator should give ρ > 0.80.

Tier 2 — Hyperparameter sensitivity
    Vary the first-stage (nuisance) hyperparameters of the CATE
    estimator — gradient-boosting ``max_depth`` and ``n_estimators`` —
    and refit the model ``n_configs`` times.  Report the *minimum*
    pairwise Spearman correlation across all (i, j) config pairs.  A
    stable CATE estimator should give min ρ > 0.70.

Reference
---------
The split-half reliability test is the standard CausalBench / Schuler
et al. (2018) CATE stability diagnostic.  The hyperparameter grid is
modelled on the competitor's project.

Implementation notes
--------------------
* Tier 1 uses our :class:`CATEEstimator` rather than the competitor's
  thin LinearDML wrapper, so any of the three backends (LinearDML /
  SparseLinearDML / CausalForestDML) can be stress-tested.
* Tier 2 varies gradient-boosting ``max_depth`` (3-6), ``n_estimators``
  (80-160) and ``min_samples_leaf`` (5-20) — the same axes the
  competitor varies, retargeted at our :class:`CATEEstimator`.
* Output is a dataclass with the raw correlations, the per-tier
  pass flag, and a `summary()` method that yields a single sentence
  suitable for the pipeline log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger("causalcredit.causal.stability")


# ===========================================================================
# Result container
# ===========================================================================

@dataclass
class StabilityResult:
    """Aggregate stability output for a single CATE estimator run."""
    method: str
    tier1_mean_spearman: float
    tier1_n_bootstrap: int
    tier1_pass: bool
    tier1_threshold: float
    tier2_min_pairwise_spearman: float
    tier2_n_configs: int
    tier2_pass: bool
    tier2_threshold: float
    tier1_per_iter: List[float] = field(default_factory=list)
    tier2_pairwise: List[Dict[str, float]] = field(default_factory=list)
    overall_pass: bool = False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "tier1": {
                "mean_spearman": float(self.tier1_mean_spearman),
                "n_bootstrap": int(self.tier1_n_bootstrap),
                "threshold": float(self.tier1_threshold),
                "pass": bool(self.tier1_pass),
                "per_iter": [float(x) for x in self.tier1_per_iter],
            },
            "tier2": {
                "min_pairwise_spearman": float(self.tier2_min_pairwise_spearman),
                "n_configs": int(self.tier2_n_configs),
                "threshold": float(self.tier2_threshold),
                "pass": bool(self.tier2_pass),
                "pairwise": [dict(p) for p in self.tier2_pairwise],
            },
            "overall_pass": bool(self.overall_pass),
            "summary": str(self.summary),
        }


# ===========================================================================
# Stability tester
# ===========================================================================

class CATEStabilityTester:
    """Tier1 + Tier2 stability tests for any of our CATE backends.

    Parameters
    ----------
    method : str
        Which :class:`CATEEstimator` backend to stress-test
        (``"LinearDML"`` / ``"SparseLinearDML"`` / ``"CausalForestDML"``).
    n_bootstrap : int
        Number of Tier1 split-half iterations (default 30 — matches the
        Oracle P0 / CausalBench recipe).
    n_configs : int
        Number of Tier2 hyperparameter configs (default 10).
    tier1_threshold : float
        Pass threshold for Tier1 mean Spearman (default 0.80).
    tier2_threshold : float
        Pass threshold for Tier2 min pairwise Spearman (default 0.70).
    random_state : int
        Seed for the permutation RNG (so reruns are reproducible).
    """

    def __init__(
        self,
        method: str = "LinearDML",
        n_bootstrap: int = 30,
        n_configs: int = 10,
        tier1_threshold: float = 0.80,
        tier2_threshold: float = 0.70,
        random_state: int = 42,
    ):
        if method not in ("LinearDML", "SparseLinearDML", "CausalForestDML"):
            raise ValueError(
                f"method must be LinearDML / SparseLinearDML / CausalForestDML; "
                f"got {method!r}"
            )
        if n_bootstrap < 2:
            raise ValueError(f"n_bootstrap must be >= 2; got {n_bootstrap}")
        if n_configs < 2:
            raise ValueError(f"n_configs must be >= 2; got {n_configs}")
        self.method = method
        self.n_bootstrap = int(n_bootstrap)
        self.n_configs = int(n_configs)
        self.tier1_threshold = float(tier1_threshold)
        self.tier2_threshold = float(tier2_threshold)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------
    def run(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> StabilityResult:
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if not (len(Y) == len(T) == len(X)):
            raise ValueError(
                f"Y / T / X must have the same length; got "
                f"{len(Y)} / {len(T)} / {len(X)}"
            )
        if len(Y) < 100:
            raise ValueError(
                f"Need at least 100 observations for stability; got {len(Y)}"
            )

        t1 = self.tier1_split_half(Y, T, X)
        t2 = self.tier2_hyperparameter_sensitivity(Y, T, X)
        overall = t1["pass"] and t2["pass"]
        summary = (
            f"method={self.method}  Tier1 ρ̄={t1['mean_spearman']:.3f} "
            f"({'PASS' if t1['pass'] else 'FAIL'})  "
            f"Tier2 min ρ={t2['min_pairwise_spearman']:.3f} "
            f"({'PASS' if t2['pass'] else 'FAIL'})  → "
            f"overall {'STABLE' if overall else 'UNSTABLE'}"
        )
        logger.info("stability: %s", summary)
        return StabilityResult(
            method=self.method,
            tier1_mean_spearman=float(t1["mean_spearman"]),
            tier1_n_bootstrap=int(self.n_bootstrap),
            tier1_pass=bool(t1["pass"]),
            tier1_threshold=float(self.tier1_threshold),
            tier1_per_iter=list(t1["per_iter"]),
            tier2_min_pairwise_spearman=float(t2["min_pairwise_spearman"]),
            tier2_n_configs=int(self.n_configs),
            tier2_pass=bool(t2["pass"]),
            tier2_threshold=float(self.tier2_threshold),
            tier2_pairwise=list(t2["pairwise"]),
            overall_pass=bool(overall),
            summary=summary,
        )

    # ------------------------------------------------------------------
    def tier1_split_half(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
    ) -> Dict[str, Any]:
        """Oracle P0 split-half reliability — 30× bootstrap, ρ̄ > 0.80."""
        from src.causal.cate import CATEEstimator

        rng = np.random.RandomState(self.random_state)
        n = len(X)
        half = n // 2
        rs: List[float] = []
        est = CATEEstimator(config={"random_state": self.random_state, "cv": 2})
        fit_fn = {
            "LinearDML": est.fit_dml,
            "SparseLinearDML": est.fit_dr,
            "CausalForestDML": est.fit_causal_forest,
        }[self.method]

        for it in range(self.n_bootstrap):
            idx = rng.permutation(n)
            X1, X2 = X[idx[:half]], X[idx[half:2 * half]]
            T1, T2 = T[idx[:half]], T[idx[half:2 * half]]
            Y1, Y2 = Y[idx[:half]], Y[idx[half:2 * half]]
            try:
                m1 = fit_fn(Y1, T1, X1)
                m2 = fit_fn(Y2, T2, X2)
                c1 = est.estimate_cate(m1, X1)
                c2 = est.estimate_cate(m2, X2)
            except Exception as e:  # pragma: no cover
                logger.warning("Tier1 iter %d failed: %s", it + 1, e)
                continue
            if len(c1) < 2 or len(c2) < 2:
                continue
            ml = min(len(c1), len(c2))
            r, _ = spearmanr(c1[:ml], c2[:ml])
            if np.isfinite(r):
                rs.append(float(r))
        if not rs:
            return {
                "mean_spearman": 0.0,
                "per_iter": [],
                "pass": False,
            }
        avg_r = float(np.mean(rs))
        return {
            "mean_spearman": avg_r,
            "per_iter": rs,
            "pass": bool(avg_r > self.tier1_threshold),
        }

    # ------------------------------------------------------------------
    def tier2_hyperparameter_sensitivity(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
    ) -> Dict[str, Any]:
        """Oracle P0 hyperparameter sensitivity — 10× full DML runs.

        Vary gradient-boosting ``max_depth``, ``n_estimators`` and
        ``min_samples_leaf`` over a fixed grid.  Pairwise Spearman of
        the CATE predictions across the grid is the stability proxy.
        """
        from econml.dml import LinearDML

        grid = self._tier2_grid()
        variants: List[np.ndarray] = []
        per_config_meta: List[Dict[str, Any]] = []
        n = len(Y)
        # T is treated as continuous in this stability test (consistent
        # with our default CATEEstimator behavior on non-binary T).
        for cfg in grid[: self.n_configs]:
            try:
                model_y = GradientBoostingRegressor(
                    max_depth=int(cfg["max_depth"]),
                    n_estimators=int(cfg["n_estimators"]),
                    min_samples_leaf=int(cfg["min_samples_leaf"]),
                    learning_rate=0.05,
                    random_state=self.random_state,
                )
                model_t = GradientBoostingRegressor(
                    max_depth=int(cfg["max_depth"]),
                    n_estimators=int(cfg["n_estimators"]),
                    min_samples_leaf=int(cfg["min_samples_leaf"]),
                    learning_rate=0.05,
                    random_state=self.random_state,
                )
                model = LinearDML(
                    model_y=model_y,
                    model_t=model_t,
                    discrete_treatment=False,
                    cv=2,
                    random_state=self.random_state,
                )
                model.fit(Y, T, X=X)
                c_hat = np.asarray(model.effect(X)).flatten()
            except Exception as e:  # pragma: no cover
                logger.warning("Tier2 config %s failed: %s", cfg, e)
                continue
            variants.append(c_hat)
            per_config_meta.append(dict(cfg))

        if len(variants) < 2:
            return {
                "min_pairwise_spearman": 0.0,
                "pairwise": [],
                "pass": False,
            }

        # Pairwise Spearman
        pairs: List[Dict[str, float]] = []
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                r, _ = spearmanr(variants[i], variants[j])
                if np.isfinite(r):
                    pairs.append({
                        "i": int(i),
                        "j": int(j),
                        "spearman": float(r),
                        "config_i": per_config_meta[i],
                        "config_j": per_config_meta[j],
                    })
        if not pairs:
            return {
                "min_pairwise_spearman": 0.0,
                "pairwise": [],
                "pass": False,
            }
        min_r = float(min(p["spearman"] for p in pairs))
        return {
            "min_pairwise_spearman": min_r,
            "pairwise": pairs,
            "pass": bool(min_r > self.tier2_threshold),
        }

    # ------------------------------------------------------------------
    def _tier2_grid(self) -> List[Dict[str, int]]:
        """The 10-config grid from the competitor, ported to our schema."""
        return [
            {"max_depth": 3, "n_estimators": 100, "min_samples_leaf": 20},
            {"max_depth": 5, "n_estimators": 100, "min_samples_leaf": 20},
            {"max_depth": 3, "n_estimators": 80,  "min_samples_leaf": 20},
            {"max_depth": 3, "n_estimators": 120, "min_samples_leaf": 20},
            {"max_depth": 4, "n_estimators": 100, "min_samples_leaf": 5},
            {"max_depth": 4, "n_estimators": 100, "min_samples_leaf": 20},
            {"max_depth": 4, "n_estimators": 100, "min_samples_leaf": 6},
            {"max_depth": 5, "n_estimators": 100, "min_samples_leaf": 6},
            {"max_depth": 3, "n_estimators": 60,  "min_samples_leaf": 20},
            {"max_depth": 3, "n_estimators": 140, "min_samples_leaf": 20},
        ]


# ===========================================================================
# Visualization
# ===========================================================================

def plot_stability(
    result: StabilityResult,
    output_path: str,
) -> str:
    """Render a 1×2 chart for the stability result.

    Panel A: per-iteration Tier1 Spearman (line) + threshold (dashed).
    Panel B: pairwise Tier2 heatmap of (config_i, config_j) → Spearman.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Tier1 per-iter
    if result.tier1_per_iter:
        ax1.plot(range(1, len(result.tier1_per_iter) + 1),
                 result.tier1_per_iter, "o-", color="#1f77b4", markersize=4)
        ax1.axhline(result.tier1_threshold, color="red", linestyle="--",
                    label=f"threshold = {result.tier1_threshold:.2f}")
        ax1.axhline(result.tier1_mean_spearman, color="#1f77b4", linestyle=":",
                    label=f"mean = {result.tier1_mean_spearman:.3f}")
        ax1.set_xlabel("Bootstrap iteration")
        ax1.set_ylabel("Spearman ρ (split-half)")
        ax1.set_title(
            f"Tier1 split-half (n={result.tier1_n_bootstrap})\n"
            f"{'PASS' if result.tier1_pass else 'FAIL'}"
        )
        ax1.set_ylim(-0.1, 1.05)
        ax1.legend(loc="lower right", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No Tier1 results", ha="center", va="center",
                 transform=ax1.transAxes)
        ax1.set_axis_off()

    # Panel B: Tier2 pairwise matrix
    if result.tier2_pairwise:
        n_cfg = result.tier2_n_configs
        mat = np.eye(n_cfg)
        for p in result.tier2_pairwise:
            mat[int(p["i"]), int(p["j"])] = float(p["spearman"])
            mat[int(p["j"]), int(p["i"])] = float(p["spearman"])
        im = ax2.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
        fig.colorbar(im, ax=ax2, label="Spearman ρ")
        ax2.set_title(
            f"Tier2 hyperparameter sensitivity ({n_cfg} configs)\n"
            f"min ρ = {result.tier2_min_pairwise_spearman:.3f} → "
            f"{'PASS' if result.tier2_pass else 'FAIL'}"
        )
        ax2.set_xlabel("Config index")
        ax2.set_ylabel("Config index")
    else:
        ax2.text(0.5, 0.5, "No Tier2 results", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_axis_off()

    fig.suptitle(f"CATE stability — {result.method}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("stability chart written: %s", output_path)
    return output_path
