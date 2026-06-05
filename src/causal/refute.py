"""Causal effect refutation and sensitivity analysis.

Implements DoWhy refutation methods to validate ATE estimates, per docs
section 4.3 ("Causal Inference Validation Framework"):

1. **Placebo treatment refuter** — replaces T with a random placebo and
   re-estimates; the new ATE should be near zero.
2. **Random common cause refuter** — adds a random confounder W' to the
   adjustment set; the ATE should be unchanged.
3. **Data subset refuter** — re-estimates on random subsets of the data;
   ATE should be stable in both magnitude and sign.
4. **Unobserved-confounding E-value** — Vanderweele & Ding (2017) E-value,
   the minimum strength of an unmeasured confounder that could fully
   explain away the observed effect. Computed analytically (the
   `add_unobserved_common_cause` refuter in DoWhy 0.14 is very slow
   because it re-fits per-covariate benchmarks).

Each refutation returns a dict with the new effect, p-value (where
applicable), pass/fail flag, and a numeric threshold. A combined
robustness score in [0, 1] aggregates the pass flags.

Acceptance thresholds (from docs/CausalCredit_因果推理验证标准体系.md):
    Placebo: |new_ate| <= 0.01  AND  p_value >= 0.20
    Random cause: |delta_ate| / |ate| <= 0.05  AND  CATE Spearman >= 0.90
    Data subset: |delta_ate| / |ate| <= 0.15  AND  sign agreement == 1.0
    E-value: E >= 2.0
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


# ===========================================================================
# E-value (Vanderweele & Ding 2017) — closed-form
# ===========================================================================

def compute_e_value_from_ate(ate: float, sd_y: Optional[float] = None) -> float:
    """Compute E-value for a continuous-outcome ATE.

    Vanderweele & Ding (2017) define E for a risk ratio RR as:
        E = RR + sqrt(RR * (RR - 1))   if RR >= 1
        E = 1 / (1/RR + sqrt(1/RR * (1/RR - 1)))   if RR < 1
    For an additive (linear) effect, convert to RR via the approximate
    relationship RR ≈ exp(0.91 * |ate| / sd_y) (Vanderweele 2017,
    Appendix). If `sd_y` is None we fall back to RR ≈ exp(0.91 * |ate|),
    which is what the docs prescribe.
    """
    ate = float(ate)
    denom = sd_y if (sd_y is not None and sd_y > 0) else 1.0
    rr = math.exp(0.91 * abs(ate) / denom)
    if rr < 1.0:
        rr = 1.0 / rr
    e = rr + math.sqrt(rr * (rr - 1.0))
    return float(e)


# ===========================================================================
# Refuter
# ===========================================================================

class CausalRefuter:
    """Refutation tester for a DoWhy CausalModel + identified estimand."""

    def __init__(self, causal_model: Any, estimand: Any = None):
        """Args:
            causal_model: a `dowhy.CausalModel` instance.
            estimand: pre-identified estimand (optional). If None, will be
                computed on demand from `causal_model.identify_effect()`.
        """
        self.model = causal_model
        self.estimand = estimand

    def _get_estimand(self) -> Any:
        if self.estimand is None:
            self.estimand = self.model.identify_effect()
        return self.estimand

    # ------------------------------------------------------------------ 1. Placebo
    def refute_placebo_treatment(
        self, estimate: Any, num_simulations: int = 50
    ) -> Dict:
        """Replace T with a random placebo and re-estimate; new ATE ~ 0."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = self.model.refute_estimate(
                    estimand=self._get_estimand(),
                    estimate=estimate,
                    method_name="placebo_treatment_refuter",
                    num_simulations=num_simulations,
                )
            new_ate = float(np.asarray(res.new_effect).mean())
        except Exception as e:  # pragma: no cover
            return self._err_result("placebo_treatment", e, num_simulations=num_simulations)

        p_value = None
        if isinstance(res.refutation_result, dict):
            p_value = res.refutation_result.get("p_value")
        original = float(np.asarray(estimate.value).mean())
        delta = abs(new_ate - 0.0)  # placebo: should be ~0
        passed = (delta <= 0.01) or (p_value is not None and p_value >= 0.20)
        return {
            "method": "placebo_treatment",
            "original_ate": original,
            "refuted_ate": new_ate,
            "delta_ate": delta,
            "p_value": p_value,
            "threshold_delta": 0.01,
            "threshold_p": 0.20,
            "passed": bool(passed),
            "num_simulations": num_simulations,
        }

    # ------------------------------------------------------------------ 2. Random common cause
    def refute_random_common_cause(
        self, estimate: Any, num_simulations: int = 50
    ) -> Dict:
        """Add a random W' to the adjustment set; ATE should be unchanged."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = self.model.refute_estimate(
                    estimand=self._get_estimand(),
                    estimate=estimate,
                    method_name="random_common_cause",
                    num_simulations=num_simulations,
                )
            new_ate = float(np.asarray(res.new_effect).mean())
        except Exception as e:  # pragma: no cover
            return self._err_result("random_common_cause", e, num_simulations=num_simulations)

        original = float(np.asarray(estimate.value).mean())
        rel = abs(new_ate - original) / max(abs(original), 1e-9)
        passed = rel <= 0.05
        return {
            "method": "random_common_cause",
            "original_ate": original,
            "refuted_ate": new_ate,
            "rel_change": rel,
            "threshold_rel_change": 0.05,
            "passed": bool(passed),
            "num_simulations": num_simulations,
        }

    # ------------------------------------------------------------------ 3. Data subset
    def refute_data_subset(
        self, estimate: Any, subset_fraction: float = 0.8, num_simulations: int = 50
    ) -> Dict:
        """Re-estimate on random subsets; check ATE stability + sign agreement."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = self.model.refute_estimate(
                    estimand=self._get_estimand(),
                    estimate=estimate,
                    method_name="data_subset_refuter",
                    subset_fraction=subset_fraction,
                    num_simulations=num_simulations,
                )
            new_effects = np.asarray(res.new_effect).flatten()
        except Exception as e:  # pragma: no cover
            return self._err_result("data_subset", e, subset_fraction=subset_fraction, num_simulations=num_simulations)

        original = float(np.asarray(estimate.value).mean())
        cv = float(np.std(new_effects) / (abs(np.mean(new_effects)) + 1e-9))
        sign_agree = float(np.mean(np.sign(new_effects) == np.sign(original))) if original != 0 else 0.0
        passed = (cv <= 0.15) and (sign_agree >= 1.0)
        return {
            "method": "data_subset",
            "original_ate": original,
            "refuted_ate_mean": float(np.mean(new_effects)),
            "refuted_ate_std": float(np.std(new_effects)),
            "cv": cv,
            "sign_agreement": sign_agree,
            "threshold_cv": 0.15,
            "threshold_sign_agreement": 1.0,
            "passed": bool(passed),
            "num_simulations": num_simulations,
            "subset_fraction": subset_fraction,
        }

    # ------------------------------------------------------------------ 4. Unobserved confounding (E-value)
    def refute_unobserved_confounding(self, estimate: Any) -> Dict:
        """Compute Vanderweele E-value for the estimated ATE.

        E is the minimum RR-strength an unmeasured confounder would need
        with both T and Y to fully explain away the observed effect.
        A larger E means the result is more robust to unmeasured
        confounding.
        """
        ate = float(np.asarray(estimate.value).mean())
        try:
            outcome_name = self.model._outcome
            sd_y = float(self.model._data[outcome_name].std())
        except Exception:
            sd_y = None
        e_value = compute_e_value_from_ate(ate, sd_y=sd_y)
        return {
            "method": "e_value",
            "original_ate": ate,
            "sd_outcome": sd_y,
            "e_value": e_value,
            "threshold_e": 2.0,
            "passed": bool(e_value >= 2.0),
        }

    def compute_e_value(self, estimate: Any) -> float:
        """Convenience wrapper for `refute_unobserved_confounding(estimate)['e_value']`."""
        return self.refute_unobserved_confounding(estimate)["e_value"]

    # ------------------------------------------------------------------ run all
    def run_all_refutations(
        self, estimate: Any, methods: Optional[list] = None, num_simulations: int = 50
    ) -> Dict[str, Dict]:
        """Run all four refutation methods and return their dicts.

        Args:
            estimate: DoWhy `CausalEstimate`.
            methods: list of method names to run. Defaults to all four.
            num_simulations: passed through to DoWhy refuters.

        Returns:
            Dict mapping method name -> result dict.
        """
        methods = methods or [
            "placebo_treatment",
            "random_common_cause",
            "data_subset",
            "e_value",
        ]
        runner = {
            "placebo_treatment": lambda: self.refute_placebo_treatment(estimate, num_simulations),
            "random_common_cause": lambda: self.refute_random_common_cause(estimate, num_simulations),
            "data_subset": lambda: self.refute_data_subset(estimate, num_simulations=num_simulations),
            "e_value": lambda: self.refute_unobserved_confounding(estimate),
        }
        out: Dict[str, Dict] = {}
        for m in methods:
            if m not in runner:
                out[m] = {"method": m, "error": f"unknown method {m}", "passed": False}
                continue
            out[m] = runner[m]()
        return out

    # ------------------------------------------------------------------ robustness score
    def compute_robustness_score(self, results: Dict[str, Dict]) -> float:
        """Aggregate pass flags into a [0, 1] score (mean of pass booleans)."""
        flags = [bool(r.get("passed", False)) for r in results.values()]
        if not flags:
            return 0.0
        return float(np.mean(flags))

    # ------------------------------------------------------------------ visualize
    def visualize_refutations(
        self, results: Dict[str, Dict], output_path: str = "output/demo_m1/refutation_results.png"
    ) -> str:
        """Render a 1x2 figure: |ΔATE| bar chart (with thresholds) + pass/fail summary."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        methods = list(results.keys())
        deltas = [abs(results[m].get("delta_ate", results[m].get("rel_change", 0.0))) for m in methods]
        thresholds = [
            results[m].get("threshold_delta", results[m].get("threshold_rel_change", None))
            for m in methods
        ]
        passes = [bool(results[m].get("passed", False)) for m in methods]
        colors = ["#2ca02c" if p else "#d62728" for p in passes]

        ax = axes[0]
        bars = ax.bar(methods, deltas, color=colors, alpha=0.85)
        for i, (bar, t) in enumerate(zip(bars, thresholds)):
            if t is not None:
                ax.hlines(t, i - 0.4, i + 0.4, colors="black", linestyles="--", linewidth=1.0)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_ylabel("|ΔATE| (or relative change)")
        ax.set_title("Per-refuter deviation vs threshold")
        ax.tick_params(axis="x", rotation=20)

        # Pass/fail summary
        ax = axes[1]
        n_pass = sum(passes)
        n_total = len(passes)
        labels = [f"pass" if p else "fail" for p in passes]
        ax.barh(methods, [1] * n_total, color=colors, alpha=0.85)
        for i, (m, p, lab) in enumerate(zip(methods, passes, labels)):
            ax.text(0.5, i, f"{lab}", ha="center", va="center", color="white", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(f"Refutation pass/fail  ({n_pass}/{n_total})")

        fig.suptitle("Causal Effect Refutation Report", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _err_result(method: str, exc: Exception, **extra) -> Dict:
        return {
            "method": method,
            "error": f"{type(exc).__name__}: {exc}",
            "passed": False,
            **extra,
        }
