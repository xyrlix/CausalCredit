"""4-Layer Causal Verification Pyramid (CCGS).

@requirement REQ-VERIFY-001
@design docs/CausalCredit_因果推理验证标准体系.md §6 (因果验证金字塔)

A composite grading framework that consolidates every individual
verification step in the CausalCredit pipeline into a single grade,
following the CausalBench / "Oracle P0" 4-layer pyramid recipe.  The
pyramid is bottom-up, where each layer checks a different aspect of
the causal claim:

::

                  ┌─────────────────────────────┐
                  │  L4  E2E Validation         │   0.20
                  │   (AUC, ECE, Demographic P) │
                  ├─────────────────────────────┤
                  │  L3  Counterfactual         │   0.25
                  │   (CCR, Immutable)          │
                  ├─────────────────────────────┤
                  │  L2  Effect Validation      │   0.30
                  │   (Refutation + BLP + CATE) │
                  ├─────────────────────────────┤
                  │  L1  Graph Validation       │   0.25
                  │   (GSI, DKCS)               │
                  └─────────────────────────────┘

                          CCGS = 0.25·L1 + 0.30·L2
                               + 0.25·L3 + 0.20·L4

Each layer score is in ``[0, 1]``.  A layer "passes" when its score
is ``>= 0.7``.  The composite **CCGS** (Composite Causal Grade
Score) passes when ``CCGS >= 0.7 AND all_layers_pass`` — the dual
requirement is deliberate: a single weak layer cannot be masked by
strong ones.

L1 — Graph Validation
    *GSI* (Graph Stability Index) = mean Jaccard similarity between
    bootstrap-discovered DAGs and the reference DAG.
    *DKCS* (Domain Knowledge Consistency Score) = per-domain-edge
    confirmation rate across bootstrap samples.
    L1 = mean(GSI, DKCS).

L2 — Effect Validation
    Aggregates 5 tests, each contributing equally:
    1. Placebo refuter  (refute.placebo_treatment_refuter)
    2. Subset refuter   (refute.data_subset_refuter)
    3. Sensitivity      (refute.unobserved_confounding_refuter)
    4. BLP              (src.causal.blp_test.BLPTest)
    5. CATE consistency (src.causal.cate.cross_validate_methods)
    L2 = (# passed) / 5.

L3 — Counterfactual Validation
    *CCR* (Counterfactual Consistency Rate) = fraction of generated
    counterfactuals that respect plausibility constraints.
    *Immutable violation* = fraction of generated counterfactuals
    that altered an immutable feature.
    L3 = min(CCR, 1 − immutable_violation / 0.2).

L4 — E2E Validation
    *AUC improvement* = (causal-model AUC) − (baseline AUC), normalized
    by a 0.05 improvement target.
    *ECE* (Expected Calibration Error), normalized so 0.1 → 0.
    *Demographic parity difference* across protected groups, normalized
    so 0.1 → 0.
    L4 = mean of available sub-scores.

Reference
---------
The pyramid recipe and CCGS weighting follow the competitor project
(``tmp/CausalCredit/src/causal/verification.py``), retargeted at our
:class:`CausalRefuter`, :class:`BLPTest`, and
``CATEEstimator.cross_validate_methods`` results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("causalcredit.causal.verification")


# ===========================================================================
# Result containers
# ===========================================================================

@dataclass
class PyramidLayerResult:
    """Container for a single pyramid layer."""
    name: str
    score: float
    pass_: bool
    components: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": float(self.score),
            "pass": bool(self.pass_),
            "components": dict(self.components),
            "notes": str(self.notes),
        }


@dataclass
class PyramidResult:
    """Container for the full pyramid output."""
    ccgs: float
    l1: PyramidLayerResult
    l2: PyramidLayerResult
    l3: PyramidLayerResult
    l4: PyramidLayerResult
    all_layers_pass: bool
    overall_pass: bool
    weight_l1: float = 0.25
    weight_l2: float = 0.30
    weight_l3: float = 0.25
    weight_l4: float = 0.20

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ccgs": float(self.ccgs),
            "l1": self.l1.to_dict(),
            "l2": self.l2.to_dict(),
            "l3": self.l3.to_dict(),
            "l4": self.l4.to_dict(),
            "all_layers_pass": bool(self.all_layers_pass),
            "overall_pass": bool(self.overall_pass),
            "weights": {
                "l1": float(self.weight_l1),
                "l2": float(self.weight_l2),
                "l3": float(self.weight_l3),
                "l4": float(self.weight_l4),
            },
        }


# ===========================================================================
# Main class
# ===========================================================================

class CausalVerificationPyramid:
    """4-layer causal verification pyramid with composite CCGS grade.

    Each ``verify_lN_*`` method takes the raw results from the
    corresponding lower-level verification modules and returns a
    :class:`PyramidLayerResult`.  The composite grade is computed by
    :meth:`compute_ccgs`.

    Pass threshold for every layer is 0.70 (configurable via
    ``threshold``).
    """

    def __init__(self, threshold: float = 0.70):
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1]; got {threshold}")
        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # L1 — Graph Validation
    # ------------------------------------------------------------------
    def verify_l1_graph(
        self,
        dag_edges: List[Tuple[str, str]],
        bootstrap_samples: Optional[List[Set[Tuple[str, str]]]] = None,
    ) -> PyramidLayerResult:
        """L1: Graph validation metrics.

        GSI = mean Jaccard similarity between each bootstrap-discovered
        DAG and the reference DAG.
        DKCS = mean per-domain-edge confirmation rate across bootstraps.

        Without bootstrap evidence, returns the placeholder ``score=1.0``
        (domain DAG is trusted).
        """
        if not bootstrap_samples:
            return PyramidLayerResult(
                name="L1_graph",
                score=1.0,
                pass_=True,
                components={"gsi": 1.0, "dkcs": 1.0, "n_bootstrap": 0},
                notes="No bootstrap data — domain DAG trusted as-is.",
            )

        ref_set: Set[Tuple[str, str]] = set(dag_edges) if dag_edges else set()
        jaccards: List[float] = []
        for sample_set in bootstrap_samples:
            sample = set(sample_set)
            inter = len(ref_set & sample)
            union = len(ref_set | sample)
            jaccards.append(inter / union if union > 0 else 1.0)
        gsi = float(np.mean(jaccards)) if jaccards else 1.0

        if ref_set:
            n_samples = len(bootstrap_samples)
            edge_conf = [
                sum(1 for s in bootstrap_samples if edge in s) / n_samples
                for edge in ref_set
            ]
            dkcs = float(np.mean(edge_conf))
        else:
            dkcs = 1.0

        score = float(np.mean([gsi, dkcs]))
        return PyramidLayerResult(
            name="L1_graph",
            score=score,
            pass_=score >= self.threshold,
            components={
                "gsi": round(gsi, 6),
                "dkcs": round(dkcs, 6),
                "n_bootstrap": len(bootstrap_samples),
                "n_ref_edges": len(ref_set),
            },
            notes=(
                "GSI = Jaccard(bootstrap ∩ ref) ; "
                "DKCS = per-domain-edge confirmation rate."
            ),
        )

    # ------------------------------------------------------------------
    # L2 — Effect Validation
    # ------------------------------------------------------------------
    def verify_l2_effect(
        self,
        placebo_result: Optional[Dict[str, Any]] = None,
        subset_result: Optional[Dict[str, Any]] = None,
        sensitivity_result: Optional[Dict[str, Any]] = None,
        blp_result: Optional[Any] = None,
        cate_spearman: Optional[Dict[str, Any]] = None,
    ) -> PyramidLayerResult:
        """L2: Causal effect validation.

        Aggregates 5 tests, each contributing equally to the layer
        score.  Each input may be ``None`` (test not run yet) — missing
        tests are excluded from both the numerator and denominator.

        Pass-detection convention: ``input.get("pass", False)``.  Our
        own ``BLPResult`` exposes ``pass_at_05`` instead, so we map
        ``BLPResult -> {"pass": blp_result.pass_at_05}`` before calling.
        """
        test_inputs = [
            ("placebo", placebo_result),
            ("subset", subset_result),
            ("sensitivity", sensitivity_result),
            ("blp", blp_result),
            ("cate_consistency", cate_spearman),
        ]
        components: Dict[str, Any] = {}
        passes: List[bool] = []
        for name, test in test_inputs:
            if test is None:
                components[f"{name}_pass"] = None
                continue
            # Translate BLPResult -> dict
            if hasattr(test, "pass_at_05") and not isinstance(test, dict):
                test = {"pass": bool(test.pass_at_05), "blp_p_value": float(test.blp_p_value)}
            p = bool(test.get("pass", False))
            components[f"{name}_pass"] = p
            if "blp_p_value" in test:
                components[f"{name}_p_value"] = float(test["blp_p_value"])
            passes.append(p)

        if not passes:
            score = 1.0
            notes = "No L2 tests provided — L2 trivially passes."
        else:
            score = sum(passes) / len(passes)
            notes = f"{sum(passes)}/{len(passes)} L2 tests passed."

        return PyramidLayerResult(
            name="L2_effect",
            score=float(score),
            pass_=float(score) >= self.threshold,
            components=components,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # L3 — Counterfactual Validation
    # ------------------------------------------------------------------
    def verify_l3_counterfactual(
        self,
        ccr: Optional[float] = None,
        immutable_violation: Optional[float] = None,
    ) -> PyramidLayerResult:
        """L3: Counterfactual validation.

        CCR (Counterfactual Consistency Rate) is the fraction of
        generated counterfactuals that satisfy plausibility constraints.
        Immutable violation is the fraction that altered an immutable
        feature (e.g. age, gender).

        L3 = min(CCR, 1 − immutable_violation / 0.2).  Missing
        components default to 1.0 (conservative — only penalize what
        we actually measured).
        """
        if ccr is None and immutable_violation is None:
            return PyramidLayerResult(
                name="L3_counterfactual",
                score=1.0,
                pass_=True,
                components={"ccr": None, "immutable_violation": None},
                notes="No counterfactual diagnostics — L3 trivially passes.",
            )

        ccr_score = max(0.0, min(1.0, ccr if ccr is not None else 1.0))
        imm_score = (
            max(0.0, 1.0 - (immutable_violation / 0.2))
            if immutable_violation is not None
            else 1.0
        )
        score = float(min(ccr_score, imm_score))
        return PyramidLayerResult(
            name="L3_counterfactual",
            score=score,
            pass_=score >= self.threshold,
            components={
                "ccr": ccr,
                "immutable_violation": immutable_violation,
                "immutable_pass": bool(imm_score >= 0.5),
            },
            notes=(
                f"L3 = min(CCR={ccr_score:.3f}, imm_score={imm_score:.3f})."
            ),
        )

    # ------------------------------------------------------------------
    # L4 — E2E Validation
    # ------------------------------------------------------------------
    def verify_l4_e2e(
        self,
        auc_baseline: float = 0.0,
        auc_causal: float = 0.0,
        ece: Optional[float] = None,
        demographic_parity_diff: Optional[float] = None,
    ) -> PyramidLayerResult:
        """L4: End-to-end validation.

        AUC improvement = (causal − baseline), normalized by a 0.05
        target.  ECE normalized so 0.1 → 0.  Demographic parity diff
        normalized so 0.1 → 0.  Missing optional components are
        excluded from the average.
        """
        auc_improvement = float(auc_causal - auc_baseline)
        auc_score = max(0.0, min(1.0, auc_improvement / 0.05))

        scores: List[float] = [auc_score]
        ece_score: Optional[float] = None
        if ece is not None:
            ece_score = max(0.0, 1.0 - float(ece) / 0.1)
            scores.append(ece_score)

        dp_score: Optional[float] = None
        if demographic_parity_diff is not None:
            dp_score = max(0.0, 1.0 - float(demographic_parity_diff) / 0.1)
            scores.append(dp_score)

        score = float(np.mean(scores))
        return PyramidLayerResult(
            name="L4_e2e",
            score=score,
            pass_=score >= self.threshold,
            components={
                "auc_baseline": float(auc_baseline),
                "auc_causal": float(auc_causal),
                "auc_improvement": round(auc_improvement, 6),
                "auc_score": round(auc_score, 6),
                "ece": ece,
                "ece_score": ece_score,
                "demographic_parity_diff": demographic_parity_diff,
                "dp_score": dp_score,
            },
            notes=(
                f"AUC +{(auc_improvement or 0) * 100:.2f} pp; "
                f"components: {[s for s in (ece_score, dp_score) if s is not None] or 'auc-only'}."
            ),
        )

    # ------------------------------------------------------------------
    # CCGS — Composite Causal Grade Score
    # ------------------------------------------------------------------
    def compute_ccgs(
        self,
        l1: PyramidLayerResult,
        l2: PyramidLayerResult,
        l3: PyramidLayerResult,
        l4: PyramidLayerResult,
        weights: Optional[Dict[str, float]] = None,
    ) -> PyramidResult:
        """Composite Causal Grade Score.

        CCGS = 0.25·L1 + 0.30·L2 + 0.25·L3 + 0.20·L4

        Pass condition: ``CCGS >= 0.7 AND all_layers_pass``.  The
        dual requirement means a weak layer cannot be masked by
        strong others — a single failure fails the whole pyramid.
        """
        w = weights or {"l1": 0.25, "l2": 0.30, "l3": 0.25, "l4": 0.20}
        wsum = sum(w.values())
        if abs(wsum - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0; got {wsum}")
        ccgs = (
            w["l1"] * l1.score
            + w["l2"] * l2.score
            + w["l3"] * l3.score
            + w["l4"] * l4.score
        )
        all_layers_pass = bool(l1.pass_ and l2.pass_ and l3.pass_ and l4.pass_)
        overall = bool(ccgs >= self.threshold and all_layers_pass)
        logger.info(
            "CCGS=%.4f  L1=%.3f  L2=%.3f  L3=%.3f  L4=%.3f  "
            "all_layers_pass=%s  overall=%s",
            ccgs, l1.score, l2.score, l3.score, l4.score,
            all_layers_pass, overall,
        )
        return PyramidResult(
            ccgs=float(ccgs),
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            all_layers_pass=all_layers_pass,
            overall_pass=overall,
            weight_l1=float(w["l1"]),
            weight_l2=float(w["l2"]),
            weight_l3=float(w["l3"]),
            weight_l4=float(w["l4"]),
        )


# ===========================================================================
# Visualization
# ===========================================================================

def plot_pyramid(
    result: PyramidResult,
    output_path: str,
) -> str:
    """Render a 1×2 chart for the pyramid result.

    Panel A: stacked bar of L1-L4 layer scores + CCGS line.
    Panel B: text summary of the layer scores and pass flags.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: bar chart
    layers = ["L1\ngraph", "L2\neffect", "L3\ncounterfactual", "L4\ne2e"]
    scores = [result.l1.score, result.l2.score, result.l3.score, result.l4.score]
    pass_colors = ["#2ca02c" if s >= 0.7 else "#d62728" for s in scores]
    bars = ax1.bar(layers, scores, color=pass_colors, alpha=0.85)
    ax1.axhline(0.7, color="black", linestyle="--", linewidth=1, label="threshold (0.70)")
    ax1.axhline(result.ccgs, color="#1f77b4", linestyle=":", linewidth=1.5,
                label=f"CCGS = {result.ccgs:.3f}")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Layer score")
    ax1.set_title(
        f"Causal Verification Pyramid (CCGS)\n"
        f"{'PASS' if result.overall_pass else 'FAIL'}  ·  "
        f"all_layers_pass={result.all_layers_pass}"
    )
    ax1.legend(loc="lower right", fontsize=8)
    for bar, score in zip(bars, scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    # Panel B: text summary
    txt = [
        f"CCGS = {result.ccgs:.4f}",
        f"Overall pass: {result.overall_pass}",
        f"All layers pass: {result.all_layers_pass}",
        "",
        f"L1 graph         = {result.l1.score:.3f}  [{'PASS' if result.l1.pass_ else 'FAIL'}]",
        f"L2 effect        = {result.l2.score:.3f}  [{'PASS' if result.l2.pass_ else 'FAIL'}]",
        f"L3 counterfactual= {result.l3.score:.3f}  [{'PASS' if result.l3.pass_ else 'FAIL'}]",
        f"L4 e2e           = {result.l4.score:.3f}  [{'PASS' if result.l4.pass_ else 'FAIL'}]",
        "",
        f"Weights: L1={result.weight_l1}, L2={result.weight_l2}, "
        f"L3={result.weight_l3}, L4={result.weight_l4}",
    ]
    ax2.text(0.05, 0.5, "\n".join(txt), family="monospace", fontsize=10,
             transform=ax2.transAxes, va="center")
    ax2.set_axis_off()
    ax2.set_title("Pyramid summary")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("pyramid chart written: %s", output_path)
    return output_path
