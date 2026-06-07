"""Unit tests for src.causal.verification (M8.6d — CCGS pyramid).

Coverage:

* L1 graph — GSI + DKCS, with and without bootstrap data
* L2 effect — pass-rate aggregation, missing tests, BLPResult passthrough
* L3 counterfactual — CCR + immutable violation
* L4 e2e — AUC improvement, ECE, demographic parity
* CCGS — weighted average, dual pass requirement
* Robustness — bad threshold, weights don't sum to 1
* Visualization — chart file is produced
"""

from __future__ import annotations

import numpy as np
import pytest

from src.causal.blp_test import BLPResult
from src.causal.verification import (
    CausalVerificationPyramid,
    PyramidLayerResult,
    PyramidResult,
    plot_pyramid,
)


# ---------------------------------------------------------------------------
# L1 — Graph
# ---------------------------------------------------------------------------
class TestL1Graph:
    def test_no_bootstrap_returns_placeholder(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l1_graph(dag_edges=[("A", "B")], bootstrap_samples=None)
        assert isinstance(res, PyramidLayerResult)
        assert res.score == 1.0
        assert res.pass_ is True
        assert "trusted" in res.notes

    def test_perfect_overlap(self):
        pyramid = CausalVerificationPyramid()
        edges = {("A", "B"), ("C", "D")}
        samples = [edges, edges, edges]
        res = pyramid.verify_l1_graph(list(edges), samples)
        assert res.score == pytest.approx(1.0, abs=1e-6)
        assert res.pass_ is True
        assert res.components["gsi"] == pytest.approx(1.0, abs=1e-6)
        assert res.components["dkcs"] == pytest.approx(1.0, abs=1e-6)

    def test_partial_overlap(self):
        pyramid = CausalVerificationPyramid()
        ref = {("A", "B"), ("C", "D")}
        # Bootstrap 1 matches perfectly; bootstrap 2 has only A->B
        samples = [ref, {("A", "B")}]
        res = pyramid.verify_l1_graph(list(ref), samples)
        # Jaccard: sample 1 = 1.0, sample 2 = 1/2
        # Mean = 0.75
        assert res.components["gsi"] == pytest.approx(0.75, abs=1e-6)
        # DKCS: A->B confirmed in 2/2 = 1.0; C->D confirmed in 1/2 = 0.5
        # Mean = 0.75
        assert res.components["dkcs"] == pytest.approx(0.75, abs=1e-6)
        assert res.score == pytest.approx(0.75, abs=1e-6)
        assert res.pass_ is True  # 0.75 >= 0.70

    def test_failing_gsi(self):
        pyramid = CausalVerificationPyramid()
        ref = {("A", "B"), ("C", "D"), ("E", "F")}
        # Bootstrap samples are all completely different
        samples = [{("X", "Y")}, {("P", "Q")}, {("R", "S")}]
        res = pyramid.verify_l1_graph(list(ref), samples)
        # GSI = 0, DKCS = 0, score = 0
        assert res.score < 0.7
        assert res.pass_ is False


# ---------------------------------------------------------------------------
# L2 — Effect
# ---------------------------------------------------------------------------
class TestL2Effect:
    def test_all_pass(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l2_effect(
            placebo_result={"pass": True},
            subset_result={"pass": True},
            sensitivity_result={"pass": True},
            blp_result={"pass": True},
            cate_spearman={"pass": True},
        )
        assert res.score == 1.0
        assert res.pass_ is True
        assert res.components["placebo_pass"] is True
        assert res.components["subset_pass"] is True

    def test_partial_pass(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l2_effect(
            placebo_result={"pass": True},
            subset_result={"pass": False},
            sensitivity_result={"pass": True},
            blp_result={"pass": False},
            cate_spearman={"pass": True},
        )
        # 3/5 pass
        assert res.score == pytest.approx(0.6, abs=1e-6)
        assert res.pass_ is False

    def test_missing_tests_excluded(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l2_effect(
            placebo_result={"pass": True},
            subset_result={"pass": True},
            sensitivity_result=None,
            blp_result=None,
            cate_spearman=None,
        )
        # 2/2 = 1.0 (missing tests excluded from both num and denom)
        assert res.score == 1.0

    def test_all_missing(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l2_effect()
        # No tests → 1.0 (trivially passes, with a "no tests" note)
        assert res.score == 1.0
        assert "trivially" in res.notes

    def test_blp_result_dataclass_passthrough(self):
        """BLPResult objects are translated to {pass, p_value} dicts."""
        pyramid = CausalVerificationPyramid()
        blp = BLPResult(
            blp_coef=0.1,
            blp_se=0.05,
            blp_t_stat=2.0,
            blp_p_value=0.01,
            n_obs=500,
            n_folds=5,
            method="LinearDML",
            pass_at_05=True,
            pass_at_10=True,
        )
        res = pyramid.verify_l2_effect(blp_result=blp)
        assert res.components["blp_pass"] is True


# ---------------------------------------------------------------------------
# L3 — Counterfactual
# ---------------------------------------------------------------------------
class TestL3Counterfactual:
    def test_no_data_returns_placeholder(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l3_counterfactual()
        assert res.score == 1.0
        assert res.pass_ is True

    def test_perfect_ccr_no_violations(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l3_counterfactual(ccr=1.0, immutable_violation=0.0)
        assert res.score == 1.0
        assert res.pass_ is True

    def test_ccr_lower_bounds_score(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l3_counterfactual(ccr=0.5, immutable_violation=0.0)
        # min(0.5, 1.0) = 0.5
        assert res.score == pytest.approx(0.5, abs=1e-6)
        assert res.pass_ is False

    def test_immutable_violation_lower_bounds_score(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l3_counterfactual(ccr=1.0, immutable_violation=0.3)
        # imm_score = 1 - 0.3/0.2 = 1 - 1.5 = -0.5, clipped to 0
        # min(1.0, 0) = 0
        assert res.score == 0.0
        assert res.pass_ is False

    def test_ccr_clipped_to_unit_interval(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l3_counterfactual(ccr=1.5, immutable_violation=0.0)
        # ccr_score = clip(1.5, 0, 1) = 1.0
        assert res.score == 1.0


# ---------------------------------------------------------------------------
# L4 — E2E
# ---------------------------------------------------------------------------
class TestL4E2E:
    def test_auc_improvement_alone(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l4_e2e(auc_baseline=0.70, auc_causal=0.72)
        # AUC improvement = 0.02, normalized: 0.02/0.05 = 0.4
        assert res.score == pytest.approx(0.4, abs=1e-6)
        assert res.pass_ is False  # 0.4 < 0.7
        assert res.components["auc_improvement"] == pytest.approx(0.02, abs=1e-6)

    def test_strong_auc_improvement(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l4_e2e(auc_baseline=0.70, auc_causal=0.78)
        # AUC improvement = 0.08, normalized to 1.0 (clipped)
        assert res.score == pytest.approx(1.0, abs=1e-6)
        assert res.pass_ is True

    def test_ece_and_dp_combine(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l4_e2e(
            auc_baseline=0.70,
            auc_causal=0.75,  # improvement = 0.05 → 1.0
            ece=0.05,          # → 0.5
            demographic_parity_diff=0.02,  # → 0.8
        )
        # mean(1.0, 0.5, 0.8) = 0.7667
        assert res.score == pytest.approx(0.7667, abs=1e-3)
        assert res.pass_ is True

    def test_zero_auc_improvement(self):
        pyramid = CausalVerificationPyramid()
        res = pyramid.verify_l4_e2e(auc_baseline=0.70, auc_causal=0.70)
        assert res.score == 0.0
        assert res.pass_ is False


# ---------------------------------------------------------------------------
# CCGS
# ---------------------------------------------------------------------------
class TestCCGS:
    def test_all_perfect_layers(self):
        pyramid = CausalVerificationPyramid()
        l1 = pyramid.verify_l1_graph([("A", "B")], None)
        l2 = pyramid.verify_l2_effect(
            placebo_result={"pass": True}, subset_result={"pass": True},
            sensitivity_result={"pass": True}, blp_result={"pass": True},
            cate_spearman={"pass": True},
        )
        l3 = pyramid.verify_l3_counterfactual()
        l4 = pyramid.verify_l4_e2e(auc_baseline=0.7, auc_causal=0.8)
        result = pyramid.compute_ccgs(l1, l2, l3, l4)
        assert isinstance(result, PyramidResult)
        assert result.ccgs == pytest.approx(1.0, abs=1e-6)
        assert result.overall_pass is True
        assert result.all_layers_pass is True

    def test_dual_requirement_fails_when_one_layer_fails(self):
        """CCGS can be high but still FAIL if one layer is below threshold."""
        pyramid = CausalVerificationPyramid(threshold=0.7)
        # L1 perfect, L2 0.6 (fail), L3 perfect, L4 strong
        l1 = pyramid.verify_l1_graph([("A", "B")], None)
        l2 = PyramidLayerResult(
            name="L2_effect", score=0.6, pass_=False,
            components={"note": "synthetic failure"},
        )
        l3 = pyramid.verify_l3_counterfactual()
        l4 = pyramid.verify_l4_e2e(auc_baseline=0.7, auc_causal=0.8)
        result = pyramid.compute_ccgs(l1, l2, l3, l4)
        # CCGS = 0.25*1 + 0.30*0.6 + 0.25*1 + 0.20*1 = 0.88
        # (above threshold) but L2 fails, so overall_pass = False
        assert result.ccgs == pytest.approx(0.88, abs=1e-6)
        assert result.all_layers_pass is False
        assert result.overall_pass is False

    def test_below_ccgs_threshold_fails(self):
        pyramid = CausalVerificationPyramid(threshold=0.7)
        l1 = pyramid.verify_l1_graph([("A", "B")], None)  # 1.0
        l2 = pyramid.verify_l2_effect(placebo_result={"pass": True},
                                       subset_result={"pass": True})  # 1.0
        l3 = pyramid.verify_l3_counterfactual()  # 1.0
        l4 = pyramid.verify_l4_e2e(auc_baseline=0.7, auc_causal=0.7)  # 0.0
        result = pyramid.compute_ccgs(l1, l2, l3, l4)
        # CCGS = 0.25 + 0.30 + 0.25 + 0 = 0.80 (passes threshold)
        # But L4 fails (score = 0 < 0.7)
        assert result.ccgs == pytest.approx(0.80, abs=1e-6)
        assert result.all_layers_pass is False
        assert result.overall_pass is False

    def test_to_dict_roundtrip(self):
        pyramid = CausalVerificationPyramid()
        l1 = pyramid.verify_l1_graph([("A", "B")], None)
        l2 = pyramid.verify_l2_effect()
        l3 = pyramid.verify_l3_counterfactual()
        l4 = pyramid.verify_l4_e2e(auc_baseline=0.7, auc_causal=0.72)
        result = pyramid.compute_ccgs(l1, l2, l3, l4)
        d = result.to_dict()
        assert "ccgs" in d
        assert "l1" in d and "l4" in d
        assert d["l1"]["name"] == "L1_graph"
        assert "weights" in d


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            CausalVerificationPyramid(threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            CausalVerificationPyramid(threshold=1.5)

    def test_weights_must_sum_to_one(self):
        pyramid = CausalVerificationPyramid()
        l1 = pyramid.verify_l1_graph([("A", "B")], None)
        l2 = pyramid.verify_l2_effect()
        l3 = pyramid.verify_l3_counterfactual()
        l4 = pyramid.verify_l4_e2e(auc_baseline=0.7, auc_causal=0.75)
        with pytest.raises(ValueError, match="sum to 1"):
            pyramid.compute_ccgs(
                l1, l2, l3, l4,
                weights={"l1": 0.4, "l2": 0.4, "l3": 0.4, "l4": 0.4},
            )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
class TestVisualization:
    def test_plot_writes_png(self, tmp_path):
        pyramid = CausalVerificationPyramid()
        l1 = pyramid.verify_l1_graph([("A", "B")], None)
        l2 = pyramid.verify_l2_effect(
            placebo_result={"pass": True}, subset_result={"pass": True},
            sensitivity_result={"pass": True}, blp_result={"pass": True},
            cate_spearman={"pass": True},
        )
        l3 = pyramid.verify_l3_counterfactual(ccr=0.9, immutable_violation=0.0)
        l4 = pyramid.verify_l4_e2e(auc_baseline=0.7, auc_causal=0.75, ece=0.05)
        result = pyramid.compute_ccgs(l1, l2, l3, l4)
        out = tmp_path / "pyramid.png"
        plot_pyramid(result, str(out))
        assert out.exists()
        assert out.stat().st_size > 1000
