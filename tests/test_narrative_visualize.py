"""Tests for src.explain.narrative_visualize."""

from __future__ import annotations

import os
import pytest


def _make_narrative():
    return {
        "model_level": {
            "top_features": [
                {"feature": "debt", "mean_abs_shap": 0.12},
                {"feature": "credit_score", "mean_abs_shap": 0.08},
                {"feature": "age", "mean_abs_shap": 0.04},
            ],
            "narrative": "P(default) is driven mainly by debt (0.12), credit_score (0.08), and age (0.04).",
        },
        "cohort_level": {
            "k": 10,
            "cohort_mean_p_default": 0.07,
            "applicant_p_default": 0.18,
            "delta": 0.11,
            "top_deviations": [
                {"feature": "debt", "z": 2.5, "applicant": 0.8, "cohort_mean": 0.3},
                {"feature": "age", "z": -1.2, "applicant": 25, "cohort_mean": 40},
            ],
            "narrative": "Among the 10 most similar training applicants, mean P(default) is 7.00%; this applicant's 18.00% is noticeably higher (Δ=+0.11).",
        },
        "individual_level": {
            "top_features": [
                {"feature": "debt", "value": 0.8, "shap": 0.20, "quadrant": "TRUSTED", "dag_paths": [["debt", "TARGET"]]},
                {"feature": "credit_score", "value": 600, "shap": -0.10, "quadrant": "TRUSTED", "dag_paths": [["credit_score", "TARGET"]]},
                {"feature": "age", "value": 25, "shap": 0.05, "quadrant": "UNTRUSTED", "dag_paths": [["age", "debt", "TARGET"]]},
            ],
            "narrative": "The dominant risk driver is debt (SHAP=+0.20, quadrant=TRUSTED).",
        },
        "robustness": {
            "stability_score": 0.85,
            "top_1_stable": 0.9,
            "top_3_stable": 0.8,
            "n_perturbations": 20,
            "noise_frac": 0.10,
            "interpretation": "Explanation is robust (stability=0.85).",
        },
    }


def test_plot_causal_waterfall_writes_png(tmp_path):
    from src.explain.narrative_visualize import plot_causal_waterfall
    n = _make_narrative()
    out = str(tmp_path / "wf.png")
    plot_causal_waterfall(n["individual_level"]["top_features"], out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000


def test_plot_causal_waterfall_handles_empty(tmp_path):
    from src.explain.narrative_visualize import plot_causal_waterfall
    out = str(tmp_path / "wf_empty.png")
    plot_causal_waterfall([], out)
    # Empty input → no file created (early return)
    assert not os.path.exists(out)


def test_plot_narrative_card_writes_png(tmp_path):
    from src.explain.narrative_visualize import plot_narrative_card
    out = str(tmp_path / "card.png")
    plot_narrative_card(_make_narrative(), out, applicant_id="HC_006355")
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000


def test_render_all_writes_two_pngs(tmp_path):
    from src.explain.narrative_visualize import render_all
    paths = render_all(_make_narrative(), str(tmp_path), applicant_id="HC_006355")
    assert len(paths) == 2
    for p in paths:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 1000
