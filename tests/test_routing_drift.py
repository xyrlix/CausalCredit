"""Tests for routing-distribution PSI in src.monitoring.drift_detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ref_routings():
    rng = np.random.default_rng(0)
    n = 1000
    # Match M7's observed distribution: REVIEW_BORDERLINE ~91%, PROCEED ~5%, etc.
    labels = rng.choice(
        ["PROCEED", "REVIEW_BORDERLINE", "REJECT_FRAUD", "REJECT_PACKAGING"],
        n,
        p=[0.05, 0.91, 0.025, 0.015],
    )
    return pd.Series(labels)


def test_routing_drift_zero_when_identical(ref_routings):
    from src.monitoring.drift_detector import DriftDetector
    detector = DriftDetector(reference_data=pd.DataFrame({"x": [0, 1]}))
    result = detector.detect_routing_drift(ref_routings, ref_routings.copy())
    assert result["psi"] == pytest.approx(0.0, abs=1e-6)
    assert result["status"] == "no_drift"


def test_routing_drift_detects_major_shift(ref_routings):
    from src.monitoring.drift_detector import DriftDetector
    detector = DriftDetector(reference_data=pd.DataFrame({"x": [0, 1]}))
    # Current: 50% PROCEED (way more clean)
    cur = pd.Series(["PROCEED"] * 500 + ["REVIEW_BORDERLINE"] * 500)
    result = detector.detect_routing_drift(ref_routings, cur)
    assert result["psi"] > 0.20
    assert result["status"] == "alert"


def test_routing_drift_detects_moderate_shift(ref_routings):
    from src.monitoring.drift_detector import DriftDetector
    detector = DriftDetector(reference_data=pd.DataFrame({"x": [0, 1]}))
    # Moderate shift: PROCEED goes 5% → 15%, REVIEW_BORDERLINE goes 91% → 80%
    rng = np.random.default_rng(42)
    cur = rng.choice(
        ["PROCEED", "REVIEW_BORDERLINE", "REJECT_FRAUD", "REJECT_PACKAGING"],
        1000,
        p=[0.15, 0.80, 0.025, 0.025],
    )
    cur = pd.Series(cur)
    result = detector.detect_routing_drift(ref_routings, cur)
    assert result["psi"] > 0.05
    # Should be at least moderate
    assert result["status"] in ("moderate", "alert")


def test_routing_drift_aligned_to_categories(ref_routings):
    from src.monitoring.drift_detector import DriftDetector
    detector = DriftDetector(reference_data=pd.DataFrame({"x": [0, 1]}))
    cats = ["PROCEED", "REVIEW_BORDERLINE", "REJECT_FRAUD", "REJECT_PACKAGING"]
    cur = ref_routings.copy()
    result = detector.detect_routing_drift(ref_routings, cur, categories=cats)
    assert set(result["ref_dist"].keys()) == set(cats)
    assert set(result["cur_dist"].keys()) == set(cats)
    # Distributions sum to 1
    assert sum(result["ref_dist"].values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(result["cur_dist"].values()) == pytest.approx(1.0, abs=1e-3)


def test_generate_drift_report_includes_routing_section(ref_routings):
    from src.monitoring.drift_detector import DriftDetector
    df_ref = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=1000)})
    df_cur = pd.DataFrame({"f1": np.random.default_rng(1).normal(size=500)})
    detector = DriftDetector(reference_data=df_ref)
    report = detector.generate_drift_report(
        current_data=df_cur,
        reference_routings=ref_routings,
        current_routings=ref_routings.copy(),
    )
    assert "## 3. Routing distribution drift" in report
    assert "PROCEED" in report
    assert "no_drift" in report


def test_routing_drift_handles_unseen_categories():
    from src.monitoring.drift_detector import DriftDetector
    detector = DriftDetector(reference_data=pd.DataFrame({"x": [0, 1]}))
    ref = pd.Series(["PROCEED"] * 100 + ["REVIEW_BORDERLINE"] * 100)
    cur = pd.Series(["PROCEED"] * 100 + ["REVIEW_DENOISED"] * 100)  # new category
    result = detector.detect_routing_drift(ref, cur)
    # Should be a meaningful drift
    assert result["psi"] > 0.05
    assert "REVIEW_DENOISED" in result["cur_dist"]
    assert result["cur_dist"]["REVIEW_DENOISED"] > 0
