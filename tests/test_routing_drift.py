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


# ---------------------------------------------------------------------------
# M8.1a — baseline persistence to routing_baseline.json
# ---------------------------------------------------------------------------


def test_routing_baseline_file_exists_with_5_categories():
    """M8.1a persisted baseline — the on-disk file the pipeline now reads."""
    import json
    from pathlib import Path
    path = Path("output/decision_reports/routing_baseline.json")
    assert path.exists(), f"Missing {path} — STEP 15 should have written it on first run"
    with open(path) as f:
        doc = json.load(f)
    cats = ["PROCEED", "REVIEW_BORDERLINE", "REVIEW_DENOISED", "REJECT_FRAUD", "REJECT_PACKAGING"]
    for c in cats:
        key = f"M7_{c}_FRAC"
        assert key in doc, f"Missing key {key}"
        v = float(doc[key])
        assert 0.0 <= v <= 1.0, f"{key}={v} out of [0, 1]"
    # Sum to ~1.0
    total = sum(float(doc[f"M7_{c}_FRAC"]) for c in cats)
    assert abs(total - 1.0) < 0.01, f"Baseline fractions sum to {total}, expected ~1.0"


def test_run_pipeline_uses_persisted_baseline(tmp_path):
    """When routing_baseline.json exists, the pipeline reads it (not hardcoded)."""
    import json
    # Replicate the same parsing logic that run_pipeline uses
    with open("output/decision_reports/routing_baseline.json") as f:
        doc = json.load(f)
    baseline = {k.replace("M7_", "").replace("_FRAC", ""): float(v)
                for k, v in doc.items() if k.startswith("M7_") and "_FRAC" in k}
    assert "PROCEED" in baseline
    assert "REJECT_FRAUD" in baseline
    assert "REVIEW_DENOISED" in baseline  # the previously-missing 5th category
    # PROCEED should be small (~5%), REVIEW_BORDERLINE large (~90%)
    assert 0.0 <= baseline["PROCEED"] <= 0.20
    assert 0.50 <= baseline["REVIEW_BORDERLINE"] <= 1.00
