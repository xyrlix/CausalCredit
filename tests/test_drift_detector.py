"""Unit tests for src.monitoring.drift_detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector, _safe_psi_from_dists


@pytest.fixture
def reference_df():
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "x_norm": rng.normal(0, 1, size=500),
        "x_uniform": rng.uniform(0, 100, size=500),
        "x_int": rng.randint(0, 10, size=500),
    })


@pytest.fixture
def current_no_drift(reference_df):
    rng = np.random.RandomState(1)
    return pd.DataFrame({
        "x_norm": rng.normal(0, 1, size=500),
        "x_uniform": rng.uniform(0, 100, size=500),
        "x_int": rng.randint(0, 10, size=500),
    })


@pytest.fixture
def current_drifted(reference_df):
    rng = np.random.RandomState(2)
    return pd.DataFrame({
        "x_norm": rng.normal(3, 2, size=500),     # shifted mean + spread
        "x_uniform": rng.uniform(50, 200, size=500),  # different support
        "x_int": rng.randint(5, 15, size=500),    # offset
    })


def test_safe_psi_identical_distributions_is_zero():
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    cur = np.array([0.25, 0.25, 0.25, 0.25])
    assert _safe_psi_from_dists(ref, cur) == pytest.approx(0.0, abs=1e-9)


def test_safe_psi_handles_zeros():
    # epsilon smoothing avoids log(0)
    ref = np.array([0.5, 0.5, 0.0])
    cur = np.array([0.0, 0.5, 0.5])
    psi = _safe_psi_from_dists(ref, cur)
    assert np.isfinite(psi)
    assert psi > 0


def test_compute_psi_no_drift(reference_df, current_no_drift):
    d = DriftDetector(reference_df)
    psi = d.compute_psi("x_norm", current_no_drift["x_norm"])
    assert psi < DriftDetector.PSI_NO_DRIFT, f"expected no drift, got PSI={psi:.4f}"


def test_compute_psi_alerts_on_drift(reference_df, current_drifted):
    d = DriftDetector(reference_df)
    psi = d.compute_psi("x_norm", current_drifted["x_norm"])
    assert psi >= DriftDetector.PSI_ALERT, f"expected alert drift, got PSI={psi:.4f}"


def test_compute_psi_unknown_feature_raises(reference_df):
    d = DriftDetector(reference_df)
    with pytest.raises(KeyError):
        d.compute_psi("not_a_feature", pd.Series([1, 2, 3]))


def test_label_psi_buckets():
    assert DriftDetector._label_psi(0.05) == "no_drift"
    assert DriftDetector._label_psi(0.15) == "moderate"
    assert DriftDetector._label_psi(0.30) == "alert"
    assert DriftDetector._label_psi(float("nan")) == "n/a"


def test_detect_feature_drift_returns_dataframe(reference_df, current_drifted):
    d = DriftDetector(reference_df)
    res = d.detect_feature_drift(current_drifted)
    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"feature", "psi", "status"}
    assert len(res) == 3
    # Drifted data should trigger at least one alert
    assert (res["status"] == "alert").sum() >= 1


def test_detect_feature_drift_subset_features(reference_df, current_no_drift):
    d = DriftDetector(reference_df)
    res = d.detect_feature_drift(current_no_drift, features=["x_norm"])
    assert len(res) == 1
    assert res.iloc[0]["feature"] == "x_norm"


def test_detect_prediction_drift_returns_psi(reference_df):
    d = DriftDetector(reference_df)
    rng = np.random.RandomState(0)
    ref_scores = pd.Series(rng.beta(2, 5, size=500))
    cur_scores = pd.Series(rng.beta(2, 5, size=500))
    res = d.detect_prediction_drift(ref_scores, cur_scores)
    assert "psi" in res and "status" in res
    assert "ref_mean" in res and "cur_mean" in res
    assert np.isfinite(res["psi"])


def test_detect_prediction_drift_flags_shift(reference_df):
    d = DriftDetector(reference_df)
    rng = np.random.RandomState(0)
    ref_scores = pd.Series(rng.beta(2, 5, size=500))
    cur_scores = pd.Series(rng.beta(5, 2, size=500))  # very different shape
    res = d.detect_prediction_drift(ref_scores, cur_scores)
    assert res["psi"] > DriftDetector.PSI_ALERT


def test_detect_concept_drift_ok_when_no_degradation(reference_df):
    d = DriftDetector(reference_df)
    res = d.detect_concept_drift(current_auc=0.78, baseline_auc=0.80)
    assert res["status"] == "ok"
    assert res["auc_drop_alert"] is False


def test_detect_concept_drift_alerts_on_drop(reference_df):
    d = DriftDetector(reference_df)
    res = d.detect_concept_drift(current_auc=0.70, baseline_auc=0.80)
    assert res["auc_drop_alert"] is True
    assert res["status"] == "alert"


def test_detect_concept_drift_with_ks(reference_df):
    d = DriftDetector(reference_df)
    res = d.detect_concept_drift(
        current_auc=0.79, baseline_auc=0.80,
        current_ks=0.20, baseline_ks=0.40,
    )
    assert res["ks_drop_alert"] is True
    assert res["status"] == "alert"


def test_compute_feature_statistics(reference_df, current_drifted):
    d = DriftDetector(reference_df)
    stats = d.compute_feature_statistics(current_drifted)
    assert isinstance(stats, pd.DataFrame)
    assert "delta_mean" in stats.columns
    # x_norm should have a non-trivial mean shift
    x_row = stats[stats["feature"] == "x_norm"].iloc[0]
    assert abs(x_row["delta_mean"]) > 1.0


def test_generate_drift_report_renders_markdown(reference_df, current_drifted):
    d = DriftDetector(reference_df)
    rng = np.random.RandomState(0)
    md = d.generate_drift_report(
        current_data=current_drifted,
        reference_scores=pd.Series(rng.beta(2, 5, size=500)),
        current_scores=pd.Series(rng.beta(5, 2, size=500)),
        baseline_auc=0.80,
        current_auc=0.70,
    )
    assert isinstance(md, str)
    assert "# Drift Report" in md
    assert "## 1. Feature drift" in md
    assert "## 2. Prediction drift" in md
    assert "## 3. Concept drift" in md
