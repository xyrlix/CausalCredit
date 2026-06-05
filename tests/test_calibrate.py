"""Unit tests for src.models.calibrate (IsotonicCalibrator)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibrate import IsotonicCalibrator


def test_calibrator_is_monotonic():
    """Output should be monotonically non-decreasing in input."""
    rng = np.random.RandomState(0)
    p_raw = rng.uniform(0, 1, size=500)
    # Mildly mis-calibrated truth: y ~ Bern(p^2)
    y_true = (rng.uniform(0, 1, size=500) < p_raw ** 2).astype(int)

    cal = IsotonicCalibrator().fit(p_raw, y_true)
    grid = np.linspace(0, 1, 50)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-9), "Calibrator output is not monotone"


def test_calibrator_output_in_unit_interval():
    rng = np.random.RandomState(0)
    p_raw = rng.uniform(0, 1, size=200)
    y_true = (rng.uniform(0, 1, size=200) < p_raw).astype(int)
    cal = IsotonicCalibrator().fit(p_raw, y_true)
    out = cal.transform(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
    assert np.all(out >= 0) and np.all(out <= 1)


def test_calibrator_improves_well_calibrated_already():
    """If raw probabilities are already calibrated, isotonic should leave them ~unchanged."""
    rng = np.random.RandomState(0)
    p_raw = rng.uniform(0, 1, size=2000)
    y_true = (rng.uniform(0, 1, size=2000) < p_raw).astype(int)
    cal = IsotonicCalibrator().fit(p_raw, y_true)
    out = cal.transform(p_raw)
    mae = np.mean(np.abs(out - p_raw))
    assert mae < 0.1, f"Calibrator distorted well-calibrated input (MAE={mae:.4f})"


def test_fit_transform_chain():
    rng = np.random.RandomState(0)
    p_raw = rng.uniform(0, 1, size=200)
    y_true = (rng.uniform(0, 1, size=200) < p_raw).astype(int)
    cal = IsotonicCalibrator()
    out = cal.fit_transform(p_raw, y_true)
    assert out.shape == p_raw.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_transform_before_fit_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="fit"):
        cal.transform(np.array([0.1, 0.5, 0.9]))


def test_calibrator_clips_out_of_range_inputs():
    """Inputs outside [0,1] should be handled (clipped) without crashing."""
    rng = np.random.RandomState(0)
    p_raw = rng.uniform(0, 1, size=200)
    y_true = (rng.uniform(0, 1, size=200) < p_raw).astype(int)
    cal = IsotonicCalibrator().fit(p_raw, y_true)
    out = cal.transform(np.array([-0.5, 0.5, 1.5]))
    assert np.all(out >= 0) and np.all(out <= 1)
