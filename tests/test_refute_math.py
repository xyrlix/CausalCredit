"""Unit tests for src.causal.refute (E-value math + helpers)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.causal.refute import compute_e_value_from_ate


def test_e_value_zero_ate_is_one():
    """No effect → no unobserved confounder needed → E ≈ 1."""
    e = compute_e_value_from_ate(ate=0.0, sd_y=1.0)
    assert e == pytest.approx(1.0, abs=1e-6)


def test_e_value_increases_with_ate():
    """Larger |ATE| should require a stronger confounder (larger E)."""
    e_small = compute_e_value_from_ate(ate=0.01, sd_y=1.0)
    e_medium = compute_e_value_from_ate(ate=0.10, sd_y=1.0)
    e_large = compute_e_value_from_ate(ate=0.50, sd_y=1.0)
    assert e_small < e_medium < e_large


def test_e_value_symmetric_in_sign():
    """E-value should depend on |ATE|, not sign."""
    e_pos = compute_e_value_from_ate(ate=0.20, sd_y=1.0)
    e_neg = compute_e_value_from_ate(ate=-0.20, sd_y=1.0)
    assert e_pos == pytest.approx(e_neg, abs=1e-6)


def test_e_value_acceptance_threshold():
    """ATE that yields E >= 2.0 is the acceptance criterion."""
    # An ATE that produces E ≥ 2 should pass; one that produces E < 2 should fail
    # Vanderweele: RR = exp(0.91*|ATE|); RR>=1.5 gives E≈2.36
    ate_passing = 0.50
    ate_failing = 0.01
    assert compute_e_value_from_ate(ate_passing, sd_y=1.0) >= 2.0
    assert compute_e_value_from_ate(ate_failing, sd_y=1.0) < 2.0


def test_e_value_formula_matches_vanderweele():
    """Check the actual formula: E = RR + sqrt(RR*(RR-1)) where RR = exp(0.91*|ATE|)."""
    ate = 0.30
    rr = math.exp(0.91 * abs(ate))
    expected_e = rr + math.sqrt(rr * (rr - 1))
    actual_e = compute_e_value_from_ate(ate=ate, sd_y=1.0)
    assert actual_e == pytest.approx(expected_e, abs=1e-4)


def test_e_value_finite_for_reasonable_inputs():
    for ate in [-1.0, -0.5, 0.0, 0.1, 0.5, 1.0]:
        for sd in [0.5, 1.0, 2.0]:
            e = compute_e_value_from_ate(ate=ate, sd_y=sd)
            assert np.isfinite(e), f"E-value not finite for ate={ate}, sd_y={sd}"
            assert e >= 1.0, f"E-value should be >= 1, got {e}"
