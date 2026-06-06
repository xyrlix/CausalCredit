"""Tests for FraudGuardConfig and configurable routing thresholds."""

from __future__ import annotations

import pytest

from src.fraud.pipeline import FraudGuard, FraudGuardConfig


def test_fraud_guard_config_defaults_match_pipeline():
    cfg = FraudGuardConfig()
    assert cfg.fraud_reject_threshold == pytest.approx(0.10)
    assert cfg.packaging_reject_threshold == pytest.approx(0.50)
    assert cfg.fraud_borderline_threshold == pytest.approx(0.05)
    assert cfg.packaging_borderline_threshold == pytest.approx(0.30)


def test_fraud_guard_config_from_dict():
    cfg = FraudGuardConfig.from_dict({
        "fraud_reject_threshold": 0.05,
        "packaging_reject_threshold": 0.40,
        "irrelevant_key": "ignored",
    })
    assert cfg.fraud_reject_threshold == pytest.approx(0.05)
    assert cfg.packaging_reject_threshold == pytest.approx(0.40)
    assert cfg.fraud_borderline_threshold == pytest.approx(0.05)  # default


def test_fraud_guard_config_from_empty_dict_returns_defaults():
    cfg = FraudGuardConfig.from_dict(None)
    assert cfg.fraud_reject_threshold == pytest.approx(0.10)
    cfg = FraudGuardConfig.from_dict({})
    assert cfg.packaging_reject_threshold == pytest.approx(0.50)


def test_fraud_routing_threshold_override_changes_routing():
    from src.fraud.pipeline import _fraud_routing
    cfg_tight = FraudGuardConfig(fraud_reject_threshold=0.03)  # tighter than default
    cfg_loose = FraudGuardConfig(
        fraud_reject_threshold=0.50, fraud_borderline_threshold=0.10
    )  # looser on both, so 0.05 is below both bands

    # fraud_score=0.05 is rejected (0.05 ≥ 0.03) under tight, but clean under loose
    assert _fraud_routing(0.05, 0.0, "OK", config=cfg_tight) == "REJECT_FRAUD"
    assert _fraud_routing(0.05, 0.0, "OK", config=cfg_loose) == "PROCEED"


def test_packaging_routing_threshold_override():
    from src.fraud.pipeline import _fraud_routing
    cfg = FraudGuardConfig(packaging_reject_threshold=0.30)
    assert _fraud_routing(0.0, 0.35, "OK", config=cfg) == "REJECT_PACKAGING"


def test_consistency_flag_threshold_in_routing_reasons():
    from src.fraud.pipeline import _routing_reasons
    cfg = FraudGuardConfig(consistency_flag_threshold=0.7)
    reasons = _routing_reasons(0.0, 0.0, consistency=0.6, config=cfg)
    assert any("0.60" in r and "0.70" in r for r in reasons), reasons


def test_fraud_guard_uses_config_in_init():
    cfg = FraudGuardConfig(fraud_reject_threshold=0.07)
    guard = FraudGuard(config=cfg)
    assert guard.config.fraud_reject_threshold == pytest.approx(0.07)
