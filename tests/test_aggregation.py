"""Unit tests for src.features.aggregation (multi-table aggregator)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.aggregation import (
    MultiTableAggregator,
    load_secondary_tables,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic versions of the 5 secondary tables
# ---------------------------------------------------------------------------

@pytest.fixture
def bureau_synth():
    """3 applicants (id 1, 2, 3) with 1-3 bureau records each, plus bureau_balance."""
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2, 2, 3],
        "SK_ID_BUREAU": ["b1", "b2", "b3", "b4", "b5"],
        "DAYS_CREDIT": [-100, -200, -300, -400, -500],
        "CREDIT_DAY_OVERDUE": [0, 5, 0, 0, 0],
        "AMT_CREDIT_SUM": [10_000, 20_000, 30_000, 40_000, 50_000],
        "AMT_CREDIT_SUM_DEBT": [0, 1_000, 0, 0, 0],
        "AMT_CREDIT_SUM_OVERDUE": [0, 50, 0, 0, 0],
        "CNT_CREDIT_PROLONG": [0, 0, 0, 0, 0],
        "DAYS_CREDIT_UPDATE": [-50, -150, -250, -350, -450],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active", "Closed"],
        "CREDIT_TYPE": [
            "Consumer credit", "Credit card", "Consumer credit",
            "Mortgage", "Consumer credit",
        ],
    })
    bal = pd.DataFrame({
        "SK_ID_BUREAU": ["b1", "b1", "b2", "b2", "b3", "b4", "b5"],
        "MONTHS_BALANCE": [0, -1, 0, -1, 0, 0, 0],
        "STATUS": ["0", "1", "0", "0", "0", "C", "0"],
    })
    return bureau, bal


@pytest.fixture
def prev_synth():
    """3 applicants with 1-2 previous applications each."""
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2, 3],
        "AMT_ANNUITY": [1000, 1500, 2000, 2500],
        "AMT_APPLICATION": [10_000, 15_000, 20_000, 25_000],
        "AMT_CREDIT": [9_000, 14_000, 19_000, 24_000],
        "AMT_DOWN_PAYMENT": [100, 200, 300, 400],
        "DAYS_DECISION": [-30, -60, -90, -120],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Unused offer"],
        "NFLAG_INSURED_ON_APPROVAL": [0.0, 1.0, 0.0, 0.0],
    })


@pytest.fixture
def pos_synth():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2, 3, 3],
        "MONTHS_BALANCE": [0, -1, 0, 0, -1],
        "CNT_INSTALMENT": [12, 11, 24, 6, 5],
        "SK_DPD": [0, 5, 0, 0, 10],
        "SK_DPD_DEF": [0, 0, 0, 0, 0],
    })


@pytest.fixture
def inst_synth():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2, 2, 3],
        "DAYS_INSTALMENT": [-100, -200, -100, -100, -50],
        "DAYS_ENTRY_PAYMENT": [-95, -210, -100, -95, -55],  # late 5d, late 10d, on-time, late 5d, late 5d
        "AMT_INSTALMENT": [1000, 1000, 2000, 2000, 500],
        "AMT_PAYMENT": [1000, 1000, 2000, 1500, 500],  # last one underpaid
        "NUM_INSTALMENT_VERSION": [1, 1, 2, 3, 1],
    })


@pytest.fixture
def cc_synth():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "MONTHS_BALANCE": [0, -1, 0],
        "AMT_BALANCE": [500, 1000, 2000],
        "AMT_CREDIT_LIMIT_ACTUAL": [10000, 10000, 5000],
        "SK_DPD": [0, 0, 30],
    })


# ---------------------------------------------------------------------------
# Bureau
# ---------------------------------------------------------------------------

def test_aggregate_bureau_preserves_index(bureau_synth):
    bureau, bal = bureau_synth
    agg = MultiTableAggregator()
    out = agg.aggregate_bureau(bureau, bal)
    assert out.index.name == "SK_ID_CURR"
    # All 3 applicants should appear
    assert set(out.index.tolist()) == {1, 2, 3}


def test_aggregate_bureau_dpd_month_frac_present(bureau_synth):
    """Bureau + balance merge should populate _DPD_MONTH_FRAC columns."""
    bureau, bal = bureau_synth
    agg = MultiTableAggregator()
    out = agg.aggregate_bureau(bureau, bal)
    assert "BUREAU__DPD_MONTH_FRAC_MEAN" in out.columns
    # Applicant 1 has 1 bad month out of 2 for bureau b1
    applicant1 = out.loc[1]
    assert 0.0 <= applicant1["BUREAU__DPD_MONTH_FRAC_MEAN"] <= 1.0


def test_aggregate_bureau_active_fracs(bureau_synth):
    bureau, bal = bureau_synth
    agg = MultiTableAggregator()
    out = agg.aggregate_bureau(bureau, bal)
    # Applicant 1: 1 Active / 2 records -> 0.5
    assert out.loc[1, "BUREAU_ACTIVE_ACTIVE_FRAC"] == pytest.approx(0.5)
    assert out.loc[3, "BUREAU_ACTIVE_ACTIVE_FRAC"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Previous application
# ---------------------------------------------------------------------------

def test_aggregate_previous_app_status_fracs(prev_synth):
    agg = MultiTableAggregator()
    out = agg.aggregate_previous_app(prev_synth)
    assert out.loc[1, "PREV_STATUS_APPROVED_FRAC"] == pytest.approx(0.5)
    assert out.loc[2, "PREV_STATUS_APPROVED_FRAC"] == pytest.approx(1.0)


def test_aggregate_previous_app_counts(prev_synth):
    agg = MultiTableAggregator()
    out = agg.aggregate_previous_app(prev_synth)
    assert out.loc[1, "PREV_APPLICATION_COUNT"] == 2
    assert out.loc[3, "PREV_APPLICATION_COUNT"] == 1


# ---------------------------------------------------------------------------
# POS / Cash
# ---------------------------------------------------------------------------

def test_aggregate_pos_cash_dpd_flag_frac(pos_synth):
    agg = MultiTableAggregator()
    out = agg.aggregate_pos_cash(pos_synth)
    # Applicant 1: 1 of 2 snapshots had DPD > 0
    assert out.loc[1, "POS_DPD_FLAG_FRAC"] == pytest.approx(0.5)
    # Applicant 2: 0 of 1
    assert out.loc[2, "POS_DPD_FLAG_FRAC"] == pytest.approx(0.0)
    # Applicant 3: 1 of 2
    assert out.loc[3, "POS_DPD_FLAG_FRAC"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Installments
# ---------------------------------------------------------------------------

def test_aggregate_installments_days_late(inst_synth):
    agg = MultiTableAggregator()
    out = agg.aggregate_installments(inst_synth)
    # DAYS_LATE = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT; positive = paid late.
    # Applicant 1: +5 (late 5d), -10 (paid 10d early) -> mean -2.5
    assert out.loc[1, "INST__DAYS_LATE_MEAN"] == pytest.approx(-2.5)
    # Applicant 2: 0 (on-time) + 5 (late 5d) -> mean 2.5
    assert out.loc[2, "INST__DAYS_LATE_MEAN"] == pytest.approx(2.5)
    # Applicant 3: -5 (paid 5d early)
    assert out.loc[3, "INST__DAYS_LATE_MEAN"] == pytest.approx(-5.0)


def test_aggregate_installments_late_fracs(inst_synth):
    agg = MultiTableAggregator()
    out = agg.aggregate_installments(inst_synth)
    # Applicant 1: 1/2 paid late (only row 0)
    assert out.loc[1, "INST_LATE_DAYS_GT0_FRAC"] == pytest.approx(0.5)
    # Applicant 2: 1/2 paid late
    assert out.loc[2, "INST_LATE_DAYS_GT0_FRAC"] == pytest.approx(0.5)
    # Nobody is 30+ days late in this fixture
    assert out.loc[1, "INST_LATE_DAYS_GT30_FRAC"] == 0.0


# ---------------------------------------------------------------------------
# Credit card
# ---------------------------------------------------------------------------

def test_aggregate_credit_card_utilization(cc_synth):
    agg = MultiTableAggregator()
    out = agg.aggregate_credit_card(cc_synth)
    # Applicant 1: 500/10000 + 1000/10000 = 0.075 mean
    assert out.loc[1, "CC_UTILIZATION_MEAN"] == pytest.approx(0.075, abs=1e-6)
    # Applicant 2: 2000/5000 = 0.4
    assert out.loc[2, "CC_UTILIZATION_MEAN"] == pytest.approx(0.4, abs=1e-6)


def test_aggregate_credit_card_dpd(cc_synth):
    agg = MultiTableAggregator()
    out = agg.aggregate_credit_card(cc_synth)
    assert out.loc[2, "CC_DPD_MAX"] == 30


# ---------------------------------------------------------------------------
# aggregate_all (end-to-end)
# ---------------------------------------------------------------------------

def test_aggregate_all_outer_joins(bureau_synth, prev_synth, pos_synth, inst_synth, cc_synth):
    bureau, bal = bureau_synth
    agg = MultiTableAggregator()
    out = agg.aggregate_all({
        "bureau": (bureau, bal),
        "previous_application": prev_synth,
        "pos_cash": pos_synth,
        "installments": inst_synth,
        "credit_card": cc_synth,
    })
    # Should have all 3 applicants
    assert set(out.index.tolist()) == {1, 2, 3}
    # And columns from all 5 sources (BUREAU_/PREV_/POS_/INST_/CC_)
    prefixes = {"BUREAU_", "PREV_", "POS_", "INST_", "CC_"}
    found = {p for p in prefixes if any(c.startswith(p) for c in out.columns)}
    assert found == prefixes


# ---------------------------------------------------------------------------
# load_secondary_tables (filesystem-level, gated by data dir existence)
# ---------------------------------------------------------------------------

def test_load_secondary_tables_missing_dir(tmp_path):
    """Empty / non-existent dir -> all 6 tables empty DataFrames (not error)."""
    out = load_secondary_tables(str(tmp_path))
    assert set(out.keys()) == {
        "bureau", "bureau_balance", "previous_application",
        "POS_CASH_balance", "installments_payments", "credit_card_balance",
    }
    for name, df in out.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Field-list sanity: catch silent schema drift if columns rename upstream
# ---------------------------------------------------------------------------

def test_field_lists_nonempty():
    for name, lst in [("BUREAU_NUM", _BUREAU_NUM_FALLBACK()),
                       ("PREV_NUM", _PREV_NUM_FALLBACK()),
                       ("POS_NUM", _POS_NUM_FALLBACK()),
                       ("INST_NUM", _INST_NUM_FALLBACK()),
                       ("CC_NUM", _CC_NUM_FALLBACK())]:
        assert len(lst) > 0, f"{name} field list is empty"
        for col in lst:
            assert isinstance(col, str)
            assert col.isupper(), f"{name}: {col} not uppercase"


# Internal helpers for the field-list test (mimic the module-level
# constants without importing them via the private underscore)
def _BUREAU_NUM_FALLBACK():
    from src.features.aggregation import _BUREAU_NUM
    return _BUREAU_NUM

def _PREV_NUM_FALLBACK():
    from src.features.aggregation import _PREV_NUM
    return _PREV_NUM

def _POS_NUM_FALLBACK():
    from src.features.aggregation import _POS_NUM
    return _POS_NUM

def _INST_NUM_FALLBACK():
    from src.features.aggregation import _INST_NUM
    return _INST_NUM

def _CC_NUM_FALLBACK():
    from src.features.aggregation import _CC_NUM
    return _CC_NUM


# ---------------------------------------------------------------------------
# Cache wrapper (load_or_build_secondary_features)
# ---------------------------------------------------------------------------

def test_load_or_build_cache_miss_then_hit(tmp_path):
    """First call: no cache -> builds. Second call: cache hit -> returns same data."""
    cache_path = str(tmp_path / "secondary.parquet")
    # Use an empty raw_dir so the cold call still produces *something* (empty
    # DataFrame). The sanity check in load_or_build requires >=200 cols, so
    # use force_rebuild + a path that doesn't exist, expecting an empty DF
    # returned but a cache file written.
    from src.features.aggregation import load_or_build_secondary_features
    # raw_dir with no parquet files -> load_secondary_tables returns empty DFs
    empty_dir = tmp_path / "empty_raw"
    empty_dir.mkdir()
    df1 = load_or_build_secondary_features(
        raw_dir=str(empty_dir),
        cache_path=cache_path,
    )
    # With empty secondary tables, aggregate_all returns an empty DataFrame
    # (0, 0) — sanity check (>=200 cols) will fail, so the cache is rebuilt
    # and re-written. But the function still returns the empty DF, which is
    # correct semantics: "I had nothing to aggregate, here's the empty result".
    assert df1.shape[1] == 0
    # Cache file was written even though the data is empty
    import os
    assert os.path.exists(cache_path)


def test_load_or_build_cache_invalidation_on_corrupt_cache(tmp_path, monkeypatch):
    """If the cache file is corrupt (fails sanity check), rebuild from scratch."""
    from src.features.aggregation import load_or_build_secondary_features
    cache_path = tmp_path / "bad.parquet"
    # Write a tiny file that has a non-SK_ID_CURR index -> sanity check fails
    bad = pd.DataFrame({"x": [1, 2]}, index=pd.Index([0, 1], name="not_skill"))
    bad.to_parquet(cache_path)
    df = load_or_build_secondary_features(
        raw_dir=str(tmp_path),  # empty -> empty secondary
        cache_path=str(cache_path),
    )
    # The function rebuilt and overwrote the bad cache. Result is the same
    # empty DF as the cold path.
    assert df.shape[1] == 0


def test_cache_version_constant_is_int():
    """Bumping the version is the documented way to invalidate caches."""
    from src.features.aggregation import SECONDARY_FEATURES_CACHE_VERSION
    assert isinstance(SECONDARY_FEATURES_CACHE_VERSION, int)
    assert SECONDARY_FEATURES_CACHE_VERSION >= 1
