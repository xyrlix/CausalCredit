"""Unit tests for src.data.temporal_guard (M8.6a).

Covers:

* MONTHS_BALANCE > 0 row removal (one issue per affected table)
* No-op behaviour when temporal column is absent or all rows are <= 0
* validate_split produces a contiguous time-ordered split
* check_split_overlap flags overlap, passes on clean split
* Edge cases — empty DataFrames, mixed empty + populated dict
* Report serializes cleanly + .passed property
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.temporal_guard import TemporalGuard, TemporalIssue, TemporalGuardReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def pos_table_with_future():
    """A POS_CASH_balance fragment with 3 of 10 rows post-application."""
    return pd.DataFrame({
        "SK_ID_CURR": list(range(10)),
        "MONTHS_BALANCE": [-12, -10, -8, -5, -3, -1, 0, 1, 2, 3],
        "CNT_INSTALMENT": [24] * 10,
        "SK_DPD": [0] * 10,
    })


@pytest.fixture
def cc_table_clean():
    """A credit_card_balance fragment that is fully historical."""
    return pd.DataFrame({
        "SK_ID_CURR": list(range(5)),
        "MONTHS_BALANCE": [-24, -12, -6, -1, 0],
        "AMT_BALANCE": [100.0, 200.0, 150.0, 50.0, 80.0],
    })


@pytest.fixture
def bureau_table_no_temporal():
    """A bureau table without MONTHS_BALANCE — should be passed through."""
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "DAYS_CREDIT": [-365, -730, -100],
        "AMT_CREDIT_SUM": [1e5, 2e5, 5e4],
    })


# ---------------------------------------------------------------------------
# scrub_secondary_tables
# ---------------------------------------------------------------------------
class TestScrubSecondaryTables:
    def test_removes_future_rows(self, pos_table_with_future):
        guard = TemporalGuard()
        cleaned, report = guard.scrub_secondary_tables({"pos": pos_table_with_future})
        # 3 rows had MONTHS_BALANCE > 0 → should be removed
        assert len(cleaned["pos"]) == 7
        assert (cleaned["pos"]["MONTHS_BALANCE"] <= 0).all()
        assert report.rows_removed["pos"] == 3
        assert report.rows_kept["pos"] == 7

    def test_emits_one_issue_per_affected_table(self, pos_table_with_future, cc_table_clean):
        guard = TemporalGuard()
        _, report = guard.scrub_secondary_tables(
            {"pos": pos_table_with_future, "cc": cc_table_clean}
        )
        # Only pos had leakage; cc was clean
        types = [i.type for i in report.issues]
        assert types == ["MONTHS_BALANCE_LEAK"]
        assert report.issues[0].table == "pos"
        assert report.issues[0].count == 3

    def test_passes_through_table_without_temporal_col(self, bureau_table_no_temporal):
        guard = TemporalGuard()
        cleaned, report = guard.scrub_secondary_tables({"bureau": bureau_table_no_temporal})
        # Identical DataFrame back out
        pd.testing.assert_frame_equal(cleaned["bureau"], bureau_table_no_temporal)
        # No issue emitted
        assert report.issues == []
        # But it IS counted in rows_kept
        assert report.rows_kept["bureau"] == 3

    def test_empty_dataframe_no_op(self):
        guard = TemporalGuard()
        empty = pd.DataFrame(columns=["MONTHS_BALANCE"])
        cleaned, report = guard.scrub_secondary_tables({"pos": empty})
        assert len(cleaned["pos"]) == 0
        assert report.issues == []

    def test_all_clean_emits_no_issues(self, cc_table_clean):
        guard = TemporalGuard()
        _, report = guard.scrub_secondary_tables({"cc": cc_table_clean})
        assert report.issues == []
        assert report.rows_removed["cc"] == 0
        assert report.passed

    def test_report_ratio_is_fraction(self, pos_table_with_future):
        guard = TemporalGuard()
        _, report = guard.scrub_secondary_tables({"pos": pos_table_with_future})
        assert report.issues[0].ratio == pytest.approx(0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# validate_split
# ---------------------------------------------------------------------------
class TestValidateSplit:
    def test_assigns_split_in_time_order(self):
        df = pd.DataFrame({
            "issue_d": pd.to_datetime(
                ["2024-01-01", "2024-06-01", "2024-12-01", "2025-03-01", "2025-09-01"]
            ),
            "x": [1, 2, 3, 4, 5],
        })
        out = TemporalGuard.validate_split(df, "issue_d", train_ratio=0.6)
        # 3 train, 2 test
        assert (out["split"] == "train").sum() == 3
        assert (out["split"] == "test").sum() == 2
        # All train dates < all test dates
        train_max = out.loc[out["split"] == "train", "issue_d"].max()
        test_min = out.loc[out["split"] == "test", "issue_d"].min()
        assert train_max < test_min

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="date column"):
            TemporalGuard.validate_split(df, "issue_d")

    def test_raises_on_bad_ratio(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"])})
        with pytest.raises(ValueError, match="train_ratio"):
            TemporalGuard.validate_split(df, "d", train_ratio=1.5)


# ---------------------------------------------------------------------------
# check_split_overlap
# ---------------------------------------------------------------------------
class TestCheckSplitOverlap:
    def test_returns_none_for_clean_split(self):
        df = pd.DataFrame({
            "d": pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01", "2025-06-01"]),
            "split": ["train", "train", "test", "test"],
        })
        assert TemporalGuard.check_split_overlap(df, "d") is None

    def test_flags_overlap(self):
        df = pd.DataFrame({
            "d": pd.to_datetime(["2024-01-01", "2024-09-01", "2024-06-01", "2025-06-01"]),
            "split": ["train", "train", "test", "test"],
        })
        issue = TemporalGuard.check_split_overlap(df, "d")
        assert issue is not None
        assert issue.type == "TEMPORAL_OVERLAP"
        assert issue.action == "WARNING"
        assert "leakage risk" in issue.detail

    def test_returns_none_if_one_side_empty(self):
        df = pd.DataFrame({
            "d": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "split": ["train", "train"],
        })
        assert TemporalGuard.check_split_overlap(df, "d") is None


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------
class TestReportSerialization:
    def test_to_dict_roundtrip(self, pos_table_with_future):
        guard = TemporalGuard()
        _, report = guard.scrub_secondary_tables({"pos": pos_table_with_future})
        d = report.to_dict()
        assert d["passed"] is True  # MONTHS_BALANCE_LEAK is EXCLUDED, not WARNING
        assert isinstance(d["issues"], list)
        assert d["issues"][0]["type"] == "MONTHS_BALANCE_LEAK"
        assert d["rows_removed"]["pos"] == 3

    def test_passed_false_when_warning_present(self):
        report = TemporalGuardReport()
        report.issues.append(TemporalIssue(
            type="TEMPORAL_OVERLAP", table="<split>",
            count=10, ratio=0.05, action="WARNING",
            detail="train max >= test min",
        ))
        assert report.passed is False
