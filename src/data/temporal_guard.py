"""Temporal data-leakage guard for the Home Credit pipeline.

@requirement REQ-DATA-LEAK-001
@design  inspired by tmp/CausalCredit/src/data/temporal_guard.py
@see     docs/CausalCredit_因果推理验证标准体系.md §4.1 (内生性 → 时间泄漏)

The Home Credit secondary tables (``POS_CASH_balance``, ``credit_card_balance``)
encode time relative to the application via ``MONTHS_BALANCE``:

* ``0``       — the application month (allowed),
* ``-N``      — N months **before** the application (allowed, historical),
* ``+N``      — N months **after** the application (**FORBIDDEN** — this is
  the actual loan-servicing history that we would not have at decision time).

Rows with ``MONTHS_BALANCE > 0`` are post-application servicing records.  If
they leak into the per-applicant aggregation, the model effectively peeks at
the answer.  This guard detects and removes them before aggregation, and
emits a structured issue list for the pipeline log.

It also exposes a ``validate_split`` helper for time-ordered train/test
splits and a ``check_split_overlap`` for detecting train/test date overlap
in datasets where an explicit date column is present (Lending Club ``issue_d``,
or any custom timestamp).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("causalcredit.data.temporal_guard")

# Columns that, by Home Credit convention, encode time relative to the
# application month. Anything > 0 is post-application (future info).
_TEMPORAL_COLS = ("MONTHS_BALANCE",)


@dataclass
class TemporalIssue:
    """A single temporal-leakage finding."""
    type: str           # MONTHS_BALANCE_LEAK / TEMPORAL_OVERLAP / DATE_GAP
    table: str          # secondary table name
    count: int          # rows affected
    ratio: float        # rows / total
    action: str         # EXCLUDED / WARNING
    detail: str = ""    # human-readable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "table": self.table,
            "count": int(self.count),
            "ratio": float(self.ratio),
            "action": self.action,
            "detail": self.detail,
        }


@dataclass
class TemporalGuardReport:
    """Aggregate result for an entire validation run."""
    issues: List[TemporalIssue] = field(default_factory=list)
    rows_removed: Dict[str, int] = field(default_factory=dict)
    rows_kept: Dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(i.action != "WARNING" for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "rows_removed": dict(self.rows_removed),
            "rows_kept": dict(self.rows_kept),
        }


class TemporalGuard:
    """Detect and remove post-application rows from Home Credit secondary tables.

    Usage::

        guard = TemporalGuard()
        cleaned, report = guard.scrub_secondary_tables({
            "pos":  pos_df,
            "cc":   cc_df,
            "inst": inst_df,
        })
        for issue in report.issues:
            log.warning("temporal-leak: %s", issue.to_dict())
    """

    def __init__(self, *, temporal_cols: tuple = _TEMPORAL_COLS):
        self._temporal_cols = tuple(temporal_cols)

    # ------------------------------------------------------------------
    # Secondary-table scrubbing (the main use case)
    # ------------------------------------------------------------------
    def scrub_secondary_tables(
        self,
        tables: Dict[str, pd.DataFrame],
    ) -> tuple:
        """Drop rows with ``MONTHS_BALANCE > 0`` from each secondary table.

        Returns ``(cleaned_tables, TemporalGuardReport)``.

        Tables without a temporal column pass through untouched.
        """
        report = TemporalGuardReport()
        cleaned: Dict[str, pd.DataFrame] = {}
        for name, df in tables.items():
            if df is None or len(df) == 0:
                cleaned[name] = df
                continue

            temporal_col = next(
                (c for c in self._temporal_cols if c in df.columns), None
            )
            if temporal_col is None:
                # No temporal info → nothing to scrub for this table.
                cleaned[name] = df
                report.rows_kept[name] = int(len(df))
                continue

            mask_future = df[temporal_col] > 0
            n_future = int(mask_future.sum())
            n_total = int(len(df))
            if n_future > 0:
                ratio = n_future / max(n_total, 1)
                report.issues.append(TemporalIssue(
                    type="MONTHS_BALANCE_LEAK",
                    table=name,
                    count=n_future,
                    ratio=ratio,
                    action="EXCLUDED",
                    detail=(
                        f"{n_future}/{n_total} ({ratio:.3%}) rows have "
                        f"{temporal_col} > 0 (post-application); excluded "
                        f"before aggregation"
                    ),
                ))
                cleaned[name] = df.loc[~mask_future].copy()
                report.rows_removed[name] = n_future
                report.rows_kept[name] = n_total - n_future
                logger.warning(
                    "temporal-leak removed: table=%s rows=%d/%d (%.3f%%)",
                    name, n_future, n_total, 100 * ratio,
                )
            else:
                cleaned[name] = df
                report.rows_removed[name] = 0
                report.rows_kept[name] = n_total
        return cleaned, report

    # ------------------------------------------------------------------
    # Date-column helpers (time-ordered splits, e.g. Lending Club issue_d)
    # ------------------------------------------------------------------
    @staticmethod
    def validate_split(
        df: pd.DataFrame,
        date_col: str,
        *,
        train_ratio: float = 0.8,
        split_col: str = "split",
    ) -> pd.DataFrame:
        """Time-ordered train/test split.

        Sort by ``date_col`` ascending, assign the first ``train_ratio``
        fraction to ``"train"`` and the remainder to ``"test"``.

        Returns a new DataFrame with an additional ``split_col``.
        """
        if date_col not in df.columns:
            raise ValueError(f"date column {date_col!r} not in DataFrame")
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(
                f"train_ratio must be in (0, 1); got {train_ratio}"
            )
        sorted_df = df.sort_values(date_col).reset_index(drop=True)
        n_train = int(len(sorted_df) * train_ratio)
        sorted_df[split_col] = "test"
        sorted_df.loc[: n_train - 1, split_col] = "train"
        return sorted_df

    @staticmethod
    def check_split_overlap(
        df: pd.DataFrame,
        date_col: str,
        *,
        split_col: str = "split",
        train_label: str = "train",
        test_label: str = "test",
    ) -> Optional[TemporalIssue]:
        """Return a TEMPORAL_OVERLAP issue if any train row is on/after the
        earliest test row by date. Returns ``None`` if the split is clean.
        """
        if date_col not in df.columns or split_col not in df.columns:
            return None
        train_dates = df.loc[df[split_col] == train_label, date_col]
        test_dates = df.loc[df[split_col] == test_label, date_col]
        if len(train_dates) == 0 or len(test_dates) == 0:
            return None
        train_max = train_dates.max()
        test_min = test_dates.min()
        if train_max >= test_min:
            return TemporalIssue(
                type="TEMPORAL_OVERLAP",
                table="<split>",
                count=int((train_dates >= test_min).sum()),
                ratio=float((train_dates >= test_min).mean()),
                action="WARNING",
                detail=(
                    f"train max date {train_max!s} >= test min date "
                    f"{test_min!s} — leakage risk"
                ),
            )
        return None
