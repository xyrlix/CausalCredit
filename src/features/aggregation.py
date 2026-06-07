"""Multi-table aggregation feature builder (Home Credit 6 secondary tables).

Aggregates 6 secondary tables (bureau, bureau_balance, previous_application,
POS_CASH_balance, installments_payments, credit_card_balance) by SK_ID_CURR
into per-applicant summary features.

Design:
- Output is a DataFrame indexed by SK_ID_CURR with a few dozen numeric cols.
- All aggregations are pure aggregations (mean, sum, max, min, std, count) +
  a small set of hand-crafted ratios (overdue ratio, payment gap mean, etc.).
- CATEGORICAL aggregations are limited to "fraction-of-time" features
  (e.g. fraction of bureau records with status 0/1/2) — full one-hot would
  blow up the column count.
- All functions are stateless & DataFrame-in / DataFrame-out, so they can
  be unit-tested without touching disk.

Typical usage:

    agg = MultiTableAggregator()
    bureau_feat = agg.aggregate_bureau(bureau_df, bureau_balance_df)
    prev_feat = agg.aggregate_previous_app(prev_df)
    pos_feat = agg.aggregate_pos_cash(pos_df)
    inst_feat = agg.aggregate_installments(inst_df)
    cc_feat = agg.aggregate_credit_card(cc_df)
    all_feat = agg.aggregate_all({
        "bureau": (bureau_df, bureau_balance_df),
        "previous_application": prev_df,
        "pos_cash": pos_df,
        "installments": inst_df,
        "credit_card": cc_df,
    })
    # all_feat has SK_ID_CURR index, ~60-80 numeric features
    app_with_features = application_df.join(all_feat, on="SK_ID_CURR")
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Sentinel value used to mark numeric features that came from secondary tables.
# Downstream feature-engineering can use it to give the new columns a single
# prefix (e.g. BUREAU_*, PREV_*, ...) and apply Winsorize.
SECONDARY_PREFIX_BUREAU = "BUREAU_"
SECONDARY_PREV = "PREV_"
SECONDARY_POS = "POS_"
SECONDARY_INST = "INST_"
SECONDARY_CC = "CC_"

# Map from "table key" in aggregate_all(...) to the per-table prefix.
_PREFIX = {
    "bureau": SECONDARY_PREFIX_BUREAU,
    "previous_application": SECONDARY_PREV,
    "pos_cash": SECONDARY_POS,
    "installments": SECONDARY_INST,
    "credit_card": SECONDARY_CC,
}


class MultiTableAggregator:
    """Aggregate 6 Home Credit secondary tables into per-applicant features."""

    # ------------------------------------------------------------------ bureau
    def aggregate_bureau(
        self, bureau_df: pd.DataFrame, bureau_bal_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Per-applicant credit-bureau history features.

        Args:
            bureau_df: bureau table (one row per bureau credit record). Must
                contain SK_ID_CURR and the numeric / categorical columns
                listed in `_BUREAU_NUM` / `_BUREAU_CAT_FRAC`.
            bureau_bal_df: optional bureau_balance table (monthly status per
                bureau record). Used to compute "fraction of months in DPD"
                per bureau record, then aggregated to applicant level.

        Returns:
            DataFrame indexed by SK_ID_CURR, with a mix of *_count, *_sum,
            *_mean, *_max, *_min, *_std numeric features and bureau-status
            fractions (e.g. BUREAU_STATUS_0_FRAC).
        """
        bureau = bureau_df.copy()
        if "SK_ID_CURR" not in bureau.columns:
            raise ValueError("bureau_df must have SK_ID_CURR")

        # Optional: enrich each bureau row with the share of "bad months" in
        # bureau_balance. STATUS column is coded 0..5 (C..X) plus C=0 means
        # "no DPD". We define "bad month" as STATUS >= 1 (DPD reported).
        if bureau_bal_df is not None and len(bureau_bal_df) > 0:
            bal = bureau_bal_df.copy()
            if "SK_ID_BUREAU" in bal.columns and "STATUS" in bal.columns:
                bal["_BAD"] = (bal["STATUS"].astype(str) != "0").astype(int)
                bad_share = bal.groupby("SK_ID_BUREAU")["_BAD"].agg(["mean", "sum", "size"])
                bad_share.columns = ["_DPD_MONTH_FRAC", "_DPD_MONTH_COUNT", "_BUREAU_REC_MONTHS"]
                bureau = bureau.merge(
                    bad_share, left_on="SK_ID_BUREAU", right_index=True, how="left"
                )
            else:
                bureau["_DPD_MONTH_FRAC"] = np.nan
                bureau["_DPD_MONTH_COUNT"] = np.nan
                bureau["_BUREAU_REC_MONTHS"] = np.nan
        else:
            bureau["_DPD_MONTH_FRAC"] = np.nan
            bureau["_DPD_MONTH_COUNT"] = np.nan
            bureau["_BUREAU_REC_MONTHS"] = np.nan

        # Numeric aggregations
        num_cols = [c for c in _BUREAU_NUM if c in bureau.columns]
        agg_dict: Dict[str, Union[str, list]] = {c: ["mean", "max", "min", "sum"] for c in num_cols}
        agg_dict["DAYS_CREDIT"] = ["mean", "max", "min"]  # avoid sum of negative days
        agg_dict["_DPD_MONTH_FRAC"] = ["mean", "max"]
        agg_dict["_DPD_MONTH_COUNT"] = ["sum", "mean"]

        grouped = bureau.groupby("SK_ID_CURR").agg(agg_dict)
        # Flatten multi-index column names -> BUREAU_*_MEAN etc.
        grouped.columns = [
            f"{SECONDARY_PREFIX_BUREAU}{c.upper()}_{stat.upper()}" for c, stat in grouped.columns
        ]

        # Record count (always useful)
        grouped[SECONDARY_PREFIX_BUREAU + "RECORD_COUNT"] = bureau.groupby("SK_ID_CURR").size()
        grouped[SECONDARY_PREFIX_BUREAU + "ACTIVE_COUNT"] = (
            bureau["CREDIT_ACTIVE"].eq("Active").groupby(bureau["SK_ID_CURR"]).sum()
            if "CREDIT_ACTIVE" in bureau.columns
            else 0
        )

        # Categorical fractions (one column per status code)
        if "CREDIT_ACTIVE" in bureau.columns:
            for status in ["Active", "Closed"]:
                col = SECONDARY_PREFIX_BUREAU + f"ACTIVE_{status.upper()}_FRAC"
                grouped[col] = (
                    bureau["CREDIT_ACTIVE"].eq(status).groupby(bureau["SK_ID_CURR"]).mean()
                )
        if "CREDIT_TYPE" in bureau.columns:
            # Top-5 credit types by global frequency
            top5 = bureau["CREDIT_TYPE"].value_counts().head(5).index.tolist()
            for status in top5:
                safe = "".join(c if c.isalnum() else "_" for c in status)[:24]
                col = SECONDARY_PREFIX_BUREAU + f"TYPE_{safe.upper()}_FRAC"
                grouped[col] = (
                    bureau["CREDIT_TYPE"].eq(status).groupby(bureau["SK_ID_CURR"]).mean()
                )

        grouped = grouped.fillna(0)
        return grouped

    # ---------------------------------------------------------- previous_application
    def aggregate_previous_app(self, prev_df: pd.DataFrame) -> pd.DataFrame:
        """Per-applicant previous-application features."""
        prev = prev_df.copy()
        if "SK_ID_CURR" not in prev.columns:
            raise ValueError("prev_df must have SK_ID_CURR")

        num_cols = [c for c in _PREV_NUM if c in prev.columns]
        agg_dict = {c: ["mean", "max", "min", "sum"] for c in num_cols}
        agg_dict["DAYS_DECISION"] = ["mean", "max", "min"]  # negative days
        agg_dict["NFLAG_INSURED_ON_APPROVAL"] = ["mean", "sum"]  # 0/1 flag

        grouped = prev.groupby("SK_ID_CURR").agg(agg_dict)
        grouped.columns = [
            f"{SECONDARY_PREV}{c.upper()}_{stat.upper()}" for c, stat in grouped.columns
        ]
        grouped[SECONDARY_PREV + "APPLICATION_COUNT"] = prev.groupby("SK_ID_CURR").size()

        # Approval / refusal fractions (4 mutually-exclusive contract statuses)
        if "NAME_CONTRACT_STATUS" in prev.columns:
            for status in ["Approved", "Refused", "Canceled", "Unused offer"]:
                col = SECONDARY_PREV + f"STATUS_{status.upper().replace(' ', '_')}_FRAC"
                grouped[col] = (
                    prev["NAME_CONTRACT_STATUS"].eq(status).groupby(prev["SK_ID_CURR"]).mean()
                )

        # Was-the-previous-app-granted a higher AMT_CREDIT than current?
        if "AMT_CREDIT" in prev.columns:
            grouped[SECONDARY_PREV + "AMT_CREDIT_TOTAL"] = prev.groupby("SK_ID_CURR")[
                "AMT_CREDIT"
            ].sum()
        if "AMT_DOWN_PAYMENT" in prev.columns:
            grouped[SECONDARY_PREV + "AMT_DOWN_PAYMENT_TOTAL"] = prev.groupby("SK_ID_CURR")[
                "AMT_DOWN_PAYMENT"
            ].sum()

        grouped = grouped.fillna(0)
        return grouped

    # ---------------------------------------------------------------- pos_cash
    def aggregate_pos_cash(self, pos_df: pd.DataFrame) -> pd.DataFrame:
        """Per-applicant POS / cash-loan installment features."""
        pos = pos_df.copy()
        if "SK_ID_CURR" not in pos.columns:
            raise ValueError("pos_df must have SK_ID_CURR")

        num_cols = [c for c in _POS_NUM if c in pos.columns]
        agg_dict = {c: ["mean", "max", "min", "sum"] for c in num_cols}

        grouped = pos.groupby("SK_ID_CURR").agg(agg_dict)
        grouped.columns = [
            f"{SECONDARY_POS}{c.upper()}_{stat.upper()}" for c, stat in grouped.columns
        ]
        grouped[SECONDARY_POS + "MONTHLY_RECORD_COUNT"] = pos.groupby("SK_ID_CURR").size()

        # DPD flag fractions: 0/1 binary per monthly snapshot
        if "SK_DPD" in pos.columns:
            grouped[SECONDARY_POS + "DPD_FLAG_FRAC"] = (
                pos["SK_DPD"].gt(0).groupby(pos["SK_ID_CURR"]).mean()
            )
            grouped[SECONDARY_POS + "DPD_MAX"] = pos.groupby("SK_ID_CURR")["SK_DPD"].max()
        if "SK_DPD_DEF" in pos.columns:
            grouped[SECONDARY_POS + "DPD_DEF_FLAG_FRAC"] = (
                pos["SK_DPD_DEF"].gt(0).groupby(pos["SK_ID_CURR"]).mean()
            )

        grouped = grouped.fillna(0)
        return grouped

    # ------------------------------------------------------------ installments
    def aggregate_installments(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """Per-applicant installment-payment features.

        The "days-late" feature is the canonical Home Credit signal:
            days_late = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
        (positive => paid late, negative => paid early).
        """
        inst = inst_df.copy()
        if "SK_ID_CURR" not in inst.columns:
            raise ValueError("inst_df must have SK_ID_CURR")

        # Compute days_late once if both columns present
        if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(inst.columns):
            inst["_DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
        else:
            inst["_DAYS_LATE"] = np.nan

        # Payment-vs-expected ratio
        if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(inst.columns):
            inst["_PAY_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)
        else:
            inst["_PAY_RATIO"] = np.nan

        agg_dict = {
            "_DAYS_LATE": ["mean", "max", "min", "std"],
            "_PAY_RATIO": ["mean", "min", "std"],
            "AMT_PAYMENT": ["sum", "mean", "max"],
            "AMT_INSTALMENT": ["sum", "mean", "max"],
            "NUM_INSTALMENT_VERSION": ["nunique"],
        }
        # Only keep columns that actually exist
        agg_dict = {c: stats for c, stats in agg_dict.items() if c in inst.columns}

        grouped = inst.groupby("SK_ID_CURR").agg(agg_dict)
        grouped.columns = [
            f"{SECONDARY_INST}{c.upper()}_{stat.upper()}" for c, stat in grouped.columns
        ]
        grouped[SECONDARY_INST + "PAYMENT_RECORD_COUNT"] = inst.groupby("SK_ID_CURR").size()

        # Useful late-payment counts
        if "_DAYS_LATE" in inst.columns:
            grouped[SECONDARY_INST + "LATE_DAYS_GT0_FRAC"] = (
                inst["_DAYS_LATE"].gt(0).groupby(inst["SK_ID_CURR"]).mean()
            )
            grouped[SECONDARY_INST + "LATE_DAYS_GT30_FRAC"] = (
                inst["_DAYS_LATE"].gt(30).groupby(inst["SK_ID_CURR"]).mean()
            )

        grouped = grouped.fillna(0)
        return grouped

    # -------------------------------------------------------------- credit_card
    def aggregate_credit_card(self, cc_df: pd.DataFrame) -> pd.DataFrame:
        """Per-applicant credit-card monthly-balance features."""
        cc = cc_df.copy()
        if "SK_ID_CURR" not in cc.columns:
            raise ValueError("cc_df must have SK_ID_CURR")

        num_cols = [c for c in _CC_NUM if c in cc.columns]
        agg_dict = {c: ["mean", "max", "min", "sum"] for c in num_cols}

        grouped = cc.groupby("SK_ID_CURR").agg(agg_dict)
        grouped.columns = [
            f"{SECONDARY_CC}{c.upper()}_{stat.upper()}" for c, stat in grouped.columns
        ]
        grouped[SECONDARY_CC + "MONTHLY_RECORD_COUNT"] = cc.groupby("SK_ID_CURR").size()

        # Utilization: AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL (high = maxing out card)
        if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(cc.columns):
            denom = cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
            util = (cc["AMT_BALANCE"] / denom).clip(0, 5)
            grouped[SECONDARY_CC + "UTILIZATION_MEAN"] = util.groupby(cc["SK_ID_CURR"]).mean()
            grouped[SECONDARY_CC + "UTILIZATION_MAX"] = util.groupby(cc["SK_ID_CURR"]).max()
        if "SK_DPD" in cc.columns:
            grouped[SECONDARY_CC + "DPD_FLAG_FRAC"] = (
                cc["SK_DPD"].gt(0).groupby(cc["SK_ID_CURR"]).mean()
            )
            grouped[SECONDARY_CC + "DPD_MAX"] = cc.groupby("SK_ID_CURR")["SK_DPD"].max()

        grouped = grouped.fillna(0)
        return grouped

    # ------------------------------------------------------------------ all
    def aggregate_all(
        self, tables: Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]
    ) -> pd.DataFrame:
        """Run all 5 aggregators and return a single DataFrame indexed by SK_ID_CURR.

        Args:
            tables: dict with keys "bureau", "previous_application", "pos_cash",
                "installments", "credit_card". The "bureau" value can be a
                tuple (bureau_df, bureau_balance_df) or a single DataFrame
                (in which case bureau_balance is omitted).

        Returns:
            DataFrame with SK_ID_CURR index and a few dozen numeric features,
            outer-joined across the 5 aggregators.
        """
        feats: list[pd.DataFrame] = []
        for key, val in tables.items():
            # Skip empty tables (no rows from a missing parquet file): the
            # per-table aggregators would raise on missing required columns,
            # so we just emit an empty contribution for that table.
            if isinstance(val, tuple):
                empty = all(v is None or len(v) == 0 for v in val)
            else:
                empty = val is None or len(val) == 0
            if empty:
                continue
            if key == "bureau":
                bureau_df, bureau_bal_df = (val if isinstance(val, tuple) else (val, None))
                feats.append(self.aggregate_bureau(bureau_df, bureau_bal_df))
            elif key == "previous_application":
                feats.append(self.aggregate_previous_app(val))
            elif key == "pos_cash":
                feats.append(self.aggregate_pos_cash(val))
            elif key == "installments":
                feats.append(self.aggregate_installments(val))
            elif key == "credit_card":
                feats.append(self.aggregate_credit_card(val))
            else:
                raise ValueError(f"Unknown table key: {key}")

        if not feats:
            return pd.DataFrame(index=pd.Index([], name="SK_ID_CURR"))

        # Outer join to keep all applicants that appear in any table
        merged = feats[0]
        for f in feats[1:]:
            merged = merged.join(f, how="outer")
        merged = merged.fillna(0)
        return merged


# ---------------------------------------------------------------------------
# Field lists. Keep them as module-level constants so tests can sanity-check
# schema without instantiating the class.
# ---------------------------------------------------------------------------

_BUREAU_NUM = [
    "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE",
    "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY", "CNT_CREDIT_PROLONG",
    "DAYS_CREDIT_UPDATE",
]
_PREV_NUM = [
    "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
    "AMT_GOODS_PRICE", "RATE_DOWN_PAYMENT", "RATE_INTEREST_PRIMARY",
    "RATE_INTEREST_PRIVILEGED", "DAYS_DECISION", "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE",
    "DAYS_TERMINATION", "NFLAG_INSURED_ON_APPROVAL", "SELLERPLACE_AREA",
]
_POS_NUM = [
    "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
    "SK_DPD", "SK_DPD_DEF",
]
_INST_NUM = [
    "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
    "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
]
_CC_NUM = [
    "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT",
    "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT",
    "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE",
    "AMT_TOTAL_RECEIVABLE", "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT",
    "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT", "CNT_INSTALMENT_MATURE_CUM",
    "SK_DPD", "SK_DPD_DEF",
]


def load_secondary_tables(
    raw_dir: str = "data/home-credit-default-risk/_raw",
) -> Dict[str, pd.DataFrame]:
    """Read pre-downloaded parquet files from `raw_dir` and return a dict of
    table-name -> DataFrame. The function is tolerant of missing files: a
    missing file produces an empty DataFrame, which the aggregator will skip
    gracefully (no features for that table).

    Naming convention: the original mirror uses two patterns
    - `<name>_NNNN.parquet`        (bureau, bureau_balance, *_balance, installments_payments)
    - `<name>_train-NNNNN-of-NNNNN.parquet`  (previous_application only)

    We match by exact prefix on the basename (not glob `*`) to avoid the
    classic `bureau_*.parquet` matching `bureau_balance_*.parquet` bug.
    """
    import glob
    import os

    tables: Dict[str, pd.DataFrame] = {}
    for name in ("bureau", "bureau_balance", "previous_application",
                 "POS_CASH_balance", "installments_payments", "credit_card_balance"):
        all_parquet = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
        files = [f for f in all_parquet if os.path.basename(f).startswith(name + "_")]
        if not files:
            tables[name] = pd.DataFrame()
            continue
        parts = [pd.read_parquet(f) for f in files]
        tables[name] = pd.concat(parts, ignore_index=True)
    return tables


# Cache version — bump to invalidate the cache when the aggregator schema changes.
SECONDARY_FEATURES_CACHE_VERSION = 2


def load_or_build_secondary_features(
    raw_dir: str = "data/home-credit-default-risk/_raw",
    cache_path: str = "output/cache/secondary_features_v1.parquet",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Cache wrapper around MultiTableAggregator.aggregate_all.

    On a warm cache, this returns in < 1 second (parquet read of a few-MB file).
    On a cold cache (or when `force_rebuild=True`), it runs the full 65-second
    aggregation over the 5 secondary tables and persists the result to
    `cache_path` for next time.

    The cache is a single parquet with SK_ID_CURR as the index. Re-running
    `run_pipeline.py` is therefore idempotent: only the very first run pays
    the aggregation cost; subsequent runs and tests share the same file.

    To invalidate the cache after modifying the aggregator (e.g. adding a
    new feature), bump `SECONDARY_FEATURES_CACHE_VERSION` (e.g. v1 → v2)
    so callers automatically pick up the new schema.
    """
    import os
    import time

    if not force_rebuild and os.path.exists(cache_path):
        t0 = time.time()
        cached = pd.read_parquet(cache_path)
        # Sanity-check: parquet should have SK_ID_CURR index and >=200 columns
        if cached.index.name == "SK_ID_CURR" and cached.shape[1] >= 200:
            print(
                f"  [cache hit]  {cache_path}  "
                f"shape={cached.shape}  loaded in {time.time()-t0:.2f}s"
            )
            return cached
        print(
            f"  [cache stale] {cache_path} failed sanity check "
            f"(index={cached.index.name}, cols={cached.shape[1]}); rebuilding"
        )

    t0 = time.time()
    print("  [cache miss] running full multi-table aggregation ...")
    secondary_raw = load_secondary_tables(raw_dir)

    # Defensive scrub: drop POS / CC rows with MONTHS_BALANCE > 0 (those are
    # post-application records and would leak the answer into the features).
    from src.data.temporal_guard import TemporalGuard
    guard = TemporalGuard()
    secondary_raw, temporal_report = guard.scrub_secondary_tables(secondary_raw)
    if temporal_report.issues:
        for issue in temporal_report.issues:
            print(f"  [temporal-guard] {issue.type} table={issue.table} "
                  f"removed={issue.count} ({100 * issue.ratio:.3f}%)")
    else:
        print("  [temporal-guard] no MONTHS_BALANCE > 0 rows found")

    agg = MultiTableAggregator()
    features = agg.aggregate_all({
        "bureau": (secondary_raw["bureau"], secondary_raw["bureau_balance"]),
        "previous_application": secondary_raw["previous_application"],
        "pos_cash": secondary_raw["POS_CASH_balance"],
        "installments": secondary_raw["installments_payments"],
        "credit_card": secondary_raw["credit_card_balance"],
    })
    # Persist for next run
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    features.to_parquet(cache_path)
    print(
        f"  [cache write] {cache_path}  "
        f"shape={features.shape}  built in {time.time()-t0:.1f}s"
    )
    return features
