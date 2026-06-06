"""Slicing helpers for fairness audits.

Maps raw Home Credit columns to a small set of bucket labels
suitable for group-fairness testing.  The buckets are chosen so
each group has at least a few hundred applicants (small groups
make the metrics statistically unreliable).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# Each slice is a (name, source_column, derived-bucket-function).
# The bucket function is a `pd.cut` or a mapping; we keep them
# simple to avoid pulling in sklearn.preprocessing.
SLICE_DEFINITIONS: List[Dict] = [
    {
        "name": "gender",
        "column": "CODE_GENDER",
        "description": "Legally protected attribute (HKMA / EU AI Act)",
    },
    {
        "name": "age_group",
        "column": "DAYS_BIRTH",
        "description": "Binned age: young / mid / old",
    },
    {
        "name": "income_group",
        "column": "AMT_INCOME_TOTAL",
        "description": "Income tertile: low / mid / high",
    },
    {
        "name": "education_group",
        "column": "NAME_EDUCATION_TYPE",
        "description": "Education level (raw categories, 5 levels)",
    },
]


def slice_dataset(X: pd.DataFrame, slice_def: Dict) -> np.ndarray:
    """Return a (n,) group-label array for the given slice.

    Missing or unknown values are mapped to ``"UNKNOWN"`` so they
    don't break the per-group loop, but they are excluded from
    between-group fairness metrics because they don't represent
    a real protected group.
    """
    col = slice_def["column"]
    if col not in X.columns:
        return np.array(["UNKNOWN"] * len(X), dtype=object)

    raw = X[col]
    name = slice_def["name"]

    if name == "age_group":
        # DAYS_BIRTH is negative days (e.g. -15000 = ~41 years old).
        # Convert to years (absolute value / 365) and bin.
        years = (-raw.astype(float) / 365.0).clip(18, 90)
        bins = [0, 35, 60, 200]
        labels = ["young", "mid", "old"]
        return pd.cut(years, bins=bins, labels=labels).astype(object).fillna("UNKNOWN").values

    if name == "income_group":
        v = raw.astype(float)
        # Use population tertiles; if not enough variance, fall back to a single bucket
        try:
            q33, q66 = v.quantile([0.333, 0.666])
        except Exception:
            return np.array(["mid"] * len(X), dtype=object)
        out = np.where(v <= q33, "low", np.where(v <= q66, "mid", "high"))
        return out.astype(object)

    if name == "education_group":
        # Map common education strings; everything else is "other"
        v = raw.astype(str)
        canon = {
            "Lower secondary": "secondary",
            "Secondary / secondary special": "secondary",
            "Incomplete higher": "higher_incomplete",
            "Higher education": "higher",
            "Academic degree": "academic",
        }
        return v.map(canon).fillna("other").values

    if name == "gender":
        v = raw.astype(str)
        # In Home Credit, values are M / F / XNA
        return np.where(v == "M", "M", np.where(v == "F", "F", "UNKNOWN")).astype(object)

    # Fallback: return raw values as strings
    return raw.astype(str).fillna("UNKNOWN").values


def build_default_slices(X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return a dict {slice_name: group_array} for all default slices."""
    return {sd["name"]: slice_dataset(X, sd) for sd in SLICE_DEFINITIONS}
