"""Data integrity validation for German Credit dataset."""

from typing import Any, Dict, List

import pandas as pd


def validate_no_nulls(df: pd.DataFrame) -> Dict[str, bool]:
    """Check for any null values in the DataFrame."""
    null_counts = df.isnull().sum()
    has_nulls = (null_counts > 0).any()
    return {
        "has_nulls": has_nulls,
        "null_columns": null_counts[null_counts > 0].to_dict() if has_nulls else {},
    }


def validate_dtypes(df: pd.DataFrame, categorical_cols: List[str],
                    numerical_cols: List[str]) -> Dict[str, Any]:
    """Validate expected data types for categorical and numerical columns."""
    issues = []
    for col in categorical_cols:
        if col not in df.columns:
            issues.append(f"Missing categorical column: {col}")
    for col in numerical_cols:
        if col not in df.columns:
            issues.append(f"Missing numerical column: {col}")
        elif col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column {col} is not numeric, found {df[col].dtype}")
    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


def validate_value_ranges(df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, Any]:
    """Validate that numerical columns have reasonable value ranges."""
    range_report = {}
    for col in numerical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            range_report[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }
    return range_report


def validate_target(df: pd.DataFrame, target_col: str = "class") -> Dict[str, Any]:
    """Validate target column distribution."""
    if target_col not in df.columns:
        return {"valid": False, "error": f"Target column '{target_col}' not found"}
    target = df[target_col]
    value_counts = target.value_counts().to_dict()
    n_total = len(target)
    return {
        "valid": True,
        "n_samples": n_total,
        "distribution": value_counts,
        "proportions": {k: round(v / n_total, 4) for k, v in value_counts.items()},
    }


def generate_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data overview report."""
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = int(df[col].nunique())
        n_missing = int(df[col].isnull().sum())
        missing_pct = round(n_missing / len(df) * 100, 2)
        rows.append({
            "column": col,
            "dtype": dtype,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "missing_pct": missing_pct,
        })
    return pd.DataFrame(rows)
