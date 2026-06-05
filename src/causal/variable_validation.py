"""Causal variable validation.

Validates that causal graph variables exist in the data and checks
basic relationships between treatments, confounders, and outcomes.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class CausalVariableValidator:
    """Validator for causal variables in the dataset."""

    def validate_treatment_variables(
        self, df: pd.DataFrame, treatments: List[str]
    ) -> Dict[str, Dict]:
        """Validate treatment variable distributions."""
        results = {}
        for tx in treatments:
            if tx not in df.columns:
                results[tx] = {"present": False, "error": f"Column '{tx}' not found"}
                continue

            col = df[tx]
            info: Dict[str, Any] = {"present": True, "dtype": str(col.dtype)}

            if pd.api.types.is_numeric_dtype(col):
                info.update({
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "mean": float(col.mean()),
                    "median": float(col.median()),
                    "std": float(col.std()),
                    "n_unique": int(col.nunique()),
                })
            else:
                info.update({
                    "n_unique": int(col.nunique()),
                    "value_counts": col.value_counts().head(10).to_dict(),
                })

            results[tx] = info
        return results

    def validate_confounders(
        self, df: pd.DataFrame, treatments: List[str],
        outcome: str, confounders: List[str],
    ) -> Dict[str, Dict]:
        """Validate confounder relationships with treatment and outcome."""
        results = {}
        outcome_col = df[outcome] if outcome in df.columns else None

        for conf in confounders:
            if conf not in df.columns:
                results[conf] = {"present": False, "error": f"Column '{conf}' not found"}
                continue

            info: Dict[str, Any] = {"present": True, "dtype": str(df[conf].dtype)}

            for tx in treatments:
                if tx in df.columns and pd.api.types.is_numeric_dtype(df[tx]):
                    if pd.api.types.is_numeric_dtype(df[conf]):
                        corr = df[tx].corr(df[conf])
                        info[f"corr_with_{tx}"] = round(float(corr), 4)
                    else:
                        try:
                            groups = df.groupby(conf, observed=False)[tx].mean()
                            info[f"mean_by_{conf}_for_{tx}"] = groups.to_dict()
                        except Exception:
                            pass

            if outcome_col is not None and pd.api.types.is_numeric_dtype(outcome_col):
                if pd.api.types.is_numeric_dtype(df[conf]):
                    corr = outcome_col.corr(df[conf])
                    info[f"corr_with_outcome"] = round(float(corr), 4)
                else:
                    try:
                        groups = df.groupby(conf, observed=False)[outcome].mean()
                        info[f"outcome_by_{conf}"] = groups.to_dict()
                    except Exception:
                        pass

            results[conf] = info
        return results

    def validate_mediators(
        self, df: pd.DataFrame, treatments: List[str],
        outcome: str, mediators: List[str],
    ) -> Dict[str, Dict]:
        """Validate mediator pathways existence."""
        results = {}
        for med in mediators:
            if med not in df.columns:
                results[med] = {"present": False, "error": f"Column '{med}' not found"}
                continue

            info: Dict[str, Any] = {"present": True, "dtype": str(df[med].dtype)}

            for tx in treatments:
                if tx in df.columns and pd.api.types.is_numeric_dtype(df[tx]):
                    if pd.api.types.is_numeric_dtype(df[med]):
                        corr = df[tx].corr(df[med])
                        info[f"corr_with_{tx}"] = round(float(corr), 4)

            if outcome in df.columns:
                if pd.api.types.is_numeric_dtype(df[outcome]) and pd.api.types.is_numeric_dtype(df[med]):
                    corr = df[outcome].corr(df[med])
                    info["corr_with_outcome"] = round(float(corr), 4)

            results[med] = info
        return results

    def validate_instruments(
        self, df: pd.DataFrame, treatment: str, instruments: List[str],
    ) -> Dict[str, Dict]:
        """Validate instrument variable strength (correlation with treatment)."""
        results = {}
        if treatment not in df.columns:
            return {inst: {"present": False, "error": "Treatment not in data"} for inst in instruments}

        for inst in instruments:
            if inst not in df.columns:
                results[inst] = {"present": False, "error": f"Column '{inst}' not found"}
                continue

            info: Dict[str, Any] = {"present": True}

            if pd.api.types.is_numeric_dtype(df[treatment]) and pd.api.types.is_numeric_dtype(df[inst]):
                corr = df[treatment].corr(df[inst])
                info["corr_with_treatment"] = round(float(corr), 4)
                info["strength"] = "strong" if abs(corr) > 0.3 else "weak"

            results[inst] = info
        return results

    def generate_quality_report(
        self,
        treatment_validation: Dict,
        confounder_validation: Dict,
        mediator_validation: Optional[Dict] = None,
    ) -> str:
        """Generate a human-readable quality report string."""
        lines = []
        lines.append("=" * 60)
        lines.append("  CAUSAL VARIABLE QUALITY REPORT")
        lines.append("=" * 60)

        lines.append("\n--- TREATMENT VARIABLES ---")
        for name, info in treatment_validation.items():
            present = info.get("present", False)
            status = "OK" if present else "MISSING"
            lines.append(f"  [{status}] {name}: {info.get('dtype', 'N/A')}")

        lines.append("\n--- CONFOUNDERS ---")
        for name, info in confounder_validation.items():
            present = info.get("present", False)
            status = "OK" if present else "MISSING"
            lines.append(f"  [{status}] {name}")

        if mediator_validation:
            lines.append("\n--- MEDIATORS ---")
            for name, info in mediator_validation.items():
                present = info.get("present", False)
                status = "OK" if present else "MISSING"
                lines.append(f"  [{status}] {name}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
