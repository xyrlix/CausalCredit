"""Model evaluation metrics and reporting."""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Model performance evaluator."""

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray,
                 y_prob: np.ndarray) -> Dict[str, Any]:
        """Compute key classification metrics."""
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)
        y_pr = np.asarray(y_prob, dtype=float)

        metrics: Dict[str, Any] = {}
        metrics["auc_roc"] = roc_auc_score(y_t, y_pr)
        metrics["accuracy"] = accuracy_score(y_t, y_p)
        metrics["precision"] = precision_score(y_t, y_p, zero_division=0)
        metrics["recall"] = recall_score(y_t, y_p, zero_division=0)
        metrics["f1_score"] = f1_score(y_t, y_p, zero_division=0)
        metrics["log_loss"] = log_loss(y_t, np.clip(y_pr, 1e-15, 1 - 1e-15))

        return {k: round(float(v), 6) for k, v in metrics.items()}

    def evaluate_subgroup(self, X: pd.DataFrame, y_true: pd.Series,
                          y_prob: np.ndarray,
                          subgroup_col: str) -> pd.DataFrame:
        """Evaluate model performance by subgroup."""
        results = []
        for group_val, idx in X.groupby(subgroup_col).groups.items():
            sub_y_true = y_true.iloc[idx]
            sub_y_prob = y_prob[idx]
            sub_y_pred = (sub_y_prob >= 0.5).astype(int)

            if len(sub_y_true) < 2:
                continue

            row = {"subgroup": str(group_val), "n": len(sub_y_true)}
            row["auc_roc"] = roc_auc_score(sub_y_true, sub_y_prob)
            row["accuracy"] = accuracy_score(sub_y_true, sub_y_pred)
            row["precision"] = precision_score(sub_y_true, sub_y_pred, zero_division=0)
            row["recall"] = recall_score(sub_y_true, sub_y_pred, zero_division=0)
            results.append(row)

        return pd.DataFrame(results)

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a formatted evaluation report string."""
        lines = ["=" * 50, "  MODEL EVALUATION REPORT", "=" * 50, ""]
        for key, value in metrics.items():
            lines.append(f"  {key:<20s}: {value}")
        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)
