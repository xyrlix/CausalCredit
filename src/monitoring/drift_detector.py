"""Model and data drift detection.

Implements Population Stability Index (PSI) for feature drift, score-bucket
drift for prediction drift, and a simple performance-degradation check for
concept drift. All measurements are reference-vs-current; the reference
dataset is supplied at construction time.

PSI bands (industry standard):
  < 0.10  no drift
  0.10–0.20  moderate drift  ("warning")
  >= 0.20  significant drift ("alert")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_psi_from_dists(ref_dist: np.ndarray, cur_dist: np.ndarray, eps: float = 1e-6) -> float:
    """PSI = Σ (cur - ref) * log(cur / ref) with epsilon smoothing."""
    ref_dist = np.clip(ref_dist, eps, None)
    cur_dist = np.clip(cur_dist, eps, None)
    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


class DriftDetector:
    """Drift detection for model monitoring."""

    PSI_NO_DRIFT = 0.10
    PSI_ALERT = 0.20

    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data.copy()
        self._bin_edges: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Feature drift (PSI)
    # ------------------------------------------------------------------
    def _bins_for(self, feature: str, bins: int) -> np.ndarray:
        if feature in self._bin_edges:
            return self._bin_edges[feature]
        ref = self.reference_data[feature].dropna().astype(float)
        if ref.nunique() <= bins:
            edges = np.unique(ref.values)
            # Make sure we have at least 2 edges
            if len(edges) < 2:
                edges = np.array([edges[0] - 1, edges[0] + 1])
        else:
            edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([ref.min() - 1e-9, ref.max() + 1e-9])
        # Ensure strict left/right padding so first/last value isn't lost
        edges[0] = -np.inf
        edges[-1] = np.inf
        self._bin_edges[feature] = edges
        return edges

    def compute_psi(self, feature: str, current: pd.Series, bins: int = 10) -> float:
        if feature not in self.reference_data.columns:
            raise KeyError(f"Feature '{feature}' not in reference data")
        ref = self.reference_data[feature].dropna().astype(float).values
        cur = current.dropna().astype(float).values
        if len(ref) == 0 or len(cur) == 0:
            return float("nan")
        edges = self._bins_for(feature, bins)
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        ref_dist = ref_counts / max(ref_counts.sum(), 1)
        cur_dist = cur_counts / max(cur_counts.sum(), 1)
        return _safe_psi_from_dists(ref_dist, cur_dist)

    @staticmethod
    def _label_psi(psi: float) -> str:
        if not np.isfinite(psi):
            return "n/a"
        if psi < DriftDetector.PSI_NO_DRIFT:
            return "no_drift"
        if psi < DriftDetector.PSI_ALERT:
            return "moderate"
        return "alert"

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        bins: int = 10,
    ) -> pd.DataFrame:
        features = features or [c for c in self.reference_data.columns if c in current_data.columns]
        rows = []
        for c in features:
            try:
                psi = self.compute_psi(c, current_data[c], bins=bins)
                rows.append({
                    "feature": c,
                    "psi": psi,
                    "status": self._label_psi(psi),
                })
            except Exception as exc:
                rows.append({"feature": c, "psi": float("nan"), "status": f"error: {exc}"})
        return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Prediction drift
    # ------------------------------------------------------------------
    def detect_prediction_drift(
        self,
        reference_scores: pd.Series,
        current_scores: pd.Series,
        bins: int = 10,
    ) -> Dict[str, float]:
        ref = reference_scores.dropna().astype(float).values
        cur = current_scores.dropna().astype(float).values
        if len(ref) == 0 or len(cur) == 0:
            return {"psi": float("nan"), "status": "n/a"}
        edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([ref.min() - 1e-9, ref.max() + 1e-9])
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        ref_dist = ref_counts / max(ref_counts.sum(), 1)
        cur_dist = cur_counts / max(cur_counts.sum(), 1)
        psi = _safe_psi_from_dists(ref_dist, cur_dist)
        return {
            "psi": psi,
            "status": self._label_psi(psi),
            "ref_mean": float(np.mean(ref)),
            "cur_mean": float(np.mean(cur)),
            "ref_std": float(np.std(ref)),
            "cur_std": float(np.std(cur)),
        }

    # ------------------------------------------------------------------
    # Concept drift
    # ------------------------------------------------------------------
    def detect_concept_drift(
        self,
        current_auc: float,
        baseline_auc: float,
        current_ks: Optional[float] = None,
        baseline_ks: Optional[float] = None,
        auc_drop_threshold: float = 0.05,
        ks_drop_threshold: float = 0.10,
    ) -> Dict[str, Any]:
        auc_drop = baseline_auc - current_auc
        result: Dict[str, Any] = {
            "auc_drop": float(auc_drop),
            "auc_drop_alert": bool(auc_drop > auc_drop_threshold),
        }
        if current_ks is not None and baseline_ks is not None:
            ks_drop = baseline_ks - current_ks
            result.update({
                "ks_drop": float(ks_drop),
                "ks_drop_alert": bool(ks_drop > ks_drop_threshold),
            })
        result["status"] = "alert" if result.get("auc_drop_alert") or result.get("ks_drop_alert", False) else "ok"
        return result

    # ------------------------------------------------------------------
    # Feature statistics comparison
    # ------------------------------------------------------------------
    def compute_feature_statistics(self, current_data: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for c in self.reference_data.columns:
            if c not in current_data.columns:
                continue
            ref = pd.to_numeric(self.reference_data[c], errors="coerce").dropna()
            cur = pd.to_numeric(current_data[c], errors="coerce").dropna()
            if len(ref) == 0 or len(cur) == 0:
                continue
            rows.append({
                "feature": c,
                "ref_mean": float(ref.mean()),
                "cur_mean": float(cur.mean()),
                "delta_mean": float(cur.mean() - ref.mean()),
                "ref_std": float(ref.std()),
                "cur_std": float(cur.std()),
                "ref_median": float(ref.median()),
                "cur_median": float(cur.median()),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    def generate_drift_report(
        self,
        current_data: pd.DataFrame,
        reference_scores: Optional[pd.Series] = None,
        current_scores: Optional[pd.Series] = None,
        baseline_auc: Optional[float] = None,
        current_auc: Optional[float] = None,
    ) -> str:
        lines = ["# Drift Report", ""]
        lines.append(f"- Reference rows: {len(self.reference_data)}")
        lines.append(f"- Current rows: {len(current_data)}")
        lines.append("")

        lines.append("## 1. Feature drift (PSI)")
        f_df = self.detect_feature_drift(current_data)
        lines.append("")
        lines.append("| Feature | PSI | Status |")
        lines.append("|---------|-----|--------|")
        for _, r in f_df.iterrows():
            psi_str = f"{r['psi']:.4f}" if np.isfinite(r["psi"]) else "n/a"
            lines.append(f"| {r['feature']} | {psi_str} | {r['status']} |")
        n_alert = (f_df["status"] == "alert").sum()
        n_mod = (f_df["status"] == "moderate").sum()
        lines.append("")
        lines.append(f"**Summary:** {n_alert} alerts, {n_mod} moderate.")
        lines.append("")

        if reference_scores is not None and current_scores is not None:
            lines.append("## 2. Prediction drift")
            pd_result = self.detect_prediction_drift(reference_scores, current_scores)
            for k, v in pd_result.items():
                if isinstance(v, float):
                    lines.append(f"- **{k}**: {v:.4f}")
                else:
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        if baseline_auc is not None and current_auc is not None:
            lines.append("## 3. Concept drift")
            cd = self.detect_concept_drift(current_auc, baseline_auc)
            for k, v in cd.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        return "\n".join(lines)
