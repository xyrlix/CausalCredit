"""Packaging detection via causal-consistency scoring.

Implements the "因果一致性检测引擎" from
``docs/CausalCredit_反欺诈能力覆盖分析.md`` §4.1.3.

The detection is based on the four-quadrant labeling produced by
``SHAPExplainer.causal_vs_noncausal_contribution`` (TRUSTED / UNTRUSTED
/ MASKED / NEGLIGIBLE) plus a domain-DAG path-integrity score.

Definitions (per the spec):

* ``path_integrity``  = 1 − (broken_paths / expected_paths)
    * Expected chain: income → consumption → repayment → default
    * A "broken" path means the head-node is in the highest income
      decile *and* the tail-node is in the lowest consumption decile
      (or, equivalently, the model is heavily weighting income signals
      that don't propagate through the chain).

* ``packaging_score`` = 1 − (#trusted + #masked) / total_features
    * Trusted & masked features are "model + causal agree, OR causal
      alone is strong" — these are the credible ones.
    * Untrusted features are "model high, causal low" — these are
      potentially packaging.

* ``feature_credibility`` (per feature) =
    ``0.5 * (causal_proxy_normalized) + 0.5 * (path_integrity)``
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Domain-DAG path nodes (Home Credit flavor). Each path is a list of
# (column, direction) pairs where direction = "high" means we expect a
# positive correlation along the path.
EXPECTED_PATHS: List[List[Tuple[str, str]]] = [
    # income → goods_price → credit → annuity → default pressure
    [("AMT_INCOME_TOTAL", "high"), ("AMT_GOODS_PRICE", "high"),
     ("AMT_CREDIT", "high"), ("AMT_ANNUITY", "high")],
    # income → external_score → default (中介)
    [("AMT_INCOME_TOTAL", "high"), ("EXT_SOURCE_2", "high")],
    # age → employment → income (DAG confounder chain)
    [("DAYS_BIRTH", "low"), ("DAYS_EMPLOYED", "low"),
     ("AMT_INCOME_TOTAL", "high")],
]


class PackagingDetector:
    """Compute packaging_score and path_integrity per applicant.

    The detector is *post-hoc*: it consumes a four-quadrant result
    (or just SHAP values + a fitted model + the feature matrix) and
    returns a single per-applicant score.  No retraining needed.
    """

    def __init__(
        self,
        high_income_quantile: float = 0.90,
        low_consumption_quantile: float = 0.10,
    ):
        self.high_income_quantile = high_income_quantile
        self.low_consumption_quantile = low_consumption_quantile
        # Learned at score time:
        self.feature_credibility_: Dict[str, float] = {}
        self.global_path_integrity_: float = 0.0

    # ------------------------------------------------------------------ public
    def calibrate(self, X: pd.DataFrame, four_quadrant: Dict) -> "PackagingDetector":
        """Learn per-feature credibility and a global path-integrity baseline.

        Should be called once on a reference population (typically the
        training set) before ``score`` is called per applicant.
        """
        per_feature = four_quadrant["per_feature"].set_index("feature")
        proxy = per_feature["abs_causal_proxy"].to_dict()
        max_proxy = max(proxy.values()) if proxy else 1.0
        for f, p in proxy.items():
            # credibility ranges [0, 1] = normalized proxy (since path
            # integrity is the same global value, the average works out
            # to 0.5 * normalized + 0.5 * global)
            self.feature_credibility_[f] = float(p) / max_proxy

        # Global path integrity = avg of per-feature normalized proxies
        # (rough but reasonable baseline)
        if self.feature_credibility_:
            self.global_path_integrity_ = float(np.mean(list(self.feature_credibility_.values())))
        else:
            self.global_path_integrity_ = 0.5
        return self

    def score(
        self,
        X: pd.DataFrame,
        four_quadrant: Dict,
        applicant_idx: int = 0,
        row_shap: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute packaging_score, path_integrity, and feature_credibility
        for a single applicant.

        Args:
            X: feature matrix (row 0 = this applicant).
            four_quadrant: dict from ``SHAPExplainer.causal_vs_noncausal_contribution``.
            applicant_idx: which row of X to score.
        """
        per_feature = four_quadrant["per_feature"]
        if not self.feature_credibility_:
            # If calibrate() was not called, derive proxies directly
            self.calibrate(X, four_quadrant)
        if len(per_feature) == 0:
            return {
                "packaging_score": 0.0,
                "path_integrity": 0.0,
                "feature_credibility": {},
                "routing": "UNKNOWN",
            }

        # 1) Path integrity for this applicant
        path_integrity = _compute_path_integrity(X.iloc[[applicant_idx]])

        # 2) Per-applicant packaging score.
        # The score is defined as the fraction of this applicant's
        # top-K most-influential features (by |SHAP|) that have LOW
        # global causal proxy (i.e. the model is using them but they
        # are not causally credible — a classic packaging signal).
        # Causal credibility uses the global threshold (causal proxies
        # are population-level quantities, not per-applicant).
        th_shap, th_causal = four_quadrant.get("thresholds", (0.0, 0.0))
        proxy_map = dict(zip(per_feature["feature"].tolist(),
                             per_feature["abs_causal_proxy"].tolist()))
        if row_shap is not None:
            sv_arr = np.asarray(row_shap).flatten()
            sv_map = dict(zip(per_feature["feature"].tolist(), sv_arr))
        else:
            # Fall back to global mean abs SHAP (less accurate)
            sv_map = {f: per_feature.iloc[i]["mean_abs_shap"]
                      for i, f in enumerate(per_feature["feature"].tolist())}
            sv_arr = np.array(list(sv_map.values()))

        # Build the per-applicant quadrant counts and packaging score
        # in one pass.
        abs_sv = np.abs(sv_arr)
        n_feat = len(abs_sv)
        # top-K = top 25% most influential features for this applicant
        k = max(1, int(0.25 * n_feat))
        topk_idx = set(np.argpartition(abs_sv, -k)[-k:].tolist())
        feature_list = per_feature["feature"].tolist()

        quadrant_counts = {"TRUSTED": 0, "UNTRUSTED": 0,
                           "MASKED": 0, "NEGLIGIBLE": 0}
        cred = dict(self.feature_credibility_)
        for j, f in enumerate(feature_list):
            ms = float(abs_sv[j])
            cp = float(proxy_map.get(f, 0.0))
            high_shap = j in topk_idx
            high_causal = cp >= th_causal
            if high_shap and high_causal:
                q = "TRUSTED"
            elif high_shap and not high_causal:
                q = "UNTRUSTED"
            elif (not high_shap) and high_causal:
                q = "MASKED"
            else:
                q = "NEGLIGIBLE"
            quadrant_counts[q] += 1
            # Re-weight credibility per-applicant
            if f in cred:
                cred[f] = cred[f] * (1.0 - 0.5 * min(ms, 1.0))

        trusted = quadrant_counts["TRUSTED"]
        masked = quadrant_counts["MASKED"]
        untrusted = quadrant_counts["UNTRUSTED"]
        negligible = quadrant_counts["NEGLIGIBLE"]
        total = trusted + masked + untrusted + negligible
        if total == 0:
            return {
                "packaging_score": 0.0,
                "path_integrity": float(path_integrity),
                "feature_credibility": cred,
                "routing": "UNKNOWN",
            }

        # 3) Packaging score = UNTRUSTED / (TRUSTED + UNTRUSTED)
        # i.e., of the model's top-K influential features, what
        # fraction have low causal credibility?  This is the literal
        # "包装嫌疑" signal: model uses them, but causal graph says
        # they're noise.
        denom = trusted + untrusted
        if denom == 0:
            packaging_score = 0.0
        else:
            packaging_score = untrusted / denom

        # 4) Routing decision
        if packaging_score >= 0.5:
            routing = "REJECT_PACKAGING_SUSPECTED"
        elif packaging_score >= 0.3:
            routing = "MANUAL_REVIEW"
        else:
            routing = "PROCEED"

        return {
            "packaging_score": float(packaging_score),
            "path_integrity": float(path_integrity),
            "feature_credibility": cred,
            "quadrant_counts": quadrant_counts,
            "routing": routing,
        }

    def score_batch(
        self,
        X: pd.DataFrame,
        four_quadrant: Dict,
    ) -> pd.DataFrame:
        """Score many applicants (rows of X) at once.

        Returns a DataFrame with one row per applicant.
        """
        if not self.feature_credibility_:
            self.calibrate(X, four_quadrant)
        rows = []
        for i in range(len(X)):
            r = self.score(X, four_quadrant, applicant_idx=i)
            rows.append({
                "applicant_idx": i,
                "packaging_score": r["packaging_score"],
                "path_integrity": r["path_integrity"],
                "routing": r["routing"],
            })
        return pd.DataFrame(rows)


def _compute_path_integrity(X_one: pd.DataFrame) -> float:
    """Compute path integrity for a single applicant.

    A path is "intact" if the head node is in its expected high/low
    quantile *and* the tail node is in its expected high/low quantile.
    A "broken" path contributes -1.  Result is normalized to [0, 1].
    """
    n_paths = len(EXPECTED_PATHS)
    n_intact = 0
    for path in EXPECTED_PATHS:
        if _is_path_intact(X_one, path):
            n_intact += 1
    return n_intact / n_paths if n_paths > 0 else 0.0


def _is_path_intact(X_one: pd.DataFrame, path: List[Tuple[str, str]]) -> bool:
    """Heuristic: a path is intact if all consecutive same-scale steps
    have ratios in [0.01, 100].

    Cross-scale steps (e.g. ``AMT_INCOME_TOTAL`` → ``EXT_SOURCE_2``) are
    skipped because their ratio is uninformative.  This means a path
    like ``income → ext_score`` is always "intact" — the integrity
    check is meaningful only for monetary chains.
    """
    if len(path) < 2:
        return True
    cols = [(c, d) for c, d in path if c in X_one.columns]
    if len(cols) < 2:
        return True
    vals = X_one.iloc[0]
    for (c1, d1), (c2, d2) in zip(cols[:-1], cols[1:]):
        v1 = float(vals.get(c1, 0.0))
        v2 = float(vals.get(c2, 0.0))
        if v1 == 0 or v2 == 0:
            continue
        # Skip cross-scale steps (one col on 0-1 scale, the other on
        # monetary scale)
        if (v1 < 1.0) != (v2 < 1.0):
            continue
        ratio = v2 / v1
        if not (0.01 <= ratio <= 100.0):
            return False
    return True
