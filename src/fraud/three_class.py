"""Three-class defaulter sub-classifier.

Home Credit ``TARGET`` is binary (default / not). The anti-fraud spec
(``docs/CausalCredit_反欺诈能力覆盖分析.md`` §1.5 + §4.1.5) requires
splitting defaulters into three sub-classes:

* **Fraudulent** — first-party fraud: 申请时即埋伏, "拿了就跑"
* **Non-malicious** — involuntary default: 收入冲击/过度负债, 干预有效
* **Systemic** — default driven by industry/macro shock, 批量发生

Since Home Credit has no ground-truth fraud labels, we construct
**pseudo-labels** from business rules over observable features.

Pipeline contract (called by ``pipeline.FraudGuard``):

1. ``fit_pseudo_labels(X, y)`` — labels all applicants (defaulters get
   sub-classes; non-defaulters get ``"non_default"``).
2. ``fit(X, y)`` — trains a 4-class LightGBM on the full population
   (non_default is included so the model can rank applicants
   jointly).  The defaulter sub-classification is the argmax over
   the 3 fraud classes.
3. ``predict_proba(X)`` — returns an ``(n, 3)`` array of fraud-class
   probabilities (fraudulent, non_malicious, systemic) for every
   applicant (rows that are confidently non-default will have all
   three probabilities very small).
4. ``fraud_score(X, default_proba)`` — composes
   ``fraud_score = P(default) * P(fraud | default)`` for downstream
   routing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Three sub-classes for defaulters + the no-default sink
DEFRAUDER_CLASSES = ["fraudulent", "non_malicious", "systemic"]
ALL_LABELS = ["non_default"] + DEFRAUDER_CLASSES  # 4 classes total

# Industries that are commonly flagged as "systemic" (cyclical,
# decline-prone).  Home Credit ORGANIZATION_TYPE is a free-text field;
# we match a small whitelist of substrings.
_SYSTEMIC_ORG_SUBSTRINGS = [
    "Industry:",
    "Transport:",
    "Construction",
    "Mining",
    "Realty",
    "Cleaning",
    "Security",
    "Agriculture",
    "Electricity",
    "Telecom",
    "Hotel",
    "Restaurant",
    "Realtor",
]


class ThreeClassFraudClassifier:
    """Pseudo-label and train a 4-class defaulter sub-classifier.

    The pseudo-label rules below are derived from the anti-fraud spec;
    they are deliberately conservative (low recall, high precision)
    so the resulting ``fraud_score`` is a *signal*, not a verdict.
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        fraud_overdue_threshold: float = 30.0,
        fraud_income_z_min: float = 1.0,
        fraud_employment_z_max: float = -1.0,
    ):
        """Args:
            params: LightGBM hyperparameters (None -> sensible defaults).
            fraud_overdue_threshold: |SK_DPD_DEF| above this in
                installments counts as "首期即违约" (only meaningful if
                M5+ secondary tables are merged in).
            fraud_income_z_min: income z-score above which we suspect
                "夸大收入".
            fraud_employment_z_max: days_employed z-score below which
                we suspect "短期就业为包装".
        """
        self.params = params or {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 50,
            "random_state": 42, "n_jobs": -1, "verbosity": -1,
        }
        self.fraud_overdue_threshold = fraud_overdue_threshold
        self.fraud_income_z_min = fraud_income_z_min
        self.fraud_employment_z_max = fraud_employment_z_max
        self.model = None
        self.label_map_: Dict[int, str] = {i: lbl for i, lbl in enumerate(ALL_LABELS)}
        self.feature_names_: List[str] = []
        self.pseudo_label_counts_: Dict[str, int] = {}

    # ------------------------------------------------------------------ labels
    def fit_pseudo_labels(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Return a pd.Series of pseudo-labels aligned with X.

        Non-defaulters get ``"non_default"``.  Defaulters (y=1) are
        bucketed into the 3 sub-classes using domain rules.
        """
        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)
        labels = pd.Series(["non_default"] * len(X), index=X.index, dtype=object)

        default_mask = y.values == 1
        if not default_mask.any():
            self.pseudo_label_counts_ = {lbl: 0 for lbl in ALL_LABELS}
            return labels

        # ---- Fraudulent default: "首期即违约" + suspicious profile
        fraud_mask = pd.Series(False, index=X.index)
        # Rule 1: first-installment default (only if M5+ INST columns exist)
        for col in ("INST__DPD_MAX", "INST__DAYS_LATE_MAX", "POS_DPD_MAX"):
            if col in X.columns:
                # POS_DPD_MAX is measured in days-past-due; for installments
                # *_MAX we use the threshold parameter
                if col.startswith("POS_"):
                    fraud_mask = fraud_mask | (X[col].fillna(0) >= self.fraud_overdue_threshold)
                else:
                    fraud_mask = fraud_mask | (X[col].fillna(0) >= self.fraud_overdue_threshold)
        # Rule 2: extremely high income + extremely short employment
        # (classic "夸大收入 + 短期就业" fraud signature)
        if "AMT_INCOME_TOTAL" in X.columns and "DAYS_EMPLOYED" in X.columns:
            income_z = _safe_zscore(X["AMT_INCOME_TOTAL"])
            emp_z = _safe_zscore(X["DAYS_EMPLOYED"].clip(upper=0))
            fraud_mask = fraud_mask | (
                (income_z >= self.fraud_income_z_min) & (emp_z <= self.fraud_employment_z_max)
            )
        # Rule 3: very low EXT_SOURCE_1 (no external credit) + high income
        if "EXT_SOURCE_1" in X.columns and "AMT_INCOME_TOTAL" in X.columns:
            ext1 = X["EXT_SOURCE_1"].fillna(0)
            income_z = _safe_zscore(X["AMT_INCOME_TOTAL"])
            fraud_mask = fraud_mask | ((ext1 < 0.05) & (income_z >= 1.5))

        # ---- Systemic default: defaulter in a cyclical/decline industry
        systemic_mask = pd.Series(False, index=X.index)
        if "ORGANIZATION_TYPE" in X.columns:
            org = X["ORGANIZATION_TYPE"].fillna("").astype(str)
            systemic_mask = org.str.contains("|".join(_SYSTEMIC_ORG_SUBSTRINGS), regex=True, na=False)

        # Apply to defaulters only
        labels[fraud_mask & default_mask] = "fraudulent"
        labels[systemic_mask & default_mask & ~fraud_mask] = "systemic"
        labels[default_mask & ~fraud_mask & ~systemic_mask] = "non_malicious"

        self.pseudo_label_counts_ = {lbl: int((labels == lbl).sum()) for lbl in ALL_LABELS}
        return labels

    # ------------------------------------------------------------------ model
    def _to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = X.copy()
        for c in Xn.columns:
            if Xn[c].dtype == "object" or str(Xn[c].dtype) == "category":
                Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
        return Xn.fillna(0.0)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        labels: Optional[pd.Series] = None,
    ) -> "ThreeClassFraudClassifier":
        """Train the 4-class LightGBM.

        Args:
            X: feature matrix (numeric or mixed).
            y: TARGET (binary, 0/1).
            labels: optional pre-computed pseudo-labels; if None, calls
                ``fit_pseudo_labels`` internally.
        """
        import lightgbm as lgb

        if labels is None:
            labels = self.fit_pseudo_labels(X, y)
        labels = labels.reset_index(drop=True)
        Xn = self._to_numeric(X.reset_index(drop=True))
        self.feature_names_ = list(Xn.columns)

        y_int = labels.map({lbl: i for i, lbl in enumerate(ALL_LABELS)}).astype(int)

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(Xn, y_int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n, 3) array of P(fraudulent), P(non_malicious), P(systemic).

        Uses the defaulter-only posterior: ``P(subclass | default=1)``.  The
        model's 4-class probabilities are first normalized over the 3
        defaulter classes so they sum to 1.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        Xn = self._to_numeric(X.reset_index(drop=True))
        full = self.model.predict_proba(Xn)  # (n, n_classes_seen)
        classes_ = list(self.model.classes_)
        # Build a full (n, 4) array, filling 0 for classes not seen in training
        n = len(Xn)
        out4 = np.zeros((n, len(ALL_LABELS)))
        for col_idx, cls in enumerate(classes_):
            if 0 <= cls < len(ALL_LABELS):
                out4[:, cls] = full[:, col_idx]
        # Restrict to defaulter classes (1, 2, 3) and renormalize
        sub = out4[:, 1:]
        s = sub.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return sub / s

    def fraud_score(self, X: pd.DataFrame, default_proba: np.ndarray) -> np.ndarray:
        """Compose fraud_score = P(default) * P(fraudulent | default).

        Args:
            X: features (same row order as ``default_proba``).
            default_proba: (n,) array of P(default) from the binary model.
        """
        p_sub = self.predict_proba(X)  # (n, 3)
        p_fraud = p_sub[:, 0]  # column 0 = P(fraudulent)
        return np.asarray(default_proba).flatten() * p_fraud

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a DataFrame sorted descending."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


def _safe_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(s.median())
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std
