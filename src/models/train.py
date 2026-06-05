"""Model training: sklearn GBT (baseline) + LightGBM (downstream causal stack).

`GBTrainer` keeps the original 5-fold-CV API (used for the AUC baseline).
`LightGBMTrainer` is the canonical trainer for the causal pipeline — it
is faster on the 30K+ row Home Credit data and feeds SHAP / DiCE.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class GBTrainer:
    """sklearn GradientBoostingClassifier trainer (baseline, for evaluation)."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        gbm_cfg = cfg.get("gbm", {})
        self.params = {
            "n_estimators": gbm_cfg.get("n_estimators", 200),
            "max_depth": gbm_cfg.get("max_depth", 5),
            "learning_rate": gbm_cfg.get("learning_rate", 0.1),
            "subsample": gbm_cfg.get("subsample", 0.8),
            "min_samples_leaf": gbm_cfg.get("min_samples_leaf", 20),
            "max_features": gbm_cfg.get("max_features", "sqrt"),
            "random_state": gbm_cfg.get("random_state", 42),
        }
        self.model: Optional[GradientBoostingClassifier] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def train_cv(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict[str, float]:
        model = GradientBoostingClassifier(**self.params)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return {
            "cv_auc_mean": float(auc.mean()),
            "cv_auc_std": float(auc.std()),
            "cv_accuracy_mean": float(acc.mean()),
            "cv_accuracy_std": float(acc.std()),
            "n_folds": n_folds,
        }

    def train_final(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
        self.model = GradientBoostingClassifier(**self.params)
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.feature_names_ = list(X.columns)
        return self.model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise RuntimeError("Model must be trained before getting feature importance")
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class LightGBMTrainer:
    """LightGBM classifier trainer — faster on big N, used downstream of GBT."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        lgbm_cfg = cfg.get("lightgbm", {})
        self.params = {
            "n_estimators": lgbm_cfg.get("n_estimators", 300),
            "max_depth": lgbm_cfg.get("max_depth", 7),
            "learning_rate": lgbm_cfg.get("learning_rate", 0.05),
            "subsample": lgbm_cfg.get("subsample", 0.8),
            "colsample_bytree": lgbm_cfg.get("colsample_bytree", 0.8),
            "min_child_samples": lgbm_cfg.get("min_child_samples", 50),
            "random_state": lgbm_cfg.get("random_state", 42),
            "n_jobs": lgbm_cfg.get("n_jobs", -1),
            "verbosity": -1,
        }
        self.model = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def train_cv(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict[str, float]:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**self.params)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return {
            "cv_auc_mean": float(auc.mean()),
            "cv_auc_std": float(auc.std()),
            "cv_accuracy_mean": float(acc.mean()),
            "cv_accuracy_std": float(acc.std()),
            "n_folds": n_folds,
        }

    def train_final(self, X: pd.DataFrame, y: pd.Series):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.feature_names_ = list(X.columns)
        return self.model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise RuntimeError("Model must be trained before getting feature importance")
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
