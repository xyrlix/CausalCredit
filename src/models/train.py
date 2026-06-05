"""GradientBoostingClassifier training with cross-validation."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class GBTrainer:
    """GradientBoostingClassifier trainer for credit default prediction."""

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
        self.feature_names_: Optional[list] = None

    def train_cv(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict[str, float]:
        """Train with stratified k-fold cross-validation and return CV scores."""
        model = GradientBoostingClassifier(**self.params)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        return {
            "cv_auc_mean": float(auc_scores.mean()),
            "cv_auc_std": float(auc_scores.std()),
            "cv_accuracy_mean": float(acc_scores.mean()),
            "cv_accuracy_std": float(acc_scores.std()),
            "n_folds": n_folds,
        }

    def train_final(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
        """Train final model on full dataset."""
        self.model = GradientBoostingClassifier(**self.params)
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.feature_names_ = list(X.columns)
        return self.model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities of positive class."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise RuntimeError("Model must be trained before getting feature importance")
        imp_df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp_df
