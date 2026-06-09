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
from sklearn.metrics import roc_auc_score, accuracy_score


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
    """LightGBM classifier trainer — faster on big N, used downstream of GBT.

    Supports CUDA build: pass ``device="cuda"`` in config to attempt GPU training.
    Auto-falls-back to CPU if the lightgbm install was compiled without CUDA.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        lgbm_cfg = cfg.get("lightgbm", {})
        self.device = _resolve_device(lgbm_cfg.get("device", "cpu"))
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
        if self.device != "cpu":
            self.params["device"] = self.device
        self.model = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.best_params_: Optional[Dict] = None
        self.study_: Optional["optuna.Study"] = None

    def train_cv(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5,
                 subsample_frac: float = 1.0, subsample_seed: int = 42,
                 early_stopping_rounds: Optional[int] = 50,
                 eval_fraction: float = 0.15) -> Dict[str, float]:
        """Run stratified K-fold cross-validation on (X, y).

        Parameters
        ----------
        subsample_frac : float in (0, 1]
            If < 1.0, take a stratified subsample of the rows before CV. Used
            to keep the 3-fold CV wall-time down on full Home Credit (~215K rows).
        early_stopping_rounds : int or None
            If given, use LightGBM early stopping with this patience. For each
            CV fold we sub-split the fold train into sub-train + sub-val; the
            model fits on sub-train and stops when sub-val AUC plateaus. The
            final model is then evaluated on the held-out fold. Default 50.
            Pass 0 or None to disable.
        eval_fraction : float in (0, 1)
            Fraction of each fold's training rows reserved for the early-stop
            validation set. Default 0.15.
        """
        import lightgbm as lgb
        if subsample_frac < 1.0:
            from sklearn.model_selection import train_test_split
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=subsample_frac, random_state=subsample_seed,
                stratify=y,
            )
            X, y = X_sub, y_sub

        use_es = early_stopping_rounds is not None and early_stopping_rounds > 0
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        auc_scores: List[float] = []
        acc_scores: List[float] = []
        best_iters: List[int] = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr_full, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr_full, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            if use_es:
                from sklearn.model_selection import train_test_split
                X_sub_tr, X_sub_va, y_sub_tr, y_sub_va = train_test_split(
                    X_tr_full, y_tr_full, test_size=eval_fraction,
                    random_state=subsample_seed + fold, stratify=y_tr_full,
                )
                model = lgb.LGBMClassifier(**self.params)
                model.fit(
                    X_sub_tr, y_sub_tr,
                    eval_set=[(X_sub_va, y_sub_va)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                )
                best_iters.append(int(model.best_iteration_ or self.params["n_estimators"]))
            else:
                model = lgb.LGBMClassifier(**self.params)
                model.fit(X_tr_full, y_tr_full)
            pred = model.predict_proba(X_va)[:, 1]
            auc_scores.append(float(roc_auc_score(y_va, pred)))
            acc_scores.append(float(accuracy_score(y_va, (pred >= 0.5).astype(int))))

        return {
            "cv_auc_mean": float(np.mean(auc_scores)),
            "cv_auc_std": float(np.std(auc_scores)),
            "cv_accuracy_mean": float(np.mean(acc_scores)),
            "cv_accuracy_std": float(np.std(acc_scores)),
            "n_folds": n_folds,
            "subsample_frac": subsample_frac,
            "early_stopping_rounds": early_stopping_rounds if use_es else 0,
            "best_iteration_mean": float(np.mean(best_iters)) if best_iters else None,
        }

    def train_final(self, X: pd.DataFrame, y: pd.Series,
                    subsample_frac: float = 1.0, subsample_seed: int = 42,
                    early_stopping_rounds: Optional[int] = 50,
                    eval_fraction: float = 0.15):
        """Fit on (X, y). If ``subsample_frac < 1.0``, take a stratified subsample first.

        If ``early_stopping_rounds`` is given, take a small eval holdout for
        early stopping (so the model is fit on ``(1 - eval_fraction)`` of the
        rows and validated on the rest). Default 50.
        """
        import lightgbm as lgb
        if subsample_frac < 1.0:
            from sklearn.model_selection import train_test_split
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=subsample_frac, random_state=subsample_seed,
                stratify=y,
            )
            X, y = X_sub, y_sub

        use_es = early_stopping_rounds is not None and early_stopping_rounds > 0
        self.model = lgb.LGBMClassifier(**self.params)
        if use_es:
            from sklearn.model_selection import train_test_split
            X_tr, X_es, y_tr, y_es = train_test_split(
                X, y, test_size=eval_fraction, random_state=subsample_seed,
                stratify=y,
            )
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
        else:
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

    def tune_hyperparams(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: int = 3600,
        n_folds: int = 3,
        subsample: int = 50_000,
        seed: int = 42,
    ) -> Dict:
        """Optuna hyperparameter search on a stratified subsample.

        Returns the best params dict (caller can merge into ``self.params``
        and re-run ``train_final``). Stores the Optuna study in
        ``self.study_`` for downstream inspection.
        """
        import optuna
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        rng = np.random.default_rng(seed)
        n = min(subsample, len(X))
        sub_idx = rng.choice(len(X), size=n, replace=False)
        X_sub = X.iloc[sub_idx].reset_index(drop=True)
        y_sub = y.iloc[sub_idx].reset_index(drop=True) if hasattr(y, "iloc") else y[sub_idx]

        base = {k: v for k, v in self.params.items() if k not in {"n_estimators"}}

        def _objective(trial: "optuna.Trial") -> float:
            params = {
                **base,
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            }
            import lightgbm as lgb
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            aucs = []
            for tr, va in cv.split(X_sub, y_sub):
                m = lgb.LGBMClassifier(**params)
                m.fit(X_sub.iloc[tr], y_sub.iloc[tr])
                aucs.append(roc_auc_score(y_sub.iloc[va], m.predict_proba(X_sub.iloc[va])[:, 1]))
            return float(np.mean(aucs))

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        self.study_ = study
        self.best_params_ = study.best_params
        return study.best_params


def _resolve_device(requested: str) -> str:
    """Return the device string for LightGBM, with safe fallback.

    LightGBM 4.5+ conda package on cuda_py3.10 ships with ``-DUSE_CUDA=1``;
    the CPU-only PyPI wheel does not. We probe by attempting a 1-iter fit
    on a tiny synthetic array — fast (~0.1s) and never touches the user's
    data, so the fallback is invisible to the caller.
    """
    requested = (requested or "cpu").lower()
    if requested == "cpu":
        return "cpu"
    if requested not in ("cuda", "gpu"):
        return "cpu"
    try:
        import lightgbm as lgb
        import numpy as _np
        rng = _np.random.default_rng(0)
        X = rng.standard_normal((64, 4)).astype(_np.float32)
        y = (X[:, 0] > 0).astype(int)
        m = lgb.LGBMClassifier(n_estimators=1, device="cuda", verbosity=-1)
        m.fit(X, y)
        return "cuda"
    except Exception:
        return "cpu"
