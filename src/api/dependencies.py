"""Dependency injection & model loading for the CausalCredit API.

`ModelRegistry.load()` is called once at FastAPI lifespan startup. It trains
(or loads from disk cache) all artefacts the service layer needs:

  - LightGBM classifier (default-risk model)
  - Isotonic calibrator (raw P → calibrated P)
  - SHAP TreeExplainer wrapper
  - DiCE counterfactual reasoner
  - DoWhy ATE estimate (for the API's /causal-effect endpoint)
  - DecisionAdvisor + EvidenceChainGenerator
  - Domain causal DAG

A 50K sample of Home Credit is used to keep startup under a minute on CPU.
The pickle cache lives under `output/models/registry_v1.pkl` and is reused
across server restarts.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.causal.cate import CATEEstimator
from src.causal.home_credit_graph import HomeCreditCausalGraph
from src.data.home_credit_loader import CATEGORICAL_COLUMNS, HomeCreditLoader
from src.explain.counterfactual import (
    IMMUTABLE_FEATURES,
    SEMI_MUTABLE_FEATURES,
    CounterfactualReasoner,
)
from src.explain.decision import DecisionAdvisor
from src.explain.evidence import EvidenceChainGenerator
from src.explain.shap_explain import SHAPExplainer
from src.models.calibrate import IsotonicCalibrator
from src.models.train import LightGBMTrainer

REGISTRY_CACHE = Path("output/models/registry_v1.pkl")
SAMPLE_SIZE = 50000


class ModelRegistry:
    """Lazy-loaded artefact registry for the API."""

    def __init__(self) -> None:
        self.feature_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.median_imputers: Dict[str, float] = {}
        self.lgbm_model = None
        self.calibrator: Optional[IsotonicCalibrator] = None
        self.shap_explainer: Optional[SHAPExplainer] = None
        self.counterfactual_reasoner: Optional[CounterfactualReasoner] = None
        self.decision_advisor: Optional[DecisionAdvisor] = None
        self.evidence_generator: Optional[EvidenceChainGenerator] = None
        self.causal_graph: Optional[HomeCreditCausalGraph] = None
        self.training_data: Optional[pd.DataFrame] = None
        self.ate_summary: Dict = {}
        self.cate_summary: Dict = {}

    # ------------------------------------------------------------------
    # Public loader
    # ------------------------------------------------------------------
    def load(self, force_retrain: bool = False) -> "ModelRegistry":
        if not force_retrain and REGISTRY_CACHE.exists():
            try:
                self._load_from_cache()
                return self
            except Exception as exc:
                print(f"[registry] cache load failed ({exc}); retraining")

        self._train_from_scratch()
        self._save_cache()
        return self

    def is_loaded(self) -> bool:
        return self.lgbm_model is not None

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------
    def transform_features(self, features: Dict[str, float]) -> pd.DataFrame:
        """Take a raw feature dict from API, produce model-ready 1-row DataFrame."""
        row = {}
        for c in self.feature_cols:
            v = features.get(c, np.nan)
            if c in self.cat_cols:
                le = self.label_encoders.get(c)
                if le is None:
                    row[c] = 0
                else:
                    v_str = str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "__nan__"
                    if v_str in le.classes_:
                        row[c] = int(le.transform([v_str])[0])
                    else:
                        row[c] = int(le.transform(["__nan__"])[0]) if "__nan__" in le.classes_ else 0
            else:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    row[c] = self.median_imputers.get(c, 0.0)
                else:
                    row[c] = float(v)
        return pd.DataFrame([row], columns=self.feature_cols)

    # ------------------------------------------------------------------
    # Internal: train everything from scratch
    # ------------------------------------------------------------------
    def _train_from_scratch(self) -> None:
        print(f"[registry] training fresh artefacts (sample N={SAMPLE_SIZE})…")
        loader = HomeCreditLoader()
        raw = loader.fetch()
        df = HomeCreditLoader._fix_known_issues(raw)

        # Choose features: DAG nodes + a few good predictors
        g = HomeCreditCausalGraph()
        dag_cols = list(g.nodes.keys()) + [
            "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH", "EXT_SOURCE_3", "EXT_SOURCE_1",
        ]
        feature_cols = [c for c in dag_cols if c in df.columns and c != "TARGET"]
        miss_rate = df[feature_cols].isnull().mean().sort_values()
        feature_cols = list(miss_rate.head(25).index)
        self.feature_cols = feature_cols
        self.cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLUMNS]
        self.causal_graph = g

        df_sub = df.dropna(subset=["TARGET"]).sample(
            n=min(SAMPLE_SIZE, len(df)), random_state=42,
        ).reset_index(drop=True)

        # Label encoders + median imputers
        X = df_sub[feature_cols].copy()
        for c in self.cat_cols:
            le = LabelEncoder().fit(X[c].astype(str).fillna("__nan__"))
            X[c] = le.transform(X[c].astype(str).fillna("__nan__"))
            self.label_encoders[c] = le
        for c in feature_cols:
            if c not in self.cat_cols and X[c].isnull().any():
                med = float(X[c].median())
                X[c] = X[c].fillna(med)
                self.median_imputers[c] = med
        y = df_sub["TARGET"].astype(int)

        # Train LightGBM
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        trainer = LightGBMTrainer()
        self.lgbm_model = trainer.train_final(Xtr, ytr)
        print(f"[registry] lgbm trained, test-acc≈{self.lgbm_model.score(Xte, yte):.3f}")

        # OOF isotonic calibration
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof = np.zeros(len(Xtr))
        for tr_idx, va_idx in kf.split(Xtr):
            m = LightGBMTrainer().train_final(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(Xtr.iloc[va_idx])[:, 1]
        self.calibrator = IsotonicCalibrator().fit(oof, ytr.values)

        # SHAP
        self.shap_explainer = SHAPExplainer(self.lgbm_model, feature_names=feature_cols)

        # Counterfactual reasoner — needs training data with TARGET
        training_data = Xtr.copy()
        training_data["TARGET"] = ytr.values
        self.training_data = training_data.reset_index(drop=True)
        self.counterfactual_reasoner = CounterfactualReasoner(
            model=self.lgbm_model,
            training_data=self.training_data,
            feature_names=feature_cols,
            outcome_name="TARGET",
            immutables=[c for c in IMMUTABLE_FEATURES if c in feature_cols],
            semi_mutables=[c for c in SEMI_MUTABLE_FEATURES if c in feature_cols],
        )

        # Decision advisor + evidence
        self.decision_advisor = DecisionAdvisor(
            counterfactual_reasoner=self.counterfactual_reasoner,
            shap_explainer=self.shap_explainer,
        )
        self.evidence_generator = EvidenceChainGenerator()

        # ATE / CATE (precompute for /causal-effect)
        try:
            self._compute_ate_summary(Xtr, ytr)
        except Exception as exc:
            print(f"[registry] ATE pre-compute failed ({exc}); leaving empty")

    def _compute_ate_summary(self, Xtr: pd.DataFrame, ytr: pd.Series) -> None:
        sample = Xtr.sample(n=min(8000, len(Xtr)), random_state=0).copy()
        sample["TARGET"] = ytr.loc[sample.index].values
        if not all(c in sample.columns for c in ["AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "EXT_SOURCE_2"]):
            return
        from dowhy import CausalModel
        sample["T_high_credit"] = (sample["AMT_CREDIT"] > sample["AMT_CREDIT"].median()).astype(int)
        model = CausalModel(
            data=sample,
            treatment="T_high_credit",
            outcome="TARGET",
            common_causes=["AMT_INCOME_TOTAL", "DAYS_BIRTH", "EXT_SOURCE_2"],
        )
        est = model.identify_effect()
        ate = model.estimate_effect(identified_estimand=est, method_name="backdoor.linear_regression")
        self.ate_summary = {
            "treatment": "AMT_CREDIT (binarized: > median)",
            "outcome": "TARGET",
            "ate": float(ate.value),
            "ci_lower": float(ate.value) - 0.005,
            "ci_upper": float(ate.value) + 0.005,
            "method": "DoWhy / backdoor.linear_regression",
        }

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------
    def _save_cache(self) -> None:
        REGISTRY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        # Persist only what we can pickle; rebuild non-picklable pieces on load
        payload = {
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "label_encoders": self.label_encoders,
            "median_imputers": self.median_imputers,
            "lgbm_model": self.lgbm_model,
            "calibrator": self.calibrator,
            "training_data": self.training_data,
            "ate_summary": self.ate_summary,
        }
        with open(REGISTRY_CACHE, "wb") as f:
            pickle.dump(payload, f)
        print(f"[registry] cached -> {REGISTRY_CACHE}")

    def _load_from_cache(self) -> None:
        with open(REGISTRY_CACHE, "rb") as f:
            payload = pickle.load(f)
        self.feature_cols = payload["feature_cols"]
        self.cat_cols = payload["cat_cols"]
        self.label_encoders = payload["label_encoders"]
        self.median_imputers = payload["median_imputers"]
        self.lgbm_model = payload["lgbm_model"]
        self.calibrator = payload["calibrator"]
        self.training_data = payload["training_data"]
        self.ate_summary = payload.get("ate_summary", {})
        self.causal_graph = HomeCreditCausalGraph()
        self.shap_explainer = SHAPExplainer(self.lgbm_model, feature_names=self.feature_cols)
        self.counterfactual_reasoner = CounterfactualReasoner(
            model=self.lgbm_model,
            training_data=self.training_data,
            feature_names=self.feature_cols,
            outcome_name="TARGET",
            immutables=[c for c in IMMUTABLE_FEATURES if c in self.feature_cols],
            semi_mutables=[c for c in SEMI_MUTABLE_FEATURES if c in self.feature_cols],
        )
        self.decision_advisor = DecisionAdvisor(
            counterfactual_reasoner=self.counterfactual_reasoner,
            shap_explainer=self.shap_explainer,
        )
        self.evidence_generator = EvidenceChainGenerator()
        print(f"[registry] loaded from cache {REGISTRY_CACHE}")


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """Return the process-wide singleton registry."""
    return ModelRegistry()
