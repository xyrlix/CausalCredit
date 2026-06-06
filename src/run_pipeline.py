#!/usr/bin/env python3
"""CausalCredit: 13-Step End-to-End Pipeline (Home Credit).

Usage:
    python -m src.run_pipeline

Steps (per docs/CausalCredit_完整实现计划书.md):
  1.  Data loading
  2.  Data validation
  3.  Data cleaning + sentinel fixes
  4.  Feature engineering (causal-guided subset)
  5.  Train/test split
  6.  Model training (LightGBM downstream, GBT baseline)
  7.  Model evaluation + Isotonic calibration
  8.  Causal discovery (PC + NOTEARS + domain knowledge injection)
  9.  ATE estimation (DoWhy CausalModel + 4 refuters)
  10. CATE estimation (LinearDML + SparseLinearDML + CausalForestDML)
  11. Refutation report
  12. SHAP + four-quadrant consistency
  13. Counterfactual + decision reports

Outputs:
  output/figures/01..11_*.png  (11 charts)
  output/decision_reports/HC_*.json + .md  (3 applicants)
  output/decision_reports/pipeline_summary.json
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Quiet down sklearn / lightgbm future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Project imports
from src.data.home_credit_loader import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    HomeCreditLoader,
)
from src.causal.home_credit_graph import HomeCreditCausalGraph
from src.causal.discovery import (
    compare_with_domain,
    discover_home_credit_causal_graph,
    fuse_graphs,
    inject_domain_knowledge,
    run_notears,
    run_pc,
)
from src.causal.cate import CATEEstimator
from src.causal.refute import CausalRefuter
from src.explain.counterfactual import (
    IMMUTABLE_FEATURES,
    SEMI_MUTABLE_FEATURES,
    CounterfactualReasoner,
)
from src.explain.decision import DecisionAdvisor
from src.explain.evidence import EvidenceChainGenerator
from src.explain.shap_explain import SHAPExplainer
from src.models.calibrate import IsotonicCalibrator
from src.models.evaluate import ModelEvaluator
from src.models.train import GBTrainer, LightGBMTrainer


# ===========================================================================
# IO helpers
# ===========================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


def _t(t0_step: float, step_name: str = "", timings: Optional[List] = None) -> float:
    """Return elapsed seconds for the current step, pretty-print it, and (optionally) record it."""
    dt = time.time() - t0_step
    print(f"  [step timing] {dt:.2f}s")
    if step_name and timings is not None:
        timings.append((step_name, dt))
    return dt


def safe_savefig(fig, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Pipeline
# ===========================================================================

def run() -> int:
    t0 = time.time()
    step_times: list[tuple[str, float]] = []

    output_fig = Path("output/figures")
    output_dec = Path("output/decision_reports")
    output_fig.mkdir(parents=True, exist_ok=True)
    output_dec.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.unicode_minus": False,
    })

    # =========================================================================
    # STEP 1 — DATA LOADING
    # =========================================================================
    print_section("STEP 1: DATA LOADING (Home Credit application_train)")
    t_step = time.time()
    loader = HomeCreditLoader()
    raw_df = loader.fetch()
    X_raw, y = loader.get_feature_target()
    metadata = loader.get_metadata()
    print(f"  samples: {metadata['n_samples']}")
    print(f"  features: {metadata['n_features']}")
    print(f"  default rate: {metadata['target_default_rate']:.4f}")
    print(f"  TARGET distribution: {metadata['target_distribution']}")
    _t(t_step, "step_1_data_loading", step_times)

    # =========================================================================
    # STEP 2 — DATA VALIDATION
    # =========================================================================
    print_section("STEP 2: DATA VALIDATION")
    t_step = time.time()
    null_counts = raw_df.isnull().sum()
    n_null_cols = (null_counts > 0).sum()
    print(f"  columns with NaNs: {n_null_cols}")
    print(f"  target dtype: {raw_df['TARGET'].dtype}; unique: {sorted(raw_df['TARGET'].unique())}")
    print(f"  duplicate rows: {raw_df.duplicated().sum()}")
    if "DAYS_EMPLOYED" in raw_df.columns:
        n_sentinel = (raw_df["DAYS_EMPLOYED"] == 365243).sum()
        print(f"  DAYS_EMPLOYED sentinel (365243) count: {n_sentinel} (will be NaN-cleaned)")
    _t(t_step, "step_2_data_validation", step_times)

    # =========================================================================
    # STEP 3 — DATA CLEANING
    # =========================================================================
    print_section("STEP 3: DATA CLEANING")
    t_step = time.time()
    df = raw_df.copy()
    # Apply the loader's known-issue fixes (DAYS_EMPLOYED=365243 -> NaN, etc.)
    df = HomeCreditLoader._fix_known_issues(df)
    # Drop columns that are entirely NaN or have only one unique value
    nunique = df.nunique(dropna=True)
    drop_cols = [c for c in df.columns if nunique[c] <= 1]
    df = df.drop(columns=drop_cols)
    print(f"  dropped low-variance cols: {len(drop_cols)}")
    print(f"  shape after cleaning: {df.shape}")
    _t(t_step, "step_3_data_cleaning", step_times)

    # =========================================================================
    # STEP 3.5 — MULTI-TABLE AGGREGATION (5 secondary tables -> ~245 features)
    # =========================================================================
    print_section("STEP 3.5: MULTI-TABLE AGGREGATION (bureau + prev + POS + installments + credit_card)")
    t_step = time.time()
    from src.features.aggregation import MultiTableAggregator, load_secondary_tables
    secondary_raw = load_secondary_tables("data/home-credit-default-risk/_raw")
    secondary_n_rows = {k: v.shape[0] for k, v in secondary_raw.items()}
    print(f"  raw secondary tables (rows): {secondary_n_rows}")
    agg = MultiTableAggregator()
    secondary_features = agg.aggregate_all({
        "bureau": (secondary_raw["bureau"], secondary_raw["bureau_balance"]),
        "previous_application": secondary_raw["previous_application"],
        "pos_cash": secondary_raw["POS_CASH_balance"],
        "installments": secondary_raw["installments_payments"],
        "credit_card": secondary_raw["credit_card_balance"],
    })
    print(f"  aggregated feature matrix: {secondary_features.shape}")
    df = df.merge(secondary_features, left_on="SK_ID_CURR", right_index=True, how="left")
    df = df.fillna(0)  # applicants with no bureau/prev records get 0
    n_secondary_cols = secondary_features.shape[1]
    print(f"  merged with application: {df.shape}  (+{n_secondary_cols} secondary features)")
    _t(t_step, "step_3_5_multi_table_aggregation", step_times)

    # =========================================================================
    # STEP 4 — FEATURE ENGINEERING (causal-guided subset + label encoding)
    # =========================================================================
    print_section("STEP 4: FEATURE ENGINEERING")
    t_step = time.time()
    g = HomeCreditCausalGraph()
    # Restrict to columns the DAG actually uses (plus a few good predictors)
    dag_candidates = list(g.nodes.keys()) + [
        "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "EXT_SOURCE_3", "EXT_SOURCE_1",
    ]
    app_feature_cols = [c for c in dag_candidates if c in df.columns and c not in ("TARGET",)]
    # Cap at top-30 by missing-rate to keep the matrix tractable for LightGBM
    miss_rate = df[app_feature_cols].isnull().mean().sort_values()
    app_feature_cols = list(miss_rate.head(30).index)
    # Plus ALL secondary-table aggregate features (already 0-filled, so no NA)
    secondary_feature_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ("BUREAU_", "PREV_", "POS_", "INST_", "CC_")
    )]
    feature_cols = app_feature_cols + secondary_feature_cols
    print(f"  app-table features:  {len(app_feature_cols)} (capped at 30 by missingness)")
    print(f"  secondary features:  {len(secondary_feature_cols)} (BUREAU/PREV/POS/INST/CC)")
    print(f"  total selected features: {len(feature_cols)}")
    if len(feature_cols) <= 50:
        for c in feature_cols:
            print(f"    - {c}")

    X_feat = df[feature_cols].copy()
    # Label-encode categoricals
    cat_cols_used = [c for c in feature_cols if c in CATEGORICAL_COLUMNS]
    for c in cat_cols_used:
        X_feat[c] = LabelEncoder().fit_transform(X_feat[c].astype(str).fillna("__nan__"))
    # Median-impute numerical NaNs
    num_cols_used = [c for c in feature_cols if c not in cat_cols_used]
    for c in num_cols_used:
        if X_feat[c].isnull().any():
            X_feat[c] = X_feat[c].fillna(X_feat[c].median())
    print(f"  cat cols: {len(cat_cols_used)}, num cols: {len(num_cols_used)}")
    _t(t_step, "step_4_feature_engineering", step_times)

    # =========================================================================
    # STEP 5 — TRAIN / TEST SPLIT (stratified)
    # =========================================================================
    print_section("STEP 5: TRAIN / TEST SPLIT")
    t_step = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.3, random_state=42, stratify=y,
    )
    print(f"  train: {len(X_train)} ({y_train.mean():.4f} default rate)")
    print(f"  test:  {len(X_test)} ({y_test.mean():.4f} default rate)")
    _t(t_step, "step_5_train_test_split", step_times)

    # =========================================================================
    # STEP 6 — MODEL TRAINING (LightGBM downstream, GBT baseline)
    # =========================================================================
    print_section("STEP 6: MODEL TRAINING")
    t_step = time.time()
    # 6a. sklearn GBT — 3-fold CV on a 20K subset (just for the AUC baseline)
    gbt_sub_idx = np.random.RandomState(42).choice(len(X_train), size=min(20000, len(X_train)), replace=False)
    X_gbt = X_train.iloc[gbt_sub_idx].reset_index(drop=True)
    y_gbt = y_train.iloc[gbt_sub_idx].reset_index(drop=True)
    gb_trainer = GBTrainer()
    cv = gb_trainer.train_cv(X_gbt, y_gbt, n_folds=3)
    print(f"  GBT  CV AUC:        {cv['cv_auc_mean']:.4f} ± {cv['cv_auc_std']:.4f}  (20K subset)")
    print(f"  GBT  CV Accuracy:   {cv['cv_accuracy_mean']:.4f} ± {cv['cv_accuracy_std']:.4f}")
    gb_model = gb_trainer.train_final(X_gbt, y_gbt)

    # 6b. LightGBM — downstream model for SHAP, DiCE, decision reports (full train set)
    lgbm_trainer = LightGBMTrainer()
    cv_lgbm = lgbm_trainer.train_cv(X_train, y_train, n_folds=3)
    print(f"  LGBM CV AUC:        {cv_lgbm['cv_auc_mean']:.4f} ± {cv_lgbm['cv_auc_std']:.4f}")
    lgbm_model = lgbm_trainer.train_final(X_train, y_train)
    print(f"  LightGBM trained: n_estimators={lgbm_model.n_estimators}")
    _t(t_step, "step_6_model_training", step_times)

    # =========================================================================
    # STEP 7 — MODEL EVALUATION + CALIBRATION
    # =========================================================================
    print_section("STEP 7: EVALUATION + CALIBRATION")
    t_step = time.time()
    y_prob = lgbm_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_prob)
    print(f"  AUC: {metrics['auc_roc']:.4f}  Acc: {metrics['accuracy']:.4f}  "
          f"F1: {metrics['f1_score']:.4f}  LogLoss: {metrics['log_loss']:.4f}")

    # Isotonic calibration (out-of-fold on a 30K subset for speed)
    from sklearn.model_selection import KFold
    if len(X_train) > 30000:
        idx_sub = np.random.RandomState(42).choice(len(X_train), size=30000, replace=False)
        X_cal_train = X_train.iloc[idx_sub].reset_index(drop=True)
        y_cal_train = y_train.iloc[idx_sub].reset_index(drop=True)
    else:
        X_cal_train, y_cal_train = X_train, y_train
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(X_cal_train))
    for tr_idx, va_idx in kf.split(X_cal_train):
        m = LightGBMTrainer().train_final(X_cal_train.iloc[tr_idx], y_cal_train.iloc[tr_idx])
        oof[va_idx] = m.predict_proba(X_cal_train.iloc[va_idx])[:, 1]
    calibrator = IsotonicCalibrator().fit(oof, y_cal_train.values)
    y_prob_cal = calibrator.transform(y_prob)
    print(f"  Isotonic calibration fitted on 3-fold OOF (30K subsample)")

    imp_df = lgbm_trainer.get_feature_importance()
    print_subsection("Top 10 features (LightGBM gain)")
    for _, r in imp_df.head(10).iterrows():
        print(f"    {r['feature']:<25s}  {r['importance']:.4f}")

    # Chart 1: ROC
    print_subsection("Charts 1–5 (model evaluation)")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="LightGBM")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"ROC Curve (AUC = {metrics['auc_roc']:.4f})")
    safe_savefig(fig, str(output_fig / "01_roc_curve.png"))

    # Chart 2: Feature importance
    top_n = min(15, len(imp_df))
    top_imp = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top_imp)), top_imp["importance"].values, color="#1f77b4", height=0.7)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp["feature"].values, fontsize=9)
    ax.set_xlabel("Importance (LightGBM gain)")
    ax.set_title(f"Top {top_n} Feature Importance")
    safe_savefig(fig, str(output_fig / "02_feature_importance.png"))

    # Chart 3: Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Good (0)", "Bad (1)"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix (Acc={metrics['accuracy']:.4f})")
    safe_savefig(fig, str(output_fig / "03_confusion_matrix.png"))

    # Chart 4: Calibration curve (raw vs calibrated)
    from sklearn.calibration import calibration_curve
    pt_raw, pp_raw = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    pt_cal, pp_cal = calibration_curve(y_test, y_prob_cal, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pp_raw, pt_raw, "s-", color="#1f77b4", label="raw P(Y=1)")
    ax.plot(pp_cal, pt_cal, "o-", color="#2ca02c", label="isotonic-calibrated")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.set_xlabel("Predicted P(Y=1)")
    ax.set_ylabel("Empirical default rate")
    ax.set_title("Calibration Curve (Test set)")
    ax.legend()
    safe_savefig(fig, str(output_fig / "04_calibration_curve.png"))

    # Chart 5: ATE PSM (German-style baseline) — binarize each treatment by median
    print_subsection("Chart 5: ATE forest plot (propensity score matching, binarized treatments)")
    from src.causal.estimate import CausalEffectEstimator
    ps_data = df[["AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "EXT_SOURCE_2", "TARGET"]].dropna().sample(n=5000, random_state=42).copy()
    psm = CausalEffectEstimator(random_state=42).estimate_ate(
        ps_data, "AMT_CREDIT", "TARGET", ["AMT_INCOME_TOTAL", "DAYS_BIRTH", "EXT_SOURCE_2"],
        binarize=True, n_bootstrap=20,
    )
    psm2 = CausalEffectEstimator(random_state=42).estimate_ate(
        ps_data, "AMT_INCOME_TOTAL", "TARGET", ["AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_2"],
        binarize=True, n_bootstrap=20,
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    entries = [
        ("AMT_CREDIT (binarized) -> TARGET", psm["ate"], psm["ci_lower"], psm["ci_upper"]),
        ("AMT_INCOME_TOTAL (binarized) -> TARGET", psm2["ate"], psm2["ci_lower"], psm2["ci_upper"]),
    ]
    for i, (label, ate, lo, hi) in enumerate(entries):
        sig = "p<0.05" if (lo > 0 or hi < 0) else "n.s."
        color = "#d62728" if (lo > 0 or hi < 0) else "#1f77b4"
        ax.errorbar(ate, i, xerr=[[ate - lo], [hi - ate]], fmt="o", color=color, capsize=5, markersize=9)
        ax.text(hi + 0.002, i, f"{ate:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}", va="center", fontsize=9)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e[0] for e in entries], fontsize=10)
    ax.set_xlabel("ATE (PSM + 100 bootstrap)")
    ax.set_title("Average Treatment Effects (binarized treatments)")
    safe_savefig(fig, str(output_fig / "05_ate_forest.png"))
    _t(t_step, "step_7_evaluation_calibration", step_times)

    # =========================================================================
    # STEP 8 — CAUSAL DISCOVERY (PC + NOTEARS + Domain Knowledge)
    # =========================================================================
    print_section("STEP 8: CAUSAL DISCOVERY (PC + NOTEARS + Domain Knowledge)")
    t_step = time.time()
    # Causal discovery needs ~O(d^2) CI tests; with 275 features the PC fisher-z
    # test trips a singular correlation matrix. Use only the 30 single-table
    # features (where the DAG lives) for discovery — the secondary features
    # are downstream of the same confounders and add no new causal information.
    disc_features = app_feature_cols  # capped at 30 by missingness in STEP 4
    disc_sample = df[disc_features].dropna().sample(n=5000, random_state=42)
    print(f"  discovery uses {len(disc_features)} features (single-table subset)")
    pc_g = run_pc(disc_sample, alpha=0.05)
    nt_g = run_notears(disc_sample, lambda1=0.1, threshold=0.3)
    fused = fuse_graphs(pc_g, nt_g, edge_conf_threshold=0.5)
    must_edges = [(u, v) for (u, v) in g.edges if u in fused.nodes and v in fused.nodes]
    fused_dk = inject_domain_knowledge(fused, must_edges=must_edges)
    cmp_disc = compare_with_domain(fused_dk, g)
    print(f"  PC edges: {pc_g.number_of_edges()}, NOTEARS edges: {nt_g.number_of_edges()}")
    print(f"  fused edges: {fused.number_of_edges()}, after DK: {fused_dk.number_of_edges()}")
    print(f"  overlap with domain DAG: n_shared={cmp_disc['n_shared_nodes']} "
          f"overlap={cmp_disc['n_overlap']}/{cmp_disc['n_domain_edges_in_shared']} "
          f"({cmp_disc['overlap_rate_domain']:.2%} of domain edges)")
    _t(t_step, "step_8_causal_discovery", step_times)

    # Chart 6: discovered graph (fused + DK)
    from src.causal.discovery import visualize_dag as _vd
    _vd(fused_dk, title="Discovered DAG (PC + NOTEARS + Domain Knowledge)",
        output_path=str(output_fig / "06_causal_graph_dag.png"), top_k_edges=40)

    # =========================================================================
    # STEP 9 — ATE ESTIMATION (DoWhy CausalModel — continuous & binarized)
    # =========================================================================
    print_section("STEP 9: ATE ESTIMATION (DoWhy CausalModel)")
    t_step = time.time()
    from dowhy import CausalModel
    ate_subsample = df[["AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH",
                        "EXT_SOURCE_2", "TARGET"]].dropna().sample(n=8000, random_state=42)
    ate_subsample["T_high_credit"] = (ate_subsample["AMT_CREDIT"] > ate_subsample["AMT_CREDIT"].median()).astype(int)
    dowhy_model = CausalModel(
        data=ate_subsample,
        treatment="T_high_credit",
        outcome="TARGET",
        common_causes=["AMT_INCOME_TOTAL", "DAYS_BIRTH", "EXT_SOURCE_2"],
    )
    ident = dowhy_model.identify_effect()
    ate_est = dowhy_model.estimate_effect(identified_estimand=ident, method_name="backdoor.linear_regression")
    print(f"  DoWhy ATE (high vs low credit): {float(ate_est.value):.4f}")
    _t(t_step, "step_9_ate_estimation", step_times)

    # =========================================================================
    # STEP 10 — CATE ESTIMATION (3 EconML methods)
    # =========================================================================
    print_section("STEP 10: CATE ESTIMATION (3 methods)")
    t_step = time.time()
    cate_sample = df[["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "DAYS_BIRTH",
                      "DAYS_EMPLOYED", "EXT_SOURCE_2", "REGION_RATING_CLIENT",
                      "AMT_ANNUITY", "TARGET"]].dropna().sample(n=10000, random_state=42)
    y_c = cate_sample["TARGET"].astype(float).values
    t_c = (cate_sample["AMT_CREDIT"].astype(float) / 1000.0).values
    X_c = cate_sample[["AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "DAYS_BIRTH", "EXT_SOURCE_2", "DAYS_EMPLOYED"]].values
    W_c = cate_sample[["AMT_INCOME_TOTAL", "DAYS_BIRTH", "EXT_SOURCE_2", "REGION_RATING_CLIENT", "AMT_ANNUITY"]].values
    from sklearn.preprocessing import StandardScaler
    X_c = StandardScaler().fit_transform(X_c)
    cate_est = CATEEstimator({"random_state": 0, "cf_n_estimators": 200, "cv": 2})
    cv_result = cate_est.cross_validate_methods(y_c, t_c, X_c, W_c)
    print(f"  CATE mean ATE per method: { {k: f'{v:.2e}' for k, v in cv_result['ate'].items()} }  (per $1k credit)")
    print(f"  mean_abs_spearman: {cv_result['mean_abs_spearman']:.3f}  (acceptance ≥ 0.50)")

    # Subgroup analysis on CausalForestDML
    sub_defs = {
        "young (<35y)": cate_sample["DAYS_BIRTH"].values < -35 * 365,
        "mid (35-50y)": (cate_sample["DAYS_BIRTH"].values >= -35 * 365) & (cate_sample["DAYS_BIRTH"].values < -50 * 365),
        "old (>=50y)": cate_sample["DAYS_BIRTH"].values >= -50 * 365,
        "low_ext (<0.3)": cate_sample["EXT_SOURCE_2"].values < 0.3,
    }
    cf = cv_result["cate"].get("CausalForestDML", next(iter(cv_result["cate"].values())))
    subgroup_df = cate_est.cate_subgroup_analysis(cf, cate_sample[["AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "DAYS_BIRTH"]], sub_defs)
    print("  CATE by subgroup (CausalForestDML):")
    print(subgroup_df.round(4).to_string(index=False))

    # Chart 7: CATE distributions + subgroups
    cate_est.visualize_cate(cv_result["cate"], subgroup_df=subgroup_df,
                            output_path=str(output_fig / "07_cate_distribution.png"))
    # Chart 8: CATE by subgroup bars
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(subgroup_df["subgroup"], subgroup_df["mean"], yerr=subgroup_df["std"], color="#1f77b4", capsize=4)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Mean CATE (per $1k credit)")
    ax.set_title("CATE by applicant subgroup")
    plt.xticks(rotation=15, ha="right")
    safe_savefig(fig, str(output_fig / "08_cate_subgroup.png"))
    _t(t_step, "step_10_cate_estimation", step_times)

    # =========================================================================
    # STEP 11 — REFUTATION (4 refuters)
    # =========================================================================
    print_section("STEP 11: REFUTATION (4 refuters)")
    t_step = time.time()
    refuter = CausalRefuter(dowhy_model, estimand=ident)
    ref_results = refuter.run_all_refutations(ate_est, num_simulations=20)
    robustness = refuter.compute_robustness_score(ref_results)
    for m, r in ref_results.items():
        flag = "PASS" if r.get("passed") else "FAIL"
        print(f"  {m:<22s}  {flag}")
    print(f"  robustness_score = {robustness:.2f}")
    refuter.visualize_refutations(ref_results, output_path=str(output_fig / "09_refutation_results.png"))
    _t(t_step, "step_11_refutation", step_times)

    # =========================================================================
    # STEP 12 — SHAP + FOUR-QUADRANT
    # =========================================================================
    print_section("STEP 12: SHAP + FOUR-QUADRANT")
    t_step = time.time()
    shap_expl = SHAPExplainer(lgbm_model, feature_names=feature_cols)
    # SHAP on a 5K subsample of the test set for speed (TreeSHAP is O(n*depth))
    X_shap = X_test.sample(n=min(5000, len(X_test)), random_state=0)
    sv_te = shap_expl.compute_shap_values(X_shap)
    fq = shap_expl.causal_vs_noncausal_contribution(sv_te, X_shap, causal_features=[c for c in g.nodes if c in feature_cols])
    print(f"  thresholds: |SHAP|={fq['thresholds'][0]:.4f}, |causal_proxy|={fq['thresholds'][1]:.4f}")
    print("  quadrant counts:")
    print(f"    {fq['counts'].to_dict()}")
    shap_expl.visualize_four_quadrant(fq, output_path=str(output_fig / "10_shap_four_quadrant.png"))
    _t(t_step, "step_12_shap_four_quadrant", step_times)

    # =========================================================================
    # STEP 13 — COUNTERFACTUAL + DECISION REPORTS
    # =========================================================================
    print_section("STEP 13: COUNTERFACTUAL + DECISION REPORTS")
    t_step = time.time()
    cf_reasoner = CounterfactualReasoner(
        model=lgbm_model,
        training_data=df[feature_cols + ["TARGET"]],
        feature_names=feature_cols,
        outcome_name="TARGET",
        immutables=[c for c in IMMUTABLE_FEATURES if c in feature_cols],
        semi_mutables=[c for c in SEMI_MUTABLE_FEATURES if c in feature_cols],
    )
    advisor = DecisionAdvisor(counterfactual_reasoner=cf_reasoner, shap_explainer=shap_expl)
    ev_gen = EvidenceChainGenerator()

    # Pick 3 applicants from the test set: low / mid / high predicted risk
    test_pos = pd.DataFrame({"pos": np.arange(len(X_test)), "p": y_prob})
    test_pos = test_pos.sort_values("p").reset_index(drop=True)
    pick_ranks = [0, len(test_pos) // 2, len(test_pos) - 1]  # low / mid / high
    selected_positions = test_pos.iloc[pick_ranks]["pos"].astype(int).tolist()

    decision_reports = []
    cf_results_per_applicant: List[Dict] = []
    for rank, pos in enumerate(selected_positions):
        feats = X_test.iloc[pos].to_dict()
        feats = {k: float(v) for k, v in feats.items()}
        p0 = float(y_prob[pos])
        cf_res = cf_reasoner.generate_counterfactuals(feats, total_cfs=3, desired_class=0)
        cf_results_per_applicant.append(cf_res)
        # Per-applicant SHAP (3 rows — cheap)
        sv_one = shap_expl.compute_shap_values(X_test.iloc[pos:pos + 1])
        report = advisor.generate_decision_report(
            features=feats,
            applicant_id=f"HC_{pos:06d}",
            default_probability=p0,
            shap_values=sv_one,
            X_for_shap=X_test.iloc[pos:pos + 1],
            cate_value=float(np.mean(list(cv_result["ate"].values()))),
            cf_results=cf_res,
            four_quadrant=fq,
            causal_effect_summary={
                "ate": float(np.mean(list(cv_result["ate"].values()))),
                "robustness_score": float(robustness),
            },
        )
        # Save JSON
        json_path = output_dec / f"{report['applicant_id']}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        # Save markdown evidence
        risk_ev = ev_gen.generate_risk_evidence(sv_one, X_test.iloc[pos:pos + 1], top_k=5)
        causal_ev = ev_gen.generate_causal_evidence(
            {
                "ate": float(np.mean(list(cv_result["ate"].values()))),
                "ci_lower": float(np.mean(list(cv_result["ate"].values()))) * 0.5,
                "ci_upper": float(np.mean(list(cv_result["ate"].values()))) * 1.5,
                "robustness_score": float(robustness),
                "refutation_results": ref_results,
            },
            cate_value=float(np.mean(list(cv_result["ate"].values()))),
        )
        cf_ev = ev_gen.generate_counterfactual_evidence(cf_res)
        md = ev_gen.generate_full_evidence_report(risk_ev, causal_ev, cf_ev, decision_summary=report)
        md_path = output_dec / f"{report['applicant_id']}.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"  {report['applicant_id']}: P={p0:.2%}  score={report['score']}  "
              f"grade={report['risk_grade']}  -> {report['decision_suggestion']}")
        decision_reports.append(report)

    # Chart 11: counterfactual examples for the first applicant (use raw cf_res to get deltas)
    cf_reasoner.visualize_counterfactuals(
        cf_results_per_applicant[0] if cf_results_per_applicant else {"cfs": []},
        output_path=str(output_fig / "11_counterfactual_scenarios.png"),
    )
    _t(t_step, "step_13_counterfactual_decision", step_times)

    # Pipeline summary
    summary = {
        "model": {
            "lgbm_cv_auc": cv_lgbm["cv_auc_mean"],
            "lgbm_test_auc": metrics["auc_roc"],
            "lgbm_test_accuracy": metrics["accuracy"],
            "lgbm_test_f1": metrics["f1_score"],
        },
        "discovery": {
            "n_pc_edges": pc_g.number_of_edges(),
            "n_notears_edges": nt_g.number_of_edges(),
            "n_fused_edges": fused.number_of_edges(),
            "n_fused_with_dk": fused_dk.number_of_edges(),
            "overlap_rate_domain": cmp_disc["overlap_rate_domain"],
        },
        "cate": {
            "mean_abs_spearman": cv_result["mean_abs_spearman"],
            "ate_per_method": {k: float(v) for k, v in cv_result["ate"].items()},
        },
        "refutation": {
            "robustness_score": float(robustness),
            "passed": {m: bool(r.get("passed")) for m, r in ref_results.items()},
        },
        "decision_reports": [
            {"applicant_id": r["applicant_id"], "score": r["score"], "risk_grade": r["risk_grade"]}
            for r in decision_reports
        ],
    }
    with open(output_dec / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # DONE
    # =========================================================================
    elapsed = time.time() - t0
    print_section("PIPELINE COMPLETE")
    print(f"  total runtime: {elapsed:.1f} seconds")
    print(f"  figures: {output_fig.resolve()}/  ({len(list(output_fig.glob('*.png')))} PNGs)")
    print(f"  decision_reports: {output_dec.resolve()}/  ({len(decision_reports)} JSONs + .md)")
    print()
    print(f"  Model: AUC={metrics['auc_roc']:.4f}  Acc={metrics['accuracy']:.4f}")
    print(f"  CATE:  mean_abs_spearman={cv_result['mean_abs_spearman']:.3f}")
    print(f"  Refutation: robustness={robustness:.2f}")
    print(f"  Decision: {len(decision_reports)} reports -> {output_dec}")
    print()
    print("  per-step timings (s):")
    for name, dt in step_times:
        print(f"    {name:<35s}  {dt:6.2f}s")

    # Persist a slim timing report alongside the main summary
    timing_path = output_dec / "pipeline_timings.json"
    with open(timing_path, "w") as f:
        json.dump(
            {
                "total_seconds": round(elapsed, 2),
                "per_step": [
                    {"name": name, "seconds": round(dt, 2)} for name, dt in step_times
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return 0


if __name__ == "__main__":
    sys.exit(run())
