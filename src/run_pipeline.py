#!/usr/bin/env python3
"""CausalCredit: End-to-End Credit Default Causal Analysis Pipeline.

Usage:
    python -m src.run_pipeline

This pipeline:
1. Loads German Credit data from sklearn's fetch_openml
2. Cleans and encodes the data
3. Builds ML features and causal features
4. Trains a GradientBoostingClassifier as base model
5. Defines a causal DAG for credit default analysis
6. Estimates ATE using manual propensity score matching with bootstrap CIs
7. Evaluates and prints comprehensive results
"""

import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, GermanCreditLoader
from src.data.preprocessing.cleaner import DataCleaner
from src.data.validator import generate_data_report, validate_no_nulls, validate_target
from src.features.builder import FeatureBuilder
from src.models.evaluate import ModelEvaluator
from src.models.train import GBTrainer
from src.causal.estimate import CausalEffectEstimator
from src.causal.graph import CreditCausalGraph
from src.causal.variable_validation import CausalVariableValidator


def print_section(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n  --- {title} ---")


def run() -> int:
    """Execute the full CausalCredit pipeline."""
    start_time = time.time()

    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================
    print_section("STEP 1: DATA LOADING")
    loader = GermanCreditLoader()
    raw_df = loader.fetch()
    X_raw, y = loader.get_feature_target()
    metadata = loader.get_metadata()

    print(f"  Dataset: German Credit (from sklearn fetch_openml)")
    print(f"  Samples: {metadata['n_samples']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Target distribution: {metadata['target_distribution']}")
    print(f"  Categorical columns: {len(CATEGORICAL_COLUMNS)}")
    print(f"  Numerical columns: {len(NUMERICAL_COLUMNS)}")

    # =========================================================================
    # 2. DATA VALIDATION
    # =========================================================================
    print_section("STEP 2: DATA VALIDATION")
    null_check = validate_no_nulls(raw_df)
    print(f"  Null values present: {null_check['has_nulls']}")

    target_check = validate_target(raw_df)
    if target_check["valid"]:
        print(f"  Target valid: yes, distribution: {target_check['distribution']}")

    report = generate_data_report(raw_df)
    print(f"  Data report: {len(report)} columns checked")

    # =========================================================================
    # 3. DATA CLEANING
    # =========================================================================
    print_section("STEP 3: DATA CLEANING")
    cleaner = DataCleaner()
    X_clean = cleaner.clean(X_raw, numerical_cols=NUMERICAL_COLUMNS)
    print(f"  Shape after cleaning: {X_clean.shape}")
    print(f"  Remaining nulls: {X_clean.isnull().sum().sum()}")

    # =========================================================================
    # 4. FEATURE ENGINEERING
    # =========================================================================
    print_section("STEP 4: FEATURE ENGINEERING")
    feature_builder = FeatureBuilder()
    X_features = feature_builder.build(
        X_clean,
        categorical_cols=CATEGORICAL_COLUMNS,
        numerical_cols=NUMERICAL_COLUMNS,
        fit=True,
    )
    feature_names = feature_builder.get_feature_names()
    print(f"  Total features after encoding + causal features: {len(feature_names)}")
    print(f"  Feature columns: {feature_names}")

    # =========================================================================
    # 5. TRAIN/TEST SPLIT
    # =========================================================================
    print_section("STEP 5: TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.3, random_state=42, stratify=y,
    )
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set:  {len(X_test)} samples")
    print(f"  Train target rate: {y_train.mean():.4f}")
    print(f"  Test target rate:  {y_test.mean():.4f}")

    # =========================================================================
    # 6. MODEL TRAINING
    # =========================================================================
    print_section("STEP 6: MODEL TRAINING (GradientBoostingClassifier)")

    trainer = GBTrainer()
    cv_results = trainer.train_cv(X_train, y_train, n_folds=5)
    print(f"  CV AUC (5-fold):     {cv_results['cv_auc_mean']:.4f} ± {cv_results['cv_auc_std']:.4f}")
    print(f"  CV Accuracy (5-fold): {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")

    model = trainer.train_final(X_train, y_train)
    y_prob = trainer.predict_proba(X_test)
    y_pred = trainer.predict(X_test)

    # =========================================================================
    # 7. MODEL EVALUATION
    # =========================================================================
    print_section("STEP 7: MODEL EVALUATION")

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_prob)

    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  Log Loss:     {metrics['log_loss']:.4f}")

    # Feature importance
    imp_df = trainer.get_feature_importance()
    print_subsection("Top 10 Feature Importance")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:<30s}: {row['importance']:.4f}")

    # =========================================================================
    # 8. CAUSAL DAG DEFINITION
    # =========================================================================
    print_section("STEP 8: CAUSAL DAG DEFINITION")

    graph = CreditCausalGraph()
    treatments = graph.get_treatment_variables()
    outcome = graph.get_outcome_variable()
    confounders = graph.get_confounders(treatments[0], outcome)

    print(f"  Treatments:  {treatments}")
    print(f"  Outcome:     {outcome}")
    print(f"  Confounders: {confounders}")
    print(f"  DAG is acyclic: {graph.validate_acyclic()}")

    # Causal assumptions
    print_subsection("Key Assumptions")
    for i, assumption in enumerate(graph.get_assumptions(), 1):
        print(f"    {i}. {assumption}")

    # =========================================================================
    # 9. CAUSAL VARIABLE VALIDATION
    # =========================================================================
    print_section("STEP 9: CAUSAL VARIABLE VALIDATION")

    validator = CausalVariableValidator()
    tx_validation = validator.validate_treatment_variables(X_raw, treatments)
    conf_validation = validator.validate_confounders(
        X_raw, treatments, "default", confounders,
    )

    has_default_col = "default" in X_raw.columns or True
    default_col = "default" if has_default_col else outcome
    if default_col not in X_raw.columns:
        X_raw["default"] = y.values

    for tx_name, info in tx_validation.items():
        if info.get("present"):
            print(f"  [{tx_name}] present, mean={info.get('mean', 'N/A')}, "
                  f"median={info.get('median', 'N/A')}")

    for conf_name, info in list(conf_validation.items())[:5]:
        status = "OK" if info.get("present") else "MISSING"
        print(f"  [{conf_name}] {status}")

    # =========================================================================
    # 10. CAUSAL EFFECT ESTIMATION (ATE)
    # =========================================================================
    print_section("STEP 10: ATE ESTIMATION (Propensity Score Matching)")

    # Prepare data for causal estimation with encoded confounders
    causal_data = X_raw.copy()
    causal_data["default"] = y.values

    raw_confounders = [c for c in confounders if c in causal_data.columns]
    print(f"  Available confounders for PSM: {raw_confounders}")

    # Build a fully numeric dataset for propensity score estimation
    # Label-encode all categorical columns
    from sklearn.preprocessing import LabelEncoder
    causal_encoded = causal_data.copy()
    for col in causal_encoded.columns:
        if not pd.api.types.is_numeric_dtype(causal_encoded[col]):
            le = LabelEncoder()
            causal_encoded[col] = le.fit_transform(causal_encoded[col].astype(str))

    estimator = CausalEffectEstimator(random_state=42)

    print_subsection("ATE: credit_amount -> default")
    ate_credit = estimator.estimate_ate(
        causal_encoded, "credit_amount", "default", raw_confounders,
        binarize=True, n_bootstrap=200,
    )
    print(f"  ATE (credit_amount binary -> default):")
    print(f"    Estimate:        {ate_credit['ate']:.6f}")
    print(f"    95% CI:          [{ate_credit['ci_lower']:.6f}, {ate_credit['ci_upper']:.6f}]")
    print(f"    Bootstrap valid: {ate_credit.get('n_bootstrap_valid', 'N/A')}/{ate_credit.get('n_bootstrap', 'N/A')}")
    print(f"    N matched:       {ate_credit.get('n_treated_matched', 'N/A')}")

    print_subsection("ATE: duration -> default")
    ate_duration = estimator.estimate_ate(
        causal_encoded, "duration", "default", raw_confounders,
        binarize=True, n_bootstrap=200,
    )
    print(f"  ATE (duration binary -> default):")
    print(f"    Estimate:        {ate_duration['ate']:.6f}")
    print(f"    95% CI:          [{ate_duration['ci_lower']:.6f}, {ate_duration['ci_upper']:.6f}]")
    print(f"    Bootstrap valid: {ate_duration.get('n_bootstrap_valid', 'N/A')}/{ate_duration.get('n_bootstrap', 'N/A')}")
    print(f"    N matched:       {ate_duration.get('n_treated_matched', 'N/A')}")

    # All treatments summary table
    print_subsection("All Treatments ATE Summary")
    ate_summary = estimator.estimate_all_treatments(
        causal_encoded, treatments, "default", raw_confounders, n_bootstrap=200,
    )
    if len(ate_summary) > 0:
        print(ate_summary.to_string(index=False))

    # =========================================================================
    # 11. SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    print_section("PIPELINE COMPLETE")
    print(f"  Total runtime: {elapsed:.2f} seconds")
    print()
    print(f"  Model Results:")
    print(f"    AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"    Accuracy:     {metrics['accuracy']:.4f}")
    print(f"    Precision:    {metrics['precision']:.4f}")
    print(f"    Recall:       {metrics['recall']:.4f}")
    print(f"    F1 Score:     {metrics['f1_score']:.4f}")
    print()
    print(f"  Causal ATE (credit_amount -> default):")
    print(f"    {ate_credit['ate']:.6f}  [{ate_credit['ci_lower']:.6f}, {ate_credit['ci_upper']:.6f}]")
    print()
    print(f"  Causal ATE (duration -> default):")
    print(f"    {ate_duration['ate']:.6f}  [{ate_duration['ci_lower']:.6f}, {ate_duration['ci_upper']:.6f}]")
    print()
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(run())
