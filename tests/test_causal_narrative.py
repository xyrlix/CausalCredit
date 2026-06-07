"""Tests for src.explain.causal_narrative.CausalNarrative."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier


@pytest.fixture
def fitted_gbt():
    """Small synthetic classifier for testing."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.lognormal(11, 0.3, n),
        "debt": rng.normal(0.3, 0.1, n),
        "credit_score": rng.normal(700, 50, n),
        "n_accounts": rng.integers(1, 10, n).astype(float),
    })
    y = (X["debt"] * 2 + (40 - X["age"]) * 0.02 + rng.normal(0, 0.1, n) > 0.7).astype(int)
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0)
    model.fit(X, y)
    return model, X, y, model.predict_proba(X)[:, 1]


@pytest.fixture
def simple_dag():
    """DAG: debt → default; age → debt; credit_score → default; income → debt."""
    g = nx.DiGraph()
    g.add_edges_from([
        ("debt", "TARGET"),
        ("age", "debt"),
        ("credit_score", "TARGET"),
        ("income", "debt"),
        ("n_accounts", "TARGET"),
    ])
    return g


def test_trace_causal_path_finds_direct_path(simple_dag):
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=list(simple_dag.nodes), dag=simple_dag, outcome_name="TARGET")
    paths = cn.trace_causal_path("debt")
    assert any("debt" in p and "TARGET" in p for p in paths)
    assert any(p == ["debt", "TARGET"] for p in paths)


def test_trace_causal_path_no_path_for_unconnected(simple_dag):
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=list(simple_dag.nodes), dag=simple_dag, outcome_name="TARGET")
    # 'age' → 'debt' → 'TARGET' is a path; 'age' → 'TARGET' direct is NOT in the DAG
    paths_age = cn.trace_causal_path("age")
    assert any("age" in p and "debt" in p for p in paths_age)
    assert all("age" != p[0] or "debt" in p for p in paths_age)


def test_trace_causal_path_returns_empty_for_missing_node(simple_dag):
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=list(simple_dag.nodes), dag=simple_dag, outcome_name="TARGET")
    assert cn.trace_causal_path("not_a_node") == []


def test_features_on_paths_to_outcome(simple_dag):
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=list(simple_dag.nodes), dag=simple_dag, outcome_name="TARGET")
    out = cn.features_on_paths_to_outcome(["debt", "credit_score", "age", "TARGET"])
    assert "debt" in out and "credit_score" in out
    assert all(len(p) > 0 for p in out["debt"])
    assert "TARGET" not in out  # outcome is excluded


def test_model_level_narrative_returns_top_features(fitted_gbt):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    sv = np.random.default_rng(0).normal(size=(50, 5))  # (n, d) with d=5
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=None)
    out = cn.model_level_narrative(sv, top_k=3)
    assert "top_features" in out and "narrative" in out
    assert len(out["top_features"]) == 3
    assert "debt" in out["narrative"] or "age" in out["narrative"]


def test_cohort_level_narrative_computes_delta(fitted_gbt):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=None)
    features = X.iloc[0].to_dict()
    out = cn.cohort_level_narrative(features, X, y_prob, k=10)
    assert "k" in out
    assert out["k"] == 10
    assert "cohort_mean_p_default" in out
    assert "applicant_p_default" in out
    assert "narrative" in out
    # Delta is applicant minus cohort
    assert out["delta"] == out["applicant_p_default"] - out["cohort_mean_p_default"]


def test_cohort_level_narrative_detects_outlier(fitted_gbt):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=None)
    # Synthesize an extreme outlier
    features = {"age": 18, "income": 1e9, "debt": 0.95, "credit_score": 300, "n_accounts": 1}
    out = cn.cohort_level_narrative(features, X, y_prob, k=10)
    # Outlier should have at least one top_deviation
    assert "top_deviations" in out
    # The deviation entries have z-scores
    for d in out["top_deviations"]:
        assert "feature" in d and "z" in d


def test_individual_level_narrative_returns_top_k(fitted_gbt, simple_dag):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=simple_dag)
    features = X.iloc[0].to_dict()
    shap_row = np.array([0.1, -0.05, 0.3, -0.02, 0.01])
    fq = {
        "per_feature": pd.DataFrame({
            "feature": list(X.columns),
            "quadrant": ["TRUSTED", "NEGLIGIBLE", "TRUSTED", "MASKED", "NEGLIGIBLE"],
        })
    }
    out = cn.individual_level_narrative(features, shap_row, four_quadrant=fq, top_k=3)
    assert "top_features" in out
    assert len(out["top_features"]) == 3
    # The dominant feature is 'debt' (largest |shap|)
    assert out["dominant_feature"] == "debt"
    # 'debt' has a direct path to TARGET
    assert "debt" in out["dominant_dag_path"] and "TARGET" in out["dominant_dag_path"]


def test_individual_level_narrative_count_quadrants(fitted_gbt):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=None)
    fq = {
        "per_feature": pd.DataFrame({
            "feature": list(X.columns),
            "quadrant": ["TRUSTED", "UNTRUSTED", "MASKED", "NEGLIGIBLE", "TRUSTED"],
        })
    }
    shap_row = np.array([0.1, 0.2, 0.3, 0.05, 0.15])
    out = cn.individual_level_narrative(X.iloc[0].to_dict(), shap_row, four_quadrant=fq, top_k=5)
    assert out["n_trusted"] == 2
    assert out["n_untrusted"] == 1
    assert out["n_masked"] == 1


def test_explanation_robustness_stable_for_clean_signal(fitted_gbt):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=None)
    features = X.iloc[0].to_dict()
    shap_row = np.array([0.01, 0.01, 0.50, 0.01, 0.01])  # debt strongly dominant
    out = cn.explanation_robustness(features, shap_row, n_perturbations=10, noise_frac=0.05)
    assert 0.0 <= out["stability_score"] <= 1.0
    # With a single dominant feature, top-1 should be stable
    assert out["top_1_stable"] >= 0.5


def test_build_full_narrative_runs_all_sections(fitted_gbt, simple_dag):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=simple_dag)
    features = X.iloc[0].to_dict()
    shap_row = np.array([0.1, -0.05, 0.3, -0.02, 0.01])
    shap_global = np.random.default_rng(0).normal(size=(50, 5))
    fq = {
        "per_feature": pd.DataFrame({
            "feature": list(X.columns),
            "quadrant": ["TRUSTED"] * 5,
        })
    }
    full = cn.build_full_narrative(
        features=features, shap_row=shap_row, shap_global=shap_global,
        X_train=X, y_prob_train=y_prob, four_quadrant=fq, run_robustness=True,
    )
    assert set(full.keys()) >= {"model_level", "cohort_level", "individual_level", "robustness"}


def test_build_full_narrative_skip_robustness(fitted_gbt):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=None)
    full = cn.build_full_narrative(
        features=X.iloc[0].to_dict(), shap_row=np.zeros(5),
        shap_global=np.zeros((10, 5)), X_train=X, y_prob_train=y_prob,
        run_robustness=False,
    )
    assert "robustness" not in full


def test_render_markdown_contains_sections(fitted_gbt):
    from src.explain.causal_narrative import CausalNarrative
    model, X, y, y_prob = fitted_gbt
    cn = CausalNarrative(model=model, feature_names=list(X.columns), dag=None)
    full = cn.build_full_narrative(
        features=X.iloc[0].to_dict(), shap_row=np.array([0.1, -0.05, 0.3, -0.02, 0.01]),
        shap_global=np.random.default_rng(0).normal(size=(20, 5)),
        X_train=X, y_prob_train=y_prob, run_robustness=False,
    )
    md = CausalNarrative.render_markdown(full)
    assert "## 因果叙事" in md
    assert "### 1. 模型层面" in md
    assert "### 2. 同类申请人对照" in md
    assert "### 3. 本申请人" in md


# ---------------------------------------------------------------------------
# M8.4a — multi-language render_markdown
# ---------------------------------------------------------------------------


def test_render_markdown_zh_default():
    """Default (Simplified Chinese) headings preserved for backwards compat."""
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=["x", "y", "z"], dag=None)
    full = cn.build_full_narrative(
        features={"x": 1.0, "y": 0.5, "z": -0.2},
        shap_row=np.array([0.1, -0.05, 0.3]),
        shap_global=np.zeros((5, 3)),
        X_train=pd.DataFrame({"x": [0.0] * 10, "y": [0.0] * 10, "z": [0.0] * 10}),
        y_prob_train=np.zeros(10), run_robustness=False,
    )
    md = CausalNarrative.render_markdown(full)
    assert "## 因果叙事" in md
    assert "### 1. 模型层面" in md
    assert "### 2. 同类申请人对照" in md
    assert "### 3. 本申请人" in md
    assert "### 4. 解释稳健性" not in md  # robustness was skipped


def test_render_markdown_zh_hk_traditional():
    """zh-HK uses Traditional Chinese + 港式措辞 (層面 / 對照 / 申請人 / 穩健性)."""
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=["x", "y", "z"], dag=None)
    full = cn.build_full_narrative(
        features={"x": 1.0, "y": 0.5, "z": -0.2},
        shap_row=np.array([0.1, -0.05, 0.3]),
        shap_global=np.zeros((5, 3)),
        X_train=pd.DataFrame({"x": [0.0] * 10, "y": [0.0] * 10, "z": [0.0] * 10}),
        y_prob_train=np.zeros(10), run_robustness=False,
    )
    md = CausalNarrative.render_markdown(full, language="zh-HK")
    assert "## 因果敘事" in md
    assert "### 1. 模型層面" in md
    assert "### 2. 同類申請人對照" in md
    assert "### 3. 本申請人" in md


def test_render_markdown_en():
    """English headings for international reviewers."""
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=["x", "y", "z"], dag=None)
    full = cn.build_full_narrative(
        features={"x": 1.0, "y": 0.5, "z": -0.2},
        shap_row=np.array([0.1, -0.05, 0.3]),
        shap_global=np.zeros((5, 3)),
        X_train=pd.DataFrame({"x": [0.0] * 10, "y": [0.0] * 10, "z": [0.0] * 10}),
        y_prob_train=np.zeros(10), run_robustness=False,
    )
    md = CausalNarrative.render_markdown(full, language="en")
    assert "## Causal Narrative" in md
    assert "### 1. Model-level" in md
    assert "### 2. Cohort comparison" in md
    assert "### 3. Individual applicant" in md


def test_render_markdown_unknown_language_falls_back_to_zh():
    """Unknown language code falls back to Simplified Chinese, not crash."""
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=["x", "y", "z"], dag=None)
    full = cn.build_full_narrative(
        features={"x": 1.0, "y": 0.5, "z": -0.2},
        shap_row=np.array([0.1, -0.05, 0.3]),
        shap_global=np.zeros((5, 3)),
        X_train=pd.DataFrame({"x": [0.0] * 10, "y": [0.0] * 10, "z": [0.0] * 10}),
        y_prob_train=np.zeros(10), run_robustness=False,
    )
    md = CausalNarrative.render_markdown(full, language="klingon")
    assert "## 因果叙事" in md
