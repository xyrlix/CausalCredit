"""Unit tests for src.fraud.three_class.ThreeClassFraudClassifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def fraud_synth():
    """Synthetic 1000-row Home Credit-like table with extreme fraud signal."""
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "AMT_INCOME_TOTAL": rng.lognormal(mean=11, sigma=0.5, size=n),
        "DAYS_EMPLOYED": -rng.integers(0, 10000, size=n),
        "EXT_SOURCE_1": rng.uniform(0, 1, size=n),
        "EXT_SOURCE_2": rng.uniform(0, 1, size=n),
        "ORGANIZATION_TYPE": rng.choice(
            ["Industry: mining", "Business Entity Type 3", "Self-employed",
             "Construction", "Trade: type 7", "Bank", "Other"],
            size=n,
        ),
        # Synthetic DPD from M5+ aggregation
        "INST__DPD_MAX": rng.choice([0, 0, 0, 5, 10, 30, 60, 90], size=n),
        "POS_DPD_MAX": rng.choice([0, 0, 0, 0, 5, 30], size=n),
    })
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.92, 0.08]))
    # Inject some clear fraud signatures
    df.loc[10:14, "INST__DPD_MAX"] = 90
    df.loc[10:14, "AMT_INCOME_TOTAL"] *= 5  # exaggerated income
    df.loc[10:14, "DAYS_EMPLOYED"] = -30  # very short employment
    y.iloc[10:14] = 1
    # Inject some clear systemic defaults (low DPD, so they don't trip fraud rules)
    df.loc[20:24, "ORGANIZATION_TYPE"] = "Industry: mining"
    df.loc[20:24, "INST__DPD_MAX"] = 0
    df.loc[20:24, "POS_DPD_MAX"] = 0
    df.loc[20:24, "AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].median()  # normal income
    y.iloc[20:24] = 1
    return df, y


def test_pseudo_labels_have_all_classes(fraud_synth):
    from src.fraud.three_class import ThreeClassFraudClassifier, ALL_LABELS
    df, y = fraud_synth
    clf = ThreeClassFraudClassifier()
    labels = clf.fit_pseudo_labels(df, y)
    counts = labels.value_counts().to_dict()
    for lbl in ALL_LABELS:
        assert lbl in counts, f"Missing label {lbl} in pseudo-labels"
    clf.pseudo_label_counts_ = counts  # save for inspection
    # The injected fraud cases should land in fraudulent
    assert (labels.iloc[10:14] == "fraudulent").all()
    # The injected systemic cases should land in systemic
    assert (labels.iloc[20:24] == "systemic").all()
    # All non-defaulters should be non_default
    assert (labels[y == 0] == "non_default").all()


def test_pseudo_label_counts_recorded(fraud_synth):
    from src.fraud.three_class import ThreeClassFraudClassifier
    df, y = fraud_synth
    clf = ThreeClassFraudClassifier()
    clf.fit_pseudo_labels(df, y)
    assert set(clf.pseudo_label_counts_.keys()) == {
        "non_default", "fraudulent", "non_malicious", "systemic"
    }
    assert clf.pseudo_label_counts_["non_default"] == int((y == 0).sum())


def test_fit_and_predict_shape(fraud_synth):
    from src.fraud.three_class import ThreeClassFraudClassifier
    df, y = fraud_synth
    clf = ThreeClassFraudClassifier(params={"n_estimators": 50, "verbosity": -1, "n_jobs": 1})
    clf.fit(df, y)
    p = clf.predict_proba(df)
    assert p.shape == (len(df), 3)
    # Each row sums to <= 1 (3 fraud classes only; non_default excluded)
    assert (p.sum(axis=1) <= 1.0 + 1e-6).all()
    assert (p >= 0).all()


def test_fraud_score_formula(fraud_synth):
    from src.fraud.three_class import ThreeClassFraudClassifier
    df, y = fraud_synth
    clf = ThreeClassFraudClassifier(params={"n_estimators": 50, "verbosity": -1, "n_jobs": 1})
    clf.fit(df, y)
    default_proba = np.full(len(df), 0.5)  # placeholder
    fraud = clf.fraud_score(df, default_proba)
    assert fraud.shape == (len(df),)
    assert (fraud >= 0).all() and (fraud <= 0.5 + 1e-6).all()


def test_feature_importance_returns_dataframe(fraud_synth):
    from src.fraud.three_class import ThreeClassFraudClassifier
    df, y = fraud_synth
    clf = ThreeClassFraudClassifier(params={"n_estimators": 50, "verbosity": -1, "n_jobs": 1})
    clf.fit(df, y)
    imp = clf.feature_importance()
    assert "feature" in imp.columns and "importance" in imp.columns
    assert len(imp) == len(df.columns)


def test_predict_without_fit_raises():
    from src.fraud.three_class import ThreeClassFraudClassifier
    clf = ThreeClassFraudClassifier()
    with pytest.raises(RuntimeError, match="not trained"):
        clf.predict_proba(pd.DataFrame({"a": [1, 2]}))


def test_pseudo_label_no_defaulters_returns_all_nd():
    """Edge case: no TARGET=1 in the population."""
    from src.fraud.three_class import ThreeClassFraudClassifier
    df = pd.DataFrame({"AMT_INCOME_TOTAL": [1, 2, 3], "DAYS_EMPLOYED": [-1, -2, -3]})
    y = pd.Series([0, 0, 0])
    clf = ThreeClassFraudClassifier()
    labels = clf.fit_pseudo_labels(df, y)
    assert (labels == "non_default").all()
    assert clf.pseudo_label_counts_["fraudulent"] == 0
    assert clf.pseudo_label_counts_["systemic"] == 0
