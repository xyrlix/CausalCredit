"""ATE estimation using manual propensity score matching.

Implements Average Treatment Effect estimation via:
1. Logistic regression for propensity score estimation
2. Nearest-neighbor matching on propensity scores
3. ATE computation with bootstrap confidence intervals

No DoWhy/EconML dependency - pure sklearn/numpy/pandas implementation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def binarize_treatment(data: pd.Series, threshold: Optional[float] = None) -> pd.Series:
    """Binarize a continuous treatment variable by median split."""
    if threshold is None:
        threshold = data.median()
    return (data >= threshold).astype(int)


def estimate_propensity_scores(X: pd.DataFrame, treatment: pd.Series) -> np.ndarray:
    """Estimate propensity scores using logistic regression.

    Returns probability of treatment=1 given confounders X.
    """
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X, treatment)
    return model.predict_proba(X)[:, 1]


def propensity_score_matching(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    n_neighbors: int = 1,
    caliper: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform propensity score matching.

    For each treated unit, find the nearest control unit(s) based on propensity score.
    Returns matched treated outcomes and matched control outcomes.

    Args:
        X: Confounder DataFrame.
        treatment: Binary treatment indicator (0/1).
        outcome: Continuous or binary outcome variable.
        n_neighbors: Number of control units to match per treated unit.
        caliper: Maximum allowed propensity score distance (in std dev units).

    Returns:
        (treated_outcomes, matched_control_outcomes)
    """
    ps = estimate_propensity_scores(X, treatment)

    treated_idx = np.where(treatment.values == 1)[0]
    control_idx = np.where(treatment.values == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("Both treated and control groups must have at least one unit.")

    ps_treated = ps[treated_idx].reshape(-1, 1)
    ps_control = ps[control_idx].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(ps_control)
    distances, indices = nn.kneighbors(ps_treated)

    if caliper is not None:
        ps_std = np.std(ps)
        caliper_dist = caliper * ps_std
        valid_mask = distances.flatten() <= caliper_dist
    else:
        valid_mask = np.ones(len(treated_idx), dtype=bool)

    outcome_vals = outcome.values

    treated_outcomes = outcome_vals[treated_idx[valid_mask]]

    if n_neighbors == 1:
        matched_control_outcomes = outcome_vals[control_idx[indices[valid_mask].flatten()]]
    else:
        matched_control_outcomes = np.mean(
            outcome_vals[control_idx[indices[valid_mask]]], axis=1
        )

    return treated_outcomes, matched_control_outcomes


def compute_ate(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    n_neighbors: int = 1,
    caliper: Optional[float] = 0.25,
) -> Dict:
    """Compute ATE using propensity score matching.

    Returns a dict with ATE estimate and related statistics.
    """
    treated_outcomes, matched_control_outcomes = propensity_score_matching(
        X, treatment, outcome, n_neighbors=n_neighbors, caliper=caliper
    )

    n_matched = len(treated_outcomes)
    diffs = treated_outcomes - matched_control_outcomes
    ate = float(np.mean(diffs))
    ate_std = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))

    return {
        "ate": ate,
        "ate_std": ate_std,
        "n_treated_matched": n_matched,
        "mean_treated_outcome": float(np.mean(treated_outcomes)),
        "mean_control_outcome": float(np.mean(matched_control_outcomes)),
        "method": "propensity_score_matching",
    }


def compute_ate_with_bootstrap(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    n_bootstrap: int = 200,
    n_neighbors: int = 1,
    caliper: Optional[float] = 0.25,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict:
    """Compute ATE with bootstrap confidence intervals.

    Returns ATE estimate with 95% CI.
    """
    rng = np.random.RandomState(random_state)
    n = len(X)

    ate_estimates = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        X_boot = X.iloc[indices].reset_index(drop=True)
        t_boot = treatment.iloc[indices].reset_index(drop=True)
        y_boot = outcome.iloc[indices].reset_index(drop=True)

        try:
            result = compute_ate(X_boot, t_boot, y_boot,
                                 n_neighbors=n_neighbors, caliper=caliper)
            ate_estimates.append(result["ate"])
        except ValueError:
            continue

    if len(ate_estimates) < 10:
        return {
            "ate": float(np.nan),
            "ci_lower": float(np.nan),
            "ci_upper": float(np.nan),
            "n_bootstrap_valid": len(ate_estimates),
            "error": "Too few valid bootstrap samples",
        }

    ate_arr = np.array(ate_estimates)
    ci_lower = float(np.percentile(ate_arr, alpha / 2 * 100))
    ci_upper = float(np.percentile(ate_arr, (1 - alpha / 2) * 100))

    base_result = compute_ate(X, treatment, outcome,
                              n_neighbors=n_neighbors, caliper=caliper)

    return {
        "ate": base_result["ate"],
        "ate_bootstrap_mean": float(np.mean(ate_arr)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": 1 - alpha,
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_valid": len(ate_estimates),
        "ate_std": base_result["ate_std"],
        "n_treated_matched": base_result["n_treated_matched"],
        "method": "propensity_score_matching_with_bootstrap",
    }


class CausalEffectEstimator:
    """Causal effect estimator using manual propensity score matching."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: List[str],
        binarize: bool = True,
        threshold: Optional[float] = None,
        n_bootstrap: int = 200,
    ) -> Dict:
        """Estimate Average Treatment Effect.

        Args:
            data: DataFrame with all variables.
            treatment_col: Name of treatment variable.
            outcome_col: Name of outcome variable.
            confounder_cols: List of confounder column names.
            binarize: Whether to binarize continuous treatment by median.
            threshold: Custom threshold for binarization.
            n_bootstrap: Number of bootstrap iterations for CI.

        Returns:
            Dict with ATE estimates and confidence intervals.
        """
        df = data.copy()
        outcome = df[outcome_col].astype(float)
        confounders = df[[c for c in confounder_cols if c in df.columns]]

        if binarize:
            treatment = binarize_treatment(df[treatment_col], threshold)
        else:
            treatment = df[treatment_col].astype(int)

        treatment_label = f"{treatment_col}_binary" if binarize else treatment_col

        result = compute_ate_with_bootstrap(
            confounders, treatment, outcome,
            n_bootstrap=n_bootstrap,
            random_state=self.random_state,
        )
        result["treatment"] = treatment_label
        result["outcome"] = outcome_col
        result["confounders"] = confounder_cols

        return result

    def estimate_all_treatments(
        self,
        data: pd.DataFrame,
        treatment_cols: List[str],
        outcome_col: str,
        confounder_cols: List[str],
        n_bootstrap: int = 200,
    ) -> pd.DataFrame:
        """Estimate ATE for all treatment variables."""
        results = []
        for tx in treatment_cols:
            if tx not in data.columns:
                continue
            r = self.estimate_ate(
                data, tx, outcome_col, confounder_cols,
                binarize=True, n_bootstrap=n_bootstrap,
            )
            results.append({
                "treatment": r["treatment"],
                "ate": r["ate"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "ate_std": r.get("ate_std", float("nan")),
                "n_matched": r.get("n_treated_matched", 0),
            })
        return pd.DataFrame(results)

    def comprehensive_analysis(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: List[str],
    ) -> Dict:
        """Run comprehensive analysis with bootstrap CIs."""
        result = self.estimate_ate(
            data, treatment_col, outcome_col, confounder_cols,
            binarize=True, n_bootstrap=200,
        )
        return result
