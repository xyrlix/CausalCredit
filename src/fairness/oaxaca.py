"""Oaxaca–Blinder decomposition of group differences in model predictions.

@requirement REQ-FAIR-002
@design docs/plans/2026-06-05-causalcredit-architecture-design.md §D9
@see docs/CausalCredit_因果推理验证标准体系.md §3.3 (公平性验证)

The classical Oaxaca (1973) / Blinder (1973) decomposition answers:

    "The mean predicted P(default) is X pp higher for Group A than for
    Group B. How much of that gap is explained by A and B having
    different feature distributions (endowments) vs. how much is
    unexplained (the 'coefficients effect', often interpreted as
    potential discrimination)?"

Formally, with two OLS regressions of the outcome on the features,
one per group::

    y_A = X_A β_A + ε_A     (fit on group A)
    y_B = X_B β_B + ε_B     (fit on group B)

the total gap is decomposed as::

    ȳ_A - ȳ_B  =  (x̄_A - x̄_B) β_B          # "explained" (endowments)
                +  x̄_A (β_A - β_B)          # "unexplained" (coefficients)

with the Blinder variant using a pooled β as the reference.

This module fits separate OLS regressions (one per group) and returns
a :class:`OaxacaBlinderResult` with:

* the overall gap,
* the explained and unexplained portions (with percentages),
* a per-feature contribution table,
* a "discrimination index" = unexplained gap / total gap (in [0, 1]).

Use case in CausalCredit
------------------------
Run after the standard group-fairness metrics (DP / EO / DI) to get an
*explanatory* decomposition: not just "is there a gap?" but "is the
gap driven by legitimate feature differences (income, employment
history) or by a residual that might be model bias?".

Reference
---------
Oaxaca, R. (1973). "Male-Female Wage Differentials in Urban Labor
Markets". International Economic Review 14(3): 693-709.
Blinder, A. S. (1973). "Wage Discrimination: Reduced Form and Structural
Estimates". Journal of Human Resources 8(4): 436-455.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("causalcredit.fairness.oaxaca")

# Convention: groups are passed as string labels. The two groups
# actually compared are the first two unique non-UNKNOWN values.
UNKNOWN_LABEL = "UNKNOWN"


@dataclass
class OaxacaBlinderResult:
    """Container for the decomposition output."""
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_y_a: float
    mean_y_b: float
    total_gap: float          # ȳ_A - ȳ_B
    explained_gap: float      # (x̄_A - x̄_B) β_B  (Blinder form)
    unexplained_gap: float    # x̄_A (β_A - β_B)
    explained_share: float    # explained / total  (in [0, 1])
    unexplained_share: float  # unexplained / total
    feature_contributions: pd.DataFrame  # feature | mean_a | mean_b | explained | unexplained
    discrimination_index: float  # |unexplained| / |total|  (in [0, 1])
    reference: str            # "B" (Blinder) or "pooled" (Oaxaca)

    def to_dict(self) -> Dict:
        return {
            "group_a": self.group_a,
            "group_b": self.group_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "mean_y_a": self.mean_y_a,
            "mean_y_b": self.mean_y_b,
            "total_gap": self.total_gap,
            "explained_gap": self.explained_gap,
            "unexplained_gap": self.unexplained_gap,
            "explained_share": self.explained_share,
            "unexplained_share": self.unexplained_share,
            "discrimination_index": self.discrimination_index,
            "reference": self.reference,
            "feature_contributions": self.feature_contributions.to_dict(orient="records"),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def oaxaca_blinder_decomposition(
    y: np.ndarray,
    X: pd.DataFrame,
    groups: np.ndarray,
    *,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    reference: str = "B",
) -> OaxacaBlinderResult:
    """Run the Oaxaca–Blinder decomposition on a binary outcome.

    Parameters
    ----------
    y : (n,) array-like of 0/1 outcomes (or continuous predictions)
    X : (n, p) DataFrame of features (numeric only — categorical
        encoding is the caller's job)
    groups : (n,) array-like of group labels. UNKNOWN / NaN entries
        are excluded.
    group_a, group_b : optional explicit group labels. If omitted, the
        first two unique non-UNKNOWN groups are used.
    reference : "B" (Blinder) uses β_B as the reference for the
        explained component; "pooled" uses a pooled OLS fit on all
        observations (the "Oaxaca threefold" variant).

    Returns
    -------
    :class:`OaxacaBlinderResult`
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if len(y) != len(X) or len(y) != len(groups):
        raise ValueError("y, X, groups must have the same length")

    # Drop UNKNOWN / NaN rows
    mask = (
        ~pd.isna(groups)
        & (groups != UNKNOWN_LABEL)
        & ~np.isnan(y)
        & np.all(~np.isnan(X), axis=1)
    )
    y = y[mask]
    X = X.loc[mask].reset_index(drop=True)
    groups = groups[mask]

    unique_groups = list(pd.Series(groups).unique())
    if group_a is None:
        group_a = unique_groups[0]
    if group_b is None:
        # Pick the first group that isn't A
        for g in unique_groups:
            if g != group_a:
                group_b = g
                break
    if group_b is None or group_b == group_a:
        raise ValueError(
            f"Need two distinct non-UNKNOWN groups; got {unique_groups!r}"
        )

    mask_a = groups == group_a
    mask_b = groups == group_b
    X_a, y_a = X.loc[mask_a], y[mask_a]
    X_b, y_b = X.loc[mask_b], y[mask_b]
    if len(X_a) < 5 or len(X_b) < 5:
        raise ValueError(
            f"Need at least 5 observations per group; got "
            f"{len(X_a)} for {group_a!r} and {len(X_b)} for {group_b!r}"
        )

    # Fit OLS per group
    model_a = LinearRegression().fit(X_a, y_a)
    model_b = LinearRegression().fit(X_b, y_b)
    if reference == "pooled":
        model_ref = LinearRegression().fit(X, y)
    elif reference == "B":
        model_ref = model_b
    elif reference == "A":
        model_ref = model_a
    else:
        raise ValueError(f"reference must be B, A, or pooled; got {reference!r}")

    # Mean feature vectors
    x_bar_a = X_a.mean(axis=0).values
    x_bar_b = X_b.mean(axis=0).values
    beta_a = model_a.coef_
    beta_b = model_b.coef_
    beta_ref = model_ref.coef_

    # Aggregate
    mean_y_a = float(y_a.mean())
    mean_y_b = float(y_b.mean())
    total_gap = mean_y_a - mean_y_b
    # ȳ_A − ȳ_B = (α_A − α_B) + x̄_A·β_A − x̄_B·β_B
    #           = (x̄_A − x̄_B)·β_ref  +  [(α_A − α_B) + x̄_A·(β_A − β_B) − x̄_B·(β_B − β_ref) + x̄_A·(β_ref − β_B)]
    # Define explained as compositional difference at the reference coefficient,
    # and unexplained as the residual — guarantees the identity exactly.
    explained_gap = float(np.dot(x_bar_a - x_bar_b, beta_ref))
    unexplained_gap = float(total_gap - explained_gap)
    if abs(total_gap) > 1e-9:
        explained_share = explained_gap / total_gap
        unexplained_share = unexplained_gap / total_gap
    else:
        # No gap → no shares
        explained_share = 0.0
        unexplained_share = 0.0
    discrimination_index = (
        abs(unexplained_gap) / (abs(explained_gap) + abs(unexplained_gap))
        if (abs(explained_gap) + abs(unexplained_gap)) > 1e-12
        else 0.0
    )

    # Per-feature contributions (Blinder-style: each feature gets its
    # share of the explained and unexplained components)
    feature_rows = []
    for i, col in enumerate(X.columns):
        endo = (x_bar_a[i] - x_bar_b[i]) * beta_ref[i]
        coef = x_bar_a[i] * (beta_a[i] - beta_b[i])
        feature_rows.append({
            "feature": col,
            "mean_group_a": float(x_bar_a[i]),
            "mean_group_b": float(x_bar_b[i]),
            "delta_mean": float(x_bar_a[i] - x_bar_b[i]),
            "beta_a": float(beta_a[i]),
            "beta_b": float(beta_b[i]),
            "explained": float(endo),
            "unexplained": float(coef),
            "abs_total": float(abs(endo) + abs(coef)),
        })
    feature_df = pd.DataFrame(feature_rows).sort_values(
        "abs_total", ascending=False
    ).reset_index(drop=True)

    return OaxacaBlinderResult(
        group_a=str(group_a),
        group_b=str(group_b),
        n_a=int(len(X_a)),
        n_b=int(len(X_b)),
        mean_y_a=mean_y_a,
        mean_y_b=mean_y_b,
        total_gap=float(total_gap),
        explained_gap=explained_gap,
        unexplained_gap=unexplained_gap,
        explained_share=float(explained_share),
        unexplained_share=float(unexplained_share),
        feature_contributions=feature_df,
        discrimination_index=float(discrimination_index),
        reference=reference,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_oaxaca_decomposition(
    result: OaxacaBlinderResult,
    output_path: str,
    top_k_features: int = 10,
) -> None:
    """Render a 2-panel chart: total-gap bar + per-feature waterfall.

    Panel 1: horizontal bar showing total / explained / unexplained.
    Panel 2: top-K feature contributions (signed bars).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: aggregate
    components = ["Total gap", "Explained", "Unexplained"]
    values = [result.total_gap, result.explained_gap, result.unexplained_gap]
    colors = ["#444", "#00aaff", "#ff7755"]
    bars = ax1.barh(components, values, color=colors)
    ax1.axvline(0, color="grey", lw=0.8)
    ax1.set_xlabel("Gap (P units)")
    ax1.set_title(
        f"Oaxaca–Blinder decomposition\n"
        f"{result.group_a} vs {result.group_b}  ·  "
        f"discrimination index = {result.discrimination_index:.2%}"
    )
    for bar, v in zip(bars, values):
        ax1.text(v + (0.001 if v >= 0 else -0.001), bar.get_y() + bar.get_height() / 2,
                 f"{v:+.4f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=9)

    # Panel 2: top-K feature contributions (split)
    df = result.feature_contributions.head(top_k_features).iloc[::-1]
    y_pos = np.arange(len(df))
    width_exp = df["explained"].values
    width_unexp = df["unexplained"].values
    ax2.barh(y_pos, width_exp, color="#00aaff", label="Explained (endowments)")
    ax2.barh(y_pos, width_unexp, left=width_exp, color="#ff7755", label="Unexplained (coefficients)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df["feature"].values, fontsize=8)
    ax2.axvline(0, color="grey", lw=0.8)
    ax2.set_xlabel("Contribution to gap (P units)")
    ax2.set_title(f"Top-{top_k_features} feature contributions")
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("oaxaca chart written: %s", output_path)
