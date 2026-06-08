"""Group-fairness metrics for binary credit-scoring classifiers.

The module is intentionally framework-free (pure numpy / pandas) so it
can audit any classifier's predictions — including the base LightGBM
model and the anti-fraud FraudGuard routing.

Notation
--------
* ``y_true``  — (n,) 0/1 ground-truth labels
* ``y_pred``  — (n,) 0/1 hard predictions at a chosen threshold
* ``y_score`` — (n,) continuous predicted probabilities
* ``groups``  — (n,) group labels (string, int, or category)
* ``A = a``   — protected attribute value

The three group-fairness metrics are all "between-group spread"
quantities.  Following HKMA / EU AI Act guidance, the **default
fair thresholds** are:

* Demographic Parity gap  ``|DP_a - DP_a'|  <  0.05``
* Equal Opportunity gap   ``|TPR_a - TPR_a'|  <  0.05``
* Disparate Impact ratio  ``min(sel_rate) / max(sel_rate)  >=  0.80``

If ANY metric is violated the slice is flagged ``WARNING``; if TWO
or more are violated it is ``UNFAIR``.

Small-group handling
--------------------
``summarize_fairness`` (and the three sub-metrics) accept a
``min_group_size`` argument (default 100).  Groups with fewer than
``min_group_size`` samples are dropped from the between-group
metric calculation — they are too small for a statistically
meaningful between-group spread (a single predicted default in a
30-person group swings the selection rate by 3 pp).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- metrics


def demographic_parity_gap(
    y_pred: np.ndarray,
    groups: np.ndarray,
    min_group_size: int = 0,
) -> float:
    """Max − min selection rate P(Ŷ=1) across groups.

    A value of 0 means perfect demographic parity.  A value of 0.10
    means the most-selected group is 10 percentage points more
    likely to get a positive prediction than the least-selected.

    Rows labelled ``UNKNOWN`` are excluded — they are missing-value
    sentinels from the slicer, not real protected groups.  Groups
    with fewer than ``min_group_size`` samples are also excluded.
    """
    _, y_pred_f, groups_f = _filter_unknown_groups(None, y_pred, groups)
    _populate_group_counts(groups_f)
    rates = _filter_small_groups(_selection_rates(y_pred_f, groups_f), min_group_size)
    if len(rates) < 2:
        return 0.0
    return float(max(rates.values()) - min(rates.values()))


def _filter_unknown_groups(
    y_true, y_pred: np.ndarray, groups: np.ndarray
):
    """Drop rows whose group label is the missing-value sentinel ``UNKNOWN``.

    The slicing module routes any unparseable / missing cell to the
    string ``"UNKNOWN"`` to keep the per-group loop from breaking;
    those rows must not be allowed to inflate or deflate the
    between-group spread (otherwise adding a few bad rows can flip a
    model from ``FAIR`` to ``UNFAIR``).  Returns numpy arrays of
    equal length.
    """
    mask = np.asarray(groups) != "UNKNOWN"
    y_true_out = np.asarray(y_true)[mask] if y_true is not None else None
    return (
        y_true_out,
        np.asarray(y_pred)[mask],
        np.asarray(groups)[mask],
    )


def _filter_small_groups(
    rates: Dict, min_group_size: int
) -> Dict:
    """Drop groups whose total sample count is below ``min_group_size``.

    Accepts a ``{group: aggregate}`` dict (e.g. selection rates) and a
    second dict of group → count, then returns only groups whose
    count meets the threshold.  The count dict defaults to the
    aggregate values themselves (caller may pass a separate
    ``_group_counts`` mapping if the aggregate is a rate, not a
    count).
    """
    if min_group_size <= 0:
        return dict(rates)
    return {g: v for g, v in rates.items() if _group_total(g) >= min_group_size}


# A module-level cache: ``_group_counts[g]`` is filled by
# ``_selection_rates`` / ``_tpr_per_group`` calls and then read by
# ``_filter_small_groups``.  This avoids recomputing the per-group
# sizes on every metric call (the slicer produces the same groups
# array for all three metrics in ``summarize_fairness``).
_group_counts: Dict[str, int] = {}


def _group_total(g) -> int:
    return int(_group_counts.get(str(g), 0))


def equal_opportunity_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    min_group_size: int = 0,
) -> float:
    """Max − min TPR across groups, computed on the y_true=1 subset.

    TPR = P(Ŷ=1 | Y=1).  Measures whether qualified applicants from
    each group have an equal chance of being accepted.
    """
    y_true_f, y_pred_f, groups_f = _filter_unknown_groups(y_true, y_pred, groups)
    _populate_group_counts(groups_f)
    tprs = _tpr_per_group(y_true_f, y_pred_f, groups_f)
    tprs = {g: v for g, v in tprs.items() if _group_total(g) >= min_group_size}
    if len(tprs) < 2:
        return 0.0
    return float(max(tprs.values()) - min(tprs.values()))


def disparate_impact_ratio(
    y_pred: np.ndarray,
    groups: np.ndarray,
    min_group_size: int = 0,
) -> float:
    """min(selection_rate) / max(selection_rate) across groups.

    The EEOC 80 % rule: ``DI >= 0.80`` is "fair".  Below 0.80 the
    classifier is potentially illegal under US equal-employment
    law; we surface this for credit decisions where the analogy
    applies.
    """
    _, y_pred_f, groups_f = _filter_unknown_groups(None, y_pred, groups)
    _populate_group_counts(groups_f)
    rates = _filter_small_groups(_selection_rates(y_pred_f, groups_f), min_group_size)
    if len(rates) < 2 or max(rates.values()) == 0:
        return 1.0
    return float(min(rates.values()) / max(rates.values()))


# ----------------------------------------------------------------- per-group


def group_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Compute per-group rates for a single slice.

    Returns a DataFrame indexed by group with columns
    ``{n, n_pos, selection_rate, tpr, fpr, fnr, auc, mean_score}``.
    AUC is computed only for groups with at least 5 positive and
    5 negative samples; otherwise it's NaN.
    """
    df = pd.DataFrame({
        "group": groups,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    })
    rows = []
    for g, sub in df.groupby("group"):
        n = len(sub)
        n_pos = int(sub["y_true"].sum())
        sel = float(sub["y_pred"].mean()) if n else 0.0
        tpr = float(((sub["y_pred"] == 1) & (sub["y_true"] == 1)).sum() / n_pos) if n_pos else 0.0
        n_neg = n - n_pos
        fpr = float(((sub["y_pred"] == 1) & (sub["y_true"] == 0)).sum() / n_neg) if n_neg else 0.0
        fnr = 1.0 - tpr if n_pos else 0.0
        if n_pos >= 5 and n_neg >= 5:
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(sub["y_true"], sub["y_score"]))
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")
        rows.append({
            "group": g,
            "n": n,
            "n_pos": n_pos,
            "selection_rate": sel,
            "tpr": tpr,
            "fpr": fpr,
            "fnr": fnr,
            "auc": auc,
            "mean_score": float(sub["y_score"].mean()) if n else 0.0,
        })
    return pd.DataFrame(rows).set_index("group")


# ------------------------------------------------------------- summary


@dataclass
class FairnessSummary:
    """One slice's overall fairness verdict."""
    slice_name: str
    n_groups: int
    n_total: int
    dp_gap: float
    eo_gap: float
    di_ratio: float
    status: str  # "FAIR" | "WARNING" | "UNFAIR"
    groups: pd.DataFrame
    violated_metrics: List[str] = field(default_factory=list)
    min_group_size: int = 0
    groups_filtered: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["groups"] = self.groups.reset_index().to_dict(orient="records")
        return d


def summarize_fairness(
    slice_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    dp_threshold: float = 0.05,
    eo_threshold: float = 0.05,
    di_threshold: float = 0.80,
    min_group_size: int = 0,
) -> FairnessSummary:
    """Compute the three metrics and return a status verdict.

    Status:
        FAIR     — all three metrics within bounds
        WARNING  — exactly one metric violated
        UNFAIR   — two or more metrics violated

    Groups with fewer than ``min_group_size`` samples are excluded
    from the between-group metric calculation (they are too small
    for a reliable between-group spread estimate).  They are still
    listed in the ``groups`` table for transparency.
    """
    # Pre-fill the module-level _group_counts cache so the three
    # sub-metric functions can use the same min_group_size filter.
    _populate_group_counts(groups)
    grp = group_rates(y_true, y_pred, y_score, groups)
    dp = demographic_parity_gap(y_pred, groups, min_group_size=min_group_size)
    eo = equal_opportunity_gap(y_true, y_pred, groups, min_group_size=min_group_size)
    di = disparate_impact_ratio(y_pred, groups, min_group_size=min_group_size)
    violated = []
    if dp > dp_threshold:
        violated.append(f"DP_gap={dp:.3f}>{dp_threshold}")
    if eo > eo_threshold:
        violated.append(f"EO_gap={eo:.3f}>{eo_threshold}")
    if di < di_threshold:
        violated.append(f"DI={di:.3f}<{di_threshold}")
    if len(violated) >= 2:
        status = "UNFAIR"
    elif len(violated) == 1:
        status = "WARNING"
    else:
        status = "FAIR"
    return FairnessSummary(
        slice_name=slice_name,
        n_groups=len(grp),
        n_total=int(grp["n"].sum()),
        dp_gap=dp,
        eo_gap=eo,
        di_ratio=di,
        status=status,
        groups=grp,
        violated_metrics=violated,
        min_group_size=min_group_size,
        groups_filtered=sorted(
            str(g) for g, n in grp["n"].items() if int(n) < min_group_size
        ),
    )


# -------------------------------------------------------------- helpers


def _populate_group_counts(groups: np.ndarray) -> None:
    """Cache per-group sample counts for the small-group filter."""
    global _group_counts
    _group_counts = {
        str(g): int(c)
        for g, c in pd.Series(np.asarray(groups)).value_counts().items()
    }


def _selection_rates(y_pred: np.ndarray, groups: np.ndarray) -> Dict:
    df = pd.DataFrame({"g": groups, "p": y_pred})
    return df.groupby("g")["p"].mean().to_dict()


def _tpr_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict:
    df = pd.DataFrame({"g": groups, "t": y_true, "p": y_pred})
    out = {}
    for g, sub in df.groupby("g"):
        n_pos = int(sub["t"].sum())
        if n_pos == 0:
            continue
        out[str(g)] = float(((sub["p"] == 1) & (sub["t"] == 1)).sum() / n_pos)
    return out
