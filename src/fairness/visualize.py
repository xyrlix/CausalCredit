"""Fairness audit visualizations.

Three canonical charts emitted by the pipeline at STEP 15:

* ``12_fairness_group_rates.png``  — per-slice, per-group
  selection_rate / TPR / FPR grouped bar chart.  This is the chart
  a regulator would ask for: "for each protected group, how often
  is the model approving applicants, and how often is it
  approving the *correct* ones?".

* ``13_fairness_metric_gaps.png`` — three subplots (DP_gap,
  EO_gap, DI_ratio) with the HKMA / EU AI Act threshold lines
  baked in.  Visually obvious whether the system is inside the
  legal band.

* ``14_fairness_status.png`` — one horizontal bar per slice
  colored by FAIR / WARNING / UNFAIR.  The "dashboard" view that
  goes on the regulator-facing summary page.

All three are pure matplotlib — no seaborn — so the same Chinese
font config that the rest of the pipeline uses (Microsoft YaHei)
keeps the slice labels readable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fairness.metrics import FairnessSummary


# ----------------------------------------------------------------- group rates


def plot_group_rates(
    summaries: Dict[str, FairnessSummary],
    output_path: str,
) -> None:
    """Per-slice, per-group stacked bars: selection_rate + TPR.

    For each slice we draw a single figure with N groups on the
    x-axis and three bars per group: selection_rate, TPR, FPR.
    The chart is split into one subplot per slice so the four
    slices don't fight for the same axis.  Groups with N<5 are
    shaded (kept but de-emphasised) to surface statistical
    noise.
    """
    n_slices = len(summaries)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4.5), sharey=False)
    if n_slices == 1:
        axes = [axes]

    for ax, (slice_name, summary) in zip(axes, summaries.items()):
        grp = summary.groups
        # Stable group order
        grp = grp.sort_index()
        x = np.arange(len(grp))
        width = 0.27
        ax.bar(x - width, grp["selection_rate"].values, width, label="Selection rate", color="#3b82f6")
        ax.bar(x, grp["tpr"].values, width, label="TPR", color="#10b981")
        ax.bar(x + width, grp["fpr"].values, width, label="FPR", color="#ef4444")
        # De-emphasise tiny groups
        for xi, n in zip(x, grp["n"].values):
            if n < 100:
                ax.text(xi, -0.07, f"n={n}", ha="center", va="top", fontsize=7, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(grp.index, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, max(0.05, float(grp[["selection_rate", "tpr", "fpr"]].values.max()) * 1.15))
        ax.set_title(f"{slice_name}\n[{summary.status}] n={summary.n_total}", fontsize=10)
        ax.set_ylabel("Rate")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Per-group fairness rates (Home Credit test set)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- metric gaps


def plot_metric_gaps(
    summaries: Dict[str, FairnessSummary],
    output_path: str,
    dp_threshold: float = 0.05,
    eo_threshold: float = 0.05,
    di_threshold: float = 0.80,
) -> None:
    """Three subplots: DP_gap, EO_gap, DI_ratio across slices.

    Each subplot draws a single bar per slice with the threshold
    marked as a horizontal red line.  Bars are colored by whether
    the slice is in compliance.
    """
    slice_names = list(summaries.keys())
    dp = [summaries[s].dp_gap for s in slice_names]
    eo = [summaries[s].eo_gap for s in slice_names]
    di = [summaries[s].di_ratio for s in slice_names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    def _bar(ax, vals, threshold, kind, ylim):
        colors = ["#10b981" if _is_ok(v, threshold, kind) else "#ef4444" for v in vals]
        bars = ax.bar(slice_names, vals, color=colors, alpha=0.85)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={threshold}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_title(kind)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8)

    _bar(axes[0], dp, dp_threshold, "Demographic Parity gap", (0, max(max(dp) * 1.2, dp_threshold * 2)))
    _bar(axes[1], eo, eo_threshold, "Equal Opportunity gap", (0, max(max(eo) * 1.2, eo_threshold * 2)))
    _bar(axes[2], di, di_threshold, "Disparate Impact ratio (>=0.80 fair)", (0, 1.05))

    fig.suptitle("Fairness metrics across slices (green=in compliance)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _is_ok(value: float, threshold: float, kind: str) -> bool:
    if kind == "Disparate Impact ratio (>=0.80 fair)":
        return value >= threshold
    return value <= threshold


# ----------------------------------------------------------------- status board


_STATUS_COLORS = {"FAIR": "#10b981", "WARNING": "#f59e0b", "UNFAIR": "#ef4444"}


def plot_status_board(
    summaries: Dict[str, FairnessSummary],
    output_path: str,
) -> None:
    """One horizontal bar per slice, colored by status.

    Acts as the executive-summary view: at a glance, which slices
    are still legal and which need remediation.
    """
    names = list(summaries.keys())
    statuses = [summaries[n].status for n in names]
    colors = [_STATUS_COLORS.get(s, "#6b7280") for s in statuses]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(names) + 1)))
    y = np.arange(len(names))
    ax.barh(y, [1] * len(names), color=colors, alpha=0.85)
    for yi, name, st, summ in zip(y, names, statuses, summaries.values()):
        ax.text(0.02, yi, f"{name}: {st}  (DP={summ.dp_gap:.3f}, EO={summ.eo_gap:.3f}, DI={summ.di_ratio:.3f})",
                va="center", ha="left", color="white", fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Fairness status — Home Credit test set", fontsize=12)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- top-level


def render_all(
    summaries: Dict[str, FairnessSummary],
    output_dir: str,
) -> List[str]:
    """Render all three charts to ``output_dir`` and return the paths."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, fn, kwargs in [
        ("12_fairness_group_rates.png", plot_group_rates, {}),
        ("13_fairness_metric_gaps.png", plot_metric_gaps, {}),
        ("14_fairness_status.png", plot_status_board, {}),
    ]:
        path = os.path.join(output_dir, name)
        fn(summaries, path, **kwargs)
        paths.append(path)
    return paths
