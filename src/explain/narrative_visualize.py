"""Visualizations for the M8.2 three-level causal narrative.

Two charts:
* ``15_causal_waterfall.png`` — top-10 SHAP contributions as a
  horizontal bar chart, color-coded by quadrant (TRUSTED/UNTRUSTED/
  MASKED/NEGLIGIBLE).  Visually obvious which features push risk
  up vs. down and which the model can be trusted on.
* ``16_narrative_card.png`` — three side-by-side panels showing the
  model-level, cohort-level, and individual-level narratives as
  card text.  Built for non-technical reviewers (compliance, ops).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_QUAD_COLORS = {
    "TRUSTED": "#10b981",     # green
    "UNTRUSTED": "#ef4444",   # red
    "MASKED": "#f59e0b",      # orange
    "NEGLIGIBLE": "#9ca3af",  # gray
    "UNKNOWN": "#6b7280",
}


def plot_causal_waterfall(
    top_features: List[Dict],
    output_path: str,
    title: str = "Top SHAP drivers (color = 4-quadrant trust label)",
) -> None:
    """Horizontal bar chart of the top features.

    ``top_features`` is the list of dicts produced by
    ``CausalNarrative.individual_level_narrative``.
    """
    if not top_features:
        return
    # Sort by abs SHAP for cleaner display
    items = sorted(top_features, key=lambda r: -abs(r.get("shap", 0.0)))
    names = [f"{r['feature']}\n[quad={r['quadrant']}]" for r in items]
    vals = [float(r["shap"]) for r in items]
    colors = [_QUAD_COLORS.get(r.get("quadrant", "UNKNOWN"), "#6b7280") for r in items]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4 * len(items) + 1.5)))
    y = np.arange(len(items))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    for yi, v in zip(y, vals):
        offset = 0.005 if v >= 0 else -0.005
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, yi, f"{v:+.4f}", va="center", ha=ha, fontsize=7, color="black")
    ax.set_xlabel("SHAP contribution to P(default)")
    ax.set_title(title, fontsize=11)
    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=_QUAD_COLORS[k], label=k) for k in ("TRUSTED", "UNTRUSTED", "MASKED", "NEGLIGIBLE")]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_narrative_card(
    narrative: Dict,
    output_path: str,
    applicant_id: str = "applicant",
) -> None:
    """Three side-by-side text panels: model / cohort / individual.

    Intended for non-technical reviewers: read it like a slide.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    panels = [
        ("Model-level", narrative.get("model_level", {}).get("narrative", "_n/a_"), "#dbeafe"),
        ("Cohort-level (k=10)", narrative.get("cohort_level", {}).get("narrative", "_n/a_"), "#fef3c7"),
        ("Individual-level", narrative.get("individual_level", {}).get("narrative", "_n/a_"), "#dcfce7"),
    ]
    for ax, (title, text, bg) in zip(axes, panels):
        ax.set_facecolor(bg)
        ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        # Word-wrap text manually (matplotlib has no built-in)
        wrapped = _wrap(text, width=55)
        ax.text(0.02, 0.85, wrapped, transform=ax.transAxes, fontsize=9, va="top", wrap=True)
        # Robustness footer if present
        r = narrative.get("robustness")
        if r:
            ax.text(
                0.02, 0.05,
                f"[Robustness] {r['interpretation']}\n"
                f"top-1 stable {r['top_1_stable']:.0%} | top-3 stable {r['top_3_stable']:.0%}",
                transform=ax.transAxes, fontsize=7, va="bottom", color="#374151", style="italic",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#94a3b8")

    fig.suptitle(f"Three-level causal narrative — {applicant_id}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _wrap(text: str, width: int = 55) -> str:
    """Naive word-wrap for matplotlib text."""
    out_lines = []
    for line in text.split("\n"):
        if not line.strip():
            out_lines.append("")
            continue
        words = line.split()
        cur = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + 1 > width and cur:
                out_lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += len(w) + 1
        if cur:
            out_lines.append(" ".join(cur))
    return "\n".join(out_lines)


def render_all(
    narrative: Dict,
    output_dir: str,
    applicant_id: str = "applicant",
) -> List[str]:
    """Render both charts and return their paths."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    p1 = os.path.join(output_dir, "15_causal_waterfall.png")
    plot_causal_waterfall(
        narrative.get("individual_level", {}).get("top_features", []),
        p1,
        title=f"Causal waterfall (top SHAP) — {applicant_id}",
    )
    paths.append(p1)
    p2 = os.path.join(output_dir, "16_narrative_card.png")
    plot_narrative_card(narrative, p2, applicant_id=applicant_id)
    paths.append(p2)
    return paths
