"""Page 2: Causal Visualization.

Shows the domain DAG (Graphviz), the pre-computed ATE, and (if cached on
disk) static visualisations from the M2 pipeline run. Strings flow
through :func:`src.frontend.i18n.t` keyed on ``ctx["lang"]`` (M8.5d).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from src.frontend.i18n import t


def render(ctx: Dict) -> None:
    registry = ctx["registry"]
    lang = ctx.get("lang", "en")

    st.title(t("causal.title", lang))
    st.caption(t("causal.caption", lang))

    # ---- Domain DAG ----
    st.subheader(t("causal.dag_header", lang))
    g = registry.causal_graph
    if g is not None:
        try:
            dot = g.get_dot_string()
            st.graphviz_chart(dot)
        except Exception as exc:
            st.warning(t("causal.dag_render_failed", lang, exc=exc))
        st.markdown(t(
            "causal.treatments_outcome", lang,
            treatments=", ".join(g.get_treatment_variables()),
            outcome=g.get_outcome_variable(),
            n_nodes=len(g.nodes), n_edges=len(g.edges),
        ))
    else:
        st.warning(t("causal.dag_unavailable", lang))

    # ---- ATE pre-compute ----
    st.subheader(t("causal.ate_header", lang))
    if registry.ate_summary:
        s = registry.ate_summary
        c1, c2, c3 = st.columns(3)
        c1.metric(t("causal.ate_estimate", lang), f"{s['ate']:+.4f}")
        c2.metric(t("causal.ate_ci_lower", lang), f"{s['ci_lower']:+.4f}")
        c3.metric(t("causal.ate_ci_upper", lang), f"{s['ci_upper']:+.4f}")
        st.caption(t("causal.ate_caption", lang,
                     treatment=s['treatment'], outcome=s['outcome'], method=s['method']))
    else:
        st.info(t("causal.ate_unavailable", lang))

    # ---- Static charts from the M2 pipeline run ----
    st.subheader(t("causal.charts_header", lang))
    fig_dir = Path("output/figures")
    if not fig_dir.exists():
        st.info(t("causal.no_charts", lang))
        return
    chart_metas = [
        ("06_causal_graph_dag.png", "Discovered DAG (PC + NOTEARS + Domain Knowledge fusion)"),
        ("07_cate_distribution.png", "CATE distributions — 3 EconML methods"),
        ("08_cate_subgroup.png", "CATE by applicant subgroup"),
        ("09_refutation_results.png", "4-refuter robustness check (DoWhy)"),
        ("10_shap_four_quadrant.png", "SHAP × causal-proxy four-quadrant"),
        ("15_causal_waterfall.png", "M8.2 — Causal narrative waterfall (top features, 4-quadrant colored)"),
        ("16_narrative_card.png", "M8.2 — Narrative card (3-panel: model / cohort / individual)"),
    ]
    available = [(p, c) for p, c in chart_metas if (fig_dir / p).exists()]
    if not available:
        st.warning(t("causal.no_charts_warn", lang))
        return
    tabs = st.tabs([f"Fig {p[:2]}" for p, _ in available])
    for tab, (path, caption) in zip(tabs, available):
        with tab:
            st.image(str(fig_dir / path), caption=caption, use_container_width=True)
