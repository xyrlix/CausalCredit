"""Page 2: Causal Visualization.

Shows the domain DAG (Graphviz), the pre-computed ATE, and (if cached on
disk) static visualisations from the M2 pipeline run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st


def render(ctx: Dict) -> None:
    registry = ctx["registry"]

    st.title("🔬 Causal Visualization")
    st.caption("Domain causal DAG, pre-computed ATE, and pipeline charts.")

    # ---- Domain DAG ----
    st.subheader("Domain Causal Graph (DAG)")
    g = registry.causal_graph
    if g is not None:
        try:
            dot = g.get_dot_string()
            st.graphviz_chart(dot)
        except Exception as exc:
            st.warning(f"Could not render DAG via Graphviz: {exc}")
        st.markdown(
            f"**Treatments:** `{', '.join(g.get_treatment_variables())}`  ·  "
            f"**Outcome:** `{g.get_outcome_variable()}`  ·  "
            f"**Nodes:** {len(g.nodes)}  ·  **Edges:** {len(g.edges)}"
        )
    else:
        st.warning("Domain DAG not available in registry.")

    # ---- ATE pre-compute ----
    st.subheader("Average Treatment Effect (ATE)")
    if registry.ate_summary:
        s = registry.ate_summary
        c1, c2, c3 = st.columns(3)
        c1.metric("ATE estimate", f"{s['ate']:+.4f}")
        c2.metric("95% CI lower", f"{s['ci_lower']:+.4f}")
        c3.metric("95% CI upper", f"{s['ci_upper']:+.4f}")
        st.caption(f"**Treatment:** {s['treatment']}  ·  **Outcome:** {s['outcome']}  ·  **Method:** {s['method']}")
    else:
        st.info("ATE pre-compute not available — re-train the registry to populate.")

    # ---- Static charts from the M2 pipeline run ----
    st.subheader("Pipeline Charts (from `output/figures/`)")
    fig_dir = Path("output/figures")
    if not fig_dir.exists():
        st.info("No charts found. Run `python -m src.run_pipeline` first.")
        return
    chart_metas = [
        ("06_causal_graph_dag.png", "Discovered DAG (PC + NOTEARS + Domain Knowledge fusion)"),
        ("07_cate_distribution.png", "CATE distributions — 3 EconML methods"),
        ("08_cate_subgroup.png", "CATE by applicant subgroup"),
        ("09_refutation_results.png", "4-refuter robustness check (DoWhy)"),
        ("10_shap_four_quadrant.png", "SHAP × causal-proxy four-quadrant"),
    ]
    available = [(p, c) for p, c in chart_metas if (fig_dir / p).exists()]
    if not available:
        st.warning("No pipeline charts on disk. Run `python -m src.run_pipeline`.")
        return
    tabs = st.tabs([f"Fig {p[:2]}" for p, _ in available])
    for tab, (path, caption) in zip(tabs, available):
        with tab:
            st.image(str(fig_dir / path), caption=caption, use_container_width=True)
