"""Page 5: Interest Rate Optimizer (M8.5g).

Counterfactual sweep across a rate grid → classify the applicant into
sleeping_dog / rate_sensitive / neutral, and recommend a profit-maximizing
rate. Strings flow through :func:`src.frontend.i18n.t` keyed on
``ctx["lang"]``.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from src.frontend.i18n import t
from src.pricing.rate_optimizer import (
    DEFAULT_COST_OF_FUNDS,
    DEFAULT_LGD,
    RateOptimizer,
)


def render(ctx: Dict) -> None:
    registry = ctx["registry"]
    preset = ctx["preset_features"]
    lang = ctx.get("lang", "en")

    st.title(t("pricing.title", lang))
    st.caption(t("pricing.caption", lang))

    # ---- Lazy-init optimizer (caches the construction cost) ----
    @st.cache_resource(show_spinner="Initializing rate optimizer…")
    def _build_optimizer(_reg_id: str):
        return RateOptimizer(
            model=registry.lgbm_model,
            feature_cols=list(registry.feature_cols),
            registry=registry,
        )

    optimizer = _build_optimizer("registry_v1")

    # ---- Sweep button ----
    if st.button(t("pricing.spinner", lang), type="primary", key="pricing_run"):
        with st.spinner(t("pricing.spinner", lang)):
            try:
                result = optimizer.score_applicant(preset, applicant_id=ctx.get("preset_name"))
                st.session_state["pricing_result"] = result.to_dict()
                st.session_state["pricing_error"] = None
            except Exception as exc:
                st.session_state["pricing_result"] = None
                st.session_state["pricing_error"] = str(exc)

    err = st.session_state.get("pricing_error")
    if err:
        st.error(f"Rate sweep failed: {err}")
        return

    res = st.session_state.get("pricing_result")
    if res is None:
        st.info("Click **" + t("pricing.spinner", lang) + "** to run the rate sweep.")
        return

    # ---- Header metrics ----
    segment = res["segment"]
    segment_color = {
        "sleeping_dog": "🟦",
        "rate_sensitive": "🟧",
        "neutral": "⚪",
    }.get(segment, "⚪")
    a, b, c, d = st.columns(4)
    a.metric(t("pricing.base_label", lang), f"{res['base_rate']*100:.2f}%")
    b.metric(t("pricing.base_pd_label", lang), f"{res['base_pd']*100:.2f}%")
    c.metric(
        t("pricing.elasticity_header", lang),
        f"{res['elasticity']:+.4f}",
    )
    d.metric(
        "Segment",
        f"{segment_color} {segment}",
    )

    # ---- Rate × P(default) grid table ----
    st.subheader(t("pricing.grid_header", lang))
    st.caption(t("pricing.grid_caption", lang))
    df = pd.DataFrame({
        "Rate (APR)": [f"{r*100:.2f}%" for r in res["rate_grid"]],
        "P(default)": [f"{p*100:.3f}%" for p in res["pd_grid"]],
        "Δ P vs base": [f"{(p - res['base_pd'])*100:+.3f} pp" for p in res["pd_grid"]],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- Recommendation panel ----
    st.subheader(t("pricing.recommendation_header", lang))
    st.caption(t(
        "pricing.recommendation_caption", lang,
        lgd=DEFAULT_LGD, cof=DEFAULT_COST_OF_FUNDS,
    ))
    delta_profit = res["expected_profit_at_recommended"] - res["expected_profit_at_base"]
    a, b, c = st.columns(3)
    a.metric(
        t("pricing.recommended_label", lang),
        f"{res['recommended_rate']*100:.2f}%",
        delta=f"{(res['recommended_rate'] - res['base_rate'])*100:+.2f} pp",
    )
    b.metric(
        t("pricing.expected_profit_label", lang),
        f"{res['expected_profit_at_recommended']:,.0f}",
    )
    c.metric(
        t("pricing.profit_delta_label", lang),
        f"{delta_profit:+,.0f}",
        delta_color="inverse" if delta_profit < 0 else "normal",
    )

    # ---- Segments explainer + reasons ----
    st.subheader(t("pricing.segments_header", lang))
    st.caption(t("pricing.segments_caption", lang))
    st.markdown("---")
    st.subheader(t("pricing.reasons_header", lang))
    for r in res.get("segment_reasons", []):
        st.markdown(f"- {r}")
