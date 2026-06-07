"""Page 3: Counterfactual Simulator.

User adjusts loan terms with sliders and sees default-probability delta in
real time, plus DiCE-generated counterfactual scenarios with feature deltas.
Strings flow through :func:`src.frontend.i18n.t` keyed on ``ctx["lang"]``
(M8.5d).
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from src.api.schemas import CounterfactualRequest
from src.frontend.i18n import t


def render(ctx: Dict) -> None:
    service = ctx["service"]
    registry = ctx["registry"]
    preset = ctx["preset_features"]
    lang = ctx.get("lang", "en")

    st.title(t("cf.title", lang))
    st.caption(t("cf.caption", lang))

    # ---- Baseline ----
    st.subheader(t("cf.baseline_header", lang))
    base_credit = int(preset["AMT_CREDIT"])
    base_annuity = int(preset["AMT_ANNUITY"])
    base_income = int(preset["AMT_INCOME_TOTAL"])
    base_emp = int(preset["DAYS_EMPLOYED"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("cf.metric_loan", lang), f"{base_credit:,}")
    c2.metric(t("cf.metric_annuity", lang), f"{base_annuity:,}")
    c3.metric(t("cf.metric_income", lang), f"{base_income:,}")
    c4.metric(t("cf.metric_emp", lang), f"{base_emp:,}")

    # ---- Intervention sliders ----
    st.subheader(t("cf.intervention_header", lang))
    s1, s2 = st.columns(2)
    with s1:
        new_credit = st.slider("AMT_CREDIT", base_credit // 4, base_credit * 2, base_credit, step=10000)
        new_annuity = st.slider("AMT_ANNUITY", base_annuity // 4, base_annuity * 2, base_annuity, step=500)
    with s2:
        new_income = st.slider("AMT_INCOME_TOTAL", base_income // 2, base_income * 3, base_income, step=5000)
        new_emp = st.slider("DAYS_EMPLOYED", base_emp - 5000, 0, base_emp, step=180)

    interventions = {}
    if new_credit != base_credit:
        interventions["AMT_CREDIT"] = new_credit
    if new_annuity != base_annuity:
        interventions["AMT_ANNUITY"] = new_annuity
    if new_income != base_income:
        interventions["AMT_INCOME_TOTAL"] = new_income
    if new_emp != base_emp:
        interventions["DAYS_EMPLOYED"] = new_emp

    if not interventions:
        st.info(t("cf.idle_hint", lang))
        return

    req = CounterfactualRequest(features=preset, interventions=interventions)
    with st.spinner(t("cf.spinner", lang)):
        resp = service.counterfactual(req)

    a, b, c, d = st.columns(4)
    a.metric(t("cf.metric_base", lang), f"{resp.baseline_probability * 100:.2f}%")
    b.metric(t("cf.metric_new", lang), f"{resp.counterfactual_probability * 100:.2f}%",
             delta=f"{resp.probability_change * 100:+.2f} pp",
             delta_color="inverse")
    c.metric(t("cf.metric_plausibility", lang), f"{resp.confidence:.2f}")
    d.metric(t("cf.metric_n_interventions", lang), len(interventions))

    st.subheader(t("cf.details_header", lang))
    st.json(interventions)

    # ---- DiCE NSGA-II generated CFs ----
    st.subheader(t("cf.dice_header", lang))
    if registry.counterfactual_reasoner is None:
        st.warning(t("cf.dice_unavailable", lang))
        return
    try:
        # Pull all features through the same encoder as the model
        feats_encoded = registry.transform_features(preset).iloc[0].to_dict()
        feats_encoded = {k: float(v) for k, v in feats_encoded.items()}
        cf_res = registry.counterfactual_reasoner.generate_counterfactuals(
            feats_encoded, total_cfs=3, desired_class=0,
        )
        cfs = cf_res.get("cfs", [])
        if not cfs:
            st.info(t("cf.dice_no_cfs", lang))
            return
        rows = []
        for cf in cfs:
            top_changes = sorted(cf["deltas"].items(), key=lambda x: -abs(x[1]))[:3]
            rows.append({
                "CF #": cf["cf_index"],
                "New P(default)": f"{cf['counterfactual_proba']:.3f}",
                "Δ P": f"{cf['delta_proba']:+.3f}",
                "Plausibility": f"{cf['causal_plausibility']:.2f}",
                "Top changes": ", ".join(f"{k}={v:+.0f}" if abs(v) > 1 else f"{k}={v:+.3f}"
                                          for k, v in top_changes),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"DiCE failed: {exc}")
