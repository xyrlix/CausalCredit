"""Page 1: Score Dashboard.

Input applicant features, get credit score / risk grade / SHAP explanation
in one panel. All user-facing strings flow through :func:`src.frontend.i18n.t`
keyed on ``ctx["lang"]`` (M8.5d).
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from src.api.schemas import CreditRequest
from src.frontend.i18n import t


def render(ctx: Dict) -> None:
    service = ctx["service"]
    preset = ctx["preset_features"]
    lang = ctx.get("lang", "en")

    st.title(t("score_dashboard.title", lang))
    st.caption(t("score_dashboard.caption", lang))

    # -------- Input form --------
    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amt_credit = st.number_input(t("score_dashboard.form_loan", lang),
                                         value=int(preset["AMT_CREDIT"]), step=10000)
            amt_annuity = st.number_input(t("score_dashboard.form_annuity", lang),
                                          value=int(preset["AMT_ANNUITY"]), step=1000)
            amt_goods = st.number_input(t("score_dashboard.form_goods", lang),
                                        value=int(preset["AMT_GOODS_PRICE"]), step=10000)
            amt_income = st.number_input(t("score_dashboard.form_income", lang),
                                         value=int(preset["AMT_INCOME_TOTAL"]), step=5000)
        with c2:
            days_birth = st.number_input(t("score_dashboard.form_dob", lang),
                                         value=int(preset["DAYS_BIRTH"]), step=365)
            days_employed = st.number_input(t("score_dashboard.form_emp", lang),
                                            value=int(preset["DAYS_EMPLOYED"]), step=180)
            ext2 = st.slider(t("score_dashboard.form_ext2", lang), 0.0, 1.0,
                             float(preset["EXT_SOURCE_2"]), 0.01)
            ext3 = st.slider(t("score_dashboard.form_ext3", lang), 0.0, 1.0,
                             float(preset["EXT_SOURCE_3"]), 0.01)
        with c3:
            region = st.selectbox(t("score_dashboard.form_region", lang), [1, 2, 3],
                                  index=int(preset["REGION_RATING_CLIENT"]) - 1)
            cnt_child = st.number_input(t("score_dashboard.form_children", lang),
                                        value=int(preset["CNT_CHILDREN"]), step=1)
            cnt_fam = st.number_input(t("score_dashboard.form_fam", lang),
                                      value=int(preset["CNT_FAM_MEMBERS"]), step=1)
            gender = st.selectbox(t("score_dashboard.form_gender", lang), ["M", "F"],
                                  index=0 if preset.get("CODE_GENDER", "M") == "M" else 1)

        with st.expander(t("score_dashboard.advanced", lang)):
            education = st.selectbox(t("score_dashboard.form_education", lang), [
                "Secondary / secondary special", "Higher education",
                "Incomplete higher", "Lower secondary",
            ], index=0 if preset.get("NAME_EDUCATION_TYPE", "").startswith("Sec") else 1)
            family = st.selectbox(t("score_dashboard.form_family", lang), [
                "Married", "Single / not married", "Civil marriage", "Separated", "Widow",
            ], index=0)
            housing = st.selectbox(t("score_dashboard.form_housing", lang), [
                "House / apartment", "With parents", "Rented apartment", "Municipal apartment",
            ], index=0)

        submitted = st.form_submit_button(
            t("score_dashboard.submit", lang), type="primary", use_container_width=True,
        )

    if not submitted:
        st.info(t("score_dashboard.idle_hint", lang))
        return

    features = {
        "AMT_CREDIT": amt_credit, "AMT_ANNUITY": amt_annuity, "AMT_GOODS_PRICE": amt_goods,
        "AMT_INCOME_TOTAL": amt_income, "DAYS_BIRTH": days_birth, "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3, "REGION_RATING_CLIENT": region,
        "CNT_CHILDREN": cnt_child, "CNT_FAM_MEMBERS": cnt_fam, "CODE_GENDER": gender,
        "NAME_EDUCATION_TYPE": education, "NAME_FAMILY_STATUS": family,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_REGISTRATION": preset.get("DAYS_REGISTRATION", -2000),
        "DAYS_ID_PUBLISH": preset.get("DAYS_ID_PUBLISH", -1500),
        "REGION_POPULATION_RELATIVE": preset.get("REGION_POPULATION_RELATIVE", 0.02),
        "OCCUPATION_TYPE": preset.get("OCCUPATION_TYPE", "Laborers"),
    }
    req = CreditRequest(features=features, include_counterfactual=False, include_explanation=True)
    with st.spinner(t("score_dashboard.spinner", lang)):
        resp = service.score(req)

    # -------- Result panel --------
    st.markdown("---")
    a, b, c, d = st.columns(4)
    a.metric(t("score_dashboard.metric_score", lang), resp.score)
    b.metric(t("score_dashboard.metric_pd", lang), f"{resp.default_probability * 100:.2f}%")
    grade_color = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "E": "🔴"}.get(resp.risk_grade, "⚪")
    c.metric(t("score_dashboard.metric_grade", lang), f"{grade_color} {resp.risk_grade}")
    d.metric(t("score_dashboard.metric_decision", lang), resp.decision_suggestion.split(" — ")[0])

    st.caption(f"{t('score_dashboard.recommendation', lang)} {resp.decision_suggestion}")

    # SHAP table
    if resp.explanation and "top_features" in resp.explanation:
        st.subheader(t("score_dashboard.shap_header", lang))
        df = pd.DataFrame(resp.explanation["top_features"])
        df["|shap|"] = df["shap"].abs()
        df = df.sort_values("|shap|", ascending=False)[["feature", "value", "shap", "direction"]]
        st.dataframe(df, use_container_width=True, hide_index=True)

    # M8.2 — Causal narrative waterfall (from pipeline cache, if present)
    from pathlib import Path
    waterfall_path = Path("output/figures/15_causal_waterfall.png")
    if waterfall_path.exists():
        st.subheader(t("score_dashboard.narrative_header", lang))
        st.caption(t("score_dashboard.narrative_caption", lang))
        st.image(str(waterfall_path), use_container_width=True)

    # Causal context
    if resp.causal_effect:
        st.subheader(t("score_dashboard.causal_header", lang))
        ce = resp.causal_effect
        st.write(
            f"**Treatment:** {ce.get('treatment')}  ·  "
            f"**ATE:** {ce.get('ate'):+.4f}  ·  "
            f"**95% CI:** [{ce.get('ci_lower'):+.4f}, {ce.get('ci_upper'):+.4f}]"
        )
        st.caption(f"Source: {ce.get('method')}")
