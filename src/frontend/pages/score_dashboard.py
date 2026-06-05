"""Page 1: Score Dashboard.

Input applicant features, get credit score / risk grade / SHAP explanation
in one panel.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from src.api.schemas import CreditRequest


def render(ctx: Dict) -> None:
    service = ctx["service"]
    preset = ctx["preset_features"]

    st.title("📊 Score Dashboard")
    st.caption("Enter applicant features and see credit score, risk grade, and top SHAP drivers.")

    # -------- Input form --------
    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amt_credit = st.number_input("AMT_CREDIT (loan)", value=int(preset["AMT_CREDIT"]), step=10000)
            amt_annuity = st.number_input("AMT_ANNUITY (yearly)", value=int(preset["AMT_ANNUITY"]), step=1000)
            amt_goods = st.number_input("AMT_GOODS_PRICE", value=int(preset["AMT_GOODS_PRICE"]), step=10000)
            amt_income = st.number_input("AMT_INCOME_TOTAL", value=int(preset["AMT_INCOME_TOTAL"]), step=5000)
        with c2:
            days_birth = st.number_input("DAYS_BIRTH (negative)", value=int(preset["DAYS_BIRTH"]), step=365)
            days_employed = st.number_input("DAYS_EMPLOYED (negative)", value=int(preset["DAYS_EMPLOYED"]), step=180)
            ext2 = st.slider("EXT_SOURCE_2 (0-1)", 0.0, 1.0, float(preset["EXT_SOURCE_2"]), 0.01)
            ext3 = st.slider("EXT_SOURCE_3 (0-1)", 0.0, 1.0, float(preset["EXT_SOURCE_3"]), 0.01)
        with c3:
            region = st.selectbox("REGION_RATING_CLIENT", [1, 2, 3],
                                  index=int(preset["REGION_RATING_CLIENT"]) - 1)
            cnt_child = st.number_input("CNT_CHILDREN", value=int(preset["CNT_CHILDREN"]), step=1)
            cnt_fam = st.number_input("CNT_FAM_MEMBERS", value=int(preset["CNT_FAM_MEMBERS"]), step=1)
            gender = st.selectbox("CODE_GENDER", ["M", "F"],
                                  index=0 if preset.get("CODE_GENDER", "M") == "M" else 1)

        with st.expander("Advanced (categorical features)"):
            education = st.selectbox("NAME_EDUCATION_TYPE", [
                "Secondary / secondary special", "Higher education",
                "Incomplete higher", "Lower secondary",
            ], index=0 if preset.get("NAME_EDUCATION_TYPE", "").startswith("Sec") else 1)
            family = st.selectbox("NAME_FAMILY_STATUS", [
                "Married", "Single / not married", "Civil marriage", "Separated", "Widow",
            ], index=0)
            housing = st.selectbox("NAME_HOUSING_TYPE", [
                "House / apartment", "With parents", "Rented apartment", "Municipal apartment",
            ], index=0)

        submitted = st.form_submit_button("🔍 Score Applicant", type="primary", use_container_width=True)

    if not submitted:
        st.info("Adjust inputs and click **Score Applicant** to run the model.")
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
    with st.spinner("Running model + SHAP…"):
        resp = service.score(req)

    # -------- Result panel --------
    st.markdown("---")
    a, b, c, d = st.columns(4)
    a.metric("Credit Score", resp.score)
    b.metric("Default Probability", f"{resp.default_probability * 100:.2f}%")
    grade_color = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "E": "🔴"}.get(resp.risk_grade, "⚪")
    c.metric("Risk Grade", f"{grade_color} {resp.risk_grade}")
    d.metric("Decision", resp.decision_suggestion.split(" — ")[0])

    st.caption(f"**Recommendation:** {resp.decision_suggestion}")

    # SHAP table
    if resp.explanation and "top_features" in resp.explanation:
        st.subheader("Top SHAP Drivers")
        df = pd.DataFrame(resp.explanation["top_features"])
        df["|shap|"] = df["shap"].abs()
        df = df.sort_values("|shap|", ascending=False)[["feature", "value", "shap", "direction"]]
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Causal context
    if resp.causal_effect:
        st.subheader("Causal Context (ATE)")
        ce = resp.causal_effect
        st.write(
            f"**Treatment:** {ce.get('treatment')}  ·  "
            f"**ATE:** {ce.get('ate'):+.4f}  ·  "
            f"**95% CI:** [{ce.get('ci_lower'):+.4f}, {ce.get('ci_upper'):+.4f}]"
        )
        st.caption(f"Source: {ce.get('method')}")
