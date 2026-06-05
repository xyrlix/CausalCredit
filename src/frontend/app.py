"""CausalCredit Streamlit Demo Application.

Interactive demo showing:
- Scoring Dashboard
- Causal Effect Visualization
- Counterfactual Simulator
- Decision Advisory Panel
"""

import streamlit as st

st.set_page_config(
    page_title="CausalCredit - Causal Inference Enhanced Credit Scoring",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

page = st.sidebar.selectbox("Navigation", [
    ":bar_chart: Scoring Dashboard",
    ":mag: Causal Effect Visualization",
    ":arrows_counterclockwise: Counterfactual Simulator",
    ":bulb: Decision Advisory Panel",
])

preset = st.sidebar.selectbox("Preset Applicant Profile", [
    "Custom",
    "Thin Credit Youth (25, no credit history)",
    "Mid-Career Adult (40, mortgage holder)",
    "Prime Customer (35, high income, high score)",
    "High-Risk Applicant (30, low income, high debt)",
])

if page == ":bar_chart: Scoring Dashboard":
    st.title("Scoring Dashboard")
    st.info("Score dashboard — enter applicant features to see credit score and risk grade.")
elif page == ":mag: Causal Effect Visualization":
    st.title("Causal Effect Visualization")
    st.info("Causal DAG + ATE results + CATE distributions.")
elif page == ":arrows_counterclockwise: Counterfactual Simulator":
    st.title("Counterfactual Simulator")
    st.info("Adjust loan terms and see how default probability changes.")
elif page == ":bulb: Decision Advisory Panel":
    st.title("Decision Advisory Panel")
    st.info("Comprehensive decision report with causal analysis and recommendations.")
