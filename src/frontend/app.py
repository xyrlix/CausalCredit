"""CausalCredit Streamlit demo.

Single-file Streamlit app that loads the same ModelRegistry as the API,
so it works standalone (no API process required). 4 pages:

  1. Score Dashboard — input applicant, see score / grade / SHAP
  2. Causal Visualization — domain DAG + ATE pre-compute
  3. Counterfactual Simulator — adjust loan terms, observe ΔP
  4. Decision Advisory Panel — full report (score + CFs + evidence)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.dependencies import get_model_registry  # noqa: E402
from src.api.schemas import (  # noqa: E402
    CounterfactualRequest,
    CreditRequest,
    ExplainRequest,
)
from src.api.services import CreditScoringService  # noqa: E402
from src.frontend.pages import (  # noqa: E402
    causal_visualization,
    counterfactual_simulator,
    decision_panel,
    score_dashboard,
)

st.set_page_config(
    page_title="CausalCredit · Causal-Inference Credit Scoring",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Shared resources (registry, service, presets)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models (one-time, ~60s on first run)…")
def boot_registry():
    reg = get_model_registry()
    reg.load()
    return reg


PRESETS = {
    "Prime Customer (35y, high income, high score)": {
        "AMT_CREDIT": 400000, "AMT_ANNUITY": 18000, "AMT_GOODS_PRICE": 380000,
        "AMT_INCOME_TOTAL": 250000, "DAYS_BIRTH": -12775, "DAYS_EMPLOYED": -3650,
        "EXT_SOURCE_2": 0.78, "EXT_SOURCE_3": 0.72, "REGION_RATING_CLIENT": 1,
        "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2, "DAYS_REGISTRATION": -2000,
        "DAYS_ID_PUBLISH": -1200, "REGION_POPULATION_RELATIVE": 0.02,
        "CODE_GENDER": "M", "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Managers",
    },
    "Mid-Career (40y, mortgage holder)": {
        "AMT_CREDIT": 700000, "AMT_ANNUITY": 32000, "AMT_GOODS_PRICE": 650000,
        "AMT_INCOME_TOTAL": 180000, "DAYS_BIRTH": -14600, "DAYS_EMPLOYED": -5200,
        "EXT_SOURCE_2": 0.55, "EXT_SOURCE_3": 0.50, "REGION_RATING_CLIENT": 2,
        "CNT_CHILDREN": 2, "CNT_FAM_MEMBERS": 4, "DAYS_REGISTRATION": -4000,
        "DAYS_ID_PUBLISH": -2200, "REGION_POPULATION_RELATIVE": 0.018,
        "CODE_GENDER": "F", "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Core staff",
    },
    "Thin Credit (25y, no history)": {
        "AMT_CREDIT": 250000, "AMT_ANNUITY": 14000, "AMT_GOODS_PRICE": 225000,
        "AMT_INCOME_TOTAL": 90000, "DAYS_BIRTH": -9100, "DAYS_EMPLOYED": -800,
        "EXT_SOURCE_2": 0.15, "EXT_SOURCE_3": 0.10, "REGION_RATING_CLIENT": 2,
        "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 1, "DAYS_REGISTRATION": -300,
        "DAYS_ID_PUBLISH": -600, "REGION_POPULATION_RELATIVE": 0.025,
        "CODE_GENDER": "M", "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Single / not married", "NAME_HOUSING_TYPE": "With parents",
        "OCCUPATION_TYPE": "Sales staff",
    },
    "High-Risk (30y, low income, high debt)": {
        "AMT_CREDIT": 900000, "AMT_ANNUITY": 45000, "AMT_GOODS_PRICE": 850000,
        "AMT_INCOME_TOTAL": 70000, "DAYS_BIRTH": -10950, "DAYS_EMPLOYED": -460,
        "EXT_SOURCE_2": 0.05, "EXT_SOURCE_3": 0.03, "REGION_RATING_CLIENT": 3,
        "CNT_CHILDREN": 3, "CNT_FAM_MEMBERS": 5, "DAYS_REGISTRATION": -1500,
        "DAYS_ID_PUBLISH": -2800, "REGION_POPULATION_RELATIVE": 0.005,
        "CODE_GENDER": "M", "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Civil marriage", "NAME_HOUSING_TYPE": "Rented apartment",
        "OCCUPATION_TYPE": "Laborers",
    },
}


def get_preset_features(name: str) -> dict:
    return dict(PRESETS.get(name, next(iter(PRESETS.values()))))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
registry = boot_registry()
service = CreditScoringService(registry)

st.sidebar.title("CausalCredit")
st.sidebar.caption(f"Loaded: {len(registry.feature_cols)} features · Cache: registry_v1.pkl")
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Score Dashboard",
        "🔬 Causal Visualization",
        "🔄 Counterfactual Simulator",
        "💡 Decision Advisory Panel",
    ],
    index=0,
)

preset_name = st.sidebar.selectbox(
    "Preset Applicant",
    list(PRESETS.keys()),
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About** — CausalCredit is a credit-scoring system that augments "
    "ML predictions with causal inference (DoWhy ATE, EconML CATE, DiCE "
    "counterfactuals)."
)


# ---------------------------------------------------------------------------
# Page dispatch
# ---------------------------------------------------------------------------
preset_features = get_preset_features(preset_name)
context = {
    "registry": registry,
    "service": service,
    "preset_features": preset_features,
    "preset_name": preset_name,
}

if page.startswith("📊"):
    score_dashboard.render(context)
elif page.startswith("🔬"):
    causal_visualization.render(context)
elif page.startswith("🔄"):
    counterfactual_simulator.render(context)
elif page.startswith("💡"):
    decision_panel.render(context)
