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
from src.frontend.i18n import (  # noqa: E402
    DEFAULT_LANG,
    LANG_LABELS,
    SUPPORTED_LANGS,
    current_language,
    t,
)
from src.frontend.pages import (  # noqa: E402
    causal_visualization,
    counterfactual_simulator,
    decision_panel,
    interest_rate_optimizer,
    score_dashboard,
)

# Page title is set before sidebar so the browser tab reflects the active
# language; sidebar radio lives below and overwrites session_state["lang"].
st.set_page_config(
    page_title=t("app.title", DEFAULT_LANG),
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
    "prime": {
        "AMT_CREDIT": 400000, "AMT_ANNUITY": 18000, "AMT_GOODS_PRICE": 380000,
        "AMT_INCOME_TOTAL": 250000, "DAYS_BIRTH": -12775, "DAYS_EMPLOYED": -3650,
        "EXT_SOURCE_2": 0.78, "EXT_SOURCE_3": 0.72, "REGION_RATING_CLIENT": 1,
        "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2, "DAYS_REGISTRATION": -2000,
        "DAYS_ID_PUBLISH": -1200, "REGION_POPULATION_RELATIVE": 0.02,
        "CODE_GENDER": "M", "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Managers",
    },
    "mid_career": {
        "AMT_CREDIT": 700000, "AMT_ANNUITY": 32000, "AMT_GOODS_PRICE": 650000,
        "AMT_INCOME_TOTAL": 180000, "DAYS_BIRTH": -14600, "DAYS_EMPLOYED": -5200,
        "EXT_SOURCE_2": 0.55, "EXT_SOURCE_3": 0.50, "REGION_RATING_CLIENT": 2,
        "CNT_CHILDREN": 2, "CNT_FAM_MEMBERS": 4, "DAYS_REGISTRATION": -4000,
        "DAYS_ID_PUBLISH": -2200, "REGION_POPULATION_RELATIVE": 0.018,
        "CODE_GENDER": "F", "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Core staff",
    },
    "thin_credit": {
        "AMT_CREDIT": 250000, "AMT_ANNUITY": 14000, "AMT_GOODS_PRICE": 225000,
        "AMT_INCOME_TOTAL": 90000, "DAYS_BIRTH": -9100, "DAYS_EMPLOYED": -800,
        "EXT_SOURCE_2": 0.15, "EXT_SOURCE_3": 0.10, "REGION_RATING_CLIENT": 2,
        "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 1, "DAYS_REGISTRATION": -300,
        "DAYS_ID_PUBLISH": -600, "REGION_POPULATION_RELATIVE": 0.025,
        "CODE_GENDER": "M", "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Single / not married", "NAME_HOUSING_TYPE": "With parents",
        "OCCUPATION_TYPE": "Sales staff",
    },
    "high_risk": {
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


def _preset_keys_for(lang: str) -> dict:
    """Return {"prime": "localised label", "mid_career": "...", ...}."""
    return {k: t(f"preset.{k}", lang) for k in PRESETS.keys()}


def get_preset_features(name: str) -> dict:
    """Look up a preset by its localised label (or its key, for tests)."""
    if name in PRESETS:
        return dict(PRESETS[name])
    return dict(next(iter(PRESETS.values())))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
registry = boot_registry()
service = CreditScoringService(registry)

# M8.5d: language picker. Stored in st.session_state["lang"].
lang = st.sidebar.radio(
    "🌐 Language",
    options=list(SUPPORTED_LANGS),
    format_func=lambda x: LANG_LABELS.get(x, x),
    index=0,
    key="lang",
)

st.sidebar.title(t("app.sidebar_title", lang))
st.sidebar.caption(t("app.sidebar_caption", lang, n_features=len(registry.feature_cols)))

# Preset labels in the active language; current selection preserved.
preset_labels = _preset_keys_for(lang)
# Reverse-lookup to find the current selection's preset key (default: "prime")
current_key = "prime"
if "preset_name" in st.session_state:
    for k, v in preset_labels.items():
        if v == st.session_state["preset_name"]:
            current_key = k
            break
    else:
        # Previous label no longer matches (e.g. language switched) — try
        # to find it in the previous-language map.
        for prev_lang in SUPPORTED_LANGS:
            for k, v in _preset_keys_for(prev_lang).items():
                if v == st.session_state["preset_name"]:
                    current_key = k
                    break
            else:
                continue
            break

preset_key = st.sidebar.selectbox(
    t("app.preset_label", lang),
    options=list(preset_labels.keys()),
    format_func=lambda x: preset_labels[x],
    index=list(preset_labels.keys()).index(current_key),
    key="preset_key_widget",
)
preset_name = preset_labels[preset_key]
st.session_state["preset_name"] = preset_name

page_keys = ["app.nav_score", "app.nav_causal", "app.nav_cf", "app.nav_decision", "app.nav_pricing"]
page = st.sidebar.radio(
    t("app.navigation", lang),
    options=page_keys,
    format_func=lambda x: t(x, lang),
    index=0,
    key="page",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"{t('app.about_heading', lang)} — {t('app.about_body', lang)}")


# ---------------------------------------------------------------------------
# Page dispatch
# ---------------------------------------------------------------------------
preset_features = get_preset_features(preset_key)
context = {
    "registry": registry,
    "service": service,
    "preset_features": preset_features,
    "preset_name": preset_name,
    "lang": lang,
}

if page == "app.nav_score":
    score_dashboard.render(context)
elif page == "app.nav_causal":
    causal_visualization.render(context)
elif page == "app.nav_cf":
    counterfactual_simulator.render(context)
elif page == "app.nav_decision":
    decision_panel.render(context)
elif page == "app.nav_pricing":
    interest_rate_optimizer.render(context)
