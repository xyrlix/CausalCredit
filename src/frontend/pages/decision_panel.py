"""Page 4: Decision Advisory Panel.

Comprehensive decision report combining score, causal effect, SHAP-driven
risk factors, and DiCE counterfactual recommendations into a single
underwriter-facing view. Tab 5 hosts the M8.2 multi-level causal narrative
(model / cohort / individual / robustness) with multi-language rendering.
All UI strings flow through :func:`src.frontend.i18n.t` keyed on
``ctx["lang"]`` (M8.5d). The narrative's own text rendering still uses
its own language codes (zh / zh-HK / en) — see ``_NARRATIVE_LABELS`` in
``src.explain.causal_narrative``.
"""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from src.api.schemas import CreditRequest
from src.explain.causal_narrative import CausalNarrative
from src.frontend.i18n import t


# ---------------------------------------------------------------------------
# Helpers (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing global SHAP + 4-quadrant (one-time, ~30s)…")
def _global_narrative_context(_registry_id: str, sample_size: int, seed: int) -> Dict:
    """Compute global SHAP, four-quadrant labels, and training probabilities.

    Cached on (sample_size, seed) — the registry itself is referenced by
    id only so Streamlit does not try to hash the (non-picklable) LGBM
    model. The returned object holds SHAP matrix + DataFrame + 4-quadrant
    labels; per-applicant SHAP rows are derived downstream.
    """
    from src.api.dependencies import get_model_registry
    reg = get_model_registry()
    if not reg.is_loaded():
        reg.load()
    feature_cols = reg.feature_cols
    training = reg.training_data[feature_cols]
    sample = training.sample(n=min(sample_size, len(training)), random_state=seed)
    sv_global = reg.shap_explainer.compute_shap_values(sample)
    fq = reg.shap_explainer.causal_vs_noncausal_contribution(
        sv_global, sample, causal_features=[],
        threshold_shap=None, threshold_causal=None,
    )
    y_prob_sample = reg.lgbm_model.predict_proba(sample)[:, 1]
    return {
        "shap_global": sv_global,
        "X_sample": sample,
        "y_prob_sample": y_prob_sample,
        "four_quadrant": fq,
        "feature_cols": list(feature_cols),
    }


def _build_dag(registry) -> nx.DiGraph:
    """Convert HomeCreditCausalGraph to networkx.DiGraph for path tracing."""
    g = nx.DiGraph()
    hcg = registry.causal_graph
    if hcg is None:
        return g
    for node_id, props in hcg.nodes.items():
        g.add_node(node_id, type=props.get("type", "unknown"),
                   label=props.get("label", node_id))
    for src, dst in hcg.edges:
        g.add_edge(src, dst)
    return g


def _render_narrative_section(narrative: Dict, language: str, lang: str) -> None:
    """Render the M8.2 narrative markdown + structured tables in a tab.

    ``language`` is the narrative-specific code (zh / zh-HK / en) — it
    controls the *content* of the narrative text. ``lang`` is the UI
    language (en / zh / zh-HK) — it controls the surrounding labels.
    """
    md = CausalNarrative.render_markdown(narrative, language=language)
    st.markdown(md)

    # ----- Structured tables (parsed from the dict for clearer display) -----
    ind = narrative.get("individual_level", {})
    if ind.get("top_features"):
        st.markdown("---")
        st.markdown(t("decision.top_features_header", lang))
        rows = []
        for f in ind["top_features"]:
            paths = f.get("dag_paths", [])
            path_str = " | ".join(" → ".join(p) for p in paths[:2]) or "—"
            rows.append({
                "Feature": f["feature"],
                "Value": (round(f["value"], 3) if f.get("value") is not None else "N/A"),
                "SHAP": round(f["shap"], 4),
                "Quadrant": f["quadrant"],
                "DAG path(s)": path_str,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Quick quadrant counts
        counts = {
            "TRUSTED": ind.get("n_trusted", 0),
            "UNTRUSTED": ind.get("n_untrusted", 0),
            "MASKED": ind.get("n_masked", 0),
        }
        c1, c2, c3 = st.columns(3)
        c1.metric(t("decision.quadrant_trusted", lang), counts["TRUSTED"])
        c2.metric(t("decision.quadrant_untrusted", lang), counts["UNTRUSTED"])
        c3.metric(t("decision.quadrant_masked", lang), counts["MASKED"])

    rob = narrative.get("robustness")
    if rob:
        st.markdown("---")
        st.markdown(t("decision.robustness_header", lang))
        r1, r2, r3 = st.columns(3)
        r1.metric("Stability score", round(rob["stability_score"], 2))
        r2.metric("Top-1 stable", f"{rob['top_1_stable']:.0%}")
        r3.metric("Top-3 stable", f"{rob['top_3_stable']:.0%}")
        st.caption(rob["interpretation"])
        st.caption(
            f"{rob['n_perturbations']} perturbations × {rob['noise_frac']*100:.0f}% Gaussian noise"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render(ctx: Dict) -> None:
    service = ctx["service"]
    registry = ctx["registry"]
    preset = ctx["preset_features"]
    preset_name = ctx["preset_name"]
    lang = ctx.get("lang", "en")

    st.title(t("decision.title", lang))
    st.caption(t("decision.caption", lang, preset_name=preset_name))

    # Sidebar options for the narrative tab
    with st.expander(t("decision.narrative_options", lang), expanded=False):
        # Narrative language is independent of UI language — the user may
        # want English UI but Chinese narrative text, etc. Default to the
        # narrative code that matches the UI language.
        default_narrative = lang if lang in ("zh", "zh-HK", "en") else "zh"
        idx = ["zh", "zh-HK", "en"].index(default_narrative)
        language = st.selectbox(
            t("decision.narrative_language", lang),
            options=["zh", "zh-HK", "en"],
            format_func=lambda x: {"zh": "简体中文", "zh-HK": "繁體 (港式)", "en": "English"}[x],
            index=idx,
        )
        run_robustness = st.checkbox(
            t("decision.narrative_robustness", lang),
            value=True,
        )

    if st.button(t("decision.generate", lang), type="primary"):
        req = CreditRequest(
            applicant_id=preset_name, features=preset,
            include_counterfactual=True, include_explanation=True,
        )
        with st.spinner(t("decision.spinner", lang)):
            resp = service.score(req)
            # ----- M8.2 narrative (computed alongside the main report) -----
            try:
                X1 = registry.transform_features(preset)
                shap_row = registry.shap_explainer.compute_shap_values(X1)
                if shap_row.ndim == 2:
                    shap_row = shap_row[0]
                ctx_g = _global_narrative_context(
                    _registry_id="registry_v1", sample_size=200, seed=0,
                )
                # Per-applicant y_prob_train (cohort uses kNN over training)
                y_prob_train_full = registry.lgbm_model.predict_proba(
                    registry.training_data[registry.feature_cols]
                )[:, 1]
                dag = _build_dag(registry)
                cn = CausalNarrative(
                    model=registry.lgbm_model,
                    feature_names=registry.feature_cols,
                    dag=dag,
                )
                narrative = cn.build_full_narrative(
                    features=preset, shap_row=shap_row,
                    shap_global=ctx_g["shap_global"],
                    X_train=registry.training_data[registry.feature_cols],
                    y_prob_train=y_prob_train_full,
                    four_quadrant=ctx_g["four_quadrant"],
                    run_robustness=run_robustness,
                )
                st.session_state["last_narrative"] = narrative
                st.session_state["last_narrative_language"] = language
            except Exception as exc:
                st.session_state["last_narrative_error"] = str(exc)
        st.session_state["last_report"] = resp.model_dump()

    resp_dict = st.session_state.get("last_report")
    if resp_dict is None:
        st.info(t("decision.idle_hint", lang))
        return

    # ---- Header ----
    a, b, c, d = st.columns(4)
    a.metric(t("decision.metric_score", lang), resp_dict["score"])
    b.metric(t("decision.metric_pd", lang), f"{resp_dict['default_probability'] * 100:.2f}%")
    grade_color = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "E": "🔴"}.get(resp_dict["risk_grade"], "⚪")
    c.metric(t("decision.metric_grade", lang), f"{grade_color} {resp_dict['risk_grade']}")
    d.metric(t("decision.metric_rec", lang), resp_dict["decision_suggestion"].split(" — ")[0])

    st.markdown(f"{t('decision.underwriting_rec', lang)} {resp_dict['decision_suggestion']}")

    # ---- Tabs ----
    t_risk, t_causal, t_cf, t_narr, t_raw = st.tabs([
        t("decision.tab_risk", lang),
        t("decision.tab_causal", lang),
        t("decision.tab_cf", lang),
        t("decision.tab_narr", lang),
        t("decision.tab_raw", lang),
    ])

    with t_risk:
        if resp_dict.get("explanation") and "top_features" in resp_dict["explanation"]:
            df = pd.DataFrame(resp_dict["explanation"]["top_features"])
            df["|shap|"] = df["shap"].abs()
            st.dataframe(
                df.sort_values("|shap|", ascending=False)[
                    ["feature", "value", "shap", "direction"]
                ],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info(t("decision.no_shap", lang))

    with t_causal:
        if resp_dict.get("causal_effect"):
            ce = resp_dict["causal_effect"]
            st.markdown(
                f"- **Treatment:** `{ce.get('treatment')}`\n"
                f"- **Outcome:** `{ce.get('outcome')}`\n"
                f"- **ATE:** {ce.get('ate'):+.4f}\n"
                f"- **95% CI:** [{ce.get('ci_lower'):+.4f}, {ce.get('ci_upper'):+.4f}]\n"
                f"- **Method:** {ce.get('method')}"
            )
        else:
            st.info(t("decision.no_causal", lang))

    with t_cf:
        cfs = resp_dict.get("counterfactual") or []
        if not cfs or "error" in (cfs[0] if cfs else {}):
            st.info(t("decision.no_cfs", lang))
        else:
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

    with t_narr:
        narr = st.session_state.get("last_narrative")
        if narr is None:
            err = st.session_state.get("last_narrative_error")
            if err:
                st.error(f"Narrative generation failed: {err}")
            else:
                st.info(t("decision.narrative_idle", lang))
        else:
            lang_now = st.session_state.get("last_narrative_language", "zh")
            if lang_now != language:
                st.caption(f"Re-rendering in **{language}**…")
                st.session_state["last_narrative_language"] = language
            _render_narrative_section(narr, language=language, lang=lang)
            with st.expander(t("decision.view_raw_narrative", lang)):
                st.json(narr)

    with t_raw:
        st.json(resp_dict)
