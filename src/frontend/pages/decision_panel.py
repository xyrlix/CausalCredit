"""Page 4: Decision Advisory Panel.

Comprehensive decision report combining score, causal effect, SHAP-driven
risk factors, and DiCE counterfactual recommendations into a single
underwriter-facing view.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from src.api.schemas import CreditRequest


def render(ctx: Dict) -> None:
    service = ctx["service"]
    registry = ctx["registry"]
    preset = ctx["preset_features"]
    preset_name = ctx["preset_name"]

    st.title("💡 Decision Advisory Panel")
    st.caption(
        f"Full decision report for preset **{preset_name}** — combines model "
        f"score, SHAP explanations, and DiCE counterfactual recommendations."
    )

    if st.button("📋 Generate decision report", type="primary"):
        req = CreditRequest(
            applicant_id=preset_name, features=preset,
            include_counterfactual=True, include_explanation=True,
        )
        with st.spinner("Generating report…"):
            resp = service.score(req)
        st.session_state["last_report"] = resp.model_dump()

    resp_dict = st.session_state.get("last_report")
    if resp_dict is None:
        st.info("Click **Generate decision report** to build the underwriter package.")
        return

    # ---- Header ----
    a, b, c, d = st.columns(4)
    a.metric("Credit Score", resp_dict["score"])
    b.metric("Default Probability", f"{resp_dict['default_probability'] * 100:.2f}%")
    grade_color = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "E": "🔴"}.get(resp_dict["risk_grade"], "⚪")
    c.metric("Risk Grade", f"{grade_color} {resp_dict['risk_grade']}")
    d.metric("Recommendation", resp_dict["decision_suggestion"].split(" — ")[0])

    st.markdown(f"> **Underwriting recommendation:** {resp_dict['decision_suggestion']}")

    # ---- Tabs ----
    t_risk, t_causal, t_cf, t_raw = st.tabs([
        "1️⃣ Risk factors (SHAP)",
        "2️⃣ Causal evidence",
        "3️⃣ Counterfactual scenarios",
        "🛠 Raw JSON",
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
            st.info("No SHAP explanation in this response.")

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
            st.info("No causal effect summary available.")

    with t_cf:
        cfs = resp_dict.get("counterfactual") or []
        if not cfs or "error" in (cfs[0] if cfs else {}):
            st.info("No counterfactual scenarios found.")
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

    with t_raw:
        st.json(resp_dict)
