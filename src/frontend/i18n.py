"""Streamlit i18n — 3 languages (en / zh / zh-HK) for the front-end.

@requirement NFR-009
@design docs/plans/2026-06-05-causalcredit-architecture-design.md §D8

Three languages supported out of the box:

* ``en``       — English (default fallback)
* ``zh``       — 简体中文
* ``zh-HK``    — 繁體中文 (港式金融業術語)

The lookup is key-based: ``t("score_dashboard.title", lang)`` returns
the same meaning in any of the 3 languages (or English if the key is
missing — the front-end never crashes on an untranslated key).

Streamlit pages call :func:`t` directly with the language chosen in the
sidebar (stored in ``st.session_state["lang"]``). :func:`language_picker`
renders the radio widget and returns the active language code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Translation tables
# ---------------------------------------------------------------------------
# Convention:
#   - Keys are dot-separated identifiers ("score_dashboard.title")
#   - The dict has the same shape in every language; missing keys fall
#     back to English, then to the key itself (so a typo is visible).
#   - Technical column names (AMT_CREDIT, EXT_SOURCE_2, ...) are NOT
#     translated — they are model inputs, not UI strings.
# ---------------------------------------------------------------------------

_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        # App-level
        "app.title": "CausalCredit · Causal-Inference Credit Scoring",
        "app.sidebar_title": "CausalCredit",
        "app.sidebar_caption": "Loaded: {n_features} features · Cache: registry_v1.pkl",
        "app.navigation": "Navigation",
        "app.nav_score": "📊 Score Dashboard",
        "app.nav_causal": "🔬 Causal Visualization",
        "app.nav_cf": "🔄 Counterfactual Simulator",
        "app.nav_decision": "💡 Decision Advisory Panel",
        "app.preset_label": "Preset Applicant",
        "app.about_heading": "**About**",
        "app.about_body": (
            "CausalCredit is a credit-scoring system that augments ML predictions "
            "with causal inference (DoWhy ATE, EconML CATE, DiCE counterfactuals)."
        ),

        # Score Dashboard
        "score_dashboard.title": "📊 Score Dashboard",
        "score_dashboard.caption": (
            "Enter applicant features and see credit score, risk grade, "
            "and top SHAP drivers."
        ),
        "score_dashboard.form_loan": "AMT_CREDIT (loan)",
        "score_dashboard.form_annuity": "AMT_ANNUITY (yearly)",
        "score_dashboard.form_goods": "AMT_GOODS_PRICE",
        "score_dashboard.form_income": "AMT_INCOME_TOTAL",
        "score_dashboard.form_dob": "DAYS_BIRTH (negative)",
        "score_dashboard.form_emp": "DAYS_EMPLOYED (negative)",
        "score_dashboard.form_ext2": "EXT_SOURCE_2 (0-1)",
        "score_dashboard.form_ext3": "EXT_SOURCE_3 (0-1)",
        "score_dashboard.form_region": "REGION_RATING_CLIENT",
        "score_dashboard.form_children": "CNT_CHILDREN",
        "score_dashboard.form_fam": "CNT_FAM_MEMBERS",
        "score_dashboard.form_gender": "CODE_GENDER",
        "score_dashboard.advanced": "Advanced (categorical features)",
        "score_dashboard.form_education": "NAME_EDUCATION_TYPE",
        "score_dashboard.form_family": "NAME_FAMILY_STATUS",
        "score_dashboard.form_housing": "NAME_HOUSING_TYPE",
        "score_dashboard.submit": "🔍 Score Applicant",
        "score_dashboard.idle_hint": "Adjust inputs and click **Score Applicant** to run the model.",
        "score_dashboard.spinner": "Running model + SHAP…",
        "score_dashboard.metric_score": "Credit Score",
        "score_dashboard.metric_pd": "Default Probability",
        "score_dashboard.metric_grade": "Risk Grade",
        "score_dashboard.metric_decision": "Decision",
        "score_dashboard.recommendation": "**Recommendation:**",
        "score_dashboard.shap_header": "Top SHAP Drivers",
        "score_dashboard.narrative_header": "Causal Narrative (M8.2)",
        "score_dashboard.narrative_caption": (
            "Top features with 4-quadrant labels — TRUSTED (model+causal agree), "
            "UNTRUSTED (model only), MASKED (causal only), NEGLIGIBLE."
        ),
        "score_dashboard.causal_header": "Causal Context (ATE)",

        # Causal Visualization
        "causal.title": "🔬 Causal Visualization",
        "causal.caption": "Domain causal DAG, pre-computed ATE, and pipeline charts.",
        "causal.dag_header": "Domain Causal Graph (DAG)",
        "causal.dag_unavailable": "Domain DAG not available in registry.",
        "causal.dag_render_failed": "Could not render DAG via Graphviz: {exc}",
        "causal.treatments_outcome": (
            "**Treatments:** `{treatments}`  ·  **Outcome:** `{outcome}`  ·  "
            "**Nodes:** {n_nodes}  ·  **Edges:** {n_edges}"
        ),
        "causal.ate_header": "Average Treatment Effect (ATE)",
        "causal.ate_estimate": "ATE estimate",
        "causal.ate_ci_lower": "95% CI lower",
        "causal.ate_ci_upper": "95% CI upper",
        "causal.ate_caption": (
            "**Treatment:** {treatment}  ·  **Outcome:** {outcome}  ·  **Method:** {method}"
        ),
        "causal.ate_unavailable": "ATE pre-compute not available — re-train the registry to populate.",
        "causal.charts_header": "Pipeline Charts (from `output/figures/`)",
        "causal.no_charts": "No charts found. Run `python -m src.run_pipeline` first.",
        "causal.no_charts_warn": "No pipeline charts on disk. Run `python -m src.run_pipeline`.",

        # Counterfactual Simulator
        "cf.title": "🔄 Counterfactual Simulator",
        "cf.caption": (
            "Adjust loan terms and see the effect on default probability. "
            "DiCE NSGA-II also generates causally-plausible scenarios that "
            "would flip the decision."
        ),
        "cf.baseline_header": "Baseline applicant",
        "cf.metric_loan": "Loan amount",
        "cf.metric_annuity": "Yearly annuity",
        "cf.metric_income": "Yearly income",
        "cf.metric_emp": "Days employed",
        "cf.intervention_header": "What-if interventions",
        "cf.idle_hint": "Move at least one slider to see the counterfactual effect.",
        "cf.spinner": "Running counterfactual…",
        "cf.metric_base": "Baseline P(default)",
        "cf.metric_new": "New P(default)",
        "cf.metric_plausibility": "Plausibility",
        "cf.metric_n_interventions": "Interventions",
        "cf.details_header": "Intervention details",
        "cf.dice_header": "DiCE NSGA-II suggested counterfactuals (causally-plausible)",
        "cf.dice_unavailable": "CounterfactualReasoner unavailable.",
        "cf.dice_no_cfs": "DiCE could not find counterfactuals for this applicant within plausibility bounds.",

        # Decision Panel
        "decision.title": "💡 Decision Advisory Panel",
        "decision.caption": (
            "Full decision report for preset **{preset_name}** — combines model "
            "score, SHAP explanations, DiCE counterfactual recommendations, and "
            "the M8.2 multi-level causal narrative."
        ),
        "decision.narrative_options": "⚙️ Narrative options",
        "decision.narrative_language": "Narrative language",
        "decision.narrative_robustness": (
            "Run robustness test (20 perturbations × TreeSHAP, ~5s extra)"
        ),
        "decision.generate": "📋 Generate decision report",
        "decision.spinner": "Generating report…",
        "decision.idle_hint": "Click **Generate decision report** to build the underwriter package.",
        "decision.metric_score": "Credit Score",
        "decision.metric_pd": "Default Probability",
        "decision.metric_grade": "Risk Grade",
        "decision.metric_rec": "Recommendation",
        "decision.underwriting_rec": "> **Underwriting recommendation:**",
        "decision.tab_risk": "1️⃣ Risk factors (SHAP)",
        "decision.tab_causal": "2️⃣ Causal evidence",
        "decision.tab_cf": "3️⃣ Counterfactual scenarios",
        "decision.tab_narr": "📖 Causal Narrative (M8.2)",
        "decision.tab_raw": "🛠 Raw JSON",
        "decision.no_shap": "No SHAP explanation in this response.",
        "decision.no_causal": "No causal effect summary available.",
        "decision.no_cfs": "No counterfactual scenarios found.",
        "decision.narrative_idle": (
            "Click **Generate decision report** above to produce the "
            "M8.2 multi-level causal narrative (model / cohort / "
            "individual / robustness)."
        ),
        "decision.view_raw_narrative": "View raw narrative dict",
        "decision.top_features_header": "**Top features (per-row SHAP, with DAG paths)**",
        "decision.robustness_header": "**Explanation robustness**",
        "decision.quadrant_trusted": "TRUSTED (model + causal agree)",
        "decision.quadrant_untrusted": "UNTRUSTED (model says, no causal)",
        "decision.quadrant_masked": "MASKED (causal hidden by model)",

        # Presets
        "preset.prime": "Prime Customer (35y, high income, high score)",
        "preset.mid_career": "Mid-Career (40y, mortgage holder)",
        "preset.thin_credit": "Thin Credit (25y, no history)",
        "preset.high_risk": "High-Risk (30y, low income, high debt)",
    },

    "zh": {
        # App-level
        "app.title": "CausalCredit · 因果推断信用评分",
        "app.sidebar_title": "CausalCredit",
        "app.sidebar_caption": "已加载：{n_features} 个特征 · 缓存：registry_v1.pkl",
        "app.navigation": "导航",
        "app.nav_score": "📊 评分仪表盘",
        "app.nav_causal": "🔬 因果可视化",
        "app.nav_cf": "🔄 反事实模拟器",
        "app.nav_decision": "💡 决策建议面板",
        "app.preset_label": "申请人预设",
        "app.about_heading": "**关于**",
        "app.about_body": (
            "CausalCredit 是一个信用评分系统，在机器学习预测基础上引入因果推断 "
            "（DoWhy ATE、EconML CATE、DiCE 反事实）。"
        ),

        # Score Dashboard
        "score_dashboard.title": "📊 评分仪表盘",
        "score_dashboard.caption": "输入申请人特征，查看信用评分、风险等级和主要 SHAP 驱动因子。",
        "score_dashboard.form_loan": "AMT_CREDIT（贷款金额）",
        "score_dashboard.form_annuity": "AMT_ANNUITY（年还款额）",
        "score_dashboard.form_goods": "AMT_GOODS_PRICE",
        "score_dashboard.form_income": "AMT_INCOME_TOTAL",
        "score_dashboard.form_dob": "DAYS_BIRTH（负值）",
        "score_dashboard.form_emp": "DAYS_EMPLOYED（负值）",
        "score_dashboard.form_ext2": "EXT_SOURCE_2（0-1）",
        "score_dashboard.form_ext3": "EXT_SOURCE_3（0-1）",
        "score_dashboard.form_region": "REGION_RATING_CLIENT",
        "score_dashboard.form_children": "CNT_CHILDREN",
        "score_dashboard.form_fam": "CNT_FAM_MEMBERS",
        "score_dashboard.form_gender": "CODE_GENDER",
        "score_dashboard.advanced": "高级（类别特征）",
        "score_dashboard.form_education": "NAME_EDUCATION_TYPE",
        "score_dashboard.form_family": "NAME_FAMILY_STATUS",
        "score_dashboard.form_housing": "NAME_HOUSING_TYPE",
        "score_dashboard.submit": "🔍 评分",
        "score_dashboard.idle_hint": "调整输入后点击 **评分** 即可运行模型。",
        "score_dashboard.spinner": "正在运行模型 + SHAP…",
        "score_dashboard.metric_score": "信用评分",
        "score_dashboard.metric_pd": "违约概率",
        "score_dashboard.metric_grade": "风险等级",
        "score_dashboard.metric_decision": "建议",
        "score_dashboard.recommendation": "**建议：**",
        "score_dashboard.shap_header": "主要 SHAP 驱动",
        "score_dashboard.narrative_header": "因果叙事（M8.2）",
        "score_dashboard.narrative_caption": (
            "主要特征按四象限标注：TRUSTED（模型+因果一致）、"
            "UNTRUSTED（仅模型）、MASKED（仅因果）、NEGLIGIBLE（可忽略）。"
        ),
        "score_dashboard.causal_header": "因果上下文（ATE）",

        # Causal Visualization
        "causal.title": "🔬 因果可视化",
        "causal.caption": "领域因果 DAG、预计算 ATE 以及 pipeline 图表。",
        "causal.dag_header": "领域因果图（DAG）",
        "causal.dag_unavailable": "注册器中暂无领域 DAG。",
        "causal.dag_render_failed": "无法通过 Graphviz 渲染 DAG：{exc}",
        "causal.treatments_outcome": (
            "**处理变量：** `{treatments}`  ·  **结果：** `{outcome}`  ·  "
            "**节点数：** {n_nodes}  ·  **边数：** {n_edges}"
        ),
        "causal.ate_header": "平均处理效应（ATE）",
        "causal.ate_estimate": "ATE 估计",
        "causal.ate_ci_lower": "95% CI 下界",
        "causal.ate_ci_upper": "95% CI 上界",
        "causal.ate_caption": (
            "**处理变量：** {treatment}  ·  **结果：** {outcome}  ·  **方法：** {method}"
        ),
        "causal.ate_unavailable": "ATE 预计算不可用，请重新训练注册器。",
        "causal.charts_header": "Pipeline 图表（来自 `output/figures/`）",
        "causal.no_charts": "未找到图表，请先运行 `python -m src.run_pipeline`。",
        "causal.no_charts_warn": "磁盘上无 pipeline 图表，请运行 `python -m src.run_pipeline`。",

        # Counterfactual Simulator
        "cf.title": "🔄 反事实模拟器",
        "cf.caption": (
            "调整贷款条款，观察其对违约概率的影响。"
            "DiCE NSGA-II 还会生成能翻转决策的因果合理场景。"
        ),
        "cf.baseline_header": "基线申请人",
        "cf.metric_loan": "贷款金额",
        "cf.metric_annuity": "年还款额",
        "cf.metric_income": "年收入",
        "cf.metric_emp": "在职天数",
        "cf.intervention_header": "反事实干预",
        "cf.idle_hint": "至少移动一个滑块以查看反事实效果。",
        "cf.spinner": "正在运行反事实…",
        "cf.metric_base": "基线 P(违约)",
        "cf.metric_new": "新 P(违约)",
        "cf.metric_plausibility": "合理性",
        "cf.metric_n_interventions": "干预数",
        "cf.details_header": "干预详情",
        "cf.dice_header": "DiCE NSGA-II 推荐的反事实（因果合理）",
        "cf.dice_unavailable": "CounterfactualReasoner 不可用。",
        "cf.dice_no_cfs": "在该申请人的合理范围内，DiCE 未能找到反事实。",

        # Decision Panel
        "decision.title": "💡 决策建议面板",
        "decision.caption": (
            "针对预设 **{preset_name}** 的完整决策报告 — 包含模型评分、"
            "SHAP 解释、DiCE 反事实建议以及 M8.2 多层因果叙事。"
        ),
        "decision.narrative_options": "⚙️ 叙事选项",
        "decision.narrative_language": "叙事语言",
        "decision.narrative_robustness": (
            "运行稳健性测试（20 次扰动 × TreeSHAP，约 5 秒）"
        ),
        "decision.generate": "📋 生成决策报告",
        "decision.spinner": "正在生成报告…",
        "decision.idle_hint": "点击 **生成决策报告** 以构建承销包。",
        "decision.metric_score": "信用评分",
        "decision.metric_pd": "违约概率",
        "decision.metric_grade": "风险等级",
        "decision.metric_rec": "建议",
        "decision.underwriting_rec": "> **承销建议：**",
        "decision.tab_risk": "1️⃣ 风险因子（SHAP）",
        "decision.tab_causal": "2️⃣ 因果证据",
        "decision.tab_cf": "3️⃣ 反事实场景",
        "decision.tab_narr": "📖 因果叙事（M8.2）",
        "decision.tab_raw": "🛠 原始 JSON",
        "decision.no_shap": "本响应中无 SHAP 解释。",
        "decision.no_causal": "暂无可用的因果效应摘要。",
        "decision.no_cfs": "未找到反事实场景。",
        "decision.narrative_idle": (
            "点击上方 **生成决策报告** 以产出 M8.2 多层因果叙事"
            "（模型 / 同类申请人 / 个体 / 稳健性）。"
        ),
        "decision.view_raw_narrative": "查看原始叙事 dict",
        "decision.top_features_header": "**主要特征（每行 SHAP，附 DAG 路径）**",
        "decision.robustness_header": "**解释稳健性**",
        "decision.quadrant_trusted": "TRUSTED（模型 + 因果一致）",
        "decision.quadrant_untrusted": "UNTRUSTED（仅模型）",
        "decision.quadrant_masked": "MASKED（被模型掩盖的因果）",

        # Presets
        "preset.prime": "优质客户（35 岁，高收入，高分）",
        "preset.mid_career": "事业中期（40 岁，房贷持有者）",
        "preset.thin_credit": "薄信用（25 岁，无历史）",
        "preset.high_risk": "高风险（30 岁，低收入，高负债）",
    },

    "zh-HK": {
        # App-level — Hong Kong banking terminology
        "app.title": "CausalCredit · 因果推斷信貸評分",
        "app.sidebar_title": "CausalCredit",
        "app.sidebar_caption": "已載入：{n_features} 個特徵 · 緩存：registry_v1.pkl",
        "app.navigation": "導航",
        "app.nav_score": "📊 評分儀表板",
        "app.nav_causal": "🔬 因果可視化",
        "app.nav_cf": "🔄 反事實模擬器",
        "app.nav_decision": "💡 決策建議面板",
        "app.preset_label": "申請人預設",
        "app.about_heading": "**關於**",
        "app.about_body": (
            "CausalCredit 係一個信貸評分系統，喺機器學習預測基礎上加入因果推斷"
            "（DoWhy ATE、EconML CATE、DiCE 反事實）。"
        ),

        # Score Dashboard
        "score_dashboard.title": "📊 評分儀表板",
        "score_dashboard.caption": "輸入申請人特徵，查看信貸評分、風險等級同主要 SHAP 驅動因子。",
        "score_dashboard.form_loan": "AMT_CREDIT（貸款金額）",
        "score_dashboard.form_annuity": "AMT_ANNUITY（年還款額）",
        "score_dashboard.form_goods": "AMT_GOODS_PRICE",
        "score_dashboard.form_income": "AMT_INCOME_TOTAL",
        "score_dashboard.form_dob": "DAYS_BIRTH（負值）",
        "score_dashboard.form_emp": "DAYS_EMPLOYED（負值）",
        "score_dashboard.form_ext2": "EXT_SOURCE_2（0-1）",
        "score_dashboard.form_ext3": "EXT_SOURCE_3（0-1）",
        "score_dashboard.form_region": "REGION_RATING_CLIENT",
        "score_dashboard.form_children": "CNT_CHILDREN",
        "score_dashboard.form_fam": "CNT_FAM_MEMBERS",
        "score_dashboard.form_gender": "CODE_GENDER",
        "score_dashboard.advanced": "進階（類別特徵）",
        "score_dashboard.form_education": "NAME_EDUCATION_TYPE",
        "score_dashboard.form_family": "NAME_FAMILY_STATUS",
        "score_dashboard.form_housing": "NAME_HOUSING_TYPE",
        "score_dashboard.submit": "🔍 評分",
        "score_dashboard.idle_hint": "調整輸入後點擊 **評分** 即可運行模型。",
        "score_dashboard.spinner": "運行緊模型 + SHAP…",
        "score_dashboard.metric_score": "信貸評分",
        "score_dashboard.metric_pd": "違約概率",
        "score_dashboard.metric_grade": "風險等級",
        "score_dashboard.metric_decision": "建議",
        "score_dashboard.recommendation": "**建議：**",
        "score_dashboard.shap_header": "主要 SHAP 驅動",
        "score_dashboard.narrative_header": "因果敘事（M8.2）",
        "score_dashboard.narrative_caption": (
            "主要特徵按四象限標註：TRUSTED（模型+因果一致）、"
            "UNTRUSTED（僅模型）、MASKED（僅因果）、NEGLIGIBLE（可忽略）。"
        ),
        "score_dashboard.causal_header": "因果上下文（ATE）",

        # Causal Visualization
        "causal.title": "🔬 因果可視化",
        "causal.caption": "領域因果 DAG、預計算 ATE 以及 pipeline 圖表。",
        "causal.dag_header": "領域因果圖（DAG）",
        "causal.dag_unavailable": "註冊器中暫無領域 DAG。",
        "causal.dag_render_failed": "無法透過 Graphviz 渲染 DAG：{exc}",
        "causal.treatments_outcome": (
            "**處理變量：** `{treatments}`  ·  **結果：** `{outcome}`  ·  "
            "**節點數：** {n_nodes}  ·  **邊數：** {n_edges}"
        ),
        "causal.ate_header": "平均處理效應（ATE）",
        "causal.ate_estimate": "ATE 估計",
        "causal.ate_ci_lower": "95% CI 下界",
        "causal.ate_ci_upper": "95% CI 上界",
        "causal.ate_caption": (
            "**處理變量：** {treatment}  ·  **結果：** {outcome}  ·  **方法：** {method}"
        ),
        "causal.ate_unavailable": "ATE 預計算不可用，請重新訓練註冊器。",
        "causal.charts_header": "Pipeline 圖表（嚟自 `output/figures/`）",
        "causal.no_charts": "搵唔到圖表，請先運行 `python -m src.run_pipeline`。",
        "causal.no_charts_warn": "磁碟上無 pipeline 圖表，請運行 `python -m src.run_pipeline`。",

        # Counterfactual Simulator
        "cf.title": "🔄 反事實模擬器",
        "cf.caption": (
            "調整貸款條款，觀察佢對違約概率嘅影響。"
            "DiCE NSGA-II 都會生成能夠翻轉決策嘅因果合理場景。"
        ),
        "cf.baseline_header": "基線申請人",
        "cf.metric_loan": "貸款金額",
        "cf.metric_annuity": "年還款額",
        "cf.metric_income": "年收入",
        "cf.metric_emp": "在職天數",
        "cf.intervention_header": "反事實干預",
        "cf.idle_hint": "至少移動一個滑桿以睇反事實效果。",
        "cf.spinner": "運行緊反事實…",
        "cf.metric_base": "基線 P(違約)",
        "cf.metric_new": "新 P(違約)",
        "cf.metric_plausibility": "合理性",
        "cf.metric_n_interventions": "干預數",
        "cf.details_header": "干預詳情",
        "cf.dice_header": "DiCE NSGA-II 推薦嘅反事實（因果合理）",
        "cf.dice_unavailable": "CounterfactualReasoner 不可用。",
        "cf.dice_no_cfs": "喺呢個申請人嘅合理範圍內，DiCE 搵唔到反事實。",

        # Decision Panel
        "decision.title": "💡 決策建議面板",
        "decision.caption": (
            "針對預設 **{preset_name}** 嘅完整決策報告 — 包含模型評分、"
            "SHAP 解釋、DiCE 反事實建議以及 M8.2 多層因果敘事。"
        ),
        "decision.narrative_options": "⚙️ 敘事選項",
        "decision.narrative_language": "敘事語言",
        "decision.narrative_robustness": (
            "運行穩健性測試（20 次擾動 × TreeSHAP，約 5 秒）"
        ),
        "decision.generate": "📋 生成決策報告",
        "decision.spinner": "生成緊報告…",
        "decision.idle_hint": "點擊 **生成決策報告** 以構建承銷包。",
        "decision.metric_score": "信貸評分",
        "decision.metric_pd": "違約概率",
        "decision.metric_grade": "風險等級",
        "decision.metric_rec": "建議",
        "decision.underwriting_rec": "> **承銷建議：**",
        "decision.tab_risk": "1️⃣ 風險因子（SHAP）",
        "decision.tab_causal": "2️⃣ 因果證據",
        "decision.tab_cf": "3️⃣ 反事實場景",
        "decision.tab_narr": "📖 因果敘事（M8.2）",
        "decision.tab_raw": "🛠 原始 JSON",
        "decision.no_shap": "本響應中無 SHAP 解釋。",
        "decision.no_causal": "暫時無可用嘅因果效應摘要。",
        "decision.no_cfs": "搵唔到反事實場景。",
        "decision.narrative_idle": (
            "點擊上方 **生成決策報告** 以產出 M8.2 多層因果敘事"
            "（模型 / 同類申請人 / 個體 / 穩健性）。"
        ),
        "decision.view_raw_narrative": "睇原始敘事 dict",
        "decision.top_features_header": "**主要特徵（每行 SHAP，附 DAG 路徑）**",
        "decision.robustness_header": "**解釋穩健性**",
        "decision.quadrant_trusted": "TRUSTED（模型 + 因果一致）",
        "decision.quadrant_untrusted": "UNTRUSTED（僅模型）",
        "decision.quadrant_masked": "MASKED（被模型掩蓋嘅因果）",

        # Presets
        "preset.prime": "優質客戶（35 歲，高收入，高分）",
        "preset.mid_career": "事業中期（40 歲，按揭持有者）",
        "preset.thin_credit": "薄信貸（25 歲，無歷史）",
        "preset.high_risk": "高風險（30 歲，低收入，高負債）",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("en", "zh", "zh-HK")
LANG_LABELS = {
    "en": "English",
    "zh": "简体中文",
    "zh-HK": "繁體 (港式)",
}


def t(key: str, lang: str = DEFAULT_LANG, **kwargs: Any) -> str:
    """Look up a translation. Falls back to English, then to the key itself.

    Format placeholders in the translation string are filled with
    ``str.format(**kwargs)``. Missing kwargs leave the placeholder in
    place (mimics Python's default format-error behaviour).
    """
    table = _STRINGS.get(lang) or _STRINGS[DEFAULT_LANG]
    text = table.get(key)
    if text is None:
        text = _STRINGS[DEFAULT_LANG].get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            # Placeholder missing — leave the original (with {name} visible)
            pass
    return text


def available_keys(lang: str = DEFAULT_LANG) -> list[str]:
    """List all keys defined for a language (debug / test helper)."""
    return list(_STRINGS.get(lang, {}).keys())


def validate_consistency() -> Dict[str, list[str]]:
    """Return languages with missing keys (compared to English).

    The Streamlit pages should never crash on an untranslated key (the
    fallback in :func:`t` handles that), but this helper lets us assert
    full coverage in tests.
    """
    base = set(_STRINGS[DEFAULT_LANG].keys())
    out: Dict[str, list[str]] = {}
    for lang in SUPPORTED_LANGS:
        if lang == DEFAULT_LANG:
            continue
        missing = sorted(base - set(_STRINGS.get(lang, {}).keys()))
        extra = sorted(set(_STRINGS.get(lang, {}).keys()) - base)
        if missing or extra:
            out[lang] = missing + [f"+{k}" for k in extra]
    return out


def language_picker(default: str = DEFAULT_LANG, key: str = "lang") -> str:
    """Render a Streamlit radio of supported languages, return the active code.

    Stores the choice in ``st.session_state[key]`` so the rest of the
    app can read it via :func:`current_language`.
    """
    import streamlit as st  # local import — module imports without Streamlit at test time

    if key not in st.session_state:
        st.session_state[key] = default
    return st.sidebar.radio(
        t("app.about_heading").split("**")[1] if "**" in t("app.about_heading") else "Language",
        options=list(SUPPORTED_LANGS),
        format_func=lambda x: LANG_LABELS.get(x, x),
        index=list(SUPPORTED_LANGS).index(st.session_state[key]),
        key=f"{key}_radio",
    ) or st.session_state[key]


def current_language(key: str = "lang", default: str = DEFAULT_LANG) -> str:
    """Return the active language code (with fallback to ``default``)."""
    try:
        import streamlit as st  # noqa
        lang = st.session_state.get(key, default)
    except Exception:
        lang = default
    return lang if lang in SUPPORTED_LANGS else default
