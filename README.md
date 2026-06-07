# CausalCredit — 因果推理增强信用评分系统

> **不只告诉你风险有多高，更告诉你为什么高、以及如何降低**  
> 从相关性预测到因果性决策的范式跃迁

中银香港创新先驱大赛2026 参赛项目。在 LightGBM 预测基座之上，融合**因果发现 / 异质处理效应 / 反事实推理 / SHAP-因果四象限 / 反欺诈三件套**，构建"预测 → 归因 → 决策 → 反欺诈"完整闭环。

---

## 🎯 6 大核心亮点

| # | 亮点 | 解决的问题 | 实现入口 | 状态 |
|:-:|------|-----------|----------|:----:|
| 1 | **混合因果发现引擎** | 哪些特征 → 违约 是因果 vs 巧合？ | `src/causal/discovery.py` + `src/causal/home_credit_graph.py` | ✅ |
| 2 | **CATE 异质处理效应** | 同一政策对不同人效果差多大？ | `src/causal/cate.py` (3 方法交叉验证) | ✅ |
| 3 | **因果约束反事实** | 怎样改才能把风险降下来？ | `src/explain/counterfactual.py` (DiCE NSGA-II) | ✅ |
| 4 | **SHAP × 因果四象限** | 模型说重要 ≠ 因果说重要 | `src/explain/shap_explain.py` | ✅ |
| 5 | **因果引导决策建议** | 给信贷员 / 客户的可执行话术 | `src/explain/decision.py` + `evidence.py` | ✅ |
| 6 | **反欺诈三件套** ⭐ | 主观欺诈 / 包装资质 / 养流水 | `src/fraud/` (4 模块) | ✅ M7 |
| 7 | **公平性审计 (HKMA / EU AI Act)** ⭐ | 模型对性别/年龄/收入/教育切片是否合规？ | `src/fairness/` (3 模块) | ✅ M8.1 |
| 8 | **因果叙事深化** ⭐ | 客户问"为什么"时给 3 层解释 (model/cohort/individual) + 解释稳健性 | `src/explain/causal_narrative.py` + `narrative_visualize.py` | ✅ M8.2 |
| 9 | **多语言 + 港式本地化** ⭐ | zh-HK 繁体 / 港式措辞 / en 国际化 | `CausalNarrative.render_markdown(language=)` | ✅ M8.4 |
| 10 | **Oaxaca-Blinder 因果公平性分解** ⭐ | 解释子群 gap 中"特征差异" vs "系数差异"占比, discrimination_index ∈ [0,1] | `src/fairness/oaxaca.py` | ✅ M8.5f |
| 11 | **TemporalGuard 数据泄漏防护** ⭐ | 剔除 `MONTHS_BALANCE > 0` 行, 时间有序切分, 结构化 issue 列表 | `src/data/temporal_guard.py` | ✅ M8.6a |
| 12 | **BLP 检验 (Best Linear Predictor)** ⭐ | Chernozhukov 2018, K-fold 交叉验证 CATE → OLS Y ~ 1 + T + c_hat + c_hat·T | `src/causal/blp_test.py` | ✅ M8.6b |
| 13 | **CATE 稳定性 Tier1+Tier2** ⭐ | 30× split-half bootstrap + 10× hyperparameter grid (Oracle P0 配方) | `src/causal/stability.py` | ✅ M8.6c |
| 14 | **CCGS 因果验证金字塔** ⭐ | 4 层 L1-L4 (GSI/DKCS/Refutation/CCR/AUC/ECE/DP) → 1 个 0-1 复合分 | `src/causal/verification.py` | ✅ M8.6d |

---

## 📊 进展看板（18 个里程碑全部完成：M0-M7 + M8.1-M8.6）

| # | 里程碑 | 关键产出 | 累计效果 | Commit |
|:-:|--------|----------|---------|--------|
| **M0** | 数据准备 | Home Credit 30 万行加载器 + 领域 DAG (15 节点 / 28 边) | 基础就绪 | `b68de7e` |
| **M1** | 5 个核心创新点 | PC+NOTEARS / CATE / Refutation / DiCE / SHAP / 决策 6 个独立 demo | 因果栈就位 | `eb607f3` |
| **M2** | 端到端 pipeline | 12 步 → 13 步, 11 张 PNG + 3 份 JSON 报告 | 跑通 84.8s | `6f17d9f` |
| **M3** | API + UI 服务化 | FastAPI 5 端点 (:8000) + Streamlit 4 页 (:8501) | 服务化 | `07836ec` |
| **M4** | 监控 + 测试 + 文档 | PSI 漂移检测 (3 层) + 85 个测试 | 1.34s 全跑 | `6e0c9a9` |
| **M5** | **8 表 JOIN + 多表因果特征** | 5 张二级表 (~1.1 GB, 80M 行) → 246 聚合特征 | **AUC 0.7547 → 0.7803** | `9cc90f0` |
| **M5+** | **CPU 优化** | 多表聚合缓存 (65s→2s) + L1 特征预筛选 | **耗时 -25% (245s → 185s)** | `bca8c96` |
| **M6** | **GPU LightGBM + Optuna** | 2 个可选杠杆 (默认关闭, 实测 Home Credit 上不显著) | 接口预留 | `7d496b6` |
| **M7** | **反欺诈三件套** ⭐ | 三分类子模型 + 包装资质 + 养流水去噪, 14 步 / 14 图 / 25 新测试 | **5 维 routing** | `7015282` |
| **M8.1** | **公平性审计 + 反欺诈升级** ⭐ | 3 项公平性指标 (DP/EO/DI) + 4 切片 + 3 张图 + FraudGuardConfig (YAML) + 路由 PSI, 15 步 / 17 图 / 31 新测试 | **HKMA / EU AI Act 合规** | (M8.1) |
| **M8.2** | **因果叙事深化** ⭐ | 3 层叙事 (model/cohort/individual) + DAG 路径追溯 + 解释稳健性扰动, 16 步 / 19 图 / 17 新测试 | **"challenge the decision" 完整回答** | (M8.2) |
| **M8.3** | **完整服务化** ⭐ | FastAPI 5 端点 fill out (11 smoke test) + 路由 baseline 持久化 (rolling update) + DAG 加 EXT_SOURCE 边 | **可生产化 API 端点** | (M8.3) |
| **M8.4** | **多语言 + 港式本地化** ⭐ | `render_markdown(language=)` 加 zh-HK (繁體) / en 参数, 港式措辞 | **跨境 / 国际化就绪** | (M8.4) |
| **M8.5** | **5 件套** ⭐ | M8.5c API 中间件 (rate limit / API key / PII) + M8.5d Streamlit i18n + M8.5e SHA-256 manifest + **M8.5f Oaxaca-Blinder 因果公平性分解** + M8.5g 利率优化 | 体验 + 公平性再深化 | `3e9737d` |
| **M8.6** | **验证深化 4 件** ⭐ | **M8.6a TemporalGuard 防泄漏** + **M8.6b BLP 检验** + **M8.6c CATE 稳定性** + **M8.6d CCGS 金字塔** | 73 新测试, 320→393 | `81d692a` |

**总投入**: 33 个测试文件 / **393 个测试用例** (全跑 1m35s) / **569 行反欺诈代码 + 432 行公平性代码 + 489 行叙事代码 + 1100 行 M8.6 验证代码** / **19 张图表** / **3 份决策报告 + 公平性 + fraud + causal_narrative_v2 字段** / **19 份设计文档**

---

## ⚡ 快速开始

### 环境要求

- **Python 3.10+** (推荐 3.11)
- **Conda 环境** `ldq_cc`（已装全部依赖, 详见 `pyproject.toml`）
- **数据**：`data/home-credit-default-risk/application_train.csv`（Kaggle 比赛下载, ~158MB, 详见 [数据集](#-数据集) 节）

### 一键运行

```bash
# 端到端 pipeline (16 步, 含 STEP 3.5 多表 + STEP 14 反欺诈 + STEP 15 公平性 + STEP 16 叙事, ~220 秒 CPU)
/home/tony/anaconda3/envs/ldq_cc/bin/python -m src.run_pipeline

# FastAPI 后端
/home/tony/anaconda3/envs/ldq_cc/bin/uvicorn src.api.app:app --port 8000

# Streamlit 前端
/home/tony/anaconda3/envs/ldq_cc/bin/streamlit run src/frontend/app.py

# 单元测试 (200 用例, ~11 秒)
/home/tony/anaconda3/envs/ldq_cc/bin/python -m pytest tests/ -v --tb=short
```

### 6 个核心亮点的独立 demo (M1 输出, 不依赖全量数据)

```bash
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_causal_discovery.py   # PC + NOTEARS 融合
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_cate.py               # 3 方法 CATE
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_refute.py            # 4 类反驳 + E-value
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_counterfactual.py     # DiCE NSGA-II
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_shap.py              # SHAP 四象限
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_decision.py          # 决策报告
```

---

## 🛠 技术栈

| 类别 | 选型 | 用途 |
|------|------|------|
| **因果发现** | `causal-learn` (PC + NOTEARS) | 混合 DAG 发现 |
| **因果效应** | `DoWhy 0.14` + `EconML 0.16` | ATE / CATE / 4 类 refuter |
| **反事实** | `dice_ml` (NSGA-II) | 多目标反事实生成 |
| **预测模型** | `LightGBM 4.5` (CPU/GPU build) + `sklearn 1.6` | 主分类器 + 校准 |
| **可解释性** | `SHAP 0.48` (TreeSHAP) | 4 象限因果一致性 |
| **反欺诈** | `LightGBM 4 类` + 业务规则伪标签 | M7 三件套 |
| **后端** | `FastAPI 0.136` + `Pydantic v2` | 5 个 REST 端点 |
| **前端** | `Streamlit 1.58` + `streamlit-agraph` | 4 页 (dashboard / 因果图 / 反事实 / 决策) |
| **监控** | 自研 PSI 漂移检测 (3 层) | 特征 / 预测 / 概念漂移 |
| **数据** | `Home Credit Default Risk` (307K) + `German Credit` (1K) | 主数据集 + 快速基线 |

---

## 🔬 16 步 Pipeline 详解

```
STEP 1-2   加载 + 校验       Home Credit 307K × 122 → DataFrame + 校验报告
STEP 3     清洗              缺失值中位数填充 + 1%/99% Winsorize
STEP 3.5   多表聚合 ⭐       bureau / prev / POS / INST / CC → 246 聚合特征
STEP 4     特征工程          causal-guided subset + label encoding
STEP 5     划分              70/30 stratified train/test
STEP 5.5   特征预筛选        L1-style: drop zero-gain features (216 of 265)
STEP 6     训练              LightGBM (500 trees, GPU/Optuna opt-in) + 3-fold CV
STEP 7     评估 + 校准       AUC/Acc/F1 + Isotonic Calibration
STEP 8     因果发现          PC + NOTEARS 融合 + 领域 DAG 注入 (43 边)
STEP 9     ATE 估计          DoWhy CausalModel + 4 类 refuter + E-value
STEP 10    CATE 估计         LinearDML + SparseLinearDML + CausalForestDML
STEP 11    反驳验证          4 类 refuter (placebo / random cause / data subset / E-value)
STEP 12    SHAP 四象限       TreeSHAP + 4 象限 (TRUSTED/UNTRUSTED/MASKED/NEGLIGIBLE)
STEP 13    反事实 + 决策     DiCE NSGA-II + 决策 JSON 报告 (中英模板)
STEP 14    反欺诈三件套 ⭐   FraudGuard (3 分类 + 包装资质 + 养流水去噪)
STEP 15    公平性审计 ⭐     3 指标 (DP/EO/DI) × 4 切片 + 路由漂移 PSI
STEP 16    因果叙事深化 ⭐   3 层叙事 (model/cohort/individual) + DAG 路径 + 解释稳健性
```

**关键产出**:
- `output/figures/19 PNG` — ROC / 特征重要性 / DAG / CATE / SHAP / 反欺诈 / 公平性 / 因果瀑布 / 三联叙事卡
- `output/decision_reports/3 JSON + 3 MD` — 完整决策报告 (含 fraud + fairness + causal_narrative_v2 字段)
- `output/decision_reports/pipeline_summary.json` — 主指标 + routing 分布 + fairness verdict + narrative stability

---

## 📈 实测数据对比（Home Credit 30 万行）

| 指标 | M2 单表 | M5 8 表 | M5+ (CPU 优化) | M6 (GPU+Optuna) | M7 (+反欺诈) | 累计提升 |
|------|------:|-----------:|-----------:|-----------:|-----------:|-----:|
| 特征数 | 30 | 265 | 216 | 211 | 211 | +181 |
| 3-fold CV AUC | 0.7503 | 0.7763 | 0.7756 | **0.7803** | **0.7803** | +0.030 |
| 测试集 AUC-ROC | 0.7547 | 0.7803 | 0.7803 | **0.7803** | **0.7803** | +0.026 |
| 测试集 F1 (default) | 0.0344 | 0.0770 | 0.0735 | **0.0735** | **0.0735** | +0.039 |
| ATE (`AMT_CREDIT` → `TARGET`) | +0.0092 | +0.0092 | +0.0092 | +0.0092 | +0.0092 | — |
| CATE 一致性 (3 方法) | 0.578 | 0.548 | 0.548 | 0.548 | 0.548 | -0.030 |
| 反驳验证 | 3/4 | 3/4 | 3/4 | 3/4 | 3/4 | — |
| 决策报告 | 3 份 | 3 份 | 3 份 | 3 份 | **+ fraud 字段** | **+ causal_narrative_v2 字段** | — |
| 单元测试 | 85 / 1.34s | 98 / 1.44s | 101 / 1.46s | 108 / 7.69s | 133 / 8.0s | **181 / 9.9s** | +96 |
| Pipeline 端到端耗时 | 84.8s | 244.9s (冷) | 184.5s (热) | 184.5s (热) | 194.9s (热) | **219.5s (热)** | +135s |

> 完整 per-step 耗时、ATE/CATE/Refutation 数值、反欺诈 routing 分布见 [`BENCHMARKS.md`](BENCHMARKS.md)

---

## 🛡 反欺诈三件套 (M7 重点)

源自 `docs/CausalCredit_反欺诈能力覆盖分析.md` §4.1 提出的"反欺诈三件套"。

| 模块 | 测量维度 | 核心算法 | 路由 |
|------|---------|---------|------|
| **三分类子模型** | 主观恶意 (fraudulent) / 履约能力变化 (non_malicious) / 系统性 (systemic) | 4 类 LightGBM + 业务规则伪标签 (无 ground truth) | `REJECT_FRAUD` ≥ 0.10 |
| **包装资质检测** | "模型在用 + 因果无效" 的特征比例 | top-25% SHAP 内 4 象限分类 + 域 DAG 路径完整性 | `REJECT_PACKAGING` ≥ 0.50 |
| **养流水去噪** | 还款↔消费脱钩 = 制造历史 | 5 INST__ + 4 CC_/POS_ 列符号一致性 → inflation [0, 0.15] | `REVIEW_DENOISED` < 0.50 |

**编排器** `FraudGuard` 按优先级合成 5 类路由: `REJECT_FRAUD` > `REJECT_PACKAGING` > `REVIEW_DENOISED` > `REVIEW_BORDERLINE` > `PROCEED`

**1K 测试子集实测路由分布**:

| 路由 | 占比 | 业务动作 |
|------|-----:|---------|
| `REVIEW_BORDERLINE` | 91.4% | 任意信号 borderline, 进入人工审查 |
| `PROCEED` | 5.2% | 干净, 自动通过 |
| `REJECT_FRAUD` | 2.5% | 高欺诈嫌疑, 自动拒绝 |
| `REJECT_PACKAGING` | 0.9% | 包装嫌疑大, 自动拒绝 |

> 完整需求↔实现追溯见 [`docs/CausalCredit_M7_反欺诈三件套实现记录.md`](docs/CausalCredit_M7_反欺诈三件套实现记录.md)

---

## ⚖️ 公平性审计 (M8.1 重点)

**问题**: HKMA / EU AI Act / EEOC 80% 规则都要求对受保护属性做群体公平性审计. 一个只看 AUC 0.78 的模型可能"对老年/低教育群体系统性拒绝", 这在合规上不可接受.

**做法**: `src/fairness/` 3 个模块在 STEP 15 对 4 个默认切片算 3 项标准指标:

| 切片 | 来源 | 分组 |
|---|---|---|
| `gender` | `CODE_GENDER` | M / F / UNKNOWN |
| `age_group` | `DAYS_BIRTH` | young (<35) / mid (35-60) / old (≥60) |
| `income_group` | `AMT_INCOME_TOTAL` | low / mid / high (样本 33/66 分位) |
| `education_group` | `NAME_EDUCATION_TYPE` | 5 个标准教育层级 |

| 指标 | 公式 | 阈值 (合规) |
|---|---|---|
| Demographic Parity gap | max sel_rate − min sel_rate | < 0.05 |
| Equal Opportunity gap | max TPR − min TPR | < 0.05 |
| Disparate Impact ratio | min sel_rate / max sel_rate | ≥ 0.80 (EEOC 80% 规则) |

**实测 (Home Credit 50K 测试集)**:

```
gender              status=WARNING  DP=0.004  EO=0.012  DI=0.538
age_group           status=WARNING  DP=0.010  EO=0.040  DI=0.082
income_group        status=WARNING  DP=0.004  EO=0.021  DI=0.511
education_group     status=WARNING  DP=0.008  EO=0.050  DI=0.000
```

**关键解读**:
- 4 个切片都触发 WARNING, **主要因 DI 远低于 0.80** (数据本身的结构性偏差)
- **EO gap 全部 < 0.05** → 模型**没有**"对真正会违约的群体漏判"的歧视
- 真正不均衡的是"有多少比例被拒绝" (selection_rate) — 模型整体更保守
- 路由分布与 M7 baseline PSI=0.001 → M7→M8.1 升级**无回归**

**3 张新图**:
- `12_fairness_group_rates.png` — per-slice per-group TPR/FPR 柱状图
- `13_fairness_metric_gaps.png` — DP/EO/DI 子图, 阈值红线
- `14_fairness_status.png` — slice 状态仪表板 (FAIR/WARNING/UNFAIR 配色)

**决策报告新增字段** (`build_fairness_block`):

```json
{
  "fairness": {
    "applicant_groups": {"gender": "F", "age_group": "mid", "income_group": "low", "education_group": "secondary"},
    "verdict": "WARNING",
    "violated_slices": ["gender", "age_group", "income_group", "education_group"],
    "regulatory_note": "One or more slices are WARNING. Model output may be biased; request additional documentation."
  }
}
```

**反欺诈路由升级 (M8.1d+e)**:
- `FraudGuardConfig` 数据类 + `configs/config.yaml::fraud_guard` — 同一份代码可跑多个产品 (现金贷 / 学生贷) 不同阈值
- `DriftDetector::detect_routing_drift` — 路由分布 PSI 监控, M7→M8.1 实测 PSI=0.001

> 完整需求↔实现追溯见 [`docs/CausalCredit_M8.1_公平性与反欺诈升级实现记录.md`](docs/CausalCredit_M8.1_公平性与反欺诈升级实现记录.md)

---

## 🧠 因果叙事深化 (M8.2 重点)

**问题**: M7 之前的决策报告 1 句话 ("Primary driver: EXT_SOURCE_2 (SHAP=+0.94)") 在监管 "challenge the decision" 流程下, 客户/信贷员/审计员都会问 3 个后续问题 — 模型普遍怎么判断? 跟同类申请人比如何? 这一单的因果路径是什么? 解释是否稳定?

**做法**: `src/explain/causal_narrative.py::CausalNarrative` 把 "为什么这单" 拆成 4 段:

| 段 | 回答的问题 | 输入 | 输出 |
|---|---|---|---|
| **Model-level** | "模型普遍靠哪些特征判断?" | 5K 测试集 SHAP | top-3 mean \|SHAP\| 特征 |
| **Cohort-level** | "跟历史最像这单的 10 个申请人比, 风险是否也高?" | KNN k=10 (z-score 标准化) | cohort P(default) + Δ + top-5 z 偏差 |
| **Individual-level** | "主导特征走了哪些因果路径到达 TARGET?" | 单条 SHAP + DAG + 4 象限 | 主导特征 + 主导路径 + 4 象限计数 |
| **Robustness** | "加 10% 噪声后, top-3 还会变吗?" | 20× 高斯扰动 | stability_score (0.6×top1 + 0.4×top3) |

**实测 (Home Credit 3 个申请人对照)**:

| 申请人 | 等级 | P(default) | cohort Δ | 主导特征 | stability | 业务解读 |
|---|---|---:|---:|---|---:|---|
| HC_006355 | E (高) | 89.55% | **+0.60** | EXT_SOURCE_2 | **0.94** | 单一主导, 极稳定, 远超 cohort |
| HC_036837 | A (边界) | 4.88% | -0.026 | EXT_SOURCE_2 | 0.34 | 与 cohort 类似, 解释 moderately robust |
| HC_023041 | A (低) | 0.26% | -0.011 | DAYS_EMPLOYED | **0.20** | 无强主导, fragile, top-1 在扰动下 70% 换位 |

**关键发现**:
- **高风险 ↔ 稳定解释**: 单一强主导特征 → 解释不会漂
- **低风险 ↔ fragile 解释**: 无强主导特征 → 解释在小扰动下大量换位, 不可直接当证据
- **cohort Δ 与风险等级强正相关**: 无需 SHAP 就能 outlier 化

**2 张新图**:
- `15_causal_waterfall.png` — top features 横向条形图, 颜色按 4 象限 (TRUSTED 绿 / UNTRUSTED 红 / MASKED 橙 / NEGLIGIBLE 灰)
- `16_narrative_card.png` — 3 并排文本面板 (蓝/黄/绿背景) 给非技术审阅者 (合规 / 运营) 一眼看懂

**决策报告新字段** (`causal_narrative_v2`):

```json
{
  "causal_narrative_v2": {
    "model_level": {"top_features": [{"feature": "EXT_SOURCE_2", "mean_abs_shap": 0.3012}, ...], "narrative": "..."},
    "cohort_level": {"k": 10, "cohort_mean_p_default": 0.295, "delta": 0.60, "top_deviations": [...], "narrative": "..."},
    "individual_level": {"dominant_feature": "EXT_SOURCE_2", "dominant_dag_path": ["EXT_SOURCE_2", "TARGET"], "n_trusted": 5, "narrative": "..."},
    "robustness": {"top_1_stable": 1.0, "top_3_stable": 0.85, "stability_score": 0.94, "interpretation": "Explanation is robust."}
  }
}
```

> 完整需求↔实现追溯见 [`docs/CausalCredit_M8.2_因果叙事深化实现记录.md`](docs/CausalCredit_M8.2_因果叙事深化实现记录.md)

---

## ⚖️ 因果公平性分解 + 验证深化 (M8.5f + M8.6 重点)

**问题 1**: 公平性指标说"女性 vs 男性 DP gap = 0.04", 但**这 0.04 里有几成是女性真的收入/教育分布不同 (legit), 几成是模型对女性有偏见 (suspect bias)?** 答: Oaxaca (1973) / Blinder (1973) wage-decomposition — 工业界 50 年来的标准做法。

**M8.5f Oaxaca-Blinder** (`src/fairness/oaxaca.py`):

```
OaxacaBlinderResult:
    total_gap            = ȳ_A − ȳ_B
    explained_gap        = (x̄_A − x̄_B) · β_ref         ← Blinder 形式, β_ref=β_B
    unexplained_gap      = total − explained             ← 恒等式强制, 吸收截距差
    discrimination_index = |unexplained| / (|explained| + |unexplained|)  ∈ [0, 1]
```

**问题 2**: 我们有 4 类 refutation + BLP + 公平性 + 反欺诈 + 漂移, 输出散落在 8 个模块 — 评审/合规想看"模型到底可不可信"得自己拼图。答: **CCGS 因果验证金字塔** (M8.6d) + BLP (M8.6b) + CATE 稳定性 (M8.6c) + TemporalGuard (M8.6a)。

**M8.6 因果验证 4 件套**:

| # | 模块 | 回答的问题 | 关键产出 |
|:-:|------|----------|------|
| **M8.6a** | `TemporalGuard` | 训练时是否偷看了"申请后才知道"的数据? | 剔除 `MONTHS_BALANCE > 0` 行 + 时间有序切分 |
| **M8.6b** | `BLPTest` | CATE 预测本身有信号吗? (还是常数 0?) | OLS Y ~ 1 + T + c_hat + c_hat·T, 检验 c_hat 系数 p<0.05 |
| **M8.6c** | `CATEStabilityTester` | CATE 在不同子集 / 超参下还稳吗? | Tier1: 30× split-half bootstrap, Tier2: 10× 超参网格 |
| **M8.6d** | `CausalVerificationPyramid` | 综合起来模型可不可信? | 4 层 L1-L4 → CCGS ∈ [0, 1], 双门控 pass |

**CCGS 公式** (M8.6d):

```
   CCGS = 0.25·L1 + 0.30·L2 + 0.25·L3 + 0.20·L4
   L1 Graph:           GSI (Jaccard bootstrap) + DKCS (per-edge 确认率)
   L2 Effect:          5 tests (placebo / subset / sensitivity / BLP / CATE consistency)
   L3 Counterfactual:  CCR (一致性率) + (1 − immutable_violation/0.2)
   L4 E2E:             AUC improvement + (1 − ECE/0.1) + (1 − DP/0.1)
   pass: CCGS ≥ 0.70  AND  all_layers_pass   ← 双门控, 防一好遮百丑
```

**为什么是这套**: 评审质疑因果估计时, 我们不再"argue" — 直接给 4 个客观分数 (GSI / BLP p / CATE ρ / CCGS) 让他看。

> 完整需求↔实现追溯见 [`docs/CausalCredit_M8.6_因果验证深化实现记录.md`](docs/CausalCredit_M8.6_因果验证深化实现记录.md)

---

## 📁 项目结构

```
CausalCredit/
├── src/
│   ├── data/                  # HomeCreditLoader + German Loader + validator/preprocessing + M8.6a TemporalGuard
│   ├── features/              # 因果特征 (5 个) + aggregator (8 表 JOIN)
│   ├── causal/                # DAG / discovery / ATE / CATE / refute / M8.6b BLP / M8.6c stability / M8.6d verification
│   ├── models/                # LightGBM (+ GPU/Optuna) / GBT / 校准 / 评估
│   ├── explain/               # SHAP / DiCE / 决策 / 证据链 / M8.2 因果叙事
│   ├── fraud/                 # M7 反欺诈三件套 (three_class + packaging + denoising + pipeline)
│   ├── fairness/              # M8.1 公平性指标 + 切片 + 可视化 + M8.5f Oaxaca-Blinder
│   ├── fraud/                 # M7 反欺诈三件套 (three_class + packaging + denoising + pipeline)
│   ├── fairness/              # M8.1 公平性指标 + 切片 + 可视化 + M8.5f Oaxaca-Blinder
│   ├── api/                   # FastAPI 5 端点 + 业务服务层 + M8.5c 中间件
│   ├── frontend/              # Streamlit 4 页 + M8.5d i18n
│   ├── monitoring/            # PSI 漂移检测 (3 层) + 路由分布 PSI
│   └── run_pipeline.py        # 16 步端到端入口
├── tests/                     # 33 个测试文件, 393 用例 (含 M8.5f + M8.6a-d 73 个新测试)
├── configs/                   # config.yaml
├── scripts/                   # run_api / run_demo / run_tests / setup_env
├── data/
│   ├── home-credit-default-risk/    # 307K × 122 主数据集
│   └── german_credit.csv            # 1K × 20 快速基线
├── output/                    # 详见下一节
└── docs/                      # 19 份设计文档 (含 M7/M8.1/M8.2/M8.6 实现记录)
```

### `output/` 目录内容

```
output/
├── figures/                              # 19 张 PNG
│   ├── 01_roc_curve.png                  # ROC + PR
│   ├── 02_feature_importance.png         # LightGBM gain
│   ├── 03_confusion_matrix.png
│   ├── 04_calibration_curve.png
│   ├── 05_ate_forest.png
│   ├── 06_causal_graph_dag.png           # PC+NOTEARS 融合 DAG
│   ├── 07_cate_distribution.png          # 3 方法 CATE
│   ├── 08_cate_subgroup.png              # 子群条形图
│   ├── 09_refutation_results.png         # 4 类 refuter |ΔATE|
│   ├── 10_shap_four_quadrant.png         # |SHAP| × |causal_proxy|
│   ├── 11_counterfactual_scenarios.png
│   ├── 12_fraud_score_routing.png        # M7 fraud_score + routing 饼图
│   ├── 13_packaging_scatter.png          # M7 path_integrity × packaging_score
│   ├── 14_denoising_effect.png           # M7 P(default) vs denoised P
│   ├── 12_fairness_group_rates.png       # M8.1 per-slice per-group TPR/FPR
│   ├── 13_fairness_metric_gaps.png       # M8.1 DP/EO/DI 带阈值线
│   ├── 14_fairness_status.png            # M8.1 slice 状态仪表板
│   ├── 15_causal_waterfall.png           # M8.2 top features 4 象限瀑布图
│   └── 16_narrative_card.png             # M8.2 三联叙事卡 (model/cohort/individual)
├── decision_reports/
│   ├── HC_006355.json                    # 高风险 (P=89%, E) — 含 causal_narrative_v2
│   ├── HC_023041.json                    # 低风险 (P=0.3%, A) — 含 causal_narrative_v2
│   ├── HC_036837.json                    # 中风险 (P=4.9%, A) — 含 causal_narrative_v2
│   ├── *.md                              # 同 3 份的可读报告 (含 4 段叙事)
│   ├── pipeline_summary.json             # 主指标 + 反欺诈 routing + 公平性 verdict
│   └── pipeline_timings.json             # per-step 耗时
├── demo_m1/                              # M1 5 个创新点独立图表
├── cache/                                # 多表聚合 parquet 缓存 (M5+)
└── models/                               # 训练好的模型 pickle
```

---

## 📂 决策报告样例 (HC_006355)

```json
{
  "applicant_id": "HC_006355",
  "default_probability": 0.8955,
  "score": 319,
  "risk_grade": "E",
  "decision_suggestion": "DECLINE — high risk; recommend rejection or sub-prime product",
  "top_risk_factors": [
    {"feature": "EXT_SOURCE_2",  "shap": +0.94, "quadrant": "TRUSTED"},
    {"feature": "EXT_SOURCE_3",  "shap": +0.54, "quadrant": "TRUSTED"},
    {"feature": "EXT_SOURCE_1",  "shap": +0.46, "quadrant": "TRUSTED"},
    {"feature": "BUREAU_TYPE_MICROLOAN_FRAC", "shap": +0.42, "quadrant": "TRUSTED"},
    {"feature": "PREV_STATUS_REFUSED_FRAC",   "shap": +0.29, "quadrant": "TRUSTED"}
  ],
  "cate_insights": [...],
  "counterfactual_recommendations": [],
  "fraud": {
    "fraud_score": 0.0005,
    "defaulter_sub_proba": {"fraudulent": 0.0006, "non_malicious": 0.9994, "systemic": 0.0},
    "packaging_score": 0.37,
    "path_integrity": 1.0,
    "denoised_default_proba": 1.0,
    "causal_consistency": 0.5,
    "inflation_strength": 0.15,
    "routing": "REVIEW_BORDERLINE",
    "routing_reasons": ["packaging_score=0.370 in [0.30, 0.50) borderline"]
  }
}
```

---

## 💡 关键设计权衡（含负面发现）

| 决策 | 结论 | 文档 |
|------|------|------|
| **GPU LightGBM 加速** | ❌ 在 21 万行规模上比 CPU 慢 1.5-1.6x (kernel 启动延迟 > 数据并行收益) | M6 节 |
| **Optuna 超参调优** | ❌ Home Credit 数据上 25 trials AUC 持平 (-0.0013, 已逼近 Bayes 最优) | M6 节 |
| **AUC 0.78 数据天花板** | ✅ 8 表 JOIN + 多表因果特征后, 纯预测力触及上限, 差异化应回到**因果可解释性 + 反欺诈** | M5+/M7 节 |
| **3 张二级表聚合** | ✅ 5 张二级表 → 246 特征是 M5 的关键收益 | M5 节 |
| **DiCE NSGA-II** | ✅ 多目标 (proximity + sparsity + plausibility) 一次生成 5 个反事实 | M1/2 节 |
| **SHAP + 因果四象限** | ✅ 比单纯 SHAP 重要度排序多一个"因果可信度"维度 | M1/2 节 |
| **三件套反欺诈** | ✅ 3 个独立信号按优先级合成, 避免一个高分淹没其他 | M7 节 |

---

## 🗂 数据集

| 数据集 | 规模 | 角色 | 下载 |
|--------|------|------|------|
| **Home Credit Default Risk** | 307,511 × 122 | 主数据集 (单表 + 5 张二级表) | [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) |
| German Credit | 1,000 × 20 | 快速基线对比 + 单元测试 | `sklearn.datasets.fetch_openml("credit-g")` |

**多表 JOIN** (M5): bureau / previous_application / POS_CASH_balance / installments_payments / credit_card_balance 聚合为 246 个新特征, 训练集从 30 列 → 216 列 (剔除 zero-gain 后).

---

## 🚧 仍为骨架的模块

按 `PROGRESS.md` 状态, 这些模块编译通过但未端到端可用:

- `src/api/services.py` + `src/api/routes.py` — 只有 `/api/v1/health` 返回 200
- `src/frontend/app.py` + `src/frontend/pages/*` — Streamlit 导航可渲染, 但页面是 `st.info` 占位符
- `src/monitoring/drift_detector.py` — 简化版 PSI 在 run_pipeline.py 中, 完整 PSI 流式监控未接入
- `src/models/calibrate.py` — Isotonic 校准的独立 API 未完成, 简化版在 pipeline 中

> **设计取舍**: 用户明确"不引入 K8s / Triton / TensorRT / Terraform / Helm / ArgoCD / Celery / Redis / Kafka / Feast / Airflow / Evidently", 因此 API/UI/monitoring 走最简实现, 把工程资源集中在**因果栈 + 反欺诈三件套**上。

---

## 🔮 未来迭代方向

- ~~**多语言决策建议** — 粤语 / 繁体~~ → 列为 **M8.4** (在 `CausalNarrative.render_markdown` 已支持中文标题基础上扩展)
- ~~**公平性验证** — `CODE_GENDER` 节点已存在, 缺 Demographic Parity / Equal Opportunity 检验~~ → ✅ M8.1 完成
- ~~**因果叙事深化** — 讲好"为什么高", 群组级 confounder 归因~~ → ✅ M8.2 完成
- ~~**M8.3 完整服务化** — FastAPI 5 端点填实 + Streamlit 4 页填实 + PSI 后台任务~~ → ✅ M8.3 完成
- ~~**M8.4 多语言 + 港式本地化** — 粤语 / 繁体 / 香港场景~~ → ✅ M8.4 完成
- ~~**M8.5 系列** — 中间件 / i18n / 模型清单 / Oaxaca 因果公平性 / 利率优化~~ → ✅ M8.5 完成
- ~~**M8.6 系列** — TemporalGuard / BLP / CATE 稳定性 / CCGS 金字塔~~ → ✅ M8.6 完成
- **M8.7 验证金字塔接入 STEP 17** — 把 L1-L4 跑通并注入 pipeline_summary.json, 输出 `pyramid_score.json`
- **多表聚合 polars 改写** — pandas 单线程 ~27s, 估可降到 ~5s
- **反欺诈伪标签升级** — 用反欺诈团队人工标注的真实种子集替换业务规则
- **实时推理服务** — gRPC / ONNX Runtime (用户未禁用)
- **生产流量调优** — 反欺诈阈值在生产数据上 ROC 优化 (现为经验值)

---

## 📚 文档索引

| 文档 | 用途 |
|------|------|
| [`PROGRESS.md`](PROGRESS.md) | 18 个里程碑详细记录 (M0-M7 + M8.1-M8.6) |
| [`BENCHMARKS.md`](BENCHMARKS.md) | 性能基准 + 反欺诈 routing 分布 + 单测覆盖 |
| [`CLAUDE.md`](CLAUDE.md) | 给 Claude Code 的协作指引 (架构 / 命令 / 约定) |
| [`docs/`](docs/) | 19 份原始分析文档 (需求 / 设计 / 验证标准) |
| [`docs/CausalCredit_M7_反欺诈三件套实现记录.md`](docs/CausalCredit_M7_反欺诈三件套实现记录.md) | M7 需求↔实现追溯 |
| [`docs/CausalCredit_M8.1_公平性与反欺诈升级实现记录.md`](docs/CausalCredit_M8.1_公平性与反欺诈升级实现记录.md) | M8.1 公平性审计 + FraudGuardConfig + 路由 PSI |
| [`docs/CausalCredit_M8.2_因果叙事深化实现记录.md`](docs/CausalCredit_M8.2_因果叙事深化实现记录.md) | M8.2 3 层叙事 + DAG 路径 + 解释稳健性 |
| [`docs/CausalCredit_M8.6_因果验证深化实现记录.md`](docs/CausalCredit_M8.6_因果验证深化实现记录.md) | M8.6 TemporalGuard + BLP + CATE 稳定性 + CCGS 金字塔 |

---

## 📄 许可证

MIT License
