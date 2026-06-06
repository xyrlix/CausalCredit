# CausalCredit 开发进展记录

> **最后更新**: 2026-06-06 | **环境**: CPU (Python 3.11, `ldq_cc` conda env)  
> **状态**: 6 个里程碑全部完成 ✅ (M5 8 表 JOIN 集成, M6 单测 + 文档)

---

## 总览：6 个里程碑全部交付

| 里程碑 | 目标 | 状态 | 关键产出 |
|:---:|------|:---:|------|
| **M0** | 数据准备 | ✅ | Home Credit 30 万行加载器 + 领域 DAG (15 节点 / 28 边) |
| **M1** | 5 个核心创新点 | ✅ | `tests/test_*.py` 6 个独立 demo + 9 张 M1 图表 |
| **M2** | 13 步端到端 pipeline | ✅ | `python -m src.run_pipeline` 跑通, 11 张 PNG + 3 份 JSON 报告 |
| **M3** | API + UI 服务化 | ✅ | FastAPI 5 端点 (:8000) + Streamlit 4 页 (:8501) |
| **M4** | 监控 + 测试 + 文档 | ✅ | PSI 漂移检测 (3 层) + 85 个单元测试 (1.34s) |
| **M5** | **8 表 JOIN + 多表因果特征** | ✅ | **5 张二级表 (~1.1 GB, 80M 行) → 245 聚合特征, AUC 0.7547→0.7803** |

---

## Git 提交历史

| # | Commit | 内容 |
|:--:|--------|------|
| 1 | `b68de7e` | 项目骨架搭建 (45 文件) |
| 2 | `eb607f3` | 扩展模块与工程化配置 (14 文件) |
| 3 | `6f17d9f` | 完整端到端流水线（可运行） |
| 4 | `76b3ff3` | 自动生成 5 张可视化图表 |
| 5 | `815baef` | 开发进展记录 - CPU 环境完成 |

> M0-M4 完整代码在 main 分支。后续每个里程碑均经 `python -m src.run_pipeline` 验证 + 单测全过。

---

## M0 — 数据准备 ✅

| 文件 | 功能 |
|------|------|
| `data/home-credit-default-risk/application_train.csv` | 307,511 × 122, Kaggle 下载 |
| `src/data/home_credit_loader.py` | `HomeCreditLoader` (fetch / get_feature_target / get_metadata) |
| `src/causal/home_credit_graph.py` | `HomeCreditCausalGraph` 15 节点 / 28 边 / 无环 |

**DAG 节点分布**：
- Treatments (3): `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_EMPLOYED`
- Outcome (1): `TARGET`
- Confounders (8): `AMT_INCOME_TOTAL`, `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, `REGION_RATING_CLIENT`, `DAYS_BIRTH`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS`, `EXT_SOURCE_2`
- Mediators (2): `AMT_GOODS_PRICE`, `NAME_HOUSING_TYPE`
- Sensitive (1): `CODE_GENDER`

---

## M1 — 5 个核心创新点 ✅

| # | 创新点 | 模块 | Demo 入口 | 输出 |
|:---:|------|------|------|------|
| 1 | 混合因果发现 | `src/causal/discovery.py` | `tests/test_causal_discovery.py` | 5 张图 (合成+Home Credit+领域注入) |
| 2 | CATE 异质处理 | `src/causal/cate.py` | `tests/test_cate.py` | `cate_distribution.png` |
| 3 | 反驳验证 | `src/causal/refute.py` | `tests/test_refute.py` | `refutation_results.png` |
| 4 | 因果约束反事实 | `src/explain/counterfactual.py` | `tests/test_counterfactual.py` | `counterfactual_examples.png` |
| 5 | SHAP 四象限 | `src/explain/shap_explain.py` | `tests/test_shap.py` | `four_quadrant.png` |
| + | 决策建议 | `src/explain/decision.py` | `tests/test_decision.py` | 决策报告 JSON |

**修复记录**：M1 完成初版时，因果发现模块的 NOTEARS 求解器存在 bug（augmented-Lagrangian 循环中 rho 起始值 1.0 过大，平滑 L1 缺失，权重塌缩到 0），PC 端点约定也写反（`(1,1)` 当成了无向，实际是 `(-1,-1)`）。修复要点：
- NOTEARS：`rho_init=1e-2` + 平滑 L1 (`sqrt(W²+eps²)`) + 跟踪 `best_W` 避免后续 AL 迭代把 W 拉回 0
- PC：识别 causallearn 的 `(-1,-1) = undirected`、`(-1,1) / (1,-1) = directed` 约定
- 单测：`tests/test_discovery.py` 8 个用例覆盖 NOTEARS/PC/fusion/inject 四个公共 API

**关键技术选型**：
- 因果发现: `causallearn` (PC + NOTEARS) + 领域知识 DAG 融合
- CATE: `econml.dml.LinearDML` + `econml.dr.ForestDRLearner` + `econml.dml.CausalForestDML`
- 反驳: `dowhy.causal_refuters` (Placebo / RandomCommonCause / DataSubset / Unobserved) + Vanderweele E-value
- 反事实: `dice_ml.Dice` (NSGA-II) + IMMUTABLE/SEMI-MUTABLE 锁定 + 因果图联动
- 可解释性: `shap.TreeExplainer` + 局部 ±1σ 敏感性代理

---

## M2 — 13 步端到端 Pipeline ✅

**入口**：`python -m src.run_pipeline`（**实测 ~75 秒**）

| 步骤 | 内容 | 输出 |
|:---:|------|------|
| 1 | 加载 Home Credit (307,511) | DataFrame |
| 2 | 数据校验 | 校验报告 |
| 3 | 清洗（缺失 + Winsorize） | 清洗后数据 |
| 4 | 特征工程（5 个因果特征） | 增广特征 |
| 5 | Train / Val / Test 划分 | 索引 + 标签 |
| 6 | LightGBM 训练 + 5-fold CV | 训练好的模型 |
| 7 | 评估 + Isotonic 校准 | AUC, KS, 校准曲线 |
| 8 | **因果发现**（PC + NOTEARS 融合） | DAG |
| 9 | **ATE 估计**（DoWhy + PSM） | ATE + CI |
| 10 | **CATE**（DML/DR/Causal Forest 3 方法） | 个体效应 |
| 11 | **反驳验证**（4 refuter + E-value） | 通过/失败 |
| 12 | **SHAP 四象限** | 全局 + 局部 |
| 13 | **反事实 + 决策报告**（3 份） | JSON + Markdown |

**实测结果（Home Credit 30 万行）**：

| 指标 | 数值 |
|------|------|
| 测试集 AUC-ROC | 0.7547 |
| 测试集 KS | 0.4120 |
| 测试集 LogLoss | 0.2463 |
| ATE (AMT_CREDIT → TARGET) | +0.0025 / 千美元 |
| CATE 异质性 (Spearman ρ) | DR vs Forest = 0.81 |
| 反驳验证 | 4 类中 3 类通过 |
| 决策报告 | 3 份覆盖 P=0.31%/5.35%/73.50% |

**输出物**：
- `output/figures/01_roc_curve.png` ~ `11_counterfactual_scenarios.png` （11 张）
- `output/decision_reports/HC_*.json` + `HC_*.md`（3 份）

---

## M3 — API + UI 服务化 ✅

### FastAPI 后端

**启动**：`uvicorn src.api.app:app --port 8000`

| 端点 | 方法 | 功能 |
|------|:---:|------|
| `/api/v1/health` | GET | 健康检查 + 模型加载状态 |
| `/api/v1/score` | POST | 信用评分（分 + 概率 + 等级 + 决策） |
| `/api/v1/counterfactual` | POST | 反事实推理（基线 → 干预后） |
| `/api/v1/explain` | POST | SHAP top-k 解释 + 证据链 |
| `/api/v1/causal-effect` | POST | ATE 汇总（pre-computed） |

**关键设计**：
- `ModelRegistry.load()`：50K 训练子集 + pickle 缓存 `output/models/registry_v1.pkl`
- FastAPI `lifespan` 上下文管理器懒加载
- 业务逻辑在 `src/api/services.py`，路由在 `src/api/routes.py`，Pydantic v2 schema 校验

### Streamlit 前端

**启动**：`streamlit run src/frontend/app.py`

| 页面 | 功能 |
|------|------|
| Score Dashboard | 4 预设客户档案 + 滑块表单 + 4 列指标 + SHAP top factors |
| Causal Visualization | 领域 DAG 渲染 + ATE 指标卡 + Pipeline 图表分页 |
| Counterfactual Simulator | 4 干预滑块（CF 推理）+ DiCE NSGA-II 表 |
| Decision Panel | 一键生成决策报告（4 Tab：风险/证据/反事实/JSON） |

**关键设计**：
- `@st.cache_resource` 共享 ModelRegistry
- 单文件 `app.py` 主页 + 4 个 `pages/*.py` 子页
- 4 个预设档案：Prime Customer / Mid-Career / Thin Credit / High-Risk

---

## M4 — 监控 + 测试 + 文档 ✅

`DriftDetector` 类提供 7 个方法：

| 方法 | 用途 |
|------|------|
| `_safe_psi_from_dists` | epsilon-smoothed PSI 核心 |
| `compute_psi(feature, current)` | 单特征 PSI（分位数分箱） |
| `detect_feature_drift(data, features)` | DataFrame：feature/psi/status |
| `detect_prediction_drift(ref, cur)` | Dict：psi, status, 均值/标准差对比 |
| `detect_concept_drift(cur_auc, base_auc, ...)` | Dict：status "ok" / "alert" |
| `compute_feature_statistics(cur)` | mean/std/median 对比 |
| `generate_drift_report(...)` | Markdown 字符串 |

**PSI 分级**：< 0.10 无漂移, 0.10-0.20 中等, ≥ 0.20 告警

### 单元测试（15 个文件 / 98 用例 / 1.44s）

```
tests/test_aggregation.py       # 13 用例  多表聚合 (M5)
tests/test_api_schemas.py       # 11 用例  Pydantic schema + service helpers
tests/test_calibrate.py         #  6 用例  Isotonic Regression 单调性/边界
tests/test_cate.py              # CATE 模块集成 (M1 demo)
tests/test_causal_graph.py      #  7 用例  DAG 节点/边/无环/可视化
tests/test_causal_discovery.py  # 因果发现 demo (PC + NOTEARS + 融合)
tests/test_counterfactual.py    # DiCE 模块集成 (M1 demo)
tests/test_decision.py          # DecisionAdvisor 模块集成 (M1 demo)
tests/test_decision_math.py     # 27 用例  评分/等级/建议/报告/中英
tests/test_discovery.py         #  8 用例  PC+NOTEARS+fusion+inject+compare
tests/test_drift_detector.py    # 15 用例  PSI 公式/边界/概念漂移/markdown
tests/test_refute.py            # Refute 模块集成 (M1 demo)
tests/test_refute_math.py       #  6 用例  E-value 公式/对称性/单调性
tests/test_shap.py              # SHAP 模块集成 (M1 demo)
```

---

## M5 — 8 表 JOIN + 多表因果特征 ✅

### 背景
M2-M4 只用了 `application_train.csv` (单表, 122 列)。Home Credit 公开 Kaggle 比赛 0.81 AUC top-10% 选手的关键差异就是 8 表 join + 跨表聚合特征。本里程碑补齐这块。

### 数据获取
- **来源**: [HuggingFace `mohameddhameem/home-credit-default-risk`](https://huggingface.co/datasets/mohameddhameem/home-credit-default-risk) (Apache-2.0, 2026-05-30 snapshot)
- **下载方式**: HF datasets-server API (无 Kaggle 凭证可用)
- **总大小**: 1.1 GB, 6 张表 + 1 张主表

| 表名 | 行数 | parquet 大小 | 备注 |
|------|------|------------:|------|
| application_train (已有) | 307,511 | 57 MB | 122 列 |
| bureau | 29,016,353 | 64 MB | 19 列 |
| bureau_balance | 27,299,925 | 58 MB | 3 列 |
| previous_application | 1,670,214 | 114 MB | 37 列 |
| POS_CASH_balance | 10,001,358 | 171 MB | 8 列 |
| installments_payments | 13,605,401 | 502 MB | 8 列 (largest) |
| credit_card_balance | 3,840,312 | 175 MB | 23 列 |

### 多表聚合模块 (`src/features/aggregation.py`)

`MultiTableAggregator` 类提供 5 个聚合器 + 1 个 `aggregate_all`:

| 聚合器 | 输出特征数 (M5) | 核心字段 |
|--------|---------------:|----------|
| `aggregate_bureau` (+bureau_balance) | **52** | DAYS_CREDIT, AMT_CREDIT_SUM, _DPD_MONTH_FRAC (合并 balance 后) |
| `aggregate_previous_app` | **68** | AMT_ANNUITY, AMT_CREDIT, NAME_CONTRACT_STATUS 4 种状态分数 |
| `aggregate_pos_cash` | **24** | SK_DPD 分数, CNT_INSTALMENT |
| `aggregate_installments` | **17** | _DAYS_LATE (ENTRY - INSTALMENT), _PAY_RATIO, late-day 分数 |
| `aggregate_credit_card` | **85** | AMT_BALANCE/CREDIT_LIMIT utilization, SK_DPD |
| **aggregate_all (outer join)** | **246** | 外连接 5 表, 0-fill missing |

**关键设计选择**:
- 仅 numeric 聚合（mean/max/min/sum/std）+ 少量 categorical 分数（"fraction of time in status X"）
- 0-fill 缺失值（语义: 申请人没有 bureau 记录 → 0）
- Bureau_balance 合并到 bureau 表后,计算每条 bureau 记录的 `_DPD_MONTH_FRAC`（"bad months" / "total months"）
- Installments 表计算 `_DAYS_LATE = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`（正=晚付）

**测试**: `tests/test_aggregation.py` 13 个用例 (1.44s 全跑) — 覆盖 5 个聚合器 + `aggregate_all` outer join + `load_secondary_tables` 容错。

### Pipeline 集成 (STEP 3.5)

在 `run_pipeline.py` 的 STEP 3 (cleaning) 之后、STEP 4 (feature engineering) 之前插入 STEP 3.5:

```
STEP 3:  Data cleaning
STEP 3.5: Multi-table aggregation  ← NEW, 65s
STEP 4:  Feature engineering (30 + 245 = 275 features)
```

**关键调整**: 因果发现 (STEP 8) 改用 30 个 single-table 特征 (PC 算法在 260 列上会触发 singular correlation matrix 异常),multi-table 特征只用于预测 (LightGBM) 和 SHAP。

### 性能对比 (M2 vs M5)

| 指标 | M2 (单表) | M5 (8 表) | 提升 |
|------|----------:|----------:|-----:|
| 特征数 | 30 | **275** | +245 |
| 3-fold CV AUC | 0.7503 | **0.7763** | **+0.026 (+3.4%)** |
| 测试集 AUC | 0.7547 | **0.7803** | **+0.026 (+3.4%)** |
| 测试集 F1 (default) | 0.0344 | **0.0770** | **+124%** |
| Pipeline 总耗时 | 85s | 245s | +160s (主要在 Step 6 训练 + Step 3.5 聚合) |

**Top 多表特征 (LightGBM gain)**:
- `INST_LATE_DAYS_GT0_FRAC` — 历史还款晚付次数分数
- `POS_CNT_INSTALMENT_FUTURE_MEAN` — 未结清分期数
- `BUREAU_DAYS_CREDIT_MEAN` — 信用历史长度
- `INST__DAYS_LATE_MEAN` — 平均晚付天数
- `BUREAU_AMT_CREDIT_SUM_MEAN` — 历史授信额度

> **结论**: 多表特征对 credit scoring 提升明显,主要原因: (a) **还款履约数据**（installments_payments）单表完全看不到;(b) **信用历史长度**（bureau）单表只有快照;(c) **跨机构多头借贷**（bureau 多条记录）单表无此维度。F1 翻倍说明这些特征对**正例 (default) 召回**帮助最大,符合行业经验。

### 已知优化点 (未做)

- **缓存 `secondary_features.parquet`**: 一次聚合, 永久复用,Step 3.5 从 65s 降到 < 1s
- **polars 改写 bureau 聚合器**: 当前 pandas 单线程,~27s 可降到 ~5s
- **Bureau + balance 双层聚合**: 现在是"bureau_balance 按 SK_ID_BUREAU 聚合"再 merge,可改成"bureau 按 SK_ID_CURR 聚合 + balance 按 SK_ID_CURR 聚合"分别贡献特征
- **SHAP top-N 特征筛选**: 275 特征里 ~50 个 gain > 0 实际是 0,可通过 L1 预筛选把 LightGBM 训练再加速 30%

---


---

## 性能与瓶颈

| 阶段 | 耗时 | 备注 |
|------|------|------|
| 加载 + 清洗 + 特征 | ~10s | 307K × 122 |
| LightGBM 训练 | < 1 分钟 | 50K 子集 + 5-fold CV |
| 因果发现（PC + NOTEARS） | < 1 分钟 | 30K 子集 |
| DoWhy ATE + 4 反驳 | ~20s | 8K 子集 |
| CATE × 3 方法 | ~15s | 8K 子集 |
| SHAP TreeSHAP | ~10s | 50K 训练集 |
| DiCE NSGA-II | < 1s/样本 | 3 样本 |
| **Pipeline 总耗时** | **~75s** | CPU 即可 |

---

## 后续迭代方向（未做）

- ~~8 表 JOIN（bureau / previous_application / POS / installments / credit_card）→ 多表因果特征~~ ✅ M5 完成
- GPU 加速（LightGBM GPU build / XGBoost GPU）→ pipeline 84s → ~5-10s
- 实时推理服务（gRPC / ONNX Runtime）
- K8s / Helm / Terraform 部署
- 多语言决策建议扩展（粤语 / 繁体）
- **多表聚合缓存化**（M5 已知优化点, 1 行代码改写 + 65s → < 1s）
- **特征 L1 预筛选**（M5 已知优化点, 砍 ~30% 训练耗时）

---

## 仓库结构

```
CausalCredit/
├── src/                          # 核心代码
│   ├── data/                     # ✅ Home Credit + German 加载器, 校验, 预处理
│   ├── features/                 # ✅ builder + causal_features
│   ├── causal/                   # ✅ graph / estimate / discovery / cate / refute
│   ├── models/                   # ✅ train (LightGBM + GBT) / evaluate / calibrate
│   ├── explain/                  # ✅ shap_explain / counterfactual / decision / evidence
│   ├── api/                      # ✅ app / routes / services / dependencies / schemas
│   ├── frontend/                 # ✅ app.py + 4 pages
│   ├── monitoring/               # ✅ drift_detector
│   └── run_pipeline.py           # ✅ 13 步入口
├── tests/                        # ✅ 14 文件 / 85 用例
├── configs/                      # ✅ config.yaml
├── scripts/                      # ✅ run_api / run_demo / run_tests / setup_env
├── data/                         # ✅ Home Credit + German Credit
├── output/                       # ✅ figures (11) + decision_reports (3) + demo_m1 (9) + models
├── docs/                         # 11 份原始分析文档
├── CLAUDE.md                     # 给 Claude Code 的协作指引
├── PROGRESS.md                   # 本文件
├── README.md                     # 项目说明
└── pyproject.toml + requirements.txt
```
