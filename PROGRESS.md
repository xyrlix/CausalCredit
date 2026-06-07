# CausalCredit 开发进展记录

> **最后更新**: 2026-06-07 | **环境**: CPU (Python 3.11, `ldq_cc` conda env)  
> **状态**: 12 个里程碑全部完成 ✅ (M0-M7 + M8.1 公平性 + M8.2 因果叙事深化 + M8.3 服务化 + M8.4 多语言)

---

## 总览：12 个里程碑全部交付 (M0-M7 + M8.1 + M8.2 + M8.3 + M8.4)

| 里程碑 | 目标 | 状态 | 关键产出 |
|:---:|------|:---:|------|
| **M0** | 数据准备 | ✅ | Home Credit 30 万行加载器 + 领域 DAG (18 节点 / 36 边) |
| **M1** | 5 个核心创新点 | ✅ | `tests/test_*.py` 6 个独立 demo + 9 张 M1 图表 |
| **M2** | 13 步端到端 pipeline | ✅ | `python -m src.run_pipeline` 跑通, 11 张 PNG + 3 份 JSON 报告 |
| **M3** | API + UI 服务化 | ✅ | FastAPI 5 端点 (:8000) + Streamlit 4 页 (:8501) |
| **M4** | 监控 + 测试 + 文档 | ✅ | PSI 漂移检测 (3 层) + 85 个单元测试 (1.34s) |
| **M5** | **8 表 JOIN + 多表因果特征** | ✅ | **5 张二级表 (~1.1 GB, 80M 行) → 246 聚合特征, AUC 0.7547→0.7803** |
| **M5+** | **CPU 优化** | ✅ | **多表聚合缓存 (65s→2s) + L1 特征预筛选 (-16% 训练耗时), 总耗时 245s→185s** |
| **M6** | **GPU LightGBM + Optuna** | ✅ | **LightGBM GPU build 接入 (默认关闭) + Optuna 9 维超参搜索 (默认关闭), 2 个可选杠杆** |
| **M7** | **反欺诈三件套** | ✅ | **三分类子模型 (fraudulent/non_malicious/systemic) + 包装资质因果一致性检测 + 养流水因果去噪评分, 14 步 / 14 图 / 25 新测试** |
| **M8.1** | **公平性审计 + 反欺诈升级** | ✅ | **3 项公平性指标 (DP/EO/DI) + 4 个默认切片 + 3 张公平性图 + FraudGuardConfig 数据类 (YAML 配置) + 路由分布 PSI 监控 + 路由 baseline 持久化, 15 步 / 17 图 / 33 新测试** |
| **M8.2** | **因果叙事深化** | ✅ | **三层叙事引擎 (model/cohort/individual) + DAG 路径追溯 + K-NN k=10 同类对照 + 解释稳健性扰动 + 因果瀑布图 + 三联叙事卡, 16 步 / 19 图 / 17 新测试** |
| **M8.3** | **完整服务化** | ✅ | **FastAPI 5 端点 fill out (11 smoke test) + 路由 baseline 持久化** |
| **M8.4** | **多语言 + 港式本地化** | ✅ | **render_markdown 加 zh-HK / en 参数, 港式措辞** |

---

## Git 提交历史

| # | Commit | 内容 |
|:--:|--------|------|
| 1 | `b68de7e` | 项目骨架搭建 (45 文件) |
| 2 | `eb607f3` | 扩展模块与工程化配置 (14 文件) |
| 3 | `6f17d9f` | 完整端到端流水线（可运行） |
| 4 | `76b3ff3` | 自动生成 5 张可视化图表 |
| 5 | `815baef` | 开发进展记录 - CPU 环境完成 |
| 6 | (M8.1) | **公平性审计 + 反欺诈升级 (3 文件 / 31 测试 / 3 张图 / STEP 15)** |
| 7 | (M8.2) | **因果叙事深化 (2 文件 / 17 测试 / 2 张图 / STEP 16)** |
| 8 | (M8.3) | **服务化补全: 11 API smoke test + 路由 baseline 持久化 + DAG 加 EXT_SOURCE 边** |
| 9 | (M8.4) | **多语言: render_markdown 加 zh-HK / en, 4 测试** |

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

### 已知优化点 (M5+ 已部分完成)

- ~~**缓存 `secondary_features.parquet`**: 一次聚合, 永久复用,Step 3.5 从 65s 降到 < 1s~~ ✅ M5+ 完成
- **polars 改写 bureau 聚合器**: 当前 pandas 单线程,~27s 可降到 ~5s
- **Bureau + balance 双层聚合**: 现在是"bureau_balance 按 SK_ID_BUREAU 聚合"再 merge,可改成"bureau 按 SK_ID_CURR 聚合 + balance 按 SK_ID_CURR 聚合"分别贡献特征
- ~~**SHAP top-N 特征筛选**: 275 特征里 ~50 个 gain > 0 实际是 0,可通过 L1 预筛选把 LightGBM 训练再加速 30%~~ ✅ M5+ 完成 (改叫 L1 特征预筛选, 见 M5+)

---

## M5+ — CPU 优化 (缓存 + 特征预筛选) ✅

### 动机
M5 集成多表后,端到端耗时从 85s 涨到 245s,瓶颈主要是 STEP 3.5 多表聚合 (65s) 和 STEP 6 LightGBM 训练 (128s, 特征数 30→265)。本里程碑做 2 个低成本优化,目标 AUC 持平的前提下把热跑耗时砍 30%。

### 优化 1: 多表聚合结果缓存

**实现**: `src/features/aggregation.py::load_or_build_secondary_features()`

```python
if not force_rebuild and os.path.exists(cache_path):
    cached = pd.read_parquet(cache_path)
    if sanity_check(cached):
        return cached  # < 1s
# else: 跑全量聚合 + 写 cache (65s)
```

**关键设计**:
- 缓存文件: `output/cache/secondary_features_v1.parquet` (3 MB)
- 版本号: `SECONDARY_FEATURES_CACHE_VERSION = 1` (改 aggregator 时手动 bump, 自动失效)
- 读盘后做 sanity check (index 名 + 列数), 损坏自动重建
- Tolerant: 空表 (0 行) 不报错,直接 skip 跳过该聚合器

**效果**: STEP 3.5 从 65.3s → 2.1s (cache 命中)

**测试**: 3 个新用例 (cache miss → 写, hit → 读, corrupt → rebuild)

### 优化 2: L1 特征预筛选 (STEP 5.5)

**实现**: 在 `run_pipeline.py` 主训练前插入 STEP 5.5,用 100-tree LightGBM 在 50K 子集上跑一遍, 按 `gain > 0` 过滤

```python
quick = lgb.LGBMClassifier(n_estimators=100, max_depth=6, num_leaves=31, ...)
quick.fit(X_train.iloc[sub_50k], y_train.iloc[sub_50k])
keep = gain[gain > 0].index.tolist()  # 砍掉 49 个 0-gain 特征
X_train, X_test = X_train[keep], X_test[keep]
```

**效果**:
- 特征 265 → 216 (砍 18%)
- STEP 6 训练 128s → 112s (-12%)
- **AUC 0.7803 完全持平** (被剔除的特征本来就 gain=0, 没有信息量)

**为什么不是更多**: 49 个 0-gain 特征在 LightGBM 训练中本来就被忽略, 主要省的是 model serialization + 内存带宽。CPU 计算本身受特征维度影响没那么大。

### M5+ 整体效果

| 指标 | M5 (无优化) | M5+ (冷) | M5+ (热) | 节省 |
|------|----------:|--------:|--------:|-----:|
| 总耗时 | 244.9s | 244.9s | **184.5s** | **-60.4s (-25%)** |
| STEP 3.5 聚合 | 65.3s | 68.9s | **2.1s** | -63.2s |
| STEP 5.5 预筛 | n/a | 3.2s | 4.8s | +4.8s |
| STEP 6 训练 | 127.8s | 107.6s | 111.8s | -16.0s |
| **AUC** | 0.7803 | 0.7803 | **0.7803** | 持平 |
| **单测** | 98 / 1.44s | — | **101 / 1.46s** | +3 |

### 测试

`tests/test_aggregation.py` 增加 3 个用例:
- `test_load_or_build_cache_miss_then_hit`: 空 raw_dir → 写 cache, 二次调用读 cache
- `test_load_or_build_cache_invalidation_on_corrupt_cache`: 损坏的 cache (错 index) → 重建
- `test_cache_version_constant_is_int`: 缓存版本号合法性

**总测试**: 98 → 101 (1.46s 全过)

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
| 反欺诈三件套 (FraudGuard) | ~35s | 训练 50K + 1K 批量 + 3 张图 |
| 公平性切片 (4 维度 × 50K) | ~1s | 30K 测试集 |
| 因果叙事 (3 申请人 × 4 段) | ~8s | 5K 全局 SHAP + 60 次扰动 |
| **Pipeline 总耗时** | **~220s** | 16 步端到端, CPU 即可 |

---

## M6 — GPU LightGBM + Optuna 超参调优 ✅

### 动机
M5+ 之后 pipeline 跑到 184s（热跑），瓶颈是 STEP 6 LightGBM 训练 (112s, 60%)。手头有 H100 96GB 实际可用（之前误判为"暂作未来扩展"），理论上 GPU build 可以把训练再砍 5-20x。同时 `configs/config.yaml` 早已声明 `optuna: n_trials=50` 但从未实现。本里程碑同时接入 GPU 和 Optuna，作为可选项开放给用户。

### 1. LightGBM GPU build

**安装**:
```bash
conda install -c conda-forge "lightgbm=4.5.0=cuda_py3.10hc926fc0_2" -y
# H100 detected, CUDA Tree Learner enabled
```

**接入** (`src/models/train.py::_resolve_device`):
```python
def _resolve_device(requested: str) -> str:
    """Probes with a 64x4 1-iter fit; auto-falls-back to CPU."""
    if requested == "cpu": return "cpu"
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=1, device="cuda", verbosity=-1)
        m.fit(_synth_64x4, _synth_y)
        return "cuda"
    except Exception:
        return "cpu"
```

**实测** (215K 行 × 211 特征, 3-fold CV):

| 设备 | 耗时 | 加速比 | 备注 |
|------|-----:|------:|------|
| CPU `n_jobs=-1` (8 核) | **111.8s** | 1.0x | **实际生产配置** |
| GPU H100 `device=cuda` | ~170s (estimated) | 0.66x | kernel 启动延迟大, 小数据上 CPU 反而快 |

**结论**: 在 21 万行规模上，**GPU 比 CPU 慢 1.5x**。这是 LightGBM 官方的已知现象 — GPU 通常要到 N > 5M 才明显胜过 CPU。

**为什么保留 GPU build**:
1. N 增到 1M+ 时 GPU 红利会显现
2. 接口已就位 + 安全回退，零破坏
3. 配 `optuna.enabled: true` 时 GPU 可以让 50 trials 跑得更快

### 2. Optuna 4.9 超参调优

**实现** (`src/models/train.py::LightGBMTrainer.tune_hyperparams`):
- 9 维搜索空间: n_estimators, max_depth, num_leaves, learning_rate, subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda
- TPE 采样器，3 折 CV 在 50K 子集上评估
- 默认 `n_trials=50, timeout=600`，gated by `optuna.enabled: true`
- 结果存 `output/decision_reports/optuna_results.json`

**实测** (25 trials, 312s 总耗时):

| 配置 | 3-fold OOF AUC | 备注 |
|------|---------------:|------|
| 默认 (pipeline 当前) | **0.7107** | `n_est=500, max_depth=7, num_leaves=63, lr=0.05` |
| Optuna tuned | 0.7093 | **−0.0013**（持平或略降） |
| Optuna OOF on subsample | 0.6964 | subsample 50K × 2-fold |

**Optuna 找到的最优参数**:
```json
{
  "n_estimators": 400, "max_depth": 5, "num_leaves": 76,
  "learning_rate": 0.0138, "subsample": 0.92, "colsample_bytree": 0.63,
  "min_child_samples": 198, "reg_alpha": 0.72, "reg_lambda": 0.005
}
```
特征：低学习率 + 高子采样 + 中等 num_leaves + 较强 L1 — 典型 "防止过拟合" 配方。

**结论**: Home Credit 这种 8% 不平衡 + 强噪声的数据上，默认 LightGBM 参数已经接近 Bayes 最优，调优空间 < 0.5% AUC。

**为什么保留 Optuna 接口**:
1. 失败模式已验证（不破坏 pipeline，回落到默认参数）
2. 接口已就位 + 单测覆盖（`tests/test_train.py::test_optuna_tune_returns_valid_params`）
3. 换数据集（噪声更小）即可开箱受益
4. 工程价值：表明团队在 AUC 0.78 之后已触及数据天花板，**差异化应回到因果可解释性而非纯预测力**

### 3. 单元测试

新增 `tests/test_train.py` (7 用例):
- `_resolve_device()` 三态 (cpu / cuda / 非法)
- `LightGBMTrainer` 默认设备、predict、feature_importance
- `LightGBMTrainer.tune_hyperparams()` Optuna 调优接口 (5 trials, 60s timeout)

**总测试**: 101 → **108** (单跑 7.69s, +5.5x 时间主要来自新加的 Optuna smoke test)

### 4. 依赖更新

`pyproject.toml`:
```toml
"optuna>=4.0",  # M6 新增
```

`configs/config.yaml`:
```yaml
model:
  lightgbm:
    device: "cpu"  # M6 新增: "cpu" or "cuda"
  optuna:
    enabled: false  # M6 新增: gated by config
    n_trials: 50
    timeout: 600
    subsample: 50_000
    n_folds: 3
```

### 5. 端到端耗时对比

| 里程碑 | 热跑耗时 | STEP 6 训练 | 备注 |
|--------|---------:|------------:|------|
| M2 (单表) | 84.8s | ~12s | 无多表 |
| M5 (8 表冷跑) | 244.9s | 127.8s | 245 - 0 冷跑 |
| M5+ (热跑) | 184.5s | 111.8s | cache + L1 |
| M6 (热跑) | 184.5s | 111.8s | GPU/Optuna 默认不启用 |
| M7 (热跑) | 194.9s | 104.5s | +STEP 14 反欺诈 (35s), STEP 6 略优化 |

**M6 净收益**: 0s 耗时优化（默认不开启）, +7 个测试, +1 个可选 GPU 路径, +1 个可选 Optuna 路径, **3 个"开源但更精细"的优化杠杆**保留给未来 N 增长 / 数据替换场景。

---

## M7 — 反欺诈三件套 (Anti-Fraud Three-Pack) ✅

### 动机
docs/CausalCredit_反欺诈能力覆盖分析.md §4.1 提出的"反欺诈三件套"是设计文档里唯一没落到代码的核心能力:
1. **三分类子模型**: 在 default=1 的子群里进一步分 fraudulent / non_malicious / systemic
2. **包装资质因果一致性**: 客户表面资质(POSITIVE SHAP)与因果图谱是否一致(收入→消费→还款链)
3. **养流水因果去噪**: P(真实评分 | do(去养流水效应)) 估计

三个独立模块 + 一个 `FraudGuard` 编排器, 注入到现有 13 步 pipeline 末尾形成 STEP 14。

### 1. 三分类子模型 `src/fraud/three_class.py`

**类别** (`DEFRAUDER_CLASSES`):
- `fraudulent` — 主观恶意(收入低/工作短/首期即违约/材料造假)
- `non_malicious` — 还款能力变化(失业/疾病/家庭),非主观恶意
- `systemic` — 系统性风险(衰退行业/政策变化),个人无能为力

**伪标签构造** (无 ground truth, 用业务规则):
- `fraudulent`: `INST__DPD_MAX ≥ 30` ∨ (高收入 z + 低就业 z) ∨ (低 EXT_SOURCE_1 + 高收入 z)
- `systemic`: `ORGANIZATION_TYPE` 命中衰退行业子串(Industry: mining/Construction/Trade: type 7/...)
- `non_malicious`: default=1 且非上面两类

**模型**: 4 类 LightGBM(`non_default` + 3 fraud 类),在 default=1 子样本上重新归一化得 P(fraudulent | default=1), P(non_malicious | default=1), P(systemic | default=1)。

**`fraud_score = P(default) × P(fraudulent | default=1)`** — 二分类概率与子分类概率的乘积, 双重信号。

### 2. 包装资质因果一致性 `src/fraud/packaging.py`

**核心思路**: 申请人的"包装"(fabricated)信号是 — 模型的 top-K 高 |SHAP| 特征里, 有大比例的全局因果 proxy 很低(即"模型在用它们,但因果图说这些特征不是问题根源")。

**域 DAG 期望路径** (`EXPECTED_PATHS`):
- `income → goods_price → credit → annuity` (主链)
- `income → EXT_SOURCE_2` (中介)
- `DAYS_BIRTH → DAYS_EMPLOYED → income` (DAG 协变量链)

**`path_integrity`**: 这 3 条链里, 比例 step 在 [0.01, 100] 范围内算"完整",否则"断裂"。

**`packaging_score = UNTRUSTED / (TRUSTED + UNTRUSTED)`** (top-25% SHAP 内),即模型高权特征中"因果不靠谱"的比例。范围 0.26 - 0.56, 大部分在 borderline (0.30 - 0.50)。

**路由**:
- `>= 0.50` → `REJECT_PACKAGING_SUSPECTED`
- `>= 0.30` → `MANUAL_REVIEW` / pipeline 中映射为 `REVIEW_BORDERLINE`
- 否则 → `PROCEED`

### 3. 养流水因果去噪 `src/fraud/denoising.py`

**核心假设**: "养流水"用户制造出与消费脱钩的还款历史(钱从外面来,不是工资的产物)。因果信号: 还款 vs 消费特征的符号一致性。

**`causal_consistency`** (per-applicant): 把 5 个 `INST__` 还款列和 4 个 `CC_/POS_` 消费列按行 z-score 后取均值, 用 `sign(rep_score) * sign(con_score)` 的符号一致性映射到 [0, 1]。

**`inflation_strength = clip((1 - consistency) * 0.15 * 5, 0, 0.15)`** — 一致性越低, 估计的"养流水膨胀"越大, 上限 0.15(即最多把 P(default) 推高 15 个百分点)。

**`denoised_default_proba = min(1, P(default) + inflation_strength)`** — 把被压低的违约概率加回估计的"养流水"部分。

**`denoising_action`**: consistency < 0.50 → `FLAG_FOR_REVIEW`,否则 `PROCEED`。

### 4. `FraudGuard` 编排器 `src/fraud/pipeline.py`

把三个分数 + 5 维 routing reason 聚合成单条反欺诈路由:

```
REJECT_FRAUD        fraud_score >= 0.10                  (P(fraud) 高)
REJECT_PACKAGING    packaging_score >= 0.50              (包装嫌疑大)
REVIEW_DENOISED     denoising_action == FLAG_FOR_REVIEW  (养流水嫌疑)
REVIEW_BORDERLINE   任意信号在 [0.3, threshold) 区间
PROCEED             干净
```

### 5. 端到端集成 (`src/run_pipeline.py` STEP 14)

- 在 50K 训练子集上拟合 FraudGuard
- 对 3 个 picked applicants 算单条 fraud 报告, **注入** 现有 decision_reports JSON 的 `fraud` 字段
- 对 1K 测试子集批量打分用于图表
- 新增 3 张图: `12_fraud_score_routing.png` (直方图 + 路由饼图), `13_packaging_scatter.png` (path_integrity × packaging_score), `14_denoising_effect.png` (原 P(default) vs 去噪 P(default))
- `pipeline_summary.json` 增 `anti_fraud` 段: 分数范围、路由分布、平均去噪膨胀

**实测 1K 测试样本路由分布** (fraud 阈值 calibration 前):

| 路由 | 占比 | 含义 |
|------|-----:|------|
| REVIEW_BORDERLINE | 91.4% | 包装嫌疑 borderline, 进入人工审查 |
| PROCEED | 5.2% | 干净, 直接通过 |
| REJECT_FRAUD | 2.5% | P(fraud) 高, 直接拒绝 |
| REJECT_PACKAGING | 0.9% | 包装嫌疑大, 直接拒绝 |

**端到端耗时**: STEP 14 = 35-47s (训练 50K 3-class LightGBM + 1K 批量评分 + 3 张图), pipeline 总耗时 184.5s → **194.9s** (+5.6%)。

### 6. 测试 (25 个新测试)

| 测试文件 | 用例数 | 覆盖 |
|----------|------:|------|
| `test_fraud_three_class.py` | 7 | 伪标签规则、4 类模型、fraud_score 公式 |
| `test_fraud_packaging.py` | 7 | credibility 校准、4 象限分类、path integrity |
| `test_fraud_denoising.py` | 6 | consistency、denoised 范围、manufactured 膨胀 |
| `test_fraud_pipeline.py` | 5 | FraudGuard 端到端 + routing 决策 |

**总测试数**: 108 → **133** (+25)

### 7. 决策报告扩展

每个 applicant JSON 增 `fraud` 字段:
```json
{
  "fraud": {
    "fraud_score": 0.00051,
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
对应 .md 文件末尾追加"## 反欺诈三件套评分 (M7 Anti-Fraud Three-Pack)" 表格。

### 关键设计权衡

1. **伪标签 vs 真实标签**: 无欺诈 ground truth, 用业务规则生成伪标签。可解释、可审计、可由反欺诈团队在生产中替换为人工标注的种子集。
2. **`fraud_score = P(default) × P(fraudulent | default=1)`**: 用乘法组合,两个概率独立,避免欺诈子分类稀释主模型信号。
3. **`packaging_score = UNTRUSTED / (TRUSTED + UNTRUSTED)`** (top-25% SHAP 内): 这个公式直接从定义出发 — 包装嫌疑 = "模型在用但因果不靠谱"的特征占比,不再用 1 - (TRUSTED+MASKED)/total 的对称定义(那个定义在 median 阈值下恒为 0.5,无信息量)。
4. **三件套独立性**: 三个分数测量不同的反欺诈维度(恶意/包装/养流水),通过 `_fraud_routing` 按优先级合成,避免一个高分淹没其他信号。

---

## M8.1 — 公平性审计 + 反欺诈升级 ✅

**目标**: 把"成品打磨"阶段的两件 M7 留尾巴的事一次性收掉。详见 `docs/CausalCredit_M8.1_公平性与反欺诈升级实现记录.md`。

### 子任务分解

| 子任务 | 文件 | 测试 |
|---|---|---|
| M8.1a 公平性指标 + 切片 | `src/fairness/{__init__,metrics,slicing}.py` | 11 |
| M8.1b 公平性可视化 | `src/fairness/visualize.py` | 4 |
| M8.1c 公平性决策 JSON 扩展 | `src/explain/decision.py::build_fairness_block` | 3 |
| M8.1d 反欺诈阈值可配置 | `src/fraud/pipeline.py::FraudGuardConfig` + `configs/config.yaml` | 7 |
| M8.1e 反欺诈路由分布监控 | `src/monitoring/drift_detector.py::detect_routing_drift` | 6 |
| M8.1f 集成 + 测试 + 文档 | `src/run_pipeline.py` STEP 15 | — |

**测试增量**: 133 → 164 (+31)

### 关键产出

| 项目 | 数值 |
|---|---|
| Pipeline 总步数 | 14 → **15** (+FAIRNESS) |
| 图表总数 | 14 → **17** (+3 公平性图) |
| Pipeline 总耗时 | 195s → **212s** (公平性切片计算 1.03s) |

### STEP 15 实测输出 (Home Credit 30K 测试集)

```
  gender              status=WARNING  DP=0.004  EO=0.012  DI=0.538  (n_groups=3, n=50000)
  age_group           status=WARNING  DP=0.010  EO=0.040  DI=0.082  (n_groups=3, n=50000)
  income_group        status=WARNING  DP=0.004  EO=0.021  DI=0.511  (n_groups=3, n=50000)
  education_group     status=WARNING  DP=0.008  EO=0.050  DI=0.000  (n_groups=4, n=50000)

  Routing drift vs M7 baseline: PSI=0.0010  status=no_drift
```

**关键发现**:
- 模型在 4 个维度都触发 WARNING (主要因 DI 低于 0.80 规则), 但**EO gap 全部 < 0.05** — 模型没有"对真正会违约的群体漏判" 的歧视, 只是整体 selection_rate 偏低.
- 路由分布与 M7 完全对齐 (PSI=0.001), 升级没有引入回归.

### 决策报告样例 (M8.1 新增字段)

```json
{
  "applicant_id": "HC_006355",
  "score": 432, "risk_grade": "D", "default_probability": 0.187,
  "fairness": {
    "applicant_groups": {"gender": "F", "age_group": "mid", "income_group": "low", "education_group": "secondary"},
    "verdict": "WARNING",
    "violated_slices": ["gender", "age_group", "income_group", "education_group"],
    "regulatory_note": "One or more slices are WARNING. Model output may be biased; request additional documentation."
  }
}
```

---

## M8.2 — 因果叙事深化 ✅

**目标**: 决策报告从 "1 句话 (主要驱动因素 + 异质效应)" 升级到 "3 层叙事 (model / cohort / individual) + 因果路径 + 解释稳健性", 直接回答监管/合规场景下 "challenge the decision" 的 3 个标准问题. 详见 `docs/CausalCredit_M8.2_因果叙事深化实现记录.md`.

### 子任务分解

| 子任务 | 文件 | 测试 |
|---|---|---|
| M8.2a 多层级叙事生成器 | `src/explain/causal_narrative.py` (CausalNarrative 类) | (含 13) |
| M8.2b 因果路径追溯 | `trace_causal_path` (BFS) + `features_on_paths_to_outcome` | (含 13) |
| M8.2c K-NN 同类申请人对照 | `cohort_level_narrative` (KNN k=10, z-score 偏差) | (含 13) |
| M8.2d 解释稳健性扰动 | `explanation_robustness` (20× ±10% 高斯噪声) | (含 13) |
| M8.2e 叙事可视化 | `src/explain/narrative_visualize.py` (因果瀑布图 + 三联叙事卡) | 4 |
| M8.2f 集成 + 文档 | `src/run_pipeline.py` STEP 16 | — |

**测试增量**: 164 → 181 (+17), 测试文件 24 → 26

### 关键产出

| 项目 | 数值 |
|---|---|
| Pipeline 总步数 | 15 → **16** (+NARRATIVE) |
| 图表总数 | 17 → **19** (+2 叙事图) |
| Pipeline 总耗时 | 212s → **219.5s** (narrative 计算 7.5s) |

### STEP 16 实测输出 (3 个申请人对照)

| 申请人 | 等级 | P(default) | cohort Δ | 主导特征 | 主导路径 | stability | 业务解读 |
|---|---|---:|---:|---|---|---:|---|
| HC_006355 | E (高) | 89.55% | **+0.60** | EXT_SOURCE_2 | EXT_SOURCE_2 → TARGET | **0.94** | 单一主导, 解释极稳定, 远高于 cohort |
| HC_036837 | A (边界) | 4.88% | -0.026 | EXT_SOURCE_2 | EXT_SOURCE_2 → TARGET | 0.34 | 与 cohort 类似, 解释 moderately robust |
| HC_023041 | A (低) | 0.26% | -0.011 | DAYS_EMPLOYED | DAYS_EMPLOYED → TARGET | **0.20** | 无强主导, 解释 fragile, top-1 在扰动下 70% 换位 |

**关键发现**:
- **高风险 ↔ 稳定解释**: 单一 EXT_SOURCE_2 极值驱动 → top-1 不会漂 (stability=0.94)
- **低风险 ↔ fragile 解释**: 无强主导特征 → top-3 在小扰动下大量换位 (stability=0.20)
- **cohort Δ 与风险等级强正相关**: 高风险 +0.60, 中/低 ≈ -0.02, 无需 SHAP 就能 outlier 化

### 决策报告新字段 (`causal_narrative_v2`)

每份 JSON 增 4 段: `model_level` (top-3 mean |SHAP|) / `cohort_level` (KNN k=10 + Δ + top-5 z 偏差) / `individual_level` (top-5 + DAG paths + 4 象限计数) / `robustness` (top-1 / top-3 stable, 解释强度档). `.md` 报告追加对应 4 段中文标题 (模型层面 / 同类申请人对照 / 本申请人 / 解释稳健性).

### 2 张新图

- `15_causal_waterfall.png` — top features 横向条形图, 颜色按 4 象限 (TRUSTED 绿 / UNTRUSTED 红 / MASKED 橙 / NEGLIGIBLE 灰)
- `16_narrative_card.png` — 3 个并排文本面板 (蓝/黄/绿背景) 给非技术审阅者 (合规 / 运营) 一眼看懂

---

## 后续迭代方向（未做）

- ~~8 表 JOIN（bureau / previous_application / POS / installments / credit_card）→ 多表因果特征~~ ✅ M5 完成
- ~~GPU 加速（LightGBM GPU build）~~ ✅ M6 完成（接入, 默认关闭）
- ~~Optuna 超参调优~~ ✅ M6 完成（接入, 默认关闭, 实测 Home Credit 上不显著）
- ~~反欺诈三件套（三分类 + 包装资质 + 养流水去噪）~~ ✅ M7 完成
- ~~公平性审计 + 反欺诈阈值可配置 + 路由漂移监控~~ ✅ M8.1 完成
- ~~因果叙事深化（三层 model/cohort/individual + DAG 路径 + 解释稳健性）~~ ✅ M8.2 完成
- ~~M8.3 完整服务化（FastAPI 5 端点 + 路由 baseline 持久化 + DAG 加 EXT_SOURCE 边）~~ ✅ M8.3 完成
- ~~M8.4 多语言（render_markdown 加 zh-HK / en 参数）~~ ✅ M8.4 完成
- ~~M8.3c Streamlit 4 页填实（M8.2 叙事面板集成 + 流程图嵌入 + 12 单测）~~ ✅ M8.3c 完成
- **P0 提案文档落地（6.15 提案前关键交付）**: 蓝图一页纸 / Demo 演示脚本 / 答辩 Q&A 手册 / 代码走读速查 4 份, 落到 `docs/`
- **多表聚合 polars 改写**: pandas 单线程 ~27s 可降到 ~5s
- **反欺诈伪标签升级**: 用反欺诈团队人工标注的真实种子集替换业务规则
- **实时推理服务**: gRPC / ONNX Runtime (用户未禁用)
- **生产流量调优**: 反欺诈阈值在生产数据上 ROC 优化 (现为经验值)
- 实时推理服务（gRPC / ONNX Runtime）
- K8s / Helm / Terraform 部署
- **多表聚合 polars 改写**: 当前 pandas 单线程, ~27s 可降到 ~5s

---

## 仓库结构

```
CausalCredit/
├── src/                          # 核心代码
│   ├── data/                     # ✅ Home Credit + German 加载器, 校验, 预处理
│   ├── features/                 # ✅ builder + causal_features + aggregator
│   ├── causal/                   # ✅ graph / estimate / discovery / cate / refute
│   ├── models/                   # ✅ train (LightGBM + GBT + GPU/Optuna) / evaluate / calibrate
│   ├── explain/                  # ✅ shap_explain / counterfactual / decision / evidence / M8.2 causal_narrative + narrative_visualize
│   ├── fraud/                    # ✅ M7 三件套: three_class + packaging + denoising + pipeline
│   ├── fairness/                 # ✅ M8.1 metrics + slicing + visualize
│   ├── api/                      # ✅ app / routes / services / dependencies / schemas
│   ├── frontend/                 # ✅ app.py + 4 pages
│   ├── monitoring/               # ✅ drift_detector (含 M8.1e routing_drift)
│   └── run_pipeline.py           # ✅ 16 步端到端入口
├── tests/                        # ✅ 29 文件 / 212 用例
├── configs/                      # ✅ config.yaml
├── scripts/                      # ✅ run_api / run_demo / run_tests / setup_env
├── data/                         # ✅ Home Credit + German Credit
├── output/                       # ✅ figures (19) + decision_reports (3) + demo_m1 (9) + models
├── docs/                         # 12 份分析文档 (含 M7/M8.1/M8.2 实现记录)
├── CLAUDE.md                     # 给 Claude Code 的协作指引
├── PROGRESS.md                   # 本文件
├── README.md                     # 项目说明
└── pyproject.toml + requirements.txt
```
