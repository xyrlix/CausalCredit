# CausalCredit 开发进展记录

> **日期**: 2026-06-05 | **环境**: CPU (Python 3.10, sklearn/pandas/matplotlib)  
> **目标环境**: GPU (计划追加 LightGBM + DoWhy + EconML + SHAP + Streamlit)

---

## Git 提交历史

| # | Commit | 内容 | 文件 |
|:--:|--------|------|:--:|
| 1 | `b68de7e` | 项目骨架搭建 | 45 |
| 2 | `eb607f3` | 扩展模块（预处理/时序/反驳验证/监控） | 14 |
| 3 | `6f17d9f` | 完整端到端流水线（可运行） | 12 |
| 4 | `76b3ff3` | 自动生成 5 张可视化图表 | 6 |

---

## 已完成 ✅

### 跑通命令

```bash
python -m src.run_pipeline    # 33 秒, 12 步全通过
```

### 运行结果（German Credit, 1000 条）

| 指标 | 值 |
|------|-----|
| AUC-ROC | 0.7965 |
| 5-fold CV AUC | 0.7459 ± 0.0318 |
| duration → 违约 ATE | **+0.149 [0.058, 0.204]** p<0.05 |
| credit_amount → 违约 ATE | +0.030 [-0.066, 0.119] n.s. |

### 实现的模块（真实代码，非骨架）

| 模块 | 文件 | 功能 |
|------|------|------|
| 数据 | `src/data/loader.py` | sklearn fetch_openml 加载 German Credit |
| 数据 | `src/data/validator.py` | 空值/类型/分布校验 |
| 数据 | `src/data/preprocessing/cleaner.py` | 缺失值填补 + Winsorize 异常值 |
| 数据 | `src/data/preprocessing/encoder.py` | LabelEncoder + StandardScaler |
| 特征 | `src/features/builder.py` | 特征工程编排器 |
| 特征 | `src/features/causal_features.py` | 5 个因果特征 (DTI/loan_burden/age_credit_interaction/credit_per_year/existing_credit_ratio) |
| 模型 | `src/models/train.py` | GradientBoostingClassifier + 5-fold CV |
| 模型 | `src/models/evaluate.py` | AUC/Acc/Precision/Recall/F1/LogLoss |
| 因果 | `src/causal/graph.py` | 完整 DAG (8 混杂 + 2 处理变量 + 无环验证) |
| 因果 | `src/causal/estimate.py` | 手动 PSM + NearestNeighbors 匹配 + 200 Bootstrap CI |
| 因果 | `src/causal/variable_validation.py` | 处理变量/混杂因子分布验证 |
| 入口 | `src/run_pipeline.py` | 12 步端到端 + 5 张图表输出到 `output/figures/` |

### 基础设施

- `pyproject.toml` + `requirements.txt` (14 核心依赖)
- `configs/config.yaml` (数据/特征/模型/API 四段配置)
- `Makefile` (install/lint/test/run-api/run-demo/clean)
- `Dockerfile` + `docker-compose.yml` (API:8000 + Demo:8501)
- `.gitignore` + `.env.example` + `.pre-commit-config.yaml`
- `scripts/run_api.sh` + `scripts/run_demo.sh` + `scripts/run_tests.sh` + `scripts/setup_env.sh`

---

## GPU 环境待做 🔧

### 第一步：环境安装

```bash
pip install lightgbm shap dowhy econml streamlit
# 配置 Kaggle API → 下载 Home Credit Default Risk 数据集
```

### 第二步：核心升级（按优先级）

| 优先级 | 文件 | 任务 |
|:---:|------|------|
| 🔴 P0 | `src/data/loader.py` | 扩展 Home Credit 8 表加载 + 多表 JOIN |
| 🔴 P0 | `src/features/aggregation.py` | 多表聚合 (bureau/previous_app/POS/installments/credit_card) |
| 🔴 P0 | `src/features/pipelines/temporal_features.py` | DPD趋势 / 行为熵 / 信用利用率趋势 |
| 🔴 P0 | `src/models/train.py` | 切换 LightGBM + Optuna 超参数优化 |
| 🔴 P0 | `src/models/calibrate.py` | Isotonic Regression 概率校准 |
| 🔴 P0 | `src/causal/cate.py` | ⭐ EconML CausalForestDML CATE |
| 🔴 P0 | `src/causal/refute.py` | ⭐ 四重反驳验证 + E-value |
| 🟡 P1 | `src/explain/shap_explain.py` | ⭐ TreeSHAP + 四象限一致性 |
| 🟡 P1 | `src/explain/counterfactual.py` | ⭐ DiCE 反事实推理 |
| 🟡 P1 | `src/explain/decision.py` | ⭐ 个性化决策建议文本 |
| 🟡 P1 | `src/explain/evidence.py` | 可追溯证据链 |
| 🟢 P2 | `src/api/services.py` | FastAPI 评分/解释/反事实 API 业务逻辑 |
| 🟢 P2 | `src/frontend/app.py` | Streamlit 4 页交互 Demo |
| 🟢 P2 | `src/monitoring/drift_detector.py` | PSI 漂移检测 |
| 🟢 P2 | `src/data/lending_club_loader.py` | Lending Club 辅助验证集 |

### GPU 环境预期目标

| 指标 | CPU 当前值 | GPU 目标值 |
|------|:--------:|:--------:|
| 数据集 | German Credit (1,000) | Home Credit (307,511 × 8表) |
| 模型 | GradientBoosting (sklearn) | LightGBM 4.x |
| AUC | 0.796 | ≥ 0.78 (大样本+更好模型) |
| CATE | ❌ 不可做 | ✅ 薄信用人群 CATE 显著差异 |
| ATE 反驳 | 手动 PSM | 4 种方法全部稳健 |
| 反事实 | ❌ | ✅ "利率降2%→违约率从X%降至Y%" |
| 前端 | ❌ | ✅ Streamlit 4 页交互 Demo |
| 推理延迟 | 33s (全量) | < 100ms (单次) |

---

## 项目当前结构

```
CausalCredit/
├── src/
│   ├── data/                    ✅ loader, validator, lending_club_loader (骨架)
│   │   └── preprocessing/      ✅ cleaner, encoder
│   ├── features/                ✅ builder, causal_features, encoding (骨架)
│   │   └── pipelines/          🔶 temporal_features (骨架)
│   ├── causal/                  ✅ graph, estimate, variable_validation
│   │                           🔶 cate, refute (骨架)
│   ├── models/                  ✅ train, evaluate  🔶 calibrate (骨架)
│   ├── explain/                🔶 全部骨架 (shap/counterfactual/decision/evidence)
│   ├── api/                    🔶 app/routes/schemas 有框架, services 骨架
│   ├── frontend/               🔶 app.py 框架, 4 pages 骨架
│   ├── monitoring/             🔶 drift_detector 骨架
│   └── utils/                   ✅ config (YAML加载), logger
├── tests/                       ✅ conftest + fixtures
├── configs/                     ✅ config.yaml
├── scripts/                     ✅ run_api/demo/tests/setup_env
├── output/figures/              ✅ 5 张图表自动生成
└── docs/                        11 份原始文档
```
