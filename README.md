# CausalCredit — 因果推理增强信用评分系统

> **不只告诉你风险有多高，更告诉你为什么高、以及如何降低**  
> 从相关性预测到因果性决策的范式跃迁

## 项目简介

CausalCredit 是一套面向金融机构的**因果推理增强信用评分系统**，参加**中银香港创新先驱大赛2026**。系统在 LightGBM 预测基座之上，融合因果发现、异质处理效应估计（CATE）、反事实推理与因果约束的可解释性，构建"预测→归因→决策→落地"的完整闭环。

## 5 大核心亮点（已实现）

| # | 亮点 | 实现 | 入口 |
|---|------|------|------|
| 1 | 混合因果发现引擎 | PC 算法 + NOTEARS 融合 + 领域知识 DAG 注入 | `src/causal/discovery.py` + `src/causal/home_credit_graph.py` |
| 2 | CATE 异质处理效应 | EconML `LinearDML` + `ForestDRLearner` + `CausalForestDML` 三方法交叉验证 | `src/causal/cate.py` |
| 3 | 因果约束反事实 | DiCE NSGA-II + IMMUTABLE/SEMI-MUTABLE 锁定 + 因果图联动传播 + plausibility 评分 | `src/explain/counterfactual.py` |
| 4 | SHAP + 因果四象限 | TreeSHAP + 局部因果代理（±1σ 敏感性）→ TRUSTED/UNTRUSTED/NEGLIGIBLE/MASKED 四象限 | `src/explain/shap_explain.py` |
| 5 | 因果引导决策 | 信用分 (300-850) + 风险等级 A-E + 中英决策建议 + 证据链 + 反事实推荐 | `src/explain/decision.py` + `src/explain/evidence.py` |

## 技术栈

- **因果推理**: DoWhy 0.14, EconML 0.16, causal-learn (PC + NOTEARS)
- **预测模型**: LightGBM 4.6 + Isotonic Regression 校准
- **可解释性**: SHAP 0.48 (TreeSHAP), DiCE (NSGA-II)
- **后端**: FastAPI 0.136 + Pydantic v2
- **前端**: Streamlit 1.58 + streamlit-agraph
- **监控**: 自研 PSI 漂移检测（特征 / 预测 / 概念三层）
- **数据**: Home Credit Default Risk (307,511 × 122) 主数据集 + German Credit (1,000) 基线

## 快速开始

### 环境要求

- Python 3.11
- Conda 环境 `ldq_cc`（已装全部依赖）
- 数据：`data/home-credit-default-risk/application_train.csv`（Kaggle 下载）

### 一键运行

```bash
# 1) 端到端 pipeline（M2，13 步，约 75 秒）
/home/tony/anaconda3/envs/ldq_cc/bin/python -m src.run_pipeline

# 2) FastAPI 后端（M3）
/home/tony/anaconda3/envs/ldq_cc/bin/uvicorn src.api.app:app --port 8000

# 3) Streamlit 前端（M3）
/home/tony/anaconda3/envs/ldq_cc/bin/streamlit run src/frontend/app.py

# 4) 单元测试（M4，85 个用例，~1.3 秒）
/home/tony/anaconda3/envs/ldq_cc/bin/python -m pytest tests/ -v
```

### 5 个核心亮点的独立 demo（M1 输出）

```bash
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_causal_discovery.py   # PC + NOTEARS 融合
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_cate.py               # 3 方法 CATE
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_refute.py            # 4 类反驳 + E-value
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_counterfactual.py     # DiCE NSGA-II
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_shap.py              # SHAP 四象限
/home/tony/anaconda3/envs/ldq_cc/bin/python tests/test_decision.py          # 决策报告
```

## 项目结构

```
CausalCredit/
├── src/
│   ├── data/                  # HomeCreditLoader + German Loader + validator/preprocessing
│   ├── features/              # 因果特征工程（5 个特征）
│   ├── causal/                # DAG / discovery / ATE / CATE / refute
│   ├── models/                # LightGBM / GBT / 校准 / 评估
│   ├── explain/               # SHAP 四象限 / DiCE 反事实 / 决策 / 证据链
│   ├── api/                   # FastAPI 5 端点 + 业务服务层
│   ├── frontend/              # Streamlit 4 页（dashboard / 因果图 / 反事实 / 决策）
│   ├── monitoring/            # PSI 漂移检测（特征 / 预测 / 概念）
│   └── run_pipeline.py        # 13 步端到端入口
├── tests/                     # 14 个测试文件，85 用例
├── configs/                   # config.yaml
├── scripts/                   # run_api / run_demo / run_tests / setup_env
├── data/                      # Home Credit + German Credit
├── output/
│   ├── figures/               # 11 张 PNG
│   ├── decision_reports/      # 3 份 JSON + Markdown
│   ├── demo_m1/               # M1 5 创新点图表
│   └── models/                # 训练好的模型 pickle 缓存
└── docs/                      # 11 份原始分析文档
```

## 13 步 Pipeline 概览

| # | 步骤 | 输出 |
|---|------|------|
| 1 | 加载 Home Credit (307,511 × 122) | DataFrame |
| 2 | 数据校验（空值 / 类型 / 分布） | 校验报告 |
| 3 | 清洗（缺失值 + Winsorize） | 清洗 DataFrame |
| 4 | 特征工程（5 个因果特征） | 增广特征 |
| 5 | 划分 train/val/test | 索引 + 标签 |
| 6 | 训练 GBT + LightGBM（5-fold CV） | 评估指标 |
| 7 | 评估 + Isotonic 校准 | AUC + 校准曲线 |
| 8 | **因果发现**（PC + NOTEARS + 融合） | DAG + 6 类图 |
| 9 | ATE 估计（DoWhy + PSM） | ATE + 95% CI |
| 10 | **CATE**（DML + DR + Causal Forest） | 个体效应 + 子群分析 |
| 11 | **反驳验证**（4 类 refuter + E-value） | 通过/失败 + 鲁棒性分 |
| 12 | **SHAP 四象限** | 全局 / 局部 / 一致性 |
| 13 | **反事实 + 决策报告** | 3 份 JSON + 11 张图 |

## 当前实测结果（Home Credit, 30 万行）

> 详细 per-step 耗时、ATE/CATE/Refutation 数值、决策报告样例见 [`BENCHMARKS.md`](BENCHMARKS.md)

| 指标 | 数值 |
|------|------|
| 测试集 AUC-ROC | 0.7547 |
| ATE（`AMT_CREDIT` → `TARGET`, DoWhy backdoor） | +0.0092 (high vs low credit) |
| CATE 一致性（3 方法 mean Spearman） | 0.578 |
| 反驳验证 | 4 类中 3 类通过（E-value = 1.96） |
| 决策报告多样性 | 3 份样本覆盖 P = 0.31% / 5.35% / 73.50% |
| 输出图表 | 11 张 PNG + 9 张 M1 demo 图 |
| 单元测试 | 85 用例 / 1.34s |
| Pipeline 端到端耗时 | **84.8 秒**（CPU, 13 步, per-step 详见 BENCHMARKS） |

## 数据集

| 数据集 | 规模 | 角色 |
|--------|------|------|
| Home Credit Default Risk | 307,511 × 122 | 主数据集（单表） |
| German Credit | 1,000 × 20 | 快速基线对比与单元测试 |

## 许可证

MIT License
