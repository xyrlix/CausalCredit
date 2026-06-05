# CausalCredit — 因果推理增强信用评分系统：实施计划与AI Agent提示词

> **项目定位**：不只预测风险，更是指导决策——因果推理增强的可解释信用评分系统  
> **比赛范围**：中银香港创新先驱大赛2026 — 大数据×理财/ESG  
> **开发周期**：6周，3人团队  
> **算力成本**：~HK$500，CPU即可训练  
> **文档版本**：v1.0  
> **编写人**：大卫-解决方案架构师  
> **日期**：2026-06-05

---

## 目录

1. [项目总览](#一项目总览)
2. [系统架构设计](#二系统架构设计)
3. [6周分阶段实施计划](#三6周分阶段实施计划)
4. [关键技术决策及理由](#四关键技术决策及理由)
5. [风险点与应对](#五风险点与应对)
6. [项目里程碑与检查点](#六项目里程碑与检查点)

---

## 一、项目总览

### 1.1 项目目标

| 目标层级 | 描述 | 衡量标准 |
|---------|------|---------|
| **核心目标** | 构建因果推理增强的信用评分系统，实现从"预测谁会违约"到"指导如何降低违约"的范式升级 | CATE估计有效、反事实建议可生成、评分可解释 |
| **比赛目标** | 在中银香港创新先驱大赛2026中展示技术创新性与业务落地价值 | 评委认可因果推理的差异化价值，Demo可交互体验 |
| **技术目标** | AUC≥0.78（预测基座）+ CATE估计通过反驳验证 + 反事实解释可量化 | 量化指标达标，因果验证通过 |
| **业务目标** | 覆盖薄信用人群评分 + 优化贷款定价 + 满足HKMA可解释性要求 | 薄信用子群AUC≥0.72，反事实建议业务合理 |

### 1.2 项目范围

**在范围内（In-Scope）：**

- Home Credit Default Risk主数据集的完整因果分析流水线
- DoWhy因果图构建与效应估计
- EconML异质处理效应（CATE）估计
- LightGBM预测基座模型
- SHAP可解释性分析
- 反事实决策建议生成
- FastAPI后端API
- Streamlit交互式Demo
- Lending Club辅助验证实验
- 技术白皮书与比赛演示材料

**不在范围内（Out-of-Scope）：**

- 实时流式推理（仅批处理+API请求模式）
- 与银行核心系统的真实对接（仅设计接口规范）
- 多语言支持（仅英文+中文界面）
- 联邦学习/隐私计算（仅设计未来扩展路径）
- German Credit Risk数据集的完整实验（仅做算法验证）

### 1.3 关键约束

| 约束维度 | 具体限制 | 应对策略 |
|---------|---------|---------|
| **时间** | 6周硬截止 | 严格按里程碑推进，Week3设Go/No-Go检查点 |
| **人力** | 3人团队 | 明确分工：因果推理工程师(A)、ML工程师(B)、全栈开发(C) |
| **算力** | CPU即可，预算~HK$500 | LightGBM+DoWhy/EconML均为CPU友好；避免深度学习 |
| **数据** | 仅开源数据集，无真实银行数据 | 用Home Credit模拟银行场景，方案文档说明真实部署路径 |
| **因果推断** | 观测数据无真实干预 | 用倾向得分匹配(PSM)+工具变量(IV)+反驳验证确保稳健性 |

### 1.4 团队分工

| 角色 | 代号 | 职责 | 主要工具 |
|------|------|------|---------|
| 因果推理工程师 | A | 因果图构建、DoWhy效应估计、EconML CATE、反驳验证 | DoWhy, EconML, networkx |
| ML工程师 | B | 数据预处理、LightGBM基座、SHAP解释、反事实推理 | LightGBM, SHAP, pandas, scikit-learn |
| 全栈开发 | C | FastAPI后端、Streamlit前端、API设计、Demo集成 | FastAPI, Streamlit, Docker |

---

## 二、系统架构设计

### 2.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     CausalCredit 系统架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    应用层 (Application)                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │   │
│  │  │ Streamlit  │  │  FastAPI   │  │  反事实决策      │   │   │
│  │  │ 交互Demo   │  │  REST API  │  │  建议引擎        │   │   │
│  │  └─────┬──────┘  └─────┬──────┘  └────────┬─────────┘   │   │
│  └────────┼───────────────┼──────────────────┼──────────────┘   │
│           │               │                  │                   │
│  ┌────────┴───────────────┴──────────────────┴──────────────┐   │
│  │                    算法层 (Algorithm)                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │   │
│  │  │ 因果推理   │  │  预测基座  │  │  可解释性        │   │   │
│  │  │ DoWhy+EconML│  │  LightGBM  │  │  SHAP+反事实     │   │   │
│  │  │            │  │            │  │                  │   │   │
│  │  │ ·因果图    │  │ ·违约预测  │  │ ·特征归因       │   │   │
│  │  │ ·ATE/ATE  │  │ ·概率校准  │  │ ·反事实场景     │   │   │
│  │  │ ·CATE     │  │ ·薄信用子群│  │ ·决策建议       │   │   │
│  │  │ ·反驳验证 │  │ ·特征工程  │  │ ·证据链生成     │   │   │
│  │  └─────┬──────┘  └─────┬──────┘  └────────┬─────────┘   │   │
│  └────────┼───────────────┼──────────────────┼──────────────┘   │
│           │               │                  │                   │
│  ┌────────┴───────────────┴──────────────────┴──────────────┐   │
│  │                    数据层 (Data)                           │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │   │
│  │  │ 数据接入   │  │  特征工程  │  │  数据存储        │   │   │
│  │  │            │  │            │  │                  │   │   │
│  │  │ ·Kaggle API│  │ ·多表Join  │  │ ·Parquet文件    │   │   │
│  │  │ ·CSV加载   │  │ ·时序特征  │  │ ·SQLite元数据   │   │   │
│  │  │ ·数据校验  │  │ ·因果特征  │  │ ·模型注册表     │   │   │
│  │  │ ·版本管理  │  │ ·编码/标准化│  │ ·实验追踪       │   │   │
│  │  └────────────┘  └────────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    基础设施层 (Infrastructure)             │   │
│  │  Docker容器化 │ Git版本控制 │ MLflow实验追踪 │ 日志监控   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

| 模块名 | 路径 | 职责 | 负责人 |
|-------|------|------|-------|
| `data/` | `src/data/` | 数据下载、加载、校验、多表关联 | B |
| `features/` | `src/features/` | 特征工程、因果特征构造、编码 | B |
| `causal/` | `src/causal/` | 因果图构建、DoWhy效应估计、EconML CATE | A |
| `models/` | `src/models/` | LightGBM训练、评估、概率校准 | B |
| `explain/` | `src/explain/` | SHAP解释、反事实推理、决策建议生成 | A+B |
| `api/` | `src/api/` | FastAPI路由、请求模型、业务逻辑 | C |
| `frontend/` | `src/frontend/` | Streamlit页面、可视化组件 | C |
| `utils/` | `src/utils/` | 配置管理、日志、通用工具 | 全员 |
| `tests/` | `tests/` | 单元测试、集成测试 | 全员 |
| `docs/` | `docs/` | 技术白皮书、API文档 | A+C |

### 2.3 数据流

```
[原始数据]                    [特征工程]                [模型推理]              [输出]
                                                                              
Kaggle Home Credit ──┐                                          
  ·application_train │──→ data/loader.py ──→ features/          ──→ models/     ──→ api/
  ·bureau            │    ·多表Join         ·builder.py          ·predict.py      ·routes.py
  ·previous_app      │    ·数据校验         ·causal_features.py  ·calibrate.py    ·schemas.py
  ·POS_CASH_balance  │    ·缺失值处理       ·encoding.py         ·evaluate.py         │
  ·installments      │    ·类型推断         ·aggregation.py                          │
  ·credit_card       │                                                               │
                      │                                                               │
Kaggle Lending Club ──┤    data/loader.py ──→ features/          ──→ causal/     ──→ explain/
  ·loan_stats         │    ·辅助数据加载     ·builder.py          ·graph.py       ·shap_explain.py
                      │                     ·rate_features.py    ·estimate.py    ·counterfactual.py
Kaggle German Credit ─┘    ·验证集构造       ·causal_features.py  ·cate.py        ·decision.py
                           ·版本快照                              ·refute.py      ·evidence.py
                                                                     │                │
                                                                     └──────┬─────────┘
                                                                            ▼
                                                                     [因果增强评分]
                                                                     score = f(pred, cate, cf)
                                                                            │
                                                                            ▼
                                                                     frontend/app.py
                                                                     ·评分仪表盘
                                                                     ·因果效应可视化
                                                                     ·反事实情景模拟
                                                                     ·决策建议面板
```

### 2.4 接口定义

#### 2.4.1 核心API接口

| 端点 | 方法 | 功能 | 请求体 | 响应体 |
|------|------|------|--------|--------|
| `/api/v1/score` | POST | 信用评分+因果分析 | `CreditRequest` | `CreditResponse` |
| `/api/v1/counterfactual` | POST | 反事实情景模拟 | `CounterfactualRequest` | `CounterfactualResponse` |
| `/api/v1/explain` | POST | SHAP可解释性分析 | `ExplainRequest` | `ExplainResponse` |
| `/api/v1/causal-effect` | POST | 因果效应查询 | `CausalEffectRequest` | `CausalEffectResponse` |
| `/api/v1/health` | GET | 健康检查 | - | `{"status": "ok"}` |

#### 2.4.2 数据模型

```python
# --- 请求模型 ---
class CreditRequest(BaseModel):
    """信用评分请求"""
    applicant_id: Optional[str] = None
    features: Dict[str, Any]  # 申请特征键值对
    include_counterfactual: bool = True  # 是否包含反事实分析
    include_explanation: bool = True     # 是否包含SHAP解释

class CounterfactualRequest(BaseModel):
    """反事实情景请求"""
    applicant_id: Optional[str] = None
    features: Dict[str, Any]           # 基线特征
    interventions: Dict[str, Any]      # 干预变量及值，如 {"RATE": 0.08, "TERM": 24}

class CausalEffectRequest(BaseModel):
    """因果效应查询请求"""
    treatment: str          # 干预变量名，如 "AMT_CREDIT"
    outcome: str            # 结果变量名，如 "TARGET"
    subgroup: Optional[Dict[str, Any]] = None  # 子群筛选条件

# --- 响应模型 ---
class CreditResponse(BaseModel):
    """信用评分响应"""
    score: int                          # 信用评分 300-850
    default_probability: float          # 违约概率 0-1
    risk_grade: str                     # 风险等级 A/B/C/D/E
    causal_effect: Optional[Dict]       # 因果效应摘要
    counterfactual: Optional[List[Dict]] # 反事实情景列表
    explanation: Optional[Dict]         # SHAP解释
    decision_suggestion: Optional[str]  # 决策建议文本

class CounterfactualResponse(BaseModel):
    """反事实情景响应"""
    baseline_probability: float         # 基线违约概率
    counterfactual_probability: float   # 反事实违约概率
    probability_change: float           # 概率变化量
    intervention_details: Dict          # 干预详情
    confidence: float                   # 置信度 0-1
```

---

## 三、6周分阶段实施计划

---

### Week 1：数据基础与因果图构建

**阶段目标**：完成数据加载、多表关联、探索性分析，构建信贷领域因果图

---

#### 任务 W1-T1：项目初始化与开发环境搭建

**负责人**：C（全栈开发）  
**预计工时**：0.5天

**具体要求**：
- 创建项目目录结构（按2.2模块划分）
- 初始化Git仓库，配置`.gitignore`（排除数据文件、模型文件、__pycache__）
- 创建`pyproject.toml`，定义依赖：`dowhy`, `econml`, `lightgbm`, `shap`, `fastapi`, `streamlit`, `pandas`, `scikit-learn`, `mlflow`, `pydantic`, `uvicorn`
- 配置`pre-commit`钩子（black格式化、isort排序、flake8检查）
- 创建`Makefile`，定义常用命令：`make install`, `make lint`, `make test`, `make run-api`, `make run-demo`
- 创建`configs/config.yaml`，集中管理所有超参数和路径配置

**交付物**：
- 可运行的项目骨架
- `pyproject.toml` + `Makefile` + `configs/config.yaml`

**验收标准**：
- `make install` 成功安装所有依赖
- `make lint` 通过代码检查
- 目录结构与2.2模块划分一致

```prompt
你是一个Python项目初始化专家。请为"CausalCredit — 因果推理增强信用评分系统"项目创建完整的项目骨架。

【任务背景】
这是一个参加中银香港创新先驱大赛2026的AI项目，技术栈为DoWhy+EconML+LightGBM+SHAP+FastAPI+Streamlit，6周3人团队开发。

【具体要求】
1. 创建以下目录结构：
```
causalcredit/
├── src/
│   ├── data/           # 数据加载、校验
│   ├── features/       # 特征工程
│   ├── causal/         # 因果推理
│   ├── models/         # 预测模型
│   ├── explain/        # 可解释性
│   ├── api/            # FastAPI后端
│   ├── frontend/       # Streamlit前端
│   └── utils/          # 通用工具
├── tests/              # 测试
├── configs/            # 配置文件
├── docs/               # 文档
├── notebooks/          # 探索性分析
├── data/               # 数据目录(gitignore)
├── models/             # 模型存储(gitignore)
└── logs/               # 日志(gitignore)
```

2. 创建`pyproject.toml`，包含以下依赖：
   - dowhy>=0.11
   - econml>=0.15
   - lightgbm>=4.0
   - shap>=0.44
   - fastapi>=0.104
   - uvicorn>=0.24
   - streamlit>=1.29
   - pandas>=2.1
   - scikit-learn>=1.3
   - mlflow>=2.9
   - pydantic>=2.5
   - pyyaml>=6.0
   - matplotlib>=3.8
   - plotly>=5.18
   - networkx>=3.2

3. 创建`Makefile`，定义命令：install, lint, test, run-api, run-demo, clean

4. 创建`configs/config.yaml`，包含以下配置节：
   - data: 数据路径、文件名
   - features: 特征工程参数
   - causal: 因果推理参数（因果图路径、反驳测试参数）
   - model: LightGBM超参数
   - api: FastAPI配置（host, port）
   - frontend: Streamlit配置

5. 创建`.gitignore`，排除data/, models/, logs/, __pycache__, *.pyc, .env

6. 每个src子目录创建`__init__.py`，data/和features/创建占位模块文件

7. 创建`src/utils/config.py`，实现配置加载函数`load_config(config_path: str) -> dict`

【输出规格】
- 输出所有文件的完整内容
- 确保pyproject.toml依赖版本兼容
- 确保Makefile命令可执行

【技术约束】
- Python 3.10+
- 使用pyproject.toml而非setup.py
- 配置文件使用YAML格式
```

---

#### 任务 W1-T2：Home Credit数据集下载与加载

**负责人**：B（ML工程师）  
**预计工时**：1天

**具体要求**：
- 从Kaggle下载Home Credit Default Risk数据集（需Kaggle API Token）
- 实现`src/data/loader.py`，支持多表加载与关联
- 数据集包含8个CSV文件：`application_{train|test}.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv`
- 实现数据校验：行数、列数、主键唯一性、外键一致性、缺失率统计
- 输出数据概览报告（每张表的shape、dtypes、缺失率、唯一值数）

**交付物**：
- `src/data/loader.py`（数据加载模块）
- `src/data/validator.py`（数据校验模块）
- `notebooks/01_data_overview.ipynb`（数据概览报告）

**验收标准**：
- 所有8个CSV文件成功加载
- 数据校验通过（无主键冲突、外键一致）
- 数据概览报告包含每张表的完整统计信息

```prompt
你是一个数据工程师。请为Home Credit Default Risk数据集实现数据加载与校验模块。

【任务背景】
CausalCredit项目使用Home Credit Default Risk作为主数据集。该数据集包含8个CSV文件，通过SK_ID_CURR（当前申请ID）和SK_ID_BUREAU（征信局记录ID）关联。数据需加载到pandas DataFrame中供后续特征工程和因果分析使用。

【具体要求】
1. 实现`src/data/loader.py`，包含以下类和函数：

```python
class HomeCreditLoader:
    """Home Credit数据集加载器"""
    def __init__(self, data_dir: str):
        """初始化，data_dir为CSV文件所在目录"""
    
    def load_table(self, table_name: str) -> pd.DataFrame:
        """加载单张表，table_name为不含.csv的文件名"""
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """加载所有8张表，返回{表名: DataFrame}字典"""
    
    def get_joined_view(self, base_table: str = "application_train") -> pd.DataFrame:
        """返回以application_train为基表的左连接视图（仅关联bureau和previous_application）"""
    
    def get_table_info(self, table_name: str) -> Dict:
        """返回单张表的元信息：行数、列数、列类型、缺失率、唯一值数"""
```

2. 实现`src/data/validator.py`，包含以下函数：

```python
def validate_primary_key(df: pd.DataFrame, key_col: str) -> bool:
    """验证主键唯一性"""

def validate_foreign_key(df: pd.DataFrame, fk_col: str, ref_df: pd.DataFrame, ref_col: str) -> bool:
    """验证外键一致性（允许部分NULL）"""

def validate_data_integrity(tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """验证所有表的数据完整性，返回{表名: [问题列表]}"""

def generate_data_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """生成数据概览报告，每行一张表，列为统计指标"""
```

3. 8个CSV文件名映射：
   - application_train.csv → 主训练集（307,511行×122列，TARGET为标签）
   - application_test.csv → 测试集（48,744行×121列，无TARGET）
   - bureau.csv → 征信局历史记录
   - bureau_balance.csv → 征信局月度余额
   - previous_application.csv → 历史申请记录
   - POS_CASH_balance.csv → POS和现金贷款月度余额
   - installments_payments.csv → 分期付款历史
   - credit_card_balance.csv → 信用卡月度余额

4. 关联键：
   - application_* ↔ bureau: SK_ID_CURR
   - bureau ↔ bureau_balance: SK_ID_BUREAU
   - application_* ↔ previous_application: SK_ID_CURR
   - previous_application ↔ POS_CASH_balance: SK_ID_PREV
   - previous_application ↔ installments_payments: SK_ID_PREV
   - previous_application ↔ credit_card_balance: SK_ID_PREV

【输出规格】
- 完整的loader.py和validator.py源代码
- 每个函数包含docstring和类型注解
- 使用logging记录加载过程
- 大表（>100MB）加载时显示进度条（tqdm）

【技术约束】
- 使用pandas读取CSV，低内存模式（low_memory=True）
- 不使用数据库，纯文件系统存储
- 配置文件中指定data_dir路径
```

---

#### 任务 W1-T3：探索性数据分析（EDA）

**负责人**：B（ML工程师）  
**预计工时**：1.5天

**具体要求**：
- TARGET分布分析（正负样本比例、是否不平衡）
- 关键特征分布：AMT_CREDIT（贷款金额）、AMT_ANNUITY（年金）、AMT_INCOME_TOTAL（收入）、DAYS_BIRTH（年龄）、DAYS_EMPLOYED（工作年限）
- 特征与TARGET的关联分析：数值特征用KS检验，分类特征用卡方检验
- 多表关联统计：每个申请人平均有多少条bureau记录、previous_application记录
- 识别潜在干预变量（treatment）：贷款利率（可从AMT_CREDIT/AMT_ANNUITY/TERM推算）、贷款期限、贷款金额
- 识别混淆变量（confounder）：收入、年龄、工作年限、教育水平
- 识别结果变量（outcome）：TARGET（违约标志）
- 薄信用人群子群分析：无bureau记录的申请人占比、特征分布差异

**交付物**：
- `notebooks/02_eda.ipynb`（完整EDA笔记本）
- `docs/eda_summary.md`（EDA结论摘要，含关键发现列表）

**验收标准**：
- TARGET分布统计完整
- Top-20关联特征排序
- 至少识别3个潜在干预变量
- 薄信用子群特征分析完成

```prompt
你是一个数据科学家，擅长金融数据的探索性分析。请对Home Credit Default Risk数据集进行完整的EDA分析。

【任务背景】
CausalCredit项目需要理解Home Credit数据集的特征分布和关联关系，以指导后续因果图构建和特征工程。特别关注：①识别因果变量（干预/混淆/结果）②薄信用人群特征③多表关联结构。

【具体要求】
创建Jupyter Notebook `notebooks/02_eda.ipynb`，包含以下分析章节：

**第1章：TARGET分布与样本平衡性**
- TARGET值分布（0=正常还款，1=违约），计算正负样本比
- 按不同维度（性别、收入类型、教育水平）的违约率差异
- 结论：是否需要采样策略（如SMOTE、类权重调整）

**第2章：关键数值特征分布**
- 对以下特征绘制分布图（正常vs违约叠加直方图）：
  AMT_CREDIT, AMT_ANNUITY, AMT_INCOME_TOTAL, AMT_GOODS_PRICE,
  DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, EXT_SOURCE_1/2/3
- 计算每个特征的KS统计量和p值
- 识别DAYS_EMPLOYED中的异常值（365243代表缺失）
- 结论：Top-10区分力最强的数值特征

**第3章：分类特征与TARGET关联**
- 对所有分类特征（FLAG_OWN_CAR, FLAG_OWN_REALTY, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE, ORGANIZATION_TYPE等）：
  - 计算各类别的违约率
  - 卡方检验p值
- 结论：Top-10区分力最强的分类特征

**第4章：多表关联结构分析**
- 统计每个SK_ID_CURR在bureau/previous_application/POS_CASH/installments/credit_card中的记录数分布
- 计算关联覆盖率（有多少比例的申请人在其他表中有记录）
- 识别"薄信用"子群：bureau记录数=0的申请人
- 结论：多表关联的丰富度、薄信用人群占比和特征

**第5章：因果变量识别**
- 干预变量（Treatment）候选：
  - 贷款利率：从AMT_CREDIT/AMT_ANNUITY/TERM推算近似利率
  - 贷款期限：TERM（如存在）
  - 贷款金额：AMT_CREDIT
- 混淆变量（Confounder）候选：
  - 收入：AMT_INCOME_TOTAL
  - 年龄：-DAYS_BIRTH/365
  - 工作年限：-DAYS_EMPLOYED/365
  - 教育水平：NAME_EDUCATION_TYPE
  - 外部评分：EXT_SOURCE_1/2/3
- 结果变量（Outcome）：TARGET
- 对每个干预变量，分析其与混淆变量和结果变量的相关性
- 结论：推荐的主要干预变量和混淆变量列表

**第6章：薄信用人群深度分析**
- 定义薄信用：bureau记录数=0 AND previous_application记录数=0
- 薄信用人群占比
- 薄信用vs非薄信用的TARGET分布差异
- 薄信用人群的特征画像（年龄、收入、职业分布）
- 结论：薄信用人群是否违约率更高？哪些替代数据特征对薄信用人群最有预测力？

【输出规格】
- 完整的Jupyter Notebook代码
- 每章末尾有Markdown结论单元格
- 图表使用matplotlib+seaborn，中文标题
- 最终输出一个EDA结论摘要字典

【技术约束】
- 使用pandas进行数据处理
- 大表操作注意内存（分块处理或采样）
- 图表保存到docs/figures/目录
- 统计检验使用scipy.stats
```

---

#### 任务 W1-T4：信贷领域因果图构建

**负责人**：A（因果推理工程师）  
**预计工时**：2天

**具体要求**：
- 基于信贷领域知识和EDA结论，构建因果有向无环图（DAG）
- 使用DoWhy的`CausalModel`定义因果图
- 因果图需包含：干预变量（贷款利率/金额/期限）、混淆变量（收入/年龄/工作年限/教育）、中介变量（负债收入比）、结果变量（TARGET）
- 实现因果图的可视化（使用graphviz或networkx）
- 编写因果图合理性检验：无环检测、d-分离验证
- 敏感性分析框架：列出哪些因果假设可能被违反，如何通过反驳测试验证

**交付物**：
- `src/causal/graph.py`（因果图定义与验证模块）
- `docs/causal_graph.md`（因果图文档：节点定义、边定义、假设说明）
- `docs/figures/causal_dag.png`（因果DAG可视化）

**验收标准**：
- 因果图包含≥15个节点、≥20条边
- 无环检测通过
- d-分离验证至少3个条件独立性关系
- 因果图可视化清晰可读

```prompt
你是一个因果推理专家，擅长在金融领域构建因果图。请为CausalCredit项目构建信贷违约的因果有向无环图（DAG）。

【任务背景】
CausalCredit的核心创新是因果推理增强信用评分。需要基于信贷领域知识和Home Credit数据集特征，构建一个合理的因果DAG，定义干预变量、混淆变量、中介变量和结果变量之间的因果关系。这个因果图将指导后续的DoWhy效应估计和EconML CATE计算。

【具体要求】
1. 实现`src/causal/graph.py`，包含以下类和函数：

```python
class CreditCausalGraph:
    """信贷违约因果图"""
    
    def __init__(self):
        """初始化因果图，定义所有节点和边"""
    
    def get_dowhy_graph(self) -> str:
        """返回DoWhy格式的因果图字符串（dot格式）"""
    
    def get_treatment_variables(self) -> List[str]:
        """返回干预变量列表"""
    
    def get_confounders(self, treatment: str, outcome: str) -> List[str]:
        """返回指定treatment-outcome对的混淆变量"""
    
    def get_mediators(self, treatment: str, outcome: str) -> List[str]:
        """返回指定treatment-outcome对的中介变量"""
    
    def get_instruments(self, treatment: str) -> List[str]:
        """返回指定treatment的工具变量"""
    
    def validate_acyclic(self) -> bool:
        """验证因果图无环"""
    
    def validate_d_separation(self) -> List[Dict]:
        """验证d-分离关系，返回[{条件, 独立对, 是否成立}]"""
    
    def visualize(self, output_path: str):
        """可视化因果图，保存为PNG"""
    
    def get_assumptions(self) -> List[str]:
        """返回因果图的关键假设列表"""
```

2. 因果图节点定义（基于Home Credit特征）：

**干预变量（Treatment）**：
- `interest_rate`：贷款利率（从AMT_CREDIT/AMT_ANNUITY推算）
- `credit_amount`：贷款金额（AMT_CREDIT）
- `credit_term`：贷款期限（推算）

**混淆变量（Confounder）**：
- `income`：收入（AMT_INCOME_TOTAL）
- `age`：年龄（-DAYS_BIRTH/365）
- `employment_years`：工作年限（-DAYS_EMPLOYED/365）
- `education_level`：教育水平（NAME_EDUCATION_TYPE编码）
- `ext_score`：外部评分（EXT_SOURCE_1/2/3加权）
- `occupation_type`：职业类型（OCCUPATION_TYPE编码）

**中介变量（Mediator）**：
- `debt_to_income`：负债收入比（AMT_ANNUITY/AMT_INCOME_TOTAL）
- `credit_utilization`：信用利用率（从credit_card_balance推算）
- `payment_history`：还款历史（从installments_payments推算）

**结果变量（Outcome）**：
- `default`：违约标志（TARGET）

**其他变量**：
- `gender`：性别（CODE_GENDER）
- `family_status`：家庭状况（NAME_FAMILY_STATUS）
- `housing_type`：住房类型（NAME_HOUSING_TYPE）
- `car_ownership`：车辆拥有（FLAG_OWN_CAR）
- `reality_ownership`：房产拥有（FLAG_OWN_REALTY）
- `region_rating`：区域评分（REGION_RATING_CLIENT）

3. 因果边定义（关键因果关系，需有领域知识支撑）：

**核心因果路径**（必须有）：
- income → credit_amount（收入影响贷款金额）
- income → debt_to_income → default（收入通过负债比影响违约）
- interest_rate → debt_to_income → default（利率通过负债比影响违约）
- interest_rate → default（利率直接影响违约）
- credit_amount → debt_to_income → default（贷款金额通过负债比影响违约）
- age → income（年龄影响收入）
- employment_years → income（工作年限影响收入）
- education_level → income → credit_amount（教育通过收入影响贷款）

**混淆路径**（必须有）：
- income → interest_rate AND income → default（收入同时影响利率和违约）
- age → employment_years AND age → default（年龄同时影响工作年限和违约）
- ext_score → interest_rate AND ext_score → default（外部评分同时影响利率和违约）

**工具变量路径**（如果有）：
- region_rating → interest_rate（区域评分影响利率，但不直接影响违约——需验证）

4. 因果图合理性检验：
- 无环检测：使用networkx.is_directed_acyclic_graph()
- d-分离验证：至少验证3个条件独立性关系，如：
  - 给定income，credit_amount ⊥ interest_rate？（验证收入是否充分控制了混淆）
  - 给定debt_to_income，interest_rate ⊥ default？（验证中介路径）
  - gender ⊥ default | income, age, education？（验证性别是否直接影响违约）

5. 敏感性分析框架：
- 列出5个关键因果假设
- 每个假设可能被违反的原因
- 对应的反驳测试方法（随机安慰剂、数据子集、添加未观测混淆）

【输出规格】
- 完整的graph.py源代码
- 因果图可视化PNG（节点用不同颜色区分：红色=干预，蓝色=混淆，绿色=中介，黄色=结果，灰色=其他）
- 因果图文档markdown

【技术约束】
- 使用DoWhy的CausalModel接受的dot格式字符串
- 可视化使用graphviz（需安装python-graphviz）
- 节点名使用Home Credit原始特征名或推算特征名
- 所有因果关系需有注释说明领域知识依据
```

---

#### 任务 W1-T5：Lending Club辅助数据集预处理

**负责人**：B（ML工程师）  
**预计工时**：0.5天

**具体要求**：
- 下载Lending Club Loan Data（200万+条）
- 筛选关键特征：loan_amnt, int_rate, installment, grade, sub_grade, annual_inc, dti, emp_length, home_ownership, loan_status
- 将loan_status映射为二分类标签（Fully Paid=0, Charged Off/Default=1，其余排除）
- 构建利率变化准实验：按grade分组，分析不同利率水平对违约率的因果效应
- 保存处理后的数据为Parquet格式

**交付物**：
- `src/data/lending_club_loader.py`
- `data/lending_club_processed.parquet`

**验收标准**：
- 处理后数据≥100万条
- 二分类标签分布合理（违约率10-25%）
- 利率-违约率按grade分组统计完成

```prompt
你是一个数据工程师。请为CausalCredit项目预处理Lending Club Loan Data辅助数据集。

【任务背景】
CausalCredit使用Lending Club数据作为辅助验证集。Lending Club是真实P2P贷款数据，包含利率(int_rate)和违约(loan_status)，利率变化可作为准实验验证因果效应估计的合理性。

【具体要求】
1. 实现`src/data/lending_club_loader.py`，包含：

```python
class LendingClubLoader:
    def __init__(self, data_dir: str):
        pass
    
    def load_raw(self) -> pd.DataFrame:
        """加载原始CSV，处理编码问题"""
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """筛选关键特征列"""
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """将loan_status映射为二分类TARGET"""
    
    def create_quasi_experiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建利率变化准实验数据"""
    
    def save_processed(self, df: pd.DataFrame, output_path: str):
        """保存为Parquet"""
```

2. 特征筛选列表：
   - loan_amnt, funded_amnt, int_rate, installment, grade, sub_grade
   - annual_inc, dti, emp_length, home_ownership
   - purpose, addr_state, delinq_2yrs
   - open_acc, total_acc, revol_bal, revol_util
   - loan_status

3. loan_status映射规则：
   - TARGET=0: Fully Paid
   - TARGET=1: Charged Off, Default
   - 排除: Current, Late (31-120 days), In Grace Period, Late (16-30 days), Issued

4. 准实验构建：
   - 按grade分组（A-G），每组内int_rate有自然变异
   - 计算每组的平均利率和违约率
   - 构建工具变量：grade→int_rate→default（grade影响利率但不直接影响违约）

5. 数据清洗：
   - int_rate: 去除%符号，转为float
   - emp_length: "10+ years"→10, "< 1 year"→0, "n/a"→NaN
   - revol_util: 去除%符号
   - 去除annual_inc极端值（>99th percentile）

【输出规格】
- 完整的lending_club_loader.py源代码
- 处理后数据统计摘要（行数、列数、TARGET分布、grade分布）

【技术约束】
- 大文件使用chunksize分块读取
- 输出使用Parquet格式（压缩比好）
- 内存控制：处理时不超过8GB
```

---

### Week 2：特征工程与预测基座

**阶段目标**：完成因果特征工程，训练LightGBM预测基座模型，建立性能基线

---

#### 任务 W2-T1：多表关联特征工程

**负责人**：B（ML工程师）  
**预计工时**：2天

**具体要求**：
- 实现多表聚合特征：对bureau/previous_application/POS_CASH/installments/credit_card按SK_ID_CURR聚合
- 聚合函数：mean, std, min, max, count, nunique, last
- 构建时序特征：最近N个月的还款行为趋势
- 构建因果特征：负债收入比、信用利用率、还款准时率
- 特征选择：基于LightGBM特征重要性初步筛选，保留Top-80特征
- 特征编码：分类特征用LabelEncoder+目标编码，数值特征用StandardScaler

**交付物**：
- `src/features/builder.py`（特征构建主模块）
- `src/features/aggregation.py`（多表聚合模块）
- `src/features/encoding.py`（特征编码模块）
- `src/features/causal_features.py`（因果特征模块）

**验收标准**：
- 最终特征数≥80（含原始+聚合+因果特征）
- 特征无数据泄漏（不使用TARGET信息）
- 分类特征编码完成
- 特征重要性排序输出

```prompt
你是一个特征工程专家，擅长金融数据的特征构建。请为CausalCredit项目实现完整的多表关联特征工程流水线。

【任务背景】
CausalCredit使用Home Credit多表关联数据集。需要从8张关联表中构建丰富的特征，特别关注因果特征（负债收入比、信用利用率等），这些特征将同时用于预测模型和因果推理。

【具体要求】
1. 实现`src/features/aggregation.py`，多表聚合特征：

```python
class MultiTableAggregator:
    """多表聚合特征构建器"""
    
    def aggregate_bureau(self, bureau_df: pd.DataFrame, bureau_bal_df: pd.DataFrame) -> pd.DataFrame:
        """聚合征信局特征：
        - 活跃贷款数、关闭贷款数
        - 历史贷款金额均值/总和
        - 历史逾期次数（DPD>0的月数）
        - 信用历史长度（月）
        - 债务总额
        """
    
    def aggregate_previous_app(self, prev_df: pd.DataFrame) -> pd.DataFrame:
        """聚合历史申请特征：
        - 历史申请次数
        - 历史批准率
        - 历史贷款金额均值
        - 历史利率（NAME_CONTRACT_TYPE分布）
        - 上次申请距今天数
        """
    
    def aggregate_pos_cash(self, pos_df: pd.DataFrame) -> pd.DataFrame:
        """聚合POS/现金贷款特征：
        - 活跃POS贷款数
        - 逾期月数占比
        - 剩余期数均值
        """
    
    def aggregate_installments(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """聚合分期付款特征：
        - 还款准时率（按时还款次数/总次数）
        - 平均延迟天数
        - 平均还款金额/应还金额比
        - 最近6个月还款趋势（线性回归斜率）
        """
    
    def aggregate_credit_card(self, cc_df: pd.DataFrame) -> pd.DataFrame:
        """聚合信用卡特征：
        - 信用卡利用率均值（AMT_BALANCE/AMT_CREDIT_LIMIT_ACTUAL）
        - 最低还款率
        - 逾期次数
        - ATM取现占比
        """
    
    def aggregate_all(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """聚合所有表，返回以SK_ID_CURR为索引的特征DataFrame"""
```

2. 实现`src/features/causal_features.py`，因果特征构建：

```python
class CausalFeatureBuilder:
    """因果特征构建器"""
    
    def build_debt_to_income(self, df: pd.DataFrame) -> pd.Series:
        """构建负债收入比 = AMT_ANNUITY / AMT_INCOME_TOTAL"""
    
    def build_credit_utilization(self, df: pd.DataFrame, cc_agg: pd.DataFrame) -> pd.Series:
        """构建信用利用率 = 信用卡余额 / 信用额度"""
    
    def build_payment_discipline(self, inst_agg: pd.DataFrame) -> pd.Series:
        """构建还款纪律评分 = 加权(准时率 * 0.5 + 金额比 * 0.3 + 趋势 * 0.2)"""
    
    def build_approx_interest_rate(self, df: pd.DataFrame) -> pd.Series:
        """构建近似利率：
        使用PMT公式反推：rate = PMT_RATE(AMT_ANNUITY, CREDIT_TERM, AMT_CREDIT)
        若无TERM，用近似：rate ≈ (AMT_ANNUITY * n - AMT_CREDIT) / (AMT_CREDIT * n / 2)
        """
    
    def build_credit_term(self, df: pd.DataFrame) -> pd.Series:
        """构建贷款期限（月）= AMT_CREDIT / AMT_ANNUITY（近似）"""
    
    def build_thin_credit_flag(self, bureau_agg: pd.DataFrame, prev_agg: pd.DataFrame) -> pd.Series:
        """构建薄信用标志：bureau记录数=0 AND previous_application记录数=0"""
    
    def build_all_causal_features(self, df: pd.DataFrame, agg_features: pd.DataFrame) -> pd.DataFrame:
        """构建所有因果特征，返回附加因果特征的DataFrame"""
```

3. 实现`src/features/encoding.py`，特征编码：

```python
class FeatureEncoder:
    """特征编码器"""
    
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.scaler = StandardScaler()
    
    def fit_transform(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """拟合并转换：
        - 二分类特征：LabelEncoder
        - 多分类特征（基数<10）：OneHotEncoder
        - 多分类特征（基数≥10）：TargetEncoder（需target）
        - 数值特征：StandardScaler
        """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的编码器转换新数据"""
```

4. 实现`src/features/builder.py`，特征构建主入口：

```python
class FeatureBuilder:
    """特征构建主入口"""
    
    def __init__(self, config: dict):
        pass
    
    def build(self, tables: Dict[str, pd.DataFrame], fit: bool = True) -> pd.DataFrame:
        """完整特征构建流水线：
        1. 多表聚合
        2. 因果特征构建
        3. 特征编码
        4. 特征选择（Top-80 by LightGBM importance）
        5. 返回最终特征矩阵
        """
```

【输出规格】
- 4个Python模块的完整源代码
- 每个聚合函数的输出特征列表
- 因果特征的计算公式和业务含义

【技术约束】
- 聚合操作使用pandas groupby + agg，避免循环
- 大表聚合注意内存，必要时分块
- 目标编码使用5折交叉编码避免数据泄漏
- 所有特征构建不使用TARGET信息
```

---

#### 任务 W2-T2：LightGBM预测基座模型训练

**负责人**：B（ML工程师）  
**预计工时**：1.5天

**具体要求**：
- 使用Week2-T1构建的特征训练LightGBM分类模型
- 5折交叉验证，报告平均AUC、Recall、Precision、F1
- 概率校准：使用Isotonic Regression校准违约概率
- 薄信用子群单独评估
- 超参数调优：使用Optuna进行贝叶斯优化（50次试验）
- 特征重要性分析：Gain-based + SHAP值
- 保存最佳模型和校准器

**交付物**：
- `src/models/train.py`（模型训练模块）
- `src/models/evaluate.py`（模型评估模块）
- `src/models/calibrate.py`（概率校准模块）
- `models/lgbm_best.txt`（最佳LightGBM模型）
- `models/calibrator.pkl`（概率校准器）
- `notebooks/03_model_evaluation.ipynb`（模型评估报告）

**验收标准**：
- 5折CV平均AUC≥0.76
- 薄信用子群AUC≥0.70
- 概率校准后Brier Score降低
- Top-20特征重要性排序输出

```prompt
你是一个机器学习工程师，擅长表格数据的树模型训练。请为CausalCredit项目训练LightGBM预测基座模型。

【任务背景】
CausalCredit使用LightGBM作为预测基座，提供违约概率预测。该预测将作为因果增强评分的基础。需要确保模型性能达标，概率校准准确，且对薄信用人群有合理表现。

【具体要求】
1. 实现`src/models/train.py`：

```python
class LightGBMTrainer:
    """LightGBM训练器"""
    
    def __init__(self, config: dict):
        """初始化超参数和训练配置"""
    
    def train_cv(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict:
        """5折交叉验证训练，返回每折的模型和指标"""
    
    def train_final(self, X: pd.DataFrame, y: pd.Series) -> lgb.Booster:
        """使用最佳超参数在全量训练集上训练最终模型"""
    
    def optuna_tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict:
        """Optuna贝叶斯优化超参数，搜索空间：
        - num_leaves: [20, 100]
        - max_depth: [3, 10]
        - learning_rate: [0.01, 0.3] log
        - n_estimators: [100, 1000]
        - min_child_samples: [5, 100]
        - subsample: [0.5, 1.0]
        - colsample_bytree: [0.5, 1.0]
        - reg_alpha: [0, 10] log
        - reg_lambda: [0, 10] log
        - scale_pos_weight: 根据正负样本比自动计算
        """
    
    def save_model(self, model: lgb.Booster, path: str):
        """保存模型"""
    
    def load_model(self, path: str) -> lgb.Booster:
        """加载模型"""
```

2. 实现`src/models/evaluate.py`：

```python
class ModelEvaluator:
    """模型评估器"""
    
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """计算评估指标：
        - AUC-ROC
        - AUC-PR
        - Precision, Recall, F1 (threshold=0.5)
        - KS统计量
        - Brier Score
        - Log Loss
        """
    
    def evaluate_subgroup(self, X: pd.DataFrame, y_true: pd.Series, y_prob: np.ndarray, 
                          subgroup_col: str) -> pd.DataFrame:
        """按子群评估模型性能，返回每组的AUC和KS"""
    
    def plot_roc_curve(self, y_true, y_prob, output_path: str):
        """绘制ROC曲线"""
    
    def plot_ks_curve(self, y_true, y_prob, output_path: str):
        """绘制KS曲线"""
    
    def plot_feature_importance(self, model, top_n: int = 20, output_path: str = None):
        """绘制特征重要性（Gain-based）"""
    
    def generate_report(self, metrics: Dict, subgroup_metrics: pd.DataFrame) -> str:
        """生成评估报告文本"""
```

3. 实现`src/models/calibrate.py`：

```python
class ProbabilityCalibrator:
    """概率校准器"""
    
    def fit_isotonic(self, y_true: pd.Series, y_prob: np.ndarray) -> IsotonicRegression:
        """拟合Isotonic Regression校准器"""
    
    def calibrate(self, y_prob: np.ndarray, calibrator) -> np.ndarray:
        """校准概率"""
    
    def evaluate_calibration(self, y_true: pd.Series, y_prob_raw: np.ndarray, 
                             y_prob_calibrated: np.ndarray) -> Dict:
        """评估校准效果：
        - 校准前后Brier Score
        - 校准前后ECE（Expected Calibration Error）
        - 可靠性图（Reliability Diagram）
        """
```

4. 训练流程：
   a. 加载特征数据
   b. 5折CV基线训练（默认超参数）
   c. Optuna超参数优化（50次试验，目标=5折CV AUC）
   d. 最佳超参数全量训练
   e. 概率校准（Isotonic Regression）
   f. 子群评估（薄信用vs非薄信用）
   g. 保存模型和评估报告

5. 评估报告需包含：
   - 整体指标表格
   - 薄信用子群指标
   - ROC曲线、KS曲线
   - 特征重要性Top-20
   - 概率校准前后对比

【输出规格】
- 3个Python模块的完整源代码
- 模型评估报告notebook
- 保存的模型文件

【技术约束】
- LightGBM使用early_stopping（50轮无改善停止）
- 交叉验证使用StratifiedKFold
- Optuna使用TPESampler
- 概率校准使用单独的验证集（20% holdout）
- 所有随机种子固定（seed=42）
```

---

#### 任务 W2-T3：因果特征与预测特征的对齐验证

**负责人**：A（因果推理工程师）  
**预计工时**：1天

**具体要求**：
- 验证因果图中定义的变量在特征工程中是否已正确构建
- 检查干预变量（interest_rate, credit_amount, credit_term）的分布合理性
- 检查混淆变量与干预变量、结果变量的相关性
- 验证中介变量（debt_to_income, credit_utilization, payment_discipline）的计算正确性
- 生成因果变量质量报告

**交付物**：
- `src/causal/variable_validation.py`
- `notebooks/04_causal_variable_validation.ipynb`
- `docs/causal_variable_report.md`

**验收标准**：
- 所有因果变量已构建且分布合理
- 混淆变量与干预变量的相关性矩阵输出
- 中介变量计算公式验证通过

```prompt
你是一个因果推理专家。请验证CausalCredit项目中因果变量与特征工程的对齐情况。

【任务背景】
Week1构建了因果图，Week2构建了特征。需要验证因果图中定义的干预变量、混淆变量、中介变量在特征工程中是否已正确构建，分布是否合理，为Week3的因果效应估计做准备。

【具体要求】
1. 实现`src/causal/variable_validation.py`：

```python
class CausalVariableValidator:
    """因果变量验证器"""
    
    def validate_treatment_variables(self, df: pd.DataFrame, 
                                      treatments: List[str]) -> Dict[str, Dict]:
        """验证干预变量：
        - 分布统计（均值、标准差、分位数）
        - 变异系数（CV>0.1才算有足够变异）
        - 异常值检测（IQR方法）
        - 与TARGET的单变量因果效应方向（正相关/负相关）
        """
    
    def validate_confounders(self, df: pd.DataFrame, 
                             treatments: List[str],
                             outcome: str,
                             confounders: List[str]) -> Dict[str, Dict]:
        """验证混淆变量：
        - 与每个treatment的相关系数
        - 与outcome的相关系数
        - 是否同时与treatment和outcome相关（混淆的必要条件）
        - VIF检验（多重共线性，VIF>10需注意）
        """
    
    def validate_mediators(self, df: pd.DataFrame,
                           treatments: List[str],
                           outcome: str,
                           mediators: List[str]) -> Dict[str, Dict]:
        """验证中介变量：
        - treatment→mediator的相关性
        - mediator→outcome的相关性
        - 中介效应占比估计（简单Sobel检验）
        """
    
    def validate_instruments(self, df: pd.DataFrame,
                              treatment: str,
                              instruments: List[str]) -> Dict[str, Dict]:
        """验证工具变量：
        - instrument→treatment的相关性（F统计量>10为强工具变量）
        - instrument与outcome的独立性（排除限制检验近似）
        """
    
    def generate_quality_report(self, validation_results: Dict) -> str:
        """生成因果变量质量报告Markdown"""
```

2. 验证内容：
   - interest_rate分布是否合理（通常5%-30%）
   - credit_amount与AMT_CREDIT是否一致
   - debt_to_income计算是否正确（AMT_ANNUITY/AMT_INCOME_TOTAL）
   - 混淆变量是否同时与treatment和outcome相关
   - 薄信用人群的因果变量是否有足够变异

3. 如果发现变量质量问题，给出修正建议：
   - 变异不足：考虑对数变换或分箱
   - 异常值过多：Winsorize处理
   - 混淆不充分：考虑添加更多混淆变量
   - 工具变量弱：考虑替换或放弃

【输出规格】
- 完整的variable_validation.py源代码
- 验证notebook，包含所有可视化
- 因果变量质量报告

【技术约束】
- 相关性使用Pearson（数值-数值）和Point-biserial（数值-二分类）
- VIF使用statsmodels.stats.outliers_influence.variance_inflation_factor
- F统计量使用statsmodels OLS回归
```

---

### Week 3：因果效应估计与CATE

**阶段目标**：完成DoWhy因果效应估计和EconML CATE计算，通过反驳验证

**🚨 Go/No-Go检查点**：本周结束评估因果效应是否显著且稳健，决定是否继续或调整方案

---

#### 任务 W3-T1：DoWhy因果效应估计（ATE）

**负责人**：A（因果推理工程师）  
**预计工时**：2天

**具体要求**：
- 使用DoWhy的CausalModel加载Week1构建的因果图
- 对3个干预变量（interest_rate, credit_amount, credit_term）分别估计ATE
- 使用多种效应估计方法：回归法、IPW（逆概率加权）、倾向得分匹配（PSM）
- 对每个ATE进行反驳验证：随机安慰剂、数据子集、添加未观测混淆
- 生成因果效应估计报告

**交付物**：
- `src/causal/estimate.py`（因果效应估计模块）
- `notebooks/05_causal_effect_estimation.ipynb`
- `docs/causal_effect_report.md`

**验收标准**：
- 3个干预变量的ATE均成功估计
- 至少2种估计方法结果一致（方向相同，量级相近）
- 反驳验证中ATE在安慰剂测试下接近0
- 反驳验证中ATE在子集测试下稳健（变化<20%）

```prompt
你是一个因果推理专家，擅长使用DoWhy进行因果效应估计。请为CausalCredit项目实现ATE估计和反驳验证。

【任务背景】
CausalCredit的核心是量化"贷款条件对违约率的因果效应"。需要使用DoWhy对3个干预变量（利率、贷款金额、贷款期限）分别估计平均处理效应（ATE），并通过反驳验证确保结果稳健。

【具体要求】
1. 实现`src/causal/estimate.py`：

```python
class CausalEffectEstimator:
    """因果效应估计器"""
    
    def __init__(self, causal_graph: CreditCausalGraph, data: pd.DataFrame):
        """初始化DoWhy CausalModel"""
    
    def estimate_ate(self, treatment: str, outcome: str, 
                     method: str = "backdoor") -> Dict:
        """估计ATE
        method选项：
        - "backdoor_regression": 后门调整+回归
        - "backdoor_ipw": 后门调整+逆概率加权
        - "backdoor_psm": 后门调整+倾向得分匹配
        
        返回：
        {
            "ate": float,           # ATE估计值
            "ate_stderr": float,    # 标准误
            "ate_ci": (float, float), # 95%置信区间
            "method": str,          # 估计方法
            "treatment": str,       # 干预变量
            "n_observations": int,  # 样本量
        }
        """
    
    def estimate_all_treatments(self, outcome: str = "default") -> pd.DataFrame:
        """对所有干预变量估计ATE，返回汇总表"""
    
    def refute_estimate(self, treatment: str, outcome: str, 
                        ate_estimate, refutation_methods: List[str] = None) -> Dict:
        """反驳验证
        refutation_methods:
        - "random_common_cause": 添加随机混淆变量
        - "placebo_treatment_refuter": 安慰剂干预（随机替换treatment）
        - "data_subset_refuter": 数据子集验证
        - "add_unobserved_confounding": 添加未观测混淆
        
        返回：
        {
            method_name: {
                "refuted_estimate": float,
                "original_estimate": float,
                "p_value": float,
                "is_robust": bool  # p>0.05且变化<20%为稳健
            }
        }
        """
    
    def comprehensive_analysis(self, treatment: str, outcome: str = "default") -> Dict:
        """综合分析：多种方法估计+反驳验证，返回完整报告"""
```

2. 对3个干预变量的分析要求：

**interest_rate → default**：
- 预期ATE方向：利率↑ → 违约率↑（正相关）
- 混淆变量：income, age, employment_years, education_level, ext_score
- 估计方法：regression + IPW + PSM
- 反驳：4种方法全部执行

**credit_amount → default**：
- 预期ATE方向：贷款金额↑ → 违约率↑（但可能非线性）
- 混淆变量：income, age, employment_years, education_level, ext_score
- 估计方法：regression + IPW
- 反驳：3种方法（排除add_unobserved_confounding，因为效应可能较复杂）

**credit_term → default**：
- 预期ATE方向：期限↑ → 违约率↑（长期贷款风险更高）
- 混淆变量：income, age, credit_amount, ext_score
- 估计方法：regression + IPW
- 反驳：3种方法

3. 特殊处理：
- 连续干预变量：DoWhy需要指定treatment为连续变量时使用线性回归方法
- 如果ATE不显著（p>0.05），记录并分析原因
- 如果反驳验证不通过，分析可能的未观测混淆

4. 报告内容：
- 每个干预变量的ATE估计结果表
- 反驳验证结果表
- 因果效应方向与预期是否一致
- 稳健性评估

【输出规格】
- 完整的estimate.py源代码
- 因果效应估计notebook
- 因果效应报告markdown

【技术约束】
- 使用dowhy CausalModel API
- 连续treatment使用linear_regression估计方法
- PSM使用caliper=0.2*std(logit(propensity))
- 反驳验证每种方法重复100次取平均
- 所有随机种子固定
```

---

#### 任务 W3-T2：EconML异质处理效应（CATE）估计

**负责人**：A（因果推理工程师）  
**预计工时**：2天

**具体要求**：
- 使用EconML的CausalForestDML估计CATE
- 对interest_rate→default的CATE进行深入分析
- 识别CATE异质性：哪些子群的因果效应最大/最小
- 特别关注薄信用人群的CATE（核心创新点）
- CATE特征重要性：哪些特征最能解释CATE的异质性
- 可视化CATE分布和子群差异

**交付物**：
- `src/causal/cate.py`（CATE估计模块）
- `notebooks/06_cate_analysis.ipynb`
- `docs/cate_report.md`

**验收标准**：
- CATE估计成功，CATE分布有显著异质性（标准差>ATE的50%）
- 薄信用人群的CATE与非薄信用人群有显著差异
- CATE特征重要性Top-5输出
- CATE可视化完成

```prompt
你是一个因果推理专家，擅长使用EconML进行异质处理效应（CATE）估计。请为CausalCredit项目实现CATE分析。

【任务背景】
CausalCredit的核心创新之一是异质处理效应（CATE）——量化"对不同类型客户，同一干预的差异化效果"。例如，降低利率对薄信用人群的违约降低效果可能远大于对优质客户。EconML的CausalForestDML是估计CATE的SOTA方法。

【具体要求】
1. 实现`src/causal/cate.py`：

```python
class CATEEstimator:
    """异质处理效应估计器"""
    
    def __init__(self, config: dict):
        pass
    
    def fit_causal_forest(self, Y: np.ndarray, T: np.ndarray, 
                          X: np.ndarray, W: np.ndarray,
                          feature_names: List[str]) -> CausalForestDML:
        """拟合CausalForestDML
        Y: 结果变量（default）
        T: 干预变量（interest_rate）
        X: 效应修饰变量（heterogeneity features，用于发现CATE异质性）
        W: 混淆变量（其他控制变量）
        
        超参数：
        - n_estimators: 1000
        - max_depth: 8
        - min_samples_leaf: 50
        - random_state: 42
        """
    
    def estimate_cate(self, model: CausalForestDML, 
                      X: np.ndarray) -> np.ndarray:
        """估计每个样本的CATE值"""
    
    def cate_subgroup_analysis(self, cate_values: np.ndarray, 
                                X: pd.DataFrame,
                                subgroup_defs: Dict[str, np.ndarray]) -> pd.DataFrame:
        """子群CATE分析
        subgroup_defs定义子群，如：
        - "thin_credit": X['thin_credit_flag'] == 1
        - "low_income": X['income'] < X['income'].quantile(0.25)
        - "young": X['age'] < 30
        - "high_ext_score": X['ext_score'] > X['ext_score'].quantile(0.75)
        
        返回每组的：平均CATE、CATE标准差、样本量
        """
    
    def cate_feature_importance(self, model: CausalForestDML,
                                feature_names: List[str]) -> pd.DataFrame:
        """CATE特征重要性：哪些特征最能解释CATE的异质性"""
    
    def cate_tree_interpretation(self, model: CausalForestDML,
                                  feature_names: List[str],
                                  max_depth: int = 3) -> Dict:
        """从因果森林中提取决策规则，解释CATE异质性"""
    
    def visualize_cate(self, cate_values: np.ndarray, X: pd.DataFrame,
                       output_dir: str):
        """可视化CATE：
        1. CATE分布直方图
        2. CATE vs 关键特征的散点图
        3. 子群CATE对比箱线图
        4. CATE特征重要性条形图
        """
```

2. CATE分析重点：

**主要干预变量：interest_rate → default**
- X（效应修饰变量）：income, age, employment_years, education_level, ext_score, thin_credit_flag, debt_to_income
- W（混淆变量）：credit_amount, credit_term, gender, family_status, region_rating

**子群定义**：
- 薄信用人群（thin_credit_flag=1）
- 低收入人群（income < 25th percentile）
- 年轻人群（age < 30）
- 高外部评分人群（ext_score > 75th percentile）
- 高负债收入比人群（debt_to_income > 75th percentile）

3. 关键分析问题：
- 薄信用人群的CATE是否显著高于非薄信用人群？（降低利率对薄信用人群效果更大？）
- CATE分布是否呈现双峰？（暗示存在两个不同的因果机制子群）
- 哪些特征最能解释CATE异质性？

4. 统计检验：
- 子群CATE差异的t检验
- CATE异质性的整体检验（CATE标准差是否显著大于0）

【输出规格】
- 完整的cate.py源代码
- CATE分析notebook（含所有可视化）
- CATE报告markdown

【技术约束】
- 使用econml.dml.CausalForestDML
- 模型训练可能需要较长时间（30万数据），考虑先用10万子集调试
- CATE估计使用model.effect(X)方法
- 特征重要性使用model.feature_importances_
- 可视化使用matplotlib+seaborn
```

---

#### 任务 W3-T3：Week3 Go/No-Go检查点评估

**负责人**：A+B+C全员  
**预计工时**：0.5天

**具体要求**：
- 汇总Week1-3所有交付物
- 评估因果效应是否显著且稳健
- 评估CATE异质性是否足够
- 评估预测基座性能是否达标
- 决定Go（继续Week4-6）或No-Go（调整方案）

**Go/No-Go标准**：

| 指标 | Go标准 | No-Go触发 |
|------|--------|----------|
| LightGBM AUC | ≥0.76 | <0.72 |
| ATE方向 | 与预期一致 | 与预期相反且无法解释 |
| ATE反驳验证 | ≥2种方法稳健 | 全部反驳失败 |
| CATE异质性 | 标准差>ATE的50% | CATE几乎无变异 |
| 薄信用CATE差异 | 显著（p<0.05） | 无显著差异 |

**交付物**：
- `docs/week3_gonogo_report.md`

**验收标准**：
- 明确的Go/No-Go决策
- 如果No-Go，有具体的调整方案

```prompt
你是一个项目管理专家。请为CausalCredit项目执行Week3 Go/No-Go检查点评估。

【任务背景】
CausalCredit项目6周计划中，Week3结束是关键检查点。需要评估因果推理核心是否成立，决定是否继续投入Week4-6的开发。

【具体要求】
1. 创建评估报告模板`docs/week3_gonogo_report.md`，包含：

**1. 交付物完成度检查**
| 交付物 | 状态 | 备注 |
|--------|------|------|
| 数据加载与校验 | ✅/❌ | |
| EDA报告 | ✅/❌ | |
| 因果图 | ✅/❌ | |
| 特征工程 | ✅/❌ | |
| LightGBM基座 | ✅/❌ | |
| ATE估计 | ✅/❌ | |
| CATE估计 | ✅/❌ | |

**2. 关键指标评估**
| 指标 | 目标值 | 实际值 | 是否达标 |
|------|--------|--------|---------|
| LightGBM 5折CV AUC | ≥0.76 | | |
| 薄信用子群AUC | ≥0.70 | | |
| interest_rate ATE方向 | 正相关 | | |
| ATE反驳验证稳健率 | ≥2/4 | | |
| CATE标准差/ATE | >0.5 | | |
| 薄信用CATE差异p值 | <0.05 | | |

**3. 风险评估**
- 列出当前Top-3风险
- 每个风险的缓解措施

**4. Go/No-Go决策**
- 决策：GO / CONDITIONAL GO / NO-GO
- 理由
- 如果CONDITIONAL GO，列出必须满足的条件和截止日期
- 如果NO-GO，列出调整方案选项

**5. Week4-6计划调整（如需要）**
- 基于Week1-3的实际进展，是否需要调整后续计划
- 资源重新分配建议

【输出规格】
- 完整的Go/No-Go报告markdown模板
- 每个评估项有明确的填写指南

【技术约束】
- 评估必须基于实际数据，不能主观臆断
- No-Go决策需要提供至少2个替代方案
```

---

### Week 4：可解释性与反事实推理

**阶段目标**：实现SHAP可解释性分析和反事实决策建议生成

---

#### 任务 W4-T1：SHAP可解释性分析

**负责人**：B（ML工程师）  
**预计工时**：1.5天

**具体要求**：
- 使用SHAP TreeExplainer对LightGBM模型进行解释
- 全局解释：SHAP特征重要性排序、SHAP依赖图、SHAP交互图
- 局部解释：单样本SHAP瀑布图、力图
- 因果特征vs非因果特征的SHAP贡献对比
- 薄信用人群的SHAP特征分布差异
- 生成可解释性报告

**交付物**：
- `src/explain/shap_explain.py`
- `notebooks/07_shap_analysis.ipynb`

**验收标准**：
- 全局SHAP重要性Top-20输出
- 至少5个关键特征的SHAP依赖图
- 薄信用vs非薄信用的SHAP差异分析完成

```prompt
你是一个可解释AI专家。请为CausalCredit项目实现SHAP可解释性分析模块。

【任务背景】
CausalCredit需要满足HKMA对AI可解释性的监管要求。SHAP是模型可解释性的标准工具，需要提供全局解释（哪些特征最重要）和局部解释（为什么这个客户被评了这个分）。

【具体要求】
1. 实现`src/explain/shap_explain.py`：

```python
class SHAPExplainer:
    """SHAP可解释性分析器"""
    
    def __init__(self, model, feature_names: List[str]):
        """初始化TreeExplainer"""
    
    def compute_shap_values(self, X: pd.DataFrame, check_additivity: bool = True) -> np.ndarray:
        """计算SHAP值"""
    
    def global_importance(self, shap_values: np.ndarray, 
                          feature_names: List[str]) -> pd.DataFrame:
        """全局特征重要性：按|SHAP|均值排序"""
    
    def dependence_plot(self, shap_values: np.ndarray, X: pd.DataFrame,
                        feature: str, interaction_feature: str = None,
                        output_path: str = None):
        """SHAP依赖图：特征值 vs SHAP值"""
    
    def interaction_plot(self, shap_values: np.ndarray, X: pd.DataFrame,
                         feature1: str, feature2: str,
                         output_path: str = None):
        """SHAP交互图：两个特征的交互效应"""
    
    def local_explanation(self, shap_values: np.ndarray, X: pd.DataFrame,
                          idx: int, output_path: str = None):
        """局部解释：单样本瀑布图"""
    
    def causal_vs_noncausal_contribution(self, shap_values: np.ndarray,
                                          feature_names: List[str],
                                          causal_features: List[str]) -> Dict:
        """因果特征vs非因果特征的SHAP贡献对比：
        - 因果特征平均|SHAP|占比
        - 非因果特征平均|SHAP|占比
        - 每个因果特征的单独贡献
        """
    
    def subgroup_shap_comparison(self, shap_values: np.ndarray, 
                                  X: pd.DataFrame,
                                  subgroup_col: str) -> pd.DataFrame:
        """子群SHAP对比：薄信用vs非薄信用的特征贡献差异"""
    
    def generate_evidence_chain(self, shap_values: np.ndarray, 
                                 X: pd.DataFrame,
                                 idx: int, top_k: int = 5) -> List[Dict]:
        """生成证据链：Top-K特征的SHAP贡献
        返回：[{"feature": str, "value": float, "shap": float, "direction": "增加/降低风险", "contribution_pct": float}]
        """
```

2. 分析要求：
   - 全局SHAP重要性Top-20特征排序
   - 关键特征SHAP依赖图：interest_rate, income, age, ext_score, debt_to_income, thin_credit_flag
   - 交互效应：interest_rate × thin_credit_flag（利率对薄信用人群的差异化影响）
   - 薄信用vs非薄信用的SHAP特征分布差异
   - 因果特征（interest_rate, debt_to_income, credit_utilization, payment_discipline）vs非因果特征的SHAP贡献占比

3. 证据链生成格式：
   ```
   客户#12345的信用评分分析：
   1. 外部评分(ext_score=0.65)：降低风险贡献15%
   2. 负债收入比(debt_to_income=0.35)：增加风险贡献12%
   3. 贷款利率(interest_rate=12%)：增加风险贡献10%
   4. 年龄(age=25)：增加风险贡献8%
   5. 薄信用标志(thin_credit=1)：增加风险贡献7%
   → 综合违约概率：18%
   ```

【输出规格】
- 完整的shap_explain.py源代码
- SHAP分析notebook
- 关键图表保存到docs/figures/shap/

【技术约束】
- 使用shap.TreeExplainer（LightGBM专用，速度快）
- 大数据集（>10万）使用shap.sample采样计算SHAP值
- 图表使用shap内置绘图+matplotlib自定义
- 证据链文本支持中英文
```

---

#### 任务 W4-T2：反事实推理与决策建议引擎

**负责人**：A（因果推理工程师）  
**预计工时**：2.5天

**具体要求**：
- 实现反事实推理：给定客户特征和干预方案，预测反事实违约概率
- 基于CATE结果生成个性化决策建议
- 实现3种反事实场景：①降低利率 ②缩短贷款期限 ③降低贷款金额
- 生成决策建议报告：推荐最优干预方案+预期效果
- 实现反事实解释："如果利率降低2%，您的违约概率将从15%降至8%"

**交付物**：
- `src/explain/counterfactual.py`（反事实推理模块）
- `src/explain/decision.py`（决策建议引擎）
- `src/explain/evidence.py`（证据链生成模块）
- `notebooks/08_counterfactual_analysis.ipynb`

**验收标准**：
- 反事实推理可生成3种场景的预测
- 决策建议包含最优干预方案和预期效果
- 反事实解释文本可读且业务合理
- 薄信用人群的反事实建议与非薄信用有显著差异

```prompt
你是一个因果推理和决策科学专家。请为CausalCredit项目实现反事实推理与决策建议引擎。

【任务背景】
CausalCredit的核心创新是"从预测到决策的范式升级"。反事实推理模块需要回答："如果调整贷款条件X，违约概率将变化Y%"，并生成个性化决策建议。这是传统信用评分系统完全不具备的能力。

【具体要求】
1. 实现`src/explain/counterfactual.py`：

```python
class CounterfactualReasoner:
    """反事实推理器"""
    
    def __init__(self, lgbm_model, cate_model, shap_explainer, 
                 feature_names: List[str], calibrator=None):
        """初始化，需要预测模型、CATE模型、SHAP解释器"""
    
    def predict_counterfactual(self, features: Dict[str, float],
                                interventions: Dict[str, float]) -> Dict:
        """预测反事实违约概率
        
        features: 基线特征 {"income": 50000, "age": 25, ...}
        interventions: 干预方案 {"interest_rate": 0.08}  # 将利率设为8%
        
        方法：
        1. 用基线特征预测基线违约概率 P(Y|X)
        2. 用CATE模型估计干预的因果效应 CATE(X, T→T')
        3. 反事实概率 = P(Y|X) + CATE(X, T→T') * ΔT
        
        返回：
        {
            "baseline_probability": float,     # 基线违约概率
            "counterfactual_probability": float, # 反事实违约概率
            "probability_change": float,        # 概率变化
            "change_direction": "降低"/"增加",
            "intervention": Dict,               # 干预方案
            "confidence": float,                # 置信度（基于CATE的方差）
        }
        """
    
    def predict_multiple_scenarios(self, features: Dict[str, float],
                                    scenarios: List[Dict[str, float]]) -> List[Dict]:
        """预测多种反事实场景"""
    
    def generate_standard_scenarios(self, features: Dict[str, float]) -> List[Dict]:
        """生成3种标准反事实场景：
        1. 降低利率2个百分点
        2. 缩短贷款期限12个月
        3. 降低贷款金额20%
        
        每种场景包含干预方案和预期效果
        """
    
    def find_optimal_intervention(self, features: Dict[str, float],
                                   target_probability: float = None,
                                   budget_constraint: Dict = None) -> Dict:
        """寻找最优干预方案
        
        目标：在约束条件下最小化违约概率
        target_probability: 目标违约概率（如降到10%以下）
        budget_constraint: 预算约束（如利率不能低于X，期限不能超过Y）
        
        方法：网格搜索+因果效应排序
        """
```

2. 实现`src/explain/decision.py`：

```python
class DecisionAdvisor:
    """决策建议引擎"""
    
    def __init__(self, counterfactual_reasoner: CounterfactualReasoner):
        pass
    
    def generate_decision_report(self, features: Dict[str, float],
                                  include_shap: bool = True) -> Dict:
        """生成完整决策报告
        
        返回：
        {
            "applicant_profile": {...},           # 申请人画像
            "risk_assessment": {...},             # 风险评估
            "causal_analysis": {...},             # 因果分析
            "counterfactual_scenarios": [...],    # 反事实场景
            "optimal_intervention": {...},        # 最优干预
            "decision_suggestion": str,           # 决策建议文本
            "evidence_chain": [...],              # 证据链
        }
        """
    
    def generate_suggestion_text(self, report: Dict, language: str = "zh") -> str:
        """生成决策建议文本
        
        格式示例（中文）：
        "建议批准该客户的贷款申请，推荐方案：利率8.5%，期限24个月。
        理由：该客户为薄信用人群，降低利率2个百分点可将其违约概率从18%降至11%，
        降幅达7个百分点，远高于非薄信用人群的3个百分点降幅。
        关键风险因素：负债收入比偏高(35%)、工作年限较短(2年)。"
        """
    
    def compare_subgroup_effect(self, features: Dict[str, float],
                                 subgroup_col: str = "thin_credit_flag") -> Dict:
        """对比不同子群的因果效应差异
        
        特别展示：薄信用人群 vs 非薄信用人群
        同一干预对不同人群的差异化效果
        """
```

3. 实现`src/explain/evidence.py`：

```python
class EvidenceChainGenerator:
    """证据链生成器"""
    
    def generate_risk_evidence(self, shap_values: np.ndarray,
                                features: Dict, top_k: int = 5) -> List[Dict]:
        """生成风险证据链"""
    
    def generate_causal_evidence(self, ate_results: Dict,
                                  cate_value: float,
                                  subgroup: str = None) -> List[Dict]:
        """生成因果证据链"""
    
    def generate_counterfactual_evidence(self, cf_result: Dict) -> str:
        """生成反事实证据文本
        
        示例："如果将利率从12%降低至10%，您的违约概率预计将从18%降至11%，
        降幅7个百分点。此效果在薄信用人群中更为显著（平均降幅9个百分点）。"
        """
    
    def generate_full_evidence_report(self, risk_evidence, causal_evidence, 
                                      cf_evidence) -> str:
        """生成完整证据报告（Markdown格式）"""
```

4. 反事实推理的技术实现细节：

**方法1：CATE调整法（主要方法）**
```
P(Y=1|do(T=t'), X) ≈ P(Y=1|T=t, X) + CATE(X) × (t' - t)
```
- 用LightGBM预测基线概率P(Y=1|T=t, X)
- 用CausalForestDML估计CATE(X)
- 反事实概率 = 基线概率 + CATE × 干预变化量

**方法2：DoWhy反事实法（验证方法）**
- 使用DoWhy的counterfactual功能
- 与方法1结果交叉验证

**置信度估计**：
- 基于CausalForestDML的方差估计
- 置信度 = 1 - normalized_variance

5. 决策建议的业务规则：
- 如果反事实违约概率可降至10%以下 → 建议批准+推荐方案
- 如果反事实违约概率可降至15%以下 → 建议批准+需附加条件
- 如果反事实违约概率仍在15%以上 → 建议拒绝+改善建议
- 薄信用人群：阈值放宽5个百分点（鼓励普惠金融）

【输出规格】
- 3个Python模块的完整源代码
- 反事实分析notebook
- 至少3个典型客户的决策建议示例

【技术约束】
- 反事实推理必须基于因果模型，不能简单用预测模型替换特征值
- CATE调整法需要确保概率在[0,1]范围内（clip处理）
- 决策建议文本需业务合理，避免过于技术化的表述
- 证据链需可追溯（每个结论有数据支撑）
```

---

### Week 5：系统集成与API开发

**阶段目标**：完成FastAPI后端和Streamlit前端，实现端到端系统集成

---

#### 任务 W5-T1：FastAPI后端开发

**负责人**：C（全栈开发）  
**预计工时**：2天

**具体要求**：
- 实现2.4定义的所有API端点
- 请求/响应模型使用Pydantic V2
- 加载训练好的模型（LightGBM + CATE模型 + 校准器）
- 实现完整的信用评分流水线：特征预处理→预测→因果分析→反事实→解释
- 错误处理、日志记录、请求验证
- API文档自动生成（Swagger/OpenAPI）

**交付物**：
- `src/api/app.py`（FastAPI应用主入口）
- `src/api/routes.py`（路由定义）
- `src/api/schemas.py`（Pydantic数据模型）
- `src/api/services.py`（业务逻辑服务层）
- `src/api/dependencies.py`（依赖注入：模型加载）

**验收标准**：
- 所有5个API端点可正常调用
- Swagger文档自动生成
- 单次评分请求延迟<500ms
- 错误请求返回合理的HTTP状态码和错误信息

```prompt
你是一个后端开发专家，擅长FastAPI。请为CausalCredit项目实现完整的FastAPI后端。

【任务背景】
CausalCredit需要提供REST API服务，供Streamlit前端和外部系统调用。API需要封装完整的信用评分流水线：特征预处理→LightGBM预测→因果效应分析→反事实推理→SHAP解释→决策建议。

【具体要求】
1. 实现`src/api/schemas.py`，Pydantic V2数据模型：

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class CreditRequest(BaseModel):
    """信用评分请求"""
    applicant_id: Optional[str] = Field(None, description="申请人ID")
    features: Dict[str, Any] = Field(..., description="申请特征键值对")
    include_counterfactual: bool = Field(True, description="是否包含反事实分析")
    include_explanation: bool = Field(True, description="是否包含SHAP解释")

class CounterfactualRequest(BaseModel):
    """反事实情景请求"""
    applicant_id: Optional[str] = Field(None, description="申请人ID")
    features: Dict[str, Any] = Field(..., description="基线特征")
    interventions: Dict[str, Any] = Field(..., description="干预方案，如{'interest_rate': 0.08}")

class ExplainRequest(BaseModel):
    """SHAP解释请求"""
    applicant_id: Optional[str] = Field(None, description="申请人ID")
    features: Dict[str, Any] = Field(..., description="申请特征")
    top_k: int = Field(5, description="返回Top-K特征解释")

class CausalEffectRequest(BaseModel):
    """因果效应查询请求"""
    treatment: str = Field(..., description="干预变量名")
    outcome: str = Field(default="default", description="结果变量名")
    subgroup: Optional[Dict[str, Any]] = Field(None, description="子群筛选条件")

class CreditResponse(BaseModel):
    """信用评分响应"""
    score: int = Field(..., description="信用评分300-850")
    default_probability: float = Field(..., description="违约概率0-1")
    risk_grade: str = Field(..., description="风险等级A/B/C/D/E")
    causal_effect: Optional[Dict] = Field(None, description="因果效应摘要")
    counterfactual: Optional[List[Dict]] = Field(None, description="反事实情景")
    explanation: Optional[Dict] = Field(None, description="SHAP解释")
    decision_suggestion: Optional[str] = Field(None, description="决策建议")

class CounterfactualResponse(BaseModel):
    """反事实情景响应"""
    baseline_probability: float
    counterfactual_probability: float
    probability_change: float
    intervention_details: Dict
    confidence: float

class ExplainResponse(BaseModel):
    """SHAP解释响应"""
    top_features: List[Dict]
    evidence_chain: List[Dict]

class CausalEffectResponse(BaseModel):
    """因果效应响应"""
    ate: float
    ate_ci: tuple
    cate_subgroup: Optional[Dict]
    refutation_results: Optional[Dict]

class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str]
```

2. 实现`src/api/dependencies.py`，模型加载依赖：

```python
from functools import lru_cache

class ModelRegistry:
    """模型注册表，单例模式"""
    def __init__(self):
        self.lgbm_model = None
        self.cate_model = None
        self.calibrator = None
        self.shap_explainer = None
        self.feature_encoder = None
        self.causal_graph = None
        self.counterfactual_reasoner = None
        self.decision_advisor = None

@lru_cache()
def get_model_registry() -> ModelRegistry:
    """获取模型注册表（启动时加载）"""
    registry = ModelRegistry()
    # 加载所有模型和组件
    ...
    return registry
```

3. 实现`src/api/services.py`，业务逻辑：

```python
class CreditScoringService:
    """信用评分服务"""
    
    def __init__(self, registry: ModelRegistry):
        pass
    
    def score(self, request: CreditRequest) -> CreditResponse:
        """完整评分流水线"""
    
    def counterfactual(self, request: CounterfactualRequest) -> CounterfactualResponse:
        """反事实分析"""
    
    def explain(self, request: ExplainRequest) -> ExplainResponse:
        """SHAP解释"""
    
    def causal_effect(self, request: CausalEffectRequest) -> CausalEffectResponse:
        """因果效应查询"""
```

4. 实现`src/api/routes.py`，路由定义：

```python
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/api/v1", tags=["CausalCredit"])

@router.post("/score", response_model=CreditResponse)
async def score_credit(request: CreditRequest, ...):
    """信用评分+因果分析"""

@router.post("/counterfactual", response_model=CounterfactualResponse)
async def counterfactual_analysis(request: CounterfactualRequest, ...):
    """反事实情景模拟"""

@router.post("/explain", response_model=ExplainResponse)
async def explain_score(request: ExplainRequest, ...):
    """SHAP可解释性分析"""

@router.post("/causal-effect", response_model=CausalEffectResponse)
async def query_causal_effect(request: CausalEffectRequest, ...):
    """因果效应查询"""

@router.get("/health")
async def health_check():
    """健康检查"""
```

5. 实现`src/api/app.py`，应用主入口：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CausalCredit API",
    description="因果推理增强信用评分系统API",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

@app.on_event("startup")
async def startup():
    """启动时加载模型"""
    get_model_registry()

app.include_router(router)
```

6. 性能要求：
- 单次/score请求延迟<500ms（不含SHAP计算）
- 单次/score请求延迟<2s（含SHAP计算）
- 并发支持：10 QPS

7. 错误处理：
- 特征缺失：返回422 + 缺失特征列表
- 特征值异常：返回422 + 异常值提示
- 模型加载失败：返回503 + 错误信息
- 反事实干预无效：返回400 + 原因

【输出规格】
- 5个Python模块的完整源代码
- API文档截图（Swagger UI）

【技术约束】
- FastAPI 0.104+
- Pydantic V2（使用model_config而非Config类）
- 模型加载使用lru_cache，避免重复加载
- 日志使用Python logging模块
- 异步端点使用async def
```

---

#### 任务 W5-T2：Streamlit交互式Demo开发

**负责人**：C（全栈开发）  
**预计工时**：2天

**具体要求**：
- 实现4个页面：①评分仪表盘 ②因果效应可视化 ③反事实情景模拟 ④决策建议面板
- 评分仪表盘：输入客户特征→显示信用评分+风险等级+违约概率
- 因果效应可视化：CATE分布图、子群CATE对比图、因果DAG
- 反事实情景模拟：滑块调整利率/期限/金额→实时显示反事实违约概率
- 决策建议面板：综合评分+因果分析+反事实→决策建议文本
- 支持预设客户画像（薄信用年轻人、中等收入中年人、优质客户等）

**交付物**：
- `src/frontend/app.py`（Streamlit主入口）
- `src/frontend/pages/score_dashboard.py`
- `src/frontend/pages/causal_visualization.py`
- `src/frontend/pages/counterfactual_simulator.py`
- `src/frontend/pages/decision_panel.py`
- `src/frontend/components/`（可复用组件）

**验收标准**：
- 4个页面均可正常访问和交互
- 反事实模拟器滑块实时响应
- 预设客户画像可一键加载
- 界面美观，适合比赛演示

```prompt
你是一个前端开发专家，擅长Streamlit。请为CausalCredit项目实现交互式Demo。

【任务背景】
CausalCredit需要一个交互式Demo用于比赛演示。评委需要能直观体验"因果推理增强信用评分"的差异化价值：输入客户信息→看到评分→调整贷款条件→看到反事实效果→获得决策建议。

【具体要求】
1. 实现`src/frontend/app.py`，主入口：

```python
import streamlit as st

st.set_page_config(
    page_title="CausalCredit — 因果推理增强信用评分",
    page_icon="🎯",
    layout="wide",
)

# 侧边栏导航
page = st.sidebar.selectbox("选择页面", [
    "📊 评分仪表盘",
    "🔬 因果效应可视化",
    "🔄 反事实情景模拟",
    "💡 决策建议面板",
])

# 预设客户画像选择
preset = st.sidebar.selectbox("预设客户画像", [
    "自定义",
    "薄信用年轻人（25岁，无信贷历史）",
    "中等收入中年人（40岁，有房贷）",
    "优质客户（35岁，高收入高评分）",
    "高风险申请人（30岁，低收入高负债）",
])
```

2. 页面1：评分仪表盘（score_dashboard.py）
- 左栏：客户特征输入表单
  - 数值特征：收入、年龄、工作年限、贷款金额、贷款期限（滑块）
  - 分类特征：教育水平、职业类型、住房类型（下拉框）
  - 薄信用标志（复选框）
- 右栏：评分结果
  - 信用评分仪表盘（300-850，用gauge chart）
  - 风险等级（A/B/C/D/E，颜色编码）
  - 违约概率（进度条）
  - Top-5风险因素（水平条形图）
  - SHAP瀑布图（简化版）

3. 页面2：因果效应可视化（causal_visualization.py）
- 因果DAG图（交互式，点击节点高亮路径）
- ATE结果表（3个干预变量的ATE及置信区间）
- CATE分布直方图
- 子群CATE对比箱线图（薄信用vs非薄信用等）
- CATE特征重要性条形图

4. 页面3：反事实情景模拟（counterfactual_simulator.py）
- 基线信息显示：当前客户特征和违约概率
- 3个干预滑块：
  - 利率调整：-5% ~ +5%（步长0.5%）
  - 期限调整：-24月 ~ +24月（步长6月）
  - 金额调整：-50% ~ +50%（步长10%）
- 实时反事实违约概率显示
- 基线vs反事实对比条形图
- 反事实解释文本："如果利率降低2%，违约概率将从X%降至Y%"

5. 页面4：决策建议面板（decision_panel.py）
- 综合评分卡片（评分+风险等级+违约概率）
- 因果分析摘要（ATE方向+CATE值）
- 最优干预方案推荐
- 决策建议文本（批准/拒绝/附条件批准）
- 证据链（风险因素+因果效应+反事实分析）
- 导出报告按钮（Markdown格式）

6. 预设客户画像数据：
```python
PRESETS = {
    "薄信用年轻人": {
        "age": 25, "income": 180000, "employment_years": 2,
        "credit_amount": 100000, "interest_rate": 0.12,
        "education_level": "Bachelor", "thin_credit_flag": 1,
        "debt_to_income": 0.25, "ext_score": 0.45,
    },
    "中等收入中年人": {
        "age": 40, "income": 500000, "employment_years": 15,
        "credit_amount": 300000, "interest_rate": 0.06,
        "education_level": "Master", "thin_credit_flag": 0,
        "debt_to_income": 0.35, "ext_score": 0.72,
    },
    "优质客户": {
        "age": 35, "income": 800000, "employment_years": 10,
        "credit_amount": 500000, "interest_rate": 0.04,
        "education_level": "PhD", "thin_credit_flag": 0,
        "debt_to_income": 0.15, "ext_score": 0.88,
    },
    "高风险申请人": {
        "age": 30, "income": 150000, "employment_years": 3,
        "credit_amount": 200000, "interest_rate": 0.15,
        "education_level": "Secondary", "thin_credit_flag": 1,
        "debt_to_income": 0.55, "ext_score": 0.30,
    },
}
```

7. UI设计要求：
- 配色方案：深蓝+金色（银行风格）
- 字体：标题用大号加粗，正文用常规
- 图表使用plotly（交互式）
- 仪表盘使用streamlit-echarts或自定义CSS
- 响应式布局（wide mode）

【输出规格】
- 完整的Streamlit应用代码
- 每个页面独立Python文件
- 可复用组件放在components/目录

【技术约束】
- Streamlit 1.29+
- 图表使用plotly（非matplotlib，因为需要交互）
- 调用FastAPI后端API（通过requests库）
- 如果API未启动，降级为直接调用模型（本地模式）
- 页面间状态通过st.session_state传递
```

---

#### 任务 W5-T3：端到端集成测试

**负责人**：C（全栈开发）  
**预计工时**：1天

**具体要求**：
- 编写端到端测试：从API请求到响应的完整流水线
- 测试所有API端点的正常和异常场景
- 测试Streamlit页面的关键交互
- 性能测试：单次评分延迟、并发能力
- 修复集成问题

**交付物**：
- `tests/test_api.py`（API测试）
- `tests/test_integration.py`（集成测试）
- `tests/test_performance.py`（性能测试）

**验收标准**：
- API测试覆盖率≥80%
- 所有端点正常和异常场景测试通过
- 单次评分延迟<500ms（不含SHAP）

```prompt
你是一个测试工程师。请为CausalCredit项目编写端到端集成测试。

【任务背景】
CausalCredit系统包含数据层、算法层、应用层三层架构，需要确保端到端流水线正常工作，API响应正确，性能达标。

【具体要求】
1. 实现`tests/test_api.py`，API端点测试：

```python
import pytest
from fastapi.testclient import TestClient

class TestScoreEndpoint:
    """测试 /api/v1/score 端点"""
    
    def test_score_normal_request(self, client):
        """正常评分请求"""
    
    def test_score_with_counterfactual(self, client):
        """包含反事实分析的评分请求"""
    
    def test_score_with_explanation(self, client):
        """包含SHAP解释的评分请求"""
    
    def test_score_missing_required_features(self, client):
        """缺少必要特征"""
    
    def test_score_invalid_feature_values(self, client):
        """特征值异常"""
    
    def test_score_thin_credit_applicant(self, client):
        """薄信用申请人评分"""

class TestCounterfactualEndpoint:
    """测试 /api/v1/counterfactual 端点"""
    
    def test_counterfactual_rate_reduction(self, client):
        """利率降低的反事实分析"""
    
    def test_counterfactual_term_reduction(self, client):
        """期限缩短的反事实分析"""
    
    def test_counterfactual_invalid_intervention(self, client):
        """无效干预变量"""

class TestExplainEndpoint:
    """测试 /api/v1/explain 端点"""
    
    def test_explain_top5(self, client):
        """Top-5特征解释"""
    
    def test_explain_top10(self, client):
        """Top-10特征解释"""

class TestCausalEffectEndpoint:
    """测试 /api/v1/causal-effect 端点"""
    
    def test_causal_effect_interest_rate(self, client):
        """利率因果效应查询"""
    
    def test_causal_effect_invalid_treatment(self, client):
        """无效干预变量"""

class TestHealthEndpoint:
    """测试 /api/v1/health 端点"""
    
    def test_health_check(self, client):
        """健康检查"""
```

2. 实现`tests/test_integration.py`，端到端集成测试：

```python
class TestEndToEnd:
    """端到端集成测试"""
    
    def test_full_scoring_pipeline(self):
        """完整评分流水线：特征输入→预测→因果分析→反事实→解释→建议"""
    
    def test_thin_credit_vs_normal(self):
        """薄信用vs非薄信用的差异化分析"""
    
    def test_counterfactual_consistency(self):
        """反事实推理一致性：相同输入相同输出"""
    
    def test_causal_effect_direction(self):
        """因果效应方向验证：利率↑→违约率↑"""
    
    def test_score_range(self):
        """评分范围验证：300-850"""
    
    def test_probability_range(self):
        """概率范围验证：0-1"""
```

3. 实现`tests/test_performance.py`，性能测试：

```python
class TestPerformance:
    """性能测试"""
    
    def test_single_score_latency(self):
        """单次评分延迟<500ms（不含SHAP）"""
    
    def test_score_with_shap_latency(self):
        """含SHAP的评分延迟<2s"""
    
    def test_counterfactual_latency(self):
        """反事实分析延迟<1s"""
    
    def test_concurrent_requests(self):
        """10并发请求全部成功"""
```

4. 测试数据：
- 创建`tests/fixtures/test_data.py`，包含5个预设测试客户
- 每个测试客户有完整的特征和预期评分范围

5. pytest配置：
- 创建`conftest.py`，定义共享fixture（TestClient、测试数据）
- 使用`@pytest.fixture(scope="session")`加载模型（只加载一次）

【输出规格】
- 3个测试文件的完整源代码
- conftest.py
- 测试数据fixture

【技术约束】
- 使用pytest + httpx（FastAPI TestClient）
- 测试不依赖外部服务
- 性能测试使用time.perf_counter()
- 并发测试使用concurrent.futures.ThreadPoolExecutor
```

---

### Week 6：文档完善与比赛准备

**阶段目标**：完成技术白皮书、比赛演示材料、最终调优

---

#### 任务 W6-T1：技术白皮书撰写

**负责人**：A（因果推理工程师）  
**预计工时**：2天

**具体要求**：
- 撰写完整的技术白皮书（15-20页），包含：
  1. 执行摘要
  2. 问题陈述与动机
  3. 相关工作综述
  4. 方法论（因果图+ATE+CATE+反事实推理）
  5. 实验设计与结果
  6. 系统架构与实现
  7. 业务价值分析
  8. 局限性与未来工作
  9. 结论
- 强调3大创新点的技术深度和业务价值
- 包含关键图表：因果DAG、CATE分布、反事实效果对比、系统架构图

**交付物**：
- `docs/CausalCredit_技术白皮书.md`
- `docs/figures/`（白皮书用图）

**验收标准**：
- 白皮书≥15页（A4格式）
- 3大创新点各有≥1页技术深度分析
- 实验结果包含量化指标
- 图表≥8张

```prompt
你是一个技术写作专家，擅长撰写AI金融领域的技术白皮书。请为CausalCredit项目撰写完整的技术白皮书。

【任务背景】
CausalCredit参加中银香港创新先驱大赛2026，需要提交技术白皮书。白皮书需面向技术评委和管理层评委，既展示技术深度，又突出业务价值。

【具体要求】
撰写`docs/CausalCredit_技术白皮书.md`，结构如下：

**1. 执行摘要（1页）**
- 一句话定位
- 3大创新点
- 关键成果数据
- 业务价值量化

**2. 问题陈述与动机（2页）**
- 传统信用评分的3大局限（只预测不决策、薄信用覆盖不足、不可解释）
- 香港市场痛点数据（新来港人士20万+/年、中小企融资缺口HK$1,200亿）
- 因果推理的范式升级价值

**3. 相关工作综述（1.5页）**
- 传统信用评分（FICO/环联/芝麻信用）
- 因果推理在金融中的应用（2024-2026前沿论文）
- 本方案的差异化定位

**4. 方法论（4页）**
- 4.1 因果图构建（DAG + 领域知识 + 数据验证）
- 4.2 平均处理效应估计（ATE：DoWhy + 3种方法 + 反驳验证）
- 4.3 异质处理效应估计（CATE：EconML CausalForestDML + 子群分析）
- 4.4 反事实推理与决策建议（CATE调整法 + 最优干预搜索）
- 4.5 因果增强评分公式：Score = f(P_pred, CATE, CF)

**5. 实验设计与结果（3页）**
- 5.1 数据集描述（Home Credit + Lending Club）
- 5.2 预测基座性能（LightGBM AUC/KS/Brier Score）
- 5.3 因果效应估计结果（ATE表 + 反驳验证表）
- 5.4 CATE异质性分析（分布图 + 子群对比）
- 5.5 反事实推理效果（3种场景对比）
- 5.6 薄信用人群专项分析

**6. 系统架构与实现（2页）**
- 三层架构图（数据层/算法层/应用层）
- API设计
- Streamlit Demo截图
- 部署方案

**7. 业务价值分析（2页）**
- 7.1 覆盖薄信用人群（拓展信贷客群）
- 7.2 优化贷款定价（基于因果效应而非仅风险预测）
- 7.3 满足HKMA可解释性要求
- 7.4 大湾区跨境信贷评估场景
- 7.5 ROI估算

**8. 局限性与未来工作（1页）**
- 因果图假设的局限性
- 观测数据vs实验数据
- 未来：联邦因果推理、实时推理、多干预优化

**9. 结论（0.5页）**

**附录**
- A. 因果图完整定义
- B. 特征工程详情
- C. 超参数配置

写作风格：
- 专业但不晦涩，技术评委看深度，管理层评委看价值
- 每个创新点用"问题→方法→效果"三段式论述
- 量化数据优先（AUC提升X%，CATE差异Y倍）
- 图表用Markdown表格+引用图片路径

【输出规格】
- 完整的技术白皮书Markdown文件
- 白皮书引用的图表保存到docs/figures/

【技术约束】
- Markdown格式，支持后续转PDF
- 图表使用相对路径引用
- 中英文术语对照（首次出现时标注英文）
- 引用格式：[作者, 年份]
```

---

#### 任务 W6-T2：比赛演示材料准备

**负责人**：C（全栈开发）  
**预计工时**：1.5天

**具体要求**：
- 准备5分钟演示脚本（中英文）
- 准备演示用预设场景（3个典型客户）
- Streamlit Demo最终打磨（UI美化、加载优化、错误处理）
- 准备Q&A预案（10个预期问题+回答）
- 录制Demo演示视频（3分钟）

**交付物**：
- `docs/demo_script.md`（演示脚本）
- `docs/qa_preparation.md`（Q&A预案）
- Demo演示视频

**验收标准**：
- 演示脚本≤5分钟
- 3个预设场景覆盖核心功能
- Q&A预案覆盖10个问题
- Demo无bug可流畅演示

```prompt
你是一个比赛演示专家。请为CausalCredit项目准备比赛演示材料。

【任务背景】
中银香港创新先驱大赛2026的决赛演示环节，每队5分钟演示+3分钟Q&A。需要准备演示脚本、预设场景和Q&A预案。

【具体要求】
1. 撰写`docs/demo_script.md`，5分钟演示脚本：

**开场（30秒）**
- 一句话定位："CausalCredit — 不只预测风险，更是指导决策"
- 痛点引入："传统信用评分只告诉你谁会违约，但不告诉你如何降低违约"

**核心演示（3分钟）**
- 场景1（1分钟）：薄信用年轻人申请贷款
  - 输入特征→评分→"传统系统会拒绝"
  - 因果分析→"降低利率2%，违约概率从18%降至11%"
  - 决策建议→"附条件批准"
- 场景2（1分钟）：中等收入客户贷款定价
  - CATE分析→"该客户对利率敏感度低，可适当上浮"
  - 反事实模拟→"利率上浮1%，违约概率仅增加1.5%"
  - 决策建议→"批准+利率上浮1%优化收益"
- 场景3（1分钟）：因果效应可视化
  - CATE分布→"不同客户对同一干预的差异化响应"
  - 薄信用vs非薄信用CATE对比→"普惠金融的因果证据"

**技术亮点（1分钟）**
- 因果DAG展示
- ATE反驳验证结果
- CATE异质性发现

**收尾（30秒）**
- 3大创新点回顾
- 业务价值量化
- "从预测到决策的范式升级"

2. 撰写`docs/qa_preparation.md`，10个预期问题+回答：

Q1: 因果推断的可信度如何保证？（观测数据vs实验数据）
Q2: 因果图构建依赖假设，如何验证？
Q3: 薄信用人群的CATE估计是否可靠（样本量小）？
Q4: 与传统评分卡相比，性能提升多少？
Q5: 如何与银行现有系统集成？
Q6: 实时推理的延迟和成本？
Q7: 数据隐私和合规性如何保证？
Q8: 反事实推理的置信度如何量化？
Q9: 方案的可扩展性（其他信贷产品）？
Q10: 与FICO/环联等现有方案的差异化？

每个回答控制在30秒内，要点式。

3. 预设场景数据：
- 3个典型客户的完整特征数据
- 每个客户的预期评分结果和反事实分析结果
- 确保Demo演示时结果一致

【输出规格】
- 演示脚本Markdown
- Q&A预案Markdown
- 预设场景数据JSON

【技术约束】
- 演示脚本时间控制精确到秒
- Q&A回答简洁有力，避免技术术语堆砌
- 预设场景需提前验证结果正确
```

---

#### 任务 W6-T3：最终调优与交付

**负责人**：A+B+C全员  
**预计工时**：1.5天

**具体要求**：
- 根据Week3-5的实验结果，微调模型和因果分析参数
- 优化Streamlit Demo的加载速度和交互体验
- 完善代码文档和README
- Docker容器化部署
- 最终验收测试

**交付物**：
- `Dockerfile` + `docker-compose.yml`
- `README.md`
- 最终版所有代码和文档

**验收标准**：
- Docker一键启动成功
- README包含完整的使用说明
- 所有Week1-6交付物完整且质量达标

```prompt
你是一个DevOps工程师和项目交付专家。请为CausalCredit项目完成最终调优与交付。

【任务背景】
CausalCredit项目6周开发即将结束，需要进行最终调优、容器化部署和项目交付。

【具体要求】
1. 创建`Dockerfile`：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（graphviz用于因果图可视化）
RUN apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# 复制源代码
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

# 暴露端口
EXPOSE 8000 8501

# 启动命令（通过docker-compose分别启动API和前端）
```

2. 创建`docker-compose.yml`：

```yaml
version: '3.8'
services:
  api:
    build: .
    command: uvicorn src.api.app:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  frontend:
    build: .
    command: streamlit run src/frontend/app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      api:
        condition: service_healthy
    environment:
      - API_URL=http://api:8000
```

3. 创建`README.md`，包含：
- 项目简介（1段话）
- 3大创新点
- 快速开始（3步：clone→docker-compose up→访问）
- 项目结构
- 技术栈
- API文档链接
- 开发指南

4. 最终调优清单：
- [ ] LightGBM模型：用全量数据重新训练最终模型
- [ ] CATE模型：调整CausalForestDML超参数（如果CATE异质性不足）
- [ ] 概率校准：验证校准效果
- [ ] API性能：缓存SHAP计算结果
- [ ] Streamlit：添加加载动画、错误提示
- [ ] 代码质量：运行lint、修复warning
- [ ] 文档完整性：检查所有模块有docstring

5. 最终验收测试：
- 运行完整测试套件：`make test`
- Docker构建测试：`docker-compose up --build`
- Demo功能测试：手动验证3个预设场景
- 性能测试：单次评分延迟<500ms

【输出规格】
- Dockerfile
- docker-compose.yml
- README.md
- 最终验收报告

【技术约束】
- Docker镜像大小<2GB
- docker-compose up后60秒内服务可用
- 所有端口可从宿主机访问
```

---

## 四、关键技术决策及理由

| 决策点 | 选择 | 备选方案 | 选择理由 |
|-------|------|---------|---------|
| **预测基座** | LightGBM | XGBoost, Neural Network | 表格数据SOTA；训练快（CPU即可）；特征重要性天然可解释；与SHAP/DoWhy集成良好 |
| **因果发现** | DoWhy（领域知识驱动） | PC算法、FCI算法（数据驱动） | 信贷领域因果结构有充分领域知识；数据驱动因果发现需要大样本且不稳定；DoWhy支持假设验证 |
| **CATE估计** | CausalForestDML | DoubleML, CausalBART, DR-Learner | 非参数方法，不假设CATE函数形式；特征重要性可解释；EconML生态成熟 |
| **可解释性** | SHAP TreeExplainer | LIME, Anchors | TreeExplainer对树模型精确且快速；SHAP值有博弈论理论基础；全局+局部解释统一 |
| **反事实推理** | CATE调整法 | GAN反事实、DoWhy counterfactual | 简单可解释；与CATE模型直接衔接；概率输出可控；无需训练额外模型 |
| **概率校准** | Isotonic Regression | Platt Scaling | 非参数方法，对任意分布都能校准；违约概率分布通常非对称，Isotonic更灵活 |
| **后端框架** | FastAPI | Flask, Django | 异步高性能；自动生成OpenAPI文档；Pydantic数据验证；类型安全 |
| **前端框架** | Streamlit | Dash, Gradio | 开发速度最快；数据可视化生态好；适合比赛Demo；无需前端经验 |
| **特征编码** | LabelEncoder+TargetEncoder | OneHotEncoder, CatBoostEncoder | 高基数分类特征用TargetEncoder避免维度爆炸；5折交叉编码防泄漏 |
| **超参数优化** | Optuna | GridSearch, RandomSearch | 贝叶斯优化效率高；50次试验即可找到近优解；与LightGBM集成良好 |

---

## 五、风险点与应对

### 5.1 技术风险

| 风险 | 严重度 | 概率 | 影响 | 缓解策略 | 触发条件 | 应急方案 |
|------|-------|------|------|---------|---------|---------|
| 因果图假设错误导致ATE方向与预期相反 | 🔴 高 | 15% | 核心创新点失效 | 多方法交叉验证；反驳测试；敏感性分析 | ATE方向与预期相反 | 重新审视因果图；调整干预变量；强调"发现意外因果效应"的价值 |
| CATE异质性不足（CATE几乎无变异） | 🟠 中高 | 20% | 核心创新点弱化 | 增加效应修饰变量；使用非线性CATE模型；子群细分 | CATE标准差<ATE的30% | 改用子群分析代替连续CATE；强调ATE的稳健性 |
| 薄信用人群CATE估计不可靠 | 🟡 中 | 30% | 普惠金融叙事弱化 | 增大薄信用子群样本；Bootstrap置信区间；迁移学习 | 薄信用CATE置信区间过宽 | 改用PSM子群分析；强调"方向性发现"而非精确估计 |
| LightGBM AUC不达标（<0.76） | 🟡 中 | 15% | 预测基座不够强 | 增加特征工程；调整超参数；尝试特征选择 | 5折CV AUC<0.76 | 降低目标至0.74；强调因果推理的增量价值而非预测性能 |
| 反事实推理概率超出[0,1] | 🟡 中 | 25% | 结果不可信 | Clip处理；使用logit空间调整；校准后调整 | 反事实概率<0或>1 | 在logit空间做CATE调整再sigmoid回概率空间 |

### 5.2 项目风险

| 风险 | 严重度 | 概率 | 缓解策略 |
|------|-------|------|---------|
| Week3 Go/No-Go检查点未通过 | 🔴 高 | 15% | 提前准备B计划：如果因果效应不显著，转向"因果发现"叙事（发现意外因果路径） |
| 6周时间不足 | 🟠 中高 | 25% | 严格按里程碑推进；Week4-5可并行开发；降低非核心功能优先级 |
| 3人团队技能不匹配 | 🟡 中 | 20% | 明确分工；交叉培训；关键模块有代码审查 |
| 比赛演示时Demo出bug | 🟡 中 | 15% | 预设场景预加载结果；离线模式兜底；充分测试 |

### 5.3 业务风险

| 风险 | 严重度 | 概率 | 缓解策略 |
|------|-------|------|---------|
| 评委质疑因果推断可信度 | 🟠 中高 | 40% | 准备反驳验证结果；强调DoWhy的三重验证机制；提供敏感性分析 |
| 评委认为方案太学术化 | 🟡 中 | 30% | Demo展示直观的反事实效果；量化业务ROI；强调HKMA可解释性合规价值 |
| 开源数据集与银行真实场景差距大 | 🟡 中 | 25% | 方案文档详细说明真实部署路径；强调方法论的可迁移性 |

---

## 六、项目里程碑与检查点

### 6.1 里程碑时间线

```
Week 1 ──── Week 2 ──── Week 3 ──── Week 4 ──── Week 5 ──── Week 6
  │            │            │            │            │            │
  ▼            ▼            ▼            ▼            ▼            ▼
M1:数据       M2:预测      M3:因果      M4:可解释    M5:系统      M6:交付
基础就绪      基座达标      推理成立      性完成        集成完成      就绪
```

### 6.2 里程碑定义

| 里程碑 | 时间点 | 关键交付物 | 验收标准 |
|-------|--------|----------|---------|
| **M1: 数据基础就绪** | Week1结束 | 数据加载器、EDA报告、因果图 | 8表加载成功、因果图≥15节点 |
| **M2: 预测基座达标** | Week2结束 | LightGBM模型、特征工程 | 5折CV AUC≥0.76 |
| **M3: 因果推理成立** 🚨 | Week3结束 | ATE+CATE估计、反驳验证 | ATE方向正确+≥2种反驳稳健 |
| **M4: 可解释性完成** | Week4结束 | SHAP分析、反事实引擎 | 证据链可生成、反事实建议可读 |
| **M5: 系统集成完成** | Week5结束 | FastAPI+Streamlit | API+Demo可正常运行 |
| **M6: 交付就绪** | Week6结束 | 白皮书、演示材料、Docker | 一键启动、演示无bug |

### 6.3 每周检查点

| 检查点 | 时间 | 检查内容 | 负责人 |
|-------|------|---------|-------|
| CP1 | Week1周五 | 数据加载+EDA+因果图 | A+B |
| CP2 | Week2周五 | 特征工程+LightGBM | B |
| **CP3** 🚨 | **Week3周五** | **ATE+CATE+Go/No-Go** | **A+B+C** |
| CP4 | Week4周五 | SHAP+反事实 | A+B |
| CP5 | Week5周五 | API+Demo集成测试 | C |
| CP6 | Week6周四 | 最终交付验收 | A+B+C |

### 6.4 关键依赖关系

```
W1-T2(数据加载) → W1-T3(EDA) → W1-T4(因果图)
                                    ↓
W2-T1(特征工程) ←───────────────────┘
     ↓
W2-T2(LightGBM) → W2-T3(因果变量验证)
                       ↓
W3-T1(ATE估计) → W3-T2(CATE估计) → W3-T3(Go/No-Go)
                                        ↓
W4-T1(SHAP) ←──────────────────────────┘
W4-T2(反事实) ←── W3-T2(CATE)
     ↓
W5-T1(FastAPI) ←── W4-T1 + W4-T2
W5-T2(Streamlit) ←── W5-T1
W5-T3(集成测试) ←── W5-T1 + W5-T2
     ↓
W6-T1(白皮书) ←── 所有前序交付物
W6-T2(演示材料) ←── W5-T2
W6-T3(最终交付) ←── 所有交付物
```

---

## 附录A：项目目录结构总览

```
causalcredit/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Home Credit数据加载器
│   │   ├── lending_club_loader.py # Lending Club数据加载器
│   │   └── validator.py           # 数据校验
│   ├── features/
│   │   ├── __init__.py
│   │   ├── builder.py             # 特征构建主入口
│   │   ├── aggregation.py         # 多表聚合
│   │   ├── causal_features.py     # 因果特征
│   │   └── encoding.py            # 特征编码
│   ├── causal/
│   │   ├── __init__.py
│   │   ├── graph.py               # 因果图定义
│   │   ├── estimate.py            # ATE估计
│   │   ├── cate.py                # CATE估计
│   │   ├── refute.py              # 反驳验证（集成在estimate.py中）
│   │   └── variable_validation.py # 因果变量验证
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py               # LightGBM训练
│   │   ├── evaluate.py            # 模型评估
│   │   └── calibrate.py           # 概率校准
│   ├── explain/
│   │   ├── __init__.py
│   │   ├── shap_explain.py        # SHAP解释
│   │   ├── counterfactual.py      # 反事实推理
│   │   ├── decision.py            # 决策建议
│   │   └── evidence.py            # 证据链生成
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI主入口
│   │   ├── routes.py              # 路由定义
│   │   ├── schemas.py             # Pydantic模型
│   │   ├── services.py            # 业务逻辑
│   │   └── dependencies.py        # 依赖注入
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── app.py                 # Streamlit主入口
│   │   ├── pages/
│   │   │   ├── score_dashboard.py
│   │   │   ├── causal_visualization.py
│   │   │   ├── counterfactual_simulator.py
│   │   │   └── decision_panel.py
│   │   └── components/
│   │       ├── charts.py
│   │       └── layouts.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # 配置管理
│       └── logger.py              # 日志工具
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── test_data.py
│   ├── test_api.py
│   ├── test_integration.py
│   └── test_performance.py
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_causal_variable_validation.ipynb
│   ├── 05_causal_effect_estimation.ipynb
│   ├── 06_cate_analysis.ipynb
│   ├── 07_shap_analysis.ipynb
│   └── 08_counterfactual_analysis.ipynb
├── configs/
│   └── config.yaml
├── docs/
│   ├── CausalCredit_技术白皮书.md
│   ├── causal_graph.md
│   ├── causal_effect_report.md
│   ├── cate_report.md
│   ├── eda_summary.md
│   ├── causal_variable_report.md
│   ├── week3_gonogo_report.md
│   ├── demo_script.md
│   ├── qa_preparation.md
│   └── figures/
├── data/                          # gitignore
├── models/                        # gitignore
├── logs/                          # gitignore
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 附录B：因果增强评分公式

### B.1 评分公式

```
CausalCredit Score = BaseScore + CausalAdjustment + CounterfactualBonus

其中：
- BaseScore = f(P_default) ∈ [300, 850]
  f(p) = 850 - 550 × p  （线性映射，p为校准后违约概率）

- CausalAdjustment = α × CATE_normalized ∈ [-50, +50]
  CATE_normalized = (CATE - CATE_mean) / CATE_std
  α = 50 （调整幅度系数）
  正CATE（利率升高增加违约风险）→ 降低评分
  负CATE（该客户对利率不敏感）→ 提高评分

- CounterfactualBonus = β × ΔP_optimal ∈ [0, +30]
  ΔP_optimal = max(0, P_baseline - P_counterfactual_optimal)
  β = 30 / max(ΔP)  （归一化到[0, 30]）
  如果存在有效干预可显著降低违约概率 → 评分加分（鼓励可改善客户）
```

### B.2 风险等级映射

| 评分范围 | 风险等级 | 含义 | 建议动作 |
|---------|---------|------|---------|
| 750-850 | A | 优质客户 | 自动批准，最优利率 |
| 650-749 | B | 良好客户 | 批准，标准利率 |
| 550-649 | C | 一般客户 | 附条件批准，可优化方案 |
| 450-549 | D | 关注客户 | 需人工审核，反事实建议 |
| 300-449 | E | 高风险客户 | 建议拒绝，改善建议 |

---

## 附录C：配置文件模板

```yaml
# configs/config.yaml

data:
  home_credit_dir: "data/home-credit-default-risk/"
  lending_club_dir: "data/lending-club/"
  processed_dir: "data/processed/"
  models_dir: "models/"

features:
  top_k: 80                    # 保留Top-K特征
  target_encoding_folds: 5     # 目标编码折数
  thin_credit_threshold: 0     # bureau记录数阈值

causal:
  graph_path: "configs/causal_graph.dot"
  ate_methods:
    - "backdoor_regression"
    - "backdoor_ipw"
    - "backdoor_psm"
  refute_methods:
    - "random_common_cause"
    - "placebo_treatment_refuter"
    - "data_subset_refuter"
    - "add_unobserved_confounding"
  refute_iterations: 100
  cate:
    n_estimators: 1000
    max_depth: 8
    min_samples_leaf: 50

model:
  lightgbm:
    objective: "binary"
    metric: "auc"
    num_leaves: 63
    max_depth: 7
    learning_rate: 0.05
    n_estimators: 1000
    min_child_samples: 50
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0.1
    reg_lambda: 0.1
    early_stopping_rounds: 50
    random_state: 42
  optuna:
    n_trials: 50
    timeout: 3600

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  log_level: "info"

frontend:
  port: 8501
  theme: "light"
  api_url: "http://localhost:8000"
```

---

*文档由大卫-解决方案架构师编写，版本v1.0，2026-06-05*
