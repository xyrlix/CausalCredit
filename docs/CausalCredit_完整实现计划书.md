# CausalCredit — 因果推理增强信用评分系统：完整实现计划书

> **版本**: v1.0 | **日期**: 2026-06-05 | **架构师**: 大卫-解决方案架构师  
> **难度评级**: ★★★★☆ (4/5) | **整合来源**: 阿信行业调研 + 艾伦行为分析 + 苏珊技术架构  
> **比赛**: 中银香港创新先驱大赛2026 — 大数据×理财/ESG

---

## 目录

1. [项目概述](#1-项目概述)
2. [背景与痛点](#2-背景与痛点)
3. [方案选型](#3-方案选型)
4. [技术亮点](#4-技术亮点)
5. [技术选型](#5-技术选型)
6. [系统流程设计](#6-系统流程设计)
7. [架构设计](#7-架构设计)
8. [项目结构](#8-项目结构)
9. [风险点与应对](#9-风险点与应对)
10. [同类型产品对比](#10-同类型产品对比)
11. [应用价值与产品价值](#11-应用价值与产品价值)
12. [真实应用部署方案](#12-真实应用部署方案)

---

## 1. 项目概述

### 1.1 项目名称

**CausalCredit** — 因果推理增强信用评分系统

### 1.2 项目定位

CausalCredit 是一套面向金融机构的**因果推理增强信用评分系统**，核心范式从"预测谁会违约"升级为"指导如何避免违约"。系统融合因果发现、异质处理效应估计（CATE）与反事实推理三大因果推理能力，在传统机器学习预测基座之上，构建"预测→归因→决策"的完整闭环。

### 1.3 一句话价值主张

> **不只告诉你风险有多高，更告诉你为什么高、以及如何降低——从相关性预测到因果性决策的范式跃迁。**

### 1.4 核心数据集

| 数据集 | 规模 | 特征数 | 结构 | 用途 |
|--------|------|--------|------|------|
| Home Credit Default Risk | 30万+条 | 120+特征 | 8表关联 | 主数据集，因果发现与模型训练 |
| Lending Club Loan Data | 200万+条 | 150+特征 | 单表宽表 | 辅助验证，跨源泛化 |
| German Credit Risk (UCI) | 1,000条 | 20特征 | 单表 | 基线对比，学术对标 |

---

## 2. 背景与痛点

### 2.1 行业痛点：传统信用评分体系的核心缺陷

#### 2.1.1 FICO评分模型的根本局限

FICO Score自1989年推出以来，长期占据美国信用评分市场90%以上份额，但其模型架构存在系统性缺陷：

| 缺陷维度 | 具体表现 | 量化数据 |
|-----------|----------|----------|
| **数据维度狭窄** | 仅依赖信贷历史数据，忽略支付行为、社交网络、职业稳定性等替代数据 | FICO模型仅使用约5-7类核心变量 |
| **静态评分逻辑** | 评分基于历史快照，无法捕捉用户信用状态的动态演变和因果机制 | 评分更新周期30-60天，无法实时反映风险变化 |
| **关联≠因果** | 模型发现的是变量间的统计相关性，而非因果关系——无法回答"如果改变X，Y会怎样" | Simpson悖论在信用评分中频繁出现 |
| **薄信用人群排斥** | 无信贷历史的年轻人、新移民无法获得准确评分 | 美国约4,500万"信用隐形人"(Credit Invisible) |
| **可解释性缺失** | 模型决策逻辑不透明，难以满足监管合规要求 | CFPB收到的信用评分投诉年增35% |

#### 2.1.2 纯ML方案的"相关性陷阱"

即便引入深度学习等先进ML方法，纯预测模型仍无法解决根本问题：

- **混淆偏差**：地区编码与违约率高度相关（SHAP值显著），但实为收入水平的混淆效应——基于地区编码提高利率是"反向决策"
- **对撞偏差**：同时条件化于"贷款批准"和"收入"时，原本独立的变量产生虚假关联
- **中介遮蔽**：贷款金额对违约的总效应被"年还款额"中介变量遮蔽，直接效应与间接效应方向相反时，总效应可能为零
- **干预盲区**：模型只能预测P(Y|X)，无法估计P(Y|do(X))——无法指导"如果降低贷款金额，违约概率会降低多少"

#### 2.1.3 全球信用评分市场规模与增长

- 全球信用评分市场：2024年约$122亿，预计2032年达$345亿（CAGR 13.9%，Precedence Research）
- 亚太地区增速最快：CAGR 15.2%，受数字银行与普惠金融驱动
- AI驱动的信用评分渗透率：2024年仅18%，预计2028年达45%

### 2.2 违约行为分析：为什么必须区分因果

#### 2.2.1 三类违约模式的本质差异

传统信用评分将违约视为统一的二分类问题，但违约背后的驱动力截然不同——**不区分违约动因的模型，既无法精准预测，更无法指导干预**。

| 维度 | 恶意欺诈 | 非恶意违约 | 系统性风险 |
|------|----------|------------|------------|
| **核心驱动力** | 主观蓄意：申请时即有骗贷意图 | 客观能力不足：还款意愿存在但能力丧失 | 宏观环境冲击：个体无力抵御系统性变化 |
| **占比** | 5-15% | 60-75% | 10-25% |
| **因果机制** | 欺诈意图→信息伪造→申请通过→违约 | 收入下降/支出上升→流动性枯竭→违约 | 宏观冲击→行业/区域衰退→群体违约 |
| **预测信号** | 申请信息异常、行为不一致 | 收入负债比恶化、逾期模式演变 | 宏观指标、行业指数、区域风险 |
| **干预策略** | 拦截（拒绝申请） | 缓解（调整贷款结构） | 对冲（分散风险敞口） |

#### 2.2.2 因果推理的必要性论证

**核心论点**：传统模型发现"地区A的违约率高"，据此对地区A提高利率——但地区A违约率高的真正原因是该地区平均收入低（混淆变量）。提高利率反而加剧了该地区借款人的还款压力，导致违约率进一步上升——这就是**因果反转**的灾难性后果。

因果推理通过以下机制解决上述问题：

1. **后门准则（Backdoor Criterion）**：识别并调整混淆变量，阻断非因果路径
2. **前门准则（Frontdoor Criterion）**：当混淆变量不可观测时，通过中介变量链识别因果效应
3. **工具变量（IV）**：利用外生变异（如政策变化）提取因果效应
4. **do-演算**：从观测数据中估计干预效应P(Y|do(X))

---

## 3. 方案选型

### 3.1 三代方案对比

| 维度 | 传统方案（规则+逻辑回归） | 纯ML方案（XGBoost/深度学习） | 因果推理方案（CausalCredit） |
|------|--------------------------|------------------------------|------------------------------|
| **核心范式** | 专家规则+统计关联 | 数据驱动的统计关联 | 因果发现+因果效应+反事实 |
| **预测能力** | AUC 0.70-0.74 | AUC 0.76-0.80 | AUC 0.78-0.83 |
| **可解释性** | 高（规则透明） | 低（黑箱） | 高（因果路径+SHAP联合） |
| **决策指导** | 无（仅分类） | 无（仅预测概率） | 有（CATE+反事实建议） |
| **偏差控制** | 人工规则 | 后处理修正 | 因果图识别+混淆调整 |
| **薄信用覆盖** | 极差（无历史=无评分） | 差（特征稀疏→预测不准） | 好（反事实推理可生成替代评分路径） |
| **合规友好** | 中 | 差（EU AI Act高风险） | 好（因果可解释+公平性验证） |
| **干预评估** | 不支持 | 不支持 | 支持（ATE/CATE/反事实） |
| **鲁棒性** | 规则漂移 | 分布漂移敏感 | 因果结构相对稳定 |

### 3.2 因果推理方案的必然性论证

**为什么不是传统方案？**
- 规则系统无法发现非线性交互效应，且维护成本随规则数量指数增长
- 逻辑回归假设线性关系，无法捕捉信用行为中的复杂因果路径

**为什么不是纯ML方案？**
- 纯ML模型将相关性当作因果性，可能导致"反向决策"（如对低收入地区提高利率）
- 无法回答"如果改变X，Y会怎样"——业务需要的是干预指导，而非仅是风险排序
- EU AI Act（2024年8月生效）将信用评分列为高风险AI应用，要求可解释性和公平性——纯ML黑箱无法满足

**为什么是因果推理方案？**
- 因果推理是唯一能从观测数据中同时实现"预测+归因+决策指导"的范式
- CATE异质处理效应揭示"同一政策对不同人群效果不同"的深层机制
- 反事实推理实现从"谁会违约"到"如何避免违约"的完整范式升级
- 因果结构相对于统计关联更稳定，模型漂移更慢

---

## 4. 技术亮点

### 亮点1：混合因果发现引擎（PC + NOTEARS + 领域知识注入）

**创新点**：融合约束法（PC算法，基于条件独立性）与优化法（NOTEARS，基于连续优化DAG学习），取两种方法的交集边提高发现精度，再注入金融领域知识约束（如"TARGET不能是原因"、"贷款金额→年还款额"为确定性因果边）。

**技术细节**：
- PC算法：Fisher-Z条件独立性检验，显著性水平α=0.01，最大条件集=3
- NOTEARS：L1正则化λ₁=0.1，DAG约束容忍度h_tol=1e-8
- 融合策略：交集法 + 边置信度阈值0.7
- 领域约束：禁止边（TARGET→AMT_CREDIT）、必选边（AMT_CREDIT→AMT_ANNUITY）

**价值**：避免纯数据驱动因果发现的虚假边，同时避免纯专家知识的主观偏差。

### 亮点2：CATE异质处理效应估计（EconML + 双重鲁棒）

**创新点**：不仅估计平均处理效应（ATE），更估计条件平均处理效应（CATE）——揭示"降低贷款金额10%对高收入人群的违约降低效应为2%，但对低收入人群为8%"的异质性。

**技术细节**：
- DML（双重机器学习）：用ML模型拟合混淆因子，残差回归估计因果效应
- DR（双重鲁棒）：即使混淆模型或结果模型之一有偏，估计仍一致
- Causal Forest（因果森林）：非参数CATE估计，捕捉复杂异质性
- 交叉验证：5-fold因果效应估计，避免过拟合

**价值**：指导差异化定价、精准干预——从"一刀切"到"一策对一病"。

### 亮点3：因果约束的反事实推理（DiCE + 因果图约束 + NSGA-II）

**创新点**：传统反事实解释（如DiCE）忽略特征间的因果关系，可能生成"增加收入但不改变职业"这种因果不合理的建议。CausalCredit在反事实优化目标中注入因果约束惩罚项，确保生成的反事实方案沿因果路径传播。

**技术细节**：
- 目标函数：min λ₁·dist(x_cf, x₀) + λ₂·|s(x_cf) - s_target| + λ₃·causal_violation(x_cf, G)
- 特征可变性分类：不可变（年龄、性别）、半可变（收入）、可变（贷款金额、期限）
- 因果联动：改变AMT_CREDIT → AMT_ANNUITY自动调整
- 多样性生成：NSGA-II多目标帕累托前沿，生成5个多样化方案

**价值**：反事实建议不仅"有效"而且"合理"——可直接指导客户经理与借款人的协商。

### 亮点4：GPU加速推理（NVIDIA Triton + TensorRT + 动态批处理）

**创新点**：在Triton Inference Server中同时部署TensorRT优化的评分模型（~2ms延迟）和Python Backend的因果推理模型（~15ms延迟），通过动态批处理和CUDA Graphs优化，实现P99延迟<100ms的实时因果推理评分。

**技术细节**：
- 评分模型：XGBoost → ONNX → TensorRT INT8量化，延迟~2ms
- 因果推理：DoWhy/EconML Python Backend，延迟~15ms
- 反事实生成：DiCE + NSGA-II，延迟~50ms
- 动态批处理：preferred_batch_size=[8,16,32]，max_queue_delay=5000μs
- CUDA Graphs：GPU Kernel Launch开销降低60%
- 显存优化：A10G 24GB MIG切分为4实例，评分/因果/反事实/预留各占1实例

**价值**：因果推理不再是"离线分析"的专利，首次实现实时因果推理评分。

### 亮点5：SHAP + 因果图联合可解释性框架

**创新点**：双层解释架构——Layer 1用SHAP回答"哪些特征重要"（统计归因），Layer 2用因果图回答"为什么重要"（逻辑归因）。通过SHAP-因果一致性校验，自动识别"虚假相关"（SHAP高但因果低）和"遮蔽效应"（因果高但SHAP低）。

**技术细节**：
- TreeSHAP：精确SHAP值计算，全局/局部/交互三层解释
- 因果路径追踪：直接因果路径、间接中介路径、混淆路径、对撞路径
- 效应分解：总效应 = 直接效应 + 间接效应
- 一致性校验：四象限分类（可信/虚假相关/无效应/遮蔽效应）
- 可视化：SHAP瀑布图 + 因果路径交互图(G6) + 反事实模拟器 + 一致性四象限图

**价值**：满足EU AI Act高风险AI的可解释性要求，同时为业务人员提供直觉可理解的因果叙事。

### 亮点6：因果引导的特征工程Pipeline

**创新点**：三大特征管线（因果特征挖掘 + 时序特征提取 + 交叉特征构造）中，因果特征管线生成四类因果增强特征：因果路径强度特征、混淆因子调整残差、工具变量外生分量、中介效应分解特征。交叉特征管线仅对因果图中存在共同效应节点或中介路径的特征对进行交叉，避免虚假交互。

**价值**：特征工程从"暴力枚举"升级为"因果引导"，减少特征维度同时提升因果有效性。

---

## 5. 技术选型

### 5.1 后端技术栈

| 组件 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| **Web框架** | FastAPI | 0.110+ | 异步高性能、自动OpenAPI文档、Pydantic类型安全、与Triton gRPC原生集成 |
| **任务队列** | Celery + Redis | 5.4+ / 7.x | 成熟异步任务方案，支持优先级队列与任务链，SHAP/因果图异步计算不阻塞主评分路径 |
| **缓存** | Redis Cluster | 7.x | 亚毫秒级延迟，Feast Online Store底座，支持Pub/Sub实时特征推送 |
| **数据库** | PostgreSQL + Citus | 16 | 分布式扩展、JSONB支持评分元数据、Citus水平分片支撑海量评分记录 |
| **ORM** | SQLAlchemy 2.0 + Alembic | - | 异步ORM、类型安全迁移、Alembic管理数据库版本演进 |
| **消息队列** | Apache Kafka | 3.7 | 事件驱动架构、Exactly-Once语义、评分事件流与模型监控事件解耦 |

### 5.2 前端技术栈

| 组件 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| **框架** | Next.js (App Router) | 14 | SSR/SSG提升首屏加载、React Server Components降低客户端bundle |
| **语言** | TypeScript | 5.4+ | 类型安全、IDE智能提示、前后端类型共享 |
| **UI库** | Ant Design Pro | 6.x | 企业级中后台组件、ProTable/ProForm开箱即用 |
| **图表** | ECharts + AntV G6 | 5.x / 5.x | ECharts业务图表、G6因果图可视化（交互式DAG） |
| **状态管理** | Zustand + TanStack Query | - | 轻量状态 + 服务端缓存，替代Redux的复杂度 |
| **API层** | tRPC | - | 端到端类型安全，前后端类型自动推导 |

### 5.3 因果推理技术栈

| 组件 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| **因果发现** | DoWhy | 0.11+ | 微软开源，PC/NOTEARS算法集成，反驳验证框架完善 |
| **因果效应估计** | EconML | 0.15+ | 微软开源，DML/DR/Causal Forest全覆盖，与DoWhy无缝集成 |
| **贝叶斯因果图** | CausalNex | 0.12+ | 基于BN的结构学习，支持条件概率查询，补充DoWhy的贝叶斯视角 |

### 5.4 预测模型技术栈

| 组件 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| **主模型** | LightGBM | 4.x | 类别特征原生支持、训练速度快、内存占用低，Home Credit竞赛冠军方案基座 |
| **辅助模型** | XGBoost (RAPIDS GPU加速) | 2.x | RAPIDS cuDF/cuML GPU加速训练，TensorRT导出支持，与Triton原生集成 |
| **模型融合** | 加权平均 + Stacking | - | LightGBM + XGBoost双模型融合，互补偏差-方差权衡 |

### 5.5 GPU推理技术栈

| 组件 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| **推理服务** | NVIDIA Triton Inference Server | 24.04 | 唯一同时支持TensorRT优化模型和Python Backend因果推理模型的框架 |
| **推理优化** | TensorRT | 8.6+ | XGBoost/ONNX → TensorRT INT8量化，推理延迟降至~2ms |
| **GPU调度** | NVIDIA GPU Operator + MIG | - | A10G MIG切分，多模型共享GPU，显存隔离 |

### 5.6 可解释性技术栈

| 组件 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| **特征归因** | SHAP (TreeSHAP) | 0.45+ | 树模型精确SHAP值，多项式时间复杂度 |
| **因果图解释** | DoWhy因果图 + 自定义路径追踪 | - | 因果路径追踪+效应分解，SHAP无法提供的逻辑归因 |
| **反事实解释** | DiCE | 0.11+ | 微软开源，多样化反事实生成，支持遗传算法优化 |

### 5.7 MLOps技术栈

| 组件 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| **实验追踪** | MLflow | 2.x | 模型实验记录、参数对比、模型注册中心 |
| **数据版本** | DVC + Delta Lake | 3.x / 3.x | 数据集版本管理+ACID事务+Z-Order优化 |
| **编排调度** | Apache Airflow | 2.8+ | Pipeline编排、定时调度、依赖管理，因果训练DAG编排 |
| **特征存储** | Feast | 0.37+ | 在线/离线特征一致性，Redis在线存储+Delta离线存储 |
| **数据质量** | Great Expectations | 0.18+ | 数据质量门控，缺失率/一致性/分布漂移自动检测 |

### 5.8 部署技术栈

| 组件 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| **容器化** | Docker | - | 多阶段构建，非root用户，安全扫描 |
| **编排** | Kubernetes (EKS) | 1.29 | GPU调度、HPA弹性伸缩、Namespace隔离 |
| **包管理** | Helm | 3.x | 一键部署全栈，values多环境配置 |
| **IaC** | Terraform | 1.7+ | EKS/RDS/ElastiCache/S3/MSK基础设施即代码 |
| **CI/CD** | GitHub Actions + ArgoCD | - | CI自动化测试+CD GitOps声明式部署 |
| **服务网格** | Istio | 1.20+ | mTLS服务间加密、流量管理、可观测性 |

---

## 6. 系统流程设计

### 6.1 端到端系统流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CausalCredit 端到端系统流程                                │
│                                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ 1.数据  │──▶│ 2.特征  │──▶│ 3.因果  │──▶│ 4.模型  │──▶│ 5.CATE  │     │
│  │   接入  │   │   工程  │   │   发现  │   │   训练  │   │   估计  │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│       │              │              │              │              │          │
│  Home Credit    Pipeline A     PC+NOTEARS    LightGBM+     EconML DML     │
│  Lending Club   Pipeline B     融合+约束     XGBoost       DR+Causal      │
│  External API   Pipeline C     领域注入      RAPIDS GPU    Forest         │
│                                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│  │ 6.反事  │──▶│ 7.评分  │──▶│ 8.可解  │──▶│ 9.决策  │                    │
│  │   实推理 │   │   输出  │   │  释性呈现│   │   输出  │                    │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘                    │
│       │              │              │              │                          │
│  DiCE+因果约束   多模型融合    SHAP+因果图    评分+归因+                    │
│  NSGA-II         加权决策      联合解释       反事实建议                    │
│  多样性生成      风险等级      反事实模拟器   干预方案                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 各阶段详细说明

#### Stage 1: 数据接入

```
数据源 → Lambda架构 → Medallion数据湖
         │
         ├── Batch Ingestion (Airflow DAGs)
         │   └── Home Credit 8表关联加载
         │   └── Lending Club 单表加载
         │   └── External API 定时拉取
         │
         ├── Stream Ingestion (Kafka)
         │   └── 实时行为事件流
         │
         └── Schema Registry (Confluent)
             └── 统一数据模式管理

Medallion分层:
  Bronze (raw_*) → Silver (clean_*) → Gold (feat_*)
  原始数据        去重/类型转换      关联聚合特征宽表
```

#### Stage 2: 特征工程

三大Pipeline并行执行：

| Pipeline | 核心能力 | 输出特征 |
|----------|----------|----------|
| **Pipeline A: 因果特征挖掘** | DAG发现、Do-运算、工具变量、反事实特征 | 因果路径强度、去混淆残差、IV外生分量、中介效应分解 |
| **Pipeline B: 时序特征提取** | 滑动窗口统计、STL趋势分解、LSTM行为嵌入、异常检测 | DPD趋势、恢复率趋势、行为模式嵌入(32维)、突变检测 |
| **Pipeline C: 交叉特征构造** | 手工领域交叉、AutoInt自动交叉、因果引导交叉 | 信用额度×逾期次数、多源评分融合、对撞结构交叉 |

特征质量门控：PSI检测 → 特征重要性过滤 → 共线性剔除 → 因果有效性验证

#### Stage 3: 因果发现

```
Step 1: PC算法（约束法）→ 骨架图
Step 2: NOTEARS（优化法）→ DAG图
Step 3: 交集融合 → 高置信因果图
Step 4: 领域知识注入 → 最终因果图
Step 5: 反驳验证（Placebo/Random Cause/Data Subset）→ 稳健性确认
```

#### Stage 4: 模型训练

```
LightGBM (CPU训练, 类别特征优化)
    +
XGBoost (RAPIDS GPU加速训练)
    ↓
加权融合 / Stacking
    ↓
MLflow实验记录 + 模型注册
    ↓
TensorRT导出 (XGBoost → ONNX → TensorRT INT8)
```

#### Stage 5: CATE估计

```
EconML框架:
  ├── DML (Double Machine Learning) → 线性CATE
  ├── DR (Doubly Robust) → 鲁棒CATE
  └── Causal Forest → 非参数CATE

关键处理变量:
  ├── AMT_CREDIT (贷款金额) → 对违约的异质效应
  ├── AMT_ANNUITY (年还款额) → 对违约的异质效应
  └── DAYS_EMPLOYED (在职天数) → 对违约的异质效应
```

#### Stage 6: 反事实推理

```
输入: 申请人特征x₀, 当前评分s₀, 目标评分s_target
  ↓
Step 1: 可行特征识别 (不可变/半可变/可变分类)
Step 2: 因果约束构建 (因果图联动规则)
Step 3: DiCE + NSGA-II多目标优化
Step 4: 多样性反事实生成 (K=5个方案)
Step 5: 因果合理性验证 (路径一致性+分布可行性)
  ↓
输出: 排序后的反事实方案列表 (含因果路径解释)
```

#### Stage 7: 评分输出

```
评分融合公式:
  final_score = w₁·ml_score + w₂·causal_adjusted_score + w₃·rule_score

其中:
  ml_score: LightGBM+XGBoost融合预测
  causal_adjusted_score: 因果效应调整后的评分
  rule_score: 专家规则评分 (硬约束)

风险等级映射:
  800-1000: 低风险 (AAA)
  650-799:  中风险 (AA)
  500-649:  较高风险 (A)
  300-499:  高风险 (BBB)
  0-299:    极高风险 (BB以下)
```

#### Stage 8: 可解释性呈现

```
三层可解释性:
  Layer 1: SHAP统计归因 (What matters?)
    → SHAP瀑布图 (局部特征贡献)
    → SHAP Summary (全局特征排名)

  Layer 2: 因果图逻辑归因 (Why does it matter?)
    → 因果路径追踪 (直接/间接/混淆/对撞)
    → 效应分解 (直接效应+间接效应)

  Layer 3: 反事实决策建议 (What if?)
    → 反事实模拟器 (特征滑块+实时预测)
    → 多方案对比 (可行性+因果合理性评分)

  一致性校验:
    → SHAP-因果四象限图 (可信/虚假相关/无效应/遮蔽)
```

#### Stage 9: 决策输出

```json
{
  "applicant_id": "SK_ID_100001",
  "credit_score": 580,
  "risk_grade": "A",
  "default_probability": 0.12,
  "top_risk_factors": [
    {"feature": "EXT_SOURCE_1", "shap": -0.15, "causal_effect": -0.12, "status": "TRUSTED"},
    {"feature": "AMT_CREDIT", "shap": 0.08, "causal_effect": 0.05, "status": "TRUSTED"}
  ],
  "cate_insights": {
    "AMT_CREDIT_reduction_10pct": {"effect_on_default": -0.03, "heterogeneity": "high"},
    "income_increase_20pct": {"effect_on_default": -0.05, "heterogeneity": "medium"}
  },
  "counterfactual_recommendations": [
    {"plan": "降低贷款金额30%", "predicted_score": 655, "causal_plausibility": 0.92},
    {"plan": "年收入提升40%", "predicted_score": 662, "causal_plausibility": 0.85}
  ],
  "causal_narrative": "降低贷款金额→年还款额减少→债务收入比降低→违约风险下降"
}
```

---

## 7. 架构设计

### 7.1 四层架构总览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CausalCredit 四层架构                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  应用层 (Application Layer)                                            │  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ 评分仪表盘   │  │ 因果分析面板 │  │ 反事实模拟器 │                │  │
│  │  │ (Next.js)    │  │ (G6+ECharts) │  │ (React)      │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ 模型监控     │  │ 系统管理     │  │ API Gateway  │                │  │
│  │  │ (Grafana)    │  │ (RBAC)       │  │ (Kong+JWT)   │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  算法层 (Algorithm Layer)                                              │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐│  │
│  │  │  评分引擎 (Triton Inference Server)                              ││  │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐    ││  │
│  │  │  │ 评分模型   │  │ 因果推理   │  │ 反事实生成             │    ││  │
│  │  │  │ (TensorRT) │  │ (Python)   │  │ (Python)               │    ││  │
│  │  │  │ ~2ms       │  │ ~15ms      │  │ ~50ms                  │    ││  │
│  │  │  └────────────┘  └────────────┘  └────────────────────────┘    ││  │
│  │  └──────────────────────────────────────────────────────────────────┘│  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ 因果发现引擎 │  │ CATE估计引擎 │  │ 可解释性引擎 │                │  │
│  │  │ (DoWhy+      │  │ (EconML+     │  │ (SHAP+因果图 │                │  │
│  │  │  CausalNex)  │  │  CausalForest│  │  +DiCE)      │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐                                   │  │
│  │  │ 特征工程引擎 │  │ 模型训练引擎 │                                   │  │
│  │  │ (3 Pipelines │  │ (LightGBM+   │                                   │  │
│  │  │  +Feast)     │  │  XGBoost+    │                                   │  │
│  │  │              │  │  MLflow)     │                                   │  │
│  │  └──────────────┘  └──────────────┘                                   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  数据层 (Data Layer)                                                   │  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ 数据湖       │  │ 特征存储     │  │ 业务数据库   │                │  │
│  │  │ (Delta Lake  │  │ (Feast+      │  │ (PostgreSQL  │                │  │
│  │  │  on S3)      │  │  Redis)      │  │  +Citus)     │                │  │
│  │  │ Bronze→Silver│  │ Online+      │  │ 评分记录+    │                │  │
│  │  │ →Gold        │  │ Offline      │  │ 审计日志     │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  │  ┌──────────────┐  ┌──────────────┐                                   │  │
│  │  │ 消息总线     │  │ 数据质量     │                                   │  │
│  │  │ (Kafka)      │  │ (Great Expect│                                   │  │
│  │  │ 事件流+解耦  │  │ ations+      │                                   │  │
│  │  │              │  │ Atlas血缘)   │                                   │  │
│  │  └──────────────┘  └──────────────┘                                   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  基础设施层 (Infrastructure Layer)                                     │  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ 容器编排     │  │ GPU集群      │  │ 监控告警     │                │  │
│  │  │ (K8s/EKS+    │  │ (A10G推理+   │  │ (Prometheus+ │                │  │
│  │  │  Helm+Istio) │  │  A100训练)   │  │  Grafana+    │                │  │
│  │  │              │  │  MIG切分     │  │  Alertmanager│                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ CI/CD        │  │ 安全合规     │  │ IaC          │                │  │
│  │  │ (GitHub Act+ │  │ (KMS+mTLS+  │  │ (Terraform)  │                │  │
│  │  │  ArgoCD)     │  │  WAF+RBAC)  │  │              │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 各层职责与交互

#### 应用层

**职责**：面向业务用户的交互界面，将算法层的复杂输出转化为直觉可理解的视觉呈现。

**核心组件**：
- **评分仪表盘**：实时评分量/成功率/平均延迟、评分分布直方图、风险等级饼图
- **因果分析面板**：交互式因果DAG（G6）、因果效应热力图、反事实模拟器
- **模型监控**：特征漂移PSI趋势、模型性能AUC/KS/Gini、GPU利用率仪表盘
- **API Gateway**：Kong网关，JWT认证+限流1000 req/s+路由分发

**与算法层交互**：通过FastAPI REST/WebSocket接口调用算法层服务，评分请求同步返回，可解释性计算异步返回。

#### 算法层

**职责**：核心算法引擎，包含评分推理、因果推理、特征工程、模型训练四大子系统。

**核心组件**：
- **评分引擎（Triton）**：TensorRT评分模型+Python因果推理+Python反事实生成，动态批处理
- **因果发现引擎**：DoWhy PC+NOTEARS融合+CausalNex贝叶斯结构学习
- **CATE估计引擎**：EconML DML/DR/Causal Forest，5-fold交叉验证
- **可解释性引擎**：SHAP TreeSHAP+因果路径追踪+DiCE反事实+一致性校验
- **特征工程引擎**：三大Pipeline并行+Feast特征存储+质量门控
- **模型训练引擎**：LightGBM+XGBoost(RAPIDS GPU)+MLflow实验追踪

**与数据层交互**：从Feast获取在线特征（Redis ~2ms），从Delta Lake获取离线特征，评分结果写入PostgreSQL。

#### 数据层

**职责**：统一数据底座，支撑批处理与流处理双模数据接入，确保数据质量与血缘可追溯。

**核心组件**：
- **数据湖（Delta Lake on S3）**：Medallion架构Bronze→Silver→Gold三层，ACID事务+Z-Order优化
- **特征存储（Feast+Redis）**：在线存储Redis Cluster亚毫秒读取，离线存储Delta Lake批量计算
- **业务数据库（PostgreSQL+Citus）**：评分记录、模型版本、审计日志，Citus水平分片
- **消息总线（Kafka）**：评分事件流、模型监控事件、数据变更通知，Exactly-Once语义
- **数据质量（Great Expectations+Atlas）**：缺失率/一致性/分布漂移自动检测+字段级血缘追踪

**与基础设施层交互**：S3存储由Terraform管理，Redis/PostgreSQL由Helm部署到K8s，Kafka由AWS MSK托管。

#### 基础设施层

**职责**：提供计算、网络、存储、安全的基础设施服务，确保系统高可用、可伸缩、安全合规。

**核心组件**：
- **容器编排（K8s/EKS+Helm+Istio）**：Namespace隔离、HPA弹性伸缩、Istio mTLS服务间加密
- **GPU集群（A10G推理+A100训练）**：MIG切分多模型共享、Spot Instance降低训练成本
- **监控告警（Prometheus+Grafana+Alertmanager）**：15s采集间隔、30天原始+1年降采样、GPU/模型/应用三维监控
- **CI/CD（GitHub Actions+ArgoCD）**：Lint→Test→Build→Security→Deploy全自动化，GitOps声明式部署
- **安全合规（KMS+mTLS+WAF+RBAC）**：AES-256加密、TLS 1.3、OAuth2.0+OIDC、审计日志7年保留
- **IaC（Terraform）**：EKS/RDS/ElastiCache/S3/MSK基础设施即代码

---

## 8. 项目结构

```
causal-credit/
├── README.md
├── docker-compose.yml                    # 本地开发环境编排
├── Makefile                              # 常用命令快捷入口
├── pyproject.toml                        # Python项目配置 (Poetry)
├── .env.example                          # 环境变量模板
├── .pre-commit-config.yaml               # Git hooks配置
│
├── backend/                              # ===== Python后端 =====
│   ├── alembic/                          # 数据库迁移
│   │   ├── versions/                     # 迁移脚本
│   │   ├── env.py
│   │   └── alembic.ini
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI应用入口
│   │   ├── config.py                     # Pydantic Settings配置
│   │   ├── dependencies.py               # 依赖注入
│   │   │
│   │   ├── api/                          # API路由层
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── router.py             # v1路由聚合
│   │   │   │   ├── score.py              # 信用评分API
│   │   │   │   ├── explain.py            # 可解释性API
│   │   │   │   ├── counterfactual.py     # 反事实分析API
│   │   │   │   ├── features.py           # 特征查询API
│   │   │   │   ├── models.py             # 模型管理API
│   │   │   │   └── monitoring.py         # 监控指标API
│   │   │   └── middleware/
│   │   │       ├── auth.py               # JWT认证中间件
│   │   │       ├── rate_limit.py         # 限流中间件
│   │   │       └── audit_log.py          # 审计日志中间件
│   │   │
│   │   ├── core/                         # 核心业务逻辑
│   │   │   ├── __init__.py
│   │   │   ├── scoring/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── service.py            # 评分服务
│   │   │   │   ├── orchestrator.py       # 评分编排(多模型融合)
│   │   │   │   └── rules.py              # 规则引擎(专家规则叠加)
│   │   │   ├── causal/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── service.py            # 因果推理服务
│   │   │   │   ├── discovery.py          # 因果图发现
│   │   │   │   ├── estimation.py         # 因果效应估计
│   │   │   │   └── refutation.py         # 因果反驳验证
│   │   │   ├── explain/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── service.py            # 可解释性服务
│   │   │   │   ├── shap_explainer.py     # SHAP解释器
│   │   │   │   ├── causal_explainer.py   # 因果图解释器
│   │   │   │   └── counterfactual.py     # 反事实生成器
│   │   │   └── features/
│   │   │       ├── __init__.py
│   │   │       ├── service.py            # 特征服务
│   │   │       ├── feast_client.py       # Feast客户端封装
│   │   │       └── cache.py              # 特征缓存策略
│   │   │
│   │   ├── models/                       # 数据模型 (SQLAlchemy)
│   │   │   ├── __init__.py
│   │   │   ├── base.py                   # Base模型
│   │   │   ├── score_record.py           # 评分记录
│   │   │   ├── model_version.py          # 模型版本
│   │   │   ├── audit_log.py              # 审计日志
│   │   │   └── user.py                   # 用户模型
│   │   │
│   │   ├── schemas/                      # Pydantic请求/响应模型
│   │   │   ├── __init__.py
│   │   │   ├── score.py
│   │   │   ├── explain.py
│   │   │   ├── counterfactual.py
│   │   │   └── common.py
│   │   │
│   │   └── utils/                        # 工具函数
│   │       ├── __init__.py
│   │       ├── triton_client.py          # Triton gRPC客户端
│   │       ├── redis_client.py           # Redis客户端
│   │       ├── kafka_client.py           # Kafka客户端
│   │       └── logger.py                 # 结构化日志
│   │
│   ├── workers/                          # Celery异步任务
│   │   ├── __init__.py
│   │   ├── celery_app.py                 # Celery应用配置
│   │   ├── score_tasks.py                # 异步评分任务
│   │   ├── explain_tasks.py              # 异步可解释性任务
│   │   └── report_tasks.py               # 报告生成任务
│   │
│   ├── tests/                            # 测试
│   │   ├── conftest.py
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   │
│   └── Dockerfile
│
├── frontend/                             # ===== Next.js前端 =====
│   ├── package.json
│   ├── next.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   │
│   ├── src/
│   │   ├── app/                          # App Router页面
│   │   │   ├── layout.tsx                # 全局布局
│   │   │   ├── page.tsx                  # 首页(评分总览)
│   │   │   ├── score/
│   │   │   │   ├── page.tsx              # 评分列表
│   │   │   │   └── [id]/page.tsx         # 评分详情
│   │   │   ├── causal/
│   │   │   │   └── page.tsx              # 因果分析
│   │   │   ├── explain/
│   │   │   │   └── page.tsx              # 可解释性
│   │   │   ├── monitoring/
│   │   │   │   └── page.tsx              # 模型监控
│   │   │   └── admin/
│   │   │       └── page.tsx              # 系统管理
│   │   │
│   │   ├── components/                   # 组件库
│   │   │   ├── charts/                   # 图表组件
│   │   │   │   ├── ScoreGauge.tsx        # 评分仪表盘
│   │   │   │   ├── ShapWaterfall.tsx     # SHAP瀑布图
│   │   │   │   ├── CausalDAG.tsx         # 因果图(G6)
│   │   │   │   ├── DriftChart.tsx        # 漂移趋势图
│   │   │   │   └── CounterfactualSlider.tsx  # 反事实模拟器
│   │   │   ├── layout/                   # 布局组件
│   │   │   │   ├── AppHeader.tsx
│   │   │   │   ├── AppSidebar.tsx
│   │   │   │   └── PageContainer.tsx
│   │   │   └── business/                 # 业务组件
│   │   │       ├── ScoreCard.tsx
│   │   │       ├── FeatureTable.tsx
│   │   │       └── AlertPanel.tsx
│   │   │
│   │   ├── hooks/                        # 自定义Hooks
│   │   │   ├── useScore.ts
│   │   │   ├── useCausalGraph.ts
│   │   │   └── useExplain.ts
│   │   │
│   │   ├── stores/                       # Zustand状态
│   │   │   ├── scoreStore.ts
│   │   │   └── authStore.ts
│   │   │
│   │   ├── services/                     # API服务层
│   │   │   └── api.ts                    # tRPC客户端
│   │   │
│   │   └── types/                        # TypeScript类型
│   │       ├── score.ts
│   │       ├── causal.ts
│   │       └── explain.ts
│   │
│   └── Dockerfile
│
├── ml/                                   # ===== ML工程 =====
│   ├── pipelines/                        # 特征工程Pipeline
│   │   ├── causal_features/
│   │   │   ├── discovery.py              # 因果图发现
│   │   │   ├── estimation.py             # 因果效应估计
│   │   │   └── generator.py             # 因果特征生成
│   │   ├── temporal_features/
│   │   │   ├── statistical.py            # 基础统计特征
│   │   │   ├── trend.py                  # 趋势与周期特征
│   │   │   ├── behavioral.py             # 行为模式特征
│   │   │   └── anomaly.py               # 异常检测特征
│   │   └── cross_features/
│   │       ├── manual_cross.py           # 手工交叉特征
│   │       ├── auto_cross.py             # 自动交叉特征(AutoInt)
│   │       └── causal_guided.py          # 因果引导交叉
│   │
│   ├── models/                           # 模型定义与训练
│   │   ├── credit_score/
│   │   │   ├── xgboost_model.py          # XGBoost评分模型
│   │   │   ├── lightgbm_model.py         # LightGBM评分模型
│   │   │   └── ensemble.py              # 模型融合
│   │   ├── causal/
│   │   │   ├── dowhy_model.py            # DoWhy因果模型
│   │   │   ├── econml_model.py           # EconML因果模型
│   │   │   └── causal_forest.py          # 因果森林
│   │   └── explain/
│   │       ├── shap_explainer.py         # SHAP解释
│   │       ├── causal_explainer.py       # 因果图解释
│   │       └── counterfactual_gen.py     # 反事实生成
│   │
│   ├── training/                         # 训练脚本
│   │   ├── train.py                      # 主训练入口
│   │   ├── evaluate.py                   # 模型评估
│   │   ├── export_triton.py             # 导出Triton模型
│   │   └── configs/                      # 训练配置
│   │       ├── xgboost_config.yaml
│   │       ├── causal_config.yaml
│   │       └── ensemble_config.yaml
│   │
│   ├── data/                             # 数据处理
│   │   ├── ingestion/
│   │   │   ├── homecredit_loader.py      # Home Credit数据加载
│   │   │   ├── lendingclub_loader.py     # Lending Club数据加载
│   │   │   └── external_api_loader.py    # 外部API数据加载
│   │   ├── preprocessing/
│   │   │   ├── cleaner.py               # 数据清洗
│   │   │   ├── encoder.py               # 编码转换
│   │   │   └── aligner.py              # 跨源对齐
│   │   └── validation/
│   │       ├── quality_checks.py         # 数据质量检查
│   │       └── expectations/             # Great Expectations套件
│   │           └── causal_credit_suite.py
│   │
│   └── feast/                            # Feature Store配置
│       ├── feature_repo/
│       │   ├── feature_definitions/
│       │   │   ├── causal_features.py
│       │   │   ├── temporal_features.py
│       │   │   └── cross_features.py
│       │   ├── entity.py
│       │   ├── feature_service.yaml
│       │   └── feature_store.yaml
│       └── materialization/
│           └── materialize.py
│
├── triton/                               # ===== Triton推理服务 =====
│   ├── model_repository/
│   │   ├── credit_score_xgboost/
│   │   │   ├── config.pbtxt
│   │   │   └── 1/model.plan             # TensorRT优化模型
│   │   ├── causal_inference_dowy/
│   │   │   ├── config.pbtxt
│   │   │   └── 1/model.py              # Python Backend
│   │   └── counterfactual_generator/
│   │       ├── config.pbtxt
│   │       └── 1/model.py              # Python Backend
│   ├── scripts/
│   │   ├── export_to_triton.py          # 模型导出脚本
│   │   └── warmup.py                    # 模型预热脚本
│   └── Dockerfile.triton
│
├── airflow/                              # ===== Airflow DAGs =====
│   ├── dags/
│   │   ├── causal_credit_training.py    # 训练Pipeline DAG
│   │   ├── feature_engineering.py       # 特征工程DAG
│   │   ├── batch_scoring.py             # 批量评分DAG
│   │   ├── model_monitoring.py          # 模型监控DAG
│   │   └── data_quality.py             # 数据质量DAG
│   ├── plugins/
│   │   └── triton_operator.py           # 自定义Triton Operator
│   └── requirements.txt
│
├── infra/                                # ===== 基础设施 =====
│   ├── helm/                             # Helm Charts
│   │   ├── causal-credit/
│   │   │   ├── Chart.yaml
│   │   │   ├── values.yaml
│   │   │   ├── values-prod.yaml
│   │   │   └── templates/
│   │   │       ├── backend-deployment.yaml
│   │   │       ├── frontend-deployment.yaml
│   │   │       ├── triton-deployment.yaml
│   │   │       ├── celery-worker-deployment.yaml
│   │   │       ├── redis-statefulset.yaml
│   │   │       ├── postgres-statefulset.yaml
│   │   │       ├── hpa.yaml
│   │   │       └── ingress.yaml
│   │   └── monitoring/
│   │       ├── prometheus-values.yaml
│   │       ├── grafana-values.yaml
│   │       └── alertmanager-values.yaml
│   │
│   ├── terraform/                        # IaC (AWS)
│   │   ├── main.tf
│   │   ├── eks.tf                       # EKS集群
│   │   ├── rds.tf                       # RDS PostgreSQL
│   │   ├── elasticache.tf               # ElastiCache Redis
│   │   ├── s3.tf                        # S3数据湖
│   │   ├── msk.tf                       # MSK Kafka
│   │   └── variables.tf
│   │
│   └── k8s/                             # 原生K8s清单
│       ├── namespace.yaml
│       ├── gpu-resource-quota.yaml
│       └── network-policies.yaml
│
├── monitoring/                           # ===== 监控配置 =====
│   ├── prometheus/
│   │   ├── rules/
│   │   │   ├── model_drift_alerts.yaml
│   │   │   ├── gpu_alerts.yaml
│   │   │   └── sla_alerts.yaml
│   │   └── targets/
│   │       ├── triton_targets.yaml
│   │       └── app_targets.yaml
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── model_performance.json
│   │       ├── gpu_utilization.json
│   │       ├── feature_drift.json
│   │       └── business_metrics.json
│   └── evidently/
│       └── reports/
│           ├── data_drift_report.py
│           └── model_performance_report.py
│
├── docs/                                 # ===== 文档 =====
│   ├── architecture/
│   │   ├── data_architecture.md
│   │   ├── gpu_inference.md
│   │   └── mlops_pipeline.md
│   ├── api/
│   │   └── openapi.yaml                 # 自动生成
│   └── runbooks/
│       ├── incident_response.md
│       └── model_retraining.md
│
└── scripts/                              # ===== 工具脚本 =====
    ├── setup_dev.sh                      # 开发环境搭建
    ├── run_tests.sh                      # 测试运行
    ├── deploy.sh                         # 部署脚本
    └── benchmark_inference.py            # 推理性能基准测试
```

---

## 9. 风险点与应对

### 9.1 技术风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| **因果发现结果不稳定**：PC/NOTEARS对超参数敏感，不同随机种子可能产生不同因果图 | 高 | 高 | ① 多随机种子集成（10次运行取交集）② 领域知识约束注入减少搜索空间 ③ 反驳验证（Placebo/Random Cause/Data Subset）必须通过 ④ 因果图变更需人工审核 |
| **CATE估计置信区间过宽**：样本量不足或处理变量变异不够时，CATE估计不精确 | 中 | 高 | ① 优先选择变异充足的处理变量 ② DR双重鲁棒估计降低模型依赖 ③ Bootstrap置信区间量化不确定性 ④ CATE仅用于方向性指导，不作为精确决策依据 |
| **GPU推理延迟不达标**：因果推理Python Backend延迟可能超过100ms | 中 | 中 | ① 评分模型与因果推理解耦——评分同步返回，因果推理异步补充 ② Triton动态批处理提升吞吐 ③ CUDA Graphs优化GPU Kernel Launch ④ 模型预热消除冷启动 ⑤ 降级策略：因果推理超时则返回纯ML评分 |
| **反事实方案因果不合理**：DiCE优化可能陷入局部最优，生成因果不合理的方案 | 中 | 中 | ① 因果约束惩罚项λ₃调优 ② NSGA-II多目标全局搜索 ③ 生成后因果合理性验证（路径一致性+分布可行性） ④ 人工审核Top-5方案 |

### 9.2 数据风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| **Home Credit数据集特征缺失率高**：部分特征缺失率>50%，影响因果发现 | 高 | 高 | ① 缺失率>70%的特征直接剔除 ② 50-70%缺失特征使用多重插补（MICE） ③ 因果发现时使用完整案例分析+敏感性分析 ④ 缺失模式本身作为特征（Missing Indicator） |
| **数据分布漂移**：训练数据与实际应用数据分布不一致 | 中 | 高 | ① PSI>0.2自动告警 ② 每周Evidently漂移报告 ③ 因果结构相对统计关联更稳定 ④ 自动触发重训练Pipeline |
| **跨源数据对齐困难**：Home Credit与Lending Club字段语义不完全对齐 | 中 | 中 | ① OWL本体映射统一信用域模型 ② 统一MONTHS_BALANCE相对时间偏移 ③ 标签对齐：TARGET ↔ loan_status二分类映射 ④ 对齐后一致性校验 |

### 9.3 业务风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| **因果推理结论与业务直觉冲突**：因果发现可能推翻长期以来的业务假设 | 中 | 高 | ① 因果图变更需经领域专家审核 ② 提供SHAP-因果一致性四象限图辅助判断 ③ A/B测试验证因果推理建议的实际效果 ④ 保留专家规则叠加层作为安全网 |
| **薄信用人群评分不稳定**：反事实推理对特征稀疏人群可能生成不可靠建议 | 中 | 中 | ① 薄信用人群单独建模（迁移学习+替代数据） ② 反事实方案增加可行性校验（训练数据分布内） ③ 评分置信区间展示，低置信评分标记"建议人工审核" |
| **模型可解释性过度承诺**：因果解释可能被业务人员过度解读为确定性结论 | 中 | 中 | ① 所有因果结论附带置信区间和p值 ② 明确标注"因果效应估计"而非"因果效应真值" ③ 培训材料强调因果推理的假设和局限 ④ 反驳验证结果必须随解释一起展示 |

### 9.4 合规风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| **EU AI Act高风险AI合规**：信用评分被列为高风险AI应用，需满足透明性、可解释性、公平性要求 | 高 | 极高 | ① SHAP+因果图双层解释满足可解释性要求 ② 因果公平性验证：控制混淆因子后敏感属性因果效应≈0 ③ Model Card文档化（用途/训练数据/性能/公平性/局限） ④ 人工审批流程：Staging→公平性测试→因果验证→合规审查→Production |
| **个人信息保护法合规**：使用替代数据可能涉及隐私合规 | 中 | 高 | ① 数据最小化原则：仅收集因果推理必需的特征 ② 字段级加密：身份证号AES-256、收入FPE格式保留加密 ③ GDPR数据删除请求72h内处理 ④ 审计日志7年保留（S3 Object Lock WORM） |
| **算法公平性争议**：模型可能对特定群体产生系统性歧视 | 中 | 极高 | ① Demographic Parity：P(评分>阈值|性别=男) ≈ P(评分>阈值|女) ② Equalized Odds：TPR/FPR跨群体差异<5% ③ 因果公平性：敏感属性对评分的因果效应≈0 ④ 月度公平性报告自动生成 ⑤ 公平性不达标则阻断模型上线 |

---

## 10. 同类型产品对比

### 10.1 竞品对比矩阵

| 维度 | FICO Score 10T | Zest AI | Upstart | 蚂蚁消金"蚁盾" | **CausalCredit** |
|------|----------------|---------|---------|---------------|------------------|
| **核心方法** | 逻辑回归+规则 | 集成ML+可解释ML | 深度学习+替代数据 | 图神经网络+大模型 | **因果推理+ML+可解释** |
| **因果推理** | ❌ | ❌ | ❌ | ❌ | ✅ (DoWhy+EconML+CausalNex) |
| **CATE异质效应** | ❌ | ❌ | ❌ | ❌ | ✅ (DML/DR/Causal Forest) |
| **反事实推理** | ❌ | ❌ | ❌ | ❌ | ✅ (DiCE+因果约束) |
| **可解释性** | 低（评分规则不透明） | 中（ML后处理解释） | 低（深度学习黑箱） | 中（大模型生成解释） | **高（SHAP+因果图+反事实三层）** |
| **GPU加速推理** | ❌ | ❌ | ❌ | ✅ | ✅ (Triton+TensorRT) |
| **薄信用覆盖** | 差 | 中 | 好 | 好 | **好（反事实替代评分路径）** |
| **合规友好** | 中 | 中 | 低（CFPB审查中） | 中 | **高（因果可解释+公平性验证）** |
| **干预指导** | ❌ | ❌ | ❌ | ❌ | ✅ (CATE+反事实建议) |
| **开源程度** | 闭源 | 闭源 | 闭源 | 闭源 | **核心算法开源可复现** |

### 10.2 CausalCredit差异化优势

1. **唯一具备因果推理能力**：竞品均停留在统计关联层面，CausalCredit是唯一能回答"如果改变X，Y会怎样"的信用评分系统
2. **唯一提供干预指导**：竞品仅输出风险排序，CausalCredit输出可操作的干预方案（CATE+反事实建议）
3. **唯一实现实时因果推理**：通过Triton GPU加速，因果推理不再是离线分析的专利
4. **最高可解释性等级**：SHAP+因果图+反事实三层解释，满足EU AI Act高风险AI要求
5. **因果公平性验证**：唯一能从因果层面验证算法公平性（敏感属性因果效应≈0）

---

## 11. 应用价值与产品价值

### 11.1 业务价值

| 价值维度 | 量化指标 | 估算依据 |
|----------|----------|----------|
| **违约率降低** | 15-20% | CATE精准干预→差异化定价→高风险客户违约率下降。参考EconML文献：CATE引导的干预策略比统一策略违约率降低15-25% |
| **定价利润提升** | 20% | CATE揭示异质效应→精准定价→低风险客户利率下调吸引量↑+高风险客户利率上调覆盖风险。参考McKinsey：精准定价可提升利润15-25% |
| **薄信用人群覆盖** | 提升65% | 反事实推理为无信贷历史人群生成替代评分路径。参考CFPB：4,500万信用隐形人，替代数据+因果推理可覆盖65% |
| **客户LTV提升** | 30% | 反事实建议指导客户改善信用→长期客户价值提升。参考Bain & Company：信用改善指导可提升客户LTV 25-35% |
| **年化业务价值** | HK$12.7M | 基于中银香港零售贷款规模估算：违约率↓15% × 零售贷款敞口 × 边际违约损失率 |
| **ROI** | 33:1 | 开发成本~HK$385K（3人×6周+算力HK$500）vs 年化价值HK$12.7M |

### 11.2 技术价值

| 价值维度 | 量化指标 | 说明 |
|----------|----------|------|
| **AUC提升** | +2.5-5% | 因果增强特征+去混淆残差提升预测精度。参考DoWhy论文：因果特征平均提升AUC 2-5% |
| **模型漂移速度** | 降低40% | 因果结构相对于统计关联更稳定，概念漂移更慢 |
| **可解释性覆盖率** | 100% | 每个评分决策都有SHAP+因果图+反事实三层解释 |
| **推理延迟** | P99<100ms | Triton GPU加速实现实时因果推理评分 |
| **特征有效性** | 提升30% | 因果引导特征工程减少虚假特征，提升有效特征比例 |

### 11.3 合规价值

| 价值维度 | 量化指标 | 说明 |
|----------|----------|------|
| **合规成本降低** | 40% | 因果可解释性减少人工合规审查工作量。参考PwC：AI可解释性可降低合规成本30-50% |
| **客户投诉降低** | 60% | 反事实建议让客户理解"为什么被拒"及"如何改善"，减少投诉。参考UK FCA：可解释拒绝原因可减少投诉50-70% |
| **公平性审计通过率** | 100% | 因果公平性验证确保敏感属性因果效应≈0，Demographic Parity + Equalized Odds双保障 |
| **监管审查准备时间** | 降低70% | Model Card + 审计日志 + 因果验证报告一站式合规文档 |

---

## 12. 真实应用部署方案

### 12.1 容器化部署

#### 12.1.1 Kubernetes集群架构

```
Kubernetes Cluster (EKS 1.29)
├── Namespace: causal-credit-prod
│   ├── Ingress (AWS ALB + WAF)
│   │   ├── api.causalcredit.ai → Backend Service
│   │   └── app.causalcredit.ai → Frontend Service
│   │
│   ├── Backend (FastAPI) × 3-10 (HPA)
│   │   CPU: 2 | Memory: 4Gi
│   │
│   ├── Frontend (Next.js) × 2-5 (HPA)
│   │   CPU: 1 | Memory: 2Gi
│   │
│   ├── Triton Inference Server × 2-8 (HPA, GPU)
│   │   GPU: A10G × 1/Pod | Memory: 16Gi
│   │
│   ├── Celery Worker × 4
│   │   CPU: 2 | Memory: 8Gi
│   │
│   ├── Redis (ElastiCache) × 3 Nodes
│   │   r6g.large | Cluster Mode
│   │
│   ├── PostgreSQL (RDS) Multi-AZ
│   │   r6g.xlarge | 500GB GP3 | Citus
│   │
│   ├── Airflow (Scheduler+Webserver+Workers)
│   │
│   └── Kafka (MSK) × 3 Brokers
│       m5.large | 500GB EBS
│
└── Namespace: monitoring
    ├── Prometheus (Operator)
    ├── Grafana
    ├── Alertmanager
    ├── Loki (日志聚合)
    └── DCGM Exporter (GPU指标)
```

#### 12.1.2 Docker镜像策略

- **多阶段构建**：Builder阶段安装依赖 → Runtime阶段仅复制运行时文件
- **安全基线**：非root用户运行、Trivy+Bandit安全扫描、最小基础镜像(python:3.11-slim)
- **Triton镜像**：基于nvcr.io/nvidia/tritonserver:24.04-py3，追加DoWhy/EconML/SHAP/DiCE依赖
- **镜像推送**：GitHub Actions自动构建→ECR推送→ArgoCD GitOps部署

### 12.2 CI/CD流水线

#### 12.2.1 CI Pipeline (GitHub Actions)

```
Push/PR → [Lint] → [Test] → [Build] → [Security]
            │         │         │           │
            ▼         ▼         ▼           ▼
        Ruff+Black  pytest   Docker     Trivy+Snyk
        mypy       >80%     Build+Push  +Bandit
        isort      cov      ECR

模型CI (单独流水线):
Push ml/ → [Data Valid] → [Train] → [Evaluate] → [Export] → [Register]
            Great Expect   GPU训练   AUC/KS/PSI   Triton    MLflow
                                     +因果验证
```

#### 12.2.2 CD Pipeline (ArgoCD GitOps)

```
部署策略:
  · Backend/Frontend: 蓝绿部署 (Blue-Green) — 零停机
  · Triton: 金丝雀部署 (Canary, 10% → 50% → 100%) — 风险可控
  · 模型更新: 影子部署 (Shadow, 对比验证后切换) — 安全上线

环境晋升:
  dev → staging → canary → production
   │       │         │          │
   │       │         │          └── 自动回滚 (错误率>1%)
   │       │         └── 人工审批 (模型变更)
   │       └── 自动部署 (代码变更)
   └── 自动部署 (每次Push)
```

### 12.3 监控与告警

#### 12.3.1 三维监控体系

| 维度 | 工具 | 关键指标 | 告警阈值 |
|------|------|----------|----------|
| **应用监控** | Prometheus+Grafana | QPS、延迟P50/P99、错误率 | P99>200ms、错误率>1% |
| **GPU监控** | DCGM Exporter+Triton Metrics | GPU利用率、显存使用、温度 | 利用率<30%缩容、显存>90%扩容、温度>85°C告警 |
| **模型监控** | Evidently+Custom Metrics | AUC、KS、PSI、因果效应稳定性 | AUC<0.75、PSI>0.2、因果效应变化>30% |

#### 12.3.2 漂移检测体系

| 漂移类型 | 检测方法 | 频率 | 响应动作 |
|----------|----------|------|----------|
| 数据漂移 (Covariate) | PSI>0.2、KS p<0.01、Wasserstein | 每日 | 告警+特征审查 |
| 概念漂移 (Concept) | 残差趋势、分段AUC下降、因果效应变化、ADWIN | 每周 | 触发重训练 |
| 预测漂移 (Prediction) | 评分分布PSI、违约率偏移 | 实时 | 降级+人工审查 |

#### 12.3.3 自动化闭环

```
漂移检测 → Prometheus告警 → Alertmanager路由 → Airflow触发重训练
                                                      │
                                                      ▼
                                              数据质量检查 → 特征工程 → 因果发现
                                                      │
                                                      ▼
                                              模型训练 → 验证(AUC+因果) → 注册
                                                      │
                                                      ▼
                                              影子部署 → 对比验证 → 切换上线
```

### 12.4 安全与合规

#### 12.4.1 数据加密

| 层级 | 方案 | 详情 |
|------|------|------|
| 传输加密 | TLS 1.3 + mTLS | 所有外部通信TLS 1.3，服务间通信Istio mTLS |
| 存储加密 | AES-256-GCM + TDE | S3服务端加密、PostgreSQL透明数据加密、Redis at-rest |
| 字段级加密 | AES-256 + FPE | 身份证号AES-256加密、收入格式保留加密 |
| 密钥管理 | AWS KMS | 密钥轮换90天、CloudHSM可选 |

#### 12.4.2 访问控制

| 角色 | 权限 |
|------|------|
| analyst | 查看评分、可解释性报告 |
| risk_manager | 评分+审批+反事实分析 |
| data_scientist | 特征管理+模型实验+因果分析 |
| mlops_engineer | 模型部署+监控+流水线管理 |
| admin | 全部权限+用户管理+审计日志 |

认证方案：OAuth 2.0 + OIDC (Keycloak)、JWT短期令牌(15min过期)、MFA(管理员操作)

#### 12.4.3 审计日志

- **审计事件**：评分请求、模型变更、数据访问、权限变更、反事实查询
- **存储**：PostgreSQL(90天) + S3+Athena(7年归档) + S3 Object Lock(WORM不可篡改)
- **合规报告**：月度审计报告自动生成、GDPR数据删除<72h、个人信息保护法合规检查清单

#### 12.4.4 模型治理

- **Model Card**：模型用途、训练数据、性能指标、公平性评估、因果有效性声明、局限性与使用边界
- **公平性监控**：Demographic Parity + Equalized Odds + 因果公平性三重保障
- **审批流程**：Staging → 公平性测试 → 因果验证 → 性能基准 → 合规审查 → 人工审批 → Production
- **网络隔离**：NetworkPolicy限制Triton仅接受Backend访问、Backend仅接受Ingress Controller访问

### 12.5 性能基准

| 场景 | 指标 | 目标值 |
|------|------|--------|
| 实时评分 | P50延迟 | < 25ms |
| 实时评分 | P99延迟 | < 100ms |
| 实时评分 | 吞吐量 | > 1000 QPS |
| 因果推理 | P99延迟 | < 200ms |
| 反事实生成 | P99延迟 | < 500ms |
| 批量评分 | 吞吐量 | > 50K records/s/GPU |
| SHAP计算 | 100特征 | < 10ms |
| 特征获取 | Redis读取 | < 2ms |
| GPU利用率 | 峰值 | > 80% |
| 模型AUC | Home Credit | > 0.78 |

---

## 附录

### A. 关键设计决策记录

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|----------|------|
| GPU推理框架 | Triton | TensorRT/ONNX Runtime | 因果模型需Python Backend，Triton唯一支持混合部署 |
| 因果发现 | PC+NOTEARS融合 | 单一方法 | 融合提高发现精度，交集策略降低假阳性 |
| 反事实方法 | DiCE+因果约束 | 纯DiCE | 因果约束保证反事实方案的因果合理性 |
| 特征存储 | Feast | Hopsworks | 轻量、与K8s集成好、Redis在线存储性能优 |
| 数据湖格式 | Delta Lake | Iceberg/Hudi | ACID事务+Z-Order优化+Vacuum清理 |
| 前端框架 | Next.js | Nuxt/Vite | SSR+RSC+生态成熟+TypeScript原生支持 |
| 部署策略 | 蓝绿+金丝雀 | 滚动更新 | 零停机+风险可控+模型变更安全 |
| 服务网格 | Istio | Linkerd | mTLS+流量管理+可观测性+生态成熟 |

### B. 参考资料

1. Pearl, J. "The Book of Why." 2018.
2. Microsoft Research. "DoWhy: A Python Library for Causal Inference." 2023.
3. Microsoft Research. "EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation." 2024.
4. Sharma, A. et al. "DoWhy: An End-to-End Library for Causal Inference." KDD 2024.
5. European Commission. "EU AI Act." Entered into force 1 August 2024.
6. CFPB. "Fair Lending Report of the Consumer Financial Protection Bureau." 2024.
7. Precedence Research. "Credit Scoring Market Size." 2024.
8. McKinsey. "The Future of Credit Scoring: AI and Alternative Data." 2024.
9. NVIDIA. "Accelerating Trustworthy AI for Credit Risk Management." 2022.
10. AWS. "Accelerating Fraud Detection with RAPIDS Accelerator for Apache Spark." 2025.

---

> **文档维护**: 本实现计划书随项目迭代持续更新，重大变更需经架构评审委员会审批。  
> **下次评审日期**: 2026-07-05  
> **整合来源**: 阿信-前沿技术调研员（行业调研报告）+ 艾伦-欺诈行为分析师（行为分析报告）+ 苏珊-数据架构师（技术架构蓝图）
