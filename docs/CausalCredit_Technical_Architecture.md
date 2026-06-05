# CausalCredit — 因果推理增强信用评分系统 技术架构蓝图

> **版本**: v1.0 | **日期**: 2026-06-05 | **架构师**: 苏珊-数据架构师  
> **难度评级**: ★★★★☆ (4/5) | **核心亮点**: GPU加速推理 + 因果可解释性

---

## 目录

- [1. 数据架构设计](#1-数据架构设计)
  - [1.1 多源数据接入层设计](#11-多源数据接入层设计)
  - [1.2 特征工程Pipeline设计](#12-特征工程pipeline设计)
  - [1.3 数据质量监控与血缘追踪](#13-数据质量监控与血缘追踪)
- [2. GPU加速推理架构](#2-gpu加速推理架构)
  - [2.1 GPU推理框架选型](#21-gpu推理框架选型)
  - [2.2 模型服务化方案](#22-模型服务化方案)
  - [2.3 GPU资源调度与弹性伸缩方案](#23-gpu资源调度与弹性伸缩方案)
- [3. 项目结构设计](#3-项目结构设计)
  - [3.1 Python后端项目结构](#31-python后端项目结构)
  - [3.2 专业前端技术选型与结构](#32-专业前端技术选型与结构)
  - [3.3 MLOps管线结构](#33-mlops管线结构)
  - [3.4 完整目录树设计](#34-完整目录树设计)
- [4. 模型可解释性技术架构](#4-模型可解释性技术架构)
  - [4.1 SHAP + 因果图联合解释方案](#41-shap--因果图联合解释方案)
  - [4.2 反事实解释生成管线](#42-反事实解释生成管线)
  - [4.3 可解释性结果的可视化呈现架构](#43-可解释性结果的可视化呈现架构)
- [5. 真实应用部署方案](#5-真实应用部署方案)
  - [5.1 容器化部署](#51-容器化部署)
  - [5.2 CI/CD流水线设计](#52-cicd流水线设计)
  - [5.3 监控与告警体系](#53-监控与告警体系)
  - [5.4 安全与合规](#54-安全与合规)

---

## 1. 数据架构设计

### 1.1 多源数据接入层设计

CausalCredit 的数据接入层采用 **Lambda架构** 融合批处理与流处理能力，以统一的数据湖为底座，支撑因果推理所需的深度时序分析与多表关联计算。

#### 1.1.1 数据源全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据源接入全景图                               │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  Home Credit │ Lending Club │  外部数据源   │      实时行为数据       │
│  (8表关联)   │  (单表宽表)   │  (API接入)   │    (事件流采集)        │
├──────────────┼──────────────┼──────────────┼────────────────────────┤
│ application  │ loan_stats   │ 征信中心API  │ 用户点击流              │
│ bureau       │              │ 反欺诈黑名单 │ 设备指纹流              │
│ bureau_bal   │              │ 宏观经济指标 │ 申请行为流              │
│ prev_app     │              │ 社交网络图谱 │ 位置变更流              │
│ pos_cash     │              │ 行业风险指数 │                        │
│ credit_card  │              │ 地理风险评分 │                        │
│ installments │              │ 企业工商数据 │                        │
│ homecredit   │              │              │                        │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
```

#### 1.1.2 Home Credit 多表关联架构

Home Credit 数据集包含8张核心表，其关联关系是因果特征挖掘的关键基础：

```
                    ┌──────────────────┐
                    │  application_    │
                    │  train/test      │  ← 主表（每笔贷款一行）
                    │  SK_ID_CURR (PK) │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                   │
          ▼                  ▼                   ▼
┌─────────────────┐ ┌────────────────┐ ┌──────────────────┐
│   bureau        │ │  previous_     │ │  POS_CASH_       │
│   SK_ID_BUREAU  │ │  application   │ │  balance         │
│   SK_ID_CURR(FK)│ │  SK_ID_PREV(PK)│ │  SK_ID_PREV(FK)  │
└────────┬────────┘ └───────┬────────┘ └──────────────────┘
         │                  │
         ▼                  ├──────────────────┐
┌─────────────────┐         │                  │
│  bureau_        │         ▼                  ▼
│  balance        │ ┌────────────────┐ ┌──────────────────┐
│  SK_ID_BUREAU   │ │  credit_card_  │ │  installments_   │
│  (FK)           │ │  balance       │ │  payments        │
└─────────────────┘ │  SK_ID_PREV    │ │  SK_ID_PREV      │
                    └────────────────┘ └──────────────────┘
```

**关联策略**：
- **1:1 关联**：`application` ← `bureau`（按 `SK_ID_CURR` 聚合，取最新记录或统计摘要）
- **1:N 关联**：`application` ← `previous_application`（按 `SK_ID_CURR` 聚合历史贷款统计特征）
- **深层关联**：`bureau` ← `bureau_balance`（按 `SK_ID_BUREAU` 聚合征信局月度状态序列）
- **时序关联**：`previous_application` ← `POS_CASH_balance` / `credit_card_balance` / `installments_payments`（按 `SK_ID_PREV` 聚合还款行为时序特征）

#### 1.1.3 统一数据接入层架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          数据接入层 (Ingestion Layer)                     │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ Batch       │  │ Stream      │  │ API         │  │ File          │  │
│  │ Ingestion   │  │ Ingestion   │  │ Gateway     │  │ Ingestion     │  │
│  │ (Airflow    │  │ (Flink/     │  │ (FastAPI    │  │ (S3/MinIO     │  │
│  │  DAGs)      │  │  Kafka)     │  │  + Rate     │  │  Event        │  │
│  │             │  │             │  │  Limiter)   │  │  Notification)│  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬────────┘  │
│         │                │                │                │            │
│         ▼                ▼                ▼                ▼            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Schema Registry (Confluent)                   │   │
│  │          统一数据模式管理 · 版本控制 · 兼容性校验                  │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Data Lake (MinIO / S3 + Delta Lake)                 │   │
│  │    Bronze Layer → Silver Layer → Gold Layer (Medallion架构)      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Medallion架构分层策略**：

| 层级 | 命名 | 内容 | 格式 | 保留期 |
|------|------|------|------|--------|
| Bronze | `raw_{source}_{table}` | 原始数据，不做任何清洗 | Parquet + Schema Enforce | 永久 |
| Silver | `clean_{source}_{table}` | 去重、类型转换、缺失值标记 | Delta Lake (ACID) | 2年 |
| Gold | `feat_{domain}_{feature_group}` | 关联聚合后的特征宽表 | Delta Lake (OPTIMIZED) | 1年 |

#### 1.1.4 Lending Club 数据融合方案

Lending Club 作为独立数据源，其单表宽表结构与 Home Credit 多表结构需要统一建模：

```
Lending Club loan_stats
         │
         ▼
┌─────────────────────┐     ┌─────────────────────┐
│  字段映射层          │     │  标准化信用域模型     │
│  (Field Mapping)    │────▶│  (Unified Credit     │
│  - loan_amnt →      │     │   Domain Model)      │
│    credit_amount    │     │  - 申请人画像域       │
│  - int_rate →       │     │  - 贷款属性域         │
│    interest_rate    │     │  - 信用历史域         │
│  - annual_inc →     │     │  - 还款行为域         │
│    annual_income    │     │  - 风险标签域         │
└─────────────────────┘     └─────────────────────┘
```

**跨源对齐策略**：
- **语义对齐**：基于 OWL 本体映射，将不同数据源的字段统一到标准信用域模型
- **粒度对齐**：Home Credit 按 `SK_ID_CURR`（贷款级别）聚合，Lending Club 天然为贷款级别
- **时序对齐**：统一使用 `MONTHS_BALANCE` 相对时间偏移量，消除绝对时间差异
- **标签对齐**：Home Credit `TARGET`（1=违约）↔ Lending Club `loan_status`（映射为二分类）

---

### 1.2 特征工程Pipeline设计

特征工程是 CausalCredit 的核心竞争力，分为三大管线：因果特征挖掘、时序特征提取、交叉特征构造。

#### 1.2.1 特征工程总体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     特征工程 Pipeline 总体架构                            │
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │ Pipeline A:  │   │ Pipeline B:  │   │ Pipeline C:  │                 │
│  │ 因果特征挖掘  │   │ 时序特征提取  │   │ 交叉特征构造  │                 │
│  │              │   │              │   │              │                 │
│  │ · DAG发现    │   │ · 滑动窗口   │   │ · 特征组合   │                 │
│  │ · Do-运算    │   │ · 趋势分解   │   │ · 自动交叉   │                 │
│  │ · 工具变量   │   │ · 周期检测   │   │ · 高阶交互   │                 │
│  │ · 反事实特征 │   │ · 异常检测   │   │ · 领域交叉   │                 │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                 │
│         │                  │                  │                          │
│         ▼                  ▼                  ▼                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                  Feature Store (Feast + Redis)                    │   │
│  │    ┌────────────┐  ┌────────────┐  ┌────────────┐               │   │
│  │    │ Online     │  │ Offline    │  │ Feature    │               │   │
│  │    │ Store      │  │ Store      │  │ Registry   │               │   │
│  │    │ (Redis     │  │ (Delta     │  │ (Feast     │               │   │
│  │    │  Cluster)  │  │  Lake)     │  │  Registry) │               │   │
│  │    └────────────┘  └────────────┘  └────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              特征质量门控 (Feature Quality Gate)                   │   │
│  │    PSI检测 · 特征重要性过滤 · 共线性剔除 · 因果有效性验证          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 1.2.2 Pipeline A: 因果特征挖掘

因果特征挖掘是 CausalCredit 区别于传统信用评分系统的核心创新，目标是识别**真正具有因果效应**的特征，而非仅具有统计相关性的特征。

**Step 1: 因果图发现 (Causal Discovery)**

```python
# 因果图发现流程
class CausalDiscoveryPipeline:
    """
    基于 PC 算法 + NOTEARS 的混合因果发现管线
    
    1. PC算法（约束法）：基于条件独立性测试构建骨架图
    2. NOTEARS（优化法）：基于连续优化的DAG学习
    3. 融合策略：取两种方法的交集边，提高发现精度
    """
    
    def discover(self, data: pd.DataFrame) -> CausalGraph:
        # Phase 1: PC Algorithm - 基于条件独立性
        pc_graph = pc_algorithm(
            data=data,
            alpha=0.01,                    # 显著性水平
            independence_test="fisherz",   # Fisher-Z检验
            max_condition_set=3            # 最大条件集大小
        )
        
        # Phase 2: NOTEARS - 基于连续优化
        notears_graph = notears_linear(
            data=data,
            lambda1=0.1,                   # L1正则化强度
            max_iter=100,
            h_tol=1e-8                     # DAG约束容忍度
        )
        
        # Phase 3: 融合 - 取交集增强置信度
        fused_graph = fuse_causal_graphs(
            graphs=[pc_graph, notears_graph],
            strategy="intersection",       # 交集策略
            edge_confidence_threshold=0.7  # 边置信度阈值
        )
        
        # Phase 4: 领域知识注入
        domain_constraints = CausalConstraints()
        domain_constraints.add_forbidden_edge("TARGET", "AMT_CREDIT")  # 目标不能是原因
        domain_constraints.add_required_edge("AMT_CREDIT", "AMT_ANNUITY")  # 领域确定性
        
        final_graph = apply_constraints(fused_graph, domain_constraints)
        return final_graph
```

**Step 2: 因果效应估计 (Causal Effect Estimation)**

```
因果效应估计方法矩阵
┌──────────────────┬──────────────────────┬──────────────────────┐
│      方法         │       适用场景        │     输出特征          │
├──────────────────┼──────────────────────┼──────────────────────┤
│ DoWhy - Backdoor │ 可观测混淆变量        │ ATE / CATE           │
│ DoWhy - IV       │ 不可观测混淆+工具变量 │ LATE                 │
│ DoWhy - Frontdoor│ 不可观测混淆+中介变量 │ 自然直接效应          │
│ EconML - DML     │ 异质因果效应          │ CATE (条件平均)       │
│ EconML - DR      │ 双重鲁棒估计          │ CATE (鲁棒)          │
│ EconML - Causal  │ 因果森林             │ CATE (非参数)         │
│        Forest    │                      │                      │
└──────────────────┴──────────────────────┴──────────────────────┘
```

**Step 3: 因果特征生成**

```python
class CausalFeatureGenerator:
    """
    基于因果图生成因果增强特征
    """
    
    def generate(self, data: pd.DataFrame, causal_graph: CausalGraph) -> pd.DataFrame:
        features = pd.DataFrame(index=data.index)
        
        # Feature 1: 因果路径强度特征
        # 计算每个变量到TARGET的所有因果路径的累积效应
        for node in causal_graph.get_causal_parents("TARGET"):
            path_effects = []
            for path in causal_graph.get_all_directed_paths(node, "TARGET"):
                effect = 1.0
                for edge in path:
                    effect *= causal_graph.get_edge_weight(edge)
                path_effects.append(effect)
            features[f"causal_path_strength_{node}"] = sum(path_effects)
        
        # Feature 2: 混淆因子调整残差
        # 对每个特征，用其混淆因子做回归，取残差作为"去混淆"特征
        for node in causal_graph.get_causal_parents("TARGET"):
            confounders = causal_graph.get_confounders(node, "TARGET")
            if confounders:
                residual = compute_confounding_residual(
                    treatment=data[node],
                    confounders=data[confounders]
                )
                features[f"deconfounded_{node}"] = residual
        
        # Feature 3: 工具变量特征
        # 利用工具变量提取外生变异
        iv_pairs = causal_graph.get_instrumental_variables("TARGET")
        for iv, treatment in iv_pairs:
            features[f"iv_exogenous_{treatment}"] = extract_iv_component(
                treatment=data[treatment],
                instrument=data[iv]
            )
        
        # Feature 4: 中介效应分解特征
        # 将总效应分解为直接效应和间接效应
        mediators = causal_graph.get_mediators_to_target()
        for treatment, mediator in mediators:
            direct, indirect = decompose_mediation_effect(
                data=data,
                treatment=treatment,
                mediator=mediator,
                outcome="TARGET",
                graph=causal_graph
            )
            features[f"direct_effect_{treatment}_via_{mediator}"] = direct
            features[f"indirect_effect_{treatment}_via_{mediator}"] = indirect
        
        return features
```

#### 1.2.3 Pipeline B: 时序特征提取

时序特征提取聚焦于信用行为的动态演变模式，是识别违约前兆的关键。

```
时序特征提取架构
┌─────────────────────────────────────────────────────────────────┐
│                     时序特征提取引擎                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 1: 基础统计特征 (tsfresh)                        │    │
│  │  · 均值/方差/偏度/峰度 (滑动窗口: 3/6/12月)             │    │
│  │  · 极值/分位数/变异系数                                  │    │
│  │  · 自相关系数 (lag=1,3,6)                               │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │  Layer 2: 趋势与周期特征                                │    │
│  │  · 线性趋势斜率 + R² (OLS回归)                          │    │
│  │  · STL分解: 趋势项 + 季节项 + 残差项                    │    │
│  │  · 变化点检测 (PELT算法)                                │    │
│  │  · 趋势加速度 (二阶差分)                                │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │  Layer 3: 行为模式特征                                  │    │
│  │  · 逾期模式编码 (连续逾期/间歇逾期/恢复模式)             │    │
│  │  · 还款行为序列嵌入 (LSTM Encoder → 32维向量)           │    │
│  │  · DPD (Days Past Due) 演化轨迹特征                     │    │
│  │  · 信用额度使用率变化率                                  │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │  Layer 4: 异常与风险特征                                │    │
│  │  · 异常分数 (Isolation Forest on 时序窗口)              │    │
│  │  · 突变检测 (CUSUM / EWMA控制图)                        │    │
│  │  · 风险累积指标 (加权逾期天数递增率)                     │    │
│  │  · 行为偏离度 (与同类客群均值的马氏距离)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**关键时序特征示例（基于 Home Credit bureau_balance）**：

| 特征名 | 计算逻辑 | 因果含义 |
|--------|----------|----------|
| `bureau_dpd_trend_6m` | 近6个月DPD的线性趋势斜率 | 逾期恶化速度 → 违约因果路径 |
| `bureau_status_transition_entropy` | 状态转移矩阵的信息熵 | 行为不稳定性 → 风险因果指标 |
| `bureau_recovery_rate_trend` | 逾期后恢复率的变化趋势 | 自愈能力 → 保护性因果因子 |
| `bureau_deterioration_acceleration` | DPD变化的二阶差分 | 风险加速效应 → 非线性因果路径 |
| `bureau_pattern_embedding` | LSTM编码的32维行为向量 | 潜在行为模式 → 隐变量因果结构 |

#### 1.2.4 Pipeline C: 交叉特征构造

```
交叉特征构造策略
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐                  │
│  │ 手工交叉 (领域驱动) │  │ 自动交叉 (数据驱动) │                  │
│  ├───────────────────┤  ├───────────────────┤                  │
│  │ 信用额度 × 逾期次数 │  │ AutoInt 多头自注意 │                  │
│  │ 贷款金额 / 年收入   │  │ 力交叉特征学习     │                  │
│  │ 在职年限 × 贷款期限 │  │                   │                  │
│  │ 外部评分 × 内部评分 │  │ DeepFM 特征交叉   │                  │
│  │ 地区风险 × 行业风险 │  │                   │                  │
│  └───────────────────┘  │ DCN-V2 显式交叉   │                  │
│                         └───────────────────┘                  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 因果引导的交叉特征 (Causal-Guided Interaction)            │  │
│  │                                                           │  │
│  │ 原理：仅对因果图中存在共同效应节点(对撞结构)或             │  │
│  │ 存在中介路径的特征对进行交叉，避免虚假交互                 │  │
│  │                                                           │  │
│  │ 示例：                                                     │  │
│  │ · AMT_CREDIT × AMT_ANNUITY (共同影响TARGET，对撞结构)     │  │
│  │ · EXT_SOURCE_1 × EXT_SOURCE_2 (多源评分融合，共同效应)    │  │
│  │ · DAYS_EMPLOYED × AMT_INCOME_TOTAL (因果链中介交互)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.2.5 特征存储与版本管理

```yaml
# Feature Store 配置 (Feast)
feature_store:
  project: causal_credit
  
  registry:
    path: postgresql://postgres:5432/feast_registry
    cache_ttl_seconds: 600
  
  provider: local
  
  online_store:
    type: redis
    redis_type: cluster
    connection_string: "redis-cluster:6379"
    max_connections: 128
  
  offline_store:
    type: delta
    path: s3://causal-credit-datalake/gold/
  
  entity:
    - name: sk_id_curr
      value_type: INT64
      description: "Home Credit 贷款申请唯一标识"
    - name: sk_id_bureau
      value_type: INT64
      description: "征信局记录唯一标识"
  
  feature_views:
    - name: causal_features
      ttl: 86400  # 24小时
      features:
        - name: causal_path_strength_credit
          dtype: FLOAT
        - name: deconfounded_amt_income
          dtype: FLOAT
        - name: iv_exogenous_days_employed
          dtype: FLOAT
    
    - name: temporal_features
      ttl: 3600   # 1小时
      features:
        - name: bureau_dpd_trend_6m
          dtype: FLOAT
        - name: bureau_recovery_rate_trend
          dtype: FLOAT
        - name: bureau_pattern_embedding
          dtype: FLOAT_LIST  # 32维向量
```

---

### 1.3 数据质量监控与血缘追踪

#### 1.3.1 数据质量监控框架

```
数据质量监控架构
┌─────────────────────────────────────────────────────────────────┐
│                    Great Expectations + Custom Rules             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  质量维度              检测规则                          │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  完整性     缺失率阈值 (关键字段 < 1%, 普通字段 < 5%)   │    │
│  │  一致性     跨表外键一致性校验 (SK_ID_CURR关联完整性)    │    │
│  │  时效性     数据到达延迟监控 (SLA: 批处理 < 4h, 流 < 5m) │    │
│  │  准确性     数值范围校验 (AMT_CREDIT > 0, EXT_SOURCE 0-1)│    │
│  │  唯一性     主键唯一性校验 (SK_ID_CURR无重复)            │    │
│  │  分布漂移   PSI > 0.2 告警, KS检验 p < 0.01 告警        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  告警通道                                                │    │
│  │  · Prometheus Alertmanager → Slack/PagerDuty            │    │
│  │  · 质量报告自动生成 (每日/每周)                          │    │
│  │  · 严重质量问题阻断下游Pipeline (Airflow Sensor)         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.3.2 数据血缘追踪

```
数据血缘追踪架构 (Apache Atlas + OpenLineage)
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  数据血缘链路示例:                                               │
│                                                                 │
│  raw_homecredit_application                                     │
│       │                                                         │
│       ▼ [Airflow DAG: etl_application]                          │
│  clean_homecredit_application                                   │
│       │                                                         │
│       ├──▶ [Feature Pipeline A] ──▶ feat_causal_app_features    │
│       │         (DoWhy因果发现)                                   │
│       ├──▶ [Feature Pipeline B] ──▶ feat_temporal_app_features  │
│       │         (tsfresh时序提取)                                 │
│       └──▶ [Feature Pipeline C] ──▶ feat_cross_app_features     │
│                 (AutoInt交叉)                                    │
│                                                                 │
│  血缘追踪能力:                                                   │
│  · 字段级血缘: 精确到每个输出特征由哪些原始字段派生               │
│  · 变换级血缘: 记录每个变换步骤的代码版本与参数                   │
│  · 影响分析: 当源字段变更时，自动分析受影响的下游特征和模型       │
│  · 合规审计: 满足GDPR/个人信息保护法的数据溯源要求               │
│                                                                 │
│  技术实现:                                                       │
│  · OpenLineage: 自动采集Airflow/Spark/Flink的血缘事件           │
│  · Apache Atlas: 血缘图存储与查询                                │
│  · Marquez: 血缘可视化UI                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. GPU加速推理架构

### 2.1 GPU推理框架选型

#### 2.1.1 框架对比分析

| 维度 | NVIDIA Triton Inference Server | TensorRT | ONNX Runtime GPU |
|------|-------------------------------|----------|------------------|
| **定位** | 完整推理服务框架 | 推理优化引擎 | 轻量推理运行时 |
| **模型格式** | ONNX / TensorRT / PyTorch / TensorFlow / Python | TensorRT Engine | ONNX |
| **动态Batch** | ✅ 原生支持动态批处理 | ❌ 需外部实现 | ⚠️ 有限支持 |
| **多模型并发** | ✅ 多模型实例并发 | ❌ 单模型 | ⚠️ 需自行管理 |
| **GPU显存管理** | ✅ 自动管理 + 显存池 | ⚠️ 手动管理 | ⚠️ 手动管理 |
| **模型热更新** | ✅ 无停机更新 | ❌ 需重启 | ❌ 需重启 |
| **推理延迟** | 极低 (TensorRT后端) | 最低 | 低 |
| **吞吐量** | 最高 (动态Batch) | 高 (单请求) | 中 |
| **因果模型支持** | ✅ Python Backend (DoWhy/EconML) | ❌ 不支持自定义算子 | ⚠️ 有限 |
| **运维复杂度** | 中 | 高 | 低 |

#### 2.1.2 选型决策：NVIDIA Triton Inference Server

**选择 Triton 作为核心推理框架，理由如下**：

1. **因果推理模型兼容性**：CausalCredit 的因果推理组件（DoWhy/EconML）无法直接编译为 TensorRT Engine，Triton 的 Python Backend 完美支持
2. **混合推理架构**：可在同一服务中同时部署 TensorRT 优化的 XGBoost/LightGBM 模型和 Python Backend 的因果推理模型
3. **动态批处理**：信用评分场景存在明显的请求波峰（工作日9-11点），动态批处理可显著提升吞吐
4. **生产级特性**：健康检查、指标暴露、模型版本管理、A/B测试等开箱即用

```
Triton 混合推理架构
┌──────────────────────────────────────────────────────────────────────┐
│                  NVIDIA Triton Inference Server                      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Model Repository                            │  │
│  │                                                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│  │
│  │  │ credit_score │  │ causal_      │  │ counterfactual_      ││  │
│  │  │ _xgboost     │  │ inference    │  │ generator            ││  │
│  │  │              │  │ _dowy        │  │                      ││  │
│  │  │ Backend:     │  │              │  │ Backend:             ││  │
│  │  │ TensorRT     │  │ Backend:     │  │ Python               ││  │
│  │  │              │  │ Python       │  │                      ││  │
│  │  │ Latency:     │  │              │  │ Latency:             ││  │
│  │  │ ~2ms         │  │ Latency:     │  │ ~50ms                ││  │
│  │  │              │  │ ~15ms        │  │                      ││  │
│  │  │ GPU: A10G    │  │ GPU: A10G    │  │ GPU: A10G            ││  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘│  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Dynamic Batcher          Instance Group         Scheduler     │  │
│  │  ┌──────────────────┐    ┌──────────────────┐   ┌──────────┐ │  │
│  │  │ max_batch_size:  │    │ credit_score:    │   │ priority │ │  │
│  │  │   64             │    │   count: 2       │   │ based    │ │  │
│  │  │ max_queue_delay: │    │   gpu: 1         │   │          │ │  │
│  │  │   5000μs         │    │ causal:          │   │ 实时 >   │ │  │
│  │  │ preferred_batch: │    │   count: 1       │   │ 批量     │ │  │
│  │  │   [8,16,32]      │    │   gpu: 1         │   │          │ │  │
│  │  └──────────────────┘    └──────────────────┘   └──────────┘ │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 2.1.3 Triton 模型配置示例

```protobuf
# config.pbtxt - credit_score_xgboost (TensorRT优化)
name: "credit_score_xgboost"
platform: "tensorrt_plan"
max_batch_size: 64

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000
  priority_levels: 2
  default_priority_level: 1
}

optimization {
  cuda {
    graphs: true
    input_copy: true
    output_copy: true
  }
}

response_cache {
  enable: true
  cache_size: 1024
}
```

```protobuf
# config.pbtxt - causal_inference_dowy (Python Backend)
name: "causal_inference_dowy"
backend: "python"
max_batch_size: 16

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 10000
}

input [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [-1]  # 动态特征维度
  }
]

output [
  {
    name: "causal_effect"
    data_type: TYPE_FP32
    dims: [-1]
  },
  {
    name: "causal_graph_adj"
    data_type: TYPE_FP32
    dims: [-1, -1]
  }
]
```

---

### 2.2 模型服务化方案

#### 2.2.1 实时推理架构

实时推理面向信贷审批场景，要求 P99 延迟 < 100ms。

```
实时推理架构 (Real-time Inference)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  客户端请求                                                          │
│  (信贷审批系统)                                                      │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────┐                       │
│  │          API Gateway (Kong)               │                       │
│  │   · 限流: 1000 req/s per tenant          │                       │
│  │   · 认证: JWT + mTLS                     │                       │
│  │   · 路由: /v1/score → Score Service      │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     │                                                │
│                     ▼                                                │
│  ┌──────────────────────────────────────────┐                       │
│  │     Score Service (FastAPI)               │                       │
│  │                                           │                       │
│  │  1. 特征获取 (Feast Online Store)         │                       │
│  │     └─ Redis Cluster: ~2ms               │                       │
│  │  2. 特征预处理 (标准化/编码)              │                       │
│  │     └─ NumPy GPU: ~1ms                   │                       │
│  │  3. 模型推理 (Triton gRPC)               │                       │
│  │     ├─ credit_score: ~2ms (TensorRT)     │                       │
│  │     ├─ causal_inference: ~15ms (Python)  │                       │
│  │     └─ counterfactual: ~50ms (Python)    │                       │
│  │  4. 结果融合与决策                        │                       │
│  │     └─ 加权融合: ~1ms                    │                       │
│  │                                           │                       │
│  │  总延迟: P50 ~25ms, P99 ~80ms            │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     │                                                │
│                     ▼                                                │
│  ┌──────────────────────────────────────────┐                       │
│  │     NVIDIA Triton (gRPC Endpoint)         │                       │
│  │     triton-server:8001                    │                       │
│  │                                           │                       │
│  │  ┌─────────┐ ┌─────────┐ ┌────────────┐ │                       │
│  │  │ Model A │ │ Model B │ │  Model C   │ │                       │
│  │  │ Score   │ │ Causal  │ │ Counter-   │ │                       │
│  │  │ (TRT)   │ │ (Py)    │ │ factual(Py)│ │                       │
│  │  └─────────┘ └─────────┘ └────────────┘ │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                      │
│  ┌──────────────────────────────────────────┐                       │
│  │     异步可解释性 (Celery Worker)           │                       │
│  │     · SHAP值计算 (非实时, 异步)           │                       │
│  │     · 因果图渲染 (非实时, 异步)           │                       │
│  │     · 反事实报告生成 (非实时, 异步)       │                       │
│  └──────────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
```

**实时推理关键优化**：

| 优化项 | 技术手段 | 效果 |
|--------|----------|------|
| 特征预取 | Feast Materialization + Redis Pipeline | 特征获取 2ms → 0.5ms |
| CUDA Graphs | Triton CUDA Graphs优化 | GPU Kernel Launch开销降低60% |
| 动态Batch | Triton Dynamic Batcher | 吞吐量提升3-5x |
| 模型预热 | Triton Warmup | 首次推理延迟降低80% |
| 响应缓存 | Triton Response Cache | 重复请求直接返回 |
| 异步可解释 | SHAP/因果图异步计算 | 不阻塞主评分路径 |

#### 2.2.2 批量推理架构

批量推理面向存量客户批量重评分、模型回测等场景，追求吞吐量最大化。

```
批量推理架构 (Batch Inference)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Airflow DAG: batch_scoring                                    │  │
│  │                                                                │  │
│  │  [Trigger] → [Load Features] → [Batch Predict] → [Save Results]│  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                         │                                            │
│                         ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Spark on Kubernetes (批量特征计算)                             │  │
│  │                                                                │  │
│  │  · 从 Delta Lake Gold 层读取特征宽表                           │  │
│  │  · 分区并行: 按 SK_ID_CURR hash 分区                          │  │
│  │  · 每分区 10000 条, 并行度 = GPU数 × 4                        │  │
│  │  · 输出: Parquet → Delta Lake                                  │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │                                        │
│                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Triton Batch Inference (HTTP/gRPC Bulk API)                   │  │
│  │                                                                │  │
│  │  · 使用 Triton 的 C++ 客户端批量发送请求                      │  │
│  │  · max_batch_size = 1024 (批量模式)                            │  │
│  │  · 多流并行: 4 CUDA Streams per GPU                           │  │
│  │  · 吞吐量: ~50,000 records/s per A10G GPU                     │  │
│  │                                                                │  │
│  │  GPU利用率优化:                                                │  │
│  │  · 批量大小自动调优 (Binary Search for optimal batch)          │  │
│  │  · FP16/INT8 量化 (XGBoost → TensorRT INT8)                  │  │
│  │  · 多GPU数据并行 (All-reduce gradient sync)                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  性能基准 (100万条记录, A10G × 2):                                   │
│  · 纯评分模型: ~20s (50K/s)                                         │
│  · 评分 + 因果推理: ~120s (8.3K/s)                                  │
│  · 评分 + 因果 + 反事实: ~600s (1.7K/s)                             │
└──────────────────────────────────────────────────────────────────────┘
```

#### 2.2.3 实时 vs 批量架构差异总结

| 维度 | 实时推理 | 批量推理 |
|------|----------|----------|
| **请求模式** | 单条/小批量, 低延迟优先 | 大批量, 高吞吐优先 |
| **特征来源** | Feast Online Store (Redis) | Delta Lake Gold Layer |
| **Batch Size** | 1-64 (动态) | 512-4096 (固定) |
| **延迟要求** | P99 < 100ms | 无严格要求 |
| **GPU利用率** | 30-60% (波动) | 90%+ (稳定) |
| **可解释性** | 异步计算, 后补 | 同步计算, 全量 |
| **容错** | 重试 + 降级 | Checkpoint + 重跑 |
| **调度** | K8s HPA (按QPS) | Airflow + Spark (按计划) |

---

### 2.3 GPU资源调度与弹性伸缩方案

#### 2.3.1 GPU资源池架构

```
GPU资源池架构
┌──────────────────────────────────────────────────────────────────────┐
│                   Kubernetes GPU Cluster                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Node Pool: gpu-inference (按需伸缩)                           │  │
│  │                                                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │  │
│  │  │ GPU Node 1   │  │ GPU Node 2   │  │ GPU Node N   │        │  │
│  │  │ A10G (24GB)  │  │ A10G (24GB)  │  │ A10G (24GB)  │        │  │
│  │  │              │  │              │  │              │        │  │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │        │  │
│  │  │ │ Triton   │ │  │ │ Triton   │ │  │ │ Triton   │ │        │  │
│  │  │ │ Pod      │ │  │ │ Pod      │ │  │ │ Pod      │ │        │  │
│  │  │ │ GPU: 1   │ │  │ │ GPU: 1   │ │  │ │ GPU: 1   │ │        │  │
│  │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │        │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │  │
│  │                                                                │  │
│  │  Min: 2 Nodes  |  Max: 8 Nodes  |  Default: 3 Nodes          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Node Pool: gpu-training (定期训练)                            │  │
│  │  · A100 (80GB) × 2-4 Nodes                                    │  │
│  │  · Spot Instance (低成本)                                      │  │
│  │  · 训练完成后自动缩容                                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  GPU 调度策略 (NVIDIA GPU Operator + MIG)                      │  │
│  │                                                                │  │
│  │  · MIG (Multi-Instance GPU): A10G 切分为 1g.6gb × 4 实例      │  │
│  │  · 时间分片: 低优先级任务使用时间分片共享GPU                    │  │
│  │  · 显存隔离: 每个Pod独占GPU显存, 避免OOM干扰                  │  │
│  │  · 优先级: 实时推理 > 批量推理 > 模型训练                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 弹性伸缩策略

```yaml
# HPA - 基于 QPS 和 GPU 利用率的混合伸缩
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-inference
  minReplicas: 2
  maxReplicas: 8
  metrics:
    # 指标1: 自定义QPS指标 (来自Prometheus)
    - type: Pods
      pods:
        metric:
          name: triton_inference_request_count
          selector:
            matchLabels:
              model: credit_score_xgboost
        target:
          type: AverageValue
          averageValue: "500"   # 每Pod 500 QPS触发扩容
    
    # 指标2: GPU利用率
    - type: Pods
      pods:
        metric:
          name: DCGM_FI_DEV_GPU_UTIL
        target:
          type: AverageValue
          averageValue: "70"    # GPU利用率 > 70% 触发扩容
    
    # 指标3: 推理队列深度
    - type: Pods
      pods:
        metric:
          name: triton_inference_queue_size
        target:
          type: AverageValue
          averageValue: "32"    # 队列深度 > 32 触发扩容
  
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # 缩容冷静期5分钟
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
```

#### 2.3.3 GPU显存优化策略

```
GPU显存优化技术栈
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. 模型量化 (Quantization)                                     │
│     · XGBoost/LightGBM → ONNX → TensorRT INT8                 │
│     · 因果推理模型: FP32 → FP16 (自动混合精度)                 │
│     · 显存节省: ~50-75%                                         │
│                                                                 │
│  2. 模型卸载 (Model Offloading)                                 │
│     · 低频模型 (反事实生成) → CPU内存驻留                       │
│     · 按需加载到GPU (首次请求触发, ~500ms冷启动)               │
│     · Triton 热卸载: 空闲 > 10min 自动卸载                     │
│                                                                 │
│  3. 显存池化 (Memory Pooling)                                   │
│     · Triton CUDA Memory Pool: 预分配显存池                     │
│     · 避免频繁malloc/free导致的显存碎片                         │
│     · 配置: --cuda-memory-pool-byte-size=0:1073741824          │
│       (GPU 0 预分配 1GB 显存池)                                 │
│                                                                 │
│  4. MIG 切分                                                    │
│     · A10G 24GB → 1g.6gb × 4 实例                              │
│     · 评分模型独占1实例, 因果推理独占1实例                      │
│     · 反事实生成与SHAP计算共享1实例                              │
│     · 预留1实例给突发流量                                       │
│                                                                 │
│  显存分配方案 (A10G 24GB):                                      │
│  ┌──────────────────────────────────────────────────┐           │
│  │  TensorRT评分模型:    2GB (INT8量化)             │           │
│  │  因果推理模型:        4GB (FP16)                 │           │
│  │  反事实生成模型:      3GB (FP16)                 │           │
│  │  CUDA Memory Pool:    1GB (预分配)               │           │
│  │  CUDA Graphs缓存:    2GB                        │           │
│  │  输入/输出缓冲区:     2GB                        │           │
│  │  系统预留:           10GB                        │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 项目结构设计

### 3.1 Python后端项目结构

#### 3.1.1 技术栈选型

| 组件 | 技术选型 | 选型理由 |
|------|----------|----------|
| **Web框架** | FastAPI 0.110+ | 异步高性能、自动OpenAPI文档、类型安全 |
| **任务队列** | Celery 5.4+ + Redis 7.x | 成熟的异步任务方案，支持优先级队列与任务链 |
| **缓存** | Redis 7.x Cluster | 亚毫秒级延迟，支持Pub/Sub与Stream |
| **数据库** | PostgreSQL 16 + Citus | 分布式扩展、JSONB支持、Citus水平分片 |
| **ORM** | SQLAlchemy 2.0 + Alembic | 异步ORM、类型安全迁移 |
| **消息队列** | Apache Kafka 3.7 | 事件驱动架构、Exactly-Once语义 |
| **gRPC客户端** | tritonclient 2.x | Triton推理服务原生客户端 |

#### 3.1.2 后端分层架构

```
后端分层架构
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  API Layer (FastAPI Routers)                            │    │
│  │  · /api/v1/score        - 信用评分                      │    │
│  │  · /api/v1/explain      - 可解释性查询                  │    │
│  │  · /api/v1/counterfactual - 反事实分析                  │    │
│  │  · /api/v1/models       - 模型管理                      │    │
│  │  · /api/v1/features     - 特征查询                      │    │
│  │  · /api/v1/monitoring   - 监控指标                      │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │  Service Layer (Business Logic)                         │    │
│  │  · ScoreService         - 评分编排                       │    │
│  │  · CausalService        - 因果推理编排                   │    │
│  │  · ExplainService       - 可解释性编排                   │    │
│  │  · FeatureService       - 特征获取与缓存                 │    │
│  │  · ModelService         - 模型版本管理                   │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │  Integration Layer (External Clients)                   │    │
│  │  · TritonClient         - GPU推理调用                    │    │
│  │  · FeastClient          - 特征存储访问                   │    │
│  │  · RedisClient          - 缓存与分布式锁                 │    │
│  │  · KafkaProducer        - 事件发布                       │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │  Data Layer (Repositories)                              │    │
│  │  · ScoreRepository      - 评分记录持久化                 │    │
│  │  · ModelRepository      - 模型元数据管理                 │    │
│  │  · AuditRepository      - 审计日志存储                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.2 专业前端技术选型与结构

#### 3.2.1 前端技术栈

| 组件 | 技术选型 | 选型理由 |
|------|----------|----------|
| **框架** | Next.js 14 (App Router) | SSR/SSG、React Server Components |
| **语言** | TypeScript 5.4+ | 类型安全、IDE智能提示 |
| **UI库** | Ant Design Pro 6.x | 企业级中后台组件、ProTable/ProForm |
| **图表** | ECharts 5.x + @antv/G6 | 因果图可视化(G6)、业务图表(ECharts) |
| **状态管理** | Zustand + TanStack Query | 轻量状态 + 服务端缓存 |
| **API层** | tRPC (端到端类型安全) | 前后端类型共享、自动类型推导 |

#### 3.2.2 前端页面架构

```
前端页面架构
┌─────────────────────────────────────────────────────────────────┐
│                    CausalCredit Dashboard                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  📊 评分总览 (Score Overview)                            │   │
│  │  · 实时评分量/成功率/平均延迟                             │   │
│  │  · 评分分布直方图 + 风险等级饼图                          │   │
│  │  · 趋势图: 24h评分量 + 模型性能                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  🔍 单笔评分详情 (Score Detail)                          │   │
│  │  · 申请人信息卡片                                         │   │
│  │  · 信用评分仪表盘 (0-1000)                               │   │
│  │  · Top-10 特征贡献 (SHAP瀑布图)                          │   │
│  │  · 因果路径图 (G6交互式因果图)                            │   │
│  │  · 反事实建议面板                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  🧠 因果分析 (Causal Analysis)                           │   │
│  │  · 全局因果图 (交互式DAG)                                 │   │
│  │  · 因果效应热力图 (特征×干预→效应)                        │   │
│  │  · 反事实模拟器 (滑块调整特征值, 实时预测)                │   │
│  │  · 因果路径排名                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  📈 模型监控 (Model Monitoring)                          │   │
│  │  · 特征漂移监控 (PSI趋势图)                              │   │
│  │  · 模型性能趋势 (AUC/KS/Gini)                            │   │
│  │  · GPU利用率仪表盘                                       │   │
│  │  · 告警历史与规则管理                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ⚙️ 系统管理 (Admin)                                     │   │
│  │  · 模型版本管理 (A/B测试配置)                             │   │
│  │  · 特征管理 (上线/下线/监控)                              │   │
│  │  · 用户权限 (RBAC)                                       │   │
│  │  · 审计日志查询                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.3 MLOps管线结构

#### 3.3.1 MLOps技术栈

| 组件 | 技术选型 | 用途 |
|------|----------|------|
| **实验追踪** | MLflow 2.x | 模型实验记录、参数对比、模型注册 |
| **数据版本** | DVC 3.x + Delta Lake | 数据集版本管理、数据血缘 |
| **编排调度** | Apache Airflow 2.8+ | Pipeline编排、定时调度、依赖管理 |
| **模型注册** | MLflow Model Registry | 模型版本管理、阶段转换(Staging→Production) |
| **模型服务** | Triton Inference Server | GPU加速推理服务 |
| **特征存储** | Feast | 特征版本管理、在线/离线一致性 |

#### 3.3.2 MLOps Pipeline 全景

```
MLOps Pipeline 全景图
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────┐  │
│  │  Data   │    │ Feature  │    │  Model   │    │   Model       │  │
│  │  Ingest │───▶│ Engineer │───▶│ Training │───▶│   Validation  │  │
│  │  (DVC)  │    │ (Feast)  │    │(MLflow)  │    │ (Great        │  │
│  │         │    │          │    │          │    │  Expectations)│  │
│  └─────────┘    └──────────┘    └──────────┘    └───────┬───────┘  │
│                                                         │          │
│           ┌─────────────────────────────────────────────┘          │
│           │                                                        │
│           ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Model      │    │   Model      │    │   Model      │        │
│  │   Registry   │───▶│   Deploy     │───▶│   Monitor    │        │
│  │   (MLflow)   │    │ (Triton+K8s) │    │ (Prometheus  │        │
│  │              │    │              │    │  + Evidently) │        │
│  └──────────────┘    └──────────────┘    └──────┬───────┘        │
│                                                   │                │
│                                                   │ 漂移告警       │
│                                                   ▼                │
│                                          ┌──────────────┐         │
│                                          │  Retrigger   │         │
│                                          │  Retraining  │         │
│                                          │  (Airflow)   │─────────┘
│                                          └──────────────┘  (闭环)
└──────────────────────────────────────────────────────────────────────┘
```

#### 3.3.3 Airflow DAG 设计

```python
# causal_credit_training_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['mlops@causalcredit.ai'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='causal_credit_training',
    default_args=default_args,
    schedule_interval='0 2 * * 0',  # 每周日凌晨2点
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['mlops', 'training', 'causal'],
) as dag:

    # Task 1: 数据质量检查
    data_quality_check = PythonOperator(
        task_id='data_quality_check',
        python_callable=run_great_expectations,
        op_kwargs={
            'suite': 'causal_credit_data_suite',
            'fail_on_error': True,
        }
    )

    # Task 2: 特征工程 (Spark on K8s)
    feature_engineering = KubernetesPodOperator(
        task_id='feature_engineering',
        name='feature-engineering',
        namespace='mlops',
        image='causalcredit/feature-pipeline:latest',
        cmds=['python', '-m', 'features.run_pipeline'],
        arguments=['--mode=full', '--output=delta://gold/features'],
        resources={
            'request_memory': '8Gi',
            'request_cpu': '4',
            'limit_memory': '16Gi',
            'limit_cpu': '8',
        }
    )

    # Task 3: 因果图发现
    causal_discovery = PythonOperator(
        task_id='causal_discovery',
        python_callable=run_causal_discovery,
        op_kwargs={
            'methods': ['pc', 'notears'],
            'fusion_strategy': 'intersection',
        }
    )

    # Task 4: 模型训练 (GPU)
    model_training = KubernetesPodOperator(
        task_id='model_training',
        name='model-training',
        namespace='mlops',
        image='causalcredit/model-training:latest',
        cmds=['python', '-m', 'models.train'],
        arguments=['--config=/config/training_config.yaml'],
        resources={
            'request_memory': '32Gi',
            'request_cpu': '8',
            'limit_memory': '64Gi',
            'limit_cpu': '16',
            'nvidia_gpu': {'limit': 1, 'request': 1},
        }
    )

    # Task 5: 模型验证
    model_validation = PythonOperator(
        task_id='model_validation',
        python_callable=validate_model,
        op_kwargs={
            'metrics': ['auc', 'ks', 'gini', 'psi'],
            'thresholds': {'auc': 0.78, 'ks': 0.45, 'psi': 0.15},
        }
    )

    # Task 6: 因果有效性验证
    causal_validation = PythonOperator(
        task_id='causal_validation',
        python_callable=validate_causal_effects,
        op_kwargs={
            'refutation_methods': ['placebo', 'random_cause', 'data_subset'],
        }
    )

    # Task 7: 模型注册
    model_registration = PythonOperator(
        task_id='model_registration',
        python_callable=register_model_to_mlflow,
        op_kwargs={
            'stage': 'Staging',
            'description': 'Weekly retrained model',
        }
    )

    # Task 8: 影子部署 (Shadow Deployment)
    shadow_deploy = PythonOperator(
        task_id='shadow_deploy',
        python_callable=deploy_shadow_model,
        op_kwargs={
            'traffic_percentage': 10,
            'duration_hours': 48,
        }
    )

    # 依赖关系
    data_quality_check >> feature_engineering >> causal_discovery
    causal_discovery >> model_training >> [model_validation, causal_validation]
    [model_validation, causal_validation] >> model_registration >> shadow_deploy
```

---

### 3.4 完整目录树设计

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

## 4. 模型可解释性技术架构

### 4.1 SHAP + 因果图联合解释方案

CausalCredit 的可解释性架构核心创新在于 **SHAP统计归因 + 因果图逻辑归因** 的双层联合解释，既回答"哪些特征重要"，又回答"为什么重要（因果路径）"。

#### 4.1.1 双层解释架构

```
SHAP + 因果图联合解释架构
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Layer 1: SHAP统计归因层 (What matters?)                      │  │
│  │                                                                │  │
│  │  输入: 特征向量 x                                             │  │
│  │  输出: 每个特征的SHAP值 φᵢ                                   │  │
│  │                                                                │  │
│  │  ┌──────────────────────────────────────────────────────┐     │  │
│  │  │  TreeSHAP (XGBoost/LightGBM)                        │     │  │
│  │  │  · 精确SHAP值计算 (多项式时间)                       │     │  │
│  │  │  · 全局解释: SHAP Summary Plot                       │     │  │
│  │  │  · 局部解释: SHAP Waterfall Plot                     │     │  │
│  │  │  · 交互解释: SHAP Interaction Values                 │     │  │
│  │  └──────────────────────────────────────────────────────┘     │  │
│  │                                                                │  │
│  │  SHAP值含义: φᵢ = E[f(x) | xᵢ] - E[f(x)]                   │  │
│  │  即: 在已知特征i取值xᵢ时，模型预测的期望变化量               │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │                                        │
│                             │ SHAP Top-K特征                         │
│                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Layer 2: 因果图逻辑归因层 (Why does it matter?)              │  │
│  │                                                                │  │
│  │  输入: SHAP Top-K特征 + 因果图 G                              │  │
│  │  输出: 因果路径解释 + 效应分解                                 │  │
│  │                                                                │  │
│  │  ┌──────────────────────────────────────────────────────┐     │  │
│  │  │  因果路径追踪                                        │     │  │
│  │  │  对每个SHAP Top-K特征, 在因果图中追踪:               │     │  │
│  │  │  · 直接因果路径: Xᵢ → Y                             │     │  │
│  │  │  · 间接因果路径: Xᵢ → M₁ → ... → Mₖ → Y           │     │  │
│  │  │  · 混淆路径: Xᵢ ← C → Y (虚假相关)                 │     │  │
│  │  │  · 对撞路径: Xᵢ → Z ← Y (条件后虚假)               │     │  │
│  │  └──────────────────────────────────────────────────────┘     │  │
│  │                                                                │  │
│  │  ┌──────────────────────────────────────────────────────┐     │  │
│  │  │  效应分解                                            │     │  │
│  │  │  总效应 = 直接效应 + 间接效应                        │     │  │
│  │  │  · 直接效应: do(Xᵢ=xᵢ') vs do(Xᵢ=xᵢ) on Y         │     │  │
│  │  │  · 间接效应: 通过中介变量M的传递效应                 │     │  │
│  │  │  · SHAP值与因果效应的一致性校验:                     │     │  │
│  │  │    若 |φᵢ| > 0 但 因果效应 ≈ 0 → 标记为"虚假相关"   │     │  │
│  │  │    若 |φᵢ| ≈ 0 但 因果效应 > 0 → 标记为"遮蔽效应"   │     │  │
│  │  └──────────────────────────────────────────────────────┘     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  联合解释输出                                                  │  │
│  │                                                                │  │
│  │  {                                                             │  │
│  │    "shap_values": {"EXT_SOURCE_1": -0.15, "AMT_CREDIT": 0.08},│  │
│  │    "causal_paths": {                                           │  │
│  │      "EXT_SOURCE_1": {                                        │  │
│  │        "path_type": "direct_causal",                          │  │
│  │        "effect": -0.12,                                       │  │
│  │        "consistency": 0.80,  // SHAP与因果效应一致性          │  │
│  │        "interpretation": "外部评分降低直接导致违约概率上升"   │  │
│  │      },                                                       │  │
│  │      "AMT_CREDIT": {                                          │  │
│  │        "path_type": "mediated",                               │  │
│  │        "mediators": ["AMT_ANNUITY", "DEBT_RATIO"],            │  │
│  │        "direct_effect": 0.03,                                 │  │
│  │        "indirect_effect": 0.05,                               │  │
│  │        "consistency": 0.75,                                   │  │
│  │        "interpretation": "贷款金额通过还款压力间接增加违约风险"│  │
│  │      }                                                        │  │
│  │    }                                                          │  │
│  │  }                                                            │  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 SHAP-因果一致性校验算法

```python
class SHAPCausalConsistencyChecker:
    """
    校验SHAP统计归因与因果逻辑归因的一致性
    不一致情况揭示模型可能存在的偏差或虚假相关
    """
    
    def check(self, shap_values: dict, causal_effects: dict, 
              causal_graph: CausalGraph) -> List[ConsistencyReport]:
        reports = []
        
        for feature, shap_val in shap_values.items():
            causal_effect = causal_effects.get(feature, 0)
            
            # Case 1: SHAP显著 + 因果显著 + 方向一致 → 可信特征
            if abs(shap_val) > self.shap_threshold and \
               abs(causal_effect) > self.causal_threshold and \
               sign(shap_val) == sign(causal_effect):
                reports.append(ConsistencyReport(
                    feature=feature,
                    status="TRUSTED",
                    shap_value=shap_val,
                    causal_effect=causal_effect,
                    consistency_score=min(abs(shap_val), abs(causal_effect)) / 
                                     max(abs(shap_val), abs(causal_effect)),
                    action="NO_ACTION"
                ))
            
            # Case 2: SHAP显著 + 因果不显著 → 虚假相关警告
            elif abs(shap_val) > self.shap_threshold and \
                 abs(causal_effect) <= self.causal_threshold:
                # 检查是否存在混淆路径
                has_confounder = causal_graph.has_confounder(feature, "TARGET")
                reports.append(ConsistencyReport(
                    feature=feature,
                    status="SPURIOUS_CORRELATION",
                    shap_value=shap_val,
                    causal_effect=causal_effect,
                    consistency_score=0.0,
                    action="FLAG_FOR_REVIEW",
                    detail=f"SHAP={shap_val:.4f}但因果效应≈0, "
                           f"{'存在混淆因子' if has_confounder else '可能为虚假相关'}"
                ))
            
            # Case 3: SHAP不显著 + 因果显著 → 遮蔽效应
            elif abs(shap_val) <= self.shap_threshold and \
                 abs(causal_effect) > self.causal_threshold:
                reports.append(ConsistencyReport(
                    feature=feature,
                    status="MASKED_EFFECT",
                    shap_value=shap_val,
                    causal_effect=causal_effect,
                    consistency_score=0.0,
                    action="INVESTIGATE_MASKING",
                    detail=f"因果效应={causal_effect:.4f}但SHAP≈0, "
                           f"可能被其他特征遮蔽"
                ))
        
        return reports
```

---

### 4.2 反事实解释生成管线

反事实解释回答"如果改变什么条件，信用评分会提升到安全区间"，是面向用户的最直观解释形式。

#### 4.2.1 反事实生成架构

```
反事实解释生成管线
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  输入: 申请人特征向量 x₀, 当前评分 s₀, 目标评分 s_target           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Step 1: 可行特征识别                                          │  │
│  │                                                                │  │
│  │  ┌──────────────────────────────────────────────────────┐     │  │
│  │  │  特征可变性分类:                                      │     │  │
│  │  │  · 不可变特征 (Immutable): 年龄、性别、教育历史       │     │  │
│  │  │  · 半可变特征 (Semi-mutable): 收入、在职年限          │     │  │
│  │  │  · 可变特征 (Mutable): 贷款金额、贷款期限、信用卡数   │     │  │
│  │  │                                                      │     │  │
│  │  │  因果约束注入:                                        │     │  │
│  │  │  · 若 X₁ → X₂ 在因果图中, 则改变X₁需传播到X₂       │     │  │
│  │  │  · 例: 增加AMT_CREDIT → AMT_ANNUITY也需相应调整     │     │  │
│  │  └──────────────────────────────────────────────────────┘     │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │                                        │
│  ┌──────────────────────────▼─────────────────────────────────────┐  │
│  │  Step 2: 多目标优化 (DiCE + 因果约束)                         │  │
│  │                                                                │  │
│  │  目标函数:                                                     │  │
│  │    min  λ₁·dist(x_cf, x₀) + λ₂·|s(x_cf) - s_target|         │  │
│  │         + λ₃·causal_violation(x_cf, G)                        │  │
│  │                                                                │  │
│  │  其中:                                                         │  │
│  │  · dist(): 特征变化距离 (加权L1/L2, 不可变特征权重=∞)        │  │
│  │  · s(): 信用评分模型                                           │  │
│  │  · causal_violation(): 因果图约束违反惩罚项                    │  │
│  │  · λ₁,λ₂,λ₃: 超参数 (可调)                                   │  │
│  │                                                                │  │
│  │  优化方法:                                                     │  │
│  │  · DiCE (Diverse Counterfactual Explanations)                 │  │
│  │  · 遗传算法 (NSGA-II 多目标帕累托前沿)                        │  │
│  │  · 梯度优化 (可微模型)                                        │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │                                        │
│  ┌──────────────────────────▼─────────────────────────────────────┐  │
│  │  Step 3: 反事实多样性生成                                     │  │
│  │                                                                │  │
│  │  生成K=5个多样化的反事实方案:                                  │  │
│  │  · 方案1: 最小变化 (仅调整1-2个特征)                          │  │
│  │  · 方案2: 收入路径 (提高收入+降低负债比)                      │  │
│  │  · 方案3: 贷款结构路径 (调整贷款金额+期限)                    │  │
│  │  · 方案4: 信用历史路径 (改善逾期记录)                          │  │
│  │  · 方案5: 综合路径 (多维度微调)                                │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │                                        │
│  ┌──────────────────────────▼─────────────────────────────────────┐  │
│  │  Step 4: 因果合理性验证                                       │  │
│  │                                                                │  │
│  │  对每个反事实方案:                                             │  │
│  │  · 验证因果图一致性: 改变是否沿因果路径传播                    │  │
│  │  · 验证反事实效应: do(x_cf) 的因果效应是否与预测一致          │  │
│  │  · 验证可行性: 反事实值是否在训练数据分布内                    │  │
│  │  · 验证鲁棒性: 微小扰动下反事实是否稳定                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  输出: 排序后的反事实方案列表 (含因果路径解释)                      │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 反事实生成核心代码

```python
class CausalCounterfactualGenerator:
    """
    因果约束的反事实解释生成器
    基于 DiCE + 因果图约束的混合方法
    """
    
    def generate(
        self,
        original_features: Dict[str, float],
        model: Any,
        causal_graph: CausalGraph,
        target_score: float,
        n_counterfactuals: int = 5
    ) -> List[CounterfactualResult]:
        
        # 1. 识别可变特征
        mutable_features = self._get_mutable_features(causal_graph)
        
        # 2. 构建因果约束
        causal_constraints = self._build_causal_constraints(causal_graph)
        
        # 3. DiCE配置
        dice_config = {
            'method': 'genetic',  # NSGA-II遗传算法
            'num_cf': n_counterfactuals,
            'features_to_vary': mutable_features,
            'permitted_range': self._get_feature_ranges(),
            'stopping_threshold': 0.05,
            'maxiterations': 500,
            'proximity_weight': 0.5,     # λ₁
            'sparsity_weight': 0.3,      # 稀疏性
            'diversity_weight': 0.2,     # 多样性
            'causal_penalty_weight': 0.4, # λ₃ 因果约束惩罚
        }
        
        # 4. 生成反事实
        dice_exp = Dice(
            model_interface=self._wrap_model(model),
            data_interface=self._prepare_data()
        )
        
        counterfactuals = dice_exp.generate_counterfactuals(
            query_instance=original_features,
            total_CFs=n_counterfactuals,
            desired_range=[target_score, 1000],
            **dice_config
        )
        
        # 5. 因果合理性验证
        validated_cfs = []
        for cf in counterfactuals:
            validation = self._validate_causal_plausibility(
                original=original_features,
                counterfactual=cf,
                graph=causal_graph
            )
            if validation.is_plausible:
                validated_cfs.append(CounterfactualResult(
                    features=cf,
                    predicted_score=model.predict(cf),
                    changes=self._compute_changes(original_features, cf),
                    causal_paths=self._trace_causal_paths(cf, causal_graph),
                    plausibility_score=validation.score
                ))
        
        # 6. 按可行性和效果排序
        return sorted(validated_cfs, key=lambda x: x.plausibility_score, reverse=True)
```

---

### 4.3 可解释性结果的可视化呈现架构

#### 4.3.1 可视化架构总览

```
可解释性可视化架构
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  可视化组件层 (React + ECharts + AntV G6)                     │  │
│  │                                                                │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │  │
│  │  │ SHAP瀑布图       │  │ 因果路径图       │  │ 反事实模拟器 │  │  │
│  │  │ (ECharts)       │  │ (AntV G6)       │  │ (React)      │  │  │
│  │  │                 │  │                 │  │              │  │  │
│  │  │ · 局部特征贡献  │  │ · 交互式DAG     │  │ · 特征滑块   │  │  │
│  │  │ · 正负向区分    │  │ · 路径高亮      │  │ · 实时预测   │  │  │
│  │  │ · 累积效应展示  │  │ · 效应标注      │  │ · 方案对比   │  │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────────┘  │  │
│  │                                                                │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │  │
│  │  │ SHAP Summary    │  │ 因果效应热力图   │  │ 一致性矩阵   │  │  │
│  │  │ (ECharts)       │  │ (ECharts)       │  │ (ECharts)    │  │  │
│  │  │                 │  │                 │  │              │  │  │
│  │  │ · 全局特征排名  │  │ · 特征×干预     │  │ · SHAP vs    │  │  │
│  │  │ · 蜂群图        │  │ · 效应大小着色  │  │   因果效应   │  │  │
│  │  │ · 特征值着色    │  │ · 置信区间      │  │ · 四象限图   │  │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  数据服务层 (FastAPI + WebSocket)                              │  │
│  │                                                                │  │
│  │  · /api/v1/explain/shap/{id}        → SHAP值 (同步)          │  │
│  │  · /api/v1/explain/causal/{id}      → 因果路径 (同步)        │  │
│  │  · /api/v1/explain/counterfactual   → 反事实方案 (异步)      │  │
│  │  · /ws/v1/explain/stream            → 实时解释流 (WebSocket) │  │
│  │  · /api/v1/explain/consistency      → 一致性报告 (同步)      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  计算引擎层 (Triton GPU加速)                                  │  │
│  │                                                                │  │
│  │  · SHAP计算: TreeSHAP on GPU (~5ms for 100 features)         │  │
│  │  · 因果路径: 图遍历算法 (~1ms)                               │  │
│  │  · 反事实: DiCE + NSGA-II on GPU (~50ms per CF)              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4.3.2 核心可视化组件设计

**组件1: 因果路径交互图 (AntV G6)**

```
交互式因果路径图设计
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────┐     -0.12      ┌─────────┐      0.08      ┌────┐│
│  │EXT_SRC_1├───────────────▶│ TARGET  │◀──────────────┤AMT ││
│  │ (外部   │  直接因果路径   │ (违约   │  直接因果路径  │CRED││
│  │  评分1) │  效应: -0.12   │  概率)  │  效应: +0.08  │IT  ││
│  └─────────┘                └────┬────┘               └──┬─┘│
│       │                           │                       │  │
│       │ 间接路径                   │                       │  │
│       │ (经AMT_ANNUITY)           │                       │  │
│       ▼                           │                       │  │
│  ┌─────────┐     +0.05      ┌────┴────┐     +0.03   ┌────┴─┐│
│  │AMT_     ├──────────────▶│DEBT_    ├──────────▶│AMT_  ││
│  │ANNUITY  │  中介效应       │RATIO    │  中介效应  │ANNUI-││
│  │         │                │         │           │TY    ││
│  └─────────┘                └─────────┘           └──────┘│
│                                                                 │
│  交互功能:                                                      │
│  · 点击节点: 展开该节点的所有因果路径                           │
│  · 悬停边: 显示效应大小、置信区间、p值                          │
│  · 拖拽: 调整布局                                              │
│  · 右键: 生成该路径的反事实解释                                 │
│  · 筛选: 按效应大小/路径类型过滤                                │
│  · 颜色编码: 红色=增加违约风险, 绿色=降低违约风险              │
│  · 线宽: 与效应绝对值成正比                                    │
└─────────────────────────────────────────────────────────────────┘
```

**组件2: SHAP-因果一致性四象限图**

```
SHAP-因果一致性四象限图
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  因果效应 │                                                     │
│     ↑     │                                                     │
│     │     │    Ⅱ 虚假相关区        Ⅰ 可信特征区                 │
│     │     │    SHAP高+因果低        SHAP高+因果高               │
│     │     │    ⚠️ 需审查            ✅ 可信赖                   │
│     │     │    · FLAG_OWN_CAR      · EXT_SOURCE_1              │
│     │     │    · CODE_GENDER       · AMT_CREDIT                │
│     │     │    · NAME_HOUSING      · DAYS_EMPLOYED             │
│     │     │      _TYPE              · BUREAU_DPD_TREND         │
│     │     │                                                     │
│  ───┼─────┼───────────────────────────────────────── SHAP值 →  │
│     │     │                                                     │
│     │     │    Ⅲ 无效应区          Ⅳ 遮蔽效应区                 │
│     │     │    SHAP低+因果低        SHAP低+因果高               │
│     │     │    ⚪ 可忽略            🔍 需调查                   │
│     │     │    · FLAG_MOBIL        · OWN_CAR_AGE               │
│     │     │    · CNT_FAM_MEMBERS   · REGION_RATING             │
│     │     │                                                     │
│     │     │                                                     │
│           └─────────────────────────────────────────────────────│
│                                                                 │
│  决策规则:                                                      │
│  · Ⅰ象限特征: 直接用于决策解释                                  │
│  · Ⅱ象限特征: 标记为"可能存在偏差", 需人工审查                 │
│  · Ⅲ象限特征: 可从模型中移除以简化                              │
│  · Ⅳ象限特征: 深入分析遮蔽原因, 可能需要特征重组               │
└─────────────────────────────────────────────────────────────────┘
```

**组件3: 反事实模拟器**

```
反事实模拟器 UI设计
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  当前评分: 580 (高风险)  ──────▶  目标评分: 650 (中等风险)     │
│  ════════════════════════════════════════════════════════════    │
│                                                                 │
│  特征调整面板:                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  AMT_CREDIT (贷款金额)                                   │  │
│  │  当前: ¥500,000  ──●────────── ¥1,000,000               │  │
│  │  调整: ¥350,000  ●────────────── ¥1,000,000    [可变]   │  │
│  │                                                          │  │
│  │  AMT_ANNUITY (年还款额) ← 因果联动                       │  │
│  │  当前: ¥60,000   ──────●──────── ¥120,000               │  │
│  │  调整: ¥42,000   ●────────────── ¥120,000    [自动联动] │  │
│  │                                                          │  │
│  │  AMT_INCOME_TOTAL (年收入)                               │  │
│  │  当前: ¥200,000  ──────────●───── ¥500,000              │  │
│  │  调整: ¥280,000  ───────────────●─ ¥500,000  [半可变]   │  │
│  │                                                          │  │
│  │  DAYS_EMPLOYED (在职天数)                                │  │
│  │  当前: 1,200天   ──────●──────── 3,000天                 │  │
│  │  调整: 1,200天   ──────●──────── 3,000天     [不可变🔒]  │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  推荐方案 (基于因果约束):                                       │
│  ┌────────┬──────────────────────┬──────────┬─────────────┐   │
│  │ 方案   │ 调整内容             │ 预测评分 │ 因果合理性  │   │
│  ├────────┼──────────────────────┼──────────┼─────────────┤   │
│  │ ★方案1 │ 贷款金额↓30%        │ 655      │ 0.92        │   │
│  │ 方案2  │ 年收入↑40%          │ 662      │ 0.85        │   │
│  │ 方案3  │ 贷款金额↓20%+收入↑20%│ 658     │ 0.88        │   │
│  └────────┴──────────────────────┴──────────┴─────────────┘   │
│                                                                 │
│  因果路径说明:                                                  │
│  "降低贷款金额 → 年还款额减少 → 债务收入比降低 → 违约风险下降" │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 真实应用部署方案

### 5.1 容器化部署

#### 5.1.1 容器化架构

```
Kubernetes容器化部署架构
┌──────────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster (EKS)                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Namespace: causal-credit-prod                                 │  │
│  │                                                                │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │  Ingress (AWS ALB Ingress Controller)                    │ │  │
│  │  │  · api.causalcredit.ai → Backend Service                │ │  │
│  │  │  · app.causalcredit.ai → Frontend Service               │ │  │
│  │  │  · WAF规则: SQL注入/XSS/CC攻击防护                      │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  │                                                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│  │
│  │  │ Backend      │  │ Frontend     │  │ Triton Inference     ││  │
│  │  │ (FastAPI)    │  │ (Next.js)    │  │ Server               ││  │
│  │  │              │  │              │  │                      ││  │
│  │  │ Replicas: 3  │  │ Replicas: 2  │  │ Replicas: 2-8 (HPA) ││  │
│  │  │ CPU: 2       │  │ CPU: 1       │  │ GPU: A10G × 1/Pod   ││  │
│  │  │ Memory: 4Gi  │  │ Memory: 2Gi  │  │ Memory: 16Gi        ││  │
│  │  │ HPA: 3-10   │  │ HPA: 2-5    │  │ HPA: 2-8            ││  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘│  │
│  │                                                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│  │
│  │  │ Celery       │  │ Redis        │  │ PostgreSQL           ││  │
│  │  │ Worker       │  │ (ElastiCache)│  │ (RDS)                ││  │
│  │  │              │  │              │  │                      ││  │
│  │  │ Replicas: 4  │  │ Cluster:     │  │ Multi-AZ            ││  │
│  │  │ CPU: 2       │  │ 3 Nodes      │  │ r6g.xlarge          ││  │
│  │  │ Memory: 8Gi  │  │ r6g.large    │  │ 500GB GP3           ││  │
│  │  │ Queue: score │  │              │  │ Citus分布式          ││  │
│  │  │   explain    │  │              │  │                      ││  │
│  │  │   report     │  │              │  │                      ││  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘│  │
│  │                                                                │  │
│  │  ┌──────────────┐  ┌──────────────┐                           │  │
│  │  │ Airflow      │  │ Kafka        │                           │  │
│  │  │ Scheduler    │  │ (MSK)        │                           │  │
│  │  │ + Webserver  │  │              │                           │  │
│  │  │ + Workers    │  │ 3 Brokers    │                           │  │
│  │  │              │  │ m5.large     │                           │  │
│  │  │ 1 each       │  │ 500GB EBS    │                           │  │
│  │  └──────────────┘  └──────────────┘                           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Namespace: monitoring                                         │  │
│  │  · Prometheus (Prometheus Operator)                            │  │
│  │  · Grafana                                                     │  │
│  │  · Alertmanager                                                │  │
│  │  · Loki (日志聚合)                                             │  │
│  │  · DCGM Exporter (GPU指标)                                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 5.1.2 Docker镜像策略

```dockerfile
# backend/Dockerfile - 多阶段构建
# Stage 1: 依赖安装
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Stage 2: 运行时
FROM python:3.11-slim AS runtime
WORKDIR /app

# 安全: 非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```dockerfile
# triton/Dockerfile.triton - 基于NVIDIA官方镜像
FROM nvcr.io/nvidia/tritonserver:24.04-py3

# 安装因果推理依赖
RUN pip install --no-cache-dir \
    dowhy==0.11.1 \
    econml==0.15.1 \
    shap==0.45.0 \
    dice-ml==0.11

# 复制模型仓库
COPY model_repository/ /models/

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

CMD ["tritonserver", "--model-repository=/models", "--grpc-port=8001", "--http-port=8000", "--metrics-port=8002"]
```

---

### 5.2 CI/CD流水线设计

#### 5.2.1 CI/CD架构

```
CI/CD流水线架构 (GitHub Actions + ArgoCD)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  CI Pipeline (GitHub Actions)                                  │  │
│  │                                                                │  │
│  │  Push/PR ──▶ [Lint] ──▶ [Test] ──▶ [Build] ──▶ [Security]   │  │
│  │                │         │         │            │              │  │
│  │                ▼         ▼         ▼            ▼              │  │
│  │            Ruff+Black  pytest   Docker      Trivy+            │  │
│  │            mypy       coverage  Build+Push  Snyk+             │  │
│  │            isort      >80%     ECR          Bandit            │  │
│  │                                                                │  │
│  │  模型CI (单独流水线):                                          │  │
│  │  Push ml/ ──▶ [Data Valid] ──▶ [Train] ──▶ [Evaluate] ──▶   │  │
│  │               Great Expect    GPU训练     AUC/KS/PSI          │  │
│  │                                            +因果验证          │  │
│  │               ──▶ [Export] ──▶ [Register]                    │  │
│  │                   Triton格式   MLflow Registry                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  CD Pipeline (ArgoCD GitOps)                                   │  │
│  │                                                                │  │
│  │  Git Repo (infra/) ──▶ ArgoCD ──▶ K8s Cluster               │  │
│  │                                                                │  │
│  │  部署策略:                                                     │  │
│  │  · Backend/Frontend: 蓝绿部署 (Blue-Green)                    │  │
│  │  · Triton: 金丝雀部署 (Canary, 10% → 50% → 100%)            │  │
│  │  · 模型更新: 影子部署 (Shadow, 对比验证后切换)                │  │
│  │                                                                │  │
│  │  环境晋升:                                                     │  │
│  │  dev → staging → canary → production                          │  │
│  │   │       │         │          │                               │  │
│  │   │       │         │          └── 自动回滚 (错误率>1%)       │  │
│  │   │       │         └── 人工审批 (模型变更)                    │  │
│  │   │       └── 自动部署 (代码变更)                              │  │
│  │   └── 自动部署 (每次Push)                                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 GitHub Actions Workflow

```yaml
# .github/workflows/backend-ci.yml
name: Backend CI/CD

on:
  push:
    branches: [main, develop]
    paths: ['backend/**']
  pull_request:
    branches: [main]
    paths: ['backend/**']

env:
  REGISTRY: 123456789.dkr.ecr.ap-northeast-1.amazonaws.com
  IMAGE_NAME: causal-credit-backend

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install ruff black mypy isort
      - run: ruff check backend/
      - run: black --check backend/
      - run: isort --check-only backend/
      - run: mypy backend/ --strict

  test:
    runs-on: ubuntu-latest
    needs: lint
    services:
      postgres:
        image: postgres:16
        env: { POSTGRES_DB: test_db, POSTGRES_PASSWORD: test }
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports: ['5432:5432']
      redis:
        image: redis:7-alpine
        ports: ['6379:6379']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: cd backend && pip install -e ".[dev]"
      - run: cd backend && pytest --cov=app --cov-report=xml -v
      - uses: codecov/codecov-action@v4
        with: { files: backend/coverage.xml }

  security:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: 'backend'
          severity: 'CRITICAL,HIGH'
      - uses: bandit/bandit-action@v1
        with: { path: backend }

  build-push:
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-northeast-1
      - uses: aws-actions/amazon-ecr-login@v2
      - uses: docker/build-push-action@v5
        with:
          context: ./backend
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-push
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - run: |
          # 更新Helm values中的镜像tag
          yq -i ".backend.image.tag = \"${{ github.sha }}\"" infra/helm/causal-credit/values-staging.yaml
          git config user.name "github-actions[bot]"
          git commit -am "chore: update staging image to ${{ github.sha }}"
          git push
          # ArgoCD自动检测到变更并部署
```

---

### 5.3 监控与告警体系

#### 5.3.1 监控架构

```
监控架构全景
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  数据采集层                                                    │  │
│  │                                                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│  │
│  │  │ 应用指标      │  │ GPU指标       │  │ 模型指标             ││  │
│  │  │ (Prometheus   │  │ (DCGM         │  │ (Evidently +         ││  │
│  │  │  + custom     │  │  Exporter +   │  │  Custom Metrics)     ││  │
│  │  │  metrics)     │  │  Triton       │  │                      ││  │
│  │  │              │  │  Metrics)     │  │                      ││  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘│  │
│  └─────────┼─────────────────┼─────────────────────┼─────────────┘  │
│            │                 │                     │                 │
│            ▼                 ▼                     ▼                 │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Prometheus (时序数据库)                                       │  │
│  │  · 采集间隔: 15s                                              │  │
│  │  · 保留期: 30天 (原始) / 1年 (降采样)                         │  │
│  │  · 远程写入: Thanos (长期存储)                                 │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                             │                                        │
│            ┌────────────────┼────────────────┐                       │
│            ▼                ▼                ▼                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Grafana      │  │ Alertmanager │  │ 自动化响应               │  │
│  │ (可视化)     │  │ (告警路由)   │  │ (Airflow触发重训练)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 5.3.2 模型漂移检测体系

```
模型漂移检测体系
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  漂移类型检测矩阵                                              │  │
│  │                                                                │  │
│  │  ┌──────────────────┬──────────────────┬──────────────────┐   │  │
│  │  │  数据漂移         │  概念漂移         │  预测漂移        │   │  │
│  │  │  (Covariate      │  (Concept        │  (Prediction     │   │  │
│  │  │   Drift)         │   Drift)         │   Drift)         │   │  │
│  │  ├──────────────────┼──────────────────┼──────────────────┤   │  │
│  │  │ 特征分布变化      │ P(Y|X)关系变化   │ 预测分布变化     │   │  │
│  │  │                  │                  │                  │   │  │
│  │  │ 检测方法:         │ 检测方法:         │ 检测方法:        │   │  │
│  │  │ · PSI > 0.2      │ · 残差趋势分析   │ · 评分分布PSI    │   │  │
│  │  │ · KS检验          │ · 分段AUC下降    │ · 违约率偏移     │   │  │
│  │  │   p < 0.01       │ · 因果效应变化   │ · 评分均值偏移   │   │  │
│  │  │ · Wasserstein    │ · ADWIN算法      │                  │   │  │
│  │  │   距离 > 阈值    │                  │                  │   │  │
│  │  │                  │                  │                  │   │  │
│  │  │ 检测频率:         │ 检测频率:         │ 检测频率:        │   │  │
│  │  │ 每日             │ 每周             │ 实时             │   │  │
│  │  │                  │                  │                  │   │  │
│  │  │ 响应动作:         │ 响应动作:         │ 响应动作:        │   │  │
│  │  │ 告警+特征审查     │ 触发重训练       │ 降级+人工审查    │   │  │
│  │  └──────────────────┴──────────────────┴──────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Evidently漂移报告 (自动生成)                                  │  │
│  │                                                                │  │
│  │  每日自动生成:                                                 │  │
│  │  · 特征级PSI热力图 (所有特征 × 最近7天)                       │  │
│  │  · Top-10漂移特征排名                                         │  │
│  │  · 漂移特征与模型性能下降的关联分析                            │  │
│  │  · 因果图结构变化检测 (新增/消失的因果边)                     │  │
│  │                                                                │  │
│  │  每周深度报告:                                                 │  │
│  │  · 模型性能趋势 (AUC/KS/Gini × 12周)                         │  │
│  │  · 分群性能分析 (按风险等级/年龄段/地区)                      │  │
│  │  · 因果效应稳定性分析                                          │  │
│  │  · 重训练建议与预期收益评估                                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 5.3.3 告警规则配置

```yaml
# monitoring/prometheus/rules/model_drift_alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: causal-credit-model-drift
  namespace: monitoring
spec:
  groups:
    - name: model_drift
      rules:
        # 特征PSI漂移告警
        - alert: FeaturePSIDrift
          expr: |
            causal_credit_feature_psi > 0.2
          for: 1h
          labels:
            severity: warning
            team: mlops
          annotations:
            summary: "Feature {{ $labels.feature }} PSI drift detected"
            description: "PSI={{ $value }} exceeds threshold 0.2"
        
        # 模型AUC下降告警
        - alert: ModelAUCDegradation
          expr: |
            causal_credit_model_auc < 0.75
            and
            causal_credit_model_auc offset 7d > 0.78
          for: 24h
          labels:
            severity: critical
            team: mlops
          annotations:
            summary: "Model AUC degraded below threshold"
            description: "Current AUC={{ $value }}, was {{ $labels.prev_auc }} 7 days ago"
        
        # 因果效应变化告警
        - alert: CausalEffectShift
          expr: |
            abs(
              causal_credit_causal_effect 
              - causal_credit_causal_effect offset 7d
            ) / causal_credit_causal_effect offset 7d > 0.3
          for: 48h
          labels:
            severity: warning
            team: mlops
          annotations:
            summary: "Causal effect for {{ $labels.feature }} shifted significantly"
            description: "Effect changed by {{ $value | humanizePercentage }}"

    - name: gpu_monitoring
      rules:
        # GPU利用率过低
        - alert: GPUUnderutilization
          expr: |
            DCGM_FI_DEV_GPU_UTIL < 30
          for: 30m
          labels:
            severity: info
          annotations:
            summary: "GPU {{ $labels.gpu }} underutilized"
            description: "Consider scaling down GPU nodes"
        
        # GPU显存使用过高
        - alert: GPUHighMemoryUsage
          expr: |
            DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL > 0.9
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "GPU {{ $labels.gpu }} memory usage > 90%"
            description: "Risk of OOM, consider scaling up"
        
        # GPU温度过高
        - alert: GPUTemperatureHigh
          expr: |
            DCGM_FI_DEV_GPU_TEMP > 85
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "GPU {{ $labels.gpu }} temperature > 85°C"
```

---

### 5.4 安全与合规

#### 5.4.1 安全架构

```
安全架构全景
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  1. 数据加密                                                   │  │
│  │                                                                │  │
│  │  传输加密:                                                     │  │
│  │  · TLS 1.3 (所有外部通信)                                     │  │
│  │  · mTLS (服务间通信, Istio Service Mesh)                      │  │
│  │  · gRPC over TLS (Triton通信)                                 │  │
│  │                                                                │  │
│  │  存储加密:                                                     │  │
│  │  · AES-256-GCM (S3/MinIO 服务端加密)                         │  │
│  │  · PostgreSQL TDE (透明数据加密)                               │  │
│  │  · Redis TLS + at-rest encryption                             │  │
│  │  · KMS (AWS Key Management Service) 密钥轮换: 90天           │  │
│  │                                                                │  │
│  │  字段级加密:                                                   │  │
│  │  · 身份证号: AES-256 加密存储, 仅授权服务解密                 │  │
│  │  · 收入信息: 格式保留加密 (FPE)                               │  │
│  │  · 评分结果: 哈希脱敏 (日志中仅显示评分等级)                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  2. 访问控制                                                   │  │
│  │                                                                │  │
│  │  身份认证:                                                     │  │
│  │  · OAuth 2.0 + OIDC (Keycloak)                                │  │
│  │  · JWT (短期令牌, 15min过期)                                  │  │
│  │  · MFA (管理员操作)                                           │  │
│  │                                                                │  │
│  │  授权模型 (RBAC + ABAC):                                      │  │
│  │  ┌──────────────┬──────────────────────────────────────────┐  │  │
│  │  │ 角色          │ 权限                                     │  │  │
│  │  ├──────────────┼──────────────────────────────────────────┤  │  │
│  │  │ analyst      │ 查看评分、可解释性报告                    │  │  │
│  │  │ risk_manager │ 评分+审批+反事实分析                      │  │  │
│  │  │ data_scientist│ 特征管理+模型实验+因果分析              │  │  │
│  │  │ mlops_engineer│ 模型部署+监控+流水线管理                │  │  │
│  │  │ admin        │ 全部权限+用户管理+审计日志               │  │  │
│  │  └──────────────┴──────────────────────────────────────────┘  │  │
│  │                                                                │  │
│  │  API安全:                                                      │  │
│  │  · Rate Limiting: 1000 req/min per API Key                    │  │
│  │  · IP白名单 (管理API)                                         │  │
│  │  · API Key轮换: 30天                                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  3. 审计日志                                                   │  │
│  │                                                                │  │
│  │  审计事件:                                                     │  │
│  │  · 评分请求 (谁、何时、对谁评分、结果)                        │  │
│  │  · 模型变更 (版本切换、参数调整)                              │  │
│  │  · 数据访问 (特征查询、批量导出)                              │  │
│  │  · 权限变更 (角色分配、API Key创建)                           │  │
│  │  · 反事实查询 (查询内容、生成结果)                            │  │
│  │                                                                │  │
│  │  存储:                                                         │  │
│  │  · PostgreSQL (近期, 90天)                                     │  │
│  │  · S3 + Athena (归档, 7年)                                    │  │
│  │  · 不可篡改: S3 Object Lock (WORM)                            │  │
│  │                                                                │  │
│  │  合规报告:                                                     │  │
│  │  · 月度审计报告 (自动生成)                                    │  │
│  │  · GDPR数据删除请求处理 (< 72h)                               │  │
│  │  · 个人信息保护法合规检查清单                                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  4. 模型治理                                                   │  │
│  │                                                                │  │
│  │  模型文档化 (Model Card):                                      │  │
│  │  · 模型用途、训练数据、性能指标                                │  │
│  │  · 公平性评估 (跨性别/年龄/地区的评分差异)                    │  │
│  │  · 因果有效性声明 (哪些因果路径经过验证)                      │  │
│  │  · 局限性与使用边界                                            │  │
│  │                                                                │  │
│  │  公平性监控:                                                   │  │
│  │  · Demographic Parity: P(评分>阈值|性别=男) ≈ P(评分>阈值|女) │  │
│  │  · Equalized Odds: TPR/FPR跨群体差异 < 5%                    │  │
│  │  · 因果公平性: 控制混淆因子后, 敏感属性对评分的因果效应 ≈ 0  │  │
│  │                                                                │  │
│  │  模型审批流程:                                                 │  │
│  │  Staging → [公平性测试] → [因果验证] → [性能基准] →          │  │
│  │  → [合规审查] → [人工审批] → Production                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 5.4.2 网络安全策略

```yaml
# infra/k8s/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: triton-network-policy
  namespace: causal-credit-prod
spec:
  podSelector:
    matchLabels:
      app: triton-inference
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # 仅允许Backend访问Triton
    - from:
        - podSelector:
            matchLabels:
              app: backend
      ports:
        - port: 8001  # gRPC
        - port: 8002  # Metrics
  egress:
    # 允许访问Redis (特征获取)
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - port: 6379
    # 允许访问S3 (模型加载)
    - to: []
      ports:
        - port: 443
    # 允许DNS
    - to: []
      ports:
        - port: 53
          protocol: UDP
---
# 限制Backend的出站访问
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-network-policy
  namespace: causal-credit-prod
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # 仅允许Ingress Controller访问
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8000
  egress:
    # 允许访问Triton
    - to:
        - podSelector:
            matchLabels:
              app: triton-inference
      ports:
        - port: 8001
    # 允许访问Redis
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - port: 6379
    # 允许访问PostgreSQL
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
      ports:
        - port: 5432
    # 允许访问Kafka
    - to:
        - podSelector:
            matchLabels:
              app: kafka
      ports:
        - port: 9092
```

---

## 附录

### A. 技术选型总览

| 领域 | 技术栈 | 版本 |
|------|--------|------|
| 后端框架 | FastAPI + Uvicorn | 0.110+ |
| 任务队列 | Celery + Redis | 5.4+ |
| 数据库 | PostgreSQL + Citus | 16 |
| 缓存 | Redis Cluster | 7.x |
| 消息队列 | Apache Kafka | 3.7 |
| GPU推理 | NVIDIA Triton Inference Server | 24.04 |
| GPU优化 | TensorRT | 8.6+ |
| 因果推理 | DoWhy + EconML | 0.11+ / 0.15+ |
| 可解释性 | SHAP + DiCE | 0.45+ / 0.11+ |
| 特征存储 | Feast | 0.37+ |
| 数据湖 | Delta Lake on S3 | 3.x |
| 数据质量 | Great Expectations | 0.18+ |
| 数据血缘 | Apache Atlas + OpenLineage | 2.3+ |
| 前端框架 | Next.js + TypeScript | 14 / 5.4+ |
| UI组件 | Ant Design Pro | 6.x |
| 图可视化 | AntV G6 | 5.x |
| MLOps | MLflow + DVC + Airflow | 2.x / 3.x / 2.8+ |
| 容器编排 | Kubernetes (EKS) | 1.29 |
| IaC | Terraform | 1.7+ |
| CI/CD | GitHub Actions + ArgoCD | - / 2.9+ |
| 监控 | Prometheus + Grafana | 2.50+ / 10.x |
| 模型监控 | Evidently | 0.4+ |
| GPU监控 | DCGM Exporter | 3.x |
| 服务网格 | Istio | 1.20+ |
| 密钥管理 | AWS KMS + Vault | - |

### B. 性能基准

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

### C. 关键设计决策记录

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|----------|------|
| GPU推理框架 | Triton | TensorRT/ONNX Runtime | 因果模型需Python Backend |
| 因果发现 | PC+NOTEARS融合 | 单一方法 | 融合提高发现精度 |
| 反事实方法 | DiCE+因果约束 | 纯DiCE | 因果约束保证合理性 |
| 特征存储 | Feast | Hopsworks | 轻量、与K8s集成好 |
| 数据湖格式 | Delta Lake | Iceberg/Hudi | ACID+Z-Order+Vacuum |
| 前端框架 | Next.js | Nuxt/Vite | SSR+RSC+生态成熟 |
| 部署策略 | 蓝绿+金丝雀 | 滚动更新 | 零停机+风险可控 |
| 服务网格 | Istio | Linkerd | mTLS+流量管理+可观测性 |

---

> **文档维护**: 本架构文档随项目迭代持续更新，重大变更需经架构评审委员会审批。  
> **下次评审日期**: 2026-07-05
