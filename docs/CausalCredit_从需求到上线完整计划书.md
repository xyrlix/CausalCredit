# CausalCredit 因果推理增强信用评分系统 — 从需求到上线完整计划书

> **版本**: v2.0 | **日期**: 2026-06-05 | **架构师**: 大卫-解决方案架构师  
> **难度评级**: ★★★★☆ (4/5) | **比赛**: 中银香港创新先驱大赛2026 — 大数据×理财/ESG  
> **整合来源**: 阿信行业调研 + 艾伦行为分析/反欺诈覆盖 + 苏珊技术架构/数据验证 + 大卫因果验证/实现计划书

---

# 第一部分：需求与数据基础

## 1. 需求背景与痛点分析

### 1.1 行业痛点：传统信用评分体系的系统性缺陷

FICO Score自1989年推出以来占据美国信用评分市场90%+份额，但其模型架构存在根本性缺陷：

| 缺陷维度 | 具体表现 | 量化数据 |
|-----------|----------|----------|
| **数据维度狭窄** | 仅依赖信贷历史数据，忽略支付行为、社交网络、职业稳定性等替代数据 | FICO模型仅使用约5-7类核心变量 |
| **静态评分逻辑** | 评分基于历史快照，无法捕捉信用状态的动态演变和因果机制 | 评分更新周期30-60天 |
| **关联≠因果** | 模型发现的是统计相关性而非因果关系，无法回答"如果改变X，Y会怎样" | Simpson悖论在信用评分中频繁出现 |
| **薄信用人群排斥** | 无信贷历史的年轻人、新移民无法获得准确评分 | 美国4,500万"信用隐形人"(CFPB, 2024) |
| **可解释性缺失** | 模型决策逻辑不透明，难以满足监管合规要求 | 70%的信贷经理不信任黑箱模型决策 |

**纯ML方案的"相关性陷阱"**：即便引入深度学习等先进ML方法，纯预测模型仍无法解决根本问题——混淆偏差导致"地区A违约率高→对地区A提高利率"的反向决策灾难；对撞偏差导致虚假关联；中介遮蔽使总效应可能为零；干预盲区使模型只能预测P(Y|X)而无法估计P(Y|do(X))。

**全球信用评分市场**：2024年约$122亿，预计2032年达$345亿（CAGR 13.9%），亚太地区增速最快（CAGR 15.2%），AI驱动的信用评分渗透率将从2024年的18%升至2028年的45%。

### 1.2 违约行为分析：为什么必须区分因果

传统信用评分将违约视为统一的二分类问题，但违约背后的驱动力截然不同——**不区分违约动因的模型，既无法精准预测，更无法指导干预**：

| 维度 | 恶意欺诈 | 非恶意违约 | 系统性风险 |
|------|----------|------------|------------|
| **核心驱动力** | 主观蓄意：申请时即有骗贷意图 | 客观能力不足：还款意愿存在但能力丧失 | 宏观环境冲击：个体无力抵御系统性变化 |
| **因果机制** | 欺诈意图→信息伪造→申请通过→违约 | 收入↓/支出↑→流动性枯竭→违约 | 宏观冲击→行业/区域衰退→群体违约 |
| **干预策略** | 拦截（拒绝申请） | 缓解（调整贷款结构） | 对冲（分散风险敞口） |
| **利率调整CATE** | ≈0（欺诈者不在乎利率） | -3%↓（显著降低还款压力） | -1%↓（对边际借款人有效） |

**因果推理的必要性论证**：传统模型发现"地区A违约率高"，据此对地区A提高利率——但地区A违约率高的真正原因是该地区平均收入低（混淆变量）。提高利率反而加剧了该地区借款人的还款压力，导致违约率进一步上升——这就是**因果反转**的灾难性后果。因果推理通过后门准则、前门准则、工具变量和do-演算，从观测数据中识别真正的因果路径，避免此类反向决策。

### 1.3 合规驱动：EU AI Act与HKMA监管要求

EU AI Act（2024年8月1日生效，2026年8月2日全面适用）将信用评分系统列为**高风险AI系统**，要求严格的透明度、公平性和可解释性。HKMA在AI治理框架中强调"可解释性和透明度"为核心原则。传统黑箱模型面临合规风险，而因果推理天然提供内在可解释性——因果图本身就是模型逻辑的透明表示，反事实推理直接回答"为什么"。

---

## 2. 数据可用性分析

### 2.1 数据集概览

| 数据集 | 规模 | 特征数 | 结构 | 可比赛使用 | 可商用 |
|--------|------|--------|------|-----------|--------|
| **Home Credit Default Risk**（主） | 307,511行 × 122列 | 218个原始字段（8表） | 多表关联 | ✅ | ❌ 需另行授权 |
| **Lending Club Loan Data**（辅） | 226万行 × 151列 | 单表宽表 | 单表 | ✅ | ✅ CC0 |
| **German Credit Risk**（基线） | 1,000行 × 20列 | 单表 | 单表 | ✅ | ✅ CC BY 4.0 |

### 2.2 因果推理核心能力特征验证

#### 因果推理特征（✅ 充分）

| 方案所需特征 | Home Credit对应字段 | 状态 | 说明 |
|-------------|-------------------|------|------|
| 贷款金额（Treatment候选） | `AMT_CREDIT` | ✅ | 连续型，变异充足，首选Treatment |
| 年还款额（中介变量） | `AMT_ANNUITY` | ✅ | 与AMT_CREDIT有确定性因果链 |
| 商品价格 | `AMT_GOODS_PRICE` | ✅ | 贷款金额的因果上游 |
| 收入总额（混淆变量） | `AMT_INCOME_TOTAL` | ✅ | 关键混淆因子 |
| 在职天数（Treatment候选） | `DAYS_EMPLOYED` | ✅ | ⚠️ 含异常值365243（失业标记） |
| 年龄（不可变特征） | `DAYS_BIRTH` | ✅ | 反事实推理中标记为不可变 |
| 性别（敏感属性） | `CODE_GENDER` | ✅ | 因果公平性验证的敏感属性 |
| 教育水平（工具变量候选） | `NAME_EDUCATION_TYPE` | ✅ | 可作为income的工具变量 |
| 职业类型 | `OCCUPATION_TYPE` | ✅ | ⚠️ 缺失率约31% |
| 地区评级 | `REGION_RATING_CLIENT` | ✅ | 典型混淆变量案例 |
| 外部评分1/2/3 | `EXT_SOURCE_1/2/3` | ✅ | ⚠️ EXT_SOURCE_1缺失率56% |
| 违约标签（Outcome） | `TARGET` | ✅ | 1=违约(8.07%), 0=正常(91.93%) |

**因果推理特征可用性结论：✅ 充分**。Treatment候选（AMT_CREDIT, DAYS_EMPLOYED）、Outcome（TARGET）、混淆变量（AMT_INCOME_TOTAL, REGION_RATING_CLIENT等）、中介变量（AMT_ANNUITY）、工具变量候选（NAME_EDUCATION_TYPE）、敏感属性（CODE_GENDER）均直接可用。

#### CATE估计特征（✅ 充分）

| 方案所需特征 | Home Credit对应字段 | 状态 |
|-------------|-------------------|------|
| Treatment变量 | `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_EMPLOYED` | ✅ |
| Outcome变量 | `TARGET` | ✅ |
| 混淆变量集 | `AMT_INCOME_TOTAL`, `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, `REGION_RATING_CLIENT`, `DAYS_BIRTH`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS` | ✅ |
| 效应修饰变量 | `CODE_GENDER`, `NAME_INCOME_TYPE`, `ORGANIZATION_TYPE`, `NAME_HOUSING_TYPE` | ✅ |

#### 反欺诈特征（🔶 有限）

| 方案所需特征 | Home Credit对应字段 | 状态 |
|-------------|-------------------|------|
| 文档提供异常 | `FLAG_DOCUMENT_2`~`FLAG_DOCUMENT_21` | ✅ |
| 社交圈违约观察 | `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE` | ✅ |
| 信用查询频率 | `AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MON/QRT/YEAR` | ✅ |
| 地址不一致 | `REG_REGION_NOT_LIVE_REGION`等6个字段 | ✅ |
| 身份信息一致性 | `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL` | 🔶 仅标志位 |
| 设备指纹/IP/生物识别 | — | ❌ 完全缺失 |
| 社交网络关系图 | — | ❌ 仅有计数，无关系图 |

#### 替代数据特征（🔶 有限）

| 方案所需特征 | Home Credit对应字段 | 状态 |
|-------------|-------------------|------|
| 履约还款数据 | installments_payments表 | ✅ |
| 信用卡消费数据 | credit_card_balance表 | 🔶 仅信用卡消费 |
| POS分期消费 | POS_CASH_balance表 | 🔶 仅分期消费 |
| 出行/社交/公共事业数据 | — | ❌ 完全缺失 |

### 2.3 因果推理可行性验证

| 维度 | 评级 | 说明 |
|------|------|------|
| Treatment候选 | ✅ 充分 | AMT_CREDIT为首选（变异充足、业务可干预、因果路径清晰） |
| Outcome定义 | ✅ 明确 | TARGET为主，可补充连续型违约严重度 |
| 混淆变量 | ✅ 充分 | 七大维度覆盖（收入能力、人口统计、地区经济、教育资本、资产状况、信用历史、外部评分） |
| 因果图构建 | ✅ 齐全 | Treatment-Outcome路径、中介链、混淆因子、工具变量、对撞变量、敏感属性、效应修饰变量均有对应字段 |
| **因果推理总体可行性** | **✅ 可行** | **Home Credit数据集完全支撑CausalCredit的因果推理核心能力** |

### 2.4 数据质量关键发现

- **缺失值**：住房建筑信息系列字段缺失率>50%（直接剔除）；`RATE_INTEREST_PRIMARY/PRIVILEGED`缺失99%（**无法直接做利率Treatment的CATE分析**，改用AMT_CREDIT）
- **异常值**：`DAYS_EMPLOYED`=365243为失业标记（替换为NaN+添加FLAG_UNEMPLOYED）；`AMT_INCOME_TOTAL`极端高值1.17亿（对数变换+Winsorize）
- **类别不平衡**：TARGET正样本仅8.07%（1:11.4），对因果推理影响有限，对预测模型需类别权重调整
- **数据泄露风险**：EXT_SOURCE_*是已知高风险泄露点，因果发现时应作为调整变量而非核心因果变量
- **真实部署映射**：约65%的特征可直接复用于银行真实场景

### 2.5 数据集可用性综合评分：3.8 / 5.0

| 维度 | 评分(1-5) | 权重 | 加权分 |
|------|----------|------|--------|
| 数据可获取性 | 5 | 10% | 0.50 |
| 因果推理特征支撑 | 5 | 25% | 1.25 |
| 反欺诈特征支撑 | 2.5 | 20% | 0.50 |
| 替代数据支撑 | 2 | 10% | 0.20 |
| 数据质量 | 3.5 | 15% | 0.525 |
| 真实部署映射 | 4 | 10% | 0.40 |
| 因果推理可行性 | 4.5 | 10% | 0.45 |

---

## 3. 我们能做什么 & 不能做什么

### 3.1 能力覆盖总表

| # | 能力点 | 评级 | 数据支撑 | 说明 |
|---|--------|------|---------|------|
| 1 | 因果发现（PC+NOTEARS融合） | ✅能做 | ✅充足变量和变异 | 核心差异化能力 |
| 2 | CATE异质处理效应估计 | ✅能做 | ✅AMT_CREDIT为Treatment，混淆变量充分 | 核心差异化能力 |
| 3 | 反事实推理与决策建议 | ✅能做 | ✅不可变/半可变/可变特征分类明确 | 核心差异化能力 |
| 4 | 白户/薄信用人群替代评分 | ✅能做 | ✅反事实推理+替代因果变量路径完整 | 第二大亮点 |
| 5 | 因果公平性验证 | ✅能做 | ✅CODE_GENDER等敏感属性+混淆变量充分 | 合规差异化 |
| 6 | SHAP+因果图联合可解释性 | ✅能做 | ✅特征归因+因果路径追踪数据齐全 | 合规差异化 |
| 7 | 包装资质申贷检测 | ✅能做 | ✅收入-职业-地区不一致检测 | 因果推理独特反欺诈价值 |
| 8 | 信用查询频率异常检测 | ✅能做 | ✅AMT_REQ_CREDIT_BUREAU_* | 反欺诈辅助 |
| 9 | 还款行为时序特征 | ✅能做 | ✅installments_payments + POS_CASH_balance | 时序Pipeline |
| 10 | LightGBM/XGBoost预测模型 | ✅能做 | ✅122列特征+30万行数据 | 预测基座 |
| 11 | 贷前准入反欺诈 | 🔶部分能做 | 🔶因果推理可辅助风险分层，但缺反欺诈决策引擎 | 需补充规则引擎 |
| 12 | 替代数据+因果建模 | 🔶部分能做 | 🔶因果建模框架完备，但替代数据源单一 | 比赛中用合成数据演示 |
| 13 | 黑产虚假优质用户识别 | 🔶部分能做 | 🔶因果一致性可识别"因果不一致"模式 | 缺细粒度时序行为一致性检测 |
| 14 | 拦截申请欺诈 | 🔶部分能做 | 🔶可构建欺诈倾向评分 | 缺欺诈标签和实时拦截引擎 |
| 15 | 多头借贷分析 | 🔶部分能做 | 🔶bureau表可聚合跨机构借贷 | 缺跨机构关联数据 |
| 16 | 识别身份冒用 | ❌做不了 | ❌无证件/设备/生物特征数据 | 架构预留IDV接口 |
| 17 | 团伙养号骗贷 | ❌做不了 | ❌无设备/IP/地址关联图数据 | 架构预留GNN模块 |
| 18 | 利率Treatment的CATE | ❌做不了 | ❌RATE_INTEREST缺失99% | 改用AMT_CREDIT间接估计 |
| 19 | 出行/社交替代数据 | ❌做不了 | ❌完全缺失 | 合成数据演示框架能力 |
| 20 | 实时行为流分析 | ❌做不了 | ❌无实时行为数据 | 架构设计支持实时流接入 |

### 3.2 关键降级方案

**降级方案1：三类违约分类（替代欺诈标签）**——由于TARGET不区分违约类型，通过因果路径聚类对违约样本分类：高负债型（AMT_CREDIT→AMT_ANNUITY→TARGET）、收入不稳定型（DAYS_EMPLOYED→AMT_INCOME_TOTAL→TARGET）、疑似欺诈型（EXT_SOURCE_*极低+特征异常）。CATE异质性分析可辅助识别：欺诈型违约对利率调整CATE≈0。

**降级方案2：轻量级身份异常检测（替代IDV系统）**——基于FLAG_WORK_PHONE/FLAG_EMAIL/FLAG_DOCUMENT_*构建contact_completeness + address_consistency + doc_pattern_anomaly + query_frequency_anomaly的identity_anomaly_score。

**降级方案3：地区-时段异常聚集检测（替代GNN团伙检测）**——对(REGION_RATING_CLIENT, WEEKDAY_APPR_PROCESS_START, HOUR_APPR_PROCESS_START)三元组计算申请密度，检测异常聚集。

---

# 第二部分：方案设计与技术选型

## 4. 特征工程方案

基于Home Credit数据集实际字段，设计完整的特征工程Pipeline，每个特征标注来源字段和衍生逻辑。

### 4.1 Pipeline A：因果特征挖掘

| 特征名 | 来源字段 | 衍生逻辑 | 因果含义 |
|--------|---------|---------|---------|
| `causal_path_strength_credit` | `AMT_CREDIT`, `AMT_ANNUITY`, `TARGET` | AMT_CREDIT→TARGET所有因果路径的累积效应 | 贷款金额对违约的因果路径强度 |
| `deconfounded_amt_income` | `AMT_INCOME_TOTAL`, `REGION_RATING_CLIENT`, `DAYS_BIRTH` | 用混淆因子做回归取残差 | 去除地区/年龄混淆后的收入效应 |
| `iv_exogenous_days_employed` | `DAYS_EMPLOYED`, `NAME_EDUCATION_TYPE` | 工具变量提取外生变异 | 教育对就业的外生影响分量 |
| `direct_effect_credit_via_annuity` | `AMT_CREDIT`, `AMT_ANNUITY`, `TARGET` | 中介效应分解：直接效应 | 贷款金额不通过年还款额的直接效应 |
| `indirect_effect_credit_via_annuity` | `AMT_CREDIT`, `AMT_ANNUITY`, `TARGET` | 中介效应分解：间接效应 | 贷款金额通过年还款额的间接效应 |
| `causal_anomaly_score` | 全部因果路径特征 | SHAP高但因果效应低的特征占比 | 因果不一致→包装/欺诈嫌疑 |
| `path_integrity_income_consumption` | `AMT_INCOME_TOTAL`, credit_card_balance: `AMT_DRAWINGS_CURRENT` | 收入→消费→还款因果链完整度 | 路径断裂→包装嫌疑 |

### 4.2 Pipeline B：时序特征提取

| 特征名 | 来源字段 | 衍生逻辑 | 因果含义 |
|--------|---------|---------|---------|
| `bureau_dpd_trend_6m` | bureau_balance: `STATUS`, `MONTHS_BALANCE` | 近6个月DPD的线性趋势斜率 | 逾期恶化速度→违约因果路径 |
| `bureau_status_transition_entropy` | bureau_balance: `STATUS` | 状态转移矩阵的信息熵 | 行为不稳定性→风险因果指标 |
| `bureau_recovery_rate_trend` | bureau_balance: `STATUS` | 逾期后恢复率的变化趋势 | 自愈能力→保护性因果因子 |
| `installment_late_days_trend` | installments_payments: `DAYS_ENTRY_PAYMENT` - `DAYS_INSTALMENT` | 逾期天数滑动窗口趋势 | 还款纪律恶化→违约前兆 |
| `credit_utilization_trend` | credit_card_balance: `AMT_BALANCE` / `AMT_CREDIT_LIMIT_ACTUAL` | 信用额度使用率6月趋势 | 过度负债→违约因果路径 |
| `repayment_amount_volatility` | installments_payments: `AMT_PAYMENT` | 还款金额变异系数 | 收入不稳定→还款能力波动 |
| `bureau_pattern_embedding` | bureau_balance全序列 | LSTM Encoder→32维向量 | 潜在行为模式→隐变量因果结构 |

### 4.3 Pipeline C：交叉特征构造

| 特征名 | 来源字段 | 衍生逻辑 | 因果含义 |
|--------|---------|---------|---------|
| `debt_to_income_ratio` | `AMT_ANNUITY` / `AMT_INCOME_TOTAL` | 债务收入比 | 核心因果中介变量 |
| `credit_to_income_ratio` | `AMT_CREDIT` / `AMT_INCOME_TOTAL` | 贷款收入比 | 过度负债因果指标 |
| `goods_to_credit_ratio` | `AMT_GOODS_PRICE` / `AMT_CREDIT` | 商品价格/贷款金额 | 超额贷款检测 |
| `ext_source_ensemble` | `EXT_SOURCE_1` × `EXT_SOURCE_2` × `EXT_SOURCE_3` | 多源评分加权融合 | 共同效应节点交叉 |
| `income_region_interaction` | `AMT_INCOME_TOTAL` × `REGION_RATING_CLIENT` | 收入×地区评级 | 因果链中介交互 |
| `employed_income_interaction` | `DAYS_EMPLOYED` × `AMT_INCOME_TOTAL` | 在职天数×收入 | 因果链中介交互 |
| `social_circle_default_rate` | `DEF_30_CNT_SOCIAL_CIRCLE` / `OBS_30_CNT_SOCIAL_CIRCLE` | 社交圈违约率 | 环境风险因果指标 |
| `doc_completeness_anomaly` | `FLAG_DOCUMENT_2`~`FLAG_DOCUMENT_21` | 文档提供完整度异常模式 | 全0或全1→欺诈信号 |

### 4.4 Pipeline D：替代数据特征（合成数据演示）

| 特征名 | 来源字段 | 衍生逻辑 | 因果含义 |
|--------|---------|---------|---------|
| `consumption_stability` | credit_card_balance: `AMT_DRAWINGS_CURRENT` | 消费频次稳定性 | 消费行为→还款能力因果链 |
| `repayment_discipline_score` | installments_payments: `AMT_PAYMENT` vs `AMT_INSTALMENT` | 准时还款率趋势 | 履约→信用品质→违约 |
| `consumption_income_ratio` | `AMT_DRAWINGS_CURRENT`聚合 / `AMT_INCOME_TOTAL` | 消费-收入比 | 收入-消费因果一致性 |

### 4.5 特征质量门控

特征工程Pipeline输出后，经过四层质量门控：
1. **PSI检测**：PSI>0.2的特征自动告警
2. **特征重要性过滤**：LightGBM特征重要性<1%的特征剔除
3. **共线性剔除**：Spearman |ρ|>0.95的特征对保留重要性更高的
4. **因果有效性验证**：对因果特征做反驳测试，验证因果效应稳健性

---

## 5. 价值亮点

### 5.1 六大技术亮点

| # | 亮点 | 创新点 | 价值 |
|---|------|--------|------|
| 1 | **混合因果发现引擎** | PC+NOTEARS融合+领域知识注入，取交集提高精度 | 避免纯数据驱动的虚假边，同时避免纯专家知识的主观偏差 |
| 2 | **CATE异质处理效应** | DML/DR/Causal Forest三方法交叉验证CATE | 揭示"同一政策对不同人群效果不同"的深层机制，指导差异化定价 |
| 3 | **因果约束反事实推理** | DiCE+因果图约束+NSGA-II，注入因果约束惩罚项 | 反事实建议不仅"有效"而且"合理"——可直接指导信贷经理 |
| 4 | **GPU加速实时因果推理** | Triton混合推理（TensorRT评分~2ms+Python因果~15ms+反事实~50ms） | 因果推理首次实现实时评分，P99<100ms |
| 5 | **SHAP+因果图联合可解释性** | 双层解释+四象限一致性校验（可信/虚假相关/无效应/遮蔽） | 满足EU AI Act高风险AI可解释性要求 |
| 6 | **因果引导特征工程** | 因果路径强度+去混淆残差+IV外生分量+中介效应分解 | 特征工程从"暴力枚举"升级为"因果引导" |

### 5.2 量化价值论证

| 业务指标 | 传统方案 | CausalCredit | 提升幅度 | 年化价值（按10万笔贷款估算） |
|---------|---------|-------------|---------|----------------------|
| 违约率 | 8.5% | 6.8% | -20% | 减少损失$3.4M/年 |
| 审批通过率 | 65% | 72% | +10.8% | 增量利润$2.1M/年 |
| 薄信用人群覆盖 | 35% | 58% | +65.7% | 增量利润$1.5M/年 |
| 定价精度 | 统一溢价 | CATE差异化 | 利润↑20% | 增量利润$4.0M/年 |
| 贷后干预效率 | 被动催收 | 主动干预 | 回收率↑40% | 减少损失$1.2M/年 |
| 合规成本 | 高 | 低 | -40% | 节省$0.5M/年 |
| **综合年化价值** | — | — | — | **$12.7M/年** |

**ROI估算**：开发成本~HK$385K（3人×6周+算力HK$500）vs 年化价值HK$12.7M → **ROI 33:1**

### 5.3 因果推理独特价值

因果推理是CausalCredit区别于所有竞品的核心——传统反欺诈依赖规则匹配（看到什么拦截什么），ML反欺诈预测谁可能欺诈（统计关联），CausalCredit理解为什么是欺诈（因果推理）→更准、更可解释、更难规避。因果一致性检测是传统ML和规则系统**无法实现**的能力：包装的资质在统计上"看起来对"，但在因果上"说不通"。

---

## 6. 竞品对比

### 6.1 六大竞品对比矩阵

| 维度 | FICO Score 10T | Zest AI | Upstart | 芝麻信用分 | 京东金融风控 | **CausalCredit** |
|------|----------------|---------|---------|----------|------------|------------------|
| **核心方法** | 逻辑回归+趋势数据 | AutoML+可解释ML | 深度学习+替代数据 | 大模型+多维行为 | ML/DL+知识图谱 | **因果推理+ML+XAI** |
| **因果推理** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ DoWhy+EconML+CausalNex |
| **CATE异质效应** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ DML/DR/Causal Forest |
| **反事实推理** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ DiCE+因果约束 |
| **可解释性** | ⚠️ 低 | ✅ 中 | ⚠️ 低 | ⚠️ 低 | ⚠️ 中 | ✅ **高（SHAP+因果图+反事实）** |
| **GPU加速** | ❌ | ❌ | ❌ | 部分 | ❌ | ✅ RAPIDS/Triton/TensorRT |
| **薄信用覆盖** | 差 | 中(98%美国人) | 好 | 好 | 好 | **好（反事实替代评分路径）** |
| **公平性审计** | ❌ | ✅ 内置 | ⚠️ 外部监控 | ❌ | ⚠️ 有限 | ✅ **因果公平性验证** |
| **干预指导** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ CATE+反事实建议 |
| **策略评估** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 因果效应估计 |
| **EU AI Act合规** | ⚠️ 部分 | ✅ 较好 | ⚠️ 待评估 | ❌ 不适用 | ❌ 不适用 | ✅ **原生合规** |
| **因果推理验证标准** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **4层递进验证体系** |

### 6.2 CausalCredit差异化优势

1. **唯一具备因果推理能力**：竞品均停留在统计关联层面，CausalCredit是唯一能回答"如果改变X，Y会怎样"的信用评分系统
2. **唯一提供干预指导**：竞品仅输出风险排序，CausalCredit输出可操作的干预方案
3. **唯一实现实时因果推理**：通过Triton GPU加速，因果推理不再是离线分析的专利
4. **最高可解释性等级**：SHAP+因果图+反事实三层解释，满足EU AI Act高风险AI要求
5. **因果推理验证标准作为独特竞争力**：4层递进验证体系（因果图→效应估计→反事实→端到端），量化指标与通过阈值，竞品均无此能力
6. **因果公平性验证**：唯一能从因果层面验证算法公平性（敏感属性因果效应≈0）

---

## 7. 模型选型

### 7.1 因果推理模型

| 组件 | 选型 | 理由 | 适用场景 |
|------|------|------|---------|
| **因果发现** | DoWhy (PC+NOTEARS) | 微软开源，PC算法（约束法）+NOTEARS（优化法）融合，反驳验证框架完善 | 从Home Credit数据中学习因果图结构 |
| **因果效应估计** | EconML (DML/DR/Causal Forest) | 微软开源，DML线性CATE+DR鲁棒CATE+CF非参数CATE全覆盖，与DoWhy无缝集成 | 估计AMT_CREDIT对TARGET的异质因果效应 |
| **贝叶斯因果图** | CausalNex | 基于BN的结构学习，支持条件概率查询，补充DoWhy的贝叶斯视角 | 不确定性量化、条件概率推理 |

### 7.2 预测模型

| 组件 | 选型 | 理由 | 适用场景 |
|------|------|------|---------|
| **主模型** | LightGBM 4.x | 类别特征原生支持、训练速度快、内存占用低，Home Credit竞赛冠军方案基座 | 主评分模型，CPU训练 |
| **辅助模型** | XGBoost 2.x (RAPIDS GPU) | RAPIDS cuDF/cuML GPU加速训练，TensorRT导出支持，与Triton原生集成 | GPU加速训练+TensorRT推理优化 |
| **模型融合** | 加权平均+Stacking | LightGBM+XGBoost双模型融合，互补偏差-方差权衡 | 最终评分输出 |

### 7.3 可解释性模型

| 组件 | 选型 | 理由 | 适用场景 |
|------|------|------|---------|
| **特征归因** | SHAP (TreeSHAP) 0.45+ | 树模型精确SHAP值，多项式时间复杂度 | "哪些特征重要"——统计归因 |
| **因果图解释** | DoWhy因果图+自定义路径追踪 | 因果路径追踪+效应分解，SHAP无法提供的逻辑归因 | "为什么重要"——逻辑归因 |
| **反事实解释** | DiCE 0.11+ | 微软开源，多样化反事实生成，支持遗传算法优化 | "如果怎样会怎样"——决策建议 |

---

## 8. 技术选型

### 8.1 后端技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| **Web框架** | FastAPI 0.110+ | 异步高性能、自动OpenAPI文档、Pydantic类型安全、与Triton gRPC原生集成 |
| **任务队列** | Celery + Redis 5.4+/7.x | 成熟异步任务方案，SHAP/因果图异步计算不阻塞主评分路径 |
| **缓存** | Redis Cluster 7.x | 亚毫秒级延迟，Feast Online Store底座 |
| **数据库** | PostgreSQL + Citus 16 | 分布式扩展、JSONB支持评分元数据、Citus水平分片 |
| **ORM** | SQLAlchemy 2.0 + Alembic | 异步ORM、类型安全迁移 |
| **消息队列** | Apache Kafka 3.7 | 事件驱动架构、Exactly-Once语义、评分事件流与模型监控事件解耦 |

### 8.2 前端技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| **框架** | Next.js (App Router) 14 | SSR/SSG提升首屏加载、React Server Components降低客户端bundle |
| **语言** | TypeScript 5.4+ | 类型安全、IDE智能提示、前后端类型共享 |
| **UI库** | Ant Design Pro 6.x | 企业级中后台组件、ProTable/ProForm开箱即用 |
| **图表** | ECharts + AntV G6 | ECharts业务图表、G6因果图可视化（交互式DAG） |
| **状态管理** | Zustand + TanStack Query | 轻量状态+服务端缓存 |
| **API层** | tRPC | 端到端类型安全，前后端类型自动推导 |

### 8.3 GPU推理技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| **推理服务** | NVIDIA Triton Inference Server 24.04 | 唯一同时支持TensorRT优化模型和Python Backend因果推理模型的框架 |
| **推理优化** | TensorRT 8.6+ | XGBoost/ONNX→TensorRT INT8量化，推理延迟降至~2ms |
| **GPU调度** | NVIDIA GPU Operator + MIG | A10G MIG切分，多模型共享GPU，显存隔离 |

### 8.4 MLOps技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| **实验追踪** | MLflow 2.x | 模型实验记录、参数对比、模型注册中心 |
| **数据版本** | DVC + Delta Lake 3.x | 数据集版本管理+ACID事务+Z-Order优化 |
| **编排调度** | Apache Airflow 2.8+ | Pipeline编排、定时调度、依赖管理 |
| **特征存储** | Feast 0.37+ | 在线/离线特征一致性，Redis在线存储+Delta离线存储 |
| **数据质量** | Great Expectations 0.18+ | 数据质量门控，缺失率/一致性/分布漂移自动检测 |

### 8.5 部署技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| **容器化** | Docker | 多阶段构建，非root用户，安全扫描 |
| **编排** | Kubernetes (EKS) 1.29 | GPU调度、HPA弹性伸缩、Namespace隔离 |
| **包管理** | Helm 3.x | 一键部署全栈，values多环境配置 |
| **IaC** | Terraform 1.7+ | EKS/RDS/ElastiCache/S3/MSK基础设施即代码 |
| **CI/CD** | GitHub Actions + ArgoCD | CI自动化测试+CD GitOps声明式部署 |
| **服务网格** | Istio 1.20+ | mTLS服务间加密、流量管理、可观测性 |

---

# 第三部分：实现与部署

## 9. 系统架构设计

### 9.1 四层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  应用层 (Application Layer)                                          │
│  评分仪表盘(Next.js) | 因果分析面板(G6+ECharts) | 反事实模拟器(React) │
│  模型监控(Grafana) | 系统管理(RBAC) | API Gateway(Kong+JWT)          │
├──────────────────────────────────────────────────────────────────────┤
│  算法层 (Algorithm Layer)                                            │
│  评分引擎(Triton: TensorRT~2ms + Python因果~15ms + 反事实~50ms)      │
│  因果发现引擎(DoWhy+CausalNex) | CATE估计引擎(EconML+CF)             │
│  可解释性引擎(SHAP+因果图+DiCE) | 特征工程引擎(3 Pipelines+Feast)    │
│  模型训练引擎(LightGBM+XGBoost RAPIDS+MLflow)                       │
├──────────────────────────────────────────────────────────────────────┤
│  数据层 (Data Layer)                                                 │
│  数据湖(Delta Lake on S3: Bronze→Silver→Gold)                       │
│  特征存储(Feast+Redis: Online+Offline)                               │
│  业务数据库(PostgreSQL+Citus) | 消息总线(Kafka)                      │
│  数据质量(Great Expectations+Atlas血缘)                              │
├──────────────────────────────────────────────────────────────────────┤
│  基础设施层 (Infrastructure Layer)                                   │
│  容器编排(K8s/EKS+Helm+Istio) | GPU集群(A10G推理+A100训练+MIG)      │
│  监控告警(Prometheus+Grafana+Alertmanager)                           │
│  CI/CD(GitHub Actions+ArgoCD) | 安全合规(KMS+mTLS+WAF+RBAC)        │
│  IaC(Terraform)                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 模块划分

| 模块 | 职责 | 核心技术 |
|------|------|---------|
| **评分服务** | 同步评分+异步可解释性 | FastAPI + Triton gRPC + Celery |
| **因果推理服务** | 因果发现+效应估计+反事实生成 | DoWhy + EconML + DiCE |
| **特征服务** | 在线/离线特征获取 | Feast + Redis + Delta Lake |
| **训练服务** | 模型训练+评估+导出 | LightGBM + XGBoost RAPIDS + MLflow |
| **监控服务** | 模型漂移+性能+GPU监控 | Prometheus + Grafana + Evidently |

### 9.3 数据流

```
申请请求 → API Gateway(Kong) → FastAPI评分服务
  → Feast获取在线特征(Redis ~2ms)
  → Triton推理: 评分模型(TensorRT ~2ms) + 因果推理(Python ~15ms)
  → 评分融合(w₁·ml_score + w₂·causal_adjusted_score + w₃·rule_score)
  → 同步返回评分结果
  → Celery异步: SHAP计算 + 因果图渲染 + 反事实报告生成
  → 结果写入PostgreSQL + Kafka事件发布
```

### 9.4 核心API接口定义

| 接口 | 方法 | 路径 | 功能 | 延迟要求 |
|------|------|------|------|---------|
| 信用评分 | POST | `/v1/score` | 输入申请人特征，输出评分+风险等级+违约概率 | P99<100ms |
| 可解释性 | GET | `/v1/explain/{score_id}` | 输出SHAP瀑布图+因果路径+一致性校验 | 异步<5s |
| 反事实分析 | POST | `/v1/counterfactual` | 输入目标评分，输出K=5个反事实方案 | <3s |
| 因果图查询 | GET | `/v1/causal/graph` | 输出当前因果DAG | <1s |
| CATE查询 | POST | `/v1/cate` | 输入处理变量+人群特征，输出CATE | <2s |
| 模型监控 | GET | `/v1/monitoring/metrics` | 输出AUC/KS/PSI/GPU利用率 | <1s |

---

## 10. 流程设计

### 10.1 九阶段端到端流程

```
Stage 1: 数据接入
  Home Credit 8表 → Lambda架构 → Medallion数据湖(Bronze→Silver→Gold)
  Lending Club单表 → 字段映射 → 统一信用域模型

Stage 2: 特征工程
  Pipeline A: 因果特征挖掘 → 因果路径强度/去混淆残差/IV外生分量/中介效应分解
  Pipeline B: 时序特征提取 → DPD趋势/恢复率趋势/行为嵌入(32维)/突变检测
  Pipeline C: 交叉特征构造 → DTI/CTI/多源评分融合/因果引导交叉
  Pipeline D: 替代数据特征 → 消费稳定性/还款纪律/消费收入比
  → Feature Store(Feast+Redis) → 特征质量门控

Stage 3: 因果发现
  PC算法(约束法) → 骨架图
  NOTEARS(优化法) → DAG图
  交集融合 → 高置信因果图
  领域知识注入(禁止边/必选边/方向约束) → 最终因果图
  反驳验证(Placebo/Random Cause/Data Subset) → 稳健性确认

Stage 4: 模型训练
  LightGBM(CPU训练, 类别特征优化) + XGBoost(RAPIDS GPU加速)
  → 加权融合/Stacking → MLflow实验记录+模型注册
  → TensorRT导出(XGBoost→ONNX→TensorRT INT8)

Stage 5: CATE估计
  EconML框架: DML(线性CATE) + DR(鲁棒CATE) + Causal Forest(非参数CATE)
  处理变量: AMT_CREDIT(首选) + AMT_ANNUITY + DAYS_EMPLOYED
  5-fold交叉验证 → CATE跨验证一致性检验

Stage 6: 反事实推理
  可行特征识别(不可变/半可变/可变分类)
  → 因果约束构建(因果图联动规则)
  → DiCE + NSGA-II多目标优化
  → 多样性反事实生成(K=5个方案)
  → 因果合理性验证(路径一致性+分布可行性)

Stage 7: 评分输出
  评分融合: final_score = w₁·ml_score + w₂·causal_adjusted_score + w₃·rule_score
  风险等级映射: 800-1000(AAA) / 650-799(AA) / 500-649(A) / 300-499(BBB) / 0-299(BB以下)

Stage 8: 可解释性呈现
  Layer 1: SHAP统计归因(What matters?) → SHAP瀑布图+Summary
  Layer 2: 因果图逻辑归因(Why?) → 因果路径追踪+效应分解
  Layer 3: 反事实决策建议(What if?) → 反事实模拟器+多方案对比
  一致性校验: SHAP-因果四象限图(可信/虚假相关/无效应/遮蔽)

Stage 9: 决策输出
  评分+风险等级+违约概率+Top风险因子+CATE洞察+反事实建议+因果叙事
```

---

## 11. 因果推理验证体系

### 11.1 四层递进验证金字塔

```
         ┌───────┐
         │ L4    │  端到端验证：A/B测试/业务指标对比
     ┌───┴───────┴───┐
     │ L3            │  反事实验证：因果约束/可行性/已知效果对比
 ┌───┴───────────────┴───┐
 │ L2                    │  因果效应验证：反驳/伪实验/敏感性/CATE一致性
┌┴───────────────────────┴┐
│ L1                      │  因果图验证：稳定性/领域一致性/独立性检验
└─────────────────────────┘
原则：下层不通过 → 上层不可验证 → 整体不可信
```

### 11.2 L1 因果图验证

| 指标 | 通过标准 | 告警标准 |
|------|----------|----------|
| 图稳定性指数 GSI | ≥ 0.80 | 0.60-0.80 |
| 领域一致性总分 DKCS | ≥ 0.90 | 0.75-0.90 |
| 禁止边违反率 | 0% | > 0% |
| 必选边覆盖率 | ≥ 90% | 70%-90% |
| 弱边比例(p>0.05) | ≤ 5% | 5%-15% |
| **综合得分 CGVS** | **≥ 0.80** | **0.60-0.80** |

### 11.3 L2 因果效应估计验证

| 指标 | 通过标准 | 告警标准 |
|------|----------|----------|
| 安慰剂/真实ATE比 | ≤ 0.10 | 0.10-0.25 |
| ATE变异系数(子集反驳) | ≤ 0.15 | 0.15-0.30 |
| IV第一阶段F统计量 | ≥ 10 | 5-10(弱IV) |
| 鲁棒性分数 Γ_critical | ≥ 1.5 | 1.2-1.5 |
| E-value | ≥ 2.0 | 1.5-2.0 |
| CATE方法间Spearman ρ | ≥ 0.75 | 0.60-0.75 |
| **综合得分 CEVS** | **≥ 0.75** | **0.55-0.75** |

### 11.4 L3 反事实推理验证

| 指标 | 通过标准 | 告警标准 |
|------|----------|----------|
| 综合因果违反率 CCR | ≤ 2% | 2%-8% |
| 不可变约束违反率 | 0% | > 0% |
| 马氏距离中位数 | ≤ 3.0 | 3.0-5.0 |
| 已知效果方向一致率 | 100% | 80%-100% |
| **综合得分 CFVS** | **≥ 0.75** | **0.55-0.75** |

### 11.5 L4 端到端验证

| 指标 | 通过标准 | 告警标准 |
|------|----------|----------|
| AUC提升(vs纯ML) | ≥ +0.01 | 0~+0.01 |
| ECE | ≤ 0.03 | 0.03-0.06 |
| Demographic Parity差异 | ≤ 0.05 | 0.05-0.10 |
| **综合得分 E2EVS** | **≥ 0.70** | **0.50-0.70** |

### 11.6 全局综合判定

```
CCGS = 0.25·CGVS + 0.30·CEVS + 0.25·CFVS + 0.20·E2EVS

CCGS ≥ 0.75  →  EXCELLENT (可全面部署)
0.60 ≤ CCGS < 0.75  →  GOOD (基本可信，需标注不确定性)
0.45 ≤ CCGS < 0.60  →  MARGINAL (仅限辅助参考)
CCGS < 0.45  →  INSUFFICIENT (需重新设计)
```

---

## 12. 部署方案

### 12.1 容器化部署

**Kubernetes集群架构（EKS 1.29）**：

| 组件 | 副本数 | 资源 | 伸缩策略 |
|------|--------|------|---------|
| Backend (FastAPI) | 3-10 | CPU:2 / Mem:4Gi | HPA by QPS |
| Frontend (Next.js) | 2-5 | CPU:1 / Mem:2Gi | HPA by CPU |
| Triton (GPU) | 2-8 | A10G×1 / Mem:16Gi | HPA by QPS+GPU利用率 |
| Celery Worker | 4 | CPU:2 / Mem:8Gi | 按队列深度 |
| Redis (ElastiCache) | 3 Nodes | r6g.large | Cluster Mode |
| PostgreSQL (RDS) | Multi-AZ | r6g.xlarge / 500GB | Citus分片 |
| Airflow | Scheduler+Web+Workers | — | — |
| Kafka (MSK) | 3 Brokers | m5.large / 500GB | — |

**GPU显存分配方案（A10G 24GB）**：TensorRT评分模型2GB(INT8) + 因果推理模型4GB(FP16) + 反事实生成3GB(FP16) + CUDA Memory Pool 1GB + CUDA Graphs 2GB + I/O缓冲2GB + 系统预留10GB

### 12.2 CI/CD流水线

```
Push/PR → [Lint: Ruff+ESLint] → [Test: pytest+Jest] → [Build: Docker+Triton导出]
→ [Security: Trivy+Bandit] → [Deploy: ArgoCD GitOps]
```

### 12.3 监控告警体系

| 监控维度 | 工具 | 关键指标 | 告警阈值 |
|---------|------|---------|---------|
| 模型性能 | Evidently+Prometheus | AUC/KS/Gini | AUC下降>0.02 |
| 特征漂移 | Evidently | PSI | PSI>0.2 |
| GPU监控 | DCGM Exporter | 利用率/显存/温度 | 利用率>90%或<10% |
| 推理延迟 | Prometheus | P50/P99 | P99>200ms |
| 业务指标 | Grafana | 评分量/成功率/违约率 | 违约率偏离>2σ |

### 12.4 安全合规

| 维度 | 方案 |
|------|------|
| 数据加密 | AES-256静态加密 + TLS 1.3传输加密 + FPE格式保留加密(收入等字段) |
| 认证授权 | OAuth2.0+OIDC + RBAC角色权限 + JWT Token |
| 审计日志 | 7年保留(S3 Object Lock WORM) + 全操作可追溯 |
| 模型治理 | Model Card文档化 + 公平性月度报告 + 因果验证报告随模型版本绑定 |
| 降级策略 | 因果推理超时→返回纯ML评分 + GPU故障→CPU降级推理 |

### 12.5 GPU推理部署

**Triton混合推理架构**：
- **credit_score_xgboost**：TensorRT后端，INT8量化，延迟~2ms，动态批处理[8,16,32]
- **causal_inference_dowy**：Python后端，FP16，延迟~15ms，动态批处理[4,8]
- **counterfactual_generator**：Python后端，FP16，延迟~50ms

**性能基准（A10G × 2）**：
- 纯评分模型：~50K records/s
- 评分+因果推理：~8.3K records/s
- 评分+因果+反事实：~1.7K records/s

---

## 13. 项目结构与实施计划

### 13.1 完整目录树

```
causal-credit/
├── backend/                    # Python后端 (FastAPI+Celery+Redis+PostgreSQL+Kafka)
│   ├── alembic/                # 数据库迁移
│   ├── app/
│   │   ├── main.py             # FastAPI应用入口
│   │   ├── config.py           # Pydantic Settings配置
│   │   ├── api/v1/             # API路由 (score/explain/counterfactual/features/models/monitoring)
│   │   ├── core/               # 核心业务 (scoring/causal/explain/features)
│   │   ├── models/             # SQLAlchemy数据模型
│   │   ├── schemas/            # Pydantic请求/响应模型
│   │   └── utils/              # 工具 (triton_client/redis_client/kafka_client/logger)
│   ├── workers/                # Celery异步任务
│   └── tests/                  # 单元/集成/E2E测试
├── frontend/                   # Next.js前端 (TS+Ant Design Pro+ECharts+G6+tRPC)
│   └── src/
│       ├── app/                # App Router页面
│       ├── components/         # 组件库 (charts/layout/business)
│       ├── hooks/              # 自定义Hooks
│       ├── stores/             # Zustand状态
│       └── types/              # TypeScript类型
├── ml/                         # ML工程
│   ├── pipelines/              # 特征工程Pipeline (causal/temporal/cross)
│   ├── models/                 # 模型定义 (credit_score/causal/explain)
│   ├── training/               # 训练脚本+配置
│   ├── data/                   # 数据处理 (ingestion/preprocessing/validation)
│   └── feast/                  # Feature Store配置
├── triton/                     # Triton推理服务
│   └── model_repository/       # 评分模型(TensorRT)+因果推理(Python)+反事实(Python)
├── airflow/                    # Airflow DAGs (训练/特征/评分/监控/质量)
├── infra/                      # 基础设施 (Helm Charts+Terraform+K8s清单)
├── monitoring/                 # 监控配置 (Prometheus+Grafana+Evidently)
├── docs/                       # 文档
└── scripts/                    # 工具脚本
```

### 13.2 六周分阶段实施计划

#### Week 1：数据基础与因果图构建

| 任务 | 交付物 | 验收标准 |
|------|--------|---------|
| W1-T1: Home Credit数据加载与多表关联 | 数据加载模块+Medallion分层 | 8表关联完整，Gold层特征宽表就绪 |
| W1-T2: 数据清洗与质量检查 | 清洗Pipeline+GE套件 | 缺失值处理完成，质量报告通过 |
| W1-T3: 因果图构建(PC+NOTEARS+领域注入) | 因果DAG+反驳验证报告 | CGVS≥0.60(WARNING以上) |
| W1-T4: 因果变量识别与映射 | Treatment/Outcome/Confounder/IV映射表 | AMT_CREDIT为Treatment确认 |
| W1-T5: 特征工程Pipeline A(因果特征) | 因果特征生成模块 | 7个因果特征生成并验证 |

#### Week 2：特征工程与预测基座

| 任务 | 交付物 | 验收标准 |
|------|--------|---------|
| W2-T1: 特征工程Pipeline B+C(时序+交叉) | 时序+交叉特征生成模块 | 15+时序特征+10+交叉特征 |
| W2-T2: LightGBM+XGBoost模型训练 | 双模型+MLflow实验记录 | AUC≥0.78 |
| W2-T3: 模型融合与TensorRT导出 | 融合模型+Triton模型文件 | 融合AUC≥0.79，TensorRT推理<5ms |

#### Week 3：因果效应估计与CATE（🚨Go/No-Go检查点）

| 任务 | 交付物 | 验收标准 |
|------|--------|---------|
| W3-T1: EconML CATE估计(DML+DR+CF) | CATE估计模块+3方法结果 | CATE方法间Spearman ρ≥0.60 |
| W3-T2: CATE反驳验证与敏感性分析 | 反驳验证报告+E-value | CEVS≥0.55(WARNING以上) |
| W3-T3: 🚨Go/No-Go检查 | 因果推理可行性评估报告 | CGVS≥0.60且CEVS≥0.55→继续；否则调整Treatment/混淆变量 |

#### Week 4：可解释性与反事实推理

| 任务 | 交付物 | 验收标准 |
|------|--------|---------|
| W4-T1: SHAP+因果图联合可解释性 | 可解释性引擎+四象限一致性图 | 每个评分决策有SHAP+因果双层解释 |
| W4-T2: DiCE反事实推理+因果约束 | 反事实生成模块+NSGA-II优化 | CCR≤5%，马氏距离中位数≤5.0 |

#### Week 5：系统集成与API开发

| 任务 | 交付物 | 验收标准 |
|------|--------|---------|
| W5-T1: FastAPI评分服务+Triton集成 | 评分API+Triton部署 | P99延迟<100ms |
| W5-T2: Next.js前端(评分仪表盘+因果面板) | 前端应用 | 评分/因果图/反事实模拟器可交互 |
| W5-T3: 端到端集成测试 | 集成测试报告 | 9阶段流程端到端通过 |

#### Week 6：文档完善与比赛准备

| 任务 | 交付物 | 验收标准 |
|------|--------|---------|
| W6-T1: 因果推理验证全量执行 | L1-L4验证报告+CCGS得分 | CCGS≥0.60(GOOD以上) |
| W6-T2: 比赛演示材料准备 | 演示PPT+Demo视频+技术白皮书 | 演示流程完整，亮点突出 |
| W6-T3: 部署方案文档化 | 部署手册+运维Runbook | 可按文档一键部署 |

---

## 14. 风险点与应对

### 14.1 技术风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|---------|
| 因果发现结果不稳定 | 高 | 高 | 多随机种子集成(10次取交集)+领域知识约束+反驳验证+人工审核 |
| CATE估计置信区间过宽 | 中 | 高 | 优先选择变异充足的Treatment+DR双重鲁棒+Bootstrap CI+CATE仅用于方向性指导 |
| GPU推理延迟不达标 | 中 | 中 | 评分与因果推理解耦(评分同步+因果异步)+动态批处理+CUDA Graphs+降级策略 |
| 反事实方案因果不合理 | 中 | 中 | 因果约束惩罚项调优+NSGA-II全局搜索+生成后验证+人工审核Top-5 |

### 14.2 数据风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|---------|
| Home Credit特征缺失率高 | 高 | 高 | >70%剔除+50-70%MICE插补+缺失指示特征+完整案例分析 |
| 数据分布漂移 | 中 | 高 | PSI>0.2告警+每周漂移报告+因果结构相对稳定+自动重训练 |
| TARGET不区分违约类型 | 高 | 中 | 因果路径聚类替代欺诈标签+CATE异质性辅助识别+方案中明确标注局限 |

### 14.3 业务风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|---------|
| 因果推理结论与业务直觉冲突 | 中 | 高 | 因果图变更需领域专家审核+SHAP-因果四象限辅助+A/B测试验证+保留专家规则安全网 |
| 薄信用人群评分不稳定 | 中 | 中 | 薄信用单独建模+反事实方案增加可行性校验+低置信评分标记"建议人工审核" |
| 模型可解释性过度承诺 | 中 | 中 | 所有因果结论附带CI和p值+标注"因果效应估计"非"真值"+反驳验证结果随解释展示 |

### 14.4 合规风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|---------|
| EU AI Act高风险AI合规 | 高 | 极高 | SHAP+因果图双层解释+因果公平性验证+Model Card文档化+人工审批流程 |
| 个人信息保护法合规 | 中 | 高 | 数据最小化+字段级加密+GDPR删除请求72h处理+审计日志7年保留 |
| 算法公平性争议 | 中 | 极高 | Demographic Parity+Equalized Odds+因果公平性验证+月度公平性报告+不达标阻断上线 |

---

> **文档版本**: v2.0  
> **编制人**: 大卫-解决方案架构师  
> **整合来源**: 阿信行业调研报告 + 艾伦欺诈行为分析/反欺诈覆盖分析 + 苏珊技术架构蓝图/数据集可用性验证 + 大卫因果推理验证标准体系/完整实现计划书
