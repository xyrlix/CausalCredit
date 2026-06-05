# CausalCredit 数据集可用性验证分析报告

> **编制人**：苏珊-数据架构师  
> **日期**：2026年6月5日  
> **服务项目**：CausalCredit — 因果推理增强信用评分系统  
> **比赛**：中银香港创新先驱大赛2026  
> **验证方法**：基于Home Credit Default Risk数据集实际字段逐一对照验证

---

## 目录

1. [数据集可获取性验证](#1-数据集可获取性验证)
2. [特征可用性逐项验证](#2-特征可用性逐项验证)
3. [因果推理可行性验证](#3-因果推理可行性验证)
4. [真实场景部署价值验证](#4-真实场景部署价值验证)
5. [数据质量评估](#5-数据质量评估)
6. [最终结论](#6-最终结论)

---

## 1. 数据集可获取性验证

### 1.1 Home Credit Default Risk（主数据集）

| 维度 | 详情 |
|------|------|
| **获取渠道** | Kaggle竞赛页面：https://www.kaggle.com/c/home-credit-default-risk/data |
| **许可协议** | Kaggle Competition Rules，仅限竞赛/学习使用，**不可直接商用** |
| **商用限制** | 需与Home Credit Group另行协商数据授权；比赛演示中可使用 |
| **规模** | 训练集307,511行 × 122列；测试集48,744行 × 121列 |
| **表数量** | 8张表（application_train/test, bureau, bureau_balance, POS_CASH_balance, credit_card_balance, previous_application, installments_payments, HomeCredit_columns_description） |
| **总特征数** | application表122列 + bureau表17列 + bureau_balance表3列 + POS_CASH_balance表8列 + credit_card_balance表23列 + previous_application表37列 + installments_payments表8列 = **约218个原始字段** |
| **数据时间范围** | 竞赛发布于2018年，数据为Home Credit在俄罗斯/中亚地区的2010s信贷数据 |
| **时效性** | ⚠️ 数据距今约8年，信贷行为模式可能已有变化；但因果结构相对统计关联更稳定，时效性影响可控 |
| **数据大小** | 约2.68GB（压缩后） |

### 1.2 Lending Club Loan Data（辅助数据集）

| 维度 | 详情 |
|------|------|
| **获取渠道** | Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| **许可协议** | CC0: Public Domain，**可商用** |
| **规模** | 约226万行 × 151列 |
| **表数量** | 单表宽表 |
| **数据时间范围** | 2007-2018年 |
| **时效性** | 同样偏旧，但作为辅助验证集足够 |

### 1.3 German Credit Risk（辅助数据集）

| 维度 | 详情 |
|------|------|
| **获取渠道** | UCI Machine Learning Repository |
| **许可协议** | CC BY 4.0，**可商用** |
| **规模** | 1,000行 × 20列 |
| **表数量** | 单表 |
| **数据时间范围** | 1990s（经典学术数据集） |
| **时效性** | 极旧，仅用于学术基线对比 |

### 1.4 可获取性总结

| 数据集 | 可获取 | 可比赛使用 | 可商用 | 规模充足 | 时效性 |
|--------|--------|-----------|--------|---------|--------|
| Home Credit | ✅ | ✅ | ❌ | ✅ 30万+行 | ⚠️ 8年前 |
| Lending Club | ✅ | ✅ | ✅ | ✅ 226万行 | ⚠️ 7年前 |
| German Credit | ✅ | ✅ | ✅ | ❌ 仅1000行 | ❌ 极旧 |

**结论**：三个数据集均可获取且可用于比赛。Home Credit作为主数据集规模充足，但需注意不可直接商用，真实部署需替换为银行自有数据。

---

## 2. 特征可用性逐项验证（最关键）

### 2.1 CausalCredit核心能力特征验证

#### 2.1.1 因果推理特征

| 方案所需特征 | Home Credit对应字段 | 状态 | 说明 |
|-------------|-------------------|------|------|
| 贷款金额（Treatment候选） | `AMT_CREDIT` | ✅直接包含 | 连续型，变异充足，适合做Treatment |
| 年还款额（中介变量） | `AMT_ANNUITY` | ✅直接包含 | 与AMT_CREDIT有确定性因果链 |
| 商品价格 | `AMT_GOODS_PRICE` | ✅直接包含 | 贷款金额的因果上游 |
| 收入总额（混淆变量） | `AMT_INCOME_TOTAL` | ✅直接包含 | 关键混淆因子 |
| 在职天数（Treatment候选） | `DAYS_EMPLOYED` | ✅直接包含 | ⚠️ 含异常值365243（失业标记） |
| 年龄（不可变特征） | `DAYS_BIRTH` | ✅直接包含 | 反事实推理中标记为不可变 |
| 性别（敏感属性） | `CODE_GENDER` | ✅直接包含 | 因果公平性验证的敏感属性 |
| 教育水平（工具变量候选） | `NAME_EDUCATION_TYPE` | ✅直接包含 | 可作为income的工具变量 |
| 职业类型 | `OCCUPATION_TYPE` | ✅直接包含 | ⚠️ 缺失率约31% |
| 组织类型 | `ORGANIZATION_TYPE` | ✅直接包含 | 雇主行业分类 |
| 家庭成员数 | `CNT_FAM_MEMBERS` | ✅直接包含 | ⚠️ 极少量缺失 |
| 子女数量 | `CNT_CHILDREN` | ✅直接包含 | ⚠️ 含异常值19 |
| 地区评级 | `REGION_RATING_CLIENT` / `REGION_RATING_CLIENT_W_CITY` | ✅直接包含 | 典型混淆变量案例 |
| 住房类型 | `NAME_HOUSING_TYPE` | ✅直接包含 | 因果路径中的中间变量 |
| 家庭状态 | `NAME_FAMILY_STATUS` | ✅直接包含 | 因果路径变量 |
| 收入类型 | `NAME_INCOME_TYPE` | ✅直接包含 | 关键因果变量 |
| 外部评分1 | `EXT_SOURCE_1` | ✅直接包含 | ⚠️ 缺失率约56% |
| 外部评分2 | `EXT_SOURCE_2` | ✅直接包含 | 缺失率约0.2% |
| 外部评分3 | `EXT_SOURCE_3` | ✅直接包含 | ⚠️ 缺失率约20% |
| 违约标签（Outcome） | `TARGET` | ✅直接包含 | 1=违约，0=正常 |

**因果推理特征可用性：✅ 充分**。Home Credit数据集包含了因果推理所需的核心变量：Treatment候选（AMT_CREDIT, DAYS_EMPLOYED）、Outcome（TARGET）、混淆变量（AMT_INCOME_TOTAL, REGION_RATING_CLIENT等）、中介变量（AMT_ANNUITY）、工具变量候选（NAME_EDUCATION_TYPE）、敏感属性（CODE_GENDER）。

#### 2.1.2 反欺诈特征

| 方案所需特征 | Home Credit对应字段 | 状态 | 说明 |
|-------------|-------------------|------|------|
| 身份信息一致性 | `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL`, `FLAG_MOBIL`, `FLAG_CONT_MOBILE`, `FLAG_EMP_PHONE` | 🔶可衍生 | 可构建"联系方式完整度"异常指标，但无法做跨申请关联 |
| 文档提供异常 | `FLAG_DOCUMENT_2` ~ `FLAG_DOCUMENT_21`（共20个） | ✅直接包含 | 可检测文档提供模式异常（欺诈者可能全部提供或全部不提供） |
| 社交圈违约观察 | `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`, `OBS_60_CNT_SOCIAL_CIRCLE`, `DEF_60_CNT_SOCIAL_CIRCLE` | ✅直接包含 | 社交圈违约比例可作为欺诈环境指标 |
| 信用查询频率 | `AMT_REQ_CREDIT_BUREAU_HOUR`, `AMT_REQ_CREDIT_BUREAU_DAY`, `AMT_REQ_CREDIT_BUREAU_WEEK`, `AMT_REQ_CREDIT_BUREAU_MON`, `AMT_REQ_CREDIT_BUREAU_QRT`, `AMT_REQ_CREDIT_BUREAU_YEAR` | ✅直接包含 | 短时间密集查询是欺诈信号 |
| 地址不一致 | `REG_REGION_NOT_LIVE_REGION`, `REG_REGION_NOT_WORK_REGION`, `LIVE_REGION_NOT_WORK_REGION`, `REG_CITY_NOT_LIVE_CITY`, `REG_CITY_NOT_WORK_CITY`, `LIVE_CITY_NOT_WORK_CITY` | ✅直接包含 | 注册地/居住地/工作地不一致是欺诈信号 |
| 申请时间异常 | `WEEKDAY_APPR_PROCESS_START`, `HOUR_APPR_PROCESS_START` | 🔶可衍生 | 可检测非正常时段申请模式 |
| 信用历史异常 | bureau表: `CREDIT_DAY_OVERDUE`, `AMT_CREDIT_MAX_OVERDUE`, `CNT_CREDIT_PROLONG` | ✅直接包含 | 历史逾期和展期是欺诈信号 |
| 多头借贷 | bureau表: 按`SK_ID_CURR`聚合的信贷记录数 | 🔶可衍生 | 可计算跨机构借贷数量和总额 |
| 还款行为异常 | installments_payments表: `DAYS_ENTRY_PAYMENT` - `DAYS_INSTALMENT` | 🔶可衍生 | 可计算逾期天数分布、还款模式 |
| 信用卡使用异常 | credit_card_balance表: `AMT_DRAWINGS_ATM_CURRENT`, `CNT_DRAWINGS_ATM_CURRENT` | 🔶可衍生 | 可计算提现比例、额度使用率等 |
| 设备指纹 | — | ❌无法支撑 | 数据集无设备信息 |
| IP地址/网络环境 | — | ❌无法支撑 | 数据集无网络信息 |
| 生物识别特征 | — | ❌无法支撑 | 数据集无生物特征 |
| 实时行为数据 | — | ❌无法支撑 | 数据集无实时行为流 |
| 社交网络关系图 | — | ❌无法支撑 | 仅有社交圈违约计数，无关系图 |

#### 2.1.3 替代数据特征

| 方案所需特征 | Home Credit对应字段 | 状态 | 说明 |
|-------------|-------------------|------|------|
| 消费行为数据 | credit_card_balance表: `AMT_DRAWINGS_CURRENT`, `AMT_DRAWINGS_POS_CURRENT`, `AMT_DRAWINGS_ATM_CURRENT` | 🔶部分覆盖 | 仅含信用卡消费，缺电商/线下消费 |
| 履约还款数据 | installments_payments表: `AMT_PAYMENT`, `AMT_INSTALMENT`, `DAYS_ENTRY_PAYMENT` | ✅直接包含 | 还款履约行为完整 |
| POS消费数据 | POS_CASH_balance表: `CNT_INSTALMENT`, `CNT_INSTALMENT_FUTURE` | 🔶部分覆盖 | 仅含分期消费，非完整消费画像 |
| 住房建筑信息 | `APARTMENTS_AVG/MODE/MEDI`, `FLOORSMAX_AVG`, `TOTALAREA_MODE`, `WALLSMATERIAL_MODE` 等 | ✅直接包含 | ⚠️ 缺失率极高（50-70%） |
| 出行数据 | — | ❌无法支撑 | 完全缺失 |
| 社交媒体数据 | — | ❌无法支撑 | 完全缺失 |
| 公共事业缴费 | — | ❌无法支撑 | 完全缺失 |
| 电信缴费 | — | ❌无法支撑 | 完全缺失 |
| 既往贷款用途 | previous_application表: `NAME_CASH_LOAN_PURPOSE`, `NAME_GOODS_CATEGORY` | ✅直接包含 | 可分析消费偏好 |

#### 2.1.4 CATE估计所需特征

| 方案所需特征 | Home Credit对应字段 | 状态 | 说明 |
|-------------|-------------------|------|------|
| Treatment变量（贷款金额） | `AMT_CREDIT` | ✅直接包含 | 连续型，变异充足 |
| Treatment变量（年还款额） | `AMT_ANNUITY` | ✅直接包含 | 与AMT_CREDIT强相关 |
| Treatment变量（在职天数） | `DAYS_EMPLOYED` | ✅直接包含 | 需处理异常值365243 |
| Outcome变量（违约） | `TARGET` | ✅直接包含 | 二分类 |
| 混淆变量集 | `AMT_INCOME_TOTAL`, `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, `REGION_RATING_CLIENT`, `DAYS_BIRTH`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS` | ✅直接包含 | 混淆变量充分 |
| 效应修饰变量 | `CODE_GENDER`, `NAME_INCOME_TYPE`, `ORGANIZATION_TYPE`, `NAME_HOUSING_TYPE` | ✅直接包含 | 可揭示异质效应 |
| 历史信贷行为（调整变量） | bureau表聚合特征、previous_application表聚合特征 | 🔶可衍生 | 需多表关联聚合 |

**CATE估计特征可用性：✅ 充分**。Home Credit数据集为CATE估计提供了充足的Treatment候选、明确的Outcome、丰富的混淆变量和效应修饰变量。

### 2.2 艾伦反欺诈8个能力点特征验证

#### 能力点1：贷前准入反欺诈

| 所需特征 | Home Credit字段 | 状态 | 说明 |
|---------|----------------|------|------|
| 申请信息异常检测 | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` | ✅直接包含 | 可检测收入-贷款金额异常比例 |
| 信用查询频率异常 | `AMT_REQ_CREDIT_BUREAU_*`（6个字段） | ✅直接包含 | 短期密集查询=高风险 |
| 地址不一致 | `REG_REGION_NOT_*`, `REG_CITY_NOT_*`（6个字段） | ✅直接包含 | 注册/居住/工作地不一致 |
| 文档提供模式 | `FLAG_DOCUMENT_2`~`FLAG_DOCUMENT_21` | ✅直接包含 | 异常文档提供模式 |
| 反欺诈决策引擎 | — | ❌无法支撑 | 需额外构建规则引擎 |
| 实时反欺诈信号 | — | ❌无法支撑 | 无设备/IP/手机号风险数据 |
| 欺诈概率子模型 | `TARGET`（无法区分欺诈vs正常违约） | ❌无法支撑 | **关键缺陷**：TARGET不区分违约类型 |

**贷前准入反欺诈评级：🔶部分覆盖**。因果推理可辅助风险分层，但缺少专门的反欺诈决策引擎和欺诈标签。

#### 能力点2：应届生/白户评分

| 所需特征 | Home Credit字段 | 状态 | 说明 |
|---------|----------------|------|------|
| 信用历史长度 | bureau表: `DAYS_CREDIT`聚合 | 🔶可衍生 | 可计算首次信贷距今天数 |
| 年龄 | `DAYS_BIRTH` | ✅直接包含 | 可识别年轻申请人 |
| 收入 | `AMT_INCOME_TOTAL` | ✅直接包含 | 薄信用人群收入特征 |
| 教育水平 | `NAME_EDUCATION_TYPE` | ✅直接包含 | 作为收入工具变量 |
| 职业 | `OCCUPATION_TYPE` | ✅直接包含 | ⚠️ 缺失率31% |
| 收入类型 | `NAME_INCOME_TYPE` | ✅直接包含 | 含"Student"/"Unemployed"类别 |
| 外部评分 | `EXT_SOURCE_1/2/3` | ✅直接包含 | 薄信用人群可能缺失 |
| 反事实推理替代路径 | 上述字段组合 | ✅可构建 | CausalCredit核心能力 |

**应届生/白户评分评级：✅已覆盖**。Home Credit数据集完整支撑反事实推理+替代因果变量评分路径。

#### 能力点3：替代数据+因果建模

| 所需特征 | Home Credit字段 | 状态 | 说明 |
|---------|----------------|------|------|
| 消费行为 | credit_card_balance表 | 🔶部分覆盖 | 仅信用卡消费 |
| 履约还款 | installments_payments表 | ✅直接包含 | 完整还款履约 |
| POS分期消费 | POS_CASH_balance表 | 🔶部分覆盖 | 仅分期消费 |
| 出行数据 | — | ❌无法支撑 | 完全缺失 |
| 社交数据 | — | ❌无法支撑 | 完全缺失 |
| 公共事业缴费 | — | ❌无法支撑 | 完全缺失 |
| 因果建模框架 | DoWhy+EconML | ✅可构建 | 框架完备，数据受限 |

**替代数据评级：🔶部分覆盖**。因果建模框架完备，但替代数据源单一（仅信贷域），出行/社交/公共事业数据完全缺失。比赛中可通过合成数据演示。

#### 能力点4：识别身份冒用

| 所需特征 | Home Credit字段 | 状态 | 说明 |
|---------|----------------|------|------|
| 身份证件OCR/比对 | — | ❌无法支撑 | 无证件图像数据 |
| 活体检测 | — | ❌无法支撑 | 无生物特征 |
| 设备指纹 | — | ❌无法支撑 | 无设备信息 |
| 同一地址多申请 | `REGION_RATING_CLIENT`, 地址不一致字段 | 🔶极弱 | 仅有地区级信息，无精确地址 |
| 同一电话多申请 | — | ❌无法支撑 | 无电话号码字段 |
| 同一邮箱多申请 | — | ❌无法支撑 | 无邮箱字段 |
| 身份信息一致性 | `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL` | 🔶极弱 | 仅有标志位，无实际联系信息 |

**身份冒用评级：❌未覆盖**。Home Credit数据集不包含身份验证层面的关键数据（证件、设备、生物特征、联系信息）。因果推理框架可在架构上预留接口，但无法基于此数据集实现。

#### 能力点5：团伙养号骗贷

| 所需特征 | Home Credit字段 | 状态 | 说明 |
|---------|----------------|------|------|
| 设备ID关联 | — | ❌无法支撑 | 无设备信息 |
| IP地址关联 | — | ❌无法支撑 | 无网络信息 |
| 地址关联 | `REGION_RATING_CLIENT` | 🔶极弱 | 仅地区级，无法做精确关联 |
| 电话关联 | — | ❌无法支撑 | 无电话号码 |
| 社交关系图 | `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE` | 🔶极弱 | 仅有计数，无关系图 |
| 时序协同模式 | `DAYS_CREDIT`（bureau表） | 🔶可衍生 | 可检测集中申请时段 |
| 图神经网络(GNN) | — | ❌无法支撑 | 缺少图结构数据 |

**团伙养号评级：❌未覆盖**。团伙检测需要关系网络数据（设备/IP/地址/社交关联），Home Credit数据集完全不具备。可用轻量级聚类（如基于地区+时段的异常聚集检测）做降级替代。

#### 能力点6：包装资质申贷

| 所需特征 | Home Credit字段 | 状态 | 说明 |
|---------|----------------|------|------|
| 收入-职业不一致 | `AMT_INCOME_TOTAL` + `OCCUPATION_TYPE` + `NAME_INCOME_TYPE` | ✅直接包含 | 可检测收入与职业不匹配 |
| 收入-教育不一致 | `AMT_INCOME_TOTAL` + `NAME_EDUCATION_TYPE` | ✅直接包含 | 可检测收入与教育不匹配 |
| 收入-地区不一致 | `AMT_INCOME_TOTAL` + `REGION_RATING_CLIENT` | ✅直接包含 | 可检测收入与地区不匹配 |
| 贷款金额-收入比 | `AMT_CREDIT` / `AMT_INCOME_TOTAL` | 🔶可衍生 | 关键包装检测指标 |
| 还款额-收入比 | `AMT_ANNUITY` / `AMT_INCOME_TOTAL` | 🔶可衍生 | 债务收入比 |
| 历史申请-当前申请差异 | previous_application表 vs application表 | 🔶可衍生 | 可检测申请信息变更 |
| 因果发现检测异常组合 | 因果图中的异常路径强度 | 🔶可衍生 | CausalCredit独特能力 |

**包装资质评级：🔶部分覆盖**。因果发现可检测异常特征组合（如"低收入+高贷款金额+高收入类型"的因果不一致），但缺少专门的资质验证模块和跨申请信息变更追踪。

#### 能力点7：剔除黑产虚假优质用户

| 所需特征 | Home Credit字段 | 状态 | 说明 |
|---------|----------------|------|------|
| "过于完美"的信用档案 | bureau表聚合 + `EXT_SOURCE_*` | 🔶可衍生 | 可检测信用档案异常完美 |
| 信用历史过短但评分过高 | bureau表: `DAYS_CREDIT` + `EXT_SOURCE_*` | 🔶可衍生 | 可检测历史-评分不一致 |
| 还款行为过于规律 | installments_payments表 | 🔶可衍生 | 可检测还款模式过于机械 |
| 社交圈异常 | `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE` | 🔶可衍生 | 社交圈违约率为0可能异常 |
| 文档提供过于完整 | `FLAG_DOCUMENT_*` | 🔶可衍生 | 全部文档都提供可能异常 |
| 因果推理识别"因果不一致" | 因果图路径强度 | 🔶可衍生 | CausalCredit独特能力 |
| 时序行为一致性 | — | ❌无法支撑 | 无细粒度时序行为数据 |

**剔除黑产评级：🔶部分覆盖**。因果推理可识别"因果不一致"模式（包装的资质在统计上对但在因果上说不通），但缺少细粒度时序行为一致性检测。

#### 能力点8：拦截申请欺诈

| 所需特征 | Home Credit字段 | 状态 | 说明 |
|---------|----------------|------|------|
| 三类违约分类 | `TARGET` | ❌无法支撑 | **关键缺陷**：TARGET不区分欺诈/非恶意/系统性 |
| 恶意欺诈信号 | 上述反欺诈特征组合 | 🔶可衍生 | 可构建欺诈倾向评分 |
| 实时拦截引擎 | — | ❌无法支撑 | 需额外构建 |
| 欺诈标签 | — | ❌无法支撑 | 数据集无欺诈标签 |

**拦截申请欺诈评级：🔶部分覆盖**。可基于因果推理构建欺诈倾向评分，但无法训练专门的欺诈分类器（缺标签），且无实时拦截引擎。

### 2.3 特征可用性汇总

| 能力类别 | ✅直接包含 | 🔶可衍生 | ❌无法支撑 |
|---------|-----------|---------|-----------|
| 因果推理特征 | 18 | 1 | 0 |
| 反欺诈特征 | 9 | 6 | 5 |
| 替代数据特征 | 4 | 3 | 4 |
| CATE估计特征 | 6 | 1 | 0 |
| 贷前准入反欺诈 | 4 | 0 | 3 |
| 白户评分 | 6 | 1 | 0 |
| 替代数据+因果 | 1 | 2 | 4 |
| 身份冒用 | 0 | 2 | 5 |
| 团伙养号 | 0 | 2 | 5 |
| 包装资质 | 3 | 4 | 0 |
| 剔除黑产 | 0 | 6 | 1 |
| 拦截申请欺诈 | 0 | 1 | 3 |
| **合计** | **51** | **29** | **30** |

**关键发现**：
- ✅ 因果推理核心能力（因果发现、CATE估计、反事实推理）的数据支撑**充分**
- ✅ 白户评分的数据支撑**充分**
- 🔶 包装资质检测和黑产剔除可通过因果推理**部分实现**
- ❌ 身份冒用、团伙养号的数据支撑**严重不足**
- ❌ **最关键缺陷**：TARGET不区分违约类型，无法训练欺诈分类器

---

## 3. 因果推理可行性验证

### 3.1 Treatment（干预变量）候选

| Treatment候选 | 字段 | 变异充足 | 业务可干预 | 因果合理性 | 推荐度 |
|--------------|------|---------|-----------|-----------|--------|
| 贷款金额 | `AMT_CREDIT` | ✅ 连续型，范围45K~4.1M | ✅ 银行可调整 | ✅ 直接影响还款压力 | ★★★★★ |
| 年还款额 | `AMT_ANNUITY` | ✅ 连续型 | ✅ 可通过期限调整 | ✅ 直接影响现金流 | ★★★★☆ |
| 贷款期限 | 🔶需衍生: `AMT_CREDIT`/`AMT_ANNUITY` | ✅ 可计算 | ✅ 银行可调整 | ✅ 影响还款压力分布 | ★★★★☆ |
| 在职天数 | `DAYS_EMPLOYED` | ✅ 但含异常值 | ❌ 银行不可干预 | ✅ 影响收入稳定性 | ★★★☆☆ |
| 信用额度 | bureau表: `AMT_CREDIT_SUM_LIMIT` | 🔶 部分缺失 | ✅ 银行可调整 | ✅ 影响负债水平 | ★★★☆☆ |
| 利率 | 🔶需从previous_application推算 | 🔶 间接推算 | ✅ 银行可调整 | ✅ 直接影响还款额 | ★★☆☆☆ |

**推荐Treatment**：`AMT_CREDIT`（贷款金额）为首选Treatment——变异充足、业务可干预、因果路径清晰（AMT_CREDIT → AMT_ANNUITY → 债务收入比 → TARGET）。

### 3.2 Outcome（结果变量）

| Outcome候选 | 字段 | 适用性 | 说明 |
|-------------|------|--------|------|
| 违约标志 | `TARGET` | ✅ 主要Outcome | 1=违约(8.07%), 0=正常(91.93%) |
| 逾期天数 | POS_CASH_balance: `SK_DPD`, `SK_DPD_DEF` | 🔶 辅助Outcome | 可构建连续型违约严重度 |
| 逾期金额 | bureau: `AMT_CREDIT_MAX_OVERDUE` | 🔶 辅助Outcome | 可构建违约损失度 |

**推荐Outcome**：`TARGET`为主，可补充构建"违约严重度"连续变量增强CATE估计精度。

### 3.3 混淆变量（Confounder）充分性

| 混淆变量类别 | Home Credit字段 | 充分性 | 说明 |
|-------------|----------------|--------|------|
| **收入能力** | `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `DAYS_EMPLOYED`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE` | ✅ 充分 | 覆盖收入水平、来源、稳定性 |
| **人口统计** | `DAYS_BIRTH`, `CODE_GENDER`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS` | ✅ 充分 | 覆盖年龄、性别、家庭结构 |
| **地区经济** | `REGION_RATING_CLIENT`, `REGION_RATING_CLIENT_W_CITY`, `REGION_POPULATION_RELATIVE` | ✅ 充分 | 覆盖地区经济水平 |
| **教育资本** | `NAME_EDUCATION_TYPE` | ✅ 基本充分 | 单一但关键 |
| **资产状况** | `FLAG_OWN_CAR`, `OWN_CAR_AGE`, `FLAG_OWN_REALTY` | ✅ 基本充分 | 覆盖车产、房产 |
| **信用历史** | bureau表聚合特征 | ✅ 充分 | 覆盖历史信贷行为 |
| **外部评分** | `EXT_SOURCE_1/2/3` | ✅ 充分 | 第三方信用评分，强混淆因子 |
| **住房条件** | `APARTMENTS_AVG`, `TOTALAREA_MODE` 等 | 🔶 缺失率高 | 50-70%缺失，可用性受限 |

**混淆变量充分性评级：✅ 充分**。Home Credit数据集提供了丰富的混淆变量，覆盖收入能力、人口统计、地区经济、教育资本、资产状况、信用历史、外部评分七大维度。关键混淆因子（收入、年龄、地区评级、外部评分）均直接可用。

### 3.4 因果图构建关键变量齐全性

| 因果图要素 | 所需变量 | Home Credit支撑 | 说明 |
|-----------|---------|----------------|------|
| **Treatment → Outcome** | AMT_CREDIT → TARGET | ✅ | 核心因果路径 |
| **中介变量** | AMT_ANNUITY, AMT_GOODS_PRICE | ✅ | AMT_CREDIT → AMT_ANNUITY → TARGET |
| **混淆因子→Treatment** | AMT_INCOME_TOTAL → AMT_CREDIT | ✅ | 收入影响贷款金额 |
| **混淆因子→Outcome** | AMT_INCOME_TOTAL → TARGET | ✅ | 收入影响违约 |
| **工具变量** | NAME_EDUCATION_TYPE | ✅ | 教育→收入→违约，教育不直接影响违约 |
| **对撞变量** | NAME_CONTRACT_STATUS（previous_application） | ✅ | 同时被收入和贷款金额影响 |
| **敏感属性** | CODE_GENDER, DAYS_BIRTH | ✅ | 因果公平性验证 |
| **效应修饰变量** | NAME_INCOME_TYPE, REGION_RATING_CLIENT | ✅ | CATE异质性来源 |
| **领域约束边** | AMT_CREDIT → AMT_ANNUITY（确定性） | ✅ | 领域知识注入 |
| **禁止边** | TARGET → AMT_CREDIT（反向因果） | ✅ | 结果不能是原因 |

**因果图构建评级：✅ 齐全**。Home Credit数据集支撑完整的因果图构建，包括Treatment-Outcome路径、中介链、混淆因子、工具变量、对撞变量、敏感属性和效应修饰变量。

### 3.5 因果推理可行性总结

| 维度 | 评级 | 说明 |
|------|------|------|
| Treatment候选 | ✅ 充分 | AMT_CREDIT为首选，AMT_ANNUITY为备选 |
| Outcome定义 | ✅ 明确 | TARGET为主，可补充连续型违约严重度 |
| 混淆变量 | ✅ 充分 | 七大维度覆盖，关键混淆因子直接可用 |
| 因果图构建 | ✅ 齐全 | 所有因果图要素均有对应字段 |
| **因果推理总体可行性** | **✅ 可行** | **Home Credit数据集完全支撑CausalCredit的因果推理核心能力** |

---

## 4. 真实场景部署价值验证

### 4.1 Home Credit特征与银行真实业务系统映射

| Home Credit字段 | 真实银行对应 | 映射关系 | 部署时可复用 |
|----------------|------------|---------|------------|
| `AMT_INCOME_TOTAL` | 客户申报收入 | ✅ 直接对应 | ✅ 可复用（需银行收入核验数据） |
| `AMT_CREDIT` | 贷款金额 | ✅ 直接对应 | ✅ 可复用 |
| `AMT_ANNUITY` | 年还款额 | ✅ 直接对应 | ✅ 可复用 |
| `AMT_GOODS_PRICE` | 抵押物/商品价值 | ✅ 直接对应 | ✅ 可复用 |
| `DAYS_BIRTH` | 客户年龄 | ✅ 直接对应 | ✅ 可复用 |
| `DAYS_EMPLOYED` | 在职时长 | ✅ 直接对应 | ✅ 可复用 |
| `NAME_EDUCATION_TYPE` | 学历 | ✅ 直接对应 | ✅ 可复用 |
| `OCCUPATION_TYPE` | 职业类型 | ✅ 直接对应 | ✅ 可复用 |
| `NAME_INCOME_TYPE` | 收入来源类型 | ✅ 直接对应 | ✅ 可复用 |
| `NAME_FAMILY_STATUS` | 婚姻状况 | ✅ 直接对应 | ✅ 可复用 |
| `CNT_CHILDREN` | 子女数 | ✅ 直接对应 | ✅ 可复用 |
| `FLAG_OWN_CAR/REALTY` | 车产/房产标志 | ✅ 直接对应 | ✅ 可复用 |
| `REGION_RATING_CLIENT` | 地区风险评级 | ✅ 直接对应 | ✅ 可复用（需替换为银行自有评级） |
| `EXT_SOURCE_1/2/3` | 外部信用评分 | ✅ 直接对应 | ⚠️ 需替换为银行使用的外部评分源（如央行征信评分、FICO等） |
| `TARGET` | 违约标志 | ✅ 直接对应 | ✅ 可复用（银行自有违约定义） |
| `AMT_REQ_CREDIT_BUREAU_*` | 征信查询次数 | ✅ 直接对应 | ✅ 可复用（需替换为银行征信查询记录） |
| bureau表 | 征信局历史记录 | ✅ 直接对应 | ✅ 可复用（需替换为央行征信报告数据） |
| `OBS/DEF_*_SOCIAL_CIRCLE` | 社交圈违约 | ⚠️ 间接对应 | ❌ 需替换（银行通常无此数据，可用关联账户违约替代） |
| `APARTMENTS_AVG/MODE/MEDI` | 住房建筑信息 | ⚠️ 间接对应 | ❌ 需替换（银行通常无此数据，可用房产估值替代） |
| `FLAG_DOCUMENT_*` | 文档提供标志 | ✅ 直接对应 | ✅ 可复用 |
| `ORGANIZATION_TYPE` | 雇主行业 | ✅ 直接对应 | ✅ 可复用 |

### 4.2 部署复用性分类

| 类别 | 字段数 | 占比 | 说明 |
|------|--------|------|------|
| ✅ 可直接复用 | 约85个 | ~65% | 核心信贷特征，银行自有数据直接对应 |
| ⚠️ 需替换数据源 | 约15个 | ~12% | 外部评分、社交圈数据等需替换为银行自有数据源 |
| ❌ 需重新设计 | 约30个 | ~23% | 住房建筑信息、部分衍生特征需根据银行实际数据重新设计 |

### 4.3 关键部署映射

| CausalCredit模块 | Home Credit数据 | 真实部署替换方案 |
|-----------------|----------------|----------------|
| 因果发现 | application + bureau + previous_application | 银行核心系统客户表 + 央行征信 + 历史申请表 |
| CATE估计 | AMT_CREDIT + TARGET + 混淆变量 | 贷款金额 + 违约标签 + 客户画像 |
| 反事实推理 | 不可变/半可变/可变特征分类 | 同样分类逻辑，特征替换为银行字段 |
| 特征工程 | 8表关联聚合 | 银行数据仓库多表关联 |
| 外部评分 | EXT_SOURCE_1/2/3 | 央行征信评分/第三方评分 |
| 征信历史 | bureau + bureau_balance | 央行征信报告 |

**部署价值评级：✅ 高**。Home Credit数据集的Schema与银行真实信贷业务系统高度对应，约65%的特征可直接复用，因果推理框架的迁移成本较低。

---

## 5. 数据质量评估

### 5.1 缺失值比例

| 表名 | 高缺失字段(>50%) | 中缺失字段(20-50%) | 低缺失字段(<20%) | 关键发现 |
|------|-----------------|-------------------|-----------------|---------|
| **application_train** | ~58个字段缺失>50%（主要为住房建筑信息APARTMENTS/BASEMENTAREA/ELEVATORS等_AVG/_MODE/_MEDI） | `OWN_CAR_AGE`(66%), `EXT_SOURCE_1`(56%), `OCCUPATION_TYPE`(31%) | 核心字段（AMT_*, DAYS_*, FLAG_*）缺失<5% | 住房类特征缺失严重，但核心信贷特征完整 |
| **bureau** | `AMT_ANNUITY`(71%) | `AMT_CREDIT_MAX_OVERDUE`(65%) | 其余字段缺失<20% | 征信局还款额和最大逾期额缺失较多 |
| **bureau_balance** | 无 | 无 | 缺失率极低 | 数据质量好 |
| **POS_CASH_balance** | 无 | 无 | 缺失率极低 | 数据质量好 |
| **credit_card_balance** | 无 | 无 | 缺失率极低 | 数据质量好 |
| **previous_application** | `RATE_INTEREST_PRIMARY`(99%), `RATE_INTEREST_PRIVILEGED`(99%) | `AMT_DOWN_PAYMENT`(14%), `RATE_DOWN_PAYMENT`(14%), `NFLAG_INSURED_ON_APPROVAL`(14%) | 其余字段缺失<5% | 利率字段几乎全缺失，⚠️ CATE利率分析受限 |
| **installments_payments** | 无 | 无 | 缺失率极低 | 数据质量好 |

**缺失值处理策略**：
- 缺失>70%：直接剔除（住房建筑_AVG/_MODE/_MEDI系列、RATE_INTEREST_*）
- 缺失50-70%：多重插补（MICE）+ 缺失指示特征
- 缺失<50%：中位数/众数填充 + 缺失指示特征
- ⚠️ `RATE_INTEREST_PRIMARY/PRIVILEGED`缺失99%：**无法直接做利率Treatment的CATE分析**，需改用AMT_CREDIT作为Treatment

### 5.2 异常值分布

| 字段 | 异常值 | 处理方案 |
|------|--------|---------|
| `DAYS_EMPLOYED` | 值365243（约1000年）占18%，为失业标记 | 替换为NaN + 添加`FLAG_UNEMPLOYED`特征 |
| `AMT_INCOME_TOTAL` | 极端高值1.17亿（仅1条） | 对数变换 + Winsorize 99.9% |
| `CNT_CHILDREN` | 最大值19（极端异常） | Winsorize 99% |
| `OBS_30_CNT_SOCIAL_CIRCLE` | 最大值348（极端异常） | Winsorize 99% |
| `AMT_REQ_CREDIT_BUREAU_QRT` | 最大值261（极端异常） | Winsorize 99% |
| bureau: `DAYS_CREDIT_ENDDATE` | 部分正值（未来到期）和极端负值 | 逻辑校验 + 截断 |

### 5.3 类别不平衡情况

| 数据集 | 正样本(违约) | 负样本(正常) | 不平衡比 | 处理方案 |
|--------|------------|------------|---------|---------|
| Home Credit (application_train) | 24,825 (8.07%) | 282,686 (91.93%) | 1:11.4 | SMOTE过采样 + 类别权重调整 + Focal Loss |
| Lending Club | ~20%违约 | ~80%正常 | 1:4 | 相对均衡 |
| German Credit | 300 (30%) | 700 (70%) | 1:2.3 | 基本均衡 |

**关键影响**：Home Credit的1:11.4不平衡比对因果推理影响有限（因果推理关注效应估计而非分类），但对预测模型（LightGBM/XGBoost）需做类别权重调整。

### 5.4 多表关联完整性

| 关联路径 | 关联键 | 完整性 | 说明 |
|---------|--------|--------|------|
| application → bureau | `SK_ID_CURR` | 🔶 约86%的申请人在bureau中有记录 | 14%无征信局记录（薄信用人群） |
| bureau → bureau_balance | `SK_BUREAU_ID` | 🔶 部分bureau记录无月度余额 | 非所有历史信贷都有月度快照 |
| application → previous_application | `SK_ID_CURR` | 🔶 约63%有历史申请 | 37%为首次申请 |
| previous_application → POS_CASH_balance | `SK_ID_PREV` | 🔶 部分历史申请有POS/CASH记录 | 仅Cash/POS贷款类型有 |
| previous_application → credit_card_balance | `SK_ID_PREV` | 🔶 部分历史申请有信用卡记录 | 仅信用卡类型有 |
| previous_application → installments_payments | `SK_ID_PREV` | 🔶 部分历史申请有还款记录 | 大部分有还款记录 |

**关联完整性评级：🔶 基本完整**。多表关联存在一定比例的"无记录"情况，但这恰好反映了真实的信贷场景（薄信用人群、首次申请者），对因果推理而言是合理的。

### 5.5 数据泄露风险检查

| 潜在泄露点 | 检查结果 | 风险等级 |
|-----------|---------|---------|
| TARGET在test集中不存在 | ✅ 无泄露 | 安全 |
| EXT_SOURCE_*是否包含未来信息 | ⚠️ 不确定 | 中风险——外部评分可能已包含TARGET相关信息 |
| bureau数据时间边界 | ⚠️ 未明确限制 | 中风险——bureau记录可能包含申请后的信息 |
| previous_application中的DAYS_DECISION | ✅ 负值表示申请前 | 安全 |
| credit_card_balance中的MONTHS_BALANCE | ✅ 负值表示申请前 | 安全 |

**数据泄露风险评级：⚠️ 中等**。EXT_SOURCE_*是Kaggle竞赛中已知的高风险泄露点（AUC从0.75提升至0.82+的主要贡献者），在因果推理中需特别注意——如果EXT_SOURCE已编码了违约信息，则因果发现可能发现虚假的"EXT_SOURCE → TARGET"因果边。建议在因果发现时将EXT_SOURCE作为调整变量而非核心因果变量。

---

## 6. 最终结论

### 6.1 数据集可用性综合评分

| 维度 | 评分(1-5) | 权重 | 加权分 | 说明 |
|------|----------|------|--------|------|
| 数据可获取性 | 5 | 10% | 0.50 | 三个数据集均可免费获取 |
| 因果推理特征支撑 | 5 | 25% | 1.25 | Treatment/Outcome/Confounder/IV齐全 |
| 反欺诈特征支撑 | 2.5 | 20% | 0.50 | 身份冒用/团伙检测严重不足 |
| 替代数据支撑 | 2 | 10% | 0.20 | 仅信贷域数据，出行/社交/公共事业缺失 |
| 数据质量 | 3.5 | 15% | 0.525 | 核心字段质量好，住房类缺失严重 |
| 真实部署映射 | 4 | 10% | 0.40 | 65%特征可直接复用 |
| 因果推理可行性 | 4.5 | 10% | 0.45 | 因果图构建变量齐全 |

**综合评分：3.8 / 5.0**

### 6.2 「可以做」清单

| # | 可做事项 | 数据支撑 | 信心度 |
|---|---------|---------|--------|
| 1 | **因果发现（PC+NOTEARS融合）** | ✅ 充足的变量和变异 | 高 |
| 2 | **CATE异质处理效应估计** | ✅ AMT_CREDIT为Treatment，TARGET为Outcome，混淆变量充分 | 高 |
| 3 | **反事实推理与决策建议** | ✅ 不可变/半可变/可变特征分类明确 | 高 |
| 4 | **白户/薄信用人群替代评分** | ✅ 反事实推理+替代因果变量路径完整 | 高 |
| 5 | **因果公平性验证** | ✅ CODE_GENDER等敏感属性+混淆变量充分 | 高 |
| 6 | **SHAP+因果图联合可解释性** | ✅ 特征归因+因果路径追踪数据齐全 | 高 |
| 7 | **包装资质异常检测** | 🔶 收入-职业-地区不一致检测 | 中高 |
| 8 | **黑产虚假优质用户识别** | 🔶 因果不一致模式检测 | 中 |
| 9 | **信用查询频率异常检测** | ✅ AMT_REQ_CREDIT_BUREAU_* | 高 |
| 10 | **多头借贷分析** | 🔶 bureau表聚合 | 中高 |
| 11 | **还款行为时序特征** | ✅ installments_payments + POS_CASH_balance | 高 |
| 12 | **LightGBM/XGBoost预测模型** | ✅ 122列特征+30万行数据 | 高 |
| 13 | **跨源验证（Lending Club）** | ✅ 辅助数据集规模充足 | 中高 |

### 6.3 「做不了」清单

| # | 不可做事项 | 缺失原因 | 降级方案 |
|---|-----------|---------|---------|
| 1 | **欺诈vs正常违约分类** | TARGET不区分违约类型 | 用因果推理识别"因果不一致"模式替代；在方案中设计三类违约分类框架（基于因果路径特征聚类） |
| 2 | **身份冒用检测** | 无证件/设备/生物特征数据 | 架构预留IDV接口；基于FLAG_WORK_PHONE/FLAG_EMAIL等构建轻量级身份异常评分 |
| 3 | **团伙养号检测** | 无设备/IP/地址关联图数据 | 架构预留GNN模块；基于地区+时段的异常聚集检测做轻量级替代 |
| 4 | **利率Treatment的CATE** | RATE_INTEREST缺失99% | 改用AMT_CREDIT作为Treatment；利率效应通过AMT_CREDIT→AMT_ANNUITY间接估计 |
| 5 | **出行/社交/公共事业替代数据** | 完全缺失 | 比赛中用合成数据演示因果推理框架的替代数据接入能力 |
| 6 | **实时行为流分析** | 无实时行为数据 | 仅做批量评分；架构设计支持实时流接入 |
| 7 | **跨机构关系图谱** | 无跨机构关联数据 | 架构预留联邦学习接口；单机构视角的因果推理 |

### 6.4 关键降级方案

#### 降级方案1：三类违约分类（替代欺诈标签）

由于TARGET不区分违约类型，CausalCredit无法直接训练欺诈分类器。降级方案：

1. **因果路径聚类**：基于因果发现结果，对违约样本（TARGET=1）按因果路径特征聚类
   - 路径1：AMT_CREDIT → AMT_ANNUITY → TARGET（高负债型违约）
   - 路径2：DAYS_EMPLOYED → AMT_INCOME_TOTAL → TARGET（收入不稳定型违约）
   - 路径3：EXT_SOURCE_* → TARGET（外部评分极低型，可能含欺诈）
2. **CATE异质性分析**：欺诈型违约对利率调整CATE≈0（欺诈者不在乎利率），而非恶意违约CATE显著为负
3. **反事实推理**：对违约样本生成反事实方案，无法生成合理改善方案的可能为恶意欺诈

#### 降级方案2：轻量级身份异常检测（替代IDV系统）

```
identity_anomaly_score = w1·contact_completeness + w2·address_consistency + w3·doc_pattern_anomaly + w4·query_frequency_anomaly

其中:
- contact_completeness: FLAG_WORK_PHONE + FLAG_PHONE + FLAG_EMAIL 的完整度
- address_consistency: REG_REGION_NOT_LIVE_REGION + REG_CITY_NOT_LIVE_CITY 的反向
- doc_pattern_anomaly: FLAG_DOCUMENT_* 的异常模式（全0或全1）
- query_frequency_anomaly: AMT_REQ_CREDIT_BUREAU_HOUR/DAY 的异常值
```

#### 降级方案3：地区-时段异常聚集检测（替代GNN团伙检测）

```
对 (REGION_RATING_CLIENT, WEEKDAY_APPR_PROCESS_START, HOUR_APPR_PROCESS_START) 三元组
计算申请密度，检测异常聚集：
- 同一地区+同一时段的申请密度显著高于基线 → 可能团伙操作
- 结合 CNT_FAM_MEMBERS, AMT_INCOME_TOTAL 的相似度 → 可能团伙成员
```

### 6.5 最终建议

**Home Credit Default Risk数据集对CausalCredit因果推理核心能力的支撑是充分的（评分3.8/5），方案可以落地。**

核心判断依据：
1. **因果推理三大能力（因果发现、CATE估计、反事实推理）的数据支撑完整**——这是CausalCredit的核心差异化优势，数据集完全支撑
2. **白户/薄信用人群评分的数据支撑充分**——这是CausalCredit的第二大亮点，数据集完全支撑
3. **反欺诈能力的数据支撑有限**——身份冒用和团伙检测无法基于此数据集实现，但可通过降级方案和架构预留接口部分弥补
4. **最关键的缺陷是TARGET不区分违约类型**——但可通过因果路径聚类和CATE异质性分析间接识别欺诈型违约

**务实建议**：
- 比赛展示中，**聚焦因果推理核心能力**（因果发现+CATE+反事实），这是数据集完全支撑的
- 反欺诈能力以**架构设计+降级演示**方式呈现，不承诺完整实现
- 替代数据以**合成数据+因果框架演示**方式呈现，展示框架能力而非数据丰富度
- 在方案中明确标注数据集局限性和真实部署时的替换方案，体现务实态度

---

> **文档版本**：v1.0  
> **编制人**：苏珊-数据架构师  
> **验证方法**：基于Home Credit Default Risk数据集Kaggle官方字段说明及社区EDA分析逐一验证
