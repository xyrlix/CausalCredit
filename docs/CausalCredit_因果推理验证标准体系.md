# CausalCredit 因果推理验证标准体系

> **版本**: v1.0 | **日期**: 2026-06-05 | **架构师**: 大卫-解决方案架构师  
> **难度评级**: ★★★★☆ (4/5) | **适用范围**: CausalCredit 因果推理全链路验证  
> **核心命题**: 因果推理不像传统ML有AUC/准确率这样直观的评估指标——我们需要一套严谨的验证标准来证明因果推理结论的可靠性

---

## 目录

1. [验证体系总览](#1-验证体系总览)
2. [因果图验证](#2-因果图验证)
3. [因果效应估计验证](#3-因果效应估计验证)
4. [反事实推理验证](#4-反事实推理验证)
5. [端到端验证](#5-端到端验证)
6. [量化指标体系总表](#6-量化指标体系总表)
7. [验证Pipeline流程](#7-验证pipeline流程)
8. [学术依据与参考文献](#8-学术依据与参考文献)

---

## 1. 验证体系总览

### 1.1 核心挑战

因果推理的验证面临根本性困难：**因果效应的"真值"（Ground Truth）在观测数据中不可直接观测**。传统ML可以通过留出集的标签计算AUC/F1，但因果效应 P(Y|do(X)) 无法从观测数据 P(Y|X) 中直接验证——这正是因果推理存在的意义，也是其验证的困境。

### 1.2 验证哲学

本体系采用**多层递进验证**（Layered Progressive Validation）哲学：

```
┌─────────────────────────────────────────────────────────┐
│                  因果推理验证金字塔                        │
│                                                          │
│                     ┌───────┐                            │
│                     │ L4    │  端到端验证                  │
│                     │业务效果│  A/B测试 / 业务指标对比       │
│                 ┌───┴───────┴───┐                        │
│                 │ L3            │  反事实推理验证            │
│                 │决策合理性      │  因果约束 / 可行性 / 对比  │
│             ┌───┴───────────────┴───┐                    │
│             │ L2                    │  因果效应估计验证      │
│             │效应准确性              │  反驳 / 伪实验 / 敏感性│
│         ┌───┴───────────────────────┴───┐                │
│         │ L1                            │  因果图验证      │
│         │结构正确性                      │  稳定性 / 一致性  │
│         └───────────────────────────────┘                │
│                                                          │
│  原则：下层不通过 → 上层不可验证 → 整体不可信              │
└─────────────────────────────────────────────────────────┘
```

**核心原则**：
- **逐层门控**：L1不通过则L2-L4无意义，L2不通过则L3-L4无意义
- **三角验证**：任何因果结论必须至少从两个独立维度获得支持
- **保守判定**：存疑时偏向"未验证"而非"通过"，避免过度自信

### 1.3 验证维度映射

| 验证层 | 核心问题 | 验证目标 | 关键方法 |
|--------|----------|----------|----------|
| L1 因果图 | 因果结构是否正确？ | 边的存在性与方向性 | 多种子稳定性 + 领域一致性 + 独立性检验 |
| L2 因果效应 | ATE/CATE估计是否准确？ | 效应量级与方向 | 反驳测试 + 伪实验 + 敏感性分析 |
| L3 反事实 | 反事实建议是否合理？ | 建议的因果一致性与可行性 | 约束违反率 + 分布可行性 + 已知效果对比 |
| L4 端到端 | 因果增强是否真正提升业务？ | 业务指标改善 | 对比实验 + A/B测试 + 校准度 |

---

## 2. 因果图验证

### 2.1 验证目标

验证混合因果发现引擎（PC + NOTEARS + 领域知识注入）输出的因果图 G = (V, E) 的结构正确性，即：
- **边存在性**：G中的边是否对应真实的因果关系？
- **边方向性**：因果方向是否正确？
- **边缺失性**：是否遗漏了重要的因果边？

### 2.2 验证方法1：多种子交集稳定性验证

#### 2.2.1 原理

因果发现算法对初始化、超参数、数据子采样敏感。如果因果结构是数据中真实存在的，那么在多种随机条件下应能稳定复现。PC算法的随机性来自条件独立性检验的顺序，NOTEARS的随机性来自初始化权重。

#### 2.2.2 方法

```
输入: 数据集 D, 种子数 K=10
对于 k = 1, 2, ..., K:
    1. 以种子 k 运行 PC 算法 → G_PC^(k)
    2. 以种子 k 运行 NOTEARS → G_NT^(k)
    3. 融合: G^(k) = G_PC^(k) ∩ G_NT^(k)  (交集法)

计算:
    边稳定性分数 S(e) = (1/K) * Σ_{k=1}^{K} I(e ∈ G^(k))
    其中 I(·) 为指示函数

输出: 稳定因果图 G_stable = {e | S(e) ≥ τ_stability}
```

#### 2.2.3 量化指标

| 指标 | 定义 | 通过标准 | 告警标准 | 计算方式 |
|------|------|----------|----------|----------|
| **边稳定性分数 S(e)** | 边e在K次运行中出现的频率 | S(e) ≥ 0.7 | 0.4 ≤ S(e) < 0.7 | K次运行中边e出现的比例 |
| **图稳定性指数 GSI** | 所有稳定边的平均稳定性 | GSI ≥ 0.80 | 0.60 ≤ GSI < 0.80 | mean(S(e) for e ∈ G_stable) |
| **核心边覆盖率** | 领域知识必选边在稳定图中的比例 | 100% | < 100% | |G_stable ∩ E_domain| / |E_domain| |
| **边一致性系数 ECC** | PC与NOTEARS的边一致率 | ECC ≥ 0.60 | 0.40 ≤ ECC < 0.60 | |E_PC ∩ E_NT| / |E_PC ∪ E_NT| |

#### 2.2.4 操作规范

- **种子数 K**：至少10次，关键场景建议20次
- **数据子采样**：每次运行随机抽取80%数据（Bootstrap子采样），增加扰动多样性
- **超参数扰动**：PC的α在[0.005, 0.01, 0.02]间扰动，NOTEARS的λ₁在[0.05, 0.1, 0.2]间扰动
- **交集策略**：采用"软交集"——边置信度加权，而非硬交集

### 2.3 验证方法2：领域知识一致性验证

#### 2.3.1 原理

金融领域存在大量已被学术文献和业务实践确认的因果关系。因果发现结果应与这些已知因果关系一致——不一致意味着发现可能存在偏差。

#### 2.3.2 金融领域已知因果知识库

构建三类领域约束：

**A. 禁止边（Must-Not-Exist）**：违反金融逻辑的因果方向

| 禁止边 | 理由 | 学术依据 |
|--------|------|----------|
| TARGET → AMT_CREDIT | 违约结果不能因果决定贷款金额（时间先后） | 因果的时间不对称性原则 |
| TARGET → DAYS_BIRTH | 违约不能改变年龄 | 物理因果律 |
| TARGET → CODE_GENDER | 违约不能改变性别 | 物理因果律 |
| AMT_ANNUITY → AMT_CREDIT | 年还款额不决定贷款金额（方向相反） | 贷款合同逻辑 |

**B. 必选边（Must-Exist）**：金融逻辑确定的因果关系

| 必选边 | 理由 | 学术依据 |
|--------|------|----------|
| AMT_CREDIT → AMT_ANNUITY | 贷款金额决定年还款额 | 贷款合同数学关系 |
| DAYS_BIRTH → DAYS_EMPLOYED | 年龄约束在职时长上限 | 生命周期约束 |
| AMT_INCOME_TOTAL → AMT_CREDIT | 收入影响可贷额度 | 信贷审批逻辑 |
| AMT_CREDIT → TARGET | 贷款金额影响违约概率 | 过度负债理论 |

**C. 方向约束（Direction-Only）**：边必须存在但方向需验证

| 约束 | 允许方向 | 禁止方向 |
|------|----------|----------|
| EXT_SOURCE_i ↔ TARGET | EXT_SOURCE_i → TARGET | TARGET → EXT_SOURCE_i |
| NAME_EDUCATION_TYPE ↔ AMT_INCOME_TOTAL | EDUCATION → INCOME | INCOME → EDUCATION |

#### 2.3.3 量化指标

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **禁止边违反率** | 发现图中出现禁止边的比例 | 0% | > 0% |
| **必选边覆盖率** | 必选边在发现图中出现的比例 | ≥ 90% | 70%-90% |
| **方向一致率** | 方向约束中方向正确的比例 | ≥ 85% | 70%-85% |
| **领域一致性总分 DKCS** | 加权综合得分 | ≥ 0.90 | 0.75-0.90 |

DKCS 计算公式：

```
DKCS = w₁·(1 - 禁止边违反率) + w₂·必选边覆盖率 + w₃·方向一致率
其中: w₁ = 0.4 (禁止边违反为致命错误, 权重最高)
      w₂ = 0.35
      w₃ = 0.25
```

### 2.4 验证方法3：条件独立性检验验证

#### 2.4.1 原理

因果图的每条边 X → Y 意味着在控制了X的所有父节点后，X与Y仍然条件依赖。如果条件独立性检验不拒绝原假设，则该边可能为虚假边。

#### 2.4.2 方法

对因果图中每条边 e: X → Y：

1. 识别X的父节点集 Pa(X)（根据因果图结构）
2. 执行条件独立性检验：X ⊥ Y | Pa(X)\{X}
3. 记录p值和检验统计量

#### 2.4.3 量化指标

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **边p值中位数** | 所有边的条件独立性p值中位数 | ≤ 0.01 | 0.01-0.05 |
| **弱边比例** | p值 > 0.05的边占比 | ≤ 5% | 5%-15% |
| **最小条件依赖强度** | 最弱的通过检验的偏相关系数绝对值 | ≥ 0.05 | 0.02-0.05 |
| **条件独立性检验一致性** | Fisher-Z与KCI检验结论一致率 | ≥ 90% | 80%-90% |

#### 2.4.4 检验方法选择

| 数据特征 | 推荐检验 | 理由 |
|----------|----------|------|
| 连续变量、线性关系 | Fisher-Z检验 | 计算高效，基于偏相关 |
| 混合变量（连续+离散） | KCI检验（Kernel Conditional Independence） | 非参数，无需分布假设 |
| 离散变量 | G²检验 | 适用于分类变量的条件独立性 |
| 大样本（N > 100K） | Fisher-Z + 校正 | 大样本下Fisher-Z渐近性良好 |

### 2.5 验证方法4：因果图与已知金融因果关系的吻合度

#### 2.5.1 原理

将发现的因果图与文献中已建立的金融因果模型进行结构对比，量化吻合程度。

#### 2.5.2 参考基准

| 基准来源 | 因果模型 | 适用场景 |
|----------|----------|----------|
| Pearl & Mackenzie (2018) | do-演算因果框架 | 因果效应识别 |
| Hoerl et al. (1962) | 岭回归因果图 | 多重共线性下的因果结构 |
| Athey & Imbens (2016) | 因果森林因果假设 | 异质处理效应 |
| 金融审慎监管框架 | 巴塞尔协议风险传导模型 | 系统性风险因果链 |

#### 2.5.3 量化指标

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **结构汉明距离 SHD** | 与参考图的最小编辑距离 | ≤ 参考图边数的20% | 20%-35% |
| **祖先-后代一致率** | 关键变量对的祖先/后代关系与参考一致的比例 | ≥ 80% | 65%-80% |
| **d-分离一致率** | 因果图蕴含的条件独立性关系与参考一致的比例 | ≥ 75% | 60%-75% |

### 2.6 因果图验证综合判定

```
因果图验证综合得分 CGVS:

CGVS = 0.30·GSI + 0.30·DKCS + 0.20·(1 - 弱边比例) + 0.20·(1 - SHD/|E_ref|)

判定规则:
    CGVS ≥ 0.80  →  PASS (因果图可信，进入L2验证)
    0.60 ≤ CGVS < 0.80  →  WARNING (需领域专家审核后决定)
    CGVS < 0.60  →  FAIL (因果图不可信，需重新发现)
```

---

## 3. 因果效应估计验证

### 3.1 验证目标

验证因果效应估计器（DML / DR / Causal Forest）输出的 ATE 和 CATE 的准确性与鲁棒性。核心挑战：**ATE/CATE的真值在观测数据中不可直接观测**，因此需要通过间接方法验证。

### 3.2 验证方法1：反驳测试（Refutation Tests）

反驳测试是DoWhy框架内置的因果效应鲁棒性验证方法，通过故意破坏因果假设来检验估计的稳健性。

#### 3.2.1 Placebo Treatment Refuter（安慰剂处理反驳）

**原理**：将处理变量替换为随机生成的变量，真实因果效应应为零。如果估计出显著效应，说明模型存在混淆偏差。

**方法**：
```
1. 将处理变量 T 替换为随机变量 T_random ~ N(0,1) 或 T_random ~ Bernoulli(0.5)
2. 使用相同的因果效应估计方法重新估计 ATE
3. 检验: ATE_placebo 是否显著偏离零
```

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **安慰剂ATE绝对值** | \|ATE_placebo\| | ≤ 0.01 | 0.01-0.05 |
| **安慰剂ATE p值** | H₀: ATE_placebo = 0 的p值 | ≥ 0.20 | 0.05-0.20 |
| **安慰剂/真实ATE比** | \|ATE_placebo\| / \|ATE_real\| | ≤ 0.10 | 0.10-0.25 |

#### 3.2.2 Random Cause Refuter（随机原因反驳）

**原理**：向数据中添加一个与结果变量无关的随机特征作为额外混淆因子，因果效应估计不应显著改变。

**方法**：
```
1. 生成随机变量 W_random ~ N(0,1)
2. 将 W_random 加入混淆因子集
3. 重新估计 ATE
4. 计算 ATE 变化率
```

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **ATE变化率** | \|ATE_new - ATE_orig\| / \|ATE_orig\| | ≤ 5% | 5%-15% |
| **CATE排序相关性** | 加入随机原因前后CATE排序的Spearman ρ | ≥ 0.90 | 0.80-0.90 |
| **置信区间重叠率** | 新旧95% CI的重叠比例 | ≥ 80% | 60%-80% |

#### 3.2.3 Data Subset Refuter（数据子集反驳）

**原理**：在不同数据子集上估计因果效应，真实效应应具有稳定性。大幅波动暗示估计对特定数据子集过拟合。

**方法**：
```
1. 随机抽取 K=10 个子集（每个含80%数据）
2. 在每个子集上估计 ATE
3. 计算ATE的跨子集变异系数
```

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **ATE变异系数 CV** | std(ATE_subsets) / mean(ATE_subsets) | ≤ 0.15 | 0.15-0.30 |
| **ATE跨子集极差比** | (max - min) / \|mean\| | ≤ 0.40 | 0.40-0.70 |
| **子集ATE符号一致率** | ATE方向一致的子集比例 | 100% | 80%-100% |

### 3.3 验证方法2：伪真实实验（Quasi-Experimental Validation）

#### 3.3.1 Natural Experiment（自然实验验证）

**原理**：利用外生冲击（政策变化、自然灾害等）作为"准实验"，验证因果效应估计的方向和量级是否与自然实验结果一致。

**适用场景与设计**：

| 自然实验 | 处理变量 | 预期效应 | 验证逻辑 |
|----------|----------|----------|----------|
| 2020 COVID收入冲击 | 收入下降（外生） | 违约率上升 | 因果估计的"收入→违约"效应方向应为正 |
| 利率政策调整 | 贷款利率变化 | 还款压力变化 | 因果估计的"利率→违约"效应方向应与政策效果一致 |
| 区域经济衰退 | 区域GDP下降 | 区域违约率上升 | 因果估计的"区域经济→违约"效应应可复现 |

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **方向一致率** | 因果估计方向与自然实验一致的比例 | 100% | 80%-100% |
| **量级吻合度** | \|ATE_estimated\| / \|ATE_natural\| | 0.5-2.0 | 0.3-0.5 或 2.0-3.0 |
| **统计显著性一致** | 因果估计与自然实验同时显著的比例 | ≥ 80% | 60%-80% |

#### 3.3.2 Instrumental Variable Validation（工具变量验证）

**原理**：使用工具变量（IV）估计的因果效应作为"准真值"基准，验证非IV方法的估计是否一致。

**候选工具变量**：

| 工具变量 | 满足的相关性条件 | 满足的外生性条件 | 文献依据 |
|----------|------------------|------------------|----------|
| 区域平均收入 | 与个体收入强相关 | 不直接影响个体违约 | Angrist & Pischke (2009) |
| 政策利率变动 | 与贷款利率强相关 | 不直接影响个体违约 | 宏观经济学标准IV |
| 同行业就业率 | 与个体就业状态相关 | 不直接影响个体违约 | 劳动经济学IV |

**两阶段最小二乘（2SLS）验证流程**：

```
Step 1: 第一阶段回归 T = α + β·Z + γ·X + ε₁
        验证: F-statistic ≥ 10 (Staiger & Stock, 1997 强工具变量标准)

Step 2: 第二阶段回归 Y = δ + θ·T̂ + η·X + ε₂
        ATE_IV = θ

Step 3: 对比 ATE_IV 与 ATE_DML / ATE_DR
        计算: |ATE_IV - ATE_DML| / |ATE_IV| = 相对偏差
```

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **IV第一阶段F统计量** | 工具变量相关性强度 | ≥ 10 | 5-10 (弱IV) |
| **IV vs DML相对偏差** | \|ATE_IV - ATE_DML\| / \|ATE_IV\| | ≤ 30% | 30%-50% |
| **IV vs DR相对偏差** | \|ATE_IV - ATE_DR\| / \|ATE_IV\| | ≤ 25% | 25%-45% |
| **Sargan过度识别检验p值** | IV外生性检验 | ≥ 0.10 | 0.05-0.10 |

### 3.4 验证方法3：敏感性分析（Sensitivity Analysis）

#### 3.4.1 Robustness Score（鲁棒性分数）

**原理**：衡量因果效应估计对未观测混淆因子的敏感程度。如果很小的未观测混淆就能使效应消失，则结论不可靠。

**方法**：基于Rosenbaum框架（2002），逐步增加未观测混淆因子的强度，观察ATE何时变为不显著。

```
对于混淆强度参数 Γ = 1.0, 1.1, 1.2, ..., 2.0:
    计算在 Γ 下 ATE 的上下界
    记录 ATE 变为不显著时的 Γ_critical

Robustness Score = Γ_critical
```

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **鲁棒性分数 Γ_critical** | ATE变为不显著的最小混淆强度 | ≥ 1.5 | 1.2-1.5 |
| **效应衰减率** | Γ从1.0到2.0时ATE的衰减比例 | ≤ 50% | 50%-75% |
| **符号稳定性阈值** | ATE符号翻转的最小混淆强度 | ≥ 1.8 | 1.3-1.8 |

#### 3.4.2 E-value（E值）

**原理**：VanderWeele & Ding (2017) 提出的E-value，量化未观测混淆因子需要多强才能完全解释观察到的效应。

**计算**：

```
对于观察到的风险比 RR:
    E-value = RR + √(RR × (RR - 1))

对于连续结果的近似:
    将ATE转换为近似风险比后计算E-value
```

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **E-value** | 未观测混淆解释效应所需的最小强度 | ≥ 2.0 | 1.5-2.0 |
| **E-value / 已知最强混淆因子效应比** | E-value相对于已知最强混淆的裕度 | ≥ 1.5 | 1.2-1.5 |

**E-value解读**：E-value = 2.0 意味着未观测混淆因子需要与结果和处理的关联都达到2.0倍风险比，才能完全解释观察到的效应——这在金融场景中是相当强的混淆，不太可能被遗漏。

#### 3.4.3 CATE跨验证一致性

**原理**：CATE的异质性模式应在不同估计方法间保持一致，否则异质性结论不可靠。

**方法**：
```
1. 使用 DML 估计 CATE_DML(x)
2. 使用 DR 估计 CATE_DR(x)
3. 使用 Causal Forest 估计 CATE_CF(x)
4. 计算三者之间的两两一致性
```

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **CATE方法间Spearman ρ** | 不同方法CATE排序的相关性 | ≥ 0.75 | 0.60-0.75 |
| **CATE方法间Pearson r** | 不同方法CATE量值的相关性 | ≥ 0.70 | 0.55-0.70 |
| **CATE符号一致率** | 不同方法CATE方向一致的比例 | ≥ 90% | 80%-90% |
| **CATE分位数组效应单调性** | CATE在分位数组间是否单调 | 单调或近似单调 | 存在反转 |

### 3.5 因果效应估计验证综合判定

```
因果效应验证综合得分 CEVS:

CEVS = 0.25·反驳测试得分 + 0.25·伪实验得分 + 0.25·敏感性得分 + 0.25·CATE一致性得分

各子得分计算:
    反驳测试得分 = 0.4·(1 - 安慰剂ATE比) + 0.3·(1 - ATE变化率) + 0.3·(1 - ATE_CV)
    伪实验得分 = 0.5·方向一致率 + 0.3·量级吻合度 + 0.2·IV相对偏差归一化
    敏感性得分 = 0.4·min(Γ_critical/2.0, 1) + 0.3·min(E-value/3.0, 1) + 0.3·(1 - 效应衰减率)
    CATE一致性得分 = 0.4·Spearman_ρ + 0.3·符号一致率 + 0.3·单调性得分

判定规则:
    CEVS ≥ 0.75  →  PASS (因果效应可信，进入L3验证)
    0.55 ≤ CEVS < 0.75  →  WARNING (需标注不确定性范围)
    CEVS < 0.55  →  FAIL (因果效应不可信，需重新估计)
```

---

## 4. 反事实推理验证

### 4.1 验证目标

验证因果约束反事实推理（DiCE + 因果图约束 + NSGA-II）生成的反事实方案的合理性与可行性。核心挑战：**反事实场景从未真实发生，无法直接验证**。

### 4.2 验证方法1：因果约束违反率

#### 4.2.1 原理

反事实方案必须满足因果图蕴含的约束关系。违反因果约束意味着方案在因果逻辑上不可行。

#### 4.2.2 因果约束类型

| 约束类型 | 定义 | 示例 | 违反检测 |
|----------|------|------|----------|
| **确定性因果约束** | 变量间存在函数关系 | AMT_ANNUITY = f(AMT_CREDIT, RATE) | 检查函数关系是否满足 |
| **方向约束** | 因果方向不可逆 | 改变AMT_ANNUITY不应独立于AMT_CREDIT | 检查下游变量是否随上游变化 |
| **不可变约束** | 特征不可改变 | DAYS_BIRTH, CODE_GENDER | 检查不可变特征是否被修改 |
| **半可变约束** | 特征可变但有合理范围 | AMT_INCOME_TOTAL变化幅度 ≤ 50% | 检查变化是否在合理范围内 |
| **中介传播约束** | 中介变量应随处理变量变化 | AMT_CREDIT↓ → AMT_ANNUITY↓ → DEBT_RATIO↓ | 检查中介路径是否完整传播 |

#### 4.2.3 量化指标

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **确定性约束违反率** | 违反确定性因果关系的方案比例 | 0% | > 0% 且 ≤ 5% |
| **方向约束违反率** | 违反因果方向约束的方案比例 | ≤ 2% | 2%-8% |
| **不可变约束违反率** | 修改不可变特征的方案比例 | 0% | > 0% |
| **中介传播完整率** | 中介路径完整传播的方案比例 | ≥ 90% | 75%-90% |
| **综合因果违反率 CCR** | 加权违反率 | ≤ 2% | 2%-8% |

CCR 计算公式：

```
CCR = 0.35·确定性违反率 + 0.25·方向违反率 + 0.20·不可变违反率 + 0.20·(1 - 中介传播完整率)
```

### 4.3 验证方法2：反事实方案在训练分布内的可行性

#### 4.3.1 原理

反事实方案的特征值应在训练数据的合理范围内，否则模型外推（Extrapolation）可能导致不可靠的预测。

#### 4.3.2 方法

```
对于每个反事实方案 x_cf:
    1. 计算马氏距离: D_M(x_cf) = √((x_cf - μ)ᵀ Σ⁻¹ (x_cf - μ))
       其中 μ, Σ 为训练数据的均值和协方差
    2. 计算特征值分位数: 对每个特征 f_i, 计算 P(X_i ≤ x_cf_i) 在训练分布中
    3. 计算局部密度比: p(x_cf) / p(x_original), 使用KDE估计
```

**量化指标**：

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **马氏距离中位数** | 所有反事实方案的马氏距离中位数 | ≤ 3.0 (χ²分布95%分位) | 3.0-5.0 |
| **极端特征比例** | 特征值超出训练数据[1%, 99%]分位数的比例 | ≤ 5% | 5%-15% |
| **密度比中位数** | p(x_cf) / p(x_orig) 的中位数 | ≥ 0.10 | 0.02-0.10 |
| **分布外检测率** | 被Isolation Forest标记为异常的方案比例 | ≤ 10% | 10%-25% |

### 4.4 验证方法3：与已知干预效果的对比

#### 4.4.1 原理

某些干预的因果效应在文献或业务实践中已有定量估计，反事实推理的预测应与这些已知效果一致。

#### 4.4.2 已知干预效果基准

| 干预 | 已知效果 | 文献来源 | 验证逻辑 |
|------|----------|----------|----------|
| 贷款金额降低10% | 违约率降低1.5-3% | Mian & Sufi (2014) | 反事实预测的效应应在1.5-3%范围内 |
| 收入增加20% | 违约率降低3-6% | 银行内部A/B测试数据 | 反事实预测的效应应在3-6%范围内 |
| 贷款期限延长12个月 | 违约率降低1-2% | 信贷实务经验 | 反事实预测的效应方向应为负 |
| 增加共同借款人 | 违约率降低4-8% | 银行风控经验 | 反事实预测的效应应在4-8%范围内 |

#### 4.4.3 量化指标

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **已知效果方向一致率** | 反事实预测方向与已知效果一致的比例 | 100% | 80%-100% |
| **量级吻合度** | 反事实预测量值落在已知效果范围内的比例 | ≥ 70% | 50%-70% |
| **相对误差中位数** | \|预测效果 - 已知效果\| / 已知效果 的中位数 | ≤ 40% | 40%-70% |

### 4.5 反事实推理验证综合判定

```
反事实验证综合得分 CFVS:

CFVS = 0.35·(1 - CCR) + 0.35·可行性得分 + 0.30·已知效果对比得分

可行性得分 = 0.3·min(1, 3.0/马氏距离中位数) + 0.3·(1 - 极端特征比例) + 0.2·min(1, 密度比/0.10) + 0.2·(1 - 分布外检测率)
已知效果对比得分 = 0.4·方向一致率 + 0.35·量级吻合度 + 0.25·(1 - 相对误差中位数)

判定规则:
    CFVS ≥ 0.75  →  PASS (反事实方案可信)
    0.55 ≤ CFVS < 0.75  →  WARNING (方案需人工审核)
    CFVS < 0.55  →  FAIL (反事实方案不可信，需重新生成)
```

---

## 5. 端到端验证

### 5.1 验证目标

验证因果增强评分系统整体是否优于纯ML评分系统——这是最终的业务价值验证。

### 5.2 验证方法1：因果增强评分 vs 纯ML评分对比实验

#### 5.2.1 实验设计

```
┌──────────────────────────────────────────────────────────────┐
│                  对比实验设计                                    │
│                                                                │
│  实验组A (Baseline): 纯ML评分                                  │
│    score_A = w₁·LightGBM_pred + w₂·XGBoost_pred              │
│                                                                │
│  实验组B (Causal-Enhanced): 因果增强评分                       │
│    score_B = w₁·ML_pred + w₂·causal_adjusted_score           │
│             + w₃·rule_score                                   │
│                                                                │
│  实验组C (Causal-Only): 纯因果评分                             │
│    score_C = f(causal_features_only)                          │
│                                                                │
│  对照维度:                                                      │
│    1. 预测性能: AUC, KS, Gini                                  │
│    2. 排序能力: NDCG@K, 累积增益曲线                            │
│    3. 校准度: Brier Score, ECE, Calibration Curve              │
│    4. 公平性: Demographic Parity, Equalized Odds               │
│    5. 稳定性: 跨时间窗口AUC方差, PSI                            │
│    6. 干预指导价值: CATE准确率, 反事实方案采纳率                 │
└──────────────────────────────────────────────────────────────┘
```

#### 5.2.2 量化指标

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **AUC提升** | AUC_B - AUC_A | ≥ +0.01 | 0 ~ +0.01 |
| **KS值提升** | KS_B - KS_A | ≥ +0.02 | 0 ~ +0.02 |
| **Brier Score降低** | BS_A - BS_B | ≥ 0.005 | 0 ~ 0.005 |
| **ECE降低** | ECE_A - ECE_B | ≥ 0.01 | 0 ~ 0.01 |
| **公平性改善** | Demographic Parity差异降低 | ≥ 20% | 0%-20% |
| **跨时间AUC方差降低** | Var(AUC_A) - Var(AUC_B) | > 0 | ≤ 0 |

### 5.3 验证方法2：A/B测试框架

#### 5.3.1 实验设计

```
A/B测试框架:

┌─────────────────────────────────────────────────────────────┐
│  流量分配 (100%)                                              │
│                                                               │
│  ├── 50% → Control Group: 纯ML评分决策                       │
│  │     └── 按传统评分模型输出审批/拒绝/定价                    │
│  │                                                           │
│  └── 50% → Treatment Group: 因果增强评分决策                  │
│        └── 按因果增强评分 + CATE干预建议输出决策               │
│                                                               │
│  最小样本量计算:                                               │
│    违约率基线 p₀ = 8%                                         │
│    期望改善 δ = 1.5% (相对改善18.75%)                         │
│    显著性水平 α = 0.05                                        │
│    统计功效 1-β = 0.80                                        │
│    → 每组最少 N = 17,200 (双比例检验)                         │
│                                                               │
│  实验周期: 8-12周 (覆盖完整还款周期)                           │
│  多重比较校正: Bonferroni (4个主要指标 → α_adj = 0.0125)      │
└─────────────────────────────────────────────────────────────┘
```

#### 5.3.2 主要观测指标

| 指标 | 统计检验 | 通过标准 | 告警标准 |
|------|----------|----------|----------|
| **违约率差异** | 双比例Z检验 (α=0.0125) | Treatment违约率显著更低 | 方向正确但不显著 |
| **审批通过率差异** | 双比例Z检验 | 通过率不降低或降低<2% | 降低2%-5% |
| **平均贷款利率差异** | 双样本t检验 | 利率更精准（方差更小） | 方向正确但不显著 |
| **客户投诉率差异** | 双比例Z检验 | 投诉率显著更低 | 方向正确但不显著 |

#### 5.3.3 护栏机制（Guardrail Metrics）

A/B测试期间必须监控的护栏指标——一旦触发立即停止实验：

| 护栏指标 | 阈值 | 行动 |
|----------|------|------|
| Treatment组违约率 > Control组 + 2% | 立即停止 | 回滚至纯ML评分 |
| 特定群体违约率差异 > 5% | 暂停+调查 | 公平性审查 |
| 评分延迟P99 > 200ms | 降级 | 切换至纯ML评分路径 |
| 客户投诉率 > Control组 × 1.5 | 暂停+调查 | 用户体验审查 |

### 5.4 验证方法3：评分校准度（Calibration）

#### 5.4.1 原理

因果增强评分的预测概率应与实际违约率一致——如果模型预测某群体违约概率为10%，则该群体的实际违约率应接近10%。

#### 5.4.2 校准度评估方法

**A. 可靠性图（Reliability Diagram）**

```
1. 将预测概率分为 B=10 个等宽分箱
2. 每个分箱计算: 预测概率均值 vs 实际违约率
3. 绘制: x轴=预测概率, y轴=实际违约率
4. 完美校准: y = x 对角线
```

**B. Expected Calibration Error (ECE)**

```
ECE = Σ_{b=1}^{B} (n_b / N) × |acc(b) - conf(b)|

其中:
    n_b = 分箱b中的样本数
    N = 总样本数
    acc(b) = 分箱b中的实际违约率
    conf(b) = 分箱b中的平均预测概率
```

**C. Brier Score分解**

```
BS = 可靠性项 - 分辨率项 + 不确定性项

可靠性项越小越好 (校准越准)
分辨率项越大越好 (区分能力越强)
```

#### 5.4.3 量化指标

| 指标 | 定义 | 通过标准 | 告警标准 |
|------|------|----------|----------|
| **ECE** | 期望校准误差 | ≤ 0.03 | 0.03-0.06 |
| **MCE** | 最大校准误差 | ≤ 0.08 | 0.08-0.15 |
| **Brier Score** | 概率预测的均方误差 | ≤ 0.12 | 0.12-0.18 |
| **可靠性项** | Brier分解的可靠性分量 | ≤ 0.02 | 0.02-0.05 |
| **分辨率项** | Brier分解的分辨率分量 | ≥ 0.05 | 0.03-0.05 |
| **Hosmer-Lemeshow p值** | 校准度拟合优度检验 | ≥ 0.05 | 0.01-0.05 |

#### 5.4.4 因果增强校准

因果增强评分的校准需额外关注：

```
因果增强校准流程:
    1. 纯ML评分 → Platt Scaling / Isotonic Regression → 校准后ML评分
    2. 因果调整评分 → 独立校准 → 校准后因果评分
    3. 融合评分 → 融合后校准 → 最终评分

    关键: 因果调整可能破坏校准度, 必须在融合后重新校准
    验证: 因果增强评分的ECE应 ≤ 纯ML评分的ECE
```

### 5.5 端到端验证综合判定

```
端到端验证综合得分 E2EVS:

E2EVS = 0.30·预测性能得分 + 0.30·A/B测试得分 + 0.25·校准度得分 + 0.15·公平性得分

预测性能得分 = 0.4·min(1, AUC提升/0.03) + 0.3·min(1, KS提升/0.04) + 0.3·min(1, Brier降低/0.01)
A/B测试得分 = 0.4·违约率改善 + 0.3·审批率维持 + 0.3·投诉率改善 (归一化到[0,1])
校准度得分 = 0.3·min(1, 0.03/ECE) + 0.3·min(1, 0.08/MCE) + 0.2·min(1, 0.12/BS) + 0.2·min(1, HL_p/0.05)
公平性得分 = 0.5·(1 - DP差异) + 0.5·(1 - EO差异)

判定规则:
    E2EVS ≥ 0.70  →  PASS (因果增强评分系统整体可信)
    0.50 ≤ E2EVS < 0.70  →  WARNING (部分指标需优化)
    E2EVS < 0.50  →  FAIL (因果增强未带来显著改善)
```

---

## 6. 量化指标体系总表

### 6.1 全指标一览

| 层级 | 维度 | 指标 | 通过标准 | 告警标准 | 权重 |
|------|------|------|----------|----------|------|
| **L1** | 稳定性 | 边稳定性分数 S(e) | ≥ 0.7 | 0.4-0.7 | 0.10 |
| **L1** | 稳定性 | 图稳定性指数 GSI | ≥ 0.80 | 0.60-0.80 | 0.10 |
| **L1** | 稳定性 | 核心边覆盖率 | 100% | < 100% | 0.05 |
| **L1** | 稳定性 | 边一致性系数 ECC | ≥ 0.60 | 0.40-0.60 | 0.05 |
| **L1** | 领域一致性 | 禁止边违反率 | 0% | > 0% | 0.10 |
| **L1** | 领域一致性 | 必选边覆盖率 | ≥ 90% | 70%-90% | 0.08 |
| **L1** | 领域一致性 | 方向一致率 | ≥ 85% | 70%-85% | 0.07 |
| **L1** | 领域一致性 | 领域一致性总分 DKCS | ≥ 0.90 | 0.75-0.90 | 0.05 |
| **L1** | 独立性检验 | 边p值中位数 | ≤ 0.01 | 0.01-0.05 | 0.08 |
| **L1** | 独立性检验 | 弱边比例 | ≤ 5% | 5%-15% | 0.07 |
| **L1** | 结构吻合 | 结构汉明距离 SHD | ≤ 20% | 20%-35% | 0.10 |
| **L1** | 结构吻合 | d-分离一致率 | ≥ 75% | 60%-75% | 0.05 |
| **L1** | **综合** | **因果图验证综合得分 CGVS** | **≥ 0.80** | **0.60-0.80** | **1.00** |
| | | | | | |
| **L2** | 反驳-安慰剂 | 安慰剂ATE绝对值 | ≤ 0.01 | 0.01-0.05 | 0.05 |
| **L2** | 反驳-安慰剂 | 安慰剂/真实ATE比 | ≤ 0.10 | 0.10-0.25 | 0.05 |
| **L2** | 反驳-随机原因 | ATE变化率 | ≤ 5% | 5%-15% | 0.05 |
| **L2** | 反驳-随机原因 | CATE排序Spearman ρ | ≥ 0.90 | 0.80-0.90 | 0.04 |
| **L2** | 反驳-子集 | ATE变异系数 CV | ≤ 0.15 | 0.15-0.30 | 0.06 |
| **L2** | 反驳-子集 | ATE符号一致率 | 100% | 80%-100% | 0.05 |
| **L2** | 伪实验 | 方向一致率 | 100% | 80%-100% | 0.08 |
| **L2** | 伪实验 | 量级吻合度 | 0.5-2.0 | 0.3-0.5或2.0-3.0 | 0.06 |
| **L2** | 伪实验 | IV第一阶段F统计量 | ≥ 10 | 5-10 | 0.06 |
| **L2** | 伪实验 | IV vs DML相对偏差 | ≤ 30% | 30%-50% | 0.05 |
| **L2** | 敏感性 | 鲁棒性分数 Γ_critical | ≥ 1.5 | 1.2-1.5 | 0.08 |
| **L2** | 敏感性 | E-value | ≥ 2.0 | 1.5-2.0 | 0.08 |
| **L2** | 敏感性 | 效应衰减率 | ≤ 50% | 50%-75% | 0.06 |
| **L2** | CATE一致性 | CATE方法间Spearman ρ | ≥ 0.75 | 0.60-0.75 | 0.08 |
| **L2** | CATE一致性 | CATE符号一致率 | ≥ 90% | 80%-90% | 0.07 |
| **L2** | CATE一致性 | CATE分位数组效应单调性 | 单调 | 存在反转 | 0.06 |
| **L2** | **综合** | **因果效应验证综合得分 CEVS** | **≥ 0.75** | **0.55-0.75** | **1.00** |
| | | | | | |
| **L3** | 因果约束 | 确定性约束违反率 | 0% | > 0%且≤5% | 0.10 |
| **L3** | 因果约束 | 不可变约束违反率 | 0% | > 0% | 0.10 |
| **L3** | 因果约束 | 中介传播完整率 | ≥ 90% | 75%-90% | 0.08 |
| **L3** | 因果约束 | 综合因果违反率 CCR | ≤ 2% | 2%-8% | 0.07 |
| **L3** | 可行性 | 马氏距离中位数 | ≤ 3.0 | 3.0-5.0 | 0.08 |
| **L3** | 可行性 | 极端特征比例 | ≤ 5% | 5%-15% | 0.07 |
| **L3** | 可行性 | 密度比中位数 | ≥ 0.10 | 0.02-0.10 | 0.06 |
| **L3** | 已知效果 | 方向一致率 | 100% | 80%-100% | 0.12 |
| **L3** | 已知效果 | 量级吻合度 | ≥ 70% | 50%-70% | 0.10 |
| **L3** | 已知效果 | 相对误差中位数 | ≤ 40% | 40%-70% | 0.08 |
| **L3** | **综合** | **反事实验证综合得分 CFVS** | **≥ 0.75** | **0.55-0.75** | **1.00** |
| | | | | | |
| **L4** | 预测性能 | AUC提升 | ≥ +0.01 | 0~+0.01 | 0.10 |
| **L4** | 预测性能 | KS值提升 | ≥ +0.02 | 0~+0.02 | 0.08 |
| **L4** | 预测性能 | Brier Score降低 | ≥ 0.005 | 0~0.005 | 0.07 |
| **L4** | A/B测试 | 违约率差异显著 | p < 0.0125 | 方向正确不显著 | 0.12 |
| **L4** | A/B测试 | 审批通过率维持 | 降低<2% | 降低2%-5% | 0.08 |
| **L4** | 校准度 | ECE | ≤ 0.03 | 0.03-0.06 | 0.10 |
| **L4** | 校准度 | MCE | ≤ 0.08 | 0.08-0.15 | 0.07 |
| **L4** | 校准度 | Hosmer-Lemeshow p值 | ≥ 0.05 | 0.01-0.05 | 0.06 |
| **L4** | 公平性 | Demographic Parity差异 | ≤ 0.05 | 0.05-0.10 | 0.08 |
| **L4** | 公平性 | Equalized Odds差异 | ≤ 0.05 | 0.05-0.10 | 0.07 |
| **L4** | 稳定性 | 跨时间AUC方差降低 | > 0 | ≤ 0 | 0.07 |
| **L4** | **综合** | **端到端验证综合得分 E2EVS** | **≥ 0.70** | **0.50-0.70** | **1.00** |

### 6.2 全局综合判定

```
CausalCredit 因果推理验证全局得分 CCGS:

CCGS = 0.25·CGVS + 0.30·CEVS + 0.25·CFVS + 0.20·E2EVS

权重说明:
    - CEVS权重最高(0.30): 因果效应估计是核心产出, 可靠性最关键
    - CGVS和CFVS次之(0.25): 结构正确性和反事实合理性同等重要
    - E2EVS权重最低(0.20): 端到端效果受多种因素影响, 不完全反映因果推理质量

全局判定规则:
    CCGS ≥ 0.75  →  EXCELLENT (因果推理系统高度可信, 可全面部署)
    0.60 ≤ CCGS < 0.75  →  GOOD (因果推理系统基本可信, 需标注不确定性)
    0.45 ≤ CCGS < 0.60  →  MARGINAL (因果推理系统部分可信, 仅限辅助参考)
    CCGS < 0.45  →  INSUFFICIENT (因果推理系统不可信, 需重新设计)
```

---

## 7. 验证Pipeline流程

### 7.1 Pipeline总览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                CausalCredit 因果推理验证 Pipeline                              │
│                                                                                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌────────────┐ │
│  │  Stage 1    │────▶│  Stage 2    │────▶│  Stage 3    │────▶│  Stage 4   │ │
│  │  因果图验证  │     │  因果效应    │     │  反事实推理  │     │  端到端验证 │ │
│  │             │     │  估计验证    │     │  验证        │     │            │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └────────────┘ │
│        │                   │                   │                   │          │
│   CGVS判定            CEVS判定            CFVS判定            E2EVS判定     │
│   PASS→继续           PASS→继续           PASS→继续           PASS→部署     │
│   WARN→审核           WARN→标注           WARN→审核           WARN→优化     │
│   FAIL→回退           FAIL→重估           FAIL→重生           FAIL→重设计   │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │  Stage 5: 综合报告生成                                                    ││
│  │  - 各层级验证结果汇总                                                     ││
│  │  - 全局综合得分 CCGS 计算                                                 ││
│  │  - 不确定性范围标注                                                       ││
│  │  - 改进建议优先级排序                                                     ││
│  │  - Model Card 更新                                                        ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Stage 1: 因果图验证Pipeline

```python
# 伪代码 - 因果图验证Pipeline

def validate_causal_graph(data, domain_constraints):
    """
    Stage 1: 因果图验证
    输入: 数据集, 领域约束
    输出: CGVS得分, 验证报告, 稳定因果图
    """
    
    # === Step 1.1: 多种子稳定性验证 ===
    K = 10  # 种子数
    graphs_pc = []
    graphs_nt = []
    
    for seed in range(K):
        # Bootstrap子采样 (80%)
        data_sample = data.sample(frac=0.8, random_state=seed)
        
        # PC算法
        g_pc = run_pc(data_sample, alpha=0.01, seed=seed)
        graphs_pc.append(g_pc)
        
        # NOTEARS算法
        g_nt = run_notears(data_sample, lambda1=0.1, seed=seed)
        graphs_nt.append(g_nt)
    
    # 计算边稳定性分数
    edge_stability = compute_edge_stability(graphs_pc, graphs_nt, K)
    
    # 生成稳定因果图 (S(e) >= 0.7)
    stable_graph = filter_stable_edges(edge_stability, threshold=0.7)
    
    # 计算GSI, ECC
    gsi = compute_gsi(edge_stability, stable_graph)
    ecc = compute_ecc(graphs_pc, graphs_nt)
    
    # === Step 1.2: 领域知识一致性验证 ===
    forbidden_violation = check_forbidden_edges(stable_graph, domain_constraints.forbidden)
    required_coverage = check_required_edges(stable_graph, domain_constraints.required)
    direction_consistency = check_direction_constraints(stable_graph, domain_constraints.directions)
    dkcs = compute_dkcs(forbidden_violation, required_coverage, direction_consistency)
    
    # === Step 1.3: 条件独立性检验验证 ===
    p_values = []
    for edge in stable_graph.edges:
        pa_x = get_parents(stable_graph, edge.source)
        p_val = conditional_independence_test(
            data, edge.source, edge.target, 
            conditioning_set=pa_x,
            method='fisher_z'
        )
        p_values.append(p_val)
    
    weak_edge_ratio = sum(1 for p in p_values if p > 0.05) / len(p_values)
    median_p = median(p_values)
    
    # === Step 1.4: 结构吻合度验证 ===
    shd = structural_hamming_distance(stable_graph, reference_graph)
    d_sep_consistency = d_separation_consistency(stable_graph, reference_graph)
    
    # === Step 1.5: 综合判定 ===
    cgvs = compute_cgvs(gsi, dkcs, weak_edge_ratio, shd)
    
    verdict = "PASS" if cgvs >= 0.80 else ("WARNING" if cgvs >= 0.60 else "FAIL")
    
    return {
        "score": cgvs,
        "verdict": verdict,
        "stable_graph": stable_graph,
        "metrics": {
            "gsi": gsi, "ecc": ecc,
            "dkcs": dkcs, "forbidden_violation": forbidden_violation,
            "weak_edge_ratio": weak_edge_ratio, "median_p_value": median_p,
            "shd": shd, "d_sep_consistency": d_sep_consistency
        }
    }
```

### 7.3 Stage 2: 因果效应估计验证Pipeline

```python
def validate_causal_effects(data, causal_graph, treatment_vars):
    """
    Stage 2: 因果效应估计验证
    输入: 数据集, 因果图, 处理变量列表
    输出: CEVS得分, 验证报告, ATE/CATE估计结果
    """
    
    results = {}
    
    for T in treatment_vars:
        # === Step 2.1: 因果效应估计 ===
        ate_dml, cate_dml = estimate_dml(data, causal_graph, T)
        ate_dr, cate_dr = estimate_dr(data, causal_graph, T)
        ate_cf, cate_cf = estimate_causal_forest(data, causal_graph, T)
        
        # === Step 2.2: 反驳测试 ===
        # Placebo Treatment
        ate_placebo = refuter_placebo(data, causal_graph, T)
        placebo_ratio = abs(ate_placebo) / abs(ate_dml) if ate_dml != 0 else float('inf')
        
        # Random Cause
        ate_random = refuter_random_cause(data, causal_graph, T)
        ate_change_rate = abs(ate_random - ate_dml) / abs(ate_dml)
        
        # Data Subset
        ate_subsets = []
        for k in range(10):
            subset = data.sample(frac=0.8, random_state=k)
            ate_sub_k, _ = estimate_dml(subset, causal_graph, T)
            ate_subsets.append(ate_sub_k)
        ate_cv = std(ate_subsets) / mean(ate_subsets)
        sign_consistency = sum(1 for a in ate_subsets if sign(a) == sign(ate_dml)) / len(ate_subsets)
        
        # === Step 2.3: 伪实验验证 ===
        # IV验证
        iv = select_instrumental_variable(causal_graph, T)
        ate_iv = estimate_2sls(data, causal_graph, T, iv)
        iv_f_stat = first_stage_f_statistic(data, T, iv)
        iv_dml_bias = abs(ate_iv - ate_dml) / abs(ate_iv) if ate_iv != 0 else float('inf')
        
        # === Step 2.4: 敏感性分析 ===
        gamma_critical = rosenbaum_sensitivity(ate_dml, data, T)
        e_value = compute_e_value(ate_dml)
        decay_rate = effect_decay_rate(data, causal_graph, T)
        
        # === Step 2.5: CATE一致性 ===
        spearman_rho = spearman_correlation(cate_dml, cate_cf)
        sign_agreement = sign_consistency_rate(cate_dml, cate_dr, cate_cf)
        monotonicity = check_cate_monotonicity(cate_dml, cate_dr, cate_cf)
        
        # === Step 2.6: 综合判定 ===
        cevs = compute_cevs(
            placebo_ratio, ate_change_rate, ate_cv,
            sign_consistency, iv_dml_bias,
            gamma_critical, e_value, decay_rate,
            spearman_rho, sign_agreement, monotonicity
        )
        
        results[T] = {
            "score": cevs,
            "ate_dml": ate_dml, "ate_dr": ate_dr, "ate_cf": ate_cf, "ate_iv": ate_iv,
            "metrics": { ... }
        }
    
    # 取所有处理变量的最低CEVS作为全局CEVS
    global_cevs = min(r["score"] for r in results.values())
    verdict = "PASS" if global_cevs >= 0.75 else ("WARNING" if global_cevs >= 0.55 else "FAIL")
    
    return {"score": global_cevs, "verdict": verdict, "per_treatment": results}
```

### 7.4 Stage 3: 反事实推理验证Pipeline

```python
def validate_counterfactuals(data, causal_graph, counterfactual_plans, known_effects):
    """
    Stage 3: 反事实推理验证
    输入: 数据集, 因果图, 反事实方案列表, 已知干预效果
    输出: CFVS得分, 验证报告
    """
    
    # === Step 3.1: 因果约束违反率 ===
    violations = {"deterministic": 0, "direction": 0, "immutable": 0, "mediation": 0}
    total = len(counterfactual_plans)
    
    for plan in counterfactual_plans:
        if violates_deterministic_constraint(plan, causal_graph):
            violations["deterministic"] += 1
        if violates_direction_constraint(plan, causal_graph):
            violations["direction"] += 1
        if violates_immutable_constraint(plan):
            violations["immutable"] += 1
        if not mediation_propagation_complete(plan, causal_graph):
            violations["mediation"] += 1
    
    ccr = compute_ccr(violations, total)
    
    # === Step 3.2: 分布可行性 ===
    mahalanobis_distances = [mahalanobis_distance(plan, data) for plan in counterfactual_plans]
    extreme_ratios = [extreme_feature_ratio(plan, data) for plan in counterfactual_plans]
    density_ratios = [kde_density_ratio(plan, data) for plan in counterfactual_plans]
    ood_rates = [isolation_forest_ood_rate(plan, data) for plan in counterfactual_plans]
    
    # === Step 3.3: 已知效果对比 ===
    direction_match = []
    magnitude_match = []
    relative_errors = []
    
    for known in known_effects:
        cf_effect = find_matching_counterfactual_effect(known, counterfactual_plans)
        if cf_effect is not None:
            direction_match.append(sign(cf_effect) == sign(known.effect))
            magnitude_match.append(known.lower <= abs(cf_effect) <= known.upper)
            relative_errors.append(abs(cf_effect - known.effect) / abs(known.effect))
    
    # === Step 3.4: 综合判定 ===
    cfvs = compute_cfvs(ccr, mahalanobis_distances, extreme_ratios, 
                         density_ratios, direction_match, magnitude_match, relative_errors)
    
    verdict = "PASS" if cfvs >= 0.75 else ("WARNING" if cfvs >= 0.55 else "FAIL")
    
    return {"score": cfvs, "verdict": verdict, "metrics": { ... }}
```

### 7.5 Stage 4: 端到端验证Pipeline

```python
def validate_end_to_end(data, ml_model, causal_model, protected_attributes):
    """
    Stage 4: 端到端验证
    输入: 数据集, ML模型, 因果增强模型, 受保护属性
    输出: E2EVS得分, 验证报告
    """
    
    # === Step 4.1: 预测性能对比 ===
    ml_scores = ml_model.predict_proba(data)
    causal_scores = causal_model.predict_proba(data)
    
    auc_ml = roc_auc_score(data.target, ml_scores)
    auc_causal = roc_auc_score(data.target, causal_scores)
    auc_lift = auc_causal - auc_ml
    
    ks_ml = compute_ks(data.target, ml_scores)
    ks_causal = compute_ks(data.target, causal_scores)
    ks_lift = ks_causal - ks_ml
    
    brier_ml = brier_score_loss(data.target, ml_scores)
    brier_causal = brier_score_loss(data.target, causal_scores)
    brier_reduction = brier_ml - brier_causal
    
    # === Step 4.2: 校准度评估 ===
    ece_causal = compute_ece(data.target, causal_scores, n_bins=10)
    mce_causal = compute_mce(data.target, causal_scores, n_bins=10)
    hl_pvalue = hosmer_lemeshow_test(data.target, causal_scores)
    
    # === Step 4.3: 公平性评估 ===
    dp_diff = demographic_parity_difference(causal_scores, data[protected_attributes])
    eo_diff = equalized_odds_difference(data.target, causal_scores, data[protected_attributes])
    
    # === Step 4.4: 稳定性评估 ===
    temporal_auc_var_ml = compute_temporal_auc_variance(ml_model, data, time_col='MONTH')
    temporal_auc_var_causal = compute_temporal_auc_variance(causal_model, data, time_col='MONTH')
    stability_improvement = temporal_auc_var_ml - temporal_auc_var_causal
    
    # === Step 4.5: 综合判定 ===
    e2evs = compute_e2evs(
        auc_lift, ks_lift, brier_reduction,
        ece_causal, mce_causal, hl_pvalue,
        dp_diff, eo_diff, stability_improvement
    )
    
    verdict = "PASS" if e2evs >= 0.70 else ("WARNING" if e2evs >= 0.50 else "FAIL")
    
    return {"score": e2evs, "verdict": verdict, "metrics": { ... }}
```

### 7.6 Stage 5: 综合报告生成

```python
def generate_validation_report(stage_results):
    """
    Stage 5: 综合报告生成
    """
    
    ccgs = (0.25 * stage_results["cgvs"] + 
            0.30 * stage_results["cevs"] + 
            0.25 * stage_results["cfvs"] + 
            0.20 * stage_results["e2evs"])
    
    if ccgs >= 0.75:
        global_verdict = "EXCELLENT"
    elif ccgs >= 0.60:
        global_verdict = "GOOD"
    elif ccgs >= 0.45:
        global_verdict = "MARGINAL"
    else:
        global_verdict = "INSUFFICIENT"
    
    # 生成报告
    report = {
        "timestamp": current_timestamp(),
        "global_score": ccgs,
        "global_verdict": global_verdict,
        "stages": {
            "L1_causal_graph": stage_results["L1"],
            "L2_causal_effect": stage_results["L2"],
            "L3_counterfactual": stage_results["L3"],
            "L4_end_to_end": stage_results["L4"]
        },
        "uncertainty_ranges": compute_uncertainty_ranges(stage_results),
        "improvement_priorities": rank_improvement_priorities(stage_results),
        "model_card_updates": generate_model_card_updates(stage_results)
    }
    
    return report
```

### 7.7 Pipeline执行规范

| 规范项 | 要求 |
|--------|------|
| **执行频率** | 每次因果图更新后全量执行；日常监控仅执行L2+L4 |
| **执行环境** | 独立验证环境，与训练/推理环境隔离 |
| **数据要求** | 使用留出验证集（Hold-out），禁止使用训练数据验证 |
| **随机种子** | 全流程固定种子确保可复现，多种子验证时使用1-20 |
| **超时控制** | Stage 1: 4h, Stage 2: 8h, Stage 3: 2h, Stage 4: 4h |
| **失败策略** | 任何Stage FAIL → 停止后续Stage → 生成失败报告 |
| **审核流程** | WARNING结果需领域专家+技术负责人双签确认 |
| **版本管理** | 每次验证结果存入MLflow，与模型版本绑定 |

---

## 8. 学术依据与参考文献

### 8.1 因果发现验证

| 方法 | 学术依据 | 关键结论 |
|------|----------|----------|
| PC算法稳定性 | Spirtes, Glymour & Scheines (2000) *Causation, Prediction, and Search* | 约束法因果发现在有限样本下可能不稳定，需多种子验证 |
| NOTEARS | Zheng et al. (2018) "DAGs with NO TEARS", NeurIPS | 连续优化DAG学习，对初始化敏感，需多起点验证 |
| 交集融合 | Constantinou & Dawid (2017) "Extended Bayesian Information Criteria", JMLR | 多方法交集可显著降低假阳性率 |
| 领域知识注入 | Andrews et al. (2020) "Tiers of strength: Incorporating domain knowledge", UAI | 领域约束可将因果发现精度提升20-40% |

### 8.2 因果效应估计验证

| 方法 | 学术依据 | 关键结论 |
|------|----------|----------|
| 反驳测试框架 | Sharma et al. (2021) "DoWhy: An End-to-End Library for Causal Inference", arXiv | 反驳测试是因果效应鲁棒性的最低验证标准 |
| 安慰剂反驳 | Rosenbaum (2002) *Observational Studies* | 安慰剂处理下效应应为零是因果识别的必要条件 |
| 工具变量验证 | Angrist & Pischke (2009) *Mostly Harmless Econometrics* | F≥10是强工具变量的经验标准 |
| 双重鲁棒估计 | Chernozhukov et al. (2018) "Double/Debiased Machine Learning", Annals of Statistics | DR估计在混淆模型或结果模型之一正确时即可一致 |
| 因果森林 | Athey & Imbens (2018) "Estimation and Inference of Heterogeneous Treatment Effects", Review of Economics and Statistics | 因果森林提供渐近正态的CATE置信区间 |
| E-value | VanderWeele & Ding (2017) "Sensitivity Analysis in Observational Research", Annals of Internal Medicine | E-value量化未观测混淆解释效应所需的最小强度 |

### 8.3 反事实推理验证

| 方法 | 学术依据 | 关键结论 |
|------|----------|----------|
| 因果约束反事实 | Karimi et al. (2021) "Algorithmic Recourse under Imperfect Causal Knowledge", NeurIPS | 忽略因果约束的反事实方案在实践中不可行 |
| DiCE | Mothilal et al. (2020) "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations", FAT* | 多样性反事实生成需配合可行性约束 |
| 马氏距离可行性 | Mahalanobis (1936); De Maesschalck et al. (2000) | 马氏距离>χ²(p,0.95)表示样本在训练分布外 |
| 反事实评估 | Joshi et al. (2019) "Towards Realistic Individual Recourse and Actionable Explanations", FAT* | 反事实方案需同时满足有效性和可行性 |

### 8.4 端到端验证

| 方法 | 学术依据 | 关键结论 |
|------|----------|----------|
| 校准度评估 | Guo et al. (2017) "On Calibration of Modern Neural Networks", ICML | ECE是校准度的标准量化指标 |
| Hosmer-Lemeshow检验 | Hosmer & Lemeshow (1980) *Applied Logistic Regression* | HL检验是校准度拟合优度的经典方法 |
| Brier Score分解 | Brier (1950); Murphy (1973) | 可靠性-分辨率-不确定性三分解提供校准度诊断 |
| 公平性验证 | Barocas & Selbst (2016) "Big Data's Disparate Impact", California Law Review | Demographic Parity + Equalized Odds是公平性双保障 |
| 因果公平性 | Kusner et al. (2017) "Counterfactual Fairness", NeurIPS | 因果公平性：敏感属性的反事实变化不改变预测 |
| A/B测试设计 | Kohavi et al. (2020) *Trustworthy Online Controlled Experiments* | 最小样本量计算和护栏机制是A/B测试的必要条件 |

### 8.5 综合性参考文献

| 文献 | 贡献 |
|------|------|
| Pearl (2009) *Causality: Models, Reasoning, and Inference* | 因果推理的数学基础：do-演算、后门准则、前门准则 |
| Peters, Janzing & Schölkopf (2017) *Elements of Causal Inference* | 因果发现的可识别性理论和验证方法 |
| Hernán & Robins (2020) *Causal Inference: What If* | 因果效应估计的靶试验框架和验证策略 |
| Imbens & Rubin (2015) *Causal Inference for Statistics, Social, and Biomedical Sciences* | 潜在结果框架下的因果推断和敏感性分析 |
| VanderWeele (2015) *Explanation in Causal Inference* | 中介分析、交互效应和因果解释的严谨框架 |

---

## 附录A：快速判定速查表

| 验证层 | 关键指标 | PASS | WARNING | FAIL |
|--------|----------|------|---------|------|
| L1 因果图 | CGVS | ≥ 0.80 | 0.60-0.80 | < 0.60 |
| L2 因果效应 | CEVS | ≥ 0.75 | 0.55-0.75 | < 0.55 |
| L3 反事实 | CFVS | ≥ 0.75 | 0.55-0.75 | < 0.55 |
| L4 端到端 | E2EVS | ≥ 0.70 | 0.50-0.70 | < 0.50 |
| **全局** | **CCGS** | **≥ 0.75** | **0.60-0.75** | **< 0.60** |

## 附录B：致命错误清单（任一触发即全局FAIL）

| 编号 | 致命错误 | 检测方法 |
|------|----------|----------|
| F1 | 因果图出现禁止边（如TARGET→DAYS_BIRTH） | 领域知识一致性验证 |
| F2 | 安慰剂反驳ATE显著不为零（p < 0.05） | Placebo Treatment Refuter |
| F3 | 反事实方案修改不可变特征（如性别、年龄） | 不可变约束检查 |
| F4 | A/B测试中Treatment组违约率显著高于Control | 护栏机制监控 |
| F5 | 因果增强评分的公平性劣于纯ML评分 | Demographic Parity + Equalized Odds |
| F6 | 因果效应方向与自然实验结论相反 | 伪实验验证 |

## 附录C：验证报告模板

```markdown
# CausalCredit 因果推理验证报告

## 基本信息
- 验证日期: YYYY-MM-DD
- 模型版本: v1.x.x
- 数据集版本: YYYY-MM-DD
- 验证执行人: [姓名]

## 全局判定
- CCGS得分: X.XX
- 判定结果: EXCELLENT / GOOD / MARGINAL / INSUFFICIENT

## L1 因果图验证
- CGVS: X.XX (PASS/WARNING/FAIL)
- GSI: X.XX
- DKCS: X.XX
- 弱边比例: X%
- [详细指标表格]

## L2 因果效应估计验证
- CEVS: X.XX (PASS/WARNING/FAIL)
- [各处理变量的详细指标]

## L3 反事实推理验证
- CFVS: X.XX (PASS/WARNING/FAIL)
- CCR: X%
- [详细指标表格]

## L4 端到端验证
- E2EVS: X.XX (PASS/WARNING/FAIL)
- AUC提升: +X.XXX
- ECE: X.XXX
- [详细指标表格]

## 不确定性范围
- ATE 95% CI: [X.XX, X.XX]
- CATE 跨方法变异系数: X.XX
- [其他不确定性标注]

## 改进建议（按优先级排序）
1. [最高优先级改进项]
2. [次高优先级改进项]
3. ...

## 致命错误检查
- F1-F6: [全部通过 / 具体失败项]

## 签署
- 技术负责人: ____________ 日期: ____________
- 领域专家: ____________ 日期: ____________
```

---

> **文档结束** | 本验证标准体系为CausalCredit因果推理的核心质量保障框架，所有因果推理结论在对外输出前必须通过本体系的逐层验证。
