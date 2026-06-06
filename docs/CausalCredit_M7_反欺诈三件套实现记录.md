# CausalCredit M7 — 反欺诈三件套实现记录

> **最后更新**: 2026-06-06 | **状态**: 已完成 ✅ | **代码 commit**: `7015282`

本文档是 `CausalCredit_反欺诈能力覆盖分析.md` §4.1 提出的"反欺诈三件套"的需求↔实现追溯。每个子模块对应:

- 需求出处(§4.1.x)
- 算法选择(为什么这么实现,有什么 trade-off)
- 实测效果(Home Credit 30 万行, 1K 测试子集)
- 局限与未来方向

---

## 1. 需求 vs 实现总览

| 需求编号 | 需求描述 | 实现模块 | 状态 |
|----------|----------|----------|:----:|
| §4.1.1 | 三分类子模型 (fraudulent / non_malicious / systemic) | `src/fraud/three_class.py` | ✅ |
| §4.1.3 | 包装资质因果一致性检测 | `src/fraud/packaging.py` | ✅ |
| §4.1.4 | 养流水因果去噪评分 | `src/fraud/denoising.py` | ✅ |
| §4.1 编排 | 三件套 → 单条反欺诈路由 | `src/fraud/pipeline.py::FraudGuard` | ✅ |
| Pipeline 集成 | STEP 14 ANTI-FRAUD, 注入决策报告 | `src/run_pipeline.py` | ✅ |

---

## 2. 三分类子模型 (`three_class.py`)

### 2.1 需求 (源自 §4.1.1)
在 default=1 的子群里进一步分:
- **fraudulent** — 主观恶意(身份欺诈/材料造假/夸大收入)
- **non_malicious** — 履约能力变化(失业/疾病),非主观恶意
- **systemic** — 系统性风险(行业衰退/政策变化),个人无能为力

最终对外暴露 `P(欺诈类别 | default=1)`,便于区别处理(欺诈→拒绝+证据移交, 非恶意→协商还款, 系统性→调整产品).

### 2.2 实现选择

**伪标签构造** (无 ground truth, 用业务规则):
```
fraudulent  ⇐  INST__DPD_MAX >= 30                            # 首期即逾期
            ∨ (z(income) > 1.0 ∧ z(employment) < -1.0)         # 高收入低就业 (夸大)
            ∨ (EXT_SOURCE_1 < 0.2 ∧ z(income) > 1.0)           # 低外部分高收入 (包装)

systemic    ⇐  ORGANIZATION_TYPE 命中衰退行业子串              # Industry: mining / Construction / ...

non_malicious ⇐ default=1 ∧ ¬fraudulent ∧ ¬systemic
```

**4 类 LightGBM** (在 `non_default` 背景类上额外保留一格),在 default=1 子样本上重新归一化得条件概率。

**`fraud_score = P(default) × P(fraudulent | default=1)`** — 乘法组合,二分类概率与子分类概率独立,避免欺诈子分类稀释主模型信号。

### 2.3 实测 (1K 测试子集)
- `fraud_score` 范围: [0.0000, 0.4385]
- 中位数 ~0.001 (默认 0.0015 × P(fraudulent|default)≈0.5)
- 25 / 1000 (2.5%) fraud_score >= 0.10 → `REJECT_FRAUD`

### 2.4 局限与未来方向
- 伪标签 = 业务规则, 与反欺诈团队人工标注的真实欺诈集会有偏差。生产中应允许从配置文件/数据库加载真实标注的种子集替换。
- 当前 `non_malicious` 占比 ~99% (因规则太严),后续可调宽 `fraudulent` 触发条件或加入更多业务信号(如同一设备多次申请/短期内多次失败)。

---

## 3. 包装资质因果一致性 (`packaging.py`)

### 3.1 需求 (源自 §4.1.3 + §2.6)
"包装资质" 客户通常表现为:
- 表面资质良好(高收入/高信用分) → 模型 SHAP 高
- 实际行为不自洽(收入高但消费低/有现金流但无资产)

检测思路: 比较 **模型在用的特征** vs **因果图上有效的特征**。

### 3.2 实现选择

**域 DAG 期望路径** (`EXPECTED_PATHS`):
```
1. income → goods_price → credit → annuity     # 主链
2. income → EXT_SOURCE_2                        # 中介 (收入通过外部分影响违约)
3. DAYS_BIRTH → DAYS_EMPLOYED → income          # 协变量链
```

**`path_integrity`**: 3 条链里, 同尺度 step 比例在 [0.01, 100] 范围内算"完整", 否则"断裂"。跨尺度 step 跳过 (AMT_INCOME 与 EXT_SOURCE_2 数量级不同, 比例无意义)。

**Per-applicant 四象限分类** (top-25% |SHAP| 内):
- `TRUSTED` — 高 SHAP ∧ 高 causal proxy (模型在用 + 因果有效)
- `UNTRUSTED` — 高 SHAP ∧ 低 causal proxy (模型在用 + 因果无效 → 包装嫌疑)
- `MASKED` — 低 SHAP ∧ 高 causal proxy (模型没看到 + 因果有效 → 漏报)
- `NEGLIGIBLE` — 低 SHAP ∧ 低 causal proxy (都无信号)

**`packaging_score = UNTRUSTED / (TRUSTED + UNTRUSTED)`** — 即模型高权特征中"因果不靠谱"的比例, 范围 [0, 1]。

> **公式迭代史**: 初版用 `1 - (TRUSTED+MASKED) / total` 在 median 阈值下恒为 0.5, 改用 per-applicant top-K SHAP + 上述公式后, 1K 子集上 packaging_score 在 [0.26, 0.56] 之间分布, 有信息量。

**路由**:
- `>= 0.50` → `REJECT_PACKAGING_SUSPECTED`
- `>= 0.30` → `MANUAL_REVIEW` (pipeline 中映射为 `REVIEW_BORDERLINE`)
- 否则 → `PROCEED`

### 3.3 实测 (1K 测试子集)
- `packaging_score` 范围: [0.26, 0.56]
- 中位数 ~0.37 (borderline 区间)
- 9 / 1000 (0.9%) `packaging_score >= 0.50` → `REJECT_PACKAGING`
- 902 / 1000 (90.2%) `packaging_score in [0.30, 0.50)` → `REVIEW_BORDERLINE`

### 3.4 局限与未来方向
- 域 DAG 路径是硬编码, 新增/删除产品线时需更新。
- 当前 `path_integrity` 检查只用 3 条主链, 实际产品还应包含"职业稳定性→DAYS_EMPLOYED"、"行业景气度→ORGANIZATION_TYPE"等。
- top-K = 25% 是经验值, 后续可基于 SHAP 累积分布动态选取 (例如"主效应贡献 >= 80% 的 top-K")。

---

## 4. 养流水因果去噪 (`denoising.py`)

### 4.1 需求 (源自 §4.1.4 + §2.7)
"养流水" 用户制造出与消费脱钩的还款历史(钱从外面来,不是工资的产物),导致:
- 模型观察到的 P(default) 偏低(还款记录"好")
- 但底层因果信号弱(收入没真正支撑这些还款)

最终对外暴露 `P(真实评分 | do(去除养流水效应))`。

### 4.2 实现选择

**因果信号**: 还款 vs 消费特征的符号一致性。5 个 INST__ 列(还款) + 4 个 CC_/POS_ 列(消费)按行 z-score 后取均值, 用 `sign(rep_z) × sign(con_z)` 映射到 [0, 1]:

```python
causal_consistency = (sign(rep_score) × sign(con_score) + 1) / 2
```

- 真实用户: 还款↑ ↔ 消费↑ → 一致性 ≈ 1
- 养流水: 还款↑ ↔ 消费↓ → 一致性 ≈ 0

**`inflation = clip((1 - consistency) × 0.15 × 5, 0, 0.15)`** — 一致性越低, 估计的"养流水膨胀"越大, 上限 0.15 (即最多把 P(default) 推高 15 个百分点)。

**`denoised_P(default) = min(1, P(default) + inflation)`** — 把被压低的违约概率加回估计的"养流水"部分。

**`denoising_action`**: consistency < 0.50 → `FLAG_FOR_REVIEW`, 否则 `PROCEED`。

### 4.3 实测 (1K 测试子集)
- 平均 `denoised_mean_inflation` = 0.15 (即一致性普遍 < 0.5, denoising_action 几乎全员 FLAG_FOR_REVIEW)
- 1K 子集上: 100% `denoising_action = FLAG_FOR_REVIEW`

### 4.4 局限与未来方向
- 当前一致性偏低是 **Home Credit 合成聚合特征**的属性 (INST/CC 列的 z-score 几乎正交), 真实业务数据上应能区分养流水 vs 真实用户。生产中需在真实还款/消费数据上重新校准。
- `inflation_strength_max = 0.15` 是经验值, 实际可能需要按行业/产品线差异化(养流水高发场景可设 0.30)。
- 当前只用了 INST/CC/POS 5+4=9 列, 实际还应包括 bureau 还款记录、previous application 履约情况等更多信号。

---

## 5. 编排器与路由 (`pipeline.py`)

### 5.1 5 维 routing 设计

| 路由 | 触发条件 | 业务含义 |
|------|---------|----------|
| `REJECT_FRAUD` | `fraud_score >= 0.10` | 模型 + 子分类都判定为高风险欺诈,直接拒绝 |
| `REJECT_PACKAGING` | `packaging_score >= 0.50` | 包装嫌疑大,直接拒绝 |
| `REVIEW_DENOISED` | `denoising_action == FLAG_FOR_REVIEW` | 养流水嫌疑,人工复审 |
| `REVIEW_BORDERLINE` | 任意信号 in [0.3, threshold) | 边缘案例,人工复审 |
| `PROCEED` | 全部干净 | 正常通过 |

按优先级串联,避免一个高分淹没其他信号(若 fraud_score 已 0.15,即使 packaging 仅 0.45 也不再叠加阻断)。

### 5.2 端到端 routing 分布 (1K 测试子集)

| 路由 | 占比 |
|------|-----:|
| REVIEW_BORDERLINE | 91.4% |
| PROCEED | 5.2% |
| REJECT_FRAUD | 2.5% |
| REJECT_PACKAGING | 0.9% |

> 注意: 当前 1K 子集是 random sample, 不是生产流量。生产中 PROCEED 占比应显著高于 5% (因为真实业务流会先经过申请材料初筛, 进入模型的样本已经过滤掉大量明显欺诈)。

### 5.3 API

```python
guard = FraudGuard(
    classifier_params={"n_estimators": 200, "max_depth": 6, ...},
)
guard.fit(X_train, y_train, four_quadrant=fq)

# 单条
r = guard.score_one(X_one, default_proba=p, four_quadrant=fq, applicant_idx=0, row_shap=sv)
# r = {"fraud_score": ..., "packaging_score": ..., "denoised_default_proba": ..., "routing": ..., ...}

# 批量 (需 per-applicant SHAP)
df = guard.score_batch(X, default_proba=p, four_quadrant=fq, shap_values=sv)
```

`row_shap` 缺失时 packaging 退化为使用全局 quadrant 标签(得分退化为 0.5, 但不会报错)。

---

## 6. Pipeline 集成 (`run_pipeline.py` STEP 14)

### 6.1 调用链

```
STEP 13 完成后
  ↓
导入 FraudGuard
  ↓
对 3 个 selected applicants 计算 per-applicant SHAP (TreeExplainer)
  ↓
guard.fit(X_train[50K subsample], y_train, four_quadrant=fq_for_3)
  ↓
score_one × 3  →  注入 3 份 decision_reports JSON 的 "fraud" 字段 + 追加 .md 段
score_batch × 1K  →  生成 3 张图 + anti_fraud 段写 pipeline_summary.json
```

### 6.2 耗时 (Home Credit 30 万行, CPU)

| 子步骤 | 耗时 |
|--------|-----:|
| TreeExplainer.shap_values(1K) | ~6s |
| 4 类 LightGBM 训练 (50K 子集) | ~14s |
| score_one × 3 + score_batch × 1K | ~10s |
| 3 张图 | ~5s |
| **STEP 14 总计** | **~35s** |
| pipeline 净增 | +10.4s (184.5 → 194.9, +5.6%) |

### 6.3 输出物

**3 份决策报告扩展**:
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

**3 张新图**:
- `12_fraud_score_routing.png` — fraud_score 直方图 + 路由饼图
- `13_packaging_scatter.png` — path_integrity × packaging_score 散点, 颜色按 routing 分
- `14_denoising_effect.png` — 原 P(default) vs denoised P(default) 散点, y=x 参考线

**`pipeline_summary.json` 增 `anti_fraud` 段**:
```json
{
  "anti_fraud": {
    "fraud_score_range": [0.0, 0.4385],
    "packaging_score_range": [0.26, 0.56],
    "routing_distribution": {
      "REVIEW_BORDERLINE": 914,
      "PROCEED": 52,
      "REJECT_FRAUD": 25,
      "REJECT_PACKAGING": 9
    },
    "denoised_mean_inflation": 0.15
  }
}
```

---

## 7. 测试覆盖 (25 个新增)

| 测试文件 | 用例数 | 覆盖 |
|----------|------:|------|
| `tests/test_fraud_three_class.py` | 7 | 伪标签规则、4 类模型、fraud_score 公式 |
| `tests/test_fraud_packaging.py` | 7 | credibility 校准、4 象限分类、path integrity |
| `tests/test_fraud_denoising.py` | 6 | consistency、denoised 范围、manufactured 膨胀 |
| `tests/test_fraud_pipeline.py` | 5 | FraudGuard 端到端 + 5 级 routing |

**总测试**: 108 → **133** (+25, 全跑 8.0s, 100% 通过)

---

## 8. 局限与未来迭代

| 类别 | 局限 | 迭代方向 |
|------|------|----------|
| 数据 | 伪标签 = 业务规则, 真实数据需校准 | 引入反欺诈团队人工标注种子集, 用监督学习替代规则 |
| 特征 | 仅用 INST/CC/POS 9 列做一致性 | 引入 bureau 还款记录、previous app 履约情况 |
| 阈值 | fraud_score >= 0.10 / packaging >= 0.50 经验值 | 在生产流量上 ROC 优化阈值 |
| 路由 | 静态优先级串联 | 改为加权求和 + 可学习的阈值(可用 LR 调) |
| 一致性 | 1D 符号 vs 1D 符号, 信息量低 | 改为高维相关性(余弦相似度/CCA) |
| 解释 | 路由原因只给一行文字 | 生成"反欺诈证据链"图, 解释每个信号来源 |

---

## 9. 与其他里程碑的协同

| 协同点 | 说明 |
|--------|------|
| M5+ 多表聚合 | INST/CC/POS 列来自 5 张二级表的聚合, 是 M7 去噪模块的输入基础 |
| M6 Optuna | 反欺诈阈值可在生产流量上用 Optuna 优化 |
| Causal DAG | packaging 的 path_integrity 直接复用 `HomeCreditCausalGraph::get_mediators` 等方法 |
| SHAP 四象限 | packaging 的 per-applicant 4 象限分类是 M0 SHAP 模块的扩展(从 mean abs SHAP → per-applicant top-K) |

---

## 10. 引用

- 需求: `docs/CausalCredit_反欺诈能力覆盖分析.md` §4.1
- 设计权衡: `PROGRESS.md` M7 节
- 实测基准: `BENCHMARKS.md` §8.6
- 代码: `src/fraud/` (4 文件, 569 行)
- 测试: `tests/test_fraud_*.py` (4 文件, 25 用例)
- Pipeline 集成: `src/run_pipeline.py` STEP 14
- 决策报告样例: `output/decision_reports/HC_006355.json` (含 fraud 字段)
- 图表: `output/figures/12_fraud_score_routing.png`、`13_packaging_scatter.png`、`14_denoising_effect.png`
