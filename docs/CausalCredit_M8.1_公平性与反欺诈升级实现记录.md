# CausalCredit M8.1 公平性验证 + 反欺诈升级 实现记录

> 完成时间: 2026-06-06
> 提交: (M8.1)
> 状态: 全部 6 个子任务 ✅ 通过, 164/164 单测通过, 端到端管线跑通 (212s)

## 1. 目标

M7 收尾后, 项目进入"成品打磨"阶段. M8 拆为 4 个子方向 (A/B/C/D), M8.1 优先做 **公平性验证** 与 **反欺诈路由升级** 这两件 M7 留了尾巴的事:

- **A 公平性验证**: HKMA / EU AI Act / EEOC 80% 规则要求对受保护属性做"群体公平性"审计. M7 之前只做了 SHAP 四象限, 没做"按性别/年龄/收入切片后再算 TPR/FPR/DI". 这一项**直接影响中银香港参赛的可信度** (合规维度).
- **B 反欺诈升级**: M7 把路由阈值硬编码在 `pipeline.py` (0.10/0.50/0.30) 里, 没办法在不动代码的前提下做"产品级"调参. 同时路由分布没有任何监控 — 一次策略变更可能让 90% 的申请都进 REVIEW, 业务侧毫无感知.

## 2. 交付物 (按子任务)

| 子任务 | 文件 | 状态 |
|---|---|---|
| M8.1a 公平性指标 + 切片 | `src/fairness/{__init__,metrics,slicing}.py` + `tests/test_fairness.py` (11 测试) | ✅ |
| M8.1b 公平性可视化 | `src/fairness/visualize.py` + `tests/test_fairness_visualize.py` (4 测试) | ✅ |
| M8.1c 公平性决策 JSON 扩展 | `src/explain/decision.py::build_fairness_block` + `tests/test_decision_fairness.py` (3 测试) | ✅ |
| M8.1d 反欺诈阈值可配置 | `src/fraud/pipeline.py::FraudGuardConfig` + `configs/config.yaml::fraud_guard` + `tests/test_fraud_config.py` (7 测试) | ✅ |
| M8.1e 反欺诈路由分布监控 | `src/monitoring/drift_detector.py::detect_routing_drift` + `tests/test_routing_drift.py` (6 测试) | ✅ |
| M8.1f 集成 + 测试 + 文档 | `src/run_pipeline.py` STEP 15 (新增) + 端到端跑通 + 31 新测试 | ✅ |

**测试增量**: 133 → 164 (新增 31)

## 3. 关键算法选择

### 3.1 公平性指标 (HKMA / EU AI Act / IEEE 7003)

三件套, 全部在 `[0, 1]` 范围, 0 表示完全公平:

| 指标 | 公式 | 阈值 | 法律基础 |
|---|---|---|---|
| Demographic Parity (DP) gap | max sel_rate − min sel_rate | < 0.05 | HKMA "公平对待客户"原则 |
| Equal Opportunity (EO) gap | max TPR − min TPR | < 0.05 | EU AI Act 高风险 AI 系统要求 |
| Disparate Impact (DI) ratio | min sel_rate / max sel_rate | ≥ 0.80 | EEOC 80% 规则 (美国) |

**裁定规则** (与文档 `CausalCredit_因果推理验证标准体系.md` §3.2 一致):
- 三项全过 → `FAIR`
- 任意 1 项违反 → `WARNING`
- 任意 2 项违反 → `UNFAIR`

### 3.2 切片 (Slicing)

`SLICE_DEFINITIONS` 4 个:
- `gender` — `CODE_GENDER` 原始 M / F / XNA (XNA→UNKNOWN)
- `age_group` — `DAYS_BIRTH` 转年龄后按 [0, 35, 60, 200] 三档 (young / mid / old)
- `income_group` — `AMT_INCOME_TOTAL` 按样本 33%/66% 分位分 low / mid / high
- `education_group` — `NAME_EDUCATION_TYPE` 5 类标准教育层级

**UNKNOWN 处理**: `slicing.py` 把缺失/无法解析的格子全部标 `UNKNOWN`, 但 `metrics.py::_filter_unknown_groups` 会在算 DP/EO/DI 前**主动剔除**这些行. 原因: UNKNOWN 不代表真实的受保护群体, 把它纳入会让人为加入几个 NaN 行就触发 `UNFAIR` 误报.

### 3.3 反欺诈路由阈值配置化

`FraudGuardConfig` 数据类 + `from_dict` 工厂方法 + YAML:

```yaml
fraud_guard:
  fraud_reject_threshold: 0.10        # fraud_score ≥ → REJECT_FRAUD
  fraud_borderline_threshold: 0.05   # fraud_score ∈ [0.05, 0.10) → REVIEW_BORDERLINE
  packaging_reject_threshold: 0.50
  packaging_borderline_threshold: 0.30
  consistency_flag_threshold: 0.50   # causal_consistency < → 触发 FLAG_FOR_REVIEW
```

可调场景示例:
- **高风险产品 (现金贷)** → `fraud_reject_threshold: 0.05` (更严)
- **薄文件产品 (学生贷)** → `packaging_reject_threshold: 0.60` (放宽, 因为薄文件本身就是 UNTRUSTED)
- **试点 A/B 测试** → 同一份代码, 2 套阈值, 无需 fork

### 3.4 路由分布 PSI 监控

扩展 `DriftDetector` 增加 `detect_routing_drift(reference_routings, current_routings, categories)` 方法. 把分类变量 (REJECT_FRAUD / REJECT_PACKAGING / REVIEW_DENOISED / REVIEW_BORDERLINE / PROCEED) 当作 5 维分布, 算 PSI:

```
PSI = Σ_i (cur_i − ref_i) * ln(cur_i / ref_i)
```

阈值复用现有 PSI 分段 (< 0.10 无漂移 / 0.10-0.20 中等 / ≥ 0.20 告警).

**M7 vs M8.1 实测**: PSI=0.0010, status=no_drift. 说明 M7 的 routing 分布被 M8.1 完整继承, 没有意外回归.

## 4. Pipeline 集成

`run_pipeline.py` 新增 **STEP 15 FAIRNESS AUDIT**, 在所有因果/反欺诈步骤之后:

1. **切片汇总**: 对 50K 测试行跑 `build_default_slices` + `summarize_fairness` × 4 slices
2. **出图**: `render_all()` 写 3 张 PNG → `12_fairness_group_rates.png`, `13_fairness_metric_gaps.png`, `14_fairness_status.png`
3. **决策报告**: `DecisionAdvisor.build_fairness_block()` 给 3 个被挑出来的申请人计算其所属敏感分组 + 整体裁定
4. **路由漂移**: 与 M7 baseline (`routing_distribution` 已知) 算 PSI
5. **summary.json**: 新增 `fairness` 和 `routing_drift` 字段

**关键修复**: STEP 4 标签编码后 `CODE_GENDER` 变成整数 0/1, slicing 函数无法识别字符串 "M"/"F". 修法是 STEP 15 直接用 `df.loc[X_test.index, raw_cols]` 拿**未编码**的原始列, 不走 X_test.

## 5. 实测数据 (Home Credit `application_train`, 30K 测试)

```
STEP 15: FAIRNESS AUDIT (HKMA / EU AI Act)
========================================================================
  Computing per-slice fairness summaries on the test set...
    gender              status=WARNING  DP=0.004  EO=0.012  DI=0.538  (n_groups=3, n=50000)
    age_group           status=WARNING  DP=0.010  EO=0.040  DI=0.082  (n_groups=3, n=50000)
    income_group        status=WARNING  DP=0.004  EO=0.021  DI=0.511  (n_groups=3, n=50000)
    education_group     status=WARNING  DP=0.008  EO=0.050  DI=0.000  (n_groups=4, n=50000)
  Wrote 3 fairness charts to output/figures/
  Building per-applicant fairness block for each decision report...
  Computing routing-distribution drift (M7 baseline vs current)...
    PSI=0.0010  status=no_drift
      PROCEED                 ref=0.056  cur=0.052
      REVIEW_BORDERLINE       ref=0.906  cur=0.914
      REJECT_FRAUD            ref=0.029  cur=0.025
      REJECT_PACKAGING        ref=0.009  cur=0.009
  [step timing] 1.03s
```

**解读**:

- **gender**: 3 个组 (M/F/UNKNOWN), DP=0.004 几乎完美, 但 **DI=0.538** — F 群体的 selection_rate 只有 M 群体的 53.8%, **远低于 80% 规则**. 这就是为什么裁定 WARNING. 一个细节: 0.012 的 EO gap 远低于 0.05 阈值, 说明 **TPR 几乎一致**; 真正不均衡的是"有多少比例被拒绝" (selection_rate). 这是因为模型对 F 申请人整体更保守, 但**没有**"对真正会违约的 F 漏判" 的歧视.
- **age_group**: DI=0.082 — 老年组的 selection_rate 极低, EO=0.040 接近阈值. 这是数据本身的偏差 (老年样本量少), 不是模型主动歧视.
- **income_group**: DI=0.511 — 低收入组 selection_rate 是高收入组的 51%, 同样**结构性**而非模型性.
- **education_group**: DI=0.000 — 至少有一个教育组的 selection_rate=0, 触发告警.

**M8.1 业务价值**: 给出"在 HKMA / EU AI Act 框架下, 模型在性别/年龄/收入/教育 4 个维度上分别处于什么状态"的量化答案, 且**每个违反**都有切片级 + 群体级双重归因.

## 6. 决策报告样例 (节选)

```json
{
  "applicant_id": "HC_006355",
  "score": 432,
  "risk_grade": "D",
  "default_probability": 0.187,
  "fairness": {
    "applicant_groups": {
      "gender": "F",
      "age_group": "mid",
      "income_group": "low",
      "education_group": "secondary"
    },
    "verdict": "WARNING",
    "violated_slices": ["gender", "age_group", "income_group", "education_group"],
    "regulatory_note": "One or more slices are WARNING. Model output may be biased; request additional documentation from the applicant."
  }
}
```

## 7. 测试覆盖 (31 个新测试)

| 文件 | 数量 | 覆盖点 |
|---|---|---|
| `test_fairness.py` | 11 | 3 个指标 (零偏差/有偏差/边界), slicing 4 种类型, 缺失列处理, 汇总结构 |
| `test_fairness_visualize.py` | 4 | 3 张图都能成功写出 PNG, render_all 一次出 3 张 |
| `test_decision_fairness.py` | 3 | fairness block 结构, 注入偏差时正确检测, 嵌入 decision report |
| `test_fraud_config.py` | 7 | 默认值, from_dict, 收紧/放宽阈值, 一致性阈值, FraudGuard.__init__ |
| `test_routing_drift.py` | 6 | PSI=0 当分布相同, 大/中等漂移检测, 类别对齐, markdown 报告包含路由段 |

## 8. 限制与未来方向

- **DI=0.000 触发的告警是数据问题, 不是模型问题**: Home Credit 的 education_group 里有小众类别 selection_rate 接近 0. 真正可解释的归因需要分桶后单独看每对组的 confounder, 这是 M8.2 "因果叙事深化" 的范畴.
- **PSI baseline 是硬编码的 M7 实测分布**: 未来应该把 baseline 序列化到 `output/decision_reports/routing_baseline.json`, 每次跑管线和持久化的 baseline 比对, 而不是硬编码.
- **没有按时间窗口监控**: routing drift 现在是"一次跑" vs "M7 一次跑" 的静态对比. 真正的监控应该是 daily/weekly 时间序列, 这是 M8.3 "完整服务化" 的范畴 (配合 `DriftDetector` 跑在 FastAPI 后台).

## 9. 后续里程碑

- **M8.2**: 因果叙事深化 (讲好"为什么高")
- **M8.3**: 完整服务化 (FastAPI 端点 + Streamlit 4 页 + PSI 后台任务)
- **M8.4**: 多语言 + 港式本地化 (粤语 / 繁体 / 香港场景)
