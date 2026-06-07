# CausalCredit M8.6 因果验证深化 实现记录

> 完成时间: 2026-06-07
> 提交: `be448e0` M8.6a + `70079cd` M8.6b + `b476a20` M8.6c + `81d692a` M8.6d
> 状态: 全部 4 个子任务 ✅ 通过, 393/393 单测通过 (1m 35s)

## 1. 目标

M8.1/M8.2 之后, 项目有了反欺诈三件套、公平性审计、三层因果叙事 — 但 **"模型到底可不可信"** 这个最关键的问题还散落在 4 个不同模块里:

- 反事实可不可信: CCR / immutable_violation 在 DiCE 里
- 效应可不可信: 4 类 refuter + BLP 在 estimate / blp_test 里
- 异质性稳不稳定: split-half bootstrap 在 stability 里
- 时间数据有没有泄漏: 旁路在 `MONTHS_BALANCE > 0` 行上

> 监管 / 合规 / 答辩评审**最常问的 1 句话**: "这套因果验证体系的总分是多少? 哪一环最弱?"

M8.6 系列 4 件 (TemporalGuard / BLP / CATE 稳定性 / CCGS 金字塔) 把这 4 件事收口, 并用 **CCGS (Composite Causal Grade Score)** 给出 1 个 0-1 的复合分。

## 2. 交付物 (按子任务)

| 子任务 | 文件 | 测试 |
|---|---|---|
| M8.6a TemporalGuard 数据泄漏防护 | `src/data/temporal_guard.py` (`TemporalGuard` + `TemporalIssue` + `TemporalGuardReport`) | 14 |
| M8.6b BLP 检验 (Best Linear Predictor) | `src/causal/blp_test.py` (`BLPTest` + `BLPResult` + `plot_blp_test`) | 17 |
| M8.6c CATE 稳定性 Tier1+Tier2 | `src/causal/stability.py` (`CATEStabilityTester` + `StabilityResult` + `plot_stability`) | 17 |
| M8.6d CCGS 因果验证金字塔 | `src/causal/verification.py` (`CausalVerificationPyramid` + `compute_ccgs` + `plot_pyramid`) | 25 |
| 顺手修复 | `src/causal/cate.py` (W=None → 省略 W kwarg, 修 econml 0.16 拒绝 0 列的 bug) | (含 M8.6b/c 失败回归) |

**测试增量**: 320 → 393 (+73)
**总测试文件数**: 29 → 33
**顺手修复的隐藏 bug**: `CATEEstimator` 旧代码把 `W=None` 转 `(n, 0)` 数组再传给 econml, 在 econml ≥ 0.15 上抛 `Found array with 0 feature(s) (shape=(400, 0)) while a minimum of 1 is required`. 这次 M8.6b/c 都依赖 DML, 必触发, 一并修了。

---

## 3. 关键算法选择

### 3.1 TemporalGuard (M8.6a) — 防御 "时间穿越"

Home Credit 二级表 (`POS_CASH_balance`, `credit_card_balance`, `installments_payments`) 都有 `MONTHS_BALANCE` 列:
- `0` = 申请当月 (允许)
- `-N` = 申请前 N 个月 (允许, 历史)
- `+N` = 申请后 N 个月 (**禁止** — 这是放款后才会产生的真实还款历史, 模型在决策时根本不知道)

如果这些行漏进聚合, 模型相当于"看见答案"。`TemporalGuard.scrub_secondary_tables(tables)`:
1. 对每张表检查 `MONTHS_BALANCE > 0` 的行
2. 算 ratio, 写一条 `TemporalIssue(type="MONTHS_BALANCE_LEAK", action="EXCLUDED")` 进 `TemporalGuardReport`
3. 删行, 留 cleaned dict

外加 2 个工具:
- `validate_split(df, date_col, train_ratio=0.8)`: 按日期升序切分, 前 80% = train, 后 20% = test, 返回带 `split` 列的 df
- `check_split_overlap(df, date_col)`: 检测 train max ≥ test min 时返回 `TEMPORAL_OVERLAP` issue (WARNING 级, 不删除, 让人决策)

**关键设计**: issue 是数据类 (`TemporalIssue`), 不是 `str` — 报告里包含 `count / ratio / action / detail`, 可直接序列化为 JSON 进 pipeline log。

### 3.2 BLP 检验 (M8.6b) — Chernozhukov et al. 2018

**问题**: 我们的 CATE 模型预测 `c_hat(X_i)`, 但**预测出的 c_hat 是否真的线性捕捉了 Y 的异质性? 还是只学到噪音?**

**方法**: OLS 回归 `Y ~ 1 + T + c_hat + c_hat·T`, 检验 `c_hat` 系数 (BLP coefficient) 的 p-value:
- p < 0.05 → CATE 预测对 Y 有**显著**的解释力, 异质性是 "真的"
- p ≥ 0.05 → CATE 没比常效应模型好多少, 异质性是 "噪音" (或模型没学到)

**实现 (`BLPTest.run`)**:
1. K-fold (默认 5) 分数据
2. 每折: 在 K-1 折上 refit CATE, 预测留出折的 c_hat → out-of-fold c_hat 整列
3. 用整列 c_hat 跑 OLS
4. `_compute_se_vec` 用经典 OLS SE (σ̂² = SSR / (n-p), cov = σ̂²·(X'X)⁻¹), 与 competitor 一致
5. t_stat = coef / se, p_value = 2 × t.sf(|t|, n-p)
6. `BLPResult` 含 blp_coef, blp_se, blp_t_stat, blp_p_value, pass_at_05, pass_at_10, design_coefs (T / c_hat / c_hat×T / intercept), cate_summary (mean/std/IQR)

**两个坑**:
1. **sklearn intercept 行为**: 用户传 `[1, T, c, cT]` 时, `LinearRegression(fit_intercept=True)` 会**丢弃**用户那列 1, 用自己的 `intercept_`, 导致 `coef_[0]` 永远是 0。修复: 设计矩阵**不**显式加 1, 让 sklearn 自己加。
2. **DGP 设计**: 测试用 DGP 必须让 Y 直接依赖 c_hat (X[:, 0]), 而不是只通过 c_hat·T。否则 c_hat 全是常数, BLP coef 退化为 0。

### 3.3 CATE 稳定性 Tier1+Tier2 (M8.6c) — Oracle P0 配方

借鉴 CausalBench / Schuler et al. 2018 的 CATE 稳定性诊断, 双 Tier:

**Tier 1 (split-half bootstrap)**:
- 30 次 (默认) 随机置换, 每次把数据切两半, 分别 refit CATE, 算两条 c_hat 的 Spearman ρ
- 报 **mean ρ**, 阈值 0.80 (PASS > 0.80)
- 答: "随机切样本, 异质性预测稳不稳定?"

**Tier 2 (hyperparameter sensitivity)**:
- 10 个 GBR 超参组合: `max_depth ∈ {3,4,5,6}`, `n_estimators ∈ {60,80,100,120,140,160}`, `min_samples_leaf ∈ {5,6,20}` (按 competitor 配方)
- 每个组合 refit LinearDML (直接调 econml, 不走 `CATEEstimator`, 因为要换 model_y / model_t), 算 C(10,2)=45 对 c_hat 的 Spearman ρ
- 报 **min pairwise ρ**, 阈值 0.70
- 答: "换一阶模型超参, 异质性预测稳不稳定?"

**overall_pass = Tier1 AND Tier2**, 输出 dataclass 含 `summary()` 1 句话供 pipeline log。

**关键设计**:
- Tier 1 用 `CATEEstimator`, 3 个 backend (LinearDML / SparseLinearDML / CausalForestDML) 都能跑
- Tier 2 直接调 `econml.dml.LinearDML(model_y=..., model_t=...)`, 因为 CATEEstimator 包装了超参, 不便替换
- T 一律按 continuous 处理 (与 `CATEEstimator` 默认一致, binary T 在 stability 测试里反而限制信号)

### 3.4 CCGS 因果验证金字塔 (M8.6d) — Composite Causal Grade Score

4 层 L1-L4 复合评分, **双门控 pass**:

```
              ┌─────────────────────────────┐
              │  L4  E2E Validation         │   0.20
              │   (AUC, ECE, Demographic P) │
              ├─────────────────────────────┤
              │  L3  Counterfactual         │   0.25
              │   (CCR, Immutable)          │
              ├─────────────────────────────┤
              │  L2  Effect Validation      │   0.30
              │   (Refutation + BLP + CATE) │
              ├─────────────────────────────┤
              │  L1  Graph Validation       │   0.25
              │   (GSI, DKCS)               │
              └─────────────────────────────┘

   CCGS = 0.25·L1 + 0.30·L2 + 0.25·L3 + 0.20·L4
   pass: CCGS ≥ 0.70 AND all_layers_pass   ← 双门控, 防一好遮百丑
```

**每层算法**:
| 层 | 指标 | 公式 |
|---|---|---|
| L1 Graph | GSI + DKCS | GSI = mean Jaccard(bootstrap ∩ ref_DAG); DKCS = per-edge 确认率; L1 = mean(GSI, DKCS) |
| L2 Effect | 5 tests pass rate | placebo / subset / sensitivity / BLP / CATE 一致性, 5 个中 pass 数 / 5 (缺测则分子分母同减) |
| L3 Counterfactual | min(CCR, imm_score) | CCR = 反事实合理率; imm_score = 1 - immutable_violation/0.2 (0.2 是不可变违反的硬上限) |
| L4 E2E | mean of available | (auc_causal − auc_baseline) / 0.05 + (1 − ECE/0.1) + (1 − DP/0.1) |

**双门控**: `overall_pass = CCGS ≥ 0.7 AND all 4 layers pass`。理由: 强 L1 (DAG 完美) + 弱 L2 (refutation 全挂) 时 CCGS 仍能 0.7+, 但模型实际不可信。**单层弱就整体挂**, 不允许平均掩盖。

**4 个关键设计**:
1. **缺失分数降级而非失败**: 没跑 CCR / ECE / DP 时, 该维度从 L 分母剔除 (不强制要求每个组件都有)
2. **BLPResult 自动转换**: `verify_l2_effect(blp_result=BLPResult(...))` 用 `pass_at_05` 当 pass 标志, caller 不用包成 dict
3. **可定制权重**: `compute_ccgs(weights={"l1": 0.4, ...})` 允许产品线调整 (现金贷 vs 信用卡侧重不同层)
4. **金字塔图**: `plot_pyramid` 2-panel — Panel A 是 L1-L4 条 + CCGS 虚线, Panel B 是文字摘要 (含 pass/fail)

---

## 4. Pipeline 集成 (STEP 17 — 计划中)

M8.6 系列**只做模块 + 单测 + 文档**, 端到端 STEP 17 接入 (L1-L4 拼起来跑出 CCGS + pyramid_score.json) 留作 M8.7+。原因: 4 个模块各自独立, 边界清晰, 集成时再处理 "CCGS L1 怎么拿到 bootstrap DAG 列表" / "L2 怎么把 4 类 refuter + BLP + CATE 一致性拼齐" 这种 IO 协调, 不在 M8.6 范围。

**计划接口** (STEP 17):
```python
# STEP 17: CCGS pyramid (planned M8.7+)
pyramid = CausalVerificationPyramid(threshold=0.7)
l1 = pyramid.verify_l1_graph(dag_edges, bootstrap_samples=discovery.bootstrap_dags)
l2 = pyramid.verify_l2_effect(
    placebo_result=refute_results["placebo"],
    subset_result=refute_results["subset"],
    sensitivity_result=refute_results["sensitivity"],
    blp_result=blp_result,                          # M8.6b
    cate_spearman=cate_cv_spearman,                 # M1 cate.cross_validate_methods
)
l3 = pyramid.verify_l3_counterfactual(
    ccr=counterfactual_reasoner.consistency_rate,
    immutable_violation=counterfactual_reasoner.immutable_violation,
)
l4 = pyramid.verify_l4_e2e(
    auc_baseline=baseline_auc, auc_causal=causal_auc,
    ece=calibrator.ece, demographic_parity_diff=fairness.dp_gap,
)
result = pyramid.compute_ccgs(l1, l2, l3, l4)
plot_pyramid(result, "output/figures/17_ccgs_pyramid.png")
# → output/decision_reports/ccgs_pyramid.json
```

**已就位 (不需重写)**:
- `refute.run_all_refutations` 4 类返回 `{pass, p_value, ...}` — L2 5 tests 中 3 个直接用
- `BLPResult.pass_at_05` — L2 BLP test 直接用
- `CATEEstimator.cross_validate_methods` — L2 CATE consistency 直接用
- `CounterfactualReasoner` 已有 `consistency_rate` 和 `immutable_violation` 字段 (M1 时代写入)
- `FairnessSummary.dp_gap` — L4 demographic parity diff 直接用

---

## 5. 实测数据

### 5.1 测试统计

| 模块 | 测试文件 | 用例 | 累计 |
|---|---|---:|---:|
| M8.6a TemporalGuard | `tests/test_temporal_guard.py` | 14 | 320 → 334 |
| M8.6b BLP | `tests/test_blp_test.py` | 17 | 334 → 351 |
| M8.6c CATE 稳定性 | `tests/test_stability.py` | 17 | 351 → 368 |
| M8.6d CCGS | `tests/test_verification.py` | 25 | 368 → 393 |

**全量 393 测试 1 分 35 秒全过**, 0 失败 / 0 跳过。

### 5.2 顺手修复的 bug

`CATEEstimator` (M1 时代写) 旧代码:
```python
def fit_dml(self, Y, T, X, W=None):
    if W is None:
        W = np.zeros((len(Y), 0))   # ← 这里! 0 列被 econml 0.15+ 拒绝
    model.fit(Y, T, X=X, W=W)
```

**修法**: 3 个 fit 方法 (`fit_dml` / `fit_dr` / `fit_causal_forest`) 全部改成:
```python
if W is None:
    model.fit(Y, T, X=X)             # 完全省略 W
else:
    model.fit(Y, T, X=X, W=W)
```

修这个之前 M8.6b/c 的 DML 调用全挂 (`Found array with 0 feature(s) (shape=(400, 0)) while a minimum of 1 is required`)。修完 M1 CATE demo 也跟着修了, 算"一次 BLP 失败找出 1 个隐藏 bug"。

### 5.3 模块单独 demo (不上 STEP 17)

```python
# M8.6a — TemporalGuard demo
from src.data.temporal_guard import TemporalGuard
guard = TemporalGuard()
cleaned, report = guard.scrub_secondary_tables({
    "pos": pos_df,    # 有 MONTHS_BALANCE 列
    "cc":  cc_df,
    "inst": inst_df,
})
print(report.passed)             # True if no WARNING issues
for issue in report.issues:
    print(issue.to_dict())       # {'type': 'MONTHS_BALANCE_LEAK', 'count': 1234, ...}

# M8.6b — BLP demo
from src.causal.blp_test import BLPTest
tester = BLPTest(n_folds=5, method="LinearDML")
res = tester.run(Y, T, X)
print(res.blp_p_value, res.pass_at_05)
plot_blp_test(res, "output/demo_m86b_blp.png")

# M8.6c — CATE 稳定性 demo
from src.causal.stability import CATEStabilityTester
tester = CATEStabilityTester(method="LinearDML", n_bootstrap=30, n_configs=10)
res = tester.run(Y, T, X)
print(res.summary)
# "method=LinearDML  Tier1 ρ̄=0.892 (PASS)  Tier2 min ρ=0.756 (PASS)  → overall STABLE"
plot_stability(res, "output/demo_m86c_stability.png")

# M8.6d — CCGS demo
from src.causal.verification import CausalVerificationPyramid
pyramid = CausalVerificationPyramid(threshold=0.7)
l1 = pyramid.verify_l1_graph([("A","B")], bootstrap_samples=None)
l2 = pyramid.verify_l2_effect(placebo_result={"pass": True}, ...)
l3 = pyramid.verify_l3_counterfactual(ccr=0.92, immutable_violation=0.0)
l4 = pyramid.verify_l4_e2e(auc_baseline=0.70, auc_causal=0.75, ece=0.05)
result = pyramid.compute_ccgs(l1, l2, l3, l4)
print(f"CCGS={result.ccgs:.3f}  pass={result.overall_pass}")
plot_pyramid(result, "output/demo_m86d_pyramid.png")
```

---

## 6. 业务价值

- **数据可信度 (M8.6a)**: `MONTHS_BALANCE > 0` 这类"时间穿越"在 M5 多表聚合里**没显式防御**, 这次补齐。后续 Lending Club (有 `issue_d` 真实日期) 接入时, `validate_split` / `check_split_overlap` 直接用, 不用重写。
- **效应可解释性 (M8.6b)**: BLP 检验是 Chernozhukov et al. 2018 的标准做法, 答辩时评审问 "CATE 是不是真的" 直接给 p-value, 不用临时编。
- **模型稳健性 (M8.6c)**: Tier1 抓"切样本", Tier2 抓"换超参", 双层防御。3 个 backend (LinearDML / SparseLinearDML / CausalForestDML) 都能跑。
- **合规可视化 (M8.6d)**: CCGS 是 1 个数字, 4 个分量。监管 / 合规的 1 句话问题 ("你这套到底可不可信?") 答 1 个数字, 详细追问再展开分层。
- **顺手修 bug**: `CATEEstimator` W=None 的隐藏问题修了, M1 时代所有 CATE demo 都受益。

---

## 7. 测试覆盖细节

### 7.1 M8.6a — 14 用例 (`test_temporal_guard.py`)
- `scrub_secondary_tables`: 空表 / 无时间列透传 / 全是 0 / 全是 +N (全删) / 混合 / 多种表字典
- `TemporalIssue.to_dict` / `TemporalGuardReport.passed`: WARNING → False
- `validate_split`: 不存在的 date_col 报错 / 非法 train_ratio 报错 / 80/20 切分顺序对
- `check_split_overlap`: 无 split 列 → None / train max ≥ test min → WARNING

### 7.2 M8.6b — 17 用例 (`test_blp_test.py`)
- DGP 强信号: BLP coef 显著, p < 0.05 → pass
- DGP 弱信号: BLP coef 不显著, p > 0.05 → fail
- K=2 折 (最小)
- `BLPResult.to_dict` 字段完整
- `plot_blp_test` PNG > 1KB
- 鲁棒性: Y/T/X 长度不齐报错 / n < 2*n_folds 报错 / 非法 method 报错 / 非法 alpha 报错

### 7.3 M8.6c — 17 用例 (`test_stability.py`)
- `tier1_split_half` 单独跑 5/30 次
- `tier2_hyperparameter_sensitivity` 单独跑 5 configs → 10 pairs
- `run` 端到端: 强信号 DGP → STABLE
- `StabilityResult.to_dict` 字段完整
- `plot_stability` PNG > 1KB
- 鲁棒性: 长度不齐 / n < 100 / 非法 method / 非法 n_bootstrap / 非法 n_configs

### 7.4 M8.6d — 25 用例 (`test_verification.py`)
- L1 Graph: 无 bootstrap placeholder / 完全重叠 → 1.0 / 部分重叠 → 0.75 / 全部错位 → fail
- L2 Effect: 全过 / 3/5 / 缺失测试排除 / BLPResult 透传
- L3 Counterfactual: 全空 placeholder / 完美 CCR / CCR 拉低 / 不可变违反拉低
- L4 E2E: AUC improvement 单维 / 强 AUC → 1.0 / 3 维组合
- CCGS: 全过 → pass / 单层 fail → fail (双门控) / to_dict 字段 / 权重不归 1 报错
- 鲁棒性: 非法 threshold (≤ 0 或 > 1) 报错
- Visualization: `plot_pyramid` PNG > 1KB

---

## 8. 限制与未来方向

- **STEP 17 未集成**: 4 个模块各自独立可用, 但还没在 `run_pipeline.py` 拼成 1 个 STEP 17 输出 CCGS。**未来**在 M8.7+ 加 STEP 17, 把 4 层收口到 1 个 `pyramid_score.json`。
- **L1 Graph bootstrap 样本来源未接**: CCGS `verify_l1_graph` 需要 `bootstrap_samples` (list of `Set[Tuple]`), 现在 `CausalDiscovery.fuse_graphs` 输出 1 个融合图, 没存 bootstrap 中间态。**未来**在 `discovery.py` 加 `run_with_bootstrap(n_bootstrap=5)`, 把每次 bootstrap 出来的图存下来给 L1 用。
- **CCGS 权重硬编码**: 默认 0.25/0.30/0.25/0.20 (按 competitor), 没产品级 override。**未来**接 `configs/config.yaml::ccgs_weights`, 类似 `fraud_guard`。
- **TemporalGuard 只查 MONTHS_BALANCE, 不查其他时间列**: 像 bureau 表的 `DAYS_CREDIT` 相对当前申请可能也有 -N / +N 风险, 但目前不在 guard 范围。**未来**加 `bureau` 时间列检查。
- **BLP 用经典 OLS SE, 不用 HC1**: 与 competitor 保持一致, 但样本 iid 性弱时 HC1 更稳。**未来**加 `use_hc1=True` 选项。

---

## 9. 后续里程碑

- **M8.7+**: STEP 17 CCGS 集成 (把 L1-L4 拼到 `run_pipeline.py`) + 1 张金字塔图 (17_ccgs_pyramid.png) + `ccgs_pyramid.json`
- **M8.8+**: TemporalGuard 拓展到 bureau / previous_application 的时间列检查
- **M9+**: P0 提案文档落地 (6.15 提案前关键交付) — 蓝图一页纸 / Demo 演示脚本 / 答辩 Q&A 手册 / 代码走读速查
