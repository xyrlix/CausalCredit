# CausalCredit 核心代码走读速查 (P0 #4 子产物)

> **目的**: 12 个核心模块 5 分钟讲解提纲 + 关键代码片段引用 + 翻车兜底  
> **使用方**: 李定泉 + 许海波 认领 (各 6 个) · 答辩前 1 周分头走读  
> **配套**: `项目总结与行动建议.html` P0 #4 子表 + 答辩 Q&A 手册

---

## 走读约定

- 每个模块 **5 分钟**: 1 分钟目标 + 2 分钟核心算法 + 1 分钟翻车点 + 1 分钟 Q&A
- 关键代码引用格式: `path:line` 直接可跳
- 测试引用: `pytest tests/test_*.py::test_* -v` 现场可跑
- 兜底话术统一用「三明治」: 承认局限 → 解释方案 → 给出 Phase 2 路径

---

## 1. 领域 DAG (M0)

**文件**: `src/causal/home_credit_graph.py:1-238` (238 行)

**5 分钟讲解稿**:
- 1 分钟: 这是 18 节点 / 36 边的领域因果图, 5 类节点 (treatment / outcome / confounder / mediator / sensitive)
- 2 分钟: 5 个核心方法 (DFS 验 acyclic / get_confounders 算 ancestor / get_instruments 找 ancestor 但非直接边 / get_mediators 找路径中间 / get_dot_string 出 Graphviz)
- 1 分钟: 翻车点 — `M8.2g` 加了 EXT_SOURCE_1/2/3 + BUREAU_TYPE_MICROLOAN_FRAC 4 个新边, 测试用 `test_ext_source_1_3_and_microloan_frac_in_dag` 校验
- 1 分钟: Q&A 见 `Q4 因果发现可信吗`

**关键代码**:
```python
# home_credit_graph.py:120-125 找 treatment 和 outcome 的共同祖先 (confounder)
ancestors_tx = self._get_ancestors(treatment)
ancestors_out = self._get_ancestors(outcome)
return [n for n, p in self.nodes.items()
        if p.get("type") in ("confounder", "sensitive")
        and n in ancestors_tx and n in ancestors_out
        and n not in (treatment, outcome)]
```

**测试**: `pytest tests/test_causal_graph.py -v` (9 测试)

---

## 2. 混合因果发现 (M1)

**文件**: `src/causal/discovery.py`

**5 分钟讲解稿**:
- 1 分钟: 5 个方法 (PC / NOTEARS / fuse / inject_domain_knowledge / compare_with_domain)
- 2 分钟: PC (Constraint-based, 条件独立测试) + NOTEARS (Score-based, 连续优化求解 DAG 矩阵) 怎么融合 — 取交集边 + 置信度加权
- 1 分钟: 翻车点 — 30 万行 PC/NOTEARS 太慢, **降采样到 30K**, 仍能 100% 召回关键边
- 1 分钟: Q&A — "召回率 100% 怎么算的?" 答: 合成数据上注入已知 DAG, 跟发现图对比

**关键代码**:
```python
# discovery.py — 融合 PC + NOTEARS, 边的存在需 2 算法都同意
def fuse_graphs(pc_graph, notears_graph, edge_conf_threshold=0.7):
    fused = nx.DiGraph()
    for u, v in list(pc_graph.edges()) + list(notears_graph.edges()):
        in_pc = pc_graph.has_edge(u, v)
        in_nt = notears_graph.has_edge(u, v)
        if in_pc and in_nt:  # 双重支持 → 高置信
            fused.add_edge(u, v, confidence=0.95)
        elif in_pc or in_nt:  # 单支持 → 低置信
            if edge_conf_threshold <= 0.5:
                fused.add_edge(u, v, confidence=0.6)
    return fused
```

**测试**: `pytest tests/test_causal_discovery.py -v` (5 测试)

---

## 3. ATE / 反驳验证 (M1)

**文件**: `src/causal/estimate.py` (ATE) + `src/causal/refute.py` (4 类 refuter)

**5 分钟讲解稿**:
- 1 分钟: ATE = 平均处理效应, 衡量"对所有申请人施加处理, 风险平均变化多少"
- 2 分钟: 4 类 refuter — Placebo (把处理换噪声) / Random common cause (加随机混杂) / Data subset (切 5 子集) / E-value (未观测混杂要多强)
- 1 分钟: 翻车点 — ATE 数值在 30 万行上 **数值很小** (0.0001 量级), 不代表没效果, 因为大多数申请人没被"处理"
- 1 分钟: Q&A — "为什么用 DoWhy 不用自己写?" 答: DoWhy 是 Microsoft 维护的工业级库, 4 类 refuter 文档完整, 自写要重写几千行

**关键代码**:
```python
# refute.py — Placebo Treatment Refuter
def refute_placebo_treatment(self, estimate):
    new_causal_model = self.causal_model.refute_estimate(
        identified_estimand=self.estimand,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
    )
    return {
        "new_ate": float(new_causal_model.new_effect.value),
        "p_value": float(new_causal_model.refutation_result["p_value"]),
        "passed": abs(new_cause.new_effect.value) <= 0.01,
    }
```

**测试**: `pytest tests/test_refute.py -v` (7 测试, 含 E-value 公式)

---

## 4. CATE 异质处理效应 (M1)

**文件**: `src/causal/cate.py`

**5 分钟讲解稿**:
- 1 分钟: CATE = 条件平均处理效应, "**这个特定申请人**处理后风险变化多少"
- 2 分钟: 3 种 EconML 方法 — `LinearDML` (线性) / `SparseLinearDML` (稀疏线性, 自动特征选择) / `CausalForestDML` (随机森林, 非线性)
- 1 分钟: 翻车点 — 3 方法的 CATE 不可能完全一致, 我们用 **Spearman ρ** 交叉验证, 接受阈值 0.5 (真实数据) / 0.7 (合成数据)
- 1 分钟: Q&A — "为什么 3 方法?" 答: 单方法有过拟合风险, 3 方法交叉验证是业界最佳实践

**关键代码**:
```python
# cate.py — 3 方法 cross_validate
def cross_validate_methods(self, X, Y, T, W) -> Dict:
    methods = {"LinearDML": self.fit_dml, "SparseLinearDML": self.fit_sparse_dml,
               "CausalForestDML": self.fit_causal_forest}
    cate_results = {name: fn(Y, T, X, W).effect(X) for name, fn in methods.items()}
    # Spearman 相关矩阵
    rho_matrix = pd.DataFrame({
        name1: [spearmanr(cate_results[name1], cate_results[name2])[0]
                for name2 in methods]
        for name1 in methods
    }, index=methods.keys())
    return {"cate_per_method": cate_results, "spearman_matrix": rho_matrix}
```

**测试**: `pytest tests/test_cate.py -v` (8 测试, 含 subgroup analysis)

---

## 5. SHAP 四象限 (M1)

**文件**: `src/explain/shap_explain.py:1-348` (348 行, **核心创新**)

**5 分钟讲解稿**:
- 1 分钟: SHAP = 模型重要性的"博弈论公平分配", 4 象限 = 模型重要 × 因果重要 的 2×2 分类
- 2 分钟: `causal_proxy()` (用 ±σ 扰动算 ∂P/∂X, 局部敏感度代理) + `causal_vs_noncausal_contribution()` 标每个特征的象限 (TRUSTED/UNTRUSTED/MASKED/NEGLIGIBLE)
- 1 分钟: 翻车点 — 阈值是动态的 (用 median), 不是硬编码 0.01, 这样不同数据集自适应
- 1 分钟: Q&A — "学术有先例吗?" 答: 组合无, 4 象限分类法原创, 见 `Q7 SHAP × 因果有先例吗`

**关键代码**:
```python
# shap_explain.py:212-228 4 象限分类
th_shap = float(np.median(list(mean_abs.values())))
th_causal = float(np.median(proxy))
for f in self.feature_names:
    high_shap = mean_abs[f] >= th_shap
    high_causal = proxy[f] >= th_causal
    if high_shap and high_causal:    q = "TRUSTED"      # 模型+因果都同意
    elif high_shap and not high_causal: q = "UNTRUSTED"  # 模型说有, 因果没说
    elif not high_shap and not high_causal: q = "NEGLIGIBLE"  # 都低, 忽略
    else: q = "MASKED"               # 因果有, 模型没看到
```

**测试**: `pytest tests/test_shap.py -v` (6 测试)

---

## 6. DiCE 反事实 (M1)

**文件**: `src/explain/counterfactual.py`

**5 分钟讲解稿**:
- 1 分钟: DiCE = Diverse Counterfactual Explanations, "怎样改才能从'违约'翻到'不违约'"
- 2 分钟: NSGA-II 多目标 (proximity / sparsity / plausibility) + IMMUTABLE 锁死 (DAYS_BIRTH/CODE_GENDER/DAYS_ID_PUBLISH/EDUCATION) + SEMI_MUTABLE 限幅 50%
- 1 分钟: 翻车点 — `causal_plausibility = 1 - clip(|delta_proba| / |cate_for_treatment|, 0, 1)`, 大 delta vs 小 CATE = 不可信
- 1 分钟: Q&A — "DiCE 找不到反事实?" 答: 通常是 feature 在搜索空间外, 我们 catch 异常返回空 list, 不让 UI 崩

**关键代码**:
```python
# counterfactual.py — IMMUTABLE 锁死
IMMUTABLE_FEATURES = {"DAYS_BIRTH", "CODE_GENDER", "DAYS_ID_PUBLISH",
                      "NAME_EDUCATION_TYPE"}
SEMI_MUTABLE_FEATURES = {"AMT_INCOME_TOTAL", "DAYS_EMPLOYED",
                          "AMT_CREDIT", "AMT_ANNUITY"}
# DiCE 内部会跳过 immutable, semi_mutable 用 bounds 限幅
```

**测试**: `pytest tests/test_counterfactual.py -v` (6 测试)

---

## 7. 决策建议 (M1)

**文件**: `src/explain/decision.py`

**5 分钟讲解稿**:
- 1 分钟: DecisionAdvisor 把 (预测 + SHAP + CATE + 反事实 + 公平性) 合成 1 份给信贷员的报告
- 2 分钟: 信用分公式 `300 + 550 * (1 - p) ** 1.5`, 截断 [300, 850]; A-E 分级 (750/A, 650/B, 550/C, 450/D, else E)
- 1 分钟: 翻车点 — 分级阈值是经验值, 业界没有统一标准; Phase 2 会用 BOCHK 实际违约率校准
- 1 分钟: Q&A — "为什么 A-E 不是 AAA-DDD?" 答: 跟国际三大信用机构 (FICO / VantageScore) 的 5 档对齐, 评审员易理解

**关键代码**:
```python
# decision.py — 信用分 + 分级
def _compute_score(p: float) -> int:
    p = max(0.0, min(1.0, p))
    return int(round(300 + 550 * (1 - p) ** 1.5))

def _compute_grade(score: int) -> str:
    if score >= 750: return "A"
    if score >= 650: return "B"
    if score >= 550: return "C"
    if score >= 450: return "D"
    return "E"
```

**测试**: `pytest tests/test_decision.py -v` (5 测试)

---

## 8. 证据链 (M1)

**文件**: `src/explain/evidence.py`

**5 分钟讲解稿**:
- 1 分钟: EvidenceChainGenerator 把模型输出转成"信贷员可读"的话术
- 2 分钟: 4 个方法 — `generate_risk_evidence` (SHAP 转中文) / `generate_causal_evidence` (CATE 转中文) / `generate_counterfactual_evidence` (反事实转中文) / `generate_audit_evidence` (审计追踪)
- 1 分钟: 翻车点 — 模板填充, 数字精度是 `%.4f`, 中文模板用 "客户违约率约 X% (Y 分位)" 而非 "Y 风险"
- 1 分钟: Q&A — "中英文切换?" 答: M8.4a 给 CausalNarrative 加了多语言, evidence 部分当前仅中文, Phase 2 扩展

**测试**: 含在 `test_decision.py` (5 测试共用 fixture)

---

## 9. 因果叙事 v2 (M8.2 — **新增, 重点**)

**文件**: `src/explain/causal_narrative.py:1-471` (471 行)

**5 分钟讲解稿**:
- 1 分钟: M8.2 把 1 句话叙事升级成 4 段 (model / cohort / individual / robustness), 是产品可解释性突破
- 2 分钟: 4 段分别做什么 — model_level (全局 top-3) / cohort_level (KNN k=10 同类对照) / individual_level (本申请人 + DAG 路径) / robustness (20× ±10% 扰动)
- 1 分钟: 翻车点 — DAG 路径用 BFS, 限制 max_length=4 防爆栈; robustness 默认开, 但 20 次 × TreeSHAP ≈ 1s, 不慢
- 1 分钟: Q&A — "为什么 M8.2 不在原版报告里?" 答: 原版给机器读, M8.2 给信贷员/客户读, 见 `Q7 SHAP × 因果有先例吗`

**关键代码**:
```python
# causal_narrative.py:329-347 build_full_narrative
def build_full_narrative(self, features, shap_row, shap_global,
                          X_train, y_prob_train, four_quadrant, run_robustness):
    out = {
        "model_level": self.model_level_narrative(shap_global, top_k=3),
        "cohort_level": self.cohort_level_narrative(features, X_train, y_prob_train, k=10),
        "individual_level": self.individual_level_narrative(features, shap_row, four_quadrant, top_k=5),
    }
    if run_robustness:
        out["robustness"] = self.explanation_robustness(features, shap_row,
                                                          n_perturbations=20, noise_frac=0.10)
    return out
```

**测试**: `pytest tests/test_causal_narrative.py -v` (17 测试, 含多语言 4 测试)

---

## 10. 反欺诈三件套 (M7 — **必走读**)

**文件**: `src/fraud/{three_class,packaging,denoising,pipeline}.py`

**5 分钟讲解稿**:
- 1 分钟: 3 个子分类器 (三分类 / 包装资质 / 养流水去噪) + 1 个 orchestrator (FraudGuard)
- 2 分钟:
  - `three_class.py`: LightGBM 4 分类 (non_default + fraudulent / non_malicious / systemic), 伪标签规则 4 条
  - `packaging.py`: `packaging_score = UNTRUSTED / (TRUSTED + UNTRUSTED)`, 路径完整性 3 条 DAG 链
  - `denoising.py`: `causal_consistency = sign(repayment_z) × sign(consumption_z)`, `inflation = clip((1-consistency) × 0.15 × 5, 0, 0.15)`
- 1 分钟: 翻车点 — 伪标签是**主动声明的概念验证**, 4 条业务规则来自零售信贷, 不是统计拟合
- 1 分钟: Q&A — 见 `Q2 反欺诈用伪标签, 学术上站不住脚`

**关键代码**:
```python
# fraud/pipeline.py — 5 级路由
def route(self, fraud_score, packaging_score, denoised_proba, baseline_proba):
    if fraud_score >= 0.10: return "REJECT_FRAUD"
    if packaging_score >= 0.50: return "REJECT_PACKAGING"
    if denoised_proba - baseline_proba > 0.05: return "REVIEW_DENOISED"
    if packaging_score >= 0.30: return "REVIEW_BORDERLINE"
    return "PROCEED"
```

**测试**: `pytest tests/test_fraud_*.py -v` (4 个文件, 25 测试)

---

## 11. 公平性审计 (M8.1)

**文件**: `src/fairness/{metrics,slicing,visualize}.py`

**5 分钟讲解稿**:
- 1 分钟: 3 项指标 × 4 默认切片 = 12 个检测
- 2 分钟: 指标定义 — `demographic_parity_gap = max-min 选中率` (阈值 < 0.05) / `equal_opportunity_gap = max-min TPR` (阈值 < 0.05) / `disparate_impact_ratio = min/max 选中率` (阈值 ≥ 0.8); 切片 — gender / age / income / education
- 1 分钟: 翻车点 — `_filter_unknown_groups` 必须过滤 UNKNOWN, 否则几个坏样本能翻转 FAIR → UNFAIR
- 1 分钟: Q&A — "DI = 0.082 这么差怎么解释?" 答: 老年组 (age_group=old) 样本极少 (n<200), DI 估计方差大, 不可靠; 但我们**主动披露**, 不掩盖

**关键代码**:
```python
# fairness/slicing.py — 4 默认切片定义
SLICE_DEFINITIONS = {
    "gender": lambda df: df["CODE_GENDER"],
    "age_group": lambda df: pd.cut(-df["DAYS_BIRTH"] / 365, bins=[0, 30, 50, 100],
                                    labels=["young", "mid", "old"]),
    "income_group": lambda df: pd.qcut(df["AMT_INCOME_TOTAL"], q=3,
                                        labels=["low", "mid", "high"]),
    "education_group": lambda df: df["NAME_EDUCATION_TYPE"],
}
```

**测试**: `pytest tests/test_fairness.py -v` (11 测试)

---

## 12. 服务化 (M8.3) + 多语言 (M8.4)

**文件**:
- `src/api/{app,services,routes,dependencies,schemas}.py` (~ 600 行)
- `src/frontend/{app,pages}.py` (~ 400 行)
- `src/explain/causal_narrative.py::render_markdown` (多语言)

**5 分钟讲解稿**:
- 1 分钟: FastAPI 5 端点 + Streamlit 4 页 + M8.4a 多语言 3 语
- 2 分钟:
  - FastAPI: lifespan 自动加载 registry_v1.pkl, 第二次启动 < 5s
  - Streamlit: 4 页 (Score / Causal / Counterfactual / Decision), 第 4 页有 5 tab (含 M8.2 叙事)
  - 多语言: render_markdown(language="zh-HK") 用 _NARRATIVE_LABELS 字典
- 1 分钟: 翻车点 — registry pickle 缓存 7.4MB, 必须 `lifespan` 上下文管理, 改代码后强制 `force_retrain=True` 重训
- 1 分钟: Q&A — "为什么不用 FastAPI 的 Depends 注入?" 答: 单进程单 registry 性能更好, 答辩时少解释

**关键代码**:
```python
# api/app.py — FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    reg = get_model_registry()
    reg.load()  # 加载或重训
    app.state.registry = reg
    yield

# explain/causal_narrative.py:415-470 多语言字典
_NARRATIVE_LABELS = {
    "zh": {"title": "因果叙事", "model": "模型层面", ...},
    "zh-HK": {"title": "因果敘事", "model": "模型層面", ...},
    "en": {"title": "Causal Narrative", "model": "Model-level", ...},
}
```

**测试**:
- `pytest tests/test_api_smoke.py -v` (11 测试, FastAPI TestClient)
- `pytest tests/test_streamlit_smoke.py -v` (12 测试, fake streamlit)

---

## 走读日历 (6.15 之前)

| 日期 | 负责人 | 模块 | 时长 |
|------|--------|------|------|
| 6.08 | 李定泉 | 1. DAG / 2. 因果发现 / 3. ATE | 1.5h |
| 6.08 | 许海波 | 4. CATE / 5. SHAP / 6. DiCE | 1.5h |
| 6.09 | 李定泉 | 7. 决策 / 8. 证据 / 9. 因果叙事 v2 | 1.5h |
| 6.09 | 许海波 | 10. 反欺诈 / 11. 公平性 / 12. 服务化 | 1.5h |
| 6.10 | 全员 | 互相走读 + 翻车演练 | 2h |
| 6.12 | 卢鸿璋 | 评委视角通读 (找新翻车点) | 2h |

**走读产物**:
- 每人 10 分钟录屏 (`output/walkthroughs/{member}_module_{n}.mp4`)
- cheat sheet 补充到本文件附录
- Q&A 手册 (`CausalCredit_答辩Q&A手册.md`) 新增发现

---

**最后更新**: 2026-06-07 · 走读认领: 李定泉 (6 模块) + 许海波 (6 模块) · 兜底: 陈天元 + 卢鸿璋
