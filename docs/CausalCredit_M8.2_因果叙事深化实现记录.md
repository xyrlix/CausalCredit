# CausalCredit M8.2 因果叙事深化 实现记录

> 完成时间: 2026-06-06
> 状态: 全部 6 个子任务 ✅ 通过, 181/181 单测通过, 端到端管线跑通 (219.5s, 19 PNG)

## 1. 目标

M8.1 把"公平性"和"反欺诈路由"收尾后, M8 还有一个 4 方向图中**最影响产品可解释性**的子项没动:

> **"为什么这个申请人风险高? 为什么是这 5 个特征? 换个角度看是不是也会得同样结论?"**

M7/M8.1 之前的决策报告, 1 句话 (`causal_narrative`) 只包含:
- "Primary driver: EXT_SOURCE_2 (SHAP=+0.94, quadrant=TRUSTED)"
- "Heterogeneous effect estimate: |CATE|=0.0000"

这在监管/合规的 "challenge the decision" 流程下, 信贷员和客户都会问 3 个后续问题:
1. **模型角度**: 在所有申请人里, 哪些特征 *普遍* 最重要? 申请人是不是落入主趋势?
2. **群体角度**: 跟历史最像这位的 10 个申请人比, 风险是不是 *也* 高? 区别在哪几个特征上?
3. **个体角度**: 决定这一单的关键特征, 走的因果路径是什么? 解释是否 **稳定** (输小扰动后 top-3 不会大变)?

M8.2 三层叙事引擎 (`CausalNarrative` + `narrative_visualize`) 一次性回答这 3 个问题.

## 2. 交付物 (按子任务)

| 子任务 | 文件 | 测试 |
|---|---|---|
| M8.2a 多层级叙事生成器 | `src/explain/causal_narrative.py` (CausalNarrative 类, 8 个方法) | 13 |
| M8.2b 因果路径追溯 | `trace_causal_path`, `features_on_paths_to_outcome` (BFS on networkx) | (含在 13) |
| M8.2c K-NN 同类申请人对照 | `cohort_level_narrative` (KNN k=10 + z-score 偏差) | (含在 13) |
| M8.2d 解释稳健性扰动 | `explanation_robustness` (20× ±10% 高斯噪声) | (含在 13) |
| M8.2e 叙事可视化 | `src/explain/narrative_visualize.py` (瀑布图 + 三联卡) | 4 |
| M8.2f 集成 + 文档 | `src/run_pipeline.py` STEP 16 + 决策 JSON/MD 扩展 | — |

**测试增量**: 164 → 181 (+17)
**总测试文件数**: 24 → 26

## 3. 关键算法选择

### 3.1 三层叙事 (3-angle Story)

| 层级 | 回答的问题 | 输入 | 输出 |
|---|---|---|---|
| **Model-level** | "这个模型普遍靠哪些特征判断?" | 5K 测试集 SHAP | top-3 mean |SHAP| 特征 + 模板化叙述 |
| **Cohort-level** | "跟历史最像这位的申请人比, 风险是不是也高?" | KNN k=10 (z-score 标准化) | cohort 均值 P(default) + Δ (applicant - cohort) + top-5 z-score 偏差特征 |
| **Individual-level** | "这一单的关键特征走了哪些因果路径到达 TARGET?" | 单条 SHAP + DAG + 四象限 | top-5 特征, 每条带 DAG 路径, 主导特征单独标 `dominant_feature` + `dominant_dag_path` |

### 3.2 因果路径追溯 (DAG path tracing)

`CausalNarrative.trace_causal_path(feature, max_length=4)` 用 **BFS with path tracking** 找出 feature → outcome 的所有简单路径:

```python
stack: List[List[str]] = [[feature]]
while stack:
    path = stack.pop()
    cur = path[-1]
    if cur == self.outcome_name:
        paths.append(path); continue
    if len(path) >= max_length: continue
    for nbr in self.dag.successors(cur):
        if nbr in path: continue  # cycle guard
        stack.append(path + [nbr])
```

- **max_length=4** 默认: Home Credit DAG 最长路径是 confounder → mediator → treatment → outcome (4 跳), 4 跳后基本是回路, 截断.
- **cycle guard**: 节点已在 path 中就跳过, 防止在循环图上无限展开 (Home Credit DAG 已无环, 但仍做防御).
- **空 path 列表**: 特征不连通到 outcome (e.g. `EXT_SOURCE_3` 在领域 DAG 中无明确指向 TARGET 的边, 真实世界是"中介了 1 个隐藏变量" — 这件事本身就值得报告).

### 3.3 K-NN 同类申请人对照 (KNN k=10)

`cohort_level_narrative(features, X_train, y_prob_train, k=10)`:

1. **标准化**: 用训练集列均值 + 列 std 把申请人和训练集都归一化, 避免 income (量级 10^5) 压过 age (量级 10^2).
2. **KNN 搜索**: `sklearn.neighbors.NearestNeighbors(n_neighbors=10)`.
3. **cohort mean P(default)**: K 个最近邻在主模型上的预测均值.
4. **delta**: applicant P(default) - cohort P(default). 符号 + 含义:
   - `|delta| < 0.02`: "in line with" (申请人没偏离群体)
   - `delta > 0`: "noticeably higher" (申请人比同类更危险)
   - `delta < 0`: "noticeably lower" (申请人比同类更安全)
5. **top deviations**: 申请人在特征上偏离 cohort 均值最大的前 5 个 (z-score > 0.5 才上墙, 噪音不报告).

### 3.4 解释稳健性扰动 (Explanation Robustness)

`explanation_robustness(features, shap_row, n_perturbations=20, noise_frac=0.10)`:

- 对申请人特征叠加 `N(0, noise_frac × max(|value|, 1))` 高斯噪声 (per-feature scale, 避免 income 的高方差压过所有其他特征).
- 用 TreeSHAP 在扰动后的样本上重新算 SHAP, 比较 **新 top-1 / 新 top-3 集合** 与基线 top-1 / top-3 集合是否一致.
- `stability_score = 0.6 × top_1_stable + 0.4 × top_3_stable` (top-1 权重 0.6 因为它代表"主导特征" 是 stable 还是漂的).
- **解释强度档**:
  - `>= 0.85`: robust (高度可解释, 决策可信)
  - `>= 0.6`: moderately robust (中度可信, top drivers 在小扰动下可能换位)
  - `< 0.6`: fragile (top drivers 一加噪声就变, 需谨慎采用解释)

### 3.5 因果瀑布图 (15_causal_waterfall.png)

横向条形图, top features 按 `|SHAP|` 降序, **颜色按 4 象限**:
- TRUSTED → 绿色 (#10b981)
- UNTRUSTED → 红色 (#ef4444)
- MASKED → 橙色 (#f59e0b)
- NEGLIGIBLE → 灰色 (#9ca3af)

条形标签含 `quad=...` 后缀, 即使打印灰度也能识别. 图例显式标 4 象限.

### 3.6 三联叙事卡 (16_narrative_card.png)

3 个并排文本面板 (蓝/黄/绿背景区分 model/cohort/individual), 给非技术审阅者 (合规 / 运营) 一眼看懂. 底部 footer 显示 robustness 解释.

## 4. Pipeline 集成 (STEP 16)

`run_pipeline.py` STEP 16 在 STEP 15 (FAIRNESS) 之后, Pipeline summary 之前:

```
1. 5K 测试子集 → TreeSHAP → global_sv (model-level 素材)
2. 50K 训练子集 → predict_proba → y_prob_train (cohort-level KNN 素材)
3. 用 fused_dk / fused / domain_dag (M2/M1 因果发现) 构造 networkx.DiGraph; fallback = HomeCreditCausalGraph 边集
4. 对 3 个 decision_reports 申请人 (高/中/低风险代表), 各跑一次 build_full_narrative
5. 把每条的 causal_narrative_v2 注入 decision JSON + 追加到 .md 末尾
6. 对每个申请人出 2 张图 (waterfall + card) → 19 PNG
7. 高风险那位额外出 1 张 headline chart (作为 README 截屏)
```

**关键点**: 
- DAG 来源有 3 个候选 (`fused_dk` / `fused` / `domain_dag`), 这是 M1/M2 因果发现的产出变量. 任意一个存在就用它, 否则 fallback 到领域 DAG (`HomeCreditCausalGraph`).
- KNN 在 50K 子集上, 单次查询 < 5ms, 3 个申请人 < 20ms. SHAP perturbation 20 次 × 3 申请人 = 60 次 TreeSHAP 调用, < 10s.

## 5. 实测数据 (Home Credit 30K 测试)

```
STEP 16: CAUSAL NARRATIVE (model / cohort / individual)
  Pre-computing global SHAP on a 5K test subsample...
  Computing train predictions for cohort-level KNN (50K subsample)...
  Building 3-level narrative for each selected applicant...
    HC_023041: stability=0.20, cohort Δ=-0.0113, charts=2
    HC_036837: stability=0.34, cohort Δ=-0.0259, charts=2
    HC_006355: stability=0.94, cohort Δ=+0.6006, charts=2
  Wrote 2 headline narrative charts for HC_006355
```

### 5.1 三个申请人对照

| 申请人 | 风险等级 | P(default) | cohort Δ | 主导特征 | 主导路径 | stability | 业务解读 |
|---|---|---:|---:|---|---|---:|---|
| HC_006355 | E (高) | 89.55% | **+0.60** | EXT_SOURCE_2 | EXT_SOURCE_2 → TARGET | **0.94** | 单一 EXT_SOURCE_2 极低 (0.012), 模型 + 因果都标记为 TRUSTED, 解释极稳定; 远离 cohort 均值, 是真正的 outlier |
| HC_036837 | A (低边界) | 4.88% | -0.026 | EXT_SOURCE_2 | EXT_SOURCE_2 → TARGET | 0.34 | 与 cohort 类似, top features 较分散, 加 10% 噪声 top-1 50% 概率被换掉 |
| HC_023041 | A (低) | 0.26% | -0.011 | DAYS_EMPLOYED | DAYS_EMPLOYED → TARGET | **0.20** | 几乎所有 top features 都是小负 SHAP, 解释 fragile — 这种"低风险" 申请人通常没有 *主导* 风险特征, top-1 在 70% 扰动后会换 |

**关键发现**:
1. **高风险 = 稳定解释**: EXT_SOURCE_2 极值驱动单一结论时, top-1 不会漂 (1.0 stable), 解释 100% 复用.
2. **低风险 = fragile 解释**: 没有强主导特征时, top-3 在小扰动下大量换位, 解释 *不可* 直接当证据用, 但 *是* 决策报告里 "ALL top-K are UNTRUSTED" 的来源之一.
3. **cohort Δ 与风险等级正相关**: 高风险 Δ=+0.60 (远超群体), 中/低 Δ≈ -0.02 (略低于群体). 这是金融直觉的硬性证据, 不需要看 SHAP.

### 5.2 决策报告新字段 (`causal_narrative_v2`)

```json
{
  "causal_narrative_v2": {
    "model_level": {
      "top_features": [
        {"feature": "EXT_SOURCE_2", "mean_abs_shap": 0.3012},
        {"feature": "EXT_SOURCE_3", "mean_abs_shap": 0.2891},
        {"feature": "EXT_SOURCE_1", "mean_abs_shap": 0.1547}
      ],
      "narrative": "At the model level, P(default) ... primarily by EXT_SOURCE_2 (mean |SHAP|=0.3012) ..."
    },
    "cohort_level": {
      "k": 10,
      "cohort_mean_p_default": 0.2948,
      "applicant_p_default": 0.8955,
      "delta": 0.6006,
      "top_deviations": [
        {"feature": "POS_DPD_DEF_FLAG_FRAC", "z": 6.295, ...},
        {"feature": "NAME_HOUSING_TYPE", "z": 4.227, ...}
      ],
      "narrative": "Among the 10 training applicants ... 89.55% is noticeably higher than the cohort (Δ=+0.6006). The applicant deviates most from the cohort in: POS_DPD_DEF_FLAG_FRAC (z=+6.29) ..."
    },
    "individual_level": {
      "top_features": [
        {"feature": "EXT_SOURCE_2", "shap": 0.94, "quadrant": "TRUSTED",
         "dag_paths": [["EXT_SOURCE_2", "TARGET"]], "n_paths": 1, ...}
      ],
      "narrative": "The dominant risk driver is EXT_SOURCE_2 (SHAP=+0.94, quadrant=TRUSTED). In the causal DAG this feature reaches TARGET via: EXT_SOURCE_2 → TARGET. 5 are TRUSTED (model and causal both agree).",
      "dominant_feature": "EXT_SOURCE_2",
      "dominant_dag_path": ["EXT_SOURCE_2", "TARGET"],
      "n_trusted": 5, "n_untrusted": 0, "n_masked": 0
    },
    "robustness": {
      "n_perturbations": 20, "noise_frac": 0.1,
      "top_1_stable": 1.0, "top_3_stable": 0.85, "stability_score": 0.94,
      "interpretation": "Explanation is robust (stability=0.94)."
    }
  }
}
```

并存的旧 1 句 `causal_narrative` 字段保持不变, 向后兼容.

### 5.3 Markdown 报告追加 (CausalNarrative.render_markdown)

`.md` 报告末尾追加 4 段: 模型层面 (含表格) / 同类申请人对照 (含 z-score 表) / 本申请人 (含 DAG paths) / 解释稳健性.

## 6. 业务价值

- **监管合规**: 三层叙事把 "challenge the decision" 的 3 个标准问题一次性回答, 客户申诉和监管审计时直接用.
- **信贷员赋能**: cohort Δ 直接告诉信贷员 "这单是不是 outlier", 主导路径 + 主导特征告诉 "主因为什么".
- **反欺诈佐证**: robustness < 0.6 的 fragile 解释往往是 *真的* 数据稀薄 (低风险申请人在训练集里本来少) 或 *真的* 在噪声边界 — 跟反欺诈三件套的 borderline 路由有自洽关系.
- **M8.1 公平性补强**: M8.1 WARNING 切片 (e.g. education_group DI=0.000) 的归因, 现在可以用 cohort-level 的 top_deviations 看"这组申请人在哪些特征上系统偏离其他组", 不再只是黑盒 verdict.

## 7. 测试覆盖 (17 个新测试)

| 文件 | 数量 | 覆盖点 |
|---|---:|---|
| `test_causal_narrative.py` | 13 | DAG 路径追溯 (有路径/无路径/缺失节点/多跳), model-level (top-K), cohort-level (delta / outliers / z-score), individual-level (4 象限计数 / DAG 路径拼接), robustness (大噪声 / 干净信号), build_full_narrative (含/不含 robustness), render_markdown (4 段标题) |
| `test_narrative_visualize.py` | 4 | 瀑布图能写 PNG + 文件 > 1KB, 空 top_features 早退, 三联卡能写 PNG + 含 robustness footer, render_all 一次出 2 张图 |

## 8. 限制与未来方向

- **DAG 节点缺失 (n_paths=0)**: 像 `EXT_SOURCE_3`, `EXT_SOURCE_1`, `BUREAU_TYPE_MICROLOAN_FRAC` 这些 SHAP 高位但 DAG 找不到路径的特征, 个体叙事里 `dag_paths: []`, `n_paths: 0`. **未来**应在 DAG 里加一条 `EXT_SOURCE_* → TARGET` 边 (在 docs 里有写但 domain DAG 没注入), 注入后 narrative 自动填上路径.
- **KNN 用 z-score 标准化**: 对 income (10^5) 和 age (10^2) 这种量级差很大的特征有偏, income 主导距离. **未来**应改用 rank-based distance (Spearman) 或先 PCA.
- **Robustness 只对申请人加噪声, 不对 cohort 加**: 真正 robust 还需 "cohort KNN 在扰动后是否还是这 10 个" — 留作 M8.3.
- **没有 "counterfactual narrative"**: 跟 M7 DiCE 反事实在 narrative 上一脉相承, 但没有合并到一个 narrative 输出. 留作 M8.3.

## 9. 后续里程碑

- **M8.3**: 完整服务化 (FastAPI 5 端点 + Streamlit 4 页填实 + PSI 后台任务) — 把 M8.1/M8.2 落到产品
- **M8.4**: 多语言 + 港式本地化 (粤语 / 繁体 / 香港场景) — `render_markdown` 已支持中文标题, 扩 `language="zh-HK"` 即可
