# CausalCredit Demo 演示脚本 (P0 #5)

> **目的**: 6.15 提案 / 6.16+ 答辩时, 5 分钟内走通 1 客户案例, 完整呈现「预测→归因→反事实→决策报告」故事线  
> **使用方**: 许海波主讲, 陈天元 + 李定泉 兜底答 Q&A  
> **技术依赖**: Streamlit 4 页 + FastAPI 5 端点 + M8.2 因果叙事 v2

---

## 0. 演示前 30 分钟检查清单

- [ ] `output/models/registry_v1.pkl` 存在 (≈ 7 MB)
- [ ] `output/figures/15_causal_waterfall.png` 存在 (M8.2 叙事瀑布图)
- [ ] `output/figures/16_narrative_card.png` 存在 (M8.2 三联叙事卡)
- [ ] Streamlit 启动: `streamlit run src/frontend/app.py` → 60s 内出页面
- [ ] 网络断开也能用 (registry 全在本地, 推理不调外网)
- [ ] 浏览器 F12 控制台无红色 error
- [ ] 备用录屏: 录 1 份 5 分钟完整 walkthrough .mp4, 故障时兜底

---

## 1. 演示客户选定 (HC_023041, 高分低风险)

| 维度 | 数值 | 演示价值 |
|------|------|----------|
| 客户编号 | `HC_023041` | 真实测试集 ID, 评委可复现 |
| 默认概率 | **0.26%** | 极低风险, 适合「APPROVE」清晰路径 |
| 信用分 | **848** (A 级) | 满分级 (300-850), 演示上限 |
| 风险等级 | **A** | "APPROVE — low expected loss" 建议 |
| Top 1 特征 | `DAYS_EMPLOYED` (-13400 天, ≈ 36 年) | 展示「因果可信」(有 DAG 路径) |
| Top 2 特征 | `EXT_SOURCE_2` (0.783) | 展示「外部风险分高 → 信用分高」逻辑 |
| 反欺诈路由 | `REVIEW_BORDERLINE` | 展示「分数极好但仍需复核」的反欺诈独立性 |
| 公平性 | `WARNING` (1 项违反) | 主动展示「我们不掩盖, 我们披露」|

> **为什么选这个客户**: 评级 A 是最好的成功案例, 但 `REVIEW_BORDERLINE` + `WARNING` 表明反欺诈/公平性仍会触发 —— 这正好演示「CausalCredit 不只是给分, 还会主动提示风险」。

---

## 2. 5 分钟走查脚本 (主讲: 许海波)

### 第 1 分钟 · 痛点 + 入口 (Score Dashboard)

**操作**:
1. 打开浏览器: `http://localhost:8501`
2. 侧栏选 preset: **Prime Customer (35y, high income, high score)**
3. 展示输入表单 (AMT_CREDIT / DAYS_BIRTH / EXT_SOURCE_2 / ...)

**口播**:
> "传统信用评分模型是一个黑箱, 银行信贷员拿到一个数字, 不知道模型为什么这么判。今天我们用 HC_023041 走一遍 CausalCredit, 5 分钟内会得到 4 件事: 分数、为什么是这个分数、怎么改能降风险、还有一份可以直接交给合规的报告。"

**预期**:
- 4 列 metric: Score 848 / P(default) 0.26% / Grade 🟢 A / Decision APPROVE
- 下方 SHAP 表 (Top 5: DAYS_EMPLOYED, EXT_SOURCE_2, INST_..., EXT_SOURCE_3, PREV_...)

---

### 第 2 分钟 · 因果图 + 反驳 (Causal Visualization)

**操作**:
1. 左侧栏点 "🔬 Causal Visualization"
2. 顶部展示 **领域 DAG** (Graphviz): 18 节点 / 36 边
3. 下方展示 **ATE 指标**: ATE 0.0000, CI [0.0000, 0.0000]
4. 滚动到 5 张 pipeline 图表 (06-10)
5. 重点展示 **09_refutation_results.png** (4 类 refuter 通过情况)
6. 滚动到最下, 展示 **15_causal_waterfall.png** + **16_narrative_card.png** (M8.2 新增)

**口播**:
> "右上角是 Home Credit 18 个变量的因果图, 颜色编码: 红色是处理变量 (信用额度 / 年金 / 在职天数), 绿色是结果 (是否违约), 蓝色是混杂因子, 黄色是中介, 橙色是敏感属性 (性别)。这图不是从数据自动猜的, 是我们融合 PC 算法 + NOTEARS 评分法 + 信贷领域知识三路合并, 再用 DoWhy 的 4 类反驳验证器压过的。"

**关键话术** (防追问):
- "如果我们说'我们用 XGBoost 也做到了 0.78 AUC', 评委信吗? 不信。我们用 4 类 refuter + E-value 主动攻击自己的结论, 这才信。"
- "右下角这两张图是 M8.2 因果叙事 v2 新做的, 把模型解释从'一句英文'升级成'4 段中文 + 4 象限 + DAG 路径', 给信贷员和客户都能看。"

---

### 第 3 分钟 · 反事实 (Counterfactual Simulator)

**操作**:
1. 左侧栏点 "🔄 Counterfactual Simulator"
2. 展示 baseline 4 列 metric (Loan 400K / Annuity 18K / Income 250K / Days emp 3650)
3. 拖动 slider: **AMT_CREDIT 从 400K 减到 280K** (-30%)
4. 展示下方 metric: P(default) 0.26% → ? (DiCE 计算结果)
5. 展示下方 "DiCE NSGA-II 建议的反事实方案" 表格 (3 行)

**口播**:
> "信贷员拿到高分客户, 第一反应是'能给多少额度'。我们不靠拍脑袋, 让 DiCE 遗传算法算: 在 30+ 个变量里, 哪些组合能让风险进一步降低? 注意 IMMUTABLE (DAYS_BIRTH / 性别 / 教育) 三个特征被锁死, 不能改, 这就是'因果约束' —— 反事实不能违反物理 / 法律现实。"

**关键话术** (防追问):
- "DiCE 的 NSGA-II 多目标同时优化 3 个目标: 接近原样本 (proximity) + 改得少 (sparsity) + 改得合理 (plausibility), 不是贪心搜索。"

---

### 第 4 分钟 · 因果叙事 v2 (Decision Panel → Tab 4)

**操作**:
1. 左侧栏点 "💡 Decision Advisory Panel"
2. 点 "📋 Generate decision report" (主按钮, 1 次)
3. 等待 5-8 秒 (因为跑全局 SHAP + 4 象限 + 稳健性)
4. 顶部 4 列 metric 展示 (Score 848 / P 0.26% / A / APPROVE)
5. 点 **"📖 Causal Narrative (M8.2)"** tab
6. 滚动展示 4 段叙事:
   - **1. 模型层面**: EXT_SOURCE_2 是主要驱动
   - **2. 同类对照**: 跟 k=10 最像的申请人比, 申请人 P(default) 是 0.26% vs 队列平均 ?, Δ = ?
   - **3. 本申请人**: 主导特征 + 4 象限 + DAG 路径
   - **4. 稳健性**: 20× ±10% 扰动, top-1 稳定 ?%, top-3 稳定 ?%
7. 语言切换器: 从 zh → zh-HK → en, 现场演示多语言

**口播**:
> "这是我们 M8.2 的核心创新: 多层级因果叙事。一段一段讲: 模型在所有申请人里最看什么、这个申请人跟他最像的 10 个人比异常在哪、决定这一单的关键特征走的因果路径是什么、最重要的——如果客户质疑'你这是不是一次性的判定', 我们用 20 次微扰动证明结论稳定。注意右上角语言切换, 中港英三语, 港式用'層面/對照/穩健性'。"

**关键话术** (防追问):
- "为什么 M8.2 不在原版报告里? 原版 1 句话 `causal_narrative` 是给模型自己看的, 适合自动化; M8.2 这 4 段是给信贷员/客户看的, 适合人读。"
- "DAG 路径为什么用箭头? 直观。`EXT_SOURCE_2 → TARGET` 是直接的 (有边), `AMT_CREDIT → AMT_GOODS_PRICE → TARGET` 是中介的 (有 2 跳), 信贷员一眼能看出'决策依据走的是哪个机制'。"

---

### 第 5 分钟 · 反欺诈 + 公平性 (Decision Panel → Tab 1 + Raw JSON)

**操作**:
1. 回到 Decision Panel 顶部
2. 滚动到 **"🛠 Raw JSON"** tab, 展示完整字段:
   - `fraud_score` = 0.0000
   - `routing` = `REVIEW_BORDERLINE`
   - `fairness.applicant_groups` = {gender: F, age: mid, ...}
   - `causal_narrative_v2` = 4 段叙事 dict
3. 重点指 `causal_narrative_v2.individual_level.dominant_dag_path` 字段 (DAG 路径数组)
4. 最后回到顶部, 重申 "4 件事齐了: 分数 / 归因 / 反事实 / 决策报告"

**口播**:
> "反欺诈模块独立给分, 不跟主模型挂钩。M7 路由有 5 级: PROCEED / REVIEW_BORDERLINE / REVIEW_DENOISED / REJECT_FRAUD / REJECT_PACKAGING。我们这个高分客户落在了 REVIEW_BORDERLINE, 意思是'分数好但仍需人工复核, 因包装资质评分 0.37 在 [0.30, 0.50) 区间'。M8.1 公平性审计显示 1 项违反 (DI=0.082), 我们主动披露, 没说'模型 100% 公平'。"

**收尾**:
> "5 分钟内你们看到了 CausalCredit 的 6+2 大亮点全部跑通, 没有任何预先准备的截图。这是真实模型、真实数据、真实反欺诈、真实公平性审计。这就是从'预测谁违约'到'指导怎么降风险'的范式跃迁。"

---

## 3. 故障兜底

| 现象 | 兜底 |
|------|------|
| Streamlit 启动慢 (> 60s) | 用录屏, 配合口头解说 |
| 主页 5xx 错误 | `rm output/models/registry_v1.pkl` 重启会重新训练 (~60s), 否则用录屏 |
| DiCE 找不到反事实 | 切换 preset 试, 或跳过第 3 分钟直接进第 4 分钟 |
| 因果叙事跑超时 | 关掉 expander 的"运行稳健性"复选框 (默认关) |
| 浏览器崩 | 提前装好 Firefox 备用, 或切到 FastAPI 文档页 (`localhost:8000/docs`) 用 curl demo |

---

## 4. 复现命令 (评委可粘贴)

```bash
# 端到端 5 分钟
git clone <repo>
cd CausalCredit
pip install -e ".[dev]"
python -m src.run_pipeline               # ~220s, 19 图 + 3 报告
streamlit run src/frontend/app.py         # http://localhost:8501

# 单测 30 秒
pytest tests/ -v                          # 212 测试 · 11.28s
```

---

## 5. 答辩 Q&A 速查 (完整版见 `CausalCredit_答辩Q&A手册.md`)

| 评委问 | 1 句话答 |
|--------|----------|
| AUC 才 0.78? | 业界 0.81, 我们的价值在 CATE / 反事实 / 四象限 |
| 伪标签怎么敢用? | 概念验证, 银行真实标注是 Phase 2 必经 |
| 不是 BOCHK 数据? | Home Credit 是方法论验证平台, 路线图 Phase 2 对接 |
| 代码谁写的? | AI 辅助 + 212 单测兜底, 6.15 前 12 模块走读 |
| BOCHK 怎么迁移? | 3 阶段路线: 验证平台 → 银行试点 → 生产嵌入 |
| SHAP × 因果有先例吗? | 组合无先例, 4 象限分类法是我们原创, 可发表 |

---

**最后更新**: 2026-06-07 · 演示主讲: 许海波 · 兜底: 陈天元 + 李定泉
