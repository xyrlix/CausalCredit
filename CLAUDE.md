# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CausalCredit — a causal-inference enhanced credit scoring system built for the BOCHK 创新先驱大赛 2026. The system layers causal discovery, CATE (heterogeneous treatment effect) estimation, counterfactual reasoning, and a three-pillar anti-fraud stack on top of a base ML scorer. End-to-end it answers not just "what is the default risk?" but "why is it high, what would change it, and is it fraud?"

The full Chinese-language design rationale is in `docs/` (see `CausalCredit_完整实现计划书.md`, `CausalCredit_因果推理验证标准体系.md`, `CausalCredit_反欺诈能力覆盖分析.md`). The current state of play, including what is implemented vs. skeleton, is in `PROGRESS.md` — read that first before assuming any module works. Per-step benchmarks and routing distributions are in `BENCHMARKS.md`.

## Common commands

All build / lint / test / run entry points are in the `Makefile`:

- `make install` — `pip install -e ".[dev]"` (full dep set: DoWhy, EconML, LightGBM, SHAP, DiCE, FastAPI, Streamlit, Optuna, shap, econml, dowhy, dice_ml, causal-learn)
- `make test` — `pytest tests/ -v --cov=src --cov-report=term-missing`
- `make lint` / `make format` — ruff (line-length 100, py311), black, isort
- `make run-api` — FastAPI on `0.0.0.0:8000`
- `make run-demo` — Streamlit on `:8501`
- `make clean` — caches only (does NOT touch `output/` or trained models)
- `docker-compose up --build` — API `:8000` + frontend `:8501`

The end-to-end pipeline (15 steps, fully working):

```bash
python -m src.run_pipeline    # ~212s on CPU, 15 steps, Home Credit 307K rows
```

**17 PNGs** land in `output/figures/` (11 from M0–M6 + 3 from M7 anti-fraud: `12_fraud_score_routing.png`, `13_packaging_scatter.png`, `14_denoising_effect.png` + 3 from M8.1 fairness: `12_fairness_group_rates.png`, `13_fairness_metric_gaps.png`, `14_fairness_status.png`).

## Architecture

The pipeline is a strictly linear 14-step assembly — `run_pipeline.py` calls one module per stage and nothing is wired up implicitly:

```
HomeCreditLoader          src/data/home_credit_loader.py     307,511 × 122 CSV
    └─> DataValidator     src/data/validator.py              schema/range checks
    └─> DataCleaner       src/data/preprocessing/            median impute + 1%/99% Winsorize
    └─> MultiTableAggregator  src/data/aggregator.py         5 secondary tables → 246 features
    └─> FeatureBuilder    src/features/builder.py            LabelEncoder + select
        └─> CausalFeatureBuilder src/features/causal_features.py    5 hand-built ratio features
    └─> LGBMFeaturePruning  src/run_pipeline.py STEP 5.5    drop zero-gain features
    └─> LightGBMTrainer   src/models/train.py               500 trees + 3-fold CV + GPU/Optuna opt-in
    └─> ModelEvaluator    src/models/evaluate.py             AUC/Acc/F1 + IsotonicCalibrator
    └─> CausalDiscovery   src/causal/discovery.py            PC + NOTEARS + domain-knowledge fusion
    └─> DoWhy ATE         src/causal/estimate.py             CausalModel + 4 refuters + E-value
    └─> CATE Estimator    src/causal/cate.py                 LinearDML + SparseLinearDML + CausalForestDML
    └─> SHAPExplainer     src/explain/shap_explain.py        TreeSHAP + 4-quadrant labels
    └─> CounterfactualReasoner src/explain/counterfactual.py DiCE NSGA-II + immutable/semi-mutable masks
    └─> FraudGuard        src/fraud/pipeline.py              3-class sub-classifier + packaging + denoising
```

### Causal DAG (the most important domain object)

Two domain DAGs coexist:

- `src/causal/graph.py::CreditCausalGraph` — German Credit, 2 treatments (`credit_amount`, `duration`), 1 outcome (`class`), 8 confounders, 1 mediator.
- `src/causal/home_credit_graph.py::HomeCreditCausalGraph` — Home Credit, 3 treatments (`AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_EMPLOYED`), 1 outcome (`TARGET`), 8 confounders, 2 mediators, 1 sensitive attribute (`CODE_GENDER`).

Both expose `get_treatment_variables`, `get_outcome_variable`, `get_confounders(tx, out)`, `get_mediators(tx, out)`, `get_instruments(tx)`, `validate_acyclic()` (DFS), and `get_dot_string()`. **Acyclicity must hold** — if you add a node/edge, re-run `validate_acyclic()`. Confounders are computed as nodes that are ancestors of *both* treatment and outcome, so adding a new backdoor path silently changes which variables the downstream causal modules (PSM, DML, refuters) use for adjustment.

### ATE / CATE / Refutation (all working end-to-end)

- `src/causal/estimate.py` — ATE via DoWhy `CausalModel.estimate_effect()` + 4 refuters (`placebo_treatment`, `random_common_cause`, `data_subset`, `e_value`) + manual PSM as fallback.
- `src/causal/cate.py` — `CATEEstimator` wraps 3 EconML methods: `LinearDML`, `SparseLinearDML`, `CausalForestDML`. Cross-method agreement (mean |Spearman|) is reported as `mean_abs_spearman` in the pipeline summary.
- `src/causal/refute.py` — `CausalRefuter` exposes the 4 refuter methods + `compute_e_value` + `compute_robustness_score`.

### Anti-Fraud Three-Pack (M7, STEP 14)

`src/fraud/` contains 4 modules wired into `FraudGuard` (`src/fraud/pipeline.py`):

1. **`three_class.py::ThreeClassFraudClassifier`** — 4-class LightGBM (non_default + fraudulent / non_malicious / systemic). Pseudo-labels are constructed from business rules since no fraud ground-truth exists: `fraudulent` triggered by `INST__DPD_MAX >= 30` OR (high income z + low employment z) OR (low EXT_SOURCE_1 + high income z); `systemic` triggered by `ORGANIZATION_TYPE` matching decline-industry substrings.
2. **`packaging.py::PackagingDetector`** — `packaging_score = UNTRUSTED / (TRUSTED + UNTRUSTED)` over the applicant's top-25% |SHAP| features. Path integrity checks 3 domain-DAG chains (income→goods→credit→annuity, income→ext_score, age→employment→income). Per-applicant SHAP values are required for a non-uniform score; falls back to global quadrant labels if not provided.
3. **`denoising.py::CausalDenoisingScorer`** — `causal_consistency = sign(repayment_z) × sign(consumption_z)` mapped to [0, 1] (5 INST__ cols vs 4 CC_/POS_ cols). `inflation = clip((1-consistency) × 0.15 × 5, 0, 0.15)`. `denoised_P(default) = P(default) + inflation`.
4. **`pipeline.py::FraudGuard`** — orchestrator with 5-level routing (`REJECT_FRAUD` ≥ 0.10 / `REJECT_PACKAGING` ≥ 0.50 / `REVIEW_DENOISED` / `REVIEW_BORDERLINE` ≥ 0.30 / `PROCEED`). Wired into the pipeline at STEP 14; the 3 picked applicants get a `fraud` field injected into their decision JSON, and a 1K test subsample is batch-scored for charts. **Routing thresholds are now in `FraudGuardConfig` (M8.1d)** — loaded from `configs/config.yaml::fraud_guard` so the same code can run multiple products (cash loan vs student loan) with different thresholds.

### Fairness audit (M8.1, STEP 15)

`src/fairness/` contains 3 modules wired into the pipeline at STEP 15:

- **`metrics.py`** — 3 standard group-fairness metrics for binary classifiers, with HKMA / EU AI Act / EEOC thresholds baked in: `demographic_parity_gap` (max-min selection rate, < 0.05), `equal_opportunity_gap` (max-min TPR, < 0.05), `disparate_impact_ratio` (min/max, ≥ 0.80). `summarize_fairness` returns a `FairnessSummary` with a status of `FAIR` / `WARNING` / `UNFAIR`.
- **`slicing.py`** — `SLICE_DEFINITIONS` for 4 default slices (`gender` from `CODE_GENDER`, `age_group` from `DAYS_BIRTH`, `income_group` from `AMT_INCOME_TOTAL`, `education_group` from `NAME_EDUCATION_TYPE`). `slice_dataset` returns a (n,) group-label array; missing/unknown values are mapped to `"UNKNOWN"` and **filtered out of all metric calculations** in `_filter_unknown_groups` (otherwise a few bad rows could flip a model from `FAIR` to `UNFAIR`).
- **`visualize.py`** — 3 PNGs: `12_fairness_group_rates.png` (per-slice grouped bars), `13_fairness_metric_gaps.png` (3-subplot with threshold lines), `14_fairness_status.png` (one bar per slice colored by status).

`DecisionAdvisor.build_fairness_block(features, X_test, y_test, y_pred_test, y_score_test)` is called per-applicant in STEP 15 and injects a `fairness` field into each decision report — applicant_groups, per-slice status, overall verdict, regulatory note. **Slicing uses the raw (unencoded) `df` columns, not the label-encoded `X_test`** — STEP 4's `LabelEncoder` would otherwise turn "M"/"F" into 0/1 and silently send everyone to UNKNOWN.

Routing drift monitoring (M8.1e) lives in `src/monitoring/drift_detector.py::detect_routing_drift` — same PSI algorithm as feature drift, but over the 5-level categorical distribution `PROCEED / REVIEW_BORDERLINE / REVIEW_DENOISED / REJECT_FRAUD / REJECT_PACKAGING`. STEP 15 compares the current batch's routing distribution to a hardcoded M7 baseline; PSI=0.001 in the test run means M7→M8.1 didn't shift routing behavior.

### What is still a skeleton

Per `PROGRESS.md` (the source of truth for status), these are placeholders that compile but do not work end-to-end — do not wire them into a path the user actually invokes:

- `src/api/services.py` + `src/api/routes.py` — route handlers are `...`; only `/api/v1/health` returns. **Stub since M3, deprioritized per user instruction (no K8s/ArgoCD/Celery/Redis).**
- `src/frontend/app.py` + `src/frontend/pages/*` — Streamlit navigation renders but pages show `st.info` placeholders. **Same status.**
- `src/models/calibrate.py` — Isotonic calibration stub (a minimal working version is inlined in `run_pipeline.py` STEP 7).
- `src/monitoring/drift_detector.py` — PSI drift detection stub (returns a 3-level flag but doesn't read from a stream).

Everything else listed in `docs/CausalCredit_完整实现计划书.md` §4.1–4.6 is implemented: discovery, CATE, refutation, counterfactual, SHAP four-quadrant, decision reports, **and now the anti-fraud three-pack**.

## Configuration and data

- `configs/config.yaml` is loaded by `src/utils/config.py::load_config()` (no env var injection; defaults to repo path). The YAML defines data dirs, LightGBM params (used by `LightGBMTrainer`), Optuna trials (gated by `optuna.enabled: false` by default), `device: "cpu"` (set to `cuda` to enable GPU build), API host/port, frontend URL.
- `.env.example` lists `HOME_CREDIT_DATA_DIR`, `LENDING_CLUB_DATA_DIR`, `KAGGLE_USERNAME/KEY` etc. for the Home Credit / Lending Club datasets; the working pipeline does NOT need them.
- `tests/fixtures/` is currently empty (only `__init__.py`). `tests/conftest.py` provides a `sample_config` dict fixture. When writing tests, add fixture data here.

## Conventions

- Ruff + black + isort all enforce line-length=100, target py311 (configs in `pyproject.toml`).
- `.pre-commit-config.yaml` runs ruff (with `--fix`) and basic whitespace/YAML/TOML checks.
- `src/run_pipeline.py` uses `matplotlib.use("Agg")` first thing — do not reorder imports or plots will fail in headless environments.
- Chinese strings are expected to render in figures; the script sets `font.sans-serif = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]` and `axes.unicode_minus = False` for matplotlib.
- Output figures use a 2-digit prefix naming convention (`01_roc_curve.png` … `14_denoising_effect.png` / `14_fairness_status.png`) — keep it when adding new charts so they slot into README/docs in the right order. M7 took 12–14 (anti-fraud); M8.1 also took 12–14 (fairness). The number-prefix namespacing is by milestone, not by chart; the README lists them in pipeline order.
- All paths in `configs/config.yaml` and `Dockerfile` are relative to repo root; running the pipeline from repo root is the safe assumption.
- When adding modules under `src/`, the package is recognized by the `src/` layout in `pyproject.toml`; tests import as `from src.X import Y`, not `from X import Y`.

## Test layout

164 tests across 24 files (~10s):

| Test file | Cases | What it covers |
|-----------|------:|----------------|
| `test_causal_discovery.py` | 5 | PC + NOTEARS + domain fusion on synthetic data |
| `test_cate.py` | 8 | 3 EconML methods, Spearman agreement, subgroup analysis |
| `test_refute.py` | 7 | 4 refuters + E-value + robustness score |
| `test_counterfactual.py` | 6 | DiCE NSGA-II + immutable/semi-mutable masks |
| `test_shap.py` | 6 | TreeSHAP + 4-quadrant + subgroup SHAP |
| `test_decision.py` | 5 | DecisionAdvisor + evidence chain |
| `test_decision_fairness.py` | 3 | `build_fairness_block` + bias detection (M8.1c) |
| `test_aggregation.py` | 16 | Bureau / prev / POS / INST / CC aggregators |
| `test_train.py` | 7 | LightGBM GPU/Optuna toggle, `_resolve_device` |
| `test_fraud_three_class.py` | 7 | Pseudo-labels + 4-class model + `fraud_score` |
| `test_fraud_packaging.py` | 7 | Calibration + 4-quadrant + path integrity |
| `test_fraud_denoising.py` | 6 | Consistency + inflation + denoised P |
| `test_fraud_pipeline.py` | 5 | FraudGuard end-to-end + 5-level routing |
| `test_fraud_config.py` | 7 | `FraudGuardConfig` + threshold override (M8.1d) |
| `test_fairness.py` | 11 | 3 metrics + 4 slices (M8.1a) |
| `test_fairness_visualize.py` | 4 | 3 fairness charts render (M8.1b) |
| `test_routing_drift.py` | 6 | Routing PSI (M8.1e) |
| Other (`test_loader`, `test_features`, `test_models`, `test_causal_graph`, `test_estimate`, `test_explain`, `test_drift`) | 47 | Earlier milestones |
