# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CausalCredit — a causal-inference enhanced credit scoring system built for the BOCHK 创新先驱大赛 2026. The system layers causal discovery, CATE (heterogeneous treatment effect) estimation, and counterfactual reasoning on top of a base ML scorer. End-to-end it answers not just "what is the default risk?" but "why is it high, and what would change it?"

The full Chinese-language design rationale is in `docs/` (see `CausalCredit_完整实现计划书.md`, `CausalCredit_因果推理验证标准体系.md`, `CausalCredit_反欺诈能力覆盖分析.md`). The current state of play, including what is implemented vs. skeleton, is in `PROGRESS.md` — read that first before assuming any module works.

## Common commands

All build / lint / test / run entry points are in the `Makefile`:

- `make install` — `pip install -e ".[dev]"` (full dep set including DoWhy, EconML, LightGBM, SHAP, DiCE, FastAPI, Streamlit)
- `make test` — `pytest tests/ -v --cov=src --cov-report=term-missing`
- `make lint` / `make format` — ruff (line-length 100, py311), black, isort
- `make run-api` — FastAPI on `0.0.0.0:8000`
- `make run-demo` — Streamlit on `:8501`
- `make clean` — caches only (does NOT touch `output/` or trained models)
- `docker-compose up --build` — API `:8000` + frontend `:8501`

The end-to-end pipeline (the only fully working entry point right now):

```bash
python -m src.run_pipeline    # ~33s on CPU, 12 steps, German Credit (1000 rows)
```

Five PNGs land in `output/figures/`: ROC, feature importance, ATE forest plot, calibration curve, confusion matrix.

## Architecture

The pipeline is a strictly linear assembly — `run_pipeline.py` calls one module per stage and nothing is wired up implicitly:

```
GermanCreditLoader        src/data/loader.py            sklearn fetch_openml("credit-g")
    └─> DataCleaner       src/data/preprocessing/       median impute + 1%/99% Winsorize
    └─> FeatureBuilder    src/features/builder.py       LabelEncoder + StandardScaler
        └─> CausalFeatureBuilder src/features/causal_features.py   5 hand-built ratio features
    └─> GBTrainer         src/models/train.py           sklearn GradientBoosting + 5-fold CV
    └─> ModelEvaluator    src/models/evaluate.py        AUC/Acc/Precision/Recall/F1/LogLoss
    └─> CreditCausalGraph src/causal/graph.py           hard-coded DAG (see below)
    └─> CausalVariableValidator src/causal/variable_validation.py
    └─> CausalEffectEstimator src/causal/estimate.py    manual PSM + NearestNeighbors + 200 bootstrap CI
```

### Causal DAG (the most important domain object)

`src/causal/graph.py` defines `CreditCausalGraph` — a hand-coded DAG used for the entire causal analysis:

- **Treatments** (2): `credit_amount`, `duration`
- **Outcome** (1): `class` (renamed to `default` in some places)
- **Confounders** (8): `age`, `job`, `housing`, `savings_status`, `checking_status`, `employment`, `credit_history`, `purpose`
- **Mediator** (1): `installment_commitment`

The graph exposes `get_treatment_variables`, `get_outcome_variable`, `get_confounders(tx, out)`, `get_mediators(tx, out)`, `get_instruments(tx)`, `validate_acyclic()` (DFS), and `get_dot_string()` for Graphviz export. Acyclicity must hold — if you add a node/edge, re-run `validate_acyclic()`. Confounders are computed as nodes that are ancestors of *both* treatment and outcome, so adding a new backdoor path silently changes which variables the PSM uses for adjustment.

### ATE estimation (currently the only causal method actually implemented)

`src/causal/estimate.py` implements ATE with **no DoWhy/EconML dependency** — pure sklearn + numpy. The flow:

1. `binarize_treatment` — median split on continuous treatments
2. `estimate_propensity_scores` — `LogisticRegression(max_iter=5000)`
3. `propensity_score_matching` — `NearestNeighbors` with optional caliper (0.25 std default)
4. `compute_ate_with_bootstrap` — 200 resamples, percentile CI

Use `CausalEffectEstimator.estimate_ate(...)` or `estimate_all_treatments(...)`. The pipeline calls it twice: for `credit_amount -> default` and `duration -> default`. With German Credit (n=1000), `duration -> default` ATE is +0.149 [0.058, 0.204] p<0.05; `credit_amount` is not significant.

### What is still a skeleton

Per `PROGRESS.md` (the source of truth for status), these are placeholders that compile but do not work end-to-end — do not wire them into a path the user actually invokes without first completing the P0/P1 work listed there:

- `src/causal/cate.py`, `src/causal/refute.py` — EconML CATE + DoWhy refutation methods (TODO bodies)
- `src/explain/*.py` — SHAP / DiCE counterfactual / decision / evidence (all stubs)
- `src/api/services.py` + `src/api/routes.py` — route handlers are `...`; only `/api/v1/health` returns
- `src/frontend/app.py` + `src/frontend/pages/*` — Streamlit navigation renders but pages show `st.info` placeholders
- `src/models/calibrate.py` — Isotonic calibration stub
- `src/features/aggregation.py`, `src/features/pipelines/temporal_features.py` — Home Credit multi-table aggregation not yet implemented
- `src/monitoring/drift_detector.py` — PSI drift detection stub

The codebase currently uses `GradientBoostingClassifier` (sklearn) for the working pipeline. `pyproject.toml` lists `lightgbm>=4.0` and `dowhy>=0.11` etc. for the planned GPU environment, but switching `GBTrainer` to LightGBM and `CausalEffectEstimator` to EconML/DoWhy is tracked as a P0 task in `PROGRESS.md` — not done.

## Configuration and data

- `configs/config.yaml` is loaded by `src/utils/config.py::load_config()` (no env var injection; defaults to repo path). The YAML defines data dirs, LightGBM params (for the future switch), Optuna trials, API host/port, frontend URL. Note the config has LightGBM hyperparameters but the actual trainer ignores them — see "skeleton" note above.
- `.env.example` lists `HOME_CREDIT_DATA_DIR`, `LENDING_CLUB_DATA_DIR`, `KAGGLE_USERNAME/KEY` etc. for the Home Credit / Lending Club datasets; the working pipeline does NOT need them.
- `tests/fixtures/` is currently empty (only `__init__.py`). `tests/conftest.py` provides a `sample_config` dict fixture. When writing tests, add fixture data here.

## Conventions

- Ruff + black + isort all enforce line-length=100, target py311 (configs in `pyproject.toml`).
- `.pre-commit-config.yaml` runs ruff (with `--fix`) and basic whitespace/YAML/TOML checks.
- `src/run_pipeline.py` uses `matplotlib.use("Agg")` first thing — do not reorder imports or plots will fail in headless environments.
- Chinese strings are expected to render in figures; the script sets `font.sans-serif = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]` and `axes.unicode_minus = False` for matplotlib.
- Output figures use a 5-prefix naming convention (`01_roc_curve.png` … `05_confusion_matrix.png`) — keep it when adding new charts so they slot into README/docs in the right order.
- All paths in `configs/config.yaml` and `Dockerfile` are relative to repo root; running the pipeline from repo root is the safe assumption.
</content>
</invoke>