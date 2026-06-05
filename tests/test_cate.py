"""End-to-end CATE demo: synthetic validation + Home Credit application.

Per the M1.2 acceptance criteria:
- Synthetic: 3 methods Spearman ρ > 0.70
- Home Credit: 3 methods Spearman ρ > 0.50
- Output: output/demo_m1/cate_distribution.png
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.causal.cate import (  # noqa: E402
    CATEEstimator,
    _is_binary_treatment,
    synthetic_cate_validation,
)
from src.data.home_credit_loader import HomeCreditLoader  # noqa: E402


def _print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def run_synthetic() -> dict:
    _print_header("Synthetic validation")
    Y, T, X, W, true_cate = synthetic_cate_validation(n=5000, seed=0)
    print(f"  Y shape={Y.shape}  T unique={np.unique(T)[:5]}  binary T? {_is_binary_treatment(T)}")
    est = CATEEstimator({"random_state": 0, "cf_n_estimators": 200, "cv": 2})
    t0 = time.time()
    out = est.cross_validate_methods(Y, T, X, W)
    print(f"  fit 3 methods in {time.time()-t0:.1f}s")
    print(f"  Spearman matrix:\n{out['spearman'].round(3)}")
    print(f"  mean_abs_spearman = {out['mean_abs_spearman']:.3f}")
    for name, cate in out["cate"].items():
        rho, _ = spearmanr(cate, true_cate)
        print(f"  vs true CATE  {name:<22s}  rho={rho:.3f}")
    # Acceptance
    assert out["mean_abs_spearman"] > 0.70, (
        f"Synthetic CATE mean_abs_spearman {out['mean_abs_spearman']:.3f} <= 0.70 threshold"
    )
    return out


def run_home_credit(sample_size: int = 30000) -> dict:
    _print_header("Home Credit application")
    loader = HomeCreditLoader()
    df = loader.fetch()
    print(f"  full shape: {df.shape}  default rate: {df['TARGET'].mean():.4f}")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    y = df_sample["TARGET"].astype(float).values
    t_raw = df_sample["AMT_CREDIT"].astype(float).values
    # Rescale T so effect sizes are interpretable (in units of "per $1k credit")
    t = t_raw / 1000.0

    Wcols = ["AMT_INCOME_TOTAL", "REGION_RATING_CLIENT", "DAYS_BIRTH",
             "DAYS_EMPLOYED", "EXT_SOURCE_2", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    Wcols = [c for c in Wcols if c in df_sample.columns]
    W = df_sample[Wcols].fillna(df_sample[Wcols].median()).values

    Xcols = ["AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "DAYS_BIRTH", "EXT_SOURCE_2", "DAYS_EMPLOYED"]
    Xcols = [c for c in Xcols if c in df_sample.columns]
    X = df_sample[Xcols].fillna(df_sample[Xcols].median()).values
    X = StandardScaler().fit_transform(X)
    print(f"  T rescaled by /1000, n={len(y)}, W cols={len(Wcols)}, X cols={len(Xcols)}")

    est = CATEEstimator({"random_state": 0, "cf_n_estimators": 300, "cv": 2})
    t0 = time.time()
    out = est.cross_validate_methods(y, t, X, W)
    print(f"  fit 3 methods in {time.time()-t0:.1f}s")
    print(f"  Spearman matrix:\n{out['spearman'].round(3)}")
    print(f"  mean_abs_spearman = {out['mean_abs_spearman']:.3f}")
    print(f"  ATE per method (per $1k credit): { {k: round(v, 4) for k, v in out['ate'].items()} }")

    # Subgroup analysis
    sub_defs = {
        "young (<35y)": df_sample["DAYS_BIRTH"].values < -35 * 365,
        "mid (35-50y)": (df_sample["DAYS_BIRTH"].values >= -35 * 365) & (df_sample["DAYS_BIRTH"].values < -50 * 365),
        "old (>=50y)": df_sample["DAYS_BIRTH"].values >= -50 * 365,
        "low_ext (<0.3)": df_sample["EXT_SOURCE_2"].fillna(0.5).values < 0.3,
        "high_ext (>=0.6)": df_sample["EXT_SOURCE_2"].fillna(0.5).values >= 0.6,
    }
    cf = out["cate"]["CausalForestDML"]
    subgroup_df = est.cate_subgroup_analysis(cf, df_sample[Xcols].fillna(0), sub_defs)
    print("\n  CATE by subgroup (CausalForestDML):")
    print(subgroup_df.round(4).to_string(index=False))

    # Feature importance from CausalForestDML
    importance = est.cate_feature_importance(out["models"]["CausalForestDML"], Xcols)
    print("\n  Feature importance (CausalForestDML):")
    print(importance.round(4).head(10).to_string(index=False))

    # Acceptance: ρ > 0.50
    assert out["mean_abs_spearman"] > 0.50, (
        f"Home Credit CATE mean_abs_spearman {out['mean_abs_spearman']:.3f} <= 0.50 threshold"
    )

    # Save visualization
    out_path = "output/demo_m1/cate_distribution.png"
    est.visualize_cate(out["cate"], subgroup_df=subgroup_df, output_path=out_path)
    print(f"\n  Saved CATE distribution plot -> {out_path}")
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)

    Path("output/demo_m1").mkdir(parents=True, exist_ok=True)

    syn = run_synthetic()
    hc = run_home_credit()

    _print_header("Summary")
    print(f"  Synthetic  mean_abs_spearman = {syn['mean_abs_spearman']:.3f}  (threshold 0.70)  PASS")
    print(f"  Home Credit mean_abs_spearman = {hc['mean_abs_spearman']:.3f}  (threshold 0.50)  PASS")
    print("\n  CATE demo complete: output/demo_m1/cate_distribution.png")
