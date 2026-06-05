"""End-to-end refutation demo: synthetic DGP + Home Credit.

Per M1.3 acceptance criteria:
- Synthetic: 3 refuters pass; E-value >= 2
- Home Credit: 1+ treatment with 4 refutation results
- Output: output/demo_m1/refutation_results.png
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.causal.refute import (  # noqa: E402
    CausalRefuter,
    compute_e_value_from_ate,
)


def _print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def run_synthetic() -> dict:
    _print_header("Synthetic validation (known ATE=0.5)")
    rng = np.random.RandomState(0)
    n = 3000
    df = pd.DataFrame({
        "X": rng.normal(size=n),
        "Z": rng.normal(size=n),
    })
    df["T"] = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(df["X"] + df["Z"])))).astype(int)
    df["Y"] = 0.5 * df["T"] + 0.3 * df["X"] + 0.5 * rng.normal(size=n)

    from dowhy import CausalModel
    model = CausalModel(data=df, treatment="T", outcome="Y", common_causes=["X", "Z"])
    ident = model.identify_effect()
    est = model.estimate_effect(identified_estimand=ident, method_name="backdoor.linear_regression")
    print(f"  ATE = {est.value:.4f}")

    refuter = CausalRefuter(model, estimand=ident)
    results = refuter.run_all_refutations(est, num_simulations=20)
    for m, r in results.items():
        flag = "PASS" if r.get("passed") else "FAIL"
        # show the salient metric
        metric = (
            r.get("delta_ate", r.get("rel_change", r.get("cv", r.get("e_value", "?"))))
        )
        print(f"  {m:<22s}  {flag}  metric={metric}")
    score = refuter.compute_robustness_score(results)
    print(f"  robustness_score = {score:.2f}")

    out_path = "output/demo_m1/refutation_synthetic.png"
    refuter.visualize_refutations(results, output_path=out_path)
    print(f"  saved -> {out_path}")
    return results


def run_home_credit(sample_size: int = 10000) -> dict:
    _print_header("Home Credit (AMT_CREDIT -> TARGET)")
    from src.data.home_credit_loader import HomeCreditLoader
    from dowhy import CausalModel

    loader = HomeCreditLoader()
    df = loader.fetch().sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Binarize T at the median to fit DoWhy's binary-treatment assumption
    t_med = df["AMT_CREDIT"].median()
    df["T_high_credit"] = (df["AMT_CREDIT"] > t_med).astype(int)
    confounders = ["AMT_INCOME_TOTAL", "REGION_RATING_CLIENT", "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2"]
    confounders = [c for c in confounders if c in df.columns]
    df_use = df[["T_high_credit", "TARGET"] + confounders].dropna()
    print(f"  using n={len(df_use)}, confounders={confounders}, T_high_rate={df_use['T_high_credit'].mean():.3f}")

    model = CausalModel(
        data=df_use,
        treatment="T_high_credit",
        outcome="TARGET",
        common_causes=confounders,
    )
    ident = model.identify_effect()
    est = model.estimate_effect(identified_estimand=ident, method_name="backdoor.linear_regression")
    print(f"  ATE (high vs low credit) = {est.value:.4f}")

    refuter = CausalRefuter(model, estimand=ident)
    t0 = time.time()
    results = refuter.run_all_refutations(est, num_simulations=20)
    print(f"  ran 4 refuters in {time.time()-t0:.1f}s")
    for m, r in results.items():
        flag = "PASS" if r.get("passed") else "FAIL"
        metric = (
            r.get("delta_ate", r.get("rel_change", r.get("cv", r.get("e_value", "?"))))
        )
        print(f"  {m:<22s}  {flag}  metric={metric}")
    score = refuter.compute_robustness_score(results)
    print(f"  robustness_score = {score:.2f}")

    out_path = "output/demo_m1/refutation_results.png"
    refuter.visualize_refutations(results, output_path=out_path)
    print(f"  saved -> {out_path}")
    return results


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    Path("output/demo_m1").mkdir(parents=True, exist_ok=True)

    # Sanity: E-value formula
    print("E-value sanity:")
    print(f"  ATE=0.5, sd_y=1  -> E={compute_e_value_from_ate(0.5, sd_y=1.0):.2f}")
    print(f"  ATE=0.1, sd_y=1  -> E={compute_e_value_from_ate(0.1, sd_y=1.0):.2f}")
    print(f"  ATE=1.0, sd_y=1  -> E={compute_e_value_from_ate(1.0, sd_y=1.0):.2f}")

    syn = run_synthetic()
    hc = run_home_credit()

    _print_header("Summary")
    print(f"  Synthetic: 4 refuters ran on DGP with ATE=0.5")
    print(f"  Home Credit: 4 refuters ran on binarized AMT_CREDIT")
    print(f"  Output: output/demo_m1/refutation_results.png")
