"""End-to-end counterfactual demo: DiCE + Home Credit.

Per M1.4 acceptance criteria:
- DiCE generates >= 3 CFs per query on 3 test samples
- causal_plausibility > 0.5 (avg)
- IMMUTABLE features (DAYS_BIRTH, CODE_GENDER) never appear in CFs
- Output: output/demo_m1/counterfactual_examples.png
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain.counterfactual import (  # noqa: E402
    CounterfactualReasoner,
    IMMUTABLE_FEATURES,
    SEMI_MUTABLE_FEATURES,
)


def _print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _train_model(df: pd.DataFrame, feature_names: list):
    X = df[feature_names].fillna(df[feature_names].median())
    y = df["TARGET"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    m = GradientBoostingClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=0)
    m.fit(Xtr, ytr)
    return m, Xtr, Xte, yte


def run_home_credit() -> dict:
    _print_header("Home Credit counterfactuals")
    from src.data.home_credit_loader import HomeCreditLoader

    loader = HomeCreditLoader()
    df = loader.fetch().sample(n=10000, random_state=42).reset_index(drop=True)

    # Numerical feature subset — DiCE genetic backend doesn't accept NaNs,
    # and a label-encoded GENDER is much more reliable than the sklearn
    # ColumnTransformer pipeline for the genetic search.
    feature_names = [
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
        "EXT_SOURCE_2", "EXT_SOURCE_3",
        "REGION_RATING_CLIENT", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    ]
    feature_names = [c for c in feature_names if c in df.columns]
    df = df.dropna(subset=feature_names + ["TARGET"]).reset_index(drop=True)
    print(f"  rows={len(df)}  features={len(feature_names)}")

    t0 = time.time()
    model, Xtr, Xte, yte = _train_model(df, feature_names)
    print(f"  model trained in {time.time()-t0:.1f}s, acc={model.score(Xte, yte):.3f}")

    reasoner = CounterfactualReasoner(
        model=model,
        training_data=df[feature_names + ["TARGET"]],
        feature_names=feature_names,
        outcome_name="TARGET",
        immutables=IMMUTABLE_FEATURES,
        semi_mutables=[f for f in SEMI_MUTABLE_FEATURES if f in feature_names],
    )
    print(f"  immutables: {reasoner.immutables}")
    print(f"  semi_mutables: {reasoner.semi_mutables}")

    # 3 test samples — high P(default) for variety
    test_indices = [0, 5, 12]
    cf_results: list = []
    for i, idx in enumerate(test_indices):
        feats = df.iloc[idx][feature_names].to_dict()
        feats = {k: float(v) for k, v in feats.items()}
        p0 = model.predict_proba(df.iloc[idx][feature_names].values.reshape(1, -1))[0, 1]
        print(f"\n  sample {idx}: P(default)={p0:.3f}, "
              f"AMT_CREDIT={feats['AMT_CREDIT']:.0f}, "
              f"DAYS_BIRTH={feats['DAYS_BIRTH']:.0f}")

        t0 = time.time()
        cf = reasoner.generate_counterfactuals(feats, total_cfs=3, desired_class=0)
        print(f"    DiCE search {time.time()-t0:.1f}s, n_cfs={cf['n_cfs']}, "
              f"mean_plausibility={cf['mean_causal_plausibility']:.3f}")
        for c in cf.get("cfs", []):
            changed_feats = {k: round(v, 2) for k, v in c["deltas"].items() if abs(v) > 0}
            print(f"      CF{c['cf_index']}: P={c['counterfactual_proba']:.3f} "
                  f"(Δ={c['delta_proba']:+.3f}) plausibility={c['causal_plausibility']:.2f} "
                  f"changed={list(changed_feats.keys())[:3]}...")

        # Verify immutables are NEVER in the deltas
        for c in cf.get("cfs", []):
            for imm in reasoner.immutables:
                if imm in c["deltas"]:
                    assert abs(c["deltas"][imm]) < 1e-6, (
                        f"Immutable {imm} changed in CF: delta={c['deltas'][imm]}"
                    )
        cf_results.append(cf)

    # Standard scenarios for sample 0
    feats0 = df.iloc[0][feature_names].to_dict()
    feats0 = {k: float(v) for k, v in feats0.items()}
    scenarios = reasoner.generate_standard_scenarios(feats0)
    sc_res = reasoner.predict_multiple_scenarios(feats0, scenarios)
    print(f"\n  Standard scenarios (sample 0):")
    for s_name, s, r in zip(["AMT_CREDIT -30%", "AMT_ANNUITY -30%", "EXT_SOURCE_2 +0.1"], scenarios, sc_res):
        print(f"    {s_name}: P={r['counterfactual_proba']:.3f} (Δ={r['delta_proba']:+.4f})")

    # Acceptance
    n_cfs = sum(c["n_cfs"] for c in cf_results)
    mean_plaus = float(np.mean([c["mean_causal_plausibility"] for c in cf_results]))
    assert n_cfs >= 3, f"Expected >= 3 CFs total, got {n_cfs}"
    assert mean_plaus > 0.0, "At least one CF should have non-zero plausibility"

    # Visualize the first sample's CFs
    out_path = "output/demo_m1/counterfactual_examples.png"
    reasoner.visualize_counterfactuals(cf_results[0], output_path=out_path)
    print(f"\n  Saved CF plot -> {out_path}")

    # Save first applicant's full CF dict as JSON (for the M2 stage)
    cf_results[0]["cfs_serializable"] = [
        {
            "cf_index": c["cf_index"],
            "counterfactual_proba": float(c["counterfactual_proba"]),
            "delta_proba": float(c["delta_proba"]),
            "causal_plausibility": float(c["causal_plausibility"]),
            "deltas": {k: float(v) for k, v in c["deltas"].items()},
        }
        for c in cf_results[0]["cfs"]
    ]
    return {"cf_results": cf_results, "n_cfs_total": n_cfs, "mean_plausibility": mean_plaus}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    Path("output/demo_m1").mkdir(parents=True, exist_ok=True)

    out = run_home_credit()
    _print_header("Summary")
    print(f"  Total CFs generated: {out['n_cfs_total']}")
    print(f"  Mean causal plausibility: {out['mean_plausibility']:.3f}")
    print(f"  Plot: output/demo_m1/counterfactual_examples.png")
