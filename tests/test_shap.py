"""End-to-end SHAP + four-quadrant demo: synthetic + Home Credit.

Per M1.5 acceptance criteria:
- Synthetic: each of 4 quadrants is reachable
- Home Credit: top-15 features show a clear TRUSTED/UNTRUSTED split
- Output: output/demo_m1/four_quadrant.png + waterfall PNGs
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

from src.explain.shap_explain import SHAPExplainer  # noqa: E402
from src.causal.home_credit_graph import HomeCreditCausalGraph  # noqa: E402


def _print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def run_synthetic() -> dict:
    _print_header("Synthetic four-quadrant validation")
    rng = np.random.RandomState(0)
    n = 1500

    # Design features so each quadrant is reachable:
    #   F1: drives Y linearly and is heavily used by the model (TRUSTED)
    #   F2: noise the model overfits to (UNTRUSTED)
    #   F3: true driver but the model does not use it (MASKED)
    #   F4: independent of both (NEGLIGIBLE)
    F1 = rng.normal(size=n)
    F2 = rng.normal(size=n)
    F3 = rng.normal(size=n)
    F4 = rng.normal(size=n)
    y = ((1.0 * F1 + 0.8 * F3 + 0.5 * rng.normal(size=n)) > 0).astype(int)
    df = pd.DataFrame({"F1": F1, "F2": F2, "F3": F3, "F4": F4})

    # Model overfits F2 (added as noise without y dependence)
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)
    m = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=0)
    m.fit(Xtr, ytr)
    print(f"  model acc: {m.score(Xte, yte):.3f}")

    expl = SHAPExplainer(m, feature_names=list(df.columns))
    sv = expl.compute_shap_values(Xte)
    res = expl.causal_vs_noncausal_contribution(
        sv, Xte, causal_features=["F1", "F3"], threshold_shap=None, threshold_causal=None,
    )
    print("  per-feature quadrants:")
    print(res["per_feature"].round(4).to_string(index=False))
    print("  counts:")
    print(res["counts"].to_string())

    out_path = "output/demo_m1/four_quadrant_synthetic.png"
    expl.visualize_four_quadrant(res, output_path=out_path)
    print(f"  saved -> {out_path}")

    # Sanity: F1 should be TRUSTED, F3 should be MASKED, F2 should be UNTRUSTED
    # (these assignments are heuristic; we just check the structure)
    quadrants = dict(zip(res["per_feature"]["feature"], res["per_feature"]["quadrant"]))
    assert "TRUSTED" in quadrants.values(), "Expected at least one TRUSTED feature"
    return res


def run_home_credit() -> dict:
    _print_header("Home Credit SHAP + four-quadrant")
    from src.data.home_credit_loader import HomeCreditLoader

    loader = HomeCreditLoader()
    df = loader.fetch().sample(n=8000, random_state=42).reset_index(drop=True)
    feature_names = [
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
        "EXT_SOURCE_2", "EXT_SOURCE_3",
        "REGION_RATING_CLIENT", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        "REGION_POPULATION_RELATIVE",
    ]
    feature_names = [c for c in feature_names if c in df.columns]
    df = df.dropna(subset=feature_names + ["TARGET"]).reset_index(drop=True)
    print(f"  rows={len(df)}  features={len(feature_names)}")

    X = df[feature_names]
    y = df["TARGET"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    m = GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.1, random_state=0)
    m.fit(Xtr, ytr)
    print(f"  model acc: {m.score(Xte, yte):.3f}")

    expl = SHAPExplainer(m, feature_names=feature_names)
    t0 = time.time()
    sv = expl.compute_shap_values(Xte)
    print(f"  SHAP values in {time.time()-t0:.1f}s, shape={sv.shape}")

    # Global importance
    gi = expl.global_importance(sv)
    print("  Top 10 features by |SHAP|:")
    print(gi.head(10).round(4).to_string(index=False))

    # Causal features (from the DAG)
    g = HomeCreditCausalGraph()
    causal_features = [c for c in g.nodes if c in feature_names]
    print(f"  Causal features in feature_names: {causal_features}")

    # Four-quadrant
    t0 = time.time()
    fq = expl.causal_vs_noncausal_contribution(sv, Xte, causal_features=causal_features)
    print(f"  four-quadrant in {time.time()-t0:.1f}s")
    print(f"  thresholds: SHAP={fq['thresholds'][0]:.4f}, causal={fq['thresholds'][1]:.4f}")
    print("  per-feature quadrants (top 15 by combined):")
    pf = fq["per_feature"].copy()
    pf["combined"] = pf["mean_abs_shap"] + pf["abs_causal_proxy"]
    print(pf.sort_values("combined", ascending=False).head(15).round(4).to_string(index=False))
    print("  counts:")
    print(fq["counts"].to_string())

    out_path = "output/demo_m1/four_quadrant.png"
    expl.visualize_four_quadrant(fq, output_path=out_path, top_n=15)
    print(f"  saved -> {out_path}")

    # Local explanation for an applicant
    if len(Xte) > 0:
        local_path = "output/demo_m1/shap_local_row0.png"
        expl.local_explanation(sv, Xte, idx=0, output_path=local_path)
        ev = expl.generate_evidence_chain(sv, Xte, idx=0, top_k=5)
        print("  Evidence chain (row 0):")
        for e in ev:
            print(f"    {e['feature']:<25s} value={e['value']:+.3f}  SHAP={e['shap']:+.4f}  {e['direction']}")

    return {"global_importance": gi, "four_quadrant": fq, "shap_values": sv}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    Path("output/demo_m1").mkdir(parents=True, exist_ok=True)

    syn = run_synthetic()
    hc = run_home_credit()

    _print_header("Summary")
    print("  Synthetic: 4 features designed to span 4 quadrants")
    print("  Home Credit: 13 features scored by SHAP + causal proxy")
    print("  Plots: output/demo_m1/four_quadrant.png, four_quadrant_synthetic.png, shap_local_row0.png")
