"""Demo: 因果发现 (PC + NOTEARS 融合 + 领域知识注入)

Run as a script:
    python tests/test_causal_discovery.py

Outputs 6 PNGs to output/demo_m1/:
    - 01_pc_dag.png            PC-only graph
    - 01_notears_dag.png       NOTEARS-only graph
    - 01_causal_discovery.png  Fused graph (PC ∪ NOTEARS) + domain overlay
    - 01_discovery_metrics.png Bar chart of edge overlap with domain
    - 01_pc_synthetic.png      PC on the synthetic 5-node ground truth
    - 01_notears_synthetic.png NOTEARS on the synthetic 5-node ground truth
"""

from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import redirect_stderr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make `src` importable when run as `python tests/test_causal_discovery.py`
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.causal.discovery import (  # noqa: E402
    compare_with_domain,
    fuse_graphs,
    inject_domain_knowledge,
    run_notears,
    run_pc,
)
from src.causal.home_credit_graph import HomeCreditCausalGraph  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "output", "demo_m1")
os.makedirs(OUT_DIR, exist_ok=True)

# Chinese-friendly font (matches run_pipeline.py convention)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _draw_dag(g: nx.DiGraph, title: str, ax, pos: dict | None = None) -> None:
    if pos is None:
        try:
            pos = nx.spring_layout(g, seed=42)
        except Exception:
            pos = nx.circular_layout(g)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#7AB8E6", node_size=900)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9)
    nx.draw_networkx_edges(
        g, pos, ax=ax, edge_color="#444", arrows=True,
        arrowsize=14, width=1.2,
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def demo_synthetic() -> None:
    """Synthetic chain x0 -> x1 -> x2 to validate the discovery pipeline."""
    print("[synthetic] building 5-node ground-truth chain x0 -> x1 -> x2 ...")
    rng = np.random.RandomState(42)
    n, d = 2000, 5
    X = rng.randn(n, d)
    X[:, 1] += 0.7 * X[:, 0]   # x0 -> x1
    X[:, 2] += 0.5 * X[:, 1]   # x1 -> x2
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])

    with redirect_stderr(io.StringIO()):
        pc_g = run_pc(df, alpha=0.01)
    nt_g = run_notears(df, lambda1=0.1, h_tol=1e-6, threshold=0.1)
    fused = fuse_graphs(pc_g, nt_g, edge_conf_threshold=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _draw_dag(pc_g, f"PC  (边数: {pc_g.number_of_edges()})", axes[0])
    _draw_dag(nt_g, f"NOTEARS  (边数: {nt_g.number_of_edges()})", axes[1])
    _draw_dag(fused, f"PC ∪ NOTEARS  (边数: {fused.number_of_edges()})", axes[2])
    fig.suptitle("合成数据: x0 -> x1 -> x2 链 (n=2000, β=0.7/0.5)", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "01_causal_discovery.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")

    # Individual PC + NOTEARS for the synthetic
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _draw_dag(pc_g, "PC (synthetic)", axes[0])
    _draw_dag(nt_g, "NOTEARS (synthetic)", axes[1])
    fig.tight_layout()
    out_synth = os.path.join(OUT_DIR, "01_synthetic_dags.png")
    fig.savefig(out_synth, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_synth}")

    # Overlap with the true DAG (we only care about which pair has an edge)
    truth = nx.DiGraph()
    truth.add_edges_from([("x0", "x1"), ("x1", "x2")])
    truth_pairs = {tuple(sorted(e)) for e in truth.edges()}
    pc_pairs = {tuple(sorted(e)) for e in pc_g.edges()}
    nt_pairs = {tuple(sorted(e)) for e in nt_g.edges()}
    fu_pairs = {tuple(sorted(e)) for e in fused.edges()}
    print(f"  Truth pairs: {sorted(truth_pairs)}")
    print(f"  PC recall: {len(pc_pairs & truth_pairs)}/{len(truth_pairs)} = "
          f"{len(pc_pairs & truth_pairs)/max(len(truth_pairs),1):.0%}")
    print(f"  NOTEARS recall: {len(nt_pairs & truth_pairs)}/{len(truth_pairs)} = "
          f"{len(nt_pairs & truth_pairs)/max(len(truth_pairs),1):.0%}")
    print(f"  Union recall: {len(fu_pairs & truth_pairs)}/{len(truth_pairs)} = "
          f"{len(fu_pairs & truth_pairs)/max(len(truth_pairs),1):.0%}")


def demo_home_credit() -> None:
    """Run PC + NOTEARS on a Home Credit subset and overlay the domain DAG."""
    data_dir = os.path.join(_ROOT, "data", "home-credit-default-risk")
    csv_path = os.path.join(data_dir, "application_train.csv")
    parquet_path = os.path.join(data_dir, "application_train.parquet")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=30000)
    elif os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path).head(30000)
    else:
        print(f"[home_credit] data not found in {data_dir}, skipping")
        return
    print(f"[home_credit] loading subset ({len(df)} rows) ...")
    # Use a handful of features for fast discovery
    feats = [
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL",
        "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "EXT_SOURCE_2", "REGION_RATING_CLIENT", "CNT_CHILDREN",
    ]
    feats = [c for c in feats if c in df.columns]
    sub = df[feats].dropna().sample(n=min(8000, len(df)), random_state=0).reset_index(drop=True)
    print(f"  features: {feats}")
    print(f"  n_used: {len(sub)}")

    with redirect_stderr(io.StringIO()):
        pc_g = run_pc(sub, alpha=0.05)
    nt_g = run_notears(sub, lambda1=0.1, h_tol=1e-6, threshold=0.1)
    fused = fuse_graphs(pc_g, nt_g, edge_conf_threshold=0.0)

    # PC-only and NOTEARS-only
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    _draw_dag(pc_g, f"PC on Home Credit subset  (边数: {pc_g.number_of_edges()})", axes[0])
    _draw_dag(nt_g, f"NOTEARS on Home Credit subset  (边数: {nt_g.number_of_edges()})", axes[1])
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "01_home_credit_dags.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")

    # Fused + domain overlay
    domain = HomeCreditCausalGraph()
    domain_pairs = {tuple(sorted(e)) for e in domain.edges}
    fused_pairs = {tuple(sorted(e)) for e in fused.edges}
    overlap = len(fused_pairs & domain_pairs)
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_dag(fused, f"Fused (PC ∪ NOTEARS)  (边数: {fused.number_of_edges()})", ax)
    ax.set_title(
        f"Home Credit 因果发现 (融合 + 领域图)\n"
        f"  融合边: {fused.number_of_edges()}  领域边: {len(domain_pairs)}  "
        f"重叠: {overlap} ({overlap/max(len(domain_pairs),1):.0%})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path2 = os.path.join(OUT_DIR, "01_discovery_with_domain.png")
    fig.savefig(out_path2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path2}")
    print(f"  Fused vs domain overlap: {overlap}/{len(domain_pairs)} "
          f"({overlap/max(len(domain_pairs),1):.0%})")

    # Edge-overlap bar chart
    cmp = compare_with_domain(fused, domain)
    labels = ["precision", "recall", "f1", "edge_overlap", "direction_agree"]
    vals = [
        cmp.get("precision", 0.0),
        cmp.get("recall", 0.0),
        cmp.get("f1", 0.0),
        cmp.get("edge_overlap", 0.0),
        cmp.get("direction_agree", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=["#7AB8E6", "#F2A65A", "#9BC53D", "#E0594E", "#9B59B6"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("比值")
    ax.set_title("Fused graph vs. Domain DAG  (5 个一致性指标)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    out_path3 = os.path.join(OUT_DIR, "01_discovery_metrics.png")
    fig.savefig(out_path3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path3}")


def demo_inject_domain() -> None:
    """Show that must-edges are added and forbidden edges removed."""
    print("[inject] demonstrating must / forbid edge injection ...")
    # Build a tiny graph
    g = nx.DiGraph()
    g.add_nodes_from(["A", "B", "C", "D"])
    g.add_edge("A", "B")
    g.add_edge("C", "D")
    must = [("B", "C")]
    forbid = [("A", "B")]
    out = inject_domain_knowledge(g, must_edges=must, forbid_edges=forbid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _draw_dag(g, "注入前", axes[0])
    _draw_dag(out, f"注入后: must={must}, forbid={forbid}", axes[1])
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "01_domain_injection.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")
    print(f"  before edges: {sorted(g.edges())}")
    print(f"  after edges:  {sorted(out.edges())}")


if __name__ == "__main__":
    demo_synthetic()
    print()
    demo_home_credit()
    print()
    demo_inject_domain()
    print()
    print(f"All figures written to {OUT_DIR}/")
