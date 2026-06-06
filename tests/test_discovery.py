"""Unit tests for src.causal.discovery (PC + NOTEARS fusion)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal.discovery import (
    compare_with_domain,
    fuse_graphs,
    inject_domain_knowledge,
    run_notears,
    run_pc,
)


@pytest.fixture
def synthetic_chain():
    """A clean x0 -> x1 -> x2 chain. Strong signal (β=0.7/0.5), n=2000."""
    rng = np.random.RandomState(42)
    n, d = 2000, 5
    X = rng.randn(n, d)
    X[:, 1] += 0.7 * X[:, 0]
    X[:, 2] += 0.5 * X[:, 1]
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
    return df, {("x0", "x1"), ("x1", "x2")}


# ---------------------------------------------------------------------------
# NOTEARS — the formerly broken one. The smoke tests above ground the fix:
# the inner L-BFGS-B used to collapse to W=0.035 because the textbook rho
# (1.0) and the AL penalty pulled all weights to zero. With rho_init=0.01,
# smooth L1 surrogate, and best-W tracking, we recover the chain.
# ---------------------------------------------------------------------------

def test_notears_recovers_chain_skeleton(synthetic_chain):
    df, truth = synthetic_chain
    g = run_notears(df, lambda1=0.1, h_tol=1e-6, threshold=0.1)
    found = {tuple(sorted(e)) for e in g.edges}
    # The chain (x0,x1) and (x1,x2) must both appear (in either direction).
    assert ("x0", "x1") in found
    assert ("x1", "x2") in found


def test_notears_weight_magnitude_is_reasonable(synthetic_chain):
    """L1-sparse solution should have meaningful weights, not collapsed to ~0."""
    df, _ = synthetic_chain
    g = run_notears(df, lambda1=0.1, h_tol=1e-6, threshold=0.1)
    weights = [d.get("weight", 0.0) for _, _, d in g.edges(data=True)]
    assert max(abs(w) for w in weights) > 0.1, (
        f"NOTEARS weights collapsed (max|weight|={max(abs(w) for w in weights):.4f})"
    )


# ---------------------------------------------------------------------------
# PC
# ---------------------------------------------------------------------------

def test_pc_finds_chain_skeleton(synthetic_chain):
    df, truth = synthetic_chain
    import io
    from contextlib import redirect_stderr
    with redirect_stderr(io.StringIO()):
        g = run_pc(df, alpha=0.01)
    found = {tuple(sorted(e)) for e in g.edges}
    assert ("x0", "x1") in found
    assert ("x1", "x2") in found


def test_pc_marks_undirected_with_edge_type(synthetic_chain):
    """PC on the chain returns the skeleton only (undirected)."""
    import io
    from contextlib import redirect_stderr
    df, _ = synthetic_chain
    with redirect_stderr(io.StringIO()):
        g = run_pc(df, alpha=0.01)
    types = {d.get("edge_type") for _, _, d in g.edges(data=True)}
    assert "undirected" in types


# ---------------------------------------------------------------------------
# Fusion + domain injection
# ---------------------------------------------------------------------------

def test_fuse_union_returns_superset(synthetic_chain):
    import io
    from contextlib import redirect_stderr
    df, _ = synthetic_chain
    with redirect_stderr(io.StringIO()):
        pc_g = run_pc(df, alpha=0.01)
    nt_g = run_notears(df, lambda1=0.1, h_tol=1e-6, threshold=0.1)
    union = fuse_graphs(pc_g, nt_g, edge_conf_threshold=0.0)
    inter = fuse_graphs(pc_g, nt_g, edge_conf_threshold=0.7)
    assert union.number_of_edges() >= pc_g.number_of_edges()
    assert union.number_of_edges() >= nt_g.number_of_edges()
    assert inter.number_of_edges() <= union.number_of_edges()


def test_inject_must_adds_edges():
    g = _two_node_graph()
    out = inject_domain_knowledge(g, must_edges=[("B", "C")], forbid_edges=[])
    assert ("B", "C") in out.edges


def test_inject_forbid_removes_edges():
    g = _two_node_graph()
    out = inject_domain_knowledge(g, must_edges=[], forbid_edges=[("A", "B")])
    assert ("A", "B") not in out.edges


# ---------------------------------------------------------------------------
# Domain comparison
# ---------------------------------------------------------------------------

def test_compare_with_domain_returns_metrics():
    import networkx as nx
    domain = nx.DiGraph()
    domain.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    discovered = nx.DiGraph()
    discovered.add_edges_from([("A", "B"), ("B", "C")])
    cmp = compare_with_domain(discovered, domain)
    for k in ("n_overlap", "overlap_rate_domain", "overlap_rate_discovered", "n_shared_nodes"):
        assert k in cmp
    # Discovered has 2 edges, both in domain -> overlap_rate_discovered = 1.0
    assert cmp["overlap_rate_discovered"] == pytest.approx(1.0, abs=1e-6)
    # The pair (A,B) is the only one in the *shared-node* subset (B is shared).
    # Domain has 1 edge in shared nodes (B->C). Discovered has 1 edge in shared
    # nodes (B->C). So n_overlap = 1.
    assert cmp["n_overlap"] == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_node_graph():
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(["A", "B", "C", "D"])
    g.add_edge("A", "B")
    g.add_edge("C", "D")
    return g
