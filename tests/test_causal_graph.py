"""Unit tests for src.causal.home_credit_graph (HomeCreditCausalGraph)."""

from __future__ import annotations

import pytest

from src.causal.home_credit_graph import HomeCreditCausalGraph


@pytest.fixture
def graph():
    return HomeCreditCausalGraph()


def test_graph_has_treatments(graph):
    treatments = graph.get_treatment_variables()
    assert len(treatments) > 0
    assert "AMT_CREDIT" in treatments


def test_graph_outcome_is_target(graph):
    assert graph.get_outcome_variable() == "TARGET"


def test_graph_is_acyclic(graph):
    assert graph.validate_acyclic() is True


def test_graph_has_confounders(graph):
    confounders = graph.get_confounders("AMT_CREDIT", "TARGET")
    assert isinstance(confounders, list)
    # Per the plan, AMT_CREDIT should have several confounders
    assert len(confounders) >= 3


def test_graph_edges_only_between_known_nodes(graph):
    node_set = set(graph.nodes.keys())
    for u, v in graph.edges:
        assert u in node_set, f"edge source {u} not in nodes"
        assert v in node_set, f"edge target {v} not in nodes"


def test_graph_dot_string_renders(graph):
    dot = graph.get_dot_string()
    assert isinstance(dot, str)
    assert "digraph" in dot.lower() or "->" in dot


def test_graph_node_count_in_range(graph):
    """Plan calls for ~15 nodes with 25-40 edges."""
    assert 10 <= len(graph.nodes) <= 25
    assert 15 <= len(graph.edges) <= 50


# ---------------------------------------------------------------------------
# M8.2g — EXT_SOURCE_1/3 + BUREAU_TYPE_MICROLOAN_FRAC added to domain DAG
# ---------------------------------------------------------------------------


def test_ext_source_1_3_and_microloan_frac_in_dag():
    """M8.2g: 3 SHAP-heavy features that previously had n_paths=0 now reachable."""
    from src.causal.home_credit_graph import HomeCreditCausalGraph
    g = HomeCreditCausalGraph()
    for f in ("EXT_SOURCE_1", "EXT_SOURCE_3", "BUREAU_TYPE_MICROLOAN_FRAC"):
        assert f in g.nodes, f"{f} not in DAG nodes"
        assert (f, "TARGET") in g.edges, f"{f} has no edge to TARGET"
    # Still acyclic
    assert g.validate_acyclic() is True
    # EXT_SOURCE_2 should also be there
    assert ("EXT_SOURCE_2", "TARGET") in g.edges


def test_trace_finds_direct_paths_for_new_features():
    """CausalNarrative should now find direct paths for these 3 features."""
    import networkx as nx
    from src.causal.home_credit_graph import HomeCreditCausalGraph
    g = HomeCreditCausalGraph()
    narr_dag = nx.DiGraph()
    narr_dag.add_nodes_from(g.nodes.keys())
    narr_dag.add_edges_from(g.edges)
    from src.explain.causal_narrative import CausalNarrative
    cn = CausalNarrative(model=None, feature_names=list(g.nodes), dag=narr_dag, outcome_name="TARGET")
    for f in ("EXT_SOURCE_1", "EXT_SOURCE_3", "BUREAU_TYPE_MICROLOAN_FRAC"):
        paths = cn.trace_causal_path(f)
        assert any("TARGET" in p for p in paths), f"{f} has no path to TARGET"
        assert any(p == [f, "TARGET"] for p in paths), f"{f} lacks direct edge path"
