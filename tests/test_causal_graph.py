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
