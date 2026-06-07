"""Smoke tests for Streamlit page render functions (M8.3c).

These tests do NOT launch a Streamlit server. They patch ``streamlit`` with
fake callables so the page modules can be imported and their ``render``
entry points invoked in-process. This catches:
  - import errors
  - missing helper functions (e.g. ``_build_dag``, ``_render_narrative_section``)
  - argument-shape regressions against the real ctx dict
  - crashes when the registry exposes the fields the pages expect
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Fake streamlit — must be installed BEFORE importing the page modules
# ---------------------------------------------------------------------------
class _FakeStreamlit:
    """Minimal Streamlit stub. Every public attribute is callable; calls
    return sensible defaults so pages can render without a real server."""

    def __init__(self):
        self._calls: List[tuple] = []
        self.session_state: Dict[str, Any] = {}

    def __getattr__(self, name: str):
        # Generic catch-all: any unknown name returns a no-op MagicMock that
        # is also callable (so e.g. ``st.spinner(...)`` works as a context mgr).
        m = MagicMock(name=f"st.{name}")
        m.__enter__ = lambda self_: None
        m.__exit__ = lambda self_, *a: None
        return m

    def title(self, *_a, **_kw): self._calls.append(("title", _a))
    def caption(self, *_a, **_kw): self._calls.append(("caption", _a))
    def subheader(self, *_a, **_kw): self._calls.append(("subheader", _a))
    def markdown(self, *_a, **_kw): self._calls.append(("markdown", _a))
    def info(self, *_a, **_kw): self._calls.append(("info", _a))
    def warning(self, *_a, **_kw): self._calls.append(("warning", _a))
    def error(self, *_a, **_kw): self._calls.append(("error", _a))
    def dataframe(self, *_a, **_kw): self._calls.append(("dataframe", _a))
    def json(self, *_a, **_kw): self._calls.append(("json", _a))
    def metric(self, *_a, **_kw): self._calls.append(("metric", _a))
    def image(self, *_a, **_kw): self._calls.append(("image", _a))
    def button(self, *_a, **_kw): return False
    def form_submit_button(self, *_a, **_kw): return False
    def form(self, *_a, **_kw):
        cm = MagicMock()
        cm.__enter__ = lambda self_: None
        cm.__exit__ = lambda self_, *a: None
        return cm
    def expander(self, *_a, **_kw):
        cm = MagicMock()
        cm.__enter__ = lambda self_: None
        cm.__exit__ = lambda self_, *a: None
        return cm
    def spinner(self, *_a, **_kw):
        cm = MagicMock()
        cm.__enter__ = lambda self_: None
        cm.__exit__ = lambda self_, *a: None
        return cm
    def tabs(self, names):
        n = len(names) if hasattr(names, "__len__") else 1
        return [MagicMock() for _ in range(n)]
    def columns(self, n):
        n = n if isinstance(n, int) else len(n)
        return [MagicMock() for _ in range(n)]
    def number_input(self, *_a, **_kw): return 0
    def slider(self, *_a, **_kw): return 0.0
    def selectbox(self, *_a, **_kw): return ""
    def radio(self, *_a, **_kw): return ""
    def checkbox(self, *_a, **_kw): return False
    def set_page_config(self, *_a, **_kw): pass
    def graphviz_chart(self, *_a, **_kw): pass
    def sidebar(self): return self
    def cache_data(self, *_a, **_kw):
        # No-op decorator — caching disabled for tests
        def deco(fn):
            return fn
        return deco
    def cache_resource(self, *_a, **_kw):
        def deco(fn):
            return fn
        return deco


@pytest.fixture
def fake_streamlit(monkeypatch):
    fs = _FakeStreamlit()
    # Register the fake `streamlit` module
    mod = types.ModuleType("streamlit")
    for attr in dir(fs):
        if not attr.startswith("_"):
            setattr(mod, attr, getattr(fs, attr))
    # Catch-all for any further names
    for name in [
        "progress", "balloons", "snow", "toast", "status", "container",
        "empty", "stop", "experimental_rerun", "rerun",
    ]:
        setattr(mod, name, MagicMock())
    monkeypatch.setitem(sys.modules, "streamlit", mod)
    return fs


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
PRESET = {
    "AMT_CREDIT": 400000, "AMT_ANNUITY": 18000, "AMT_GOODS_PRICE": 380000,
    "AMT_INCOME_TOTAL": 250000, "DAYS_BIRTH": -12775, "DAYS_EMPLOYED": -3650,
    "EXT_SOURCE_2": 0.78, "EXT_SOURCE_3": 0.72, "REGION_RATING_CLIENT": 1,
    "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2, "DAYS_REGISTRATION": -2000,
    "DAYS_ID_PUBLISH": -1200, "REGION_POPULATION_RELATIVE": 0.02,
    "CODE_GENDER": "M", "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
    "OCCUPATION_TYPE": "Managers",
}


class _FakeRegistry:
    feature_cols = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                    "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED",
                    "EXT_SOURCE_2", "EXT_SOURCE_3", "REGION_RATING_CLIENT",
                    "CNT_CHILDREN", "CNT_FAM_MEMBERS"]
    counterfactual_reasoner = None
    causal_graph = None
    ate_summary = {"ate": 0.05, "ci_lower": 0.04, "ci_upper": 0.06,
                   "treatment": "AMT_CREDIT (binarized)", "outcome": "TARGET",
                   "method": "DoWhy"}
    training_data = None

    def transform_features(self, f): raise NotImplementedError  # tested separately


class _FakeService:
    def score(self, request):
        # Return a stub object with the attributes pages read
        from types import SimpleNamespace
        return SimpleNamespace(
            score=720, default_probability=0.08, risk_grade="B",
            decision_suggestion="APPROVE with standard terms",
            explanation={"top_features": [
                {"feature": "EXT_SOURCE_2", "value": 0.78, "shap": 0.05,
                 "direction": "decreases_default"},
            ]},
            causal_effect=self._registry.ate_summary,
            counterfactual=[],
        )

    def counterfactual(self, request):
        from types import SimpleNamespace
        return SimpleNamespace(
            baseline_probability=0.08, counterfactual_probability=0.06,
            probability_change=-0.02, confidence=0.8, intervention_details={},
        )

    def __init__(self, registry):
        self._registry = registry


@pytest.fixture
def ctx():
    reg = _FakeRegistry()
    svc = _FakeService(reg)
    return {
        "registry": reg, "service": svc,
        "preset_features": dict(PRESET), "preset_name": "Prime Customer",
    }


# ---------------------------------------------------------------------------
# Test: pages import cleanly with fake streamlit
# ---------------------------------------------------------------------------
def test_score_dashboard_imports(fake_streamlit):
    from src.frontend.pages import score_dashboard
    assert callable(score_dashboard.render)


def test_causal_visualization_imports(fake_streamlit):
    from src.frontend.pages import causal_visualization
    assert callable(causal_visualization.render)


def test_counterfactual_simulator_imports(fake_streamlit):
    from src.frontend.pages import counterfactual_simulator
    assert callable(counterfactual_simulator.render)


def test_decision_panel_imports(fake_streamlit):
    from src.frontend.pages import decision_panel
    assert callable(decision_panel.render)


# ---------------------------------------------------------------------------
# Test: helpers exist with correct signatures
# ---------------------------------------------------------------------------
def test_decision_panel_helpers_exposed():
    """The M8.3c narrative tab depends on these module-level helpers."""
    from src.frontend.pages import decision_panel
    assert callable(getattr(decision_panel, "_global_narrative_context", None))
    assert callable(getattr(decision_panel, "_build_dag", None))
    assert callable(getattr(decision_panel, "_render_narrative_section", None))


def test_decision_panel_build_dag_uses_home_credit_graph():
    """_build_dag should consult registry.causal_graph and return networkx.DiGraph."""
    import networkx as nx
    from src.frontend.pages.decision_panel import _build_dag
    from src.causal.home_credit_graph import HomeCreditCausalGraph
    reg = _FakeRegistry()
    reg.causal_graph = HomeCreditCausalGraph()
    g = _build_dag(reg)
    assert isinstance(g, nx.DiGraph)
    assert "TARGET" in g.nodes
    assert len(g.edges) > 0


def test_decision_panel_build_dag_handles_none_graph():
    from src.frontend.pages.decision_panel import _build_dag
    reg = _FakeRegistry()
    reg.causal_graph = None
    g = _build_dag(reg)
    import networkx as nx
    assert isinstance(g, nx.DiGraph)
    assert len(g.nodes) == 0


# ---------------------------------------------------------------------------
# Test: each render() runs without raising
# ---------------------------------------------------------------------------
def test_causal_visualization_runs(fake_streamlit, ctx):
    from src.frontend.pages import causal_visualization
    causal_visualization.render(ctx)  # should not raise


def test_counterfactual_simulator_runs(fake_streamlit, ctx):
    from src.frontend.pages import counterfactual_simulator
    counterfactual_simulator.render(ctx)  # should not raise


def test_decision_panel_runs_with_no_report(fake_streamlit, ctx):
    """On first load (no `last_report` in session_state) the panel shows
    the info hint and returns — must not crash."""
    from src.frontend.pages import decision_panel
    decision_panel.render(ctx)


# ---------------------------------------------------------------------------
# Test: M8.3c narrative tab contract
# ---------------------------------------------------------------------------
def test_decision_panel_renders_narrative_section_zh(fake_streamlit):
    """The narrative tab renderer must accept a narrative dict + language
    string and produce markdown / dataframe calls without raising."""
    import numpy as np
    import pandas as pd
    from src.frontend.pages.decision_panel import _render_narrative_section
    from src.explain.causal_narrative import CausalNarrative

    cn = CausalNarrative(model=None, feature_names=["a", "b", "c"], dag=None)
    X_train = pd.DataFrame({"a": [0.0] * 10, "b": [0.0] * 10, "c": [0.0] * 10})
    full = cn.build_full_narrative(
        features={"a": 1.0, "b": 0.5, "c": -0.2},
        shap_row=np.array([0.1, -0.05, 0.3]),
        shap_global=np.zeros((5, 3)),
        X_train=X_train,
        y_prob_train=np.zeros(10),
        run_robustness=False,
    )
    _render_narrative_section(full, language="zh")


def test_decision_panel_renders_narrative_section_en(fake_streamlit):
    import numpy as np
    import pandas as pd
    from src.frontend.pages.decision_panel import _render_narrative_section
    from src.explain.causal_narrative import CausalNarrative

    cn = CausalNarrative(model=None, feature_names=["a", "b", "c"], dag=None)
    X_train = pd.DataFrame({"a": [0.0] * 10, "b": [0.0] * 10, "c": [0.0] * 10})
    full = cn.build_full_narrative(
        features={"a": 1.0, "b": 0.5, "c": -0.2},
        shap_row=np.array([0.1, -0.05, 0.3]),
        shap_global=np.zeros((5, 3)),
        X_train=X_train, y_prob_train=np.zeros(10),
        run_robustness=False,
    )
    _render_narrative_section(full, language="en")
    _render_narrative_section(full, language="zh-HK")
    _render_narrative_section(full, language="klingon")  # falls back silently
