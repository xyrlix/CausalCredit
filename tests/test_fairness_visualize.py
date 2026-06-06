"""Tests for src.fairness.visualize."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


def _make_summaries():
    """Build a small set of FairnessSummary objects on a 1K synthetic frame."""
    from src.fairness.metrics import summarize_fairness
    rng = np.random.default_rng(7)
    n = 1000
    g = rng.choice(["M", "F", "UNKNOWN"], n, p=[0.45, 0.45, 0.10])
    y_t = rng.choice([0, 1], n, p=[0.92, 0.08])
    p_pred = 0.08 + 0.10 * (g == "F")
    y_p = (rng.uniform(size=n) < p_pred).astype(int)
    y_s = p_pred + rng.normal(scale=0.05, size=n)

    return {
        "gender": summarize_fairness("gender", y_t, y_p, y_s, g),
    }


def test_plot_group_rates_writes_png(tmp_path):
    from src.fairness.visualize import plot_group_rates
    out = str(tmp_path / "groups.png")
    plot_group_rates(_make_summaries(), out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000


def test_plot_metric_gaps_writes_png(tmp_path):
    from src.fairness.visualize import plot_metric_gaps
    out = str(tmp_path / "gaps.png")
    plot_metric_gaps(_make_summaries(), out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000


def test_plot_status_board_writes_png(tmp_path):
    from src.fairness.visualize import plot_status_board
    out = str(tmp_path / "status.png")
    plot_status_board(_make_summaries(), out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000


def test_render_all_writes_three_pngs(tmp_path):
    from src.fairness.visualize import render_all
    paths = render_all(_make_summaries(), str(tmp_path))
    assert len(paths) == 3
    for p in paths:
        assert os.path.exists(p)
