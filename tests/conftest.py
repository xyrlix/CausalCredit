"""Test fixtures and shared configuration for pytest."""

import pytest


@pytest.fixture
def sample_config():
    """Return a minimal config dict for testing."""
    return {
        "data": {"home_credit_dir": "tests/fixtures/data/"},
        "features": {"top_k": 80, "target_encoding_folds": 5},
        "model": {"lightgbm": {"random_state": 42}},
        "api": {"host": "127.0.0.1", "port": 8000},
    }
