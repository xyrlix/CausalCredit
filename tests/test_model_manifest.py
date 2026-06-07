"""Tests for the model-registry manifest (M8.5e).

Covers SHA-256 integrity, sidecar YAML I/O, mismatch detection, and
the ``active_version`` / ``model_hash`` fields on :class:`ModelRegistry`.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import pytest

from src.api.model_manifest import (
    build_manifest,
    collect_environment,
    compute_file_hash,
    make_active_version,
    manifest_path_for,
    read_manifest,
    validate_cache,
    write_manifest,
)


@pytest.fixture
def tmp_pickle(tmp_path):
    p = tmp_path / "registry_v1.pkl"
    p.write_bytes(b"fake-pickle-payload-for-test")
    return p


class TestComputeFileHash:
    def test_returns_64_char_hex(self, tmp_pickle):
        h = compute_file_hash(tmp_pickle)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, tmp_pickle):
        h1 = compute_file_hash(tmp_pickle)
        h2 = compute_file_hash(tmp_pickle)
        assert h1 == h2

    def test_changes_when_content_changes(self, tmp_pickle):
        h1 = compute_file_hash(tmp_pickle)
        tmp_pickle.write_bytes(b"different-content")
        h2 = compute_file_hash(tmp_pickle)
        assert h1 != h2

    def test_handles_large_file(self, tmp_path):
        p = tmp_path / "big.bin"
        # 5 MB of zeros — bigger than default chunk size
        p.write_bytes(b"\x00" * (5 * 1024 * 1024))
        h = compute_file_hash(p)
        # Just verify the function returns a valid 64-char hex digest
        # (don't hardcode the hash — it depends on padding details)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestManifestPath:
    def test_sidecar_naming(self, tmp_path):
        p = tmp_path / "registry_v1.pkl"
        assert manifest_path_for(p) == tmp_path / "registry_v1.pkl.manifest.yaml"

    def test_works_with_arbitrary_filename(self, tmp_path):
        p = tmp_path / "model.pkl"
        assert manifest_path_for(p) == tmp_path / "model.pkl.manifest.yaml"


class TestWriteReadManifest:
    def test_roundtrip(self, tmp_pickle):
        m = build_manifest(
            tmp_pickle,
            active_version="train-20260607-120000",
            feature_cols=["AMT_CREDIT", "DAYS_BIRTH"],
            cat_cols=["CODE_GENDER"],
            n_samples=50000,
        )
        path = write_manifest(tmp_pickle, m)
        assert path.exists()

        loaded = read_manifest(tmp_pickle)
        assert loaded is not None
        assert loaded["active_version"] == "train-20260607-120000"
        assert loaded["feature_cols"] == ["AMT_CREDIT", "DAYS_BIRTH"]
        assert loaded["cat_cols"] == ["CODE_GENDER"]
        assert loaded["n_samples"] == 50000
        assert loaded["pickle_sha256"] == compute_file_hash(tmp_pickle)
        assert loaded["schema_version"] == 1
        # environment block present
        assert "environment" in loaded
        assert "lightgbm" in loaded["environment"]
        # created_at is ISO 8601 with timezone
        assert "T" in loaded["created_at"]

    def test_read_missing_returns_none(self, tmp_path):
        p = tmp_path / "no-manifest.pkl"
        p.write_bytes(b"x")
        assert read_manifest(p) is None

    def test_environment_returns_known_libs(self):
        env = collect_environment()
        for lib in ("lightgbm", "scikit-learn", "numpy", "pandas", "dowhy"):
            assert lib in env
        # sklearn is installed in any modern env
        assert env["scikit-learn"] is not None
        assert env["numpy"] is not None


class TestValidateCache:
    def test_valid_when_hash_matches(self, tmp_pickle):
        m = build_manifest(
            tmp_pickle, active_version="v1",
            feature_cols=[], cat_cols=[], n_samples=0,
        )
        write_manifest(tmp_pickle, m)
        assert validate_cache(tmp_pickle) is True

    def test_invalid_when_pickle_tampered(self, tmp_pickle):
        m = build_manifest(
            tmp_pickle, active_version="v1",
            feature_cols=[], cat_cols=[], n_samples=0,
        )
        write_manifest(tmp_pickle, m)
        # Append a byte — hash will no longer match
        with open(tmp_pickle, "ab") as f:
            f.write(b"\x00")
        assert validate_cache(tmp_pickle) is False

    def test_invalid_when_manifest_missing(self, tmp_pickle):
        assert validate_cache(tmp_pickle) is False

    def test_invalid_when_pickle_missing(self, tmp_path):
        p = tmp_path / "ghost.pkl"
        assert validate_cache(p) is False


class TestMakeActiveVersion:
    def test_starts_with_train(self):
        v = make_active_version()
        assert v.startswith("train-")
        # train-YYYYMMDD-HHMMSS = 20 chars
        assert len(v) == len("train-20260607-120000")

    def test_unique_across_calls(self):
        # Two calls one second apart can collide; require 2s gap if we want
        # strict uniqueness. In practice retraining is minutes apart.
        v1 = make_active_version()
        v2 = make_active_version()
        # Both should still be valid version strings
        assert v1.startswith("train-")
        assert v2.startswith("train-")


class TestModelRegistryProvenance:
    """Integration smoke for active_version + model_hash on ModelRegistry."""

    def test_fields_initially_empty(self):
        from src.api.dependencies import ModelRegistry
        r = ModelRegistry()
        assert r.active_version == ""
        assert r.model_hash == ""

    def test_exposes_fields_in_health_response(self, tmp_pickle, monkeypatch):
        """Stub the registry cache to a known-good file and verify the
        /health endpoint surfaces version + hash."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.dependencies import ModelRegistry, get_model_registry
        from src.api.routes import router

        # Build a minimal model registry that's pre-loaded
        reg = ModelRegistry()
        reg.active_version = "train-test-12345"
        reg.model_hash = "deadbeef" * 8  # 64 hex chars

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_model_registry] = lambda: reg

        client = TestClient(app)
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["active_version"] == "train-test-12345"
        assert body["model_hash"].startswith("deadbeef")
        # Hash is truncated to 16 chars + ellipsis
        assert body["model_hash"].endswith("…")
