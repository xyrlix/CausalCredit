"""Model registry manifest — SHA-256 integrity + provenance.

Sits next to ``registry_v1.pkl`` as ``registry_v1.pkl.manifest.yaml`` and
records:

* the SHA-256 of the pickle file (integrity)
* training timestamp + active version string
* feature schema (column names + types)
* library versions for reproducibility (lightgbm, scikit-learn, dowhy)

Verified on every load — a corrupted or swapped pickle triggers
retrain-from-scratch rather than a silent bad-score.

@requirement NFR-005
@design docs/plans/2026-06-05-causalcredit-architecture-design.md §D5
"""

from __future__ import annotations

import hashlib
import importlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("causalcredit.manifest")

MANIFEST_SUFFIX = ".manifest.yaml"


def compute_file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in 1 MB chunks (works for large pickles)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_version(pkg: str) -> Optional[str]:
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return None


def collect_environment() -> Dict[str, Optional[str]]:
    """Library versions for the manifest (best-effort)."""
    return {
        "lightgbm": _try_version("lightgbm"),
        "scikit-learn": _try_version("sklearn"),
        "dowhy": _try_version("dowhy"),
        "econml": _try_version("econml"),
        "shap": _try_version("shap"),
        "numpy": _try_version("numpy"),
        "pandas": _try_version("pandas"),
    }


def manifest_path_for(pickle_path: Path) -> Path:
    """Sidecar manifest path: ``foo.pkl`` → ``foo.pkl.manifest.yaml``."""
    return pickle_path.with_name(pickle_path.name + MANIFEST_SUFFIX)


def build_manifest(
    pickle_path: Path,
    *,
    active_version: str,
    feature_cols: List[str],
    cat_cols: List[str],
    n_samples: int,
) -> Dict[str, Any]:
    """Assemble the manifest dict (caller writes via :func:`write_manifest`)."""
    return {
        "schema_version": 1,
        "active_version": active_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pickle_sha256": compute_file_hash(pickle_path),
        "pickle_size_bytes": pickle_path.stat().st_size,
        "feature_cols": list(feature_cols),
        "cat_cols": list(cat_cols),
        "n_samples": int(n_samples),
        "environment": collect_environment(),
    }


def write_manifest(pickle_path: Path, manifest: Dict[str, Any]) -> Path:
    """Persist the manifest as YAML next to the pickle."""
    path = manifest_path_for(pickle_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    logger.info("manifest written: %s (version=%s, sha256=%s)",
                path, manifest.get("active_version"), manifest.get("pickle_sha256")[:12])
    return path


def read_manifest(pickle_path: Path) -> Optional[Dict[str, Any]]:
    """Load the sidecar manifest, or return None if it doesn't exist."""
    path = manifest_path_for(pickle_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_cache(pickle_path: Path) -> bool:
    """Return True if the pickle on disk matches its sidecar manifest.

    A missing manifest, mismatched hash, or unreadable file all return
    False — the caller should treat that as a signal to retrain rather
    than load stale bytes.
    """
    if not pickle_path.exists():
        return False
    manifest = read_manifest(pickle_path)
    if manifest is None:
        logger.warning("no manifest for %s; treating cache as invalid", pickle_path)
        return False
    expected = manifest.get("pickle_sha256")
    if not expected:
        return False
    actual = compute_file_hash(pickle_path)
    if actual != expected:
        logger.warning("manifest hash mismatch for %s: expected=%s actual=%s",
                       pickle_path, expected[:12], actual[:12])
        return False
    return True


def make_active_version() -> str:
    """Default version string: ``train-YYYYMMDD-HHMMSS`` (UTC)."""
    return "train-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
