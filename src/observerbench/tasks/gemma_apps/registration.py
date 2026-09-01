"""Digest seal for the prospective matched Gemma APPS replication.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from observerbench.provenance import file_sha256, json_sha256


CONFIG_SCHEMA = "observerbench.controlarena_apps_gemma.v0"
MANIFEST_SCHEMA = "observerbench.controlarena_apps_gemma.source_manifest.v0"
MANIFEST_STATUS = "frozen_before_gemma_monitor_outcomes"


def _repo_root(path: Path) -> Path:
    candidate = path.resolve()
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src/observerbench").exists():
            return parent
    raise ValueError(f"could not find ObserverBench repository above {path}")


def load_gemma_apps_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported Gemma APPS replication config")
    if payload.get("status") != "prospectively_frozen_before_gemma_monitor_outcomes":
        raise ValueError("Gemma APPS config does not declare its prospective status")
    return payload


def _source_paths(config: Mapping[str, Any], repo_root: Path) -> tuple[Path, ...]:
    raw = config.get("seal", {}).get("source_files")
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) for item in raw):
        raise ValueError("Gemma APPS config requires a nonempty seal.source_files list")
    if len(raw) != len(set(raw)):
        raise ValueError("Gemma APPS source file list contains duplicates")
    paths = tuple(repo_root / item for item in raw)
    missing = [path.relative_to(repo_root).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing sealed source files: {', '.join(missing)}")
    return paths


def _artifact_hashes(
    config: Mapping[str, Any], repo_root: Path
) -> dict[str, str]:
    return {
        path.relative_to(repo_root).as_posix(): file_sha256(path)
        for path in _source_paths(config, repo_root)
    }


def _bundle_sha256(artifact_hashes: Mapping[str, str]) -> str:
    return json_sha256(dict(sorted(artifact_hashes.items())))


def seal_gemma_apps_sources(
    config_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write the local source seal before any Gemma monitor outcome is read."""

    config_path = Path(config_path).resolve()
    repo_root = _repo_root(config_path)
    config = load_gemma_apps_config(config_path)
    declared_manifest = repo_root / str(config["seal"]["manifest"])
    output = Path(manifest_path).resolve() if manifest_path is not None else declared_manifest
    if output != declared_manifest.resolve():
        raise ValueError("manifest path differs from the path declared by the config")
    if output.exists():
        raise FileExistsError("Gemma APPS source manifest already exists; verify it instead")
    artifacts = _artifact_hashes(config, repo_root)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "status": MANIFEST_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration_scope": "local digest seal; not an independent timestamp",
        "config_path": config_path.relative_to(repo_root).as_posix(),
        "config_sha256": file_sha256(config_path),
        "config_json_sha256": json_sha256(config),
        "artifact_hashes": artifacts,
        "source_bundle_sha256": _bundle_sha256(artifacts),
        "dataset_revision": str(config["dataset"]["revision"]),
        "model_revision": str(config["model"]["revision"]),
        "sae_revision": str(config["gemma_scope"]["revision"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_gemma_apps_source_manifest(
    config_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reject a run if any frozen configuration or source file changed."""

    config_path = Path(config_path).resolve()
    repo_root = _repo_root(config_path)
    config = load_gemma_apps_config(config_path)
    declared_manifest = (repo_root / str(config["seal"]["manifest"])).resolve()
    supplied = Path(manifest_path).resolve() if manifest_path is not None else declared_manifest
    if supplied != declared_manifest:
        raise ValueError("supplied manifest differs from the config's declared manifest")
    payload = json.loads(supplied.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("status") != MANIFEST_STATUS:
        raise ValueError("invalid Gemma APPS source manifest schema or status")
    expected_paths = [
        path.relative_to(repo_root).as_posix() for path in _source_paths(config, repo_root)
    ]
    observed = payload.get("artifact_hashes")
    if not isinstance(observed, dict) or sorted(observed) != sorted(expected_paths):
        raise ValueError("source manifest file set differs from the frozen config")
    current = _artifact_hashes(config, repo_root)
    changed = [path for path in sorted(current) if current[path] != observed.get(path)]
    if changed:
        raise ValueError(f"sealed Gemma APPS sources changed: {', '.join(changed)}")
    checks = {
        "config_sha256": file_sha256(config_path),
        "config_json_sha256": json_sha256(config),
        "source_bundle_sha256": _bundle_sha256(current),
    }
    for key, value in checks.items():
        if payload.get(key) != value:
            raise ValueError(f"Gemma APPS manifest {key} does not match")
    for key, value in (
        ("dataset_revision", config["dataset"]["revision"]),
        ("model_revision", config["model"]["revision"]),
        ("sae_revision", config["gemma_scope"]["revision"]),
    ):
        if payload.get(key) != value:
            raise ValueError(f"Gemma APPS manifest {key} does not match")
    return payload


__all__ = [
    "CONFIG_SCHEMA",
    "MANIFEST_SCHEMA",
    "load_gemma_apps_config",
    "seal_gemma_apps_sources",
    "verify_gemma_apps_source_manifest",
]
