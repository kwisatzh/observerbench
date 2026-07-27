"""Small provenance helpers for persisted ObserverBench results.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def component_provenance(component: Any) -> dict[str, Any]:
    """Describe a runtime component without assuming a base class."""

    payload: dict[str, Any] = {
        "name": str(component.name),
        "class": f"{type(component).__module__}.{type(component).__qualname__}",
    }
    component_version = getattr(component, "__version__", None)
    if component_version is not None:
        payload["version"] = str(component_version)
    if is_dataclass(component):
        payload["parameters"] = asdict(component)
    return payload


def _git_value(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def runtime_provenance(repo: str | Path | None = None) -> dict[str, Any]:
    """Return package, dependency, Python, and best-effort Git provenance."""

    repo_path = Path(repo).resolve() if repo is not None else Path.cwd().resolve()
    payload: dict[str, Any] = {
        "package_version": package_version("observerbench"),
        "python_version": platform.python_version(),
        "dependencies": {
            name: package_version(name)
            for name in ("numpy", "pandas", "matplotlib", "torch")
        },
        "source_revision": "unknown",
        "source_dirty": None,
    }
    try:
        root = Path(_git_value(repo_path, "rev-parse", "--show-toplevel"))
        payload["source_revision"] = _git_value(root, "rev-parse", "HEAD")
        payload["source_dirty"] = bool(
            _git_value(root, "status", "--porcelain", "--untracked-files=no")
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return payload


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(encoded).hexdigest()


def portable_artifact_path(
    path: str | Path,
    repo_root: str | Path | None = None,
) -> str:
    """Return a provenance label that does not expose a local absolute path."""

    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    if repo_root is not None:
        try:
            return candidate.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
    return f"external/{candidate.name}"


def source_hashes(
    paths: Iterable[str | Path],
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    return {
        portable_artifact_path(path, repo_root): file_sha256(path)
        for path in paths
    }
