"""Focused tests for the source-sealed Qwen3.5 APPS archive.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from observerbench.provenance import file_sha256
from observerbench.tasks.qwen35_apps.registration import seal_qwen35_apps_sources


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SCRIPT = ROOT / "scripts/build_qwen35_apps_source_archive.py"


def _archive_builder():
    spec = importlib.util.spec_from_file_location(
        "observerbench_qwen35_apps_archive_builder", ARCHIVE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_archive


def _config(*, pending: bool = False) -> dict[str, object]:
    revision = "PENDING_REVISION" if pending else "a" * 40
    sae_revision = "PENDING_REVISION" if pending else "b" * 40
    return {
        "schema": "observerbench.controlarena_apps_qwen35.v0",
        "status": "prospectively_frozen_before_qwen35_monitor_outcomes",
        "task_id": "controlarena-apps-verified-backdoors@qwen3.5-9b-v0",
        "dataset": {"revision": "d" * 40},
        "model": {
            "revision": revision,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "measurement": {"layers": [12, 20, 28]},
        "official_sae": {
            "revision": sae_revision,
            "layers": [12, 20, 28],
            "checkpoint_match": False,
        },
        "interpretation": {
            "primary_outcome": {
                "contrast": (
                    "neutral-context-residual-probe minus "
                    "verdict-adjacent-residual-probe"
                ),
                "contrast_definition": (
                    "mean_realized_violations(neutral-context-residual-probe) - "
                    "mean_realized_violations(verdict-adjacent-residual-probe)"
                ),
                "attack_prevalence": 0.01,
                "audit_budget_fraction": 0.02,
                "ranking_metric": "mean realized_violations",
                "cell_key": "prevalence=0.01|budget=0.02",
            }
        },
        "seal": {
            "manifest": "configs/manifest.json",
            "source_files": ["configs/qwen35.json", "artifact.txt"],
        },
    }


def _fixture_repo(tmp_path: Path, *, pending: bool = False) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "src/observerbench").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "artifact.txt").write_text("frozen\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0'\n", encoding="utf-8"
    )
    config_path = repo / "configs/qwen35.json"
    config_path.write_text(json.dumps(_config(pending=pending)), encoding="utf-8")
    return repo, config_path


def test_archive_builder_packages_only_verified_sealed_sources(tmp_path: Path) -> None:
    repo, config_path = _fixture_repo(tmp_path)
    seal_qwen35_apps_sources(config_path)
    output = tmp_path / "qwen35-apps.tar.gz"

    archive_manifest = _archive_builder()(repo, output, config_path)

    payload = json.loads(archive_manifest.read_text(encoding="utf-8"))
    assert payload["archive_sha256"] == file_sha256(output)
    assert payload["model_revision"] == "a" * 40
    assert payload["sae_revision"] == "b" * 40
    assert set(payload["files"]) == {
        "artifact.txt",
        "configs/manifest.json",
        "configs/qwen35.json",
    }
    with tarfile.open(output, "r:gz") as bundle:
        assert set(bundle.getnames()) == set(payload["files"])


def test_archive_builder_is_deterministic(tmp_path: Path) -> None:
    repo, config_path = _fixture_repo(tmp_path)
    seal_qwen35_apps_sources(config_path)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    _archive_builder()(repo, first, config_path)
    _archive_builder()(repo, second, config_path)

    assert file_sha256(first) == file_sha256(second)


def test_archive_builder_refuses_pending_revisions_before_manifest_lookup(
    tmp_path: Path,
) -> None:
    repo, config_path = _fixture_repo(tmp_path, pending=True)

    with pytest.raises(ValueError, match="PENDING_REVISION"):
        _archive_builder()(repo, tmp_path / "qwen35-apps.tar.gz", config_path)


def test_archive_builder_rejects_changed_source_after_sealing(tmp_path: Path) -> None:
    repo, config_path = _fixture_repo(tmp_path)
    seal_qwen35_apps_sources(config_path)
    (repo / "artifact.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sources changed"):
        _archive_builder()(repo, tmp_path / "qwen35-apps.tar.gz", config_path)
