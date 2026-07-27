"""Tests for the sealed Copy-v2 Colab source archive.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from observerbench.provenance import file_sha256, json_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SCRIPT = REPO_ROOT / "scripts/build_qwen_phase10_source_archive.py"


def _archive_builder():
    spec = importlib.util.spec_from_file_location(
        "observerbench_phase10_archive_builder", ARCHIVE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_archive


def _sealed_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    phase10 = repo / "configs/revision/phase10"
    phase10.mkdir(parents=True)
    (repo / "README.md").write_text("ObserverBench\n", encoding="utf-8")
    producer = repo / "producer.py"
    producer.write_text("VALUE = 1\n", encoding="utf-8")

    copy_v1_path = phase10 / "copy_v1_token_banks.json"
    copy_v1 = {
        "token_pool_size": 2,
        "token_pool_sha256": json_sha256((101, 202)),
    }
    copy_v1_path.write_text(
        json.dumps(copy_v1, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema": "observerbench.qwen_induction_phase10.v1",
        "status": "frozen_before_copy_v2_outcomes",
        "data_version": "copy-v2",
        "token_pool": {
            "exclude_copy_v1_allocated_tokens": {
                "source_artifact": (
                    "configs/revision/phase10/copy_v1_token_banks.json"
                ),
                "source_artifact_sha256": file_sha256(copy_v1_path),
                "source_token_pool_size": copy_v1["token_pool_size"],
                "source_token_pool_sha256": copy_v1["token_pool_sha256"],
            }
        },
    }
    config_path = phase10 / "qwen2_5_7b_induction_copy_v2.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_hashes = {"producer.py": file_sha256(producer)}
    source_manifest_path = phase10 / "qwen_copy_v2_source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema": (
                    "observerbench.qwen_induction_phase10.source_manifest.v1"
                ),
                "status": "frozen_before_copy_v2_outcomes",
                "config_sha256": json_sha256(config),
                "source_bundle_sha256": json_sha256(source_hashes),
                "artifact_hashes": source_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return repo, config_path, source_manifest_path


def test_archive_builder_accepts_only_a_complete_matching_seal(
    tmp_path: Path,
) -> None:
    repo, _config, _manifest = _sealed_repo(tmp_path)
    output = tmp_path / "copy-v2.tar.gz"
    archive_manifest = _archive_builder()(repo, output)

    assert output.is_file()
    payload = json.loads(archive_manifest.read_text(encoding="utf-8"))
    assert payload["archive_sha256"] == file_sha256(output)
    assert set(payload["files"]) == {
        "README.md",
        "configs/revision/phase10/copy_v1_token_banks.json",
        "configs/revision/phase10/qwen2_5_7b_induction_copy_v2.json",
        "configs/revision/phase10/qwen_copy_v2_source_manifest.json",
        "producer.py",
    }


def test_archive_builder_rejects_config_not_bound_by_source_manifest(
    tmp_path: Path,
) -> None:
    repo, config_path, _manifest = _sealed_repo(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["new_unsealed_field"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="source manifest and frozen config differ"):
        _archive_builder()(repo, tmp_path / "copy-v2.tar.gz")


def test_archive_builder_rejects_mismatched_source_bundle_digest(
    tmp_path: Path,
) -> None:
    repo, _config, manifest_path = _sealed_repo(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_bundle_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched bundle digest"):
        _archive_builder()(repo, tmp_path / "copy-v2.tar.gz")


def test_archive_builder_rejects_changed_copy_v1_exclusion_asset(
    tmp_path: Path,
) -> None:
    repo, _config, _manifest = _sealed_repo(tmp_path)
    copy_v1 = repo / "configs/revision/phase10/copy_v1_token_banks.json"
    copy_v1.write_text('{"token_pool_size": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="exclusion artifact differs"):
        _archive_builder()(repo, tmp_path / "copy-v2.tar.gz")
