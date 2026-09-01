"""Registration and matched-design tests for the Qwen3.5 APPS replication.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observerbench.tasks.qwen35_apps.registration import (
    load_qwen35_apps_config,
    require_resolved_revisions,
    seal_qwen35_apps_sources,
    unresolved_revision_fields,
    verify_qwen35_apps_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
QWEN35_CONFIG = (
    ROOT / "configs/revision/ai_control/controlarena_apps_qwen3_5_9b_v0.json"
)
QWEN_CONFIG = ROOT / "configs/revision/ai_control/controlarena_apps_qwen2_5_7b_v0.json"
QWEN_FOLLOWUP = ROOT / "configs/revision/ai_control/controlarena_apps_qwen_followup_v1.json"


def _fixture_config(*, pending: bool = False) -> dict[str, object]:
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


def _fixture_repo(tmp_path: Path, *, pending: bool = False) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "src/observerbench").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "artifact.txt").write_text("frozen\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0'\n", encoding="utf-8"
    )
    config_path = repo / "configs/qwen35.json"
    config_path.write_text(json.dumps(_fixture_config(pending=pending)), encoding="utf-8")
    return repo, config_path, repo / "configs/manifest.json"


def test_qwen35_design_matches_frozen_apps_contract() -> None:
    config = load_qwen35_apps_config(QWEN35_CONFIG)
    qwen = json.loads(QWEN_CONFIG.read_text(encoding="utf-8"))
    followup = json.loads(QWEN_FOLLOWUP.read_text(encoding="utf-8"))

    assert config["task_id"] == "controlarena-apps-verified-backdoors@qwen3.5-9b-v0"
    assert config["dataset"] == qwen["dataset"]
    assert config["protocol"] == qwen["protocol"]
    assert config["measurement"]["prompt_variants"][:2] == qwen["measurement"]["prompt_variants"]
    assert config["measurement"]["layers"] == [12, 20, 28]
    assert config["measurement"]["projection_dim"] == qwen["measurement"]["projection_dim"]
    assert config["measurement"]["projection_seed"] == qwen["measurement"]["projection_seed"]
    assert config["measurement"]["ridge_grid"] == qwen["measurement"]["ridge_grid"]
    for key in (
        "prevalences",
        "audit_budget_fractions",
        "population_size",
        "replicates",
        "seed",
    ):
        assert config["sensitivity"][key] == followup["sensitivity"][key]


def test_qwen35_freezes_thinking_layers_primary_contrast_and_sae_mismatch() -> None:
    config = load_qwen35_apps_config(QWEN35_CONFIG)

    assert config["model"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert config["official_sae"]["layers"] == [12, 20, 28]
    assert config["official_sae"]["trained_model_id"] == "Qwen/Qwen3.5-9B-Base"
    assert config["official_sae"]["evaluated_model_id"] == "Qwen/Qwen3.5-9B"
    assert config["official_sae"]["checkpoint_match"] is False
    assert "base-to-posttrain mismatch" in config["official_sae"]["mismatch"]
    assert config["interpretation"]["primary_outcome"]["attack_prevalence"] == 0.01
    assert config["interpretation"]["primary_outcome"]["audit_budget_fraction"] == 0.02
    assert config["interpretation"]["primary_outcome"]["contrast"].startswith(
        "neutral-context-residual-probe minus"
    )
    assert unresolved_revision_fields(config) == ()
    require_resolved_revisions(config)
    assert config["model"]["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert config["official_sae"]["revision"] == (
        "7bb2370d360e566b2d8acb8b3d15a0b2b88f52a8"
    )


def test_source_seal_requires_exact_commit_hashes(tmp_path: Path) -> None:
    _repo, config_path, manifest_path = _fixture_repo(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["revision"] = "main"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="exact 40-character commit hashes"):
        seal_qwen35_apps_sources(config_path, manifest_path)


def test_local_source_seal_verifies_then_rejects_a_changed_file(tmp_path: Path) -> None:
    repo, config_path, manifest_path = _fixture_repo(tmp_path)

    seal_qwen35_apps_sources(config_path, manifest_path)
    verify_qwen35_apps_source_manifest(config_path, manifest_path)
    with pytest.raises(FileExistsError, match="already exists"):
        seal_qwen35_apps_sources(config_path, manifest_path)
    (repo / "artifact.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sources changed"):
        verify_qwen35_apps_source_manifest(config_path, manifest_path)
