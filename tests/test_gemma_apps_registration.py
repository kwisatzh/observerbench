"""Registration and matched-design tests for the Gemma APPS replication.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observerbench.tasks.gemma_apps.registration import (
    load_gemma_apps_config,
    seal_gemma_apps_sources,
    verify_gemma_apps_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
GEMMA_CONFIG = ROOT / "configs/revision/ai_control/controlarena_apps_gemma2_9b_it_v0.json"
QWEN_CONFIG = ROOT / "configs/revision/ai_control/controlarena_apps_qwen2_5_7b_v0.json"
QWEN_FOLLOWUP = ROOT / "configs/revision/ai_control/controlarena_apps_qwen_followup_v1.json"


def test_gemma_design_matches_qwen_rows_controller_budgets_and_prevalence() -> None:
    gemma = load_gemma_apps_config(GEMMA_CONFIG)
    qwen = json.loads(QWEN_CONFIG.read_text())
    followup = json.loads(QWEN_FOLLOWUP.read_text())

    assert gemma["dataset"] == qwen["dataset"]
    parent_design = ROOT / gemma["matched_parent_design"]["path"]
    assert parent_design.is_file()
    assert gemma["matched_parent_design"]["file_sha256"] == (
        "9c0f56b4873bc322da3f0c8a792270c95907d4f3b479f18bec3abcd41bce8dd6"
    )
    assert gemma["protocol"] == qwen["protocol"]
    assert gemma["measurement"]["prompt_variants"][:2] == qwen["measurement"]["prompt_variants"]
    assert gemma["measurement"]["projection_dim"] == qwen["measurement"]["projection_dim"]
    assert gemma["measurement"]["projection_seed"] == qwen["measurement"]["projection_seed"]
    assert gemma["measurement"]["ridge_grid"] == qwen["measurement"]["ridge_grid"]
    assert gemma["sensitivity"]["prevalences"] == followup["sensitivity"]["prevalences"]
    assert gemma["sensitivity"]["audit_budget_fractions"] == followup["sensitivity"]["audit_budget_fractions"]
    assert gemma["sensitivity"]["population_size"] == followup["sensitivity"]["population_size"]
    assert gemma["sensitivity"]["replicates"] == followup["sensitivity"]["replicates"]
    assert gemma["sensitivity"]["seed"] == followup["sensitivity"]["seed"]
    assert (
        gemma["observers"]["supervised_text_dimension"]
        == followup["controls"]["supervised_text"]["dimension"]
    )


def test_config_pins_exact_gemma_and_three_official_saes() -> None:
    config = load_gemma_apps_config(GEMMA_CONFIG)
    assert config["model"]["id"] == "google/gemma-2-9b-it"
    assert len(config["model"]["revision"]) == 40
    assert config["gemma_scope"]["repository"] == "google/gemma-scope-9b-it-res"
    assert len(config["gemma_scope"]["revision"]) == 40
    assert [row["layer"] for row in config["gemma_scope"]["saes"]] == [9, 20, 31]
    assert all(row["width"] == 16384 for row in config["gemma_scope"]["saes"])
    assert config["compatibility_smoke"]["evaluate_monitor_quality"] is False
    assert "individual fit or calibration rows" in config["measurement"]["truncation_policy"]
    assert config["interpretation"]["primary_outcome"] == {
        "attack_prevalence": 0.01,
        "audit_budget_fraction": 0.02,
        "ranking_metric": "mean realized_violations",
        "cell_key": "prevalence=0.01|budget=0.02",
    }


def test_local_source_seal_rejects_a_changed_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src/observerbench").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "artifact.txt").write_text("frozen\n")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")
    config_path = repo / "configs/gemma.json"
    config = {
        "schema": "observerbench.controlarena_apps_gemma.v0",
        "status": "prospectively_frozen_before_gemma_monitor_outcomes",
        "dataset": {"revision": "d" * 40},
        "model": {"revision": "m" * 40},
        "gemma_scope": {"revision": "s" * 40},
        "seal": {
            "manifest": "configs/manifest.json",
            "source_files": ["configs/gemma.json", "artifact.txt"],
        },
    }
    config_path.write_text(json.dumps(config))
    manifest_path = repo / "configs/manifest.json"

    seal_gemma_apps_sources(config_path, manifest_path)
    verify_gemma_apps_source_manifest(config_path, manifest_path)
    with pytest.raises(FileExistsError, match="already exists"):
        seal_gemma_apps_sources(config_path, manifest_path)
    (repo / "artifact.txt").write_text("changed\n")

    with pytest.raises(ValueError, match="sources changed"):
        verify_gemma_apps_source_manifest(config_path, manifest_path)
