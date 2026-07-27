"""Tests for model-free Qwen induction artifact serialization.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from observerbench.provenance import file_sha256
from observerbench.tasks.qwen_induction.artifacts import (
    PRESELECTION_MANIFEST_SCHEMA,
    QWEN_INDUCTION_SCIENTIFIC_CONFIG_SHA256,
    TOKEN_BANKS_SCHEMA,
    load_phase09_config,
    validate_effect_artifacts,
    validate_exact_scientific_config,
    validate_phase09_config,
    write_effect_artifacts,
    write_frozen_design_artifacts,
    write_preselection_artifacts,
)
from observerbench.tasks.qwen_induction.design import (
    SEQUENCE_BANKS,
    SEQUENCE_FAMILIES,
    SequenceBankCounts,
    build_mask_design,
    build_sequence_design,
)
from observerbench.tasks.qwen_induction.effect_task import (
    QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS,
    QWEN_INDUCTION_MODEL_NAME,
    QWEN_INDUCTION_MODEL_REVISION,
    load_qwen_induction_effect_prediction_task,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = (
    REPO_ROOT / "configs/revision/phase09/qwen2_5_7b_induction_full.json"
)
SMOKE_CONFIG = (
    REPO_ROOT / "configs/revision/phase09/qwen2_5_0_5b_induction_smoke.json"
)
HEAD_LABELS = tuple(f"L{4 + 2 * index}H{index}" for index in range(8))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _small_scientific_config() -> dict[str, object]:
    config = load_phase09_config(FULL_CONFIG, require_scientific=True)
    config["token_pool"]["per_split_size"] = 64
    for bank in config["sequence_design"]["prompts_per_family"]:
        config["sequence_design"]["prompts_per_family"][bank] = 1
    validate_phase09_config(config, require_scientific=True)
    return config


def _designs(config: dict[str, object]):
    counts = SequenceBankCounts(
        reference=1,
        discovery=1,
        head_fit=1,
        head_test=1,
        calibration=1,
        locked_test=1,
    )
    sequence = build_sequence_design(
        range(10_000, 11_000),
        bank_counts=counts,
        per_split_size=config["token_pool"]["per_split_size"],
        seed=config["sequence_design"]["seed"],
    )
    masks = build_mask_design(
        HEAD_LABELS,
        seed=config["mask_design"]["seed"],
    )
    selected_heads = [
        {
            "component_index": index,
            "head_label": label,
            "layer": 4 + 2 * index,
            "head": index,
            "kv_group": index % 4,
        }
        for index, label in enumerate(HEAD_LABELS)
    ]
    return sequence, masks, selected_heads


def _write_design(root: Path):
    config = _small_scientific_config()
    sequence, masks, selected_heads = _designs(config)
    write_preselection_artifacts(
        sequence, config, root, exact_scientific_config=False
    )
    proof_dir = root / "test_gate_proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)
    discovery_gate = proof_dir / "discovery_gate.json"
    confirmation_gate = proof_dir / "confirmation_gate.json"
    discovery_gate.write_text('{"passed": true}\n', encoding="utf-8")
    confirmation_gate.write_text('{"passed": true}\n', encoding="utf-8")
    write_frozen_design_artifacts(
        masks,
        selected_heads,
        config,
        root,
        all_design_gates_pass=True,
        gate_artifacts={
            "discovery_gate": discovery_gate,
            "confirmation_gate": confirmation_gate,
        },
        exact_scientific_config=False,
    )
    return config, sequence, masks


def _effect(bits: str, prompt_index: int) -> float:
    linear = sum((index + 1) * 0.02 * int(bit) for index, bit in enumerate(bits))
    pair = 0.015 * int(bits[1]) * int(bits[6])
    return linear + pair + 0.001 * prompt_index * sum(map(int, bits))


def _measurement_rows(root: Path) -> list[dict[str, object]]:
    prompts = _read_csv(root / "design/prompts.csv")
    masks_by_split = {
        "train": _read_csv(root / "design/calibration_masks.csv"),
        "test": _read_csv(root / "design/test_masks.csv"),
    }
    rows: list[dict[str, object]] = []
    for prompt_index, prompt in enumerate(prompts):
        clean = 3.0 + prompt_index * 0.01
        for mask in masks_by_split[prompt["split"]]:
            effect = _effect(mask["mask_bits"], prompt_index)
            rows.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "mask_id": mask["mask_id"],
                    "mask_bits": mask["mask_bits"],
                    "clean_margin": clean,
                    "ablated_margin": clean - effect,
                    "drop_from_clean": effect,
                }
            )
    return rows


def test_phase09_loaders_accept_frozen_full_and_engineering_smoke_configs() -> None:
    full = load_phase09_config(FULL_CONFIG, require_scientific=True)
    smoke = load_phase09_config(SMOKE_CONFIG)

    assert full["model"]["id"] == QWEN_INDUCTION_MODEL_NAME
    assert full["model"]["revision"] == QWEN_INDUCTION_MODEL_REVISION
    assert full["model"]["expected_kv_heads"] == 4
    assert smoke["status"] == "engineering_smoke_only"
    assert validate_exact_scientific_config(full) is full
    assert len(QWEN_INDUCTION_SCIENTIFIC_CONFIG_SHA256) == 64
    changed = deepcopy(full)
    changed["clean_gate"]["minimum_candidate_accuracy_overall"] = 0.90
    with pytest.raises(ValueError, match="frozen production digest"):
        validate_exact_scientific_config(changed)
    with pytest.raises(ValueError, match="scientific Phase-09"):
        load_phase09_config(SMOKE_CONFIG, require_scientific=True)


def test_engineering_smoke_preselection_requires_explicit_opt_out(
    tmp_path: Path,
) -> None:
    smoke = load_phase09_config(SMOKE_CONFIG)
    smoke["token_pool"]["per_split_size"] = 64
    smoke["sequence_design"]["families"] = [
        {
            "sequence_length": family.sequence_length,
            "repeat_gap": family.induction_gap,
        }
        for family in SEQUENCE_FAMILIES
    ]
    for bank in smoke["sequence_design"]["prompts_per_family"]:
        smoke["sequence_design"]["prompts_per_family"][bank] = 1
    counts = SequenceBankCounts(
        reference=1,
        discovery=1,
        head_fit=1,
        head_test=1,
        calibration=1,
        locked_test=1,
    )
    sequence = build_sequence_design(
        range(20_000, 21_000),
        bank_counts=counts,
        per_split_size=64,
        seed=smoke["sequence_design"]["seed"],
    )

    with pytest.raises(ValueError, match="scientific Phase-09"):
        write_preselection_artifacts(sequence, smoke, tmp_path)
    manifest_path = write_preselection_artifacts(
        sequence,
        smoke,
        tmp_path,
        require_scientific=False,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"]["requested_name"] == "Qwen/Qwen2.5-0.5B"
    assert manifest["scientific_outcomes_included"] is False


def test_design_bridge_preserves_all_prompts_masks_pools_and_hashes(
    tmp_path: Path,
) -> None:
    config, sequence, masks = _write_design(tmp_path)
    design_dir = tmp_path / "design"

    prompts_all = _read_csv(design_dir / "prompts_all.csv")
    prompts = _read_csv(design_dir / "prompts.csv")
    calibration = _read_csv(design_dir / "calibration_masks.csv")
    test = _read_csv(design_dir / "test_masks.csv")
    token_banks = json.loads((design_dir / "token_banks.json").read_text())
    preselection = json.loads(
        (design_dir / "preselection_manifest.json").read_text()
    )
    manifest = json.loads((design_dir / "design_manifest.json").read_text())

    assert len(prompts_all) == len(SEQUENCE_BANKS) * len(SEQUENCE_FAMILIES)
    assert len(prompts) == 2 * len(SEQUENCE_FAMILIES)
    assert {row["split"] for row in prompts} == {"train", "test"}
    assert {
        row["bank"] for row in prompts_all if row["prompt_id"] in {
            prompt["prompt_id"] for prompt in prompts if prompt["split"] == "train"
        }
    } == {"calibration"}
    assert {
        row["bank"] for row in prompts_all if row["prompt_id"] in {
            prompt["prompt_id"] for prompt in prompts if prompt["split"] == "test"
        }
    } == {"locked_test"}

    assert token_banks["schema"] == TOKEN_BANKS_SCHEMA
    assert len(token_banks["banks"]) == len(SEQUENCE_BANKS)
    assert len({bank["token_bank_id"] for bank in token_banks["banks"]}) == 6
    assert preselection["schema"] == PRESELECTION_MANIFEST_SCHEMA
    assert preselection["scientific_outcomes_included"] is False
    assert preselection["sequence_design_sha256"] == sequence.design_sha256

    assert len(calibration) == len(test) == 128
    assert calibration[0]["mask_bits"] == "00000000"
    assert {row["mask_bits"] for row in calibration}.isdisjoint(
        {row["mask_bits"] for row in test}
    )
    assert len(
        {row["mask_bits"] for row in calibration + test}
    ) == 256
    assert [int(row["measurement_order"]) for row in calibration] == list(
        range(1, 129)
    )

    expected_pool_by_mask = {
        mask_id: pool.pool_id
        for pool in masks.action_pools
        for mask_id in pool.mask_ids[1:]
    }
    assert {row["mask_id"]: row["pool_id"] for row in test} == (
        expected_pool_by_mask
    )
    assert len({row["pool_id"] for row in test}) == 16
    assert {sum(row["pool_id"] == pool for row in test) for pool in {
        row["pool_id"] for row in test
    }} == {8}

    assert manifest["mask_design_sha256"] == masks.design_sha256
    assert manifest["measurement_budgets"] == list(
        QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS
    )
    assert manifest["model"] == {
        "requested_name": QWEN_INDUCTION_MODEL_NAME,
        "requested_revision": QWEN_INDUCTION_MODEL_REVISION,
    }
    assert manifest["config_sha256"]
    for label, digest in manifest["artifact_hashes"].items():
        assert file_sha256(design_dir / label) == digest
    validate_phase09_config(config, require_scientific=True)


def test_effect_bridge_reconstructs_shards_and_loads_public_task(
    tmp_path: Path,
) -> None:
    config, _sequence, masks = _write_design(tmp_path)
    effect_manifest_path = write_effect_artifacts(
        _measurement_rows(tmp_path),
        config,
        tmp_path,
        shard_size=100,
        exact_scientific_config=False,
    )

    validate_effect_artifacts(tmp_path)
    manifest = json.loads(effect_manifest_path.read_text())
    assert manifest["measurement_rows"] == {"train": 512, "test": 512}
    assert len(manifest["artifacts"]) == 14
    assert len(list((tmp_path / "effects/shards/train").glob("*.csv"))) == 6
    assert len(list((tmp_path / "effects/shards/test").glob("*.csv"))) == 6

    task = load_qwen_induction_effect_prediction_task(
        tmp_path,
        measurement_budget=128,
        verify_hashes=True,
    )
    assert len(task.measurements) == 128
    assert len(task.queries) == len(task.targets) == 128
    assert task.card.metadata["underlying_train_effect_cells"] == 512
    assert task.card.metadata["underlying_test_effect_cells"] == 512
    assert task.card.metadata["model_revision"] == QWEN_INDUCTION_MODEL_REVISION
    assert {query.features.mask_id for query in task.queries} == {
        mask.mask_id for mask in masks.heldout_masks
    }
    assert {query.features.candidate_pool for query in task.queries} == {
        pool.pool_id for pool in masks.action_pools
    }

    no_op = task.measurements[0]
    assert no_op.features.mask_bits == "00000000"
    assert no_op.observed_effect == pytest.approx(0.0)


def test_effect_validation_rejects_a_modified_shard(tmp_path: Path) -> None:
    config, _sequence, _masks = _write_design(tmp_path)
    write_effect_artifacts(
        _measurement_rows(tmp_path),
        config,
        tmp_path,
        shard_size=200,
        exact_scientific_config=False,
    )
    shard = sorted((tmp_path / "effects/shards/test").glob("*.csv"))[0]
    shard.write_text(shard.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_effect_artifacts(tmp_path)
