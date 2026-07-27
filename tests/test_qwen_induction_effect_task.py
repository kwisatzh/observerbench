# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from observerbench import run_effect_prediction_task
from observerbench.provenance import file_sha256
from observerbench.tasks import (
    finite_effect_measurement_budgets,
    finite_effect_task_ids,
    finite_effect_task_versions,
    load_finite_effect_task,
)
from observerbench.tasks.qwen_induction.effect_task import (
    InductionMaskFeatures,
    QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS,
    QWEN_INDUCTION_EFFECT_TASK_NAME,
    QWEN_INDUCTION_MODEL_NAME,
    QWEN_INDUCTION_MODEL_REVISION,
    QwenInductionAdditiveRidgeBaseline,
    load_qwen_induction_effect_prediction_task,
    qwen_induction_additive_baseline_card,
    qwen_induction_effect_task_version,
)


SELECTED_HEAD_FIELDS = [
    "component_index",
    "head_label",
    "layer",
    "head",
    "kv_group",
]
CALIBRATION_MASK_FIELDS = [
    "measurement_order",
    "mask_id",
    "mask_bits",
    "n_heads",
    "bank",
    "pool_id",
]
TEST_MASK_FIELDS = [
    "mask_id",
    "mask_bits",
    "n_heads",
    "bank",
    "pool_id",
]
PROMPT_FIELDS = [
    "prompt_id",
    "split",
    "family_id",
    "cluster_id",
    "token_bank_id",
    "input_ids",
    "target_token_id",
    "distractor_token_id_1",
    "distractor_token_id_2",
    "source_key_position",
    "source_value_position",
    "query_position",
    "sequence_length",
    "repeat_gap",
]
EFFECT_FIELDS = [
    "schema_version",
    "prompt_id",
    "split",
    "family_id",
    "cluster_id",
    "mask_id",
    "mask_bits",
    "bank",
    "pool_id",
    "clean_margin",
    "ablated_margin",
    "drop_from_clean",
]


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def _mask_row(
    value: int,
    *,
    order: int | None,
    bank: str,
    pool_id: str = "",
) -> dict[str, object]:
    bits = f"{value:08b}"
    return {
        "measurement_order": "" if order is None else order,
        "mask_id": f"mask_{value:03d}",
        "mask_bits": bits,
        "n_heads": bits.count("1"),
        "bank": bank,
        "pool_id": pool_id,
    }


def _effect(bits: str) -> float:
    coefficients = tuple(0.025 * (index + 1) for index in range(8))
    return sum(coefficient * int(bit) for coefficient, bit in zip(coefficients, bits))


def _make_artifacts(root: Path) -> Path:
    design = root / "design"
    effects = root / "effects"

    selected_heads = [
        {
            "component_index": index,
            "head_label": f"L{4 + 2 * index}H{index}",
            "layer": 4 + 2 * index,
            "head": index,
            "kv_group": index // 2,
        }
        for index in range(8)
    ]
    _write_csv(design / "selected_heads.csv", selected_heads, SELECTED_HEAD_FIELDS)

    anchors = [0, *(1 << shift for shift in range(8))]
    calibration_values = anchors + [
        value for value in range(256) if value not in anchors
    ][: 128 - len(anchors)]
    test_values = [value for value in range(256) if value not in calibration_values]
    calibration = [
        _mask_row(value, order=index + 1, bank="calibration")
        for index, value in enumerate(calibration_values)
    ]
    test = [
        _mask_row(
            value,
            order=None,
            bank="test",
            pool_id=f"pool_{index // 8:02d}",
        )
        for index, value in enumerate(test_values)
    ]
    _write_csv(
        design / "calibration_masks.csv",
        calibration,
        CALIBRATION_MASK_FIELDS,
    )
    _write_csv(design / "test_masks.csv", test, TEST_MASK_FIELDS)

    prompts = []
    for split, token_offset in (("train", 0), ("test", 100)):
        for index in range(2):
            input_ids = [
                10 + token_offset + index,
                20 + token_offset + index,
                30 + token_offset + index,
                31 + token_offset + index,
                40 + token_offset + index,
                41 + token_offset + index,
                50 + token_offset + index,
                20 + token_offset + index,
            ]
            prompts.append(
                {
                    "prompt_id": f"prompt_{split}_{index}",
                    "split": split,
                    "family_id": f"length_08_gap_06_{index}",
                    "cluster_id": f"cluster_{split}_{index}",
                    "token_bank_id": f"bank_{split}",
                    "input_ids": " ".join(map(str, input_ids)),
                    "target_token_id": 30 + token_offset + index,
                    "distractor_token_id_1": 40 + token_offset + index,
                    "distractor_token_id_2": 50 + token_offset + index,
                    "source_key_position": 1,
                    "source_value_position": 2,
                    "query_position": 7,
                    "sequence_length": len(input_ids),
                    "repeat_gap": 6,
                }
            )
    _write_csv(design / "prompts.csv", prompts, PROMPT_FIELDS)

    gate_dir = design / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    discovery_gate = gate_dir / "discovery_gate.json"
    confirmation_gate = gate_dir / "confirmation_gate.json"
    discovery_gate.write_text('{"passed": true}\n', encoding="utf-8")
    confirmation_gate.write_text('{"passed": true}\n', encoding="utf-8")

    design_artifacts = (
        design / "selected_heads.csv",
        design / "calibration_masks.csv",
        design / "test_masks.csv",
        design / "prompts.csv",
        discovery_gate,
        confirmation_gate,
    )
    design_manifest = {
        "schema": "observerbench.qwen_induction_design_manifest.v1",
        "status": "frozen_before_outcomes",
        "design_id": "qwen_induction_fixture_v1",
        "data_version": "copy-v1",
        "all_design_gates_pass": True,
        "measurement_budgets": list(QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS),
        "model": {
            "requested_name": QWEN_INDUCTION_MODEL_NAME,
            "requested_revision": QWEN_INDUCTION_MODEL_REVISION,
        },
        "gate_artifact_hashes": {
            "discovery_gate": {
                "path": "gates/discovery_gate.json",
                "sha256": file_sha256(discovery_gate),
            },
            "confirmation_gate": {
                "path": "gates/confirmation_gate.json",
                "sha256": file_sha256(confirmation_gate),
            },
        },
        "artifact_hashes": {
            path.relative_to(design).as_posix(): file_sha256(path)
            for path in design_artifacts
        },
    }
    (design / "design_manifest.json").write_text(
        json.dumps(design_manifest, sort_keys=True),
        encoding="utf-8",
    )

    effect_paths = []
    for split, masks in (("train", calibration), ("test", test)):
        rows = []
        for prompt_index, prompt in enumerate(
            row for row in prompts if row["split"] == split
        ):
            clean = 3.0 + 0.1 * prompt_index
            for mask in masks:
                effect = _effect(str(mask["mask_bits"]))
                rows.append(
                    {
                        "schema_version": "observerbench.qwen_induction_effect_rows.v1",
                        "prompt_id": prompt["prompt_id"],
                        "split": split,
                        "family_id": prompt["family_id"],
                        "cluster_id": prompt["cluster_id"],
                        "mask_id": mask["mask_id"],
                        "mask_bits": mask["mask_bits"],
                        "bank": mask["bank"],
                        "pool_id": mask["pool_id"],
                        "clean_margin": repr(clean),
                        "ablated_margin": repr(clean - effect),
                        "drop_from_clean": repr(effect),
                    }
                )
        path = effects / f"{split}_effects.csv"
        _write_csv(path, rows, EFFECT_FIELDS)
        effect_paths.append(path)

    effect_manifest = {
        "schema": "observerbench.qwen_induction_effect_run.v1",
        "status": "complete_locked_test_outcomes",
        "design_manifest_sha256": file_sha256(design / "design_manifest.json"),
        "model": {
            "requested_name": QWEN_INDUCTION_MODEL_NAME,
            "requested_revision": QWEN_INDUCTION_MODEL_REVISION,
        },
        "intervention": {
            "site": "final_query_head_z",
            "replacement": "family_conditioned_reference_mean",
            "n_selected_heads": 8,
        },
        "artifacts": {
            path.name: file_sha256(path) for path in effect_paths
        },
    }
    (effects / "effect_manifest.json").write_text(
        json.dumps(effect_manifest, sort_keys=True),
        encoding="utf-8",
    )
    return root


def test_versions_are_exact_and_reject_unknown_budget() -> None:
    assert tuple(
        qwen_induction_effect_task_version(budget)
        for budget in QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS
    ) == (
        "copy-v1-b016",
        "copy-v1-b040",
        "copy-v1-b064",
        "copy-v1-b128",
    )
    with pytest.raises(ValueError, match="unsupported Qwen induction measurement budget"):
        qwen_induction_effect_task_version(32)


def test_closed_registry_exposes_qwen_versions_and_loads_tables(tmp_path: Path) -> None:
    versions = tuple(
        qwen_induction_effect_task_version(budget)
        for budget in QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS
    )
    assert finite_effect_task_versions(QWEN_INDUCTION_EFFECT_TASK_NAME) == versions
    assert finite_effect_measurement_budgets(QWEN_INDUCTION_EFFECT_TASK_NAME) == (
        QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS
    )
    task_id = f"{QWEN_INDUCTION_EFFECT_TASK_NAME}@{versions[0]}"
    assert task_id in finite_effect_task_ids()
    root = _make_artifacts(tmp_path / "qwen_registry")
    task = load_finite_effect_task(task_id, artifacts_root=root)
    assert task.name == QWEN_INDUCTION_EFFECT_TASK_NAME
    assert task.measurement_budget == 16


def test_checked_loader_exposes_eight_head_features_and_targets(tmp_path: Path) -> None:
    root = _make_artifacts(tmp_path / "qwen")
    task = load_qwen_induction_effect_prediction_task(
        root,
        measurement_budget=16,
    )

    assert task.name == QWEN_INDUCTION_EFFECT_TASK_NAME
    assert task.version == "copy-v1-b016"
    assert task.measurement_budget == 16
    assert len(task.queries) == len(task.targets) == 128
    assert isinstance(task.measurements[0].features, InductionMaskFeatures)
    assert task.measurements[0].features.mask_bits == "00000000"
    assert task.measurements[0].features.n_heads == 0
    assert len(task.measurements[0].features.head_labels) == 8
    assert task.measurements[0].metadata["n_train_prompts"] == 2
    assert task.queries[0].metadata["n_test_prompts"] == 2
    assert task.card.metadata["model_inference_included"] is False
    assert task.card.metadata["model_revision"] == QWEN_INDUCTION_MODEL_REVISION
    targets = {target.query_id: target.observed_effect for target in task.targets}
    for query in task.queries:
        assert targets[query.query_id] == pytest.approx(_effect(query.features.mask_bits))


@pytest.mark.parametrize("budget", QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS)
def test_all_frozen_budgets_load(tmp_path: Path, budget: int) -> None:
    root = _make_artifacts(tmp_path / f"qwen_{budget}")
    task = load_qwen_induction_effect_prediction_task(root, measurement_budget=budget)
    assert task.measurement_budget == budget
    assert task.card.metadata["underlying_train_effect_cells"] == 2 * budget


def test_additive_baseline_recovers_linear_fixture_without_model_runtime(
    tmp_path: Path,
) -> None:
    root = _make_artifacts(tmp_path / "qwen")
    task = load_qwen_induction_effect_prediction_task(root, measurement_budget=16)
    predictor = QwenInductionAdditiveRidgeBaseline(ridge=1e-10)

    result = run_effect_prediction_task(
        task,
        predictor,
        qwen_induction_additive_baseline_card(ridge=1e-10),
        outdir=tmp_path / "baseline",
    )

    assert result.n_queries == 128
    assert result.metrics["mae"] < 1e-8
    assert (tmp_path / "baseline" / "effect_predictions.csv").is_file()
    assert (tmp_path / "baseline" / "task_card.json").is_file()


def test_checked_loader_rejects_tampered_effect_table(tmp_path: Path) -> None:
    root = _make_artifacts(tmp_path / "qwen")
    path = root / "effects" / "test_effects.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen artifact hash mismatch"):
        load_qwen_induction_effect_prediction_task(root, measurement_budget=16)


def test_checked_loader_rejects_wrong_model_revision(tmp_path: Path) -> None:
    root = _make_artifacts(tmp_path / "qwen")
    manifest_path = root / "effects" / "effect_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["requested_revision"] = "wrong-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned base model"):
        load_qwen_induction_effect_prediction_task(root, measurement_budget=16)
