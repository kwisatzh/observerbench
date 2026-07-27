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
    finite_effect_task_specs,
    finite_effect_task_versions,
    get_finite_effect_task_spec,
    load_finite_effect_task,
)
from observerbench.tasks.ioi import (
    IOI_EFFECT_MEASUREMENT_BUDGETS,
    IOI_EFFECT_MODEL_REVISION,
    IOI_EFFECT_TASK_NAME,
    IOIAdditiveRidgeBaseline,
    IOIMaskFeatures,
    ioi_additive_baseline_card,
    ioi_effect_task_version,
    load_ioi_effect_prediction_task,
)


HEAD_GROUPS = ("P",) * 3 + ("B",) * 8 + ("E",) * 2


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def _mask_row(value: int, *, order: int | None, candidate: bool) -> dict[str, object]:
    bits = f"{value:013b}"
    mask = tuple(int(bit) for bit in bits)
    row: dict[str, object] = {
        "measurement_order": "" if order is None else order,
        "mask_id": f"mask_{value:04d}",
        "mask_bits": bits,
        "n_heads": sum(mask),
        "n_P": sum(bit for bit, group in zip(mask, HEAD_GROUPS) if group == "P"),
        "n_B": sum(bit for bit, group in zip(mask, HEAD_GROUPS) if group == "B"),
        "n_E": sum(bit for bit, group in zip(mask, HEAD_GROUPS) if group == "E"),
        "bank": "candidate" if candidate else "calibration",
        "pool_id": "candidate_pool_00" if candidate else "",
        "pool_index": 0 if candidate else "",
        "sampling_stratum": "broad" if candidate else "",
        "size_match_cell": f"n_heads_{sum(mask):02d}" if candidate else "",
        "within_cell_index": 0 if candidate else "",
    }
    return row


def _effect(bits: str) -> float:
    coefficients = tuple(0.05 * (index + 1) for index in range(13))
    return 0.2 + sum(coef * int(bit) for coef, bit in zip(coefficients, bits))


def _make_artifacts(root: Path) -> Path:
    design = root / "design"
    effects = root / "ioi_effects"
    core_values = [0, *(1 << shift for shift in range(13))]
    calibration_values = core_values + [
        value for value in range(1, 8192) if value not in core_values
    ][: 160 - len(core_values)]
    candidate_values = [7001, 8191]
    calibration = [
        _mask_row(value, order=index + 1, candidate=False)
        for index, value in enumerate(calibration_values)
    ]
    candidates = [
        _mask_row(value, order=None, candidate=True) for value in candidate_values
    ]
    mask_fields = [
        "measurement_order",
        "mask_id",
        "mask_bits",
        "n_heads",
        "n_P",
        "n_B",
        "n_E",
        "bank",
        "pool_id",
        "pool_index",
        "sampling_stratum",
        "size_match_cell",
        "within_cell_index",
    ]
    _write_csv(design / "calibration_masks.csv", calibration, mask_fields[:7])
    _write_csv(design / "candidate_masks.csv", candidates, mask_fields[1:])
    _write_csv(design / "masks.csv", [*calibration, *candidates], mask_fields)

    prompts = [
        {
            "prompt_id": f"prompt_{split}_{index}",
            "split": split,
            "template_id": "abba_fixture",
            "structure": "ABBA",
        }
        for split in ("train", "test")
        for index in range(2)
    ]
    _write_csv(
        design / "prompts.csv",
        prompts,
        ["prompt_id", "split", "template_id", "structure"],
    )
    design_hashes = {
        path.name: file_sha256(path)
        for path in (
            design / "calibration_masks.csv",
            design / "candidate_masks.csv",
            design / "masks.csv",
            design / "prompts.csv",
        )
    }
    design_manifest = {
        "schema": "observerbench.ioi_phase5_design_manifest.v1",
        "status": "frozen_before_outcomes",
        "design_id": "ioi_fixture_v1",
        "all_design_gates_pass": True,
        "artifact_hashes": design_hashes,
    }
    (design / "design_manifest.json").write_text(
        json.dumps(design_manifest, sort_keys=True),
        encoding="utf-8",
    )

    effect_fields = [
        "schema_version",
        "prompt_id",
        "split",
        "template_id",
        "structure",
        "mask_id",
        "mask_bits",
        "bank",
        "pool_id",
        "clean_ld",
        "ablated_ld",
        "drop_from_clean",
    ]
    effect_paths = []
    for split in ("train", "test"):
        rows = []
        for prompt in (row for row in prompts if row["split"] == split):
            for mask in (*calibration, *candidates):
                value = _effect(str(mask["mask_bits"]))
                rows.append(
                    {
                        "schema_version": "observerbench.ioi_effect_rows.v1",
                        **prompt,
                        "mask_id": mask["mask_id"],
                        "mask_bits": mask["mask_bits"],
                        "bank": mask["bank"],
                        "pool_id": mask["pool_id"],
                        "clean_ld": 2.0,
                        "ablated_ld": 2.0 - value,
                        "drop_from_clean": value,
                    }
                )
        path = effects / "shards" / split / "effects_0000_0162.csv"
        _write_csv(path, rows, effect_fields)
        effect_paths.append(path)
    effect_manifest = {
        "schema": "observerbench.ioi_effect_run.v1",
        "status": "complete_unopened_confirmatory_outcomes",
        "design_manifest_sha256": file_sha256(design / "design_manifest.json"),
        "model": {
            "requested_name": "gpt2-small",
            "requested_revision": IOI_EFFECT_MODEL_REVISION,
        },
        "artifacts": {
            path.relative_to(effects).as_posix(): file_sha256(path)
            for path in effect_paths
        },
    }
    (effects / "effect_manifest.json").write_text(
        json.dumps(effect_manifest, sort_keys=True),
        encoding="utf-8",
    )
    return root


def test_registry_exposes_exact_versions_and_budgets() -> None:
    specs = finite_effect_task_specs()
    expected_versions = tuple(
        ioi_effect_task_version(budget) for budget in IOI_EFFECT_MEASUREMENT_BUDGETS
    )

    assert finite_effect_task_ids() == tuple(sorted(finite_effect_task_ids()))
    assert tuple(spec.task_id for spec in specs) == finite_effect_task_ids()
    assert finite_effect_task_versions(IOI_EFFECT_TASK_NAME) == expected_versions
    assert finite_effect_measurement_budgets(IOI_EFFECT_TASK_NAME) == (
        IOI_EFFECT_MEASUREMENT_BUDGETS
    )
    first_ioi = get_finite_effect_task_spec(
        f"{IOI_EFFECT_TASK_NAME}@{ioi_effect_task_version(20)}"
    )
    assert first_ioi.measurement_budget == 20
    with pytest.raises(KeyError, match="unknown finite-effect task ID"):
        get_finite_effect_task_spec("not-a-task@v1")
    with pytest.raises(KeyError, match="unknown finite-effect task"):
        finite_effect_task_versions("not-a-task")


def test_checked_loader_exposes_mask_records_and_heldout_targets(tmp_path: Path) -> None:
    root = _make_artifacts(tmp_path / "phase05")
    task = load_ioi_effect_prediction_task(root, measurement_budget=20)

    assert task.name == IOI_EFFECT_TASK_NAME
    assert task.version == ioi_effect_task_version(20)
    assert task.measurement_budget == 20
    assert len(task.queries) == len(task.targets) == 2
    assert isinstance(task.measurements[0].features, IOIMaskFeatures)
    assert task.measurements[0].features.mask_bits == "0" * 13
    assert task.measurements[0].metadata["n_train_prompts"] == 2
    assert task.queries[0].query_id.startswith("mask_")
    assert task.queries[0].metadata["n_test_prompts"] == 2
    assert task.card.metadata["model_inference_included"] is False
    assert task.card.metadata["measurement_unit"] == "distinct intervention mask"
    target_by_id = {row.query_id: row.observed_effect for row in task.targets}
    for query in task.queries:
        assert target_by_id[query.query_id] == pytest.approx(
            _effect(query.features.mask_bits)
        )


def test_registry_loader_and_additive_baseline_run_without_model_runtime(
    tmp_path: Path,
) -> None:
    root = _make_artifacts(tmp_path / "phase05")
    task_id = f"{IOI_EFFECT_TASK_NAME}@{ioi_effect_task_version(20)}"
    task = load_finite_effect_task(task_id, artifacts_root=root)
    predictor = IOIAdditiveRidgeBaseline(ridge=1e-10)

    result = run_effect_prediction_task(
        task,
        predictor,
        ioi_additive_baseline_card(ridge=1e-10),
        outdir=tmp_path / "baseline",
    )

    assert result.measurement_budget == 20
    assert result.n_queries == 2
    assert result.metrics["mae"] < 1e-8
    assert (tmp_path / "baseline" / "task_card.json").is_file()
    assert (tmp_path / "baseline" / "observer_card.json").is_file()


def test_checked_loader_rejects_tampered_effect_table(tmp_path: Path) -> None:
    root = _make_artifacts(tmp_path / "phase05")
    shard = root / "ioi_effects" / "shards" / "test" / "effects_0000_0162.csv"
    shard.write_text(shard.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen artifact hash mismatch"):
        load_ioi_effect_prediction_task(root, measurement_budget=20)
