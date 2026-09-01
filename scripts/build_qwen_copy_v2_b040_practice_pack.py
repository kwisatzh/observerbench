#!/usr/bin/env python3
"""Build the open, model-free Qwen Copy-v2 budget-40 practice pack."""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


PACK_SCHEMA = "observerbench.open_practice_pack.v1"
TASK_ID = "induction-qwen2-5-7b-finite-effects@copy-v2-b040"
MEASUREMENT_BUDGET = 40
EXPECTED_DATA_VERSION = "copy-v2"
EXPECTED_MODEL = "Qwen/Qwen2.5-7B"
EXPECTED_MODEL_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
EXPECTED_EFFECT_ROW_SCHEMA = "observerbench.qwen_induction_effect_rows.v1"
EXPECTED_DESIGN_SCHEMA = "observerbench.qwen_induction_design_manifest.v1"
EXPECTED_EFFECT_SCHEMA = "observerbench.qwen_induction_effect_run.v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "results"
    / "revision"
    / "phase10"
    / "qwen_induction_copy_v2_complete"
    / "copy_v2"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "practice" / "qwen_copy_v2_b040"

CALIBRATION_FIELDS = (
    "measurement_order",
    "measurement_id",
    "mask_bits",
    "n_heads",
    "observed_mean_effect",
    "n_train_prompts",
    "h0",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "h7",
)
QUERY_FIELDS = (
    "query_id",
    "mask_bits",
    "n_heads",
    "pool_id",
    "h0",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "h7",
)
TARGET_FIELDS = ("query_id", "observed_mean_effect", "n_test_prompts")
ACTION_FIELDS = (
    "target_fraction",
    "target",
    "pool_id",
    "query_id",
    "mask_bits",
    "n_heads",
    "is_noop",
    "actual_target_loss",
    "n_test_prompts",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _verify_file_hashes(
    root: Path,
    hashes: Mapping[str, object],
    required: Iterable[str],
) -> None:
    for label in required:
        expected = str(hashes.get(label, ""))
        relative = Path(label)
        if not expected or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"manifest lacks a safe hash for {label}")
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen source hash mismatch: {path}")


def _checked_source(source_root: Path) -> tuple[Path, Path, dict[str, object]]:
    design_dir = source_root / "design"
    effects_dir = source_root / "effects"
    design_path = design_dir / "design_manifest.json"
    effect_path = effects_dir / "effect_manifest.json"
    design = _read_json(design_path)
    effect = _read_json(effect_path)

    if design.get("schema") != EXPECTED_DESIGN_SCHEMA:
        raise ValueError("unexpected Copy-v2 design schema")
    if design.get("status") != "frozen_before_outcomes":
        raise ValueError("Copy-v2 design is not frozen before outcomes")
    if design.get("data_version") != EXPECTED_DATA_VERSION:
        raise ValueError("practice pack requires Copy-v2")
    if design.get("scientific_claim_allowed") is not True:
        raise ValueError("Copy-v2 design is not eligible for scientific use")
    if design.get("all_design_gates_pass") is not True:
        raise ValueError("Copy-v2 design gates did not pass")
    if design.get("measurement_budgets") != [16, 40, 64, 128]:
        raise ValueError("Copy-v2 nested budgets changed")
    model = design.get("model")
    if not isinstance(model, dict) or (
        model.get("requested_name") != EXPECTED_MODEL
        or model.get("requested_revision") != EXPECTED_MODEL_REVISION
    ):
        raise ValueError("Copy-v2 model identity changed")

    if effect.get("schema") != EXPECTED_EFFECT_SCHEMA:
        raise ValueError("unexpected Copy-v2 effect schema")
    if effect.get("status") != "complete_locked_test_outcomes":
        raise ValueError("Copy-v2 locked outcomes are incomplete")
    if effect.get("data_version") != EXPECTED_DATA_VERSION:
        raise ValueError("Copy-v2 effect data version changed")
    if effect.get("scientific_claim_allowed") is not True:
        raise ValueError("Copy-v2 effects are not eligible for scientific use")
    if effect.get("design_manifest_sha256") != _sha256(design_path):
        raise ValueError("Copy-v2 effects are not bound to the frozen design")

    design_hashes = design.get("artifact_hashes")
    effect_hashes = effect.get("artifacts")
    if not isinstance(design_hashes, dict) or not isinstance(effect_hashes, dict):
        raise ValueError("Copy-v2 manifests lack artifact hashes")
    _verify_file_hashes(
        design_dir,
        design_hashes,
        (
            "selected_heads.csv",
            "calibration_masks.csv",
            "test_masks.csv",
            "prompts.csv",
        ),
    )
    _verify_file_hashes(
        effects_dir,
        effect_hashes,
        ("train_effects.csv", "test_effects.csv"),
    )
    return design_dir, effects_dir, design


def _mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("effect aggregation requires finite nonempty values")
    return math.fsum(values) / len(values)


def _effect_cells(
    path: Path,
    *,
    split: str,
    expected_masks: set[str],
) -> tuple[dict[str, list[float]], int]:
    grouped: dict[str, list[float]] = {mask_id: [] for mask_id in expected_masks}
    prompt_ids: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for row in _read_csv(path):
        if row.get("schema_version") != EXPECTED_EFFECT_ROW_SCHEMA:
            raise ValueError(f"unexpected effect-row schema in {path}")
        if row.get("split") != split:
            raise ValueError(f"unexpected split in {path}")
        prompt_id = row["prompt_id"]
        mask_id = row["mask_id"]
        key = (prompt_id, mask_id)
        if mask_id not in grouped or key in seen:
            raise ValueError(f"unexpected or duplicate prompt-mask cell: {key}")
        seen.add(key)
        prompt_ids.add(prompt_id)
        clean = float(row["clean_margin"])
        ablated = float(row["ablated_margin"])
        effect = float(row["drop_from_clean"])
        if not all(math.isfinite(value) for value in (clean, ablated, effect)):
            raise ValueError(f"non-finite effect cell: {key}")
        if not math.isclose(clean - ablated, effect, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"effect arithmetic changed: {key}")
        grouped[mask_id].append(effect)
    expected_cells = len(expected_masks) * len(prompt_ids)
    if len(seen) != expected_cells or any(
        len(values) != len(prompt_ids) for values in grouped.values()
    ):
        raise ValueError(f"incomplete prompt-by-mask table: {path}")
    return grouped, len(prompt_ids)


def _features(mask_bits: str) -> dict[str, int]:
    if len(mask_bits) != 8 or set(mask_bits) - {"0", "1"}:
        raise ValueError(f"invalid eight-head mask: {mask_bits}")
    return {f"h{index}": int(bit) for index, bit in enumerate(mask_bits)}


def build_pack(source_root: Path, output_root: Path) -> dict[str, object]:
    """Verify the frozen run and export its small, fully open practice view."""

    design_dir, effects_dir, design = _checked_source(source_root)
    selected_heads = sorted(
        _read_csv(design_dir / "selected_heads.csv"),
        key=lambda row: int(row["component_index"]),
    )
    if [int(row["component_index"]) for row in selected_heads] != list(range(8)):
        raise ValueError("selected head indices changed")

    calibration_masks = sorted(
        _read_csv(design_dir / "calibration_masks.csv"),
        key=lambda row: int(row["measurement_order"]),
    )
    test_masks = _read_csv(design_dir / "test_masks.csv")
    if len(calibration_masks) != 128 or len(test_masks) != 128:
        raise ValueError("Copy-v2 must partition all masks into 128/128")
    if calibration_masks[0]["mask_bits"] != "00000000":
        raise ValueError("first calibration measurement must be exact no-op")
    if len({row["mask_bits"] for row in calibration_masks + test_masks}) != 256:
        raise ValueError("Copy-v2 mask partition is not the full Boolean cube")
    pool_counts: dict[str, int] = {}
    for row in test_masks:
        pool_counts[row["pool_id"]] = pool_counts.get(row["pool_id"], 0) + 1
    if len(pool_counts) != 16 or set(pool_counts.values()) != {8}:
        raise ValueError("Copy-v2 must contain 16 pools of eight test masks")

    first_masks = calibration_masks[:MEASUREMENT_BUDGET]
    train_by_mask, n_train_prompts = _effect_cells(
        effects_dir / "train_effects.csv",
        split="train",
        expected_masks={row["mask_id"] for row in calibration_masks},
    )
    test_by_mask, n_test_prompts = _effect_cells(
        effects_dir / "test_effects.csv",
        split="test",
        expected_masks={row["mask_id"] for row in test_masks},
    )

    calibration_rows: list[dict[str, object]] = []
    for row in first_masks:
        bits = row["mask_bits"]
        calibration_rows.append(
            {
                "measurement_order": int(row["measurement_order"]),
                "measurement_id": row["mask_id"],
                "mask_bits": bits,
                "n_heads": bits.count("1"),
                "observed_mean_effect": repr(_mean(train_by_mask[row["mask_id"]])),
                "n_train_prompts": n_train_prompts,
                **_features(bits),
            }
        )

    query_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for row in sorted(test_masks, key=lambda value: value["mask_id"]):
        bits = row["mask_bits"]
        query_rows.append(
            {
                "query_id": row["mask_id"],
                "mask_bits": bits,
                "n_heads": bits.count("1"),
                "pool_id": row["pool_id"],
                **_features(bits),
            }
        )
        target_rows.append(
            {
                "query_id": row["mask_id"],
                "observed_mean_effect": repr(_mean(test_by_mask[row["mask_id"]])),
                "n_test_prompts": n_test_prompts,
            }
        )

    confirmation_gate = _read_json(design_dir / "gates" / "confirmation_gate.json")
    selected_full_effect = float(confirmation_gate["selected_mean_effect"])
    target_specs = (
        (0.25, selected_full_effect * 0.25),
        (0.50, selected_full_effect * 0.50),
        (0.75, selected_full_effect * 0.75),
    )
    action_rows: list[dict[str, object]] = []
    masks_by_pool: dict[str, list[dict[str, str]]] = {}
    for row in test_masks:
        masks_by_pool.setdefault(row["pool_id"], []).append(row)
    for fraction, target in target_specs:
        for pool_id in sorted(masks_by_pool):
            for row in sorted(masks_by_pool[pool_id], key=lambda value: value["mask_id"]):
                bits = row["mask_bits"]
                actual_loss = _mean(
                    [abs(effect - target) for effect in test_by_mask[row["mask_id"]]]
                )
                action_rows.append(
                    {
                        "target_fraction": f"{fraction:.2f}",
                        "target": repr(target),
                        "pool_id": pool_id,
                        "query_id": row["mask_id"],
                        "mask_bits": bits,
                        "n_heads": bits.count("1"),
                        "is_noop": "false",
                        "actual_target_loss": repr(actual_loss),
                        "n_test_prompts": n_test_prompts,
                    }
                )
            action_rows.append(
                {
                    "target_fraction": f"{fraction:.2f}",
                    "target": repr(target),
                    "pool_id": pool_id,
                    "query_id": "analytic_noop",
                    "mask_bits": "00000000",
                    "n_heads": 0,
                    "is_noop": "true",
                    "actual_target_loss": repr(target),
                    "n_test_prompts": n_test_prompts,
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    generated = {
        "calibration_measurements.csv": calibration_rows,
        "queries.csv": query_rows,
        "targets.csv": target_rows,
        "action_outcomes.csv": action_rows,
    }
    fields = {
        "calibration_measurements.csv": CALIBRATION_FIELDS,
        "queries.csv": QUERY_FIELDS,
        "targets.csv": TARGET_FIELDS,
        "action_outcomes.csv": ACTION_FIELDS,
    }
    for name, rows in generated.items():
        _write_csv(output_root / name, fields[name], rows)

    source_files = {
        "design_manifest": design_dir / "design_manifest.json",
        "effect_manifest": effects_dir / "effect_manifest.json",
        "calibration_masks": design_dir / "calibration_masks.csv",
        "test_masks": design_dir / "test_masks.csv",
        "train_effects": effects_dir / "train_effects.csv",
        "test_effects": effects_dir / "test_effects.csv",
    }
    card: dict[str, object] = {
        "schema": PACK_SCHEMA,
        "task_id": TASK_ID,
        "participation_class": "open_practice",
        "leaderboard_eligible": False,
        "sealed": False,
        "targets_public": True,
        "purpose": (
            "Fit an observer on 40 cached intervention measurements, predict 128 "
            "held-out mask effects, and test whether those predictions choose useful actions."
        ),
        "model": {
            "name": EXPECTED_MODEL,
            "parameters": "7B",
            "revision": EXPECTED_MODEL_REVISION,
            "inference_required": False,
        },
        "measurement_budget": MEASUREMENT_BUDGET,
        "n_calibration_measurements": len(calibration_rows),
        "n_queries": len(query_rows),
        "n_train_prompts_per_measurement": n_train_prompts,
        "n_test_prompts_per_query": n_test_prompts,
        "feature_order": [f"h{index}" for index in range(8)],
        "heads": [
            {
                "feature": f"h{index}",
                "label": row["head_label"],
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "kv_group": int(row["kv_group"]),
            }
            for index, row in enumerate(selected_heads)
        ],
        "prediction_contract": {
            "input": "calibration_measurements.csv and queries.csv",
            "submission_columns": ["schema_version", "query_id", "predicted_effect"],
            "submission_schema_version": "observerbench.effect_predictions.v0",
            "prediction_metrics": ["mae", "rmse", "mean_error", "max_absolute_error"],
        },
        "action_contract": {
            "targets": [
                {"fraction_of_confirmed_full_panel_effect": fraction, "value": target}
                for fraction, target in target_specs
            ],
            "candidate_set": "eight frozen masks in each of 16 pools plus analytic no-op",
            "controller": (
                "Choose the candidate minimizing |predicted_mean_effect - target|; "
                "the analytic no-op predicts effect 0 and loss target."
            ),
            "tie_break": "predicted loss, then fewer heads, then query_id",
            "realized_loss": "mean over held-out prompts of |prompt_effect - target|",
            "no_op": {
                "query_id": "analytic_noop",
                "effect": 0.0,
                "realized_loss": "target",
            },
            "metrics": [
                "mean_selected_action_loss",
                "mean_oracle_action_loss",
                "mean_regret",
                "oracle_action_match_rate",
                "noop_selection_rate",
            ],
        },
        "scope": [
            "This is an open practice task with public targets, not a sealed leaderboard.",
            "It uses one pinned Qwen2.5-7B base checkpoint and one synthetic copy distribution.",
            "The fixed controller converts mean-effect predictions into actions; direct-risk observers require a richer submission contract.",
        ],
        "source": {
            "design_id": design["design_id"],
            "data_version": EXPECTED_DATA_VERSION,
            "source_path": str(source_root.relative_to(REPO_ROOT)),
            "sha256": {label: _sha256(path) for label, path in source_files.items()},
        },
        "files": {
            name: {"sha256": _sha256(output_root / name), "rows": len(rows)}
            for name, rows in generated.items()
        },
    }
    (output_root / "task_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    card = build_pack(args.source_root.resolve(), args.output.resolve())
    print(
        f"Built {card['task_id']} as {card['participation_class']} at "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
