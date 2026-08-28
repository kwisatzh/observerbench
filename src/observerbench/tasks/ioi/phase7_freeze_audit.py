"""Independent pre-outcome audit for the repaired Phase-7 freeze.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance
from observerbench.tasks.ioi.phase5_analysis import _design_run, _validate_effect_manifest
from observerbench.tasks.ioi.phase5_effects import load_locked_ioi_design
from observerbench.tasks.ioi.phase6_risk import _head_quadratic_design
from observerbench.tasks.ioi.phase7_confirmation import (
    DIRECT_RISK,
    EXACT_NOOP,
    NATURAL_MEAN,
    NOOP_BITS,
    PROTOCOL_SCHEMA_V2,
    SCIENTIFIC_STATUS,
    TARGET_POLICY,
    load_phase5_train_calibration_only,
    load_phase7_protocol,
    load_verified_phase7_design,
    validate_phase7_freeze,
    verify_clean_pretest,
    verify_protocol_sources,
)
from observerbench.tasks.ioi.stage2d import ridge_fit


AUDIT_SCHEMA = "observerbench.ioi_phase07_preoutcome_audit.v1"
AUDIT_STATUS = "independent_recomputation_passed_v2_outcomes_unopened"
AUDIT_FILENAME = "preoutcome_audit.json"
SOURCE_FILES = (
    "src/observerbench/tasks/ioi/phase7_confirmation.py",
    "src/observerbench/tasks/ioi/phase7_pretest.py",
    "src/observerbench/tasks/ioi/phase7_freeze_audit.py",
    "src/observerbench/tasks/ioi/phase7_measurement.py",
    "src/observerbench/tasks/ioi/phase7_evaluation.py",
    "src/observerbench/tasks/ioi/phase5_effects.py",
    "src/observerbench/tasks/ioi/phase5_analysis.py",
    "src/observerbench/tasks/ioi/phase6_risk.py",
    "src/observerbench/tasks/ioi/stage2d.py",
    "scripts/run_ioi_phase07_selected_measurement.py",
    "scripts/evaluate_ioi_phase07_confirmation.py",
)


def phase7_source_hashes(repository_root: str | Path) -> dict[str, str]:
    """Hash the closed implementation set used after the pre-outcome audit."""

    root = Path(repository_root)
    return {relative: file_sha256(root / relative) for relative in SOURCE_FILES}


def _assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: list[str],
    label: str,
) -> float:
    if set(actual.columns) != set(expected.columns):
        raise ValueError(f"{label} columns changed")
    actual = actual[expected.columns].sort_values(sort_by).reset_index(drop=True)
    expected = expected.sort_values(sort_by).reset_index(drop=True)
    numeric = [
        column
        for column in expected.columns
        if pd.api.types.is_numeric_dtype(expected[column])
        and not pd.api.types.is_bool_dtype(expected[column])
    ]
    maximum = 0.0
    for column in numeric:
        left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(left) | np.isfinite(right)
        if not np.array_equal(np.isfinite(left), np.isfinite(right)):
            raise ValueError(f"{label} non-finite pattern changed: {column}")
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(left[finite] - right[finite]))))
        if not np.allclose(left, right, atol=1e-12, rtol=1e-12, equal_nan=True):
            raise ValueError(f"{label} numeric values changed: {column}")
    for column in expected.columns:
        if column in numeric:
            continue
        left = actual[column].fillna("").astype(str).tolist()
        right = expected[column].fillna("").astype(str).tolist()
        if left != right:
            raise ValueError(f"{label} metadata changed: {column}")
    return maximum


def independently_recompute_phase7_freeze(
    design_dir: str | Path,
    pretest_dir: str | Path,
    freeze_dir: str | Path,
    phase5_design_dir: str | Path,
    phase5_effects_dir: str | Path,
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Recompute the complete freeze without calling the freeze writer."""

    protocol = load_phase7_protocol(protocol_path)
    if protocol["schema"] != PROTOCOL_SCHEMA_V2:
        raise ValueError("the independent audit requires the repaired v2 protocol")
    _design_manifest, _protocol, _prompts, calibration, candidates = (
        load_verified_phase7_design(design_dir, protocol_path)
    )
    verify_clean_pretest(
        pretest_dir, design_dir=design_dir, protocol_path=protocol_path
    )
    phase5_prompts, phase5_masks, _ = load_locked_ioi_design(phase5_design_dir)
    effect_manifest = _validate_effect_manifest(phase5_effects_dir, phase5_design_dir)
    phase5_calibration = phase5_masks.loc[
        phase5_masks["bank"].astype(str) == "calibration"
    ].copy()
    phase5_calibration["measurement_order"] = pd.to_numeric(
        phase5_calibration["measurement_order"], errors="raise"
    ).astype(int)
    phase5_calibration = phase5_calibration.sort_values("measurement_order").reset_index(
        drop=True
    )
    if phase5_calibration["mask_bits"].astype(str).str.zfill(13).tolist() != calibration.sort_values(
        "measurement_order"
    )["mask_bits"].astype(str).str.zfill(13).tolist():
        raise ValueError("v2 calibration bank differs from Phase 5")
    train, train_paths = load_phase5_train_calibration_only(
        phase5_effects_dir,
        phase5_prompts=phase5_prompts,
        phase5_masks=phase5_masks,
        effect_manifest=effect_manifest,
    )
    calibration_ids = phase5_calibration["mask_id"].astype(str).tolist()
    ordered_candidates = candidates.sort_values(
        ["pool_index", "is_noop", "n_heads", "sampling_stratum", "mask_id"]
    ).reset_index(drop=True)
    combined = pd.concat(
        [phase5_calibration, ordered_candidates], ignore_index=True, sort=False
    )
    x, terms = _head_quadratic_design(_design_run(combined).masks)
    n_calibration = len(phase5_calibration)
    if x.shape != (len(combined), 92) or np.linalg.matrix_rank(x[:n_calibration]) != 92:
        raise ValueError("independent v2 design-rank check failed")
    target = float(protocol["target"])
    ridge = float(protocol["ridge"])
    grouped = train.assign(
        target_loss=np.abs(train["drop_from_clean"].to_numpy(float) - target)
    ).groupby("mask_id")
    risk_response = grouped["target_loss"].mean()
    mean_response = grouped["drop_from_clean"].mean()
    y_risk = np.asarray([risk_response[item] for item in calibration_ids], dtype=float)
    y_mean = np.asarray([mean_response[item] for item in calibration_ids], dtype=float)
    beta_risk = ridge_fit(x[:n_calibration], y_risk, ridge)
    beta_mean = ridge_fit(x[:n_calibration], y_mean, ridge)

    candidate_x = x[n_calibration:]
    risk_prediction = candidate_x @ beta_risk
    predicted_mean = candidate_x @ beta_mean
    mean_plugin = np.abs(predicted_mean - target)
    noop = ordered_candidates["is_noop"].astype(bool).to_numpy()
    risk_prediction[noop] = abs(target)
    predicted_mean[noop] = 0.0
    mean_plugin[noop] = abs(target)
    prediction_rows: list[dict[str, Any]] = []
    for selector, losses, means in (
        (DIRECT_RISK, risk_prediction, np.full(len(risk_prediction), np.nan)),
        (NATURAL_MEAN, mean_plugin, predicted_mean),
    ):
        prediction_rows.extend(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "selector": selector,
                "target": target,
                "measurement_budget": 160,
                "mask_id": str(row.mask_id),
                "pool_id": str(row.pool_id),
                "is_noop": bool(row.is_noop),
                "n_heads": int(row.n_heads),
                "predicted_target_loss": float(loss),
                "predicted_mean_effect": float(mean) if np.isfinite(mean) else np.nan,
                "noop_loss_set_analytically": bool(row.is_noop),
            }
            for row, loss, mean in zip(
                ordered_candidates.itertuples(index=False), losses, means
            )
        )
    expected_predictions = pd.DataFrame(prediction_rows)

    action_rows: list[dict[str, Any]] = []
    for selector, group in expected_predictions.groupby("selector", sort=True):
        for pool_id, pool in group.groupby("pool_id", sort=True):
            pool = pool.reset_index(drop=True)
            index = int(
                np.lexsort(
                    (
                        pool["mask_id"].astype(str).to_numpy(),
                        pool["n_heads"].to_numpy(int),
                        pool["predicted_target_loss"].to_numpy(float),
                    )
                )[0]
            )
            row = pool.iloc[index]
            action_rows.append(
                {
                    "scientific_status": SCIENTIFIC_STATUS,
                    "selector": str(selector),
                    "target": target,
                    "policy": TARGET_POLICY,
                    "pool_id": str(pool_id),
                    "selected_mask_id": str(row["mask_id"]),
                    "selected_is_noop": bool(row["is_noop"]),
                    "selected_head_count": int(row["n_heads"]),
                    "predicted_target_loss": float(row["predicted_target_loss"]),
                }
            )
    for pool_id, pool in ordered_candidates.groupby("pool_id", sort=True):
        row = pool.loc[pool["is_noop"].astype(bool)].iloc[0]
        action_rows.append(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "selector": EXACT_NOOP,
                "target": target,
                "policy": TARGET_POLICY,
                "pool_id": str(pool_id),
                "selected_mask_id": str(row["mask_id"]),
                "selected_is_noop": True,
                "selected_head_count": 0,
                "predicted_target_loss": target,
            }
        )
    expected_actions = pd.DataFrame(action_rows).sort_values(
        ["selector", "pool_id"]
    ).reset_index(drop=True)
    selected_ids = set(
        expected_actions.loc[
            ~expected_actions["selected_is_noop"].astype(bool), "selected_mask_id"
        ].astype(str)
    )
    expected_selected = ordered_candidates.loc[
        ordered_candidates["mask_id"].astype(str).isin(selected_ids)
    ].copy()
    expected_coefficients = pd.DataFrame(
        [
            {
                "selector": selector,
                "term": term,
                "coefficient": float(value),
                "measurement_budget": 160,
                "ridge": ridge,
            }
            for selector, values in (
                (DIRECT_RISK, beta_risk),
                (NATURAL_MEAN, beta_mean),
            )
            for term, value in zip(terms, values)
        ]
    )

    freeze_root = Path(freeze_dir)
    actual_coefficients = pd.read_csv(freeze_root / "observer_coefficients.csv")
    actual_predictions = pd.read_csv(
        freeze_root / "candidate_predictions.csv", dtype={"mask_id": str, "pool_id": str}
    )
    actual_actions = pd.read_csv(
        freeze_root / "fixed_actions.csv",
        dtype={"pool_id": str, "selected_mask_id": str},
    )
    actual_selected = pd.read_csv(
        freeze_root / "selected_measurement_masks.csv",
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    errors = {
        "coefficients_max_abs_error": _assert_frame_matches(
            actual_coefficients,
            expected_coefficients,
            sort_by=["selector", "term"],
            label="observer coefficients",
        ),
        "predictions_max_abs_error": _assert_frame_matches(
            actual_predictions,
            expected_predictions,
            sort_by=["selector", "pool_id", "mask_id"],
            label="candidate predictions",
        ),
        "actions_max_abs_error": _assert_frame_matches(
            actual_actions,
            expected_actions,
            sort_by=["selector", "pool_id"],
            label="fixed actions",
        ),
        "selected_union_max_abs_error": _assert_frame_matches(
            actual_selected,
            expected_selected,
            sort_by=["mask_id"],
            label="selected measurement union",
        ),
    }
    if (actual_selected["mask_bits"].astype(str).str.zfill(13) == NOOP_BITS).any():
        raise ValueError("independent audit found no-op in the measured union")
    return {
        "errors": errors,
        "counts": {
            "phase5_train_calibration_shards_read": len(train_paths),
            "phase5_train_calibration_cells_read": len(train),
            "phase5_candidate_mask_outcome_rows_read": 0,
            "phase5_validation_or_test_outcome_rows_read": 0,
            "phase7_candidate_outcome_rows_read": 0,
            "coefficients": len(expected_coefficients),
            "candidate_predictions": len(expected_predictions),
            "fixed_actions": len(expected_actions),
            "selected_unique_nonnoop_masks": len(expected_selected),
        },
        "train_paths": train_paths,
    }


def run_phase7_preoutcome_audit(
    design_dir: str | Path,
    pretest_dir: str | Path,
    freeze_dir: str | Path,
    phase5_design_dir: str | Path,
    phase5_effects_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
) -> Path:
    """Write the independent v2 audit seal before any v2 candidate measurement."""

    repository_root = Path(protocol_path).resolve().parents[2]
    protocol = load_phase7_protocol(protocol_path)
    verify_protocol_sources(protocol, repository_root)
    validate_phase7_freeze(
        freeze_dir,
        design_dir=design_dir,
        pretest_dir=pretest_dir,
        phase5_design_dir=phase5_design_dir,
        phase5_effects_dir=phase5_effects_dir,
        protocol_path=protocol_path,
    )
    result = independently_recompute_phase7_freeze(
        design_dir,
        pretest_dir,
        freeze_dir,
        phase5_design_dir,
        phase5_effects_dir,
        protocol_path=protocol_path,
    )
    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a Phase-7 pre-outcome audit is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "all_checks_pass": True,
        "protocol_schema": protocol["schema"],
        "input_hashes": {
            "protocol": file_sha256(protocol_path),
            "design_manifest": file_sha256(Path(design_dir) / "design_manifest.json"),
            "pretest_manifest": file_sha256(Path(pretest_dir) / "pretest_manifest.json"),
            "prediction_action_manifest": file_sha256(
                Path(freeze_dir) / "prediction_action_manifest.json"
            ),
            "selected_measurement_masks": file_sha256(
                Path(freeze_dir) / "selected_measurement_masks.csv"
            ),
            "template_head_means": file_sha256(
                Path(phase5_effects_dir) / "template_head_means.npz"
            ),
        },
        "phase5_train_calibration_shard_hashes": {
            path.relative_to(Path(phase5_effects_dir)).as_posix(): file_sha256(path)
            for path in result["train_paths"]
        },
        "source_code_hashes": phase7_source_hashes(repository_root),
        "independent_recomputation": result["errors"],
        "counts": result["counts"],
        "v1_partial_evidence_verified_by_hash_only": True,
        "v1_candidate_outcome_values_read": False,
        "v2_candidate_outcome_values_read": False,
        "runtime": runtime_provenance(),
        "next_allowed_stage": "Measure only the frozen v2 selected non-noop union.",
    }
    write_json(output / AUDIT_FILENAME, audit)
    return output


def validate_phase7_preoutcome_audit(
    audit_dir: str | Path,
    *,
    design_dir: str | Path,
    pretest_dir: str | Path,
    freeze_dir: str | Path,
    phase5_effects_dir: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Require the independent audit and every frozen source hash to remain intact."""

    repository_root = Path(protocol_path).resolve().parents[2]
    protocol = load_phase7_protocol(protocol_path)
    if protocol["schema"] != PROTOCOL_SCHEMA_V2:
        raise ValueError("candidate access is disabled for the superseded v1 protocol")
    path = Path(audit_dir) / AUDIT_FILENAME
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("status") != AUDIT_STATUS
        or audit.get("all_checks_pass") is not True
    ):
        raise ValueError("Phase-7 v2 pre-outcome audit did not pass")
    # Check the audited implementation before opening separately archived raw
    # measurements.  This keeps source-integrity failures diagnostic even in a
    # public checkout, where the large measurement shards are intentionally
    # absent.
    if audit.get("source_code_hashes") != phase7_source_hashes(repository_root):
        raise ValueError("Phase-7 v2 audited source code changed")
    expected_inputs = {
        "protocol": file_sha256(protocol_path),
        "design_manifest": file_sha256(Path(design_dir) / "design_manifest.json"),
        "pretest_manifest": file_sha256(Path(pretest_dir) / "pretest_manifest.json"),
        "prediction_action_manifest": file_sha256(
            Path(freeze_dir) / "prediction_action_manifest.json"
        ),
        "selected_measurement_masks": file_sha256(
            Path(freeze_dir) / "selected_measurement_masks.csv"
        ),
        "template_head_means": file_sha256(
            Path(phase5_effects_dir) / "template_head_means.npz"
        ),
    }
    if audit.get("input_hashes") != expected_inputs:
        raise ValueError("Phase-7 v2 pre-outcome audit input changed")
    if audit.get("v2_candidate_outcome_values_read") is not False:
        raise ValueError("Phase-7 v2 audit accessed a candidate outcome")
    verify_protocol_sources(protocol, repository_root)
    return audit
