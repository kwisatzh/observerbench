"""Post-review target and transformed-mean sensitivity for canonical IOI.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This study reuses the sealed Phase-7 design and clean gate.  It fits three
same-basis observers at targets 0.5, 1.0, and 1.5, freezes every action, and
then identifies the exact selected-mask union that still requires model
inference.  Phase-7 outcomes are not read by the fit or action freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance
from observerbench.tasks.ioi.phase5_analysis import _design_run, _validate_effect_manifest
from observerbench.tasks.ioi.phase5_effects import load_locked_ioi_design
from observerbench.tasks.ioi.phase6_risk import _head_quadratic_design
from observerbench.tasks.ioi.phase7_confirmation import (
    EXACT_NOOP,
    NOOP_BITS,
    TARGET_POLICY,
    load_phase5_train_calibration_only,
    load_verified_phase7_design,
    validate_phase7_freeze,
    verify_clean_pretest,
)
from observerbench.tasks.ioi.phase7_measurement import load_phase7_measurement_inputs
from observerbench.tasks.ioi.stage2d import ridge_fit


PROTOCOL_SCHEMA = "observerbench.ioi_phase08_target_sensitivity.v1"
SCIENTIFIC_STATUS = "post_review_post_confirmatory_target_sensitivity"
FREEZE_SCHEMA = "observerbench.ioi_phase08_prediction_action_freeze.v1"
FREEZE_STATUS = "all_sensitivity_actions_frozen_before_new_outcomes"
AUDIT_SCHEMA = "observerbench.ioi_phase08_preoutcome_audit.v1"
AUDIT_STATUS = "deterministic_recomputation_passed_new_outcomes_unopened"
AUDIT_FILENAME = "preoutcome_audit.json"

DIRECT_RISK = "direct_risk_head_pair_quadratic"
NATURAL_MEAN = "natural_mean_effect_head_pair_quadratic"
TRANSFORMED_MEAN = "transformed_mean_head_pair_quadratic"
SELECTORS = (DIRECT_RISK, NATURAL_MEAN, TRANSFORMED_MEAN)

REQUIRED_DOWNSTREAM_SOURCE_FILES = (
    "src/observerbench/core.py",
    "src/observerbench/provenance.py",
    "src/observerbench/tasks/ioi/phase5_analysis.py",
    "src/observerbench/tasks/ioi/phase5_design.py",
    "src/observerbench/tasks/ioi/phase5_effects.py",
    "src/observerbench/tasks/ioi/heads.py",
    "src/observerbench/tasks/ioi/phase2_capacity.py",
    "src/observerbench/tasks/ioi/phase6_risk.py",
    "src/observerbench/tasks/ioi/phase7_confirmation.py",
    "src/observerbench/tasks/ioi/phase7_freeze_audit.py",
    "src/observerbench/tasks/ioi/phase7_measurement.py",
    "src/observerbench/tasks/ioi/phase7_evaluation.py",
    "src/observerbench/tasks/ioi/phase8_sensitivity.py",
    "src/observerbench/tasks/ioi/phase8_measurement.py",
    "src/observerbench/tasks/ioi/phase8_evaluation.py",
    "src/observerbench/tasks/ioi/stage2d.py",
    "scripts/freeze_ioi_phase08_sensitivity.py",
    "scripts/audit_ioi_phase08_preoutcome.py",
    "scripts/run_ioi_phase08_selected_measurement.py",
    "scripts/evaluate_ioi_phase08_sensitivity.py",
)


@dataclass(frozen=True)
class Phase8Paths:
    """Existing sealed sources used by the sensitivity freeze."""

    phase7_protocol: Path
    phase7_design: Path
    phase7_pretest: Path
    phase7_freeze: Path
    phase7_audit: Path
    phase7_measurement: Path
    phase5_design: Path
    phase5_effects: Path


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_phase8_protocol(path: str | Path) -> dict[str, Any]:
    """Load the explicit post-confirmatory sensitivity protocol."""

    protocol = _read_json(path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"expected {PROTOCOL_SCHEMA}")
    if protocol.get("status") != SCIENTIFIC_STATUS:
        raise ValueError("Phase-8 must remain labeled post-confirmatory")
    if tuple(map(float, protocol.get("targets", ()))) != (0.5, 1.0, 1.5):
        raise ValueError("Phase-8 requires targets 0.5, 1.0, and 1.5")
    if tuple(map(str, protocol.get("selectors", ()))) != SELECTORS:
        raise ValueError("Phase-8 selector family changed")
    if int(protocol.get("measurement_budget", -1)) != 160:
        raise ValueError("Phase-8 requires all 160 calibration masks")
    if float(protocol.get("ridge", np.nan)) != 1e-6:
        raise ValueError("Phase-8 ridge changed")
    if int(protocol.get("candidate_pool_count", -1)) != 48:
        raise ValueError("Phase-8 requires the 48 Phase-7 pools")
    if int(protocol.get("candidate_pool_size_including_noop", -1)) != 31:
        raise ValueError("Phase-8 requires the 31-action Phase-7 pools")
    if protocol.get("new_outcome_access_rule") != (
        "Freeze predictions, actions, and the new selected-mask union before "
        "reading any newly measured outcome."
    ):
        raise ValueError("Phase-8 new-outcome access rule changed")
    hashes = protocol.get("source_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("Phase-8 protocol must pin its inherited sources")
    return protocol


def verify_phase8_protocol_sources(
    protocol: Mapping[str, Any], repository_root: str | Path
) -> None:
    """Verify every inherited Phase-5 and Phase-7 source named in the protocol."""

    root = Path(repository_root)
    for relative, expected in protocol["source_hashes"].items():
        path = root / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise ValueError(f"Phase-8 inherited source changed: {relative}")


def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    result = values.astype(str).str.lower().map({"true": True, "false": False})
    if result.isna().any():
        raise ValueError(f"invalid boolean column: {column}")
    return result.astype(bool)


def _verify_phase7_measurement_manifest(
    measurement_dir: str | Path,
    *,
    expected_selected_hash: str,
) -> dict[str, Any]:
    """Verify the complete 89-mask Phase-7 measurement without reading values."""

    root = Path(measurement_dir)
    manifest = _read_json(root / "measurement_manifest.json")
    if manifest.get("schema") != "observerbench.ioi_phase07_selected_measurement.v1":
        raise ValueError("unexpected inherited Phase-7 measurement schema")
    if manifest.get("status") != "all_frozen_selected_nonnoop_outcomes_measured":
        raise ValueError("inherited Phase-7 selected measurement is incomplete")
    counts = manifest.get("counts", {})
    if int(counts.get("selected_unique_nonnoop_masks", -1)) != 89:
        raise ValueError("Phase-7 measured union no longer contains 89 masks")
    if int(counts.get("effect_cells", -1)) != 89 * 512:
        raise ValueError("Phase-7 measured cell count changed")
    if manifest.get("source_hashes", {}).get("selected_measurement_masks") != expected_selected_hash:
        raise ValueError("Phase-7 measurement used a different selected-mask union")
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        path = root / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise ValueError(f"inherited Phase-7 measurement changed: {relative}")
    return manifest


def load_phase8_fit_inputs(
    paths: Phase8Paths,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load only design metadata and Phase-5 train/calibration outcomes."""

    design_manifest, _protocol, _prompts, calibration, candidates = (
        load_verified_phase7_design(paths.phase7_design, paths.phase7_protocol)
    )
    verify_clean_pretest(
        paths.phase7_pretest,
        design_dir=paths.phase7_design,
        protocol_path=paths.phase7_protocol,
    )
    phase7_freeze = validate_phase7_freeze(
        paths.phase7_freeze,
        design_dir=paths.phase7_design,
        pretest_dir=paths.phase7_pretest,
        phase5_design_dir=paths.phase5_design,
        phase5_effects_dir=paths.phase5_effects,
        protocol_path=paths.phase7_protocol,
    )
    inherited_selected_path = paths.phase7_freeze / "selected_measurement_masks.csv"
    _verify_phase7_measurement_manifest(
        paths.phase7_measurement,
        expected_selected_hash=file_sha256(inherited_selected_path),
    )
    inherited = pd.read_csv(
        inherited_selected_path,
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    inherited["mask_bits"] = inherited["mask_bits"].astype(str).str.zfill(13)
    inherited["is_noop"] = _bool_column(inherited, "is_noop")
    if len(inherited) != 89 or inherited["mask_id"].astype(str).duplicated().any():
        raise ValueError("inherited Phase-7 selected-mask table changed")
    if inherited["is_noop"].any() or (inherited["mask_bits"] == NOOP_BITS).any():
        raise ValueError("Phase-7 measured union contains no-op")

    phase5_prompts, phase5_masks, _ = load_locked_ioi_design(paths.phase5_design)
    effect_manifest = _validate_effect_manifest(paths.phase5_effects, paths.phase5_design)
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
        raise ValueError("Phase-8 calibration bank differs from Phase 5 and Phase 7")
    train, _paths = load_phase5_train_calibration_only(
        paths.phase5_effects,
        phase5_prompts=phase5_prompts,
        phase5_masks=phase5_masks,
        effect_manifest=effect_manifest,
    )
    if len(train) != 192 * 160:
        raise ValueError("Phase-8 fit requires the complete Phase-5 train table")
    if set(train["split"].astype(str)) != {"train"} or set(train["bank"].astype(str)) != {
        "calibration"
    }:
        raise ValueError("Phase-8 fit received an outcome outside train/calibration")
    if phase7_freeze.get("phase7_candidate_outcomes_loaded") is not False:
        raise ValueError("Phase-7 freeze provenance changed")
    return design_manifest, phase5_calibration, candidates, inherited, train


def compute_phase8_freeze_tables(
    calibration: pd.DataFrame,
    candidates: pd.DataFrame,
    inherited_measured: pd.DataFrame,
    train_effects: pd.DataFrame,
    *,
    targets: Sequence[float] = (0.5, 1.0, 1.5),
    ridge: float = 1e-6,
) -> dict[str, pd.DataFrame]:
    """Fit all observers and return deterministic pre-outcome tables."""

    calibration = calibration.sort_values("measurement_order").reset_index(drop=True)
    candidates = candidates.copy()
    candidates["mask_bits"] = candidates["mask_bits"].astype(str).str.zfill(13)
    candidates["is_noop"] = _bool_column(candidates, "is_noop")
    candidates = candidates.sort_values(
        ["pool_index", "is_noop", "n_heads", "sampling_stratum", "mask_id"]
    ).reset_index(drop=True)
    if len(calibration) != 160 or len(candidates) != 48 * 31:
        raise ValueError("Phase-8 inherited design dimensions changed")
    if candidates.groupby("pool_id").size().nunique() != 1 or set(
        candidates.groupby("pool_id").size()
    ) != {31}:
        raise ValueError("Phase-8 candidate pools are incomplete")
    combined = pd.concat([calibration, candidates], ignore_index=True, sort=False)
    design, terms = _head_quadratic_design(_design_run(combined).masks)
    if design.shape[1] != 92 or np.linalg.matrix_rank(design[:160]) != 92:
        raise ValueError("Phase-8 92-column rank gate failed")
    calibration_ids = calibration["mask_id"].astype(str).tolist()
    train = train_effects.assign(mask_id=train_effects["mask_id"].astype(str))
    grouped = train.groupby("mask_id")
    mean_response = grouped["drop_from_clean"].mean()
    y_mean = np.asarray([mean_response[item] for item in calibration_ids], dtype=float)
    beta_mean = ridge_fit(design[:160], y_mean, ridge)
    candidate_design = design[160:]
    predicted_mean = candidate_design @ beta_mean
    noop = candidates["is_noop"].to_numpy(bool)

    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for target in map(float, targets):
        risk_response = train.assign(
            target_loss=np.abs(train["drop_from_clean"].to_numpy(float) - target)
        ).groupby("mask_id")["target_loss"].mean()
        y_risk = np.asarray([risk_response[item] for item in calibration_ids], dtype=float)
        y_transformed = np.abs(y_mean - target)
        fits = {
            DIRECT_RISK: (ridge_fit(design[:160], y_risk, ridge), y_risk),
            NATURAL_MEAN: (beta_mean, y_mean),
            TRANSFORMED_MEAN: (
                ridge_fit(design[:160], y_transformed, ridge),
                y_transformed,
            ),
        }
        for selector, (coefficient, response) in fits.items():
            if selector == NATURAL_MEAN:
                predicted_loss = np.abs(predicted_mean - target)
                means = predicted_mean.copy()
            else:
                predicted_loss = candidate_design @ coefficient
                means = np.full(len(candidates), np.nan)
            negative_fraction = float((predicted_loss[~noop] < 0.0).mean())
            predicted_loss = predicted_loss.copy()
            predicted_loss[noop] = abs(target)
            if selector == NATURAL_MEAN:
                means[noop] = 0.0
            prediction_rows.extend(
                {
                    "scientific_status": SCIENTIFIC_STATUS,
                    "selector": selector,
                    "target": target,
                    "measurement_budget": 160,
                    "mask_id": str(mask.mask_id),
                    "pool_id": str(mask.pool_id),
                    "is_noop": bool(mask.is_noop),
                    "n_heads": int(mask.n_heads),
                    "predicted_target_loss": float(loss),
                    "predicted_mean_effect": (
                        float(mean) if np.isfinite(mean) else np.nan
                    ),
                    "noop_loss_set_analytically": bool(mask.is_noop),
                }
                for mask, loss, mean in zip(
                    candidates.itertuples(index=False), predicted_loss, means
                )
            )
            coefficient_rows.extend(
                {
                    "selector": selector,
                    "target": target,
                    "term": term,
                    "coefficient": float(value),
                    "measurement_budget": 160,
                    "ridge": ridge,
                }
                for term, value in zip(terms, coefficient)
            )
            diagnostic_rows.append(
                {
                    "selector": selector,
                    "target": target,
                    "design_rank": 92,
                    "n_columns": 92,
                    "train_calibration_mae": float(
                        np.mean(np.abs(design[:160] @ coefficient - response))
                    ),
                    "candidate_negative_prediction_fraction_before_noop_override": negative_fraction,
                }
            )
    predictions = pd.DataFrame(prediction_rows)

    action_rows: list[dict[str, Any]] = []
    grouping = ["selector", "target", "pool_id"]
    for keys, pool in predictions.groupby(grouping, sort=True):
        pool = pool.reset_index(drop=True)
        if len(pool) != 31 or int(pool["is_noop"].sum()) != 1:
            raise ValueError("each Phase-8 action must choose among 30 masks and no-op")
        selected = int(
            np.lexsort(
                (
                    pool["mask_id"].astype(str).to_numpy(),
                    pool["n_heads"].to_numpy(int),
                    pool["predicted_target_loss"].to_numpy(float),
                )
            )[0]
        )
        row = pool.iloc[selected]
        action_rows.append(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "selector": str(keys[0]),
                "target": float(keys[1]),
                "policy": TARGET_POLICY,
                "pool_id": str(keys[2]),
                "selected_mask_id": str(row["mask_id"]),
                "selected_is_noop": bool(row["is_noop"]),
                "selected_head_count": int(row["n_heads"]),
                "predicted_target_loss": float(row["predicted_target_loss"]),
            }
        )
    for target in map(float, targets):
        for pool_id, pool in candidates.groupby("pool_id", sort=True):
            row = pool.loc[pool["is_noop"]]
            if len(row) != 1:
                raise ValueError("each Phase-8 pool lacks a unique analytic no-op")
            no_op = row.iloc[0]
            action_rows.append(
                {
                    "scientific_status": SCIENTIFIC_STATUS,
                    "selector": EXACT_NOOP,
                    "target": target,
                    "policy": TARGET_POLICY,
                    "pool_id": str(pool_id),
                    "selected_mask_id": str(no_op["mask_id"]),
                    "selected_is_noop": True,
                    "selected_head_count": 0,
                    "predicted_target_loss": abs(target),
                }
            )
    actions = pd.DataFrame(action_rows).sort_values(
        ["target", "selector", "pool_id"]
    ).reset_index(drop=True)
    if len(actions) != 4 * 3 * 48:
        raise AssertionError("Phase-8 fixed action count changed")

    selected_ids = set(
        actions.loc[~actions["selected_is_noop"], "selected_mask_id"].astype(str)
    )
    all_selected = candidates.loc[candidates["mask_id"].astype(str).isin(selected_ids)].copy()
    all_selected = all_selected.sort_values("mask_id").reset_index(drop=True)
    if len(all_selected) != len(selected_ids) or all_selected["is_noop"].any():
        raise ValueError("Phase-8 selected-mask union is invalid")
    inherited_ids = set(inherited_measured["mask_id"].astype(str))
    reused = all_selected.loc[all_selected["mask_id"].astype(str).isin(inherited_ids)].copy()
    new = all_selected.loc[~all_selected["mask_id"].astype(str).isin(inherited_ids)].copy()
    if set(reused["mask_id"].astype(str)) != inherited_ids:
        raise ValueError("Phase-8 union does not reuse all 89 Phase-7 measured masks")
    if set(reused["mask_id"].astype(str)) & set(new["mask_id"].astype(str)):
        raise ValueError("Phase-8 reused and new mask unions overlap")
    return {
        "candidate_predictions.csv": predictions,
        "observer_coefficients.csv": pd.DataFrame(coefficient_rows),
        "fit_diagnostics.csv": pd.DataFrame(diagnostic_rows),
        "fixed_actions.csv": actions,
        "all_selected_masks.csv": all_selected,
        "reused_phase7_masks.csv": reused.sort_values("mask_id").reset_index(drop=True),
        "new_measurement_masks.csv": new.sort_values("mask_id").reset_index(drop=True),
    }


def freeze_phase8_sensitivity(
    paths: Phase8Paths,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    repository_root: str | Path,
) -> Path:
    """Freeze every sensitivity action without reading a Phase-7 outcome value."""

    protocol = load_phase8_protocol(protocol_path)
    verify_phase8_protocol_sources(protocol, repository_root)
    design, calibration, candidates, inherited, train = load_phase8_fit_inputs(paths)
    tables = compute_phase8_freeze_tables(
        calibration,
        candidates,
        inherited,
        train,
        targets=tuple(map(float, protocol["targets"])),
        ridge=float(protocol["ridge"]),
    )
    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a Phase-8 action freeze is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    counts = {
        "candidate_prediction_rows": len(tables["candidate_predictions.csv"]),
        "fixed_actions": len(tables["fixed_actions.csv"]),
        "selected_unique_nonnoop_masks": len(tables["all_selected_masks.csv"]),
        "reused_phase7_measured_masks": len(tables["reused_phase7_masks.csv"]),
        "new_selected_masks_to_measure": len(tables["new_measurement_masks.csv"]),
        "new_effect_cells_to_measure": 512 * len(tables["new_measurement_masks.csv"]),
        "exact_noop_baseline_action_rows": int(
            (tables["fixed_actions.csv"]["selector"] == EXACT_NOOP).sum()
        ),
        "fitted_selector_noop_selections": int(
            (
                (tables["fixed_actions.csv"]["selector"] != EXACT_NOOP)
                & tables["fixed_actions.csv"]["selected_is_noop"]
            ).sum()
        ),
        "total_selected_noop_action_rows": int(
            tables["fixed_actions.csv"]["selected_is_noop"].sum()
        ),
    }
    manifest = {
        "schema": FREEZE_SCHEMA,
        "status": FREEZE_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design["design_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "new_candidate_outcomes_loaded": False,
        "phase7_outcome_values_loaded_during_fit": False,
        "fit_outcomes": "Phase-5 train/calibration only",
        "targets": list(map(float, protocol["targets"])),
        "selectors": list(SELECTORS),
        "basis_columns": 92,
        "ridge": float(protocol["ridge"]),
        "counts": counts,
        "source_bindings": {
            "phase7_design_manifest": file_sha256(paths.phase7_design / "design_manifest.json"),
            "phase7_pretest_manifest": file_sha256(paths.phase7_pretest / "pretest_manifest.json"),
            "phase7_freeze_manifest": file_sha256(paths.phase7_freeze / "prediction_action_manifest.json"),
            "phase7_selected_masks": file_sha256(paths.phase7_freeze / "selected_measurement_masks.csv"),
            "phase7_measurement_manifest": file_sha256(paths.phase7_measurement / "measurement_manifest.json"),
            "phase5_design_manifest": file_sha256(paths.phase5_design / "design_manifest.json"),
            "phase5_effect_manifest": file_sha256(paths.phase5_effects / "effect_manifest.json"),
        },
        "artifact_hashes": {
            name: file_sha256(output / name) for name in tables
        },
        "runtime": runtime_provenance(),
        "next_allowed_stage": (
            "Run the deterministic pre-outcome audit, then measure only "
            "new_measurement_masks.csv."
        ),
    }
    write_json(output / "prediction_action_manifest.json", manifest)
    return output


def validate_phase8_freeze(
    freeze_dir: str | Path,
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Validate the immutable Phase-8 action freeze."""

    root = Path(freeze_dir)
    manifest = _read_json(root / "prediction_action_manifest.json")
    if manifest.get("schema") != FREEZE_SCHEMA or manifest.get("status") != FREEZE_STATUS:
        raise ValueError("Phase-8 action freeze is missing or incomplete")
    if manifest.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("Phase-8 protocol changed after action freeze")
    if manifest.get("new_candidate_outcomes_loaded") is not False:
        raise ValueError("Phase-8 freeze accessed a new candidate outcome")
    if manifest.get("phase7_outcome_values_loaded_during_fit") is not False:
        raise ValueError("Phase-8 fit accessed inherited outcome values")
    for name, expected in manifest.get("artifact_hashes", {}).items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Phase-8 freeze artifact changed: {name}")
    expected_counts = {
        "candidate_prediction_rows": 3 * 3 * 48 * 31,
        "fixed_actions": 4 * 3 * 48,
        "selected_unique_nonnoop_masks": 237,
        "reused_phase7_measured_masks": 89,
        "new_selected_masks_to_measure": 148,
        "new_effect_cells_to_measure": 148 * 512,
        "exact_noop_baseline_action_rows": 3 * 48,
        "fitted_selector_noop_selections": 12,
        "total_selected_noop_action_rows": 156,
    }
    for name, expected in expected_counts.items():
        if int(manifest.get("counts", {}).get(name, -1)) != expected:
            raise ValueError(f"Phase-8 frozen count changed: {name}")
    return manifest


def audit_phase8_freeze(
    paths: Phase8Paths,
    freeze_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    repository_root: str | Path,
    downstream_source_files: Sequence[str],
) -> Path:
    """Recompute frozen tables and bind the downstream code before inference."""

    protocol = load_phase8_protocol(protocol_path)
    verify_phase8_protocol_sources(protocol, repository_root)
    manifest = validate_phase8_freeze(freeze_dir, protocol_path=protocol_path)
    _design, calibration, candidates, inherited, train = load_phase8_fit_inputs(paths)
    expected = compute_phase8_freeze_tables(
        calibration,
        candidates,
        inherited,
        train,
        targets=tuple(map(float, protocol["targets"])),
        ridge=float(protocol["ridge"]),
    )
    maximum = 0.0
    for name, frame in expected.items():
        actual = pd.read_csv(
            Path(freeze_dir) / name,
            dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
        )
        if set(actual.columns) != set(frame.columns) or len(actual) != len(frame):
            raise ValueError(f"Phase-8 recomputation shape changed: {name}")
        actual = actual[frame.columns].fillna("").astype(str).sort_values(list(frame.columns)).reset_index(drop=True)
        wanted = frame.fillna("").astype(str).sort_values(list(frame.columns)).reset_index(drop=True)
        if not actual.equals(wanted):
            # CSV round trips can change the final few digits.  Re-read numeric
            # columns and compare separately before rejecting the seal.
            raw = pd.read_csv(Path(freeze_dir) / name)
            for column in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]):
                    left = pd.to_numeric(raw[column], errors="coerce").to_numpy(float)
                    right = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                    if not np.allclose(left, right, atol=1e-12, rtol=1e-12, equal_nan=True):
                        raise ValueError(f"Phase-8 recomputation changed numeric values: {name}:{column}")
                    finite = np.isfinite(left) & np.isfinite(right)
                    if finite.any():
                        maximum = max(maximum, float(np.max(np.abs(left[finite] - right[finite]))))
                elif raw[column].fillna("").astype(str).tolist() != frame[column].fillna("").astype(str).tolist():
                    raise ValueError(f"Phase-8 recomputation changed metadata: {name}:{column}")
    if set(downstream_source_files) != set(REQUIRED_DOWNSTREAM_SOURCE_FILES):
        raise ValueError("Phase-8 audit must bind the complete downstream source set")
    root = Path(repository_root)
    source_hashes = {
        relative: file_sha256(root / relative) for relative in downstream_source_files
    }
    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a Phase-8 pre-outcome audit is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "protocol_sha256": file_sha256(protocol_path),
        "freeze_manifest_sha256": file_sha256(
            Path(freeze_dir) / "prediction_action_manifest.json"
        ),
        "recomputed_all_predictions_actions_and_unions": True,
        "new_outcome_values_loaded": False,
        "maximum_numeric_recomputation_difference": maximum,
        "frozen_counts": manifest["counts"],
        "downstream_source_hashes": source_hashes,
        "runtime": runtime_provenance(),
    }
    write_json(output / AUDIT_FILENAME, audit)
    return output


def validate_phase8_audit(
    audit_dir: str | Path,
    freeze_dir: str | Path,
    *,
    protocol_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate the pre-outcome recomputation and bound downstream code."""

    audit = _read_json(Path(audit_dir) / AUDIT_FILENAME)
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != AUDIT_STATUS:
        raise ValueError("Phase-8 pre-outcome audit is absent")
    if audit.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("Phase-8 audit used a different protocol")
    if audit.get("freeze_manifest_sha256") != file_sha256(
        Path(freeze_dir) / "prediction_action_manifest.json"
    ):
        raise ValueError("Phase-8 audit used a different freeze")
    if audit.get("new_outcome_values_loaded") is not False:
        raise ValueError("Phase-8 audit accessed a new outcome")
    recorded = audit.get("downstream_source_hashes", {})
    if set(recorded) != set(REQUIRED_DOWNSTREAM_SOURCE_FILES):
        raise ValueError("Phase-8 audit downstream source set is incomplete")
    root = Path(repository_root)
    for relative, expected in recorded.items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Phase-8 downstream source changed after audit: {relative}")
    return audit
