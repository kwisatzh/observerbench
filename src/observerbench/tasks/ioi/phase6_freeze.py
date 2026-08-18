"""Train-only observer fit and action freeze for prospective Phase-6 IOI.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This module consumes only the sealed Phase-6 design and the reference/train
calibration run.  It freezes candidate predictions and one fixed mask per
observer, candidate pool, target, and policy before any test forward pass.
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
from observerbench.tasks.ioi.phase2_capacity import build_capacity_design
from observerbench.tasks.ioi.phase5_analysis import _design_run
from observerbench.tasks.ioi.phase5_effects import validate_effect_rows
from observerbench.tasks.ioi.phase6_confirmatory import (
    DESIGN_SCHEMA,
    PHASE6_STATUS,
    _head_quadratic_rank,
    load_phase6_protocol,
)
from observerbench.tasks.ioi.phase6_measurement import (
    CALIBRATION_SCHEMA,
    CALIBRATION_STATUS,
)
from observerbench.tasks.ioi.phase6_risk import _head_quadratic_design
from observerbench.tasks.ioi.stage2d import ridge_fit


FREEZE_SCHEMA = "observerbench.ioi_phase06_prediction_action_freeze.v1"
FREEZE_STATUS = "predictions_and_actions_frozen_before_phase6_test_outcomes"
DIRECT_RISK = "direct_risk"
NATURAL_MEAN = "natural_mean_effect"
JENSEN_SCORE = "target_specific_jensen_score"
TARGET_POLICY = "target_loss"
COST_POLICY = "cost_aware"
QUADRATIC_MODEL = "head_pair_quadratic_screen"


@dataclass(frozen=True)
class Phase6FreezeConfig:
    measurement_budget: int = 160
    targets: tuple[float, ...] = (0.5, 1.0, 1.5)
    ridge: float = 1e-6
    head_cost_penalty: float = 0.02

    def __post_init__(self) -> None:
        if self.measurement_budget != 160:
            raise ValueError("Phase-6 freeze requires the full 160-mask calibration bank")
        if self.targets != (0.5, 1.0, 1.5):
            raise ValueError("Phase-6 targets changed")
        if self.ridge != 1e-6:
            raise ValueError("Phase-6 ridge changed")
        if self.head_cost_penalty != 0.02:
            raise ValueError("Phase-6 head-cost penalty changed")


def _verify_design_and_protocol(
    design_dir: str | Path,
    protocol_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(design_dir)
    manifest_path = root / "design_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DESIGN_SCHEMA:
        raise ValueError("unexpected Phase-6 design manifest schema")
    if manifest.get("scientific_status") != PHASE6_STATUS:
        raise ValueError("Phase-6 scientific status changed")
    if manifest.get("contains_model_outcomes") is not False:
        raise ValueError("design manifest contains outcomes")
    if manifest.get("predictions_frozen") is not False:
        raise ValueError("design is not at the pre-fit stage")
    if manifest.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("protocol changed after the design was sealed")
    for filename, expected in manifest["artifact_hashes"].items():
        if file_sha256(root / filename) != expected:
            raise ValueError(f"sealed design artifact changed: {filename}")

    protocol = load_phase6_protocol(protocol_path)
    prompts = pd.read_csv(root / "prompts.csv", dtype={"prompt_id": str})
    calibration = pd.read_csv(
        root / "calibration_masks.csv",
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    candidates = pd.read_csv(
        root / "candidate_masks.csv",
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    calibration["pool_id"] = calibration["pool_id"].fillna("").astype(str)
    if _head_quadratic_rank(calibration) != 92:
        raise ValueError("the frozen 92-column quadratic rank gate failed")
    return manifest, protocol, prompts, calibration, candidates


def _verify_calibration_run(
    calibration_dir: str | Path,
    design_dir: str | Path,
    *,
    prompts: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(calibration_dir)
    manifest_path = root / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unexpected Phase-6 calibration manifest schema")
    if manifest.get("status") != CALIBRATION_STATUS:
        raise ValueError("Phase-6 calibration run is not complete and test-sealed")
    if manifest.get("design_manifest_sha256") != file_sha256(
        Path(design_dir) / "design_manifest.json"
    ):
        raise ValueError("calibration run used a different design")
    if manifest.get("accessed_prompt_splits") != ["reference", "train"]:
        raise ValueError("calibration run accessed an unauthorized prompt split")
    if manifest.get("accessed_mask_banks") != ["calibration"]:
        raise ValueError("calibration run accessed an unauthorized mask bank")
    if int(manifest.get("test_prompt_forward_passes", -1)) != 0:
        raise ValueError("test prompts were opened before the prediction freeze")
    if int(manifest.get("candidate_mask_forward_passes", -1)) != 0:
        raise ValueError("candidate masks were opened before the prediction freeze")
    counts = manifest.get("counts", {})
    expected_counts = {
        "reference_prompts": 512,
        "train_prompts": 192,
        "calibration_masks": 160,
        "train_calibration_effect_cells": 192 * 160,
    }
    for name, expected in expected_counts.items():
        if int(counts.get(name, -1)) != expected:
            raise ValueError(f"calibration manifest count changed: {name}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("calibration manifest has no artifact hash index")
    if any(Path(str(label)).is_absolute() or ".." in Path(str(label)).parts for label in artifacts):
        raise ValueError("calibration artifact labels must be root-relative")
    expected_static = {"template_head_means.npz", "clean_scores_train.csv"}
    shard_paths = tuple(sorted((root / "shards" / "train").glob("effects_*.csv")))
    if not shard_paths or int(counts.get("shards", -1)) != len(shard_paths):
        raise ValueError("calibration shard count differs from its manifest")
    expected_labels = expected_static | {
        path.relative_to(root).as_posix() for path in shard_paths
    }
    if set(map(str, artifacts)) != expected_labels:
        raise ValueError("calibration manifest does not index the exact allowed artifacts")
    for label, expected in artifacts.items():
        if file_sha256(root / str(label)) != expected:
            raise ValueError(f"calibration artifact changed: {label}")

    with np.load(root / "template_head_means.npz", allow_pickle=False) as cache:
        means = np.asarray(cache["means"])
        templates = tuple(map(str, cache["templates"].tolist()))
    expected_templates = tuple(
        sorted(prompts.loc[prompts["split"] == "reference", "template_id"].astype(str).unique())
    )
    if means.shape != (8, 13, 64):
        raise ValueError("template-conditioned reference mean array has wrong shape")
    if templates != expected_templates:
        raise ValueError("reference means do not match the frozen templates")
    clean = pd.read_csv(root / "clean_scores_train.csv", dtype={"prompt_id": str})
    expected_train_ids = set(
        prompts.loc[prompts["split"] == "train", "prompt_id"].astype(str)
    )
    if len(clean) != 192 or set(clean["prompt_id"].astype(str)) != expected_train_ids:
        raise ValueError("clean train scores do not match the frozen train prompts")
    if set(clean["split"].astype(str)) != {"train"}:
        raise ValueError("clean score artifact contains a non-train prompt")
    if not np.isfinite(clean["clean_ld"].to_numpy(float)).all():
        raise ValueError("clean train scores contain non-finite values")

    effects = pd.concat(
        [
            pd.read_csv(
                path,
                dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
            )
            for path in shard_paths
        ],
        ignore_index=True,
    )
    train_ids = prompts.loc[prompts["split"] == "train", "prompt_id"].astype(str).tolist()
    calibration_ids = calibration.sort_values("measurement_order")["mask_id"].astype(str).tolist()
    validate_effect_rows(effects, prompt_ids=train_ids, mask_ids=calibration_ids)
    if set(effects["split"].astype(str)) != {"train"}:
        raise ValueError("calibration effects include a non-train prompt")
    if set(effects["bank"].astype(str)) != {"calibration"}:
        raise ValueError("calibration effects include a candidate mask")
    if len(effects) != 192 * 160:
        raise ValueError("train/calibration table is not the exact Cartesian design")
    observed = effects[["mask_id", "mask_bits"]].drop_duplicates().copy()
    expected = calibration[["mask_id", "mask_bits"]].copy()
    for frame in (observed, expected):
        frame["mask_id"] = frame["mask_id"].astype(str)
        frame["mask_bits"] = frame["mask_bits"].astype(str).str.zfill(13)
    if not observed.set_index("mask_id").sort_index().equals(
        expected.set_index("mask_id").sort_index()
    ):
        raise ValueError("calibration mask mapping differs from the sealed design")
    return manifest, effects


def _design_for_model(
    combined: pd.DataFrame,
    model: str,
) -> tuple[np.ndarray, list[str]]:
    run = _design_run(combined)
    if model == QUADRATIC_MODEL:
        return _head_quadratic_design(run.masks)
    return build_capacity_design(run, model)


def _fit_one(
    design: np.ndarray,
    response: np.ndarray,
    *,
    n_calibration: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficient = ridge_fit(design[:n_calibration], response, ridge)
    fitted = design[:n_calibration] @ coefficient
    candidate = design[n_calibration:] @ coefficient
    return coefficient, fitted, candidate


def fit_phase6_observers(
    calibration: pd.DataFrame,
    candidates: pd.DataFrame,
    train_effects: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    config: Phase6FreezeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit every frozen observer using train/calibration effects only."""

    calibration = calibration.sort_values("measurement_order").reset_index(drop=True)
    candidates = candidates.sort_values(
        ["pool_index", "n_heads", "sampling_stratum", "mask_id"]
    ).reset_index(drop=True)
    calibration_ids = calibration["mask_id"].astype(str).tolist()
    effect_ids = set(train_effects["mask_id"].astype(str))
    if effect_ids != set(calibration_ids):
        raise ValueError("observer fit received effects outside the calibration bank")
    if set(train_effects["split"].astype(str)) != {"train"}:
        raise ValueError("observer fit received a non-train prompt")
    if set(train_effects["bank"].astype(str)) != {"calibration"}:
        raise ValueError("observer fit received candidate-mask effects")
    if len(train_effects) != 192 * len(calibration):
        raise ValueError("observer fit requires exactly 192 prompts per calibration mask")

    combined = pd.concat([calibration, candidates], ignore_index=True, sort=False)
    n_calibration = len(calibration)
    train = train_effects.assign(mask_id=train_effects["mask_id"].astype(str))
    mean_effect = train.groupby("mask_id")["drop_from_clean"].mean()
    direct_models = tuple(map(str, protocol["direct_risk_models"]))
    natural_models = tuple(map(str, protocol["mean_effect_models"]))
    jensen_models = tuple(map(str, protocol["jensen_score_sensitivity_models"]))
    expected_direct = (
        "additive_head",
        "count_additive",
        "count_plus_PE_bin4",
        "count_plus_all_bin4",
        QUADRATIC_MODEL,
    )
    if direct_models != expected_direct:
        raise ValueError("the frozen direct-risk model list changed")
    if natural_models != ("count_plus_all_bin4", QUADRATIC_MODEL):
        raise ValueError("the frozen natural mean-effect model list changed")
    if jensen_models != (QUADRATIC_MODEL,):
        raise ValueError("the frozen Jensen model list changed")

    designs = {
        model: _design_for_model(combined, model)
        for model in dict.fromkeys((*direct_models, *natural_models, *jensen_models))
    }
    if np.linalg.matrix_rank(designs[QUADRATIC_MODEL][0][:n_calibration]) != 92:
        raise ValueError("quadratic calibration design lost rank before fit")

    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    def record_fit(
        *,
        selector_family: str,
        model: str,
        target: float | None,
        response: np.ndarray,
        prediction_transform: str,
    ) -> None:
        design, terms = designs[model]
        coefficient, fitted, candidate_raw = _fit_one(
            design,
            response,
            n_calibration=n_calibration,
            ridge=config.ridge,
        )
        singular = np.linalg.svd(design[:n_calibration], compute_uv=False)
        nonzero = singular[singular > singular.max() * 1e-10]
        target_values = config.targets if target is None else (float(target),)
        coefficients.extend(
            {
                "scientific_status": PHASE6_STATUS,
                "selector_family": selector_family,
                "model": model,
                "target": target,
                "measurement_budget": config.measurement_budget,
                "term": term,
                "coefficient": float(value),
            }
            for term, value in zip(terms, coefficient)
        )
        for current_target in target_values:
            diagnostics.append(
                {
                    "scientific_status": PHASE6_STATUS,
                    "selector_family": selector_family,
                    "model": model,
                    "target": float(current_target),
                    "fit_is_shared_across_targets": target is None,
                    "measurement_budget": config.measurement_budget,
                    "ridge": config.ridge,
                    "design_rank": int(np.linalg.matrix_rank(design[:n_calibration])),
                    "n_columns": len(terms),
                    "nonzero_condition_number": float(nonzero[0] / nonzero[-1]),
                    "train_calibration_fit_mae": float(np.mean(np.abs(fitted - response))),
                    "response_min": float(np.min(response)),
                    "response_max": float(np.max(response)),
                }
            )
            if prediction_transform == "natural_plugin":
                predicted_loss = np.abs(candidate_raw - float(current_target))
                predicted_mean = candidate_raw
            else:
                predicted_loss = candidate_raw
                predicted_mean = np.full(len(candidate_raw), np.nan)
            predictions.extend(
                {
                    "scientific_status": PHASE6_STATUS,
                    "selector_family": selector_family,
                    "model": model,
                    "target": float(current_target),
                    "measurement_budget": config.measurement_budget,
                    "mask_id": str(mask.mask_id),
                    "predicted_target_loss": float(loss),
                    "predicted_mean_effect": (
                        float(mean) if np.isfinite(mean) else np.nan
                    ),
                }
                for mask, loss, mean in zip(
                    candidates.itertuples(index=False), predicted_loss, predicted_mean
                )
            )

    for target in config.targets:
        response = (
            train.assign(
                target_loss=np.abs(train["drop_from_clean"].to_numpy(float) - target)
            )
            .groupby("mask_id")["target_loss"]
            .mean()
        )
        ordered_response = np.asarray([response[mask_id] for mask_id in calibration_ids])
        for model in direct_models:
            record_fit(
                selector_family=DIRECT_RISK,
                model=model,
                target=target,
                response=ordered_response,
                prediction_transform="raw_risk",
            )

    mean_response = np.asarray([mean_effect[mask_id] for mask_id in calibration_ids])
    for model in natural_models:
        record_fit(
            selector_family=NATURAL_MEAN,
            model=model,
            target=None,
            response=mean_response,
            prediction_transform="natural_plugin",
        )

    for target in config.targets:
        jensen_response = np.abs(mean_response - target)
        for model in jensen_models:
            record_fit(
                selector_family=JENSEN_SCORE,
                model=model,
                target=target,
                response=jensen_response,
                prediction_transform="raw_jensen_score",
            )

    prediction_frame = pd.DataFrame(predictions)
    diagnostic_frame = pd.DataFrame(diagnostics)
    negative = (
        prediction_frame.groupby(["selector_family", "model", "target"])[
            "predicted_target_loss"
        ]
        .apply(lambda values: float((values < 0.0).mean()))
        .rename("candidate_negative_prediction_fraction")
        .reset_index()
    )
    diagnostic_frame = diagnostic_frame.merge(
        negative,
        on=["selector_family", "model", "target"],
        how="left",
        validate="one_to_one",
    )
    return prediction_frame, pd.DataFrame(coefficients), diagnostic_frame


def select_phase6_actions(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    config: Phase6FreezeConfig,
) -> pd.DataFrame:
    """Freeze one deterministic candidate mask per observer/pool/target/policy."""

    meta = candidates[["mask_id", "pool_id", "n_heads", "size_match_cell"]].copy()
    meta["mask_id"] = meta["mask_id"].astype(str)
    frame = predictions.merge(meta, on="mask_id", how="left", validate="many_to_one")
    if frame[["pool_id", "n_heads"]].isna().any().any():
        raise ValueError("predictions include a mask outside the candidate bank")
    rows: list[dict[str, Any]] = []
    grouping = ["selector_family", "model", "target", "pool_id"]
    for keys, group in frame.groupby(grouping, sort=True):
        if len(group) != 32:
            raise ValueError("every frozen action must choose among exactly 32 masks")
        group = group.reset_index(drop=True)
        ids = group["mask_id"].astype(str).to_numpy()
        counts = group["n_heads"].to_numpy(int)
        predicted = group["predicted_target_loss"].to_numpy(float)
        if not np.isfinite(predicted).all():
            raise ValueError("candidate predictions contain a non-finite value")
        for policy, objective in (
            (TARGET_POLICY, predicted),
            (COST_POLICY, predicted + config.head_cost_penalty * counts),
        ):
            selected = int(np.lexsort((ids, counts, objective))[0])
            rows.append(
                {
                    "scientific_status": PHASE6_STATUS,
                    "selector_family": str(keys[0]),
                    "model": str(keys[1]),
                    "target": float(keys[2]),
                    "pool_id": str(keys[3]),
                    "measurement_budget": config.measurement_budget,
                    "policy": policy,
                    "selected_mask_id": str(ids[selected]),
                    "selected_head_count": int(counts[selected]),
                    "selected_size_match_cell": str(
                        group.iloc[selected]["size_match_cell"]
                    ),
                    "predicted_target_loss": float(predicted[selected]),
                    "predicted_objective": float(objective[selected]),
                    "negative_prediction_selected": bool(predicted[selected] < 0.0),
                }
            )
    return pd.DataFrame(rows)


def freeze_phase6_predictions_and_actions(
    design_dir: str | Path,
    calibration_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    config: Phase6FreezeConfig = Phase6FreezeConfig(),
) -> Path:
    """Write the immutable prediction/action seal; never read Phase-6 test outcomes."""

    design_manifest, protocol, prompts, calibration, candidates = _verify_design_and_protocol(
        design_dir, protocol_path
    )
    calibration_manifest, train_effects = _verify_calibration_run(
        calibration_dir,
        design_dir,
        prompts=prompts,
        calibration=calibration,
    )
    predictions, coefficients, diagnostics = fit_phase6_observers(
        calibration,
        candidates,
        train_effects,
        protocol=protocol,
        config=config,
    )
    actions = select_phase6_actions(predictions, candidates, config=config)
    expected_prediction_rows = (5 + 2 + 1) * 3 * len(candidates)
    expected_action_rows = (5 + 2 + 1) * 3 * 48 * 2
    if len(predictions) != expected_prediction_rows:
        raise AssertionError("unexpected frozen prediction row count")
    if len(actions) != expected_action_rows:
        raise AssertionError("unexpected frozen action row count")

    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "prediction/action output is not empty; a freeze seal is never overwritten"
        )
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "candidate_predictions.csv": predictions,
        "observer_coefficients.csv": coefficients,
        "fit_diagnostics.csv": diagnostics,
        "fixed_actions.csv": actions,
    }
    for filename, frame in frames.items():
        frame.to_csv(output / filename, index=False)
    manifest = {
        "schema": FREEZE_SCHEMA,
        "status": FREEZE_STATUS,
        "scientific_status": PHASE6_STATUS,
        "design_id": design_manifest["design_id"],
        "design_manifest_sha256": file_sha256(
            Path(design_dir) / "design_manifest.json"
        ),
        "calibration_manifest_sha256": file_sha256(
            Path(calibration_dir) / "calibration_manifest.json"
        ),
        "protocol_sha256": file_sha256(protocol_path),
        "accessed_prompt_splits": ["reference", "train"],
        "accessed_mask_banks_for_outcomes": ["calibration"],
        "test_prompt_forward_passes": 0,
        "candidate_mask_forward_passes": 0,
        "contains_phase6_test_outcomes": False,
        "selection_rule": protocol["selection_rule"],
        "hypotheses": protocol["hypotheses"],
        "multiplicity_boundary": protocol["multiplicity_boundary"],
        "models": {
            "direct_risk": protocol["direct_risk_models"],
            "natural_mean_effect": protocol["mean_effect_models"],
            "target_specific_jensen_score": protocol[
                "jensen_score_sensitivity_models"
            ],
        },
        "counts": {
            "train_calibration_effect_cells": len(train_effects),
            "candidate_masks": len(candidates),
            "candidate_pools": int(candidates["pool_id"].nunique()),
            "prediction_rows": len(predictions),
            "coefficient_rows": len(coefficients),
            "diagnostic_rows": len(diagnostics),
            "fixed_action_rows": len(actions),
        },
        "source_seals": {
            "design": design_manifest["artifact_hashes"],
            "calibration": calibration_manifest["artifacts"],
        },
        "artifact_hashes": {
            filename: file_sha256(output / filename) for filename in frames
        },
        "runtime": runtime_provenance(),
        "next_allowed_stage": (
            "After a separate blind audit verifies this manifest, measure clean test "
            "scores and only the frozen candidate masks selected or required by the "
            "prespecified evaluation. No design, fit, prediction, or action may change."
        ),
    }
    write_json(output / "prediction_action_manifest.json", manifest)
    return output


def validate_phase6_prediction_action_seal(
    freeze_dir: str | Path,
    *,
    design_dir: str | Path,
    calibration_dir: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Verify the immutable seal before a separate test runner is allowed to start."""

    root = Path(freeze_dir)
    manifest = json.loads(
        (root / "prediction_action_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != FREEZE_SCHEMA or manifest.get("status") != FREEZE_STATUS:
        raise ValueError("prediction/action seal is absent or has the wrong status")
    expected_sources = {
        "design_manifest_sha256": file_sha256(
            Path(design_dir) / "design_manifest.json"
        ),
        "calibration_manifest_sha256": file_sha256(
            Path(calibration_dir) / "calibration_manifest.json"
        ),
        "protocol_sha256": file_sha256(protocol_path),
    }
    for field, expected in expected_sources.items():
        if manifest.get(field) != expected:
            raise ValueError(f"prediction/action seal source changed: {field}")
    if manifest.get("contains_phase6_test_outcomes") is not False:
        raise ValueError("prediction/action seal contains test outcomes")
    if int(manifest.get("test_prompt_forward_passes", -1)) != 0:
        raise ValueError("test prompts were opened before the seal")
    if int(manifest.get("candidate_mask_forward_passes", -1)) != 0:
        raise ValueError("candidate masks were opened before the seal")
    for filename, expected in manifest["artifact_hashes"].items():
        if file_sha256(root / filename) != expected:
            raise ValueError(f"frozen prediction/action artifact changed: {filename}")
    return manifest
