"""Evaluate the frozen post-confirmatory Phase-8 IOI sensitivity.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

All targets and references are reported.  There is no success gate: the study
is a secondary, post-review target-sensitivity analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance
from observerbench.tasks.ioi.phase7_evaluation import (
    _load_selected_effects,
    fixed_action_outcomes,
)
from observerbench.tasks.ioi.phase7_measurement import (
    load_phase7_measurement_inputs,
    phase7_measurement_source_hashes,
    validate_selected_effect_shard,
)
from observerbench.tasks.ioi.phase8_measurement import (
    MEASUREMENT_SCHEMA,
    MEASUREMENT_STATUS,
    load_phase8_measurement_inputs,
    phase8_measurement_source_hashes,
)
from observerbench.tasks.ioi.phase8_sensitivity import (
    DIRECT_RISK,
    EXACT_NOOP,
    NATURAL_MEAN,
    SCIENTIFIC_STATUS,
    TRANSFORMED_MEAN,
    Phase8Paths,
    _bool_column,
    load_phase8_protocol,
)


EVALUATION_SCHEMA = "observerbench.ioi_phase08_evaluation.v1"
EVALUATION_STATUS = "post_confirmatory_secondary_sensitivity_complete"
REFERENCES = (NATURAL_MEAN, TRANSFORMED_MEAN, EXACT_NOOP)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_new_effects(
    measurement_dir: str | Path,
    *,
    prompts: pd.DataFrame,
    masks: pd.DataFrame,
    clean: pd.DataFrame,
    expected_sources: Mapping[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, tuple[Path, ...]]:
    root = Path(measurement_dir)
    manifest = _read_json(root / "measurement_manifest.json")
    if manifest.get("schema") != MEASUREMENT_SCHEMA or manifest.get("status") != MEASUREMENT_STATUS:
        raise ValueError("Phase-8 new selected-mask measurement is incomplete")
    if manifest.get("source_hashes") != dict(expected_sources):
        raise ValueError("Phase-8 measurement used different frozen sources")
    expected_labels = {
        "measurement_run_spec.json",
        "measurement_progress.json",
        *(
            f"shards/test/effects_{start:04d}_{min(start + 16, len(masks)):04d}.csv"
            for start in range(0, len(masks), 16)
        ),
    }
    if set(manifest.get("artifact_hashes", {})) != expected_labels:
        raise ValueError("Phase-8 measurement artifact set changed")
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Phase-8 measurement artifact changed: {relative}")
    paths = tuple(sorted((root / "shards" / "test").glob("effects_*.csv")))
    if len(paths) != 10:
        raise ValueError("Phase-8 measurement must contain ten deterministic shards")
    frames: list[pd.DataFrame] = []
    ordered = masks.sort_values("mask_id").reset_index(drop=True)
    for start in range(0, len(ordered), 16):
        stop = min(start + 16, len(ordered))
        path = root / "shards" / "test" / f"effects_{start:04d}_{stop:04d}.csv"
        rows = pd.read_csv(
            path,
            dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
        )
        validate_selected_effect_shard(
            rows,
            prompts=prompts,
            masks=ordered.iloc[start:stop].reset_index(drop=True),
            clean=clean,
        )
        frames.append(rows)
    effects = pd.concat(frames, ignore_index=True)
    if len(effects) != 148 * 512 or effects.duplicated(["prompt_id", "mask_id"]).any():
        raise ValueError("Phase-8 new outcome table is incomplete")
    return manifest, effects, paths


def _outcome_cells(outcomes: pd.DataFrame) -> pd.DataFrame:
    cells = outcomes.groupby(
        ["target", "selector", "unordered_name_pair_id", "pool_id"],
        as_index=False,
    ).agg(actual_target_loss=("actual_target_loss", "mean"), prompt_count=("prompt_id", "size"))
    if set(cells["prompt_count"].astype(int)) != {16}:
        raise ValueError("every Phase-8 pair-pool cell must average 16 prompts")
    if len(cells) != 3 * 4 * 32 * 48:
        raise ValueError("Phase-8 pair-pool cell table is incomplete")
    return cells


def _loss_matrix(
    cells: pd.DataFrame,
    *,
    target: float,
    selector: str,
) -> pd.DataFrame:
    frame = cells.loc[
        np.isclose(cells["target"].to_numpy(float), float(target))
        & (cells["selector"].astype(str) == selector)
    ]
    matrix = frame.pivot(
        index="unordered_name_pair_id",
        columns="pool_id",
        values="actual_target_loss",
    ).sort_index().sort_index(axis=1)
    if matrix.shape != (32, 48) or not np.isfinite(matrix.to_numpy(float)).all():
        raise ValueError("Phase-8 contrast matrix is incomplete")
    return matrix


def secondary_contrasts(
    outcomes: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Report each target and a fixed-target equally weighted aggregate."""

    cells = _outcome_cells(outcomes)
    targets = tuple(map(float, protocol["targets"]))
    bootstrap = protocol["bootstrap"]
    repeats = int(bootstrap["repeats"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    pair_draws = rng.integers(0, 32, size=(repeats, 32))
    pool_draws = rng.integers(0, 48, size=(repeats, 48))
    rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    sign_rows: list[dict[str, Any]] = []
    for reference in REFERENCES:
        differences: dict[float, np.ndarray] = {}
        references: dict[float, np.ndarray] = {}
        candidates: dict[float, np.ndarray] = {}
        for target in targets:
            candidate_frame = _loss_matrix(
                cells, target=target, selector=DIRECT_RISK
            )
            reference_frame = _loss_matrix(
                cells, target=target, selector=reference
            ).reindex(index=candidate_frame.index, columns=candidate_frame.columns)
            if reference_frame.isna().any().any():
                raise ValueError("Phase-8 reference labels do not align with direct risk")
            candidate_values = candidate_frame.to_numpy(float)
            reference_values = reference_frame.to_numpy(float)
            difference = reference_values - candidate_values
            differences[target] = difference
            references[target] = reference_values
            candidates[target] = candidate_values
            draws = np.empty(repeats, dtype=float)
            for index in range(repeats):
                draws[index] = float(
                    difference[np.ix_(pair_draws[index], pool_draws[index])].mean()
                )
            reference_mean = float(reference_values.mean())
            candidate_mean = float(candidate_values.mean())
            mean = float(difference.mean())
            rows.append(
                {
                    "analysis_status": SCIENTIFIC_STATUS,
                    "target_scope": f"target_{target:g}",
                    "target": target,
                    "candidate": DIRECT_RISK,
                    "reference": reference,
                    "reference_mean_loss": reference_mean,
                    "candidate_mean_loss": candidate_mean,
                    "absolute_loss_reduction": mean,
                    "relative_loss_reduction": mean / reference_mean,
                    "q025": float(np.quantile(draws, 0.025, method="linear")),
                    "q975": float(np.quantile(draws, 0.975, method="linear")),
                    "bootstrap_repeats": repeats,
                    "secondary_no_success_gate": True,
                }
            )
            pool_means = difference.mean(axis=0)
            sign_rows.append(
                {
                    "target_scope": f"target_{target:g}",
                    "reference": reference,
                    "positive_pool_count": int((pool_means > 0).sum()),
                    "zero_pool_count": int((pool_means == 0).sum()),
                    "negative_pool_count": int((pool_means < 0).sum()),
                    "diagnostic_only": True,
                }
            )
            for pair_index in range(32):
                for pool_index in range(48):
                    cell_rows.append(
                        {
                            "target": target,
                            "reference": reference,
                            "pair_index": pair_index,
                            "pool_index": pool_index,
                            "reference_minus_direct_loss": float(
                                difference[pair_index, pool_index]
                            ),
                        }
                    )

        # The same resampled pair/pool indices apply to all three fixed targets;
        # targets themselves are never resampled.
        aggregate_difference = np.mean(
            np.stack([differences[target] for target in targets], axis=0), axis=0
        )
        aggregate_reference = np.mean(
            np.stack([references[target] for target in targets], axis=0), axis=0
        )
        aggregate_candidate = np.mean(
            np.stack([candidates[target] for target in targets], axis=0), axis=0
        )
        draws = np.empty(repeats, dtype=float)
        for index in range(repeats):
            draws[index] = float(
                aggregate_difference[
                    np.ix_(pair_draws[index], pool_draws[index])
                ].mean()
            )
        reference_mean = float(aggregate_reference.mean())
        candidate_mean = float(aggregate_candidate.mean())
        mean = float(aggregate_difference.mean())
        rows.append(
            {
                "analysis_status": SCIENTIFIC_STATUS,
                "target_scope": "all_three_equal_weight",
                "target": np.nan,
                "candidate": DIRECT_RISK,
                "reference": reference,
                "reference_mean_loss": reference_mean,
                "candidate_mean_loss": candidate_mean,
                "absolute_loss_reduction": mean,
                "relative_loss_reduction": mean / reference_mean,
                "q025": float(np.quantile(draws, 0.025, method="linear")),
                "q975": float(np.quantile(draws, 0.975, method="linear")),
                "bootstrap_repeats": repeats,
                "secondary_no_success_gate": True,
            }
        )
        pool_means = aggregate_difference.mean(axis=0)
        sign_rows.append(
            {
                "target_scope": "all_three_equal_weight",
                "reference": reference,
                "positive_pool_count": int((pool_means > 0).sum()),
                "zero_pool_count": int((pool_means == 0).sum()),
                "negative_pool_count": int((pool_means < 0).sum()),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(cell_rows), pd.DataFrame(sign_rows)


def evaluate_phase8_sensitivity(
    paths: Phase8Paths,
    freeze_dir: str | Path,
    audit_dir: str | Path,
    new_measurement_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    repository_root: str | Path,
) -> Path:
    """Combine sealed old/new rows and report every secondary contrast."""

    protocol = load_phase8_protocol(protocol_path)
    (
        freeze,
        _design,
        prompts,
        inherited_masks,
        new_masks,
        clean,
        _means,
        _templates,
    ) = load_phase8_measurement_inputs(
        paths,
        freeze_dir,
        audit_dir,
        protocol_path=protocol_path,
        repository_root=repository_root,
    )
    phase7_sources = phase7_measurement_source_hashes(
        paths.phase7_design,
        paths.phase7_pretest,
        paths.phase7_freeze,
        paths.phase7_audit,
        paths.phase5_effects,
        protocol_path=paths.phase7_protocol,
    )
    _manifest, old_effects, old_paths = _load_selected_effects(
        paths.phase7_measurement,
        prompts=prompts,
        selected=inherited_masks,
        clean=clean,
        expected_source_hashes=phase7_sources,
    )
    phase8_sources = phase8_measurement_source_hashes(
        paths, freeze_dir, audit_dir, protocol_path=protocol_path
    )
    new_manifest, new_effects, new_paths = _load_new_effects(
        new_measurement_dir,
        prompts=prompts,
        masks=new_masks,
        clean=clean,
        expected_sources=phase8_sources,
    )
    effects = pd.concat([old_effects, new_effects], ignore_index=True)
    if len(effects) != 237 * 512 or effects.duplicated(["prompt_id", "mask_id"]).any():
        raise ValueError("combined Phase-8 outcome union is not 237 by 512")

    actions = pd.read_csv(Path(freeze_dir) / "fixed_actions.csv")
    actions["selected_is_noop"] = _bool_column(actions, "selected_is_noop")
    if len(actions) != 4 * 3 * 48:
        raise ValueError("Phase-8 frozen action table is incomplete")
    outcome_frames: list[pd.DataFrame] = []
    for target in map(float, protocol["targets"]):
        target_actions = actions.loc[
            np.isclose(actions["target"].to_numpy(float), target)
        ].copy()
        frame = fixed_action_outcomes(
            prompts, target_actions, effects, target=target
        )
        frame.insert(0, "target", target)
        outcome_frames.append(frame)
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    if len(outcomes) != 3 * 4 * 48 * 512:
        raise AssertionError("Phase-8 fixed-action outcome count changed")

    contrasts, cell_differences, pool_signs = secondary_contrasts(
        outcomes, protocol=protocol
    )
    summary = outcomes.groupby(["target", "selector"], as_index=False).agg(
        mean_target_loss=("actual_target_loss", "mean"),
        mean_finite_effect=("finite_effect", "mean"),
        selected_noop_fraction=("selected_is_noop", "mean"),
        mean_selected_head_count=("selected_head_count", "mean"),
    )
    fit_diagnostics = pd.read_csv(Path(freeze_dir) / "fit_diagnostics.csv")
    selected_diagnostics = actions.groupby(["target", "selector"], as_index=False).agg(
        selected_action_count=("pool_id", "size"),
        selected_noop_count=("selected_is_noop", "sum"),
        selected_raw_negative_count=(
            "predicted_target_loss",
            lambda values: int((pd.to_numeric(values) < 0).sum()),
        ),
        selected_raw_negative_fraction=(
            "predicted_target_loss",
            lambda values: float((pd.to_numeric(values) < 0).mean()),
        ),
    )
    prediction_diagnostics = fit_diagnostics.merge(
        selected_diagnostics,
        on=["target", "selector"],
        how="left",
        validate="one_to_one",
    )

    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a Phase-8 evaluation is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "secondary_contrasts.csv": contrasts,
        "pair_pool_differences.csv": cell_differences,
        "pool_signs.csv": pool_signs,
        "observer_summary.csv": summary,
        "prediction_diagnostics.csv": prediction_diagnostics,
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    outcomes.to_csv(
        output / "fixed_action_prompt_losses.csv.gz",
        index=False,
        compression="gzip",
    )
    transformed = prediction_diagnostics.loc[
        prediction_diagnostics["selector"] == TRANSFORMED_MEAN
    ]
    digest = {
        "scientific_status": SCIENTIFIC_STATUS,
        "secondary_no_success_gate": True,
        "targets": list(map(float, protocol["targets"])),
        "contrasts": contrasts.to_dict(orient="records"),
        "transformed_mean_prediction_diagnostics": transformed.to_dict(
            orient="records"
        ),
        "reused_phase7_masks": 89,
        "newly_measured_masks": int(
            new_manifest["counts"]["new_selected_unique_nonnoop_masks"]
        ),
        "combined_selected_mask_union": 237,
        "noop_ablation_forward_passes": 0,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(output / "result_digest.json", digest)
    artifacts = [
        *frames,
        "fixed_action_prompt_losses.csv.gz",
        "result_digest.json",
    ]
    manifest = {
        "schema": EVALUATION_SCHEMA,
        "status": EVALUATION_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "secondary_no_success_gate": True,
        "protocol_sha256": file_sha256(protocol_path),
        "freeze_manifest_sha256": file_sha256(
            Path(freeze_dir) / "prediction_action_manifest.json"
        ),
        "preoutcome_audit_sha256": file_sha256(
            Path(audit_dir) / "preoutcome_audit.json"
        ),
        "phase7_measurement_manifest_sha256": file_sha256(
            paths.phase7_measurement / "measurement_manifest.json"
        ),
        "phase8_measurement_manifest_sha256": file_sha256(
            Path(new_measurement_dir) / "measurement_manifest.json"
        ),
        "phase7_shard_hashes": {
            path.relative_to(paths.phase7_measurement).as_posix(): file_sha256(path)
            for path in old_paths
        },
        "phase8_shard_hashes": {
            path.relative_to(Path(new_measurement_dir)).as_posix(): file_sha256(path)
            for path in new_paths
        },
        "artifact_hashes": {
            name: file_sha256(output / name) for name in artifacts
        },
        "runtime": runtime_provenance(),
    }
    write_json(output / "evaluation_manifest.json", manifest)
    return output
