"""Frozen pair-cluster by pool evaluation for Phase 7.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance
from observerbench.tasks.ioi.phase7_confirmation import (
    DIRECT_RISK,
    EXACT_NOOP,
    NATURAL_MEAN,
    SCIENTIFIC_STATUS,
)
from observerbench.tasks.ioi.phase5_effects import validate_effect_rows
from observerbench.tasks.ioi.phase7_measurement import (
    MEASUREMENT_SCHEMA,
    MEASUREMENT_STATUS,
    load_phase7_measurement_inputs,
    phase7_measurement_source_hashes,
)


EVALUATION_SCHEMA = "observerbench.ioi_phase07_evaluation.v1"
EVALUATION_STATUS = "frozen_joint_primary_evaluation_complete"


def validate_phase7_measured_union(
    effects: pd.DataFrame,
    *,
    prompts: pd.DataFrame,
    selected: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    """Require the exact frozen prompt-by-selected-mask Cartesian product."""

    validate_effect_rows(
        effects,
        prompt_ids=prompts["prompt_id"].astype(str).tolist(),
        mask_ids=selected["mask_id"].astype(str).tolist(),
    )
    if set(effects["split"].astype(str)) != {"test"}:
        raise ValueError("Phase-7 measurement contains a non-test prompt")
    if set(effects["bank"].astype(str)) != {"candidate"}:
        raise ValueError("Phase-7 measurement contains a non-candidate mask")
    observed_mapping = effects[["mask_id", "mask_bits", "pool_id"]].drop_duplicates()
    expected_mapping = selected[["mask_id", "mask_bits", "pool_id"]].copy()
    for frame in (observed_mapping, expected_mapping):
        frame["mask_id"] = frame["mask_id"].astype(str)
        frame["mask_bits"] = frame["mask_bits"].astype(str).str.zfill(13)
        frame["pool_id"] = frame["pool_id"].astype(str)
        frame.sort_values("mask_id", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not observed_mapping.equals(expected_mapping):
        raise ValueError("Phase-7 measured mask union differs from the frozen union")
    clean_by_id = clean.set_index(clean["prompt_id"].astype(str))["clean_ld"]
    expected_clean = effects["prompt_id"].astype(str).map(clean_by_id).to_numpy(float)
    if not np.allclose(
        effects["clean_ld"].to_numpy(float), expected_clean, atol=1e-6, rtol=0
    ):
        raise ValueError("Phase-7 measurement clean scores differ from the pretest")


def _load_selected_effects(
    measurement_dir: str | Path,
    *,
    prompts: pd.DataFrame,
    selected: pd.DataFrame,
    clean: pd.DataFrame,
    expected_source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, tuple[Path, ...]]:
    root = Path(measurement_dir)
    manifest = json.loads(
        (root / "measurement_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != MEASUREMENT_SCHEMA or manifest.get("status") != MEASUREMENT_STATUS:
        raise ValueError("Phase-7 selected measurement is not complete")
    if manifest.get("source_hashes") != dict(expected_source_hashes):
        raise ValueError("Phase-7 measurement used different frozen sources")
    spec_path = root / "measurement_run_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("source_hashes") != dict(expected_source_hashes):
        raise ValueError("Phase-7 measurement spec used different frozen sources")
    expected_labels = {
        "measurement_run_spec.json",
        "measurement_progress.json",
        *(
            f"shards/test/effects_{start:04d}_{min(start + 16, len(selected)):04d}.csv"
            for start in range(0, len(selected), 16)
        ),
    }
    if set(map(str, manifest.get("artifact_hashes", {}))) != expected_labels:
        raise ValueError("Phase-7 measurement artifact set differs from the frozen scope")
    for label, expected in manifest.get("artifact_hashes", {}).items():
        if file_sha256(root / label) != expected:
            raise ValueError(f"Phase-7 measurement artifact changed: {label}")
    if int(manifest.get("unselected_candidate_masks_measured", -1)) != 0:
        raise ValueError("Phase-7 measurement accessed an unselected mask")
    if int(manifest.get("noop_ablation_forward_passes", -1)) != 0:
        raise ValueError("Phase-7 measurement ran a no-op ablation")
    paths = tuple(sorted((root / "shards" / "test").glob("effects_*.csv")))
    if not paths:
        raise FileNotFoundError("Phase-7 selected effect shards are missing")
    effects = pd.concat(
        [
            pd.read_csv(
                path,
                dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
            )
            for path in paths
        ],
        ignore_index=True,
    )
    if effects.duplicated(["prompt_id", "mask_id"]).any():
        raise ValueError("Phase-7 selected effects contain duplicate cells")
    counts = manifest.get("counts", {})
    expected_cells = int(counts.get("effect_cells", -1))
    expected_masks = int(counts.get("selected_unique_nonnoop_masks", -1))
    if len(effects) != expected_cells or effects["mask_id"].astype(str).nunique() != expected_masks:
        raise ValueError("Phase-7 selected effects do not match the sealed measurement count")
    if {
        path.relative_to(root).as_posix() for path in paths
    } != {label for label in expected_labels if label.startswith("shards/")}:
        raise ValueError("Phase-7 measurement has an unindexed or missing effect shard")
    validate_phase7_measured_union(
        effects, prompts=prompts, selected=selected, clean=clean
    )
    if not np.isfinite(
        effects[["clean_ld", "ablated_ld", "drop_from_clean"]].to_numpy(float)
    ).all():
        raise ValueError("Phase-7 selected effects contain a non-finite value")
    return manifest, effects, paths


def fixed_action_outcomes(
    prompts: pd.DataFrame,
    actions: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    target: float,
) -> pd.DataFrame:
    """Materialize every frozen action on every prompt, with analytic no-op."""

    matrix = effects.pivot(index="prompt_id", columns="mask_id", values="drop_from_clean")
    ordered_prompts = prompts.sort_values("prompt_id").reset_index(drop=True)
    matrix = matrix.reindex(index=ordered_prompts["prompt_id"].astype(str))
    rows: list[pd.DataFrame] = []
    for action in actions.itertuples(index=False):
        if bool(action.selected_is_noop):
            values = np.zeros(len(ordered_prompts), dtype=float)
        else:
            mask_id = str(action.selected_mask_id)
            if mask_id not in matrix.columns:
                raise ValueError(f"selected action lacks a measured effect: {mask_id}")
            values = matrix[mask_id].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"selected action has incomplete outcomes: {mask_id}")
        frame = ordered_prompts[
            [
                "prompt_id",
                "template_id",
                "structure",
                "unordered_name_pair_id",
                "pair_orientation",
            ]
        ].copy()
        frame["selector"] = str(action.selector)
        frame["pool_id"] = str(action.pool_id)
        frame["selected_mask_id"] = str(action.selected_mask_id)
        frame["selected_is_noop"] = bool(action.selected_is_noop)
        frame["selected_head_count"] = int(action.selected_head_count)
        frame["finite_effect"] = values
        frame["actual_target_loss"] = np.abs(values - float(target))
        rows.append(frame)
    outcomes = pd.concat(rows, ignore_index=True)
    expected = len(prompts) * len(actions)
    if len(outcomes) != expected:
        raise AssertionError("Phase-7 fixed-action outcome count changed")
    noop_loss = outcomes.loc[outcomes["selector"] == EXACT_NOOP, "actual_target_loss"]
    if len(noop_loss) != len(prompts) * 48 or not np.allclose(noop_loss, target):
        raise ValueError("exact no-op loss is not analytic and constant")
    return outcomes


def _cell_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    cells = outcomes.groupby(
        ["selector", "unordered_name_pair_id", "pool_id"], as_index=False
    ).agg(
        actual_target_loss=("actual_target_loss", "mean"),
        prompt_count=("prompt_id", "size"),
    )
    if set(cells["prompt_count"].astype(int)) != {16}:
        raise ValueError("every pair-cluster by pool action cell must average 16 prompts")
    expected = 3 * 32 * 48
    if len(cells) != expected:
        raise ValueError("Phase-7 action cell table is incomplete")
    return cells


def _comparison_matrix(
    cells: pd.DataFrame,
    *,
    reference: str,
    candidate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    candidate_frame = cells.loc[cells["selector"] == candidate]
    reference_frame = cells.loc[cells["selector"] == reference]
    candidate_matrix = candidate_frame.pivot(
        index="unordered_name_pair_id", columns="pool_id", values="actual_target_loss"
    ).sort_index().sort_index(axis=1)
    reference_matrix = reference_frame.pivot(
        index="unordered_name_pair_id", columns="pool_id", values="actual_target_loss"
    ).reindex(index=candidate_matrix.index, columns=candidate_matrix.columns)
    if candidate_matrix.shape != (32, 48) or reference_matrix.shape != (32, 48):
        raise ValueError("Phase-7 contrast matrix must be 32 pair clusters by 48 pools")
    candidate_values = candidate_matrix.to_numpy(float)
    reference_values = reference_matrix.to_numpy(float)
    if not np.isfinite(candidate_values).all() or not np.isfinite(reference_values).all():
        raise ValueError("Phase-7 contrast matrix contains a missing value")
    return (
        reference_values - candidate_values,
        reference_values,
        candidate_values,
        candidate_matrix.index.astype(str).tolist(),
        candidate_matrix.columns.astype(str).tolist(),
    )


def paired_cluster_pool_contrasts(
    outcomes: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the frozen two-axis paired bootstrap to both primary references."""

    cells = _cell_table(outcomes)
    bootstrap = protocol["bootstrap"]
    repeats = int(bootstrap["repeats"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    pair_draws = rng.integers(0, 32, size=(repeats, 32))
    pool_draws = rng.integers(0, 48, size=(repeats, 48))
    comparisons = (
        ("H1a_estimand", NATURAL_MEAN),
        ("H1b_intervention_value", EXACT_NOOP),
    )
    rows: list[dict[str, Any]] = []
    cell_rows: list[pd.DataFrame] = []
    sign_rows: list[dict[str, Any]] = []
    for comparison_id, reference in comparisons:
        difference, reference_values, candidate_values, pair_ids, pool_ids = (
            _comparison_matrix(cells, reference=reference, candidate=DIRECT_RISK)
        )
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
                "comparison_id": comparison_id,
                "candidate": DIRECT_RISK,
                "reference": reference,
                "target": float(protocol["target"]),
                "pair_clusters": 32,
                "candidate_pools": 48,
                "reference_mean_loss": reference_mean,
                "candidate_mean_loss": candidate_mean,
                "absolute_loss_reduction": mean,
                "relative_loss_reduction": mean / reference_mean,
                "q025": float(np.quantile(draws, 0.025, method="linear")),
                "q975": float(np.quantile(draws, 0.975, method="linear")),
                "bootstrap_repeats": repeats,
            }
        )
        pair_grid, pool_grid = np.meshgrid(pair_ids, pool_ids, indexing="ij")
        cell_rows.append(
            pd.DataFrame(
                {
                    "comparison_id": comparison_id,
                    "unordered_name_pair_id": pair_grid.ravel(),
                    "pool_id": pool_grid.ravel(),
                    "reference_minus_direct_loss": difference.ravel(),
                    "reference_loss": reference_values.ravel(),
                    "direct_risk_loss": candidate_values.ravel(),
                }
            )
        )
        pool_means = difference.mean(axis=0)
        sign_rows.append(
            {
                "comparison_id": comparison_id,
                "positive_pool_count": int((pool_means > 0.0).sum()),
                "zero_pool_count": int((pool_means == 0.0).sum()),
                "negative_pool_count": int((pool_means < 0.0).sum()),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows), pd.concat(cell_rows, ignore_index=True), pd.DataFrame(sign_rows)


def apply_joint_primary_gate(
    contrasts: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Require success against both same-basis mean plug-in and exact no-op."""

    hypotheses = protocol["primary_hypotheses"]
    records: dict[str, Any] = {}
    for comparison_id in ("H1a_estimand", "H1b_intervention_value"):
        row = contrasts.loc[contrasts["comparison_id"] == comparison_id]
        if len(row) != 1:
            raise ValueError(f"missing unique Phase-7 primary contrast: {comparison_id}")
        value = row.iloc[0]
        threshold = float(hypotheses[comparison_id]["minimum_relative_loss_reduction"])
        checks = {
            "relative_reduction_at_least_five_percent": bool(
                value["relative_loss_reduction"] >= threshold
            ),
            "paired_cluster_pool_interval_lower_strictly_positive": bool(
                value["q025"] > 0.0
            ),
        }
        records[comparison_id] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "relative_loss_reduction": float(value["relative_loss_reduction"]),
            "absolute_loss_reduction": float(value["absolute_loss_reduction"]),
            "q025": float(value["q025"]),
            "q975": float(value["q975"]),
        }
    joint = bool(all(item["passed"] for item in records.values()))
    return {
        "joint_primary_passed": joint,
        "comparisons": records,
        "joint_success_rule": hypotheses["joint_success_rule"],
        "licensed_claim": (
            "On these eight validated canonical templates and new prompt strings, the "
            "direct-risk observer selected fixed interventions with lower finite-effect "
            "target loss than both the same-basis mean-effect plug-in observer and exact "
            "no-op at the pilot-informed target t=1.0."
            if joint
            else "The joint Phase-7 primary claim is not licensed. Report both comparisons."
        ),
    }


def template_sensitivity(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Descriptive per-template directions; these do not rescue the primary gate."""

    means = outcomes.groupby(
        ["template_id", "selector"], as_index=False
    )["actual_target_loss"].mean()
    pivot = means.pivot(index="template_id", columns="selector", values="actual_target_loss")
    rows = []
    for template, row in pivot.iterrows():
        for comparison_id, reference in (
            ("H1a_estimand", NATURAL_MEAN),
            ("H1b_intervention_value", EXACT_NOOP),
        ):
            rows.append(
                {
                    "template_id": str(template),
                    "comparison_id": comparison_id,
                    "reference_mean_loss": float(row[reference]),
                    "direct_risk_mean_loss": float(row[DIRECT_RISK]),
                    "reference_minus_direct_loss": float(row[reference] - row[DIRECT_RISK]),
                    "descriptive_only": True,
                }
            )
    return pd.DataFrame(rows)


def evaluate_phase7_confirmation(
    design_dir: str | Path,
    pretest_dir: str | Path,
    freeze_dir: str | Path,
    audit_dir: str | Path,
    phase5_design_dir: str | Path,
    phase5_effects_dir: str | Path,
    measurement_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
) -> Path:
    """Evaluate the frozen joint primary test and write a complete audit trail."""

    freeze, design, prompts, selected, clean, _means, _templates = (
        load_phase7_measurement_inputs(
            design_dir,
            pretest_dir,
            freeze_dir,
            audit_dir,
            phase5_design_dir,
            phase5_effects_dir,
            protocol_path=protocol_path,
        )
    )
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    expected_sources = phase7_measurement_source_hashes(
        design_dir,
        pretest_dir,
        freeze_dir,
        audit_dir,
        phase5_effects_dir,
        protocol_path=protocol_path,
    )
    measurement, effects, shard_paths = _load_selected_effects(
        measurement_dir,
        prompts=prompts,
        selected=selected,
        clean=clean,
        expected_source_hashes=expected_sources,
    )
    if measurement.get("design_id") != design.get("design_id"):
        raise ValueError("Phase-7 measurement used a different design")
    actions = pd.read_csv(Path(freeze_dir) / "fixed_actions.csv")
    actions["selected_is_noop"] = actions["selected_is_noop"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if actions["selected_is_noop"].isna().any() or len(actions) != 144:
        raise ValueError("Phase-7 fixed action table is invalid")
    outcomes = fixed_action_outcomes(
        prompts, actions, effects, target=float(protocol["target"])
    )
    contrasts, cells, pool_signs = paired_cluster_pool_contrasts(
        outcomes, protocol=protocol
    )
    audit = apply_joint_primary_gate(contrasts, protocol=protocol)
    templates = template_sensitivity(outcomes)
    summary = outcomes.groupby("selector", as_index=False).agg(
        mean_target_loss=("actual_target_loss", "mean"),
        mean_finite_effect=("finite_effect", "mean"),
        selected_noop_fraction=("selected_is_noop", "mean"),
        mean_selected_head_count=("selected_head_count", "mean"),
    )

    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a Phase-7 evaluation is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "primary_contrasts.csv": contrasts,
        "pair_pool_cells.csv": cells,
        "pool_signs.csv": pool_signs,
        "template_sensitivity.csv": templates,
        "observer_summary.csv": summary,
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    outcomes.to_csv(output / "fixed_action_prompt_losses.csv.gz", index=False, compression="gzip")
    write_json(output / "hypothesis_audit.json", audit)
    digest = {
        "scientific_status": SCIENTIFIC_STATUS,
        "target": float(protocol["target"]),
        "target_is_pilot_informed": True,
        "clean_pretest_passed_before_candidate_measurement": True,
        "joint_primary": audit,
        "primary_contrasts": contrasts.to_dict(orient="records"),
        "selected_unique_nonnoop_masks_measured": int(
            measurement["counts"]["selected_unique_nonnoop_masks"]
        ),
        "unselected_candidate_masks_measured": 0,
        "noop_ablation_forward_passes": 0,
    }
    write_json(output / "result_digest.json", digest)
    artifacts = [
        *frames,
        "fixed_action_prompt_losses.csv.gz",
        "hypothesis_audit.json",
        "result_digest.json",
    ]
    manifest = {
        "schema": EVALUATION_SCHEMA,
        "status": EVALUATION_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design["design_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "design_manifest_sha256": file_sha256(Path(design_dir) / "design_manifest.json"),
        "pretest_manifest_sha256": file_sha256(Path(pretest_dir) / "pretest_manifest.json"),
        "prediction_action_manifest_sha256": file_sha256(
            Path(freeze_dir) / "prediction_action_manifest.json"
        ),
        "preoutcome_audit_sha256": file_sha256(
            Path(audit_dir) / "preoutcome_audit.json"
        ),
        "measurement_manifest_sha256": file_sha256(
            Path(measurement_dir) / "measurement_manifest.json"
        ),
        "measured_shard_hashes": {
            path.relative_to(Path(measurement_dir)).as_posix(): file_sha256(path)
            for path in shard_paths
        },
        "joint_primary_passed": bool(audit["joint_primary_passed"]),
        "artifact_hashes": {
            name: file_sha256(output / name) for name in artifacts
        },
        "runtime": runtime_provenance(),
    }
    write_json(output / "evaluation_manifest.json", manifest)
    return output
