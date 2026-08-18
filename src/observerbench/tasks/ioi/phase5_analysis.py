"""Confirmatory fitting and held-out action selection for Phase-5 IOI.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.effect_prediction import EffectObserverCard, EffectTaskCard
from observerbench.provenance import (
    file_sha256,
    json_sha256,
    runtime_provenance,
    source_hashes,
)
from observerbench.tasks.ioi.heads import head_records
from observerbench.tasks.ioi.phase2_capacity import (
    LoadedIOIRun,
    build_capacity_design,
)
from observerbench.tasks.ioi.phase5_effects import load_locked_ioi_design
from observerbench.tasks.ioi.stage2d import parse_mask_bits, ridge_fit


PHASE5_MODELS: tuple[str, ...] = (
    "additive_head",
    "count_additive",
    "count_plus_PE_bin4",
    "count_plus_all_bin4",
)
SINGLETON_MODEL = "singleton_sum"
PINNED_GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


@dataclass(frozen=True)
class IOIPhase5AnalysisConfig:
    budgets: tuple[int, ...] = (20, 40, 80, 160)
    targets: tuple[float, ...] = (0.5, 1.0, 1.5)
    models: tuple[str, ...] = PHASE5_MODELS
    ridge: float = 1e-6
    target_tolerance: float = 0.25
    head_cost_penalty: float = 0.02
    bootstrap_repeats: int = 2000
    seed: int = 25051
    regret_reduction_fraction_min: float = 0.15
    within_tolerance_improvement_min: float = 0.05
    paired_interval_excludes_zero: bool = True
    requires_size_matched_direction: bool = True
    requires_cost_aware_direction: bool = True

    def __post_init__(self) -> None:
        if not self.budgets or tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("budgets must be a non-empty increasing tuple")
        if not self.targets or not np.isfinite(self.targets).all():
            raise ValueError("targets must be non-empty and finite")
        if not set(self.models).issubset(PHASE5_MODELS):
            raise ValueError("unsupported Phase-5 model")
        if self.ridge < 0.0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")
        if self.target_tolerance <= 0.0 or not np.isfinite(self.target_tolerance):
            raise ValueError("target_tolerance must be finite and positive")
        if self.head_cost_penalty < 0.0 or not np.isfinite(self.head_cost_penalty):
            raise ValueError("head_cost_penalty must be finite and non-negative")
        if self.bootstrap_repeats <= 0:
            raise ValueError("bootstrap_repeats must be positive")
        if not 0.0 <= self.regret_reduction_fraction_min <= 1.0:
            raise ValueError("regret_reduction_fraction_min must lie in [0, 1]")
        if not 0.0 <= self.within_tolerance_improvement_min <= 1.0:
            raise ValueError("within_tolerance_improvement_min must lie in [0, 1]")

    @classmethod
    def from_protocol(
        cls,
        protocol: Mapping[str, Any],
        *,
        bootstrap_repeats: int | None = None,
    ) -> "IOIPhase5AnalysisConfig":
        """Build analysis settings from the frozen confirmatory protocol."""

        if protocol.get("schema") != "observerbench.ioi_selection_protocol.v2":
            raise ValueError("expected observerbench.ioi_selection_protocol.v2")
        tolerance_metrics = [
            str(metric)
            for metric in protocol.get("secondary_metrics", ())
            if re.fullmatch(r"within_[0-9]+(?:\.[0-9]+)?", str(metric))
        ]
        if len(tolerance_metrics) != 1:
            raise ValueError("protocol must name exactly one within_<tolerance> metric")
        success = protocol.get("success")
        if not isinstance(success, Mapping):
            raise ValueError("protocol success thresholds are missing")
        return cls(
            budgets=tuple(int(value) for value in protocol["measurement_budgets"]),
            targets=tuple(float(value) for value in protocol["targets"]),
            models=tuple(str(value) for value in protocol["models"]),
            ridge=float(protocol["ridge"]),
            target_tolerance=float(tolerance_metrics[0].split("_", 1)[1]),
            head_cost_penalty=float(protocol["head_cost_penalty"]),
            bootstrap_repeats=(
                int(protocol["bootstrap_repeats"])
                if bootstrap_repeats is None
                else int(bootstrap_repeats)
            ),
            seed=int(protocol["seed"]),
            regret_reduction_fraction_min=float(
                success["regret_reduction_fraction_min"]
            ),
            within_tolerance_improvement_min=float(
                success["within_tolerance_improvement_min"]
            ),
            paired_interval_excludes_zero=bool(
                success["paired_interval_excludes_zero"]
            ),
            requires_size_matched_direction=bool(
                success["requires_size_matched_direction"]
            ),
            requires_cost_aware_direction=bool(
                success["requires_cost_aware_direction"]
            ),
        )


@dataclass(frozen=True)
class IOIPhase5EvaluationConfig:
    """Evaluation-only settings frozen after measurement and before outcomes."""

    primary_budget: int = 160
    targets: tuple[float, ...] = (0.5, 1.0, 1.5)
    target_tolerance: float = 0.25
    head_cost_penalty: float = 0.02
    bootstrap_repeats: int = 2000
    seed: int = 25051
    fixed_action_loss_reduction_fraction_min: float = 0.15
    within_tolerance_improvement_min: float = 0.05
    paired_interval_excludes_zero: bool = True
    requires_size_matched_direction: bool = True
    requires_cost_aware_direction: bool = True

    def __post_init__(self) -> None:
        if self.primary_budget <= 0:
            raise ValueError("primary_budget must be positive")
        if not self.targets or not np.isfinite(self.targets).all():
            raise ValueError("targets must be non-empty and finite")
        if self.target_tolerance <= 0.0 or not np.isfinite(self.target_tolerance):
            raise ValueError("target_tolerance must be finite and positive")
        if self.head_cost_penalty < 0.0 or not np.isfinite(self.head_cost_penalty):
            raise ValueError("head_cost_penalty must be finite and non-negative")
        if self.bootstrap_repeats <= 0:
            raise ValueError("bootstrap_repeats must be positive")
        if not 0.0 <= self.fixed_action_loss_reduction_fraction_min <= 1.0:
            raise ValueError(
                "fixed_action_loss_reduction_fraction_min must lie in [0, 1]"
            )
        if not 0.0 <= self.within_tolerance_improvement_min <= 1.0:
            raise ValueError("within_tolerance_improvement_min must lie in [0, 1]")

    @classmethod
    def from_protocol(
        cls,
        protocol: Mapping[str, Any],
        *,
        bootstrap_repeats: int | None = None,
    ) -> "IOIPhase5EvaluationConfig":
        if protocol.get("schema") != "observerbench.ioi_evaluation_protocol.v3":
            raise ValueError("expected observerbench.ioi_evaluation_protocol.v3")
        if protocol.get("status") != "frozen_after_measurement_before_heldout_open":
            raise ValueError("evaluation protocol was not frozen at the required boundary")
        success = protocol.get("success")
        if not isinstance(success, Mapping):
            raise ValueError("evaluation success thresholds are missing")
        return cls(
            primary_budget=int(protocol["primary_budget"]),
            targets=tuple(float(value) for value in protocol["targets"]),
            target_tolerance=float(protocol["target_tolerance"]),
            head_cost_penalty=float(protocol["head_cost_penalty"]),
            bootstrap_repeats=(
                int(protocol["bootstrap_repeats"])
                if bootstrap_repeats is None
                else int(bootstrap_repeats)
            ),
            seed=int(protocol["seed"]),
            fixed_action_loss_reduction_fraction_min=float(
                success["fixed_action_loss_reduction_fraction_min"]
            ),
            within_tolerance_improvement_min=float(
                success["within_tolerance_improvement_min"]
            ),
            paired_interval_excludes_zero=bool(
                success["paired_interval_excludes_zero"]
            ),
            requires_size_matched_direction=bool(
                success["requires_size_matched_direction"]
            ),
            requires_cost_aware_direction=bool(
                success["requires_cost_aware_direction"]
            ),
        )


def _load_split_effects(effects_dir: str | Path, split: str) -> tuple[pd.DataFrame, tuple[Path, ...]]:
    paths = tuple(sorted((Path(effects_dir) / "shards" / split).glob("effects_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no effect shards found for split {split!r}")
    rows = pd.concat(
        [
            pd.read_csv(
                path,
                dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
            )
            for path in paths
        ],
        ignore_index=True,
    )
    if rows.duplicated(["prompt_id", "mask_id"]).any():
        raise ValueError(f"duplicate prompt-mask effects in {split}")
    if set(rows["split"].astype(str)) != {split}:
        raise ValueError(f"{split} shard directory contains another split")
    if not np.isfinite(rows[["clean_ld", "ablated_ld", "drop_from_clean"]]).all().all():
        raise ValueError(f"{split} effects contain non-finite values")
    return rows, paths


def _validate_effect_manifest(
    effects_dir: str | Path,
    design_dir: str | Path,
) -> dict[str, Any]:
    """Verify the complete measurement artifact without interpreting outcomes."""

    root = Path(effects_dir)
    manifest_path = root / "effect_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("effect_manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "observerbench.ioi_effect_run.v1":
        raise ValueError("unexpected IOI effect manifest schema")
    if manifest.get("status") != "complete_unopened_confirmatory_outcomes":
        raise ValueError("confirmatory effect measurement is not complete and sealed")
    if manifest.get("design_manifest_sha256") != file_sha256(
        Path(design_dir) / "design_manifest.json"
    ):
        raise ValueError("effect measurement used a different frozen design")
    model = manifest.get("model")
    if not isinstance(model, Mapping) or model.get("requested_revision") != PINNED_GPT2_REVISION:
        raise ValueError("effect measurement did not use the pinned GPT-2 revision")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("effect manifest has no artifact hashes")
    if any(
        Path(str(label)).is_absolute() or ".." in Path(str(label)).parts
        for label in artifacts
    ):
        raise ValueError("effect artifact paths must be effects-root-relative")

    prompts, masks, _design_manifest = load_locked_ioi_design(design_dir)
    outcome_prompts = int(prompts["split"].isin(("train", "validation", "test")).sum())
    reference_prompts = int((prompts["split"] == "reference").sum())
    mask_count = int(len(masks))
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("effect manifest has no count record")
    expected_counts = {
        "reference_prompts": reference_prompts,
        "outcome_prompts": outcome_prompts,
        "masks": mask_count,
        "effect_cells": outcome_prompts * mask_count,
    }
    for name, expected in expected_counts.items():
        if int(counts.get(name, -1)) != expected:
            raise ValueError(f"effect manifest {name} differs from the frozen design")

    expected_labels = {
        "template_head_means.npz",
        "clean_scores_train.csv",
        "clean_scores_validation.csv",
        "clean_scores_test.csv",
    }
    expected_labels.update(
        path.relative_to(root).as_posix()
        for split in ("train", "validation", "test")
        for path in sorted((root / "shards" / split).glob("effects_*.csv"))
    )
    if set(map(str, artifacts)) != expected_labels:
        raise ValueError("effect manifest does not index the exact artifact set")
    expected_shards = int(counts.get("shards", -1))
    if expected_shards != len(expected_labels) - 4:
        raise ValueError("effect manifest shard count disagrees with its artifact index")
    for label, expected_sha256 in artifacts.items():
        path = root / str(label)
        if not path.is_file():
            raise FileNotFoundError(f"effect artifact is missing: {label}")
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"effect artifact hash mismatch: {label}")
    return manifest


def _validate_split_cartesian(
    rows: pd.DataFrame,
    prompts: pd.DataFrame,
    masks: pd.DataFrame,
    *,
    split: str,
) -> None:
    """Require the exact frozen prompt-by-mask effect table and mask mapping."""

    required = {"prompt_id", "mask_id", "mask_bits", "bank", "pool_id", "split"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{split} effects are missing columns: {missing}")
    prompt_ids = set(
        prompts.loc[prompts["split"] == split, "prompt_id"].astype(str)
    )
    mask_ids = set(masks["mask_id"].astype(str))
    if set(rows["prompt_id"].astype(str)) != prompt_ids:
        raise ValueError(f"{split} effects do not contain the exact prompt set")
    if set(rows["mask_id"].astype(str)) != mask_ids:
        raise ValueError(f"{split} effects do not contain the exact mask set")
    if len(rows) != len(prompt_ids) * len(mask_ids):
        raise ValueError(f"{split} effects are not a complete prompt-by-mask Cartesian table")
    if rows.duplicated(["prompt_id", "mask_id"]).any():
        raise ValueError(f"{split} effects contain duplicate prompt-mask cells")
    mapping_columns = ["mask_bits", "bank", "pool_id"]
    expected_mapping = masks[["mask_id", *mapping_columns]].copy()
    observed = rows[["mask_id", *mapping_columns]].drop_duplicates().copy()
    if observed["mask_id"].astype(str).duplicated().any():
        raise ValueError(f"{split} maps one mask id to multiple metadata records")
    for frame in (expected_mapping, observed):
        frame["mask_id"] = frame["mask_id"].astype(str)
        for column in mapping_columns:
            frame[column] = frame[column].fillna("").astype(str)
    expected_mapping = expected_mapping.set_index("mask_id").sort_index()
    observed_mapping = observed.set_index("mask_id").reindex(expected_mapping.index)
    if not observed_mapping.equals(expected_mapping):
        raise ValueError(
            f"{split} mask_id-to-bits/bank/pool mapping differs from the design"
        )


def _design_run(masks: pd.DataFrame) -> LoadedIOIRun:
    ordered = masks.reset_index(drop=True).copy()
    matrix = np.stack(
        [parse_mask_bits(bits, 13) for bits in ordered["mask_bits"].astype(str)],
        axis=0,
    )
    subset = ordered.copy()
    subset.insert(0, "subset_idx", np.arange(len(subset), dtype=int))
    return LoadedIOIRun(
        prefix="phase5",
        source=Path("phase5"),
        heads=pd.DataFrame(head_records()),
        subset=subset,
        masks=matrix,
        prompt_drops=np.empty((0, len(subset)), dtype=float),
        mean_drops=np.zeros(len(subset), dtype=float),
        input_files=(),
    )


def _singleton_predictions(
    calibration: pd.DataFrame,
    train_mean: Mapping[str, float],
    candidate_matrix: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    singleton = calibration[calibration["n_heads"] == 1]
    if len(singleton) != 13:
        raise ValueError("the singleton-sum observer requires all 13 singleton masks")
    effects = np.zeros(13, dtype=float)
    coefficients: list[dict[str, object]] = []
    for row in singleton.itertuples(index=False):
        bits = parse_mask_bits(row.mask_bits, 13)
        head = int(np.flatnonzero(bits)[0])
        effects[head] = float(train_mean[str(row.mask_id)])
        coefficients.append(
            {
                "model": SINGLETON_MODEL,
                "measurement_budget": 14,
                "term": f"head_{head}",
                "coefficient": effects[head],
            }
        )
    return candidate_matrix @ effects, coefficients


def fit_phase5_observers(
    design_dir: str | Path,
    effects_dir: str | Path,
    outdir: str | Path,
    *,
    config: IOIPhase5AnalysisConfig,
) -> Path:
    """Fit only on train-prompt calibration rows and freeze predictions.

    This function never opens validation or test shard directories. Its output
    manifest is the gate required before held-out outcomes may be evaluated.
    """

    prompts, masks, design_manifest = load_locked_ioi_design(design_dir)
    effect_manifest = _validate_effect_manifest(effects_dir, design_dir)
    train_effects, train_paths = _load_split_effects(effects_dir, "train")
    _validate_split_cartesian(
        train_effects,
        prompts,
        masks,
        split="train",
    )

    calibration = masks[masks["bank"] == "calibration"].copy()
    candidates = masks[masks["bank"] == "candidate"].copy()
    calibration["measurement_order"] = pd.to_numeric(
        calibration["measurement_order"], errors="raise"
    ).astype(int)
    calibration = calibration.sort_values("measurement_order").reset_index(drop=True)
    candidates = candidates.sort_values("mask_id").reset_index(drop=True)
    if max(config.budgets) != len(calibration):
        raise ValueError("primary budget must use the complete calibration bank")

    train_mean_series = train_effects.groupby("mask_id")["drop_from_clean"].mean()
    calibration_ids = calibration["mask_id"].astype(str).tolist()
    missing = sorted(set(calibration_ids) - set(train_mean_series.index.astype(str)))
    if missing:
        raise ValueError(f"train effects lack calibration masks: {missing[:3]}")
    train_mean = {str(key): float(value) for key, value in train_mean_series.items()}

    combined = pd.concat([calibration, candidates], ignore_index=True, sort=False)
    run = _design_run(combined)
    candidate_rows = np.arange(len(calibration), len(combined), dtype=int)
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for model in config.models:
        design, columns = build_capacity_design(run, model)
        for budget in config.budgets:
            measurement_rows = np.arange(budget, dtype=int)
            y = np.asarray(
                [train_mean[mask_id] for mask_id in calibration_ids[:budget]],
                dtype=float,
            )
            coefficients = ridge_fit(design[measurement_rows], y, config.ridge)
            prediction = design[candidate_rows] @ coefficients
            singular = np.linalg.svd(design[measurement_rows], compute_uv=False)
            nonzero = singular[singular > singular.max() * 1e-10]
            rank_rows.append(
                {
                    "model": model,
                    "measurement_budget": budget,
                    "design_rank": int(np.linalg.matrix_rank(design[measurement_rows])),
                    "n_columns": int(len(columns)),
                    "condition_nonzero": float(nonzero[0] / nonzero[-1]),
                }
            )
            coefficient_rows.extend(
                {
                    "model": model,
                    "measurement_budget": budget,
                    "term": term,
                    "coefficient": float(value),
                }
                for term, value in zip(columns, coefficients)
            )
            prediction_rows.extend(
                {
                    "model": model,
                    "measurement_budget": budget,
                    "n_measurements": budget,
                    "mask_id": str(mask.mask_id),
                    "predicted_effect": float(value),
                }
                for mask, value in zip(candidates.itertuples(index=False), prediction)
            )

    singleton_prediction, singleton_coefficients = _singleton_predictions(
        calibration,
        train_mean,
        run.masks[candidate_rows],
    )
    coefficient_rows.extend(singleton_coefficients)
    prediction_rows.extend(
        {
            "model": SINGLETON_MODEL,
            "measurement_budget": 14,
            "n_measurements": 14,
            "mask_id": str(mask.mask_id),
            "predicted_effect": float(value),
        }
        for mask, value in zip(candidates.itertuples(index=False), singleton_prediction)
    )

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "candidate_predictions.csv"
    coefficient_path = output / "observer_coefficients.csv"
    rank_path = output / "fit_rank_diagnostics.csv"
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    pd.DataFrame(coefficient_rows).to_csv(coefficient_path, index=False)
    pd.DataFrame(rank_rows).to_csv(rank_path, index=False)

    prediction_sha256 = file_sha256(prediction_path)
    manifest = {
        "schema": "observerbench.ioi_phase5_fit.v2",
        "status": "fit_frozen_before_validation_test",
        "config": asdict(config),
        "config_sha256": json_sha256(asdict(config)),
        "design_manifest_sha256": file_sha256(Path(design_dir) / "design_manifest.json"),
        "design_id": design_manifest.get("design_id"),
        "design_schema": design_manifest.get("schema"),
        "effect_manifest_sha256": file_sha256(
            Path(effects_dir) / "effect_manifest.json"
        ),
        "effect_manifest_status": effect_manifest.get("status"),
        "train_effect_sources": source_hashes(train_paths, effects_dir),
        "frozen_prediction": {
            "relative_path": prediction_path.name,
            "sha256": prediction_sha256,
            "rows": len(prediction_rows),
        },
        "outputs": source_hashes(
            [prediction_path, coefficient_path, rank_path], output
        ),
        "runtime": runtime_provenance(),
    }
    write_json(output / "fit_manifest.json", manifest)
    return output


def _choose(
    candidates: pd.DataFrame,
    prediction: np.ndarray,
    actual: np.ndarray,
    target: float,
    *,
    head_cost_penalty: float,
    target_tolerance: float,
) -> dict[str, float | int | str]:
    ids = candidates["mask_id"].astype(str).to_numpy()
    counts = candidates["n_heads"].to_numpy(int)
    predicted_error = np.abs(prediction - target)
    actual_error = np.abs(actual - target)
    selected = int(np.lexsort((ids, counts, predicted_error))[0])
    cost_prediction = predicted_error + head_cost_penalty * counts
    cost_actual = actual_error + head_cost_penalty * counts
    cost_selected = int(np.lexsort((ids, counts, cost_prediction))[0])
    return {
        "selected_mask_id": str(ids[selected]),
        "selected_head_count": int(counts[selected]),
        "predicted_effect": float(prediction[selected]),
        "actual_effect": float(actual[selected]),
        "fixed_action_target_loss": float(actual_error[selected]),
        "within_tolerance": int(actual_error[selected] <= target_tolerance),
        "cost_selected_mask_id": str(ids[cost_selected]),
        "cost_selected_head_count": int(counts[cost_selected]),
        "cost_aware_fixed_action_loss": float(cost_actual[cost_selected]),
    }


def _prediction_metrics(
    predictions: pd.DataFrame,
    actual_mean: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["model", "measurement_budget"], sort=False):
        actual = np.asarray([actual_mean[str(mask)] for mask in group["mask_id"]], dtype=float)
        predicted = group["predicted_effect"].to_numpy(float)
        residual = predicted - actual
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        rows.append(
            {
                "model": keys[0],
                "measurement_budget": int(keys[1]),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "r2": float(1.0 - np.sum(residual**2) / denominator),
            }
        )
    return pd.DataFrame(rows)


def _selection_rows(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    evaluation_config: IOIPhase5EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_by_prompt = {
        prompt_id: group.set_index("mask_id")["drop_from_clean"].astype(float)
        for prompt_id, group in effects.groupby("prompt_id", sort=False)
    }
    standard_rows: list[dict[str, object]] = []
    size_rows: list[dict[str, object]] = []
    for (model, budget), model_predictions in predictions.groupby(
        ["model", "measurement_budget"], sort=False
    ):
        predicted_by_id = model_predictions.set_index("mask_id")["predicted_effect"]
        for pool_id, pool in candidates.groupby("pool_id", sort=True):
            pool = pool.sort_values("mask_id").reset_index(drop=True)
            predicted = predicted_by_id.reindex(pool["mask_id"].astype(str)).to_numpy(float)
            if not np.isfinite(predicted).all():
                raise ValueError(f"missing predictions for {model}, {pool_id}")
            for prompt_id, actual_by_id in effect_by_prompt.items():
                actual = actual_by_id.reindex(pool["mask_id"].astype(str)).to_numpy(float)
                if not np.isfinite(actual).all():
                    raise ValueError(f"missing candidate effects for {prompt_id}, {pool_id}")
                for target in evaluation_config.targets:
                    row: dict[str, object] = {
                        "model": model,
                        "measurement_budget": int(budget),
                        "prompt_id": str(prompt_id),
                        "pool_id": str(pool_id),
                        "target": float(target),
                    }
                    row.update(
                        _choose(
                            pool,
                            predicted,
                            actual,
                            float(target),
                            head_cost_penalty=evaluation_config.head_cost_penalty,
                            target_tolerance=evaluation_config.target_tolerance,
                        )
                    )
                    standard_rows.append(row)
                    for size_cell, cell in pool.groupby("size_match_cell", sort=True):
                        local = cell.index.to_numpy(int)
                        size_row: dict[str, object] = {
                            "model": model,
                            "measurement_budget": int(budget),
                            "prompt_id": str(prompt_id),
                            "pool_id": str(pool_id),
                            "target": float(target),
                            "size_match_cell": str(size_cell),
                        }
                        size_row.update(
                            _choose(
                                pool.iloc[local].reset_index(drop=True),
                                predicted[local],
                                actual[local],
                                float(target),
                                head_cost_penalty=0.0,
                                target_tolerance=evaluation_config.target_tolerance,
                            )
                        )
                        size_rows.append(size_row)
    return pd.DataFrame(standard_rows), pd.DataFrame(size_rows)


def _best_fixed_action_oracle(
    candidates: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    targets: Sequence[float],
) -> pd.DataFrame:
    """Return a point-only lower bound that must use one mask for all prompts."""

    effect_by_prompt = effects.pivot(
        index="prompt_id", columns="mask_id", values="drop_from_clean"
    )
    rows: list[dict[str, object]] = []
    for pool_id, pool in candidates.groupby("pool_id", sort=True):
        pool = pool.sort_values("mask_id").reset_index(drop=True)
        ids = pool["mask_id"].astype(str).to_numpy()
        counts = pool["n_heads"].to_numpy(int)
        actual = effect_by_prompt.reindex(columns=ids).to_numpy(float)
        if not np.isfinite(actual).all():
            raise ValueError(f"best-fixed oracle lacks effects for {pool_id}")
        for target in targets:
            mean_loss = np.mean(np.abs(actual - float(target)), axis=0)
            selected = int(np.lexsort((ids, counts, mean_loss))[0])
            rows.append(
                {
                    "pool_id": str(pool_id),
                    "target": float(target),
                    "best_fixed_action_mask_id": str(ids[selected]),
                    "best_fixed_action_head_count": int(counts[selected]),
                    "best_fixed_action_mean_target_loss": float(mean_loss[selected]),
                }
            )
    return pd.DataFrame(rows)


def _exact_mean_effect_oracle(
    candidates: pd.DataFrame,
    effects: pd.DataFrame,
    best_fixed_oracle: pd.DataFrame,
    *,
    targets: Sequence[float],
) -> pd.DataFrame:
    """Audit actions chosen with exact held-out mean effects.

    This is a post-outcome diagnostic, not a deployable observer. It separates
    exact recovery of the mean effect from the absolute-loss objective used to
    judge one fixed action across heterogeneous prompts.
    """

    effect_by_prompt = effects.pivot(
        index="prompt_id", columns="mask_id", values="drop_from_clean"
    )
    rows: list[dict[str, object]] = []
    for pool_id, pool in candidates.groupby("pool_id", sort=True):
        pool = pool.sort_values("mask_id").reset_index(drop=True)
        ids = pool["mask_id"].astype(str).to_numpy()
        counts = pool["n_heads"].to_numpy(int)
        actual = effect_by_prompt.reindex(columns=ids).to_numpy(float)
        if not np.isfinite(actual).all():
            raise ValueError(f"exact-mean oracle lacks effects for {pool_id}")
        exact_mean = actual.mean(axis=0)
        for target in targets:
            mean_target_error = np.abs(exact_mean - float(target))
            selected = int(np.lexsort((ids, counts, mean_target_error))[0])
            rows.append(
                {
                    "pool_id": str(pool_id),
                    "target": float(target),
                    "exact_mean_mask_id": str(ids[selected]),
                    "exact_mean_head_count": int(counts[selected]),
                    "exact_mean_effect": float(exact_mean[selected]),
                    "exact_mean_to_target_error": float(mean_target_error[selected]),
                    "exact_mean_fixed_action_loss": float(
                        np.mean(np.abs(actual[:, selected] - float(target)))
                    ),
                }
            )

    result = pd.DataFrame(rows)
    oracle_columns = [
        "pool_id",
        "target",
        "best_fixed_action_mask_id",
        "best_fixed_action_head_count",
        "best_fixed_action_mean_target_loss",
    ]
    result = result.merge(
        best_fixed_oracle[oracle_columns],
        on=["pool_id", "target"],
        how="left",
        validate="one_to_one",
    )
    if result["best_fixed_action_mask_id"].isna().any():
        raise ValueError("best-fixed oracle does not cover every exact-mean action")
    result["same_mask_as_best_fixed"] = (
        result["exact_mean_mask_id"] == result["best_fixed_action_mask_id"]
    )
    result["fixed_action_loss_gap"] = (
        result["exact_mean_fixed_action_loss"]
        - result["best_fixed_action_mean_target_loss"]
    )
    return result.sort_values(["pool_id", "target"]).reset_index(drop=True)


def _summary(
    selection: pd.DataFrame,
    size_selection: pd.DataFrame,
    best_fixed_oracle: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "fixed_action_target_loss",
        "within_tolerance",
        "selected_head_count",
        "cost_aware_fixed_action_loss",
    ]
    summary = selection.groupby(["model", "measurement_budget"], as_index=False)[metrics].mean()
    size = (
        size_selection.groupby(["model", "measurement_budget"], as_index=False)[
            "fixed_action_target_loss"
        ]
        .mean()
        .rename(
            columns={
                "fixed_action_target_loss": "size_matched_fixed_action_target_loss"
            }
        )
    )
    oracle_loss = float(best_fixed_oracle["best_fixed_action_mean_target_loss"].mean())
    summary["secondary_best_fixed_action_oracle_loss"] = oracle_loss
    summary["secondary_best_fixed_action_regret"] = (
        summary["fixed_action_target_loss"] - oracle_loss
    )
    return summary.merge(size, on=["model", "measurement_budget"], how="left")


def _two_way_draws(
    values: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> np.ndarray:
    matrix = values.pivot(index="prompt_id", columns="pool_id", values="value").to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("paired prompt-pool contrast is incomplete")
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        prompt_draw = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        pool_draw = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = float(matrix[np.ix_(prompt_draw, pool_draw)].mean())
    return draws


def _two_way_cluster_draws(
    values: pd.DataFrame,
    prompt_clusters: pd.DataFrame,
    *,
    cluster_column: str,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Resample name-pair clusters and pools, retaining every row per cluster."""

    if prompt_clusters["prompt_id"].astype(str).duplicated().any():
        raise ValueError("prompt cluster metadata must map each prompt exactly once")
    merged = values.merge(
        prompt_clusters[["prompt_id", cluster_column]],
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )
    if merged[cluster_column].isna().any():
        raise ValueError(f"missing {cluster_column} for a held-out prompt")
    matrix = merged.pivot(index="prompt_id", columns="pool_id", values="value")
    if not np.isfinite(matrix.to_numpy(float)).all():
        raise ValueError("clustered prompt-pool contrast is incomplete")
    cluster_by_prompt = (
        merged[["prompt_id", cluster_column]]
        .drop_duplicates()
        .set_index("prompt_id")[cluster_column]
        .reindex(matrix.index)
    )
    cluster_ids = sorted(cluster_by_prompt.astype(str).unique())
    row_indices = {
        cluster: np.flatnonzero(cluster_by_prompt.astype(str).to_numpy() == cluster)
        for cluster in cluster_ids
    }
    rng = np.random.default_rng(seed)
    array = matrix.to_numpy(float)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled_clusters = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        sampled_rows = np.concatenate(
            [row_indices[cluster_ids[cluster]] for cluster in sampled_clusters]
        )
        sampled_pools = rng.integers(0, array.shape[1], size=array.shape[1])
        draws[index] = float(array[np.ix_(sampled_rows, sampled_pools)].mean())
    return draws


def _contrast_table(
    selection: pd.DataFrame,
    size_selection: pd.DataFrame,
    prompt_clusters: pd.DataFrame,
    *,
    primary_budget: int,
    evaluation_config: IOIPhase5EvaluationConfig,
) -> pd.DataFrame:
    candidates = ("count_plus_PE_bin4", "count_plus_all_bin4")
    references = ("additive_head", "count_additive")
    records: list[dict[str, object]] = []

    def paired(frame: pd.DataFrame, metric: str, reference: str, candidate: str) -> pd.DataFrame:
        keys = ["prompt_id", "pool_id", "target"]
        if "size_match_cell" in frame.columns:
            keys.append("size_match_cell")
        subset = frame[frame["measurement_budget"] == primary_budget]
        pivot = subset.pivot_table(index=keys, columns="model", values=metric, aggfunc="mean")
        difference = pivot[reference] - pivot[candidate]
        return difference.groupby(level=["prompt_id", "pool_id"]).mean().rename("value").reset_index()

    for reference in references:
        for candidate in candidates:
            metrics = {
                "fixed_action_target_loss_reduction": paired(
                    selection,
                    "fixed_action_target_loss",
                    reference,
                    candidate,
                ),
                "within_tolerance_improvement": paired(
                    selection, "within_tolerance", candidate, reference
                ),
                "cost_aware_fixed_action_loss_reduction": paired(
                    selection,
                    "cost_aware_fixed_action_loss",
                    reference,
                    candidate,
                ),
                "size_matched_fixed_action_loss_reduction": paired(
                    size_selection,
                    "fixed_action_target_loss",
                    reference,
                    candidate,
                ),
            }
            row: dict[str, object] = {
                "reference": reference,
                "candidate": candidate,
                "measurement_budget": primary_budget,
            }
            for offset, (name, values) in enumerate(metrics.items()):
                draws = _two_way_draws(
                    values,
                    repeats=evaluation_config.bootstrap_repeats,
                    seed=evaluation_config.seed + 1000 * offset + 17 * len(records),
                )
                row[f"{name}_mean"] = float(values["value"].mean())
                row[f"{name}_q025"] = float(np.quantile(draws, 0.025))
                row[f"{name}_q975"] = float(np.quantile(draws, 0.975))
                if name == "fixed_action_target_loss_reduction":
                    for cluster_offset, cluster_column in enumerate(
                        ("ordered_name_pair_id", "unordered_name_pair_id")
                    ):
                        cluster_draws = _two_way_cluster_draws(
                            values,
                            prompt_clusters,
                            cluster_column=cluster_column,
                            repeats=evaluation_config.bootstrap_repeats,
                            seed=(
                                evaluation_config.seed
                                + 10000 * (cluster_offset + 1)
                                + 17 * len(records)
                            ),
                        )
                        prefix = cluster_column.removesuffix("_id")
                        row[f"{name}_{prefix}_q025"] = float(
                            np.quantile(cluster_draws, 0.025)
                        )
                        row[f"{name}_{prefix}_q975"] = float(
                            np.quantile(cluster_draws, 0.975)
                        )
            reference_loss = float(
                selection[
                    (selection["model"] == reference)
                    & (selection["measurement_budget"] == primary_budget)
                ]["fixed_action_target_loss"].mean()
            )
            row["fixed_action_target_loss_reduction_fraction"] = float(
                row["fixed_action_target_loss_reduction_mean"]
                / max(reference_loss, 1e-12)
            )
            records.append(row)
    return pd.DataFrame(records)


def _success_audit(
    contrasts: pd.DataFrame,
    *,
    evaluation_config: IOIPhase5EvaluationConfig,
) -> dict[str, Any]:
    primary = contrasts[contrasts["candidate"] == "count_plus_all_bin4"]
    rows = []
    for row in primary.itertuples(index=False):
        gates = {
            "fixed_action_loss_reduction_fraction": (
                row.fixed_action_target_loss_reduction_fraction
                >= evaluation_config.fixed_action_loss_reduction_fraction_min
            ),
            "fixed_action_loss_reduction_interval": (
                row.fixed_action_target_loss_reduction_q025 > 0.0
                if evaluation_config.paired_interval_excludes_zero
                else True
            ),
            "within_tolerance_improvement": (
                row.within_tolerance_improvement_mean
                >= evaluation_config.within_tolerance_improvement_min
            ),
            "size_matched_direction": (
                row.size_matched_fixed_action_loss_reduction_mean > 0.0
                if evaluation_config.requires_size_matched_direction
                else True
            ),
            "cost_aware_direction": (
                row.cost_aware_fixed_action_loss_reduction_mean > 0.0
                if evaluation_config.requires_cost_aware_direction
                else True
            ),
        }
        rows.append(
            {
                "reference": row.reference,
                "gates": gates,
                "secondary_name_pair_sensitivity": {
                    "ordered_pair_ci_direction_positive": (
                        row.fixed_action_target_loss_reduction_ordered_name_pair_q025
                        > 0.0
                    ),
                    "unordered_pair_ci_direction_positive": (
                        row.fixed_action_target_loss_reduction_unordered_name_pair_q025
                        > 0.0
                    ),
                },
                "all_gates_pass": bool(all(gates.values())),
            }
        )
    return {
        "schema": "observerbench.ioi_phase5_success_audit.v2",
        "primary_estimand": "mean_per_prompt_fixed_action_target_loss",
        "thresholds": {
            "fixed_action_loss_reduction_fraction_min": (
                evaluation_config.fixed_action_loss_reduction_fraction_min
            ),
            "paired_interval_excludes_zero": (
                evaluation_config.paired_interval_excludes_zero
            ),
            "within_tolerance_improvement_min": (
                evaluation_config.within_tolerance_improvement_min
            ),
            "requires_size_matched_direction": (
                evaluation_config.requires_size_matched_direction
            ),
            "requires_cost_aware_direction": (
                evaluation_config.requires_cost_aware_direction
            ),
        },
        "comparisons": rows,
        "all_gates_pass": bool(rows and all(row["all_gates_pass"] for row in rows)),
    }


def _write_cards(
    output: Path,
    design_manifest: Mapping[str, Any],
    predictions: pd.DataFrame,
    *,
    evaluation_config: IOIPhase5EvaluationConfig,
) -> None:
    cards = output / "cards"
    cards.mkdir(exist_ok=True)
    task = EffectTaskCard(
        task_name="ioi-heldout-effect-selection",
        task_version="1.0",
        summary="Predict finite IOI head-subset effects and select a held-out intervention.",
        model_or_substrate="GPT-2 small IOI",
        access_regime="forward template-conditioned mean ablations",
        estimand="clean-minus-ablated IOI logit-difference response",
        intervention_family="13 published IOI heads under frozen subset masks",
        measurement_design="160 calibration masks and ten disjoint 32-mask candidate pools",
        validation_target=(
            "mean held-out per-prompt target loss of one fixed selected mask"
        ),
        train_split="frozen train names/prompts and calibration masks",
        evaluation_split="frozen test names/prompts and candidate masks",
        primary_metrics=(
            "fixed_action_target_loss",
            f"within_{evaluation_config.target_tolerance:g}",
        ),
        known_scope_limits=(
            "One model, head family, position, and ablation convention.",
            "Templates occur in every split; this is name and prompt holdout, not unseen-template generalization.",
        ),
        metadata={"design_id": design_manifest.get("design_id")},
    )
    write_json(cards / "task_card.json", task.to_dict())
    for model, group in predictions.groupby("model", sort=False):
        card = EffectObserverCard(
            observer_name=str(model),
            observer_version="1.0",
            observer_family="finite head-subset effect predictor",
            access_regime="forward measurements",
            measurement_basis=str(model),
            fit_procedure=(
                "sum of singleton effects"
                if model == SINGLETON_MODEL
                else "ridge regression on the frozen mask-feature basis"
            ),
            implementation="observerbench.tasks.ioi.phase5_analysis",
            known_failure_modes=(
                "Predictions depend on the frozen mask distribution and ablation basis.",
            ),
            metadata={
                "measurement_budgets": sorted(group["measurement_budget"].astype(int).unique().tolist())
            },
        )
        write_json(cards / f"observer_card_{model}.json", card.to_dict())


def _read_frozen_predictions(
    fit_dir: str | Path,
    fit_manifest: Mapping[str, Any],
    candidates: pd.DataFrame,
    *,
    config: IOIPhase5AnalysisConfig,
) -> pd.DataFrame:
    """Load the exact prediction artifact committed by the train-only fit."""

    frozen = fit_manifest.get("frozen_prediction")
    if not isinstance(frozen, Mapping):
        raise ValueError("fit manifest lacks the required frozen_prediction record")
    relative_path = frozen.get("relative_path")
    expected_sha256 = frozen.get("sha256")
    expected_rows = frozen.get("rows")
    if relative_path != "candidate_predictions.csv":
        raise ValueError("frozen prediction path must be candidate_predictions.csv")
    if not isinstance(expected_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise ValueError("frozen prediction record has no valid SHA-256")
    if not isinstance(expected_rows, int) or expected_rows <= 0:
        raise ValueError("frozen prediction record has no valid row count")

    prediction_path = Path(fit_dir) / relative_path
    if not prediction_path.is_file():
        raise FileNotFoundError("the frozen candidate prediction artifact is missing")
    actual_sha256 = file_sha256(prediction_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("candidate prediction hash changed after fit freeze")

    predictions = pd.read_csv(prediction_path, dtype={"mask_id": str, "model": str})
    required = {
        "model",
        "measurement_budget",
        "n_measurements",
        "mask_id",
        "predicted_effect",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"frozen predictions are missing columns: {missing}")
    if len(predictions) != expected_rows:
        raise ValueError("frozen prediction row count changed after fit freeze")
    if predictions.duplicated(["model", "measurement_budget", "mask_id"]).any():
        raise ValueError("frozen predictions contain duplicate model-budget-mask rows")
    predictions["measurement_budget"] = pd.to_numeric(
        predictions["measurement_budget"], errors="raise"
    ).astype(int)
    predictions["n_measurements"] = pd.to_numeric(
        predictions["n_measurements"], errors="raise"
    ).astype(int)
    predictions["predicted_effect"] = pd.to_numeric(
        predictions["predicted_effect"], errors="raise"
    ).astype(float)
    if not np.isfinite(predictions["predicted_effect"]).all():
        raise ValueError("frozen predictions contain non-finite values")
    if not (
        predictions["measurement_budget"] == predictions["n_measurements"]
    ).all():
        raise ValueError("measurement_budget and n_measurements disagree")

    expected_groups = {
        (model, budget)
        for model in config.models
        for budget in config.budgets
    } | {(SINGLETON_MODEL, 14)}
    actual_groups = set(
        predictions[["model", "measurement_budget"]].itertuples(
            index=False, name=None
        )
    )
    if actual_groups != expected_groups:
        raise ValueError("frozen predictions do not match the configured model budgets")
    candidate_ids = set(candidates["mask_id"].astype(str))
    for keys, group in predictions.groupby(["model", "measurement_budget"]):
        if set(group["mask_id"].astype(str)) != candidate_ids:
            raise ValueError(f"frozen predictions do not cover every candidate for {keys}")
    return predictions


def _verify_frozen_train_sources(
    fit_manifest: Mapping[str, Any],
    effect_manifest: Mapping[str, Any],
) -> None:
    """Tie every train shard used for fitting to the sealed effect manifest."""

    fit_sources = fit_manifest.get("train_effect_sources")
    artifacts = effect_manifest.get("artifacts")
    if not isinstance(fit_sources, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("fit or effect manifest lacks source hashes")
    sealed_train = {
        str(label): str(digest)
        for label, digest in artifacts.items()
        if str(label).startswith("shards/train/")
    }
    recorded_train = {
        str(label): str(digest) for label, digest in fit_sources.items()
    }
    if not sealed_train or recorded_train != sealed_train:
        raise ValueError("fit train sources do not match the sealed train shard hashes")


def evaluate_phase5_observers(
    design_dir: str | Path,
    effects_dir: str | Path,
    fit_dir: str | Path,
    outdir: str | Path,
    *,
    config: IOIPhase5AnalysisConfig,
    evaluation_config: IOIPhase5EvaluationConfig,
    evaluation_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Open held-out effects only after the train-only fit artifact is frozen."""

    fit_manifest_path = Path(fit_dir) / "fit_manifest.json"
    if not fit_manifest_path.exists():
        raise FileNotFoundError("fit_manifest.json must exist before held-out evaluation")
    fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
    if fit_manifest.get("schema") != "observerbench.ioi_phase5_fit.v2":
        raise ValueError("held-out evaluation requires the Phase-5 fit v2 manifest")
    if fit_manifest.get("status") != "fit_frozen_before_validation_test":
        raise ValueError("held-out evaluation requires a frozen train-only fit artifact")
    expected_config_sha256 = fit_manifest.get("config_sha256")
    if expected_config_sha256 != json_sha256(fit_manifest.get("config")):
        raise ValueError("fit manifest config record does not match its frozen hash")
    if expected_config_sha256 != json_sha256(asdict(config)):
        raise ValueError("evaluation config differs from the frozen fit config")
    if evaluation_config.targets != config.targets:
        raise ValueError("evaluation amendment changed the frozen target set")
    if evaluation_config.primary_budget != max(config.budgets):
        raise ValueError("evaluation primary budget must remain the frozen maximum")
    if evaluation_config.target_tolerance != config.target_tolerance:
        raise ValueError("evaluation amendment changed the frozen target tolerance")
    if evaluation_config.head_cost_penalty != config.head_cost_penalty:
        raise ValueError("evaluation amendment changed the frozen head-cost penalty")

    prompts, masks, design_manifest = load_locked_ioi_design(design_dir)
    if fit_manifest.get("design_manifest_sha256") != file_sha256(
        Path(design_dir) / "design_manifest.json"
    ):
        raise ValueError("design manifest changed after fit freeze")
    if fit_manifest.get("design_id") != design_manifest.get("design_id"):
        raise ValueError("design id differs from the frozen fit design")
    effect_manifest = _validate_effect_manifest(effects_dir, design_dir)
    _verify_frozen_train_sources(fit_manifest, effect_manifest)
    candidates = masks[masks["bank"] == "candidate"].copy()
    predictions = _read_frozen_predictions(
        fit_dir,
        fit_manifest,
        candidates,
        config=config,
    )
    validation, validation_paths = _load_split_effects(effects_dir, "validation")
    test, test_paths = _load_split_effects(effects_dir, "test")
    for split, frame in (("validation", validation), ("test", test)):
        _validate_split_cartesian(
            frame,
            prompts,
            masks,
            split=split,
        )

    candidate_ids = set(candidates["mask_id"].astype(str))
    validation = validation[validation["mask_id"].astype(str).isin(candidate_ids)].copy()
    test = test[test["mask_id"].astype(str).isin(candidate_ids)].copy()
    validation_mean = validation.groupby("mask_id")["drop_from_clean"].mean().to_dict()
    test_mean = test.groupby("mask_id")["drop_from_clean"].mean().to_dict()
    prediction_metrics = pd.concat(
        [
            _prediction_metrics(predictions, validation_mean).assign(split="validation"),
            _prediction_metrics(predictions, test_mean).assign(split="test"),
        ],
        ignore_index=True,
    )
    selection, size_selection = _selection_rows(
        predictions,
        candidates,
        test,
        evaluation_config=evaluation_config,
    )
    best_fixed_oracle = _best_fixed_action_oracle(
        candidates,
        test,
        targets=evaluation_config.targets,
    )
    exact_mean_oracle = _exact_mean_effect_oracle(
        candidates,
        test,
        best_fixed_oracle,
        targets=evaluation_config.targets,
    )
    summary = _summary(selection, size_selection, best_fixed_oracle)
    test_prompt_clusters = prompts.loc[
        prompts["split"] == "test",
        ["prompt_id", "io_name", "s_name"],
    ].copy()
    test_prompt_clusters["ordered_name_pair_id"] = (
        test_prompt_clusters["io_name"].astype(str)
        + "::"
        + test_prompt_clusters["s_name"].astype(str)
    )
    test_prompt_clusters["unordered_name_pair_id"] = test_prompt_clusters.apply(
        lambda row: "::".join(sorted((str(row.io_name), str(row.s_name)))),
        axis=1,
    )
    contrasts = _contrast_table(
        selection,
        size_selection,
        test_prompt_clusters,
        primary_budget=evaluation_config.primary_budget,
        evaluation_config=evaluation_config,
    )
    audit = _success_audit(contrasts, evaluation_config=evaluation_config)

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_metrics.to_csv(output / "prediction_metrics.csv", index=False)
    selection.to_csv(output / "selection_rows.csv", index=False)
    size_selection.to_csv(output / "size_matched_selection_rows.csv", index=False)
    best_fixed_oracle.to_csv(output / "best_fixed_action_oracle.csv", index=False)
    exact_mean_oracle.to_csv(output / "exact_mean_effect_oracle.csv", index=False)
    summary.to_csv(output / "selection_summary.csv", index=False)
    contrasts.to_csv(output / "selection_contrasts.csv", index=False)
    write_json(output / "success_audit.json", audit)
    _write_cards(
        output,
        design_manifest,
        predictions,
        evaluation_config=evaluation_config,
    )
    manifest = {
        "schema": "observerbench.ioi_phase5_evaluation.v2",
        "status": "complete_confirmatory_evaluation",
        "fit_config": asdict(config),
        "evaluation_config": asdict(evaluation_config),
        "evaluation_protocol_sha256": evaluation_protocol_sha256,
        "fit_manifest_sha256": file_sha256(fit_manifest_path),
        "effect_manifest_sha256": file_sha256(
            Path(effects_dir) / "effect_manifest.json"
        ),
        "heldout_sources": source_hashes(
            [*validation_paths, *test_paths], effects_dir
        ),
        "outputs": source_hashes(
            [
                output / "prediction_metrics.csv",
                output / "selection_rows.csv",
                output / "size_matched_selection_rows.csv",
                output / "best_fixed_action_oracle.csv",
                output / "exact_mean_effect_oracle.csv",
                output / "selection_summary.csv",
                output / "selection_contrasts.csv",
                output / "success_audit.json",
            ],
            output,
        ),
        "runtime": runtime_provenance(),
    }
    write_json(output / "evaluation_manifest.json", manifest)
    return audit
