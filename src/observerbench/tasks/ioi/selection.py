"""Held-out intervention selection for finite IOI effect observers.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import runtime_provenance, source_hashes
from observerbench.tasks.ioi.phase2_capacity import (
    CAPACITY_MODELS,
    LoadedIOIRun,
    build_capacity_design,
    load_head_subset_run,
)
from observerbench.tasks.ioi.stage2d import kfold_indices


DEFAULT_MODELS: tuple[str, ...] = (
    "additive_head",
    "count_additive",
    "count_plus_PE_bin4",
    "count_plus_all_bin4",
)


@dataclass(frozen=True)
class IOISelectionConfig:
    """Frozen choices for a held-out finite-effect selection evaluation."""

    repetitions: int = 30
    mask_folds: int = 5
    prompt_fit_fraction: float = 0.5
    targets: tuple[float, ...] = tuple(np.arange(0.25, 2.01, 0.25))
    models: tuple[str, ...] = DEFAULT_MODELS
    measurement_budgets: tuple[int, ...] = ()
    ridge: float = 1e-6
    target_tolerance: float = 0.25
    head_cost_penalty: float = 0.02
    seed: int = 19000

    def __post_init__(self) -> None:
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.mask_folds < 2:
            raise ValueError("mask_folds must be at least two")
        if not 0.0 < self.prompt_fit_fraction < 1.0:
            raise ValueError("prompt_fit_fraction must lie strictly between zero and one")
        if not self.targets or not np.isfinite(self.targets).all():
            raise ValueError("targets must be a non-empty finite sequence")
        if not set(self.models).issubset(CAPACITY_MODELS):
            unknown = sorted(set(self.models) - set(CAPACITY_MODELS))
            raise ValueError(f"unknown capacity models: {unknown}")
        if any(budget <= 0 for budget in self.measurement_budgets):
            raise ValueError("measurement budgets must be positive")
        if self.ridge < 0.0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")
        if self.target_tolerance <= 0.0 or not np.isfinite(self.target_tolerance):
            raise ValueError("target_tolerance must be positive and finite")
        if self.head_cost_penalty < 0.0 or not np.isfinite(self.head_cost_penalty):
            raise ValueError("head_cost_penalty must be finite and non-negative")


def _ridge_coefficients(
    features: np.ndarray,
    effects: np.ndarray,
    indices: np.ndarray,
    ridge: float,
) -> np.ndarray:
    design = np.asarray(features[indices], dtype=float)
    response = np.asarray(effects[indices], dtype=float)
    regularizer = ridge * np.eye(design.shape[1], dtype=float)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ response,
    )


def _nested_measurement_indices(
    available: np.ndarray,
    *,
    budget: int | None,
    seed: int,
) -> np.ndarray:
    if budget is None or budget >= len(available):
        return np.sort(available)
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(available)[:budget])


def _selection_row(
    *,
    prediction: np.ndarray,
    actual: np.ndarray,
    candidate_indices: np.ndarray,
    head_counts: np.ndarray,
    target: float,
    target_tolerance: float,
    head_cost_penalty: float,
) -> dict[str, float | int]:
    predicted_loss = np.abs(prediction - target)
    selected_local = int(np.lexsort((candidate_indices, head_counts, predicted_loss))[0])
    actual_loss = np.abs(actual - target)
    oracle_local = int(np.lexsort((candidate_indices, head_counts, actual_loss))[0])

    selected_count = int(head_counts[selected_local])
    matched = np.flatnonzero(head_counts == selected_count)
    matched_oracle_local = int(matched[np.argmin(actual_loss[matched])])

    cost_predicted_loss = predicted_loss + head_cost_penalty * head_counts
    cost_selected_local = int(
        np.lexsort((candidate_indices, head_counts, cost_predicted_loss))[0]
    )
    cost_actual_loss = actual_loss + head_cost_penalty * head_counts
    cost_oracle_local = int(np.argmin(cost_actual_loss))

    absolute_error = float(actual_loss[selected_local])
    oracle_error = float(actual_loss[oracle_local])
    matched_oracle_error = float(actual_loss[matched_oracle_local])
    return {
        "target": float(target),
        "selected_subset_idx": int(candidate_indices[selected_local]),
        "selected_head_count": selected_count,
        "predicted_effect": float(prediction[selected_local]),
        "actual_effect": float(actual[selected_local]),
        "absolute_target_error": absolute_error,
        "oracle_target_error": oracle_error,
        "oracle_regret": absolute_error - oracle_error,
        "same_size_oracle_target_error": matched_oracle_error,
        "same_size_oracle_regret": absolute_error - matched_oracle_error,
        "within_tolerance": int(absolute_error <= target_tolerance),
        "cost_selected_subset_idx": int(candidate_indices[cost_selected_local]),
        "cost_selected_head_count": int(head_counts[cost_selected_local]),
        "cost_aware_loss": float(cost_actual_loss[cost_selected_local]),
        "cost_aware_oracle_loss": float(cost_actual_loss[cost_oracle_local]),
        "cost_aware_regret": float(
            cost_actual_loss[cost_selected_local] - cost_actual_loss[cost_oracle_local]
        ),
    }


def evaluate_selection(
    run: LoadedIOIRun,
    *,
    config: IOISelectionConfig,
) -> pd.DataFrame:
    """Fit on held-out prompts/masks and score selected candidate interventions.

    The fit response averages only the fit prompts. The selected intervention is
    evaluated only on the disjoint evaluation prompts. Candidate masks are held
    out from fitting in every fold. All observers share the same prompt splits,
    mask folds, measurement masks, targets, and candidate actuators.
    """

    n_prompts, n_masks = run.prompt_drops.shape
    if n_prompts < 2:
        raise ValueError("selection evaluation requires at least two prompts")
    if n_masks != len(run.subset):
        raise ValueError("prompt effects and subset table have inconsistent mask counts")
    nonclean = np.flatnonzero(run.masks.any(axis=1))
    if len(nonclean) < config.mask_folds:
        raise ValueError("not enough non-clean masks for the requested folds")

    designs = {
        model: build_capacity_design(run, model)[0]
        for model in config.models
    }
    budgets: tuple[int | None, ...] = (
        tuple(config.measurement_budgets) if config.measurement_budgets else (None,)
    )
    rows: list[dict[str, object]] = []
    n_fit_prompts = int(round(config.prompt_fit_fraction * n_prompts))

    for repetition in range(config.repetitions):
        repetition_seed = config.seed + repetition
        rng = np.random.default_rng(repetition_seed)
        prompt_order = rng.permutation(n_prompts)
        fit_prompts = np.sort(prompt_order[:n_fit_prompts])
        eval_prompts = np.sort(prompt_order[n_fit_prompts:])
        fit_effect = run.prompt_drops[fit_prompts].mean(axis=0)
        eval_effect = run.prompt_drops[eval_prompts].mean(axis=0)

        shuffled_nonclean = nonclean[
            np.random.default_rng(repetition_seed + 100_000).permutation(len(nonclean))
        ]
        folds = kfold_indices(
            len(shuffled_nonclean),
            config.mask_folds,
            seed=repetition_seed + 200_000,
        )
        for fold, (train_local, candidate_local) in enumerate(folds):
            available = np.sort(shuffled_nonclean[train_local])
            candidates = np.sort(shuffled_nonclean[candidate_local])
            candidate_counts = run.masks[candidates].sum(axis=1).astype(int)
            for budget in budgets:
                measurements = _nested_measurement_indices(
                    available,
                    budget=budget,
                    seed=repetition_seed + 10_000 * (fold + 1),
                )
                budget_label = "all" if budget is None else str(budget)
                for model, features in designs.items():
                    coefficients = _ridge_coefficients(
                        features,
                        fit_effect,
                        measurements,
                        config.ridge,
                    )
                    prediction = features[candidates] @ coefficients
                    actual = eval_effect[candidates]
                    for target in config.targets:
                        row: dict[str, object] = {
                            "repetition": repetition,
                            "fold": fold,
                            "model": model,
                            "measurement_budget": budget_label,
                            "n_measurements": int(len(measurements)),
                            "n_fit_prompts": int(len(fit_prompts)),
                            "n_eval_prompts": int(len(eval_prompts)),
                            "n_candidates": int(len(candidates)),
                        }
                        row.update(
                            _selection_row(
                                prediction=prediction,
                                actual=actual,
                                candidate_indices=candidates,
                                head_counts=candidate_counts,
                                target=float(target),
                                target_tolerance=config.target_tolerance,
                                head_cost_penalty=config.head_cost_penalty,
                            )
                        )
                        rows.append(row)
    return pd.DataFrame(rows)


def summarize_selection(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selection outcomes without treating repeated splits as IID."""

    metrics = [
        "absolute_target_error",
        "oracle_regret",
        "same_size_oracle_regret",
        "within_tolerance",
        "selected_head_count",
        "cost_aware_loss",
        "cost_aware_regret",
    ]
    per_repetition = (
        rows.groupby(["measurement_budget", "model", "repetition"], as_index=False)[metrics]
        .mean()
    )
    summary = (
        per_repetition.groupby(["measurement_budget", "model"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def paired_selection_contrasts(
    rows: pd.DataFrame,
    *,
    reference_models: Sequence[str] = ("additive_head", "count_additive"),
    candidate_models: Sequence[str] = ("count_plus_PE_bin4", "count_plus_all_bin4"),
) -> pd.DataFrame:
    """Return repetition-level paired improvements for primary outcomes."""

    keys = ["measurement_budget", "repetition", "fold", "target"]
    metrics = ["oracle_regret", "absolute_target_error", "within_tolerance"]
    averaged = rows.groupby([*keys, "model"], as_index=False)[metrics].mean()
    records: list[dict[str, object]] = []
    for budget, budget_rows in averaged.groupby("measurement_budget", sort=False):
        for reference in reference_models:
            for candidate in candidate_models:
                left = budget_rows[budget_rows["model"] == reference]
                right = budget_rows[budget_rows["model"] == candidate]
                merged = left.merge(right, on=keys, suffixes=("_reference", "_candidate"))
                for repetition, group in merged.groupby("repetition"):
                    records.append(
                        {
                            "measurement_budget": budget,
                            "reference": reference,
                            "candidate": candidate,
                            "repetition": int(repetition),
                            "oracle_regret_reduction": float(
                                (group["oracle_regret_reference"] - group["oracle_regret_candidate"]).mean()
                            ),
                            "absolute_error_reduction": float(
                                (
                                    group["absolute_target_error_reference"]
                                    - group["absolute_target_error_candidate"]
                                ).mean()
                            ),
                            "within_tolerance_improvement": float(
                                (
                                    group["within_tolerance_candidate"]
                                    - group["within_tolerance_reference"]
                                ).mean()
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _contrast_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    metrics = [
        "oracle_regret_reduction",
        "absolute_error_reduction",
        "within_tolerance_improvement",
    ]
    for keys, group in contrasts.groupby(
        ["measurement_budget", "reference", "candidate"], sort=False
    ):
        row: dict[str, object] = dict(
            zip(("measurement_budget", "reference", "candidate"), keys)
        )
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
            row[f"{metric}_positive_fraction"] = float(np.mean(values > 0.0))
        records.append(row)
    return pd.DataFrame(records)


def run_selection_analysis(
    input_run: str | Path,
    outdir: str | Path,
    *,
    label: str,
    config: IOISelectionConfig,
) -> Path:
    """Run and persist one exploratory selection analysis."""

    run = load_head_subset_run(input_run)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = evaluate_selection(run, config=config)
    summary = summarize_selection(rows)
    contrasts = paired_selection_contrasts(rows)
    contrast_summary = _contrast_summary(contrasts)

    rows.to_csv(out / "selection_decisions.csv", index=False)
    summary.to_csv(out / "selection_summary.csv", index=False)
    contrasts.to_csv(out / "selection_paired_repetitions.csv", index=False)
    contrast_summary.to_csv(out / "selection_contrasts.csv", index=False)

    manifest = {
        "schema": "observerbench.ioi_selection_run.v1",
        "status": "exploratory_existing_outcomes",
        "label": label,
        "config": asdict(config),
        "input": {
            "source": str(run.source),
            "prefix": run.prefix,
            "n_prompts": int(run.prompt_drops.shape[0]),
            "n_masks": int(run.prompt_drops.shape[1]),
            "n_nonclean_masks": int(run.masks.any(axis=1).sum()),
            "hashes": source_hashes(run.input_files),
        },
        "outputs": source_hashes(
            [
                out / "selection_decisions.csv",
                out / "selection_summary.csv",
                out / "selection_paired_repetitions.csv",
                out / "selection_contrasts.csv",
            ]
        ),
        "runtime": runtime_provenance(),
    }
    write_json(out / "selection_manifest.json", manifest)
    return out

