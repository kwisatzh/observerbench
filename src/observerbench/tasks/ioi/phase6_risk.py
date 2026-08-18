"""Post-outcome exploratory direct-risk observers for Phase-5 IOI.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This module deliberately does not present its results as confirmatory.  The
Phase-5 test outcomes had already been opened before this analysis was
conceived.  It reuses the frozen prompts, masks, models, budgets, and targets,
but changes the fitted response from mean finite effect to expected absolute
target loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256, runtime_provenance, source_hashes
from observerbench.tasks.ioi.phase2_capacity import build_capacity_design
from observerbench.tasks.ioi.phase5_analysis import (
    IOIPhase5AnalysisConfig,
    PHASE5_MODELS,
    _design_run,
    _load_split_effects,
    _read_frozen_predictions,
    _two_way_cluster_draws,
    _two_way_draws,
    _validate_effect_manifest,
    _validate_split_cartesian,
)
from observerbench.tasks.ioi.phase5_effects import load_locked_ioi_design
from observerbench.tasks.ioi.stage2d import ridge_fit


EXPLORATORY_STATUS = "exploratory_post_outcome_not_confirmatory"
SELECTOR_DIRECT_RISK = "direct_risk"
SELECTOR_MEAN_EFFECT = "mean_effect"
POLICY_TARGET = "target_loss"
POLICY_COST = "cost_aware"
HEAD_QUADRATIC_MODEL = "head_pair_quadratic_screen"


@dataclass(frozen=True)
class IOIRiskExploratoryConfig:
    """Settings retained from Phase 5 for the post-outcome risk analysis."""

    budgets: tuple[int, ...] = (20, 40, 80, 160)
    primary_budget: int = 160
    targets: tuple[float, ...] = (0.5, 1.0, 1.5)
    models: tuple[str, ...] = PHASE5_MODELS
    ridge: float = 1e-6
    target_tolerance: float = 0.25
    head_cost_penalty: float = 0.02
    bootstrap_repeats: int = 2000
    seed: int = 26061

    def __post_init__(self) -> None:
        if not self.budgets or tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("budgets must be a non-empty increasing tuple")
        if self.primary_budget != max(self.budgets):
            raise ValueError("primary_budget must be the largest retained budget")
        if not self.targets or not np.isfinite(self.targets).all():
            raise ValueError("targets must be non-empty and finite")
        if not self.models or not set(self.models).issubset(PHASE5_MODELS):
            raise ValueError("unsupported structural risk model")
        if self.ridge < 0.0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")
        if self.target_tolerance <= 0.0 or not np.isfinite(self.target_tolerance):
            raise ValueError("target_tolerance must be finite and positive")
        if self.head_cost_penalty < 0.0 or not np.isfinite(self.head_cost_penalty):
            raise ValueError("head_cost_penalty must be finite and non-negative")
        if self.bootstrap_repeats <= 0:
            raise ValueError("bootstrap_repeats must be positive")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        bootstrap_repeats: int | None = None,
    ) -> "IOIRiskExploratoryConfig":
        if value.get("schema") != "observerbench.ioi_risk_exploratory.v1":
            raise ValueError("expected observerbench.ioi_risk_exploratory.v1")
        if value.get("status") != EXPLORATORY_STATUS:
            raise ValueError("risk analysis must be marked exploratory and post-outcome")
        return cls(
            budgets=tuple(int(item) for item in value["measurement_budgets"]),
            primary_budget=int(value["primary_budget"]),
            targets=tuple(float(item) for item in value["targets"]),
            models=tuple(str(item) for item in value["models"]),
            ridge=float(value["ridge"]),
            target_tolerance=float(value["target_tolerance"]),
            head_cost_penalty=float(value["head_cost_penalty"]),
            bootstrap_repeats=(
                int(value["bootstrap_repeats"])
                if bootstrap_repeats is None
                else int(bootstrap_repeats)
            ),
            seed=int(value["seed"]),
        )


def _validate_retained_design(
    exploratory: IOIRiskExploratoryConfig,
    phase5: IOIPhase5AnalysisConfig,
) -> None:
    """Reject silent changes to the frozen Phase-5 decision problem."""

    comparisons = {
        "measurement budgets": (exploratory.budgets, phase5.budgets),
        "targets": (exploratory.targets, phase5.targets),
        "models": (exploratory.models, phase5.models),
        "ridge": (exploratory.ridge, phase5.ridge),
        "target tolerance": (
            exploratory.target_tolerance,
            phase5.target_tolerance,
        ),
        "head-cost penalty": (
            exploratory.head_cost_penalty,
            phase5.head_cost_penalty,
        ),
    }
    changed = [name for name, (left, right) in comparisons.items() if left != right]
    if changed:
        raise ValueError(f"exploratory risk analysis changed Phase-5 {', '.join(changed)}")


def fit_direct_risk_predictions(
    masks: pd.DataFrame,
    train_effects: pd.DataFrame,
    *,
    config: IOIRiskExploratoryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit target-specific expected absolute loss on calibration masks.

    The response for mask ``m`` and target ``t`` is
    ``mean_train_prompt |effect(prompt, m) - t|``.  The feature bases, ridge
    value, nested measurement order, and candidate masks are unchanged from
    Phase 5.
    """

    calibration = masks[masks["bank"] == "calibration"].copy()
    candidates = masks[masks["bank"] == "candidate"].copy()
    calibration["measurement_order"] = pd.to_numeric(
        calibration["measurement_order"], errors="raise"
    ).astype(int)
    calibration = calibration.sort_values("measurement_order").reset_index(drop=True)
    candidates = candidates.sort_values("mask_id").reset_index(drop=True)
    if len(calibration) != max(config.budgets):
        raise ValueError("the complete retained calibration bank is required")

    combined = pd.concat([calibration, candidates], ignore_index=True, sort=False)
    run = _design_run(combined)
    candidate_rows = np.arange(len(calibration), len(combined), dtype=int)
    train = train_effects.copy()
    train["mask_id"] = train["mask_id"].astype(str)
    calibration_ids = calibration["mask_id"].astype(str).tolist()

    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for target in config.targets:
        target_risk = (
            train.assign(target_loss=np.abs(train["drop_from_clean"] - float(target)))
            .groupby("mask_id")["target_loss"]
            .mean()
        )
        missing = sorted(set(calibration_ids) - set(target_risk.index.astype(str)))
        if missing:
            raise ValueError(f"train effects lack calibration masks: {missing[:3]}")
        for model in config.models:
            design, columns = build_capacity_design(run, model)
            for budget in config.budgets:
                measurement_rows = np.arange(budget, dtype=int)
                response = np.asarray(
                    [float(target_risk[mask_id]) for mask_id in calibration_ids[:budget]],
                    dtype=float,
                )
                coefficients = ridge_fit(
                    design[measurement_rows], response, config.ridge
                )
                fitted = design[measurement_rows] @ coefficients
                prediction = design[candidate_rows] @ coefficients
                singular = np.linalg.svd(design[measurement_rows], compute_uv=False)
                nonzero = singular[singular > singular.max() * 1e-10]
                diagnostic_rows.append(
                    {
                        "analysis_status": EXPLORATORY_STATUS,
                        "target": float(target),
                        "model": model,
                        "measurement_budget": int(budget),
                        "design_rank": int(
                            np.linalg.matrix_rank(design[measurement_rows])
                        ),
                        "n_columns": int(len(columns)),
                        "condition_nonzero": float(nonzero[0] / nonzero[-1]),
                        "calibration_fit_mae": float(
                            np.mean(np.abs(fitted - response))
                        ),
                        "candidate_negative_prediction_fraction": float(
                            np.mean(prediction < 0.0)
                        ),
                    }
                )
                coefficient_rows.extend(
                    {
                        "analysis_status": EXPLORATORY_STATUS,
                        "target": float(target),
                        "model": model,
                        "measurement_budget": int(budget),
                        "term": term,
                        "coefficient": float(value),
                    }
                    for term, value in zip(columns, coefficients)
                )
                prediction_rows.extend(
                    {
                        "analysis_status": EXPLORATORY_STATUS,
                        "selector_family": SELECTOR_DIRECT_RISK,
                        "target": float(target),
                        "model": model,
                        "measurement_budget": int(budget),
                        "mask_id": str(mask.mask_id),
                        "predicted_target_loss": float(value),
                    }
                    for mask, value in zip(
                        candidates.itertuples(index=False), prediction
                    )
                )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(diagnostic_rows),
    )


def _head_quadratic_design(mask_matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Intercept, 13 head indicators, and all 78 distinct head products."""

    masks = np.asarray(mask_matrix, dtype=float)
    if masks.ndim != 2 or masks.shape[1] != 13:
        raise ValueError("the full head-quadratic screen requires 13-bit masks")
    pairs = tuple(combinations(range(13), 2))
    pair_block = np.column_stack([masks[:, left] * masks[:, right] for left, right in pairs])
    design = np.column_stack([np.ones(len(masks)), masks, pair_block])
    columns = [
        "intercept",
        *[f"head_{head}" for head in range(13)],
        *[f"head_{left}:head_{right}" for left, right in pairs],
    ]
    if design.shape[1] != 92:
        raise AssertionError("the full head-quadratic basis must have 92 columns")
    return design, columns


def fit_head_quadratic_risk_screen(
    masks: pd.DataFrame,
    train_effects: pd.DataFrame,
    *,
    config: IOIRiskExploratoryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the bounded post-hoc 13-head quadratic screen at budget 160.

    This uses the same ridge value without test tuning.  It is intentionally
    kept outside ``config.models`` because it was introduced after inspecting
    the first direct-risk results.
    """

    calibration = masks[masks["bank"] == "calibration"].copy()
    candidates = masks[masks["bank"] == "candidate"].copy()
    calibration["measurement_order"] = pd.to_numeric(
        calibration["measurement_order"], errors="raise"
    ).astype(int)
    calibration = calibration.sort_values("measurement_order").reset_index(drop=True)
    candidates = candidates.sort_values("mask_id").reset_index(drop=True)
    if len(calibration) != config.primary_budget:
        raise ValueError("quadratic screen requires the full 160-mask calibration bank")
    combined = pd.concat([calibration, candidates], ignore_index=True, sort=False)
    run = _design_run(combined)
    design, columns = _head_quadratic_design(run.masks)
    calibration_rows = np.arange(len(calibration), dtype=int)
    candidate_rows = np.arange(len(calibration), len(combined), dtype=int)
    calibration_ids = calibration["mask_id"].astype(str).tolist()
    train = train_effects.copy()
    train["mask_id"] = train["mask_id"].astype(str)

    predictions: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for target in config.targets:
        target_risk = (
            train.assign(target_loss=np.abs(train["drop_from_clean"] - float(target)))
            .groupby("mask_id")["target_loss"]
            .mean()
        )
        response = np.asarray(
            [float(target_risk[mask_id]) for mask_id in calibration_ids],
            dtype=float,
        )
        coefficient = ridge_fit(design[calibration_rows], response, config.ridge)
        fitted = design[calibration_rows] @ coefficient
        prediction = design[candidate_rows] @ coefficient
        singular = np.linalg.svd(design[calibration_rows], compute_uv=False)
        nonzero = singular[singular > singular.max() * 1e-10]
        diagnostics.append(
            {
                "analysis_status": EXPLORATORY_STATUS,
                "screen_status": "bounded_posthoc_screen_no_ridge_tuning",
                "target": float(target),
                "model": HEAD_QUADRATIC_MODEL,
                "measurement_budget": config.primary_budget,
                "ridge": config.ridge,
                "design_rank": int(np.linalg.matrix_rank(design[calibration_rows])),
                "n_columns": int(len(columns)),
                "condition_nonzero": float(nonzero[0] / nonzero[-1]),
                "calibration_fit_mae": float(np.mean(np.abs(fitted - response))),
                "candidate_negative_prediction_fraction": float(
                    np.mean(prediction < 0.0)
                ),
            }
        )
        coefficients.extend(
            {
                "analysis_status": EXPLORATORY_STATUS,
                "screen_status": "bounded_posthoc_screen_no_ridge_tuning",
                "target": float(target),
                "model": HEAD_QUADRATIC_MODEL,
                "measurement_budget": config.primary_budget,
                "term": term,
                "coefficient": float(value),
            }
            for term, value in zip(columns, coefficient)
        )
        predictions.extend(
            {
                "analysis_status": EXPLORATORY_STATUS,
                "screen_status": "bounded_posthoc_screen_no_ridge_tuning",
                "selector_family": SELECTOR_DIRECT_RISK,
                "target": float(target),
                "model": HEAD_QUADRATIC_MODEL,
                "measurement_budget": config.primary_budget,
                "mask_id": str(mask.mask_id),
                "predicted_target_loss": float(value),
            }
            for mask, value in zip(candidates.itertuples(index=False), prediction)
        )
    return pd.DataFrame(predictions), pd.DataFrame(coefficients), pd.DataFrame(diagnostics)


def fit_head_quadratic_mean_screen(
    masks: pd.DataFrame,
    train_effects: pd.DataFrame,
    *,
    config: IOIRiskExploratoryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the capacity-matched mean-effect control for the quadratic screen."""

    calibration = masks[masks["bank"] == "calibration"].copy()
    candidates = masks[masks["bank"] == "candidate"].copy()
    calibration["measurement_order"] = pd.to_numeric(
        calibration["measurement_order"], errors="raise"
    ).astype(int)
    calibration = calibration.sort_values("measurement_order").reset_index(drop=True)
    candidates = candidates.sort_values("mask_id").reset_index(drop=True)
    if len(calibration) != config.primary_budget:
        raise ValueError("quadratic screen requires the full 160-mask calibration bank")
    combined = pd.concat([calibration, candidates], ignore_index=True, sort=False)
    run = _design_run(combined)
    design, columns = _head_quadratic_design(run.masks)
    calibration_rows = np.arange(len(calibration), dtype=int)
    candidate_rows = np.arange(len(calibration), len(combined), dtype=int)
    calibration_ids = calibration["mask_id"].astype(str).tolist()
    train_mean = (
        train_effects.assign(mask_id=train_effects["mask_id"].astype(str))
        .groupby("mask_id")["drop_from_clean"]
        .mean()
    )
    response = np.asarray(
        [float(train_mean[mask_id]) for mask_id in calibration_ids], dtype=float
    )
    coefficient = ridge_fit(design[calibration_rows], response, config.ridge)
    fitted = design[calibration_rows] @ coefficient
    predicted_effect = design[candidate_rows] @ coefficient
    singular = np.linalg.svd(design[calibration_rows], compute_uv=False)
    nonzero = singular[singular > singular.max() * 1e-10]
    diagnostics = pd.DataFrame(
        [
            {
                "analysis_status": EXPLORATORY_STATUS,
                "screen_status": "capacity_matched_posthoc_mean_effect_control",
                "model": HEAD_QUADRATIC_MODEL,
                "measurement_budget": config.primary_budget,
                "ridge": config.ridge,
                "design_rank": int(np.linalg.matrix_rank(design[calibration_rows])),
                "n_columns": int(len(columns)),
                "condition_nonzero": float(nonzero[0] / nonzero[-1]),
                "calibration_fit_mae": float(np.mean(np.abs(fitted - response))),
            }
        ]
    )
    coefficients = pd.DataFrame(
        [
            {
                "analysis_status": EXPLORATORY_STATUS,
                "screen_status": "capacity_matched_posthoc_mean_effect_control",
                "model": HEAD_QUADRATIC_MODEL,
                "measurement_budget": config.primary_budget,
                "term": term,
                "coefficient": float(value),
            }
            for term, value in zip(columns, coefficient)
        ]
    )
    predictions: list[dict[str, object]] = []
    for target in config.targets:
        predictions.extend(
            {
                "analysis_status": EXPLORATORY_STATUS,
                "screen_status": "capacity_matched_posthoc_mean_effect_control",
                "selector_family": SELECTOR_MEAN_EFFECT,
                "target": float(target),
                "model": HEAD_QUADRATIC_MODEL,
                "measurement_budget": config.primary_budget,
                "mask_id": str(mask.mask_id),
                "predicted_target_loss": float(abs(value - float(target))),
            }
            for mask, value in zip(
                candidates.itertuples(index=False), predicted_effect
            )
        )
    return pd.DataFrame(predictions), coefficients, diagnostics


def _mean_effect_risk_predictions(
    mean_predictions: pd.DataFrame,
    *,
    targets: Sequence[float],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in targets:
        frame = mean_predictions.copy()
        frame["analysis_status"] = EXPLORATORY_STATUS
        frame["selector_family"] = SELECTOR_MEAN_EFFECT
        frame["target"] = float(target)
        frame["predicted_target_loss"] = np.abs(
            frame["predicted_effect"].to_numpy(float) - float(target)
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)[
        [
            "analysis_status",
            "selector_family",
            "target",
            "model",
            "measurement_budget",
            "mask_id",
            "predicted_target_loss",
        ]
    ]


def select_fixed_masks(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    head_cost_penalty: float,
) -> pd.DataFrame:
    """Choose one fixed mask per selector/model/budget/pool/target."""

    candidate_meta = candidates[
        ["mask_id", "pool_id", "n_heads", "size_match_cell"]
    ].copy()
    candidate_meta["mask_id"] = candidate_meta["mask_id"].astype(str)
    frame = predictions.merge(
        candidate_meta,
        on="mask_id",
        how="left",
        validate="many_to_one",
    )
    if frame[["pool_id", "n_heads"]].isna().any().any():
        raise ValueError("predictions include masks outside the candidate bank")

    rows: list[dict[str, object]] = []
    group_columns = [
        "selector_family",
        "model",
        "measurement_budget",
        "pool_id",
        "target",
    ]
    for keys, group in frame.groupby(group_columns, sort=True):
        group = group.reset_index(drop=True)
        ids = group["mask_id"].astype(str).to_numpy()
        counts = group["n_heads"].to_numpy(int)
        predicted = group["predicted_target_loss"].to_numpy(float)
        for policy, objective in (
            (POLICY_TARGET, predicted),
            (POLICY_COST, predicted + head_cost_penalty * counts),
        ):
            selected = int(np.lexsort((ids, counts, objective))[0])
            rows.append(
                {
                    "analysis_status": EXPLORATORY_STATUS,
                    "selector_family": str(keys[0]),
                    "model": str(keys[1]),
                    "measurement_budget": int(keys[2]),
                    "pool_id": str(keys[3]),
                    "target": float(keys[4]),
                    "policy": policy,
                    "selected_mask_id": str(ids[selected]),
                    "selected_head_count": int(counts[selected]),
                    "selected_size_match_cell": str(
                        group.iloc[selected]["size_match_cell"]
                    ),
                    "predicted_target_loss": float(predicted[selected]),
                    "predicted_objective": float(objective[selected]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_fixed_masks(
    decisions: pd.DataFrame,
    test_effects: pd.DataFrame,
    *,
    target_tolerance: float,
    head_cost_penalty: float,
) -> pd.DataFrame:
    """Evaluate every fixed decision on every test prompt."""

    candidate_effects = test_effects[
        ["prompt_id", "mask_id", "drop_from_clean"]
    ].copy()
    candidate_effects["mask_id"] = candidate_effects["mask_id"].astype(str)
    rows = decisions.merge(
        candidate_effects,
        left_on="selected_mask_id",
        right_on="mask_id",
        how="left",
        validate="many_to_many",
    ).drop(columns="mask_id")
    if rows["drop_from_clean"].isna().any():
        raise ValueError("selected masks are missing test-prompt effects")
    rows["actual_target_loss"] = np.abs(
        rows["drop_from_clean"].to_numpy(float) - rows["target"].to_numpy(float)
    )
    rows["within_tolerance"] = (
        rows["actual_target_loss"] <= target_tolerance
    ).astype(int)
    rows["actual_objective"] = rows["actual_target_loss"] + np.where(
        rows["policy"] == POLICY_COST,
        head_cost_penalty * rows["selected_head_count"].to_numpy(int),
        0.0,
    )
    return rows


def _risk_prediction_metrics(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    test_effects: pd.DataFrame,
) -> pd.DataFrame:
    test_risk_rows: list[pd.DataFrame] = []
    for target in sorted(predictions["target"].unique()):
        actual = (
            test_effects.assign(
                actual_target_loss=np.abs(
                    test_effects["drop_from_clean"] - float(target)
                )
            )
            .groupby("mask_id", as_index=False)["actual_target_loss"]
            .mean()
        )
        actual["target"] = float(target)
        test_risk_rows.append(actual)
    actual_risk = pd.concat(test_risk_rows, ignore_index=True)
    actual_risk["mask_id"] = actual_risk["mask_id"].astype(str)
    frame = predictions.merge(
        actual_risk,
        on=["mask_id", "target"],
        how="left",
        validate="many_to_one",
    ).merge(
        candidates[["mask_id", "pool_id"]].assign(
            mask_id=lambda value: value["mask_id"].astype(str)
        ),
        on="mask_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["selector_family", "model", "measurement_budget", "target"],
        sort=True,
    ):
        residual = (
            group["predicted_target_loss"].to_numpy(float)
            - group["actual_target_loss"].to_numpy(float)
        )
        actual = group["actual_target_loss"].to_numpy(float)
        predicted = group["predicted_target_loss"].to_numpy(float)
        rank_correlation = float(
            pd.Series(actual).rank().corr(pd.Series(predicted).rank())
        )
        rows.append(
            {
                "analysis_status": EXPLORATORY_STATUS,
                "selector_family": str(keys[0]),
                "model": str(keys[1]),
                "measurement_budget": int(keys[2]),
                "target": float(keys[3]),
                "test_candidate_mae": float(np.mean(np.abs(residual))),
                "test_candidate_rmse": float(np.sqrt(np.mean(residual**2))),
                "test_candidate_rank_correlation": rank_correlation,
            }
        )
    return pd.DataFrame(rows)


def _selector_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    return (
        outcomes.groupby(
            [
                "analysis_status",
                "selector_family",
                "model",
                "measurement_budget",
                "policy",
            ],
            as_index=False,
        )[
            [
                "actual_target_loss",
                "actual_objective",
                "within_tolerance",
                "selected_head_count",
            ]
        ]
        .mean()
    )


def _comparison_pairs(models: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Same-basis comparisons plus all-pairs risk against simpler means."""

    pairs = [(model, model) for model in models]
    all_pairs = "count_plus_all_bin4"
    pairs.extend(
        (all_pairs, reference)
        for reference in models
        if (all_pairs, reference) not in pairs
    )
    return tuple(pairs)


def _contrast_specs(
    models: Sequence[str],
) -> tuple[tuple[str, str, str, str], ...]:
    """Return decision-calibration and within-risk structural comparisons."""

    specs = [
        (SELECTOR_DIRECT_RISK, candidate, SELECTOR_MEAN_EFFECT, reference)
        for candidate, reference in _comparison_pairs(models)
    ]
    all_pairs = "count_plus_all_bin4"
    specs.extend(
        (SELECTOR_DIRECT_RISK, all_pairs, SELECTOR_DIRECT_RISK, reference)
        for reference in models
        if reference != all_pairs
    )
    return tuple(specs)


def paired_selector_contrasts(
    outcomes: pd.DataFrame,
    prompt_clusters: pd.DataFrame,
    *,
    config: IOIRiskExploratoryConfig,
    comparison_specs: Sequence[tuple[str, str, str, str]] | None = None,
) -> pd.DataFrame:
    """Paired prompt-by-pool contrasts with name-pair sensitivities.

    Positive values favor the direct-risk selector.  Inference is restricted
    to the retained 160-measurement primary budget; all budgets remain in the
    descriptive output.
    """

    primary = outcomes[outcomes["measurement_budget"] == config.primary_budget]
    records: list[dict[str, object]] = []
    record_index = 0
    specs = (
        tuple(comparison_specs)
        if comparison_specs is not None
        else _contrast_specs(config.models)
    )
    for candidate_family, risk_model, reference_family, mean_model in specs:
        for policy in (POLICY_TARGET, POLICY_COST):
            risk = primary[
                (primary["selector_family"] == candidate_family)
                & (primary["model"] == risk_model)
                & (primary["policy"] == policy)
            ]
            mean = primary[
                (primary["selector_family"] == reference_family)
                & (primary["model"] == mean_model)
                & (primary["policy"] == policy)
            ]
            keys = ["prompt_id", "pool_id", "target"]
            paired = mean.merge(
                risk,
                on=keys,
                suffixes=("_mean", "_risk"),
                validate="one_to_one",
            )
            if paired.empty:
                raise ValueError(f"empty selector comparison: {risk_model}, {mean_model}")
            scopes: list[tuple[str, pd.DataFrame]] = [("pooled", paired)]
            scopes.extend(
                (f"target_{target:g}", paired[paired["target"] == target])
                for target in config.targets
            )
            metrics = [
                (
                    "objective_loss_reduction",
                    paired["actual_objective_mean"]
                    - paired["actual_objective_risk"],
                    paired["actual_objective_mean"],
                    paired["actual_objective_risk"],
                )
            ]
            if policy == POLICY_TARGET:
                metrics.append(
                    (
                        "within_tolerance_improvement",
                        paired["within_tolerance_risk"]
                        - paired["within_tolerance_mean"],
                        paired["within_tolerance_mean"],
                        paired["within_tolerance_risk"],
                    )
                )
            for scope_name, scoped in scopes:
                scoped_index = scoped.index
                for metric_name, difference, reference_value, candidate_value in metrics:
                    values = scoped[keys].copy()
                    values["value"] = difference.loc[scoped_index].to_numpy(float)
                    values = (
                        values.groupby(["prompt_id", "pool_id"], as_index=False)["value"]
                        .mean()
                    )
                    draws = _two_way_draws(
                        values,
                        repeats=config.bootstrap_repeats,
                        seed=config.seed + 101 * record_index,
                    )
                    row: dict[str, object] = {
                        "analysis_status": EXPLORATORY_STATUS,
                        "candidate_selector_family": candidate_family,
                        "candidate_model": risk_model,
                        "reference_selector_family": reference_family,
                        "reference_model": mean_model,
                        "measurement_budget": config.primary_budget,
                        "policy": policy,
                        "target_scope": scope_name,
                        "metric": metric_name,
                        "mean": float(values["value"].mean()),
                        "q025": float(np.quantile(draws, 0.025)),
                        "q975": float(np.quantile(draws, 0.975)),
                        "reference_mean": float(
                            reference_value.loc[scoped_index].mean()
                        ),
                        "candidate_mean": float(
                            candidate_value.loc[scoped_index].mean()
                        ),
                    }
                    if metric_name == "objective_loss_reduction":
                        row["reduction_fraction"] = float(
                            row["mean"] / max(row["reference_mean"], 1e-12)
                        )
                    else:
                        row["reduction_fraction"] = np.nan
                    for cluster_offset, cluster_column in enumerate(
                        ("ordered_name_pair_id", "unordered_name_pair_id")
                    ):
                        cluster_draws = _two_way_cluster_draws(
                            values,
                            prompt_clusters,
                            cluster_column=cluster_column,
                            repeats=config.bootstrap_repeats,
                            seed=(
                                config.seed
                                + 100_000 * (cluster_offset + 1)
                                + 101 * record_index
                            ),
                        )
                        prefix = cluster_column.removesuffix("_id")
                        row[f"{prefix}_q025"] = float(
                            np.quantile(cluster_draws, 0.025)
                        )
                        row[f"{prefix}_q975"] = float(
                            np.quantile(cluster_draws, 0.975)
                        )
                    records.append(row)
                    record_index += 1
    return pd.DataFrame(records)


def pool_contrasts(
    outcomes: pd.DataFrame,
    *,
    config: IOIRiskExploratoryConfig,
    comparison_specs: Sequence[tuple[str, str, str, str]],
) -> pd.DataFrame:
    """Expose the pool-level signs behind each primary-budget average."""

    primary = outcomes[outcomes["measurement_budget"] == config.primary_budget]
    rows: list[dict[str, object]] = []
    for candidate_family, candidate_model, reference_family, reference_model in comparison_specs:
        for policy in (POLICY_TARGET, POLICY_COST):
            candidate = primary[
                (primary["selector_family"] == candidate_family)
                & (primary["model"] == candidate_model)
                & (primary["policy"] == policy)
            ]
            reference = primary[
                (primary["selector_family"] == reference_family)
                & (primary["model"] == reference_model)
                & (primary["policy"] == policy)
            ]
            paired = reference.merge(
                candidate,
                on=["prompt_id", "pool_id", "target"],
                suffixes=("_reference", "_candidate"),
                validate="one_to_one",
            )
            paired["objective_loss_reduction"] = (
                paired["actual_objective_reference"]
                - paired["actual_objective_candidate"]
            )
            for pool_id, group in paired.groupby("pool_id", sort=True):
                rows.append(
                    {
                        "analysis_status": EXPLORATORY_STATUS,
                        "candidate_selector_family": candidate_family,
                        "candidate_model": candidate_model,
                        "reference_selector_family": reference_family,
                        "reference_model": reference_model,
                        "measurement_budget": config.primary_budget,
                        "policy": policy,
                        "pool_id": str(pool_id),
                        "mean_objective_loss_reduction": float(
                            group["objective_loss_reduction"].mean()
                        ),
                        "positive_direction": int(
                            group["objective_loss_reduction"].mean() > 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def budget_descriptives(
    outcomes: pd.DataFrame,
    *,
    model: str,
) -> pd.DataFrame:
    """Descriptive direct-risk versus same-basis mean-effect budget curve."""

    direct = outcomes[
        (outcomes["selector_family"] == SELECTOR_DIRECT_RISK)
        & (outcomes["model"] == model)
    ]
    mean = outcomes[
        (outcomes["selector_family"] == SELECTOR_MEAN_EFFECT)
        & (outcomes["model"] == model)
    ]
    paired = mean.merge(
        direct,
        on=["measurement_budget", "policy", "prompt_id", "pool_id", "target"],
        suffixes=("_mean", "_risk"),
        validate="one_to_one",
    )
    paired["objective_loss_reduction"] = (
        paired["actual_objective_mean"] - paired["actual_objective_risk"]
    )
    rows: list[dict[str, object]] = []
    for (budget, policy), group in paired.groupby(
        ["measurement_budget", "policy"], sort=True
    ):
        by_pool = group.groupby("pool_id")["objective_loss_reduction"].mean()
        reference_mean = float(group["actual_objective_mean"].mean())
        reduction = float(group["objective_loss_reduction"].mean())
        rows.append(
            {
                "analysis_status": EXPLORATORY_STATUS,
                "model": model,
                "measurement_budget": int(budget),
                "policy": str(policy),
                "mean_objective_loss_reduction": reduction,
                "reduction_fraction": reduction / max(reference_mean, 1e-12),
                "positive_pool_count": int((by_pool > 0.0).sum()),
                "pool_count": int(len(by_pool)),
            }
        )
    return pd.DataFrame(rows)


def fixed_action_oracle(
    candidates: pd.DataFrame,
    test_effects: pd.DataFrame,
    *,
    targets: Sequence[float],
    head_cost_penalty: float,
) -> pd.DataFrame:
    """Point-only best fixed mask for each retained pool, target, and policy."""

    effect_matrix = test_effects.pivot(
        index="prompt_id", columns="mask_id", values="drop_from_clean"
    )
    rows: list[dict[str, object]] = []
    for pool_id, pool in candidates.groupby("pool_id", sort=True):
        pool = pool.sort_values("mask_id").reset_index(drop=True)
        ids = pool["mask_id"].astype(str).to_numpy()
        counts = pool["n_heads"].to_numpy(int)
        actual = effect_matrix.reindex(columns=ids).to_numpy(float)
        if not np.isfinite(actual).all():
            raise ValueError(f"fixed-action oracle lacks effects for {pool_id}")
        for target in targets:
            target_loss = np.abs(actual - float(target))
            for policy, cost in (
                (POLICY_TARGET, np.zeros(len(ids), dtype=float)),
                (POLICY_COST, head_cost_penalty * counts),
            ):
                mean_objective = target_loss.mean(axis=0) + cost
                selected = int(np.lexsort((ids, counts, mean_objective))[0])
                rows.append(
                    {
                        "analysis_status": EXPLORATORY_STATUS,
                        "pool_id": str(pool_id),
                        "target": float(target),
                        "policy": policy,
                        "oracle_mask_id": str(ids[selected]),
                        "oracle_head_count": int(counts[selected]),
                        "oracle_mean_objective": float(mean_objective[selected]),
                    }
                )
    return pd.DataFrame(rows)


def decision_quality(
    outcomes: pd.DataFrame,
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a point-only best-fixed regret to every fixed decision."""

    group_columns = [
        "analysis_status",
        "selector_family",
        "model",
        "measurement_budget",
        "pool_id",
        "target",
        "policy",
        "selected_mask_id",
        "selected_head_count",
    ]
    quality = outcomes.groupby(group_columns, as_index=False).agg(
        selected_mean_target_loss=("actual_target_loss", "mean"),
        selected_mean_objective=("actual_objective", "mean"),
        selected_within_tolerance=("within_tolerance", "mean"),
    )
    quality = quality.merge(
        oracle,
        on=["analysis_status", "pool_id", "target", "policy"],
        how="left",
        validate="many_to_one",
    )
    if quality["oracle_mean_objective"].isna().any():
        raise ValueError("decision quality is missing a fixed-action oracle cell")
    quality["best_fixed_action_regret"] = (
        quality["selected_mean_objective"] - quality["oracle_mean_objective"]
    )
    return quality


def _selection_change_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "measurement_budget", "pool_id", "target", "policy"]
    direct = decisions[decisions["selector_family"] == SELECTOR_DIRECT_RISK][
        [*keys, "selected_mask_id"]
    ].rename(columns={"selected_mask_id": "risk_mask_id"})
    mean = decisions[decisions["selector_family"] == SELECTOR_MEAN_EFFECT][
        [*keys, "selected_mask_id"]
    ].rename(columns={"selected_mask_id": "mean_mask_id"})
    paired = direct.merge(mean, on=keys, validate="one_to_one")
    paired["selection_changed"] = paired["risk_mask_id"] != paired["mean_mask_id"]
    return (
        paired.groupby(["model", "measurement_budget", "policy"], as_index=False)[
            "selection_changed"
        ]
        .mean()
        .assign(analysis_status=EXPLORATORY_STATUS)
    )


def _primary_result_digest(
    contrasts: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    quality: pd.DataFrame,
    selection_changes: pd.DataFrame,
    pool_results: pd.DataFrame,
    budget_results: pd.DataFrame,
    *,
    config: IOIRiskExploratoryConfig,
) -> dict[str, Any]:
    all_pairs = "count_plus_all_bin4"
    main = contrasts[
        (contrasts["candidate_model"] == all_pairs)
        & (contrasts["reference_selector_family"] == SELECTOR_MEAN_EFFECT)
        & (contrasts["reference_model"] == all_pairs)
        & (contrasts["policy"] == POLICY_TARGET)
        & (contrasts["target_scope"] == "pooled")
        & (contrasts["metric"] == "objective_loss_reduction")
    ]
    if len(main) != 1:
        raise ValueError("the primary exploratory contrast is missing or duplicated")
    row = main.iloc[0]
    per_target = contrasts[
        (contrasts["candidate_model"] == all_pairs)
        & (contrasts["reference_selector_family"] == SELECTOR_MEAN_EFFECT)
        & (contrasts["reference_model"] == all_pairs)
        & (contrasts["policy"] == POLICY_TARGET)
        & (contrasts["target_scope"] != "pooled")
        & (contrasts["metric"] == "objective_loss_reduction")
    ][["target_scope", "mean", "q025", "q975", "reduction_fraction"]]
    direct_metrics = prediction_metrics[
        (prediction_metrics["selector_family"] == SELECTOR_DIRECT_RISK)
        & (prediction_metrics["model"] == all_pairs)
        & (prediction_metrics["measurement_budget"] == config.primary_budget)
    ]
    mean_metrics = prediction_metrics[
        (prediction_metrics["selector_family"] == SELECTOR_MEAN_EFFECT)
        & (prediction_metrics["model"] == all_pairs)
        & (prediction_metrics["measurement_budget"] == config.primary_budget)
    ]
    cost = contrasts[
        (contrasts["candidate_model"] == all_pairs)
        & (contrasts["reference_selector_family"] == SELECTOR_MEAN_EFFECT)
        & (contrasts["reference_model"] == all_pairs)
        & (contrasts["policy"] == POLICY_COST)
        & (contrasts["target_scope"] == "pooled")
        & (contrasts["metric"] == "objective_loss_reduction")
    ]
    if len(cost) != 1:
        raise ValueError("the cost-aware exploratory contrast is missing or duplicated")
    cost_row = cost.iloc[0]
    quality_primary = quality[
        (quality["model"] == all_pairs)
        & (quality["measurement_budget"] == config.primary_budget)
        & (quality["policy"] == POLICY_TARGET)
    ]
    regret_by_selector = quality_primary.groupby("selector_family")[
        "best_fixed_action_regret"
    ].mean()
    mean_regret = float(regret_by_selector[SELECTOR_MEAN_EFFECT])
    risk_regret = float(regret_by_selector[SELECTOR_DIRECT_RISK])
    change = selection_changes[
        (selection_changes["model"] == all_pairs)
        & (selection_changes["measurement_budget"] == config.primary_budget)
        & (selection_changes["policy"] == POLICY_TARGET)
    ]
    if len(change) != 1:
        raise ValueError("the all-pairs selection-change diagnostic is missing")
    structural = contrasts[
        (contrasts["candidate_model"] == all_pairs)
        & (contrasts["reference_selector_family"] == SELECTOR_DIRECT_RISK)
        & (contrasts["policy"] == POLICY_TARGET)
        & (contrasts["metric"] == "objective_loss_reduction")
    ][
        [
            "reference_model",
            "target_scope",
            "mean",
            "q025",
            "q975",
            "reduction_fraction",
            "ordered_name_pair_q025",
            "unordered_name_pair_q025",
        ]
    ]
    quadratic = contrasts[
        (contrasts["candidate_model"] == HEAD_QUADRATIC_MODEL)
        & (contrasts["policy"] == POLICY_TARGET)
        & (contrasts["metric"] == "objective_loss_reduction")
        & (
            contrasts["reference_model"].isin(
                (
                    HEAD_QUADRATIC_MODEL,
                    all_pairs,
                    "count_plus_PE_bin4",
                    "additive_head",
                    "count_additive",
                )
            )
        )
    ][
        [
            "reference_selector_family",
            "reference_model",
            "target_scope",
            "mean",
            "q025",
            "q975",
            "reduction_fraction",
            "ordered_name_pair_q025",
            "unordered_name_pair_q025",
        ]
    ]
    main_pool = pool_results[
        (pool_results["candidate_model"] == all_pairs)
        & (pool_results["reference_selector_family"] == SELECTOR_MEAN_EFFECT)
        & (pool_results["reference_model"] == all_pairs)
    ]
    pool_signs = {
        policy: {
            "positive_pool_count": int(
                main_pool[main_pool["policy"] == policy]["positive_direction"].sum()
            ),
            "pool_count": int((main_pool["policy"] == policy).sum()),
        }
        for policy in (POLICY_TARGET, POLICY_COST)
    }
    return {
        "schema": "observerbench.ioi_risk_exploratory_digest.v1",
        "status": EXPLORATORY_STATUS,
        "disclosure": (
            "This analysis was conceived after the Phase-5 test outcomes were opened. "
            "No interval or threshold has confirmatory status."
        ),
        "primary_descriptive_comparison": {
            "candidate": "direct-risk all-pairs selector",
            "reference": "mean-effect all-pairs selector",
            "measurement_budget": config.primary_budget,
            "mean_test_target_loss_reduction": float(row["mean"]),
            "reduction_fraction": float(row["reduction_fraction"]),
            "prompt_pool_interval": [float(row["q025"]), float(row["q975"])],
            "ordered_name_pair_interval": [
                float(row["ordered_name_pair_q025"]),
                float(row["ordered_name_pair_q975"]),
            ],
            "unordered_name_pair_interval": [
                float(row["unordered_name_pair_q025"]),
                float(row["unordered_name_pair_q975"]),
            ],
        },
        "per_target_descriptive_comparisons": per_target.to_dict(orient="records"),
        "secondary_cost_aware_comparison": {
            "mean_test_objective_reduction": float(cost_row["mean"]),
            "reduction_fraction": float(cost_row["reduction_fraction"]),
            "prompt_pool_interval": [
                float(cost_row["q025"]),
                float(cost_row["q975"]),
            ],
            "ordered_name_pair_interval": [
                float(cost_row["ordered_name_pair_q025"]),
                float(cost_row["ordered_name_pair_q975"]),
            ],
            "unordered_name_pair_interval": [
                float(cost_row["unordered_name_pair_q025"]),
                float(cost_row["unordered_name_pair_q975"]),
            ],
        },
        "secondary_best_fixed_action_regret_point_only": {
            "mean_effect_all_pairs": mean_regret,
            "direct_risk_all_pairs": risk_regret,
            "regret_reduction_fraction": float(
                (mean_regret - risk_regret) / max(mean_regret, 1e-12)
            ),
        },
        "all_pairs_direct_risk_against_simpler_direct_risk": structural.to_dict(
            orient="records"
        ),
        "all_pairs_fixed_selection_changed_fraction": float(
            change.iloc[0]["selection_changed"]
        ),
        "all_pairs_pool_signs": pool_signs,
        "all_pairs_budget_descriptives": budget_results.to_dict(orient="records"),
        "bounded_head_quadratic_screen": {
            "status": "posthoc_screen_no_ridge_tuning",
            "comparisons": quadratic.to_dict(orient="records"),
        },
        "all_pairs_test_candidate_prediction_mae": {
            "direct_risk": float(direct_metrics["test_candidate_mae"].mean()),
            "mean_effect_implied_risk": float(
                mean_metrics["test_candidate_mae"].mean()
            ),
        },
        "interpretation_rule": (
            "Treat the sign, size, and sensitivity intervals as hypothesis-generating "
            "only; a fresh outcome-sealed replication must decide the claim."
        ),
    }


def run_risk_exploratory_analysis(
    design_dir: str | Path,
    effects_dir: str | Path,
    mean_fit_dir: str | Path,
    confirmatory_dir: str | Path,
    outdir: str | Path,
    *,
    config: IOIRiskExploratoryConfig,
    phase5_config: IOIPhase5AnalysisConfig,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run and label the complete post-outcome direct-risk exploration."""

    _validate_retained_design(config, phase5_config)
    confirmatory_manifest_path = Path(confirmatory_dir) / "evaluation_manifest.json"
    if not confirmatory_manifest_path.is_file():
        raise FileNotFoundError("completed Phase-5 evaluation manifest is required")
    confirmatory_manifest = json.loads(
        confirmatory_manifest_path.read_text(encoding="utf-8")
    )
    if confirmatory_manifest.get("status") != "complete_confirmatory_evaluation":
        raise ValueError("Phase-5 evaluation was not complete before this exploration")

    prompts, masks, design_manifest = load_locked_ioi_design(design_dir)
    effect_manifest = _validate_effect_manifest(effects_dir, design_dir)
    train, train_paths = _load_split_effects(effects_dir, "train")
    test, test_paths = _load_split_effects(effects_dir, "test")
    for split, frame in (("train", train), ("test", test)):
        _validate_split_cartesian(frame, prompts, masks, split=split)

    candidates = masks[masks["bank"] == "candidate"].copy()
    candidate_ids = set(candidates["mask_id"].astype(str))
    test_candidates = test[test["mask_id"].astype(str).isin(candidate_ids)].copy()
    risk_predictions, coefficients, fit_diagnostics = fit_direct_risk_predictions(
        masks,
        train,
        config=config,
    )
    (
        quadratic_predictions,
        quadratic_coefficients,
        quadratic_diagnostics,
    ) = fit_head_quadratic_risk_screen(masks, train, config=config)
    (
        quadratic_mean_predictions,
        quadratic_mean_coefficients,
        quadratic_mean_diagnostics,
    ) = fit_head_quadratic_mean_screen(masks, train, config=config)

    mean_fit_manifest_path = Path(mean_fit_dir) / "fit_manifest.json"
    mean_fit_manifest = json.loads(mean_fit_manifest_path.read_text(encoding="utf-8"))
    mean_predictions = _read_frozen_predictions(
        mean_fit_dir,
        mean_fit_manifest,
        candidates,
        config=phase5_config,
    )
    mean_predictions = mean_predictions[
        mean_predictions["model"].isin(config.models)
        & mean_predictions["measurement_budget"].isin(config.budgets)
    ].copy()
    implied_risk_predictions = _mean_effect_risk_predictions(
        mean_predictions,
        targets=config.targets,
    )
    all_predictions = pd.concat(
        [
            risk_predictions,
            implied_risk_predictions,
            quadratic_predictions,
            quadratic_mean_predictions,
        ],
        ignore_index=True,
    )
    decisions = select_fixed_masks(
        all_predictions,
        candidates,
        head_cost_penalty=config.head_cost_penalty,
    )
    outcomes = evaluate_fixed_masks(
        decisions,
        test_candidates,
        target_tolerance=config.target_tolerance,
        head_cost_penalty=config.head_cost_penalty,
    )
    prediction_metrics = _risk_prediction_metrics(
        all_predictions,
        candidates,
        test_candidates,
    )
    summary = _selector_summary(outcomes)
    selection_changes = _selection_change_summary(decisions)

    oracle = fixed_action_oracle(
        candidates,
        test_candidates,
        targets=config.targets,
        head_cost_penalty=config.head_cost_penalty,
    )
    quality = decision_quality(outcomes, oracle)

    prompt_clusters = prompts.loc[
        prompts["split"] == "test", ["prompt_id", "io_name", "s_name"]
    ].copy()
    prompt_clusters["ordered_name_pair_id"] = (
        prompt_clusters["io_name"].astype(str)
        + "::"
        + prompt_clusters["s_name"].astype(str)
    )
    prompt_clusters["unordered_name_pair_id"] = prompt_clusters.apply(
        lambda row: "::".join(sorted((str(row.io_name), str(row.s_name)))),
        axis=1,
    )
    comparison_specs = (
        *_contrast_specs(config.models),
        (
            SELECTOR_DIRECT_RISK,
            HEAD_QUADRATIC_MODEL,
            SELECTOR_MEAN_EFFECT,
            HEAD_QUADRATIC_MODEL,
        ),
        (
            SELECTOR_DIRECT_RISK,
            HEAD_QUADRATIC_MODEL,
            SELECTOR_MEAN_EFFECT,
            "count_plus_all_bin4",
        ),
        (
            SELECTOR_DIRECT_RISK,
            HEAD_QUADRATIC_MODEL,
            SELECTOR_DIRECT_RISK,
            "count_plus_all_bin4",
        ),
        (
            SELECTOR_DIRECT_RISK,
            HEAD_QUADRATIC_MODEL,
            SELECTOR_DIRECT_RISK,
            "count_plus_PE_bin4",
        ),
        (
            SELECTOR_DIRECT_RISK,
            HEAD_QUADRATIC_MODEL,
            SELECTOR_DIRECT_RISK,
            "additive_head",
        ),
        (
            SELECTOR_DIRECT_RISK,
            HEAD_QUADRATIC_MODEL,
            SELECTOR_DIRECT_RISK,
            "count_additive",
        ),
    )
    contrasts = paired_selector_contrasts(
        outcomes,
        prompt_clusters,
        config=config,
        comparison_specs=comparison_specs,
    )
    pool_results = pool_contrasts(
        outcomes,
        config=config,
        comparison_specs=comparison_specs,
    )
    budget_results = budget_descriptives(
        outcomes,
        model="count_plus_all_bin4",
    )
    digest = _primary_result_digest(
        contrasts,
        prediction_metrics,
        quality,
        selection_changes,
        pool_results,
        budget_results,
        config=config,
    )

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "direct_risk_predictions.csv": risk_predictions,
        "direct_risk_coefficients.csv": coefficients,
        "fit_diagnostics.csv": fit_diagnostics,
        "quadratic_screen_predictions.csv": quadratic_predictions,
        "quadratic_screen_coefficients.csv": quadratic_coefficients,
        "quadratic_screen_fit_diagnostics.csv": quadratic_diagnostics,
        "quadratic_screen_mean_effect_predictions.csv": quadratic_mean_predictions,
        "quadratic_screen_mean_effect_coefficients.csv": quadratic_mean_coefficients,
        "quadratic_screen_mean_effect_fit_diagnostics.csv": quadratic_mean_diagnostics,
        "all_selector_predictions.csv": all_predictions,
        "fixed_mask_decisions.csv": decisions,
        "test_prompt_outcomes.csv": outcomes,
        "risk_prediction_metrics.csv": prediction_metrics,
        "selector_summary.csv": summary,
        "selection_change_summary.csv": selection_changes,
        "fixed_action_oracle.csv": oracle,
        "decision_quality.csv": quality,
        "paired_contrasts.csv": contrasts,
        "pool_contrasts.csv": pool_results,
        "budget_descriptives.csv": budget_results,
    }
    for name, frame in paths.items():
        frame.to_csv(output / name, index=False)
    write_json(output / "result_digest.json", digest)
    readme = (
        "# Exploratory post-outcome IOI risk observer\n\n"
        f"Status: `{EXPLORATORY_STATUS}`.\n\n"
        "This analysis was conceived after the Phase-5 test outcomes had been opened. "
        "It fits only the retained train-prompt calibration effects, but its conception "
        "and interpretation are post-outcome. Treat every estimate and interval here as "
        "hypothesis-generating, not confirmatory. No manuscript source was changed.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")

    artifact_paths = [output / name for name in paths]
    artifact_paths.extend([output / "result_digest.json", output / "README.md"])
    manifest = {
        "schema": "observerbench.ioi_risk_exploratory_run.v1",
        "status": EXPLORATORY_STATUS,
        "post_outcome_disclosure": {
            "phase5_test_already_opened": True,
            "phase5_confirmatory_manifest": file_sha256(
                confirmatory_manifest_path
            ),
            "claim_status": "hypothesis_generating_only",
        },
        "retained_decision_problem": {
            "design_id": design_manifest.get("design_id"),
            "measurement_budgets": list(config.budgets),
            "targets": list(config.targets),
            "models": list(config.models),
            "candidate_masks": int(len(candidates)),
            "candidate_pools": int(candidates["pool_id"].nunique()),
        },
        "changed_estimand": {
            "old_response": "mean_train_prompt_effect",
            "new_response": "mean_train_prompt_absolute_target_loss",
            "new_response_is_target_specific": True,
        },
        "bounded_posthoc_screen": {
            "model": HEAD_QUADRATIC_MODEL,
            "basis": "intercept + 13 head indicators + 78 distinct head-pair products",
            "measurement_budget": config.primary_budget,
            "ridge": config.ridge,
            "ridge_tuned_on_test": False,
            "capacity_matched_mean_effect_control": True,
            "screen_status": "conceived_after_initial_direct_risk_results",
        },
        "config": asdict(config),
        "config_sha256": (
            file_sha256(config_path)
            if config_path is not None
            else json_sha256(asdict(config))
        ),
        "inputs": {
            "design_manifest": file_sha256(
                Path(design_dir) / "design_manifest.json"
            ),
            "effect_manifest": file_sha256(
                Path(effects_dir) / "effect_manifest.json"
            ),
            "mean_fit_manifest": file_sha256(mean_fit_manifest_path),
            "phase5_confirmatory_manifest": file_sha256(
                confirmatory_manifest_path
            ),
            "train_effect_sources": source_hashes(train_paths, effects_dir),
            "test_effect_sources": source_hashes(test_paths, effects_dir),
            "effect_manifest_status": effect_manifest.get("status"),
        },
        "outputs": source_hashes(artifact_paths, output),
        "runtime": runtime_provenance(),
    }
    write_json(output / "exploratory_manifest.json", manifest)
    return digest
