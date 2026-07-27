"""Model-free fitting and action analysis for Qwen induction effects.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from observerbench.tasks.ioi.stage2d import ridge_fit


NO_EFFECT = "no_effect"
ADDITIVE = "additive"
QUADRATIC = "quadratic"
NATURAL_MEAN = "natural_mean"
TRANSFORMED_MEAN = "transformed_mean"
DIRECT_RISK = "direct_risk"
EXACT_NOOP = "exact_noop"

MEAN_EFFECT_MODELS = (NO_EFFECT, ADDITIVE, QUADRATIC)
ACTION_SELECTORS = (NATURAL_MEAN, TRANSFORMED_MEAN, DIRECT_RISK, EXACT_NOOP)


def parse_mask_bits(value: str | Sequence[int], n_components: int = 8) -> np.ndarray:
    """Return one checked binary mask in component order."""

    if isinstance(value, str):
        text = value.strip()
        if len(text) != n_components or set(text) - {"0", "1"}:
            raise ValueError(f"mask bits must contain exactly {n_components} binary digits")
        result = np.asarray([int(bit) for bit in text], dtype=float)
    else:
        result = np.asarray(tuple(value), dtype=float)
        if result.shape != (n_components,) or not np.isin(result, (0.0, 1.0)).all():
            raise ValueError(f"mask must have shape ({n_components},) and be binary")
    return result


def masks_from_frame(frame: pd.DataFrame, n_components: int = 8) -> np.ndarray:
    """Compose a binary mask matrix from a table with a ``mask_bits`` column."""

    if "mask_bits" not in frame:
        raise ValueError("mask table lacks mask_bits")
    return np.stack(
        [parse_mask_bits(value, n_components) for value in frame["mask_bits"].astype(str)],
        axis=0,
    )


def mask_design_matrix(
    masks: np.ndarray,
    family: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the fixed additive or all-pairs head-mask basis."""

    masks = np.asarray(masks, dtype=float)
    if masks.ndim != 2 or masks.shape[1] != 8 or not np.isin(masks, (0.0, 1.0)).all():
        raise ValueError("Qwen induction masks must be a binary n-by-8 matrix")
    main_names = tuple(f"head_{index}" for index in range(8))
    base = np.column_stack([np.ones(len(masks)), masks])
    if family == ADDITIVE:
        return base, ("intercept", *main_names)
    if family == QUADRATIC:
        pairs = tuple(combinations(range(8), 2))
        pair_values = np.column_stack(
            [masks[:, left] * masks[:, right] for left, right in pairs]
        )
        pair_names = tuple(f"head_{left}:head_{right}" for left, right in pairs)
        return np.column_stack([base, pair_values]), (
            "intercept",
            *main_names,
            *pair_names,
        )
    if family == NO_EFFECT:
        return np.ones((len(masks), 1), dtype=float), ("intercept",)
    raise ValueError(f"unknown mean-effect model: {family}")


def _folds(n_rows: int, n_folds: int, seed: int) -> tuple[np.ndarray, ...]:
    if n_rows < 2:
        raise ValueError("ridge selection requires at least two masks")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    return tuple(block for block in np.array_split(order, min(n_folds, n_rows)) if len(block))


def choose_ridge(
    design: np.ndarray,
    response: np.ndarray,
    ridge_grid: Iterable[float],
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[float, pd.DataFrame]:
    """Choose ridge by deterministic mask-level cross-validation."""

    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    grid = tuple(float(value) for value in ridge_grid)
    if design.ndim != 2 or response.shape != (len(design),):
        raise ValueError("ridge design and response shapes differ")
    if not grid or any(value <= 0.0 or not np.isfinite(value) for value in grid):
        raise ValueError("ridge grid must contain finite positive values")
    folds = _folds(len(design), n_folds, seed)
    rows: list[dict[str, float | int]] = []
    for ridge in grid:
        errors: list[float] = []
        for fold_index, test in enumerate(folds):
            train = np.setdiff1d(np.arange(len(design)), test, assume_unique=True)
            coefficient = ridge_fit(design[train], response[train], ridge)
            fold_mae = float(np.mean(np.abs(design[test] @ coefficient - response[test])))
            errors.append(fold_mae)
            rows.append({"ridge": ridge, "fold": fold_index, "mae": fold_mae})
        rows.append({"ridge": ridge, "fold": -1, "mae": float(np.mean(errors))})
    diagnostics = pd.DataFrame(rows)
    means = diagnostics.loc[diagnostics["fold"] == -1].sort_values(
        ["mae", "ridge"], kind="mergesort"
    )
    return float(means.iloc[0]["ridge"]), diagnostics


@dataclass(frozen=True)
class FittedMaskModel:
    family: str
    ridge: float
    terms: tuple[str, ...]
    coefficient: np.ndarray

    def predict(self, masks: np.ndarray) -> np.ndarray:
        design, terms = mask_design_matrix(masks, self.family)
        if terms != self.terms:
            raise ValueError("prediction terms differ from fitted mask model")
        return design @ self.coefficient


def fit_mask_model(
    masks: np.ndarray,
    response: np.ndarray,
    family: str,
    ridge_grid: Iterable[float],
    *,
    seed: int,
) -> tuple[FittedMaskModel, pd.DataFrame]:
    """Fit one fixed-basis response model after mask-level ridge selection."""

    design, terms = mask_design_matrix(masks, family)
    response = np.asarray(response, dtype=float)
    if family == NO_EFFECT:
        coefficient = np.asarray([0.0])
        diagnostics = pd.DataFrame([{"ridge": 0.0, "fold": -1, "mae": float(np.mean(np.abs(response))) }])
        return FittedMaskModel(family, 0.0, terms, coefficient), diagnostics
    ridge, diagnostics = choose_ridge(design, response, ridge_grid, seed=seed)
    coefficient = ridge_fit(design, response, ridge)
    return FittedMaskModel(family, ridge, terms, coefficient), diagnostics


def _validate_effect_cells(
    effects: pd.DataFrame,
    masks: pd.DataFrame,
    *,
    expected_split: str,
) -> pd.DataFrame:
    effects = effects.copy()
    if "effect" not in effects and "drop_from_clean" in effects:
        effects["effect"] = pd.to_numeric(
            effects["drop_from_clean"], errors="raise"
        ).astype(float)
    required = {"prompt_id", "family_id", "split", "mask_id", "mask_bits", "effect"}
    missing = required - set(effects)
    if missing:
        raise ValueError(f"effect table lacks columns: {sorted(missing)}")
    aliases = {
        "calibration": {"calibration", "train"},
        "locked_test": {"locked_test", "test"},
    }.get(expected_split, {expected_split})
    observed_splits = set(effects["split"].astype(str))
    if len(observed_splits) != 1 or not observed_splits.issubset(aliases):
        raise ValueError(f"effect table is not confined to {expected_split}")
    if effects[["prompt_id", "mask_id"]].duplicated().any():
        raise ValueError("effect cells must be unique by prompt and mask")
    if {"mask_id", "mask_bits"} - set(masks):
        raise ValueError("mask table lacks mask_id or mask_bits")
    if masks["mask_id"].astype(str).duplicated().any():
        raise ValueError("mask IDs must be unique")
    mask_ids = masks["mask_id"].astype(str).tolist()
    mask_bits_by_id: dict[str, str] = {}
    for row in masks[["mask_id", "mask_bits"]].itertuples(index=False):
        mask_id = str(row.mask_id)
        bits = str(row.mask_bits)
        parse_mask_bits(bits)
        mask_bits_by_id[mask_id] = bits
    effects["mask_id"] = effects["mask_id"].astype(str)
    effects["mask_bits"] = effects["mask_bits"].astype(str)
    if any(
        mask_bits_by_id.get(str(row.mask_id)) != str(row.mask_bits)
        for row in effects[["mask_id", "mask_bits"]].itertuples(index=False)
    ):
        raise ValueError("effect mask bits differ from the frozen mask design")
    effects["effect"] = pd.to_numeric(effects["effect"], errors="raise")
    if not np.isfinite(effects["effect"].to_numpy(float)).all():
        raise ValueError("effect values must be finite")
    grouped = effects.groupby("mask_id", sort=False)
    if set(grouped.groups) != set(mask_ids):
        raise ValueError("effect table and mask table contain different mask IDs")
    prompt_counts = grouped["prompt_id"].nunique()
    if prompt_counts.nunique() != 1:
        raise ValueError("each mask must be measured on the same prompt count")
    prompt_sets = [set(group["prompt_id"].astype(str)) for _, group in grouped]
    if any(values != prompt_sets[0] for values in prompt_sets[1:]):
        raise ValueError("each mask must be measured on the same prompt IDs")
    if (effects.groupby("prompt_id")["family_id"].nunique() != 1).any():
        raise ValueError("prompt family IDs changed across masks")
    return effects.copy()


def freeze_mean_effect_predictions(
    calibration_effects: pd.DataFrame,
    calibration_masks: pd.DataFrame,
    locked_masks: pd.DataFrame,
    *,
    budgets: Sequence[int] = (16, 40, 64, 128),
    ridge_grid: Sequence[float] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0),
    seed: int = 9173,
) -> dict[str, pd.DataFrame]:
    """Fit and freeze every held-out mask prediction before locked effects."""

    calibration_masks = calibration_masks.sort_values("measurement_order").reset_index(drop=True)
    if len(calibration_masks) != 128 or len(locked_masks) != 128:
        raise ValueError("the exhaustive design requires 128 calibration and 128 locked masks")
    if calibration_masks.iloc[0]["mask_bits"] != "00000000":
        raise ValueError("the first calibration mask must be exact no-op")
    effects = _validate_effect_cells(
        calibration_effects,
        calibration_masks,
        expected_split="calibration",
    )
    mean_by_mask = effects.groupby("mask_id")["effect"].mean()
    locked_bits = masks_from_frame(locked_masks)
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    diagnostic_rows: list[pd.DataFrame] = []
    for budget in map(int, budgets):
        if budget <= 0 or budget > len(calibration_masks):
            raise ValueError(f"invalid measurement budget: {budget}")
        subset = calibration_masks.iloc[:budget]
        train_bits = masks_from_frame(subset)
        response = np.asarray([mean_by_mask[str(mask_id)] for mask_id in subset["mask_id"]])
        for model_index, family in enumerate(MEAN_EFFECT_MODELS):
            model, diagnostics = fit_mask_model(
                train_bits,
                response,
                family,
                ridge_grid,
                seed=seed + budget * 10 + model_index,
            )
            predictions = model.predict(locked_bits)
            prediction_rows.extend(
                {
                    "measurement_budget": budget,
                    "model": family,
                    "mask_id": str(mask.mask_id),
                    "mask_bits": str(mask.mask_bits),
                    "predicted_mean_effect": float(prediction),
                }
                for mask, prediction in zip(locked_masks.itertuples(index=False), predictions)
            )
            coefficient_rows.extend(
                {
                    "measurement_budget": budget,
                    "model": family,
                    "ridge": model.ridge,
                    "term": term,
                    "coefficient": float(value),
                }
                for term, value in zip(model.terms, model.coefficient)
            )
            diagnostic = diagnostics.copy()
            diagnostic.insert(0, "measurement_budget", budget)
            diagnostic.insert(1, "model", family)
            diagnostic_rows.append(diagnostic)
    return {
        "predictions": pd.DataFrame(prediction_rows),
        "coefficients": pd.DataFrame(coefficient_rows),
        "ridge_diagnostics": pd.concat(diagnostic_rows, ignore_index=True),
    }


def freeze_actions(
    calibration_effects: pd.DataFrame,
    calibration_masks: pd.DataFrame,
    locked_masks: pd.DataFrame,
    targets: Sequence[float],
    *,
    ridge_grid: Sequence[float] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0),
    seed: int = 9199,
) -> dict[str, pd.DataFrame]:
    """Freeze same-basis selector predictions and actions for all targets."""

    calibration_masks = calibration_masks.sort_values("measurement_order").reset_index(drop=True)
    if len(calibration_masks) != 128 or len(locked_masks) != 128:
        raise ValueError("action freeze requires the complete exhaustive design")
    if "pool_id" not in locked_masks:
        raise ValueError("locked masks lack action-pool identities")
    pool_sizes = locked_masks.groupby("pool_id", sort=True).size()
    if len(pool_sizes) != 16 or set(pool_sizes.astype(int)) != {8}:
        raise ValueError("locked masks must form 16 fixed pools of eight")
    pool_ids = tuple(str(value) for value in pool_sizes.index)
    effects = _validate_effect_cells(
        calibration_effects,
        calibration_masks,
        expected_split="calibration",
    )
    calibration_ids = calibration_masks["mask_id"].astype(str).tolist()
    grouped = effects.groupby("mask_id")
    mean_response = grouped["effect"].mean()
    y_mean = np.asarray([mean_response[mask_id] for mask_id in calibration_ids])
    train_design, terms = mask_design_matrix(masks_from_frame(calibration_masks), QUADRATIC)
    test_design, test_terms = mask_design_matrix(masks_from_frame(locked_masks), QUADRATIC)
    if terms != test_terms or np.linalg.matrix_rank(train_design) != 37:
        raise ValueError("the 37-column quadratic calibration design must have full rank")
    mean_ridge, mean_diagnostics = choose_ridge(
        train_design, y_mean, ridge_grid, seed=seed
    )
    beta_mean = ridge_fit(train_design, y_mean, mean_ridge)
    predicted_mean = test_design @ beta_mean

    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    diagnostic_rows: list[pd.DataFrame] = []
    for target_index, target_value in enumerate(targets):
        target = float(target_value)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("action targets must be finite and positive")
        y_risk = np.asarray(
            [
                np.mean(np.abs(grouped.get_group(mask_id)["effect"].to_numpy(float) - target))
                for mask_id in calibration_ids
            ]
        )
        y_transformed = np.abs(y_mean - target)
        response_by_selector = {
            TRANSFORMED_MEAN: y_transformed,
            DIRECT_RISK: y_risk,
        }
        fits: dict[str, tuple[np.ndarray, float]] = {
            NATURAL_MEAN: (beta_mean, mean_ridge),
        }
        for selector_index, selector in enumerate((TRANSFORMED_MEAN, DIRECT_RISK), start=1):
            response = response_by_selector[selector]
            ridge, diagnostics = choose_ridge(
                train_design,
                response,
                ridge_grid,
                seed=seed + 100 * target_index + selector_index,
            )
            fits[selector] = (ridge_fit(train_design, response, ridge), ridge)
            diagnostic = diagnostics.copy()
            diagnostic.insert(0, "target", target)
            diagnostic.insert(1, "selector", selector)
            diagnostic_rows.append(diagnostic)
        for selector, (coefficient, ridge) in fits.items():
            if selector == NATURAL_MEAN:
                predicted_loss = np.abs(predicted_mean - target)
            else:
                predicted_loss = test_design @ coefficient
            prediction_rows.extend(
                {
                    "selector": selector,
                    "target": target,
                    "pool_id": str(mask.pool_id),
                    "mask_id": str(mask.mask_id),
                    "mask_bits": str(mask.mask_bits),
                    "n_heads": int(str(mask.mask_bits).count("1")),
                    "predicted_target_loss": float(loss),
                    "predicted_mean_effect": (
                        float(mean) if selector == NATURAL_MEAN else np.nan
                    ),
                    "is_noop": False,
                }
                for mask, loss, mean in zip(
                    locked_masks.itertuples(index=False), predicted_loss, predicted_mean
                )
            )
            coefficient_rows.extend(
                {
                    "selector": selector,
                    "target": target,
                    "ridge": ridge,
                    "term": term,
                    "coefficient": float(value),
                }
                for term, value in zip(terms, coefficient)
            )
    predictions = pd.DataFrame(prediction_rows)
    noops = pd.DataFrame(
        [
            {
                "selector": selector,
                "target": float(target),
                "pool_id": pool_id,
                "mask_id": "analytic_noop",
                "mask_bits": "00000000",
                "n_heads": 0,
                "predicted_target_loss": abs(float(target)),
                "predicted_mean_effect": 0.0 if selector == NATURAL_MEAN else np.nan,
                "is_noop": True,
            }
            for target in targets
            for selector in (NATURAL_MEAN, TRANSFORMED_MEAN, DIRECT_RISK)
            for pool_id in pool_ids
        ]
    )
    predictions = pd.concat([predictions, noops], ignore_index=True)
    action_rows: list[dict[str, object]] = []
    for (selector, target, pool_id), pool in predictions.groupby(
        ["selector", "target", "pool_id"], sort=True
    ):
        if len(pool) != 9 or int(pool["is_noop"].sum()) != 1:
            raise ValueError("each learned policy must choose among eight masks and no-op")
        chosen = pool.sort_values(
            ["predicted_target_loss", "n_heads", "mask_id"], kind="mergesort"
        ).iloc[0]
        action_rows.append(
            {
                "selector": selector,
                "target": float(target),
                "pool_id": pool_id,
                "selected_mask_id": str(chosen["mask_id"]),
                "selected_mask_bits": str(chosen["mask_bits"]),
                "selected_is_noop": bool(chosen["is_noop"]),
                "predicted_target_loss": float(chosen["predicted_target_loss"]),
            }
        )
    for target in map(float, targets):
        for pool_id in pool_ids:
            action_rows.append(
                {
                    "selector": EXACT_NOOP,
                    "target": target,
                    "pool_id": pool_id,
                    "selected_mask_id": "analytic_noop",
                    "selected_mask_bits": "00000000",
                    "selected_is_noop": True,
                    "predicted_target_loss": abs(target),
                }
            )
    mean_diag = mean_diagnostics.copy()
    mean_diag.insert(0, "target", np.nan)
    mean_diag.insert(1, "selector", NATURAL_MEAN)
    diagnostic_rows.append(mean_diag)
    return {
        "candidate_predictions": predictions.sort_values(
            ["target", "selector", "pool_id", "is_noop", "mask_id"]
        ).reset_index(drop=True),
        "fixed_actions": pd.DataFrame(action_rows).sort_values(
            ["target", "selector", "pool_id"]
        ).reset_index(drop=True),
        "coefficients": pd.DataFrame(coefficient_rows),
        "ridge_diagnostics": pd.concat(diagnostic_rows, ignore_index=True),
    }


def evaluate_mean_effect_predictions(
    prediction_freeze: pd.DataFrame,
    locked_effects: pd.DataFrame,
    locked_masks: pd.DataFrame,
) -> pd.DataFrame:
    """Score frozen mask-level predictions on the locked prompt population."""

    effects = _validate_effect_cells(
        locked_effects, locked_masks, expected_split="locked_test"
    )
    observed = effects.groupby("mask_id")["effect"].mean().rename("observed_mean_effect")
    joined = prediction_freeze.merge(observed, left_on="mask_id", right_index=True, validate="many_to_one")
    rows: list[dict[str, object]] = []
    for (budget, model), group in joined.groupby(["measurement_budget", "model"], sort=True):
        error = group["predicted_mean_effect"].to_numpy(float) - group[
            "observed_mean_effect"
        ].to_numpy(float)
        truth = group["observed_mean_effect"].to_numpy(float)
        denominator = float(np.sum((truth - truth.mean()) ** 2))
        rows.append(
            {
                "measurement_budget": int(budget),
                "model": str(model),
                "n_masks": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "r2": (
                    float(1.0 - np.sum(error**2) / denominator)
                    if denominator > 0.0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_fixed_actions(
    fixed_actions: pd.DataFrame,
    locked_effects: pd.DataFrame,
    locked_masks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed actions and pool oracles from locked prompt-level cells."""

    effects = _validate_effect_cells(
        locked_effects, locked_masks, expected_split="locked_test"
    )
    effects_by_mask = {mask_id: group for mask_id, group in effects.groupby("mask_id")}
    prompt_ids = tuple(sorted(effects["prompt_id"].astype(str).unique()))
    rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    for (target, pool_id), actions in fixed_actions.groupby(["target", "pool_id"], sort=True):
        pool_masks = locked_masks.loc[locked_masks["pool_id"].astype(str) == str(pool_id)]
        risks: dict[str, float] = {"analytic_noop": abs(float(target))}
        for mask_id in pool_masks["mask_id"].astype(str):
            cell = effects_by_mask[mask_id].set_index("prompt_id").loc[list(prompt_ids)]
            risks[mask_id] = float(np.mean(np.abs(cell["effect"].to_numpy(float) - float(target))))
        oracle_id = min(
            risks,
            key=lambda item: (
                risks[item],
                0 if item == "analytic_noop" else str(
                    pool_masks.set_index("mask_id").loc[item, "mask_bits"]
                ).count("1"),
                item,
            ),
        )
        oracle_rows.append(
            {
                "target": float(target),
                "pool_id": str(pool_id),
                "oracle_mask_id": oracle_id,
                "oracle_target_loss": risks[oracle_id],
            }
        )
        for action in actions.itertuples(index=False):
            actual_loss = risks[str(action.selected_mask_id)]
            rows.append(
                {
                    "selector": str(action.selector),
                    "target": float(target),
                    "pool_id": str(pool_id),
                    "selected_mask_id": str(action.selected_mask_id),
                    "selected_is_noop": bool(action.selected_is_noop),
                    "actual_target_loss": actual_loss,
                    "oracle_target_loss": risks[oracle_id],
                    "regret": actual_loss - risks[oracle_id],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(oracle_rows)


def _percentile_summary(
    point: float,
    draws: np.ndarray,
    *,
    interval: float,
) -> tuple[float, float, float]:
    if not 0.0 < interval < 1.0:
        raise ValueError("bootstrap interval must lie strictly between zero and one")
    alpha = (1.0 - interval) / 2.0
    lower, upper = np.quantile(np.asarray(draws, dtype=float), [alpha, 1.0 - alpha])
    return float(point), float(lower), float(upper)


def _mask_cluster_effects(
    effects: pd.DataFrame,
    masks: pd.DataFrame,
    *,
    expected_split: str,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Return a complete mask-by-prompt-cluster finite-effect matrix."""

    checked = _validate_effect_cells(effects, masks, expected_split=expected_split)
    if "cluster_id" not in checked:
        raise ValueError("bootstrap analysis requires frozen prompt cluster IDs")
    checked["cluster_id"] = checked["cluster_id"].astype(str)
    mask_ids = tuple(masks["mask_id"].astype(str))
    cluster_ids = tuple(sorted(checked["cluster_id"].unique()))
    grouped = checked.groupby(["mask_id", "cluster_id"], sort=False)["effect"].mean()
    matrix = np.empty((len(mask_ids), len(cluster_ids)), dtype=float)
    for mask_index, mask_id in enumerate(mask_ids):
        for cluster_index, cluster_id in enumerate(cluster_ids):
            try:
                matrix[mask_index, cluster_index] = float(grouped.loc[(mask_id, cluster_id)])
            except KeyError:
                raise ValueError(
                    "locked effects do not form a complete mask-by-cluster design"
                ) from None
    if not np.isfinite(matrix).all():
        raise ValueError("mask-by-cluster effects must be finite")
    return mask_ids, cluster_ids, matrix


def _mask_cluster_losses(
    effects: pd.DataFrame,
    masks: pd.DataFrame,
    *,
    expected_split: str,
    target: float,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Return E_prompt-in-cluster |effect-target| for every mask and cluster."""

    checked = _validate_effect_cells(effects, masks, expected_split=expected_split)
    if "cluster_id" not in checked:
        raise ValueError("bootstrap analysis requires frozen prompt cluster IDs")
    checked["cluster_id"] = checked["cluster_id"].astype(str)
    checked["target_loss"] = np.abs(checked["effect"].to_numpy(float) - float(target))
    mask_ids = tuple(masks["mask_id"].astype(str))
    cluster_ids = tuple(sorted(checked["cluster_id"].unique()))
    grouped = checked.groupby(["mask_id", "cluster_id"], sort=False)[
        "target_loss"
    ].mean()
    matrix = np.empty((len(mask_ids), len(cluster_ids)), dtype=float)
    for mask_index, mask_id in enumerate(mask_ids):
        for cluster_index, cluster_id in enumerate(cluster_ids):
            try:
                matrix[mask_index, cluster_index] = float(
                    grouped.loc[(mask_id, cluster_id)]
                )
            except KeyError:
                raise ValueError(
                    "locked losses do not form a complete mask-by-cluster design"
                ) from None
    if not np.isfinite(matrix).all():
        raise ValueError("mask-by-cluster target losses must be finite")
    return mask_ids, cluster_ids, matrix


def bootstrap_prediction_contrasts(
    prediction_freeze: pd.DataFrame,
    locked_effects: pd.DataFrame,
    locked_masks: pd.DataFrame,
    *,
    repeats: int = 5000,
    seed: int = 9173,
    interval: float = 0.95,
) -> pd.DataFrame:
    """Paired two-way bootstrap for additive-minus-quadratic held-out MAE.

    Each draw resamples prompt clusters and masks independently, while both
    frozen models receive the same draw. Positive contrasts favor the
    interaction-aware quadratic observer.
    """

    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    mask_ids, cluster_ids, effect_matrix = _mask_cluster_effects(
        locked_effects,
        locked_masks,
        expected_split="locked_test",
    )
    required = {"measurement_budget", "model", "mask_id", "predicted_mean_effect"}
    missing = required - set(prediction_freeze)
    if missing:
        raise ValueError(f"prediction freeze lacks columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    n_masks, n_clusters = effect_matrix.shape
    for budget, group in prediction_freeze.groupby("measurement_budget", sort=True):
        relevant = group.loc[group["model"].isin((ADDITIVE, QUADRATIC))].copy()
        pivot = relevant.pivot(index="mask_id", columns="model", values="predicted_mean_effect")
        if set(pivot.index.astype(str)) != set(mask_ids) or set(pivot.columns) != {
            ADDITIVE,
            QUADRATIC,
        }:
            raise ValueError("prediction freeze does not cover both models on every locked mask")
        pivot.index = pivot.index.astype(str)
        additive = np.asarray([pivot.loc[mask_id, ADDITIVE] for mask_id in mask_ids], dtype=float)
        quadratic = np.asarray([pivot.loc[mask_id, QUADRATIC] for mask_id in mask_ids], dtype=float)
        truth = effect_matrix.mean(axis=1)
        point = float(
            np.mean(np.abs(additive - truth))
            - np.mean(np.abs(quadratic - truth))
        )
        draws = np.empty(repeats, dtype=float)
        for repeat in range(repeats):
            sampled_masks = rng.integers(0, n_masks, size=n_masks)
            sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
            sampled_truth = effect_matrix[np.ix_(sampled_masks, sampled_clusters)].mean(axis=1)
            draws[repeat] = (
                np.mean(np.abs(additive[sampled_masks] - sampled_truth))
                - np.mean(np.abs(quadratic[sampled_masks] - sampled_truth))
            )
        estimate, lower, upper = _percentile_summary(point, draws, interval=interval)
        rows.append(
            {
                "measurement_budget": int(budget),
                "contrast": "additive_mae_minus_quadratic_mae",
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "quadratic_better": bool(lower > 0.0),
                "n_masks": n_masks,
                "n_prompt_clusters": len(cluster_ids),
                "bootstrap_repeats": repeats,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_action_contrasts(
    fixed_actions: pd.DataFrame,
    locked_effects: pd.DataFrame,
    locked_masks: pd.DataFrame,
    *,
    repeats: int = 5000,
    seed: int = 9173,
    interval: float = 0.95,
) -> pd.DataFrame:
    """Paired pool-and-cluster bootstrap of controls minus direct-risk loss.

    Positive contrasts mean that the direct-risk selector achieved lower
    realized target loss than the same-basis comparator.
    """

    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    required = {
        "selector",
        "target",
        "pool_id",
        "selected_mask_id",
        "selected_is_noop",
    }
    missing = required - set(fixed_actions)
    if missing:
        raise ValueError(f"fixed actions lack columns: {sorted(missing)}")
    expected_selectors = {NATURAL_MEAN, TRANSFORMED_MEAN, DIRECT_RISK, EXACT_NOOP}
    pool_ids = tuple(sorted(locked_masks["pool_id"].astype(str).unique()))
    if len(pool_ids) != 16:
        raise ValueError("action bootstrap requires the sixteen frozen action pools")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for target, target_actions in fixed_actions.groupby("target", sort=True):
        mask_ids, cluster_ids, loss_matrix = _mask_cluster_losses(
            locked_effects,
            locked_masks,
            expected_split="locked_test",
            target=float(target),
        )
        mask_index = {mask_id: index for index, mask_id in enumerate(mask_ids)}
        n_clusters = len(cluster_ids)
        if set(target_actions["selector"].astype(str)) != expected_selectors:
            raise ValueError("every target must contain all four frozen selectors")
        losses: dict[str, np.ndarray] = {}
        for selector in sorted(expected_selectors):
            selector_actions = target_actions.loc[
                target_actions["selector"].astype(str) == selector
            ].copy()
            if set(selector_actions["pool_id"].astype(str)) != set(pool_ids):
                raise ValueError("selector actions do not cover every frozen pool")
            selector_actions = selector_actions.set_index(
                selector_actions["pool_id"].astype(str)
            )
            values = np.empty((len(pool_ids), n_clusters), dtype=float)
            for pool_index, pool_id in enumerate(pool_ids):
                action = selector_actions.loc[pool_id]
                if isinstance(action, pd.DataFrame):
                    raise ValueError("selector has duplicate actions within a pool")
                mask_id = str(action["selected_mask_id"])
                if bool(action["selected_is_noop"]):
                    if mask_id != "analytic_noop":
                        raise ValueError("no-op action has a non-analytic mask ID")
                    target_loss = np.full(n_clusters, abs(float(target)), dtype=float)
                else:
                    if mask_id not in mask_index:
                        raise ValueError("selected action is absent from locked effects")
                    allowed = locked_masks.loc[
                        locked_masks["pool_id"].astype(str) == pool_id,
                        "mask_id",
                    ].astype(str)
                    if mask_id not in set(allowed):
                        raise ValueError("selected action lies outside its frozen pool")
                    target_loss = loss_matrix[mask_index[mask_id]]
                values[pool_index] = target_loss
            losses[selector] = values

        direct = losses[DIRECT_RISK]
        for comparator in (NATURAL_MEAN, TRANSFORMED_MEAN, EXACT_NOOP):
            difference = losses[comparator] - direct
            point = float(difference.mean())
            draws = np.empty(repeats, dtype=float)
            for repeat in range(repeats):
                sampled_pools = rng.integers(0, len(pool_ids), size=len(pool_ids))
                sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
                draws[repeat] = float(
                    difference[np.ix_(sampled_pools, sampled_clusters)].mean()
                )
            estimate, lower, upper = _percentile_summary(
                point, draws, interval=interval
            )
            rows.append(
                {
                    "target": float(target),
                    "reference": DIRECT_RISK,
                    "comparator": comparator,
                    "contrast_comparator_minus_reference": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "direct_risk_better": bool(lower > 0.0),
                    "n_action_pools": len(pool_ids),
                    "n_prompt_clusters": n_clusters,
                    "bootstrap_repeats": repeats,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_aggregate_action_contrasts(
    fixed_actions: pd.DataFrame,
    locked_effects: pd.DataFrame,
    locked_masks: pd.DataFrame,
    *,
    repeats: int = 5000,
    seed: int = 9173,
    interval: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap equal-weight action contrasts across the three targets.

    Every draw reuses one pool sample and one prompt-cluster sample for all
    targets and selectors. Positive comparator-minus-direct-risk contrasts
    favor the direct-risk selector.
    """

    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    required = {
        "selector",
        "target",
        "pool_id",
        "selected_mask_id",
        "selected_is_noop",
    }
    missing = required - set(fixed_actions)
    if missing:
        raise ValueError(f"fixed actions lack columns: {sorted(missing)}")
    expected_selectors = {NATURAL_MEAN, TRANSFORMED_MEAN, DIRECT_RISK, EXACT_NOOP}
    targets = tuple(sorted(map(float, fixed_actions["target"].unique())))
    if len(targets) != 3:
        raise ValueError("aggregate action bootstrap requires the three frozen targets")
    pool_ids = tuple(sorted(locked_masks["pool_id"].astype(str).unique()))
    if len(pool_ids) != 16:
        raise ValueError("action bootstrap requires the sixteen frozen action pools")

    differences: dict[str, list[np.ndarray]] = {
        NATURAL_MEAN: [],
        TRANSFORMED_MEAN: [],
        EXACT_NOOP: [],
    }
    common_cluster_ids: tuple[str, ...] | None = None
    for target in targets:
        target_actions = fixed_actions.loc[
            fixed_actions["target"].astype(float) == target
        ].copy()
        mask_ids, cluster_ids, loss_matrix = _mask_cluster_losses(
            locked_effects,
            locked_masks,
            expected_split="locked_test",
            target=target,
        )
        if common_cluster_ids is None:
            common_cluster_ids = cluster_ids
        elif cluster_ids != common_cluster_ids:
            raise ValueError("all targets must use the same frozen prompt clusters")
        mask_index = {mask_id: index for index, mask_id in enumerate(mask_ids)}
        if set(target_actions["selector"].astype(str)) != expected_selectors:
            raise ValueError("every target must contain all four frozen selectors")

        losses: dict[str, np.ndarray] = {}
        for selector in sorted(expected_selectors):
            selector_actions = target_actions.loc[
                target_actions["selector"].astype(str) == selector
            ].copy()
            if set(selector_actions["pool_id"].astype(str)) != set(pool_ids):
                raise ValueError("selector actions do not cover every frozen pool")
            selector_actions = selector_actions.set_index(
                selector_actions["pool_id"].astype(str)
            )
            values = np.empty((len(pool_ids), len(cluster_ids)), dtype=float)
            for pool_index, pool_id in enumerate(pool_ids):
                action = selector_actions.loc[pool_id]
                if isinstance(action, pd.DataFrame):
                    raise ValueError("selector has duplicate actions within a pool")
                mask_id = str(action["selected_mask_id"])
                if bool(action["selected_is_noop"]):
                    if mask_id != "analytic_noop":
                        raise ValueError("no-op action has a non-analytic mask ID")
                    target_loss = np.full(
                        len(cluster_ids), abs(target), dtype=float
                    )
                else:
                    if mask_id not in mask_index:
                        raise ValueError("selected action is absent from locked effects")
                    allowed = locked_masks.loc[
                        locked_masks["pool_id"].astype(str) == pool_id,
                        "mask_id",
                    ].astype(str)
                    if mask_id not in set(allowed):
                        raise ValueError("selected action lies outside its frozen pool")
                    target_loss = loss_matrix[mask_index[mask_id]]
                values[pool_index] = target_loss
            losses[selector] = values

        direct = losses[DIRECT_RISK]
        for comparator in differences:
            differences[comparator].append(losses[comparator] - direct)

    if common_cluster_ids is None:
        raise AssertionError("three frozen targets produced no prompt clusters")
    rng = np.random.default_rng(seed)
    sampled_pools = rng.integers(
        0, len(pool_ids), size=(repeats, len(pool_ids))
    )
    sampled_clusters = rng.integers(
        0, len(common_cluster_ids), size=(repeats, len(common_cluster_ids))
    )
    rows: list[dict[str, object]] = []
    for comparator in (NATURAL_MEAN, TRANSFORMED_MEAN, EXACT_NOOP):
        difference = np.stack(differences[comparator], axis=0)
        point = float(difference.mean())
        draws = np.empty(repeats, dtype=float)
        for repeat in range(repeats):
            sampled = np.take(difference, sampled_pools[repeat], axis=1)
            sampled = np.take(sampled, sampled_clusters[repeat], axis=2)
            draws[repeat] = float(sampled.mean())
        estimate, lower, upper = _percentile_summary(
            point, draws, interval=interval
        )
        rows.append(
            {
                "aggregation": "equal_weight_across_targets",
                "reference": DIRECT_RISK,
                "comparator": comparator,
                "contrast_comparator_minus_reference": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "direct_risk_better": bool(lower > 0.0),
                "n_targets": len(targets),
                "n_action_pools": len(pool_ids),
                "n_prompt_clusters": len(common_cluster_ids),
                "bootstrap_repeats": repeats,
            }
        )
    return pd.DataFrame(rows)


def effect_dispersion_decomposition(
    locked_effects: pd.DataFrame,
    locked_masks: pd.DataFrame,
    targets: Sequence[float],
) -> pd.DataFrame:
    """Compute risk = absolute mean error + prompt-dispersion penalty."""

    effects = _validate_effect_cells(
        locked_effects, locked_masks, expected_split="locked_test"
    )
    mask_meta = locked_masks.set_index(locked_masks["mask_id"].astype(str))
    rows: list[dict[str, object]] = []
    for target in map(float, targets):
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("dispersion targets must be finite and positive")
        for mask_id, group in effects.groupby("mask_id", sort=True):
            values = group["effect"].to_numpy(float)
            mean_effect = float(values.mean())
            risk = float(np.mean(np.abs(values - target)))
            mean_error = abs(mean_effect - target)
            penalty = risk - mean_error
            if penalty < -1e-12:
                raise AssertionError("dispersion penalty violates Jensen's inequality")
            meta = mask_meta.loc[str(mask_id)]
            rows.append(
                {
                    "target": target,
                    "mask_id": str(mask_id),
                    "mask_bits": str(meta["mask_bits"]),
                    "pool_id": str(meta["pool_id"]),
                    "mean_effect": mean_effect,
                    "realized_target_loss": risk,
                    "absolute_mean_error": mean_error,
                    "dispersion_penalty": max(0.0, penalty),
                    "n_prompts": len(values),
                }
            )
    return pd.DataFrame(rows)


def prediction_error_diagnostics(
    prediction_freeze: pd.DataFrame,
    locked_effects: pd.DataFrame,
    locked_masks: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return mask, density, and prompt-level summaries for frozen predictions."""

    effects = _validate_effect_cells(
        locked_effects, locked_masks, expected_split="locked_test"
    )
    if "cluster_id" not in effects:
        raise ValueError("prompt-cell diagnostics require frozen cluster_id values")
    required = {
        "measurement_budget",
        "model",
        "mask_id",
        "mask_bits",
        "predicted_mean_effect",
    }
    missing = required - set(prediction_freeze)
    if missing:
        raise ValueError(f"prediction freeze lacks columns: {sorted(missing)}")
    prediction_freeze = prediction_freeze.copy()
    prediction_freeze["mask_id"] = prediction_freeze["mask_id"].astype(str)
    prediction_freeze["mask_bits"] = prediction_freeze["mask_bits"].astype(str)
    prediction_freeze["model"] = prediction_freeze["model"].astype(str)
    prediction_freeze["predicted_mean_effect"] = pd.to_numeric(
        prediction_freeze["predicted_mean_effect"], errors="raise"
    )
    budgets = pd.to_numeric(
        prediction_freeze["measurement_budget"], errors="raise"
    ).to_numpy(float)
    if (
        not np.isfinite(budgets).all()
        or (budgets <= 0.0).any()
        or not np.equal(budgets, np.floor(budgets)).all()
        or not np.isfinite(
            prediction_freeze["predicted_mean_effect"].to_numpy(float)
        ).all()
    ):
        raise ValueError("prediction freeze contains invalid numeric values")
    prediction_freeze["measurement_budget"] = budgets.astype(int)
    if prediction_freeze[
        ["measurement_budget", "model", "mask_id"]
    ].duplicated().any():
        raise ValueError("prediction freeze contains duplicate mask predictions")
    if set(prediction_freeze["model"]) != set(MEAN_EFFECT_MODELS):
        raise ValueError("prediction freeze contains unexpected model families")
    mask_meta = locked_masks[["mask_id", "mask_bits"]].copy()
    mask_meta["mask_id"] = mask_meta["mask_id"].astype(str)
    mask_meta["mask_bits"] = mask_meta["mask_bits"].astype(str)
    mask_meta["n_heads"] = mask_meta["mask_bits"].astype(str).str.count("1")
    mask_bits_by_id = dict(zip(mask_meta["mask_id"], mask_meta["mask_bits"]))
    for (_budget, _model), group in prediction_freeze.groupby(
        ["measurement_budget", "model"], sort=False
    ):
        if set(group["mask_id"]) != set(mask_bits_by_id):
            raise ValueError("prediction freeze does not cover every locked mask once")
        if any(
            mask_bits_by_id.get(str(row.mask_id)) != str(row.mask_bits)
            for row in group[["mask_id", "mask_bits"]].itertuples(index=False)
        ):
            raise ValueError("prediction mask bits differ from the frozen design")
    observed = effects.groupby("mask_id", sort=True)["effect"].mean().rename(
        "observed_mean_effect"
    )
    mask_errors = prediction_freeze.merge(
        mask_meta[["mask_id", "n_heads"]],
        on="mask_id",
        how="left",
        validate="many_to_one",
    ).merge(observed, left_on="mask_id", right_index=True, validate="many_to_one")
    if mask_errors["n_heads"].isna().any():
        raise ValueError("prediction freeze contains masks outside the locked design")
    mask_errors["error"] = (
        mask_errors["predicted_mean_effect"].to_numpy(float)
        - mask_errors["observed_mean_effect"].to_numpy(float)
    )
    mask_errors["absolute_error"] = np.abs(mask_errors["error"])
    mask_errors["squared_error"] = mask_errors["error"] ** 2

    density_rows: list[dict[str, object]] = []
    for (budget, model, density), group in mask_errors.groupby(
        ["measurement_budget", "model", "n_heads"], sort=True
    ):
        density_rows.append(
            {
                "measurement_budget": int(budget),
                "model": str(model),
                "n_heads": int(density),
                "n_masks": len(group),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "bias": float(group["error"].mean()),
            }
        )

    prompt_cells = prediction_freeze[
        ["measurement_budget", "model", "mask_id", "predicted_mean_effect"]
    ].merge(
        effects[["prompt_id", "family_id", "cluster_id", "mask_id", "effect"]],
        on="mask_id",
        how="inner",
        validate="many_to_many",
    )
    prompt_cells["error"] = (
        prompt_cells["predicted_mean_effect"].to_numpy(float)
        - prompt_cells["effect"].to_numpy(float)
    )
    prompt_cells["absolute_error"] = np.abs(prompt_cells["error"])
    prompt_cells["squared_error"] = prompt_cells["error"] ** 2
    prompt_rows: list[dict[str, object]] = []
    groupings = (
        (["measurement_budget", "model"], "all"),
        (["measurement_budget", "model", "family_id"], None),
    )
    for columns, fixed_family in groupings:
        for keys, group in prompt_cells.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            budget, model = keys[:2]
            family = fixed_family if fixed_family is not None else str(keys[2])
            prompt_rows.append(
                {
                    "measurement_budget": int(budget),
                    "model": str(model),
                    "family_id": family,
                    "n_prompt_mask_cells": len(group),
                    "n_prompt_clusters": int(group["cluster_id"].nunique()),
                    "mae": float(group["absolute_error"].mean()),
                    "rmse": float(np.sqrt(group["squared_error"].mean())),
                    "bias": float(group["error"].mean()),
                }
            )
    return {
        "mask_errors": mask_errors.sort_values(
            ["measurement_budget", "model", "mask_id"]
        ).reset_index(drop=True),
        "density_summary": pd.DataFrame(density_rows),
        "prompt_error_summary": pd.DataFrame(prompt_rows),
    }


def _strict_boolean_values(series: pd.Series, label: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"{label} contains missing values")
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise").to_numpy(float)
        if not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError(f"{label} must contain only booleans or 0/1")
        return numeric.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "0", "1")).all():
        raise ValueError(f"{label} must contain only booleans or 0/1")
    return normalized.isin(("true", "1")).to_numpy(bool)


def intervention_outcome_diagnostics(
    locked_effects: pd.DataFrame,
    locked_masks: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Summarize retained post-intervention accuracy and target-NLL outcomes."""

    effects = _validate_effect_cells(
        locked_effects, locked_masks, expected_split="locked_test"
    )
    required = {
        "clean_candidate_correct",
        "ablated_candidate_correct",
        "clean_top1_correct",
        "ablated_top1_correct",
        "clean_target_nll",
        "ablated_target_nll",
    }
    missing = required - set(effects)
    if missing:
        raise ValueError(f"intervention outcomes lack columns: {sorted(missing)}")
    mask_meta = locked_masks[["mask_id", "mask_bits"]].copy()
    mask_meta["mask_id"] = mask_meta["mask_id"].astype(str)
    mask_meta["n_heads"] = mask_meta["mask_bits"].astype(str).str.count("1")
    effects = effects.merge(
        mask_meta[["mask_id", "n_heads"]],
        on="mask_id",
        how="left",
        validate="many_to_one",
    )
    if effects["n_heads"].isna().any():
        raise ValueError("intervention outcomes contain masks outside the locked design")
    for column in (
        "clean_candidate_correct",
        "ablated_candidate_correct",
        "clean_top1_correct",
        "ablated_top1_correct",
    ):
        effects[column] = _strict_boolean_values(effects[column], column)
    effects["clean_target_nll"] = pd.to_numeric(
        effects["clean_target_nll"], errors="raise"
    )
    effects["ablated_target_nll"] = pd.to_numeric(
        effects["ablated_target_nll"], errors="raise"
    )
    nll = effects[["clean_target_nll", "ablated_target_nll"]].to_numpy(float)
    if not np.isfinite(nll).all() or (nll < 0.0).any():
        raise ValueError("target NLL values must be finite and nonnegative")
    clean_columns = (
        "clean_candidate_correct",
        "clean_top1_correct",
        "clean_target_nll",
    )
    for _prompt_id, group in effects.groupby("prompt_id", sort=False):
        if any(group[column].nunique(dropna=False) != 1 for column in clean_columns):
            raise ValueError("clean intervention baselines changed across masks")
    effects["target_nll_increase"] = (
        effects["ablated_target_nll"] - effects["clean_target_nll"]
    )

    def summarize(columns: list[str]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for keys, group in effects.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {column: value for column, value in zip(columns, keys)}
            row.update(
                {
                    "n_cells": len(group),
                    "clean_candidate_accuracy": float(
                        group["clean_candidate_correct"].mean()
                    ),
                    "ablated_candidate_accuracy": float(
                        group["ablated_candidate_correct"].mean()
                    ),
                    "clean_top1_accuracy": float(group["clean_top1_correct"].mean()),
                    "ablated_top1_accuracy": float(
                        group["ablated_top1_correct"].mean()
                    ),
                    "mean_target_nll_increase": float(
                        group["target_nll_increase"].mean()
                    ),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    return {
        "by_mask": summarize(["mask_id", "mask_bits", "n_heads"]),
        "by_density": summarize(["n_heads"]),
        "by_family": summarize(["family_id"]),
    }


__all__ = [
    "ACTION_SELECTORS",
    "ADDITIVE",
    "DIRECT_RISK",
    "EXACT_NOOP",
    "FittedMaskModel",
    "MEAN_EFFECT_MODELS",
    "NATURAL_MEAN",
    "NO_EFFECT",
    "QUADRATIC",
    "TRANSFORMED_MEAN",
    "choose_ridge",
    "bootstrap_aggregate_action_contrasts",
    "bootstrap_action_contrasts",
    "bootstrap_prediction_contrasts",
    "effect_dispersion_decomposition",
    "intervention_outcome_diagnostics",
    "evaluate_fixed_actions",
    "evaluate_mean_effect_predictions",
    "fit_mask_model",
    "freeze_actions",
    "freeze_mean_effect_predictions",
    "mask_design_matrix",
    "masks_from_frame",
    "parse_mask_bits",
    "prediction_error_diagnostics",
]
