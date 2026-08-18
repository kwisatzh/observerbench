"""Nonlinear suffix-loop control experiment for ObserverBench Phase 5.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The plant intervenes on the readout-position residual after block 0 and reruns
block 1, the final layer norm, and the target head at every step.  Its affine
shadow uses the baseline symmetric finite-response gain along the same fixed
direction.  The controlled direction family fixes the observer self-gain while
varying the response omitted by that observer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance
from observerbench.tasks.trained_ctl1 import (
    TinyTransformerRegressor,
    TrainedTransformerCtl1Config,
    _fit_ridge,
    generate_data,
    train_model,
)


@dataclass
class NonlinearSuffixConfig(TrainedTransformerCtl1Config):
    """Frozen settings for the nonlinear suffix-loop study."""

    seeds: tuple[int, ...] = tuple(range(12))
    gamma_values_natural: tuple[float, ...] = (1.15, 0.0)
    completed_blocks: int = 1
    loop_steps: int = 15
    controller_gain: float = 0.35
    observer_self_gain: float = 0.85
    finite_delta: float = 0.05
    rho_targets: tuple[float, ...] = (-1.25, -0.75, 0.0, 0.75, 1.25)
    bias_targets: tuple[float, ...] = (-0.25, 0.0, 0.25)
    relative_offsets: tuple[float, ...] = (-0.5, -0.25, 0.25, 0.5)
    max_action_diagnostic: float = 1.0
    direction_gain_tolerance: float = 2e-3
    manipulability_tolerance: float = 1e-10
    bootstrap_repeats: int = 5000
    bootstrap_seed: int = 25052


@dataclass(frozen=True)
class PrefixObserver:
    name: str
    intercept: float
    gradient: np.ndarray

    def estimate(self, residual: np.ndarray) -> float:
        return float(self.intercept + np.asarray(residual, dtype=float) @ self.gradient)


@dataclass(frozen=True)
class DirectionSolution:
    direction: np.ndarray
    observer_gain: float
    true_gain: float
    rho: float
    manipulability_denominator: float
    target_gain_error: float
    iterations: int


@dataclass(frozen=True)
class CleanResidualDistanceScale:
    """Within-model scale from pairwise clean-archetype readout distances."""

    archetype_count: int
    pair_count: int
    nonzero_pair_count: int
    median: float
    maximum: float
    median_degenerate: bool
    maximum_degenerate: bool
    tolerance: float


def _tokens_for_archetypes(seq_len: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if seq_len != 4:
        raise ValueError("the frozen TinyTransformer fixture requires seq_len=4")
    archetypes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = []
    for x1, x2 in archetypes:
        rows.append([0, 2 if x1 else 1, 4 if x2 else 3, 5])
    return np.asarray(rows, dtype=np.int64), archetypes


def extract_prefix_residuals(
    model: TinyTransformerRegressor,
    tokens: np.ndarray,
    completed_blocks: int,
) -> np.ndarray:
    """Extract the full prefix residual stream on CPU."""

    model.eval()
    device = next(model.parameters()).device
    token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
    with torch.no_grad():
        residual = model.encode_prefix(token_tensor, completed_blocks=completed_blocks)
    return residual.detach().cpu().numpy()


def suffix_values(
    model: TinyTransformerRegressor,
    prefix: np.ndarray,
    completed_blocks: int,
) -> np.ndarray:
    """Evaluate the nonlinear suffix for one or more full residual streams."""

    model.eval()
    device = next(model.parameters()).device
    residual = torch.as_tensor(prefix, dtype=torch.float32, device=device)
    with torch.no_grad():
        target, _features = model.decode_suffix(residual, completed_blocks=completed_blocks)
    return target.detach().cpu().numpy()


def _local_suffix_gradient(
    model: TinyTransformerRegressor,
    prefix: np.ndarray,
    completed_blocks: int,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    residual = torch.as_tensor(prefix[None], dtype=torch.float32, device=device).clone().detach()
    residual.requires_grad_(True)
    target, _features = model.decode_suffix(residual, completed_blocks=completed_blocks)
    gradient = torch.autograd.grad(target.sum(), residual)[0][0, -1]
    return gradient.detach().cpu().numpy().astype(float)


def _symmetric_gain(
    model: TinyTransformerRegressor,
    prefix: np.ndarray,
    direction: np.ndarray,
    completed_blocks: int,
    delta: float,
) -> float:
    pair = np.repeat(prefix[None], 2, axis=0)
    pair[0, -1] += delta * direction
    pair[1, -1] -= delta * direction
    values = suffix_values(model, pair, completed_blocks)
    return float((values[0] - values[1]) / (2.0 * delta))


def solve_controlled_direction(
    model: TinyTransformerRegressor,
    prefix: np.ndarray,
    observer_gradient: np.ndarray,
    *,
    completed_blocks: int,
    observer_gain: float,
    rho_target: float,
    delta: float,
    manipulability_tolerance: float,
    gain_tolerance: float,
) -> DirectionSolution:
    """Solve two gain constraints and refine against the finite suffix response.

    The base component fixes ``w_E^T d``.  A component of the local true
    gradient orthogonal to ``w_E`` varies the true response without changing
    the observer response.  A scalar Newton refinement targets the symmetric
    finite response at ``delta`` rather than only its differential limit.
    """

    w_e = np.asarray(observer_gradient, dtype=float)
    w_norm_sq = float(w_e @ w_e)
    if w_norm_sq <= manipulability_tolerance:
        raise ValueError(f"observer gradient is degenerate: squared norm={w_norm_sq:.6g}")
    w_t = _local_suffix_gradient(model, prefix, completed_blocks)
    omitted_axis = w_t - (float(w_t @ w_e) / w_norm_sq) * w_e
    denominator = float(omitted_axis @ omitted_axis)
    if denominator <= manipulability_tolerance:
        raise ValueError(
            "omitted response is not manipulable: "
            f"orthogonal true-gradient squared norm={denominator:.6g}"
        )

    base = (observer_gain / w_norm_sq) * w_e
    target_gain = observer_gain * (1.0 + rho_target)
    coefficient = (target_gain - float(w_t @ base)) / denominator

    def make_direction(value: float) -> np.ndarray:
        direction = base + value * omitted_axis
        # Remove roundoff in the equality constraint.
        correction = (observer_gain - float(w_e @ direction)) / w_norm_sq
        return direction + correction * w_e

    iterations = 0
    for iterations in range(1, 26):
        direction = make_direction(coefficient)
        measured = _symmetric_gain(model, prefix, direction, completed_blocks, delta)
        error = measured - target_gain
        if abs(error) <= gain_tolerance:
            break
        step = max(1e-4, 1e-3 * max(1.0, abs(coefficient)))
        plus = _symmetric_gain(
            model,
            prefix,
            make_direction(coefficient + step),
            completed_blocks,
            delta,
        )
        minus = _symmetric_gain(
            model,
            prefix,
            make_direction(coefficient - step),
            completed_blocks,
            delta,
        )
        derivative = (plus - minus) / (2.0 * step)
        if not np.isfinite(derivative) or abs(derivative) <= 1e-10:
            raise ValueError(
                "finite-response direction refinement is singular: "
                f"derivative={derivative:.6g}, rho_target={rho_target:.6g}"
            )
        coefficient -= error / derivative
    direction = make_direction(coefficient)
    measured = _symmetric_gain(model, prefix, direction, completed_blocks, delta)
    target_error = float(measured - target_gain)
    if not np.isfinite(measured) or abs(target_error) > gain_tolerance:
        raise ValueError(
            "finite-response direction did not reach the frozen target: "
            f"error={target_error:.6g}, tolerance={gain_tolerance:.6g}"
        )
    measured_observer_gain = float(w_e @ direction)
    return DirectionSolution(
        direction=direction,
        observer_gain=measured_observer_gain,
        true_gain=measured,
        rho=float((measured - measured_observer_gain) / measured_observer_gain),
        manipulability_denominator=denominator,
        target_gain_error=target_error,
        iterations=iterations,
    )


def _fit_prefix_observers(
    model: TinyTransformerRegressor,
    train: dict[str, np.ndarray],
    cfg: NonlinearSuffixConfig,
) -> tuple[dict[str, PrefixObserver], np.ndarray]:
    prefix = extract_prefix_residuals(model, train["tokens"], cfg.completed_blocks)
    readout = prefix[:, -1, :]
    target = suffix_values(model, prefix, cfg.completed_blocks)
    probe_x1 = _fit_ridge(readout, train["x1"], cfg.ridge)
    probe_x2 = _fit_ridge(readout, train["x2"], cfg.ridge)
    probe_int = _fit_ridge(readout, train["interaction"], cfg.ridge)
    x1_hat = np.c_[np.ones(len(readout)), readout] @ probe_x1
    x2_hat = np.c_[np.ones(len(readout)), readout] @ probe_x2
    int_hat = np.c_[np.ones(len(readout)), readout] @ probe_int
    fo_coef = _fit_ridge(np.c_[x1_hat, x2_hat], target, cfg.ridge)
    li_coef = _fit_ridge(np.c_[x1_hat, x2_hat, int_hat], target, cfg.ridge)

    def compose(name: str, coef: np.ndarray, probes: Iterable[np.ndarray]) -> PrefixObserver:
        probe_list = list(probes)
        intercept = float(coef[0] + sum(c * p[0] for c, p in zip(coef[1:], probe_list)))
        gradient = np.sum(
            np.stack([c * p[1:] for c, p in zip(coef[1:], probe_list)]),
            axis=0,
        )
        return PrefixObserver(name, intercept, gradient)

    return {
        "first_order": compose("first_order", fo_coef, (probe_x1, probe_x2)),
        "lifted_interaction": compose(
            "lifted_interaction",
            li_coef,
            (probe_x1, probe_x2, probe_int),
        ),
    }, readout


def _support_fraction(direction: np.ndarray, clean_readout: np.ndarray) -> float:
    centered = clean_readout - clean_readout.mean(axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0 or singular[0] <= 1e-12:
        return 0.0
    rank = int(np.sum(singular > singular[0] * 1e-7))
    basis = vh[:rank].T
    projected = basis @ (basis.T @ direction)
    return float(np.linalg.norm(projected) / max(np.linalg.norm(direction), 1e-12))


def _clean_residual_distance_scale(
    clean_readout: np.ndarray,
    tolerance: float,
) -> CleanResidualDistanceScale:
    """Measure clean variation at the same residual site used by the rollout.

    With the frozen fixture, four clean archetypes yield six pairwise distances.
    The empirical median is a typical clean task-variation scale and the maximum
    is the observed clean envelope.  Neither quantity estimates the support of
    the full activation distribution.
    """

    residuals = np.asarray(clean_readout, dtype=float)
    if residuals.ndim != 2 or residuals.shape[0] < 2:
        raise ValueError("clean readout must contain at least two residual vectors")
    if tolerance < 0.0:
        raise ValueError("clean-distance tolerance must be nonnegative")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("clean readout contains a non-finite value")
    upper_i, upper_j = np.triu_indices(residuals.shape[0], k=1)
    distances = np.linalg.norm(residuals[upper_i] - residuals[upper_j], axis=1)
    median = float(np.median(distances))
    maximum = float(np.max(distances))
    return CleanResidualDistanceScale(
        archetype_count=int(residuals.shape[0]),
        pair_count=int(len(distances)),
        nonzero_pair_count=int(np.sum(distances > tolerance)),
        median=median,
        maximum=maximum,
        median_degenerate=bool(median <= tolerance),
        maximum_degenerate=bool(maximum <= tolerance),
        tolerance=float(tolerance),
    )


def _displacement_scale_diagnostics(
    displacement_l2: float,
    clean_scale: CleanResidualDistanceScale,
) -> dict[str, Any]:
    """Normalize a rollout displacement without hiding collapsed clean scales."""

    displacement = float(displacement_l2)
    if not np.isfinite(displacement) or displacement < 0.0:
        raise ValueError("residual displacement must be finite and nonnegative")

    def ratio(denominator: float, degenerate: bool) -> float:
        if degenerate:
            return float("nan")
        return float(displacement / denominator)

    return {
        "clean_archetype_count": clean_scale.archetype_count,
        "clean_pairwise_distance_count": clean_scale.pair_count,
        "clean_pairwise_nonzero_distance_count": clean_scale.nonzero_pair_count,
        "clean_pairwise_distance_median": clean_scale.median,
        "clean_pairwise_distance_max": clean_scale.maximum,
        "clean_pairwise_median_degenerate": clean_scale.median_degenerate,
        "clean_pairwise_max_degenerate": clean_scale.maximum_degenerate,
        "residual_displacement_over_clean_pairwise_median": ratio(
            clean_scale.median,
            clean_scale.median_degenerate,
        ),
        "residual_displacement_over_clean_pairwise_max": ratio(
            clean_scale.maximum,
            clean_scale.maximum_degenerate,
        ),
    }


def _certificate(
    observer_error_initial: float,
    mismatch_initial: float,
    rho: float,
    pole: float,
    steps: int,
) -> float:
    pole_power = pole ** steps
    return float(
        pole_power * observer_error_initial
        - mismatch_initial
        - rho * (1.0 - pole_power) * observer_error_initial
    )


def _rollout(
    model: TinyTransformerRegressor,
    prefix: np.ndarray,
    observer: PrefixObserver,
    direction: np.ndarray,
    target_ref: float,
    clean_scale: CleanResidualDistanceScale,
    cfg: NonlinearSuffixConfig,
) -> dict[str, Any]:
    r0 = prefix[-1].copy()
    z0 = float(suffix_values(model, prefix[None], cfg.completed_blocks)[0])
    zhat0 = observer.estimate(r0)
    g_e = float(observer.gradient @ direction)
    g_t = _symmetric_gain(
        model,
        prefix,
        direction,
        cfg.completed_blocks,
        cfg.finite_delta,
    )
    if abs(g_e) <= 1e-12:
        raise ValueError("observer has zero response along rollout direction")
    rho = float((g_t - g_e) / g_e)
    pole = float(1.0 - cfg.controller_gain * g_e)
    m0 = float(z0 - zhat0)
    ehat0 = float(target_ref - zhat0)
    predicted_final = _certificate(ehat0, m0, rho, pole, cfg.loop_steps)
    pole_only_final = float((pole ** cfg.loop_steps) * ehat0)
    bias_only_final = float(pole_only_final - m0)

    working = prefix.copy()
    scalar_displacement = 0.0
    true_errors: list[float] = []
    shadow_errors: list[float] = []
    actions: list[float] = []
    z_final = z0
    z_shadow_final = z0
    for step in range(cfg.loop_steps + 1):
        z = float(suffix_values(model, working[None], cfg.completed_blocks)[0])
        zhat = observer.estimate(working[-1])
        z_shadow = float(z0 + g_t * scalar_displacement)
        true_errors.append(float(target_ref - z))
        shadow_errors.append(float(target_ref - z_shadow))
        z_final = z
        z_shadow_final = z_shadow
        if step == cfg.loop_steps:
            break
        action = float(cfg.controller_gain * (target_ref - zhat))
        actions.append(action)
        scalar_displacement += action
        working[-1] += action * direction

    actual_final = true_errors[-1]
    shadow_final = shadow_errors[-1]
    residual_displacement_l2 = float(abs(scalar_displacement) * np.linalg.norm(direction))
    return {
        "z0": z0,
        "zhat0": zhat0,
        "target_ref": float(target_ref),
        "initial_true_error": float(true_errors[0]),
        "initial_observer_error": ehat0,
        "initial_mismatch": m0,
        "observer_gain": g_e,
        "true_symmetric_gain": g_t,
        "rho_measured": rho,
        "observer_pole": pole,
        "certificate_final_error": predicted_final,
        "pole_only_final_error": pole_only_final,
        "bias_only_final_error": bias_only_final,
        "actual_final_error": actual_final,
        "shadow_final_error": shadow_final,
        "certificate_residual": float(actual_final - predicted_final),
        "actual_integrated_squared_error": float(np.sum(np.square(true_errors))),
        "shadow_integrated_squared_error": float(np.sum(np.square(shadow_errors))),
        "final_suffix_value": z_final,
        "final_shadow_value": z_shadow_final,
        "final_suffix_shadow_abs_gap": float(abs(z_final - z_shadow_final)),
        "final_suffix_move_abs": float(abs(z_final - z0)),
        "scalar_displacement": float(scalar_displacement),
        "residual_displacement_l2": residual_displacement_l2,
        "direction_norm": float(np.linalg.norm(direction)),
        "max_abs_action": float(max((abs(value) for value in actions), default=0.0)),
        "would_clip_fraction": float(
            np.mean(np.abs(actions) > cfg.max_action_diagnostic) if actions else 0.0
        ),
        **_displacement_scale_diagnostics(residual_displacement_l2, clean_scale),
    }


def _controlled_rows_for_seed(
    model: TinyTransformerRegressor,
    observers: dict[str, PrefixObserver],
    archetype_prefixes: np.ndarray,
    archetypes: list[tuple[int, int]],
    clean_readout: np.ndarray,
    cfg: NonlinearSuffixConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    base_observer = observers["first_order"]
    clean_scale = _clean_residual_distance_scale(
        clean_readout,
        cfg.manipulability_tolerance,
    )
    for archetype_index, ((x1, x2), prefix) in enumerate(zip(archetypes, archetype_prefixes)):
        z0 = float(suffix_values(model, prefix[None], cfg.completed_blocks)[0])
        for rho_target in cfg.rho_targets:
            try:
                solution = solve_controlled_direction(
                    model,
                    prefix,
                    base_observer.gradient,
                    completed_blocks=cfg.completed_blocks,
                    observer_gain=cfg.observer_self_gain,
                    rho_target=rho_target,
                    delta=cfg.finite_delta,
                    manipulability_tolerance=cfg.manipulability_tolerance,
                    gain_tolerance=cfg.direction_gain_tolerance,
                )
            except ValueError as exc:
                failures.append({
                    "seed": seed,
                    "archetype_index": archetype_index,
                    "x1": x1,
                    "x2": x2,
                    "rho_target": rho_target,
                    "reason": str(exc),
                })
                continue
            support_fraction = _support_fraction(solution.direction, clean_readout)
            for bias in cfg.bias_targets:
                local_observer = PrefixObserver(
                    name="controlled_first_order_gradient",
                    intercept=float(z0 - bias - prefix[-1] @ base_observer.gradient),
                    gradient=base_observer.gradient,
                )
                for offset in cfg.relative_offsets:
                    rollout = _rollout(
                        model,
                        prefix,
                        local_observer,
                        solution.direction,
                        z0 + offset,
                        clean_scale,
                        cfg,
                    )
                    rows.append({
                        "seed": seed,
                        "gamma": float(cfg.gamma),
                        "archetype_index": archetype_index,
                        "x1": x1,
                        "x2": x2,
                        "rho_target": rho_target,
                        "bias_target": bias,
                        "relative_offset": offset,
                        "direction_support_fraction": support_fraction,
                        "manipulability_denominator": solution.manipulability_denominator,
                        "direction_target_gain_error": solution.target_gain_error,
                        "direction_solver_iterations": solution.iterations,
                        **rollout,
                    })
    return rows, failures


def _natural_rows_for_seed(
    model: TinyTransformerRegressor,
    observers: dict[str, PrefixObserver],
    archetype_prefixes: np.ndarray,
    archetypes: list[tuple[int, int]],
    clean_readout: np.ndarray,
    cfg: NonlinearSuffixConfig,
    seed: int,
    gamma: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean_scale = _clean_residual_distance_scale(
        clean_readout,
        cfg.manipulability_tolerance,
    )
    directions: dict[str, np.ndarray] = {}
    for name, observer in observers.items():
        norm_sq = float(observer.gradient @ observer.gradient)
        if norm_sq <= cfg.manipulability_tolerance:
            raise ValueError(f"natural {name} direction is degenerate for seed {seed}, gamma {gamma}")
        directions[name] = cfg.observer_self_gain * observer.gradient / norm_sq
    for archetype_index, ((x1, x2), prefix) in enumerate(zip(archetypes, archetype_prefixes)):
        z0 = float(suffix_values(model, prefix[None], cfg.completed_blocks)[0])
        for estimator_name, observer in observers.items():
            for direction_name, direction in directions.items():
                support_fraction = _support_fraction(direction, clean_readout)
                for offset in cfg.relative_offsets:
                    rollout = _rollout(
                        model,
                        prefix,
                        observer,
                        direction,
                        z0 + offset,
                        clean_scale,
                        cfg,
                    )
                    rows.append({
                        "seed": seed,
                        "gamma": gamma,
                        "archetype_index": archetype_index,
                        "x1": x1,
                        "x2": x2,
                        "estimator": estimator_name,
                        "direction_provider": direction_name,
                        "relative_offset": offset,
                        "direction_support_fraction": support_fraction,
                        **rollout,
                    })
    return rows


def _leave_one_seed_out(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    predictors = {
        "full_certificate": "certificate_final_error",
        "pole_only": "pole_only_final_error",
        "bias_only": "bias_only_final_error",
    }
    rows = []
    for held_seed in sorted(frame["seed"].unique()):
        train = frame[frame["seed"] != held_seed]
        test = frame[frame["seed"] == held_seed]
        for name, column in predictors.items():
            design = np.c_[np.ones(len(train)), train[column].to_numpy(float)]
            coef, *_ = np.linalg.lstsq(design, train["actual_final_error"].to_numpy(float), rcond=None)
            prediction = np.c_[np.ones(len(test)), test[column].to_numpy(float)] @ coef
            residual = test["actual_final_error"].to_numpy(float) - prediction
            rows.append({
                "held_out_seed": int(held_seed),
                "predictor": name,
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "calibration_intercept": float(coef[0]),
                "calibration_slope": float(coef[1]),
            })
    result = pd.DataFrame(rows)
    means = result.groupby("predictor")["mae"].mean().to_dict()
    return result, {str(key): float(value) for key, value in means.items()}


def _bootstrap_seed_difference(
    cv: pd.DataFrame,
    comparator: str,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    pivot = cv.pivot(index="held_out_seed", columns="predictor", values="mae")
    paired = (pivot[comparator] - pivot["full_certificate"]).to_numpy(float)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(paired), size=len(paired))
        draws[index] = float(np.mean(paired[sample]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_mae_advantage": float(np.mean(paired)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _displacement_group_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize rollout displacement against within-model clean scales."""

    if frame.empty:
        return {
            "condition_count": 0,
            "seed_count": 0,
            "clean_scale_count": 0,
        }
    scale_keys = [key for key in ("seed", "gamma") if key in frame.columns]
    scales = frame.drop_duplicates(scale_keys) if scale_keys else frame.iloc[:1]

    result: dict[str, Any] = {
        "condition_count": int(len(frame)),
        "seed_count": int(frame["seed"].nunique()) if "seed" in frame else 0,
        "clean_scale_count": int(len(scales)),
        "clean_scales_with_degenerate_median": int(
            scales["clean_pairwise_median_degenerate"].astype(bool).sum()
        ),
        "clean_scales_with_degenerate_max": int(
            scales["clean_pairwise_max_degenerate"].astype(bool).sum()
        ),
    }
    displacement = frame["residual_displacement_l2"].to_numpy(float)
    result.update({
        "residual_displacement_l2_median": float(np.median(displacement)),
        "residual_displacement_l2_p95": float(np.quantile(displacement, 0.95)),
        "residual_displacement_l2_max": float(np.max(displacement)),
    })
    for scale_name in ("median", "max"):
        column = f"residual_displacement_over_clean_pairwise_{scale_name}"
        valid = frame[column].to_numpy(float)
        valid = valid[np.isfinite(valid)]
        prefix = f"ratio_to_clean_pairwise_{scale_name}"
        result[f"{prefix}_valid_condition_count"] = int(len(valid))
        if len(valid) == 0:
            result[f"{prefix}_median"] = None
            result[f"{prefix}_p95"] = None
            result[f"{prefix}_max"] = None
            result[f"{prefix}_fraction_above_one"] = None
        else:
            result[f"{prefix}_median"] = float(np.median(valid))
            result[f"{prefix}_p95"] = float(np.quantile(valid, 0.95))
            result[f"{prefix}_max"] = float(np.max(valid))
            result[f"{prefix}_fraction_above_one"] = float(np.mean(valid > 1.0))
    return result


def _residual_displacement_diagnostic(
    controlled: pd.DataFrame,
    natural: pd.DataFrame,
) -> dict[str, Any]:
    """Build diagnostic-only summaries without changing scientific gates."""

    natural_diag = natural[natural["estimator"] == natural["direction_provider"]]
    natural_rows = []
    for (gamma, estimator), group in natural_diag.groupby(["gamma", "estimator"]):
        natural_rows.append({
            "gamma": float(gamma),
            "estimator": str(estimator),
            **_displacement_group_summary(group),
        })
    controlled_by_rho = []
    for rho, group in controlled.groupby("rho_target"):
        controlled_by_rho.append({
            "rho_target": float(rho),
            **_displacement_group_summary(group),
        })
    return {
        "status": "diagnostic_only_not_a_success_gate",
        "normalization": (
            "For each seed and trained model, divide ||h_T-h_0|| at the intervened "
            "readout position by the median or maximum of all pairwise distances "
            "among the four clean archetype residuals at that same position."
        ),
        "interpretation": (
            "The median is a typical task-archetype variation scale; the maximum "
            "is the observed clean-archetype envelope. A ratio above one is not, "
            "by itself, proof that a rollout is outside the model's activation manifold."
        ),
        "limitation": (
            "Four archetypes provide only six pairwise distances. They cover the "
            "frozen binary fixture, not nuisance variation or the full clean activation distribution."
        ),
        "natural_diagonal_by_gamma_estimator": natural_rows,
        "controlled_overall": _displacement_group_summary(controlled),
        "controlled_by_rho_target": controlled_by_rho,
    }


def _natural_factorial_analysis(
    natural: pd.DataFrame,
    cfg: NonlinearSuffixConfig,
) -> dict[str, Any]:
    """Summarize the frozen natural 2x2 without treating it as a new gate.

    The natural run already crosses the first-order and lifted estimators with
    both observer-derived directions.  The diagonal cells answer a system-level
    question.  The crossed cells can isolate a factor only when the resulting
    feedback loop remains well conditioned and within the experiment's clean-
    archetype displacement scale.  We therefore report the factorial and its
    diagnostics together.
    """

    metric_columns = [
        "actual_integrated_squared_error",
        "observer_gain",
        "observer_pole",
    ]
    seed_cells = natural.groupby(
        ["seed", "gamma", "estimator", "direction_provider"],
        as_index=False,
    )[metric_columns].mean()
    by_gamma: dict[str, Any] = {}
    estimators = ("first_order", "lifted_interaction")

    for gamma_index, gamma in enumerate(sorted(seed_cells["gamma"].unique())):
        gamma_rows = natural[np.isclose(natural["gamma"], gamma)]
        gamma_seed_cells = seed_cells[np.isclose(seed_cells["gamma"], gamma)]
        pivot = gamma_seed_cells.pivot(
            index="seed",
            columns=["estimator", "direction_provider"],
            values="actual_integrated_squared_error",
        )
        ff = pivot[("first_order", "first_order")]
        fl = pivot[("first_order", "lifted_interaction")]
        lf = pivot[("lifted_interaction", "first_order")]
        ll = pivot[("lifted_interaction", "lifted_interaction")]
        contrast_values = {
            "diagonal_pair_fo_minus_lifted": ff - ll,
            "estimator_fo_minus_lifted_at_fo_direction": ff - lf,
            "estimator_fo_minus_lifted_at_lifted_direction": fl - ll,
            "direction_fo_minus_lifted_at_fo_estimator": ff - fl,
            "direction_fo_minus_lifted_at_lifted_estimator": lf - ll,
            # Positive means that replacing the estimator helps more with the
            # lifted direction than with the first-order direction.
            "lifted_pair_compatibility_interaction": (fl - ll) - (ff - lf),
        }
        rng = np.random.default_rng(cfg.bootstrap_seed + 200 + gamma_index)
        sample_indices = rng.integers(
            0,
            len(pivot),
            size=(cfg.bootstrap_repeats, len(pivot)),
        )
        contrasts: dict[str, Any] = {}
        for name, values in contrast_values.items():
            paired = values.to_numpy(float)
            bootstrap_means = paired[sample_indices].mean(axis=1)
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            contrasts[name] = {
                "mean_ise_difference": float(paired.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "positive_seed_count": int(np.sum(paired > 0.0)),
                "seed_count": int(len(paired)),
            }

        cells: list[dict[str, Any]] = []
        for estimator in estimators:
            for direction in estimators:
                conditions = gamma_rows[
                    (gamma_rows["estimator"] == estimator)
                    & (gamma_rows["direction_provider"] == direction)
                ]
                seeds = gamma_seed_cells[
                    (gamma_seed_cells["estimator"] == estimator)
                    & (gamma_seed_cells["direction_provider"] == direction)
                ]
                outside = conditions[
                    "residual_displacement_over_clean_pairwise_max"
                ].to_numpy(float)
                cells.append({
                    "estimator": estimator,
                    "direction_provider": direction,
                    "condition_count": int(len(conditions)),
                    "seed_count": int(len(seeds)),
                    "mean_ise": float(
                        seeds["actual_integrated_squared_error"].mean()
                    ),
                    "mean_observer_gain": float(seeds["observer_gain"].mean()),
                    "nonpositive_observer_gain_seed_count": int(
                        np.sum(seeds["observer_gain"] <= 0.0)
                    ),
                    "unstable_observer_pole_seed_count": int(
                        np.sum(np.abs(seeds["observer_pole"]) >= 1.0)
                    ),
                    "would_clip_condition_count": int(
                        np.sum(conditions["would_clip_fraction"] > 0.0)
                    ),
                    "mean_would_clip_step_fraction": float(
                        conditions["would_clip_fraction"].mean()
                    ),
                    "max_abs_action": float(conditions["max_abs_action"].max()),
                    "displacement_exceeds_clean_pairwise_max_condition_count": int(
                        np.sum(outside > 1.0)
                    ),
                    "displacement_exceeds_clean_pairwise_max_fraction": float(
                        np.mean(outside > 1.0)
                    ),
                })

        crossed = [
            cell for cell in cells
            if cell["estimator"] != cell["direction_provider"]
        ]
        crossed_checks = {
            "all_seed_mean_observer_gains_positive": all(
                cell["nonpositive_observer_gain_seed_count"] == 0
                for cell in crossed
            ),
            "all_seed_mean_observer_poles_stable": all(
                cell["unstable_observer_pole_seed_count"] == 0
                for cell in crossed
            ),
            "no_crossed_condition_would_clip": all(
                cell["would_clip_condition_count"] == 0 for cell in crossed
            ),
            "no_crossed_displacement_exceeds_clean_pairwise_max": all(
                cell["displacement_exceeds_clean_pairwise_max_condition_count"] == 0
                for cell in crossed
            ),
        }
        crossed_pass = bool(all(crossed_checks.values()))
        by_gamma[str(float(gamma))] = {
            "cells": cells,
            "contrasts": contrasts,
            "crossed_arm_checks": crossed_checks,
            "crossed_arms_pass_conditioning_and_clean_scale_diagnostics": crossed_pass,
            "attribution_status": (
                "bounded_factorial_interpretation_available"
                if crossed_pass
                else "crossed_arms_fail_diagnostics_diagonal_result_remains_system_level"
            ),
        }

    return {
        "status": "secondary_analysis_of_factorial_cells_frozen_before_outcomes",
        "uncertainty_unit": "training_seed",
        "contrast_sign": (
            "Positive first-order-minus-lifted contrasts favor lifted. "
            "Positive compatibility interaction means the lifted estimator helps "
            "more with the lifted direction than with the first-order direction."
        ),
        "clean_scale_limitation": (
            "The clean envelope contains only the four frozen task archetypes; "
            "exceeding it is a scale warning, not proof of leaving the activation manifold."
        ),
        "by_gamma": by_gamma,
    }


def analyze_suffix_results(
    controlled: pd.DataFrame,
    natural: pd.DataFrame,
    failures: pd.DataFrame,
    cfg: NonlinearSuffixConfig,
) -> dict[str, Any]:
    cv, cv_mae = _leave_one_seed_out(controlled)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    bootstrap = {
        name: _bootstrap_seed_difference(cv, name, cfg.bootstrap_repeats, rng)
        for name in ("pole_only", "bias_only")
    }
    reductions = {
        name: float((cv_mae[name] - cv_mae["full_certificate"]) / max(cv_mae[name], 1e-12))
        for name in ("pole_only", "bias_only")
    }
    pivot = cv.pivot(index="held_out_seed", columns="predictor", values="mae")
    seeds_improved = int(np.sum(
        (pivot["full_certificate"] < pivot["pole_only"])
        & (pivot["full_certificate"] < pivot["bias_only"])
    ))
    cert = controlled["certificate_final_error"].to_numpy(float)
    actual = controlled["actual_final_error"].to_numpy(float)
    correlation = float(np.corrcoef(cert, actual)[0, 1])
    sign_agreement = float(np.mean(np.signbit(cert) == np.signbit(actual)))
    raw_residual_mae = float(np.mean(np.abs(cert - actual)))
    residual_fraction = float(raw_residual_mae / max(np.mean(np.abs(actual)), 1e-12))

    natural_diag = natural[natural["estimator"] == natural["direction_provider"]]
    natural_means = natural_diag.groupby(["seed", "gamma", "estimator"], as_index=False)[
        ["actual_integrated_squared_error", "shadow_integrated_squared_error"]
    ].mean()
    ranking_by_gamma: dict[str, dict[str, Any]] = {}
    for gamma_index, (gamma, group) in enumerate(natural_means.groupby("gamma")):
        actual_pivot = group.pivot(index="seed", columns="estimator", values="actual_integrated_squared_error")
        shadow_pivot = group.pivot(index="seed", columns="estimator", values="shadow_integrated_squared_error")
        actual_diff = actual_pivot["first_order"] - actual_pivot["lifted_interaction"]
        shadow_diff = shadow_pivot["first_order"] - shadow_pivot["lifted_interaction"]
        agrees = np.signbit(actual_diff.to_numpy()) == np.signbit(shadow_diff.to_numpy())
        natural_rng = np.random.default_rng(cfg.bootstrap_seed + 100 + gamma_index)
        paired = actual_diff.to_numpy(float)
        samples = natural_rng.integers(0, len(paired), size=(cfg.bootstrap_repeats, len(paired)))
        bootstrap_means = np.mean(paired[samples], axis=1)
        interval = np.quantile(bootstrap_means, [0.025, 0.975])
        first_order_mean = float(actual_pivot["first_order"].mean())
        ranking_by_gamma[str(float(gamma))] = {
            "seeds_ranked_correctly": int(np.sum(agrees)),
            "seed_count": int(len(agrees)),
            "seeds_lifted_ise_lower": int(np.sum(actual_diff > 0.0)),
            "mean_actual_fo_minus_lifted_ise": float(actual_diff.mean()),
            "mean_actual_fo_minus_lifted_ise_ci95_low": float(interval[0]),
            "mean_actual_fo_minus_lifted_ise_ci95_high": float(interval[1]),
            "pooled_lifted_ise_reduction_fraction": float(
                actual_diff.mean() / max(first_order_mean, 1e-12)
            ),
            "mean_shadow_fo_minus_lifted_ise": float(shadow_diff.mean()),
        }

    edge = controlled[np.isclose(np.abs(controlled["relative_offset"]), 0.5)]
    nonlinear_by_seed = edge.groupby("seed").agg(
        suffix_shadow_gap=("final_suffix_shadow_abs_gap", "mean"),
        suffix_move=("final_suffix_move_abs", "mean"),
        final_error_gap=("certificate_residual", lambda values: float(np.mean(np.abs(values)))),
        final_error=("actual_final_error", lambda values: float(np.mean(np.abs(values)))),
    )
    nonlinear_by_seed["suffix_nonlinearity_ratio"] = (
        nonlinear_by_seed["suffix_shadow_gap"] / nonlinear_by_seed["suffix_move"].clip(lower=1e-12)
    )
    nonlinear_by_seed["final_error_nonlinearity_ratio"] = (
        nonlinear_by_seed["final_error_gap"] / nonlinear_by_seed["final_error"].clip(lower=1e-12)
    )
    nonlinear_seed_count = int(np.sum(nonlinear_by_seed["suffix_nonlinearity_ratio"] >= 0.05))
    final_error_nonlinearity_ratio = float(
        edge["certificate_residual"].abs().mean()
        / max(edge["actual_final_error"].abs().mean(), 1e-12)
    )
    clip_fraction = float(controlled["would_clip_fraction"].mean())
    displacement_diagnostic = _residual_displacement_diagnostic(controlled, natural)
    primary_gamma_key = str(float(cfg.gamma))
    ranking_primary = ranking_by_gamma.get(primary_gamma_key, {"seeds_ranked_correctly": 0})
    gates = {
        "full_certificate_reduction_vs_pole_at_least_20pct": reductions["pole_only"] >= 0.20,
        "full_certificate_reduction_vs_bias_at_least_20pct": reductions["bias_only"] >= 0.20,
        "full_certificate_improves_at_least_10_of_12_seeds": seeds_improved >= 10,
        "seed_bootstrap_ci_above_zero_vs_pole": bootstrap["pole_only"]["ci95_low"] > 0.0,
        "seed_bootstrap_ci_above_zero_vs_bias": bootstrap["bias_only"]["ci95_low"] > 0.0,
        "certificate_correlation_at_least_0_8": correlation >= 0.80,
        "certificate_sign_agreement_at_least_0_8": sign_agreement >= 0.80,
        "certificate_residual_at_most_25pct_actual": residual_fraction <= 0.25,
        "natural_ranking_at_least_10_of_12_seeds": int(ranking_primary["seeds_ranked_correctly"]) >= 10,
        "suffix_nonlinearity_at_least_5pct_in_8_of_12_seeds": nonlinear_seed_count >= 8,
        "final_error_nonlinearity_at_least_5pct": final_error_nonlinearity_ratio >= 0.05,
        "diagnostic_clipping_below_0_5pct": clip_fraction <= 0.005,
        "no_manipulability_failures": failures.empty,
    }
    return {
        "schema": "observerbench.nonlinear_suffix_analysis.v3",
        "controlled_condition_count": int(len(controlled)),
        "natural_condition_count": int(len(natural)),
        "manipulability_failure_count": int(len(failures)),
        "leave_one_seed_out_mae": cv_mae,
        "mae_reduction_fraction": reductions,
        "seeds_full_certificate_beats_both": seeds_improved,
        "seed_bootstrap_mae_advantage": bootstrap,
        "certificate_actual_correlation": correlation,
        "certificate_sign_agreement": sign_agreement,
        "certificate_raw_residual_mae": raw_residual_mae,
        "certificate_residual_fraction_of_actual": residual_fraction,
        "natural_diagonal_ranking": ranking_by_gamma,
        "natural_factorial_secondary": _natural_factorial_analysis(natural, cfg),
        "nonlinearity": {
            "seeds_suffix_ratio_at_least_5pct": nonlinear_seed_count,
            "mean_suffix_nonlinearity_ratio": float(nonlinear_by_seed["suffix_nonlinearity_ratio"].mean()),
            "mean_final_error_nonlinearity_ratio": final_error_nonlinearity_ratio,
            "diagnostic_clip_fraction": clip_fraction,
        },
        "residual_displacement_diagnostic": displacement_diagnostic,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "leave_one_seed_out_rows": cv.to_dict(orient="records"),
        "nonlinearity_by_seed": nonlinear_by_seed.reset_index().to_dict(orient="records"),
    }


def run_nonlinear_suffix_experiment(
    cfg: NonlinearSuffixConfig,
    outdir: str | Path,
    *,
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train all frozen seeds, run controlled and natural loops, and persist them."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    controlled_rows: list[dict[str, Any]] = []
    natural_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    token_rows, archetypes = _tokens_for_archetypes(cfg.seq_len)

    for seed in cfg.seeds:
        for gamma in cfg.gamma_values_natural:
            run_cfg = replace(cfg, seed=int(seed), gamma=float(gamma), device="cpu")
            train, _test = generate_data(run_cfg)
            model, train_info = train_model(run_cfg, train)
            observers, train_readout = _fit_prefix_observers(model, train, run_cfg)
            archetype_prefixes = extract_prefix_residuals(
                model,
                token_rows,
                run_cfg.completed_blocks,
            )
            clean_readout = archetype_prefixes[:, -1, :]
            clean_scale = _clean_residual_distance_scale(
                clean_readout,
                run_cfg.manipulability_tolerance,
            )
            training_rows.append({
                "seed": int(seed),
                "gamma": float(gamma),
                "final_train_loss": float(train_info["final_train_loss"]),
                "device": str(train_info["device"]),
                "train_readout_rank": int(np.linalg.matrix_rank(train_readout - train_readout.mean(axis=0))),
                "clean_archetype_count": clean_scale.archetype_count,
                "clean_pairwise_distance_count": clean_scale.pair_count,
                "clean_pairwise_nonzero_distance_count": clean_scale.nonzero_pair_count,
                "clean_pairwise_distance_median": clean_scale.median,
                "clean_pairwise_distance_max": clean_scale.maximum,
                "clean_pairwise_median_degenerate": clean_scale.median_degenerate,
                "clean_pairwise_max_degenerate": clean_scale.maximum_degenerate,
            })
            natural_rows.extend(_natural_rows_for_seed(
                model,
                observers,
                archetype_prefixes,
                archetypes,
                clean_readout,
                run_cfg,
                int(seed),
                float(gamma),
            ))
            if np.isclose(gamma, cfg.gamma):
                seed_rows, seed_failures = _controlled_rows_for_seed(
                    model,
                    observers,
                    archetype_prefixes,
                    archetypes,
                    clean_readout,
                    run_cfg,
                    int(seed),
                )
                controlled_rows.extend(seed_rows)
                failures.extend(seed_failures)

    controlled = pd.DataFrame(controlled_rows)
    natural = pd.DataFrame(natural_rows)
    failure_frame = pd.DataFrame(failures, columns=[
        "seed", "archetype_index", "x1", "x2", "rho_target", "reason"
    ])
    if controlled.empty:
        raise RuntimeError("all controlled directions failed; inspect manipulability_failures.csv")
    analysis = analyze_suffix_results(controlled, natural, failure_frame, cfg)

    controlled.to_csv(out / "controlled_conditions.csv", index=False)
    natural.to_csv(out / "natural_factorial_conditions.csv", index=False)
    failure_frame.to_csv(out / "manipulability_failures.csv", index=False)
    pd.DataFrame(training_rows).to_csv(out / "training_summary.csv", index=False)
    write_json(out / "analysis.json", analysis)
    pd.DataFrame(analysis["leave_one_seed_out_rows"]).to_csv(
        out / "leave_one_seed_out_metrics.csv",
        index=False,
    )
    pd.DataFrame(analysis["nonlinearity_by_seed"]).to_csv(
        out / "nonlinearity_by_seed.csv",
        index=False,
    )
    displacement = analysis["residual_displacement_diagnostic"]
    displacement_rows = [
        {
            "condition_family": "natural_diagonal",
            **row,
        }
        for row in displacement["natural_diagonal_by_gamma_estimator"]
    ]
    displacement_rows.append({
        "condition_family": "controlled_overall",
        "gamma": float(cfg.gamma),
        **displacement["controlled_overall"],
    })
    displacement_rows.extend({
        "condition_family": "controlled_by_rho_target",
        "gamma": float(cfg.gamma),
        **row,
    } for row in displacement["controlled_by_rho_target"])
    pd.DataFrame(displacement_rows).to_csv(
        out / "residual_displacement_summary.csv",
        index=False,
    )
    write_json(out / "config.json", asdict(cfg))
    provenance = runtime_provenance(Path(__file__).resolve().parents[3])
    if protocol_path is not None:
        provenance["protocol_sha256"] = file_sha256(protocol_path)
        provenance["protocol_name"] = Path(protocol_path).name
    write_json(out / "provenance.json", provenance)
    return analysis
