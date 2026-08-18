"""Closed-loop trained-transformer Ctl-2 task for ObserverBench.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

Ctl-1 measured one-shot target/collateral geometry: fit an observer, form an
observer-derived residual direction, apply one proportional intervention, and
score target movement and collateral. Ctl-2 closes the loop. The controller
repeatedly reads the observer estimate on the edited residual state, applies the
same proportional law, and accumulates target error, actuation energy, and
collateral over time.

This task is intentionally small. It is not a new controller or a transformer
plant: the loop adds a control vector directly to the model's final residual and
then evaluates affine readouts.  The Phase-1 design factors the state estimator
from the actuation direction so each source of error can be tested while the
other is held fixed.

v7 additions:
  * an oracle_target arm, used as a gain/controller sanity check;
  * divergence criteria based on final-vs-initial error growth, not a stale
    absolute threshold alone;
  * per-example, per-step trajectories and fan plots, so the closed-loop result
    is visible as a trajectory rather than only as unbounded ratios.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None

from observerbench.core import ObserverResult, write_json
from observerbench.cards import results_to_dataframe, write_cards
from observerbench.control import (
    AffineStateEstimator,
    AffineSupport,
    direction_support_metrics,
    fit_affine_support,
    loop_gain_diagnostics,
    positive_gain_matched_controller_gain,
    project_direction_to_target_gain,
)
from observerbench.metrics import mse, mae, r2_score
from observerbench.provenance import file_sha256, json_sha256, runtime_provenance
from observerbench.tasks.trained_ctl1 import (
    TrainedTransformerCtl1Config,
    generate_data,
    make_data,
    train_model,
    extract_hidden,
    _fit_ridge,
    _predict_ridge,
    normalize_direction,
)


@dataclass
class TrainedTransformerCtl2Config(TrainedTransformerCtl1Config):
    # Closed-loop-specific settings.
    loop_steps: int = 15
    controller_gain: float = 0.35
    max_strength: float = 1.0
    target_tolerance: float = 0.10

    # Old absolute threshold kept for backward compatibility, but the primary
    # divergence flag below uses a bounded, growth-based definition.
    divergence_threshold: float = 4.0
    divergence_abs_error_threshold: float = 2.0
    divergence_initial_error_multiplier: float = 2.0
    divergence_final_mse_multiplier: float = 4.0

    # Include oracle arm by default. It uses the true target readout as the
    # observer estimate and the true target-head direction as the actuation
    # direction. It answers the control-reviewer question: at this gain, does an
    # accurate observer/controller stay stable?
    include_oracle: bool = True

    # If True, use target_ref from config. If False, set a per-example target
    # offset above the model's initial output. The default fixed target mirrors
    # Ctl-1 and is easier to interpret.
    use_relative_target: bool = False
    relative_target_offset: float = 0.75

    # Phase-1 arm design.  The default preserves the frozen v7 diagonal
    # comparison.  Reviewer-facing runs use ``factorial_3x3`` and report the
    # first-order/lifted 2x2 as the primary causal comparison.
    arm_design: str = "diagonal"  # diagonal | factorial_2x2 | factorial_3x3
    direction_support_mode: str = "unprojected"  # unprojected | projected | both
    # Level calibration fits target levels on held-out perturbed states. Response
    # calibration instead fits within-baseline finite deltas and preserves each
    # test trajectory's initial estimate.
    include_affine_calibration: bool = False
    include_response_gain_calibration: bool = False
    include_gain_matched_control: bool = False

    # Residual-support and held-out calibration settings.
    support_relative_tolerance: float = 1e-6
    convex_hull_tolerance: float = 1e-5
    n_calibration: int = 512
    calibration_strength: float = 0.50
    calibration_points: int = 5

    # Full per-example trajectories are useful for one diagnostic run but too
    # large for seed/gamma sweeps. Aggregate and four-state summaries are always
    # written.
    write_per_example_outputs: bool = True
    write_per_step_examples: bool = True
    write_observer_cards: bool = True
    write_plots: bool = True


@dataclass(frozen=True)
class Ctl2Arm:
    """One independently composed estimator--direction--controller arm."""

    estimator_name: str
    direction_name: str
    support_mode: str = "unprojected"
    estimator_calibration: str = "none"  # none | level_affine | response_gain
    controller_mode: str = "base"  # base | gain_matched

    @property
    def arm_id(self) -> str:
        return (
            f"est-{self.estimator_name}__dir-{self.direction_name}"
            f"__support-{self.support_mode}__cal-{self.estimator_calibration}"
            f"__controller-{self.controller_mode}"
        )


def _linear_from_probe(h: np.ndarray, probe: np.ndarray) -> np.ndarray:
    return probe[0] + h @ probe[1:]


def _probe_estimator(
    name: str,
    coef: np.ndarray,
    probes: tuple[np.ndarray, ...],
) -> AffineStateEstimator:
    """Compose a ridge observer and its latent probes into residual space."""

    weights = np.asarray(coef[1:], dtype=float)
    if len(weights) != len(probes):
        raise ValueError("observer coefficients and probes do not match")
    intercept = float(coef[0] + sum(weight * probe[0] for weight, probe in zip(weights, probes)))
    gradient = np.sum(
        np.stack([weight * np.asarray(probe[1:], dtype=float) for weight, probe in zip(weights, probes)]),
        axis=0,
    )
    return AffineStateEstimator(name=name, intercept=intercept, gradient=gradient)


def _observer_estimate(
    h: np.ndarray,
    observer_name: str,
    coef: np.ndarray | None,
    probe_x1: np.ndarray,
    probe_x2: np.ndarray,
    probe_int: np.ndarray,
    target_vec: np.ndarray,
    target_bias: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    x1h = _linear_from_probe(h, probe_x1)
    x2h = _linear_from_probe(h, probe_x2)
    ih = _linear_from_probe(h, probe_int)
    if observer_name == "first_order":
        assert coef is not None
        zhat = coef[0] + coef[1] * x1h + coef[2] * x2h
    elif observer_name == "lifted_interaction":
        assert coef is not None
        zhat = coef[0] + coef[1] * x1h + coef[2] * x2h + coef[3] * ih
    elif observer_name == "oracle_target":
        zhat = target_bias + h @ target_vec
    else:
        raise ValueError(f"unknown observer {observer_name}")
    return zhat, {"x1_hat": x1h, "x2_hat": x2h, "interaction_hat": ih}


def _support_modes(cfg: TrainedTransformerCtl2Config) -> tuple[str, ...]:
    if cfg.direction_support_mode == "both":
        return ("unprojected", "projected")
    if cfg.direction_support_mode not in {"unprojected", "projected"}:
        raise ValueError("direction_support_mode must be unprojected, projected, or both")
    return (cfg.direction_support_mode,)


def _base_arm_pairs(cfg: TrainedTransformerCtl2Config) -> list[tuple[str, str]]:
    if cfg.arm_design == "diagonal":
        names = ["first_order", "lifted_interaction"]
        if cfg.include_oracle:
            names.append("oracle_target")
        return [(name, name) for name in names]
    if cfg.arm_design == "factorial_2x2":
        names = ["first_order", "lifted_interaction"]
        return [(estimator, direction) for estimator in names for direction in names]
    if cfg.arm_design == "factorial_3x3":
        if not cfg.include_oracle:
            raise ValueError("factorial_3x3 requires include_oracle=True")
        names = ["first_order", "lifted_interaction", "oracle_target"]
        return [(estimator, direction) for estimator in names for direction in names]
    raise ValueError("arm_design must be diagonal, factorial_2x2, or factorial_3x3")


def enumerate_ctl2_arms(cfg: TrainedTransformerCtl2Config) -> list[Ctl2Arm]:
    """Enumerate primary and calibration arms deterministically."""

    primary = [
        Ctl2Arm(estimator, direction, support_mode)
        for support_mode in _support_modes(cfg)
        for estimator, direction in _base_arm_pairs(cfg)
    ]
    arms = list(primary)
    if cfg.include_affine_calibration:
        arms.extend(
            Ctl2Arm(
                arm.estimator_name,
                arm.direction_name,
                arm.support_mode,
                estimator_calibration="level_affine",
            )
            for arm in primary
        )
    if cfg.include_response_gain_calibration:
        arms.extend(
            Ctl2Arm(
                arm.estimator_name,
                arm.direction_name,
                arm.support_mode,
                estimator_calibration="response_gain",
            )
            for arm in primary
        )
    if cfg.include_gain_matched_control:
        arms.extend(
            Ctl2Arm(
                arm.estimator_name,
                arm.direction_name,
                arm.support_mode,
                controller_mode="gain_matched",
            )
            for arm in primary
        )
    return arms


def _settling_steps(abs_err: np.ndarray, tol: float) -> np.ndarray:
    """First step from which all remaining errors are below tol; nan if never."""
    n, _t = abs_err.shape
    out = np.full(n, np.nan)
    ok = abs_err <= tol
    suffix_ok = np.flip(np.cumprod(np.flip(ok.astype(int), axis=1), axis=1), axis=1).astype(bool)
    for i in range(n):
        hits = np.where(suffix_ok[i])[0]
        if len(hits):
            out[i] = float(hits[0])
    return out


def _prepare_observers(cfg: TrainedTransformerCtl2Config):
    train, test = generate_data(cfg)
    calibration_rng = np.random.default_rng(cfg.seed + 1_000_003)
    calibration = make_data(max(1, int(cfg.n_calibration)), cfg, calibration_rng)
    model, train_info = train_model(cfg, train)
    train_pred, _train_feats, h_train = extract_hidden(model, train, cfg)
    test_pred, _test_feats, h_test = extract_hidden(model, test, cfg)
    _calibration_pred, _calibration_feats, h_calibration = extract_hidden(model, calibration, cfg)

    # Residual probes for latent variables and nuisance.
    probe_x1 = _fit_ridge(h_train, train["x1"], cfg.ridge)
    probe_x2 = _fit_ridge(h_train, train["x2"], cfg.ridge)
    probe_int = _fit_ridge(h_train, train["interaction"], cfg.ridge)
    probe_nuis = _fit_ridge(h_train, train["nuisance"], cfg.ridge)

    x1h_tr = _predict_ridge(h_train, probe_x1)
    x2h_tr = _predict_ridge(h_train, probe_x2)
    ih_tr = _predict_ridge(h_train, probe_int)
    x1h_te = _predict_ridge(h_test, probe_x1)
    x2h_te = _predict_ridge(h_test, probe_x2)
    ih_te = _predict_ridge(h_test, probe_int)

    fo_coef = _fit_ridge(np.c_[x1h_tr, x2h_tr], train["target_clean"], cfg.ridge)
    li_coef = _fit_ridge(np.c_[x1h_tr, x2h_tr, ih_tr], train["target_clean"], cfg.ridge)

    target_vec = model.target_head.weight.detach().cpu().numpy().reshape(-1)
    target_bias = float(model.target_head.bias.detach().cpu().numpy().reshape(-1)[0])
    nuisance_vec = probe_nuis[1:]
    nuisance_bias = float(probe_nuis[0])

    estimators = {
        "first_order": _probe_estimator("first_order", fo_coef, (probe_x1, probe_x2)),
        "lifted_interaction": _probe_estimator(
            "lifted_interaction",
            li_coef,
            (probe_x1, probe_x2, probe_int),
        ),
    }
    if cfg.include_oracle:
        estimators["oracle_target"] = AffineStateEstimator(
            "oracle_target",
            target_bias,
            target_vec,
        )

    support = fit_affine_support(
        h_train,
        labels=np.c_[train["x1"], train["x2"]],
        relative_tolerance=cfg.support_relative_tolerance,
    )
    clean_keys = np.c_[train["x1"], train["x2"]]
    _unique_clean_keys, clean_inverse = np.unique(clean_keys, axis=0, return_inverse=True)
    clean_residual_centroids = np.stack([
        h_train[clean_inverse == idx].mean(axis=0)
        for idx in range(clean_inverse.max() + 1)
    ])
    clean_norms = np.linalg.norm(h_train, axis=1)
    clean_pairwise = np.linalg.norm(
        clean_residual_centroids[:, None, :] - clean_residual_centroids[None, :, :],
        axis=2,
    )
    upper = clean_pairwise[np.triu_indices(len(clean_residual_centroids), k=1)]
    augmented_centroids = np.vstack([
        clean_residual_centroids.T,
        np.ones(len(clean_residual_centroids), dtype=float),
    ])
    barycentric_map = np.linalg.pinv(augmented_centroids)

    observers = {
        "first_order": {"coef": fo_coef, "estimator": estimators["first_order"]},
        "lifted_interaction": {"coef": li_coef, "estimator": estimators["lifted_interaction"]},
    }
    if cfg.include_oracle:
        observers["oracle_target"] = {"coef": None, "estimator": estimators["oracle_target"]}

    for name, obs in observers.items():
        # The residual-space gradient of the affine estimator is the raw
        # observer-derived direction used in v7.
        obs["raw_direction"] = obs["estimator"].gradient.copy()
        direction, raw_target_gain, direction_scale, norm_diag = normalize_direction(obs["raw_direction"], target_vec, cfg)
        projected_direction = None
        projected_scale = float("nan")
        projected_error = ""
        try:
            projected_direction, projected_scale = project_direction_to_target_gain(
                direction,
                support=support,
                target_gradient=target_vec,
                target_gain=cfg.target_actuation_gain,
            )
        except ValueError as exc:
            projected_error = str(exc)
        obs.update({
            "direction": direction,
            "directions": {
                "unprojected": direction,
                "projected": projected_direction,
            },
            "raw_target_gain": raw_target_gain,
            "direction_scale": direction_scale,
            "norm_diag": norm_diag,
            "projected_direction_scale": projected_scale,
            "projected_direction_error": projected_error,
        })

        # Pin the composition: the public affine estimator must reproduce the
        # task's legacy observer calculation before directions are crossed.
        legacy, _ = _observer_estimate(
            h_test,
            name,
            obs["coef"],
            probe_x1,
            probe_x2,
            probe_int,
            target_vec,
            target_bias,
        )
        if not np.allclose(obs["estimator"].estimate_batch(h_test), legacy, rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"composed affine estimator does not reproduce {name}")

    env = {
        "train": train,
        "test": test,
        "calibration": calibration,
        "model": model,
        "train_info": train_info,
        "h_train": h_train,
        "h_test": h_test,
        "h_calibration": h_calibration,
        "test_pred": test_pred,
        "target_vec": target_vec,
        "target_bias": target_bias,
        "nuisance_vec": nuisance_vec,
        "nuisance_bias": nuisance_bias,
        "probe_x1": probe_x1,
        "probe_x2": probe_x2,
        "probe_int": probe_int,
        "probe_nuis": probe_nuis,
        "x1h_te": x1h_te,
        "x2h_te": x2h_te,
        "ih_te": ih_te,
        "observers": observers,
        "estimators": estimators,
        "support": support,
        "clean_residual_centroids": clean_residual_centroids,
        "clean_residual_norm_mean": float(np.mean(clean_norms)),
        "clean_residual_norm_std": float(np.std(clean_norms)),
        "clean_residual_norm_min": float(np.min(clean_norms)),
        "clean_residual_norm_max": float(np.max(clean_norms)),
        "clean_centroid_pairwise_rms": float(np.sqrt(np.mean(upper ** 2))) if len(upper) else 0.0,
        "barycentric_map": barycentric_map,
    }
    return env


class GainMatchIneligible(ValueError):
    """Raised when a non-positive observer self-gain cannot be gain matched."""


def _level_calibrated_estimator(
    cfg: TrainedTransformerCtl2Config,
    env: Dict,
    estimator: AffineStateEstimator,
    direction: np.ndarray,
) -> tuple[AffineStateEstimator, dict[str, float]]:
    """Fit one target-level affine correction on a separate perturbation split."""

    if cfg.calibration_points < 2:
        raise ValueError("calibration_points must be at least two")
    strengths = np.linspace(
        -float(cfg.calibration_strength),
        float(cfg.calibration_strength),
        int(cfg.calibration_points),
    )
    h0 = np.asarray(env["h_calibration"], dtype=float)
    states = (h0[:, None, :] + strengths[None, :, None] * direction[None, None, :]).reshape(-1, h0.shape[1])
    true_target = env["target_bias"] + states @ env["target_vec"]
    raw_estimate = estimator.estimate_batch(states)
    calibration_coef = _fit_ridge(raw_estimate[:, None], true_target, cfg.ridge)
    calibrated = estimator.affine_calibration(
        offset=float(calibration_coef[0]),
        scale=float(calibration_coef[1]),
        name=f"{estimator.name}_affine_calibrated",
    )
    calibrated_estimate = calibrated.estimate_batch(states)
    return calibrated, {
        "affine_calibration_offset": float(calibration_coef[0]),
        "affine_calibration_scale": float(calibration_coef[1]),
        "calibration_raw_mae": float(np.mean(np.abs(raw_estimate - true_target))),
        "calibration_affine_mae": float(np.mean(np.abs(calibrated_estimate - true_target))),
        "calibration_n_states": float(len(states)),
    }


def _response_gain_calibration(
    cfg: TrainedTransformerCtl2Config,
    env: Dict,
    estimator: AffineStateEstimator,
    direction: np.ndarray,
) -> dict[str, float]:
    """Fit a scalar finite-response gain from within-baseline deltas.

    Clean-baseline levels cancel before fitting. During a rollout the corrected
    response is anchored to that example's original estimate, so this diagnostic
    changes response gain without pretending to repair initial estimation bias.
    """

    if cfg.calibration_points < 2:
        raise ValueError("calibration_points must be at least two")
    strengths = np.linspace(
        -float(cfg.calibration_strength),
        float(cfg.calibration_strength),
        int(cfg.calibration_points),
    )
    h0 = np.asarray(env["h_calibration"], dtype=float)
    states = h0[:, None, :] + strengths[None, :, None] * direction[None, None, :]
    flat_states = states.reshape(-1, h0.shape[1])
    raw_level = estimator.estimate_batch(flat_states).reshape(len(h0), len(strengths))
    true_level = (env["target_bias"] + flat_states @ env["target_vec"]).reshape(
        len(h0), len(strengths)
    )
    raw_baseline = estimator.estimate_batch(h0)[:, None]
    true_baseline = (env["target_bias"] + h0 @ env["target_vec"])[:, None]
    raw_delta = raw_level - raw_baseline
    true_delta = true_level - true_baseline
    denominator = float(np.sum(raw_delta ** 2))
    if denominator <= 1e-18:
        raise ValueError("observer has no measurable response along the calibration direction")
    scale = float(np.sum(raw_delta * true_delta) / denominator)
    calibrated_delta = scale * raw_delta
    return {
        "response_calibration_scale": scale,
        "response_calibration_raw_mae": float(np.mean(np.abs(raw_delta - true_delta))),
        "response_calibration_corrected_mae": float(np.mean(np.abs(calibrated_delta - true_delta))),
        "response_calibration_raw_gain": float(estimator.gradient @ direction),
        "response_calibration_target_gain": float(env["target_vec"] @ direction),
        "calibration_n_states": float(raw_delta.size),
    }


def _arm_label(cfg: TrainedTransformerCtl2Config, arm: Ctl2Arm) -> str:
    if (
        cfg.arm_design == "diagonal"
        and arm.estimator_name == arm.direction_name
        and arm.support_mode == "unprojected"
        and arm.estimator_calibration == "none"
        and arm.controller_mode == "base"
    ):
        return arm.estimator_name
    return arm.arm_id


def _resolve_arm(
    cfg: TrainedTransformerCtl2Config,
    env: Dict,
    arm: Ctl2Arm,
) -> tuple[AffineStateEstimator, np.ndarray, float, float, dict[str, float | bool]]:
    if arm.estimator_name not in env["estimators"]:
        raise KeyError(f"unknown estimator {arm.estimator_name}")
    if arm.direction_name not in env["observers"]:
        raise KeyError(f"unknown direction {arm.direction_name}")

    estimator = env["estimators"][arm.estimator_name]
    direction = env["observers"][arm.direction_name]["directions"].get(arm.support_mode)
    if direction is None:
        reason = env["observers"][arm.direction_name].get("projected_direction_error", "direction unavailable")
        raise ValueError(f"{arm.direction_name}/{arm.support_mode}: {reason}")
    direction = np.asarray(direction, dtype=float)

    calibration_metrics: dict[str, float | bool] = {
        "affine_calibration_offset": 0.0,
        "affine_calibration_scale": 1.0,
        "calibration_raw_mae": float("nan"),
        "calibration_affine_mae": float("nan"),
        "response_calibration_scale": 1.0,
        "response_calibration_raw_mae": float("nan"),
        "response_calibration_corrected_mae": float("nan"),
        "response_calibration_raw_gain": float("nan"),
        "response_calibration_target_gain": float("nan"),
        "calibration_n_states": 0.0,
    }
    response_scale = 1.0
    if arm.estimator_calibration == "level_affine":
        estimator, fitted = _level_calibrated_estimator(cfg, env, estimator, direction)
        calibration_metrics.update(fitted)
    elif arm.estimator_calibration == "response_gain":
        fitted = _response_gain_calibration(cfg, env, estimator, direction)
        calibration_metrics.update(fitted)
        response_scale = float(fitted["response_calibration_scale"])
    elif arm.estimator_calibration != "none":
        raise ValueError(f"unknown estimator calibration {arm.estimator_calibration}")

    effective_estimator_gradient = response_scale * estimator.gradient
    base_gain = loop_gain_diagnostics(
        env["target_vec"],
        effective_estimator_gradient,
        direction,
        cfg.controller_gain,
    )
    controller_gain = float(cfg.controller_gain)
    gain_match_eligible = bool(base_gain.observer_self_gain > 1e-12)
    if arm.controller_mode == "gain_matched":
        matched = positive_gain_matched_controller_gain(
            base_controller_gain=cfg.controller_gain,
            reference_self_gain=cfg.target_actuation_gain,
            arm_self_gain=base_gain.observer_self_gain,
        )
        if matched is None:
            raise GainMatchIneligible(
                f"{arm.arm_id}: observer self-gain {base_gain.observer_self_gain:.6g} is non-positive"
            )
        controller_gain = matched
    elif arm.controller_mode != "base":
        raise ValueError(f"unknown controller mode {arm.controller_mode}")

    calibration_metrics.update({
        "gain_match_eligible": gain_match_eligible,
        "base_controller_gain": float(cfg.controller_gain),
        "effective_controller_gain": float(controller_gain),
        "controller_gain_ratio_vs_base": float(controller_gain / max(abs(cfg.controller_gain), 1e-12)),
    })
    return estimator, direction, response_scale, controller_gain, calibration_metrics


def _rollout_arm(
    cfg: TrainedTransformerCtl2Config,
    env: Dict,
    arm: Ctl2Arm,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimator, direction, response_scale, controller_gain, arm_calibration = _resolve_arm(cfg, env, arm)
    arm_id = _arm_label(cfg, arm)
    direction_record = env["observers"][arm.direction_name]
    h0 = env["h_test"].copy()
    h = h0.copy()
    n = h.shape[0]
    target_vec = env["target_vec"]
    target_bias = env["target_bias"]
    nuisance_vec = env["nuisance_vec"]
    nuisance_bias = env["nuisance_bias"]
    support: AffineSupport = env["support"]
    clean_centroids = np.asarray(env["clean_residual_centroids"], dtype=float)
    clean_centroid_norms = np.linalg.norm(clean_centroids, axis=1)
    clean_norm_mean = float(env.get("clean_residual_norm_mean", np.mean(clean_centroid_norms)))
    clean_norm_std_value = float(env.get("clean_residual_norm_std", np.std(clean_centroid_norms)))
    clean_norm_min = float(env.get("clean_residual_norm_min", np.min(clean_centroid_norms)))
    clean_norm_max = float(env.get("clean_residual_norm_max", np.max(clean_centroid_norms)))
    clean_pairwise = np.linalg.norm(
        clean_centroids[:, None, :] - clean_centroids[None, :, :],
        axis=2,
    )
    clean_pairwise_upper = clean_pairwise[np.triu_indices(len(clean_centroids), k=1)]
    clean_centroid_pairwise_rms = float(env.get(
        "clean_centroid_pairwise_rms",
        np.sqrt(np.mean(clean_pairwise_upper ** 2)) if len(clean_pairwise_upper) else 0.0,
    ))
    barycentric_map = np.asarray(env.get(
        "barycentric_map",
        np.linalg.pinv(np.vstack([clean_centroids.T, np.ones(len(clean_centroids))])),
    ))

    gain_diagnostics = loop_gain_diagnostics(
        target_vec,
        response_scale * estimator.gradient,
        direction,
        controller_gain,
    )
    support_metrics = direction_support_metrics(direction, support)

    target_initial = target_bias + h @ target_vec
    nuisance_initial = nuisance_bias + h @ nuisance_vec
    if cfg.use_relative_target:
        target_ref = target_initial + cfg.relative_target_offset
    else:
        target_ref = np.full(n, cfg.target_ref, dtype=float)

    rows = []
    trajectory_example_rows = []
    strength_history = []
    raw_strength_history = []
    clip_history = []
    h_history_targets = []
    h_history_collateral = []
    h_history_obs = []
    h_history_off_support = []
    h_history_delta_from_initial = []
    h_history_nearest_clean = []
    h_history_norm_z = []
    h_history_convex_extrapolation = []
    previous_recorded_h = None
    response_anchor = None

    for t in range(cfg.loop_steps + 1):
        target_t = target_bias + h @ target_vec
        nuisance_t = nuisance_bias + h @ nuisance_vec
        raw_zhat_t = estimator.estimate_batch(h)
        if response_anchor is None:
            response_anchor = raw_zhat_t.copy()
        if arm.estimator_calibration == "response_gain":
            zhat_t = response_anchor + response_scale * (raw_zhat_t - response_anchor)
        else:
            zhat_t = raw_zhat_t
        err_true = target_ref - target_t
        err_obs = target_ref - zhat_t
        collateral_delta = nuisance_t - nuisance_initial
        if t < cfg.loop_steps:
            raw_next_strength = controller_gain * err_obs
            next_strength = np.clip(raw_next_strength, -cfg.max_strength, cfg.max_strength)
            next_clip_active = np.abs(raw_next_strength - next_strength) > 1e-12
        else:
            raw_next_strength = np.zeros(n, dtype=float)
            next_strength = np.zeros(n, dtype=float)
            next_clip_active = np.zeros(n, dtype=bool)
        if previous_recorded_h is None:
            state_step_delta = np.zeros(n, dtype=float)
        else:
            state_step_delta = np.linalg.norm(h - previous_recorded_h, axis=1)
        state_delta_from_initial = np.linalg.norm(h - h0, axis=1)
        off_support_distance = np.asarray(support.off_support_distance(h), dtype=float)
        normalized_off_support = np.asarray(support.normalized_off_support_distance(h), dtype=float)
        centered_state_norm = np.linalg.norm(h - support.center, axis=1)
        nearest_clean_distance = np.min(np.linalg.norm(h[:, None, :] - clean_centroids[None, :, :], axis=2), axis=1)
        residual_norm = np.linalg.norm(h, axis=1)
        clean_norm_std = max(clean_norm_std_value, 1e-12)
        residual_norm_z = (residual_norm - clean_norm_mean) / clean_norm_std
        augmented_states = np.c_[h, np.ones(n, dtype=float)]
        barycentric_weights = augmented_states @ barycentric_map.T
        convex_extrapolation = np.sum(np.maximum(-barycentric_weights, 0.0), axis=1)
        previous_recorded_h = h.copy()
        h_history_targets.append(target_t.copy())
        h_history_collateral.append(collateral_delta.copy())
        h_history_obs.append(zhat_t.copy())
        h_history_off_support.append(off_support_distance.copy())
        h_history_delta_from_initial.append(state_delta_from_initial.copy())
        h_history_nearest_clean.append(nearest_clean_distance.copy())
        h_history_norm_z.append(residual_norm_z.copy())
        h_history_convex_extrapolation.append(convex_extrapolation.copy())
        rows.append({
            "observer": arm_id,
            "estimator": arm.estimator_name,
            "direction_provider": arm.direction_name,
            "support_mode": arm.support_mode,
            "estimator_calibration": arm.estimator_calibration,
            "controller_mode": arm.controller_mode,
            "step": t,
            "target_mse": float(np.mean(err_true ** 2)),
            "target_mae": float(np.mean(np.abs(err_true))),
            "observer_error_mae": float(np.mean(np.abs(zhat_t - target_t))),
            "collateral_abs_delta": float(np.mean(np.abs(collateral_delta))),
            "collateral_rms_delta": float(np.sqrt(np.mean(collateral_delta ** 2))),
            "mean_observer_estimate": float(np.mean(zhat_t)),
            "mean_true_target": float(np.mean(target_t)),
            "mean_target_ref": float(np.mean(target_ref)),
            "mean_residual_off_support_l2": float(np.mean(off_support_distance)),
            "mean_residual_off_support_normalized": float(np.mean(normalized_off_support)),
            "mean_nearest_clean_residual_l2": float(np.mean(nearest_clean_distance)),
            "mean_residual_delta_from_initial_l2": float(np.mean(state_delta_from_initial)),
            "mean_abs_residual_norm_z": float(np.mean(np.abs(residual_norm_z))),
            "mean_convex_hull_extrapolation_mass": float(np.mean(convex_extrapolation)),
            "fraction_outside_clean_convex_hull": float(
                np.mean(convex_extrapolation > cfg.convex_hull_tolerance)
            ),
            "next_clip_fraction": float(np.mean(next_clip_active)) if t < cfg.loop_steps else float("nan"),
        })
        trajectory_example_rows.extend({
            "observer": arm_id,
            "estimator": arm.estimator_name,
            "direction_provider": arm.direction_name,
            "support_mode": arm.support_mode,
            "estimator_calibration": arm.estimator_calibration,
            "controller_mode": arm.controller_mode,
            "example_idx": int(i),
            "step": int(t),
            "target": float(target_t[i]),
            "target_ref": float(target_ref[i]),
            "target_error": float(err_true[i]),
            "target_abs_error": float(abs(err_true[i])),
            "target_squared_error": float(err_true[i] ** 2),
            "observer_estimate": float(zhat_t[i]),
            "observer_bias": float(zhat_t[i] - target_t[i]),
            "observer_abs_bias": float(abs(zhat_t[i] - target_t[i])),
            "collateral_delta": float(collateral_delta[i]),
            "collateral_abs_delta": float(abs(collateral_delta[i])),
            "residual_l2_delta_from_initial": float(state_delta_from_initial[i]),
            "residual_l2_delta_from_prev_step": float(state_step_delta[i]),
            "residual_l2_from_support_center": float(centered_state_norm[i]),
            "residual_off_support_l2": float(off_support_distance[i]),
            "residual_off_support_normalized": float(normalized_off_support[i]),
            "nearest_clean_residual_l2": float(nearest_clean_distance[i]),
            "residual_norm_z": float(residual_norm_z[i]),
            "convex_hull_extrapolation_mass": float(convex_extrapolation[i]),
            "next_raw_control_strength": float(raw_next_strength[i]),
            "next_control_strength": float(next_strength[i]),
            "next_control_clipped": bool(next_clip_active[i]),
        } for i in range(n))
        if t == cfg.loop_steps:
            break
        strength = next_strength
        strength_history.append(strength.copy())
        raw_strength_history.append(raw_next_strength.copy())
        clip_history.append(next_clip_active.copy())
        h = h + strength[:, None] * direction[None, :]

    target_mat = np.stack(h_history_targets, axis=1)
    collateral_mat = np.stack(h_history_collateral, axis=1)
    obs_mat = np.stack(h_history_obs, axis=1)
    off_support_mat = np.stack(h_history_off_support, axis=1)
    delta_from_initial_mat = np.stack(h_history_delta_from_initial, axis=1)
    nearest_clean_mat = np.stack(h_history_nearest_clean, axis=1)
    norm_z_mat = np.stack(h_history_norm_z, axis=1)
    convex_extrapolation_mat = np.stack(h_history_convex_extrapolation, axis=1)
    ref_mat = target_ref[:, None]
    err_mat = ref_mat - target_mat
    abs_err = np.abs(err_mat)
    sq_err = err_mat ** 2
    settling = _settling_steps(abs_err, cfg.target_tolerance)
    strengths = np.stack(strength_history, axis=1) if strength_history else np.zeros((n, 0))
    raw_strengths = np.stack(raw_strength_history, axis=1) if raw_strength_history else np.zeros((n, 0))
    clipped = np.stack(clip_history, axis=1) if clip_history else np.zeros((n, 0), dtype=bool)

    # Integrated metrics over the trajectory. We include the initial step in ISE
    # because a controller with a better observer should reduce this integral.
    ise = np.sum(sq_err, axis=1)
    iae = np.sum(abs_err, axis=1)
    cumulative_collateral = np.sum(np.abs(collateral_mat), axis=1)
    final_collateral = np.abs(collateral_mat[:, -1])
    total_energy = np.sum(np.abs(strengths), axis=1) * float(np.sum(np.abs(direction))) if strengths.shape[1] else np.zeros(n)
    overshoot = np.maximum(0.0, target_mat - ref_mat)
    undershoot = np.maximum(0.0, ref_mat - target_mat)

    initial_abs_error = abs_err[:, 0]
    final_abs_error = abs_err[:, -1]
    initial_sq_error = sq_err[:, 0]
    final_sq_error = sq_err[:, -1]
    old_abs_diverged = np.max(abs_err, axis=1) > cfg.divergence_threshold
    growth_diverged = (
        (final_abs_error > cfg.divergence_abs_error_threshold) &
        (final_abs_error > cfg.divergence_initial_error_multiplier * np.maximum(initial_abs_error, 1e-12))
    )
    mse_growth_diverged = final_sq_error > cfg.divergence_final_mse_multiplier * np.maximum(initial_sq_error, 1e-12)
    worsened = final_sq_error > initial_sq_error

    initial_target_mse = float(np.mean(initial_sq_error))
    final_target_mse = float(np.mean(final_sq_error))
    target_improvement_mse = initial_target_mse - final_target_mse
    direction_on = support.project_direction(direction)
    direction_off = direction - direction_on
    effective_estimator_gradient = response_scale * estimator.gradient
    estimator_on_gain = float(effective_estimator_gradient @ direction_on)
    estimator_off_gain = float(effective_estimator_gradient @ direction_off)
    target_on_gain = float(target_vec @ direction_on)
    target_off_gain = float(target_vec @ direction_off)
    unprojected_direction = np.asarray(direction_record["directions"]["unprojected"], dtype=float)
    projected_before_rescale = support.project_direction(unprojected_direction)
    clip_fraction = float(np.mean(clipped)) if clipped.size else 0.0
    examples_ever_clipped = float(np.mean(np.any(clipped, axis=1))) if clipped.size else 0.0
    clipped_command_mass_fraction = (
        float(np.sum(np.abs(raw_strengths - strengths)) / max(np.sum(np.abs(raw_strengths)), 1e-12))
        if raw_strengths.size else 0.0
    )
    predicted_final_true_error_mae = float("nan")
    affine_true_error_prediction_residual_mae = float("nan")
    if clip_fraction == 0.0 and np.isfinite(gain_diagnostics.normalized_omitted_response):
        pole_power = gain_diagnostics.observer_error_pole ** cfg.loop_steps
        initial_observer_error = target_ref - obs_mat[:, 0]
        initial_target_observer_mismatch = target_mat[:, 0] - obs_mat[:, 0]
        predicted_final_true_error = (
            pole_power * initial_observer_error
            - initial_target_observer_mismatch
            - gain_diagnostics.normalized_omitted_response
            * (1.0 - pole_power)
            * initial_observer_error
        )
        actual_final_true_error = target_ref - target_mat[:, -1]
        predicted_final_true_error_mae = float(np.mean(np.abs(predicted_final_true_error)))
        affine_true_error_prediction_residual_mae = float(
            np.mean(np.abs(predicted_final_true_error - actual_final_true_error))
        )
    metrics = {
        "arm_id": arm_id,
        "estimator_name": arm.estimator_name,
        "direction_name": arm.direction_name,
        "direction_support_mode": arm.support_mode,
        "estimator_calibration": arm.estimator_calibration,
        "controller_mode": arm.controller_mode,
        "loop_steps": float(cfg.loop_steps),
        "model_test_mse": mse(env["test_pred"], env["test"]["target"]),
        "model_test_r2": r2_score(env["test"]["target_clean"], env["test_pred"]),
        "observer_initial_r2_vs_plant_target": r2_score(target_mat[:, 0], obs_mat[:, 0]),
        "observer_initial_mae_vs_plant_target": mae(target_mat[:, 0], obs_mat[:, 0]),
        "observer_initial_r2_vs_synthetic_clean_target": r2_score(env["test"]["target_clean"], obs_mat[:, 0]),
        "observer_initial_mae_vs_synthetic_clean_target": mae(env["test"]["target_clean"], obs_mat[:, 0]),
        # Deprecated v7 aliases. These historically referred to the synthetic
        # clean label, not the actual target-head plant readout.
        "observer_initial_r2_vs_true_target": r2_score(env["test"]["target_clean"], obs_mat[:, 0]),
        "observer_initial_mae_vs_true_target": mae(env["test"]["target_clean"], obs_mat[:, 0]),
        "initial_target_mse": initial_target_mse,
        "final_target_mse": final_target_mse,
        "final_over_initial_target_mse": float(final_target_mse / max(initial_target_mse, 1e-12)),
        "target_improvement_mse": target_improvement_mse,
        "target_improvement_fraction": target_improvement_mse / max(initial_target_mse, 1e-12),
        "integrated_squared_error": float(np.mean(ise)),
        "integrated_absolute_error": float(np.mean(iae)),
        "final_target_mae": float(np.mean(final_abs_error)),
        "cumulative_collateral_abs": float(np.mean(cumulative_collateral)),
        "final_collateral_abs": float(np.mean(final_collateral)),
        "max_collateral_abs": float(np.mean(np.max(np.abs(collateral_mat), axis=1))),
        "actuation_energy_l1": float(np.mean(total_energy)),
        "mean_abs_strength": float(np.mean(np.abs(strengths))) if strengths.size else 0.0,
        "mean_strength": float(np.mean(strengths)) if strengths.size else 0.0,
        "max_abs_strength": float(np.max(np.abs(strengths))) if strengths.size else 0.0,
        "overshoot_mean": float(np.mean(np.max(overshoot, axis=1))),
        "undershoot_mean": float(np.mean(np.max(undershoot, axis=1))),
        "settled_fraction": float(np.mean(~np.isnan(settling))),
        "settling_step_mean": float(np.nanmean(settling)) if np.any(~np.isnan(settling)) else float("nan"),
        # The continuous errors above are primary.  These bounded threshold
        # flags remain secondary diagnostics for backward compatibility.
        "large_error_trajectory_rate": float(np.mean(growth_diverged)),
        "final_mse_growth_flag_rate": float(np.mean(mse_growth_diverged)),
        "divergence_rate": float(np.mean(growth_diverged)),
        "divergence_rate_mse_growth": float(np.mean(mse_growth_diverged)),
        "divergence_rate_old_abs_threshold": float(np.mean(old_abs_diverged)),
        "target_error_worsened_rate": float(np.mean(worsened)),
        "mean_observer_bias_initial": float(np.mean(obs_mat[:, 0] - target_mat[:, 0])),
        "mean_observer_bias_final": float(np.mean(obs_mat[:, -1] - target_mat[:, -1])),
        "observer_bias_mae_path": float(np.mean(np.abs(obs_mat - target_mat))),
        "effective_target_gain": gain_diagnostics.target_gain,
        "observer_self_gain": gain_diagnostics.observer_self_gain,
        "observer_to_target_gain_ratio": float(
            gain_diagnostics.observer_self_gain / max(abs(gain_diagnostics.target_gain), 1e-12)
        ),
        "mismatch_projection": gain_diagnostics.mismatch_projection,
        "omitted_response_gain": gain_diagnostics.omitted_response_gain,
        "normalized_omitted_response": gain_diagnostics.normalized_omitted_response,
        "true_to_observer_response_ratio": gain_diagnostics.true_to_observer_response_ratio,
        "true_response_direction_compatible": gain_diagnostics.true_response_direction_compatible,
        "predicted_final_true_error_mae_unsaturated_affine": predicted_final_true_error_mae,
        "affine_true_error_prediction_residual_mae": affine_true_error_prediction_residual_mae,
        "observer_error_pole_unsaturated": gain_diagnostics.observer_error_pole,
        "observer_direction_sign_compatible": gain_diagnostics.sign_compatible,
        "locally_convergent_unsaturated": gain_diagnostics.locally_convergent_unsaturated,
        "target_gain_on_support_component": target_on_gain,
        "target_gain_off_support_component": target_off_gain,
        "observer_gain_on_support_component": estimator_on_gain,
        "observer_gain_off_support_component": estimator_off_gain,
        "effective_collateral_gain": float(nuisance_vec @ direction),
        "raw_target_gain": float(direction_record["raw_target_gain"]),
        "direction_norm": float(np.linalg.norm(direction)),
        "raw_direction_norm": float(np.linalg.norm(direction_record["raw_direction"])),
        "normalization_mode": direction_record["norm_diag"].get("normalization_mode"),
        "orientation_sign": float(direction_record["norm_diag"].get("orientation_sign", float("nan"))),
        "raw_parallel_norm": float(direction_record["norm_diag"].get("raw_parallel_norm", float("nan"))),
        "raw_orthogonal_norm": float(direction_record["norm_diag"].get("raw_orthogonal_norm", float("nan"))),
        "fixed_parallel_norm": float(direction_record["norm_diag"].get("fixed_parallel_norm", float("nan"))),
        "target_head_norm": float(direction_record["norm_diag"].get("target_head_norm", float("nan"))),
        "post_normalization_target_gain_error": float(target_vec @ direction - cfg.target_actuation_gain),
        "projected_direction_scale": float(direction_record.get("projected_direction_scale", float("nan"))),
        "projected_target_gain_before_rescale": float(target_vec @ projected_before_rescale),
        "direction_norm_ratio_vs_unprojected": float(
            np.linalg.norm(direction) / max(np.linalg.norm(unprojected_direction), 1e-12)
        ),
        "residual_support_rank": float(support.rank),
        "residual_support_n_centroids": float(support.n_centroids),
        "residual_support_absolute_tolerance": float(support.absolute_tolerance),
        "residual_support_clean_rms_radius": float(support.clean_rms_radius),
        "mean_residual_off_support_l2_path": float(np.mean(off_support_mat)),
        "final_residual_off_support_l2": float(np.mean(off_support_mat[:, -1])),
        "max_residual_off_support_l2": float(np.mean(np.max(off_support_mat, axis=1))),
        "mean_residual_off_support_normalized_path": float(
            np.mean(off_support_mat) / max(support.clean_rms_radius, 1e-12)
        ),
        "mean_residual_delta_from_initial_l2_path": float(np.mean(delta_from_initial_mat)),
        "final_residual_delta_from_initial_l2": float(np.mean(delta_from_initial_mat[:, -1])),
        "max_residual_delta_from_initial_l2": float(np.mean(np.max(delta_from_initial_mat, axis=1))),
        "mean_residual_delta_from_initial_normalized_path": float(
            np.mean(delta_from_initial_mat) / max(support.clean_rms_radius, 1e-12)
        ),
        "mean_nearest_clean_residual_l2_path": float(np.mean(nearest_clean_mat)),
        "final_nearest_clean_residual_l2": float(np.mean(nearest_clean_mat[:, -1])),
        "max_nearest_clean_residual_l2": float(np.mean(np.max(nearest_clean_mat, axis=1))),
        "mean_nearest_clean_residual_normalized_path": float(
            np.mean(nearest_clean_mat) / max(support.clean_rms_radius, 1e-12)
        ),
        "mean_abs_residual_norm_z_path": float(np.mean(np.abs(norm_z_mat))),
        "max_abs_residual_norm_z": float(np.mean(np.max(np.abs(norm_z_mat), axis=1))),
        "mean_convex_hull_extrapolation_mass_path": float(np.mean(convex_extrapolation_mat)),
        "final_convex_hull_extrapolation_mass": float(np.mean(convex_extrapolation_mat[:, -1])),
        "max_convex_hull_extrapolation_mass": float(
            np.mean(np.max(convex_extrapolation_mat, axis=1))
        ),
        "outside_clean_convex_hull_step_fraction": float(
            np.mean(convex_extrapolation_mat > cfg.convex_hull_tolerance)
        ),
        "convex_hull_extrapolation_tolerance": float(cfg.convex_hull_tolerance),
        "clean_residual_norm_mean": clean_norm_mean,
        "clean_residual_norm_std": clean_norm_std_value,
        "clean_residual_norm_min": clean_norm_min,
        "clean_residual_norm_max": clean_norm_max,
        "clean_centroid_pairwise_rms": clean_centroid_pairwise_rms,
        "control_clip_fraction": clip_fraction,
        "examples_ever_clipped_fraction": examples_ever_clipped,
        "clipped_command_mass_fraction": clipped_command_mass_fraction,
        "gamma": float(cfg.gamma),
        "nuisance_interaction_weight": float(cfg.nuisance_interaction_weight),
        "controller_gain": float(controller_gain),
        "max_strength": float(cfg.max_strength),
        "target_ref_mean": float(np.mean(target_ref)),
        "divergence_abs_error_threshold": float(cfg.divergence_abs_error_threshold),
        "divergence_initial_error_multiplier": float(cfg.divergence_initial_error_multiplier),
        "divergence_final_mse_multiplier": float(cfg.divergence_final_mse_multiplier),
    }
    metrics.update(support_metrics)
    metrics.update(arm_calibration)

    # Per-example metrics for paired comparisons.
    per_example = pd.DataFrame({
        "observer": arm_id,
        "estimator": arm.estimator_name,
        "direction_provider": arm.direction_name,
        "support_mode": arm.support_mode,
        "estimator_calibration": arm.estimator_calibration,
        "controller_mode": arm.controller_mode,
        "example_idx": np.arange(n),
        "initial_target": target_mat[:, 0],
        "final_target": target_mat[:, -1],
        "target_ref": target_ref,
        "integrated_squared_error": ise,
        "integrated_absolute_error": iae,
        "initial_target_abs_error": initial_abs_error,
        "final_target_abs_error": final_abs_error,
        "initial_target_squared_error": initial_sq_error,
        "final_target_squared_error": final_sq_error,
        "final_over_initial_squared_error": final_sq_error / np.maximum(initial_sq_error, 1e-12),
        "cumulative_collateral_abs": cumulative_collateral,
        "final_collateral_abs": final_collateral,
        "actuation_energy_l1": total_energy,
        "settling_step": settling,
        "max_abs_error": np.max(abs_err, axis=1),
        "old_abs_threshold_diverged": old_abs_diverged,
        "growth_diverged": growth_diverged,
        "mse_growth_diverged": mse_growth_diverged,
        "target_error_worsened": worsened,
        "observer_bias_abs_mean": np.mean(np.abs(obs_mat - target_mat), axis=1),
        "mean_residual_off_support_l2": np.mean(off_support_mat, axis=1),
        "final_residual_off_support_l2": off_support_mat[:, -1],
        "max_residual_off_support_l2": np.max(off_support_mat, axis=1),
        "mean_residual_delta_from_initial_l2": np.mean(delta_from_initial_mat, axis=1),
        "final_residual_delta_from_initial_l2": delta_from_initial_mat[:, -1],
        "mean_nearest_clean_residual_l2": np.mean(nearest_clean_mat, axis=1),
        "final_nearest_clean_residual_l2": nearest_clean_mat[:, -1],
        "mean_abs_residual_norm_z": np.mean(np.abs(norm_z_mat), axis=1),
        "mean_convex_hull_extrapolation_mass": np.mean(convex_extrapolation_mat, axis=1),
        "final_convex_hull_extrapolation_mass": convex_extrapolation_mat[:, -1],
        "x1": np.asarray(env["test"].get("x1", np.zeros(n)), dtype=float),
        "x2": np.asarray(env["test"].get("x2", np.zeros(n)), dtype=float),
        "control_clip_fraction": np.mean(clipped, axis=1) if clipped.size else np.zeros(n),
        "ever_clipped": np.any(clipped, axis=1) if clipped.size else np.zeros(n, dtype=bool),
    })

    traj = pd.DataFrame(rows)
    traj_examples = pd.DataFrame(trajectory_example_rows)
    return metrics, traj, per_example, traj_examples


def _rollout_observer(
    cfg: TrainedTransformerCtl2Config,
    env: Dict,
    observer_name: str,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible diagonal wrapper used by v7 smoke tests."""

    return _rollout_arm(cfg, env, Ctl2Arm(observer_name, observer_name))


def _plot_fan(traj_examples: pd.DataFrame, value_col: str, outpath: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for observer_name, g in traj_examples.groupby("observer"):
        q = g.groupby("step")[value_col].quantile([0.1, 0.5, 0.9]).unstack()
        mean = g.groupby("step")[value_col].mean()
        steps = q.index.values
        ax.plot(steps, mean.values, marker="o", label=f"{observer_name} mean")
        ax.fill_between(steps, q[0.1].values, q[0.9].values, alpha=0.15)
    ax.set_xlabel("closed-loop step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _factorial_contrasts(results: List[ObserverResult]) -> pd.DataFrame:
    """Compute estimate and direction contrasts for the primary 2x2 cells."""

    primary = {
        (
            str(result.metrics["direction_support_mode"]),
            str(result.metrics["estimator_name"]),
            str(result.metrics["direction_name"]),
        ): result.metrics
        for result in results
        if result.metrics["estimator_calibration"] == "none"
        and result.metrics["controller_mode"] == "base"
        and result.metrics["estimator_name"] in {"first_order", "lifted_interaction"}
        and result.metrics["direction_name"] in {"first_order", "lifted_interaction"}
    }
    metrics = [
        "integrated_squared_error",
        "integrated_absolute_error",
        "final_target_mse",
        "cumulative_collateral_abs",
        "actuation_energy_l1",
        "observer_bias_mae_path",
        "large_error_trajectory_rate",
        "observer_self_gain",
        "observer_error_pole_unsaturated",
        "mean_residual_off_support_l2_path",
    ]
    rows: list[dict[str, float | str]] = []
    for support_mode in sorted({key[0] for key in primary}):
        cells = {
            (estimator, direction): primary.get((support_mode, estimator, direction))
            for estimator in ["first_order", "lifted_interaction"]
            for direction in ["first_order", "lifted_interaction"]
        }
        if any(value is None for value in cells.values()):
            continue
        for metric in metrics:
            ff = float(cells[("first_order", "first_order")][metric])
            lf = float(cells[("lifted_interaction", "first_order")][metric])
            fl = float(cells[("first_order", "lifted_interaction")][metric])
            ll = float(cells[("lifted_interaction", "lifted_interaction")][metric])
            rows.append({
                "support_mode": support_mode,
                "metric": metric,
                "estimate_effect_at_first_order_direction_fo_minus_lifted": ff - lf,
                "estimate_effect_at_lifted_direction_fo_minus_lifted": fl - ll,
                "direction_effect_at_first_order_estimate_fo_minus_lifted": ff - fl,
                "direction_effect_at_lifted_estimate_fo_minus_lifted": lf - ll,
                "estimate_by_direction_interaction": (ff - lf) - (fl - ll),
            })
    return pd.DataFrame(rows)


def _paired_factorial_deltas(per_example: pd.DataFrame) -> pd.DataFrame:
    primary = per_example[
        (per_example["estimator_calibration"] == "none")
        & (per_example["controller_mode"] == "base")
        & per_example["estimator"].isin(["first_order", "lifted_interaction"])
        & per_example["direction_provider"].isin(["first_order", "lifted_interaction"])
    ]
    metrics = [
        "integrated_squared_error",
        "integrated_absolute_error",
        "final_target_abs_error",
        "final_target_squared_error",
        "cumulative_collateral_abs",
        "actuation_energy_l1",
        "observer_bias_abs_mean",
        "mean_residual_off_support_l2",
    ]
    rows: list[dict[str, float | int | str]] = []
    for support_mode in sorted(primary["support_mode"].unique()):
        support_rows = primary[primary["support_mode"] == support_mode]
        for fixed_factor, contrast_factor in [
            ("direction_provider", "estimator"),
            ("estimator", "direction_provider"),
        ]:
            for fixed_name in ["first_order", "lifted_interaction"]:
                first = support_rows[
                    (support_rows[fixed_factor] == fixed_name)
                    & (support_rows[contrast_factor] == "first_order")
                ].set_index("example_idx")
                lifted = support_rows[
                    (support_rows[fixed_factor] == fixed_name)
                    & (support_rows[contrast_factor] == "lifted_interaction")
                ].set_index("example_idx")
                if len(first) == 0 or not first.index.equals(lifted.index):
                    continue
                for metric in metrics:
                    deltas = first[metric].astype(float) - lifted[metric].astype(float)
                    rows.extend({
                        "support_mode": support_mode,
                        "contrast": f"{contrast_factor}_first_order_minus_lifted",
                        "fixed_factor": fixed_factor,
                        "fixed_name": fixed_name,
                        "metric": metric,
                        "example_idx": int(example_idx),
                        "x1": float(first.loc[example_idx, "x1"]),
                        "x2": float(first.loc[example_idx, "x2"]),
                        "delta": float(delta),
                    } for example_idx, delta in deltas.items())
    return pd.DataFrame(rows)


def _primary_plot_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["estimator_calibration"] == "none")
        & (frame["controller_mode"] == "base")
        & frame["estimator"].isin(["first_order", "lifted_interaction"])
        & frame["direction_provider"].isin(["first_order", "lifted_interaction"])
    ]


def run_trained_transformer_ctl2(cfg: TrainedTransformerCtl2Config, outdir: str | Path) -> List[ObserverResult]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    env = _prepare_observers(cfg)

    results: List[ObserverResult] = []
    trajectories: list[pd.DataFrame] = []
    per_examples: list[pd.DataFrame] = []
    trajectory_examples: list[pd.DataFrame] = []
    skipped_arms: list[dict[str, str]] = []
    for arm in enumerate_ctl2_arms(cfg):
        try:
            metrics, traj, per_ex, traj_ex = _rollout_arm(cfg, env, arm)
        except GainMatchIneligible as exc:
            skipped_arms.append({"arm_id": arm.arm_id, "reason": str(exc)})
            continue
        trajectories.append(traj)
        per_examples.append(per_ex)
        trajectory_examples.append(traj_ex)
        results.append(ObserverResult(
            task="trained_transformer_ctl2_closed_loop",
            observer=_arm_label(cfg, arm),
            access_regime="white-box final-residual representation; additive residual intervention loop",
            observer_family="independently composed affine estimator and fixed residual-space direction",
            metrics=metrics,
            metadata={
                "estimand": "target-head state along an additive final-residual intervention trajectory",
                "measurement_design": (
                    "cross first-order, lifted, and oracle estimators with independently held directions; "
                    "optionally project directions into the affine span of clean residual states and calibrate on "
                    "separate finite probes"
                ),
                "validation_target": (
                    "paired integrated/final target error, fitted nuisance-probe displacement, self-gain, "
                    "clipping, and distance from the clean residual states"
                ),
                "notes": (
                    "This is an additive final-residual loop with affine readouts, not a rerun transformer plant. "
                    "Only within-direction estimator contrasts isolate estimation effects."
                ),
                "estimator": arm.estimator_name,
                "direction_provider": arm.direction_name,
                "support_mode": arm.support_mode,
                "estimator_calibration": arm.estimator_calibration,
                "controller_mode": arm.controller_mode,
            },
        ))

    if not results:
        raise RuntimeError("Ctl-2 produced no eligible arms")

    # Keep the v7 lifted-diagonal ratios when a matching reference exists, but
    # make the direction/support condition explicit for factorial runs.
    reference_by_condition = {
        (
            result.metrics["direction_support_mode"],
            result.metrics["estimator_calibration"],
            result.metrics["controller_mode"],
        ): result.metrics
        for result in results
        if result.metrics["estimator_name"] == "lifted_interaction"
        and result.metrics["direction_name"] == "lifted_interaction"
    }
    for result in results:
        key = (
            result.metrics["direction_support_mode"],
            result.metrics["estimator_calibration"],
            result.metrics["controller_mode"],
        )
        reference = reference_by_condition.get(key)
        if reference is None:
            continue
        result.metrics["ise_ratio_vs_lifted_diagonal_same_condition"] = (
            result.metrics["integrated_squared_error"] / max(reference["integrated_squared_error"], 1e-12)
        )
        result.metrics["collateral_ratio_vs_lifted_diagonal_same_condition"] = (
            result.metrics["cumulative_collateral_abs"] / max(reference["cumulative_collateral_abs"], 1e-12)
        )
        result.metrics["final_mse_ratio_vs_lifted_diagonal_same_condition"] = (
            result.metrics["final_target_mse"] / max(reference["final_target_mse"], 1e-12)
        )
        result.metrics["energy_ratio_vs_lifted_diagonal_same_condition"] = (
            result.metrics["actuation_energy_l1"] / max(reference["actuation_energy_l1"], 1e-12)
        )
        if cfg.arm_design == "diagonal" and cfg.direction_support_mode == "unprojected":
            result.metrics["ise_ratio_vs_lifted"] = result.metrics["ise_ratio_vs_lifted_diagonal_same_condition"]
            result.metrics["cumulative_collateral_ratio_vs_lifted"] = result.metrics[
                "collateral_ratio_vs_lifted_diagonal_same_condition"
            ]
            result.metrics["final_mse_ratio_vs_lifted"] = result.metrics[
                "final_mse_ratio_vs_lifted_diagonal_same_condition"
            ]
            result.metrics["energy_ratio_vs_lifted"] = result.metrics[
                "energy_ratio_vs_lifted_diagonal_same_condition"
            ]

    results_to_dataframe(results).to_csv(out / "observerbench_results.csv", index=False)
    result_frame = pd.DataFrame([{"observer": result.observer, **result.metrics} for result in results])
    result_frame.to_csv(out / "trained_transformer_ctl2_results.csv", index=False)
    traj_df = pd.concat(trajectories, ignore_index=True)
    per_ex_df = pd.concat(per_examples, ignore_index=True)
    traj_ex_df = pd.concat(trajectory_examples, ignore_index=True)
    traj_df.to_csv(out / "trained_transformer_ctl2_trajectory_summary.csv", index=False)
    if cfg.write_per_example_outputs:
        per_ex_df.to_csv(out / "trained_transformer_ctl2_per_example.csv", index=False)
    if cfg.write_per_step_examples:
        traj_ex_df.to_csv(out / "trained_transformer_ctl2_per_step_examples.csv", index=False)

    archetype_metrics = [
        "integrated_squared_error",
        "final_target_squared_error",
        "cumulative_collateral_abs",
        "actuation_energy_l1",
        "observer_bias_abs_mean",
        "mean_residual_delta_from_initial_l2",
        "mean_nearest_clean_residual_l2",
        "mean_abs_residual_norm_z",
        "mean_convex_hull_extrapolation_mass",
        "control_clip_fraction",
    ]
    archetypes = per_ex_df.groupby(
        [
            "observer",
            "estimator",
            "direction_provider",
            "support_mode",
            "estimator_calibration",
            "controller_mode",
            "x1",
            "x2",
        ],
        as_index=False,
    ).agg(
        n_examples=("example_idx", "size"),
        **{metric: (metric, "mean") for metric in archetype_metrics},
    )
    archetypes.to_csv(out / "trained_transformer_ctl2_state_archetypes.csv", index=False)

    factorial = _factorial_contrasts(results)
    factorial.to_csv(out / "trained_transformer_ctl2_factorial_contrasts.csv", index=False)
    paired = _paired_factorial_deltas(per_ex_df)
    if cfg.write_per_example_outputs:
        paired.to_csv(out / "trained_transformer_ctl2_factorial_paired_deltas.csv", index=False)
    if len(paired):
        paired.groupby(
            ["support_mode", "contrast", "fixed_factor", "fixed_name", "metric"],
            as_index=False,
        )["delta"].agg(["mean", "min", "max"]).reset_index().to_csv(
            out / "trained_transformer_ctl2_factorial_paired_summary.csv",
            index=False,
        )

    # Preserve the legacy diagonal paired output for fast-reproduction users.
    if cfg.write_per_example_outputs and {"first_order", "lifted_interaction"}.issubset(set(per_ex_df["observer"])):
        fo_ex = per_ex_df[per_ex_df["observer"] == "first_order"].set_index("example_idx")
        li_ex = per_ex_df[per_ex_df["observer"] == "lifted_interaction"].set_index("example_idx")
        legacy = pd.DataFrame({"example_idx": fo_ex.index})
        for column in [
            "integrated_squared_error", "integrated_absolute_error", "final_target_abs_error",
            "final_target_squared_error", "cumulative_collateral_abs", "final_collateral_abs",
            "actuation_energy_l1", "observer_bias_abs_mean", "max_abs_error",
            "final_over_initial_squared_error",
        ]:
            legacy[f"delta_{column}_fo_minus_lifted"] = fo_ex[column].values - li_ex[column].values
        legacy.to_csv(out / "trained_transformer_ctl2_paired_deltas.csv", index=False)

    if cfg.write_observer_cards:
        write_cards(results, out / "observer_cards")
    write_json(out / "trained_transformer_ctl2_config.json", asdict(cfg))
    write_json(out / "trained_transformer_ctl2_train_info.json", env["train_info"])
    write_json(out / "trained_transformer_ctl2_skipped_arms.json", skipped_arms)
    write_json(out / "trained_transformer_ctl2_support.json", {
        "rank": env["support"].rank,
        "n_centroids": env["support"].n_centroids,
        "relative_tolerance": cfg.support_relative_tolerance,
        "absolute_tolerance": env["support"].absolute_tolerance,
        "clean_rms_radius": env["support"].clean_rms_radius,
        "singular_values": env["support"].singular_values.tolist(),
        "clean_residual_norm_mean": env["clean_residual_norm_mean"],
        "clean_residual_norm_std": env["clean_residual_norm_std"],
        "clean_residual_norm_min": env["clean_residual_norm_min"],
        "clean_residual_norm_max": env["clean_residual_norm_max"],
        "clean_centroid_pairwise_rms": env["clean_centroid_pairwise_rms"],
    })

    summary = {
        "schema": "observerbench.ctl2.phase01.v2",
        "arm_design": cfg.arm_design,
        "n_completed_arms": len(results),
        "n_skipped_arms": len(skipped_arms),
        "residual_support_rank": env["support"].rank,
        "primary_contrasts_file": "trained_transformer_ctl2_factorial_contrasts.csv",
        "continuous_errors_are_primary": True,
        "thresholded_large_error_rates_are_secondary": True,
    }
    if cfg.arm_design == "diagonal" and {"first_order", "lifted_interaction"}.issubset(set(result_frame["observer"])):
        by_name = result_frame.set_index("observer")
        fo = by_name.loc["first_order"]
        lifted = by_name.loc["lifted_interaction"]
        summary.update({
            "delta_ise_fo_minus_lifted": float(fo["integrated_squared_error"] - lifted["integrated_squared_error"]),
            "delta_cumulative_collateral_fo_minus_lifted": float(
                fo["cumulative_collateral_abs"] - lifted["cumulative_collateral_abs"]
            ),
            "delta_final_target_mse_fo_minus_lifted": float(fo["final_target_mse"] - lifted["final_target_mse"]),
            "delta_large_error_rate_fo_minus_lifted": float(
                fo["large_error_trajectory_rate"] - lifted["large_error_trajectory_rate"]
            ),
            "delta_divergence_rate_fo_minus_lifted": float(fo["divergence_rate"] - lifted["divergence_rate"]),
            "delta_worsened_rate_fo_minus_lifted": float(
                fo["target_error_worsened_rate"] - lifted["target_error_worsened_rate"]
            ),
            "fo_ise_ratio_vs_lifted": float(
                fo["integrated_squared_error"] / max(lifted["integrated_squared_error"], 1e-12)
            ),
            "fo_collateral_ratio_vs_lifted": float(
                fo["cumulative_collateral_abs"] / max(lifted["cumulative_collateral_abs"], 1e-12)
            ),
            "fo_energy_ratio_vs_lifted": float(
                fo["actuation_energy_l1"] / max(lifted["actuation_energy_l1"], 1e-12)
            ),
            "strong_success_candidate": bool(
                fo["integrated_squared_error"] <= 1.25 * lifted["integrated_squared_error"]
                and fo["cumulative_collateral_abs"] >= 2.0 * lifted["cumulative_collateral_abs"]
            ),
            "alternative_success_candidate": bool(
                fo["integrated_squared_error"] > lifted["integrated_squared_error"]
            ),
        })
        if "oracle_target" in by_name.index:
            oracle = by_name.loc["oracle_target"]
            summary.update({
                "oracle_integrated_squared_error": float(oracle["integrated_squared_error"]),
                "oracle_final_target_mse": float(oracle["final_target_mse"]),
                "oracle_divergence_rate": float(oracle["divergence_rate"]),
                "oracle_target_error_worsened_rate": float(oracle["target_error_worsened_rate"]),
                "oracle_sanity_converges": bool(
                    oracle["divergence_rate"] == 0.0
                    and oracle["target_error_worsened_rate"] < 0.5
                ),
            })
    write_json(out / "trained_transformer_ctl2_pairwise_summary.json", summary)

    repo_root = Path(__file__).resolve().parents[3]
    source_files = [
        repo_root / "src" / "observerbench" / "control.py",
        repo_root / "src" / "observerbench" / "tasks" / "trained_ctl2.py",
        repo_root / "src" / "observerbench" / "tasks" / "trained_ctl1.py",
    ]
    write_json(out / "trained_transformer_ctl2_run_manifest.json", {
        "schema": "observerbench.run_manifest.v0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_provenance(repo_root),
        "config_sha256": json_sha256(asdict(cfg)),
        "source_sha256": {
            str(path.relative_to(repo_root)): file_sha256(path)
            for path in source_files
        },
        "result_sha256": file_sha256(out / "trained_transformer_ctl2_results.csv"),
        "train_info_sha256": file_sha256(out / "trained_transformer_ctl2_train_info.json"),
    })

    if not cfg.write_plots:
        return results

    plot_traj = _primary_plot_rows(traj_df)
    plot_examples = _primary_plot_rows(traj_ex_df)
    for value, ylabel, filename, title in [
        ("target_mse", "mean target MSE", "ctl2_target_mse_trajectory.png", "Ctl-2: target tracking"),
        ("collateral_abs_delta", "mean |fitted nuisance-probe displacement|", "ctl2_collateral_trajectory.png", "Ctl-2: fitted nuisance-probe movement"),
        ("observer_error_mae", "observer MAE vs plant target", "ctl2_observer_bias_trajectory.png", "Ctl-2: observer bias"),
        ("mean_residual_off_support_l2", "mean distance from clean affine span", "ctl2_off_support_trajectory.png", "Ctl-2: affine-span departure"),
        ("mean_nearest_clean_residual_l2", "mean distance to nearest clean residual", "ctl2_nearest_clean_trajectory.png", "Ctl-2: distance from observed residual states"),
        ("mean_convex_hull_extrapolation_mass", "mean negative barycentric mass", "ctl2_convex_hull_trajectory.png", "Ctl-2: extrapolation beyond the clean convex hull"),
    ]:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        for arm_id, group in plot_traj.groupby("observer"):
            ax.plot(group["step"], group[value], marker="o", label=arm_id)
        ax.set_xlabel("control step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)

    _plot_fan(
        plot_examples,
        "target_squared_error",
        out / "ctl2_target_mse_fan.png",
        "Ctl-2: target-error trajectories",
        "per-example target squared error",
    )
    _plot_fan(
        plot_examples,
        "collateral_abs_delta",
        out / "ctl2_collateral_fan.png",
        "Ctl-2: collateral trajectories",
        "per-example |collateral displacement|",
    )
    _plot_fan(
        plot_examples,
        "observer_abs_bias",
        out / "ctl2_observer_bias_fan.png",
        "Ctl-2: observer-bias trajectories",
        "per-example |observer - plant target|",
    )
    _plot_fan(
        plot_examples,
        "residual_off_support_l2",
        out / "ctl2_off_support_fan.png",
        "Ctl-2: clean affine-span departure",
        "per-example distance from clean affine span",
    )
    _plot_fan(
        plot_examples,
        "nearest_clean_residual_l2",
        out / "ctl2_nearest_clean_fan.png",
        "Ctl-2: distance from observed residual states",
        "per-example distance to nearest clean residual",
    )

    scatter = plot_examples[plot_examples["step"] > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for arm_id, group in scatter.groupby("observer"):
        ax.scatter(group["residual_off_support_l2"], group["observer_abs_bias"], s=8, alpha=0.18, label=arm_id)
    ax.set_xlabel("residual distance from clean affine support")
    ax.set_ylabel("|observer - plant target|")
    ax.set_title("Ctl-2: observer bias versus support departure")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "ctl2_bias_vs_off_support.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for arm_id, group in scatter.groupby("observer"):
        ax.scatter(group["nearest_clean_residual_l2"], group["observer_abs_bias"], s=8, alpha=0.18, label=arm_id)
    ax.set_xlabel("distance to nearest clean residual")
    ax.set_ylabel("|observer - plant target|")
    ax.set_title("Ctl-2: observer bias versus clean-state distance")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "ctl2_bias_vs_nearest_clean.png", dpi=180)
    plt.close(fig)

    summary_rows = _primary_plot_rows(per_ex_df).drop_duplicates("observer")
    primary_results = result_frame[result_frame["observer"].isin(summary_rows["observer"])]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(primary_results))
    width = 0.35
    ax.bar(
        x - width / 2,
        primary_results["integrated_squared_error"],
        width,
        label="integrated target squared error",
    )
    ax.bar(
        x + width / 2,
        primary_results["cumulative_collateral_abs"],
        width,
        label="cumulative fitted nuisance-probe displacement",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(primary_results["observer"], rotation=25, ha="right")
    ax.set_title("Ctl-2 primary estimator--direction arms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ctl2_summary_bar.png", dpi=180)
    plt.close(fig)

    return results
