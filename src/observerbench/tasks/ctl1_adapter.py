"""Public composition adapter for the bundled analytic Ctl-1 task.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from observerbench.control import ClippedProportionalController
from observerbench.core import (
    API_CONTRACT_VERSION,
    ControlRequest,
    Controller,
    Observer,
    ObserverResult,
    ObserverTask,
    write_json,
)
from observerbench.metrics import mae, mse, r2_score
from observerbench.provenance import component_provenance, runtime_provenance
from observerbench.tasks.ctl1_analytic import (
    CollateralTaskConfig,
    generate_data,
    normalize_direction,
    nuisance_readout,
    target_readout,
)


_CONFIG_METADATA_KEYS = {"mode", "quick", "task"}
RESULT_SCHEMA_VERSION = "observerbench.external_task_run.v0"


def _task_config(config: Mapping[str, Any]) -> CollateralTaskConfig:
    field_names = {item.name for item in fields(CollateralTaskConfig)}
    unknown = set(config) - field_names - _CONFIG_METADATA_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown analytic Ctl-1 configuration fields: {names}")
    if config.get("task", "ctl1_analytic") != "ctl1_analytic":
        raise ValueError("Analytic Ctl-1 adapter requires task='ctl1_analytic'")
    return CollateralTaskConfig(**{key: value for key, value in config.items() if key in field_names})


def _fixed_direction(observations: list[Any]) -> np.ndarray:
    directions = [np.asarray(observation.direction, dtype=float) for observation in observations]
    if not directions:
        raise ValueError("Analytic Ctl-1 requires at least one test state")
    if any(direction.shape != (3,) for direction in directions):
        raise ValueError("Analytic Ctl-1 directions must have shape (3,)")
    if not all(np.isfinite(direction).all() for direction in directions):
        raise ValueError("Analytic Ctl-1 directions must be finite")
    reference = directions[0]
    if not all(np.allclose(direction, reference, rtol=1e-10, atol=1e-12) for direction in directions[1:]):
        raise ValueError("Analytic Ctl-1 requires a fixed direction across test states")
    return reference.copy()


def _scalar_estimates(observations: list[Any]) -> np.ndarray:
    estimates = np.asarray([observation.estimate for observation in observations], dtype=float)
    if estimates.ndim != 1 or not np.isfinite(estimates).all():
        raise ValueError("Analytic Ctl-1 estimates must be finite scalars")
    return estimates


def _controller_actions(
    controller: Controller[np.ndarray, float],
    states: np.ndarray,
    estimates: np.ndarray,
    target: float,
) -> np.ndarray:
    actions = np.asarray(
        [
            controller.action(
                ControlRequest(
                    state=state.copy(),
                    target=target,
                    estimate=float(estimate),
                    step=0,
                )
            )
            for state, estimate in zip(states, estimates)
        ],
        dtype=float,
    )
    if actions.shape != estimates.shape or not np.isfinite(actions).all():
        raise ValueError("Analytic Ctl-1 controller actions must be finite scalars")
    return actions


def run_ctl1_analytic_observer(
    *,
    observer: Observer[np.ndarray, float, np.ndarray],
    controller: Controller[np.ndarray, float],
    config: Mapping[str, Any],
    outdir: Path,
) -> ObserverResult:
    """Evaluate one external observer on the existing one-shot Ctl-1 fixture."""

    cfg = _task_config(config)
    _train, test = generate_data(cfg)
    states = np.asarray(test["activation"], dtype=float)
    observations = [observer.observe(state.copy()) for state in states]
    estimates = _scalar_estimates(observations)
    raw_direction = _fixed_direction(observations)

    tr = target_readout(cfg)
    nr = nuisance_readout(cfg)
    direction, raw_target_gain, direction_scale = normalize_direction(raw_direction, tr, cfg)
    target_gain = float(tr @ direction)
    collateral_gain = float(nr @ direction)
    actions = _controller_actions(controller, states, estimates, cfg.target_ref)

    delta = actions[:, None] * direction[None, :]
    target_after = test["target"] + delta @ tr
    nuisance_after = test["nuisance"] + delta @ nr
    baseline_target_mse = mse(np.full_like(test["target"], cfg.target_ref), test["target"])
    control_target_mse = mse(np.full_like(target_after, cfg.target_ref), target_after)
    collateral_per_target_gain = (
        collateral_gain / target_gain if abs(target_gain) > 1e-12 else float("nan")
    )

    expected_controller = make_ctl1_analytic_controller(cfg)
    fixed_controller = bool(
        type(controller) is ClippedProportionalController
        and np.isclose(controller.gain, expected_controller.gain)
        and np.isclose(controller.max_action, expected_controller.max_action)
    )
    runtime = runtime_provenance(Path(__file__).resolve().parents[3])
    result = ObserverResult(
        task="ctl1_analytic",
        observer=observer.name,
        access_regime="external observer over the analytic activation state",
        observer_family="externally supplied estimator and fixed direction provider",
        metrics={
            "observer_r2": r2_score(test["target"], estimates),
            "observer_mae": mae(test["target"], estimates),
            "baseline_target_mse": baseline_target_mse,
            "control_target_mse": control_target_mse,
            "target_improvement_mse": baseline_target_mse - control_target_mse,
            "collateral_abs_delta": float(np.mean(np.abs(nuisance_after - test["nuisance"]))),
            "actuation_energy_l1": float(np.mean(np.sum(np.abs(delta), axis=1))),
            "mean_abs_strength": float(np.mean(np.abs(actions))),
            "mean_strength": float(np.mean(actions)),
            "raw_direction_x1": float(raw_direction[0]),
            "raw_direction_x2": float(raw_direction[1]),
            "raw_direction_interaction": float(raw_direction[2]),
            "direction_x1": float(direction[0]),
            "direction_x2": float(direction[1]),
            "direction_interaction": float(direction[2]),
            "raw_target_gain": raw_target_gain,
            "direction_scale": direction_scale,
            "effective_target_gain": target_gain,
            "effective_collateral_gain": collateral_gain,
            "collateral_per_target_gain": collateral_per_target_gain,
            "gamma": float(cfg.gamma),
            "nuisance_interaction_weight": float(cfg.nuisance_interaction_weight),
        },
        metadata={
            "api_contract_version": API_CONTRACT_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "comparison_mode": "benchmark_fixed_controller" if fixed_controller else "custom_controller",
            "benchmark_comparable_controller": fixed_controller,
            "observer_provenance": dict(observer.provenance),
            "estimator": component_provenance(observer.estimator),
            "direction_provider": component_provenance(observer.direction_provider),
            "controller": component_provenance(controller),
            "runtime": runtime,
            "task_config": asdict(cfg),
            "estimand": "one-shot finite-control target state in [x1, x2, x1*x2] coordinates",
            "measurement_design": (
                "external scalar estimate and fixed three-coordinate direction "
                "on the bundled analytic Ctl-1 test split"
            ),
            "validation_target": "target tracking with movement of the fixed nuisance readout reported separately",
            "notes": (
                "The task normalizes the supplied fixed direction using the same "
                "target-gain rule as the bundled paper observer evaluation."
            ),
        },
    )

    write_json(outdir / "observer_result.json", result.to_dict())
    write_json(
        outdir / "run_metadata.json",
        {
            "schema": RESULT_SCHEMA_VERSION,
            "api_contract_version": API_CONTRACT_VERSION,
            "package_version": runtime["package_version"],
            "source_revision": runtime["source_revision"],
            "source_dirty": runtime["source_dirty"],
            "task": result.task,
            "observer": observer.name,
            "observer_provenance": dict(observer.provenance),
            "estimator": component_provenance(observer.estimator),
            "direction_provider": component_provenance(observer.direction_provider),
            "controller": component_provenance(controller),
            "comparison_mode": result.metadata["comparison_mode"],
            "config": asdict(cfg),
        },
    )
    return result


def make_ctl1_analytic_task() -> ObserverTask[np.ndarray, float, np.ndarray, ObserverResult]:
    """Return a local task composition point without mutating the registry."""

    return ObserverTask(
        name="ctl1_analytic",
        summary="Evaluate an external observer on the bundled analytic Ctl-1 fixture.",
        runner=run_ctl1_analytic_observer,
    )


def make_ctl1_analytic_controller(cfg: CollateralTaskConfig) -> ClippedProportionalController:
    """Construct the controller used by the bundled analytic Ctl-1 task."""

    return ClippedProportionalController(gain=cfg.controller_gain, max_action=cfg.max_strength)


def evaluate_ctl1_analytic_observer(
    observer: Observer[np.ndarray, float, np.ndarray],
    *,
    config: Mapping[str, Any],
    outdir: str | Path,
) -> ObserverResult:
    """Run the benchmark-comparable Ctl-1 path with its controller held fixed."""

    cfg = _task_config(config)
    from observerbench.core import run_observer_task

    return run_observer_task(
        make_ctl1_analytic_task(),
        observer,
        make_ctl1_analytic_controller(cfg),
        config=config,
        outdir=outdir,
    )
