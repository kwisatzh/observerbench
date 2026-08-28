"""Tests for supplying an outside observer to the bundled analytic task.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pytest

from observerbench import Observer, run_observer_task
from observerbench.control import (
    AffineStateEstimator,
    ClippedProportionalController,
    FixedDirectionProvider,
)
from observerbench.observers import LinearObserver, lifted_pair_features
from observerbench.tasks import (
    evaluate_ctl1_analytic_observer,
    make_ctl1_analytic_controller,
    make_ctl1_analytic_task,
    task_names,
)
from observerbench.tasks.ctl1_analytic import (
    CollateralTaskConfig,
    evaluate_observer,
    generate_data,
    observer_direction_in_activation_space,
    target_readout,
)


class OutsideOracleEstimator:
    name = "outside-oracle-estimator"

    def __init__(self, readout: np.ndarray) -> None:
        self.readout = np.asarray(readout, dtype=float)

    def estimate(self, state: np.ndarray) -> float:
        return float(self.readout @ state)


class OutsideFixedDirection:
    name = "outside-fixed-direction"

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = np.asarray(vector, dtype=float)

    def direction(self, state: np.ndarray) -> np.ndarray:
        del state
        return self.vector.copy()


class OutsideStateDependentDirection:
    name = "outside-state-dependent-direction"

    def direction(self, state: np.ndarray) -> np.ndarray:
        return np.array([1.0 + state[0], 0.0, 0.0])


def small_config() -> CollateralTaskConfig:
    return CollateralTaskConfig(
        n_train=64,
        n_test=64,
        p_feature=0.5,
        seed=7,
        target_noise=0.0,
        collateral_noise=0.0,
    )


def test_outside_observer_runs_on_existing_ctl1_task_without_registration(tmp_path: Path) -> None:
    cfg = small_config()
    readout = target_readout(cfg)
    observer = Observer(
        name="outside-oracle",
        estimator=OutsideOracleEstimator(readout),
        direction_provider=OutsideFixedDirection(readout),
        provenance={"implementation": "tests.outside-oracle", "version": "test"},
    )
    task_names_before = task_names()

    result = run_observer_task(
        make_ctl1_analytic_task(),
        observer,
        make_ctl1_analytic_controller(cfg),
        config={"task": "ctl1_analytic", "mode": "quick", **asdict(cfg)},
        outdir=tmp_path / "outside-observer",
    )

    assert task_names() == task_names_before
    assert result.task == "ctl1_analytic"
    assert result.observer == "outside-oracle"
    assert result.metrics["observer_r2"] == pytest.approx(1.0)
    assert result.metrics["effective_target_gain"] == pytest.approx(cfg.target_actuation_gain)
    assert result.metadata["estimator"]["name"] == "outside-oracle-estimator"
    assert result.metadata["direction_provider"]["name"] == "outside-fixed-direction"
    assert result.metadata["controller"]["name"] == "clipped_proportional"
    assert result.metadata["comparison_mode"] == "benchmark_fixed_controller"
    assert result.metadata["observer_provenance"]["version"] == "test"

    saved_result = json.loads((tmp_path / "outside-observer" / "observer_result.json").read_text())
    saved_metadata = json.loads((tmp_path / "outside-observer" / "run_metadata.json").read_text())
    assert saved_result["observer"] == "outside-oracle"
    assert saved_metadata["schema"] == "observerbench.external_task_run.v0"
    assert saved_metadata["package_version"] == version("observerbench")
    assert saved_metadata["source_revision"]


def test_benchmark_wrapper_owns_controller_settings(tmp_path: Path) -> None:
    cfg = small_config()
    readout = target_readout(cfg)
    observer = Observer(
        name="outside-oracle",
        estimator=OutsideOracleEstimator(readout),
        direction_provider=OutsideFixedDirection(readout),
    )

    result = evaluate_ctl1_analytic_observer(
        observer,
        config=asdict(cfg),
        outdir=tmp_path / "fixed-controller",
    )

    assert result.metadata["benchmark_comparable_controller"] is True
    assert result.metadata["controller"]["parameters"] == {
        "gain": cfg.controller_gain,
        "max_action": cfg.max_strength,
        "name": "clipped_proportional",
    }


def test_custom_controller_run_is_marked_noncomparable(tmp_path: Path) -> None:
    cfg = small_config()
    readout = target_readout(cfg)
    observer = Observer(
        name="outside-oracle",
        estimator=OutsideOracleEstimator(readout),
        direction_provider=OutsideFixedDirection(readout),
    )

    result = run_observer_task(
        make_ctl1_analytic_task(),
        observer,
        ClippedProportionalController(gain=0.1, max_action=cfg.max_strength),
        config=asdict(cfg),
        outdir=tmp_path / "custom-controller",
    )

    assert result.metadata["benchmark_comparable_controller"] is False
    assert result.metadata["comparison_mode"] == "custom_controller"


def test_ctl1_adapter_rejects_state_dependent_direction(tmp_path: Path) -> None:
    cfg = small_config()
    observer = Observer(
        name="invalid-direction",
        estimator=OutsideOracleEstimator(target_readout(cfg)),
        direction_provider=OutsideStateDependentDirection(),
    )

    with pytest.raises(ValueError, match="requires a fixed direction"):
        run_observer_task(
            make_ctl1_analytic_task(),
            observer,
            make_ctl1_analytic_controller(cfg),
            config=asdict(cfg),
            outdir=tmp_path / "invalid-direction",
        )


def test_ctl1_adapter_matches_existing_evaluation_semantics(tmp_path: Path) -> None:
    cfg = small_config()
    train, test = generate_data(cfg)
    fitted = LinearObserver("lifted_interaction", lifted_pair_features, ridge=cfg.ridge)
    fitted.fit(train, train["target"])
    assert fitted.coef_ is not None
    direction = observer_direction_in_activation_space("lifted_interaction", fitted)
    observer = Observer(
        name="lifted-interaction-adapter",
        estimator=AffineStateEstimator(
            name="prefit-lifted-estimator",
            intercept=float(fitted.coef_[0]),
            gradient=fitted.coef_[1:],
        ),
        direction_provider=FixedDirectionProvider(
            name="prefit-lifted-direction",
            vector=direction,
        ),
    )

    adapted = run_observer_task(
        make_ctl1_analytic_task(),
        observer,
        make_ctl1_analytic_controller(cfg),
        config=asdict(cfg),
        outdir=tmp_path / "compatibility",
    )
    existing = evaluate_observer("lifted_interaction", fitted, train, test, cfg)

    for metric in (
        "observer_r2",
        "observer_mae",
        "baseline_target_mse",
        "control_target_mse",
        "target_improvement_mse",
        "collateral_abs_delta",
        "actuation_energy_l1",
        "mean_abs_strength",
        "mean_strength",
        "effective_target_gain",
        "effective_collateral_gain",
    ):
        assert adapted.metrics[metric] == pytest.approx(existing.metrics[metric])
