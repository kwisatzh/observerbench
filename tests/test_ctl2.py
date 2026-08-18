from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from observerbench.control import AffineStateEstimator, fit_affine_support
from observerbench.tasks.trained_ctl2 import (
    Ctl2Arm,
    GainMatchIneligible,
    TrainedTransformerCtl2Config,
    _response_gain_calibration,
    _rollout_arm,
    _rollout_observer,
    enumerate_ctl2_arms,
)
from observerbench.tasks.registry import run_registered_task


def tiny_ctl2_config(**overrides):
    config = {
        "task": "trained_ctl2",
        "mode": "quick",
        "quick": True,
        "seed": 0,
        "device": "cpu",
        "n_train": 160,
        "n_test": 64,
        "train_steps": 20,
        "d_model": 12,
        "n_heads": 2,
        "n_layers": 1,
        "d_mlp": 32,
        "batch_size": 64,
        "loop_steps": 4,
        "controller_gain": 0.35,
        "max_strength": 1.0,
        "gamma": 1.15,
        "nuisance_interaction_weight": 0.0,
        "include_oracle": True,
    }
    config.update(overrides)
    return config


def run_ctl2(tmp_path: Path, **overrides) -> Path:
    outdir = tmp_path / overrides.pop("dirname", "ctl2")
    run_registered_task("trained_ctl2", tiny_ctl2_config(**overrides), Path("configs/trained_ctl2.yaml"), outdir)
    return outdir


def synthetic_ctl2_env() -> dict:
    h_test = np.asarray(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.2, 0.2],
            [0.4, 0.1],
        ],
        dtype=float,
    )
    target_vec = np.asarray([1.0, 0.0], dtype=float)
    target = h_test @ target_vec
    first_direction = np.asarray([0.5, 0.1], dtype=float)
    lifted_direction = first_direction.copy()
    oracle_direction = np.asarray([1.0, 0.0], dtype=float)
    estimators = {
        "first_order": AffineStateEstimator("first_order", 0.0, target_vec),
        "lifted_interaction": AffineStateEstimator("lifted_interaction", 0.0, target_vec),
        "oracle_target": AffineStateEstimator("oracle_target", 0.0, target_vec),
    }
    support = fit_affine_support(h_test, relative_tolerance=1e-8)

    def direction_record(estimator_name: str, vector: np.ndarray) -> dict:
        return {
            "coef": None,
            "estimator": estimators[estimator_name],
            "direction": vector,
            "directions": {"unprojected": vector, "projected": vector},
            "raw_direction": vector,
            "raw_target_gain": float(target_vec @ vector),
            "norm_diag": {},
            "projected_direction_scale": 1.0,
            "projected_direction_error": "",
        }

    return {
        "test": {"target": target.copy(), "target_clean": target.copy()},
        "calibration": {"target": target.copy(), "target_clean": target.copy()},
        "test_pred": target.copy(),
        "h_test": h_test,
        "h_calibration": h_test.copy(),
        "target_vec": target_vec,
        "target_bias": 0.0,
        "nuisance_vec": np.asarray([0.0, 1.0], dtype=float),
        "nuisance_bias": 0.0,
        "probe_x1": np.asarray([0.0, 1.0, 0.0], dtype=float),
        "probe_x2": np.asarray([0.0, 0.0, 1.0], dtype=float),
        "probe_int": np.asarray([0.0, 0.5, 0.5], dtype=float),
        "observers": {
            "first_order": direction_record("first_order", first_direction),
            "lifted_interaction": direction_record("lifted_interaction", lifted_direction),
            "oracle_target": direction_record("oracle_target", oracle_direction),
        },
        "estimators": estimators,
        "support": support,
        "clean_residual_centroids": h_test.copy(),
    }


def synthetic_ctl2_config(**overrides) -> TrainedTransformerCtl2Config:
    cfg = TrainedTransformerCtl2Config(
        seed=0,
        n_train=4,
        n_test=4,
        train_steps=0,
        device="cpu",
        loop_steps=4,
        controller_gain=0.5,
        max_strength=1.0,
        target_ref=1.0,
        gamma=0.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def rollout_synthetic(observer: str, **overrides):
    return _rollout_observer(synthetic_ctl2_config(**overrides), synthetic_ctl2_env(), observer)


def test_ctl2_synthetic_quick_smoke_no_training() -> None:
    metrics, traj, per_example, per_step = rollout_synthetic("first_order")

    assert metrics["loop_steps"] == 4.0
    assert set(traj["step"]) == {0, 1, 2, 3, 4}
    assert per_example["integrated_squared_error"].notna().all()
    assert per_step.groupby("example_idx")["step"].nunique().min() == 5
    assert metrics["observer_self_gain"] == pytest.approx(0.5)
    assert metrics["observer_error_pole_unsaturated"] == pytest.approx(0.75)
    assert metrics["control_clip_fraction"] == 0.0
    assert metrics["final_residual_delta_from_initial_l2"] >= 0.0
    assert metrics["final_nearest_clean_residual_l2"] >= 0.0
    assert metrics["final_convex_hull_extrapolation_mass"] >= 0.0


def test_ctl2_factorial_arm_enumeration_is_exact() -> None:
    cfg = synthetic_ctl2_config(arm_design="factorial_3x3", direction_support_mode="both")
    arms = enumerate_ctl2_arms(cfg)

    assert len(arms) == 18
    assert len({arm.arm_id for arm in arms}) == 18
    assert Ctl2Arm("first_order", "lifted_interaction", "projected") in arms

    calibrated = enumerate_ctl2_arms(
        synthetic_ctl2_config(
            arm_design="factorial_2x2",
            include_response_gain_calibration=True,
        )
    )
    assert len(calibrated) == 8
    assert Ctl2Arm(
        "first_order",
        "lifted_interaction",
        estimator_calibration="response_gain",
    ) in calibrated


def test_ctl2_response_calibration_fits_deltas_and_preserves_initial_estimate() -> None:
    env = synthetic_ctl2_env()
    estimator = AffineStateEstimator(
        "first_order",
        0.3,
        np.asarray([0.25, 1.0]),
    )
    env["estimators"]["first_order"] = estimator
    direction = env["observers"]["first_order"]["direction"]
    fitted = _response_gain_calibration(
        synthetic_ctl2_config(),
        env,
        estimator,
        direction,
    )

    expected_scale = float((env["target_vec"] @ direction) / (estimator.gradient @ direction))
    assert fitted["response_calibration_scale"] == pytest.approx(expected_scale)
    assert fitted["response_calibration_corrected_mae"] < 1e-12

    raw_metrics, _raw_traj, _raw_examples, raw_steps = _rollout_arm(
        synthetic_ctl2_config(loop_steps=2),
        env,
        Ctl2Arm("first_order", "first_order"),
    )
    calibrated_metrics, _cal_traj, _cal_examples, calibrated_steps = _rollout_arm(
        synthetic_ctl2_config(loop_steps=2),
        env,
        Ctl2Arm("first_order", "first_order", estimator_calibration="response_gain"),
    )

    raw_initial = raw_steps[raw_steps["step"] == 0]["observer_estimate"].to_numpy()
    calibrated_initial = calibrated_steps[calibrated_steps["step"] == 0]["observer_estimate"].to_numpy()
    assert np.allclose(raw_initial, calibrated_initial)
    assert raw_metrics["observer_self_gain"] == pytest.approx(estimator.gradient @ direction)
    assert calibrated_metrics["observer_self_gain"] == pytest.approx(env["target_vec"] @ direction)


def test_ctl2_crossed_estimator_and_direction_are_independent() -> None:
    env = synthetic_ctl2_env()
    env["estimators"]["first_order"] = AffineStateEstimator(
        "first_order",
        0.0,
        np.asarray([-1.0, 0.0]),
    )
    metrics, *_ = _rollout_arm(
        synthetic_ctl2_config(loop_steps=1),
        env,
        Ctl2Arm("first_order", "oracle_target"),
    )

    assert metrics["effective_target_gain"] == pytest.approx(1.0)
    assert metrics["observer_self_gain"] == pytest.approx(-1.0)
    assert metrics["observer_direction_sign_compatible"] is False
    assert metrics["observer_error_pole_unsaturated"] == pytest.approx(1.5)


def test_ctl2_unsaturated_observer_error_follows_reported_pole() -> None:
    metrics, _traj, _per_example, per_step = rollout_synthetic("first_order", loop_steps=3)
    pole = float(metrics["observer_error_pole_unsaturated"])
    one = per_step[per_step["example_idx"] == 0].sort_values("step")
    errors = (one["target_ref"] - one["observer_estimate"]).to_numpy()

    assert np.allclose(errors[1:], pole * errors[:-1])


def test_ctl2_omitted_response_condition_matches_affine_rollout() -> None:
    observer_errors = []
    target_changes = {}
    for rho in (-1.5, -1.0, 0.0, 0.5):
        env = deepcopy(synthetic_ctl2_env())
        env["h_test"][:, 1] = 0.0
        env["h_calibration"] = env["h_test"].copy()
        env["target_vec"] = np.asarray([1.0, 1.0])
        env["target_bias"] = 0.0
        target = env["h_test"] @ env["target_vec"]
        env["test"] = {"target": target.copy(), "target_clean": target.copy()}
        env["calibration"] = {"target": target.copy(), "target_clean": target.copy()}
        env["test_pred"] = target.copy()
        env["estimators"]["first_order"] = AffineStateEstimator(
            "first_order",
            0.0,
            np.asarray([1.0, 0.0]),
        )
        direction = np.asarray([1.0, rho])
        record = env["observers"]["first_order"]
        record["estimator"] = env["estimators"]["first_order"]
        record["direction"] = direction
        record["directions"] = {"unprojected": direction, "projected": direction}
        record["raw_direction"] = direction
        record["raw_target_gain"] = float(env["target_vec"] @ direction)
        env["support"] = fit_affine_support(env["h_test"], relative_tolerance=1e-8)
        env["clean_residual_centroids"] = env["h_test"].copy()

        metrics, _traj, _examples, per_step = _rollout_arm(
            synthetic_ctl2_config(
                loop_steps=12,
                controller_gain=0.5,
                max_strength=10.0,
                use_relative_target=True,
                relative_target_offset=1.0,
            ),
            env,
            Ctl2Arm("first_order", "first_order"),
        )
        one = per_step[per_step["example_idx"] == 0].sort_values("step")
        observer_errors.append(
            (one["target_ref"] - one["observer_estimate"]).to_numpy()
        )
        target_changes[rho] = float(one["target"].iloc[-1] - one["target"].iloc[0])

        assert metrics["normalized_omitted_response"] == pytest.approx(rho)
        assert metrics["true_to_observer_response_ratio"] == pytest.approx(1.0 + rho)
        assert metrics["affine_true_error_prediction_residual_mae"] < 1e-12
        assert metrics["control_clip_fraction"] == 0.0

    assert all(np.allclose(observer_errors[0], errors) for errors in observer_errors[1:])
    assert abs(target_changes[-1.0]) < 1e-12
    assert target_changes[-1.5] < 0.0


def test_ctl2_negative_self_gain_is_ineligible_for_gain_matching() -> None:
    env = synthetic_ctl2_env()
    env["estimators"]["first_order"] = AffineStateEstimator(
        "first_order",
        0.0,
        np.asarray([-1.0, 0.0]),
    )
    with pytest.raises(GainMatchIneligible, match="non-positive"):
        _rollout_arm(
            synthetic_ctl2_config(loop_steps=1),
            env,
            Ctl2Arm("first_order", "oracle_target", controller_mode="gain_matched"),
        )


def test_ctl2_clipping_fraction_excludes_terminal_record() -> None:
    metrics, _traj, _per_example, per_step = rollout_synthetic(
        "first_order",
        loop_steps=2,
        controller_gain=10.0,
        max_strength=0.05,
    )
    transitions = per_step[per_step["step"] < 2]
    terminal = per_step[per_step["step"] == 2]

    assert metrics["control_clip_fraction"] == pytest.approx(
        transitions["next_control_clipped"].astype(float).mean()
    )
    assert not terminal["next_control_clipped"].any()


def test_ctl2_genuine_loop_without_training_reads_current_state() -> None:
    _metrics, _traj, _per_example, per_step = rollout_synthetic("first_order")
    step0 = per_step[per_step["step"] == 0].set_index("example_idx")
    step1 = per_step[per_step["step"] == 1].set_index("example_idx")
    joined = step0.join(step1, lsuffix="_0", rsuffix="_1")
    controlled = joined["next_control_strength_0"].abs() > 1e-8

    assert controlled.any()
    assert (joined.loc[controlled, "residual_l2_delta_from_prev_step_1"] > 1e-8).all()
    assert (joined.loc[controlled, "target_1"] - joined.loc[controlled, "target_0"]).abs().min() > 1e-8
    assert (
        joined.loc[controlled, "observer_estimate_1"] - joined.loc[controlled, "observer_estimate_0"]
    ).abs().min() > 1e-8
    assert np.allclose(joined.loc[controlled, "observer_estimate_1"], joined.loc[controlled, "target_1"])


def test_ctl2_oracle_remains_stable_without_training() -> None:
    _metrics, _traj, _per_example, per_step = rollout_synthetic("oracle_target")
    oracle = per_step.sort_values(["example_idx", "step"])

    for _example_idx, group in oracle.groupby("example_idx"):
        after_first = group.query("step >= 1")["target_abs_error"].to_numpy()
        assert (after_first[1:] <= after_first[:-1] + 1e-12).all()


def test_ctl2_gamma_zero_first_order_and_lifted_match_without_training() -> None:
    fo_metrics, *_ = rollout_synthetic("first_order", gamma=0.0)
    li_metrics, *_ = rollout_synthetic("lifted_interaction", gamma=0.0)

    for metric in [
        "integrated_squared_error",
        "final_target_mse",
        "cumulative_collateral_abs",
        "divergence_rate",
        "divergence_rate_mse_growth",
        "target_error_worsened_rate",
    ]:
        assert abs(float(fo_metrics[metric]) - float(li_metrics[metric])) < 1e-12


@pytest.mark.slow
def test_ctl2_genuine_loop_reads_current_edited_state(tmp_path: Path) -> None:
    outdir = run_ctl2(tmp_path, dirname="loop")
    per_step = pd.read_csv(outdir / "trained_transformer_ctl2_per_step_examples.csv")
    first_order = per_step[per_step["observer"] == "first_order"]

    step0 = first_order[first_order["step"] == 0].set_index("example_idx")
    step1 = first_order[first_order["step"] == 1].set_index("example_idx")
    joined = step0.join(step1, lsuffix="_0", rsuffix="_1")
    controlled = joined["next_control_strength_0"].abs() > 1e-8

    assert controlled.any()
    assert (joined.loc[controlled, "residual_l2_delta_from_prev_step_1"] > 1e-8).any()
    assert (joined.loc[controlled, "target_1"] - joined.loc[controlled, "target_0"]).abs().max() > 1e-8
    assert (
        joined.loc[controlled, "observer_estimate_1"] - joined.loc[controlled, "observer_estimate_0"]
    ).abs().max() > 1e-8


@pytest.mark.slow
def test_ctl2_oracle_target_is_stable_at_configured_gain(tmp_path: Path) -> None:
    outdir = run_ctl2(tmp_path, dirname="oracle")
    per_step = pd.read_csv(outdir / "trained_transformer_ctl2_per_step_examples.csv")
    oracle = per_step[per_step["observer"] == "oracle_target"]

    for _example_idx, group in oracle.groupby("example_idx"):
        after_first = group.sort_values("step").query("step >= 1")["target_abs_error"].to_numpy()
        if len(after_first) > 1:
            assert (after_first[1:] <= after_first[:-1] + 1e-8).all()


@pytest.mark.slow
def test_ctl2_gamma_zero_first_order_and_lifted_match(tmp_path: Path) -> None:
    outdir = run_ctl2(
        tmp_path,
        dirname="gamma_zero",
        gamma=0.0,
        nuisance_interaction_weight=0.0,
        seed=1,
        train_steps=30,
    )
    results = pd.read_csv(outdir / "trained_transformer_ctl2_results.csv").set_index("observer")
    first_order = results.loc["first_order"]
    lifted = results.loc["lifted_interaction"]

    for metric in [
        "integrated_squared_error",
        "final_target_mse",
        "cumulative_collateral_abs",
        "divergence_rate",
        "divergence_rate_mse_growth",
        "target_error_worsened_rate",
    ]:
        denom = max(abs(float(lifted[metric])), 1.0)
        assert abs(float(first_order[metric]) - float(lifted[metric])) / denom < 0.35


@pytest.mark.slow
def test_ctl2_per_step_outputs_have_multiple_steps_per_example(tmp_path: Path) -> None:
    outdir = run_ctl2(tmp_path, dirname="outputs")
    per_step_path = outdir / "trained_transformer_ctl2_per_step_examples.csv"

    assert per_step_path.exists()
    per_step = pd.read_csv(per_step_path)
    assert per_step.groupby(["observer", "example_idx"])["step"].nunique().min() > 1
    assert (outdir / "ctl2_target_mse_fan.png").exists()
    assert (outdir / "ctl2_collateral_fan.png").exists()
    assert (outdir / "ctl2_observer_bias_fan.png").exists()


@pytest.mark.slow
def test_ctl2_compact_sweep_outputs_keep_summaries_and_provenance(tmp_path: Path) -> None:
    outdir = run_ctl2(
        tmp_path,
        dirname="compact",
        arm_design="factorial_2x2",
        direction_support_mode="both",
        include_oracle=False,
        write_per_example_outputs=False,
        write_per_step_examples=False,
        write_observer_cards=False,
        write_plots=False,
    )

    assert not (outdir / "trained_transformer_ctl2_per_example.csv").exists()
    assert not (outdir / "trained_transformer_ctl2_per_step_examples.csv").exists()
    assert (outdir / "trained_transformer_ctl2_state_archetypes.csv").exists()
    assert (outdir / "trained_transformer_ctl2_factorial_paired_summary.csv").exists()
    assert (outdir / "trained_transformer_ctl2_run_manifest.json").exists()
