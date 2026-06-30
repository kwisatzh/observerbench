from __future__ import annotations

from pathlib import Path

import pandas as pd

from observerbench.tasks.registry import run_registered_task


def tiny_ctl2_config(**overrides):
    config = {
        "task": "trained_ctl2",
        "mode": "quick",
        "quick": False,
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


def test_ctl2_oracle_target_is_stable_at_configured_gain(tmp_path: Path) -> None:
    outdir = run_ctl2(tmp_path, dirname="oracle")
    per_step = pd.read_csv(outdir / "trained_transformer_ctl2_per_step_examples.csv")
    oracle = per_step[per_step["observer"] == "oracle_target"]

    for _example_idx, group in oracle.groupby("example_idx"):
        after_first = group.sort_values("step").query("step >= 1")["target_abs_error"].to_numpy()
        if len(after_first) > 1:
            assert (after_first[1:] <= after_first[:-1] + 1e-8).all()


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


def test_ctl2_per_step_outputs_have_multiple_steps_per_example(tmp_path: Path) -> None:
    outdir = run_ctl2(tmp_path, dirname="outputs")
    per_step_path = outdir / "trained_transformer_ctl2_per_step_examples.csv"

    assert per_step_path.exists()
    per_step = pd.read_csv(per_step_path)
    assert per_step.groupby(["observer", "example_idx"])["step"].nunique().min() > 1
    assert (outdir / "ctl2_target_mse_fan.png").exists()
    assert (outdir / "ctl2_collateral_fan.png").exists()
    assert (outdir / "ctl2_observer_bias_fan.png").exists()
