from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from pathlib import Path

import pandas as pd

from observerbench.tasks.ctl1_analytic import CollateralTaskConfig, run_task


def test_ctl1_analytic_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "ctl1_analytic"
    results = run_task(
        CollateralTaskConfig(
            seed=0,
            n_train=128,
            n_test=96,
            gamma=1.15,
            nuisance_interaction_weight=0.0,
        ),
        outdir,
    )

    assert {result.observer for result in results} == {"first_order", "lifted_interaction"}
    result_csv = outdir / "collateral_task_results.csv"
    assert result_csv.exists()
    df = pd.read_csv(result_csv)
    assert set(df["observer"]) == {"first_order", "lifted_interaction"}
    assert (df["control_target_mse"] >= 0.0).all()
    assert (df["collateral_abs_delta"] >= 0.0).all()
    assert (outdir / "collateral_task_target_vs_collateral.png").exists()


def test_ctl1_manifold_respecting_check_recomputes_product(tmp_path: Path) -> None:
    outdir = tmp_path / "ctl1_manifold"
    results = run_task(
        CollateralTaskConfig(
            seed=1,
            n_train=256,
            n_test=128,
            gamma=1.15,
            include_manifold_respecting=True,
        ),
        outdir,
    )

    manifold = [result for result in results if result.task == "collateral_base_manifold_control"]
    assert {result.observer for result in manifold} == {"first_order", "lifted_interaction"}
    assert all(result.metrics["manifold_constraint_error_max"] < 1e-12 for result in manifold)
    examples = pd.read_csv(outdir / "collateral_task_manifold_examples.csv")
    assert (
        examples["interaction_after"] - examples["x1_after"] * examples["x2_after"]
    ).abs().max() < 1e-12
    assert (outdir / "collateral_task_manifold_target_vs_collateral.png").exists()
