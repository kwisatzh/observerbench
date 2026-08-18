"""Tests for the capacity-matched IOI Phase-2 analysis.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from observerbench.tasks.ioi.phase2_capacity import (
    IOIPhase2Config,
    build_capacity_design,
    load_head_subset_run,
    run_capacity_analysis,
)
from observerbench.tasks.ioi.stage2c import write_stage2c_fixture


ROOT = Path(__file__).resolve().parents[1]


def test_pair_blocks_have_matched_four_rank_in_fixture(tmp_path: Path) -> None:
    source = write_stage2c_fixture(tmp_path / "input", n_prompts=8, seed=0)
    run = load_head_subset_run(source)
    base, _ = build_capacity_design(run, "count_additive")
    base_rank = int(np.linalg.matrix_rank(base))

    for pair in ("PB", "PE", "BE"):
        design, columns = build_capacity_design(
            run,
            f"count_plus_{pair}_bin4",
        )
        assert len(columns) == base.shape[1] + 4
        assert int(np.linalg.matrix_rank(design)) == base_rank + 4


def test_capacity_analysis_runs_on_synthetic_fixture(tmp_path: Path) -> None:
    source = write_stage2c_fixture(tmp_path / "input", n_prompts=8, seed=0)
    out = tmp_path / "output"
    run_capacity_analysis(
        source,
        out,
        label="synthetic test",
        config=IOIPhase2Config(
            k_folds=3,
            cv_repeats=2,
            bootstrap_repeats=8,
            seed=0,
        ),
        include_mobius=True,
    )

    assert (out / "model_comparison.csv").exists()
    assert (out / "add_one_vs_count_additive.csv").exists()
    assert (out / "leave_one_out_contrasts.csv").exists()
    assert (out / "mobius_bootstrap_summary.csv").exists()
    audit = pd.read_csv(out / "capacity_audit.csv")
    singles = audit[audit["model"].isin(
        [
            "count_plus_PB_bin4",
            "count_plus_PE_bin4",
            "count_plus_BE_bin4",
        ]
    )]
    assert set(singles["rank_added_vs_count_additive"]) == {4}


def test_checked_phase2_outputs_pin_revised_claim() -> None:
    base = ROOT / "results/revision/phase02"
    stage2b = base / "ioi_stage2b_capacity"
    stage2c = base / "ioi_stage2c_capacity"

    b_count = pd.read_csv(stage2b / "add_one_vs_count_additive.csv").set_index("contrast")
    b_add = pd.read_csv(stage2b / "add_one_vs_additive_head.csv").set_index("contrast")
    c_add = pd.read_csv(stage2c / "add_one_vs_additive_head.csv").set_index("contrast")
    c_loo = pd.read_csv(stage2c / "leave_one_out_contrasts.csv").set_index("contrast")
    mobius = pd.read_csv(stage2c / "mobius_bootstrap_summary.csv").set_index("term")

    assert b_count.loc["add_PE_bin4_vs_count", "q05"] > 0
    assert b_add.loc["add_PE_bin4_vs_additive", "q05"] > 0
    assert b_add.loc["add_all_bin4_vs_additive", "q05"] > 0
    assert c_add.loc["add_PE_bin4_vs_additive", "q05"] > 0
    assert c_add.loc["add_PB_bin4_vs_additive", "q95"] < 0
    assert c_loo.loc["remove_PB_bin4", "q05"] > 0
    assert c_loo.loc["remove_PE_bin4", "mean"] > c_loo.loc["remove_PB_bin4", "mean"]
    assert mobius.loc["PE_minus_PB", "q05"] > 0
