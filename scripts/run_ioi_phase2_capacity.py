"""Run the capacity-matched IOI Phase-2 re-analysis.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.phase2_capacity import (
    IOIPhase2Config,
    run_capacity_analysis,
    write_combined_claim_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2b-input",
        type=Path,
        default=ROOT / "results/revision/phase02/inputs/stage2b_mean_end",
    )
    parser.add_argument(
        "--stage2c-input",
        type=Path,
        default=ROOT / "results/frozen/ioi/stage2c_primary_stratified_mean_end",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/phase02",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = IOIPhase2Config(
        k_folds=args.k_folds,
        cv_repeats=args.cv_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    stage2b_out = args.outdir / "ioi_stage2b_capacity"
    stage2c_out = args.outdir / "ioi_stage2c_capacity"
    run_capacity_analysis(
        args.stage2b_input,
        stage2b_out,
        label="Stage 2b anchored broad random design",
        config=config,
        include_design_sensitivity=True,
    )
    run_capacity_analysis(
        args.stage2c_input,
        stage2c_out,
        label="Stage 2c primary-stratified design",
        config=config,
        include_mobius=True,
    )
    write_combined_claim_audit(stage2b_out, stage2c_out, args.outdir)


if __name__ == "__main__":
    main()
