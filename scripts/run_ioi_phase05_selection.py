"""Run the exploratory Phase 5 held-out IOI intervention selection audit.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.selection import IOISelectionConfig, run_selection_analysis


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
        default=ROOT / "results/revision/phase05/ioi_selection_pilot",
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=19000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = dict(repetitions=args.repetitions, seed=args.seed)
    run_selection_analysis(
        args.stage2b_input,
        args.outdir / "broad",
        label="existing Stage 2b anchored broad masks",
        config=IOISelectionConfig(
            measurement_budgets=(20, 40, 80, 120),
            **common,
        ),
    )
    run_selection_analysis(
        args.stage2c_input,
        args.outdir / "primary_stratified",
        label="existing Stage 2c primary-stratified masks",
        config=IOISelectionConfig(
            measurement_budgets=(20, 40, 80, 160),
            **common,
        ),
    )


if __name__ == "__main__":
    main()

