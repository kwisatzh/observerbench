"""Freeze Phase-6 train-only observer predictions and fixed actions.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.phase6_freeze import (
    Phase6FreezeConfig,
    freeze_phase6_predictions_and_actions,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit only from the sealed Phase-6 train/calibration table, then hash "
            "all candidate predictions and fixed actions before test inference."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase06_fresh_confirmation_v1.json",
    )
    parser.add_argument(
        "--design-dir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/design",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/calibration",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/prediction_freeze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = freeze_phase6_predictions_and_actions(
        args.design_dir,
        args.calibration_dir,
        args.outdir,
        protocol_path=args.protocol,
        config=Phase6FreezeConfig(),
    )
    print(f"Wrote immutable Phase-6 prediction/action seal to {output}")


if __name__ == "__main__":
    main()
