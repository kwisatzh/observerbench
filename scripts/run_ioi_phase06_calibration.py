"""Measure only Phase-6 reference means and train/calibration effects.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.phase6_measurement import (
    Phase6CalibrationConfig,
    run_phase6_calibration,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the approved Phase-6 reference + train/calibration stage. "
            "This command cannot evaluate test prompts or candidate masks."
        )
    )
    parser.add_argument(
        "--design-dir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/design",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/calibration",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--reference-batch-size", type=int, default=64)
    parser.add_argument("--mask-shard-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_phase6_calibration(
        args.design_dir,
        args.outdir,
        config=Phase6CalibrationConfig(
            device=args.device,
            pair_batch_size=args.pair_batch_size,
            reference_batch_size=args.reference_batch_size,
            mask_shard_size=args.mask_shard_size,
        ),
    )
    print(f"Wrote Phase-6 train/calibration measurements to {output}")


if __name__ == "__main__":
    main()
