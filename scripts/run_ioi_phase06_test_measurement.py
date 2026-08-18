"""Measure the complete sealed Phase-6 IOI held-out candidate surface.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.phase6_test_measurement import (
    Phase6TestMeasurementConfig,
    run_phase6_test_measurement,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT / "results/revision/phase06/ioi_fresh_confirmation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "After the independent seal audit, measure every frozen Phase-6 test "
            "prompt and candidate mask. This command never fits or selects an action."
        )
    )
    parser.add_argument("--design-dir", type=Path, default=PHASE6 / "design")
    parser.add_argument("--calibration-dir", type=Path, default=PHASE6 / "calibration")
    parser.add_argument("--freeze-dir", type=Path, default=PHASE6 / "prediction_freeze")
    parser.add_argument("--outdir", type=Path, default=PHASE6 / "test_measurement")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/revision/ioi_phase06_fresh_confirmation_v1.json",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--clean-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_phase6_test_measurement(
        args.design_dir,
        args.calibration_dir,
        args.freeze_dir,
        args.outdir,
        protocol_path=args.protocol,
        config=Phase6TestMeasurementConfig(
            device=args.device,
            pair_batch_size=args.pair_batch_size,
            clean_batch_size=args.clean_batch_size,
        ),
    )
    print(f"Wrote sealed Phase-6 held-out measurements to {output}")


if __name__ == "__main__":
    main()
