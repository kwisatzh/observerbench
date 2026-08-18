"""Evaluate the sealed prospective Phase-6 IOI confirmation.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.phase6_evaluation import evaluate_phase6_confirmation


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify every Phase-6 seal, then evaluate the already-frozen actions "
            "under the prespecified H1, H2, sensitivity, and clean-task gates."
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
        "--freeze-dir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/prediction_freeze",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/test_measurement",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/phase06/ioi_fresh_confirmation/evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = evaluate_phase6_confirmation(
        args.design_dir,
        args.calibration_dir,
        args.freeze_dir,
        args.test_dir,
        args.outdir,
        protocol_path=args.protocol,
    )
    print(f"Wrote sealed prospective Phase-6 evaluation to {output}")


if __name__ == "__main__":
    main()
