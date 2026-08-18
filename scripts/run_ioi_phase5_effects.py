"""Measure the locked Phase-5 GPT-2 IOI intervention design.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.phase5_effects import (
    IOIPhase5EffectConfig,
    run_ioi_phase5_effects,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design-dir",
        type=Path,
        default=ROOT / "results/revision/phase05/design",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/phase05/ioi_effects",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--reference-batch-size", type=int, default=64)
    parser.add_argument("--mask-shard-size", type=int, default=16)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small design prefix and mark all outputs as non-claim smoke data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ioi_phase5_effects(
        args.design_dir,
        args.outdir,
        config=IOIPhase5EffectConfig(
            device=args.device,
            pair_batch_size=args.pair_batch_size,
            reference_batch_size=args.reference_batch_size,
            mask_shard_size=args.mask_shard_size,
            max_reference_prompts_per_template=4 if args.smoke else None,
            max_outcome_prompts_per_split=4 if args.smoke else None,
            max_masks=16 if args.smoke else None,
        ),
    )


if __name__ == "__main__":
    main()
