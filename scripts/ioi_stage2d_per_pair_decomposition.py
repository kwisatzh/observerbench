#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.stage2d import IOIStage2dConfig, run_ioi_stage2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IOI Stage 2d per-pair decomposition.")
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--outdir", default="runs/ioi_stage2d")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = IOIStage2dConfig(
        input_run=args.input_run,
        k_folds=args.k_folds,
        bootstrap_repeats=args.bootstrap_repeats,
        ridge=args.ridge,
        seed=args.seed,
    )
    run_ioi_stage2d(cfg, Path(args.outdir))
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
