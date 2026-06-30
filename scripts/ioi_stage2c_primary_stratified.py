#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.stage2c import IOIStage2cConfig, run_ioi_stage2c


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IOI Stage 2c primary-stratified diagnostic.")
    parser.add_argument("--outdir", default="runs/ioi_stage2c")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ablation", default="mean")
    parser.add_argument("--n-prompts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = IOIStage2cConfig(quick=args.quick, ablation=args.ablation, n_prompts=args.n_prompts, seed=args.seed)
    run_ioi_stage2c(cfg, Path(args.outdir))
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
