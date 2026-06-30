#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.stage2b import IOIStage2bConfig, run_ioi_stage2b


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IOI Stage 2b head-subset diagnostic.")
    parser.add_argument("--outdir", default="runs/ioi_stage2b")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ablation", default="mean")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = IOIStage2bConfig(quick=args.quick, ablation=args.ablation, seed=args.seed)
    run_ioi_stage2b(cfg, Path(args.outdir))
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
