#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import argparse
from pathlib import Path

from observerbench.tasks.ioi.stage1 import IOIStage1Config, run_ioi_stage1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IOI Stage 1 self-repair diagnostic.")
    parser.add_argument("--outdir", default="runs/ioi_stage1")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ablation", default="mean")
    parser.add_argument("--n-prompts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = IOIStage1Config(quick=args.quick, ablation=args.ablation, n_prompts=args.n_prompts, seed=args.seed)
    run_ioi_stage1(cfg, Path(args.outdir))
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
