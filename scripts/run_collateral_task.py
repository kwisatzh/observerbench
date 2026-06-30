#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from observerbench.tasks.registry import run_registered_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analytic Ctl-1 collateral task.")
    parser.add_argument("--outdir", default="runs/collateral_task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=4000)
    parser.add_argument("--gamma", type=float, default=1.15)
    parser.add_argument("--nuisance-interaction-weight", type=float, default=0.0)
    parser.add_argument("--include-interaction-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "task": "ctl1_analytic",
        "seed": args.seed,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "gamma": args.gamma,
        "nuisance_interaction_weight": args.nuisance_interaction_weight,
        "include_interaction_only": args.include_interaction_only,
    }
    outdir = Path(args.outdir)
    results = run_registered_task("ctl1_analytic", config, Path("configs/ctl1_analytic.yaml"), outdir)
    print(f"\nWrote results to {outdir}\n")
    print(pd.DataFrame([{"observer": r.observer, **r.metrics} for r in results]).to_string(index=False))


if __name__ == "__main__":
    main()
