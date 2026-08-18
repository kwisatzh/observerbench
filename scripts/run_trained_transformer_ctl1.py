#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from observerbench.tasks.registry import run_registered_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained-transformer Ctl-1 task.")
    parser.add_argument("--outdir", default="runs/trained_transformer_ctl1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--d-mlp", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--nuisance-interaction-weight", type=float, default=None)
    parser.add_argument("--target-ref", type=float, default=None)
    parser.add_argument("--target-actuation-gain", type=float, default=None)
    parser.add_argument("--direction-normalization", choices=["parallel_orthogonal", "target_gain_scale"], default=None)
    parser.add_argument("--orthogonal-scale", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {"task": "trained_ctl1", "seed": args.seed, "device": args.device, "quick": args.quick}
    for cli_name, field_name in [
        ("n_train", "n_train"),
        ("n_test", "n_test"),
        ("train_steps", "train_steps"),
        ("d_model", "d_model"),
        ("n_layers", "n_layers"),
        ("n_heads", "n_heads"),
        ("d_mlp", "d_mlp"),
        ("batch_size", "batch_size"),
        ("gamma", "gamma"),
        ("nuisance_interaction_weight", "nuisance_interaction_weight"),
        ("target_ref", "target_ref"),
        ("target_actuation_gain", "target_actuation_gain"),
        ("direction_normalization", "direction_normalization"),
        ("orthogonal_scale", "orthogonal_scale"),
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            config[field_name] = value

    outdir = Path(args.outdir)
    results = run_registered_task("trained_ctl1", config, Path("configs/trained_ctl1.yaml"), outdir)
    print(f"\nWrote results to {outdir}\n")
    print(pd.DataFrame([{"observer": r.observer, **r.metrics} for r in results]).to_string(index=False))


if __name__ == "__main__":
    main()
