#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from observerbench.tasks.registry import run_registered_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-loop trained-transformer Ctl-2 task.")
    parser.add_argument("--outdir", default="runs/trained_transformer_ctl2")
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
    parser.add_argument("--loop-steps", type=int, default=None)
    parser.add_argument("--controller-gain", type=float, default=None)
    parser.add_argument("--max-strength", type=float, default=None)
    parser.add_argument("--target-ref", type=float, default=None)
    parser.add_argument("--target-actuation-gain", type=float, default=None)
    parser.add_argument("--direction-normalization", choices=["parallel_orthogonal", "target_gain_scale"], default=None)
    parser.add_argument("--orthogonal-scale", type=float, default=None)
    parser.add_argument("--aux-feature-weight", type=float, default=None)
    parser.add_argument("--use-relative-target", action="store_true")
    parser.add_argument("--relative-target-offset", type=float, default=None)
    parser.add_argument("--no-oracle", action="store_true", help="Disable oracle_target control arm")
    parser.add_argument("--divergence-abs-error-threshold", type=float, default=None)
    parser.add_argument("--divergence-initial-error-multiplier", type=float, default=None)
    parser.add_argument("--divergence-final-mse-multiplier", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {"task": "trained_ctl2", "seed": args.seed, "device": args.device, "quick": args.quick}
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
        ("loop_steps", "loop_steps"),
        ("controller_gain", "controller_gain"),
        ("max_strength", "max_strength"),
        ("target_ref", "target_ref"),
        ("target_actuation_gain", "target_actuation_gain"),
        ("direction_normalization", "direction_normalization"),
        ("orthogonal_scale", "orthogonal_scale"),
        ("aux_feature_weight", "aux_feature_weight"),
        ("relative_target_offset", "relative_target_offset"),
        ("divergence_abs_error_threshold", "divergence_abs_error_threshold"),
        ("divergence_initial_error_multiplier", "divergence_initial_error_multiplier"),
        ("divergence_final_mse_multiplier", "divergence_final_mse_multiplier"),
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            config[field_name] = value
    if args.use_relative_target:
        config["use_relative_target"] = True
    if args.no_oracle:
        config["include_oracle"] = False

    outdir = Path(args.outdir)
    results = run_registered_task("trained_ctl2", config, Path("configs/trained_ctl2.yaml"), outdir)
    print(f"\nWrote results to {outdir}\n")
    print(pd.DataFrame([{"observer": r.observer, **r.metrics} for r in results]).to_string(index=False))


if __name__ == "__main__":
    main()
