#!/usr/bin/env python3
"""Run the compact multi-seed Ctl-2 estimator--direction sweep.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from observerbench.config import load_config
from observerbench.core import write_json
from observerbench.provenance import (
    json_sha256,
    portable_artifact_path,
    runtime_provenance,
)
from observerbench.tasks.trained_ctl2 import (
    TrainedTransformerCtl2Config,
    run_trained_transformer_ctl2,
)
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "revision" / "trained_ctl2_phase01_default_v3.yaml"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "revision" / "phase01" / "ctl2_multiseed_sweep"


def _gamma_label(gamma: float) -> str:
    return f"gamma_{gamma:.2f}".replace("-", "m").replace(".", "p")


def _config(base: dict, *, seed: int, gamma: float, device: str | None) -> TrainedTransformerCtl2Config:
    names = {item.name for item in fields(TrainedTransformerCtl2Config)}
    kwargs = {key: value for key, value in base.items() if key in names}
    kwargs.update({
        "seed": int(seed),
        "gamma": float(gamma),
        "arm_design": "factorial_2x2",
        "direction_support_mode": "both",
        "include_oracle": False,
        "include_affine_calibration": False,
        "include_response_gain_calibration": True,
        "include_gain_matched_control": False,
        "write_per_example_outputs": False,
        "write_per_step_examples": False,
        "write_observer_cards": False,
        "write_plots": False,
    })
    if device is not None:
        kwargs["device"] = device
    return TrainedTransformerCtl2Config(**kwargs)


def _collect(outdir: Path, gammas: list[float], seeds: list[int]) -> None:
    result_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    for gamma in gammas:
        for seed in seeds:
            run_dir = outdir / _gamma_label(gamma) / f"seed_{seed}"
            result = pd.read_csv(run_dir / "trained_transformer_ctl2_results.csv")
            result = pd.concat(
                [
                    pd.DataFrame({"sweep_gamma": [gamma] * len(result), "seed": [seed] * len(result)}),
                    result.reset_index(drop=True),
                ],
                axis=1,
            )
            result_frames.append(result)
            contrast = pd.read_csv(run_dir / "trained_transformer_ctl2_factorial_contrasts.csv")
            contrast = pd.concat(
                [
                    pd.DataFrame({"sweep_gamma": [gamma] * len(contrast), "seed": [seed] * len(contrast)}),
                    contrast.reset_index(drop=True),
                ],
                axis=1,
            )
            contrast_frames.append(contrast)
    pd.concat(result_frames, ignore_index=True).to_csv(outdir / "phase01_sweep_results.csv", index=False)
    pd.concat(contrast_frames, ignore_index=True).to_csv(
        outdir / "phase01_sweep_factorial_contrasts.csv",
        index=False,
    )


def _seed_summary(frame: pd.DataFrame, group: list[str], metric: str) -> pd.DataFrame:
    return frame.groupby(group, as_index=False)[metric].agg(
        mean="mean",
        seed_min="min",
        seed_max="max",
    )


def _plot_sweep(outdir: Path) -> None:
    frame = pd.read_csv(outdir / "phase01_sweep_results.csv")
    primary = frame[
        (frame["estimator_calibration"] == "none")
        & (frame["controller_mode"] == "base")
    ]
    summary = _seed_summary(
        primary,
        ["sweep_gamma", "direction_support_mode", "estimator_name", "direction_name"],
        "integrated_squared_error",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    colors = {"first_order": "tab:red", "lifted_interaction": "tab:blue"}
    linestyles = {"first_order": "-", "lifted_interaction": "--"}
    for ax, support_mode in zip(axes, ("unprojected", "projected")):
        panel = summary[summary["direction_support_mode"] == support_mode]
        for (estimator, direction), cell in panel.groupby(["estimator_name", "direction_name"]):
            cell = cell.sort_values("sweep_gamma")
            ax.plot(
                cell["sweep_gamma"],
                cell["mean"],
                color=colors[estimator],
                linestyle=linestyles[direction],
                marker="o",
                label=f"est={estimator}; dir={direction}",
            )
            ax.fill_between(
                cell["sweep_gamma"],
                cell["seed_min"],
                cell["seed_max"],
                color=colors[estimator],
                alpha=0.10,
            )
        ax.set_title(f"{support_mode} directions")
        ax.set_xlabel("interaction strength gamma")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("integrated target squared error")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Ctl-2 factorial: mean and training-seed min--max")
    fig.tight_layout()
    fig.savefig(outdir / "ctl2_phase01_factorial_ise_by_gamma.png", dpi=180)
    plt.close(fig)

    diagonal = frame[
        (frame["direction_support_mode"] == "unprojected")
        & (frame["controller_mode"] == "base")
        & (frame["estimator_name"] == frame["direction_name"])
    ]
    calibration = _seed_summary(
        diagonal,
        ["sweep_gamma", "estimator_name", "estimator_calibration"],
        "integrated_squared_error",
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for (estimator, mode), cell in calibration.groupby(["estimator_name", "estimator_calibration"]):
        cell = cell.sort_values("sweep_gamma")
        ax.plot(
            cell["sweep_gamma"],
            cell["mean"],
            color=colors[estimator],
            linestyle="-" if mode == "none" else ":",
            marker="o",
            label=f"{estimator}; {mode}",
        )
        ax.fill_between(
            cell["sweep_gamma"],
            cell["seed_min"],
            cell["seed_max"],
            color=colors[estimator],
            alpha=0.10,
        )
    ax.set_xlabel("interaction strength gamma")
    ax.set_ylabel("integrated target squared error")
    ax.set_title("Ctl-2 unprojected diagonal: finite-response calibration")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ctl2_phase01_response_calibration_by_gamma.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.15, 1.5])
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base = load_config(args.base_config)
    args.outdir.mkdir(parents=True, exist_ok=True)
    planned = [
        (gamma, seed, args.outdir / _gamma_label(gamma) / f"seed_{seed}")
        for gamma in args.gammas
        for seed in args.seeds
    ]
    existing = [path for _gamma, _seed, path in planned if (path / "trained_transformer_ctl2_results.csv").exists()]
    if existing and not args.resume:
        raise SystemExit(
            "Refusing to overwrite completed runs; pass --resume to keep them and run only missing cells."
        )

    for gamma, seed, run_dir in planned:
        if (run_dir / "trained_transformer_ctl2_results.csv").exists():
            print(f"keep gamma={gamma:g} seed={seed}: {run_dir}")
            continue
        print(f"run gamma={gamma:g} seed={seed}: {run_dir}")
        cfg = _config(base, seed=seed, gamma=gamma, device=args.device)
        run_trained_transformer_ctl2(cfg, run_dir)

    _collect(args.outdir, list(args.gammas), list(args.seeds))
    _plot_sweep(args.outdir)
    write_json(args.outdir / "phase01_sweep_manifest.json", {
        "schema": "observerbench.ctl2.phase01_sweep.v0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_provenance(REPO_ROOT),
        "base_config": portable_artifact_path(args.base_config, REPO_ROOT),
        "base_config_sha256": json_sha256(base),
        "gammas": list(args.gammas),
        "seeds": list(args.seeds),
        "n_runs": len(planned),
        "arm_design": "factorial_2x2",
        "direction_support_mode": "both",
        "calibration_modes": ["none", "response_gain"],
        "uncertainty_unit": "training seed",
        "per_example_rows_are_not_independent": True,
    })
    print(args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
