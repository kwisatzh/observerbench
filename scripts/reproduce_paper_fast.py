#!/usr/bin/env python3
"""Regenerate ObserverBench paper figures and tables from frozen summaries.

This script is deliberately CPU-only. It reads CSV/JSON summaries under
results/frozen, writes paper/generated_figures, and never imports
TransformerLens, downloads GPT-2, or trains a model.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = REPO_ROOT / "results" / "frozen"
DEFAULT_OUTDIR = REPO_ROOT / "paper" / "generated_figures"
MPL_CACHE = REPO_ROOT / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Generated:
    result_id: str
    paths: tuple[Path, ...]


def _read_csv(frozen: Path, relpath: str) -> pd.DataFrame:
    path = frozen / relpath
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen input: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, outdir: Path, filename: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"None of these columns exist: {', '.join(names)}")


def _err(ax: plt.Axes, x, y, yerr=None, label=None, marker="o") -> None:
    ax.errorbar(x, y, yerr=yerr, marker=marker, linewidth=2, capsize=3, label=label)


def _plot_ratio_pair(
    df: pd.DataFrame,
    x_col: str,
    ratio_cols: tuple[str, ...],
    target_cols: tuple[str, ...],
    ratio_std_cols: tuple[str, ...],
    target_std_cols: tuple[str, ...],
    title: str,
    x_label: str,
    outdir: Path,
    filename: str,
) -> Path:
    ratio_col = _first_existing(df, ratio_cols)
    target_col = _first_existing(df, target_cols)
    ratio_std = df[_first_existing(df, ratio_std_cols)] if any(c in df.columns for c in ratio_std_cols) else None
    target_std = df[_first_existing(df, target_std_cols)] if any(c in df.columns for c in target_std_cols) else None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _err(axes[0], df[x_col], df[ratio_col], ratio_std, label="first-order / lifted")
    axes[0].axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("collateral ratio")
    axes[0].set_title("Collateral burden")
    axes[0].legend()

    _err(axes[1], df[x_col], df[target_col], target_std, label="first-order vs lifted")
    axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("target improvement fraction")
    axes[1].set_title("Target usefulness")
    axes[1].legend()

    fig.suptitle(title)
    return _save(fig, outdir, filename)


def figure_01(frozen: Path, outdir: Path) -> Generated:
    df = _read_csv(frozen, "ctl1_analytic/collateral_sweeps_v2/gamma_sweep_pairwise_summary.csv")
    path = _plot_ratio_pair(
        df,
        "gamma",
        ("fo_lifted_collateral_ratio_mean",),
        ("fo_target_improvement_fraction_of_lifted_mean",),
        ("fo_lifted_collateral_ratio_std",),
        ("fo_target_improvement_fraction_of_lifted_std",),
        "Figure 1: Analytic Ctl-1 interaction-strength sweep",
        "interaction strength gamma",
        outdir,
        "figure_01_ctl1_analytic_interaction_sweep.png",
    )
    return Generated("figure_01", (path,))


def figure_02(frozen: Path, outdir: Path) -> Generated:
    df = _read_csv(frozen, "ctl1_analytic/collateral_sweeps_v2/nuisance_weight_pairwise_summary.csv")
    path = _plot_ratio_pair(
        df,
        "nuisance_interaction_weight",
        ("fo_lifted_collateral_ratio_mean",),
        ("fo_target_improvement_fraction_of_lifted_mean",),
        ("fo_lifted_collateral_ratio_std",),
        ("fo_target_improvement_fraction_of_lifted_std",),
        "Figure 2: Analytic nuisance-placement sweep",
        "nuisance interaction weight",
        outdir,
        "figure_02_ctl1_analytic_nuisance_sweep.png",
    )
    return Generated("figure_02", (path,))


def figure_03(frozen: Path, outdir: Path) -> Generated:
    df = _read_csv(frozen, "trained_ctl1/trained_transformer_sweeps_v5_clean/gamma_sweep_pairwise_summary.csv")
    path = _plot_ratio_pair(
        df,
        "gamma",
        ("fo_lifted_collateral_ratio_median", "fo_lifted_collateral_ratio_mean"),
        ("fo_target_improvement_fraction_of_lifted_median", "fo_target_improvement_fraction_of_lifted_mean"),
        ("fo_lifted_collateral_ratio_std",),
        ("fo_target_improvement_fraction_of_lifted_std",),
        "Figure 3: Trained-transformer Ctl-1 interaction-strength sweep",
        "interaction strength gamma",
        outdir,
        "figure_03_trained_ctl1_interaction_sweep.png",
    )
    return Generated("figure_03", (path,))


def figure_04(frozen: Path, outdir: Path) -> Generated:
    df = _read_csv(
        frozen,
        "trained_ctl1/trained_transformer_sweeps_v5_clean/nuisance_weight_pairwise_summary.csv",
    )
    path = _plot_ratio_pair(
        df,
        "nuisance_interaction_weight",
        ("fo_lifted_collateral_ratio_median", "fo_lifted_collateral_ratio_mean"),
        ("fo_target_improvement_fraction_of_lifted_median", "fo_target_improvement_fraction_of_lifted_mean"),
        ("fo_lifted_collateral_ratio_std",),
        ("fo_target_improvement_fraction_of_lifted_std",),
        "Figure 4: Trained-transformer Ctl-1 nuisance-placement sweep",
        "nuisance interaction weight",
        outdir,
        "figure_04_trained_ctl1_nuisance_sweep.png",
    )
    return Generated("figure_04", (path,))


def _plot_ctl2_metric(ax: plt.Axes, traj: pd.DataFrame, metric: str, ylabel: str) -> None:
    metric_df = traj[traj["metric"] == metric]
    for observer, group in metric_df.groupby("observer", sort=False):
        group = group.sort_values("step")
        ax.plot(group["step"], group["mean"], marker="o", linewidth=2, label=observer)
        ax.fill_between(group["step"], group["q10"], group["q90"], alpha=0.16)
    ax.set_xlabel("closed-loop step")
    ax.set_ylabel(ylabel)
    ax.legend()


def figure_05(frozen: Path, outdir: Path) -> Generated:
    traj = _read_csv(frozen, "ctl2/default_v7_clean/ctl2_trajectory_quantiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    _plot_ctl2_metric(axes[0], traj, "target_squared_error", "target squared error")
    axes[0].set_title("Target error")
    _plot_ctl2_metric(axes[1], traj, "observer_abs_bias", "absolute observer bias")
    axes[1].set_title("Observer bias")
    fig.suptitle("Figure 5: Ctl-2 closed-loop trajectories")
    path = _save(fig, outdir, "figure_05_ctl2_closed_loop_trajectories.png")
    return Generated("figure_05", (path,))


def figure_06(frozen: Path, outdir: Path) -> Generated:
    df = _read_csv(frozen, "ctl2/sweeps_v7/gamma_sweep_pairwise_summary.csv")
    y_col = _first_existing(
        df,
        ("fo_minus_lifted_divergence_rate_median", "fo_minus_lifted_divergence_rate_mean"),
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(df["gamma"], df[y_col], marker="o", linewidth=2)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("interaction strength gamma")
    ax.set_ylabel("divergence-rate delta, first-order minus lifted")
    ax.set_title("Figure 6: Ctl-2 divergence-rate delta")
    path = _save(fig, outdir, "figure_06_ctl2_divergence_rate_delta.png")
    return Generated("figure_06", (path,))


def figure_07(frozen: Path, outdir: Path) -> Generated:
    traj = _read_csv(frozen, "ctl2/default_v7_clean/ctl2_trajectory_quantiles.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _plot_ctl2_metric(ax, traj, "collateral_abs_delta", "absolute collateral displacement")
    ax.set_title("Figure 7: Ctl-2 collateral trajectories")
    path = _save(fig, outdir, "figure_07_ctl2_collateral_trajectories.png")
    return Generated("figure_07", (path,))


def figure_08(frozen: Path, outdir: Path) -> Generated:
    summary = _read_csv(frozen, "ioi/stage1_both_end/ioi_stage1_summary.csv")
    row = summary[summary["ablation"] == "mean"].iloc[0] if "ablation" in summary else summary.iloc[0]
    values = {
        "P": float(row.get("drop_primary", row.get("drop_P"))),
        "B": float(row.get("drop_backup", row.get("drop_B"))),
        "additive": float(row.get("additive_prediction", row.get("drop_P", 0.0) + row.get("drop_B", 0.0))),
        "P+B": float(row.get("drop_both", row.get("drop_P+B"))),
        "interaction": float(row["interaction"]),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(values.keys(), values.values())
    ax.set_ylabel("drop in IOI logit difference")
    ax.set_title("Figure 8: Stage 1 IOI feasibility")
    path = _save(fig, outdir, "figure_08_ioi_stage1_self_repair.png")
    return Generated("figure_08", (path,))


def figure_09(frozen: Path, outdir: Path) -> Generated:
    boot = _read_csv(frozen, "ioi/stage2b_mean_end/ioi_stage2b_bootstrap_summary.csv")
    pred = _read_csv(frozen, "ioi/stage2b_mean_end/ioi_stage2b_kfold_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    boot = boot.sort_values("mae_mean")
    x = np.arange(len(boot))
    axes[0].bar(x, boot["mae_mean"])
    axes[0].errorbar(
        x,
        boot["mae_mean"],
        yerr=[boot["mae_mean"] - boot["mae_q05"], boot["mae_q95"] - boot["mae_mean"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(boot["model"], rotation=25, ha="right")
    axes[0].set_ylabel("bootstrap MAE")
    axes[0].set_title("Bootstrap MAE")

    scatter = pred[pred["model"] == "additive_head"]
    axes[1].scatter(scatter["observed"], scatter["predicted"], s=22, alpha=0.65)
    lo = float(min(scatter["observed"].min(), scatter["predicted"].min()))
    hi = float(max(scatter["observed"].max(), scatter["predicted"].max()))
    axes[1].plot([lo, hi], [lo, hi], color="0.35", linestyle="--", linewidth=1)
    axes[1].set_xlabel("observed held-out drop")
    axes[1].set_ylabel("predicted drop")
    axes[1].set_title("Held-out predictions, additive_head")

    fig.suptitle("Figure 9: Stage 2b random head-level subsets")
    path = _save(fig, outdir, "figure_09_ioi_stage2b_random_subsets.png")
    return Generated("figure_09", (path,))


def figure_10(frozen: Path, outdir: Path) -> Generated:
    delta = _read_csv(frozen, "ioi/stage2c_primary_stratified_mean_end/ioi_stage2c_paired_delta_mae.csv")
    errors = _read_csv(frozen, "ioi/stage2c_primary_stratified_mean_end/ioi_stage2c_primary_count_errors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    delta = delta.sort_values("delta_mae_mean", ascending=False)
    x = np.arange(len(delta))
    axes[0].bar(x, delta["delta_mae_mean"])
    axes[0].errorbar(
        x,
        delta["delta_mae_mean"],
        yerr=[delta["delta_mae_mean"] - delta["delta_mae_q05"], delta["delta_mae_q95"] - delta["delta_mae_mean"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(delta["model"], rotation=25, ha="right")
    axes[0].set_ylabel("paired MAE reduction vs additive_head")
    axes[0].set_title("Paired delta MAE")

    agg = errors[errors["has_B"].isna()].copy() if "has_B" in errors else errors.copy()
    for model, group in agg.groupby("model", sort=False):
        group = group.sort_values("n_P")
        axes[1].plot(group["n_P"], group["mae"], marker="o", linewidth=2, label=model)
    axes[1].set_xlabel("number of primary Name Movers ablated")
    axes[1].set_ylabel("held-out MAE")
    axes[1].set_title("Errors by primary coverage")
    axes[1].legend()

    fig.suptitle("Figure 10: Stage 2c primary-stratified subsets")
    path = _save(fig, outdir, "figure_10_ioi_stage2c_primary_stratified.png")
    return Generated("figure_10", (path,))


def _write_markdown_table(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_01(frozen: Path, outdir: Path) -> Generated:
    df = _read_csv(frozen, "ctl2/sweeps_v7/gamma_sweep_pairwise_summary.csv")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "gamma": row["gamma"],
                "delta_ise_median": row["fo_minus_lifted_integrated_squared_error_median"],
                "delta_collateral_median": row["fo_minus_lifted_cumulative_collateral_abs_median"],
                "delta_divergence_rate_median": row["fo_minus_lifted_divergence_rate_median"],
                "delta_final_target_mse_median": row["fo_minus_lifted_final_target_mse_median"],
            }
        )
    table = pd.DataFrame(rows)
    csv_path = outdir / "table_01_ctl2_interaction_sweep.csv"
    md_path = outdir / "table_01_ctl2_interaction_sweep.md"
    table.to_csv(csv_path, index=False)
    _write_markdown_table(md_path, list(table.columns), table.values.tolist())
    return Generated("table_01", (csv_path, md_path))


def table_02(frozen: Path, outdir: Path) -> Generated:
    boot = _read_csv(frozen, "ioi/stage2d_per_pair/ioi_stage2d_bootstrap_summary.csv").set_index("model")
    comp = _read_csv(frozen, "ioi/stage2d_per_pair/ioi_stage2d_model_comparison.csv").set_index("model")
    order = [
        "additive_head",
        "count_additive",
        "count_plus_PB_count",
        "count_plus_PE_count",
        "count_plus_BE_count",
        "count_plus_all_pairs",
    ]
    rows = []
    for model in order:
        rows.append(
            {
                "model": model,
                "mae_mean": boot.loc[model, "mae_mean"],
                "r2_mean": boot.loc[model, "r2_mean"],
                "delta_mae_vs_additive_mean": comp.loc[model, "delta_mae_vs_additive_mean"],
                "delta_mae_vs_count_additive_mean": comp.loc[model, "delta_mae_vs_count_additive_mean"],
            }
        )
    table = pd.DataFrame(rows)
    csv_path = outdir / "table_02_ioi_stage2d_per_pair.csv"
    md_path = outdir / "table_02_ioi_stage2d_per_pair.md"
    table.to_csv(csv_path, index=False)
    _write_markdown_table(md_path, list(table.columns), table.values.tolist())
    return Generated("table_02", (csv_path, md_path))


GENERATORS: dict[str, Callable[[Path, Path], Generated]] = {
    "figure_01": figure_01,
    "figure_02": figure_02,
    "figure_03": figure_03,
    "figure_04": figure_04,
    "figure_05": figure_05,
    "figure_06": figure_06,
    "figure_07": figure_07,
    "figure_08": figure_08,
    "figure_09": figure_09,
    "figure_10": figure_10,
    "table_01": table_01,
    "table_02": table_02,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--only", choices=sorted(GENERATORS), default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frozen = args.frozen_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    keys = [args.only] if args.only else list(GENERATORS)
    generated = [GENERATORS[key](frozen, outdir) for key in keys]
    manifest = {
        "schema": "observerbench.paper_reproduction_fast.v0",
        "frozen_dir": str(frozen.relative_to(REPO_ROOT) if frozen.is_relative_to(REPO_ROOT) else frozen),
        "outdir": str(outdir.relative_to(REPO_ROOT) if outdir.is_relative_to(REPO_ROOT) else outdir),
        "generated": [
            {
                "result_id": item.result_id,
                "paths": [
                    str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
                    for path in item.paths
                ],
            }
            for item in generated
        ],
        "cpu_only": True,
        "downloads_models": False,
        "trains_models": False,
    }
    manifest_path = outdir / "reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for item in generated:
        for path in item.paths:
            print(path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
