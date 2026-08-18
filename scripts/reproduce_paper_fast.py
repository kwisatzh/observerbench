#!/usr/bin/env python3
"""Regenerate ObserverBench paper figures and tables from frozen summaries.

This script is deliberately CPU-only. It reads checked CSV/JSON summaries,
writes only the requested output tree apart from an ephemeral plotting cache,
and never imports TransformerLens, downloads GPT-2, or trains a model.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = REPO_ROOT / "results" / "frozen"
DEFAULT_REVISION = REPO_ROOT / "results" / "revision"
DEFAULT_OUTDIR = REPO_ROOT / "paper" / "generated_current"
MPL_CACHE = Path(
    os.environ.get(
        "MPLCONFIGDIR",
        Path(tempfile.gettempdir()) / "observerbench-matplotlib-cache",
    )
)
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from observerbench.cards import validate_observer_card_bundle
from observerbench.paper_cards import write_ctl2_card_tex
from observerbench.provenance import file_sha256, portable_artifact_path


@dataclass(frozen=True)
class Generated:
    result_id: str
    paths: tuple[Path, ...]


def _read_csv(frozen: Path, relpath: str) -> pd.DataFrame:
    path = frozen / relpath
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen input: {path}")
    return pd.read_csv(path)


def _read_revision_csv(revision: Path, relpath: str) -> pd.DataFrame:
    path = revision / relpath
    if not path.exists():
        raise FileNotFoundError(f"Missing checked revision input: {path}")
    return pd.read_csv(path)


def _read_revision_json(revision: Path, relpath: str) -> dict:
    path = revision / relpath
    if not path.exists():
        raise FileNotFoundError(f"Missing checked revision input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def _seed_summary(frame: pd.DataFrame, group: list[str], metric: str) -> pd.DataFrame:
    return frame.groupby(group, as_index=False)[metric].agg(
        mean="mean",
        seed_min="min",
        seed_max="max",
    )


def revision_ctl2_factorial(revision: Path, outdir: Path) -> Generated:
    frame = _read_revision_csv(
        revision,
        "phase04/ctl2_multiseed_sweep_current/phase01_sweep_results.csv",
    )
    primary = frame[
        (frame["estimator_calibration"] == "none")
        & (frame["controller_mode"] == "base")
    ]
    summary = _seed_summary(
        primary,
        [
            "sweep_gamma",
            "direction_support_mode",
            "estimator_name",
            "direction_name",
        ],
        "integrated_squared_error",
    )
    colors = {"first_order": "tab:red", "lifted_interaction": "tab:blue"}
    linestyles = {"first_order": "-", "lifted_interaction": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, support_mode in zip(axes, ("unprojected", "projected")):
        panel = summary[summary["direction_support_mode"] == support_mode]
        for (estimator, direction), cell in panel.groupby(
            ["estimator_name", "direction_name"],
            sort=False,
        ):
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
    fig.suptitle("Checked Ctl-2 factorial: mean and training-seed min--max")
    path = _save(
        fig,
        outdir,
        "revision_ctl2_factorial_ise_by_gamma.png",
    )
    return Generated("revision_ctl2_factorial", (path,))


def revision_ctl2_calibration(revision: Path, outdir: Path) -> Generated:
    frame = _read_revision_csv(
        revision,
        "phase04/ctl2_multiseed_sweep_current/phase01_sweep_results.csv",
    )
    diagonal = frame[
        (frame["direction_support_mode"] == "unprojected")
        & (frame["controller_mode"] == "base")
        & (frame["estimator_name"] == frame["direction_name"])
    ]
    summary = _seed_summary(
        diagonal,
        ["sweep_gamma", "estimator_name", "estimator_calibration"],
        "integrated_squared_error",
    )
    colors = {"first_order": "tab:red", "lifted_interaction": "tab:blue"}
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for (estimator, mode), cell in summary.groupby(
        ["estimator_name", "estimator_calibration"],
        sort=False,
    ):
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
    ax.set_title("Checked Ctl-2 finite-response calibration")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    path = _save(
        fig,
        outdir,
        "revision_ctl2_response_calibration_by_gamma.png",
    )
    return Generated("revision_ctl2_calibration", (path,))


def revision_ctl2_table(revision: Path, outdir: Path) -> Generated:
    audit = _read_revision_json(revision, "phase04/ctl2_phase04_audit.json")
    rows = [
        row
        for row in audit["multiseed"]["fixed_direction_ise_effects"]
        if row["gamma"] > 0 and row["support_mode"] == "unprojected"
    ]
    table = pd.DataFrame([
        {
            "gamma": row["gamma"],
            "fixed_direction": row["fixed_direction"],
            "mean_fo_minus_lifted_ise": row["mean"],
            "seed_min": row["seed_min"],
            "seed_max": row["seed_max"],
            "positive_seeds": row["n_positive_fo_minus_lifted"],
            "n_seeds": row["n_seeds"],
        }
        for row in rows
    ])
    csv_path = outdir / "revision_ctl2_factorial_table.csv"
    md_path = outdir / "revision_ctl2_factorial_table.md"
    tex_path = outdir / "revision_ctl2_factorial_table.tex"
    table.to_csv(csv_path, index=False)
    _write_markdown_table(md_path, list(table.columns), table.values.tolist())
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"$\gamma$ & Fixed direction & Mean FO--lifted ISE & Seed min--max \\",
        r"\midrule",
    ]
    direction_labels = {
        "first_order": "first order",
        "lifted_interaction": "lifted",
    }
    for row in rows:
        tex_lines.append(
            f"{row['gamma']:.2f} & {direction_labels[row['fixed_direction']]} & "
            f"{row['mean']:.2f} & {row['seed_min']:.3g}--{row['seed_max']:.4g} \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Ctl-2 estimator effects with direction fixed. All listed contrasts are positive in all six training seeds. The $\gamma=0$ contrasts are numerical zero and are omitted.}",
        r"\label{tab:ctl2-factorial}",
        r"\end{table}",
        "",
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    return Generated("revision_ctl2_table", (csv_path, md_path, tex_path))


def _interval_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
) -> None:
    labels = ["PB", "PE", "BE"]
    values = frame.assign(
        pair=frame["contrast"].str.extract(r"_(PB|PE|BE)_", expand=False)
    ).set_index("pair").loc[labels]
    means = values["mean"].to_numpy()
    lower = means - values["q05"].to_numpy()
    upper = values["q95"].to_numpy() - means
    x = np.arange(len(labels))
    ax.bar(x, means, color=["tab:blue", "tab:orange", "tab:green"])
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out MAE reduction")
    ax.set_title(title)


def _revision_ioi_capacity(
    revision: Path,
    outdir: Path,
    *,
    stage: str,
    result_id: str,
    filename: str,
    title: str,
) -> Generated:
    base = f"phase02/{stage}_capacity"
    add_one = _read_revision_csv(revision, f"{base}/add_one_vs_count_additive.csv")
    leave_one_out = _read_revision_csv(revision, f"{base}/leave_one_out_contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    _interval_panel(axes[0], add_one, title="Add one pair to count-additive")
    _interval_panel(axes[1], leave_one_out, title="Remove one pair from all-pairs")
    fig.suptitle(title)
    path = _save(fig, outdir, filename)
    return Generated(result_id, (path,))


def revision_ioi_stage2b(revision: Path, outdir: Path) -> Generated:
    return _revision_ioi_capacity(
        revision,
        outdir,
        stage="ioi_stage2b",
        result_id="revision_ioi_stage2b",
        filename="revision_ioi_stage2b_capacity_matched.png",
        title="Checked IOI Stage 2b: capacity-matched pair contrasts",
    )


def revision_ioi_stage2c(revision: Path, outdir: Path) -> Generated:
    return _revision_ioi_capacity(
        revision,
        outdir,
        stage="ioi_stage2c",
        result_id="revision_ioi_stage2c",
        filename="revision_ioi_stage2c_capacity_matched.png",
        title="Checked IOI Stage 2c: capacity-matched pair contrasts",
    )


def revision_ioi_mobius(revision: Path, outdir: Path) -> Generated:
    frame = _read_revision_csv(
        revision,
        "phase02/ioi_stage2c_capacity/mobius_bootstrap_summary.csv",
    )
    labels = ["PB", "PE", "BE", "PBE"]
    values = frame.set_index("term").loc[labels]
    means = values["mean"].to_numpy()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x, means)
    ax.errorbar(
        x,
        means,
        yerr=np.vstack(
            [
                means - values["q05"].to_numpy(),
                values["q95"].to_numpy() - means,
            ]
        ),
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("direct group-mask interaction")
    ax.set_title("Checked IOI direct interactions: prompt-bootstrap 5--95%")
    path = _save(fig, outdir, "revision_ioi_mobius_intervals.png")
    return Generated("revision_ioi_mobius", (path,))


def revision_ioi_table(revision: Path, outdir: Path) -> Generated:
    rows: list[dict[str, object]] = []
    for stage in ("ioi_stage2b", "ioi_stage2c"):
        base = f"phase02/{stage}_capacity"
        for analysis, filename in (
            ("add_one_vs_count", "add_one_vs_count_additive.csv"),
            ("leave_one_out", "leave_one_out_contrasts.csv"),
        ):
            frame = _read_revision_csv(revision, f"{base}/{filename}")
            pairs = frame.assign(
                pair=frame["contrast"].str.extract(r"_(PB|PE|BE)_", expand=False)
            ).dropna(subset=["pair"])
            for _, row in pairs.iterrows():
                rows.append(
                    {
                        "stage": stage.replace("ioi_", ""),
                        "analysis": analysis,
                        "pair": row["pair"],
                        "mae_reduction_mean": float(row["mean"]),
                        "q05": float(row["q05"]),
                        "q95": float(row["q95"]),
                    }
                )
    table = pd.DataFrame(rows)
    contrast_csv = outdir / "revision_ioi_capacity_contrasts.csv"
    contrast_md = outdir / "revision_ioi_capacity_contrasts.md"
    contrast_tex = outdir / "revision_ioi_capacity_contrasts.tex"
    table.to_csv(contrast_csv, index=False)
    _write_markdown_table(contrast_md, list(table.columns), table.values.tolist())
    by_key = table.set_index(["stage", "analysis", "pair"])
    tex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"Design & Pair & $\Delta_{\mathrm{add}}$ & $\Delta_{\mathrm{remove}}$ \\",
        r"\midrule",
    ]
    design_labels = {"stage2b": "Broad random", "stage2c": "Primary stratified"}
    pair_labels = {"PB": r"$P\times B$", "PE": r"$P\times E$", "BE": r"$B\times E$"}
    for stage in ("stage2b", "stage2c"):
        for pair_index, pair in enumerate(("PB", "PE", "BE")):
            add = by_key.loc[(stage, "add_one_vs_count", pair)]
            remove = by_key.loc[(stage, "leave_one_out", pair)]
            design = design_labels[stage] if pair_index == 0 else ""
            tex_lines.append(
                f"{design} & {pair_labels[pair]} & "
                f"{add['mae_reduction_mean']:.4f} [{add['q05']:.4f}, {add['q95']:.4f}] & "
                f"{remove['mae_reduction_mean']:.4f} [{remove['q05']:.4f}, {remove['q95']:.4f}] \\\\"
            )
        if stage == "stage2b":
            tex_lines.append(r"\addlinespace")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Capacity-matched IOI pair contrasts. Positive values are paired reductions in held-out MAE; brackets give 5--95\% prompt-bootstrap intervals. Add-one and leave-one-out answer different conditional questions.}",
        r"\label{tab:ioi-capacity}",
        r"\end{table}",
        "",
    ])
    contrast_tex.write_text("\n".join(tex_lines), encoding="utf-8")

    mobius = _read_revision_csv(
        revision,
        "phase02/ioi_stage2c_capacity/mobius_bootstrap_summary.csv",
    )
    mobius = mobius[["term", "point", "mean", "q05", "q95"]]
    mobius_csv = outdir / "revision_ioi_mobius_intervals.csv"
    mobius_md = outdir / "revision_ioi_mobius_intervals.md"
    mobius.to_csv(mobius_csv, index=False)
    _write_markdown_table(mobius_md, list(mobius.columns), mobius.values.tolist())
    return Generated(
        "revision_ioi_table",
        (contrast_csv, contrast_md, contrast_tex, mobius_csv, mobius_md),
    )


def revision_observer_cards(revision: Path, outdir: Path) -> Generated:
    source = revision / "phase04/observer_cards"
    bundle_path = source / "observer_card.json"
    markdown_path = source / "observer_card.md"
    if not bundle_path.exists() or not markdown_path.exists():
        raise FileNotFoundError(
            "Missing checked Phase-4 ObserverCards; run "
            "python scripts/build_phase04_artifact.py first."
        )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    validate_observer_card_bundle(bundle)
    destination = outdir / "observer_cards"
    destination.mkdir(parents=True, exist_ok=True)
    json_out = destination / "observer_card.json"
    markdown_out = destination / "observer_card.md"
    shutil.copyfile(bundle_path, json_out)
    shutil.copyfile(markdown_path, markdown_out)
    tex_out = write_ctl2_card_tex(
        bundle["cards"],
        destination / "ctl2_observer_card.tex",
    )
    return Generated("revision_observer_cards", (json_out, markdown_out, tex_out))


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

REVISION_GENERATORS: dict[str, Callable[[Path, Path], Generated]] = {
    "revision_ctl2_factorial": revision_ctl2_factorial,
    "revision_ctl2_calibration": revision_ctl2_calibration,
    "revision_ctl2_table": revision_ctl2_table,
    "revision_ioi_stage2b": revision_ioi_stage2b,
    "revision_ioi_stage2c": revision_ioi_stage2c,
    "revision_ioi_mobius": revision_ioi_mobius,
    "revision_ioi_table": revision_ioi_table,
    "revision_observer_cards": revision_observer_cards,
}

CURRENT_PAPER_IDS = (
    "figure_01",
    "figure_02",
    "figure_03",
    "figure_04",
    "figure_08",
    "revision_ctl2_factorial",
    "revision_ctl2_calibration",
    "revision_ctl2_table",
    "revision_ioi_stage2b",
    "revision_ioi_stage2c",
    "revision_ioi_mobius",
    "revision_ioi_table",
    "revision_observer_cards",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--revision-dir", type=Path, default=DEFAULT_REVISION)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--only",
        choices=sorted(GENERATORS | REVISION_GENERATORS),
        default=None,
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Generate the superseded v7 figure/table set instead of the current paper set.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frozen = args.frozen_dir.resolve()
    revision = args.revision_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.only:
        keys = [args.only]
    elif args.legacy:
        keys = list(GENERATORS)
    else:
        keys = list(CURRENT_PAPER_IDS)
    generated = [
        GENERATORS[key](frozen, outdir)
        if key in GENERATORS
        else REVISION_GENERATORS[key](revision, outdir)
        for key in keys
    ]
    manifest = {
        "schema": "observerbench.paper_reproduction_fast.v1",
        "frozen_dir": portable_artifact_path(frozen, REPO_ROOT),
        "revision_dir": portable_artifact_path(revision, REPO_ROOT),
        "outdir": portable_artifact_path(outdir, REPO_ROOT),
        "selection": "one" if args.only else "legacy" if args.legacy else "current_paper",
        "current_default_ids": list(CURRENT_PAPER_IDS),
        "generated": [
            {
                "result_id": item.result_id,
                "paths": [
                    path.relative_to(outdir).as_posix()
                    for path in item.paths
                ],
                "sha256": {
                    path.relative_to(outdir).as_posix(): file_sha256(path)
                    for path in item.paths
                },
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
