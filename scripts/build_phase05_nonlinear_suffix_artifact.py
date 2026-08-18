#!/usr/bin/env python3
"""Build checked paper artifacts for the Phase-5 nonlinear suffix experiment.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results/revision/phase05/nonlinear_suffix_v3_support"
DEFAULT_PROTOCOL = REPO_ROOT / "configs/revision/ctl2_phase05_nonlinear_suffix_v2.json"
DEFAULT_OUTDIR = REPO_ROOT / "paper/generated_phase05/nonlinear_suffix"
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, portable_artifact_path, source_hashes


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} changed: recomputed={actual}, recorded={expected}")


def _natural_seed_effects(natural: pd.DataFrame) -> pd.DataFrame:
    diagonal = natural[natural["estimator"] == natural["direction_provider"]]
    means = diagonal.groupby(["seed", "gamma", "estimator"], as_index=False)[
        "actual_integrated_squared_error"
    ].mean()
    pivot = means.pivot(
        index=["seed", "gamma"],
        columns="estimator",
        values="actual_integrated_squared_error",
    ).reset_index()
    pivot["first_order_minus_lifted"] = (
        pivot["first_order"] - pivot["lifted_interaction"]
    )
    return pivot.sort_values(["gamma", "seed"]).reset_index(drop=True)


def _factorial_cell_rows(natural: pd.DataFrame) -> pd.DataFrame:
    seed_cells = natural.groupby(
        ["seed", "gamma", "estimator", "direction_provider"],
        as_index=False,
    )[[
        "actual_integrated_squared_error",
        "observer_gain",
        "observer_pole",
    ]].mean()
    rows: list[dict[str, Any]] = []
    for (gamma, estimator, direction), conditions in natural.groupby(
        ["gamma", "estimator", "direction_provider"],
    ):
        seeds = seed_cells[
            np.isclose(seed_cells["gamma"], gamma)
            & (seed_cells["estimator"] == estimator)
            & (seed_cells["direction_provider"] == direction)
        ]
        outside = conditions[
            "residual_displacement_over_clean_pairwise_max"
        ].to_numpy(float)
        rows.append({
            "gamma": float(gamma),
            "estimator": str(estimator),
            "direction_provider": str(direction),
            "condition_count": int(len(conditions)),
            "seed_count": int(len(seeds)),
            "mean_ise": float(
                seeds["actual_integrated_squared_error"].mean()
            ),
            "mean_observer_gain": float(seeds["observer_gain"].mean()),
            "nonpositive_observer_gain_seed_count": int(
                np.sum(seeds["observer_gain"] <= 0.0)
            ),
            "unstable_observer_pole_seed_count": int(
                np.sum(np.abs(seeds["observer_pole"]) >= 1.0)
            ),
            "would_clip_condition_count": int(
                np.sum(conditions["would_clip_fraction"] > 0.0)
            ),
            "mean_would_clip_step_fraction": float(
                conditions["would_clip_fraction"].mean()
            ),
            "max_abs_action": float(conditions["max_abs_action"].max()),
            "displacement_exceeds_clean_pairwise_max_condition_count": int(
                np.sum(outside > 1.0)
            ),
            "displacement_exceeds_clean_pairwise_max_fraction": float(
                np.mean(outside > 1.0)
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["gamma", "estimator", "direction_provider"],
    ).reset_index(drop=True)


def _factorial_contrast_rows(
    natural: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_repeats: int,
) -> pd.DataFrame:
    seed_cells = natural.groupby(
        ["seed", "gamma", "estimator", "direction_provider"],
        as_index=False,
    )["actual_integrated_squared_error"].mean()
    rows: list[dict[str, Any]] = []
    for gamma_index, gamma in enumerate(sorted(seed_cells["gamma"].unique())):
        gamma_rows = seed_cells[np.isclose(seed_cells["gamma"], gamma)]
        pivot = gamma_rows.pivot(
            index="seed",
            columns=["estimator", "direction_provider"],
            values="actual_integrated_squared_error",
        )
        ff = pivot[("first_order", "first_order")]
        fl = pivot[("first_order", "lifted_interaction")]
        lf = pivot[("lifted_interaction", "first_order")]
        ll = pivot[("lifted_interaction", "lifted_interaction")]
        contrasts = {
            "diagonal_pair_fo_minus_lifted": ff - ll,
            "estimator_fo_minus_lifted_at_fo_direction": ff - lf,
            "estimator_fo_minus_lifted_at_lifted_direction": fl - ll,
            "direction_fo_minus_lifted_at_fo_estimator": ff - fl,
            "direction_fo_minus_lifted_at_lifted_estimator": lf - ll,
            "lifted_pair_compatibility_interaction": (fl - ll) - (ff - lf),
        }
        for name, values in contrasts.items():
            paired = values.to_numpy(float)
            low, high = _bootstrap_mean_interval(
                paired,
                seed=bootstrap_seed + 200 + gamma_index,
                repeats=bootstrap_repeats,
            )
            rows.append({
                "gamma": float(gamma),
                "contrast": name,
                "mean_ise_difference": float(paired.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "positive_seed_count": int(np.sum(paired > 0.0)),
                "seed_count": int(len(paired)),
            })
    return pd.DataFrame(rows)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(repeats, len(values)))
    means = np.mean(values[samples], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _validate(
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    analysis: dict[str, Any],
    provenance: dict[str, Any],
    controlled: pd.DataFrame,
    natural: pd.DataFrame,
    cv: pd.DataFrame,
    failures: pd.DataFrame,
    residual_summary: pd.DataFrame,
    natural_effects: pd.DataFrame,
    factorial_cells: pd.DataFrame,
    factorial_contrasts: pd.DataFrame,
) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "protocol_is_frozen_v2": (
            protocol.get("schema") == "observerbench.nonlinear_suffix_protocol.v2"
            and protocol.get("status") == "frozen_before_new_outcomes"
        ),
        "protocol_hash_matches_recorded_provenance": (
            file_sha256(protocol_path) == provenance.get("protocol_sha256")
            and protocol_path.name == provenance.get("protocol_name")
        ),
        "analysis_schema_is_v3": (
            analysis.get("schema") == "observerbench.nonlinear_suffix_analysis.v3"
        ),
        "analysis_and_every_preregistered_gate_pass": (
            bool(analysis.get("all_gates_pass"))
            and bool(analysis.get("gates"))
            and all(bool(value) for value in analysis["gates"].values())
        ),
        "source_row_counts_match_analysis": (
            len(controlled) == int(analysis["controlled_condition_count"])
            and len(natural) == int(analysis["natural_condition_count"])
            and len(failures) == int(analysis["manipulability_failure_count"])
        ),
        "source_seed_sets_match_protocol": (
            sorted(controlled["seed"].unique().tolist()) == protocol["seeds"]
            and sorted(natural["seed"].unique().tolist()) == protocol["seeds"]
            and sorted(cv["held_out_seed"].unique().tolist()) == protocol["seeds"]
        ),
        "natural_gamma_values_match_protocol": (
            sorted(natural["gamma"].unique().tolist())
            == sorted(float(value) for value in protocol["gamma_values_natural"])
        ),
        "no_manipulability_failure_is_recorded": failures.empty,
        "support_diagnostic_is_present_and_bounded": (
            analysis.get("residual_displacement_diagnostic", {}).get("status")
            == "diagnostic_only_not_a_success_gate"
            and not residual_summary.empty
            and int(
                residual_summary[
                    residual_summary["condition_family"] == "controlled_overall"
                ]["condition_count"].iloc[0]
            )
            == len(controlled)
            and float(
                residual_summary[
                    residual_summary["condition_family"] == "controlled_overall"
                ]["ratio_to_clean_pairwise_max_fraction_above_one"].iloc[0]
            )
            < 0.01
            and bool(
                (
                    residual_summary[
                        residual_summary["condition_family"] == "natural_diagonal"
                    ]["ratio_to_clean_pairwise_max_fraction_above_one"]
                    == 0.0
                ).all()
            )
        ),
    }

    cv_means = cv.groupby("predictor")["mae"].mean()
    for predictor, recorded in analysis["leave_one_seed_out_mae"].items():
        _assert_close(
            float(cv_means[predictor]),
            float(recorded),
            f"leave-one-seed-out MAE for {predictor}",
        )

    certificate = controlled["certificate_final_error"].to_numpy(float)
    actual = controlled["actual_final_error"].to_numpy(float)
    _assert_close(
        float(np.corrcoef(certificate, actual)[0, 1]),
        float(analysis["certificate_actual_correlation"]),
        "certificate correlation",
    )
    _assert_close(
        float(np.mean(np.signbit(certificate) == np.signbit(actual))),
        float(analysis["certificate_sign_agreement"]),
        "certificate sign agreement",
    )
    checks["certificate_metrics_recompute_exactly"] = True

    gamma_values = sorted(float(value) for value in protocol["gamma_values_natural"])
    for gamma_index, gamma in enumerate(gamma_values):
        rows = natural_effects[np.isclose(natural_effects["gamma"], gamma)]
        values = rows["first_order_minus_lifted"].to_numpy(float)
        recorded = analysis["natural_diagonal_ranking"][str(float(gamma))]
        low, high = _bootstrap_mean_interval(
            values,
            seed=int(protocol["bootstrap_seed"]) + 100 + gamma_index,
            repeats=int(protocol["bootstrap_repeats"]),
        )
        _assert_close(
            float(values.mean()),
            float(recorded["mean_actual_fo_minus_lifted_ise"]),
            f"natural mean difference for gamma={gamma}",
        )
        _assert_close(
            low,
            float(recorded["mean_actual_fo_minus_lifted_ise_ci95_low"]),
            f"natural lower interval for gamma={gamma}",
        )
        _assert_close(
            high,
            float(recorded["mean_actual_fo_minus_lifted_ise_ci95_high"]),
            f"natural upper interval for gamma={gamma}",
        )
    checks["natural_effects_and_bootstrap_intervals_recompute_exactly"] = True

    factorial = analysis["natural_factorial_secondary"]
    if factorial.get("status") != "secondary_analysis_of_factorial_cells_frozen_before_outcomes":
        raise ValueError("factorial analysis status is missing or incorrect")
    for gamma in gamma_values:
        recorded_gamma = factorial["by_gamma"][str(float(gamma))]
        recorded_cells = {
            (row["estimator"], row["direction_provider"]): row
            for row in recorded_gamma["cells"]
        }
        source_cells = factorial_cells[np.isclose(factorial_cells["gamma"], gamma)]
        for row in source_cells.to_dict(orient="records"):
            recorded = recorded_cells[(row["estimator"], row["direction_provider"])]
            for field in (
                "mean_ise",
                "mean_observer_gain",
                "mean_would_clip_step_fraction",
                "max_abs_action",
                "displacement_exceeds_clean_pairwise_max_fraction",
            ):
                _assert_close(float(row[field]), float(recorded[field]), field)
            for field in (
                "condition_count",
                "seed_count",
                "nonpositive_observer_gain_seed_count",
                "unstable_observer_pole_seed_count",
                "would_clip_condition_count",
                "displacement_exceeds_clean_pairwise_max_condition_count",
            ):
                if int(row[field]) != int(recorded[field]):
                    raise ValueError(
                        f"factorial cell {field} changed for gamma={gamma}: "
                        f"recomputed={row[field]}, recorded={recorded[field]}"
                    )
        source_contrasts = factorial_contrasts[
            np.isclose(factorial_contrasts["gamma"], gamma)
        ]
        for row in source_contrasts.to_dict(orient="records"):
            recorded = recorded_gamma["contrasts"][row["contrast"]]
            for field in (
                "mean_ise_difference",
                "ci95_low",
                "ci95_high",
            ):
                _assert_close(float(row[field]), float(recorded[field]), field)
            for field in ("positive_seed_count", "seed_count"):
                if int(row[field]) != int(recorded[field]):
                    raise ValueError(
                        f"factorial contrast {field} changed for gamma={gamma}, "
                        f"contrast={row['contrast']}"
                    )
    checks["factorial_cells_contrasts_and_intervals_recompute_exactly"] = True

    gamma_on = factorial["by_gamma"]["1.15"]
    gamma_off = factorial["by_gamma"]["0.0"]
    checks["interaction_crossed_arms_are_flagged_not_claimed"] = (
        not gamma_on["crossed_arms_pass_conditioning_and_clean_scale_diagnostics"]
        and gamma_on["attribution_status"]
        == "crossed_arms_fail_diagnostics_diagonal_result_remains_system_level"
    )
    checks["first_order_null_factorial_is_well_conditioned"] = bool(
        gamma_off["crossed_arms_pass_conditioning_and_clean_scale_diagnostics"]
    )

    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Phase-5 artifact validation failed: {failed}")
    return checks


def _summary_rows(
    *,
    analysis: dict[str, Any],
    natural_effects: pd.DataFrame,
) -> pd.DataFrame:
    cv_mae = analysis["leave_one_seed_out_mae"]
    rows: list[dict[str, Any]] = []
    for comparator, label in (
        ("bias_only", "Bias-only MAE - full-certificate MAE"),
        ("pole_only", "Pole-only MAE - full-certificate MAE"),
    ):
        interval = analysis["seed_bootstrap_mae_advantage"][comparator]
        rows.append({
            "comparison": label,
            "metric_a_label": comparator.replace("_", " "),
            "metric_a_mean": float(cv_mae[comparator]),
            "metric_b_label": "full certificate",
            "metric_b_mean": float(cv_mae["full_certificate"]),
            "difference_a_minus_b": float(interval["mean_mae_advantage"]),
            "ci95_low": float(interval["ci95_low"]),
            "ci95_high": float(interval["ci95_high"]),
            "relative_reduction_fraction": float(
                analysis["mae_reduction_fraction"][comparator]
            ),
            "seed_successes": int(analysis["seeds_full_certificate_beats_both"]),
            "seed_count": int(natural_effects["seed"].nunique()),
        })

    for gamma in (1.15, 0.0):
        gamma_rows = natural_effects[np.isclose(natural_effects["gamma"], gamma)]
        recorded = analysis["natural_diagonal_ranking"][str(float(gamma))]
        rows.append({
            "comparison": f"First-order ISE - lifted ISE, gamma={gamma:g}",
            "metric_a_label": "first order",
            "metric_a_mean": float(gamma_rows["first_order"].mean()),
            "metric_b_label": "lifted interaction",
            "metric_b_mean": float(gamma_rows["lifted_interaction"].mean()),
            "difference_a_minus_b": float(
                recorded["mean_actual_fo_minus_lifted_ise"]
            ),
            "ci95_low": float(
                recorded["mean_actual_fo_minus_lifted_ise_ci95_low"]
            ),
            "ci95_high": float(
                recorded["mean_actual_fo_minus_lifted_ise_ci95_high"]
            ),
            "relative_reduction_fraction": float(
                recorded["pooled_lifted_ise_reduction_fraction"]
            ),
            "seed_successes": int(recorded["seeds_lifted_ise_lower"]),
            "seed_count": int(recorded["seed_count"]),
        })
    return pd.DataFrame(rows)


def _write_latex(summary: pd.DataFrame, path: Path) -> None:
    labels = {
        "Bias-only MAE - full-certificate MAE": "Bias vs. full cert.",
        "Pole-only MAE - full-certificate MAE": "Pole vs. full cert.",
        "First-order ISE - lifted ISE, gamma=1.15": r"FO vs. lifted, $\gamma=1.15$",
        "First-order ISE - lifted ISE, gamma=0": r"FO vs. lifted, $\gamma=0$",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Comparison & A & B & A--B [95\% CI] & Seeds \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        comparison = labels[row.comparison]
        interval = (
            f"{row.difference_a_minus_b:.3f} "
            f"[{row.ci95_low:.3f}, {row.ci95_high:.3f}]"
        )
        lines.append(
            f"{comparison} & {row.metric_a_mean:.3f} & "
            f"{row.metric_b_mean:.3f} & {interval} & "
            f"{row.seed_successes}/{row.seed_count} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Study-1 nonlinear-suffix results. A--B is positive when "
            r"the full certificate or lifted pair has lower error. Intervals "
            r"resample the 12 training seeds.}"
        ),
        r"\label{tab:phase5-nonlinear-suffix}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_factorial_latex(cells: pd.DataFrame, path: Path) -> None:
    labels = {
        "first_order": "FO",
        "lifted_interaction": "Lifted",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}rllrrrrr@{}}",
        r"\toprule",
        (
            r"$\gamma$ & Estimate & Direction & ISE & "
            r"$g_E\!\leq\!0$ & $|p|\!\geq\!1$ & "
            r"Would clip & Beyond clean max \\"
        ),
        r"\midrule",
    ]
    ordered = cells.sort_values(
        ["gamma", "estimator", "direction_provider"],
        ascending=[False, True, True],
    )
    previous_gamma: float | None = None
    for row in ordered.itertuples(index=False):
        if previous_gamma is not None and not np.isclose(row.gamma, previous_gamma):
            lines.append(r"\addlinespace")
        lines.append(
            f"{row.gamma:g} & {labels[row.estimator]} & "
            f"{labels[row.direction_provider]} & {row.mean_ise:.3f} & "
            f"{row.nonpositive_observer_gain_seed_count}/{row.seed_count} & "
            f"{row.unstable_observer_pole_seed_count}/{row.seed_count} & "
            f"{row.would_clip_condition_count}/{row.condition_count} & "
            f"{row.displacement_exceeds_clean_pairwise_max_condition_count}/"
            f"{row.condition_count} \\\\"
        )
        previous_gamma = float(row.gamma)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Secondary nonlinear-suffix estimator--direction factorial. "
            r"Counts for $g_E$ and $p$ use 12 seed-level means; the last two "
            r"columns use 192 rollout conditions per cell. The crossed arms at "
            r"$\gamma=1.15$ fail loop-conditioning and clean-archetype scale "
            r"diagnostics, so they do not identify an estimator-only advantage. "
            r"The clean maximum is a scale check, not a manifold boundary.}"
        ),
        r"\label{tab:nonlinear-suffix-factorial}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(
    *,
    controlled: pd.DataFrame,
    natural_effects: pd.DataFrame,
    analysis: dict[str, Any],
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, (ax_pred, ax_control) = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
        gridspec_kw={"width_ratios": [1.0, 1.12]},
    )

    certificate = controlled["certificate_final_error"].to_numpy(float)
    actual = controlled["actual_final_error"].to_numpy(float)
    ax_pred.scatter(
        certificate,
        actual,
        s=5,
        alpha=0.13,
        color="#0072B2",
        edgecolors="none",
        rasterized=True,
        label="Individual conditions",
    )
    condition_means = controlled.groupby(
        ["rho_target", "bias_target", "relative_offset"],
        as_index=False,
    )[["certificate_final_error", "actual_final_error"]].mean()
    ax_pred.scatter(
        condition_means["certificate_final_error"],
        condition_means["actual_final_error"],
        s=12,
        facecolors="none",
        edgecolors="0.15",
        linewidths=0.55,
        label="Means across seeds",
    )
    axis_limit = 3.2
    ax_pred.plot(
        [-axis_limit, axis_limit],
        [-axis_limit, axis_limit],
        color="0.35",
        linewidth=0.8,
        linestyle="--",
        zorder=0,
    )
    ax_pred.set_xlim(-axis_limit, axis_limit)
    ax_pred.set_ylim(-axis_limit, axis_limit)
    ax_pred.set_aspect("equal", adjustable="box")
    ax_pred.set_xlabel("Certificate prediction")
    ax_pred.set_ylabel("Actual final tracking error")
    ax_pred.set_title("(a) Predicting closed-loop error", loc="left")
    ax_pred.text(
        0.04,
        0.96,
        (
            f"raw $r$ = {analysis['certificate_actual_correlation']:.3f}\n"
            f"sign agreement = {100 * analysis['certificate_sign_agreement']:.1f}%\n"
            "residual MAE = "
            f"{100 * analysis['certificate_residual_fraction_of_actual']:.1f}% "
            r"of mean $|e_T|$"
        ),
        transform=ax_pred.transAxes,
        va="top",
        ha="left",
        fontsize=7.4,
    )
    ax_pred.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#0072B2",
                markeredgecolor="none",
                markersize=4,
                label="Individual conditions",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="0.15",
                markersize=4,
                label="Means across seeds",
            ),
        ],
        loc="lower right",
        frameon=False,
        handletextpad=0.4,
    )

    effects = natural_effects.pivot(
        index="seed",
        columns="gamma",
        values="first_order_minus_lifted",
    )
    gamma_off = effects[0.0]
    gamma_on = effects[1.15]
    jitter = np.linspace(-0.075, 0.075, len(effects))
    for index, seed in enumerate(effects.index):
        ax_control.plot(
            [gamma_off.loc[seed], gamma_on.loc[seed]],
            [jitter[index], 1.0 + jitter[index]],
            color="0.78",
            linewidth=0.65,
            zorder=1,
        )
    ax_control.scatter(
        gamma_off,
        jitter,
        s=17,
        facecolors="white",
        edgecolors="0.28",
        linewidths=0.7,
        zorder=2,
    )
    ax_control.scatter(
        gamma_on,
        1.0 + jitter,
        s=17,
        color="#0072B2",
        edgecolors="white",
        linewidths=0.35,
        zorder=2,
    )

    for y, gamma, color, marker, filled in (
        (0.0, 0.0, "0.1", "o", False),
        (1.0, 1.15, "#0072B2", "s", True),
    ):
        recorded = analysis["natural_diagonal_ranking"][str(float(gamma))]
        mean = float(recorded["mean_actual_fo_minus_lifted_ise"])
        low = float(recorded["mean_actual_fo_minus_lifted_ise_ci95_low"])
        high = float(recorded["mean_actual_fo_minus_lifted_ise_ci95_high"])
        ax_control.errorbar(
            mean,
            y,
            xerr=np.array([[mean - low], [high - mean]]),
            fmt=marker,
            markersize=6.0,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            color=color,
            capsize=2.5,
            elinewidth=1.4,
            zorder=4,
        )

    ax_control.axvline(0.0, color="0.35", linewidth=0.8, linestyle="--", zorder=0)
    ax_control.set_yticks([0.0, 1.0], [r"$\gamma=0$", r"$\gamma=1.15$"])
    ax_control.set_ylim(-0.25, 1.25)
    ax_control.set_xlim(-0.3, 1.85)
    ax_control.set_xlabel("ISE difference (first-order pair - lifted pair)")
    ax_control.set_title("(b) Diagonal pair comparison", loc="left")
    ax_control.text(
        1.82,
        1.23,
        "29.0% lower ISE\n11/12 seeds",
        ha="right",
        va="top",
        fontsize=7.4,
    )
    ax_control.text(
        1.82,
        0.0,
        "1.8% lower ISE\n6/12 seeds; CI includes 0",
        ha="right",
        va="center",
        fontsize=7.4,
    )
    ax_control.grid(axis="x", color="0.9", linewidth=0.5)

    for ax in (ax_pred, ax_control):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.085, right=0.99, top=0.91, bottom=0.18, wspace=0.34)
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


def _contains_private_path(paths: list[Path]) -> bool:
    banned = ("/Users/", "/Downloads/", "observerbench-review-phase01")
    for path in paths:
        if path.suffix.lower() not in {".json", ".csv", ".tex"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(value in text for value in banned):
            return True
    return False


def build(
    *,
    repo_root: Path,
    results_dir: Path,
    protocol_path: Path,
    outdir: Path,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    analysis_path = results_dir / "analysis.json"
    provenance_path = results_dir / "provenance.json"
    controlled_path = results_dir / "controlled_conditions.csv"
    natural_path = results_dir / "natural_factorial_conditions.csv"
    cv_path = results_dir / "leave_one_seed_out_metrics.csv"
    failures_path = results_dir / "manipulability_failures.csv"
    residual_summary_path = results_dir / "residual_displacement_summary.csv"

    protocol = _read_json(protocol_path)
    analysis = _read_json(analysis_path)
    provenance = _read_json(provenance_path)
    controlled = pd.read_csv(controlled_path)
    natural = pd.read_csv(natural_path)
    cv = pd.read_csv(cv_path)
    failures = pd.read_csv(failures_path)
    residual_summary = pd.read_csv(residual_summary_path)
    natural_effects = _natural_seed_effects(natural)
    factorial_cells = _factorial_cell_rows(natural)
    factorial_contrasts = _factorial_contrast_rows(
        natural,
        bootstrap_seed=int(protocol["bootstrap_seed"]),
        bootstrap_repeats=int(protocol["bootstrap_repeats"]),
    )
    checks = _validate(
        protocol_path=protocol_path,
        protocol=protocol,
        analysis=analysis,
        provenance=provenance,
        controlled=controlled,
        natural=natural,
        cv=cv,
        failures=failures,
        residual_summary=residual_summary,
        natural_effects=natural_effects,
        factorial_cells=factorial_cells,
        factorial_contrasts=factorial_contrasts,
    )

    summary = _summary_rows(analysis=analysis, natural_effects=natural_effects)
    csv_path = outdir / "nonlinear_suffix_summary.csv"
    tex_path = outdir / "nonlinear_suffix_summary.tex"
    png_path = outdir / "nonlinear_suffix_certificate_and_control.png"
    pdf_path = outdir / "nonlinear_suffix_certificate_and_control.pdf"
    factorial_cells_path = outdir / "nonlinear_suffix_factorial_cells.csv"
    factorial_contrasts_path = outdir / "nonlinear_suffix_factorial_contrasts.csv"
    factorial_tex_path = outdir / "nonlinear_suffix_factorial.tex"
    summary.to_csv(csv_path, index=False)
    _write_latex(summary, tex_path)
    factorial_cells.to_csv(factorial_cells_path, index=False)
    factorial_contrasts.to_csv(factorial_contrasts_path, index=False)
    _write_factorial_latex(factorial_cells, factorial_tex_path)
    _plot(
        controlled=controlled,
        natural_effects=natural_effects,
        analysis=analysis,
        png_path=png_path,
        pdf_path=pdf_path,
    )

    input_paths = [
        protocol_path,
        analysis_path,
        provenance_path,
        controlled_path,
        natural_path,
        cv_path,
        failures_path,
        residual_summary_path,
    ]
    output_paths = [
        csv_path,
        tex_path,
        png_path,
        pdf_path,
        factorial_cells_path,
        factorial_contrasts_path,
        factorial_tex_path,
    ]
    manifest = {
        "schema": "observerbench.phase05.nonlinear_suffix_artifact.v2",
        "result_status": (
            "preregistered_primary_gates_pass_secondary_factorial_attribution_negative"
        ),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "input_hashes": source_hashes(input_paths, repo_root),
        "code_hashes": source_hashes(
            [repo_root / "scripts/build_phase05_nonlinear_suffix_artifact.py"],
            repo_root,
        ),
        "output_hashes": source_hashes(output_paths, repo_root),
        "contains_private_local_path": _contains_private_path(output_paths),
        "commands_to_reproduce": [
            "python scripts/build_phase05_nonlinear_suffix_artifact.py",
        ],
    }
    manifest_path = outdir / "nonlinear_suffix_artifact_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    outdir = args.outdir.resolve()
    manifest = build(
        repo_root=repo_root,
        results_dir=args.results_dir.resolve(),
        protocol_path=args.protocol.resolve(),
        outdir=outdir,
    )
    print(portable_artifact_path(outdir, repo_root))
    return 0 if manifest["all_checks_pass"] and not manifest["contains_private_local_path"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
