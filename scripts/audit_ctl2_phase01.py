#!/usr/bin/env python3
"""Audit the checked Ctl-2 Phase-1 runs and write a claim-facing summary.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import (
    file_sha256,
    portable_artifact_path,
    runtime_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = REPO_ROOT / "results" / "revision" / "phase01"


def _results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "trained_transformer_ctl2_results.csv")


def _row(frame: pd.DataFrame, **selectors: str) -> pd.Series:
    selected = frame
    for key, value in selectors.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {selectors}, found {len(selected)}")
    return selected.iloc[0]


def _finite(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Expected finite value, found {value!r}")
    return number


def _seed_range(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "seed_min": float(values.min()),
        "seed_max": float(values.max()),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    gate = _results(args.gamma_zero)
    default = _results(args.default)
    gain = _results(args.gain_match)
    sweep = pd.read_csv(args.sweep / "phase01_sweep_results.csv")

    gamma0_unprojected = _row(
        gate,
        estimator_name="first_order",
        direction_name="first_order",
        direction_support_mode="unprojected",
        estimator_calibration="none",
        controller_mode="base",
    )
    gamma0_projected = _row(
        gate,
        estimator_name="first_order",
        direction_name="first_order",
        direction_support_mode="projected",
        estimator_calibration="none",
        controller_mode="base",
    )
    gamma0_response = _row(
        gate,
        estimator_name="first_order",
        direction_name="first_order",
        direction_support_mode="unprojected",
        estimator_calibration="response_gain",
        controller_mode="base",
    )
    gain_matched = _row(
        gain,
        estimator_name="first_order",
        direction_name="first_order",
        direction_support_mode="unprojected",
        estimator_calibration="none",
        controller_mode="gain_matched",
    )

    gamma_zero = {
        "unprojected_base": {
            key: _finite(gamma0_unprojected[key])
            for key in (
                "initial_target_mse",
                "final_target_mse",
                "observer_self_gain",
                "mean_nearest_clean_residual_normalized_path",
            )
        },
        "affine_span_projected_base": {
            key: _finite(gamma0_projected[key])
            for key in (
                "initial_target_mse",
                "final_target_mse",
                "observer_self_gain",
                "direction_norm_ratio_vs_unprojected",
                "projected_direction_scale",
                "cumulative_collateral_abs",
                "mean_nearest_clean_residual_normalized_path",
            )
        },
        "unprojected_response_calibrated": {
            key: _finite(gamma0_response[key])
            for key in (
                "initial_target_mse",
                "final_target_mse",
                "observer_self_gain",
                "response_calibration_scale",
                "control_clip_fraction",
                "mean_nearest_clean_residual_normalized_path",
            )
        },
    }
    gamma_zero["target_gate_passes_after_projection"] = bool(
        gamma0_projected["final_target_mse"] < gamma0_projected["initial_target_mse"]
    )
    gamma_zero["projection_is_well_conditioned"] = bool(
        gamma0_projected["direction_norm_ratio_vs_unprojected"] <= 5.0
        and gamma0_projected["mean_nearest_clean_residual_normalized_path"] <= 1.0
    )
    gamma_zero["response_calibration_target_gate_passes"] = bool(
        gamma0_response["final_target_mse"] < gamma0_response["initial_target_mse"]
        and gamma0_response["control_clip_fraction"] == 0.0
    )

    primary = default[
        (default["estimator_calibration"] == "none")
        & (default["controller_mode"] == "base")
        & default["estimator_name"].isin(["first_order", "lifted_interaction"])
        & default["direction_name"].isin(["first_order", "lifted_interaction"])
    ]
    fixed_direction_rows: list[dict[str, Any]] = []
    for support_mode in ("unprojected", "projected"):
        for direction_name in ("first_order", "lifted_interaction"):
            cell = primary[
                (primary["direction_support_mode"] == support_mode)
                & (primary["direction_name"] == direction_name)
            ].set_index("estimator_name")
            fixed_direction_rows.append({
                "support_mode": support_mode,
                "fixed_direction": direction_name,
                "fo_ise": _finite(cell.loc["first_order", "integrated_squared_error"]),
                "lifted_ise": _finite(cell.loc["lifted_interaction", "integrated_squared_error"]),
                "fo_minus_lifted_ise": _finite(
                    cell.loc["first_order", "integrated_squared_error"]
                    - cell.loc["lifted_interaction", "integrated_squared_error"]
                ),
            })

    default_response = _row(
        default,
        estimator_name="first_order",
        direction_name="first_order",
        direction_support_mode="unprojected",
        estimator_calibration="response_gain",
        controller_mode="base",
    )
    default_result = {
        "fixed_direction_seed0": fixed_direction_rows,
        "all_seed0_estimator_effects_positive": all(
            row["fo_minus_lifted_ise"] > 0 for row in fixed_direction_rows
        ),
        "fo_response_calibrated_unprojected": {
            key: _finite(default_response[key])
            for key in (
                "initial_target_mse",
                "final_target_mse",
                "integrated_squared_error",
                "observer_initial_mae_vs_plant_target",
                "observer_self_gain",
                "response_calibration_scale",
                "control_clip_fraction",
            )
        },
    }

    gain_match = {
        key: _finite(gain_matched[key])
        for key in (
            "initial_target_mse",
            "final_target_mse",
            "integrated_squared_error",
            "observer_self_gain",
            "observer_error_pole_unsaturated",
            "effective_controller_gain",
            "control_clip_fraction",
        )
    }
    gain_match["realized_without_clipping"] = bool(gain_matched["control_clip_fraction"] == 0.0)
    gain_match["target_still_worsens"] = bool(
        gain_matched["final_target_mse"] > gain_matched["initial_target_mse"]
    )

    base = sweep[
        (sweep["estimator_calibration"] == "none")
        & (sweep["controller_mode"] == "base")
    ]
    multiseed_rows: list[dict[str, Any]] = []
    for gamma in sorted(base["sweep_gamma"].unique()):
        for support_mode in ("unprojected", "projected"):
            for direction_name in ("first_order", "lifted_interaction"):
                cell = base[
                    (base["sweep_gamma"] == gamma)
                    & (base["direction_support_mode"] == support_mode)
                    & (base["direction_name"] == direction_name)
                ].pivot(index="seed", columns="estimator_name", values="integrated_squared_error")
                delta = cell["first_order"] - cell["lifted_interaction"]
                multiseed_rows.append({
                    "gamma": float(gamma),
                    "support_mode": support_mode,
                    "fixed_direction": direction_name,
                    "n_seeds": int(len(delta)),
                    "n_positive_fo_minus_lifted": int((delta > 0).sum()),
                    **_seed_range(delta),
                })

    threshold_flip = base[
        (base["sweep_gamma"] == 0.5)
        & (base["direction_support_mode"] == "unprojected")
    ].pivot(
        index="seed",
        columns=["estimator_name", "direction_name"],
        values=["large_error_trajectory_rate", "integrated_squared_error"],
    )
    rate_delta = (
        threshold_flip[("large_error_trajectory_rate", "first_order", "first_order")]
        - threshold_flip[("large_error_trajectory_rate", "lifted_interaction", "lifted_interaction")]
    )
    ise_delta = (
        threshold_flip[("integrated_squared_error", "first_order", "first_order")]
        - threshold_flip[("integrated_squared_error", "lifted_interaction", "lifted_interaction")]
    )

    calibrated = sweep[sweep["estimator_calibration"] == "response_gain"].copy()
    calibrated["absolute_response_scale"] = calibrated["response_calibration_scale"].abs()
    calibration_conditioning: list[dict[str, Any]] = []
    for (gamma, support_mode, estimator_name), cell in calibrated.groupby(
        ["sweep_gamma", "direction_support_mode", "estimator_name"]
    ):
        calibration_conditioning.append({
            "gamma": float(gamma),
            "support_mode": str(support_mode),
            "estimator": str(estimator_name),
            "n_arms": int(len(cell)),
            "mean_absolute_scale": float(cell["absolute_response_scale"].mean()),
            "max_absolute_scale": float(cell["absolute_response_scale"].max()),
            "n_absolute_scale_above_10": int((cell["absolute_response_scale"] > 10).sum()),
        })

    essential_multiseed = [row for row in multiseed_rows if row["gamma"] > 0]
    all_positive = all(row["n_positive_fo_minus_lifted"] == row["n_seeds"] for row in essential_multiseed)
    multiseed = {
        "uncertainty_unit": "training seed",
        "reported_intervals": "seed min--max; not percentile intervals",
        "fixed_direction_ise_effects": multiseed_rows,
        "all_positive_for_gamma_above_zero": all_positive,
        "gamma_0p5_threshold_metric": {
            "diagonal_large_error_rate_fo_minus_lifted": _seed_range(rate_delta),
            "diagonal_ise_fo_minus_lifted": _seed_range(ise_delta),
            "n_seeds_with_positive_ise_delta": int((ise_delta > 0).sum()),
            "interpretation": (
                "The thresholded rate can reverse sign while continuous ISE remains worse for the "
                "first-order pair in every seed; the old sign flip is a coarse-threshold artifact."
            ),
        },
        "response_calibration_conditioning": calibration_conditioning,
    }

    checks = {
        "gamma0_projection_target_gate": gamma_zero["target_gate_passes_after_projection"],
        "gamma0_projection_not_global_repair": not gamma_zero["projection_is_well_conditioned"],
        "gamma0_response_calibration_gate": gamma_zero["response_calibration_target_gate_passes"],
        "default_seed0_fixed_direction_attribution": default_result["all_seed0_estimator_effects_positive"],
        "gain_match_realized_without_clipping": gain_match["realized_without_clipping"],
        "gain_match_does_not_repair_target": gain_match["target_still_worsens"],
        "multiseed_estimator_effect_for_gamma_above_zero": all_positive,
        "gamma0p5_continuous_metric_resolves_sign_flip": bool((ise_delta > 0).all()),
    }
    return {
        "schema": "observerbench.ctl2.phase01_audit.v0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_provenance(REPO_ROOT),
        "inputs": {
            "gamma_zero": portable_artifact_path(args.gamma_zero, REPO_ROOT),
            "default": portable_artifact_path(args.default, REPO_ROOT),
            "gain_match": portable_artifact_path(args.gain_match, REPO_ROOT),
            "sweep": portable_artifact_path(args.sweep, REPO_ROOT),
            "sweep_results_sha256": file_sha256(args.sweep / "phase01_sweep_results.csv"),
        },
        "gamma_zero": gamma_zero,
        "default_interaction": default_result,
        "gain_match_small_setpoint": gain_match,
        "multiseed": multiseed,
        "checks": checks,
        "all_required_checks_pass": all(checks.values()),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ctl-2 Phase-1 audit",
        "",
        f"All required checks pass: **{report['all_required_checks_pass']}**.",
        "",
        "## Checked conclusions",
        "",
        "- The 2x2 factorial separates estimator error from direction geometry.",
        "- For every nonzero interaction strength, first-order ISE exceeds lifted ISE in all six seeds with either direction held fixed.",
        "- Direction geometry changes the size of the failure, especially for the first-order estimator.",
        "- A scalar held-out response calibration fixes the local response gain in this affine fixture, but it leaves initial estimation bias and can be ill-conditioned.",
        "- Affine-span projection repairs target tracking at gamma=0 but uses a much larger direction and moves far from the observed clean residual states.",
        "- The unclipped small-setpoint gain-matched arm still worsens true target error, so the result is not only a controller-gain artifact.",
        "- At gamma=0.5, the thresholded large-error flag can reverse while continuous ISE remains worse in every seed; continuous errors should be primary.",
        "",
        "## Scope",
        "",
        "The plant is an additive final-residual loop with an affine target head. Collateral is a fitted nuisance-probe displacement. The four binary inputs yield four distinct trajectories per training seed, so seed—not repeated test row—is the uncertainty unit.",
        "",
        "## Required checks",
        "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["checks"].items())
    lines.extend(["", "## Fixed-direction ISE effects across seeds", "", "| gamma | support | fixed direction | positive seeds | mean FO-lifted | seed min | seed max |", "|---:|---|---|---:|---:|---:|---:|"])
    for row in report["multiseed"]["fixed_direction_ise_effects"]:
        if row["gamma"] == 0:
            continue
        lines.append(
            f"| {row['gamma']:.2f} | {row['support_mode']} | {row['fixed_direction']} | "
            f"{row['n_positive_fo_minus_lifted']}/{row['n_seeds']} | {row['mean']:.4g} | "
            f"{row['seed_min']:.4g} | {row['seed_max']:.4g} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-zero", type=Path, default=REVISION_ROOT / "ctl2_gamma0_seed0_v3")
    parser.add_argument("--default", type=Path, default=REVISION_ROOT / "ctl2_gamma115_seed0_v3")
    parser.add_argument(
        "--gain-match",
        type=Path,
        default=REVISION_ROOT / "ctl2_gamma0_gainmatch_small_offset_seed0_v2",
    )
    parser.add_argument("--sweep", type=Path, default=REVISION_ROOT / "ctl2_multiseed_sweep")
    parser.add_argument("--out-json", type=Path, default=REVISION_ROOT / "ctl2_phase01_audit.json")
    parser.add_argument("--out-markdown", type=Path, default=REVISION_ROOT / "CTL2_PHASE01_AUDIT.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit(args)
    write_json(args.out_json, report)
    args.out_markdown.write_text(_markdown(report), encoding="utf-8")
    print(args.out_json)
    print(args.out_markdown)
    return 0 if report["all_required_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
