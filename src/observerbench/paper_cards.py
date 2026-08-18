"""ObserverCard compositions for the checked paper result families.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .cards import (
    CTL2_THRESHOLDS,
    IOI_THRESHOLDS,
    validate_observer_card,
    write_observer_card_bundle_from_cards,
)
from .provenance import portable_artifact_path, runtime_provenance, source_hashes


def _provenance(paths: Iterable[Path], repo_root: Path) -> dict[str, Any]:
    files = tuple(paths)
    runtime = runtime_provenance(repo_root)
    return {
        "source_files": source_hashes(files, repo_root),
        "source_revision": runtime["source_revision"],
        "package_version": runtime["package_version"],
    }


def _checked_card(card: dict[str, Any]) -> dict[str, Any]:
    validate_observer_card(card)
    return card


def ctl1_sweep_card(repo_root: str | Path, *, trained: bool) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if trained:
        base = root / "results/frozen/trained_ctl1/trained_transformer_sweeps_v5_clean"
        task_name = "trained_ctl1"
        substrate = "tiny trained Transformer residual stream"
        access = "white-box learned residual representation"
        gamma_value = 1.15
        result_id = "figure_03"
        observer_name = "first_order_vs_lifted_trained_sweep"
        scope = [
            "The auxiliary feature heads are supervised.",
            "Six trained models are the uncertainty unit.",
            "Small denominators make some collateral ratios sensitive.",
        ]
    else:
        base = root / "results/frozen/ctl1_analytic/collateral_sweeps_v2"
        task_name = "ctl1_analytic"
        substrate = "analytic three-coordinate activation geometry"
        access = "white-box latent state"
        gamma_value = 0.50
        result_id = "figure_01"
        observer_name = "first_order_vs_lifted_analytic_sweep"
        scope = [
            "This is a coordinate-level one-shot intervention.",
            "The base-coordinate robustness check remains a continuous edit, not a realizable token intervention.",
            "Ten random seeds are the uncertainty unit.",
        ]

    gamma_path = base / "gamma_sweep_pairwise_summary.csv"
    nuisance_path = base / "nuisance_weight_pairwise_summary.csv"
    gamma = pd.read_csv(gamma_path)
    null = gamma.loc[gamma["gamma"].eq(0.0)].iloc[0]
    point = gamma.loc[gamma["gamma"].eq(gamma_value)].iloc[0]
    collateral_key = (
        "fo_lifted_collateral_ratio_median"
        if trained
        else "fo_lifted_collateral_ratio_mean"
    )
    target_key = (
        "fo_target_improvement_fraction_of_lifted_median"
        if trained
        else "fo_target_improvement_fraction_of_lifted_mean"
    )
    mse_key = (
        "fo_target_mse_ratio_to_lifted_median"
        if trained
        else "fo_target_mse_ratio_to_lifted_mean"
    )
    metrics = {
        "null_gamma": 0.0,
        "null_collateral_ratio_first_order_over_lifted": float(null[collateral_key]),
        "reported_gamma": gamma_value,
        "collateral_ratio_first_order_over_lifted": float(point[collateral_key]),
        "target_improvement_fraction_of_lifted": float(point[target_key]),
        "target_mse_ratio_first_order_over_lifted": float(point[mse_key]),
        "n_seeds": int(point["n"]),
    }
    failures = [
        "The first-order estimator--direction pair pays more collateral than the lifted pair at the reported interaction setting.",
        "The collateral ranking changes when the nuisance readout moves, so the observer name alone does not determine collateral cost.",
    ]
    card = {
        "task_name": task_name,
        "model_or_substrate": substrate,
        "access_regime": access,
        "observer_family": "first-order and lifted estimator--direction pairs",
        "observer_name": observer_name,
        "estimand": "one-shot target response and fixed-readout collateral movement",
        "measurement_design": "sweep interaction strength and nuisance placement while matching initial target authority",
        "validation_target": "target usefulness with low collateral movement, including a gamma-zero null",
        "primary_metrics": metrics,
        "thresholds": {
            "null_collateral_ratio_tolerance": 0.01,
            "target_competitive_fraction_min": 0.80,
            "collateral_ratio_caution": 1.20,
        },
        "failure_modes_detected": failures,
        "recommendation": "Use the lifted pair when collateral movement matters in this fixture, and report direction--nuisance overlap rather than attributing collateral to the observer name.",
        "known_scope_limits": scope,
        "commands_to_reproduce": [
            f"python scripts/reproduce_paper_fast.py --only {result_id}",
        ],
        "result_status": "frozen",
        "reproducibility_provenance": _provenance(
            [gamma_path, nuisance_path],
            root,
        ),
    }
    return _checked_card(card)


def ctl2_revision_card(audit_path: str | Path) -> dict[str, Any]:
    path = Path(audit_path).resolve()
    root = path.parent
    while root.parent != root and not (root / "pyproject.toml").exists():
        root = root.parent
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["multiseed"]["fixed_direction_ise_effects"]
    nonzero_unprojected = [
        row
        for row in rows
        if row["gamma"] > 0 and row["support_mode"] == "unprojected"
    ]
    gamma_zero = [row for row in rows if row["gamma"] == 0]
    conditioning = report["multiseed"]["response_calibration_conditioning"]
    gain = report["gain_match_small_setpoint"]
    projection = report["gamma_zero"]["affine_span_projected_base"]
    threshold = report["multiseed"]["gamma_0p5_threshold_metric"]
    default_response = report["default_interaction"]["fo_response_calibrated_unprojected"]
    gamma_1p15 = {
        row["fixed_direction"]: row
        for row in nonzero_unprojected
        if row["gamma"] == 1.15
    }
    sweep_path = root / report["inputs"]["sweep"] / "phase01_sweep_results.csv"
    metrics = {
        "nonzero_gamma_settings": len({row["gamma"] for row in nonzero_unprojected}),
        "gamma_settings": sorted({float(row["gamma"]) for row in rows}),
        "fixed_direction_contrasts_per_gamma": 2,
        "training_seeds": int(nonzero_unprojected[0]["n_seeds"]),
        "positive_seed_level_fixed_direction_contrasts": int(
            sum(row["n_positive_fo_minus_lifted"] for row in nonzero_unprojected)
        ),
        "total_seed_level_fixed_direction_contrasts": int(
            sum(row["n_seeds"] for row in nonzero_unprojected)
        ),
        "gamma_zero_max_absolute_factorial_effect": float(
            max(abs(row["mean"]) for row in gamma_zero)
        ),
        "gamma_0p5_diagonal_ise_delta_mean": float(
            threshold["diagonal_ise_fo_minus_lifted"]["mean"]
        ),
        "gamma_0p5_threshold_flag_delta_mean": float(
            threshold["diagonal_large_error_rate_fo_minus_lifted"]["mean"]
        ),
        "gamma_1p15_first_order_direction_ise_delta_mean": float(
            gamma_1p15["first_order"]["mean"]
        ),
        "gamma_1p15_first_order_direction_ise_delta_seed_min": float(
            gamma_1p15["first_order"]["seed_min"]
        ),
        "gamma_1p15_first_order_direction_ise_delta_seed_max": float(
            gamma_1p15["first_order"]["seed_max"]
        ),
        "gamma_1p15_lifted_direction_ise_delta_mean": float(
            gamma_1p15["lifted_interaction"]["mean"]
        ),
        "gamma_1p15_lifted_direction_ise_delta_seed_min": float(
            gamma_1p15["lifted_interaction"]["seed_min"]
        ),
        "gamma_1p15_lifted_direction_ise_delta_seed_max": float(
            gamma_1p15["lifted_interaction"]["seed_max"]
        ),
        "gain_matched_initial_target_mse": float(gain["initial_target_mse"]),
        "gain_matched_final_target_mse": float(gain["final_target_mse"]),
        "gain_matched_observer_error_pole": float(gain["observer_error_pole_unsaturated"]),
        "gain_matched_control_clip_fraction": float(gain["control_clip_fraction"]),
        "response_calibrated_initial_target_mse": float(default_response["initial_target_mse"]),
        "response_calibrated_final_target_mse": float(default_response["final_target_mse"]),
        "response_calibrated_observer_self_gain": float(default_response["observer_self_gain"]),
        "projected_direction_norm_ratio": float(projection["direction_norm_ratio_vs_unprojected"]),
        "projected_cumulative_collateral_abs": float(projection["cumulative_collateral_abs"]),
        "projected_mean_nearest_clean_normalized_path": float(
            projection["mean_nearest_clean_residual_normalized_path"]
        ),
        "max_absolute_response_calibration_scale": float(
            max(row["max_absolute_scale"] for row in conditioning)
        ),
    }
    card = {
        "task_name": "trained_ctl2",
        "model_or_substrate": "tiny trained Transformer final-residual representation",
        "access_regime": "white-box affine residual readouts with additive residual updates",
        "observer_family": "factorial first-order and interaction-aware estimators and directions",
        "observer_name": "first_order_estimator_factorial_evaluation",
        "estimand": "target-head state along a repeated additive residual intervention",
        "measurement_design": "hold direction fixed while comparing estimators, then hold estimator fixed while comparing directions across interaction strengths and model seeds",
        "validation_target": "integrated true target error, observer self-gain, clipping, fitted nuisance movement, and distance from clean residual states",
        "primary_metrics": metrics,
        "thresholds": {
            **CTL2_THRESHOLDS,
            "gamma_zero_max_absolute_factorial_effect": 1e-5,
            "all_nonzero_seed_level_estimator_contrasts_must_be_positive": True,
            "unsaturated_observer_error_convergence": "0 < K*g_E < 2",
        },
        "failure_modes_detected": [
            "The uncalibrated first-order estimator has greater ISE than the lifted estimator with either direction fixed at every nonzero interaction strength in all six seeds.",
            "Matching the observer-error pole does not repair true-target tracking.",
            "Affine-span projection is ill-conditioned and does not establish an on-manifold actuator.",
            "Some scalar response calibrations require very large gains.",
        ],
        "recommendation": "Do not use the uncalibrated first-order estimator for this control target. If response calibration is used, report its conditioning and validate true-target tracking; treat affine-span projection as a diagnostic only.",
        "known_scope_limits": [
            "The plant is an additive final-residual loop, not a rerun Transformer.",
            "The target is the model target head; collateral is a fitted nuisance probe.",
            "The local convergence condition applies only to the unsaturated scalar observer-error mode.",
            "Four binary inputs yield four trajectories per trained model, so training seed is the uncertainty unit.",
        ],
        "commands_to_reproduce": [
            "python scripts/build_phase04_artifact.py",
            "python scripts/reproduce_paper_fast.py --only revision_ctl2_factorial",
        ],
        "result_status": "checked_revision",
        "reproducibility_provenance": _provenance([path, sweep_path], root),
    }
    return _checked_card(card)


def _contrast(frame: pd.DataFrame, name: str) -> pd.Series:
    hit = frame.loc[frame["contrast"].eq(name)]
    if len(hit) != 1:
        raise ValueError(f"Expected one IOI contrast named {name!r}, found {len(hit)}")
    return hit.iloc[0]


def ioi_capacity_card(results_dir: str | Path) -> dict[str, Any]:
    directory = Path(results_dir).resolve()
    root = directory
    while root.parent != root and not (root / "pyproject.toml").exists():
        root = root.parent
    stage = "stage2b" if "stage2b" in directory.name else "stage2c" if "stage2c" in directory.name else None
    if stage is None:
        raise ValueError(f"Cannot infer IOI capacity stage from {directory.name!r}")

    model_path = directory / "model_comparison.csv"
    capacity_path = directory / "capacity_audit.csv"
    add_additive_path = directory / "add_one_vs_additive_head.csv"
    add_count_path = directory / "add_one_vs_count_additive.csv"
    loo_path = directory / "leave_one_out_contrasts.csv"
    manifest_path = directory / "run_manifest.json"
    model = pd.read_csv(model_path).set_index("model")
    capacity = pd.read_csv(capacity_path)
    add_additive = pd.read_csv(add_additive_path)
    add_count = pd.read_csv(add_count_path)
    loo = pd.read_csv(loo_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_add = _contrast(add_additive, "add_all_bin4_vs_additive")
    all_count = _contrast(add_count, "add_all_bin4_vs_count")
    pe_add = _contrast(add_count, "add_PE_bin4_vs_count")
    pb_add = _contrast(add_count, "add_PB_bin4_vs_count")
    pb_additive = _contrast(add_additive, "add_PB_bin4_vs_additive")
    pe_loo = _contrast(loo, "remove_PE_bin4")
    pb_loo = _contrast(loo, "remove_PB_bin4")
    single = capacity[capacity["model"].isin(
        ["count_plus_PB_bin4", "count_plus_PE_bin4", "count_plus_BE_bin4"]
    )]
    metrics: dict[str, Any] = {
        "n_subsets": int(capacity["n_subsets"].iloc[0]),
        "n_evaluated_nonclean_subsets": int(capacity["n_subsets"].iloc[0] - 1),
        "n_prompts": int(manifest["n_prompts"]),
        "cross_validation_folds": int(manifest["config"]["k_folds"]),
        "cross_validation_repeats": int(manifest["config"]["cv_repeats"]),
        "prompt_bootstrap_repeats": int(manifest["config"]["bootstrap_repeats"]),
        "count_additive_design_rank": int(capacity.loc[
            capacity["model"].eq("count_additive"), "design_rank"
        ].iloc[0]),
        "rank_added_per_single_pair": int(single["rank_added_vs_count_additive"].iloc[0]),
        "additive_head_mae": float(model.loc["additive_head", "mae"]),
        "additive_head_r2": float(model.loc["additive_head", "r2"]),
        "count_additive_mae": float(model.loc["count_additive", "mae"]),
        "capacity_matched_all_pairs_mae": float(model.loc["count_plus_all_bin4", "mae"]),
        "capacity_matched_all_pairs_r2": float(model.loc["count_plus_all_bin4", "r2"]),
        "all_pairs_delta_mae_vs_additive_mean": float(all_add["mean"]),
        "all_pairs_delta_mae_vs_additive_q05": float(all_add["q05"]),
        "all_pairs_delta_mae_vs_count_mean": float(all_count["mean"]),
        "all_pairs_delta_mae_vs_count_q05": float(all_count["q05"]),
        "PE_add_one_delta_mae_mean": float(pe_add["mean"]),
        "PE_add_one_delta_mae_q05": float(pe_add["q05"]),
        "PE_add_one_delta_mae_q95": float(pe_add["q95"]),
        "PB_add_one_delta_mae_mean": float(pb_add["mean"]),
        "PB_add_one_delta_mae_q05": float(pb_add["q05"]),
        "PB_add_one_delta_mae_q95": float(pb_add["q95"]),
        "PB_add_one_vs_additive_delta_mae_mean": float(pb_additive["mean"]),
        "PB_add_one_vs_additive_delta_mae_q05": float(pb_additive["q05"]),
        "PB_add_one_vs_additive_delta_mae_q95": float(pb_additive["q95"]),
        "PE_leave_one_out_delta_mae_mean": float(pe_loo["mean"]),
        "PE_leave_one_out_delta_mae_q05": float(pe_loo["q05"]),
        "PE_leave_one_out_delta_mae_q95": float(pe_loo["q95"]),
        "PB_leave_one_out_delta_mae_mean": float(pb_loo["mean"]),
        "PB_leave_one_out_delta_mae_q05": float(pb_loo["q05"]),
        "PB_leave_one_out_delta_mae_q95": float(pb_loo["q95"]),
    }
    source_paths = [
        model_path,
        capacity_path,
        add_additive_path,
        add_count_path,
        loo_path,
        manifest_path,
    ]
    mobius_path = directory / "mobius_bootstrap_summary.csv"
    if mobius_path.exists():
        mobius = pd.read_csv(mobius_path).set_index("term")
        metrics.update({
            "direct_PE_point": float(mobius.loc["PE", "point"]),
            "direct_PB_point": float(mobius.loc["PB", "point"]),
            "direct_PE_minus_PB_q05": float(mobius.loc["PE_minus_PB", "q05"]),
        })
        source_paths.append(mobius_path)

    design = (
        "anchored broad-random head masks"
        if stage == "stage2b"
        else "primary-stratified head masks"
    )
    limits = [
        "GPT-2-small IOI is a known-circuit diagnostic, not a general LLM benchmark.",
        "Prompt-bootstrap intervals condition on fixed masks, repeated cross-validation splits, prompt templates, and interaction bases.",
        "The pair ranking is conditional: add-one and leave-one-out contrasts answer different questions.",
    ]
    if stage == "stage2b":
        coverage_path = directory / "design_coverage.json"
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        metrics.update({
            "full_pb_corner_count": int(coverage["full_pb_corner_count"]),
            "high_pb_coverage_count": int(coverage["high_pb_coverage_count"]),
            "median_normalized_pb_exposure": float(coverage["median_normalized_pb_exposure"]),
        })
        source_paths.append(coverage_path)
        limits.append("The broad design includes two forced full-PB anchors and has limited leverage near the full PB corner.")
    card = {
        "task_name": f"ioi_{stage}",
        "model_or_substrate": "GPT-2-small IOI head-subset intervention outputs",
        "access_regime": "frozen prompt-level head-ablation effects",
        "observer_family": "capacity-matched count-bin interaction predictors",
        "observer_name": "capacity_matched_all_pairs",
        "estimand": "held-out drop in IOI logit difference for a head-ablation subset",
        "measurement_design": f"ten repeated five-fold splits over {design}; each single pair receives four added design ranks",
        "validation_target": "held-out MAE against both per-head and count-additive baselines, with add-one and leave-one-out pair contrasts",
        "primary_metrics": metrics,
        "thresholds": {
            **IOI_THRESHOLDS,
            "rank_added_per_single_pair_required": 4,
        },
        "failure_modes_detected": [
            "A per-head-only observer leaves held-out error captured by the capacity-matched interaction family.",
            "A single add-one contrast understates conditional PB contribution; leave-one-out analysis is also required.",
        ],
        "recommendation": "Use the capacity-matched all-pairs predictor for this intervention design. Keep per-head additivity as a strong baseline, and report PE as dominant while describing PB as conditional.",
        "known_scope_limits": limits,
        "commands_to_reproduce": [
            "python scripts/run_ioi_phase2_capacity.py",
            f"python scripts/reproduce_paper_fast.py --only revision_ioi_{stage}",
        ],
        "result_status": "checked_revision",
        "reproducibility_provenance": _provenance(source_paths, root),
    }
    return _checked_card(card)


def build_phase4_cards(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    ctl2_audit = root / "results/revision/phase04/ctl2_phase04_audit.json"
    if not ctl2_audit.exists():
        ctl2_audit = root / "results/revision/phase01/ctl2_phase01_audit.json"
    return [
        ctl1_sweep_card(root, trained=False),
        ctl1_sweep_card(root, trained=True),
        ctl2_revision_card(ctl2_audit),
        ioi_capacity_card(root / "results/revision/phase02/ioi_stage2b_capacity"),
        ioi_capacity_card(root / "results/revision/phase02/ioi_stage2c_capacity"),
    ]


def write_phase4_observer_cards(
    repo_root: str | Path,
    outdir: str | Path,
) -> tuple[Path, Path]:
    root = Path(repo_root).resolve()
    cards = build_phase4_cards(root)
    source_results = sorted({
        source
        for card in cards
        for source in card["reproducibility_provenance"]["source_files"]
    })
    return write_observer_card_bundle_from_cards(
        cards,
        outdir,
        source_results=source_results,
    )


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_ctl2_card_tex(cards: list[dict[str, Any]], outpath: str | Path) -> Path:
    card = next(card for card in cards if card["task_name"] == "trained_ctl2")
    metrics = card["primary_metrics"]
    positive = metrics["positive_seed_level_fixed_direction_contrasts"]
    total = metrics["total_seed_level_fixed_direction_contrasts"]
    primary_result = (
        f"FO ISE exceeds lifted ISE in {positive}/{total} seed-level contrasts "
        "across four nonzero interaction strengths with each direction held fixed."
    )
    diagnostics = (
        f"Gain-matched target MSE: {metrics['gain_matched_initial_target_mse']:.4g} to "
        f"{metrics['gain_matched_final_target_mse']:.3f}; projected direction norm ratio: "
        f"{metrics['projected_direction_norm_ratio']:.2f}; maximum response-calibration scale: "
        f"{metrics['max_absolute_response_calibration_scale']:.0f}."
    )
    rows = [
        ("Task / observer", "Ctl-2 / first-order estimator factorial evaluation"),
        ("Estimand", card["estimand"]),
        ("Measurement design", card["measurement_design"]),
        ("Validation target", card["validation_target"]),
        ("Result status", "Checked revision; source files and hashes are in the JSON card."),
        ("Primary result", primary_result),
        ("Diagnostics", diagnostics),
        ("Recommendation", card["recommendation"]),
        ("Scope", card["known_scope_limits"][0] + " " + card["known_scope_limits"][2]),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}p{0.22\linewidth}p{0.70\linewidth}@{}}",
        r"\toprule",
    ]
    for label, value in rows:
        lines.append(f"{_tex_escape(label)} & {_tex_escape(value)} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{A checked ObserverCard generated from the Ctl-2 audit. The full JSON and Markdown bundle contains the remaining task cards, thresholds, and source hashes.}",
        r"\label{tab:observercard}",
        r"\end{table}",
        "",
    ])
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
