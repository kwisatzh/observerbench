from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List
import pandas as pd
from .core import ObserverCard, ObserverResult


CARD_SCHEMA_VERSION = "observerbench.observer_card_bundle.v1"

CARD_REQUIRED_FIELDS = {
    "task_name",
    "model_or_substrate",
    "access_regime",
    "observer_family",
    "observer_name",
    "estimand",
    "measurement_design",
    "validation_target",
    "primary_metrics",
    "thresholds",
    "failure_modes_detected",
    "recommendation",
    "known_scope_limits",
    "commands_to_reproduce",
}

OBSERVER_CARD_BUNDLE_SCHEMA = {
    "schema_version": CARD_SCHEMA_VERSION,
    "type": "object",
    "required": ["schema_version", "source_results", "cards"],
    "card_required": sorted(CARD_REQUIRED_FIELDS),
}

CTL2_THRESHOLDS = {
    "integrated_squared_error_ratio_vs_reference_max": 1.25,
    "cumulative_collateral_ratio_vs_reference_max": 2.0,
    "divergence_rate_max": 0.05,
    "divergence_rate_mse_growth_max": 0.05,
    "target_error_worsened_rate_max": 0.25,
    "observer_bias_mae_path_max": 0.5,
}

IOI_THRESHOLDS = {
    "main_success_requires_beating_additive_head": True,
    "main_success_requires_beating_count_additive": True,
    "paired_delta_q05_success_min": 0.0,
    "p_delta_gt_0_success_min": 0.95,
}


def infer_failure_modes(metrics: Dict[str, float]) -> List[str]:
    failures: List[str] = []
    if metrics.get("observer_r2", 1.0) < 0.8:
        failures.append("Observer has weak held-out predictive fidelity for the target estimand.")
    if metrics.get("control_target_mse", 0.0) > metrics.get("baseline_target_mse", float("inf")):
        failures.append("Closed-loop control is worse than doing nothing on the target metric.")
    if metrics.get("target_improvement_fraction_vs_best", 1.0) < 0.8:
        failures.append("Target-control improvement is less than 80% of the best observer on this task.")
    if metrics.get("collateral_ratio_vs_best", 1.0) > 2.0:
        failures.append("Collateral movement is more than 2x the best observer on this task.")
    # High nuisance overlap is diagnostic rather than automatically disqualifying.
    # If target and collateral are both near the best observer, the benchmark should
    # say the cheap observer is sufficient instead of issuing a generic warning.
    if (
        abs(metrics.get("collateral_per_target_gain", 0.0)) > 0.5
        and metrics.get("target_improvement_fraction_vs_best", 1.0) < 0.98
        and metrics.get("collateral_ratio_vs_best", 1.0) > 1.2
    ):
        failures.append("Observer-derived actuation direction has high nuisance overlap per unit target gain.")
    return failures


def infer_recommendation(metrics: Dict[str, float], failures: List[str]) -> str:
    target_frac = metrics.get("target_improvement_fraction_vs_best", 1.0)
    collateral_ratio = metrics.get("collateral_ratio_vs_best", 1.0)
    target_mse_ratio = metrics.get("target_mse_ratio_vs_best", 1.0)

    # Null/equivalence case: the cheap observer is good enough. This is important
    # because ObserverBench should tell us when not to escalate.
    if target_frac >= 0.98 and collateral_ratio <= 1.2 and target_mse_ratio <= 1.2:
        return "This observer is sufficient under this benchmark configuration; richer observers do not materially improve target or collateral metrics."
    if target_frac >= 0.8 and collateral_ratio > 2.0:
        return "Acceptable for target-only control, but not recommended when collateral movement matters; escalate to a richer observer or lower-risk actuator."
    if target_frac < 0.8 and collateral_ratio <= 1.5:
        return "Low collateral but insufficient target control; use mainly as a diagnostic observer."
    if target_frac >= 0.8 and collateral_ratio <= 1.5:
        return "Use this observer under the benchmark configuration: target control is strong and collateral is near the best observed value."
    if failures:
        return "Use with caution; benchmark surfaced the failure modes listed above."
    return "Observer is acceptable under this benchmark configuration."


def write_cards(results: Iterable[ObserverResult], outdir: str | Path) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for res in results:
        metrics = res.metrics
        failures = res.known_failure_modes or infer_failure_modes(metrics)
        # Recommendations are result-driven by default. A caller may still set a
        # recommendation explicitly for imported legacy results, but MVP tasks
        # should leave it empty.
        recommendation = res.recommendation or infer_recommendation(metrics, failures)
        card = ObserverCard(
            observer=res.observer,
            task=res.task,
            access_regime=res.access_regime,
            estimand=res.metadata.get("estimand", "control-relevant latent state"),
            measurement_design=res.metadata.get("measurement_design", res.observer_family),
            validation_target=res.metadata.get("validation_target", "target control with low collateral"),
            metrics=metrics,
            known_failure_modes=failures,
            recommendation=recommendation,
            notes=res.metadata.get("notes", ""),
        )
        safe_name = f"{res.task}__{res.observer}".replace("/", "_").replace(" ", "_")
        card.write(out / f"{safe_name}.md")


def results_to_dataframe(results: Iterable[ObserverResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        failures = r.known_failure_modes or infer_failure_modes(r.metrics)
        recommendation = r.recommendation or infer_recommendation(r.metrics, failures)
        row = {
            "task": r.task,
            "observer": r.observer,
            "access_regime": r.access_regime,
            "observer_family": r.observer_family,
            "recommendation": recommendation,
        }
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _numeric_metrics(row: pd.Series) -> dict[str, Any]:
    skip = {
        "task",
        "observer",
        "model",
        "access_regime",
        "observer_family",
        "recommendation",
        "columns",
        "mode",
    }
    metrics: dict[str, Any] = {}
    for key, value in row.items():
        if key in skip:
            continue
        converted = _jsonable(value)
        if isinstance(converted, (int, float, bool)) or converted is None:
            metrics[key] = converted
    return metrics


def _first_existing_result_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        "trained_transformer_ctl2_results.csv",
        "trained_transformer_ctl1_results.csv",
        "collateral_task_results.csv",
        "ioi_stage2d_model_comparison.csv",
        "ioi_stage2d_fit_summary.csv",
        "ioi_stage2c_fit_summary.csv",
        "ioi_stage2b_fit_summary.csv",
        "ioi_stage1_summary.csv",
        "observerbench_results.csv",
    ]
    for name in candidates:
        candidate = path / name
        if candidate.exists():
            return candidate
    matches = sorted(path.glob("*.csv"))
    if not matches:
        raise FileNotFoundError(f"No result CSV found under {path}")
    return matches[0]


def _task_name_from_csv(path: Path) -> str:
    name = path.name
    if "ctl2" in name:
        return "trained_ctl2"
    if "ctl1" in name or "collateral_task" in name:
        return "trained_ctl1" if "trained_transformer_ctl1" in name else "ctl1_analytic"
    if "stage2d" in name:
        return "ioi_stage2d"
    if "stage2c" in name:
        return "ioi_stage2c"
    if "stage2b" in name:
        return "ioi_stage2b"
    if "stage1" in name:
        return "ioi_stage1"
    return "unknown_task"


def _task_defaults(task_name: str) -> dict[str, str | list[str]]:
    defaults: dict[str, dict[str, str | list[str]]] = {
        "ctl1_analytic": {
            "model_or_substrate": "analytic latent activation geometry",
            "access_regime": "white-box latent state",
            "observer_family": "linear observers over first-order or lifted basis",
            "estimand": "finite-control target state with configurable interaction component",
            "measurement_design": "fit observer labels, convert coefficients to fixed activation-space actuator direction, then measure target and collateral movement",
            "validation_target": "target control with low movement of a fixed nuisance readout",
            "known_scope_limits": ["Analytic Ctl-1 reproduction task only.", "No language-model claim is made by this task."],
        },
        "trained_ctl1": {
            "model_or_substrate": "tiny trained transformer residual stream",
            "access_regime": "white-box residual representation",
            "observer_family": "linear probes over learned residual features",
            "estimand": "one-shot control-relevant target state in the residual stream",
            "measurement_design": "train latent probes and compare observer-derived residual actuation directions",
            "validation_target": "target tracking with low fixed-nuisance collateral",
            "known_scope_limits": ["Tiny trained-transformer Ctl-1 reproduction task.", "Not a general language-model benchmark."],
        },
        "trained_ctl2": {
            "model_or_substrate": "tiny trained transformer residual stream",
            "access_regime": "white-box residual representation with iterative edited-state readout",
            "observer_family": "linear/oracle observers over learned residual features",
            "estimand": "closed-loop control-relevant target state in the residual stream",
            "measurement_design": "repeat observer measurement on the current edited residual state, apply clipped proportional control, and update residual state",
            "validation_target": "integrated target tracking error, cumulative collateral, observer bias, and divergence along the closed loop",
            "known_scope_limits": ["Ctl-2 reproduction task only.", "Quick runs are smoke tests and do not replace frozen paper outputs."],
        },
        "ioi_stage1": {
            "model_or_substrate": "GPT-2-small IOI head-ablation diagnostic",
            "access_regime": "activation patching/head ablation outputs",
            "observer_family": "whole-group IOI head intervention diagnostic",
            "estimand": "drop in IOI logit difference under group ablation",
            "measurement_design": "compare P, B, and P+B drops under mean ablation",
            "validation_target": "non-additive self-repair diagnostic",
            "known_scope_limits": ["Known-answer IOI diagnostic.", "Mean ablation is primary; zero ablation is robustness only."],
        },
        "ioi_stage2b": {
            "model_or_substrate": "GPT-2-small IOI head-subset outputs",
            "access_regime": "head-subset intervention outputs",
            "observer_family": "random head-subset predictive models",
            "estimand": "subset drop in IOI logit difference",
            "measurement_design": "fit subset-level predictors on random head subsets",
            "validation_target": "whether additive head terms are sufficient in the random-subset regime",
            "known_scope_limits": ["Known-answer IOI diagnostic.", "Random subset regime should not be overread as primary-stratified behavior."],
        },
        "ioi_stage2c": {
            "model_or_substrate": "GPT-2-small IOI primary-stratified outputs",
            "access_regime": "head-subset intervention outputs",
            "observer_family": "primary-stratified head-subset predictive models",
            "estimand": "subset drop in IOI logit difference",
            "measurement_design": "fit predictors on primary-stratified head subsets",
            "validation_target": "whether interaction/count terms matter under primary-stratified sampling",
            "known_scope_limits": ["Known-answer IOI diagnostic.", "Does not introduce new IOI task claims."],
        },
        "ioi_stage2d": {
            "model_or_substrate": "GPT-2-small IOI Stage 2c postprocess outputs",
            "access_regime": "postprocess of frozen/head-subset intervention outputs",
            "observer_family": "per-pair decomposition models with count-additive control",
            "estimand": "held-out prediction error for IOI subset drop",
            "measurement_design": "compare additive_head, count_additive, and pair-count models with paired bootstrap delta MAE",
            "validation_target": "P x E dominance, P x B smaller self-repair term, and count-additive control",
            "known_scope_limits": ["Postprocess only; does not rerun GPT-2.", "A model is not a main success unless it beats additive_head and count_additive."],
        },
    }
    return defaults.get(task_name, {
        "model_or_substrate": "unknown",
        "access_regime": "unknown",
        "observer_family": "unknown",
        "estimand": "unknown",
        "measurement_design": "unknown",
        "validation_target": "unknown",
        "known_scope_limits": ["Unknown result format."],
    })


def _thresholds_for(task_name: str) -> dict[str, Any]:
    if task_name == "trained_ctl2":
        return dict(CTL2_THRESHOLDS)
    if task_name.startswith("ioi_"):
        return dict(IOI_THRESHOLDS)
    return {
        "target_improvement_fraction_vs_best_min": 0.8,
        "collateral_ratio_vs_best_caution": 2.0,
        "target_mse_ratio_vs_best_near_tie": 1.2,
    }


def failure_modes_from_metrics(task_name: str, metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if task_name == "trained_ctl2":
        if (metrics.get("divergence_rate") or 0.0) > thresholds["divergence_rate_max"]:
            failures.append(f"divergence_rate exceeds threshold {thresholds['divergence_rate_max']}.")
        if (metrics.get("divergence_rate_mse_growth") or 0.0) > thresholds["divergence_rate_mse_growth_max"]:
            failures.append(f"divergence_rate_mse_growth exceeds threshold {thresholds['divergence_rate_mse_growth_max']}.")
        if (metrics.get("target_error_worsened_rate") or 0.0) > thresholds["target_error_worsened_rate_max"]:
            failures.append(f"target_error_worsened_rate exceeds threshold {thresholds['target_error_worsened_rate_max']}.")
        if (metrics.get("observer_bias_mae_path") or 0.0) > thresholds["observer_bias_mae_path_max"]:
            failures.append(f"observer_bias_mae_path exceeds threshold {thresholds['observer_bias_mae_path_max']}.")
        if (metrics.get("ise_ratio_vs_lifted") or 0.0) > thresholds["integrated_squared_error_ratio_vs_reference_max"]:
            failures.append("integrated_squared_error ratio is worse than the reference threshold.")
        if (metrics.get("cumulative_collateral_ratio_vs_lifted") or 0.0) > thresholds["cumulative_collateral_ratio_vs_reference_max"]:
            failures.append("cumulative_collateral_abs ratio is worse than the reference threshold.")
        return failures

    if task_name == "ioi_stage2d":
        beats_add = bool(metrics.get("beats_additive_head", False))
        beats_count = bool(metrics.get("beats_count_additive", False))
        if beats_count and not beats_add:
            failures.append("Beats count_additive but not additive_head; not a main success.")
        if not bool(metrics.get("main_success", False)):
            failures.append("Does not satisfy the main success rule against both baselines.")
        return failures

    if task_name.startswith("ioi_"):
        return failures

    if (metrics.get("target_improvement_fraction_vs_best") or 1.0) < thresholds["target_improvement_fraction_vs_best_min"]:
        failures.append("Target-control improvement is below threshold.")
    if (metrics.get("collateral_ratio_vs_best") or 1.0) > thresholds["collateral_ratio_vs_best_caution"]:
        failures.append("Collateral movement exceeds caution threshold.")
    return failures


def recommendation_from_metrics(task_name: str, metrics: dict[str, Any], failures: list[str]) -> str:
    if task_name == "trained_ctl2":
        if any("divergence_rate" in failure or "target_error_worsened_rate" in failure for failure in failures):
            return "Not recommended for closed-loop use under this configuration; divergence or target-worsening metrics exceed thresholds."
        if failures:
            return "Use with caution; closed-loop target tracking may be acceptable, but collateral or observer-bias thresholds were exceeded."
        return "Recommended as stable under this Ctl-2 configuration; closed-loop divergence and observer-bias metrics are within thresholds."

    if task_name == "ioi_stage2b":
        return "Random subset regime: additive terms are often sufficient; treat interaction wins as weak unless paired deltas clearly beat additive baselines."
    if task_name == "ioi_stage2c":
        return "Primary-stratified regime: interaction/count terms matter; compare against additive baselines before calling a model successful."
    if task_name == "ioi_stage2d":
        if bool(metrics.get("is_dominant_single_pair", False)):
            return "Stage 2d: this is the dominant single-pair interaction by paired Delta MAE; the paper fixture identifies P x E as dominant while P x B is smaller."
        if failures:
            return "Stage 2d: not a main success under the required two-baseline rule; do not label success from count_additive alone."
        return "Stage 2d: passes the two-baseline rule; interpret alongside the P x E dominant and smaller P x B self-repair terms."
    if task_name == "ioi_stage1":
        interaction = metrics.get("interaction")
        if interaction is not None and interaction > 0:
            return "Stage 1 diagnostic shows positive whole-group self-repair interaction under mean ablation."
        return "Stage 1 diagnostic does not show a positive whole-group self-repair interaction under this configuration."

    if not failures:
        return "Metrics are within configured thresholds for this reproduction task."
    if any("Collateral" in failure or "collateral" in failure for failure in failures):
        return "Acceptable for target-only control, but not recommended when collateral movement matters."
    return "Use with caution; benchmark surfaced metric-threshold failures."


def _primary_metrics(task_name: str, row: pd.Series) -> dict[str, Any]:
    metrics = _numeric_metrics(row)
    if task_name == "trained_ctl2":
        wanted = [
            "integrated_squared_error",
            "cumulative_collateral_abs",
            "divergence_rate",
            "divergence_rate_mse_growth",
            "target_error_worsened_rate",
            "observer_bias_mae_path",
            "ise_ratio_vs_lifted",
            "cumulative_collateral_ratio_vs_lifted",
        ]
        return {key: metrics.get(key) for key in wanted if key in metrics}
    if task_name == "ioi_stage2d":
        wanted = [
            "mae_mean",
            "mae",
            "delta_mae_vs_additive_mean",
            "delta_mae_vs_additive_q05",
            "delta_mae_vs_additive_q95",
            "p_delta_vs_additive_gt_0",
            "delta_mae_vs_count_additive_mean",
            "delta_mae_vs_count_additive_q05",
            "delta_mae_vs_count_additive_q95",
            "p_delta_vs_count_additive_gt_0",
            "beats_additive_head",
            "beats_count_additive",
            "main_success",
            "is_dominant_single_pair",
        ]
        out = {key: metrics.get(key) for key in wanted if key in metrics}
        if "dominant_single_pair" in row:
            out["dominant_single_pair"] = _jsonable(row["dominant_single_pair"])
        return out
    if task_name == "ioi_stage1":
        wanted = [
            "drop_P",
            "drop_B",
            "drop_P+B",
            "interaction",
            "interaction_fraction_of_joint",
            "backup_conditional_amplification",
        ]
        return {key: metrics.get(key) for key in wanted if key in metrics}
    return metrics


def _enrich_stage2d(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns or "delta_mae_vs_additive_mean" not in df.columns:
        return df
    out = df.copy()
    single_pairs = {
        "count_plus_PB_count": "P_B",
        "count_plus_PE_count": "P_E",
        "count_plus_BE_count": "B_E",
    }
    candidates = out[out["model"].isin(single_pairs)]
    if candidates.empty:
        return out
    dominant_idx = candidates["delta_mae_vs_additive_mean"].astype(float).idxmax()
    dominant_model = str(out.loc[dominant_idx, "model"])
    dominant_pair = single_pairs[dominant_model]
    out["dominant_single_pair"] = dominant_pair
    out["is_dominant_single_pair"] = out["model"].eq(dominant_model)
    return out


def _cards_from_dataframe(df: pd.DataFrame, task_name: str, results_path: Path, outdir: Path) -> list[dict[str, Any]]:
    if task_name == "ioi_stage2d":
        df = _enrich_stage2d(df)
    defaults = _task_defaults(task_name)
    thresholds = _thresholds_for(task_name)
    cards: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        observer_name = str(row.get("observer", row.get("model", "task_summary")))
        metrics = _primary_metrics(task_name, row)
        failures = failure_modes_from_metrics(task_name, metrics, thresholds)
        recommendation = recommendation_from_metrics(task_name, metrics, failures)
        card = {
            "task_name": task_name,
            "model_or_substrate": defaults["model_or_substrate"],
            "access_regime": str(row.get("access_regime", defaults["access_regime"])),
            "observer_family": str(row.get("observer_family", defaults["observer_family"])),
            "observer_name": observer_name,
            "estimand": defaults["estimand"],
            "measurement_design": defaults["measurement_design"],
            "validation_target": defaults["validation_target"],
            "primary_metrics": metrics,
            "thresholds": thresholds,
            "failure_modes_detected": failures,
            "recommendation": recommendation,
            "known_scope_limits": list(defaults["known_scope_limits"]),
            "commands_to_reproduce": [
                f"observerbench make-card --results {results_path} --outdir {outdir}",
            ],
        }
        validate_observer_card(card)
        cards.append(card)
    return cards


def load_cards_from_results(results: str | Path, outdir: str | Path) -> list[dict[str, Any]]:
    results_path = Path(results)
    out_path = Path(outdir)
    csv_path = _first_existing_result_csv(results_path)
    task_name = _task_name_from_csv(csv_path)
    df = pd.read_csv(csv_path)
    return _cards_from_dataframe(df, task_name, results_path, out_path)


def validate_observer_card(card: dict[str, Any]) -> None:
    missing = CARD_REQUIRED_FIELDS - set(card)
    if missing:
        raise ValueError(f"ObserverCard missing fields: {sorted(missing)}")
    for key in [
        "task_name",
        "model_or_substrate",
        "access_regime",
        "observer_family",
        "observer_name",
        "estimand",
        "measurement_design",
        "validation_target",
        "recommendation",
    ]:
        if not isinstance(card[key], str):
            raise ValueError(f"ObserverCard field {key} must be a string")
    if not isinstance(card["primary_metrics"], dict):
        raise ValueError("ObserverCard primary_metrics must be an object")
    if not isinstance(card["thresholds"], dict):
        raise ValueError("ObserverCard thresholds must be an object")
    for key in ["failure_modes_detected", "known_scope_limits", "commands_to_reproduce"]:
        if not isinstance(card[key], list) or not all(isinstance(item, str) for item in card[key]):
            raise ValueError(f"ObserverCard field {key} must be a list of strings")


def validate_observer_card_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema_version") != CARD_SCHEMA_VERSION:
        raise ValueError("Unknown observer card schema version")
    cards = bundle.get("cards")
    if not isinstance(cards, list) or not cards:
        raise ValueError("ObserverCard bundle must contain a non-empty cards list")
    for card in cards:
        if not isinstance(card, dict):
            raise ValueError("ObserverCard bundle cards must be objects")
        validate_observer_card(card)


def observer_cards_to_markdown(cards: list[dict[str, Any]]) -> str:
    lines = ["# ObserverCards", ""]
    for card in cards:
        lines.extend(
            [
                f"## {card['task_name']} / {card['observer_name']}",
                "",
                f"**Task name.** {card['task_name']}",
                f"**Observer name.** {card['observer_name']}",
                f"**Model or substrate.** {card['model_or_substrate']}",
                f"**Access regime.** {card['access_regime']}",
                f"**Observer family.** {card['observer_family']}",
                "",
                f"**Estimand.** {card['estimand']}",
                f"**Measurement design.** {card['measurement_design']}",
                f"**Validation target.** {card['validation_target']}",
                "",
                "### Primary Metrics",
            ]
        )
        for key, value in card["primary_metrics"].items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "### Thresholds"])
        for key, value in card["thresholds"].items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "### Failure Modes Detected"])
        if card["failure_modes_detected"]:
            lines.extend(f"- {failure}" for failure in card["failure_modes_detected"])
        else:
            lines.append("- None detected under configured thresholds.")
        lines.extend(["", "### Recommendation", card["recommendation"], "", "### Scope limits"])
        lines.extend(f"- {limit}" for limit in card["known_scope_limits"])
        lines.extend(["", "### Commands to reproduce"])
        lines.extend(f"- `{command}`" for command in card["commands_to_reproduce"])
        lines.append("")
    return "\n".join(lines)


def write_observer_card_bundle(results: str | Path, outdir: str | Path) -> tuple[Path, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cards = load_cards_from_results(results, out)
    bundle = {
        "schema_version": CARD_SCHEMA_VERSION,
        "source_results": str(results),
        "cards": cards,
    }
    validate_observer_card_bundle(bundle)
    json_path = out / "observer_card.json"
    md_path = out / "observer_card.md"
    json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(observer_cards_to_markdown(cards), encoding="utf-8")
    return json_path, md_path
