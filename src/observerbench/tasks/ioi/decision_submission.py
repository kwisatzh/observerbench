"""Compose IOI effect-table submissions with the fixed action selector.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.effect_prediction import (
    EffectObserverCard, EffectTaskCard, FiniteEffectMeasurement,
    FiniteEffectPredictionTask, FiniteEffectQuery, FiniteEffectTarget,
    evaluate_effect_predictions, read_effect_predictions,
)
from observerbench.provenance import file_sha256, runtime_provenance, source_hashes
from observerbench.tasks.ioi.phase5_analysis import _choose, _two_way_cluster_draws


TASK_ID = "ioi-gpt2-small-action-selection@phase5-noop-v1-b160"
PREDICTION_TASK_ID = "ioi-gpt2-small-finite-effects@phase5-test-v1-b160"
PACK_SCHEMA = "observerbench.ioi_decision_replay.v1"
RESULT_SCHEMA = "observerbench.ioi_decision_result.v1"
DEFAULT_PACK = Path(__file__).resolve().parents[4] / "practice" / "ioi_decision_v1"
TARGETS = (0.5, 1.0, 1.5)
NOOP = "analytic_noop"


def load_decision_pack(root: str | Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Check the open replay pack without opening held-out responses."""
    root = Path(root)
    card = json.loads((root / "task_card.json").read_text())
    if (card.get("schema") != PACK_SCHEMA or card.get("task_id") != TASK_ID
            or card.get("prediction_task_id") != PREDICTION_TASK_ID):
        raise ValueError("unsupported IOI decision task or prediction task")
    if (card.get("targets") != list(TARGETS) or card.get("primary_target") != 1.0
            or card.get("selector") != "absolute_mean_error_then_head_count_then_mask_id"
            or card.get("include_noop") is not True
            or card.get("participation_class") != "open_replay"):
        raise ValueError("target, selector, no-op, or participation contract differs")
    required = {"queries.csv", "calibration_measurements.csv", "responses.npz", "prompt_clusters.csv"}
    if not required.issubset(card.get("files", {})):
        raise ValueError("incomplete decision pack")
    for name, digest in card["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("pack file escapes its root or is missing")
        if file_sha256(path) != digest:
            raise ValueError(f"decision-pack hash mismatch: {name}")
    queries = pd.read_csv(root / "queries.csv", dtype={"query_id": str, "mask_bits": str, "pool_id": str})
    calibration = pd.read_csv(root / "calibration_measurements.csv", dtype={"measurement_id": str, "mask_bits": str})
    if queries["query_id"].duplicated().any() or calibration["measurement_id"].duplicated().any():
        raise ValueError("duplicate query or measurement IDs")
    if (len(queries) != card["n_queries"] or len(calibration) != 160
            or set(queries.query_id) & set(calibration.measurement_id)
            or NOOP in set(queries.query_id)):
        raise ValueError("query counts, fit/test separation, or no-op is invalid")
    for row in queries.itertuples():
        if (len(row.mask_bits) != 13 or set(row.mask_bits) - {"0", "1"}
                or row.n_heads != row.mask_bits.count("1") or not row.pool_id):
            raise ValueError("invalid head mask or pool")
    if queries.groupby("pool_id").size().to_dict() != card["pool_sizes"]:
        raise ValueError("candidate pool membership differs")
    return card, queries, calibration


def select_decision_actions(queries: pd.DataFrame, predictions: dict[str, float]) -> dict:
    """Use the paper's selector; no held-out response is an input."""
    if set(predictions) != set(queries.query_id) or not np.isfinite(list(predictions.values())).all():
        raise ValueError("predictions must cover every query exactly with finite values")
    selected = {}
    for pool_id, group in queries.groupby("pool_id", sort=True):
        pool = group.rename(columns={"query_id": "mask_id"})[["mask_id", "n_heads"]]
        pool = pd.concat([pool, pd.DataFrame([{"mask_id": NOOP, "n_heads": 0}])], ignore_index=True)
        predicted = np.asarray([0.0 if name == NOOP else predictions[name] for name in pool.mask_id])
        for target in TARGETS:
            choice = _choose(pool, predicted, np.zeros(len(pool)), target,
                             head_cost_penalty=0.0, target_tolerance=0.25)
            selected[(str(pool_id), target)] = {
                "selected_mask_id": choice["selected_mask_id"],
                "selected_head_count": choice["selected_head_count"],
                "predicted_effect": choice["predicted_effect"],
            }
    return selected


def evaluate_ioi_decision_csv(
    task_id: str, *, pack_root: str | Path, predictions_path: str | Path,
    observer_card_path: str | Path, outdir: str | Path,
) -> dict:
    """Return prediction scores, selected actions, and paired action contrasts."""
    if task_id != TASK_ID:
        raise ValueError(f"IOI decision task must be {TASK_ID}")
    root, output = Path(pack_root), Path(outdir)
    card, queries, calibration = load_decision_pack(root)
    payload = json.loads(Path(observer_card_path).read_text())
    if payload.pop("schema_version", None) != "observerbench.effect_observer_card.v0":
        raise ValueError("unsupported observer card schema")
    observer = EffectObserverCard(**payload)
    for key, expected in (("prediction_task_id", PREDICTION_TASK_ID), ("decision_task_id", TASK_ID),
                          ("prediction_estimand", "mean_finite_effect")):
        if key in observer.metadata and observer.metadata[key] != expected:
            raise ValueError(f"observer {key} does not match this decision contract")
    supplied = read_effect_predictions(predictions_path)
    prediction_map = {row.query_id: row.predicted_effect for row in supplied}
    # Selection is completed for the submission and references before reading Y.
    choices = {"submission": select_decision_actions(queries, prediction_map)}
    for name, record in card["references"].items():
        if name == "submission" or name == NOOP:
            raise ValueError("reserved reference name")
        if record["predictions"] not in card["files"]:
            raise ValueError("reference predictions must be covered by the pack hashes")
        reference = read_effect_predictions(root / record["predictions"])
        choices[name] = select_decision_actions(queries, {r.query_id: r.predicted_effect for r in reference})
    choices[NOOP] = {(pool, target): {"selected_mask_id": NOOP, "selected_head_count": 0, "predicted_effect": 0.0}
                    for pool, target in choices["submission"]}

    clusters = pd.read_csv(root / "prompt_clusters.csv", dtype=str)
    with np.load(root / "responses.npz", allow_pickle=False) as stored:
        y = stored["responses"].copy()
        mask_ids = stored["mask_ids"].astype(str).tolist()
        prompt_ids = stored["prompt_ids"].astype(str).tolist()
    if (len(set(mask_ids)) != len(mask_ids) or set(mask_ids) != set(queries.query_id)
            or len(set(prompt_ids)) != len(prompt_ids)
            or len(prompt_ids) != card["n_prompts"]
            or clusters.isna().any().any()
            or clusters.prompt_id.duplicated().any() or set(clusters.prompt_id) != set(prompt_ids)
            or y.shape != (len(prompt_ids), len(mask_ids)) or not np.isfinite(y).all()):
        raise ValueError("incomplete or non-finite held-out response matrix")
    response_by_mask = {mask: y[:, i] for i, mask in enumerate(mask_ids)}
    response_by_mask[NOOP] = np.zeros(len(prompt_ids))

    prediction_card = EffectTaskCard(
        task_name=PREDICTION_TASK_ID.split("@")[0], task_version=PREDICTION_TASK_ID.split("@")[1],
        summary="Frozen GPT-2-small IOI mean-effect prediction, paired with a separate action contract",
        model_or_substrate=card["model"], access_regime=card["information_boundary"],
        estimand="mean_finite_effect", intervention_family="template-conditioned final-token mean ablation",
        measurement_design="160 calibration masks; 320 held-out candidate masks",
        validation_target="mean response over 256 held-out prompts", train_split="train", evaluation_split="test",
    )
    task = FiniteEffectPredictionTask(
        name=prediction_card.task_name, version=prediction_card.task_version, card=prediction_card,
        measurements=[FiniteEffectMeasurement(r.measurement_id, r.mask_bits, r.observed_effect) for r in calibration.itertuples()],
        queries=[FiniteEffectQuery(r.query_id, r.mask_bits) for r in queries.itertuples()],
        targets=[FiniteEffectTarget(mask, float(response_by_mask[mask].mean())) for mask in queries.query_id],
    )
    output.mkdir(parents=True, exist_ok=True)
    prediction_result = evaluate_effect_predictions(task, supplied, observer, outdir=output)
    frames, action_rows = {}, []
    for name, selection in choices.items():
        rows = []
        for (pool, target), action in selection.items():
            effects = response_by_mask[action["selected_mask_id"]]
            losses = np.abs(effects - target)
            action_rows.append({"observer": name, "pool_id": pool, "target": target, **action,
                                "actual_mean_effect": float(effects.mean()), "action_loss": float(losses.mean())})
            rows.extend({"prompt_id": prompt, "pool_id": pool, "target": target, "value": float(loss)}
                        for prompt, loss in zip(prompt_ids, losses))
        frames[name] = pd.DataFrame(rows)

    target_results = []
    for target in TARGETS:
        own = frames["submission"].query("target == @target")
        values = _two_way_cluster_draws(own, clusters, cluster_column="cluster_id",
                                        repeats=card["bootstrap_repeats"], seed=card["bootstrap_seed"])
        row = {"target": target, "action_loss": float(own.value.mean()),
               "action_loss_interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
               "noop_loss": target, "excess_loss_over_noop": float(own.value.mean() - target),
               "selected_noop_pools": sum(a["selected_mask_id"] == NOOP for (p, t), a in choices["submission"].items() if t == target),
               "references": {}}
        for name, frame in frames.items():
            if name == "submission":
                continue
            reference = frame.query("target == @target")
            paired = own.merge(reference, on=["prompt_id", "pool_id", "target"], validate="one_to_one", suffixes=("_own", "_ref"))
            paired["value"] = paired.value_own - paired.value_ref
            delta = _two_way_cluster_draws(paired, clusters, cluster_column="cluster_id",
                                           repeats=card["bootstrap_repeats"], seed=card["bootstrap_seed"])
            row["references"][name] = {"action_loss": float(reference.value.mean()),
                "submission_minus_reference": float(paired.value.mean()),
                "paired_interval_95": np.quantile(delta, [0.025, 0.975]).tolist()}
        target_results.append(row)
    result = {
        "schema_version": RESULT_SCHEMA, "task_id": TASK_ID,
        "prediction_task_id": PREDICTION_TASK_ID, "observer_name": observer.observer_name,
        "observer_version": observer.observer_version, "access_regime": observer.access_regime,
        "participation_class": "open_replay", "sealed_score": False,
        "prediction_metrics": dict(prediction_result.metrics), "primary_target": 1.0,
        "by_target": target_results, "uncertainty": card["uncertainty"],
        "task_card_sha256": file_sha256(root / "task_card.json"),
        "prediction_sha256": file_sha256(predictions_path),
        "observer_card_sha256": file_sha256(observer_card_path),
        "scope": card["scope"],
        "runtime": runtime_provenance(Path(__file__).resolve().parents[4]),
        "scorer_source_hashes": source_hashes(
            [__file__, inspect.getfile(_choose), inspect.getfile(evaluate_effect_predictions)],
            Path(__file__).resolve().parents[4]),
    }
    write_json(output / "decision_evaluation.json", result)
    write_json(output / "decision_task_card.json", card)
    pd.DataFrame(action_rows).to_csv(output / "selected_actions.csv", index=False)
    lines = [f"Observer: {observer.observer_name}", f"Task: {TASK_ID}",
             "Open replay: not a sealed leaderboard submission.",
             f"Prediction MAE: {prediction_result.metrics['mae']:.4f}"]
    for row in target_results:
        lines.append(f"Target {row['target']:g}: action loss {row['action_loss']:.4f}; no-op {row['noop_loss']:.4f}; excess {row['excess_loss_over_noop']:+.4f}")
        for name, contrast in row["references"].items():
            lo, hi = contrast["paired_interval_95"]
            lines.append(f"  versus {name}: {contrast['submission_minus_reference']:+.4f} [{lo:+.4f}, {hi:+.4f}] (negative favors submission)")
    (output / "scorecard.txt").write_text("\n".join(lines) + "\n")
    return result
