"""Package existing IOI measurements and replay submitted effect tables.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.effect_prediction import EffectObserverCard, FiniteEffectPrediction, write_effect_predictions
from observerbench.provenance import file_sha256
from observerbench.tasks.ioi.decision_submission import (
    DEFAULT_PACK, PACK_SCHEMA, PREDICTION_TASK_ID, TASK_ID, TARGETS, evaluate_ioi_decision_csv,
)
from observerbench.tasks.ioi.effect_task import load_ioi_effect_prediction_task
from observerbench.tasks.ioi.phase5_analysis import _load_split_effects


def build_pack(out: Path) -> None:
    source = ROOT / "results/revision/phase05"
    task = load_ioi_effect_prediction_task(source, measurement_budget=160)
    out.mkdir(parents=True, exist_ok=True)
    (out / "submissions").mkdir(exist_ok=True)
    candidates = pd.read_csv(source / "design/candidate_masks.csv", dtype={"mask_bits": str})
    queries = candidates.rename(columns={"mask_id": "query_id"})[["query_id", "mask_bits", "pool_id", "n_heads"]]
    queries = queries.sort_values("query_id")
    queries.to_csv(out / "queries.csv", index=False)
    pd.DataFrame([{"measurement_id": r.measurement_id, "mask_bits": r.features.mask_bits,
                   "observed_effect": r.observed_effect} for r in task.measurements]).to_csv(
                       out / "calibration_measurements.csv", index=False)
    prompts = pd.read_csv(source / "design/prompts.csv").query("split == 'test'").sort_values("prompt_id")
    name_pairs = [tuple(sorted((r.io_name, r.s_name))) for r in prompts.itertuples()]
    pair_ids = {pair: f"pair_{i:03d}" for i, pair in enumerate(sorted(set(name_pairs)))}
    pd.DataFrame({"prompt_id": prompts.prompt_id.tolist(), "cluster_id": [pair_ids[p] for p in name_pairs]}).to_csv(
        out / "prompt_clusters.csv", index=False)
    effects, paths = _load_split_effects(source / "ioi_effects", "test")
    response = effects[effects.mask_id.isin(queries.query_id)].pivot(index="prompt_id", columns="mask_id", values="drop_from_clean")
    response = response.reindex(index=prompts.prompt_id, columns=queries.query_id)
    if response.shape != (256, 320) or not np.isfinite(response.to_numpy()).all():
        raise ValueError("expected all 81,920 held-out candidate responses")
    np.savez_compressed(out / "responses.npz", responses=response.to_numpy(),
                        mask_ids=np.asarray(response.columns, dtype=str), prompt_ids=np.asarray(response.index, dtype=str))
    stored_predictions = pd.read_csv(source / "ioi_fit/candidate_predictions.csv")
    entries = json.loads((ROOT / "leaderboards/effect/ioi-gpt2-small-finite-effects-b160/results.json").read_text())["rows"]
    references, submissions = {}, {}
    for key, model, observer in (("additive", "additive_head", "ioi-additive-ridge"),
                                 ("all_pairs", "count_plus_all_bin4", "ioi-count-plus-all-pairs-ridge")):
        entry = next(row for row in entries if row["observer_name"] == observer)
        selected = stored_predictions[(stored_predictions.model == model) & (stored_predictions.measurement_budget == 160)]
        if set(selected.mask_id) != set(queries.query_id) or len(selected) != 320:
            raise ValueError(f"incomplete stored reference: {model}")
        prediction_file = f"submissions/{key}.csv"
        card_file = f"submissions/{key}.json"
        write_effect_predictions(out / prediction_file, [FiniteEffectPrediction(r.mask_id, r.predicted_effect) for r in selected.itertuples()])
        observer_card = EffectObserverCard(
            observer_name=observer, observer_version="1.0.0", observer_family=entry["observer_family"],
            access_regime=entry["access_regime"], measurement_basis="canonical 13-head IOI ablation panel",
            fit_procedure="frozen Phase 5 ridge fit on 160 train-prompt calibration masks",
            implementation="observerbench.tasks.ioi.phase5_analysis.fit_phase5_observers",
            metadata={"prediction_estimand": "mean_finite_effect", "prediction_task_id": PREDICTION_TASK_ID},
        )
        write_json(out / card_file, observer_card.to_dict())
        references[key] = {"predictions": prediction_file, "display_name": entry["display_name"]}
        submissions[key] = {**entry, "predictions": prediction_file, "observer_card": card_file}
    atp_root = ROOT / "results/revision/published_baselines/ioi_attribution_patching_v1"
    for key, variant, observer in (("atp_raw", "raw", "ioi-attribution-patching-raw"),
                                    ("atp_calibrated", "scalar_calibrated", "ioi-attribution-patching-scalar-calibrated")):
        prediction_file, card_file = f"submissions/{key}.csv", f"submissions/{key}.json"
        shutil.copyfile(atp_root / variant / "b160/effect_predictions.csv", out / prediction_file)
        shutil.copyfile(atp_root / variant / "b160/observer_card.json", out / card_file)
        entry = next(row for row in entries if row["observer_name"] == observer)
        references[key] = {"predictions": prediction_file, "display_name": entry["display_name"]}
        submissions[key] = {**entry, "predictions": prediction_file, "observer_card": card_file}
    write_json(out / "submissions.json", submissions)
    card = {
        "schema": PACK_SCHEMA, "task_id": TASK_ID, "prediction_task_id": PREDICTION_TASK_ID,
        "model": "GPT-2-small (124M parameters)", "model_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "participation_class": "open_replay", "include_noop": True,
        "targets": list(TARGETS), "primary_target": 1.0,
        "selector": "absolute_mean_error_then_head_count_then_mask_id",
        "information_boundary": "160 train-prompt finite calibration effects; white-box gradients declared separately",
        "loss": "mean absolute distance from target, over held-out prompts and ten equally weighted action pools",
        "n_queries": len(queries), "n_prompts": len(prompts), "n_source_clusters": len(pair_ids),
        "pool_sizes": queries.groupby("pool_id").size().to_dict(), "references": references,
        "bootstrap_repeats": 2000, "bootstrap_seed": 20260904,
        "uncertainty": "95% paired percentile intervals resampling unordered-name-pair clusters and candidate pools independently; frozen fits and selections, not independent refits",
        "scope": "Secondary open replay of the original Phase 5 panel with exact no-op added. Not the later Phase 7 confirmation, not an independent replication, and not a sealed score. All three existing targets are reported; target 1 is the display target. No model inference or refitting occurs.",
        "source_files": {str(p.relative_to(ROOT)): file_sha256(p) for p in [source / "design/candidate_masks.csv", source / "design/prompts.csv", source / "ioi_fit/candidate_predictions.csv", *paths]},
        "files": {str(p.relative_to(out)): file_sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.suffix in {".csv", ".npz", ".json"} and p.name != "task_card.json"},
    }
    write_json(out / "task_card.json", card)
    print(f"Packed {len(queries)} masks x {len(prompts)} prompts ({len(pair_ids)} name-pair clusters): {out}")


def score_pack(pack: Path) -> None:
    submissions = json.loads((pack / "submissions.json").read_text())
    rows = []
    for key, submission in submissions.items():
        output = ROOT / "results/revision/ioi_decision_replay_v1" / key
        result = evaluate_ioi_decision_csv(TASK_ID, pack_root=pack,
            predictions_path=pack / submission["predictions"], observer_card_path=pack / submission["observer_card"], outdir=output)
        metrics = {"mae": result["prediction_metrics"]["mae"]}
        for target in result["by_target"]:
            if target["target"] == 1.0:
                metrics["action_loss"] = target["action_loss"]
                metrics["excess_loss_over_noop"] = target["excess_loss_over_noop"]
            else:
                metrics[f"action_loss_t{target['target']:.1f}"] = target["action_loss"]
        rows.append({"task_id": TASK_ID, "observer_name": submission["observer_name"],
            "display_name": submission["display_name"], "observer_family": submission["observer_family"],
            "access_regime": submission["access_regime"], "requires_white_box_access": submission["requires_white_box_access"],
            "measurement_budget": 160, "metrics": metrics, "result_status": "decision-tested open replay; secondary analysis",
            "source_result": str((output / "decision_evaluation.json").relative_to(ROOT)),
            "source_sha256": file_sha256(output / "decision_evaluation.json")})
        print((output / "scorecard.txt").read_text())
    leaderboard = ROOT / "leaderboards/decision/ioi-gpt2-small-action-selection-b160"
    leaderboard.mkdir(parents=True, exist_ok=True)
    directions = {key: "lower" for key in rows[0]["metrics"]}
    rows.append({"task_id": TASK_ID, "observer_name": "analytic_noop", "display_name": "Exact no-op",
        "observer_family": "Analytic action reference; no predictions or measurements required",
        "access_regime": "none", "measurement_budget": 0, "result_status": "reference; exact no-op",
        "metrics": {"action_loss": 1.0, "excess_loss_over_noop": 0.0,
                    **{f"action_loss_t{t:.1f}": t for t in TARGETS if t != 1.0}}})
    write_json(leaderboard / "results.json", {"schema_version": "observerbench.decision_leaderboard.v1",
        "task_id": TASK_ID, "primary_target": 1.0, "participation_class": "open_replay",
        "noop_loss_by_target": {str(t): t for t in TARGETS},
        "metric_direction": directions, "rows": rows})
    pd.DataFrame([{**{k: row[k] for k in ("observer_name", "access_regime", "result_status")}, **row["metrics"]} for row in rows]).to_csv(leaderboard / "results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--score", action="store_true", help="Also score all four frozen submissions and write the decision ladder")
    args = parser.parse_args()
    build_pack(args.pack_dir)
    if args.score:
        score_pack(args.pack_dir)
