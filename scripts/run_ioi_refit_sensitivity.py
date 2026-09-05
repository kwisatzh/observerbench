"""Refit the existing IOI observers on disjoint halves of the training population.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance, source_hashes
from observerbench.tasks.ioi.attribution_patching import load_template_head_means, measure_ioi_attribution_map
from observerbench.tasks.ioi.decision_submission import DEFAULT_PACK, NOOP, TARGETS, load_decision_pack, select_decision_actions
from observerbench.tasks.ioi.phase2_capacity import build_capacity_design
from observerbench.tasks.ioi.phase5_analysis import PINNED_GPT2_REVISION, _design_run, _load_split_effects
from observerbench.tasks.ioi.phase5_effects import load_locked_ioi_design
from observerbench.tasks.ioi.stage2d import ridge_fit

ROOT = Path(__file__).resolve().parents[1]


def training_halves(prompts, seeds):
    """Keep repeated appearances of the same unordered name pair together."""
    keys = np.asarray(["|".join(sorted((str(r.io_name), str(r.s_name)))) for r in prompts.itertuples()])
    groups = np.unique(keys)
    yield "full", np.arange(len(prompts)), len(groups)
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(groups)
        for half, selected in enumerate(np.array_split(order, 2)):
            yield f"{seed}-{half}", np.flatnonzero(np.isin(keys, selected)), len(selected)


def fit_predictions(train_masks, query_masks, effects, head_effects, ridge):
    """Use the existing additive/interaction designs and scalar AtP calibration."""
    predictions = {}
    for name, family in (("additive", "additive_head"), ("all_pairs", "count_plus_all_bin4")):
        x, _ = build_capacity_design(_design_run(train_masks), family)
        q, _ = build_capacity_design(_design_run(query_masks), family)
        predictions[name] = q @ ridge_fit(x, effects, ridge)
    x = np.asarray([[int(v) for v in bits] for bits in train_masks.mask_bits])
    q = np.asarray([[int(v) for v in bits] for bits in query_masks.mask_bits])
    raw = x @ head_effects
    gain = float(raw @ effects / (raw @ raw))
    predictions["atp_raw"] = q @ head_effects
    predictions["atp_calibrated"] = gain * predictions["atp_raw"]
    return predictions, gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "results/revision/phase05")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/revision/ioi_refit_sensitivity_20260904")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "configs/revision/ioi_refit_sensitivity_20260904.json"
    protocol = json.loads(protocol_path.read_text())
    write_json(args.outdir / "protocol.json", protocol)
    prompts, _, _ = load_locked_ioi_design(args.artifacts_root / "design")
    prompts = prompts[prompts.split == "train"].reset_index(drop=True)
    _, queries, calibration = load_decision_pack(DEFAULT_PACK)
    train_masks = pd.read_csv(args.artifacts_root / "design/calibration_masks.csv", dtype={"mask_bits": str, "mask_id": str})
    train_masks = train_masks.set_index("mask_id").loc[calibration.measurement_id].reset_index()
    query_masks = pd.read_csv(args.artifacts_root / "design/candidate_masks.csv", dtype={"mask_bits": str, "mask_id": str})
    query_masks = query_masks.set_index("mask_id").loc[queries.query_id].reset_index()
    training_path = args.outdir / "training_effects.npz"
    if training_path.exists():
        with np.load(training_path, allow_pickle=False) as stored:
            if (stored["prompt_ids"].astype(str).tolist() != prompts.prompt_id.tolist()
                    or stored["mask_ids"].astype(str).tolist() != train_masks.mask_id.tolist()):
                raise ValueError("training cache differs from the frozen design")
            matrix = stored["effects"].copy()
    else:
        train, shards = _load_split_effects(args.artifacts_root / "ioi_effects", "train")
        matrix = train.pivot(index="prompt_id", columns="mask_id", values="drop_from_clean").loc[prompts.prompt_id, train_masks.mask_id].to_numpy()
        np.savez_compressed(training_path,effects=matrix,prompt_ids=prompts.prompt_id.to_numpy(dtype=str),mask_ids=train_masks.mask_id.to_numpy(dtype=str))
        write_json(args.outdir/"training_input_provenance.json", {"sha256":file_sha256(training_path),
            "original_shards":{p.name:file_sha256(p) for p in shards},
            "scope":"Original training responses only. The held-out response matrix stays in the separate replay pack."})
    if not np.isfinite(matrix).all():
        raise ValueError("incomplete training response matrix")
    gradient_path = args.outdir / "per_prompt_attribution.npz"
    cost_path = args.outdir / "gradient_measurement_cost.json"
    if not gradient_path.exists():
        import torch
        from transformer_lens import HookedTransformer
        started = time.perf_counter()
        model = HookedTransformer.from_pretrained("gpt2-small", revision=PINNED_GPT2_REVISION, device=args.device)
        model.eval()
        loaded = time.perf_counter()
        means, templates = load_template_head_means(args.artifacts_root / "ioi_effects/template_head_means.npz")
        print(f"Model loaded; measuring gradients on {len(prompts)} training prompts", flush=True)
        atp = measure_ioi_attribution_map(model, args.artifacts_root / "design", means, templates,
            model_revision=PINNED_GPT2_REVISION, batch_size=args.batch_size, per_prompt_effects_path=gradient_path)
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()
        measured = time.perf_counter()
        write_json(cost_path, {"device": args.device, "batch_size": args.batch_size, "n_prompts": len(prompts),
            "model_load_seconds": loaded-started, "gradient_measurement_seconds": measured-loaded,
            "scope": "Actual pinned-model gradient measurement; excludes acquisition of cached template means and finite calibration effects. Not full observer construction cost.",
            "model_revision": PINNED_GPT2_REVISION, "runtime": runtime_provenance(ROOT)})
        write_json(args.outdir / "attribution_map.json", atp.to_dict())
        del model
    with np.load(gradient_path, allow_pickle=False) as stored:
        if stored["prompt_ids"].astype(str).tolist() != prompts.prompt_id.tolist() or str(stored["model_revision"]) != PINNED_GPT2_REVISION:
            raise ValueError("gradient cache does not match the frozen training prompts/model")
        gradients = stored["effects"].copy()
    if gradients.shape != (len(prompts), 13) or not np.isfinite(gradients).all():
        raise ValueError("incomplete per-prompt gradients")

    predictions, actions, fits = [], [], []
    for fit_id, indices, n_clusters in training_halves(prompts, protocol["seeds"]):
        started = time.perf_counter()
        fitted, gain = fit_predictions(train_masks, query_masks, matrix[indices].mean(axis=0), gradients[indices].mean(axis=0), protocol["ridge"])
        fit_seconds = time.perf_counter()-started
        for name, values in fitted.items():
            prediction_map = dict(zip(queries.query_id, map(float, values)))
            selected = select_decision_actions(queries, prediction_map)
            predictions.extend({"fit_id": fit_id, "observer": name, "query_id": key, "predicted_effect": value} for key, value in prediction_map.items())
            actions.extend({"fit_id": fit_id, "observer": name, "pool_id": pool, "target": target, **choice} for (pool, target), choice in selected.items())
        fits.append({"fit_id": fit_id, "n_prompts": len(indices), "n_name_pairs": n_clusters,
                     "scalar_gain": gain, "four_method_fit_seconds": fit_seconds,
                     "fit_and_selection_seconds": time.perf_counter()-started})
    pred = pd.DataFrame(predictions)
    chosen = pd.DataFrame(actions)
    pred.to_csv(args.outdir / "predictions.csv", index=False)
    chosen.to_csv(args.outdir / "selected_actions.csv", index=False)
    pd.DataFrame(fits).to_csv(args.outdir / "fits.csv", index=False)
    # All fits and choices are now fixed. Only this stage opens held-out responses.
    with np.load(DEFAULT_PACK / "responses.npz", allow_pickle=False) as stored:
        y = stored["responses"]
        responses = dict(zip(stored["mask_ids"].astype(str), y.T))
    means = {key: float(value.mean()) for key, value in responses.items()}
    responses[NOOP] = np.zeros(y.shape[0])
    chosen["action_loss"] = [float(np.abs(responses[r.selected_mask_id]-r.target).mean()) for r in chosen.itertuples()]
    pred["absolute_error"] = [abs(r.predicted_effect-means[r.query_id]) for r in pred.itertuples()]
    mae = pred.groupby(["fit_id", "observer"]).absolute_error.mean()
    scores = chosen.groupby(["fit_id", "observer", "target"]).action_loss.mean().reset_index()
    scores["mae"] = [float(mae.loc[(r.fit_id, r.observer)]) for r in scores.itertuples()]
    scores.to_csv(args.outdir / "refit_scores.csv", index=False)
    summary = []
    for (name, target), frame in scores[scores.fit_id != "full"].groupby(["observer", "target"]):
        summary.append({"observer": name, "target": target, "n_fits": len(frame),
                        "mean_mae": float(frame.mae.mean()), "mae_range_5_95": np.quantile(frame.mae,[.05,.95]).tolist(),
                        "mean_action_loss": float(frame.action_loss.mean()),
                        "action_loss_range_5_95": np.quantile(frame.action_loss,[.05,.95]).tolist(),
                        "fraction_beating_noop": float((frame.action_loss < target).mean())})
    comparisons = []
    for target in TARGETS:
        pivot = scores[(scores.fit_id != "full") & (scores.target == target)].pivot(index="fit_id",columns="observer",values=["mae","action_loss"])
        for a,b in (("all_pairs","additive"),("atp_calibrated","atp_raw")):
            dm = pivot["mae"][a]-pivot["mae"][b]
            dl = pivot["action_loss"][a]-pivot["action_loss"][b]
            comparisons.append({"target":target,"contrast":f"{a} minus {b}","mean_mae_change":float(dm.mean()),
                "mean_loss_change":float(dl.mean()),"loss_change_range_5_95":np.quantile(dl,[.05,.95]).tolist(),
                "fraction_lower_mae":float((dm<0).mean()),"fraction_lower_loss":float((dl<0).mean()),
                "fraction_lower_mae_but_higher_loss":float(((dm<0)&(dl>0)).mean())})
    result = {"schema":"observerbench.ioi_refit_sensitivity.results.v1", "scope":protocol["scope"],
              "summary":summary,"comparisons":comparisons,"full_fit":scores[scores.fit_id=="full"].to_dict("records"),
              "protocol_sha256":file_sha256(protocol_path),"gradient_sha256":file_sha256(gradient_path),
              "training_input_sha256":file_sha256(training_path),
              "source_hashes":source_hashes([Path(__file__),ROOT/"src/observerbench/tasks/ioi/attribution_patching.py"],ROOT),
              "runtime":runtime_provenance(ROOT)}
    write_json(args.outdir / "results.json",result)
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
