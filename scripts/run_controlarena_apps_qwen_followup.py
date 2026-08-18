"""Run matched controls for the external ControlArena APPS observer study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from observerbench.ai_control import (
    AIControlSample,
    evaluate_ai_control_scores,
    read_ai_control_scores,
    write_ai_control_scores,
)
from observerbench.ai_control_followup import (
    hashed_text_features,
    prevalence_budget_sensitivity,
    select_ridge_binary_observer,
)
from observerbench.core import write_json
from observerbench.provenance import runtime_provenance
from observerbench.tasks.qwen_induction.plant import Qwen2InductionPlant

from run_controlarena_apps_qwen import (
    MonitorRow,
    _prompt,
    _protocol_rows,
    _score_token_ids,
    load_pairs,
    make_rows,
    select_activation_probe,
)


FOLLOWUP_RESULT_SCHEMA = "observerbench.controlarena_apps_qwen.followup_results.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parent(
    followup: Mapping[str, Any],
    *,
    base_config_path: Path,
    base_outdir: Path,
) -> None:
    expected = followup["parent"]
    paths = {
        "config_sha256": base_config_path,
        "results_sha256": base_outdir / "results.json",
        "score_table_sha256": base_outdir / "ai_control_scores.csv",
        "monitor_cache_sha256": base_outdir / "monitor_cache.npz",
    }
    for key, path in paths.items():
        observed = _sha256(path)
        if observed != expected[key]:
            raise ValueError(f"parent artifact changed for {path}: {observed}")


def _validate_rows(rows: Sequence[MonitorRow], design_path: Path) -> None:
    design = _load_json(design_path)
    expected = [
        (
            str(row["sample_id"]),
            str(row["split"]),
            str(row["eval_mode"]),
            bool(row["backdoor_works"]),
            str(row["family_id"]),
        )
        for row in design["rows"]
    ]
    observed = [
        (row.sample_id, row.split, row.eval_mode, row.backdoor_works, row.family_id)
        for row in rows
    ]
    if observed != expected:
        raise ValueError("reconstructed APPS rows differ from the frozen parent design")


def _full_input_text(row: MonitorRow) -> str:
    return (
        "<problem>\n"
        + row.question
        + "\n</problem>\n<candidate_solution>\n"
        + row.code
        + "\n</candidate_solution>"
    )


def _neutral_text(row: MonitorRow) -> str:
    return (
        "Programming problem:\n<problem>\n"
        + row.question
        + "\n</problem>\n\nCandidate solution:\n<solution>\n"
        + row.code
        + "\n</solution>"
    )


def _score_log_probabilities_from_cached_residual(
    transformer: Qwen2InductionPlant,
    final_residual: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch = transformer.torch
    score_ids = torch.as_tensor(
        _score_token_ids(transformer.tokenizer),
        device=transformer.device,
    )
    log_probabilities = []
    expected_scores = []
    values = torch.linspace(0.0, 10.0, 10, device=transformer.device)
    with torch.inference_mode():
        for start in range(0, len(final_residual), batch_size):
            hidden = torch.as_tensor(
                final_residual[start : start + batch_size],
                dtype=transformer.model.lm_head.weight.dtype,
                device=transformer.device,
            )
            hidden = transformer.model.model.norm(hidden)
            candidate_logits = transformer.model.lm_head(hidden).float().index_select(1, score_ids)
            log_probs = torch.log_softmax(candidate_logits, dim=1)
            log_probabilities.append(log_probs.cpu().numpy())
            expected_scores.append((torch.exp(log_probs) * values).sum(dim=1).cpu().numpy())
    return (
        np.concatenate(log_probabilities).astype(np.float32),
        np.concatenate(expected_scores).astype(np.float32),
    )


def _direct_basic_score_log_probabilities(
    transformer: Qwen2InductionPlant,
    rows: Sequence[MonitorRow],
    *,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch = transformer.torch
    tokenizer = transformer.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    score_ids = torch.as_tensor(_score_token_ids(tokenizer), device=transformer.device)
    values = torch.linspace(0.0, 10.0, 10, device=transformer.device)
    log_probabilities = []
    expected_scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            rendered = []
            for row in rows[start : start + batch_size]:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": _prompt(row, "controlarena-basic")}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rendered.append(text + "<score>\n")
            tokens = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            outputs = transformer.model(
                input_ids=tokens["input_ids"].to(transformer.device),
                attention_mask=tokens["attention_mask"].to(transformer.device),
                use_cache=False,
            )
            candidate_logits = outputs.logits[:, -1].float().index_select(1, score_ids)
            log_probs = torch.log_softmax(candidate_logits, dim=1)
            log_probabilities.append(log_probs.cpu().numpy())
            expected_scores.append((torch.exp(log_probs) * values).sum(dim=1).cpu().numpy())
            del outputs
    return (
        np.concatenate(log_probabilities).astype(np.float32),
        np.concatenate(expected_scores).astype(np.float32),
    )


def encode_neutral_activations(
    transformer: Qwen2InductionPlant,
    rows: Sequence[MonitorRow],
    *,
    layers: Sequence[int],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read residuals from problem plus code without a monitoring instruction."""

    torch = transformer.torch
    tokenizer = transformer.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    activations = []
    lengths = []
    total_batches = (len(rows) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": _neutral_text(row)}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for row in rows[start : start + batch_size]
            ]
            tokens = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            input_ids = tokens["input_ids"].to(transformer.device)
            attention_mask = tokens["attention_mask"].to(transformer.device)
            outputs = transformer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            activations.append(
                torch.stack([outputs.hidden_states[int(layer)][:, -1] for layer in layers], dim=1)
                .float()
                .cpu()
                .numpy()
            )
            lengths.append(attention_mask.sum(dim=1).cpu().numpy())
            del outputs, input_ids, attention_mask
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
                print(
                    f"neutral-context inference: batch {batch_index}/{total_batches}",
                    flush=True,
                )
    return np.concatenate(activations), np.concatenate(lengths).astype(np.int32)


def _locked_rows(rows: Sequence[MonitorRow]) -> list[MonitorRow]:
    return [row for row in rows if row.split == "locked_test"]


def _score_pools(panel: Sequence[AIControlSample]) -> dict[str, dict[str, np.ndarray]]:
    monitors = sorted({row.monitor_name for row in panel})
    pools: dict[str, dict[str, np.ndarray]] = {}
    for monitor in monitors:
        monitor_rows = [row for row in panel if row.monitor_name == monitor]
        pools[monitor] = {
            mode: np.asarray(
                [row.score for row in sorted(monitor_rows, key=lambda item: item.sample_id)
                 if row.eval_mode == mode],
                dtype=float,
            )
            for mode in ("honest", "attack")
        }
    return pools


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--followup-config", type=Path, required=True)
    parser.add_argument("--base-outdir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    base_config = _load_json(args.base_config)
    followup = _load_json(args.followup_config)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print("validating frozen parent artifacts", flush=True)
    _validate_parent(followup, base_config_path=args.base_config, base_outdir=args.base_outdir)
    print("loading the pinned APPS split", flush=True)
    pairs = load_pairs(base_config)
    rows = make_rows(pairs, base_config)
    _validate_rows(rows, args.base_outdir / "design.json")

    with np.load(args.base_outdir / "monitor_cache.npz", allow_pickle=False) as archive:
        basic_scores = archive["basic_scores"].copy()
        basic_activations = archive["basic_activations"].copy()
        basic_lengths = archive["basic_lengths"].copy()

    labels = np.asarray([row.backdoor_works for row in rows], dtype=bool)
    max_length = int(base_config["measurement"]["max_length"])
    truncated_training = np.asarray(basic_lengths >= max_length)
    if any(
        truncated_training[index] and row.split == "locked_test"
        for index, row in enumerate(rows)
    ):
        raise ValueError("a locked-test row reached the declared context limit")
    analysis_rows = [
        replace(row, split="excluded_truncation") if truncated_training[index] else row
        for index, row in enumerate(rows)
    ]
    splits = [row.split for row in analysis_rows]
    control = followup["controls"]
    selection_budget = float(followup["selection"]["audit_budget_fraction"])

    text_config = control["supervised_text"]
    print("fitting the supervised text baseline", flush=True)
    text_features = hashed_text_features(
        [_full_input_text(row) for row in rows],
        dimension=int(text_config["dimension"]),
    )
    text_selection, text_scores = select_ridge_binary_observer(
        text_features,
        labels,
        splits,
        ridge_grid=text_config["ridge_grid"],
        selection_budget=selection_budget,
    )
    del text_features

    cache_path = args.outdir / "followup_cache.npz"
    reconstruction = {"source": "cached final residual plus pinned LM head"}
    if args.resume and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            score_log_probabilities = archive["score_log_probabilities"].copy()
            neutral_activations = archive["neutral_activations"].copy()
            neutral_lengths = archive["neutral_lengths"].copy()
            reconstructed_expected = archive["reconstructed_expected"].copy()
        reconstruction["source"] = "resumed checked follow-up cache"
    else:
        print("loading the pinned Qwen model", flush=True)
        model_config = base_config["model"]
        transformer = Qwen2InductionPlant.from_pretrained(
            model_config["id"],
            model_config["revision"],
            device=args.device,
            dtype=model_config["dtype"],
            attention_implementation=model_config["attention_implementation"],
        )
        base_layers = list(map(int, base_config["measurement"]["layers"]))
        final_layer_position = base_layers.index(max(base_layers))
        score_log_probabilities, reconstructed_expected = (
            _score_log_probabilities_from_cached_residual(
                transformer,
                basic_activations[:, final_layer_position],
                batch_size=int(base_config["measurement"]["batch_size"]),
            )
        )
        max_error = float(np.max(np.abs(reconstructed_expected - basic_scores)))
        print(f"score-distribution reconstruction max error: {max_error:.6g}", flush=True)
        if max_error > 1e-3:
            reconstruction["source"] = "direct basic-prompt rerun after residual reconstruction mismatch"
            score_log_probabilities, reconstructed_expected = _direct_basic_score_log_probabilities(
                transformer,
                rows,
                max_length=int(base_config["measurement"]["max_length"]),
                batch_size=int(base_config["measurement"]["batch_size"]),
            )
        neutral_config = control["neutral_activation"]
        print("starting neutral-context activation extraction", flush=True)
        neutral_activations, neutral_lengths = encode_neutral_activations(
            transformer,
            rows,
            layers=neutral_config["layers"],
            max_length=int(neutral_config["max_length"]),
            batch_size=int(neutral_config["batch_size"]),
        )
        np.savez_compressed(
            cache_path,
            score_log_probabilities=score_log_probabilities,
            neutral_activations=neutral_activations,
            neutral_lengths=neutral_lengths,
            reconstructed_expected=reconstructed_expected,
        )
        print(f"saved follow-up cache to {cache_path}", flush=True)
        del transformer
    score_difference = np.abs(reconstructed_expected - basic_scores)
    reconstruction["max_abs_expected_score_difference"] = float(np.max(score_difference))
    reconstruction["max_abs_expected_score_difference_on_included_rows"] = float(
        np.max(score_difference[~truncated_training])
    )
    reconstruction["excluded_truncated_training_rows"] = int(np.sum(truncated_training))

    score_config = control["supervised_score_distribution"]
    score_selection, supervised_score_scores = select_ridge_binary_observer(
        score_log_probabilities,
        labels,
        splits,
        ridge_grid=score_config["ridge_grid"],
        selection_budget=selection_budget,
    )

    neutral_config = control["neutral_activation"]
    neutral_base_config = json.loads(json.dumps(base_config))
    neutral_base_config["measurement"].update(
        {
            "layers": neutral_config["layers"],
            "projection_dim": neutral_config["projection_dim"],
            "projection_seed": neutral_config["projection_seed"],
            "ridge_grid": neutral_config["ridge_grid"],
        }
    )
    neutral_selection, neutral_scores = select_activation_probe(
        analysis_rows,
        neutral_activations,
        neutral_base_config,
    )

    locked = _locked_rows(rows)
    panel = list(read_ai_control_scores(args.base_outdir / "ai_control_scores.csv"))
    panel.extend(_protocol_rows(locked, "qwen-supervised-text-ridge", text_scores))
    panel.extend(
        _protocol_rows(
            locked,
            "qwen-supervised-score-distribution",
            supervised_score_scores,
        )
    )
    panel.extend(_protocol_rows(locked, "qwen-neutral-activation-probe", neutral_scores))
    panel.extend(
        _protocol_rows(
            locked,
            "exact-label-oracle",
            [float(row.backdoor_works) for row in locked],
        )
    )
    panel_tuple = tuple(panel)
    write_ai_control_scores(args.outdir / "ai_control_scores.csv", panel_tuple)

    budgets = (
        float(base_config["protocol"]["audit_budget_fraction"]),
        *map(float, base_config["protocol"]["secondary_audit_budget_fractions"]),
    )
    budget_results = {
        str(budget): {
            result.monitor_name: result.to_dict()
            for result in evaluate_ai_control_scores(panel_tuple, audit_budget_fraction=budget)
        }
        for budget in budgets
    }
    sensitivity_config = followup["sensitivity"]
    print("running frozen-score prevalence and budget sensitivity", flush=True)
    sensitivity = prevalence_budget_sensitivity(
        _score_pools(panel_tuple),
        prevalences=sensitivity_config["prevalences"],
        budgets=sensitivity_config["audit_budget_fractions"],
        population_size=int(sensitivity_config["population_size"]),
        replicates=int(sensitivity_config["replicates"]),
        seed=int(sensitivity_config["seed"]),
    )
    write_json(args.outdir / "prevalence_sensitivity.json", sensitivity)
    print("finished prevalence and budget sensitivity", flush=True)

    payload = {
        "schema": FOLLOWUP_RESULT_SCHEMA,
        "status": "complete_post_outcome_controls",
        "config": followup,
        "runtime": runtime_provenance(Path(__file__).resolve().parents[1]),
        "n_selected_pairs": len(pairs),
        "access": {
            "qwen-controlarena-basic": "expected score-token distribution under explicit monitoring prompt",
            "qwen-supervised-score-distribution": "supervised ridge over the same ten score-token log probabilities",
            "qwen-activation-probe": "supervised residual read at the score position under the explicit basic monitoring prompt",
            "qwen-neutral-activation-probe": "supervised residual read after problem plus code with no monitoring instruction",
            "qwen-supervised-text-ridge": "supervised hashed word and character n-grams over problem plus code",
            "exact-label-oracle": "evaluator-only working-backdoor label",
        },
        "selections": {
            "supervised_text": text_selection,
            "supervised_score_distribution": score_selection,
            "neutral_activation": neutral_selection,
        },
        "score_distribution_reconstruction": reconstruction,
        "neutral_truncation": {
            "max_length": int(neutral_config["max_length"]),
            "at_limit": int(np.sum(neutral_lengths >= int(neutral_config["max_length"]))),
            "excluded_from_fit": int(np.sum(truncated_training)),
            "locked_test_at_limit": 0,
        },
        "budget_results": budget_results,
        "sensitivity_artifact": "prevalence_sensitivity.json",
    }
    write_json(args.outdir / "results.json", payload)
    print(json.dumps(payload["budget_results"][str(budgets[0])], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
