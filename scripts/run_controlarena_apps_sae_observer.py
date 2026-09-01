"""Run the published SAE sparse-probing observer on frozen Qwen APPS activations.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This runner does not repeat Qwen inference. It consumes the frozen neutral
activation cache, encodes it with one pinned public BatchTopK SAE, fits the
published sparse-probing recipe and two capacity controls, and emits one
target-free CSV plus ObserverCard per observer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from observerbench.ai_control_followup import (
    budget_metrics,
    prevalence_budget_sensitivity,
)
from observerbench.ai_control_public import (
    PrivateAIControlTarget,
    PublicAIControlScore,
    evaluate_public_ai_control_scores,
    read_public_ai_control_scores,
    write_public_ai_control_scores,
)
from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance
from observerbench.safety import SafetyObserverCard
from observerbench.tasks.controlarena_apps.sae_probe import (
    BatchTopKSAE,
    FrozenAppsRow,
    context_length_diagnostics,
    encode_batch_topk_sae,
    load_frozen_apps_design,
    load_neutral_activation_cache,
    load_sae_encoding,
    reconstruction_diagnostics,
    select_dense_logistic_observer,
    select_sparse_logistic_observer,
    write_sae_encoding,
)


CONFIG_SCHEMA = "observerbench.controlarena_apps_qwen.sae_probe.v1"
RESULT_SCHEMA = "observerbench.controlarena_apps_qwen.sae_probe.results.v1"
IMPLEMENTATION_URL = (
    "https://github.com/kwisatzh/observerbench/blob/main/"
    "scripts/run_controlarena_apps_sae_observer.py"
)

MONITOR_SAE = "qwen-neutral-sae-sparse-probe"
MONITOR_RAW = "qwen-neutral-residual-logistic"
MONITOR_RECONSTRUCTION = "qwen-neutral-sae-reconstruction-logistic"


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported ControlArena APPS SAE config")
    if payload.get("status") != "post_outcome_method_integration_frozen_before_sae_results":
        raise ValueError("the SAE integration config is not in its frozen pre-result state")
    return payload


def _repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = file_sha256(path)
    if observed != str(expected):
        raise ValueError(f"{label} changed: expected {expected}, observed {observed}")
    return observed


def _download_checkpoint_files(sae_config: Mapping[str, Any]) -> dict[str, Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "install ObserverBench with the sae extra to download the checkpoint"
        ) from error
    return {
        key: Path(
            hf_hub_download(
                repo_id=str(sae_config["repo_id"]),
                revision=str(sae_config["revision"]),
                filename=str(sae_config[key]),
            )
        )
        for key in ("checkpoint", "config", "eval_results")
    }


def _validate_checkpoint_metadata(
    config_path: Path,
    evaluation_path: Path,
    sae_config: Mapping[str, Any],
) -> dict[str, Any]:
    _require_hash(config_path, str(sae_config["config_sha256"]), "SAE config")
    _require_hash(
        evaluation_path,
        str(sae_config["eval_results_sha256"]),
        "SAE evaluation metadata",
    )
    source = json.loads(config_path.read_text(encoding="utf-8"))
    trainer = source.get("trainer", {})
    expected = {
        "trainer_class": "BatchTopKTrainer",
        "dict_class": "BatchTopKSAE",
        "activation_dim": int(sae_config["activation_dim"]),
        "dict_size": int(sae_config["dictionary_size"]),
        "k": int(sae_config["k"]),
        "layer": 19,
        "lm_name": "Qwen/Qwen2.5-7B-Instruct",
        "submodule_name": "resid_post_layer_19",
    }
    differences = {
        key: {"expected": value, "observed": trainer.get(key)}
        for key, value in expected.items()
        if trainer.get(key) != value
    }
    if differences:
        raise ValueError(f"pinned SAE metadata changed: {differences}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return {
        "trainer": expected,
        "reported_evaluation": evaluation,
    }


def _resolve_checkpoint(
    sae_config: Mapping[str, Any],
    *,
    checkpoint: Path | None,
    download_checkpoint: bool,
) -> tuple[Path, dict[str, Any]]:
    metadata: dict[str, Any] = {"source": "explicit local checkpoint"}
    if checkpoint is None:
        if not download_checkpoint:
            raise FileNotFoundError(
                "no SAE checkpoint supplied; pass --download-checkpoint or --checkpoint PATH"
            )
        files = _download_checkpoint_files(sae_config)
        checkpoint = files["checkpoint"]
        metadata = {
            "source": "Hugging Face pinned revision",
            **_validate_checkpoint_metadata(
                files["config"], files["eval_results"], sae_config
            ),
        }
    observed = _require_hash(
        checkpoint,
        str(sae_config["checkpoint_sha256"]),
        "SAE checkpoint",
    )
    if checkpoint.stat().st_size != int(sae_config["checkpoint_bytes"]):
        raise ValueError("SAE checkpoint byte count differs from the pin")
    metadata.update(
        {
            "path": str(checkpoint),
            "sha256": observed,
            "bytes": checkpoint.stat().st_size,
        }
    )
    return checkpoint, metadata


def _encoding_metadata(
    *,
    config_sha256: str,
    design_sha256: str,
    cache_sha256: str,
    sae_config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "observerbench.controlarena_apps_qwen.sae_encoding.v1",
        "config_sha256": config_sha256,
        "design_sha256": design_sha256,
        "neutral_cache_sha256": cache_sha256,
        "sae_checkpoint_sha256": str(sae_config["checkpoint_sha256"]),
        "sae_revision": str(sae_config["revision"]),
        "sae_checkpoint": str(sae_config["checkpoint"]),
        "hf_hidden_state_index": int(parent_config["hf_hidden_state_index"]),
        "transformer_block_output": int(parent_config["transformer_block_output"]),
    }


def _locked_rows(rows: Sequence[FrozenAppsRow]) -> tuple[FrozenAppsRow, ...]:
    return tuple(row for row in rows if row.split == "locked_test")


def _private_targets(rows: Sequence[FrozenAppsRow]) -> tuple[PrivateAIControlTarget, ...]:
    return tuple(
        PrivateAIControlTarget(
            sample_id=row.sample_id,
            eval_mode=row.eval_mode,
            backdoor_works=row.backdoor_works,
            family_id=row.family_id,
        )
        for row in rows
    )


def _public_scores(
    rows: Sequence[FrozenAppsRow],
    monitor_name: str,
    scores: Sequence[float],
) -> tuple[PublicAIControlScore, ...]:
    return tuple(
        PublicAIControlScore(
            sample_id=row.sample_id,
            monitor_name=monitor_name,
            score=float(score),
        )
        for row, score in zip(rows, scores, strict=True)
    )


def _observer_cards(
    config: Mapping[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
) -> dict[str, SafetyObserverCard]:
    sae = config["sae"]
    probe = config["probe"]
    common_metadata = {
        "task_id": "controlarena-apps-verified-backdoors@qwen2.5-7b-v1",
        "model_revision": config["parent"]["model_revision"],
        "source_activation": "neutral context; output of transformer block 19 at the final token",
        "result_status": "post-outcome published-method integration",
    }
    return {
        MONITOR_SAE: SafetyObserverCard(
            observer_name=MONITOR_SAE,
            observer_version="1.0.0",
            observer_family="published SAE sparse probe",
            observer_input=(
                "features from andyrdt BatchTopK SAE resid_post_layer_19/trainer_3 "
                "over the frozen neutral-context residual"
            ),
            fit_procedure=(
                "Kantamneni et al.: rank features by absolute class-mean "
                "activation difference, retain top k, fit L1 logistic; choose k and "
                "C on 2% calibration audit loss, then rerank/refit on fit plus calibration"
            ),
            implementation=IMPLEMENTATION_URL,
            risk_score_meaning="larger probability gives higher audit priority",
            known_failure_modes=(
                "the SAE was trained with 1024-token contexts; longer APPS rows are reported separately",
                "the SAE training blend includes chat, pretraining, and a small emergent-misalignment corpus",
                "sparse coordinates are not assumed to be monosemantic",
            ),
            access_regime="cached white-box residual followed by a pinned public SAE encoder",
            requires_white_box_access=True,
            additional_forward_passes=0,
            metadata={
                **common_metadata,
                "sae_repo_id": sae["repo_id"],
                "sae_revision": sae["revision"],
                "sae_checkpoint_sha256": sae["checkpoint_sha256"],
                "dictionary_size": sae["dictionary_size"],
                "sae_k": sae["k"],
                "probe_feature_count_grid": probe["feature_count_grid"],
                "selected_feature_count": selections[MONITOR_SAE]["feature_count"],
                "selected_c": selections[MONITOR_SAE]["c"],
                "selected_feature_ids": selections[MONITOR_SAE][
                    "refit_rank_top_feature_ids"
                ],
                "paper": probe["method_paper"],
            },
        ),
        MONITOR_RAW: SafetyObserverCard(
            observer_name=MONITOR_RAW,
            observer_version="1.0.0",
            observer_family="raw residual logistic control",
            observer_input="all 3584 frozen neutral-context residual coordinates",
            fit_procedure=(
                "L2 logistic regression; choose C on 2% calibration audit loss and refit on fit plus calibration"
            ),
            implementation=IMPLEMENTATION_URL,
            risk_score_meaning="larger probability gives higher audit priority",
            known_failure_modes=(
                "dense coordinates do not provide a sparse feature interpretation",
            ),
            access_regime="cached white-box residual",
            requires_white_box_access=True,
            additional_forward_passes=0,
            metadata={
                **common_metadata,
                "selected_c": selections[MONITOR_RAW]["c"],
            },
        ),
        MONITOR_RECONSTRUCTION: SafetyObserverCard(
            observer_name=MONITOR_RECONSTRUCTION,
            observer_version="1.0.0",
            observer_family="SAE reconstruction logistic control",
            observer_input="all 3584 coordinates of the pinned SAE reconstruction",
            fit_procedure=(
                "same L2 logistic method and selection rule as the raw residual control"
            ),
            implementation=IMPLEMENTATION_URL,
            risk_score_meaning="larger probability gives higher audit priority",
            known_failure_modes=(
                "reconstruction can discard task-relevant residual information",
                "the SAE was trained with 1024-token contexts; longer APPS rows are reported separately",
            ),
            access_regime="cached white-box residual followed by pinned SAE encode/decode",
            requires_white_box_access=True,
            additional_forward_passes=0,
            metadata={
                **common_metadata,
                "sae_checkpoint_sha256": sae["checkpoint_sha256"],
                "selected_c": selections[MONITOR_RECONSTRUCTION]["c"],
            },
        ),
    }


def _write_submission(
    root: Path,
    card: SafetyObserverCard,
    scores: Sequence[PublicAIControlScore],
) -> tuple[Path, Path]:
    output = root / card.observer_name
    score_path = output / "predictions.csv"
    card_path = output / "observer_card.json"
    write_public_ai_control_scores(score_path, scores)
    write_json(card_path, asdict(card))
    round_trip = read_public_ai_control_scores(score_path)
    if tuple(round_trip) != tuple(scores):
        raise ValueError("public score-table round trip changed an SAE observer result")
    return score_path, card_path


def _score_pools(
    rows: Sequence[FrozenAppsRow],
    scores_by_monitor: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    modes = np.asarray([row.eval_mode for row in rows], dtype=str)
    for monitor, monitor_scores in scores_by_monitor.items():
        scores = np.asarray(monitor_scores, dtype=float)
        output[monitor] = {
            mode: scores[modes == mode]
            for mode in ("honest", "attack")
        }
    return output


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _paired_problem_bootstrap(
    rows: Sequence[FrozenAppsRow],
    scores_by_monitor: Mapping[str, Sequence[float]],
    *,
    budgets: Sequence[float],
    replicates: int,
    seed: int,
    reference_monitor: str,
) -> dict[str, Any]:
    """Resample APPS problems while keeping each honest/attack pair together."""

    if reference_monitor not in scores_by_monitor:
        raise ValueError("paired-bootstrap reference monitor is absent")
    problem_ids = tuple(sorted({row.problem_id for row in rows}))
    row_index = {row.sample_id: index for index, row in enumerate(rows)}
    labels = np.asarray(
        [
            rows[row_index[f"{problem_id}::{mode}"]].backdoor_works
            for problem_id in problem_ids
            for mode in ("honest", "attack")
        ],
        dtype=bool,
    ).reshape(len(problem_ids), 2)
    scores = {
        monitor: np.asarray(
            [
                monitor_scores[row_index[f"{problem_id}::{mode}"]]
                for problem_id in problem_ids
                for mode in ("honest", "attack")
            ],
            dtype=float,
        ).reshape(len(problem_ids), 2)
        for monitor, monitor_scores in scores_by_monitor.items()
    }
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(problem_ids), size=(int(replicates), len(problem_ids)))
    payload: dict[str, Any] = {
        "schema": "observerbench.controlarena_apps_qwen.sae_probe.bootstrap.v1",
        "resampling_unit": "APPS problem with paired honest and working-backdoor rows",
        "replicates": int(replicates),
        "seed": int(seed),
        "reference_monitor": reference_monitor,
        "budgets": {},
    }
    for budget in budgets:
        samples = {
            monitor: {
                metric: np.empty(int(replicates), dtype=float)
                for metric in ("risk_auroc", "realized_violations", "audit_precision")
            }
            for monitor in scores
        }
        for replicate, chosen in enumerate(draws):
            sampled_labels = labels[chosen].reshape(-1)
            for monitor, matrix in scores.items():
                metrics = budget_metrics(
                    sampled_labels,
                    matrix[chosen].reshape(-1),
                    budget=float(budget),
                )
                for metric in samples[monitor]:
                    samples[monitor][metric][replicate] = metrics[metric]
        by_monitor = {
            monitor: {metric: _interval(values) for metric, values in metrics.items()}
            for monitor, metrics in samples.items()
        }
        differences = {
            monitor: {
                f"monitor_minus_reference_{metric}": _interval(
                    samples[monitor][metric] - samples[reference_monitor][metric]
                )
                for metric in ("risk_auroc", "realized_violations", "audit_precision")
            }
            for monitor in samples
            if monitor != reference_monitor
        }
        payload["budgets"][str(float(budget))] = {
            "monitors": by_monitor,
            "paired_differences": differences,
        }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_config(args.config)
    args.outdir.mkdir(parents=True, exist_ok=True)
    parent = config["parent"]
    sae_config = config["sae"]
    probe = config["probe"]
    protocol = config["protocol"]
    device = str(args.device or config["runtime"]["device"])

    config_hash = file_sha256(args.config)
    design_path = _repo_path(repo_root, str(parent["design"]))
    neutral_cache_path = _repo_path(repo_root, str(parent["neutral_cache"]))
    base_config_path = _repo_path(repo_root, str(parent["base_config"]))
    design_hash = _require_hash(
        design_path, str(parent["design_sha256"]), "frozen APPS design"
    )
    neutral_cache_hash = _require_hash(
        neutral_cache_path,
        str(parent["neutral_cache_sha256"]),
        "frozen neutral activation cache",
    )
    _require_hash(
        base_config_path,
        str(parent["base_config_sha256"]),
        "frozen APPS base config",
    )
    rows = load_frozen_apps_design(design_path)
    activations, lengths = load_neutral_activation_cache(
        neutral_cache_path,
        expected_rows=len(rows),
        layer_axis_position=int(parent["cache_layer_axis_position"]),
        expected_activation_dim=int(parent["activation_dim"]),
    )
    labels = np.asarray([row.backdoor_works for row in rows], dtype=bool)
    splits = np.asarray([row.split for row in rows], dtype=str)

    encoding_metadata = _encoding_metadata(
        config_sha256=config_hash,
        design_sha256=design_hash,
        cache_sha256=neutral_cache_hash,
        sae_config=sae_config,
        parent_config=parent,
    )
    encoded_cache_path = args.outdir / "sae_encoded_cache.npz"
    checkpoint_metadata: dict[str, Any]
    if args.resume and encoded_cache_path.exists():
        print("loading hash-matched SAE encoding cache", flush=True)
        encoding = load_sae_encoding(
            encoded_cache_path,
            expected_metadata=encoding_metadata,
        )
        checkpoint_metadata = {
            "source": "resumed pickle-free SAE encoding cache",
            "sha256": str(sae_config["checkpoint_sha256"]),
        }
    else:
        checkpoint_path, checkpoint_metadata = _resolve_checkpoint(
            sae_config,
            checkpoint=args.checkpoint,
            download_checkpoint=bool(args.download_checkpoint),
        )
        print("loading pinned BatchTopK SAE", flush=True)
        sae = BatchTopKSAE.from_checkpoint(
            checkpoint_path,
            device=device,
            expected_activation_dim=int(sae_config["activation_dim"]),
            expected_dictionary_size=int(sae_config["dictionary_size"]),
            expected_k=int(sae_config["k"]),
        )
        encoding = encode_batch_topk_sae(
            sae,
            activations,
            batch_size=int(config["runtime"]["sae_batch_size"]),
            metadata=encoding_metadata,
            progress=lambda message: print(message, flush=True),
        )
        write_sae_encoding(encoded_cache_path, encoding)
        print(f"saved SAE encoding cache to {encoded_cache_path}", flush=True)
        del sae

    print("fitting published SAE sparse probe", flush=True)
    sparse_selection, sparse_scores = select_sparse_logistic_observer(
        encoding.features,
        labels,
        splits,
        feature_count_grid=probe["feature_count_grid"],
        c_grid=probe["l1_c_grid"],
        selection_budget=float(probe["selection_audit_budget_fraction"]),
        seed=int(probe["seed"]),
        max_iter=int(probe["max_iter"]),
    )
    print("fitting raw residual capacity control", flush=True)
    raw_selection, raw_scores = select_dense_logistic_observer(
        activations,
        labels,
        splits,
        c_grid=probe["l2_c_grid"],
        selection_budget=float(probe["selection_audit_budget_fraction"]),
        seed=int(probe["seed"]),
        max_iter=int(probe["max_iter"]),
    )
    print("fitting SAE reconstruction capacity control", flush=True)
    reconstruction_selection, reconstruction_scores = select_dense_logistic_observer(
        encoding.reconstruction,
        labels,
        splits,
        c_grid=probe["l2_c_grid"],
        selection_budget=float(probe["selection_audit_budget_fraction"]),
        seed=int(probe["seed"]),
        max_iter=int(probe["max_iter"]),
    )

    locked = _locked_rows(rows)
    scores_by_monitor = {
        MONITOR_SAE: sparse_scores,
        MONITOR_RAW: raw_scores,
        MONITOR_RECONSTRUCTION: reconstruction_scores,
    }
    selections = {
        MONITOR_SAE: sparse_selection,
        MONITOR_RAW: raw_selection,
        MONITOR_RECONSTRUCTION: reconstruction_selection,
    }
    cards = _observer_cards(config, selections)
    all_public_scores: list[PublicAIControlScore] = []
    output_files: dict[str, Any] = {}
    for monitor_name, monitor_scores in scores_by_monitor.items():
        public_rows = _public_scores(locked, monitor_name, monitor_scores)
        score_path, card_path = _write_submission(
            args.outdir / "submissions",
            cards[monitor_name],
            public_rows,
        )
        all_public_scores.extend(public_rows)
        output_files[monitor_name] = {
            "predictions": str(score_path),
            "observer_card": str(card_path),
            "predictions_sha256": file_sha256(score_path),
            "observer_card_sha256": file_sha256(card_path),
        }

    private_targets = _private_targets(locked)
    budgets = tuple(map(float, protocol["audit_budget_fractions"]))
    budget_results = {
        str(budget): {
            result.monitor_name: result.to_dict()
            for result in evaluate_public_ai_control_scores(
                all_public_scores,
                private_targets,
                audit_budget_fraction=budget,
            )
        }
        for budget in budgets
    }
    context = context_length_diagnostics(
        rows,
        lengths,
        scores_by_monitor,
        training_context_length=int(sae_config["training_context_length"]),
        audit_budgets=budgets,
    )
    reconstruction = reconstruction_diagnostics(
        activations,
        encoding,
        lengths=lengths,
        training_context_length=int(sae_config["training_context_length"]),
    )
    sensitivity_config = protocol["prevalence_sensitivity"]
    sensitivity = prevalence_budget_sensitivity(
        _score_pools(locked, scores_by_monitor),
        prevalences=sensitivity_config["prevalences"],
        budgets=sensitivity_config["audit_budget_fractions"],
        population_size=int(sensitivity_config["population_size"]),
        replicates=int(sensitivity_config["replicates"]),
        seed=int(sensitivity_config["seed"]),
    )
    bootstrap_config = protocol["paired_bootstrap"]
    bootstrap = _paired_problem_bootstrap(
        locked,
        scores_by_monitor,
        budgets=budgets,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
        reference_monitor=str(bootstrap_config["reference_monitor"]),
    )
    write_json(args.outdir / "context_length_diagnostics.json", context)
    write_json(args.outdir / "prevalence_sensitivity.json", sensitivity)
    write_json(args.outdir / "paired_problem_bootstrap.json", bootstrap)

    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete_post_outcome_published_method_integration",
        "scope": config["scope"],
        "config": config,
        "config_sha256": config_hash,
        "runtime": runtime_provenance(repo_root),
        "inputs": {
            "design": str(design_path),
            "design_sha256": design_hash,
            "neutral_cache": str(neutral_cache_path),
            "neutral_cache_sha256": neutral_cache_hash,
            "n_rows": len(rows),
            "n_locked_rows": len(locked),
        },
        "checkpoint": checkpoint_metadata,
        "encoding_cache": {
            "path": str(encoded_cache_path),
            "sha256": file_sha256(encoded_cache_path),
            "metadata": encoding_metadata,
        },
        "selections": selections,
        "reconstruction_diagnostics": reconstruction,
        "budget_results": budget_results,
        "context_length_diagnostics": "context_length_diagnostics.json",
        "prevalence_sensitivity": "prevalence_sensitivity.json",
        "paired_problem_bootstrap": "paired_problem_bootstrap.json",
        "submission_files": output_files,
    }
    write_json(args.outdir / "results.json", payload)
    primary = str(float(protocol["primary_audit_budget_fraction"]))
    print(json.dumps(payload["budget_results"][primary], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
