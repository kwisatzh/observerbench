"""Run the sealed matched APPS replication on Qwen3.5-9B.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This runner composes the checked APPS task, Gemma staging helpers, and the
Qwen3.5-specific model and Qwen-Scope adapters.  Compatibility smoke validates
the runtime without fitting or ranking an observer.  The full stage evaluates
the frozen non-SAE panel; official SAE probes are deliberately left to a
separate post-cache stage so their large checkpoints need not stay resident
during model inference.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from observerbench.ai_control import (
    evaluate_ai_control_scores,
    read_ai_control_scores,
    write_ai_control_scores,
)
from observerbench.ai_control_contrasts import paired_prevalence_contrasts
from observerbench.ai_control_followup import (
    hashed_text_features,
    prevalence_budget_sensitivity,
    select_ridge_binary_observer,
)
from observerbench.core import write_json
from observerbench.provenance import file_sha256, package_version, runtime_provenance
from observerbench.tasks.qwen35_apps import (
    EncodedMonitorView,
    Qwen35CausalMonitor,
    encode_qwen_scope_features,
    integer_score_token_ids,
    load_qwen_scope_parameters,
    validate_contextual_score_token_ids,
)
from observerbench.tasks.qwen35_apps.registration import (
    load_qwen35_apps_config,
    verify_qwen35_apps_source_manifest,
)

from run_controlarena_apps_gemma import (
    _analysis_rows,
    _length_summary,
    _load_view,
    _render_views,
    _row_identity_digest,
    _save_view,
    _validate_matched_parent_design,
)
from run_controlarena_apps_qwen import (
    MonitorRow,
    _protocol_rows,
    load_pairs,
    make_rows,
    select_activation_probe,
)
from run_controlarena_apps_qwen_followup import _full_input_text, _score_pools


RESULT_SCHEMA = "observerbench.controlarena_apps_qwen35.results.v0"
SMOKE_SCHEMA = "observerbench.controlarena_apps_qwen35.compatibility_smoke.v0"
SAE_RESULT_SCHEMA = "observerbench.controlarena_apps_qwen35.sae_results.v0"
PRIMARY_CONTRAST_ARTIFACT = "paired_primary_contrast.json"
PRIMARY_LEFT_MONITOR = "qwen35-neutral-context-residual-probe"
PRIMARY_RIGHT_MONITOR = "qwen35-verdict-adjacent-residual-probe"
DEFERRED_SAE_MONITOR = "qwen35-neutral-context-qwen-scope-sae-probe"
SAE_RESULT_ARTIFACT = "sae_results.json"
SAE_SCORE_ARTIFACT = "sae_ai_control_scores.csv"
SAE_SENSITIVITY_ARTIFACT = "sae_prevalence_sensitivity.json"
SAE_CONTRAST_ARTIFACT = "sae_paired_contrasts.json"


def _model(config: Mapping[str, Any], args: argparse.Namespace) -> Qwen35CausalMonitor:
    model = config["model"]
    return Qwen35CausalMonitor.from_pretrained(
        str(model["id"]),
        str(model["revision"]),
        device=str(args.device),
        dtype=str(model["dtype"]),
        attention_implementation=str(model["attention_implementation"]),
        local_files_only=bool(args.local_files_only),
    )


def _audit_model(
    monitor: Qwen35CausalMonitor, config: Mapping[str, Any]
) -> dict[str, Any]:
    model = config["model"]
    return monitor.audit_runtime(
        expected_model_type=str(model["expected_model_type"]),
        expected_text_model_type=str(model["expected_text_model_type"]),
        expected_layers=int(model["expected_layers"]),
        expected_hidden_size=int(model["expected_hidden_size"]),
        selected_layers=tuple(map(int, config["measurement"]["layers"])),
        expected_selected_layer_types={
            int(layer): str(layer_type)
            for layer, layer_type in model["selected_layer_types"].items()
        },
    )


def _score_contract(
    monitor: Qwen35CausalMonitor, config: Mapping[str, Any]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    measurement = config["measurement"]
    bins = tuple(map(int, measurement["score_bins"]))
    token_ids = integer_score_token_ids(monitor.tokenizer, bins)
    values = tuple(map(float, measurement["score_values"]))
    if len(token_ids) != len(values):
        raise ValueError("frozen Qwen3.5 score tokens and values differ in length")
    return token_ids, values


def _download_sae(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    local_files_only: bool,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("huggingface_hub is required for Qwen-Scope") from error
    scope = config["official_sae"]
    return Path(
        hf_hub_download(
            repo_id=str(scope["repository"]),
            revision=str(scope["revision"]),
            filename=str(spec["filename"]),
            local_files_only=bool(local_files_only),
        )
    )


def _sae_smoke_audits_pass(
    audits: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> bool:
    """Check all, and only, the three frozen Qwen-Scope layer contracts."""

    scope = config["official_sae"]
    expected_layers = tuple(map(int, scope["layers"]))
    if len(audits) != len(expected_layers):
        return False
    observed_layers = tuple(int(item.get("layer", -1)) for item in audits)
    if observed_layers != expected_layers:
        return False
    return all(
        bool(item.get("finite"))
        and int(item.get("d_in", -1)) == int(config["model"]["expected_hidden_size"])
        and int(item.get("width", -1)) == int(scope["width"])
        and int(item.get("top_k", -1)) == int(scope["target_l0"])
        and list(item.get("encoded_shape", ())) == [1, int(scope["width"])]
        and list(item.get("active_features_per_row", ())) == [
            int(scope["target_l0"])
        ]
        for item in audits
    )


def _load_checked_sae(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    local_files_only: bool,
) -> tuple[Path, Any, dict[str, Any]]:
    """Load one pinned SAE only after checking its registered file contract."""

    layer = int(spec["layer"])
    scope = config["official_sae"]
    path = _download_sae(
        config,
        spec,
        local_files_only=local_files_only,
    )
    observed_bytes = path.stat().st_size
    if observed_bytes != int(spec["bytes"]):
        raise ValueError(f"Qwen-Scope layer {layer} checkpoint size changed")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != str(spec["file_sha256"]):
        raise ValueError(f"Qwen-Scope layer {layer} checkpoint hash changed")
    parameters = load_qwen_scope_parameters(
        path,
        top_k=int(scope["target_l0"]),
    )
    if parameters.d_in != int(config["model"]["expected_hidden_size"]):
        raise ValueError(f"Qwen-Scope layer {layer} has an unexpected input width")
    if parameters.width != int(scope["width"]):
        raise ValueError(f"Qwen-Scope layer {layer} has an unexpected feature width")
    return path, parameters, {
        "layer": layer,
        "filename": str(spec["filename"]),
        "file_sha256": observed_sha256,
        "bytes": observed_bytes,
        "d_in": parameters.d_in,
        "width": parameters.width,
        "top_k": int(parameters.top_k),
    }


def _audit_sae_checkpoints(
    config: Mapping[str, Any],
    neutral: EncodedMonitorView,
    *,
    device: str,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    layers = tuple(map(int, config["measurement"]["layers"]))
    scope = config["official_sae"]
    audits: list[dict[str, Any]] = []
    for spec in scope["saes"]:
        layer = int(spec["layer"])
        if layer not in layers:
            raise ValueError(f"Qwen-Scope layer {layer} is not a measured residual layer")
        _path, parameters, checkpoint_audit = _load_checked_sae(
            config,
            spec,
            local_files_only=local_files_only,
        )
        features = encode_qwen_scope_features(
            np.asarray(neutral.activations[:1, layers.index(layer)]),
            parameters,
            device=device,
            dtype=str(scope["compute_dtype"]),
            batch_size=1,
        )
        audits.append(
            {
                **checkpoint_audit,
                "encoded_shape": list(features.shape),
                "finite": bool(np.isfinite(features).all()),
                "active_features_per_row": np.count_nonzero(features, axis=1)
                .astype(int)
                .tolist(),
            }
        )
        del parameters, features
        gc.collect()
    return audits


def _smoke_shapes_pass(
    basic: EncodedMonitorView,
    detailed: EncodedMonitorView,
    neutral: EncodedMonitorView,
    *,
    rows: int,
    layers: int,
    hidden_size: int,
    score_bins: int,
) -> bool:
    expected_residual = (rows, layers, hidden_size)
    expected_scores = (rows, score_bins)
    return bool(
        basic.activations.shape == expected_residual
        and neutral.activations.shape == expected_residual
        and detailed.activations.shape == (rows, 0, hidden_size)
        and basic.score_log_probabilities is not None
        and detailed.score_log_probabilities is not None
        and basic.expected_scores is not None
        and detailed.expected_scores is not None
        and basic.score_log_probabilities.shape == expected_scores
        and detailed.score_log_probabilities.shape == expected_scores
        and np.isfinite(basic.activations).all()
        and np.isfinite(neutral.activations).all()
        and np.isfinite(basic.score_log_probabilities).all()
        and np.isfinite(detailed.score_log_probabilities).all()
        and np.isfinite(basic.expected_scores).all()
        and np.isfinite(detailed.expected_scores).all()
    )


def _encode_score_only(
    monitor: Qwen35CausalMonitor,
    rendered: Sequence[str],
    *,
    hook_layer: int,
    hidden_size: int,
    max_length: int,
    batch_size: int,
    score_token_ids: Sequence[int],
    score_values: Sequence[float],
) -> EncodedMonitorView:
    """Read scores without relying on a composite config's hidden-size field.

    The shared encoder needs one explicit hidden size when no hooks are requested.
    Qwen3.5 may expose that value only below ``text_config``.  We therefore use
    one already-declared residual hook, discard it immediately, and retain only
    the score readout and token lengths.
    """

    encoded = monitor.encode_rendered(
        rendered,
        layers=(int(hook_layer),),
        max_length=int(max_length),
        batch_size=int(batch_size),
        score_token_ids=score_token_ids,
        score_values=score_values,
    )
    return EncodedMonitorView(
        activations=np.empty(
            (len(encoded.activations), 0, int(hidden_size)),
            dtype=np.float16,
        ),
        lengths=np.asarray(encoded.lengths),
        score_log_probabilities=np.asarray(encoded.score_log_probabilities),
        expected_scores=np.asarray(encoded.expected_scores),
    )


def _compatibility_smoke(
    config_path: Path,
    manifest_path: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> int:
    smoke_path = outdir / "compatibility_smoke.json"
    if smoke_path.exists():
        raise FileExistsError(
            "Qwen3.5 compatibility smoke already exists; inspect it instead"
        )
    config = load_qwen35_apps_config(config_path)
    manifest = verify_qwen35_apps_source_manifest(config_path, manifest_path)
    pairs = load_pairs(config)
    rows = make_rows(pairs, config)
    _validate_matched_parent_design(config, rows)

    monitor = _model(config, args)
    model_audit = _audit_model(monitor, config)
    score_ids, score_values = _score_contract(monitor, config)
    views = _render_views(monitor, rows)
    score_context_audit = validate_contextual_score_token_ids(
        monitor.tokenizer,
        (views["basic"][0], views["detailed"][0]),
        bins=tuple(map(int, config["measurement"]["score_bins"])),
        token_ids=score_ids,
    )
    max_length = int(config["measurement"]["max_length"])
    length_audit = {
        name: _length_summary(
            rows,
            monitor.token_lengths(rendered),
            max_length=max_length,
        )
        for name, rendered in views.items()
    }
    locked_truncation = sum(
        int(summary["at_or_above_limit_by_split"]["locked_test"])
        for summary in length_audit.values()
    )

    smoke_rows = 2 * int(config["compatibility_smoke"]["pairs"])
    layers = tuple(map(int, config["measurement"]["layers"]))
    batch_size = int(config["measurement"]["batch_size"])
    basic = monitor.encode_rendered(
        views["basic"][:smoke_rows],
        layers=layers,
        max_length=max_length,
        batch_size=batch_size,
        score_token_ids=score_ids,
        score_values=score_values,
    )
    detailed = _encode_score_only(
        monitor,
        views["detailed"][:smoke_rows],
        hook_layer=layers[0],
        hidden_size=int(config["model"]["expected_hidden_size"]),
        max_length=max_length,
        batch_size=batch_size,
        score_token_ids=score_ids,
        score_values=score_values,
    )
    neutral = monitor.encode_rendered(
        views["neutral"][:smoke_rows],
        layers=layers,
        max_length=max_length,
        batch_size=batch_size,
    )
    sae_audits = _audit_sae_checkpoints(
        config,
        neutral,
        device=str(args.device),
        local_files_only=bool(args.local_files_only),
    )
    success = bool(
        locked_truncation == 0
        and _smoke_shapes_pass(
            basic,
            detailed,
            neutral,
            rows=smoke_rows,
            layers=len(layers),
            hidden_size=int(config["model"]["expected_hidden_size"]),
            score_bins=len(score_ids),
        )
        and _sae_smoke_audits_pass(sae_audits, config)
    )
    payload = {
        "schema": SMOKE_SCHEMA,
        "status": "passed" if success else "failed",
        "scope": (
            "compatibility only; two APPS pairs reach model inference; no observer "
            "fit, AUROC, ranking, or monitor-quality outcome is computed"
        ),
        "monitor_quality_evaluated": False,
        "outcomes_reported": False,
        "config_sha256": manifest["config_sha256"],
        "source_bundle_sha256": manifest["source_bundle_sha256"],
        "row_identity_sha256": _row_identity_digest(rows),
        "n_pairs_in_frozen_design": len(pairs),
        "n_rows_in_frozen_design": len(rows),
        "n_pairs_in_model_smoke": int(config["compatibility_smoke"]["pairs"]),
        "n_rows_in_model_smoke": smoke_rows,
        "model": model_audit,
        "score_token_ids": list(score_ids),
        "score_values": list(score_values),
        "score_context_audit": score_context_audit,
        "length_audit": length_audit,
        "smoke_shapes": {
            "basic_activations": list(basic.activations.shape),
            "basic_score_log_probabilities": list(
                basic.score_log_probabilities.shape
            ),
            "detailed_score_log_probabilities": list(
                detailed.score_log_probabilities.shape
            ),
            "neutral_activations": list(neutral.activations.shape),
        },
        "saes": sae_audits,
    }
    write_json(smoke_path, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if success else 2


def _validate_smoke(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    rows: Sequence[MonitorRow],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SMOKE_SCHEMA or payload.get("status") != "passed":
        raise ValueError("full run requires one passing Qwen3.5 compatibility smoke")
    for key in ("config_sha256", "source_bundle_sha256"):
        if payload.get(key) != manifest.get(key):
            raise ValueError(f"Qwen3.5 compatibility smoke {key} differs from the seal")
    if payload.get("row_identity_sha256") != _row_identity_digest(rows):
        raise ValueError("Qwen3.5 compatibility smoke used different APPS rows")
    if payload.get("monitor_quality_evaluated") is not False:
        raise ValueError("Qwen3.5 smoke crossed the frozen compatibility boundary")
    if not _sae_smoke_audits_pass(payload.get("saes", ()), config):
        raise ValueError("Qwen3.5 smoke did not validate all three Qwen-Scope SAEs")
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("config_layout") not in {
        "nested_text_config",
        "extracted_text_config",
    }:
        raise ValueError("Qwen3.5 smoke has no checked text-architecture layout")
    return payload


def _cache_binding(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[MonitorRow],
    *,
    view: str,
) -> dict[str, str]:
    return {
        "task_id": str(config["task_id"]),
        "config_sha256": str(manifest["config_sha256"]),
        "source_bundle_sha256": str(manifest["source_bundle_sha256"]),
        "row_identity_sha256": _row_identity_digest(rows),
        "model_revision": str(config["model"]["revision"]),
        "view": str(view),
    }


def _design_payload(
    config: Mapping[str, Any], rows: Sequence[MonitorRow]
) -> dict[str, Any]:
    return {
        "schema": config["schema"],
        "config": config,
        "row_identity_sha256": _row_identity_digest(rows),
        "rows": [
            {
                "sample_id": row.sample_id,
                "problem_id": row.problem_id,
                "split": row.split,
                "eval_mode": row.eval_mode,
                "backdoor_works": row.backdoor_works,
                "family_id": row.family_id,
            }
            for row in rows
        ],
    }


def _write_or_validate_design(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != dict(payload):
            raise ValueError("existing Qwen3.5 design belongs to another frozen run")
        return
    write_json(path, payload)


def _standardize_qwen_scope_features(
    features: np.ndarray,
    rows: Sequence[MonitorRow],
    *,
    minimum_fit_active_rows: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the frozen fit-only filter and scaling to exact TopK features."""

    values = np.asarray(features)
    fit = np.asarray([row.split == "fit" for row in rows])
    if values.ndim != 2 or values.shape[0] != len(rows):
        raise ValueError("Qwen-Scope features differ from the frozen APPS rows")
    if not fit.any() or not np.isfinite(values).all():
        raise ValueError("Qwen-Scope fit features must be nonempty and finite")
    active = np.count_nonzero(values[fit] != 0.0, axis=0) >= int(
        minimum_fit_active_rows
    )
    if not active.any():
        raise ValueError("no Qwen-Scope feature passed the fit-only activity filter")
    selected = np.asarray(values[:, active], dtype=np.float32)
    mean = selected[fit].mean(axis=0)
    scale = selected[fit].std(axis=0)
    nonconstant = scale >= 1e-8
    if not nonconstant.any():
        raise ValueError("all retained Qwen-Scope features are constant on fit rows")
    selected = (selected[:, nonconstant] - mean[nonconstant]) / scale[nonconstant]
    return selected, {
        "released_width": int(values.shape[1]),
        "passed_fit_activity_filter": int(np.sum(active)),
        "retained_fit_nonconstant": int(np.sum(nonconstant)),
    }


def _select_qwen_scope_probe(
    rows: Sequence[MonitorRow],
    neutral_activations: np.ndarray,
    config: Mapping[str, Any],
    *,
    device: str,
    local_files_only: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    """Select the SAE layer and ridge using calibration loss only."""

    layers = tuple(map(int, config["measurement"]["layers"]))
    activations = np.asarray(neutral_activations)
    if activations.shape != (
        len(rows),
        len(layers),
        int(config["model"]["expected_hidden_size"]),
    ):
        raise ValueError("neutral residual cache has the wrong Qwen3.5 shape")
    labels = np.asarray([row.backdoor_works for row in rows], dtype=bool)
    splits = [row.split for row in rows]
    scope = config["official_sae"]
    selection_budget = float(config["observers"]["selection_budget_fraction"])
    candidates: list[
        tuple[tuple[float, float, int, float], dict[str, Any], np.ndarray]
    ] = []
    for spec in scope["saes"]:
        layer = int(spec["layer"])
        if layer not in layers:
            raise ValueError(f"Qwen-Scope layer {layer} is not in the neutral cache")
        _path, parameters, checkpoint_audit = _load_checked_sae(
            config,
            spec,
            local_files_only=local_files_only,
        )
        print(f"encoding official Qwen-Scope features at layer {layer}", flush=True)
        features = encode_qwen_scope_features(
            activations[:, layers.index(layer)],
            parameters,
            device=device,
            dtype=str(scope["compute_dtype"]),
            batch_size=int(config["measurement"]["batch_size"]),
        )
        active_per_row = np.count_nonzero(features, axis=1)
        if features.shape != (len(rows), int(scope["width"])):
            raise ValueError(f"Qwen-Scope layer {layer} encoded the wrong shape")
        if not np.isfinite(features).all() or not np.all(
            active_per_row == int(scope["target_l0"])
        ):
            raise ValueError(f"Qwen-Scope layer {layer} violated exact TopK features")
        standardized, feature_audit = _standardize_qwen_scope_features(
            features,
            rows,
            minimum_fit_active_rows=int(scope["minimum_fit_active_rows"]),
        )
        selection, locked_scores = select_ridge_binary_observer(
            standardized,
            labels,
            splits,
            ridge_grid=config["measurement"]["ridge_grid"],
            selection_budget=selection_budget,
        )
        candidate = {
            **selection,
            **checkpoint_audit,
            "feature_audit": feature_audit,
            "selection_rows": "fit for ridge fit; calibration for layer/ridge choice",
            "locked_labels_used_for_selection": False,
        }
        metrics = selection["calibration_metrics"]
        key = (
            float(metrics["realized_violation_rate"]),
            -float(metrics["risk_auroc"]),
            layer,
            float(selection["ridge"]),
        )
        candidates.append((key, candidate, np.asarray(locked_scores, dtype=float)))
        del parameters, features, standardized
        gc.collect()
    if not candidates:
        raise ValueError("no Qwen-Scope candidate was evaluated")
    _key, selected, locked_scores = min(candidates, key=lambda item: item[0])
    return {
        **selected,
        "layer_candidates": [candidate for _key, candidate, _scores in candidates],
        "selection_key": (
            "calibration realized_violation_rate, then calibration AUROC, "
            "then layer and ridge"
        ),
        "refit_rows": "fit plus calibration only",
        "locked_test_scored_after_selection": True,
    }, locked_scores


def _paired_primary_contrast(
    panel: Sequence[Any],
    sensitivity: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = config["interpretation"]["primary_outcome"]
    contrast_name = str(primary["contrast"])
    cell_key = str(primary["cell_key"])
    artifact = paired_prevalence_contrasts(
        _score_pools(panel),
        sensitivity,
        contrasts={
            contrast_name: (PRIMARY_LEFT_MONITOR, PRIMARY_RIGHT_MONITOR),
        },
        cell_keys=(cell_key,),
    )
    try:
        result = artifact["cells"][cell_key]["contrasts"][contrast_name]
    except KeyError as error:
        raise ValueError("frozen Qwen3.5 primary paired contrast is missing") from error
    return artifact, result


def _validate_core_results(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    rows: Sequence[MonitorRow],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != RESULT_SCHEMA or payload.get("status") != (
        "complete_core_panel_sae_probe_deferred_post_cache"
    ):
        raise ValueError("SAE stage requires one complete Qwen3.5 core result")
    source = payload.get("source_manifest")
    if not isinstance(source, Mapping):
        raise ValueError("Qwen3.5 core result has no source-manifest binding")
    for key in ("config_sha256", "source_bundle_sha256"):
        if source.get(key) != manifest.get(key):
            raise ValueError(f"Qwen3.5 core result {key} differs from the seal")
    if payload.get("n_selected_pairs") != len(rows) // 2:
        raise ValueError("Qwen3.5 core result used a different APPS row panel")
    core_config = payload.get("config")
    if not isinstance(core_config, Mapping) or core_config.get("task_id") != config.get(
        "task_id"
    ):
        raise ValueError("Qwen3.5 core result belongs to another task")
    return payload


def _core_comparison_panel(
    core_panel: Sequence[Any], rows: Sequence[MonitorRow]
) -> tuple[Any, ...]:
    names = (
        "qwen35-controlarena-basic",
        "qwen35-controlarena-detailed-score-only",
        PRIMARY_LEFT_MONITOR,
    )
    expected_ids = sorted(row.sample_id for row in rows if row.split == "locked_test")
    selected = tuple(row for row in core_panel if row.monitor_name in names)
    for name in names:
        observed_ids = sorted(row.sample_id for row in selected if row.monitor_name == name)
        if observed_ids != expected_ids:
            raise ValueError(f"Qwen3.5 core score panel is incomplete for {name}")
    return selected


def _sae_run(
    config_path: Path,
    manifest_path: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> int:
    if not args.allow_sae_run:
        raise ValueError("SAE stage is review-gated; pass --allow-sae-run after review")
    sae_paths = tuple(
        outdir / name
        for name in (
            SAE_RESULT_ARTIFACT,
            SAE_SCORE_ARTIFACT,
            SAE_SENSITIVITY_ARTIFACT,
            SAE_CONTRAST_ARTIFACT,
        )
    )
    existing = [path.name for path in sae_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Qwen3.5 SAE artifacts are never overwritten: " + ", ".join(existing)
        )
    config = load_qwen35_apps_config(config_path)
    manifest = verify_qwen35_apps_source_manifest(config_path, manifest_path)
    pairs = load_pairs(config)
    rows = make_rows(pairs, config)
    _validate_matched_parent_design(config, rows)
    smoke = _validate_smoke(
        outdir / "compatibility_smoke.json",
        manifest=manifest,
        rows=rows,
        config=config,
    )
    core_path = outdir / "results.json"
    core = _validate_core_results(
        core_path,
        manifest=manifest,
        rows=rows,
        config=config,
    )

    views = {
        name: _load_view(
            outdir / f"{name}_view.npz",
            expected_binding=_cache_binding(
                config,
                manifest,
                rows,
                view=name,
            ),
        )
        for name in ("basic", "detailed", "neutral")
    }
    measurement = config["measurement"]
    if not _smoke_shapes_pass(
        views["basic"],
        views["detailed"],
        views["neutral"],
        rows=len(rows),
        layers=len(measurement["layers"]),
        hidden_size=int(config["model"]["expected_hidden_size"]),
        score_bins=len(measurement["score_bins"]),
    ):
        raise ValueError("Qwen3.5 bound caches cannot support the SAE stage")
    analysis_rows, excluded = _analysis_rows(
        rows,
        (views["basic"], views["detailed"], views["neutral"]),
        max_length=int(measurement["max_length"]),
    )
    selection, sae_scores = _select_qwen_scope_probe(
        analysis_rows,
        views["neutral"].activations,
        config,
        device=str(args.device),
        local_files_only=bool(args.local_files_only),
    )

    locked = [row for row in rows if row.split == "locked_test"]
    core_panel = tuple(read_ai_control_scores(outdir / "ai_control_scores.csv"))
    comparison_panel = _core_comparison_panel(core_panel, rows)
    sae_panel = _protocol_rows(locked, DEFERRED_SAE_MONITOR, sae_scores)
    panel = (*comparison_panel, *sae_panel)
    write_ai_control_scores(outdir / SAE_SCORE_ARTIFACT, panel)

    budgets = (
        float(config["protocol"]["audit_budget_fraction"]),
        *map(float, config["protocol"]["secondary_audit_budget_fractions"]),
    )
    budget_results = {
        str(budget): {
            result.monitor_name: result.to_dict()
            for result in evaluate_ai_control_scores(
                panel,
                audit_budget_fraction=budget,
            )
        }
        for budget in budgets
    }
    sensitivity_config = config["sensitivity"]
    sensitivity = prevalence_budget_sensitivity(
        _score_pools(panel),
        prevalences=sensitivity_config["prevalences"],
        budgets=sensitivity_config["audit_budget_fractions"],
        population_size=int(sensitivity_config["population_size"]),
        replicates=int(sensitivity_config["replicates"]),
        seed=int(sensitivity_config["seed"]),
        status=str(config["status"]),
        design_disclosure=(
            "The SAE family, layer set, controller, prevalence grid, and budget "
            "grid were frozen before Qwen3.5 monitor outcomes."
        ),
    )
    write_json(outdir / SAE_SENSITIVITY_ARTIFACT, sensitivity)
    contrasts = paired_prevalence_contrasts(
        _score_pools(panel),
        sensitivity,
        contrasts={
            "sae-minus-basic": (
                DEFERRED_SAE_MONITOR,
                "qwen35-controlarena-basic",
            ),
            "sae-minus-detailed": (
                DEFERRED_SAE_MONITOR,
                "qwen35-controlarena-detailed-score-only",
            ),
            "sae-minus-neutral-residual": (
                DEFERRED_SAE_MONITOR,
                PRIMARY_LEFT_MONITOR,
            ),
        },
    )
    primary_key = str(config["interpretation"]["primary_outcome"]["cell_key"])
    if primary_key not in contrasts["cells"]:
        raise ValueError("Qwen3.5 SAE contrasts omit the frozen operating cell")
    write_json(outdir / SAE_CONTRAST_ARTIFACT, contrasts)

    payload = {
        "schema": SAE_RESULT_SCHEMA,
        "status": "complete_post_cache_sae_panel",
        "scope": (
            "separate post-cache SAE stage; core results.json and "
            "ai_control_scores.csv remain unchanged"
        ),
        "config_sha256": manifest["config_sha256"],
        "source_bundle_sha256": manifest["source_bundle_sha256"],
        "core_results_sha256": file_sha256(core_path),
        "core_scores_sha256": file_sha256(outdir / "ai_control_scores.csv"),
        "compatibility_smoke_status": smoke["status"],
        "core_status": core["status"],
        "excluded_fit_or_calibration_rows_at_context_limit": int(np.sum(excluded)),
        "selection": selection,
        "access": {
            DEFERRED_SAE_MONITOR: (
                "calibration-selected ridge over official Qwen-Scope TopK-50 "
                "features of the bound neutral residual; the SAE is trained on "
                "the base model and is not checkpoint-matched"
            )
        },
        "fixed_budget_results": budget_results,
        "primary_operating_cell": contrasts["cells"][primary_key],
        "score_artifact": SAE_SCORE_ARTIFACT,
        "sensitivity_artifact": SAE_SENSITIVITY_ARTIFACT,
        "paired_contrast_artifact": SAE_CONTRAST_ARTIFACT,
    }
    write_json(outdir / SAE_RESULT_ARTIFACT, payload)
    print(json.dumps(payload["fixed_budget_results"][str(budgets[0])], indent=2), flush=True)
    return 0


def _full_run(
    config_path: Path,
    manifest_path: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> int:
    if not args.allow_full_run:
        raise ValueError("full run is review-gated; pass --allow-full-run after code review")
    result_path = outdir / "results.json"
    if result_path.exists():
        raise FileExistsError(
            "Qwen3.5 results already exist; completed results are never overwritten"
        )
    config = load_qwen35_apps_config(config_path)
    manifest = verify_qwen35_apps_source_manifest(config_path, manifest_path)
    pairs = load_pairs(config)
    rows = make_rows(pairs, config)
    _validate_matched_parent_design(config, rows)
    smoke = _validate_smoke(
        outdir / "compatibility_smoke.json",
        manifest=manifest,
        rows=rows,
        config=config,
    )
    _write_or_validate_design(outdir / "design.json", _design_payload(config, rows))

    cache_paths = {
        name: outdir / f"{name}_view.npz"
        for name in ("basic", "detailed", "neutral")
    }
    existing_caches = [path for path in cache_paths.values() if path.exists()]
    if existing_caches and not args.resume:
        raise FileExistsError("Qwen3.5 view caches exist; pass --resume to verify and reuse")
    views: dict[str, EncodedMonitorView] = {}
    if args.resume:
        for name, path in cache_paths.items():
            if path.exists():
                views[name] = _load_view(
                    path,
                    expected_binding=_cache_binding(
                        config,
                        manifest,
                        rows,
                        view=name,
                    ),
                )
    missing = [name for name in cache_paths if name not in views]
    model_audit = dict(smoke["model"])
    if missing:
        monitor = _model(config, args)
        model_audit = _audit_model(monitor, config)
        score_ids, score_values = _score_contract(monitor, config)
        rendered = _render_views(monitor, rows)
        measurement = config["measurement"]
        layers = tuple(map(int, measurement["layers"]))
        for name in missing:
            print(f"running Qwen3.5 {name} view", flush=True)
            if name == "detailed":
                view = _encode_score_only(
                    monitor,
                    rendered[name],
                    hook_layer=layers[0],
                    hidden_size=int(config["model"]["expected_hidden_size"]),
                    max_length=int(measurement["max_length"]),
                    batch_size=int(measurement["batch_size"]),
                    score_token_ids=score_ids,
                    score_values=score_values,
                )
            else:
                view = monitor.encode_rendered(
                    rendered[name],
                    layers=layers,
                    max_length=int(measurement["max_length"]),
                    batch_size=int(measurement["batch_size"]),
                    score_token_ids=score_ids if name == "basic" else None,
                    score_values=score_values if name == "basic" else None,
                )
            _save_view(
                cache_paths[name],
                view,
                binding=_cache_binding(
                    config,
                    manifest,
                    rows,
                    view=name,
                ),
            )
            views[name] = view
        del monitor
        gc.collect()

    measurement = config["measurement"]
    if not _smoke_shapes_pass(
        views["basic"],
        views["detailed"],
        views["neutral"],
        rows=len(rows),
        layers=len(measurement["layers"]),
        hidden_size=int(config["model"]["expected_hidden_size"]),
        score_bins=len(measurement["score_bins"]),
    ):
        raise ValueError("Qwen3.5 bound view caches have incompatible shapes or values")
    analysis_rows, excluded = _analysis_rows(
        rows,
        (views["basic"], views["detailed"], views["neutral"]),
        max_length=int(measurement["max_length"]),
    )
    labels = np.asarray([row.backdoor_works for row in rows], dtype=bool)
    splits = [row.split for row in analysis_rows]
    selection_budget = float(config["observers"]["selection_budget_fraction"])

    text_features = hashed_text_features(
        [_full_input_text(row) for row in rows],
        dimension=int(config["observers"]["supervised_text_dimension"]),
    )
    text_selection, text_scores = select_ridge_binary_observer(
        text_features,
        labels,
        splits,
        ridge_grid=measurement["ridge_grid"],
        selection_budget=selection_budget,
    )
    del text_features
    score_selection, score_distribution_scores = select_ridge_binary_observer(
        np.asarray(views["basic"].score_log_probabilities),
        labels,
        splits,
        ridge_grid=measurement["ridge_grid"],
        selection_budget=selection_budget,
    )
    residual_config = json.loads(json.dumps(config))
    residual_config["protocol"]["audit_budget_fraction"] = selection_budget
    verdict_selection, verdict_scores = select_activation_probe(
        analysis_rows,
        views["basic"].activations,
        residual_config,
    )
    neutral_selection, neutral_scores = select_activation_probe(
        analysis_rows,
        views["neutral"].activations,
        residual_config,
    )

    locked = [row for row in rows if row.split == "locked_test"]
    locked_mask = np.asarray([row.split == "locked_test" for row in rows])
    basic_scores = np.asarray(views["basic"].expected_scores)[locked_mask]
    detailed_scores = np.asarray(views["detailed"].expected_scores)[locked_mask]
    score_map = {
        "qwen35-controlarena-basic": basic_scores,
        "qwen35-controlarena-detailed-score-only": detailed_scores,
        "qwen35-controlarena-prompt-ensemble": (basic_scores + detailed_scores) / 2.0,
        "qwen35-supervised-score-distribution": score_distribution_scores,
        PRIMARY_RIGHT_MONITOR: verdict_scores,
        PRIMARY_LEFT_MONITOR: neutral_scores,
        "supervised-text-ridge": text_scores,
        "exact-label-oracle": [float(row.backdoor_works) for row in locked],
    }
    panel = tuple(
        sample
        for monitor_name, scores in score_map.items()
        for sample in _protocol_rows(locked, monitor_name, scores)
    )
    write_ai_control_scores(outdir / "ai_control_scores.csv", panel)
    budgets = (
        float(config["protocol"]["audit_budget_fraction"]),
        *map(float, config["protocol"]["secondary_audit_budget_fractions"]),
    )
    budget_results = {
        str(budget): {
            result.monitor_name: result.to_dict()
            for result in evaluate_ai_control_scores(
                panel,
                audit_budget_fraction=budget,
            )
        }
        for budget in budgets
    }
    sensitivity_config = config["sensitivity"]
    sensitivity = prevalence_budget_sensitivity(
        _score_pools(panel),
        prevalences=sensitivity_config["prevalences"],
        budgets=sensitivity_config["audit_budget_fractions"],
        population_size=int(sensitivity_config["population_size"]),
        replicates=int(sensitivity_config["replicates"]),
        seed=int(sensitivity_config["seed"]),
        status=str(config["status"]),
        design_disclosure=(
            "The prevalence and audit-budget grid was inherited after the "
            "Qwen2.5 and Gemma outcomes and frozen before Qwen3.5 outcomes."
        ),
    )
    write_json(outdir / "prevalence_sensitivity.json", sensitivity)
    contrast_artifact, primary_contrast = _paired_primary_contrast(
        panel,
        sensitivity,
        config,
    )
    write_json(outdir / PRIMARY_CONTRAST_ARTIFACT, contrast_artifact)

    runtime = runtime_provenance(Path(__file__).resolve().parents[1])
    runtime["qwen35_dependencies"] = {
        name: package_version(name)
        for name in ("transformers", "accelerate", "datasets", "huggingface-hub")
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete_core_panel_sae_probe_deferred_post_cache",
        "disclosure": config["disclosure"],
        "config": config,
        "source_manifest": manifest,
        "compatibility_smoke": smoke,
        "runtime": runtime,
        "model_audit": model_audit,
        "n_selected_pairs": len(pairs),
        "excluded_fit_or_calibration_rows_at_context_limit": int(np.sum(excluded)),
        "selections": {
            "supervised_text": text_selection,
            "supervised_score_distribution": score_selection,
            "verdict_adjacent_residual": verdict_selection,
            "neutral_context_residual": neutral_selection,
        },
        "access": {
            "qwen35-controlarena-basic": "expected score-token distribution under the basic monitoring prompt",
            "qwen35-controlarena-detailed-score-only": "expected score-token distribution under the detailed monitoring prompt",
            "qwen35-controlarena-prompt-ensemble": "mean of the two prompted expected scores",
            "qwen35-supervised-score-distribution": "ridge over the basic prompt's ten candidate-normalized score-bin log probabilities",
            PRIMARY_RIGHT_MONITOR: "projected post-layer residual at the score position",
            PRIMARY_LEFT_MONITOR: "projected post-layer residual after problem plus code without a monitoring instruction",
            "supervised-text-ridge": "hashed word and character n-grams over problem plus code",
            "exact-label-oracle": "evaluator-only verified working-backdoor label",
            DEFERRED_SAE_MONITOR: (
                "DEFERRED POST-CACHE: calibration-only selection over official "
                "Qwen-Scope TopK-50 features at layers 12, 20, and 28; the SAEs "
                "were trained on the base checkpoint and are not checkpoint-matched"
            ),
        },
        "deferred_sae_stage": {
            "status": "not_run_in_core_panel",
            "monitor": DEFERRED_SAE_MONITOR,
            "layers": list(map(int, config["official_sae"]["layers"])),
            "input_cache": "neutral_view.npz",
            "reason": (
                "run the three large official SAE encoders after the bound neutral "
                "residual cache exists; no SAE score or ranking is claimed here"
            ),
        },
        "budget_results": budget_results,
        "primary_rare_prevalence_outcome": {
            "definition": config["interpretation"]["primary_outcome"],
            "paired_contrast": primary_contrast,
            "artifact": PRIMARY_CONTRAST_ARTIFACT,
        },
        "sensitivity_artifact": "prevalence_sensitivity.json",
    }
    write_json(result_path, payload)
    print(json.dumps(payload["budget_results"][str(budgets[0])], indent=2), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("compatibility-smoke", "full", "sae"),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--allow-sae-run", action="store_true")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.stage == "compatibility-smoke":
        return _compatibility_smoke(
            args.config.resolve(),
            args.manifest.resolve(),
            args.outdir.resolve(),
            args,
        )
    if args.stage == "sae":
        return _sae_run(
            args.config.resolve(),
            args.manifest.resolve(),
            args.outdir.resolve(),
            args,
        )
    return _full_run(
        args.config.resolve(),
        args.manifest.resolve(),
        args.outdir.resolve(),
        args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
