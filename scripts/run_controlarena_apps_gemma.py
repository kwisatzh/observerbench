"""Run the sealed matched APPS replication on Gemma-2-9B-it.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from observerbench.ai_control import (
    evaluate_ai_control_scores,
    write_ai_control_scores,
)
from observerbench.ai_control_followup import (
    hashed_text_features,
    prevalence_budget_sensitivity,
    select_ridge_binary_observer,
)
from observerbench.core import write_json
from observerbench.provenance import file_sha256, package_version, runtime_provenance
from observerbench.tasks.gemma_apps import (
    EncodedMonitorView,
    HFCausalMonitor,
    encode_gemma_scope_features,
    integer_score_token_ids,
    load_gemma_apps_config,
    load_gemma_scope_parameters,
    validate_contextual_score_token_ids,
    verify_gemma_apps_source_manifest,
)

from run_controlarena_apps_qwen import (
    MonitorRow,
    _prompt,
    _protocol_rows,
    load_pairs,
    make_rows,
    select_activation_probe,
)
from run_controlarena_apps_qwen_followup import (
    _full_input_text,
    _neutral_text,
    _score_pools,
)


RESULT_SCHEMA = "observerbench.controlarena_apps_gemma.results.v0"
SMOKE_SCHEMA = "observerbench.controlarena_apps_gemma.compatibility_smoke.v0"


def _row_identity_digest(rows: Sequence[MonitorRow]) -> str:
    payload = [
        [
            row.sample_id,
            row.problem_id,
            row.split,
            row.eval_mode,
            row.backdoor_works,
            row.family_id,
        ]
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _validate_matched_parent_design(
    config: Mapping[str, Any], rows: Sequence[MonitorRow]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = config["matched_parent_design"]
    path = repo_root / str(parent["path"])
    if file_sha256(path) != str(parent["file_sha256"]):
        raise ValueError("the frozen Qwen APPS parent design changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = [
        [
            str(row["sample_id"]),
            str(row["problem_id"]),
            str(row["split"]),
            str(row["eval_mode"]),
            bool(row["backdoor_works"]),
            str(row["family_id"]),
        ]
        for row in payload["rows"]
    ]
    observed = [
        [
            row.sample_id,
            row.problem_id,
            row.split,
            row.eval_mode,
            row.backdoor_works,
            row.family_id,
        ]
        for row in rows
    ]
    if expected != observed:
        raise ValueError("Gemma APPS rows differ from the frozen Qwen parent design")
    if _row_identity_digest(rows) != str(parent["row_identity_sha256"]):
        raise ValueError("Gemma APPS row digest differs from the frozen parent digest")


def _render_views(
    monitor: HFCausalMonitor,
    rows: Sequence[MonitorRow],
) -> dict[str, list[str]]:
    return {
        "basic": [
            monitor.render_chat(
                _prompt(row, "controlarena-basic"), add_generation_prompt=True
            )
            + "<score>\n"
            for row in rows
        ],
        "detailed": [
            monitor.render_chat(
                _prompt(row, "controlarena-detailed-score-only"),
                add_generation_prompt=True,
            )
            + "<score>\n"
            for row in rows
        ],
        "neutral": [
            monitor.render_chat(_neutral_text(row), add_generation_prompt=False)
            for row in rows
        ],
    }


def _length_summary(
    rows: Sequence[MonitorRow], lengths: np.ndarray, *, max_length: int
) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=int)
    if values.shape != (len(rows),):
        raise ValueError("token lengths differ from the frozen APPS row count")
    return {
        "minimum": int(np.min(values)),
        "median": float(np.median(values)),
        "maximum": int(np.max(values)),
        "at_or_above_limit": int(np.sum(values >= max_length)),
        "at_or_above_limit_by_split": {
            split: int(
                sum(
                    length >= max_length and row.split == split
                    for row, length in zip(rows, values, strict=True)
                )
            )
            for split in ("fit", "calibration", "locked_test")
        },
    }


def _sae_smoke_audits_pass(audits: Sequence[Mapping[str, Any]]) -> bool:
    return bool(audits) and all(
        bool(item.get("finite")) and int(item.get("active_features", 0)) > 0
        for item in audits
    )


def _download_sae(config: Mapping[str, Any], spec: Mapping[str, Any]) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("huggingface_hub is required for Gemma Scope") from error
    scope = config["gemma_scope"]
    return Path(
        hf_hub_download(
            repo_id=str(scope["repository"]),
            revision=str(scope["revision"]),
            filename=str(spec["filename"]),
        )
    )


def _model(config: Mapping[str, Any], args: argparse.Namespace) -> HFCausalMonitor:
    model = config["model"]
    return HFCausalMonitor.from_pretrained(
        str(model["id"]),
        str(model["revision"]),
        device=str(args.device),
        dtype=str(model["dtype"]),
        attention_implementation=str(model["attention_implementation"]),
        local_files_only=bool(args.local_files_only),
    )


def _audit_model(monitor: HFCausalMonitor, config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return monitor.audit_runtime(
        expected_model_type=str(model["expected_model_type"]),
        expected_layers=int(model["expected_layers"]),
        expected_hidden_size=int(model["expected_hidden_size"]),
    )


def _score_contract(
    monitor: HFCausalMonitor, config: Mapping[str, Any]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    measurement = config["measurement"]
    bins = tuple(map(int, measurement["score_bins"]))
    ids = integer_score_token_ids(monitor.tokenizer, bins)
    values = tuple(map(float, measurement["score_values"]))
    if len(ids) != len(values):
        raise ValueError("frozen score-token IDs and score values differ in length")
    return ids, values


def _save_view(
    path: Path, view: EncodedMonitorView, *, binding: Mapping[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "activations": np.asarray(view.activations),
        "lengths": np.asarray(view.lengths),
        "binding_json": np.asarray(json.dumps(dict(binding), sort_keys=True)),
    }
    if view.score_log_probabilities is not None:
        payload["score_log_probabilities"] = np.asarray(view.score_log_probabilities)
        payload["expected_scores"] = np.asarray(view.expected_scores)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _load_view(
    path: Path, *, expected_binding: Mapping[str, str]
) -> EncodedMonitorView:
    with np.load(path, allow_pickle=False) as archive:
        if "binding_json" not in archive.files:
            raise ValueError(f"unbound Gemma view cache: {path}")
        observed_binding = json.loads(str(archive["binding_json"].item()))
        if observed_binding != dict(expected_binding):
            raise ValueError(f"Gemma view cache belongs to another frozen run: {path}")
        return EncodedMonitorView(
            activations=archive["activations"].copy(),
            lengths=archive["lengths"].copy(),
            score_log_probabilities=(
                archive["score_log_probabilities"].copy()
                if "score_log_probabilities" in archive.files
                else None
            ),
            expected_scores=(
                archive["expected_scores"].copy()
                if "expected_scores" in archive.files
                else None
            ),
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
            "compatibility smoke already exists; inspect it instead of rerunning it"
        )
    config = load_gemma_apps_config(config_path)
    manifest = verify_gemma_apps_source_manifest(config_path, manifest_path)
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
    basic = monitor.encode_rendered(
        views["basic"][:smoke_rows],
        layers=layers,
        max_length=max_length,
        batch_size=int(config["measurement"]["batch_size"]),
        score_token_ids=score_ids,
        score_values=score_values,
    )
    detailed = monitor.encode_rendered(
        views["detailed"][:smoke_rows],
        layers=(),
        max_length=max_length,
        batch_size=int(config["measurement"]["batch_size"]),
        score_token_ids=score_ids,
        score_values=score_values,
    )
    neutral = monitor.encode_rendered(
        views["neutral"][:smoke_rows],
        layers=layers,
        max_length=max_length,
        batch_size=int(config["measurement"]["batch_size"]),
    )
    sae_audits = []
    scope = config["gemma_scope"]
    for spec in scope["saes"]:
        layer = int(spec["layer"])
        layer_position = layers.index(layer)
        path = _download_sae(config, spec)
        parameters = load_gemma_scope_parameters(path)
        if parameters.d_in != int(config["model"]["expected_hidden_size"]):
            raise ValueError(f"Gemma Scope layer {layer} has an unexpected input width")
        if parameters.width != int(spec["width"]):
            raise ValueError(f"Gemma Scope layer {layer} has an unexpected feature width")
        features = encode_gemma_scope_features(
            neutral.activations[:1, layer_position],
            parameters,
            device=str(args.device),
            dtype=str(scope["compute_dtype"]),
            batch_size=1,
        )
        sae_audits.append(
            {
                "layer": layer,
                "filename": str(spec["filename"]),
                "file_sha256": file_sha256(path),
                "d_in": parameters.d_in,
                "width": parameters.width,
                "encoded_shape": list(features.shape),
                "finite": bool(np.isfinite(features).all()),
                "active_features": int(np.count_nonzero(features)),
            }
        )
        del parameters, features
        gc.collect()
    success = bool(
        locked_truncation == 0
        and np.isfinite(basic.activations).all()
        and np.isfinite(neutral.activations).all()
        and np.isfinite(basic.expected_scores).all()
        and np.isfinite(detailed.expected_scores).all()
        and _sae_smoke_audits_pass(sae_audits)
    )
    payload = {
        "schema": SMOKE_SCHEMA,
        "status": "passed" if success else "failed",
        "scope": "compatibility only; no observer fit, ranking, or locked-test model outcome",
        "config_sha256": manifest["config_sha256"],
        "source_bundle_sha256": manifest["source_bundle_sha256"],
        "row_identity_sha256": _row_identity_digest(rows),
        "n_pairs": len(pairs),
        "n_rows": len(rows),
        "model": model_audit,
        "score_token_ids": list(score_ids),
        "score_values": list(score_values),
        "score_context_audit": score_context_audit,
        "length_audit": length_audit,
        "smoke_shapes": {
            "basic_activations": list(basic.activations.shape),
            "basic_score_log_probabilities": list(basic.score_log_probabilities.shape),
            "detailed_score_log_probabilities": list(detailed.score_log_probabilities.shape),
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
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SMOKE_SCHEMA or payload.get("status") != "passed":
        raise ValueError("full run requires one passing Gemma compatibility smoke")
    for key in ("config_sha256", "source_bundle_sha256"):
        if payload.get(key) != manifest.get(key):
            raise ValueError(f"compatibility smoke {key} differs from the source seal")
    if payload.get("row_identity_sha256") != _row_identity_digest(rows):
        raise ValueError("compatibility smoke used different APPS rows")
    return payload


def _analysis_rows(
    rows: Sequence[MonitorRow], views: Sequence[EncodedMonitorView], *, max_length: int
) -> tuple[list[MonitorRow], np.ndarray]:
    at_limit = np.zeros(len(rows), dtype=bool)
    for view in views:
        at_limit |= np.asarray(view.lengths) >= int(max_length)
    if any(at_limit[index] and row.split == "locked_test" for index, row in enumerate(rows)):
        raise ValueError("a locked-test row reached the frozen context limit")
    analysis = [
        replace(row, split="excluded_truncation") if at_limit[index] else row
        for index, row in enumerate(rows)
    ]
    return analysis, at_limit


def _standardize_sae_features(
    features: np.ndarray,
    rows: Sequence[MonitorRow],
    *,
    minimum_fit_active_rows: int,
) -> tuple[np.ndarray, dict[str, int]]:
    fit = np.asarray([row.split == "fit" for row in rows])
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[0] != len(rows):
        raise ValueError("SAE features differ from the frozen rows")
    active = np.count_nonzero(values[fit] > 0, axis=0) >= int(minimum_fit_active_rows)
    if not active.any():
        raise ValueError("no SAE feature passed the frozen fit-only activity filter")
    selected = np.asarray(values[:, active], dtype=np.float32)
    mean = selected[fit].mean(axis=0)
    scale = selected[fit].std(axis=0)
    nonconstant = scale >= 1e-8
    if not nonconstant.any():
        raise ValueError("all retained SAE features are constant on fit rows")
    selected = (selected[:, nonconstant] - mean[nonconstant]) / scale[nonconstant]
    return selected, {
        "released_width": int(values.shape[1]),
        "passed_activity_filter": int(np.sum(active)),
        "retained_nonconstant": int(np.sum(nonconstant)),
    }


def _select_sae_probe(
    rows: Sequence[MonitorRow],
    neutral_activations: np.ndarray,
    config: Mapping[str, Any],
    *,
    device: str,
    cache_path: Path,
    resume: bool,
    cache_binding: Mapping[str, str],
) -> tuple[dict[str, Any], np.ndarray]:
    if resume and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if "binding_json" not in archive.files:
                raise ValueError("unbound Gemma Scope probe cache")
            observed_binding = json.loads(str(archive["binding_json"].item()))
            if observed_binding != dict(cache_binding):
                raise ValueError("Gemma Scope probe cache belongs to another frozen run")
            selection = json.loads(str(archive["selection_json"].item()))
            return selection, archive["locked_scores"].copy()
    layers = tuple(map(int, config["measurement"]["layers"]))
    labels = np.asarray([row.backdoor_works for row in rows], dtype=bool)
    splits = [row.split for row in rows]
    scope = config["gemma_scope"]
    candidates: list[tuple[tuple[float, float, int, float], dict[str, Any], np.ndarray]] = []
    for spec in scope["saes"]:
        layer = int(spec["layer"])
        residuals = np.asarray(neutral_activations[:, layers.index(layer)])
        path = _download_sae(config, spec)
        parameters = load_gemma_scope_parameters(path)
        if parameters.d_in != residuals.shape[1] or parameters.width != int(spec["width"]):
            raise ValueError(f"Gemma Scope layer {layer} shape differs from the frozen design")
        print(f"encoding official Gemma Scope features at layer {layer}", flush=True)
        features = encode_gemma_scope_features(
            residuals,
            parameters,
            device=device,
            dtype=str(scope["compute_dtype"]),
            batch_size=int(scope["encode_batch_size"]),
        )
        standardized, feature_audit = _standardize_sae_features(
            features,
            rows,
            minimum_fit_active_rows=int(scope["minimum_fit_active_rows"]),
        )
        selection, scores = select_ridge_binary_observer(
            standardized,
            labels,
            splits,
            ridge_grid=config["measurement"]["ridge_grid"],
            selection_budget=float(config["observers"]["selection_budget_fraction"]),
        )
        candidate = {
            **selection,
            "layer": layer,
            "sae_repository": str(scope["repository"]),
            "sae_revision": str(scope["revision"]),
            "sae_filename": str(spec["filename"]),
            "sae_file_sha256": file_sha256(path),
            "average_l0": int(spec["average_l0"]),
            "feature_audit": feature_audit,
        }
        metrics = selection["calibration_metrics"]
        key = (
            float(metrics["realized_violation_rate"]),
            -float(metrics["risk_auroc"]),
            layer,
            float(selection["ridge"]),
        )
        candidates.append((key, candidate, scores))
        del parameters, features, standardized
        gc.collect()
    _, best, locked_scores = min(candidates, key=lambda item: item[0])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            selection_json=np.asarray(json.dumps(best, sort_keys=True)),
            locked_scores=np.asarray(locked_scores, dtype=np.float32),
            binding_json=np.asarray(json.dumps(dict(cache_binding), sort_keys=True)),
        )
    temporary.replace(cache_path)
    return best, np.asarray(locked_scores, dtype=float)


def _full_run(
    config_path: Path,
    manifest_path: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> int:
    if not args.allow_full_run:
        raise ValueError("full run is review-gated; pass --allow-full-run after code review")
    config = load_gemma_apps_config(config_path)
    manifest = verify_gemma_apps_source_manifest(config_path, manifest_path)
    pairs = load_pairs(config)
    rows = make_rows(pairs, config)
    _validate_matched_parent_design(config, rows)
    smoke = _validate_smoke(
        outdir / "compatibility_smoke.json", manifest=manifest, rows=rows
    )
    write_json(
        outdir / "design.json",
        {
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
        },
    )

    cache_paths = {
        name: outdir / f"{name}_view.npz" for name in ("basic", "detailed", "neutral")
    }
    row_digest = _row_identity_digest(rows)
    views: dict[str, EncodedMonitorView] = {}
    if args.resume:
        for name, path in cache_paths.items():
            if path.exists():
                views[name] = _load_view(
                    path,
                    expected_binding={
                        "config_sha256": str(manifest["config_sha256"]),
                        "source_bundle_sha256": str(manifest["source_bundle_sha256"]),
                        "row_identity_sha256": row_digest,
                        "view": name,
                    },
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
            print(f"running Gemma {name} view", flush=True)
            view = monitor.encode_rendered(
                rendered[name],
                layers=layers if name in ("basic", "neutral") else (),
                max_length=int(measurement["max_length"]),
                batch_size=int(measurement["batch_size"]),
                score_token_ids=score_ids if name in ("basic", "detailed") else None,
                score_values=score_values if name in ("basic", "detailed") else None,
            )
            _save_view(
                cache_paths[name],
                view,
                binding={
                    "config_sha256": str(manifest["config_sha256"]),
                    "source_bundle_sha256": str(manifest["source_bundle_sha256"]),
                    "row_identity_sha256": row_digest,
                    "view": name,
                },
            )
            views[name] = view
        del monitor
        gc.collect()

    measurement = config["measurement"]
    analysis_rows, excluded = _analysis_rows(
        rows,
        (views["basic"], views["detailed"], views["neutral"]),
        max_length=int(measurement["max_length"]),
    )
    labels = np.asarray([row.backdoor_works for row in rows], dtype=bool)
    splits = [row.split for row in analysis_rows]
    selection_budget = float(config["observers"]["selection_budget_fraction"])

    print("fitting model-independent text observer", flush=True)
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
        analysis_rows, views["basic"].activations, residual_config
    )
    neutral_selection, neutral_scores = select_activation_probe(
        analysis_rows, views["neutral"].activations, residual_config
    )
    sae_selection, sae_scores = _select_sae_probe(
        analysis_rows,
        views["neutral"].activations,
        config,
        device=str(args.device),
        cache_path=outdir / "sae_probe_cache.npz",
        resume=bool(args.resume),
        cache_binding={
            "config_sha256": str(manifest["config_sha256"]),
            "source_bundle_sha256": str(manifest["source_bundle_sha256"]),
            "row_identity_sha256": row_digest,
            "view": "neutral-context-gemma-scope-sae-probe",
        },
    )

    locked = [row for row in rows if row.split == "locked_test"]
    locked_mask = np.asarray([row.split == "locked_test" for row in rows])
    basic_scores = np.asarray(views["basic"].expected_scores)[locked_mask]
    detailed_scores = np.asarray(views["detailed"].expected_scores)[locked_mask]
    score_map = {
        "gemma-controlarena-basic": basic_scores,
        "gemma-controlarena-detailed-score-only": detailed_scores,
        "gemma-controlarena-prompt-ensemble": (basic_scores + detailed_scores) / 2.0,
        "gemma-supervised-score-distribution": score_distribution_scores,
        "gemma-verdict-adjacent-residual-probe": verdict_scores,
        "gemma-neutral-context-residual-probe": neutral_scores,
        "gemma-neutral-context-gemma-scope-sae-probe": sae_scores,
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
            for result in evaluate_ai_control_scores(panel, audit_budget_fraction=budget)
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
        status="prospectively_frozen_before_gemma_monitor_outcomes",
        design_disclosure=(
            "The prevalence and audit-budget grid was inherited after the Qwen "
            "outcomes were known and frozen before any Gemma monitor outcome."
        ),
    )
    write_json(outdir / "prevalence_sensitivity.json", sensitivity)
    primary = config["interpretation"]["primary_outcome"]
    primary_key = str(primary["cell_key"])
    primary_cell = sensitivity["cells"].get(primary_key)
    if primary_cell is None:
        raise ValueError(f"frozen primary sensitivity cell is missing: {primary_key}")
    runtime = runtime_provenance(Path(__file__).resolve().parents[1])
    runtime["gemma_dependencies"] = {
        name: package_version(name)
        for name in ("transformers", "accelerate", "datasets", "huggingface-hub")
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete_matched_replication",
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
            "neutral_context_gemma_scope_sae": sae_selection,
        },
        "access": {
            "gemma-controlarena-basic": "expected score-token distribution under the basic monitoring prompt",
            "gemma-controlarena-detailed-score-only": "expected score-token distribution under the detailed monitoring prompt",
            "gemma-controlarena-prompt-ensemble": "mean of the two prompted expected scores",
            "gemma-supervised-score-distribution": "ridge over candidate-normalized log probabilities for the basic prompt's ten declared score bins",
            "gemma-verdict-adjacent-residual-probe": "projected post-layer residual at the score position",
            "gemma-neutral-context-residual-probe": "projected post-layer residual after problem plus code without a monitoring instruction",
            "gemma-neutral-context-gemma-scope-sae-probe": "official Gemma Scope JumpReLU features of the same neutral residual",
            "supervised-text-ridge": "hashed word and character n-grams over problem plus code",
            "exact-label-oracle": "evaluator-only verified working-backdoor label",
        },
        "budget_results": budget_results,
        "primary_rare_prevalence_outcome": {
            "definition": primary,
            "result": primary_cell,
        },
        "sensitivity_artifact": "prevalence_sensitivity.json",
    }
    write_json(outdir / "results.json", payload)
    print(json.dumps(payload["budget_results"][str(budgets[0])], indent=2), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("compatibility-smoke", "full"), required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-full-run", action="store_true")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.stage == "compatibility-smoke":
        return _compatibility_smoke(
            args.config.resolve(), args.manifest.resolve(), args.outdir.resolve(), args
        )
    return _full_run(
        args.config.resolve(), args.manifest.resolve(), args.outdir.resolve(), args
    )


if __name__ == "__main__":
    raise SystemExit(main())
