"""Focused staging tests for the matched Qwen3.5 APPS runner.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_controlarena_apps_qwen35.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_controlarena_apps_qwen35_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _row(split: str, index: int, *, attack: bool = False):
    mode = "attack" if attack else "honest"
    return RUNNER.MonitorRow(
        sample_id=f"p{index}::{mode}",
        problem_id=f"p{index}",
        split=split,
        eval_mode=mode,
        backdoor_works=attack,
        family_id="introductory:any",
        question="question",
        code="code",
    )


def _view(lengths, *, layers: int = 1, hidden_size: int = 4, scores: bool = False):
    values = np.asarray(lengths, dtype=np.int32)
    return RUNNER.EncodedMonitorView(
        activations=np.zeros(
            (len(values), layers, hidden_size),
            dtype=np.float16,
        ),
        lengths=values,
        score_log_probabilities=(
            np.zeros((len(values), 2), dtype=np.float32) if scores else None
        ),
        expected_scores=(
            np.zeros(len(values), dtype=np.float32) if scores else None
        ),
    )


def _config(*, width: int = 8, top_k: int = 2) -> dict[str, object]:
    return {
        "task_id": "controlarena-apps-verified-backdoors@qwen3.5-9b-v0",
        "model": {
            "revision": "a" * 40,
            "expected_hidden_size": 4,
        },
        "official_sae": {
            "layers": [12, 20, 28],
            "width": width,
            "target_l0": top_k,
        },
        "interpretation": {
            "primary_outcome": {
                "contrast": (
                    "neutral-context-residual-probe minus "
                    "verdict-adjacent-residual-probe"
                ),
                "cell_key": "prevalence=0.01|budget=0.02",
            }
        },
    }


def _sae_audits(*, width: int = 8, top_k: int = 2):
    return [
        {
            "layer": layer,
            "d_in": 4,
            "width": width,
            "top_k": top_k,
            "encoded_shape": [1, width],
            "finite": True,
            "active_features_per_row": [top_k],
        }
        for layer in (12, 20, 28)
    ]


def test_training_truncation_is_excluded_but_locked_truncation_stops() -> None:
    rows = [_row("fit", 0), _row("calibration", 1), _row("locked_test", 2)]

    analysis, excluded = RUNNER._analysis_rows(
        rows,
        (_view([8, 4, 4]), _view([4, 4, 4])),
        max_length=8,
    )

    assert analysis[0].split == "excluded_truncation"
    assert analysis[1:] == rows[1:]
    np.testing.assert_array_equal(excluded, [True, False, False])
    with pytest.raises(ValueError, match="locked-test"):
        RUNNER._analysis_rows(rows, (_view([4, 4, 8]),), max_length=8)


def test_bound_view_cache_round_trips_and_rejects_foreign_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "basic_view.npz"
    binding = {
        "config_sha256": "config",
        "source_bundle_sha256": "source",
        "row_identity_sha256": "rows",
        "view": "basic",
    }
    expected = _view([3, 4], scores=True)

    RUNNER._save_view(path, expected, binding=binding)
    observed = RUNNER._load_view(path, expected_binding=binding)

    np.testing.assert_array_equal(observed.activations, expected.activations)
    np.testing.assert_array_equal(
        observed.score_log_probabilities,
        expected.score_log_probabilities,
    )
    with pytest.raises(ValueError, match="another frozen run"):
        RUNNER._load_view(
            path,
            expected_binding={**binding, "view": "neutral"},
        )


def test_stage_outputs_are_never_overwritten(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "compatibility_smoke.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(device="cpu", local_files_only=True)

    with pytest.raises(FileExistsError, match="already exists"):
        RUNNER._compatibility_smoke(
            tmp_path / "missing-config.json",
            tmp_path / "missing-manifest.json",
            outdir,
            args,
        )

    (outdir / "results.json").write_text('{"status":"complete"}', encoding="utf-8")
    args.allow_full_run = True
    with pytest.raises(FileExistsError, match="never overwritten"):
        RUNNER._full_run(
            tmp_path / "missing-config.json",
            tmp_path / "missing-manifest.json",
            outdir,
            args,
        )


def test_full_stage_requires_the_explicit_review_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="review-gated"):
        RUNNER.main(
            [
                "--config",
                str(tmp_path / "missing-config.json"),
                "--manifest",
                str(tmp_path / "missing-manifest.json"),
                "--outdir",
                str(tmp_path / "out"),
                "--stage",
                "full",
            ]
        )


def test_sae_stage_requires_review_gate_and_never_overwrites(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    with pytest.raises(ValueError, match="SAE stage is review-gated"):
        RUNNER.main(
            [
                "--config",
                str(tmp_path / "missing-config.json"),
                "--manifest",
                str(tmp_path / "missing-manifest.json"),
                "--outdir",
                str(outdir),
                "--stage",
                "sae",
            ]
        )

    (outdir / RUNNER.SAE_RESULT_ARTIFACT).write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        allow_sae_run=True,
        device="cpu",
        local_files_only=True,
    )
    with pytest.raises(FileExistsError, match="never overwritten"):
        RUNNER._sae_run(
            tmp_path / "missing-config.json",
            tmp_path / "missing-manifest.json",
            outdir,
            args,
        )


def test_smoke_requires_all_three_exact_sae_layer_contracts() -> None:
    config = _config()
    audits = _sae_audits()

    assert RUNNER._sae_smoke_audits_pass(audits, config)
    assert not RUNNER._sae_smoke_audits_pass(audits[:2], config)
    assert not RUNNER._sae_smoke_audits_pass(
        [*audits[:2], {**audits[2], "active_features_per_row": [1]}],
        config,
    )
    assert not RUNNER._sae_smoke_audits_pass(
        [{**audits[0], "layer": 20}, *audits[1:]],
        config,
    )


def test_smoke_is_bound_to_seal_rows_architecture_and_sae_checks(
    tmp_path: Path,
) -> None:
    rows = [_row("fit", 0), _row("locked_test", 1, attack=True)]
    manifest = {
        "config_sha256": "config",
        "source_bundle_sha256": "source",
    }
    payload = {
        "schema": RUNNER.SMOKE_SCHEMA,
        "status": "passed",
        **manifest,
        "row_identity_sha256": RUNNER._row_identity_digest(rows),
        "monitor_quality_evaluated": False,
        "model": {"config_layout": "extracted_text_config"},
        "saes": _sae_audits(),
    }
    path = tmp_path / "compatibility_smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert RUNNER._validate_smoke(
        path,
        manifest=manifest,
        rows=rows,
        config=_config(),
    ) == payload

    changed = {**payload, "source_bundle_sha256": "other"}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the seal"):
        RUNNER._validate_smoke(
            path,
            manifest=manifest,
            rows=rows,
            config=_config(),
        )


def test_score_only_view_uses_one_checked_hook_then_discards_it() -> None:
    class _Monitor:
        def __init__(self) -> None:
            self.layers = None

        def encode_rendered(self, rendered, **kwargs):
            self.layers = kwargs["layers"]
            rows = len(rendered)
            return RUNNER.EncodedMonitorView(
                activations=np.ones((rows, 1, 4), dtype=np.float16),
                lengths=np.full(rows, 3, dtype=np.int32),
                score_log_probabilities=np.zeros((rows, 2), dtype=np.float32),
                expected_scores=np.ones(rows, dtype=np.float32),
            )

    monitor = _Monitor()
    observed = RUNNER._encode_score_only(
        monitor,
        ["a", "b"],
        hook_layer=12,
        hidden_size=4,
        max_length=8,
        batch_size=2,
        score_token_ids=(1, 2),
        score_values=(0.0, 1.0),
    )

    assert monitor.layers == (12,)
    assert observed.activations.shape == (2, 0, 4)
    np.testing.assert_array_equal(observed.expected_scores, [1.0, 1.0])


def test_qwen_scope_standardization_uses_fit_rows_and_counts_negative_topk() -> None:
    rows = [
        _row("fit", 0),
        _row("fit", 1, attack=True),
        _row("calibration", 2),
        _row("locked_test", 3, attack=True),
    ]
    features = np.asarray(
        [
            [-1.0, 0.0, 4.0],
            [1.0, -2.0, 4.0],
            [3.0, -3.0, 8.0],
            [5.0, -4.0, 9.0],
        ],
        dtype=np.float32,
    )

    standardized, audit = RUNNER._standardize_qwen_scope_features(
        features,
        rows,
        minimum_fit_active_rows=2,
    )

    assert standardized.shape == (4, 1)
    np.testing.assert_allclose(standardized[:2, 0], [-1.0, 1.0])
    assert audit == {
        "released_width": 3,
        "passed_fit_activity_filter": 2,
        "retained_fit_nonconstant": 1,
    }


def test_core_result_gate_binds_status_seal_and_row_panel(tmp_path: Path) -> None:
    rows = [_row("fit", 0), _row("fit", 0, attack=True)]
    manifest = {"config_sha256": "config", "source_bundle_sha256": "source"}
    config = _config()
    payload = {
        "schema": RUNNER.RESULT_SCHEMA,
        "status": "complete_core_panel_sae_probe_deferred_post_cache",
        "source_manifest": manifest,
        "n_selected_pairs": 1,
        "config": {"task_id": config["task_id"]},
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert RUNNER._validate_core_results(
        path,
        manifest=manifest,
        rows=rows,
        config=config,
    ) == payload

    path.write_text(
        json.dumps({**payload, "status": "incomplete"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete Qwen3.5 core result"):
        RUNNER._validate_core_results(
            path,
            manifest=manifest,
            rows=rows,
            config=config,
        )


def test_sae_checkpoint_audit_checks_hash_shape_and_exact_topk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {}
    specs = []
    for layer in (12, 20, 28):
        path = tmp_path / f"layer{layer}.sae.pt"
        path.write_bytes(f"layer-{layer}".encode())
        files[layer] = path
        specs.append(
            {
                "layer": layer,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "file_sha256": RUNNER.file_sha256(path),
            }
        )
    config = _config()
    config["measurement"] = {"layers": [12, 20, 28]}
    config["official_sae"].update(
        {
            "compute_dtype": "float32",
            "saes": specs,
        }
    )
    parameters = SimpleNamespace(d_in=4, width=8, top_k=2)
    monkeypatch.setattr(
        RUNNER,
        "_download_sae",
        lambda _config, spec, **_kwargs: files[int(spec["layer"])],
    )
    monkeypatch.setattr(
        RUNNER,
        "load_qwen_scope_parameters",
        lambda _path, *, top_k: SimpleNamespace(
            d_in=parameters.d_in,
            width=parameters.width,
            top_k=top_k,
        ),
    )

    def _encode(_residual, _parameters, **_kwargs):
        features = np.zeros((1, 8), dtype=np.float32)
        features[0, :2] = [2.0, 1.0]
        return features

    monkeypatch.setattr(RUNNER, "encode_qwen_scope_features", _encode)
    neutral = RUNNER.EncodedMonitorView(
        activations=np.zeros((1, 3, 4), dtype=np.float16),
        lengths=np.ones(1, dtype=np.int32),
    )

    audits = RUNNER._audit_sae_checkpoints(
        config,
        neutral,
        device="cpu",
        local_files_only=True,
    )

    assert RUNNER._sae_smoke_audits_pass(audits, config)
    assert [item["active_features_per_row"] for item in audits] == [[2], [2], [2]]


def test_sae_layer_selection_uses_calibration_metrics_not_locked_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row("fit", 0),
        _row("fit", 1, attack=True),
        _row("calibration", 2),
        _row("calibration", 3, attack=True),
        _row("locked_test", 4),
        _row("locked_test", 5, attack=True),
    ]
    config = _config(width=4, top_k=2)
    config.update(
        {
            "measurement": {
                "layers": [12, 20, 28],
                "batch_size": 2,
                "ridge_grid": [0.1, 1.0],
            },
            "observers": {"selection_budget_fraction": 0.5},
        }
    )
    config["official_sae"].update(
        {
            "compute_dtype": "float32",
            "minimum_fit_active_rows": 1,
            "saes": [
                {"layer": layer, "filename": f"layer{layer}.sae.pt"}
                for layer in (12, 20, 28)
            ],
        }
    )
    current_layer = {"value": None}

    def _load(_config, spec, **_kwargs):
        layer = int(spec["layer"])
        current_layer["value"] = layer
        return Path(spec["filename"]), SimpleNamespace(), {
            "layer": layer,
            "filename": spec["filename"],
        }

    def _encode(_residuals, _parameters, **_kwargs):
        layer = int(current_layer["value"])
        features = np.zeros((len(rows), 4), dtype=np.float32)
        features[:, 0] = np.arange(1, len(rows) + 1, dtype=np.float32)
        features[:, 1] = float(layer)
        return features

    losses = iter((0.3, 0.1, 0.2))
    observed_locked_labels = []

    def _select(_features, labels, splits, **_kwargs):
        observed_locked_labels.append(
            tuple(bool(label) for label, split in zip(labels, splits) if split == "locked_test")
        )
        loss = next(losses)
        return {
            "ridge": 0.1,
            "calibration_metrics": {
                "realized_violation_rate": loss,
                "risk_auroc": 0.5,
            },
        }, np.asarray([100.0 + loss, 200.0 + loss])

    monkeypatch.setattr(RUNNER, "_load_checked_sae", _load)
    monkeypatch.setattr(RUNNER, "encode_qwen_scope_features", _encode)
    monkeypatch.setattr(RUNNER, "select_ridge_binary_observer", _select)
    activations = np.zeros((len(rows), 3, 4), dtype=np.float16)

    selection, scores = RUNNER._select_qwen_scope_probe(
        rows,
        activations,
        config,
        device="cpu",
        local_files_only=True,
    )

    assert selection["layer"] == 20
    assert selection["locked_labels_used_for_selection"] is False
    assert selection["refit_rows"] == "fit plus calibration only"
    np.testing.assert_allclose(scores, [100.1, 200.1])
    assert observed_locked_labels == [(False, True)] * 3


def test_primary_contrast_freezes_direction_and_operating_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    name = _config()["interpretation"]["primary_outcome"]["contrast"]

    def _paired(scores, sensitivity, *, contrasts, cell_keys):
        captured.update(
            scores=scores,
            sensitivity=sensitivity,
            contrasts=contrasts,
            cell_keys=cell_keys,
        )
        return {
            "cells": {
                "prevalence=0.01|budget=0.02": {
                    "contrasts": {name: {"paired_difference": {"mean": -1.0}}}
                }
            }
        }

    monkeypatch.setattr(RUNNER, "_score_pools", lambda _panel: {"scores": {}})
    monkeypatch.setattr(RUNNER, "paired_prevalence_contrasts", _paired)

    artifact, result = RUNNER._paired_primary_contrast(
        [object()],
        {"cells": {}},
        _config(),
    )

    assert artifact["cells"]
    assert result["paired_difference"]["mean"] == -1.0
    assert captured["contrasts"] == {
        name: (
            RUNNER.PRIMARY_LEFT_MONITOR,
            RUNNER.PRIMARY_RIGHT_MONITOR,
        )
    }
    assert captured["cell_keys"] == ("prevalence=0.01|budget=0.02",)
