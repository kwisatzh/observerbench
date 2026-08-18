"""Resumable measurement of the frozen new Phase-8 selected-mask union.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The runner has no path to the 1,440-mask candidate bank.  It validates the
Phase-8 freeze and pre-outcome audit, reuses the sealed Phase-7 inputs, and
loads only ``new_measurement_masks.csv`` before model inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256, runtime_provenance
from observerbench.tasks.ioi.phase5_effects import (
    GPT2_SMALL_REVISION,
    _model_tokens,
    _score_mask_shard,
)
from observerbench.tasks.ioi.phase7_confirmation import NOOP_BITS
from observerbench.tasks.ioi.phase7_measurement import (
    load_phase7_measurement_inputs,
    validate_selected_effect_shard,
)
from observerbench.tasks.ioi.phase8_sensitivity import (
    AUDIT_FILENAME,
    SCIENTIFIC_STATUS,
    Phase8Paths,
    _bool_column,
    load_phase8_protocol,
    validate_phase8_audit,
    validate_phase8_freeze,
    verify_phase8_protocol_sources,
)


MEASUREMENT_SPEC_SCHEMA = "observerbench.ioi_phase08_selected_measurement_spec.v1"
MEASUREMENT_SCHEMA = "observerbench.ioi_phase08_selected_measurement.v1"
MEASUREMENT_STATUS = "all_frozen_new_selected_mask_outcomes_measured"


@dataclass(frozen=True)
class Phase8MeasurementConfig:
    """Compute-only settings; scientific choices are already frozen."""

    model_name: str = "gpt2-small"
    model_revision: str = GPT2_SMALL_REVISION
    device: str = "cpu"
    pair_batch_size: int = 128
    mask_shard_size: int = 16
    seed: int = 28081

    def __post_init__(self) -> None:
        if self.model_name != "gpt2-small" or self.model_revision != GPT2_SMALL_REVISION:
            raise ValueError("Phase-8 measurement requires pinned GPT-2-small")
        if self.pair_batch_size <= 0 or self.mask_shard_size != 16:
            raise ValueError("Phase-8 uses positive batches and 16-mask shards")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _spans(count: int, size: int) -> tuple[tuple[int, int], ...]:
    return tuple((start, min(start + size, count)) for start in range(0, count, size))


def load_phase8_measurement_inputs(
    paths: Phase8Paths,
    freeze_dir: str | Path,
    audit_dir: str | Path,
    *,
    protocol_path: str | Path,
    repository_root: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    tuple[str, ...],
]:
    """Validate every seal and return only prompts and the 148 new masks."""

    protocol = load_phase8_protocol(protocol_path)
    verify_phase8_protocol_sources(protocol, repository_root)
    freeze = validate_phase8_freeze(freeze_dir, protocol_path=protocol_path)
    validate_phase8_audit(
        audit_dir,
        freeze_dir,
        protocol_path=protocol_path,
        repository_root=repository_root,
    )
    (
        _phase7_freeze,
        design,
        prompts,
        inherited,
        clean,
        means,
        templates,
    ) = load_phase7_measurement_inputs(
        paths.phase7_design,
        paths.phase7_pretest,
        paths.phase7_freeze,
        paths.phase7_audit,
        paths.phase5_design,
        paths.phase5_effects,
        protocol_path=paths.phase7_protocol,
    )
    new = pd.read_csv(
        Path(freeze_dir) / "new_measurement_masks.csv",
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    new["mask_bits"] = new["mask_bits"].astype(str).str.zfill(13)
    new["is_noop"] = _bool_column(new, "is_noop")
    if len(new) != 148 or new["mask_id"].astype(str).duplicated().any():
        raise ValueError("Phase-8 new selected-mask bank must contain 148 unique masks")
    if new["is_noop"].any() or (new["mask_bits"] == NOOP_BITS).any():
        raise ValueError("analytic no-op entered the Phase-8 measurement bank")
    if set(new["mask_id"].astype(str)) & set(inherited["mask_id"].astype(str)):
        raise ValueError("Phase-8 new measurement repeats a Phase-7 measured mask")
    expected = freeze["source_bindings"]
    current = {
        "phase7_design_manifest": file_sha256(paths.phase7_design / "design_manifest.json"),
        "phase7_pretest_manifest": file_sha256(paths.phase7_pretest / "pretest_manifest.json"),
        "phase7_freeze_manifest": file_sha256(paths.phase7_freeze / "prediction_action_manifest.json"),
        "phase7_selected_masks": file_sha256(paths.phase7_freeze / "selected_measurement_masks.csv"),
        "phase7_measurement_manifest": file_sha256(paths.phase7_measurement / "measurement_manifest.json"),
        "phase5_design_manifest": file_sha256(paths.phase5_design / "design_manifest.json"),
        "phase5_effect_manifest": file_sha256(paths.phase5_effects / "effect_manifest.json"),
    }
    if expected != current:
        raise ValueError("Phase-8 inherited source binding changed after freeze")
    return freeze, design, prompts, inherited, new, clean, means, templates


def phase8_measurement_source_hashes(
    paths: Phase8Paths,
    freeze_dir: str | Path,
    audit_dir: str | Path,
    *,
    protocol_path: str | Path,
) -> dict[str, str]:
    """Bind measurement and evaluation to all frozen sources."""

    return {
        "protocol": file_sha256(protocol_path),
        "phase8_freeze_manifest": file_sha256(
            Path(freeze_dir) / "prediction_action_manifest.json"
        ),
        "phase8_new_measurement_masks": file_sha256(
            Path(freeze_dir) / "new_measurement_masks.csv"
        ),
        "phase8_preoutcome_audit": file_sha256(Path(audit_dir) / AUDIT_FILENAME),
        "phase7_design_manifest": file_sha256(paths.phase7_design / "design_manifest.json"),
        "phase7_pretest_manifest": file_sha256(paths.phase7_pretest / "pretest_manifest.json"),
        "phase7_freeze_manifest": file_sha256(paths.phase7_freeze / "prediction_action_manifest.json"),
        "phase7_selected_masks": file_sha256(paths.phase7_freeze / "selected_measurement_masks.csv"),
        "phase7_measurement_manifest": file_sha256(paths.phase7_measurement / "measurement_manifest.json"),
        "clean_scores_test": file_sha256(paths.phase7_pretest / "clean_scores_test.csv"),
        "template_head_means": file_sha256(paths.phase5_effects / "template_head_means.npz"),
        "measurement_source": file_sha256(Path(__file__)),
        "evaluation_source": file_sha256(Path(__file__).with_name("phase8_evaluation.py")),
    }


def run_phase8_selected_measurement(
    paths: Phase8Paths,
    freeze_dir: str | Path,
    audit_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    repository_root: str | Path,
    config: Phase8MeasurementConfig,
) -> Path:
    """Measure the frozen 148-mask set difference and nothing else."""

    freeze, design, prompts, _inherited, new, clean, means, templates = (
        load_phase8_measurement_inputs(
            paths,
            freeze_dir,
            audit_dir,
            protocol_path=protocol_path,
            repository_root=repository_root,
        )
    )
    new = new.sort_values("mask_id").reset_index(drop=True)
    sources = phase8_measurement_source_hashes(
        paths, freeze_dir, audit_dir, protocol_path=protocol_path
    )
    spans = _spans(len(new), config.mask_shard_size)
    spec = {
        "schema": MEASUREMENT_SPEC_SCHEMA,
        "status": "sealed_before_new_outcomes",
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design["design_id"],
        "config": asdict(config),
        "source_hashes": sources,
        "selected_mask_order_sha256": json_sha256(new["mask_id"].astype(str).tolist()),
        "counts": {
            "test_prompts": len(prompts),
            "new_selected_unique_nonnoop_masks": len(new),
            "new_effect_cells": len(prompts) * len(new),
            "mask_shards": len(spans),
            "reused_phase7_masks_remeasured": 0,
            "unselected_candidate_masks_measured": 0,
            "noop_ablation_forward_passes": 0,
        },
        "forbidden_operations": {
            "refit_observers": True,
            "reselect_actions": True,
            "measure_inherited_masks_again": True,
            "measure_unselected_masks": True,
            "filter_prompts": True,
        },
    }
    output = Path(outdir)
    shard_root = output / "shards" / "test"
    output.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    spec_path = output / "measurement_run_spec.json"
    if spec_path.exists():
        if _read_json(spec_path) != spec:
            raise ValueError("Phase-8 measurement resume differs from sealed spec")
    else:
        write_json(spec_path, spec)

    complete = output / "measurement_manifest.json"
    if complete.exists():
        manifest = _read_json(complete)
        if manifest.get("schema") != MEASUREMENT_SCHEMA or manifest.get("status") != MEASUREMENT_STATUS:
            raise ValueError("Phase-8 completed measurement manifest is invalid")
        for relative, expected in manifest.get("artifact_hashes", {}).items():
            if file_sha256(output / relative) != expected:
                raise ValueError(f"completed Phase-8 measurement changed: {relative}")
        return output

    completed: dict[tuple[int, int], Path] = {}
    for start, stop in spans:
        path = shard_root / f"effects_{start:04d}_{stop:04d}.csv"
        if not path.exists():
            continue
        masks = new.iloc[start:stop].reset_index(drop=True)
        rows = pd.read_csv(
            path,
            dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
        )
        validate_selected_effect_shard(rows, prompts=prompts, masks=masks, clean=clean)
        completed[(start, stop)] = path

    if len(completed) != len(spans):
        import torch
        from transformer_lens import HookedTransformer

        torch.manual_seed(config.seed)
        model = HookedTransformer.from_pretrained(
            config.model_name,
            device=config.device,
            revision=config.model_revision,
        )
        model.eval()
        if int(model.cfg.n_layers) != 12 or int(model.cfg.n_heads) != 12:
            raise ValueError("loaded model is not GPT-2-small")
        tokens, io_tokens, s_tokens = _model_tokens(model, prompts)
        if not np.array_equal(io_tokens, prompts["answer_token_id"].to_numpy(int)):
            raise ValueError("Phase-8 IO tokenization differs from the sealed design")
        if not np.array_equal(
            s_tokens, prompts["counterfactual_token_id"].to_numpy(int)
        ):
            raise ValueError("Phase-8 subject tokenization differs from the sealed design")
        template_to_index = {value: index for index, value in enumerate(templates)}
        clean_values = (
            clean.sort_values("prompt_id")
            .set_index("prompt_id")
            .reindex(prompts["prompt_id"].astype(str))["clean_ld"]
            .to_numpy(float)
        )
        for start, stop in spans:
            if (start, stop) in completed:
                continue
            path = shard_root / f"effects_{start:04d}_{stop:04d}.csv"
            masks = new.iloc[start:stop].reset_index(drop=True)
            rows = _score_mask_shard(
                model,
                prompts,
                masks,
                tokens,
                io_tokens,
                s_tokens,
                clean_values,
                template_to_index,
                means,
                pair_batch_size=config.pair_batch_size,
            )
            validate_selected_effect_shard(rows, prompts=prompts, masks=masks, clean=clean)
            temporary = path.with_suffix(".tmp")
            rows.to_csv(temporary, index=False)
            temporary.replace(path)
            completed[(start, stop)] = path
            write_json(
                output / "measurement_progress.json",
                {
                    "status": "partial" if len(completed) < len(spans) else "complete",
                    "completed_new_selected_masks": sum(stop - start for start, stop in completed),
                    "completed_new_effect_cells": len(prompts)
                    * sum(stop - start for start, stop in completed),
                    "reused_phase7_masks_remeasured": 0,
                    "unselected_candidate_masks_measured": 0,
                    "noop_ablation_forward_passes": 0,
                    "shard_hashes": {
                        item.relative_to(output).as_posix(): file_sha256(item)
                        for item in sorted(completed.values())
                    },
                },
            )

    if len(completed) != len(spans):
        raise ValueError("Phase-8 selected measurement did not complete")
    progress = output / "measurement_progress.json"
    artifacts = [spec_path, progress, *sorted(completed.values())]
    manifest = {
        "schema": MEASUREMENT_SCHEMA,
        "status": MEASUREMENT_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design["design_id"],
        "source_hashes": sources,
        "measurement_spec_sha256": file_sha256(spec_path),
        "counts": spec["counts"],
        "fixed_actions": freeze["counts"]["fixed_actions"],
        "accessed_prompt_splits": ["test"],
        "accessed_mask_bank": "frozen Phase-8 new selected-mask set difference only",
        "reused_phase7_masks_remeasured": 0,
        "unselected_candidate_masks_measured": 0,
        "noop_ablation_forward_passes": 0,
        "artifact_hashes": {
            path.relative_to(output).as_posix(): file_sha256(path) for path in artifacts
        },
        "runtime": runtime_provenance(),
        "next_allowed_stage": "Combine with hash-verified Phase-7 rows and evaluate all frozen targets and selectors.",
    }
    write_json(complete, manifest)
    return output
