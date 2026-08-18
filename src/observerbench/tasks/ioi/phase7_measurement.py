"""Measure only the frozen non-noop Phase-7 actions.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The runner validates the clean pretest and action freeze before loading a model.
It has no path to score an unselected Phase-7 candidate mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256, runtime_provenance
from observerbench.tasks.ioi.phase5_effects import (
    GPT2_SMALL_REVISION,
    _model_tokens,
    _score_mask_shard,
    validate_effect_rows,
)
from observerbench.tasks.ioi.phase7_confirmation import (
    NOOP_BITS,
    SCIENTIFIC_STATUS,
    load_phase7_protocol,
    load_verified_phase7_design,
    validate_phase7_freeze,
    verify_clean_pretest,
)
from observerbench.tasks.ioi.phase7_freeze_audit import (
    AUDIT_FILENAME,
    validate_phase7_preoutcome_audit,
)


MEASUREMENT_SPEC_SCHEMA = "observerbench.ioi_phase07_selected_measurement_spec.v1"
MEASUREMENT_SCHEMA = "observerbench.ioi_phase07_selected_measurement.v1"
MEASUREMENT_STATUS = "all_frozen_selected_nonnoop_outcomes_measured"


@dataclass(frozen=True)
class Phase7MeasurementConfig:
    """Compute-only settings; scientific choices are already frozen."""

    model_name: str = "gpt2-small"
    model_revision: str = GPT2_SMALL_REVISION
    device: str = "cpu"
    pair_batch_size: int = 128
    mask_shard_size: int = 16
    seed: int = 27071

    def __post_init__(self) -> None:
        if self.model_name != "gpt2-small" or self.model_revision != GPT2_SMALL_REVISION:
            raise ValueError("Phase-7 measurement requires the pinned GPT-2-small revision")
        if self.pair_batch_size <= 0 or self.mask_shard_size != 16:
            raise ValueError("Phase-7 uses positive batches and deterministic 16-mask shards")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _spans(count: int, size: int) -> tuple[tuple[int, int], ...]:
    return tuple((start, min(start + size, count)) for start in range(0, count, size))


def _verify_clean_scores(
    pretest_dir: str | Path,
    prompts: pd.DataFrame,
) -> pd.DataFrame:
    root = Path(pretest_dir)
    manifest = _read_json(root / "pretest_manifest.json")
    clean_path = root / "clean_scores_test.csv"
    if manifest["artifact_hashes"].get("clean_scores_test.csv") != file_sha256(clean_path):
        raise ValueError("Phase-7 clean pretest scores changed")
    clean = pd.read_csv(clean_path, dtype={"prompt_id": str})
    if len(clean) != len(prompts) or clean["prompt_id"].astype(str).duplicated().any():
        raise ValueError("clean pretest scores are not one-to-one with prompts")
    expected = set(prompts["prompt_id"].astype(str))
    if set(clean["prompt_id"].astype(str)) != expected:
        raise ValueError("clean pretest scores use a different prompt bank")
    if not np.isfinite(clean["clean_ld"].to_numpy(float)).all():
        raise ValueError("clean pretest scores contain a non-finite value")
    return clean


def load_phase7_measurement_inputs(
    design_dir: str | Path,
    pretest_dir: str | Path,
    freeze_dir: str | Path,
    audit_dir: str | Path,
    phase5_design_dir: str | Path,
    phase5_effects_dir: str | Path,
    *,
    protocol_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    tuple[str, ...],
]:
    """Validate all seals and expose only selected masks and frozen prompts."""

    validate_phase7_preoutcome_audit(
        audit_dir,
        design_dir=design_dir,
        pretest_dir=pretest_dir,
        freeze_dir=freeze_dir,
        phase5_effects_dir=phase5_effects_dir,
        protocol_path=protocol_path,
    )
    freeze = validate_phase7_freeze(
        freeze_dir,
        design_dir=design_dir,
        pretest_dir=pretest_dir,
        phase5_design_dir=phase5_design_dir,
        phase5_effects_dir=phase5_effects_dir,
        protocol_path=protocol_path,
    )
    verify_clean_pretest(
        pretest_dir, design_dir=design_dir, protocol_path=protocol_path
    )
    design, protocol, prompts, _calibration, _actions = load_verified_phase7_design(
        design_dir, protocol_path
    )
    selected = pd.read_csv(
        Path(freeze_dir) / "selected_measurement_masks.csv",
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    selected["mask_bits"] = selected["mask_bits"].astype(str).str.zfill(13)
    selected["is_noop"] = selected["is_noop"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if selected["is_noop"].isna().any():
        raise ValueError("selected measurement bank has invalid no-op flags")
    if selected.empty or selected["mask_id"].astype(str).duplicated().any():
        raise ValueError("selected measurement bank must contain unique masks")
    if (selected["mask_bits"] == NOOP_BITS).any() or selected["is_noop"].astype(bool).any():
        raise ValueError("analytic no-op entered the ablation measurement bank")
    clean = _verify_clean_scores(pretest_dir, prompts)
    cache_path = Path(phase5_effects_dir) / "template_head_means.npz"
    expected_cache = protocol["pilot_and_source_hashes"][
        "results/revision/phase05/ioi_effects/template_head_means.npz"
    ]
    if file_sha256(cache_path) != expected_cache:
        raise ValueError("Phase-5 template-conditioned mean cache changed")
    with np.load(cache_path, allow_pickle=False) as cache:
        means = np.asarray(cache["means"])
        templates = tuple(map(str, cache["templates"].tolist()))
    expected_templates = tuple(sorted(prompts["template_id"].astype(str).unique()))
    if means.shape != (8, 13, 64) or templates != expected_templates:
        raise ValueError("Phase-5 reference means do not match canonical Phase-7 templates")
    if not np.isfinite(means).all():
        raise ValueError("template-conditioned mean cache contains a non-finite value")
    return freeze, design, prompts, selected, clean, means, templates


def validate_selected_effect_shard(
    rows: pd.DataFrame,
    *,
    prompts: pd.DataFrame,
    masks: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    """Validate one exact prompt-by-selected-mask shard."""

    validate_effect_rows(
        rows,
        prompt_ids=prompts["prompt_id"].astype(str).tolist(),
        mask_ids=masks["mask_id"].astype(str).tolist(),
    )
    if set(rows["split"].astype(str)) != {"test"}:
        raise ValueError("selected effect shard contains a non-test prompt")
    if set(rows["bank"].astype(str)) != {"candidate"}:
        raise ValueError("selected effect shard contains a non-candidate mask")
    observed = rows[["mask_id", "mask_bits", "pool_id"]].drop_duplicates().copy()
    expected = masks[["mask_id", "mask_bits", "pool_id"]].copy()
    for frame in (observed, expected):
        frame["mask_id"] = frame["mask_id"].astype(str)
        frame["mask_bits"] = frame["mask_bits"].astype(str).str.zfill(13)
        frame["pool_id"] = frame["pool_id"].astype(str)
        frame.sort_values("mask_id", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not observed.equals(expected):
        raise ValueError("selected effect shard mask mapping changed")
    clean_by_id = clean.set_index(clean["prompt_id"].astype(str))["clean_ld"]
    expected_clean = rows["prompt_id"].astype(str).map(clean_by_id).to_numpy(float)
    if not np.allclose(rows["clean_ld"].to_numpy(float), expected_clean, atol=1e-6, rtol=0):
        raise ValueError("selected effect shard does not reuse clean pretest scores")


def phase7_measurement_source_hashes(
    design_dir: str | Path,
    pretest_dir: str | Path,
    freeze_dir: str | Path,
    audit_dir: str | Path,
    phase5_effects_dir: str | Path,
    *,
    protocol_path: str | Path,
) -> dict[str, str]:
    """Return the complete v2 source binding used by measurement and evaluation."""

    return {
        "protocol": file_sha256(protocol_path),
        "design_manifest": file_sha256(Path(design_dir) / "design_manifest.json"),
        "pretest_manifest": file_sha256(Path(pretest_dir) / "pretest_manifest.json"),
        "prediction_action_manifest": file_sha256(
            Path(freeze_dir) / "prediction_action_manifest.json"
        ),
        "selected_measurement_masks": file_sha256(
            Path(freeze_dir) / "selected_measurement_masks.csv"
        ),
        "clean_scores_test": file_sha256(Path(pretest_dir) / "clean_scores_test.csv"),
        "template_head_means": file_sha256(
            Path(phase5_effects_dir) / "template_head_means.npz"
        ),
        "preoutcome_audit": file_sha256(Path(audit_dir) / AUDIT_FILENAME),
        "measurement_source": file_sha256(Path(__file__)),
        "evaluation_source": file_sha256(
            Path(__file__).with_name("phase7_evaluation.py")
        ),
    }


def run_phase7_selected_measurement(
    design_dir: str | Path,
    pretest_dir: str | Path,
    freeze_dir: str | Path,
    audit_dir: str | Path,
    phase5_design_dir: str | Path,
    phase5_effects_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    config: Phase7MeasurementConfig,
) -> Path:
    """Measure the frozen union of selected non-noop masks, and nothing else."""

    freeze, design, prompts, selected, clean, means, templates = (
        load_phase7_measurement_inputs(
            design_dir,
            pretest_dir,
            freeze_dir,
            audit_dir,
            phase5_design_dir,
            phase5_effects_dir,
            protocol_path=protocol_path,
        )
    )
    selected = selected.sort_values("mask_id").reset_index(drop=True)
    source_hashes = phase7_measurement_source_hashes(
        design_dir,
        pretest_dir,
        freeze_dir,
        audit_dir,
        phase5_effects_dir,
        protocol_path=protocol_path,
    )
    spans = _spans(len(selected), config.mask_shard_size)
    spec = {
        "schema": MEASUREMENT_SPEC_SCHEMA,
        "status": "sealed_before_candidate_outcomes",
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design["design_id"],
        "config": asdict(config),
        "source_hashes": source_hashes,
        "selected_mask_order_sha256": json_sha256(selected["mask_id"].astype(str).tolist()),
        "counts": {
            "test_prompts": len(prompts),
            "selected_unique_nonnoop_masks": len(selected),
            "effect_cells": len(prompts) * len(selected),
            "mask_shards": len(spans),
            "unselected_candidate_masks_measured": 0,
            "noop_ablation_forward_passes": 0,
        },
        "forbidden_operations": {
            "refit_observers": True,
            "reselect_actions": True,
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
            raise ValueError("Phase-7 measurement resume differs from its sealed spec")
    else:
        write_json(spec_path, spec)

    complete_path = output / "measurement_manifest.json"
    if complete_path.exists():
        manifest = _read_json(complete_path)
        if manifest.get("schema") != MEASUREMENT_SCHEMA or manifest.get("status") != MEASUREMENT_STATUS:
            raise ValueError("Phase-7 measurement manifest is incomplete")
        for label, expected in manifest["artifact_hashes"].items():
            if file_sha256(output / label) != expected:
                raise ValueError(f"completed Phase-7 measurement changed: {label}")
        return output

    completed: list[Path] = []
    completed_masks = 0
    for start, stop in spans:
        path = shard_root / f"effects_{start:04d}_{stop:04d}.csv"
        if not path.exists():
            continue
        masks = selected.iloc[start:stop].reset_index(drop=True)
        rows = pd.read_csv(
            path,
            dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
        )
        validate_selected_effect_shard(
            rows, prompts=prompts, masks=masks, clean=clean
        )
        completed.append(path)
        completed_masks += len(masks)

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
            raise ValueError("measurement IO tokenization differs from the sealed design")
        if not np.array_equal(
            s_tokens, prompts["counterfactual_token_id"].to_numpy(int)
        ):
            raise ValueError("measurement subject tokenization differs from the sealed design")
        template_to_index = {value: index for index, value in enumerate(templates)}
        completed_set = set(completed)
        for start, stop in spans:
            path = shard_root / f"effects_{start:04d}_{stop:04d}.csv"
            if path in completed_set:
                continue
            masks = selected.iloc[start:stop].reset_index(drop=True)
            rows = _score_mask_shard(
                model,
                prompts,
                masks,
                tokens,
                io_tokens,
                s_tokens,
                clean.sort_values("prompt_id")
                .set_index("prompt_id")
                .reindex(prompts["prompt_id"].astype(str))["clean_ld"]
                .to_numpy(float),
                template_to_index,
                means,
                pair_batch_size=config.pair_batch_size,
            )
            validate_selected_effect_shard(
                rows, prompts=prompts, masks=masks, clean=clean
            )
            temporary = path.with_suffix(".tmp")
            rows.to_csv(temporary, index=False)
            temporary.replace(path)
            completed.append(path)
            completed_set.add(path)
            completed_masks += len(masks)
            write_json(
                output / "measurement_progress.json",
                {
                    "status": "partial" if completed_masks < len(selected) else "complete",
                    "completed_selected_masks": completed_masks,
                    "completed_effect_cells": completed_masks * len(prompts),
                    "unselected_candidate_masks_measured": 0,
                    "noop_ablation_forward_passes": 0,
                    "shard_hashes": {
                        item.relative_to(output).as_posix(): file_sha256(item)
                        for item in sorted(completed)
                    },
                },
            )

    completed = sorted(shard_root.glob("effects_*.csv"))
    if completed_masks != len(selected) or len(completed) != len(spans):
        raise ValueError("Phase-7 selected measurement did not complete")
    artifacts = [spec_path, output / "measurement_progress.json", *completed]
    manifest = {
        "schema": MEASUREMENT_SCHEMA,
        "status": MEASUREMENT_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design["design_id"],
        "source_hashes": source_hashes,
        "measurement_spec_sha256": file_sha256(spec_path),
        "counts": spec["counts"],
        "fixed_actions": freeze["counts"]["fixed_actions"],
        "accessed_prompt_splits": ["test"],
        "accessed_mask_bank": "frozen selected non-noop union only",
        "unselected_candidate_masks_measured": 0,
        "noop_ablation_forward_passes": 0,
        "artifact_hashes": {
            path.relative_to(output).as_posix(): file_sha256(path) for path in artifacts
        },
        "runtime": runtime_provenance(),
        "next_allowed_stage": "Apply the frozen pair-cluster by pool evaluation; do not refit.",
    }
    write_json(complete_path, manifest)
    return output
