"""Calibration-only measurement stage for the prospective Phase-6 IOI study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This stage can read reference and train prompts and the inherited 160-mask
calibration bank.  It has no code path that evaluates test prompts or candidate
masks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance, source_hashes
from observerbench.tasks.ioi.phase5_effects import (
    GPT2_SMALL_REVISION,
    _model_tokens,
    _score_clean,
    _score_mask_shard,
    measure_template_head_means,
    validate_effect_rows,
)
from observerbench.tasks.ioi.phase6_confirmatory import DESIGN_SCHEMA, DESIGN_STATUS


CALIBRATION_SCHEMA = "observerbench.ioi_phase06_calibration_run.v1"
CALIBRATION_STATUS = "phase6_train_calibration_complete_test_unopened"


@dataclass(frozen=True)
class Phase6CalibrationConfig:
    model_name: str = "gpt2-small"
    model_revision: str = GPT2_SMALL_REVISION
    device: str = "cpu"
    pair_batch_size: int = 128
    reference_batch_size: int = 64
    mask_shard_size: int = 16
    seed: int = 26062

    def __post_init__(self) -> None:
        if self.model_revision != GPT2_SMALL_REVISION:
            raise ValueError("Phase 6 requires the pinned GPT-2-small revision")
        if min(self.pair_batch_size, self.reference_batch_size, self.mask_shard_size) <= 0:
            raise ValueError("all batch and shard sizes must be positive")


def _load_sealed_design(
    design_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(design_dir)
    manifest_path = root / "design_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DESIGN_SCHEMA or manifest.get("status") != DESIGN_STATUS:
        raise ValueError("Phase-6 design is not in its outcome-free sealed state")
    if manifest.get("contains_model_outcomes") is not False:
        raise ValueError("design manifest unexpectedly contains outcomes")
    if manifest.get("phase6_forward_passes_performed") is not False:
        raise ValueError("design manifest says a Phase-6 forward pass already occurred")
    for filename, expected in manifest["artifact_hashes"].items():
        if file_sha256(root / filename) != expected:
            raise ValueError(f"sealed design artifact changed: {filename}")
    prompts = pd.read_csv(root / "prompts.csv", dtype={"prompt_id": str})
    calibration = pd.read_csv(
        root / "calibration_masks.csv",
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    if set(prompts["split"]) != {"reference", "train", "test"}:
        raise ValueError("unexpected Phase-6 prompt split")
    if len(calibration) != 160 or set(calibration["bank"]) != {"calibration"}:
        raise ValueError("calibration stage requires exactly the inherited 160 masks")
    if calibration["pool_id"].fillna("").astype(str).ne("").any():
        raise ValueError("calibration masks cannot belong to candidate pools")
    reference = prompts[prompts["split"] == "reference"].reset_index(drop=True)
    train = prompts[prompts["split"] == "train"].reset_index(drop=True)
    if set(reference["split"]) != {"reference"} or set(train["split"]) != {"train"}:
        raise ValueError("calibration loader exposed an unexpected prompt split")
    calibration["pool_id"] = calibration["pool_id"].fillna("").astype(str)
    return reference, train, calibration, manifest


def run_phase6_calibration(
    design_dir: str | Path,
    outdir: str | Path,
    *,
    config: Phase6CalibrationConfig,
) -> Path:
    """Measure reference means and train/calibration effects only."""

    import torch
    from transformer_lens import HookedTransformer

    reference, train, calibration, design_manifest = _load_sealed_design(design_dir)
    if len(reference) != 512 or len(train) != 192:
        raise ValueError("Phase-6 reference/train prompt counts changed")
    if set(reference["io_name"]) & set(train["io_name"]):
        raise ValueError("reference and train name banks overlap")

    output = Path(outdir)
    shard_root = output / "shards" / "train"
    output.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=config.device,
        revision=config.model_revision,
    )
    model.eval()

    reference_tokens, _reference_io, _reference_s = _model_tokens(model, reference)
    means, templates = measure_template_head_means(
        model,
        reference,
        reference_tokens,
        batch_size=config.reference_batch_size,
    )
    cache_path = output / "template_head_means.npz"
    temporary_cache = output / "template_head_means.tmp"
    with temporary_cache.open("wb") as handle:
        np.savez_compressed(handle, means=means, templates=np.asarray(templates))
    temporary_cache.replace(cache_path)

    tokens, io_tokens, s_tokens = _model_tokens(model, train)
    clean_scores = _score_clean(
        model,
        train,
        tokens,
        io_tokens,
        s_tokens,
        batch_size=config.pair_batch_size,
    )
    clean_rows = train[
        ["prompt_id", "split", "template_id", "structure", "unordered_name_pair_id"]
    ].copy()
    clean_rows["clean_ld"] = clean_scores
    clean_path = output / "clean_scores_train.csv"
    clean_rows.to_csv(clean_path, index=False)

    template_to_index = {name: index for index, name in enumerate(templates)}
    shard_paths: list[Path] = []
    for start in range(0, len(calibration), config.mask_shard_size):
        stop = min(start + config.mask_shard_size, len(calibration))
        path = shard_root / f"effects_{start:04d}_{stop:04d}.csv"
        shard_masks = calibration.iloc[start:stop].reset_index(drop=True)
        if path.exists():
            existing = pd.read_csv(
                path,
                dtype={"prompt_id": str, "mask_id": str, "mask_bits": str},
            )
            validate_effect_rows(
                existing,
                prompt_ids=train["prompt_id"].astype(str).tolist(),
                mask_ids=shard_masks["mask_id"].astype(str).tolist(),
            )
            shard_paths.append(path)
            continue
        rows = _score_mask_shard(
            model,
            train,
            shard_masks,
            tokens,
            io_tokens,
            s_tokens,
            clean_scores,
            template_to_index,
            means,
            pair_batch_size=config.pair_batch_size,
        )
        validate_effect_rows(
            rows,
            prompt_ids=train["prompt_id"].astype(str).tolist(),
            mask_ids=shard_masks["mask_id"].astype(str).tolist(),
        )
        temporary = path.with_suffix(".tmp")
        rows.to_csv(temporary, index=False)
        temporary.replace(path)
        shard_paths.append(path)

    expected_cells = len(train) * len(calibration)
    manifest = {
        "schema": CALIBRATION_SCHEMA,
        "status": CALIBRATION_STATUS,
        "config": asdict(config),
        "design_manifest_sha256": file_sha256(
            Path(design_dir) / "design_manifest.json"
        ),
        "design_id": design_manifest["design_id"],
        "accessed_prompt_splits": ["reference", "train"],
        "accessed_mask_banks": ["calibration"],
        "test_prompt_forward_passes": 0,
        "candidate_mask_forward_passes": 0,
        "counts": {
            "reference_prompts": len(reference),
            "train_prompts": len(train),
            "calibration_masks": len(calibration),
            "train_calibration_effect_cells": expected_cells,
            "shards": len(shard_paths),
        },
        "model": {
            "requested_name": config.model_name,
            "requested_revision": config.model_revision,
            "resolved_name": model.cfg.model_name,
            "n_layers": int(model.cfg.n_layers),
            "n_heads": int(model.cfg.n_heads),
            "d_head": int(model.cfg.d_head),
            "dtype": str(model.cfg.dtype),
            "device": str(model.cfg.device),
        },
        "artifacts": source_hashes(
            [cache_path.resolve(), clean_path.resolve(), *(path.resolve() for path in shard_paths)],
            output.resolve(),
        ),
        "runtime": runtime_provenance(),
        "next_allowed_stage": (
            "Fit frozen observers and hash predictions and fixed actions. No test-prompt "
            "or candidate-mask forward pass is allowed before that seal exists."
        ),
    }
    write_json(output / "calibration_manifest.json", manifest)
    return output
