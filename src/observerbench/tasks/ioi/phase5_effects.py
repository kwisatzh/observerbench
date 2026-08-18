"""Packed, resumable GPT-2 IOI finite-effect measurement.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance, source_hashes
from observerbench.tasks.ioi.heads import head_records
from observerbench.tasks.ioi.stage2d import parse_mask_bits


EFFECT_SCHEMA_VERSION = "observerbench.ioi_effect_rows.v1"
GPT2_SMALL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


@dataclass(frozen=True)
class IOIPhase5EffectConfig:
    model_name: str = "gpt2-small"
    model_revision: str = GPT2_SMALL_REVISION
    device: str = "cpu"
    pair_batch_size: int = 128
    reference_batch_size: int = 64
    mask_shard_size: int = 16
    reference_split: str = "reference"
    outcome_splits: tuple[str, ...] = ("train", "validation", "test")
    max_reference_prompts_per_template: int | None = None
    max_outcome_prompts_per_split: int | None = None
    max_masks: int | None = None
    seed: int = 25051

    def __post_init__(self) -> None:
        if not self.model_revision:
            raise ValueError("model_revision must be non-empty")
        if self.pair_batch_size <= 0 or self.reference_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.mask_shard_size <= 0:
            raise ValueError("mask_shard_size must be positive")
        if not self.outcome_splits:
            raise ValueError("outcome_splits must be non-empty")
        if self.reference_split in self.outcome_splits:
            raise ValueError("reference prompts cannot also be outcome prompts")
        for name, value in (
            ("max_reference_prompts_per_template", self.max_reference_prompts_per_template),
            ("max_outcome_prompts_per_split", self.max_outcome_prompts_per_split),
            ("max_masks", self.max_masks),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when supplied")


def _required_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_locked_ioi_design(design_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load a frozen Phase-5 prompt/mask design and verify its source hashes."""

    root = Path(design_dir)
    prompts_path = root / "prompts.csv"
    masks_path = root / "masks.csv"
    manifest_path = root / "design_manifest.json"
    if not prompts_path.exists() or not masks_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("design_dir must contain prompts.csv, masks.csv, and design_manifest.json")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"frozen_before_outcomes", "frozen"}:
        raise ValueError("IOI effects require a design frozen before model outcomes")
    expected_hashes = manifest.get("source_hashes", manifest.get("hashes", {}))
    for path in (prompts_path, masks_path):
        expected = expected_hashes.get(path.name)
        if isinstance(expected, Mapping):
            expected = expected.get("sha256")
        if expected is not None and expected != file_sha256(path):
            raise ValueError(f"locked design hash mismatch for {path.name}")

    prompts = pd.read_csv(prompts_path, dtype={"prompt_id": str, "template_id": str})
    masks = pd.read_csv(masks_path, dtype={"mask_id": str, "mask_bits": str, "pool_id": str})
    _required_columns(
        prompts,
        ("prompt_id", "split", "template_id", "structure", "io_name", "s_name", "prompt"),
        name="prompts.csv",
    )
    _required_columns(
        masks,
        ("mask_id", "mask_bits", "bank", "pool_id", "n_heads", "n_P", "n_B", "n_E"),
        name="masks.csv",
    )
    if prompts["prompt_id"].duplicated().any():
        raise ValueError("prompt_id must be unique")
    if prompts["prompt"].duplicated().any():
        raise ValueError("prompt text must be unique")
    if masks["mask_id"].duplicated().any() or masks["mask_bits"].duplicated().any():
        raise ValueError("mask ids and bit strings must be unique")
    if not masks["mask_bits"].str.fullmatch(r"[01]{13}").all():
        raise ValueError("every mask_bits value must contain exactly 13 binary digits")
    return prompts, masks, manifest


def validate_effect_rows(
    rows: pd.DataFrame,
    *,
    prompt_ids: Sequence[str],
    mask_ids: Sequence[str],
    zero_tolerance: float = 1e-6,
) -> None:
    """Fail on missing, duplicate, non-finite, or inconsistent effect cells."""

    _required_columns(
        rows,
        ("prompt_id", "mask_id", "clean_ld", "ablated_ld", "drop_from_clean", "mask_bits"),
        name="effect rows",
    )
    expected = len(prompt_ids) * len(mask_ids)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} effect cells, found {len(rows)}")
    if rows.duplicated(["prompt_id", "mask_id"]).any():
        raise ValueError("effect rows contain duplicate prompt-mask keys")
    if set(rows["prompt_id"].astype(str)) != set(map(str, prompt_ids)):
        raise ValueError("effect rows do not contain the expected prompts")
    if set(rows["mask_id"].astype(str)) != set(map(str, mask_ids)):
        raise ValueError("effect rows do not contain the expected masks")
    values = rows[["clean_ld", "ablated_ld", "drop_from_clean"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("effect scores must be finite")
    residual = rows["clean_ld"] - rows["ablated_ld"] - rows["drop_from_clean"]
    if float(np.abs(residual).max()) > zero_tolerance:
        raise ValueError("drop_from_clean has an inconsistent sign or value")
    clean_rows = rows[rows["mask_bits"].astype(str) == "0" * 13]
    if len(clean_rows) and float(clean_rows["drop_from_clean"].abs().max()) > zero_tolerance:
        raise ValueError("the clean mask must have zero finite effect")


def _model_tokens(model: Any, prompts: pd.DataFrame) -> tuple[list[Any], np.ndarray, np.ndarray]:
    token_rows = []
    io_tokens = []
    s_tokens = []
    for row in prompts.itertuples(index=False):
        tokens = model.to_tokens(str(row.prompt), prepend_bos=True)[0].detach().cpu()
        token_rows.append(tokens)
        io_tokens.append(int(model.to_single_token(" " + str(row.io_name))))
        s_tokens.append(int(model.to_single_token(" " + str(row.s_name))))
    return token_rows, np.asarray(io_tokens, dtype=int), np.asarray(s_tokens, dtype=int)


def _batched(indices: Sequence[int], size: int) -> Iterable[np.ndarray]:
    values = np.asarray(indices, dtype=int)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _length_buckets(token_rows: Sequence[Any]) -> list[np.ndarray]:
    lengths = np.asarray([len(row) for row in token_rows], dtype=int)
    return [np.flatnonzero(lengths == length) for length in sorted(set(lengths))]


def _logit_difference(logits: Any, io_token: Any, s_token: Any) -> Any:
    import torch

    batch = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch, -1, io_token] - logits[batch, -1, s_token]


def _hook_layout() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    by_layer: dict[int, list[tuple[int, int]]] = {}
    for canonical, record in enumerate(head_records()):
        by_layer.setdefault(int(record["layer"]), []).append(
            (canonical, int(record["head"]))
        )
    return {
        layer: (
            np.asarray([item[0] for item in items], dtype=int),
            np.asarray([item[1] for item in items], dtype=int),
        )
        for layer, items in by_layer.items()
    }


def measure_template_head_means(
    model: Any,
    reference_prompts: pd.DataFrame,
    token_rows: Sequence[Any],
    *,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Measure final-position head-z means for each frozen template."""

    import torch

    templates = sorted(reference_prompts["template_id"].astype(str).unique())
    template_to_index = {name: idx for idx, name in enumerate(templates)}
    layout = _hook_layout()
    hook_names = [f"blocks.{layer}.attn.hook_z" for layer in layout]
    sums = np.zeros((len(templates), 13, model.cfg.d_head), dtype=np.float64)
    counts = np.zeros(len(templates), dtype=int)

    for template, frame in reference_prompts.groupby("template_id", sort=True):
        local_indices = frame.index.to_numpy(int)
        template_index = template_to_index[str(template)]
        local_tokens = [token_rows[index] for index in local_indices]
        for bucket in _length_buckets(local_tokens):
            prompt_indices = local_indices[bucket]
            for batch_indices in _batched(prompt_indices, batch_size):
                tokens = torch.stack([token_rows[index] for index in batch_indices]).to(model.cfg.device)
                with torch.inference_mode():
                    _logits, cache = model.run_with_cache(
                        tokens,
                        names_filter=lambda name: name in hook_names,
                    )
                for layer, (canonical, heads) in layout.items():
                    values = cache[f"blocks.{layer}.attn.hook_z"][:, -1, heads, :]
                    sums[template_index, canonical, :] += values.detach().cpu().numpy().sum(axis=0)
                counts[template_index] += len(batch_indices)
    if np.any(counts == 0):
        raise ValueError("every template must have at least one reference prompt")
    return (sums / counts[:, None, None]).astype(np.float32), templates


def _make_ablation_hooks(
    *,
    torch_module: Any,
    pair_masks: Any,
    pair_templates: Any,
    template_head_means: Any,
) -> list[tuple[str, Any]]:
    hooks = []
    for layer, (canonical_np, heads_np) in _hook_layout().items():
        canonical = torch_module.as_tensor(canonical_np, device=pair_masks.device)
        heads = torch_module.as_tensor(heads_np, device=pair_masks.device)

        def replace(z: Any, hook: Any, *, canonical=canonical, heads=heads) -> Any:
            del hook
            output = z.clone()
            selected = pair_masks.index_select(1, canonical).unsqueeze(-1)
            means = template_head_means.index_select(0, pair_templates).index_select(1, canonical)
            current = output[:, -1, heads, :]
            output[:, -1, heads, :] = torch_module.where(selected, means, current)
            return output

        hooks.append((f"blocks.{layer}.attn.hook_z", replace))
    return hooks


def _score_clean(
    model: Any,
    prompts: pd.DataFrame,
    token_rows: Sequence[Any],
    io_tokens: np.ndarray,
    s_tokens: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    import torch

    scores = np.full(len(prompts), np.nan, dtype=np.float32)
    for bucket in _length_buckets(token_rows):
        for batch_indices in _batched(bucket, batch_size):
            tokens = torch.stack([token_rows[index] for index in batch_indices]).to(model.cfg.device)
            io = torch.as_tensor(io_tokens[batch_indices], device=model.cfg.device)
            subject = torch.as_tensor(s_tokens[batch_indices], device=model.cfg.device)
            with torch.inference_mode():
                logits = model(tokens, return_type="logits")
            scores[batch_indices] = _logit_difference(logits, io, subject).detach().cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("clean scores contain non-finite values")
    return scores


def _score_mask_shard(
    model: Any,
    prompts: pd.DataFrame,
    masks: pd.DataFrame,
    token_rows: Sequence[Any],
    io_tokens: np.ndarray,
    s_tokens: np.ndarray,
    clean_scores: np.ndarray,
    template_to_index: Mapping[str, int],
    template_head_means: np.ndarray,
    *,
    pair_batch_size: int,
) -> pd.DataFrame:
    import torch

    mask_matrix = np.stack([parse_mask_bits(bits, 13) for bits in masks["mask_bits"]]).astype(bool)
    template_rows = prompts["template_id"].astype(str).map(template_to_index).to_numpy()
    if np.any(pd.isna(template_rows)):
        raise ValueError("outcome prompt template lacks a reference mean")
    mean_tensor = torch.as_tensor(template_head_means, device=model.cfg.device)
    records: list[dict[str, object]] = []
    prompt_buckets = _length_buckets(token_rows)

    for bucket in prompt_buckets:
        pairs = np.asarray(
            [(prompt_idx, mask_idx) for prompt_idx in bucket for mask_idx in range(len(masks))],
            dtype=int,
        )
        for batch in _batched(np.arange(len(pairs)), pair_batch_size):
            selected_pairs = pairs[batch]
            prompt_indices = selected_pairs[:, 0]
            mask_indices = selected_pairs[:, 1]
            tokens = torch.stack([token_rows[index] for index in prompt_indices]).to(model.cfg.device)
            io = torch.as_tensor(io_tokens[prompt_indices], device=model.cfg.device)
            subject = torch.as_tensor(s_tokens[prompt_indices], device=model.cfg.device)
            pair_masks = torch.as_tensor(mask_matrix[mask_indices], device=model.cfg.device)
            pair_templates = torch.as_tensor(template_rows[prompt_indices], device=model.cfg.device)
            hooks = _make_ablation_hooks(
                torch_module=torch,
                pair_masks=pair_masks,
                pair_templates=pair_templates,
                template_head_means=mean_tensor,
            )
            with torch.inference_mode():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
            ablated = _logit_difference(logits, io, subject).detach().cpu().numpy()
            for local, (prompt_idx, mask_idx) in enumerate(selected_pairs):
                prompt = prompts.iloc[int(prompt_idx)]
                mask = masks.iloc[int(mask_idx)]
                clean = float(clean_scores[int(prompt_idx)])
                # The all-zero row is the identity intervention. Reuse the
                # separately measured clean score instead of routing it through
                # no-op hooks, which can trigger backend-dependent roundoff.
                score = (
                    clean
                    if str(mask.mask_bits) == "0" * 13
                    else float(ablated[local])
                )
                records.append(
                    {
                        "schema_version": EFFECT_SCHEMA_VERSION,
                        "prompt_id": str(prompt.prompt_id),
                        "split": str(prompt.split),
                        "template_id": str(prompt.template_id),
                        "structure": str(prompt.structure),
                        "mask_id": str(mask.mask_id),
                        "mask_bits": str(mask.mask_bits),
                        "bank": str(mask.bank),
                        "pool_id": str(mask.pool_id),
                        "clean_ld": clean,
                        "ablated_ld": score,
                        "drop_from_clean": clean - score,
                    }
                )
    return pd.DataFrame(records)


def run_ioi_phase5_effects(
    design_dir: str | Path,
    outdir: str | Path,
    *,
    config: IOIPhase5EffectConfig,
) -> Path:
    """Measure all frozen calibration/candidate masks with packed hooks."""

    import json
    import torch
    from transformer_lens import HookedTransformer

    prompts, masks, design_manifest = load_locked_ioi_design(design_dir)
    reference = prompts[prompts["split"] == config.reference_split].copy().reset_index(drop=True)
    outcome = prompts[prompts["split"].isin(config.outcome_splits)].copy().reset_index(drop=True)
    smoke_limits = any(
        value is not None
        for value in (
            config.max_reference_prompts_per_template,
            config.max_outcome_prompts_per_split,
            config.max_masks,
        )
    )
    if config.max_reference_prompts_per_template is not None:
        reference = (
            reference.groupby("template_id", sort=True, group_keys=False)
            .head(config.max_reference_prompts_per_template)
            .reset_index(drop=True)
        )
    if config.max_outcome_prompts_per_split is not None:
        outcome = (
            outcome.groupby("split", sort=True, group_keys=False)
            .head(config.max_outcome_prompts_per_split)
            .reset_index(drop=True)
        )
    if config.max_masks is not None:
        masks = masks.head(config.max_masks).reset_index(drop=True)
    if reference.empty or outcome.empty:
        raise ValueError("the locked design must contain reference and outcome prompts")
    if set(reference["io_name"]) & set(outcome["io_name"]):
        raise ValueError("reference and outcome IO names must be disjoint")

    output = Path(outdir)
    shards = output / "shards"
    output.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    for split in config.outcome_splits:
        (shards / split).mkdir(parents=True, exist_ok=True)
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
    cache_tmp = output / "template_head_means.tmp"
    with cache_tmp.open("wb") as handle:
        np.savez_compressed(handle, means=means, templates=np.asarray(templates))
    cache_tmp.replace(cache_path)

    token_rows, io_tokens, s_tokens = _model_tokens(model, outcome)
    clean_scores = _score_clean(
        model,
        outcome,
        token_rows,
        io_tokens,
        s_tokens,
        batch_size=config.pair_batch_size,
    )
    clean_rows = outcome[["prompt_id", "split", "template_id", "structure"]].copy()
    clean_rows["clean_ld"] = clean_scores
    clean_paths: list[Path] = []
    for split in config.outcome_splits:
        clean_path = output / f"clean_scores_{split}.csv"
        clean_rows[clean_rows["split"] == split].to_csv(clean_path, index=False)
        clean_paths.append(clean_path)

    template_to_index = {name: idx for idx, name in enumerate(templates)}
    shard_paths: list[Path] = []
    for start in range(0, len(masks), config.mask_shard_size):
        stop = min(start + config.mask_shard_size, len(masks))
        paths = {
            split: shards / split / f"effects_{start:04d}_{stop:04d}.csv"
            for split in config.outcome_splits
        }
        shard_masks = masks.iloc[start:stop].reset_index(drop=True)
        if all(path.exists() for path in paths.values()):
            for split, path in paths.items():
                existing = pd.read_csv(
                    path,
                    dtype={"prompt_id": str, "mask_id": str, "mask_bits": str},
                )
                validate_effect_rows(
                    existing,
                    prompt_ids=outcome.loc[
                        outcome["split"] == split, "prompt_id"
                    ].astype(str).tolist(),
                    mask_ids=shard_masks["mask_id"].astype(str).tolist(),
                )
                shard_paths.append(path)
            continue
        rows = _score_mask_shard(
            model,
            outcome,
            shard_masks,
            token_rows,
            io_tokens,
            s_tokens,
            clean_scores,
            template_to_index,
            means,
            pair_batch_size=config.pair_batch_size,
        )
        for split, path in paths.items():
            split_rows = rows[rows["split"] == split].copy()
            validate_effect_rows(
                split_rows,
                prompt_ids=outcome.loc[
                    outcome["split"] == split, "prompt_id"
                ].astype(str).tolist(),
                mask_ids=shard_masks["mask_id"].astype(str).tolist(),
            )
            temporary = path.with_suffix(".tmp")
            split_rows.to_csv(temporary, index=False)
            temporary.replace(path)
            shard_paths.append(path)

    manifest = {
        "schema": "observerbench.ioi_effect_run.v1",
        "status": (
            "complete_smoke_not_for_claims"
            if smoke_limits
            else "complete_unopened_confirmatory_outcomes"
        ),
        "config": asdict(config),
        "design_manifest_sha256": file_sha256(Path(design_dir) / "design_manifest.json"),
        "design_schema": design_manifest.get("schema"),
        "counts": {
            "reference_prompts": int(len(reference)),
            "outcome_prompts": int(len(outcome)),
            "masks": int(len(masks)),
            "effect_cells": int(len(outcome) * len(masks)),
            "shards": int(len(shard_paths)),
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
            [path.resolve() for path in (cache_path, *clean_paths, *shard_paths)],
            output.resolve(),
        ),
        "runtime": runtime_provenance(),
        "analysis_guard": (
            "Fit artifacts must be frozen from train-prompt calibration rows before "
            "validation or test candidate effects are loaded."
        ),
    }
    write_json(output / "effect_manifest.json", manifest)
    return output
