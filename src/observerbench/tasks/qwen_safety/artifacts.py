"""Frozen design and activation-cache artifacts for Qwen safety.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256
from observerbench.tasks.qwen_safety.design import (
    QWEN_SAFETY_DESIGN_SCHEMA,
    QwenSafetyDesign,
    QwenSafetyDesignConfig,
    QwenSafetyPrompt,
)


QWEN_SAFETY_CACHE_SCHEMA = "observerbench.qwen_safety.activation_cache.v0"
QWEN_SAFETY_CACHE_MANIFEST_SCHEMA = "observerbench.qwen_safety.cache_manifest.v0"
PROMPT_COLUMNS = (
    "prompt_id",
    "bank",
    "pair_id",
    "family_id",
    "operation",
    "template",
    "resource",
    "granted_resource",
    "unsafe_if_allowed",
    "severity",
    "benign_value",
    "action_span",
    "user_prompt",
)


def _design_payload(design: QwenSafetyDesign) -> dict[str, Any]:
    return {
        "schema_version": design.schema_version,
        "config": asdict(design.config),
        "resource_banks": {bank: list(resources) for bank, resources in design.resource_banks.items()},
        "prompts": [asdict(prompt) for prompt in design.prompts],
        "design_sha256": design.design_sha256,
    }


def write_qwen_safety_design(design: QwenSafetyDesign, outdir: str | Path) -> tuple[Path, Path]:
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    design_path = output / "qwen_safety_design.json"
    prompt_path = output / "qwen_safety_prompts.csv"
    write_json(design_path, _design_payload(design))
    with prompt_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROMPT_COLUMNS)
        writer.writeheader()
        for prompt in design.prompts:
            row = asdict(prompt)
            writer.writerow({column: row[column] for column in PROMPT_COLUMNS})
    return design_path, prompt_path


def load_qwen_safety_design(path: str | Path) -> QwenSafetyDesign:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != QWEN_SAFETY_DESIGN_SCHEMA:
        raise ValueError("unsupported Qwen safety design schema")
    config = QwenSafetyDesignConfig(**payload["config"])
    prompts = tuple(QwenSafetyPrompt(**row) for row in payload["prompts"])
    resources = {bank: tuple(values) for bank, values in payload["resource_banks"].items()}
    design = QwenSafetyDesign(
        config=config,
        prompts=prompts,
        resource_banks=resources,
        design_sha256=str(payload["design_sha256"]),
    )
    check_payload = {
        "schema_version": design.schema_version,
        "config": asdict(design.config),
        "resource_banks": {key: value for key, value in design.resource_banks.items()},
        "prompts": [asdict(prompt) for prompt in design.prompts],
    }
    if json_sha256(check_payload) != design.design_sha256:
        raise ValueError("Qwen safety design hash mismatch")
    return design


def write_activation_cache(
    path: str | Path,
    *,
    prompt_ids: Sequence[str],
    layer_indices: Sequence[int],
    activations: np.ndarray,
    candidate_margins: Sequence[float],
    block_minus_allow_margins: Sequence[float] | None = None,
    candidate_correct: Sequence[bool],
    top1_correct: Sequence[bool],
    sequence_lengths: Sequence[int],
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    activations = np.asarray(activations, dtype=np.float16)
    n_prompts = len(prompt_ids)
    layers = tuple(map(int, layer_indices))
    if activations.ndim != 3 or activations.shape[:2] != (n_prompts, len(layers)):
        raise ValueError("activation cache must have prompt-by-layer-by-hidden shape")
    if len(set(prompt_ids)) != n_prompts:
        raise ValueError("activation cache prompt IDs must be unique")
    arrays = {
        "prompt_ids": np.asarray(prompt_ids),
        "layer_indices": np.asarray(layers, dtype=np.int64),
        "activations": activations,
        "candidate_margins": np.asarray(candidate_margins, dtype=np.float32),
        "candidate_correct": np.asarray(candidate_correct, dtype=np.bool_),
        "top1_correct": np.asarray(top1_correct, dtype=np.bool_),
        "sequence_lengths": np.asarray(sequence_lengths, dtype=np.int64),
    }
    if block_minus_allow_margins is not None:
        arrays["block_minus_allow_margins"] = np.asarray(
            block_minus_allow_margins, dtype=np.float32
        )
    for name, values in arrays.items():
        if name in {"layer_indices", "activations"}:
            continue
        if len(values) != n_prompts:
            raise ValueError(f"{name} length differs from prompt count")
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    manifest_path = output.with_suffix(".manifest.json")
    write_json(
        manifest_path,
        {
            "schema": QWEN_SAFETY_CACHE_MANIFEST_SCHEMA,
            "cache_schema": QWEN_SAFETY_CACHE_SCHEMA,
            "cache_file": output.name,
            "cache_sha256": file_sha256(output),
            "n_prompts": n_prompts,
            "layer_indices": layers,
            "hidden_size": int(activations.shape[2]),
            "metadata": dict(metadata),
        },
    )
    return output, manifest_path


def load_activation_cache(path: str | Path, *, verify_hash: bool = True) -> dict[str, Any]:
    source = Path(path)
    manifest_path = source.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != QWEN_SAFETY_CACHE_MANIFEST_SCHEMA:
        raise ValueError("unsupported Qwen safety cache manifest")
    if verify_hash and file_sha256(source) != manifest.get("cache_sha256"):
        raise ValueError("Qwen safety activation-cache hash mismatch")
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    if list(map(int, payload["layer_indices"])) != list(map(int, manifest["layer_indices"])):
        raise ValueError("cache layers differ from manifest")
    if payload["activations"].shape != (
        int(manifest["n_prompts"]),
        len(manifest["layer_indices"]),
        int(manifest["hidden_size"]),
    ):
        raise ValueError("activation-cache shape differs from manifest")
    payload["manifest"] = manifest
    return payload


__all__ = [
    "PROMPT_COLUMNS",
    "QWEN_SAFETY_CACHE_MANIFEST_SCHEMA",
    "QWEN_SAFETY_CACHE_SCHEMA",
    "load_activation_cache",
    "load_qwen_safety_design",
    "write_activation_cache",
    "write_qwen_safety_design",
]
