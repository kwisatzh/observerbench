"""Frozen prompt and mask design for the confirmatory Phase 5 IOI study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from itertools import permutations
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.provenance import file_sha256, json_sha256
from observerbench.tasks.ioi.heads import head_records
from observerbench.tasks.ioi.phase2_capacity import (
    LoadedIOIRun,
    build_capacity_design,
)


N_HEADS = 13
PROMPT_SPLITS: tuple[str, ...] = ("reference", "train", "validation", "test")
DESIGN_MODELS: tuple[str, ...] = (
    "additive_head",
    "count_additive",
    "count_plus_PE_bin4",
    "count_plus_all_bin4",
)


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    structure: str
    text: str


# Keep these literal. Changing wording, order, or identifiers creates a new
# confirmatory protocol version.
PHASE5_TEMPLATES: tuple[PromptTemplate, ...] = (
    PromptTemplate(
        "abba_store",
        "ABBA",
        "When {io} and {s} went to the store, {s} gave a bottle of milk to",
    ),
    PromptTemplate(
        "abba_park",
        "ABBA",
        "After {io} and {s} visited the park, {s} handed a folded map to",
    ),
    PromptTemplate(
        "abba_office",
        "ABBA",
        "While {io} and {s} worked in the office, {s} passed a blue folder to",
    ),
    PromptTemplate(
        "abba_meeting",
        "ABBA",
        "Because {io} and {s} attended the meeting, {s} sent a short note to",
    ),
    PromptTemplate(
        "baba_store",
        "BABA",
        "When {s} and {io} went to the store, {s} gave a bottle of milk to",
    ),
    PromptTemplate(
        "baba_park",
        "BABA",
        "After {s} and {io} visited the park, {s} handed a folded map to",
    ),
    PromptTemplate(
        "baba_office",
        "BABA",
        "While {s} and {io} worked in the office, {s} passed a blue folder to",
    ),
    PromptTemplate(
        "baba_meeting",
        "BABA",
        "Because {s} and {io} attended the meeting, {s} sent a short note to",
    ),
)


@dataclass(frozen=True)
class Phase5Design:
    templates: pd.DataFrame
    names: pd.DataFrame
    prompts: pd.DataFrame
    calibration_masks: pd.DataFrame
    candidate_masks: pd.DataFrame
    rank_diagnostics: pd.DataFrame
    leakage_audit: Mapping[str, object]
    protocol: Mapping[str, object]


def _stable_key(seed: int, namespace: str, value: str) -> str:
    return json_sha256({"seed": seed, "namespace": namespace, "value": value})


def _stable_id(namespace: str, payload: object) -> str:
    return f"{namespace}_{json_sha256(payload)[:16]}"


def _validate_mask_bits(bits: str, *, allow_clean: bool = False) -> str:
    value = str(bits).strip()
    if len(value) != N_HEADS or set(value) - {"0", "1"}:
        raise ValueError(f"invalid {N_HEADS}-bit mask: {bits!r}")
    if not allow_clean and "1" not in value:
        raise ValueError("clean mask is not an intervention mask")
    return value


def load_legacy_mask_bits(paths: Iterable[str | Path]) -> set[str]:
    """Load mask bits from legacy subset CSVs for a strict leakage gate."""

    masks: set[str] = set()
    for path_like in paths:
        path = Path(path_like)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "mask_bits" not in reader.fieldnames:
                raise ValueError(f"{path} has no mask_bits column")
            for row in reader:
                masks.add(_validate_mask_bits(row["mask_bits"], allow_clean=True))
    return masks


def deterministic_name_split(
    names: Sequence[str],
    *,
    seed: int,
    split_counts: Mapping[str, int],
) -> dict[str, tuple[str, ...]]:
    """Split a frozen name bank without depending on input order or RNG state."""

    cleaned = [str(name).strip() for name in names]
    if any(not name or " " in name for name in cleaned):
        raise ValueError("names must be non-empty single words")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("name candidates must be unique")
    if tuple(split_counts) != PROMPT_SPLITS:
        raise ValueError(f"name splits must be ordered as {PROMPT_SPLITS}")
    if sum(split_counts.values()) > len(cleaned):
        raise ValueError("the requested name split exceeds the candidate bank")
    ordered = sorted(cleaned, key=lambda name: _stable_key(seed, "name", name))
    splits: dict[str, tuple[str, ...]] = {}
    start = 0
    for split, count in split_counts.items():
        stop = start + int(count)
        splits[split] = tuple(ordered[start:stop])
        start = stop
    return splits


def validate_single_token_names(
    names: Sequence[str],
    encode: Callable[[str], Sequence[int]],
) -> dict[str, int]:
    """Require each leading-space name to map to one distinct token."""

    token_by_name: dict[str, int] = {}
    name_by_token: dict[int, str] = {}
    for name in names:
        tokens = tuple(int(token) for token in encode(f" {name}"))
        if len(tokens) != 1:
            raise ValueError(f"name {name!r} is not one leading-space token: {tokens}")
        token = tokens[0]
        if token in name_by_token:
            raise ValueError(
                f"names {name_by_token[token]!r} and {name!r} share token {token}"
            )
        token_by_name[name] = token
        name_by_token[token] = name
    return token_by_name


def _template_role_sequence(template: PromptTemplate) -> str:
    parts = template.text.split("{")
    roles = []
    for part in parts[1:]:
        role = part.split("}", 1)[0]
        if role in {"io", "s"}:
            roles.append("A" if role == "io" else "B")
    roles.append("A")
    return "".join(roles)


def template_frame() -> pd.DataFrame:
    rows = []
    for template in PHASE5_TEMPLATES:
        role_sequence = _template_role_sequence(template)
        if role_sequence != template.structure:
            raise ValueError(
                f"{template.template_id} has {role_sequence}, expected {template.structure}"
            )
        rows.append(
            {
                "template_id": template.template_id,
                "structure": template.structure,
                "template": template.text,
                "template_hash": json_sha256(
                    {
                        "template_id": template.template_id,
                        "structure": template.structure,
                        "template": template.text,
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def build_prompt_design(
    name_splits: Mapping[str, Sequence[str]],
    *,
    prompt_counts_by_split: Mapping[str, int],
    seed: int,
    token_by_name: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Build a balanced, deterministic prompt table with no model outcomes."""

    rows: list[dict[str, object]] = []
    if tuple(prompt_counts_by_split) != PROMPT_SPLITS:
        raise ValueError(f"prompt splits must be ordered as {PROMPT_SPLITS}")
    for split in PROMPT_SPLITS:
        prompts_in_split = int(prompt_counts_by_split[split])
        if prompts_in_split % len(PHASE5_TEMPLATES):
            raise ValueError("every prompt split must divide evenly across templates")
        per_template = prompts_in_split // len(PHASE5_TEMPLATES)
        split_names = tuple(name_splits[split])
        pairs = tuple(permutations(split_names, 2))
        if len(pairs) < per_template:
            raise ValueError(f"not enough distinct name pairs for {split}")
        for template in PHASE5_TEMPLATES:
            ordered_pairs = sorted(
                pairs,
                key=lambda pair: _stable_key(
                    seed,
                    f"prompt:{split}:{template.template_id}",
                    f"{pair[0]}:{pair[1]}",
                ),
            )
            for io_name, subject_name in ordered_pairs[:per_template]:
                prompt = template.text.format(io=io_name, s=subject_name)
                identity = {
                    "split": split,
                    "template_id": template.template_id,
                    "io_name": io_name,
                    "s_name": subject_name,
                    "prompt": prompt,
                }
                row: dict[str, object] = {
                    "prompt_id": _stable_id("prompt", identity),
                    "split": split,
                    "template_id": template.template_id,
                    "structure": template.structure,
                    "io_name": io_name,
                    "s_name": subject_name,
                    "prompt": prompt,
                    "answer": io_name,
                    "counterfactual_answer": subject_name,
                }
                if token_by_name is not None:
                    row["answer_token_id"] = token_by_name[io_name]
                    row["counterfactual_token_id"] = token_by_name[subject_name]
                rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["split", "template_id", "prompt_id"]
    )
    return frame.reset_index(drop=True)


def _mask_counts(bits: str) -> tuple[int, int, int, int]:
    mask = np.fromiter((int(bit) for bit in bits), dtype=int)
    return int(mask.sum()), int(mask[:3].sum()), int(mask[3:11].sum()), int(
        mask[11:].sum()
    )


def _mask_row(bits: str) -> dict[str, object]:
    n_heads, n_p, n_b, n_e = _mask_counts(bits)
    return {
        "mask_id": _stable_id("mask", {"mask_bits": bits}),
        "mask_bits": bits,
        "n_heads": n_heads,
        "n_P": n_p,
        "n_B": n_b,
        "n_E": n_e,
    }


def _mask_universe(excluded: set[str], *, seed: int) -> list[str]:
    masks = [
        f"{value:0{N_HEADS}b}"
        for value in range(1, 2**N_HEADS)
        if f"{value:0{N_HEADS}b}" not in excluded
    ]
    return sorted(masks, key=lambda bits: _stable_key(seed, "mask", bits))


def _candidate_masks(
    universe: Sequence[str],
    *,
    seed: int,
    pool_count: int,
    head_counts: Sequence[int],
    masks_per_stratum_by_head_count: Mapping[int, int],
) -> pd.DataFrame:
    available = set(universe)
    rows: list[dict[str, object]] = []
    for pool_index in range(pool_count):
        pool_id = f"candidate_pool_{pool_index:02d}"
        for n_heads in head_counts:
            masks_per_cell = int(masks_per_stratum_by_head_count[n_heads])
            for stratum, predicate in (
                ("broad", lambda n_p: n_p <= 1),
                ("primary_heavy", lambda n_p: n_p >= 2),
            ):
                eligible = [
                    bits
                    for bits in available
                    if _mask_counts(bits)[0] == n_heads
                    and predicate(_mask_counts(bits)[1])
                ]
                ordered = sorted(
                    eligible,
                    key=lambda bits: _stable_key(
                        seed,
                        f"candidate:{pool_id}:{n_heads}:{stratum}",
                        bits,
                    ),
                )
                chosen = ordered[:masks_per_cell]
                if len(chosen) != masks_per_cell:
                    raise ValueError(
                        f"not enough masks for {pool_id}, count {n_heads}, {stratum}"
                    )
                for within_cell, bits in enumerate(chosen):
                    available.remove(bits)
                    row = _mask_row(bits)
                    row.update(
                        {
                            "pool_id": pool_id,
                            "pool_index": pool_index,
                            "sampling_stratum": stratum,
                            "size_match_cell": f"n_heads_{n_heads:02d}",
                            "within_cell_index": within_cell,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["pool_index", "n_heads", "sampling_stratum", "mask_id"]
    ).reset_index(drop=True)


def _run_for_masks(masks: Sequence[str]) -> LoadedIOIRun:
    matrix = np.asarray([[int(bit) for bit in bits] for bits in masks], dtype=int)
    subset = pd.DataFrame([_mask_row(bits) for bits in masks])
    subset.insert(0, "subset_idx", np.arange(len(subset), dtype=int))
    heads = pd.DataFrame(head_records())
    return LoadedIOIRun(
        prefix="phase5_design",
        source=Path("phase5_design"),
        heads=heads,
        subset=subset,
        masks=matrix,
        prompt_drops=np.empty((0, len(matrix)), dtype=float),
        mean_drops=np.zeros(len(matrix), dtype=float),
        input_files=(),
    )


def _ordered_calibration_masks(
    universe: Sequence[str],
    *,
    count: int,
    initial_masks: Sequence[str] = (),
) -> list[str]:
    """Greedily order rows for early rank, then regularized D-optimality."""

    combined = tuple(initial_masks) + tuple(universe)
    run = _run_for_masks(combined)
    design, _ = build_capacity_design(run, "count_plus_all_bin4")
    scales = np.sqrt(np.mean(design**2, axis=0))
    scales[scales < 1e-12] = 1.0
    standardized = design / scales
    attainable_rank = int(np.linalg.matrix_rank(standardized))
    initial = standardized[: len(initial_masks)]
    candidate_rows = standardized[len(initial_masks) :]
    available = np.ones(len(candidate_rows), dtype=bool)
    selected: list[int] = []
    row_basis: list[np.ndarray] = []

    for row in initial:
        residual = row.copy()
        for basis in row_basis:
            residual -= float(residual @ basis) * basis
        norm = float(np.linalg.norm(residual))
        if norm > 1e-9:
            row_basis.append(residual / norm)

    for _ in range(count):
        if len(row_basis) < attainable_rank:
            residual_sq = np.einsum("ij,ij->i", candidate_rows, candidate_rows)
            for basis in row_basis:
                projection = candidate_rows @ basis
                residual_sq -= projection * projection
            residual_sq[~available] = -np.inf
            index = int(np.argmax(residual_sq))
            residual = candidate_rows[index].copy()
            for basis in row_basis:
                residual -= float(residual @ basis) * basis
            # Re-orthogonalize once to make the nested rank order stable.
            for basis in row_basis:
                residual -= float(residual @ basis) * basis
            norm = float(np.linalg.norm(residual))
            if norm <= 1e-9:
                raise RuntimeError("calibration row search stalled before full rank")
            row_basis.append(residual / norm)
        else:
            selected_design = np.vstack(
                [initial, candidate_rows[np.asarray(selected, dtype=int)]]
            )
            gram = selected_design.T @ selected_design
            gram += 1e-6 * np.eye(gram.shape[0], dtype=float)
            inverse = np.linalg.inv(gram)
            leverage = np.einsum(
                "ij,jk,ik->i", candidate_rows, inverse, candidate_rows
            )
            leverage[~available] = -np.inf
            index = int(np.argmax(leverage))
        selected.append(index)
        available[index] = False
    return [universe[index] for index in selected]


def build_mask_design(
    *,
    legacy_masks: set[str],
    seed: int,
    calibration_count: int,
    candidate_pool_count: int,
    candidate_head_counts: Sequence[int],
    masks_per_stratum_by_head_count: Mapping[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = ["0" * N_HEADS]
    anchors.extend(
        "".join("1" if index == head else "0" for index in range(N_HEADS))
        for head in range(N_HEADS)
    )
    universe = _mask_universe(legacy_masks | set(anchors), seed=seed)
    candidates = _candidate_masks(
        universe,
        seed=seed,
        pool_count=candidate_pool_count,
        head_counts=candidate_head_counts,
        masks_per_stratum_by_head_count=masks_per_stratum_by_head_count,
    )
    candidate_bits = set(candidates["mask_bits"].astype(str))
    calibration_universe = [bits for bits in universe if bits not in candidate_bits]
    calibration_bits = _ordered_calibration_masks(
        calibration_universe,
        count=calibration_count - len(anchors),
        initial_masks=anchors,
    )
    calibration_bits = anchors + calibration_bits
    calibration = pd.DataFrame([_mask_row(bits) for bits in calibration_bits])
    calibration.insert(
        0, "measurement_order", np.arange(1, len(calibration) + 1, dtype=int)
    )
    return calibration, candidates


def calibration_rank_diagnostics(
    calibration: pd.DataFrame,
    *,
    budgets: Sequence[int],
) -> pd.DataFrame:
    masks = calibration.sort_values("measurement_order")["mask_bits"].astype(str)
    run = _run_for_masks(tuple(masks))
    rows: list[dict[str, object]] = []
    for model in DESIGN_MODELS:
        design, columns = build_capacity_design(run, model)
        attainable_rank = int(np.linalg.matrix_rank(design))
        for budget in budgets:
            if budget > len(design):
                raise ValueError(f"budget {budget} exceeds calibration design")
            prefix = design[:budget]
            singular = np.linalg.svd(prefix, compute_uv=False)
            tolerance = (
                singular.max() * max(prefix.shape) * np.finfo(float).eps
                if len(singular)
                else 0.0
            )
            nonzero = singular[singular > tolerance]
            observed_rank = int(len(nonzero))
            condition = (
                float(nonzero[0] / nonzero[-1]) if len(nonzero) else float("inf")
            )
            rows.append(
                {
                    "model": model,
                    "budget": int(budget),
                    "n_columns": len(columns),
                    "attainable_rank": attainable_rank,
                    "observed_rank": observed_rank,
                    "rank_fraction": observed_rank / attainable_rank,
                    "full_attainable_rank": observed_rank == attainable_rank,
                    "smallest_nonzero_singular_value": (
                        float(nonzero[-1]) if len(nonzero) else 0.0
                    ),
                    "nonzero_condition_number": condition,
                    "prefix_hash": json_sha256(
                        calibration.sort_values("measurement_order")
                        .head(budget)["mask_bits"]
                        .astype(str)
                        .tolist()
                    ),
                }
            )
    return pd.DataFrame(rows)


def audit_design(
    design: Phase5Design,
    *,
    legacy_masks: set[str],
) -> dict[str, object]:
    prompts = design.prompts
    names = design.names
    calibration = design.calibration_masks
    candidates = design.candidate_masks
    protocol = design.protocol
    budgets = tuple(int(value) for value in protocol["measurement_budgets"])
    candidate_head_counts = tuple(
        int(value) for value in protocol["candidate_head_counts"]
    )

    names_by_split = {
        split: set(names.loc[names["split"] == split, "name"].astype(str))
        for split in PROMPT_SPLITS
    }
    calibration_bits = set(calibration["mask_bits"].astype(str))
    candidate_bits = set(candidates["mask_bits"].astype(str))
    anchor_bits = ["0" * N_HEADS]
    anchor_bits.extend(
        "".join("1" if index == head else "0" for index in range(N_HEADS))
        for head in range(N_HEADS)
    )
    variable_calibration_bits = calibration_bits - set(anchor_bits)
    expected_pool_size = int(protocol["candidate_pool_size"])
    pool_sizes = candidates.groupby("pool_id").size()
    per_pool_count_balance = candidates.groupby(["pool_id", "n_heads"]).size()
    per_pool_strata = candidates.groupby(["pool_id", "sampling_stratum"]).size()
    rank_at_primary = design.rank_diagnostics[
        design.rank_diagnostics["budget"] == max(budgets)
    ]

    gates = {
        "eight_literal_templates": len(design.templates) == 8
        and set(design.templates["structure"]) == {"ABBA", "BABA"},
        "prompt_count": len(prompts) == int(protocol["prompt_count"]),
        "outcome_prompt_count": int(
            prompts["split"].isin(("train", "validation", "test")).sum()
        )
        == int(protocol["outcome_prompt_count"]),
        "reference_prompt_count": int((prompts["split"] == "reference").sum())
        == int(protocol["reference_prompt_count"]),
        "unique_prompt_ids": prompts["prompt_id"].is_unique,
        "prompt_split_counts": prompts.groupby("split").size().to_dict()
        == {key: int(value) for key, value in protocol["prompt_counts_by_split"].items()},
        "template_counts_per_split": all(
            set(
                prompts.loc[prompts["split"] == split]
                .groupby("template_id")
                .size()
                .tolist()
            )
            == {int(protocol["prompts_per_template_by_split"][split])}
            for split in PROMPT_SPLITS
        ),
        "name_banks_pairwise_disjoint": sum(
            len(values) for values in names_by_split.values()
        )
        == len(set().union(*names_by_split.values())),
        "prompt_names_respect_split": all(
            {
                str(row.io_name),
                str(row.s_name),
            }
            <= names_by_split[str(row.split)]
            for row in prompts.itertuples(index=False)
        ),
        "calibration_count": len(calibration)
        == int(protocol["calibration_mask_count"]),
        "candidate_count": len(candidates)
        == int(protocol["new_nonclean_mask_count"]),
        "unique_masks": len(calibration_bits) == len(calibration)
        and len(candidate_bits) == len(candidates),
        "calibration_anchor_prefix": calibration.sort_values("measurement_order")
        .head(len(anchor_bits))["mask_bits"]
        .astype(str)
        .tolist()
        == anchor_bits,
        "nonclean_candidate_masks": "0" * N_HEADS not in candidate_bits,
        "calibration_candidate_disjoint": not (calibration_bits & candidate_bits),
        "candidate_legacy_masks_excluded": not (candidate_bits & legacy_masks),
        "variable_calibration_legacy_masks_excluded": not (
            variable_calibration_bits & legacy_masks
        ),
        "candidate_pool_sizes": len(pool_sizes)
        == int(protocol["candidate_pool_count"])
        and bool((pool_sizes == expected_pool_size).all()),
        "candidate_head_count_balance": all(
            int(per_pool_count_balance.loc[(pool_id, n_heads)])
            == 2
            * int(protocol["candidate_masks_per_stratum_by_head_count"][str(n_heads)])
            for pool_id in pool_sizes.index
            for n_heads in candidate_head_counts
        ),
        "candidate_stratum_balance": set(
            per_pool_strata.index.get_level_values("sampling_stratum")
        )
        == {"broad", "primary_heavy"}
        and per_pool_strata.nunique() == 1,
        "nested_budget_prefixes": tuple(sorted(budgets)) == budgets
        and max(budgets) == len(calibration)
        and calibration["measurement_order"].tolist()
        == list(range(1, len(calibration) + 1)),
        "full_rank_at_primary_budget": bool(
            rank_at_primary["full_attainable_rank"].all()
        ),
    }
    return {
        "schema": "observerbench.ioi_phase5_design_audit.v1",
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "counts": {
            "templates": len(design.templates),
            "names": len(names),
            "prompts": len(prompts),
            "legacy_masks": len(legacy_masks),
            "calibration_masks": len(calibration),
            "candidate_masks": len(candidates),
            "candidate_pools": int(candidates["pool_id"].nunique()),
        },
    }


def prepare_phase5_design(
    protocol: Mapping[str, object],
    *,
    legacy_masks: set[str],
    encode_name: Callable[[str], Sequence[int]] | None = None,
) -> Phase5Design:
    """Materialize the locked design in memory without evaluating GPT-2."""

    expected_templates = [
        {
            "template_id": template.template_id,
            "structure": template.structure,
            "template": template.text,
        }
        for template in PHASE5_TEMPLATES
    ]
    if protocol.get("templates") != expected_templates:
        raise ValueError("protocol templates do not match PHASE5_TEMPLATES")

    seed = int(protocol["seed"])
    names = tuple(str(name) for name in protocol["name_candidates"])
    splits = deterministic_name_split(
        names,
        seed=seed,
        split_counts={
            split: int(protocol["name_count_by_split"][split])
            for split in PROMPT_SPLITS
        },
    )
    token_by_name = validate_single_token_names(names, encode_name) if encode_name else None
    name_rows = []
    for split, split_names in splits.items():
        for name in split_names:
            row: dict[str, object] = {
                "name_id": _stable_id("name", {"name": name}),
                "name": name,
                "split": split,
            }
            if token_by_name is not None:
                row["leading_space_token_id"] = token_by_name[name]
            name_rows.append(row)
    name_frame = pd.DataFrame(name_rows).sort_values(["split", "name_id"])
    name_frame = name_frame.reset_index(drop=True)
    prompts = build_prompt_design(
        splits,
        prompt_counts_by_split={
            split: int(protocol["prompt_counts_by_split"][split])
            for split in PROMPT_SPLITS
        },
        seed=seed,
        token_by_name=token_by_name,
    )
    calibration, candidates = build_mask_design(
        legacy_masks=legacy_masks,
        seed=seed,
        calibration_count=int(protocol["calibration_mask_count"]),
        candidate_pool_count=int(protocol["candidate_pool_count"]),
        candidate_head_counts=tuple(protocol["candidate_head_counts"]),
        masks_per_stratum_by_head_count={
            int(n_heads): int(count)
            for n_heads, count in protocol[
                "candidate_masks_per_stratum_by_head_count"
            ].items()
        },
    )
    diagnostics = calibration_rank_diagnostics(
        calibration,
        budgets=tuple(protocol["measurement_budgets"]),
    )
    provisional = Phase5Design(
        templates=template_frame(),
        names=name_frame,
        prompts=prompts,
        calibration_masks=calibration,
        candidate_masks=candidates,
        rank_diagnostics=diagnostics,
        leakage_audit={},
        protocol=protocol,
    )
    audit = audit_design(provisional, legacy_masks=legacy_masks)
    if not bool(audit["all_gates_pass"]):
        failed = [name for name, passed in audit["gates"].items() if not passed]
        raise ValueError(f"Phase 5 design gates failed: {', '.join(failed)}")
    return Phase5Design(
        templates=provisional.templates,
        names=provisional.names,
        prompts=provisional.prompts,
        calibration_masks=provisional.calibration_masks,
        candidate_masks=provisional.candidate_masks,
        rank_diagnostics=provisional.rank_diagnostics,
        leakage_audit=audit,
        protocol=protocol,
    )


def load_phase5_protocol(path: str | Path) -> dict[str, object]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema") != "observerbench.ioi_selection_protocol.v2":
        raise ValueError("expected observerbench.ioi_selection_protocol.v2")
    return protocol


def write_phase5_design(
    design: Phase5Design,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    legacy_paths: Sequence[str | Path],
) -> Path:
    """Write only design inputs and provenance; no model outcome is accepted."""

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "templates.csv": design.templates,
        "names.csv": design.names,
        "prompts.csv": design.prompts,
        "calibration_masks.csv": design.calibration_masks,
        "candidate_masks.csv": design.candidate_masks,
        "rank_diagnostics.csv": design.rank_diagnostics,
    }
    calibration_for_runner = design.calibration_masks.copy()
    calibration_for_runner["bank"] = "calibration"
    calibration_for_runner["pool_id"] = ""
    candidates_for_runner = design.candidate_masks.copy()
    candidates_for_runner["bank"] = "candidate"
    frames["masks.csv"] = pd.concat(
        [calibration_for_runner, candidates_for_runner],
        ignore_index=True,
        sort=False,
    )
    for filename, frame in frames.items():
        frame.to_csv(output / filename, index=False)
    audit_path = output / "leakage_audit.json"
    audit_path.write_text(
        json.dumps(design.leakage_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_paths = [output / filename for filename in frames]
    artifact_paths.append(audit_path)
    manifest = {
        "schema": "observerbench.ioi_phase5_design_manifest.v1",
        "status": "frozen_before_outcomes",
        "design_id": _stable_id(
            "ioi_phase05_design",
            {
                "protocol": design.protocol,
                "calibration_masks": design.calibration_masks["mask_bits"].tolist(),
                "candidate_masks": design.candidate_masks["mask_bits"].tolist(),
                "prompt_ids": design.prompts["prompt_id"].tolist(),
            },
        ),
        "contains_model_outcomes": False,
        "tokenization_validated": "leading_space_token_id" in design.names.columns,
        "protocol_hash": file_sha256(protocol_path),
        "legacy_mask_source_hashes": {
            Path(path).name: file_sha256(path) for path in legacy_paths
        },
        "source_hashes": {
            filename: file_sha256(output / filename)
            for filename in ("prompts.csv", "masks.csv")
        },
        "artifact_hashes": {
            path.name: file_sha256(path) for path in artifact_paths
        },
        "all_design_gates_pass": bool(design.leakage_audit["all_gates_pass"]),
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "design_manifest.json").write_text(manifest_text, encoding="utf-8")
    (output / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return output
