"""Prospective Phase-6 IOI decision-risk confirmation.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The protocol is explicitly pilot-informed.  This module separates design,
calibration measurement, prediction freeze, and test evaluation so candidate
effects cannot be consulted while observers or fixed actions are chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.provenance import file_sha256, json_sha256
from observerbench.tasks.ioi.phase5_design import (
    N_HEADS,
    PHASE5_TEMPLATES,
    PromptTemplate,
    _candidate_masks,
    _stable_id,
    _stable_key,
    _mask_universe,
    validate_single_token_names,
)


PHASE6_SCHEMA = "observerbench.ioi_phase06_fresh_confirmation.v1"
PHASE6_STATUS = "pilot_informed_prospective_fresh_template_name_mask_confirmation"
DESIGN_SCHEMA = "observerbench.ioi_phase06_design_manifest.v1"
DESIGN_STATUS = "frozen_before_phase6_forward_passes"
NAME_SPLITS: tuple[str, ...] = ("reference", "train", "test")


@dataclass(frozen=True)
class Phase6Design:
    """Outcome-free inputs for the Phase-6 calibration and test stages."""

    templates: pd.DataFrame
    names: pd.DataFrame
    pair_clusters: pd.DataFrame
    prompts: pd.DataFrame
    calibration_masks: pd.DataFrame
    candidate_masks: pd.DataFrame
    audit: Mapping[str, Any]
    protocol: Mapping[str, Any]


def load_phase6_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate the pilot-informed prospective protocol."""

    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema") != PHASE6_SCHEMA:
        raise ValueError(f"expected {PHASE6_SCHEMA}")
    if protocol.get("status") != PHASE6_STATUS:
        raise ValueError("Phase 6 must disclose its pilot-informed prospective status")
    targets = tuple(float(value) for value in protocol["targets"])
    primary = tuple(float(value) for value in protocol["primary_targets"])
    stress = tuple(float(value) for value in protocol["stress_test_targets"])
    if targets != (0.5, 1.0, 1.5) or primary != (0.5, 1.0) or stress != (1.5,):
        raise ValueError("the frozen Phase-6 target hierarchy changed")
    if int(protocol["candidate_pool_count"]) != 48:
        raise ValueError("the power-audited design requires 48 candidate pools")
    if int(protocol["candidate_pool_size"]) != 32:
        raise ValueError("every candidate pool must contain 32 masks")
    if int(protocol["measurement_budget"]) != 160:
        raise ValueError("the Phase-6 observer fit requires 160 calibration masks")
    if float(protocol["ridge"]) != 1e-6:
        raise ValueError("the frozen ridge value is 1e-6")
    if "preregister" in json.dumps(protocol).lower():
        raise ValueError("use prospective/frozen wording; this is not externally preregistered")
    return protocol


def _phase6_templates(protocol: Mapping[str, Any]) -> tuple[PromptTemplate, ...]:
    templates = tuple(
        PromptTemplate(
            str(item["template_id"]),
            str(item["structure"]),
            str(item["template"]),
        )
        for item in protocol["templates"]
    )
    if len(templates) != 8 or len({item.template_id for item in templates}) != 8:
        raise ValueError("Phase 6 requires eight unique templates")
    if [item.structure for item in templates].count("ABBA") != 4:
        raise ValueError("Phase 6 requires four ABBA templates")
    if [item.structure for item in templates].count("BABA") != 4:
        raise ValueError("Phase 6 requires four BABA templates")
    old_text = {item.text for item in PHASE5_TEMPLATES}
    old_ids = {item.template_id for item in PHASE5_TEMPLATES}
    if old_text & {item.text for item in templates}:
        raise ValueError("Phase-6 primary templates must be new")
    if old_ids & {item.template_id for item in templates}:
        raise ValueError("Phase-6 template identifiers must be new")
    for item in templates:
        roles = [
            "A" if part.split("}", 1)[0] == "io" else "B"
            for part in item.text.split("{")[1:]
            if part.split("}", 1)[0] in {"io", "s"}
        ]
        roles.append("A")
        if "".join(roles) != item.structure:
            raise ValueError(
                f"template {item.template_id} has role sequence {''.join(roles)}"
            )
    return templates


def _template_frame(templates: Sequence[PromptTemplate]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "template_id": item.template_id,
                "structure": item.structure,
                "template": item.text,
                "template_hash": json_sha256(
                    {
                        "template_id": item.template_id,
                        "structure": item.structure,
                        "template": item.text,
                    }
                ),
            }
            for item in templates
        ]
    )


def _select_name_splits(
    protocol: Mapping[str, Any],
    *,
    phase5_names: pd.DataFrame,
    encode_name: Callable[[str], Sequence[int]],
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    candidates = tuple(str(value).strip() for value in protocol["name_candidates"])
    if len(set(candidates)) != len(candidates):
        raise ValueError("Phase-6 name candidates must be unique")
    old_names = set(phase5_names["name"].astype(str))
    overlap = old_names & set(candidates)
    if overlap:
        raise ValueError(f"Phase-6 candidate names overlap Phase 5: {sorted(overlap)[:3]}")
    token_by_name = validate_single_token_names(candidates, encode_name)
    old_tokens = set(
        pd.to_numeric(
            phase5_names.get("leading_space_token_id", pd.Series(dtype=int)),
            errors="coerce",
        ).dropna().astype(int)
    )
    if old_tokens & set(token_by_name.values()):
        raise ValueError("Phase-6 leading-space name tokens overlap Phase 5")

    seed = int(protocol["seed"])
    required = sum(int(protocol["name_count_by_split"][split]) for split in NAME_SPLITS)
    ordered = sorted(
        candidates,
        key=lambda name: _stable_key(seed, "phase6:name", name),
    )
    if len(ordered) < required:
        raise ValueError("not enough Phase-6 names for disjoint reference/train/test banks")
    selected = ordered[:required]
    splits: dict[str, tuple[str, ...]] = {}
    offset = 0
    rows: list[dict[str, Any]] = []
    for split in NAME_SPLITS:
        count = int(protocol["name_count_by_split"][split])
        values = tuple(selected[offset : offset + count])
        splits[split] = values
        offset += count
        for name in values:
            rows.append(
                {
                    "name_id": _stable_id("phase6_name", {"name": name}),
                    "name": name,
                    "split": split,
                    "leading_space_token_id": token_by_name[name],
                    "selection_rank": ordered.index(name),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "name_id"]).reset_index(drop=True), splits


def build_test_pair_clusters(
    names: Sequence[str],
    *,
    seed: int,
) -> pd.DataFrame:
    """Hash-partition test names into disjoint unordered pairs exactly once."""

    if len(names) % 2:
        raise ValueError("the test name bank must have even size")
    ordered = sorted(names, key=lambda name: _stable_key(seed, "phase6:test-pair", name))
    rows = []
    for pair_index in range(len(ordered) // 2):
        name_a, name_b = sorted(ordered[2 * pair_index : 2 * pair_index + 2])
        rows.append(
            {
                "pair_index": pair_index,
                "name_a": name_a,
                "name_b": name_b,
                "unordered_name_pair_id": _stable_id(
                    "phase6_pair", {"name_a": name_a, "name_b": name_b}
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("pair_index").reset_index(drop=True)


def _prompt_row(
    *,
    split: str,
    template: PromptTemplate,
    io_name: str,
    s_name: str,
    token_by_name: Mapping[str, int],
    unordered_name_pair_id: str,
    pair_orientation: str = "not_applicable",
) -> dict[str, Any]:
    prompt = template.text.format(io=io_name, s=s_name)
    identity = {
        "phase": 6,
        "split": split,
        "template_id": template.template_id,
        "io_name": io_name,
        "s_name": s_name,
        "prompt": prompt,
    }
    return {
        "prompt_id": _stable_id("phase6_prompt", identity),
        "split": split,
        "template_id": template.template_id,
        "structure": template.structure,
        "unordered_name_pair_id": unordered_name_pair_id,
        "pair_orientation": pair_orientation,
        "io_name": io_name,
        "s_name": s_name,
        "prompt": prompt,
        "answer": io_name,
        "counterfactual_answer": s_name,
        "answer_token_id": int(token_by_name[io_name]),
        "counterfactual_token_id": int(token_by_name[s_name]),
    }


def build_phase6_prompts(
    templates: Sequence[PromptTemplate],
    names: pd.DataFrame,
    splits: Mapping[str, Sequence[str]],
    pair_clusters: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    token_by_name = names.set_index("name")["leading_space_token_id"].astype(int).to_dict()
    seed = int(protocol["seed"])
    rows: list[dict[str, Any]] = []
    for split in ("reference", "train"):
        count = int(protocol["prompt_counts_by_split"][split])
        if count % len(templates):
            raise ValueError(f"{split} prompts must balance over templates")
        per_template = count // len(templates)
        pairs = tuple(permutations(tuple(splits[split]), 2))
        if len(pairs) < per_template:
            raise ValueError(f"not enough {split} name pairs")
        for template in templates:
            ordered = sorted(
                pairs,
                key=lambda pair: _stable_key(
                    seed,
                    f"phase6:{split}:{template.template_id}",
                    f"{pair[0]}::{pair[1]}",
                ),
            )
            for io_name, s_name in ordered[:per_template]:
                pair_id = _stable_id(
                    f"phase6_{split}_pair",
                    {"io": io_name, "s": s_name},
                )
                rows.append(
                    _prompt_row(
                        split=split,
                        template=template,
                        io_name=io_name,
                        s_name=s_name,
                        token_by_name=token_by_name,
                        unordered_name_pair_id=pair_id,
                    )
                )
    for pair in pair_clusters.itertuples(index=False):
        orientations = (
            (str(pair.name_a), str(pair.name_b), "a_to_b"),
            (str(pair.name_b), str(pair.name_a), "b_to_a"),
        )
        for io_name, s_name, orientation in orientations:
            for template in templates:
                rows.append(
                    _prompt_row(
                        split="test",
                        template=template,
                        io_name=io_name,
                        s_name=s_name,
                        token_by_name=token_by_name,
                        unordered_name_pair_id=str(pair.unordered_name_pair_id),
                        pair_orientation=orientation,
                    )
                )
    return pd.DataFrame(rows).sort_values(
        ["split", "template_id", "prompt_id"]
    ).reset_index(drop=True)


def build_phase6_candidate_masks(
    protocol: Mapping[str, Any],
    *,
    excluded_bits: set[str],
) -> pd.DataFrame:
    seed = int(protocol["seed"])
    universe = _mask_universe(set(excluded_bits), seed=seed)
    candidates = _candidate_masks(
        universe,
        seed=seed,
        pool_count=int(protocol["candidate_pool_count"]),
        head_counts=tuple(int(value) for value in protocol["candidate_head_counts"]),
        masks_per_stratum_by_head_count={
            int(key): int(value)
            for key, value in protocol["candidate_masks_per_stratum_by_head_count"].items()
        },
    )
    candidates["pool_id"] = candidates["pool_index"].map(
        lambda value: f"phase06_candidate_pool_{int(value):02d}"
    )
    candidates["bank"] = "candidate"
    return candidates


def _head_quadratic_rank(calibration: pd.DataFrame) -> int:
    """Rank of intercept + 13 heads + all 78 distinct head products."""

    ordered = calibration.sort_values("measurement_order")
    masks = np.asarray(
        [[int(bit) for bit in bits] for bits in ordered["mask_bits"].astype(str)],
        dtype=float,
    )
    pair_block = np.column_stack(
        [masks[:, left] * masks[:, right] for left, right in combinations(range(N_HEADS), 2)]
    )
    return int(np.linalg.matrix_rank(np.column_stack([np.ones(len(masks)), masks, pair_block])))


def audit_phase6_design(
    design: Phase6Design,
    *,
    phase5_names: pd.DataFrame,
    phase5_masks: pd.DataFrame,
    excluded_bits: set[str],
) -> dict[str, Any]:
    protocol = design.protocol
    prompts = design.prompts
    candidates = design.candidate_masks
    calibration = design.calibration_masks
    pair_clusters = design.pair_clusters
    names = design.names
    names_by_split = {
        split: set(names.loc[names["split"] == split, "name"].astype(str))
        for split in NAME_SPLITS
    }
    pair_degree = pd.concat(
        [
            pair_clusters[["name_a"]].rename(columns={"name_a": "name"}),
            pair_clusters[["name_b"]].rename(columns={"name_b": "name"}),
        ],
        ignore_index=True,
    )["name"].value_counts()
    test_prompts = prompts[prompts["split"] == "test"]
    io_prompt_degree = test_prompts["io_name"].value_counts()
    s_prompt_degree = test_prompts["s_name"].value_counts()
    pool_sizes = candidates.groupby("pool_id").size()
    expected_alloc = {
        int(key): int(value)
        for key, value in protocol["candidate_masks_per_stratum_by_head_count"].items()
    }
    phase5_bits = set(phase5_masks["mask_bits"].astype(str))
    candidate_bits = set(candidates["mask_bits"].astype(str))
    calibration_bits = set(calibration["mask_bits"].astype(str))
    gates = {
        "eight_new_balanced_templates": len(design.templates) == 8
        and design.templates["template_id"].is_unique
        and design.templates["structure"].value_counts().to_dict()
        == {"ABBA": 4, "BABA": 4},
        "name_split_counts": names.groupby("split").size().to_dict()
        == {
            split: int(protocol["name_count_by_split"][split])
            for split in NAME_SPLITS
        },
        "name_banks_disjoint": sum(map(len, names_by_split.values()))
        == len(set().union(*names_by_split.values())),
        "names_disjoint_phase5": not (
            set(names["name"].astype(str)) & set(phase5_names["name"].astype(str))
        ),
        "name_tokens_unique": names["leading_space_token_id"].is_unique,
        "prompt_split_counts": prompts.groupby("split").size().to_dict()
        == {
            split: int(protocol["prompt_counts_by_split"][split])
            for split in NAME_SPLITS
        },
        "prompt_ids_and_text_unique": prompts["prompt_id"].is_unique
        and prompts["prompt"].is_unique,
        "prompt_names_respect_splits": all(
            {str(row.io_name), str(row.s_name)} <= names_by_split[str(row.split)]
            for row in prompts.itertuples(index=False)
        ),
        "test_pair_cluster_count": len(pair_clusters)
        == int(protocol["test_unordered_pair_cluster_count"]),
        "test_pair_edges_unique": not pair_clusters.duplicated(["name_a", "name_b"]).any(),
        "test_names_belong_to_one_pair": set(pair_degree.tolist()) == {1},
        "test_pair_roles_balanced": set(io_prompt_degree.tolist()) == {8}
        and set(s_prompt_degree.tolist()) == {8},
        "sixteen_test_prompts_per_pair": set(
            prompts.loc[prompts["split"] == "test"]
            .groupby("unordered_name_pair_id")
            .size()
            .tolist()
        )
        == {int(protocol["test_prompts_per_pair_cluster"])},
        "calibration_is_exact_ordered_phase5_bank": len(calibration) == 160
        and calibration.sort_values("measurement_order")["measurement_order"].astype(int).tolist()
        == list(range(1, 161))
        and calibration.sort_values("measurement_order")["mask_bits"].astype(str).tolist()
        == phase5_masks.loc[phase5_masks["bank"] == "calibration"]
        .sort_values("measurement_order")["mask_bits"]
        .astype(str)
        .tolist(),
        "candidate_count": len(candidates) == int(protocol["new_candidate_mask_count"]),
        "candidate_masks_unique": candidates["mask_id"].is_unique
        and candidates["mask_bits"].is_unique,
        "candidate_masks_fresh": not (candidate_bits & excluded_bits)
        and not (candidate_bits & phase5_bits)
        and not (candidate_bits & calibration_bits),
        "candidate_pool_sizes": len(pool_sizes) == int(protocol["candidate_pool_count"])
        and set(pool_sizes.tolist()) == {int(protocol["candidate_pool_size"])},
        "candidate_strata_balanced": all(
            len(group[group["sampling_stratum"] == stratum]) == 16
            for _, group in candidates.groupby("pool_id")
            for stratum in ("broad", "primary_heavy")
        ),
        "candidate_head_count_cells": all(
            len(
                group[
                    (group["n_heads"] == n_heads)
                    & (group["sampling_stratum"] == stratum)
                ]
            )
            == count
            for _, group in candidates.groupby("pool_id")
            for n_heads, count in expected_alloc.items()
            for stratum in ("broad", "primary_heavy")
        ),
        "quadratic_calibration_rank_92": _head_quadratic_rank(calibration) == 92,
        "no_outcome_columns": not any(
            column in prompts.columns or column in candidates.columns
            for column in ("clean_ld", "ablated_ld", "drop_from_clean", "target_loss")
        ),
    }
    return {
        "schema": "observerbench.ioi_phase06_design_audit.v1",
        "status": DESIGN_STATUS,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "counts": {
            "templates": len(design.templates),
            "names": len(names),
            "pair_clusters": len(pair_clusters),
            "prompts": len(prompts),
            "reference_prompts": int((prompts["split"] == "reference").sum()),
            "train_prompts": int((prompts["split"] == "train").sum()),
            "test_prompts": int((prompts["split"] == "test").sum()),
            "calibration_masks": len(calibration),
            "candidate_masks": len(candidates),
            "candidate_pools": int(candidates["pool_id"].nunique()),
        },
    }


def prepare_phase6_design(
    protocol: Mapping[str, Any],
    *,
    phase5_names: pd.DataFrame,
    phase5_masks: pd.DataFrame,
    excluded_bits: set[str],
    encode_name: Callable[[str], Sequence[int]],
) -> Phase6Design:
    """Construct all prompts and masks without a model forward pass."""

    templates = _phase6_templates(protocol)
    name_frame, splits = _select_name_splits(
        protocol,
        phase5_names=phase5_names,
        encode_name=encode_name,
    )
    pairs = build_test_pair_clusters(
        splits["test"],
        seed=int(protocol["seed"]),
    )
    prompts = build_phase6_prompts(
        templates,
        name_frame,
        splits,
        pairs,
        protocol=protocol,
    )
    calibration = phase5_masks[phase5_masks["bank"] == "calibration"].copy()
    calibration["bank"] = "calibration"
    calibration["pool_id"] = ""
    candidates = build_phase6_candidate_masks(protocol, excluded_bits=excluded_bits)
    provisional = Phase6Design(
        templates=_template_frame(templates),
        names=name_frame,
        pair_clusters=pairs,
        prompts=prompts,
        calibration_masks=calibration,
        candidate_masks=candidates,
        audit={},
        protocol=protocol,
    )
    audit = audit_phase6_design(
        provisional,
        phase5_names=phase5_names,
        phase5_masks=phase5_masks,
        excluded_bits=excluded_bits,
    )
    if not audit["all_gates_pass"]:
        failed = [name for name, passed in audit["gates"].items() if not passed]
        raise ValueError(f"Phase-6 design gates failed: {', '.join(failed)}")
    return Phase6Design(**{**provisional.__dict__, "audit": audit})


def write_phase6_design(
    design: Phase6Design,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    phase5_design_dir: str | Path,
    exclusion_paths: Iterable[str | Path],
) -> Path:
    """Freeze the outcome-free design and a hash-indexed manifest."""

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    masks = pd.concat(
        [design.calibration_masks, design.candidate_masks],
        ignore_index=True,
        sort=False,
    )
    frames = {
        "templates.csv": design.templates,
        "names.csv": design.names,
        "pair_clusters.csv": design.pair_clusters,
        "prompts.csv": design.prompts,
        "calibration_masks.csv": design.calibration_masks,
        "candidate_masks.csv": design.candidate_masks,
        "masks.csv": masks,
    }
    for filename, frame in frames.items():
        frame.to_csv(output / filename, index=False)
    audit_path = output / "leakage_audit.json"
    audit_path.write_text(
        json.dumps(design.audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_paths = [output / filename for filename in frames]
    artifact_paths.append(audit_path)
    phase5_root = Path(phase5_design_dir)
    manifest = {
        "schema": DESIGN_SCHEMA,
        "status": DESIGN_STATUS,
        "scientific_status": PHASE6_STATUS,
        "design_id": _stable_id(
            "ioi_phase06_design",
            {
                "protocol": design.protocol,
                "prompt_ids": design.prompts["prompt_id"].tolist(),
                "calibration_masks": design.calibration_masks["mask_bits"].tolist(),
                "candidate_masks": design.candidate_masks["mask_bits"].tolist(),
            },
        ),
        "contains_model_outcomes": False,
        "phase6_forward_passes_performed": False,
        "predictions_frozen": False,
        "tokenization_validated": True,
        "all_design_gates_pass": bool(design.audit["all_gates_pass"]),
        "protocol_sha256": file_sha256(protocol_path),
        "phase5_design_manifest_sha256": file_sha256(
            phase5_root / "design_manifest.json"
        ),
        "phase5_calibration_masks_sha256": file_sha256(
            phase5_root / "calibration_masks.csv"
        ),
        "exclusion_source_hashes": {
            Path(path).name: file_sha256(path) for path in exclusion_paths
        },
        "artifact_hashes": {
            path.name: file_sha256(path) for path in artifact_paths
        },
        "locked_sources": {
            "prompts.csv": file_sha256(output / "prompts.csv"),
            "calibration_masks.csv": file_sha256(output / "calibration_masks.csv"),
            "candidate_masks.csv": file_sha256(output / "candidate_masks.csv"),
        },
        "next_allowed_stage": (
            "Measure new template-conditioned reference means and only the 160 "
            "calibration masks on train prompts; freeze predictions and actions "
            "before measuring any candidate mask on test prompts."
        ),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "design_manifest.json").write_text(text, encoding="utf-8")
    (output / "manifest.json").write_text(text, encoding="utf-8")
    return output
