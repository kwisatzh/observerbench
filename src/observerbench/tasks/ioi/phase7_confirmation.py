"""Outcome-sealed canonical-template IOI action confirmation.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

Phase 7 composes the Phase-5 calibration surface, Phase-6 fresh names, and the
existing quadratic observer basis.  Design, clean pretest, action freeze, and
held-out measurement remain separate stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256, runtime_provenance
from observerbench.tasks.ioi.phase5_analysis import (
    _design_run,
    _validate_effect_manifest,
    _validate_split_cartesian,
)
from observerbench.tasks.ioi.phase5_design import (
    N_HEADS,
    PHASE5_TEMPLATES,
    _candidate_masks,
    _mask_universe,
    _stable_id,
    _stable_key,
    load_legacy_mask_bits,
    template_frame,
)
from observerbench.tasks.ioi.phase5_effects import (
    load_locked_ioi_design,
    validate_effect_rows,
)
from observerbench.tasks.ioi.phase6_risk import _head_quadratic_design
from observerbench.tasks.ioi.stage2d import ridge_fit


PROTOCOL_SCHEMA_V1 = "observerbench.ioi_phase07_canonical_noop_confirmation.v1"
PROTOCOL_SCHEMA_V2 = "observerbench.ioi_phase07_canonical_noop_confirmation.v2"
PROTOCOL_SCHEMA = PROTOCOL_SCHEMA_V1
SCIENTIFIC_STATUS = "pilot_informed_outcome_sealed_canonical_template_confirmation"
DESIGN_SCHEMA = "observerbench.ioi_phase07_design.v1"
DESIGN_STATUS = "outcome_free_design_frozen"
PRETEST_SCHEMA = "observerbench.ioi_phase07_clean_pretest.v1"
PRETEST_PASS_STATUS = "clean_pretest_passed_candidate_outcomes_unopened"
FREEZE_SCHEMA = "observerbench.ioi_phase07_prediction_action_freeze.v1"
FREEZE_STATUS = "actions_frozen_candidate_outcomes_unopened"
DIRECT_RISK = "direct_risk_head_pair_quadratic"
NATURAL_MEAN = "natural_mean_effect_head_pair_quadratic"
EXACT_NOOP = "exact_noop"
TARGET_POLICY = "target_loss"
NOOP_BITS = "0" * N_HEADS
V1_PROTOCOL_SHA256 = "852d05da70ba13e870baf797906a190325d3eb82cbf1452c75983383e683e0f2"
V1_ABORTED_EVIDENCE = {
    "results/revision/phase07/ioi_canonical_noop_confirmation/selected_measurement/measurement_run_spec.json": "23426bd642f760b431495d285ed9393d5032525730701baf8a2fa50532409a51",
    "results/revision/phase07/ioi_canonical_noop_confirmation/selected_measurement/measurement_progress.json": "77ce34f5426a680d3acfd4bc91cfebad8bb38dd8a49b1af1f2fd3fe665a5b8bc",
    "results/revision/phase07/ioi_canonical_noop_confirmation/selected_measurement/shards/test/effects_0000_0016.csv": "b14e7319940a5a71c73b2e83164105ade23a9f01e9e713ab09841d723303cc5e",
}


@dataclass(frozen=True)
class Phase7Design:
    """All model-outcome-free Phase-7 inputs."""

    templates: pd.DataFrame
    names: pd.DataFrame
    pair_clusters: pd.DataFrame
    prompts: pd.DataFrame
    calibration_masks: pd.DataFrame
    candidate_actions: pd.DataFrame
    audit: Mapping[str, Any]
    protocol: Mapping[str, Any]


def load_phase7_protocol(path: str | Path) -> dict[str, Any]:
    """Load and enforce the frozen Phase-7 scientific choices."""

    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = protocol.get("schema")
    if schema not in {PROTOCOL_SCHEMA_V1, PROTOCOL_SCHEMA_V2}:
        raise ValueError(
            f"expected {PROTOCOL_SCHEMA_V1} or {PROTOCOL_SCHEMA_V2}"
        )
    if protocol.get("status") != SCIENTIFIC_STATUS:
        raise ValueError("Phase-7 pilot-informed status changed")
    if float(protocol.get("target", np.nan)) != 1.0:
        raise ValueError("Phase 7 has one frozen target: 1.0")
    if int(protocol.get("measurement_budget", -1)) != 160:
        raise ValueError("Phase 7 requires all 160 Phase-5 calibration masks")
    if float(protocol.get("ridge", np.nan)) != 1e-6:
        raise ValueError("Phase-7 ridge changed")
    if int(protocol.get("candidate_pool_count", -1)) != 48:
        raise ValueError("Phase 7 requires 48 action pools")
    if int(protocol.get("fresh_nonnoop_masks_per_pool", -1)) != 30:
        raise ValueError("Phase 7 requires 30 fresh interventions per pool")
    if int(protocol.get("candidate_pool_size_including_noop", -1)) != 31:
        raise ValueError("Phase 7 requires 30 interventions plus one no-op per pool")
    if "preregister" in str(protocol.get("status", "")).lower():
        raise ValueError("Phase 7 is frozen locally, not externally preregistered")
    if schema == PROTOCOL_SCHEMA_V2:
        repair = protocol.get("v1_aborted_attempt")
        if not isinstance(repair, Mapping):
            raise ValueError("Phase-7 v2 must disclose the aborted v1 attempt")
        required = {
            "status": "aborted_after_one_selected_only_shard",
            "partial_effect_cells": 8192,
            "partial_selected_masks": 16,
            "candidate_outcome_values_inspected": False,
            "evaluation_run": False,
        }
        for key, expected in required.items():
            if repair.get(key) != expected:
                raise ValueError(f"Phase-7 v2 aborted-attempt disclosure changed: {key}")
        if repair.get("v1_protocol_sha256") != V1_PROTOCOL_SHA256:
            raise ValueError("Phase-7 v2 no longer pins the superseded v1 protocol")
        evidence = repair.get("artifact_hashes")
        if evidence != V1_ABORTED_EVIDENCE:
            raise ValueError("Phase-7 v2 must hash all three v1 partial-run artifacts")
    return protocol


def verify_protocol_sources(protocol: Mapping[str, Any], root: str | Path) -> None:
    """Verify every pilot and inherited artifact named in the protocol."""

    base = Path(root)
    for section in ("pilot_and_source_hashes", "legacy_mask_exclusions"):
        for relative, expected in protocol[section].items():
            path = base / str(relative)
            if not path.is_file() or file_sha256(path) != str(expected):
                raise ValueError(f"frozen Phase-7 source changed: {relative}")
    repair = protocol.get("v1_aborted_attempt")
    if isinstance(repair, Mapping):
        for relative, expected in repair["artifact_hashes"].items():
            path = base / str(relative)
            if not path.is_file() or file_sha256(path) != str(expected):
                raise ValueError(f"frozen Phase-7 v1 audit evidence changed: {relative}")


def build_fresh_pair_clusters(
    names: Sequence[str],
    phase6_pairs: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Build the first deterministic perfect matching disjoint from Phase 6."""

    values = tuple(map(str, names))
    if len(values) != 64 or len(set(values)) != 64:
        raise ValueError("Phase 7 requires 64 unique Phase-6 test names")
    old_edges = {
        tuple(sorted((str(row.name_a), str(row.name_b))))
        for row in phase6_pairs.itertuples(index=False)
    }
    ordered = sorted(values, key=lambda name: _stable_key(seed, "phase7:pair", name))
    left, right = ordered[:32], ordered[32:]
    selected: list[tuple[str, str]] | None = None
    selected_offset: int | None = None
    for offset in range(32):
        edges = [
            tuple(sorted((left[index], right[(index + offset) % 32])))
            for index in range(32)
        ]
        if not (set(edges) & old_edges):
            selected = edges
            selected_offset = offset
            break
    if selected is None or selected_offset is None:
        raise ValueError("the frozen Phase-7 pairing rule found no disjoint matching")
    rows = [
        {
            "pair_index": index,
            "name_a": edge[0],
            "name_b": edge[1],
            "unordered_name_pair_id": _stable_id(
                "phase7_pair", {"name_a": edge[0], "name_b": edge[1]}
            ),
            "cyclic_offset": selected_offset,
        }
        for index, edge in enumerate(selected)
    ]
    return pd.DataFrame(rows).sort_values("pair_index").reset_index(drop=True)


def build_phase7_prompts(
    names: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Combine fresh names and pairs with literal Phase-5 templates."""

    token_by_name = (
        names.set_index("name")["leading_space_token_id"].astype(int).to_dict()
    )
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        orientations = (
            (str(pair.name_a), str(pair.name_b), "a_to_b"),
            (str(pair.name_b), str(pair.name_a), "b_to_a"),
        )
        for io_name, s_name, orientation in orientations:
            for template in PHASE5_TEMPLATES:
                prompt = template.text.format(io=io_name, s=s_name)
                identity = {
                    "phase": 7,
                    "template_id": template.template_id,
                    "io_name": io_name,
                    "s_name": s_name,
                    "prompt": prompt,
                }
                rows.append(
                    {
                        "prompt_id": _stable_id("phase7_prompt", identity),
                        "split": "test",
                        "template_id": template.template_id,
                        "structure": template.structure,
                        "unordered_name_pair_id": str(pair.unordered_name_pair_id),
                        "pair_orientation": orientation,
                        "io_name": io_name,
                        "s_name": s_name,
                        "prompt": prompt,
                        "answer": io_name,
                        "counterfactual_answer": s_name,
                        "answer_token_id": int(token_by_name[io_name]),
                        "counterfactual_token_id": int(token_by_name[s_name]),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["template_id", "unordered_name_pair_id", "pair_orientation"]
    ).reset_index(drop=True)


def build_phase7_candidate_actions(
    protocol: Mapping[str, Any],
    *,
    excluded_bits: set[str],
) -> pd.DataFrame:
    """Build 48 fresh 30-mask pools and append an analytic no-op to each."""

    seed = int(protocol["seed"])
    nonnoop = _candidate_masks(
        _mask_universe(set(excluded_bits) | {NOOP_BITS}, seed=seed),
        seed=seed,
        pool_count=int(protocol["candidate_pool_count"]),
        head_counts=tuple(map(int, protocol["candidate_head_counts"])),
        masks_per_stratum_by_head_count={
            int(key): int(value)
            for key, value in protocol["fresh_masks_per_stratum_by_head_count"].items()
        },
    )
    nonnoop["pool_id"] = nonnoop["pool_index"].map(
        lambda value: f"phase07_action_pool_{int(value):02d}"
    )
    nonnoop["bank"] = "candidate"
    nonnoop["is_noop"] = False
    nonnoop["physical_mask_id"] = nonnoop["mask_id"].astype(str)
    noops = pd.DataFrame(
        [
            {
                "mask_id": f"phase07_exact_noop_{pool_index:02d}",
                "mask_bits": NOOP_BITS,
                "n_heads": 0,
                "n_P": 0,
                "n_B": 0,
                "n_E": 0,
                "pool_id": f"phase07_action_pool_{pool_index:02d}",
                "pool_index": pool_index,
                "sampling_stratum": "no_op",
                "size_match_cell": "n_heads_00",
                "within_cell_index": 0,
                "bank": "candidate",
                "is_noop": True,
                "physical_mask_id": "exact_noop",
            }
            for pool_index in range(int(protocol["candidate_pool_count"]))
        ]
    )
    return pd.concat([nonnoop, noops], ignore_index=True, sort=False).sort_values(
        ["pool_index", "is_noop", "n_heads", "sampling_stratum", "mask_id"]
    ).reset_index(drop=True)


def audit_phase7_design(
    design: Phase7Design,
    *,
    phase5_prompts: pd.DataFrame,
    phase6_prompts: pd.DataFrame,
    phase6_pairs: pd.DataFrame,
    excluded_bits: set[str],
) -> dict[str, Any]:
    """Run every outcome-free design gate."""

    prompts = design.prompts
    pairs = design.pair_clusters
    actions = design.candidate_actions
    nonnoop = actions.loc[~actions["is_noop"].astype(bool)]
    noops = actions.loc[actions["is_noop"].astype(bool)]
    old_edges = {
        tuple(sorted((str(row.name_a), str(row.name_b))))
        for row in phase6_pairs.itertuples(index=False)
    }
    new_edges = {
        tuple(sorted((str(row.name_a), str(row.name_b))))
        for row in pairs.itertuples(index=False)
    }
    old_prompt_text = set(phase5_prompts["prompt"].astype(str)) | set(
        phase6_prompts["prompt"].astype(str)
    )
    template_counts = prompts.groupby("template_id").size()
    pair_counts = prompts.groupby("unordered_name_pair_id").size()
    pool_sizes = actions.groupby("pool_id").size()
    nonnoop_pool_sizes = nonnoop.groupby("pool_id").size()
    noop_pool_sizes = noops.groupby("pool_id").size()
    allocation = {
        int(key): int(value)
        for key, value in design.protocol["fresh_masks_per_stratum_by_head_count"].items()
    }
    gates = {
        "literal_phase5_templates": design.templates["template"].astype(str).tolist()
        == template_frame()["template"].astype(str).tolist(),
        "phase6_test_name_bank_reused_exactly": len(design.names) == 64
        and set(design.names["split"].astype(str)) == {"test"}
        and design.names["name"].is_unique
        and design.names["leading_space_token_id"].is_unique,
        "fresh_pair_matching": len(pairs) == 32
        and len(new_edges) == 32
        and not (new_edges & old_edges),
        "each_name_in_one_pair": len(set(pairs["name_a"]) | set(pairs["name_b"])) == 64,
        "test_prompt_count": len(prompts) == 512,
        "prompt_ids_unique": prompts["prompt_id"].is_unique,
        "prompt_strings_unique": prompts["prompt"].is_unique,
        "all_prompt_strings_new": not (set(prompts["prompt"].astype(str)) & old_prompt_text),
        "template_balance": len(template_counts) == 8 and set(template_counts) == {64},
        "pair_balance": len(pair_counts) == 32 and set(pair_counts) == {16},
        "both_orientations": all(
            set(group["pair_orientation"].astype(str)) == {"a_to_b", "b_to_a"}
            for _, group in prompts.groupby("unordered_name_pair_id")
        ),
        "action_pool_count_and_size": len(pool_sizes) == 48 and set(pool_sizes) == {31},
        "thirty_fresh_masks_per_pool": set(nonnoop_pool_sizes) == {30},
        "one_noop_per_pool": set(noop_pool_sizes) == {1},
        "nonnoop_ids_and_bits_unique": nonnoop["mask_id"].is_unique
        and nonnoop["mask_bits"].is_unique,
        "nonnoop_masks_fresh": not (
            set(nonnoop["mask_bits"].astype(str)) & set(excluded_bits)
        ),
        "only_noop_bits_repeat": set(noops["mask_bits"].astype(str)) == {NOOP_BITS}
        and int(actions["mask_bits"].astype(str).value_counts().max()) == 48,
        "prespecified_nonnoop_cells": all(
            len(
                group[
                    (group["n_heads"].astype(int) == n_heads)
                    & (group["sampling_stratum"].astype(str) == stratum)
                ]
            )
            == count
            for _, group in nonnoop.groupby("pool_id")
            for n_heads, count in allocation.items()
            for stratum in ("broad", "primary_heavy")
        ),
        "no_outcome_columns": not {
            "clean_ld",
            "ablated_ld",
            "drop_from_clean",
            "target_loss",
        }
        & (set(prompts) | set(actions)),
    }
    return {
        "schema": "observerbench.ioi_phase07_design_audit.v1",
        "status": DESIGN_STATUS,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "counts": {
            "templates": len(design.templates),
            "names": len(design.names),
            "pair_clusters": len(pairs),
            "test_prompts": len(prompts),
            "calibration_masks": len(design.calibration_masks),
            "candidate_action_rows": len(actions),
            "fresh_nonnoop_masks": len(nonnoop),
            "no_op_rows": len(noops),
            "action_pools": int(actions["pool_id"].nunique()),
        },
    }


def prepare_phase7_design(
    protocol: Mapping[str, Any],
    *,
    phase5_prompts: pd.DataFrame,
    phase5_masks: pd.DataFrame,
    phase6_names: pd.DataFrame,
    phase6_pairs: pd.DataFrame,
    phase6_prompts: pd.DataFrame,
    excluded_bits: set[str],
) -> Phase7Design:
    """Construct the full design without a model forward pass."""

    names = phase6_names.loc[phase6_names["split"].astype(str) == "test"].copy()
    names = names.sort_values("name").reset_index(drop=True)
    pairs = build_fresh_pair_clusters(names["name"], phase6_pairs, seed=int(protocol["seed"]))
    prompts = build_phase7_prompts(names, pairs)
    calibration = phase5_masks.loc[phase5_masks["bank"].astype(str) == "calibration"].copy()
    calibration["pool_id"] = ""
    actions = build_phase7_candidate_actions(protocol, excluded_bits=excluded_bits)
    provisional = Phase7Design(
        templates=template_frame(),
        names=names,
        pair_clusters=pairs,
        prompts=prompts,
        calibration_masks=calibration,
        candidate_actions=actions,
        audit={},
        protocol=protocol,
    )
    audit = audit_phase7_design(
        provisional,
        phase5_prompts=phase5_prompts,
        phase6_prompts=phase6_prompts,
        phase6_pairs=phase6_pairs,
        excluded_bits=excluded_bits,
    )
    if not audit["all_gates_pass"]:
        failed = [name for name, passed in audit["gates"].items() if not passed]
        raise ValueError(f"Phase-7 design gates failed: {', '.join(failed)}")
    return Phase7Design(**{**provisional.__dict__, "audit": audit})


def write_phase7_design(
    design: Phase7Design,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
    source_paths: Mapping[str, str | Path],
) -> Path:
    """Write a hash-indexed, model-outcome-free design."""

    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a frozen Phase-7 design is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "templates.csv": design.templates,
        "names.csv": design.names,
        "pair_clusters.csv": design.pair_clusters,
        "prompts.csv": design.prompts,
        "calibration_masks.csv": design.calibration_masks,
        "candidate_actions.csv": design.candidate_actions,
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    write_json(output / "design_audit.json", dict(design.audit))
    artifacts = [*frames, "design_audit.json"]
    manifest = {
        "schema": DESIGN_SCHEMA,
        "status": DESIGN_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": _stable_id(
            "ioi_phase07_design",
            {
                "protocol": design.protocol,
                "prompts": design.prompts["prompt_id"].tolist(),
                "pairs": design.pair_clusters["unordered_name_pair_id"].tolist(),
                "actions": design.candidate_actions["mask_id"].tolist(),
            },
        ),
        "contains_model_outcomes": False,
        "candidate_effect_forward_passes": 0,
        "protocol_sha256": file_sha256(protocol_path),
        "source_hashes": {
            label: file_sha256(path) for label, path in source_paths.items()
        },
        "artifact_hashes": {
            name: file_sha256(output / name) for name in artifacts
        },
        "all_design_gates_pass": bool(design.audit["all_gates_pass"]),
        "next_allowed_stage": (
            "Run the clean-only pretest. Do not fit Phase-7 observers and do not "
            "measure a Phase-7 candidate effect unless every clean gate passes."
        ),
    }
    write_json(output / "design_manifest.json", manifest)
    return output


def _verify_artifact_index(root: Path, index: Mapping[str, Any], label: str) -> None:
    if not isinstance(index, Mapping) or not index:
        raise ValueError(f"{label} lacks an artifact hash index")
    for raw_name, expected in index.items():
        relative = Path(str(raw_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} contains a non-portable path")
        path = root / relative
        if not path.is_file() or file_sha256(path) != str(expected):
            raise ValueError(f"{label} artifact changed: {raw_name}")


def load_verified_phase7_design(
    design_dir: str | Path,
    protocol_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verify the design seal and return prompts, calibration, and actions."""

    root = Path(design_dir)
    manifest = json.loads((root / "design_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != DESIGN_SCHEMA or manifest.get("status") != DESIGN_STATUS:
        raise ValueError("Phase-7 design is not frozen")
    if manifest.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("Phase-7 design scientific status changed")
    if manifest.get("contains_model_outcomes") is not False:
        raise ValueError("Phase-7 design contains model outcomes")
    if manifest.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("Phase-7 protocol changed after the design freeze")
    _verify_artifact_index(root, manifest.get("artifact_hashes", {}), "design")
    protocol = load_phase7_protocol(protocol_path)
    prompts = pd.read_csv(root / "prompts.csv", dtype={"prompt_id": str})
    calibration = pd.read_csv(
        root / "calibration_masks.csv",
        dtype={"mask_id": str, "mask_bits": str, "pool_id": str},
    )
    actions = pd.read_csv(
        root / "candidate_actions.csv",
        dtype={
            "mask_id": str,
            "mask_bits": str,
            "pool_id": str,
            "physical_mask_id": str,
        },
    )
    calibration["mask_bits"] = calibration["mask_bits"].astype(str).str.zfill(13)
    calibration["pool_id"] = calibration["pool_id"].fillna("").astype(str)
    actions["mask_bits"] = actions["mask_bits"].astype(str).str.zfill(13)
    actions["is_noop"] = actions["is_noop"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if actions["is_noop"].isna().any():
        raise ValueError("candidate action no-op flags are invalid")
    return manifest, protocol, prompts, calibration, actions


def verify_clean_pretest(
    pretest_dir: str | Path,
    *,
    design_dir: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Require the pre-outcome clean gate to have passed."""

    root = Path(pretest_dir)
    manifest = json.loads((root / "pretest_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PRETEST_SCHEMA:
        raise ValueError("unexpected Phase-7 clean pretest schema")
    if manifest.get("status") != PRETEST_PASS_STATUS:
        raise ValueError("Phase-7 clean pretest did not pass; candidate outcomes stay sealed")
    if manifest.get("design_manifest_sha256") != file_sha256(
        Path(design_dir) / "design_manifest.json"
    ):
        raise ValueError("clean pretest used a different design")
    if manifest.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("clean pretest used a different protocol")
    if manifest.get("candidate_mask_forward_passes") != 0:
        raise ValueError("clean pretest accessed a candidate mask")
    if manifest.get("candidate_outcomes_authorized") is not True:
        raise ValueError("clean pretest did not authorize the freeze stage")
    _verify_artifact_index(root, manifest.get("artifact_hashes", {}), "pretest")
    return manifest


def load_phase5_train_calibration_only(
    effects_dir: str | Path,
    *,
    phase5_prompts: pd.DataFrame,
    phase5_masks: pd.DataFrame,
    effect_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[Path, ...]]:
    """Read only the ten train shards that contain the 160 calibration masks."""

    calibration = phase5_masks.loc[
        phase5_masks["bank"].astype(str) == "calibration"
    ].copy()
    calibration["measurement_order"] = pd.to_numeric(
        calibration["measurement_order"], errors="raise"
    ).astype(int)
    calibration = calibration.sort_values("measurement_order").reset_index(drop=True)
    if len(calibration) != 160:
        raise ValueError("Phase-7 fit requires 160 Phase-5 calibration masks")
    first_ids = phase5_masks.iloc[: len(calibration)]["mask_id"].astype(str).tolist()
    if first_ids != calibration["mask_id"].astype(str).tolist():
        raise ValueError("Phase-5 calibration masks are not the first 160 shard rows")

    root = Path(effects_dir)
    artifact_hashes = effect_manifest.get("artifacts", {})
    paths: list[Path] = []
    frames: list[pd.DataFrame] = []
    train_prompt_ids = phase5_prompts.loc[
        phase5_prompts["split"].astype(str) == "train", "prompt_id"
    ].astype(str).tolist()
    for start in range(0, len(calibration), 16):
        stop = start + 16
        relative = f"shards/train/effects_{start:04d}_{stop:04d}.csv"
        path = root / relative
        if artifact_hashes.get(relative) != file_sha256(path):
            raise ValueError(f"Phase-5 calibration shard changed: {relative}")
        frame = pd.read_csv(
            path,
            dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
        )
        expected_masks = calibration.iloc[start:stop]
        validate_effect_rows(
            frame,
            prompt_ids=train_prompt_ids,
            mask_ids=expected_masks["mask_id"].astype(str).tolist(),
        )
        frames.append(frame)
        paths.append(path)
    train = pd.concat(frames, ignore_index=True)
    _validate_split_cartesian(
        train, phase5_prompts, calibration, split="train"
    )
    return train, tuple(paths)


def fit_and_freeze_phase7_actions(
    design_dir: str | Path,
    pretest_dir: str | Path,
    phase5_design_dir: str | Path,
    phase5_effects_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
) -> Path:
    """Fit on Phase-5 train/calibration outcomes and freeze Phase-7 actions."""

    design_manifest, protocol, _prompts, calibration, candidates = (
        load_verified_phase7_design(design_dir, protocol_path)
    )
    pretest_manifest = verify_clean_pretest(
        pretest_dir, design_dir=design_dir, protocol_path=protocol_path
    )
    phase5_prompts, phase5_masks, _ = load_locked_ioi_design(phase5_design_dir)
    effect_manifest = _validate_effect_manifest(phase5_effects_dir, phase5_design_dir)
    phase5_calibration = phase5_masks.loc[
        phase5_masks["bank"].astype(str) == "calibration"
    ].copy()
    phase5_calibration["measurement_order"] = pd.to_numeric(
        phase5_calibration["measurement_order"], errors="raise"
    ).astype(int)
    phase5_calibration = phase5_calibration.sort_values("measurement_order").reset_index(
        drop=True
    )
    if phase5_calibration["mask_bits"].astype(str).str.zfill(13).tolist() != calibration.sort_values(
        "measurement_order"
    )["mask_bits"].astype(str).str.zfill(13).tolist():
        raise ValueError("Phase-7 calibration bank differs from sealed Phase 5")
    calibration_ids = phase5_calibration["mask_id"].astype(str).tolist()
    train, train_paths = load_phase5_train_calibration_only(
        phase5_effects_dir,
        phase5_prompts=phase5_prompts,
        phase5_masks=phase5_masks,
        effect_manifest=effect_manifest,
    )
    if len(train) != 192 * 160 or set(train["bank"].astype(str)) != {"calibration"}:
        raise ValueError("Phase-7 fit requires the exact Phase-5 train/calibration table")

    ordered_candidates = candidates.sort_values(
        ["pool_index", "is_noop", "n_heads", "sampling_stratum", "mask_id"]
    ).reset_index(drop=True)
    combined = pd.concat(
        [phase5_calibration, ordered_candidates], ignore_index=True, sort=False
    )
    design, terms = _head_quadratic_design(_design_run(combined).masks)
    n_calibration = len(phase5_calibration)
    if design.shape[1] != 92 or np.linalg.matrix_rank(design[:n_calibration]) != 92:
        raise ValueError("Phase-7 quadratic calibration rank gate failed")
    target = float(protocol["target"])
    ridge = float(protocol["ridge"])
    grouped = train.assign(
        target_loss=np.abs(train["drop_from_clean"].to_numpy(float) - target)
    ).groupby("mask_id")
    risk_response = grouped["target_loss"].mean()
    mean_response = grouped["drop_from_clean"].mean()
    y_risk = np.asarray([risk_response[item] for item in calibration_ids], dtype=float)
    y_mean = np.asarray([mean_response[item] for item in calibration_ids], dtype=float)
    beta_risk = ridge_fit(design[:n_calibration], y_risk, ridge)
    beta_mean = ridge_fit(design[:n_calibration], y_mean, ridge)
    candidate_design = design[n_calibration:]
    raw_risk = candidate_design @ beta_risk
    predicted_mean = candidate_design @ beta_mean
    mean_plugin = np.abs(predicted_mean - target)
    noop = ordered_candidates["is_noop"].astype(bool).to_numpy()
    raw_risk[noop] = abs(target)
    mean_plugin[noop] = abs(target)
    predicted_mean[noop] = 0.0

    prediction_rows: list[dict[str, Any]] = []
    for selector, losses, means in (
        (DIRECT_RISK, raw_risk, np.full(len(raw_risk), np.nan)),
        (NATURAL_MEAN, mean_plugin, predicted_mean),
    ):
        prediction_rows.extend(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "selector": selector,
                "target": target,
                "measurement_budget": 160,
                "mask_id": str(row.mask_id),
                "pool_id": str(row.pool_id),
                "is_noop": bool(row.is_noop),
                "n_heads": int(row.n_heads),
                "predicted_target_loss": float(loss),
                "predicted_mean_effect": float(mean) if np.isfinite(mean) else np.nan,
                "noop_loss_set_analytically": bool(row.is_noop),
            }
            for row, loss, mean in zip(
                ordered_candidates.itertuples(index=False), losses, means
            )
        )
    predictions = pd.DataFrame(prediction_rows)

    action_rows: list[dict[str, Any]] = []
    for selector, group in predictions.groupby("selector", sort=True):
        for pool_id, pool in group.groupby("pool_id", sort=True):
            if len(pool) != 31 or int(pool["is_noop"].sum()) != 1:
                raise ValueError("every Phase-7 selection requires 30 masks and one no-op")
            pool = pool.reset_index(drop=True)
            selected = int(
                np.lexsort(
                    (
                        pool["mask_id"].astype(str).to_numpy(),
                        pool["n_heads"].to_numpy(int),
                        pool["predicted_target_loss"].to_numpy(float),
                    )
                )[0]
            )
            row = pool.iloc[selected]
            action_rows.append(
                {
                    "scientific_status": SCIENTIFIC_STATUS,
                    "selector": str(selector),
                    "target": target,
                    "policy": TARGET_POLICY,
                    "pool_id": str(pool_id),
                    "selected_mask_id": str(row["mask_id"]),
                    "selected_is_noop": bool(row["is_noop"]),
                    "selected_head_count": int(row["n_heads"]),
                    "predicted_target_loss": float(row["predicted_target_loss"]),
                }
            )
    for pool_id, pool in ordered_candidates.groupby("pool_id", sort=True):
        noop_row = pool.loc[pool["is_noop"].astype(bool)]
        if len(noop_row) != 1:
            raise ValueError("exact-noop baseline lacks one no-op action")
        row = noop_row.iloc[0]
        action_rows.append(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "selector": EXACT_NOOP,
                "target": target,
                "policy": TARGET_POLICY,
                "pool_id": str(pool_id),
                "selected_mask_id": str(row["mask_id"]),
                "selected_is_noop": True,
                "selected_head_count": 0,
                "predicted_target_loss": target,
            }
        )
    actions = pd.DataFrame(action_rows).sort_values(["selector", "pool_id"]).reset_index(
        drop=True
    )
    if len(actions) != 3 * 48:
        raise AssertionError("Phase-7 fixed action count changed")
    selected_ids = set(
        actions.loc[~actions["selected_is_noop"].astype(bool), "selected_mask_id"].astype(str)
    )
    selected_masks = ordered_candidates.loc[
        ordered_candidates["mask_id"].astype(str).isin(selected_ids)
    ].copy()
    if selected_masks["is_noop"].astype(bool).any():
        raise AssertionError("analytic no-op entered the ablation measurement bank")
    if set(selected_masks["mask_id"].astype(str)) != selected_ids:
        raise ValueError("selected action union is incomplete")

    coefficients = pd.DataFrame(
        [
            {
                "selector": selector,
                "term": term,
                "coefficient": float(value),
                "measurement_budget": 160,
                "ridge": ridge,
            }
            for selector, values in ((DIRECT_RISK, beta_risk), (NATURAL_MEAN, beta_mean))
            for term, value in zip(terms, values)
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "selector": DIRECT_RISK,
                "target": target,
                "design_rank": 92,
                "n_columns": 92,
                "train_calibration_mae": float(
                    np.mean(np.abs(design[:n_calibration] @ beta_risk - y_risk))
                ),
                "candidate_negative_prediction_fraction_before_noop_override": float(
                    (raw_risk[~noop] < 0.0).mean()
                ),
            },
            {
                "selector": NATURAL_MEAN,
                "target": target,
                "design_rank": 92,
                "n_columns": 92,
                "train_calibration_mae": float(
                    np.mean(np.abs(design[:n_calibration] @ beta_mean - y_mean))
                ),
                "candidate_negative_prediction_fraction_before_noop_override": 0.0,
            },
        ]
    )

    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("a Phase-7 action freeze is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "candidate_predictions.csv": predictions,
        "observer_coefficients.csv": coefficients,
        "fit_diagnostics.csv": diagnostics,
        "fixed_actions.csv": actions,
        "selected_measurement_masks.csv": selected_masks,
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    manifest = {
        "schema": FREEZE_SCHEMA,
        "status": FREEZE_STATUS,
        "scientific_status": SCIENTIFIC_STATUS,
        "design_id": design_manifest["design_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "design_manifest_sha256": file_sha256(Path(design_dir) / "design_manifest.json"),
        "pretest_manifest_sha256": file_sha256(Path(pretest_dir) / "pretest_manifest.json"),
        "phase5_design_manifest_sha256": file_sha256(
            Path(phase5_design_dir) / "design_manifest.json"
        ),
        "phase5_effect_manifest_sha256": file_sha256(
            Path(phase5_effects_dir) / "effect_manifest.json"
        ),
        "phase5_train_shard_hashes": {
            path.relative_to(Path(phase5_effects_dir)).as_posix(): file_sha256(path)
            for path in train_paths
        },
        "accessed_outcome_rows": "Phase-5 train/calibration only",
        "phase5_validation_or_test_rows_loaded": False,
        "phase7_candidate_outcomes_loaded": False,
        "phase7_candidate_mask_forward_passes": 0,
        "clean_pretest_passed": True,
        "clean_pretest_gate": pretest_manifest["gate"],
        "target": target,
        "basis_columns": 92,
        "ridge": ridge,
        "counts": {
            "phase5_train_calibration_cells": len(train),
            "candidate_action_rows": len(ordered_candidates),
            "fixed_actions": len(actions),
            "selected_unique_nonnoop_masks": len(selected_masks),
            "analytic_noop_action_rows": int(actions["selected_is_noop"].sum()),
        },
        "artifact_hashes": {
            name: file_sha256(output / name) for name in frames
        },
        "runtime": runtime_provenance(),
        "next_allowed_stage": (
            "After a separate seal validation, measure only selected_measurement_masks.csv "
            "on every frozen test prompt. Do not measure any unselected candidate."
        ),
        "source_effect_manifest_model": effect_manifest["model"],
    }
    write_json(output / "prediction_action_manifest.json", manifest)
    return output


def validate_phase7_freeze(
    freeze_dir: str | Path,
    *,
    design_dir: str | Path,
    pretest_dir: str | Path,
    phase5_design_dir: str | Path,
    phase5_effects_dir: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Validate all Phase-7 pre-outcome seals without reading outcomes."""

    root = Path(freeze_dir)
    manifest = json.loads(
        (root / "prediction_action_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != FREEZE_SCHEMA or manifest.get("status") != FREEZE_STATUS:
        raise ValueError("Phase-7 action freeze is missing")
    expected = {
        "protocol_sha256": file_sha256(protocol_path),
        "design_manifest_sha256": file_sha256(Path(design_dir) / "design_manifest.json"),
        "pretest_manifest_sha256": file_sha256(Path(pretest_dir) / "pretest_manifest.json"),
        "phase5_design_manifest_sha256": file_sha256(
            Path(phase5_design_dir) / "design_manifest.json"
        ),
        "phase5_effect_manifest_sha256": file_sha256(
            Path(phase5_effects_dir) / "effect_manifest.json"
        ),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Phase-7 freeze source changed: {field}")
    if manifest.get("phase7_candidate_outcomes_loaded") is not False:
        raise ValueError("Phase-7 freeze accessed a candidate outcome")
    if int(manifest.get("phase7_candidate_mask_forward_passes", -1)) != 0:
        raise ValueError("Phase-7 freeze ran a candidate intervention")
    verify_clean_pretest(pretest_dir, design_dir=design_dir, protocol_path=protocol_path)
    _verify_artifact_index(root, manifest.get("artifact_hashes", {}), "action freeze")
    actions = pd.read_csv(root / "fixed_actions.csv")
    actions["selected_is_noop"] = actions["selected_is_noop"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if actions["selected_is_noop"].isna().any():
        raise ValueError("fixed-action no-op flags are invalid")
    selected = pd.read_csv(
        root / "selected_measurement_masks.csv", dtype={"mask_id": str, "mask_bits": str}
    )
    selected["mask_bits"] = selected["mask_bits"].astype(str).str.zfill(13)
    required = set(
        actions.loc[~actions["selected_is_noop"].astype(bool), "selected_mask_id"].astype(str)
    )
    if required != set(selected["mask_id"].astype(str)):
        raise ValueError("selected measurement union differs from fixed actions")
    if (selected["mask_bits"] == NOOP_BITS).any():
        raise ValueError("no-op must not enter the intervention measurement bank")
    if len(actions) != 144 or set(actions["selector"].astype(str)) != {
        DIRECT_RISK,
        NATURAL_MEAN,
        EXACT_NOOP,
    }:
        raise ValueError("fixed-action selector table changed")
    if not all(
        len(group) == 48 and group["pool_id"].astype(str).nunique() == 48
        for _, group in actions.groupby("selector")
    ):
        raise ValueError("every selector must freeze one action in each of 48 pools")
    return manifest


def load_legacy_exclusions(
    protocol: Mapping[str, Any],
    root: str | Path,
) -> set[str]:
    """Load the exact legacy mask files named by the protocol."""

    base = Path(root)
    paths: list[Path] = []
    for relative, expected in protocol["legacy_mask_exclusions"].items():
        path = base / str(relative)
        if file_sha256(path) != str(expected):
            raise ValueError(f"legacy mask source changed: {relative}")
        paths.append(path)
    return load_legacy_mask_bits(paths)


def protocol_source_paths(protocol: Mapping[str, Any], root: str | Path) -> dict[str, Path]:
    """Return source paths in the same labels used by the protocol."""

    base = Path(root)
    paths = {
        str(relative): base / str(relative)
        for section in ("pilot_and_source_hashes", "legacy_mask_exclusions")
        for relative in protocol[section]
    }
    repair = protocol.get("v1_aborted_attempt")
    if isinstance(repair, Mapping):
        paths.update(
            {
                str(relative): base / str(relative)
                for relative in repair["artifact_hashes"]
            }
        )
    return paths
