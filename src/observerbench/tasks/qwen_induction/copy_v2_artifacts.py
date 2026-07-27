"""Hash-bound, model-free artifacts for the Qwen induction Copy-v2 study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The candidate serializer accepts an already-built :class:`SequenceDesign` and
cannot run a model.  The preselection serializer accepts only a passed
``CopyV2EligibilityResult`` and freezes the selected prompt tables before any
attention or intervention access.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256
from observerbench.tasks.qwen_induction.artifacts import (
    PRESELECTION_MANIFEST_SCHEMA,
    TOKEN_BANKS_SCHEMA,
    _PROMPTS_ALL_COLUMNS,
    _bank_id_by_name,
    _sequence_row,
)
from observerbench.tasks.qwen_induction.design import (
    SEQUENCE_BANKS,
    SEQUENCE_FAMILIES,
    SequenceDesign,
    validate_sequence_design,
)
from observerbench.tasks.qwen_induction.effect_task import _PROMPT_COLUMNS
from observerbench.tasks.qwen_induction.eligibility import (
    COPY_V2_CANDIDATE_MARGIN_MINIMUM,
    COPY_V2_CELL_COVERAGE_MINIMUM,
    COPY_V2_ELIGIBILITY_SCHEMA,
    COPY_V2_ELIGIBILITY_STATUS_PASS,
    COPY_V2_FAMILY_COVERAGE_MINIMUM,
    COPY_V2_OVERALL_COVERAGE_MINIMUM,
    COPY_V2_RESERVOIR_MULTIPLIER,
    COPY_V2_SELECTION_SEED,
    CopyV2EligibilityResult,
    evaluate_copy_v2_clean_eligibility,
    write_copy_v2_eligibility_artifacts,
)


COPY_V2_CONFIG_SCHEMA = "observerbench.qwen_induction_phase10.v1"
COPY_V2_DATA_VERSION = "copy-v2"
COPY_V2_CANDIDATE_MANIFEST_SCHEMA = (
    "observerbench.qwen_induction.copy_v2.candidate_reservoir.v1"
)
COPY_V2_EXCLUSION_SCHEMA = (
    "observerbench.qwen_induction.copy_v2.copy_v1_token_exclusion.v1"
)
COPY_V2_CANDIDATE_STATUS = "candidate_reservoir_frozen_before_clean_scoring"
COPY_V2_PRESELECTION_STATUS = (
    "clean_eligible_prompts_frozen_before_attention_or_intervention"
)

_CONFIG_TO_DESIGN_BANK = {
    "reference": "reference",
    "discovery": "discovery",
    "head_fit": "head_fit",
    "head_confirmation": "head_test",
    "calibration": "calibration",
    "locked_test": "locked_test",
}
_DESIGN_TO_CONFIG_BANK = {
    design: config for config, design in _CONFIG_TO_DESIGN_BANK.items()
}
_DECISION_COLUMNS = {
    "eligible",
    "eligibility_reason",
    "selection_sha256",
    "selection_rank_within_cell",
    "selected",
}
_ZERO_ACCESS_AUDIT = {
    "attention_scores_loaded": False,
    "intervention_metadata_loaded": False,
    "intervention_outcomes_loaded": False,
    "intervention_forward_passes": 0,
    "candidate_effect_cells": 0,
}


def _load_config(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        config = dict(value)
    else:
        try:
            config = json.loads(Path(value).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Copy-v2 config is missing: {value}") from None
        except json.JSONDecodeError as error:
            raise ValueError("Copy-v2 config is not valid JSON") from error
    if config.get("schema") != COPY_V2_CONFIG_SCHEMA:
        raise ValueError("unexpected Copy-v2 config schema")
    if config.get("data_version") != COPY_V2_DATA_VERSION:
        raise ValueError("Copy-v2 config has an unexpected data version")
    if config.get("predecessor", {}).get("continue_predecessor_run") is not False:
        raise ValueError("Copy-v2 must not continue the Copy-v1 run")
    return config


def _counts_from_config(
    config: Mapping[str, Any], field: str
) -> dict[str, int]:
    try:
        raw = config["sequence_design"][field]
    except (KeyError, TypeError):
        raise ValueError(f"Copy-v2 config lacks sequence_design.{field}") from None
    if not isinstance(raw, Mapping) or set(raw) != set(_CONFIG_TO_DESIGN_BANK):
        raise ValueError(f"sequence_design.{field} must define the six banks")
    counts: dict[str, int] = {}
    for config_bank, design_bank in _CONFIG_TO_DESIGN_BANK.items():
        value = raw[config_bank]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field}.{config_bank} must be a positive integer")
        counts[design_bank] = int(value)
    return counts


def _validated_counts(config: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    final = _counts_from_config(config, "final_prompts_per_family")
    reservoir = _counts_from_config(config, "reservoir_prompts_per_family")
    multiplier = config.get("sequence_design", {}).get("reservoir_multiplier")
    if multiplier != COPY_V2_RESERVOIR_MULTIPLIER:
        raise ValueError("Copy-v2 reservoir multiplier differs from the eligibility protocol")
    for bank in SEQUENCE_BANKS:
        if reservoir[bank] != COPY_V2_RESERVOIR_MULTIPLIER * final[bank]:
            raise ValueError(f"Copy-v2 reservoir is not exactly two-times for {bank}")
    return final, reservoir


def _safe_relative_path(root: Path, label: str) -> Path:
    relative = Path(str(label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {label}")
    return root / relative


def _write_csv_atomic(
    frame: pd.DataFrame, path: Path, columns: Sequence[str]
) -> None:
    if tuple(frame.columns) != tuple(columns):
        raise ValueError(f"unexpected columns for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        columns=list(columns),
        lineterminator="\n",
        float_format="%.17g",
    )
    temporary.replace(path)


def _read_csv_exact(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Copy-v2 artifact is missing: {path}") from None
    if tuple(frame.columns) != tuple(columns):
        raise ValueError(f"unexpected columns in {path}")
    return frame


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _load_copy_v1_token_banks(path: str | Path) -> tuple[dict[str, set[int]], dict[str, Any]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Copy-v1 token-bank artifact is missing: {source}") from None
    except json.JSONDecodeError as error:
        raise ValueError("Copy-v1 token-bank artifact is not valid JSON") from error
    if payload.get("schema") != TOKEN_BANKS_SCHEMA:
        raise ValueError("unexpected Copy-v1 token-bank schema")
    rows = payload.get("banks")
    if not isinstance(rows, list) or {row.get("bank") for row in rows} != set(
        SEQUENCE_BANKS
    ):
        raise ValueError("Copy-v1 token banks must define all six banks")
    by_bank: dict[str, set[int]] = {}
    all_tokens: set[int] = set()
    for row in rows:
        bank = str(row["bank"])
        raw_tokens = row.get("token_ids")
        if not isinstance(raw_tokens, list):
            raise ValueError(f"Copy-v1 token bank {bank} lacks token IDs")
        try:
            tokens = tuple(int(token) for token in raw_tokens)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Copy-v1 token bank {bank} has invalid token IDs") from error
        if any(token < 0 for token in tokens) or len(tokens) != len(set(tokens)):
            raise ValueError(f"Copy-v1 token bank {bank} has invalid or duplicate tokens")
        if int(row.get("token_count", -1)) != len(tokens):
            raise ValueError(f"Copy-v1 token bank {bank} has the wrong token count")
        if str(row.get("token_sha256", "")) != json_sha256(tuple(sorted(tokens))):
            raise ValueError(f"Copy-v1 token bank {bank} has a mismatched token hash")
        if all_tokens.intersection(tokens):
            raise ValueError("Copy-v1 token banks are not disjoint")
        by_bank[bank] = set(tokens)
        all_tokens.update(tokens)
    return by_bank, payload


def _candidate_rows(design: SequenceDesign) -> pd.DataFrame:
    bank_ids = _bank_id_by_name(design)
    rows = [
        _sequence_row(
            example,
            token_bank_id=bank_ids[example.bank],
            split=None,
        )
        for example in design.examples
    ]
    return pd.DataFrame(rows, columns=_PROMPTS_ALL_COLUMNS)


def _candidate_token_payload(design: SequenceDesign) -> dict[str, Any]:
    bank_ids = _bank_id_by_name(design)
    return {
        "schema": TOKEN_BANKS_SCHEMA,
        "data_version": COPY_V2_DATA_VERSION,
        "role": "preassigned_candidate_reservoir",
        "sequence_design_sha256": design.design_sha256,
        "token_pool_size": design.token_pool_size,
        "token_pool_sha256": design.token_pool_sha256,
        "per_split_size": design.per_split_size,
        "banks": [
            {
                "bank": bank.bank,
                "token_bank_id": bank_ids[bank.bank],
                "token_ids": bank.token_ids,
                "token_count": bank.token_count,
                "token_sha256": bank.token_sha256,
            }
            for bank in design.token_banks
        ],
    }


def _zero_access(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} lacks an access audit")
    for key, expected in _ZERO_ACCESS_AUDIT.items():
        if value.get(key) != expected:
            raise ValueError(f"{label} does not certify zero intervention access: {key}")


def write_copy_v2_candidate_artifacts(
    sequence_design: SequenceDesign,
    config: Mapping[str, Any] | str | Path,
    artifacts_root: str | Path,
    *,
    copy_v1_token_banks_path: str | Path,
    scientific_claim_allowed: bool = True,
) -> Path:
    """Serialize the frozen two-times candidate reservoirs without model access."""

    frozen = _load_config(config)
    _final, reservoir = _validated_counts(frozen)
    validate_sequence_design(sequence_design)
    sequence = frozen.get("sequence_design", {})
    token = frozen.get("token_pool", {})
    if sequence_design.seed != int(sequence.get("seed", -1)):
        raise ValueError("candidate sequence seed differs from the Copy-v2 config")
    if sequence_design.per_split_size != int(token.get("per_split_size", -1)):
        raise ValueError("candidate token-bank size differs from the Copy-v2 config")
    if sequence_design.bank_counts.as_dict() != reservoir:
        raise ValueError("SequenceDesign does not contain the registered two-times reservoirs")

    copy_v1_source = Path(copy_v1_token_banks_path)
    exclusion_config = token.get("exclude_copy_v1_allocated_tokens", {})
    copy_v1_by_bank, copy_v1_payload = _load_copy_v1_token_banks(copy_v1_source)
    if (
        file_sha256(copy_v1_source)
        != str(exclusion_config.get("source_artifact_sha256", ""))
        or int(copy_v1_payload.get("token_pool_size", -1))
        != int(exclusion_config.get("source_token_pool_size", -2))
        or str(copy_v1_payload.get("token_pool_sha256", ""))
        != str(exclusion_config.get("source_token_pool_sha256", ""))
    ):
        raise ValueError("Copy-v1 token-bank source differs from the config binding")
    excluded = set().union(*copy_v1_by_bank.values())
    candidate_tokens = {
        token_id
        for bank in sequence_design.token_banks
        for token_id in bank.token_ids
    }
    overlap = excluded.intersection(candidate_tokens)
    if overlap:
        raise ValueError(
            f"Copy-v2 candidate token banks reuse {len(overlap)} Copy-v1 tokens"
        )

    root = Path(artifacts_root)
    final_output = root / "candidate_design"
    if final_output.exists():
        existing = verify_copy_v2_candidate_artifacts(root, frozen)
        if existing.get("sequence_design_sha256") != sequence_design.design_sha256:
            raise ValueError("existing candidate reservoir differs from recomputation")
        return final_output / "candidate_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    output = root / ".candidate_design.incomplete"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    prompts_path = output / "prompts_all.csv"
    token_banks_path = output / "token_banks.json"
    copy_v1_path = output / "copy_v1_token_banks.json"
    exclusion_path = output / "copy_v1_token_exclusion.json"

    _write_csv_atomic(_candidate_rows(sequence_design), prompts_path, _PROMPTS_ALL_COLUMNS)
    write_json(token_banks_path, _candidate_token_payload(sequence_design))
    write_json(copy_v1_path, copy_v1_payload)
    exclusion_payload = {
        "schema": COPY_V2_EXCLUSION_SCHEMA,
        "status": "copy_v1_allocated_tokens_excluded_before_candidate_generation",
        "source_data_version": "copy-v1",
        "source_artifact_sha256": file_sha256(copy_v1_path),
        "banks": {
            bank: {
                "token_count": len(tokens),
                "token_ids_sha256": json_sha256(tuple(sorted(tokens))),
            }
            for bank, tokens in sorted(copy_v1_by_bank.items())
        },
        "excluded_token_count": len(excluded),
        "excluded_token_ids_sha256": json_sha256(tuple(sorted(excluded))),
        "candidate_overlap_count": 0,
    }
    write_json(exclusion_path, exclusion_payload)
    artifacts = (prompts_path, token_banks_path, copy_v1_path, exclusion_path)
    manifest = {
        "schema": COPY_V2_CANDIDATE_MANIFEST_SCHEMA,
        "status": COPY_V2_CANDIDATE_STATUS,
        "data_version": COPY_V2_DATA_VERSION,
        "scientific_claim_allowed": bool(scientific_claim_allowed),
        "config_sha256": json_sha256(frozen),
        "sequence_design_sha256": sequence_design.design_sha256,
        "reservoir_multiplier": COPY_V2_RESERVOIR_MULTIPLIER,
        "reservoir_counts_per_family": dict(reservoir),
        "prompt_count": len(sequence_design.examples),
        "copy_v1_excluded_token_count": len(excluded),
        "artifact_hashes": {
            path.name: file_sha256(path) for path in artifacts
        },
        "access_audit": {
            "model_forward_passes": 0,
            **_ZERO_ACCESS_AUDIT,
        },
    }
    manifest_path = output / "candidate_manifest.json"
    write_json(manifest_path, manifest)
    output.replace(final_output)
    verify_copy_v2_candidate_artifacts(root, frozen)
    return final_output / "candidate_manifest.json"


def _verified_hashes(root: Path, manifest: Mapping[str, Any]) -> None:
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("Copy-v2 manifest has no artifact hashes")
    for label, expected in hashes.items():
        path = _safe_relative_path(root, str(label))
        if not path.is_file() or file_sha256(path) != str(expected):
            raise ValueError(f"Copy-v2 artifact hash mismatch: {label}")


def _require_hash_labels(
    manifest: Mapping[str, Any], required: set[str], label: str
) -> None:
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, Mapping) or not required.issubset(hashes):
        missing = sorted(required - set(hashes or {}))
        raise ValueError(f"{label} does not bind required artifacts: {missing}")


def _validate_candidate_rows(
    rows: pd.DataFrame,
    token_payload: Mapping[str, Any],
    reservoir: Mapping[str, int],
    excluded: set[int],
) -> None:
    if rows.empty or rows["prompt_id"].duplicated().any():
        raise ValueError("candidate prompt IDs must be nonempty and globally unique")
    expected_families = {family.family_id for family in SEQUENCE_FAMILIES}
    if set(rows["bank"]) != set(SEQUENCE_BANKS) or set(rows["family_id"]) != expected_families:
        raise ValueError("candidate table has unexpected banks or families")
    counts = rows.groupby(["bank", "family_id"]).size()
    for bank in SEQUENCE_BANKS:
        for family in expected_families:
            if int(counts.get((bank, family), 0)) != int(reservoir[bank]):
                raise ValueError(f"candidate reservoir count changed for {bank}/{family}")

    bank_rows = token_payload.get("banks")
    if not isinstance(bank_rows, list) or {row.get("bank") for row in bank_rows} != set(
        SEQUENCE_BANKS
    ):
        raise ValueError("candidate token-bank artifact does not contain six banks")
    bank_tokens: dict[str, set[int]] = {}
    bank_ids: dict[str, str] = {}
    all_tokens: set[int] = set()
    for raw in bank_rows:
        bank = str(raw["bank"])
        tokens = {int(value) for value in raw.get("token_ids", [])}
        if len(tokens) != int(raw.get("token_count", -1)):
            raise ValueError(f"candidate token count changed for {bank}")
        if str(raw.get("token_sha256")) != json_sha256(tuple(sorted(tokens))):
            raise ValueError(f"candidate token hash changed for {bank}")
        if all_tokens.intersection(tokens):
            raise ValueError("candidate token banks are not disjoint")
        if excluded.intersection(tokens):
            raise ValueError("candidate token banks contain excluded Copy-v1 tokens")
        bank_tokens[bank] = tokens
        bank_ids[bank] = str(raw.get("token_bank_id", ""))
        all_tokens.update(tokens)

    for row in rows.itertuples(index=False):
        bank = str(row.bank)
        try:
            tokens = tuple(map(int, str(row.input_ids).split()))
            key_positions = tuple(map(int, str(row.key_positions).split()))
            source_values = tuple(
                map(int, str(row.source_value_positions).split())
            )
        except ValueError as error:
            raise ValueError("candidate prompt contains invalid integer metadata") from error
        sequence_length = int(row.sequence_length)
        query = int(row.query_position)
        if (
            len(tokens) != sequence_length
            or query != sequence_length - 1
            or len(key_positions) != 3
            or len(source_values) != 3
            or source_values != tuple(position + 1 for position in key_positions)
        ):
            raise ValueError("candidate prompt structure changed")
        if str(row.token_bank_id) != bank_ids[bank] or not set(tokens).issubset(
            bank_tokens[bank]
        ):
            raise ValueError("candidate prompt crossed its frozen token bank")
        source_key = int(row.source_key_position)
        source_value = int(row.source_value_position)
        target = int(row.target_token_id)
        distractors = (
            int(row.distractor_token_id_1),
            int(row.distractor_token_id_2),
        )
        if (
            source_value != source_key + 1
            or tokens[query] != tokens[source_key]
            or tokens[source_value] != target
            or tuple(tokens[position] for position in source_values[1:]) != distractors
            or len(set(tokens[:-1])) != len(tokens) - 1
        ):
            raise ValueError("candidate induction fixture changed")


def verify_copy_v2_candidate_artifacts(
    artifacts_root: str | Path,
    config: Mapping[str, Any] | str | Path,
) -> Mapping[str, Any]:
    """Verify hashes, two-times cells, token exclusion, and zero outcome access."""

    frozen = _load_config(config)
    _final, reservoir = _validated_counts(frozen)
    root = Path(artifacts_root)
    output = root / "candidate_design"
    try:
        manifest = json.loads(
            (output / "candidate_manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise FileNotFoundError("Copy-v2 candidate manifest is missing") from None
    if (
        manifest.get("schema") != COPY_V2_CANDIDATE_MANIFEST_SCHEMA
        or manifest.get("status") != COPY_V2_CANDIDATE_STATUS
        or manifest.get("data_version") != COPY_V2_DATA_VERSION
        or manifest.get("config_sha256") != json_sha256(frozen)
        or not isinstance(manifest.get("scientific_claim_allowed"), bool)
    ):
        raise ValueError("Copy-v2 candidate manifest identity changed")
    if manifest.get("reservoir_counts_per_family") != reservoir:
        raise ValueError("Copy-v2 candidate manifest reservoir counts changed")
    _zero_access(manifest.get("access_audit"), "candidate manifest")
    if manifest["access_audit"].get("model_forward_passes") != 0:
        raise ValueError("candidate reservoir was not outcome-free")
    _require_hash_labels(
        manifest,
        {
            "prompts_all.csv",
            "token_banks.json",
            "copy_v1_token_banks.json",
            "copy_v1_token_exclusion.json",
        },
        "candidate manifest",
    )
    _verified_hashes(output, manifest)

    rows = _read_csv_exact(output / "prompts_all.csv", _PROMPTS_ALL_COLUMNS)
    token_payload = json.loads((output / "token_banks.json").read_text(encoding="utf-8"))
    if (
        token_payload.get("schema") != TOKEN_BANKS_SCHEMA
        or token_payload.get("data_version") != COPY_V2_DATA_VERSION
        or token_payload.get("role") != "preassigned_candidate_reservoir"
        or token_payload.get("sequence_design_sha256")
        != manifest.get("sequence_design_sha256")
    ):
        raise ValueError("candidate token-bank identity changed")
    copy_v1_by_bank, copy_v1_payload = _load_copy_v1_token_banks(
        output / "copy_v1_token_banks.json"
    )
    exclusion_config = frozen["token_pool"]["exclude_copy_v1_allocated_tokens"]
    if (
        file_sha256(output / "copy_v1_token_banks.json")
        != str(exclusion_config["source_artifact_sha256"])
        or int(copy_v1_payload.get("token_pool_size", -1))
        != int(exclusion_config["source_token_pool_size"])
        or str(copy_v1_payload.get("token_pool_sha256", ""))
        != str(exclusion_config["source_token_pool_sha256"])
    ):
        raise ValueError("copied Copy-v1 token banks differ from the config binding")
    excluded = set().union(*copy_v1_by_bank.values())
    exclusion = json.loads(
        (output / "copy_v1_token_exclusion.json").read_text(encoding="utf-8")
    )
    if (
        exclusion.get("schema") != COPY_V2_EXCLUSION_SCHEMA
        or exclusion.get("candidate_overlap_count") != 0
        or exclusion.get("excluded_token_count") != len(excluded)
        or exclusion.get("excluded_token_ids_sha256")
        != json_sha256(tuple(sorted(excluded)))
        or exclusion.get("source_artifact_sha256")
        != file_sha256(output / "copy_v1_token_banks.json")
    ):
        raise ValueError("Copy-v1 token exclusion proof changed")
    _validate_candidate_rows(rows, token_payload, reservoir, excluded)
    if int(manifest.get("prompt_count", -1)) != len(rows):
        raise ValueError("candidate prompt count differs from its manifest")
    return manifest


def load_copy_v2_candidate_reservoir(
    artifacts_root: str | Path,
    config: Mapping[str, Any] | str | Path,
) -> pd.DataFrame:
    """Load the verified model-free reservoir in the existing prompt schema."""

    verify_copy_v2_candidate_artifacts(artifacts_root, config)
    return _read_csv_exact(
        Path(artifacts_root) / "candidate_design" / "prompts_all.csv",
        _PROMPTS_ALL_COLUMNS,
    )


def _eligibility_protocol_matches_config(
    result: CopyV2EligibilityResult,
    config: Mapping[str, Any],
    final_counts: Mapping[str, int],
) -> None:
    if not result.passed or result.summary.get("status") != COPY_V2_ELIGIBILITY_STATUS_PASS:
        raise ValueError("Copy-v2 preselection requires a passed clean eligibility result")
    if result.summary.get("schema") != COPY_V2_ELIGIBILITY_SCHEMA:
        raise ValueError("unexpected Copy-v2 eligibility schema")
    protocol = result.summary.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("eligibility result lacks protocol metadata")
    eligibility_config = config.get("clean_eligibility", {})
    coverage_config = eligibility_config.get("coverage_gate", {})
    selection_config = eligibility_config.get("selection", {})
    expected = {
        "reservoir_multiplier": COPY_V2_RESERVOIR_MULTIPLIER,
        "candidate_margin_minimum": COPY_V2_CANDIDATE_MARGIN_MINIMUM,
        "selection_seed": COPY_V2_SELECTION_SEED,
    }
    for key, value in expected.items():
        observed = protocol.get(key)
        if isinstance(value, float):
            if not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"eligibility protocol changed: {key}")
        elif observed != value:
            raise ValueError(f"eligibility protocol changed: {key}")
    if (
        not math.isclose(
            float(eligibility_config.get("minimum_candidate_margin")),
            COPY_V2_CANDIDATE_MARGIN_MINIMUM,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or selection_config.get("seed") != COPY_V2_SELECTION_SEED
        or selection_config.get("method") != protocol.get("selection_rule")
    ):
        raise ValueError("eligibility result does not match the Copy-v2 config")
    minima = protocol.get("coverage_minima")
    if not isinstance(minima, Mapping) or any(
        not math.isclose(float(minima.get(label, -1.0)), expected_value)
        for label, expected_value in (
            ("overall", COPY_V2_OVERALL_COVERAGE_MINIMUM),
            ("family_pooled", COPY_V2_FAMILY_COVERAGE_MINIMUM),
            ("bank_family", COPY_V2_CELL_COVERAGE_MINIMUM),
        )
    ):
        raise ValueError("eligibility coverage minima changed")
    if (
        float(coverage_config.get("minimum_overall", -1.0))
        != COPY_V2_OVERALL_COVERAGE_MINIMUM
        or float(coverage_config.get("minimum_per_family_pooled_across_banks", -1.0))
        != COPY_V2_FAMILY_COVERAGE_MINIMUM
        or float(coverage_config.get("minimum_per_bank_family", -1.0))
        != COPY_V2_CELL_COVERAGE_MINIMUM
        or protocol.get("required_counts_per_family") != dict(final_counts)
    ):
        raise ValueError("eligibility coverage protocol differs from config")
    _zero_access(result.summary.get("access_audit"), "eligibility result")


def _recompute_eligibility(
    candidates: pd.DataFrame,
    result: CopyV2EligibilityResult,
    final_counts: Mapping[str, int],
) -> tuple[CopyV2EligibilityResult, pd.DataFrame]:
    decisions = result.decisions.copy()
    if set(_PROMPTS_ALL_COLUMNS) - set(decisions):
        raise ValueError("eligibility decisions do not retain the candidate prompt schema")
    supplied_candidates = decisions.loc[:, _PROMPTS_ALL_COLUMNS].astype(str)
    expected_candidates = candidates.astype(str)
    supplied_candidates = supplied_candidates.sort_values("prompt_id").reset_index(drop=True)
    expected_candidates = expected_candidates.sort_values("prompt_id").reset_index(drop=True)
    if not supplied_candidates.equals(expected_candidates):
        raise ValueError("eligibility decisions differ from the frozen candidate reservoir")
    score_columns = [
        column
        for column in decisions.columns
        if column not in set(_PROMPTS_ALL_COLUMNS) | _DECISION_COLUMNS
    ]
    if not {"candidate_correct", "candidate_margin"}.issubset(score_columns):
        raise ValueError("eligibility decisions do not retain the clean scores")
    scores = decisions.loc[:, ["prompt_id", *score_columns]].copy()
    recomputed = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=final_counts,
    )
    for field in (
        "schema",
        "status",
        "passed",
        "protocol",
        "gates",
        "counts",
        "input_hashes",
        "access_audit",
        "next_allowed_stage",
    ):
        if result.summary.get(field) != recomputed.summary.get(field):
            raise ValueError(
                f"eligibility manifest {field} differs from clean-only recomputation"
            )
    for supplied, expected, label in (
        (result.decisions, recomputed.decisions, "decisions"),
        (result.selected_ids, recomputed.selected_ids, "selection"),
        (result.coverage, recomputed.coverage, "coverage"),
    ):
        if tuple(supplied.columns) != tuple(expected.columns):
            raise ValueError(f"eligibility {label} differs from clean-only recomputation")
        left = supplied.copy().fillna("").reset_index(drop=True)
        right = expected.copy().fillna("").reset_index(drop=True)
        if label == "decisions":
            for column in _PROMPTS_ALL_COLUMNS:
                left[column] = left[column].astype(str)
                right[column] = right[column].astype(str)
        try:
            pd.testing.assert_frame_equal(
                left,
                right,
                check_dtype=False,
                check_exact=False,
                rtol=0.0,
                atol=1e-15,
            )
        except AssertionError as error:
            raise ValueError(
                f"eligibility {label} differs from clean-only recomputation"
            ) from error
    return recomputed, scores


def _selected_prompt_frames(
    candidates: pd.DataFrame,
    selected_ids: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected_ids["prompt_id"].astype(str).duplicated().any():
        raise ValueError("selected prompt IDs are duplicated")
    selection = selected_ids[
        ["prompt_id", "bank", "family_id", "selection_rank_within_cell"]
    ].copy()
    merged = candidates.merge(
        selection,
        on=["prompt_id", "bank", "family_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(selection):
        raise ValueError("selected prompt IDs are outside the candidate reservoir")
    merged["selection_rank_within_cell"] = pd.to_numeric(
        merged["selection_rank_within_cell"], errors="raise"
    ).astype(int)
    merged["cluster_id"] = [
        f"qcluster_{bank}_{rank - 1:04d}"
        for bank, rank in zip(merged["bank"], merged["selection_rank_within_cell"])
    ]
    bank_order = {bank: index for index, bank in enumerate(SEQUENCE_BANKS)}
    family_order = {
        family.family_id: index for index, family in enumerate(SEQUENCE_FAMILIES)
    }
    merged["_bank_order"] = merged["bank"].map(bank_order)
    merged["_family_order"] = merged["family_id"].map(family_order)
    merged = merged.sort_values(
        ["_bank_order", "_family_order", "selection_rank_within_cell"],
        kind="mergesort",
    ).drop(columns=["_bank_order", "_family_order", "selection_rank_within_cell"])
    all_prompts = merged.loc[:, _PROMPTS_ALL_COLUMNS].reset_index(drop=True)

    adapter = all_prompts.loc[
        all_prompts["bank"].isin(("calibration", "locked_test"))
    ].copy()
    adapter.insert(
        1,
        "split",
        adapter["bank"].map({"calibration": "train", "locked_test": "test"}),
    )
    adapter = adapter.drop(
        columns=["bank", "key_positions", "source_value_positions"]
    ).loc[:, _PROMPT_COLUMNS]
    return all_prompts, adapter.reset_index(drop=True)


def _normalized_source_hashes(source_hashes: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("Copy-v2 preselection requires producer source hashes")
    normalized: dict[str, str] = {}
    for label, digest in source_hashes.items():
        name = str(label)
        value = str(digest)
        if not name or len(value) != 64:
            raise ValueError("producer source hashes require nonempty labels and SHA-256 values")
        try:
            int(value, 16)
        except ValueError:
            raise ValueError("producer source hash is not hexadecimal") from None
        normalized[name] = value.lower()
    return dict(sorted(normalized.items()))


def write_copy_v2_preselection_artifacts(
    result: CopyV2EligibilityResult,
    config: Mapping[str, Any] | str | Path,
    artifacts_root: str | Path,
    *,
    runtime_record: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> Path:
    """Freeze passed clean-only selections in the existing prompt schemas."""

    frozen = _load_config(config)
    final_counts, _reservoir = _validated_counts(frozen)
    _eligibility_protocol_matches_config(result, frozen, final_counts)
    root = Path(artifacts_root)
    candidate_manifest = verify_copy_v2_candidate_artifacts(root, frozen)
    candidates = load_copy_v2_candidate_reservoir(root, frozen)
    recomputed, _scores = _recompute_eligibility(candidates, result, final_counts)
    if not recomputed.passed:
        raise ValueError("recomputed Copy-v2 eligibility did not pass")

    final_output = root / "design"
    if final_output.exists():
        existing = verify_copy_v2_preselection_artifacts(root, frozen)
        existing_clean_hash = existing.get("clean_eligibility", {}).get(
            "clean_scores_canonical_sha256"
        )
        recomputed_clean_hash = recomputed.summary["input_hashes"][
            "clean_scores_sha256"
        ]
        if existing_clean_hash != recomputed_clean_hash:
            raise ValueError("existing preselection differs from recomputation")
        return final_output / "preselection_manifest.json"
    output = root / ".design.incomplete"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    eligibility_dir = output / "eligibility"
    eligibility_manifest = write_copy_v2_eligibility_artifacts(
        recomputed, eligibility_dir
    )
    all_prompts, adapter_prompts = _selected_prompt_frames(
        candidates, recomputed.selected_ids
    )
    prompts_all_path = output / "prompts_all.csv"
    prompts_path = output / "prompts.csv"
    token_banks_path = output / "token_banks.json"
    _write_csv_atomic(all_prompts, prompts_all_path, _PROMPTS_ALL_COLUMNS)
    _write_csv_atomic(adapter_prompts, prompts_path, _PROMPT_COLUMNS)
    _copy_atomic(root / "candidate_design" / "token_banks.json", token_banks_path)

    runtime = dict(runtime_record)
    if not runtime:
        raise ValueError("Copy-v2 preselection requires a nonempty runtime record")
    normalized_sources = _normalized_source_hashes(source_hashes)
    artifact_paths = (
        prompts_all_path,
        prompts_path,
        token_banks_path,
        *sorted(path for path in eligibility_dir.iterdir() if path.is_file()),
    )
    eligibility_payload = json.loads(eligibility_manifest.read_text(encoding="utf-8"))
    _zero_access(eligibility_payload.get("access_audit"), "eligibility manifest")
    manifest = {
        "schema": PRESELECTION_MANIFEST_SCHEMA,
        "status": COPY_V2_PRESELECTION_STATUS,
        "data_version": COPY_V2_DATA_VERSION,
        "scientific_claim_allowed": bool(
            runtime.get("scientific_claim_allowed", False)
        ),
        "scientific_outcomes_included": False,
        "attention_outputs_included": False,
        "intervention_outcomes_included": False,
        "config_sha256": json_sha256(frozen),
        "sequence_design_sha256": candidate_manifest["sequence_design_sha256"],
        "candidate_manifest": {
            "path": "../candidate_design/candidate_manifest.json",
            "sha256": file_sha256(
                root / "candidate_design" / "candidate_manifest.json"
            ),
        },
        "clean_eligibility": {
            "manifest_path": "eligibility/eligibility_manifest.json",
            "manifest_sha256": file_sha256(eligibility_manifest),
            "clean_scores_canonical_sha256": eligibility_payload["input_hashes"][
                "clean_scores_sha256"
            ],
            "coverage_artifact": "eligibility/coverage.csv",
            "selection_artifact": "eligibility/selected_prompt_ids.csv",
            "decision_artifact": "eligibility/candidate_decisions.csv",
            "passed": True,
        },
        "prompt_counts": {
            bank: int((all_prompts["bank"] == bank).sum()) for bank in SEQUENCE_BANKS
        },
        "runtime_record": runtime,
        "runtime_record_sha256": json_sha256(runtime),
        "producer_source_hashes": normalized_sources,
        "producer_source_hashes_sha256": json_sha256(normalized_sources),
        "artifact_hashes": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in artifact_paths
        },
        "access_audit": {
            "clean_scores_loaded": True,
            **_ZERO_ACCESS_AUDIT,
        },
        "next_allowed_stage": "head_discovery",
    }
    manifest_path = output / "preselection_manifest.json"
    write_json(manifest_path, manifest)
    output.replace(final_output)
    verify_copy_v2_preselection_artifacts(root, frozen)
    return final_output / "preselection_manifest.json"


def verify_copy_v2_preselection_artifacts(
    artifacts_root: str | Path,
    config: Mapping[str, Any] | str | Path,
) -> Mapping[str, Any]:
    """Verify selected prompts, clean decision binding, and zero interventions."""

    frozen = _load_config(config)
    final_counts, _reservoir = _validated_counts(frozen)
    root = Path(artifacts_root)
    candidate_manifest = verify_copy_v2_candidate_artifacts(root, frozen)
    output = root / "design"
    try:
        manifest = json.loads(
            (output / "preselection_manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise FileNotFoundError("Copy-v2 preselection manifest is missing") from None
    if (
        manifest.get("schema") != PRESELECTION_MANIFEST_SCHEMA
        or manifest.get("status") != COPY_V2_PRESELECTION_STATUS
        or manifest.get("data_version") != COPY_V2_DATA_VERSION
        or manifest.get("config_sha256") != json_sha256(frozen)
        or manifest.get("sequence_design_sha256")
        != candidate_manifest.get("sequence_design_sha256")
    ):
        raise ValueError("Copy-v2 preselection manifest identity changed")
    if any(
        manifest.get(field) is not False
        for field in (
            "scientific_outcomes_included",
            "attention_outputs_included",
            "intervention_outcomes_included",
        )
    ):
        raise ValueError("Copy-v2 preselection contains forbidden outcomes")
    _zero_access(manifest.get("access_audit"), "preselection manifest")
    if manifest["access_audit"].get("clean_scores_loaded") is not True:
        raise ValueError("Copy-v2 preselection did not bind clean scores")
    _require_hash_labels(
        manifest,
        {
            "prompts_all.csv",
            "prompts.csv",
            "token_banks.json",
            "eligibility/eligibility_manifest.json",
            "eligibility/candidate_decisions.csv",
            "eligibility/selected_prompt_ids.csv",
            "eligibility/coverage.csv",
        },
        "preselection manifest",
    )
    _verified_hashes(output, manifest)
    candidate_binding = manifest.get("candidate_manifest")
    if (
        not isinstance(candidate_binding, Mapping)
        or candidate_binding.get("path")
        != "../candidate_design/candidate_manifest.json"
        or candidate_binding.get("sha256")
        != file_sha256(root / "candidate_design" / "candidate_manifest.json")
    ):
        raise ValueError("preselection is not bound to the candidate manifest")

    eligibility_binding = manifest.get("clean_eligibility")
    if not isinstance(eligibility_binding, Mapping) or eligibility_binding.get(
        "passed"
    ) is not True:
        raise ValueError("preselection is not bound to passed eligibility")
    expected_eligibility_paths = {
        "manifest_path": "eligibility/eligibility_manifest.json",
        "coverage_artifact": "eligibility/coverage.csv",
        "selection_artifact": "eligibility/selected_prompt_ids.csv",
        "decision_artifact": "eligibility/candidate_decisions.csv",
    }
    if any(
        eligibility_binding.get(field) != expected
        for field, expected in expected_eligibility_paths.items()
    ):
        raise ValueError("preselection clean-eligibility paths changed")
    eligibility_manifest_path = _safe_relative_path(
        output, str(eligibility_binding.get("manifest_path", ""))
    )
    if file_sha256(eligibility_manifest_path) != eligibility_binding.get(
        "manifest_sha256"
    ):
        raise ValueError("eligibility manifest hash changed")
    eligibility_payload = json.loads(
        eligibility_manifest_path.read_text(encoding="utf-8")
    )
    if (
        eligibility_payload.get("schema") != COPY_V2_ELIGIBILITY_SCHEMA
        or eligibility_payload.get("status") != COPY_V2_ELIGIBILITY_STATUS_PASS
        or eligibility_payload.get("passed") is not True
        or eligibility_binding.get("clean_scores_canonical_sha256")
        != eligibility_payload.get("input_hashes", {}).get("clean_scores_sha256")
    ):
        raise ValueError("preselection clean-score or eligibility binding changed")
    _zero_access(eligibility_payload.get("access_audit"), "eligibility manifest")
    eligibility_hashes = eligibility_payload.get("artifact_hashes")
    if not isinstance(eligibility_hashes, Mapping):
        raise ValueError("eligibility manifest has no artifact hashes")
    for label, expected in eligibility_hashes.items():
        path = _safe_relative_path(eligibility_manifest_path.parent, str(label))
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"eligibility artifact hash mismatch: {label}")

    sources = _normalized_source_hashes(manifest.get("producer_source_hashes", {}))
    if json_sha256(sources) != manifest.get("producer_source_hashes_sha256"):
        raise ValueError("preselection source-hash binding changed")
    runtime = manifest.get("runtime_record")
    if not isinstance(runtime, Mapping) or not runtime:
        raise ValueError("preselection runtime record is missing")
    if json_sha256(runtime) != manifest.get("runtime_record_sha256"):
        raise ValueError("preselection runtime binding changed")
    if (
        manifest.get("scientific_claim_allowed")
        is not runtime.get("scientific_claim_allowed")
        or manifest.get("scientific_claim_allowed")
        is not candidate_manifest.get("scientific_claim_allowed")
    ):
        raise ValueError("preselection scientific-claim mode changed")

    if file_sha256(output / "token_banks.json") != file_sha256(
        root / "candidate_design" / "token_banks.json"
    ):
        raise ValueError("selected design token banks differ from the candidate reservoir")
    all_prompts = _read_csv_exact(output / "prompts_all.csv", _PROMPTS_ALL_COLUMNS)
    adapter = _read_csv_exact(output / "prompts.csv", _PROMPT_COLUMNS)
    selected = _read_csv_exact(
        output / "eligibility" / "selected_prompt_ids.csv",
        (
            "prompt_id",
            "bank",
            "family_id",
            "selection_rank_within_cell",
            "selection_sha256",
        ),
    )
    candidates = load_copy_v2_candidate_reservoir(root, frozen)
    decision_frame = pd.read_csv(
        output / "eligibility" / "candidate_decisions.csv",
        float_precision="round_trip",
    )
    selected_frame = pd.read_csv(
        output / "eligibility" / "selected_prompt_ids.csv"
    )
    coverage_frame = pd.read_csv(output / "eligibility" / "coverage.csv")
    serialized_result = CopyV2EligibilityResult(
        passed=True,
        decisions=decision_frame,
        selected_ids=selected_frame,
        coverage=coverage_frame,
        summary=eligibility_payload,
    )
    _eligibility_protocol_matches_config(
        serialized_result, frozen, final_counts
    )
    recomputed, _scores = _recompute_eligibility(
        candidates, serialized_result, final_counts
    )
    expected_all, expected_adapter = _selected_prompt_frames(
        candidates, recomputed.selected_ids
    )
    if all_prompts.to_csv(index=False) != expected_all.astype(str).to_csv(index=False):
        raise ValueError("selected prompt contents differ from clean eligibility")
    if adapter.to_csv(index=False) != expected_adapter.astype(str).to_csv(index=False):
        raise ValueError("adapter prompt contents differ from clean eligibility")
    expected_total = 0
    expected_families = {family.family_id for family in SEQUENCE_FAMILIES}
    counts = all_prompts.groupby(["bank", "family_id"]).size()
    for bank in SEQUENCE_BANKS:
        expected_total += final_counts[bank] * len(expected_families)
        for family in expected_families:
            if int(counts.get((bank, family), 0)) != final_counts[bank]:
                raise ValueError(f"selected prompt count changed for {bank}/{family}")
    if (
        len(all_prompts) != expected_total
        or set(all_prompts["prompt_id"]) != set(selected["prompt_id"])
        or all_prompts["prompt_id"].duplicated().any()
    ):
        raise ValueError("selected prompt table differs from clean eligibility")
    expected_adapter = all_prompts.loc[
        all_prompts["bank"].isin(("calibration", "locked_test")), "prompt_id"
    ]
    if set(adapter["prompt_id"]) != set(expected_adapter):
        raise ValueError("adapter prompt table differs from selected train/test prompts")
    if set(adapter["split"]) != {"train", "test"}:
        raise ValueError("adapter prompt table lost its train/test split")
    if manifest.get("prompt_counts") != {
        bank: int((all_prompts["bank"] == bank).sum()) for bank in SEQUENCE_BANKS
    }:
        raise ValueError("preselection prompt counts differ from selected prompts")
    return manifest


__all__ = [
    "COPY_V2_CANDIDATE_MANIFEST_SCHEMA",
    "COPY_V2_CANDIDATE_STATUS",
    "COPY_V2_CONFIG_SCHEMA",
    "COPY_V2_DATA_VERSION",
    "COPY_V2_EXCLUSION_SCHEMA",
    "COPY_V2_PRESELECTION_STATUS",
    "load_copy_v2_candidate_reservoir",
    "verify_copy_v2_candidate_artifacts",
    "verify_copy_v2_preselection_artifacts",
    "write_copy_v2_candidate_artifacts",
    "write_copy_v2_preselection_artifacts",
]
