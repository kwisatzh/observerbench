"""Clean-only prompt eligibility for the bounded Qwen copy-v2 study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This module consumes frozen candidate metadata and clean scores only.  It has
no model, attention, mask, ablation, or intervention dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Integral
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256
from observerbench.tasks.qwen_induction.design import (
    SEQUENCE_BANKS,
    SEQUENCE_FAMILIES,
    SequenceBankCounts,
)


COPY_V2_ELIGIBILITY_SCHEMA = (
    "observerbench.qwen_induction.copy_v2.clean_eligibility.v1"
)
COPY_V2_ELIGIBILITY_STATUS_PASS = "passed_clean_eligibility_prompts_frozen"
COPY_V2_ELIGIBILITY_STATUS_FAIL = (
    "failed_clean_eligibility_stop_interventions_forbidden"
)
COPY_V2_RESERVOIR_MULTIPLIER = 2
COPY_V2_CANDIDATE_MARGIN_MINIMUM = math.log(4.0)
COPY_V2_OVERALL_COVERAGE_MINIMUM = 0.80
COPY_V2_FAMILY_COVERAGE_MINIMUM = 0.75
COPY_V2_CELL_COVERAGE_MINIMUM = 0.70
COPY_V2_SELECTION_SEED = 12119

_REQUIRED_CANDIDATE_COLUMNS = ("prompt_id", "bank", "family_id")
_REQUIRED_SCORE_COLUMNS = (
    "prompt_id",
    "candidate_correct",
    "candidate_margin",
    "eager_sdpa_candidate_prediction_agreement",
    "finite_candidate_logits",
    "finite_target_nll",
)
_RESERVED_DECISION_COLUMNS = {
    "eligible",
    "eligibility_reason",
    "selection_sha256",
    "selection_rank_within_cell",
    "selected",
}
_ALLOWED_CANDIDATE_COLUMNS = {
    "prompt_id",
    "bank",
    "family_id",
    "cluster_id",
    "token_bank_id",
    "input_ids",
    "target_token_id",
    "distractor_token_id_1",
    "distractor_token_id_2",
    "key_positions",
    "source_value_positions",
    "source_key_position",
    "source_value_position",
    "query_position",
    "sequence_length",
    "repeat_gap",
}
_ALLOWED_CLEAN_SCORE_COLUMNS = {
    "prompt_id",
    "bank",
    "family_id",
    "candidate_correct",
    "candidate_margin",
    "top1_correct",
    "target_nll",
    "candidate_predicted_token_id",
    "predicted_token_id",
    "eager_sdpa_candidate_prediction_agreement",
    "sdpa_candidate_margin",
    "sdpa_candidate_predicted_token_id",
    "eager_candidate_margin",
    "eager_candidate_predicted_token_id",
    "stage_candidate_margin",
    "stage_candidate_predicted_token_id",
    "finite_target_nll",
    "candidate_logits_finite",
    "finite_candidate_logits",
    "sdpa_candidate_logits_finite",
    "eager_candidate_logits_finite",
}


@dataclass(frozen=True)
class CopyV2EligibilityResult:
    """Complete clean-only decision record before any intervention access."""

    passed: bool
    decisions: pd.DataFrame
    selected_ids: pd.DataFrame
    coverage: pd.DataFrame
    summary: Mapping[str, Any]


def _required_counts(
    counts: SequenceBankCounts | Mapping[str, int],
) -> dict[str, int]:
    if isinstance(counts, SequenceBankCounts):
        raw = counts.as_dict()
    elif isinstance(counts, Mapping):
        raw = dict(counts)
    else:
        raise TypeError("required counts must be SequenceBankCounts or a mapping")
    if set(raw) != set(SEQUENCE_BANKS):
        raise ValueError(f"required counts must define exactly {SEQUENCE_BANKS}")
    result: dict[str, int] = {}
    for bank in SEQUENCE_BANKS:
        value = raw[bank]
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(f"required count for {bank} must be a positive integer")
        result[bank] = int(value)
    return result


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")


def _normalized_booleans(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("candidate_correct contains missing values")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise").to_numpy(float)
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError("candidate_correct must contain only booleans or 0/1")
        return pd.Series(values.astype(bool), index=series.index)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "0", "1")).all():
        raise ValueError("candidate_correct must contain only booleans or 0/1")
    return normalized.isin(("true", "1"))


def _table_sha256(frame: pd.DataFrame) -> str:
    """Hash a table independently of input row and column order."""

    columns = sorted(map(str, frame.columns))
    ordered = frame.loc[:, columns].copy()
    sort_columns = [
        column
        for column in ("bank", "family_id", "prompt_id")
        if column in ordered
    ]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns, kind="mergesort")
    encoded = ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _selection_sha256(prompt_id: str) -> str:
    payload = f"{COPY_V2_SELECTION_SEED}:{prompt_id}".encode("utf-8")
    return sha256(payload).hexdigest()


def _validate_inputs(
    candidates: pd.DataFrame,
    clean_scores: pd.DataFrame,
    required: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(candidates, _REQUIRED_CANDIDATE_COLUMNS, "candidate table")
    _require_columns(clean_scores, _REQUIRED_SCORE_COLUMNS, "clean-score table")
    unexpected_candidates = set(candidates) - _ALLOWED_CANDIDATE_COLUMNS
    unexpected_scores = set(clean_scores) - _ALLOWED_CLEAN_SCORE_COLUMNS
    if unexpected_candidates or unexpected_scores:
        raise ValueError(
            "clean eligibility accepts only frozen fixture metadata and declared "
            "clean-score fields; unexpected columns: "
            f"{sorted(unexpected_candidates | unexpected_scores)}"
        )
    reserved = _RESERVED_DECISION_COLUMNS.intersection(
        set(candidates) | set(clean_scores)
    )
    if reserved:
        raise ValueError(f"eligibility output columns were supplied as inputs: {sorted(reserved)}")

    candidate_rows = candidates.copy()
    score_rows = clean_scores.copy()
    for frame, label in (
        (candidate_rows, "candidate"),
        (score_rows, "clean-score"),
    ):
        if frame.empty or frame["prompt_id"].isna().any():
            raise ValueError(f"{label} rows must be nonempty with prompt IDs")
        frame["prompt_id"] = frame["prompt_id"].astype(str)
        if (frame["prompt_id"].str.len() == 0).any():
            raise ValueError(f"{label} prompt IDs must be nonempty")
        if frame["prompt_id"].duplicated().any():
            raise ValueError(f"{label} prompt IDs must be globally unique")

    candidate_rows["bank"] = candidate_rows["bank"].astype(str)
    candidate_rows["family_id"] = candidate_rows["family_id"].astype(str)
    expected_families = tuple(family.family_id for family in SEQUENCE_FAMILIES)
    if set(candidate_rows["bank"]) != set(SEQUENCE_BANKS):
        raise ValueError("candidate reservoirs must contain all and only the six frozen banks")
    if set(candidate_rows["family_id"]) != set(expected_families):
        raise ValueError("candidate reservoirs must contain all and only the four frozen families")

    if set(score_rows["prompt_id"]) != set(candidate_rows["prompt_id"]):
        raise ValueError("clean scores must cover exactly the candidate prompt IDs")
    for column in ("bank", "family_id"):
        if column in score_rows:
            supplied = score_rows.set_index("prompt_id")[column].astype(str)
            expected = candidate_rows.set_index("prompt_id")[column].astype(str)
            if not supplied.sort_index().equals(expected.sort_index()):
                raise ValueError(f"clean-score {column} differs from the candidate reservoir")
            score_rows = score_rows.drop(columns=column)

    overlap = (set(candidate_rows) & set(score_rows)) - {"prompt_id"}
    if overlap:
        raise ValueError(f"candidate and clean-score columns overlap: {sorted(overlap)}")
    score_rows["candidate_correct"] = _normalized_booleans(
        score_rows["candidate_correct"]
    )
    score_rows["eager_sdpa_candidate_prediction_agreement"] = _normalized_booleans(
        score_rows["eager_sdpa_candidate_prediction_agreement"]
    )
    score_rows["finite_target_nll"] = _normalized_booleans(
        score_rows["finite_target_nll"]
    )
    score_rows["finite_candidate_logits"] = _normalized_booleans(
        score_rows["finite_candidate_logits"]
    )
    margins = pd.to_numeric(score_rows["candidate_margin"], errors="raise").to_numpy(
        float
    )
    score_rows["candidate_margin"] = margins

    cell_counts = candidate_rows.groupby(["bank", "family_id"], sort=False).size()
    for bank in SEQUENCE_BANKS:
        for family_id in expected_families:
            observed = int(cell_counts.get((bank, family_id), 0))
            expected = COPY_V2_RESERVOIR_MULTIPLIER * int(required[bank])
            if observed != expected:
                raise ValueError(
                    f"candidate reservoir {bank}/{family_id} has {observed} rows; "
                    f"expected exactly {expected}"
                )
    return candidate_rows, score_rows


def evaluate_copy_v2_clean_eligibility(
    candidates: pd.DataFrame,
    clean_scores: pd.DataFrame,
    *,
    required_counts: SequenceBankCounts | Mapping[str, int],
) -> CopyV2EligibilityResult:
    """Apply the fixed clean rule and freeze deterministic prompt selections.

    Every bank-by-family reservoir must contain exactly twice its required
    downstream count.  Eligibility uses only the declared clean score, and
    selection among eligible rows uses an outcome-independent SHA-256 order.
    """

    required = _required_counts(required_counts)
    candidate_rows, score_rows = _validate_inputs(
        candidates, clean_scores, required
    )
    candidate_input_sha256 = _table_sha256(candidate_rows)
    clean_scores_sha256 = _table_sha256(score_rows)
    decisions = candidate_rows.merge(
        score_rows,
        on="prompt_id",
        how="inner",
        validate="one_to_one",
    )
    decisions["eligible"] = (
        decisions["candidate_correct"].astype(bool)
        & decisions["eager_sdpa_candidate_prediction_agreement"].astype(bool)
        & decisions["finite_candidate_logits"].astype(bool)
        & decisions["finite_target_nll"].astype(bool)
        & (
            decisions["candidate_margin"].to_numpy(float)
            >= COPY_V2_CANDIDATE_MARGIN_MINIMUM
        )
    )
    decisions["eligibility_reason"] = np.select(
        [
            ~decisions["finite_candidate_logits"].astype(bool),
            ~decisions["finite_target_nll"].astype(bool),
            ~decisions["eager_sdpa_candidate_prediction_agreement"].astype(bool),
            ~decisions["candidate_correct"].astype(bool),
            decisions["candidate_margin"].to_numpy(float)
            < COPY_V2_CANDIDATE_MARGIN_MINIMUM,
        ],
        [
            "nonfinite_candidate_logits",
            "nonfinite_target_nll",
            "eager_sdpa_candidate_prediction_mismatch",
            "candidate_incorrect",
            "candidate_margin_below_ln4",
        ],
        default="eligible",
    )
    decisions["selection_sha256"] = [
        _selection_sha256(str(prompt)) for prompt in decisions["prompt_id"]
    ]
    decisions["selection_rank_within_cell"] = 0
    decisions["selected"] = False

    bank_order = {bank: index for index, bank in enumerate(SEQUENCE_BANKS)}
    family_ids = tuple(family.family_id for family in SEQUENCE_FAMILIES)
    family_order = {family: index for index, family in enumerate(family_ids)}
    for bank in SEQUENCE_BANKS:
        for family_id in family_ids:
            eligible_indices = decisions.index[
                (decisions["bank"] == bank)
                & (decisions["family_id"] == family_id)
                & decisions["eligible"]
            ]
            ordered = decisions.loc[eligible_indices].sort_values(
                ["selection_sha256", "prompt_id"], kind="mergesort"
            )
            for rank, index in enumerate(ordered.index, start=1):
                decisions.at[index, "selection_rank_within_cell"] = rank
                if rank <= required[bank]:
                    decisions.at[index, "selected"] = True

    decisions["_bank_order"] = decisions["bank"].map(bank_order)
    decisions["_family_order"] = decisions["family_id"].map(family_order)
    decisions = decisions.sort_values(
        ["_bank_order", "_family_order", "selection_sha256", "prompt_id"],
        kind="mergesort",
    ).drop(columns=["_bank_order", "_family_order"])
    decisions = decisions.reset_index(drop=True)
    decisions["selection_rank_within_cell"] = decisions[
        "selection_rank_within_cell"
    ].astype(int)
    decisions["selected"] = decisions["selected"].astype(bool)

    selected_ids = decisions.loc[
        decisions["selected"],
        [
            "prompt_id",
            "bank",
            "family_id",
            "selection_rank_within_cell",
            "selection_sha256",
        ],
    ].copy()
    selected_ids["_bank_order"] = selected_ids["bank"].map(bank_order)
    selected_ids["_family_order"] = selected_ids["family_id"].map(family_order)
    selected_ids = selected_ids.sort_values(
        ["_bank_order", "_family_order", "selection_rank_within_cell"],
        kind="mergesort",
    ).drop(columns=["_bank_order", "_family_order"])
    selected_ids = selected_ids.reset_index(drop=True)

    coverage_rows: list[dict[str, Any]] = []

    def append_coverage(
        scope: str,
        frame: pd.DataFrame,
        *,
        minimum: float,
        required_count: int,
        bank: str = "",
        family_id: str = "",
    ) -> None:
        candidate_count = len(frame)
        eligible_count = int(frame["eligible"].sum())
        selected_count = int(frame["selected"].sum())
        coverage = eligible_count / candidate_count
        coverage_rows.append(
            {
                "scope": scope,
                "bank": bank,
                "family_id": family_id,
                "candidate_count": candidate_count,
                "required_count": int(required_count),
                "eligible_count": eligible_count,
                "selected_count": selected_count,
                "coverage": float(coverage),
                "minimum_coverage": float(minimum),
                "coverage_passed": bool(coverage >= minimum),
                "exact_fill": bool(selected_count == required_count),
            }
        )

    append_coverage(
        "overall",
        decisions,
        minimum=COPY_V2_OVERALL_COVERAGE_MINIMUM,
        required_count=sum(required[bank] for bank in SEQUENCE_BANKS)
        * len(family_ids),
    )
    for family_id in family_ids:
        family = decisions.loc[decisions["family_id"] == family_id]
        append_coverage(
            "family",
            family,
            minimum=COPY_V2_FAMILY_COVERAGE_MINIMUM,
            required_count=sum(required.values()),
            family_id=family_id,
        )
    for bank in SEQUENCE_BANKS:
        for family_id in family_ids:
            cell = decisions.loc[
                (decisions["bank"] == bank)
                & (decisions["family_id"] == family_id)
            ]
            append_coverage(
                "bank_family",
                cell,
                minimum=COPY_V2_CELL_COVERAGE_MINIMUM,
                required_count=required[bank],
                bank=bank,
                family_id=family_id,
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage_passed = bool(coverage["coverage_passed"].all())
    exact_fill = bool(
        coverage.loc[coverage["scope"] == "bank_family", "exact_fill"].all()
    )
    passed = bool(coverage_passed and exact_fill)
    summary = {
        "schema": COPY_V2_ELIGIBILITY_SCHEMA,
        "status": (
            COPY_V2_ELIGIBILITY_STATUS_PASS
            if passed
            else COPY_V2_ELIGIBILITY_STATUS_FAIL
        ),
        "passed": passed,
        "protocol": {
            "reservoir_assignment": "preassigned_disjoint_bank_by_family",
            "reservoir_multiplier": COPY_V2_RESERVOIR_MULTIPLIER,
            "eligibility_rule": (
                "candidate_correct AND candidate_margin >= ln(4) AND "
                "discovery eager/SDPA prediction agreement AND finite "
                "candidate logits and target NLL"
            ),
            "candidate_margin_minimum": COPY_V2_CANDIDATE_MARGIN_MINIMUM,
            "selection_seed": COPY_V2_SELECTION_SEED,
            "selection_rule": (
                "ascending_sha256_of_selection_seed_colon_prompt_id_then_prompt_id"
            ),
            "coverage_minima": {
                "overall": COPY_V2_OVERALL_COVERAGE_MINIMUM,
                "family_pooled": COPY_V2_FAMILY_COVERAGE_MINIMUM,
                "bank_family": COPY_V2_CELL_COVERAGE_MINIMUM,
            },
            "required_counts_per_family": dict(required),
        },
        "gates": {
            "overall_coverage": bool(
                coverage.loc[
                    coverage["scope"] == "overall", "coverage_passed"
                ].iloc[0]
            ),
            "every_family_pooled_coverage": bool(
                coverage.loc[
                    coverage["scope"] == "family", "coverage_passed"
                ].all()
            ),
            "every_bank_family_coverage": bool(
                coverage.loc[
                    coverage["scope"] == "bank_family", "coverage_passed"
                ].all()
            ),
            "exact_fill": exact_fill,
        },
        "counts": {
            "candidate_rows": len(decisions),
            "eligible_rows": int(decisions["eligible"].sum()),
            "selected_rows": len(selected_ids),
        },
        "input_hashes": {
            "candidate_table_sha256": candidate_input_sha256,
            "clean_scores_sha256": clean_scores_sha256,
        },
        "access_audit": {
            "clean_score_rows_consumed": len(score_rows),
            "attention_scores_loaded": False,
            "intervention_metadata_loaded": False,
            "intervention_outcomes_loaded": False,
            "intervention_forward_passes": 0,
            "candidate_effect_cells": 0,
        },
        "next_allowed_stage": (
            "freeze selected prompt IDs before head discovery"
            if passed
            else "STOP. Do not scan heads or measure any intervention outcome."
        ),
    }
    return CopyV2EligibilityResult(
        passed=passed,
        decisions=decisions,
        selected_ids=selected_ids,
        coverage=coverage,
        summary=summary,
    )


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def write_copy_v2_eligibility_artifacts(
    result: CopyV2EligibilityResult,
    output_dir: str | Path,
) -> Path:
    """Write an immutable, hash-bound eligibility decision bundle."""

    output = Path(output_dir)
    artifacts = {
        "candidate_decisions.csv": result.decisions,
        "selected_prompt_ids.csv": result.selected_ids,
        "coverage.csv": result.coverage,
    }
    payload = dict(result.summary)
    if output.exists():
        manifest_path = output / "eligibility_manifest.json"
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError("existing eligibility bundle is incomplete") from error
        hashes = existing.get("artifact_hashes")
        if (
            existing.get("schema") != payload.get("schema")
            or existing.get("input_hashes") != payload.get("input_hashes")
            or existing.get("passed") != payload.get("passed")
            or not isinstance(hashes, Mapping)
            or set(hashes) != set(artifacts)
            or any(
                not (output / name).is_file()
                or file_sha256(output / name) != hashes[name]
                for name in artifacts
            )
        ):
            raise ValueError("existing eligibility bundle differs from recomputation")
        return manifest_path

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name, frame in artifacts.items():
        _write_csv_atomic(frame, staging / name)
    payload["artifact_hashes"] = {
        name: file_sha256(staging / name) for name in artifacts
    }
    manifest_path = staging / "eligibility_manifest.json"
    temporary = staging / ".eligibility_manifest.json.tmp"
    write_json(temporary, payload)
    temporary.replace(manifest_path)
    staging.replace(output)
    return output / "eligibility_manifest.json"


__all__ = [
    "COPY_V2_CANDIDATE_MARGIN_MINIMUM",
    "COPY_V2_CELL_COVERAGE_MINIMUM",
    "COPY_V2_ELIGIBILITY_SCHEMA",
    "COPY_V2_ELIGIBILITY_STATUS_FAIL",
    "COPY_V2_ELIGIBILITY_STATUS_PASS",
    "COPY_V2_FAMILY_COVERAGE_MINIMUM",
    "COPY_V2_OVERALL_COVERAGE_MINIMUM",
    "COPY_V2_RESERVOIR_MULTIPLIER",
    "COPY_V2_SELECTION_SEED",
    "CopyV2EligibilityResult",
    "evaluate_copy_v2_clean_eligibility",
    "write_copy_v2_eligibility_artifacts",
]
