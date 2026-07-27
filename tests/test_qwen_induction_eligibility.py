"""Tests for the bounded clean-only Qwen copy-v2 eligibility layer.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observerbench.provenance import file_sha256
from observerbench.tasks.qwen_induction.design import (
    SEQUENCE_BANKS,
    SEQUENCE_FAMILIES,
    SequenceBankCounts,
)
from observerbench.tasks.qwen_induction.eligibility import (
    COPY_V2_CANDIDATE_MARGIN_MINIMUM,
    COPY_V2_ELIGIBILITY_STATUS_FAIL,
    COPY_V2_ELIGIBILITY_STATUS_PASS,
    COPY_V2_SELECTION_SEED,
    evaluate_copy_v2_clean_eligibility,
    write_copy_v2_eligibility_artifacts,
)


SMALL_COUNTS = SequenceBankCounts(
    reference=2,
    discovery=2,
    head_fit=2,
    head_test=2,
    calibration=2,
    locked_test=2,
)


def _tables(
    counts: SequenceBankCounts = SMALL_COUNTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    required = counts.as_dict()
    for bank_index, bank in enumerate(SEQUENCE_BANKS):
        for family_index, family in enumerate(SEQUENCE_FAMILIES):
            for reservoir_index in range(2 * required[bank]):
                prompt_id = (
                    f"candidate_{bank_index}_{family_index}_{reservoir_index:03d}"
                )
                candidates.append(
                    {
                        "prompt_id": prompt_id,
                        "bank": bank,
                        "family_id": family.family_id,
                        "input_ids": f"{bank_index} {family_index} {reservoir_index}",
                    }
                )
                scores.append(
                    {
                        "prompt_id": prompt_id,
                        "candidate_correct": True,
                        "candidate_margin": COPY_V2_CANDIDATE_MARGIN_MINIMUM + 0.5,
                        "eager_sdpa_candidate_prediction_agreement": True,
                        "finite_candidate_logits": True,
                        "finite_target_nll": True,
                        "top1_correct": reservoir_index % 2 == 0,
                        "target_nll": 0.1 + 0.001 * reservoir_index,
                    }
                )
    return pd.DataFrame(candidates), pd.DataFrame(scores)


def _cell(
    frame: pd.DataFrame,
    bank: str = "reference",
    family_id: str | None = None,
) -> pd.DataFrame:
    family_id = family_id or SEQUENCE_FAMILIES[0].family_id
    return frame.loc[
        (frame["bank"] == bank) & (frame["family_id"] == family_id)
    ]


def test_pass_freezes_lowest_sha_ids_independently_of_input_order() -> None:
    candidates, scores = _tables()
    first = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=SMALL_COUNTS,
    )
    second = evaluate_copy_v2_clean_eligibility(
        candidates.sample(frac=1.0, random_state=7).reset_index(drop=True),
        scores.sample(frac=1.0, random_state=11).reset_index(drop=True),
        required_counts=SMALL_COUNTS,
    )

    assert first.passed
    assert first.summary["status"] == COPY_V2_ELIGIBILITY_STATUS_PASS
    assert len(first.decisions) == 2 * len(first.selected_ids)
    assert first.selected_ids.equals(second.selected_ids)
    assert first.summary["input_hashes"] == second.summary["input_hashes"]
    assert set(first.coverage["scope"]) == {"overall", "family", "bank_family"}
    assert len(first.coverage) == 1 + len(SEQUENCE_FAMILIES) + (
        len(SEQUENCE_BANKS) * len(SEQUENCE_FAMILIES)
    )

    decision_cell = _cell(first.decisions)
    expected = decision_cell.sort_values(
        ["selection_sha256", "prompt_id"], kind="mergesort"
    ).head(SMALL_COUNTS.reference)
    selected = decision_cell.loc[decision_cell["selected"]]
    assert set(selected["prompt_id"]) == set(expected["prompt_id"])
    assert set(first.decisions["eligibility_reason"]) == {"eligible"}
    assert first.summary["protocol"]["selection_seed"] == (
        COPY_V2_SELECTION_SEED
    )
    independently_ranked = decision_cell.assign(
        expected_sha256=[
            sha256(f"12119:{prompt_id}".encode("utf-8")).hexdigest()
            for prompt_id in decision_cell["prompt_id"]
        ]
    ).sort_values(["expected_sha256", "prompt_id"], kind="mergesort")
    assert set(selected["prompt_id"]) == set(
        independently_ranked.head(SMALL_COUNTS.reference)["prompt_id"]
    )
    assert first.summary["access_audit"] == {
        "clean_score_rows_consumed": len(scores),
        "attention_scores_loaded": False,
        "intervention_metadata_loaded": False,
        "intervention_outcomes_loaded": False,
        "intervention_forward_passes": 0,
        "candidate_effect_cells": 0,
    }


def test_fixed_rule_is_correct_and_inclusive_at_ln4() -> None:
    candidates, scores = _tables()
    prompt_ids = _cell(candidates)["prompt_id"].tolist()
    scores.loc[
        scores["prompt_id"] == prompt_ids[0], "candidate_margin"
    ] = math.log(4.0)
    scores.loc[
        scores["prompt_id"] == prompt_ids[1], "candidate_margin"
    ] = np.nextafter(math.log(4.0), -math.inf)
    scores.loc[
        scores["prompt_id"] == prompt_ids[2], "candidate_correct"
    ] = False
    scores.loc[
        scores["prompt_id"] == prompt_ids[3],
        "eager_sdpa_candidate_prediction_agreement",
    ] = False

    result = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=SMALL_COUNTS,
    )
    indexed = result.decisions.set_index("prompt_id")

    assert bool(indexed.loc[prompt_ids[0], "eligible"])
    assert indexed.loc[prompt_ids[0], "eligibility_reason"] == "eligible"
    assert not bool(indexed.loc[prompt_ids[1], "eligible"])
    assert indexed.loc[prompt_ids[1], "eligibility_reason"] == (
        "candidate_margin_below_ln4"
    )
    assert not bool(indexed.loc[prompt_ids[2], "eligible"])
    assert indexed.loc[prompt_ids[2], "eligibility_reason"] == (
        "candidate_incorrect"
    )
    assert not bool(indexed.loc[prompt_ids[3], "eligible"])
    assert indexed.loc[prompt_ids[3], "eligibility_reason"] == (
        "eager_sdpa_candidate_prediction_mismatch"
    )


def test_all_four_gates_are_recorded_and_failure_stops() -> None:
    candidates, scores = _tables()
    # Three of four candidates pass in every cell: family and cell gates pass,
    # exact fill succeeds, but 0.75 overall coverage fails the stricter 0.80 gate.
    for bank in SEQUENCE_BANKS:
        for family in SEQUENCE_FAMILIES:
            prompt_id = _cell(candidates, bank, family.family_id).iloc[0]["prompt_id"]
            scores.loc[scores["prompt_id"] == prompt_id, "candidate_correct"] = False
    result = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=SMALL_COUNTS,
    )

    assert not result.passed
    assert result.summary["status"] == COPY_V2_ELIGIBILITY_STATUS_FAIL
    assert result.summary["gates"] == {
        "overall_coverage": False,
        "every_family_pooled_coverage": True,
        "every_bank_family_coverage": True,
        "exact_fill": True,
    }
    assert result.summary["next_allowed_stage"].startswith("STOP")

    # One eligible row in a four-row cell also fails its cell coverage and fill.
    candidates, scores = _tables()
    prompt_ids = _cell(candidates)["prompt_id"].tolist()
    scores.loc[
        scores["prompt_id"].isin(prompt_ids[:3]), "candidate_correct"
    ] = False
    result = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=SMALL_COUNTS,
    )
    assert not result.summary["gates"]["every_bank_family_coverage"]
    assert not result.summary["gates"]["exact_fill"]

    # With ten candidates per cell, 0.70 passes every cell but fails the
    # stricter 0.75 family-pooled gate while leaving overall coverage above 0.80.
    five_counts = SequenceBankCounts(
        reference=5,
        discovery=5,
        head_fit=5,
        head_test=5,
        calibration=5,
        locked_test=5,
    )
    candidates, scores = _tables(five_counts)
    target_family = SEQUENCE_FAMILIES[0].family_id
    for bank in SEQUENCE_BANKS:
        failed_ids = _cell(candidates, bank, target_family).head(3)["prompt_id"]
        scores.loc[
            scores["prompt_id"].isin(failed_ids), "candidate_correct"
        ] = False
    result = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=five_counts,
    )
    assert result.summary["gates"]["overall_coverage"]
    assert not result.summary["gates"]["every_family_pooled_coverage"]
    assert result.summary["gates"]["every_bank_family_coverage"]
    assert result.summary["gates"]["exact_fill"]


def test_rejects_malformed_or_intervention_tainted_inputs() -> None:
    candidates, scores = _tables()
    malformed = candidates.iloc[:-1].copy()
    with pytest.raises(ValueError, match="expected exactly"):
        evaluate_copy_v2_clean_eligibility(
            malformed,
            scores.loc[scores["prompt_id"].isin(malformed["prompt_id"])],
            required_counts=SMALL_COUNTS,
        )

    missing = scores.iloc[:-1]
    with pytest.raises(ValueError, match="cover exactly"):
        evaluate_copy_v2_clean_eligibility(
            candidates,
            missing,
            required_counts=SMALL_COUNTS,
        )

    tainted = scores.assign(mask_id="forbidden")
    with pytest.raises(ValueError, match="unexpected columns"):
        evaluate_copy_v2_clean_eligibility(
            candidates,
            tainted,
            required_counts=SMALL_COUNTS,
        )

    invalid = scores.copy()
    invalid["candidate_correct"] = invalid["candidate_correct"].astype(object)
    invalid.loc[0, "candidate_correct"] = "maybe"
    with pytest.raises(ValueError, match="booleans or 0/1"):
        evaluate_copy_v2_clean_eligibility(
            candidates,
            invalid,
            required_counts=SMALL_COUNTS,
        )


def test_nonfinite_clean_scores_remain_in_the_coverage_denominator() -> None:
    candidates, scores = _tables()
    prompt_id = str(scores.iloc[0]["prompt_id"])
    scores.loc[scores["prompt_id"] == prompt_id, "candidate_margin"] = float("nan")
    scores.loc[
        scores["prompt_id"] == prompt_id, "finite_candidate_logits"
    ] = False

    result = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=SMALL_COUNTS,
    )
    row = result.decisions.set_index("prompt_id").loc[prompt_id]
    assert bool(row["eligible"]) is False
    assert row["eligibility_reason"] == "nonfinite_candidate_logits"
    overall = result.coverage.loc[result.coverage["scope"] == "overall"].iloc[0]
    assert int(overall["candidate_count"]) == len(candidates)

def test_writes_immutable_hash_bound_bundle(tmp_path: Path) -> None:
    candidates, scores = _tables()
    result = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=SMALL_COUNTS,
    )
    output = tmp_path / "eligibility"
    manifest_path = write_copy_v2_eligibility_artifacts(result, output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["passed"] is True
    assert manifest["artifact_hashes"] == {
        name: file_sha256(output / name)
        for name in (
            "candidate_decisions.csv",
            "selected_prompt_ids.csv",
            "coverage.csv",
        )
    }
    written_decisions = pd.read_csv(output / "candidate_decisions.csv")
    assert len(written_decisions) == len(candidates)
    assert {"eligible", "eligibility_reason", "selected"}.issubset(
        written_decisions
    )
    assert write_copy_v2_eligibility_artifacts(result, output) == manifest_path
