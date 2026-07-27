"""Tests for immutable Qwen Copy-v2 candidate and preselection artifacts.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from observerbench.core import write_json
from observerbench.provenance import file_sha256, json_sha256
from observerbench.tasks.qwen_induction.artifacts import (
    TOKEN_BANKS_SCHEMA,
    _PROMPTS_ALL_COLUMNS,
)
from observerbench.tasks.qwen_induction.copy_v2_artifacts import (
    COPY_V2_CANDIDATE_STATUS,
    COPY_V2_DATA_VERSION,
    COPY_V2_PRESELECTION_STATUS,
    load_copy_v2_candidate_reservoir,
    verify_copy_v2_candidate_artifacts,
    verify_copy_v2_preselection_artifacts,
    write_copy_v2_candidate_artifacts,
    write_copy_v2_preselection_artifacts,
)
from observerbench.tasks.qwen_induction.design import (
    SEQUENCE_BANKS,
    SequenceBankCounts,
    build_sequence_design,
)
from observerbench.tasks.qwen_induction.effect_task import _PROMPT_COLUMNS
from observerbench.tasks.qwen_induction.eligibility import (
    evaluate_copy_v2_clean_eligibility,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COPY_V2_CONFIG = (
    REPO_ROOT
    / "configs/revision/phase10/qwen2_5_7b_induction_copy_v2.json"
)


def _config() -> dict:
    config = deepcopy(json.loads(COPY_V2_CONFIG.read_text(encoding="utf-8")))
    config["token_pool"]["per_split_size"] = 64
    config["sequence_design"]["final_prompts_per_family"] = {
        "reference": 1,
        "discovery": 1,
        "head_fit": 1,
        "head_confirmation": 1,
        "calibration": 1,
        "locked_test": 1,
    }
    config["sequence_design"]["reservoir_prompts_per_family"] = {
        key: 2
        for key in config["sequence_design"]["final_prompts_per_family"]
    }
    return config


def _sequence(config: dict):
    return build_sequence_design(
        range(10_000, 11_000),
        bank_counts=SequenceBankCounts(
            reference=2,
            discovery=2,
            head_fit=2,
            head_test=2,
            calibration=2,
            locked_test=2,
        ),
        per_split_size=config["token_pool"]["per_split_size"],
        seed=config["sequence_design"]["seed"],
    )


def _copy_v1_payload(token_by_bank: dict[str, int] | None = None) -> dict:
    if token_by_bank is None:
        token_by_bank = {
            bank: 100 + index for index, bank in enumerate(SEQUENCE_BANKS)
        }
    token_pool = tuple(sorted(map(int, token_by_bank.values())))
    return {
        "schema": TOKEN_BANKS_SCHEMA,
        "token_pool_size": len(token_pool),
        "token_pool_sha256": json_sha256(token_pool),
        "banks": [
            {
                "bank": bank,
                "token_ids": [int(token_by_bank[bank])],
                "token_count": 1,
                "token_sha256": json_sha256((int(token_by_bank[bank]),)),
            }
            for bank in SEQUENCE_BANKS
        ],
    }


def _write_copy_v1(path: Path, payload: dict | None = None) -> Path:
    write_json(path, _copy_v1_payload() if payload is None else payload)
    return path


def _bind_copy_v1(config: dict, source: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    binding = config["token_pool"]["exclude_copy_v1_allocated_tokens"]
    binding["source_artifact_sha256"] = file_sha256(source)
    binding["source_token_pool_size"] = int(payload["token_pool_size"])
    binding["source_token_pool_sha256"] = str(payload["token_pool_sha256"])


def _candidate_bundle(tmp_path: Path):
    config = _config()
    sequence = _sequence(config)
    source = _write_copy_v1(tmp_path / "copy_v1_token_banks.json")
    _bind_copy_v1(config, source)
    manifest = write_copy_v2_candidate_artifacts(
        sequence,
        config,
        tmp_path / "artifacts",
        copy_v1_token_banks_path=source,
    )
    return config, sequence, tmp_path / "artifacts", manifest


def _passing_result(root: Path, config: dict):
    candidates = load_copy_v2_candidate_reservoir(root, config)
    scores = pd.DataFrame(
        {
            "prompt_id": candidates["prompt_id"],
            "candidate_correct": True,
            "candidate_margin": 2.0,
            "eager_sdpa_candidate_prediction_agreement": True,
            "finite_candidate_logits": True,
            "finite_target_nll": True,
            "top1_correct": False,
            "target_nll": 4.0,
        }
    )
    counts = {
        "reference": 1,
        "discovery": 1,
        "head_fit": 1,
        "head_test": 1,
        "calibration": 1,
        "locked_test": 1,
    }
    return evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts=counts,
    )


def test_candidate_reservoir_is_two_times_disjoint_and_outcome_free(
    tmp_path: Path,
) -> None:
    config, _sequence_design, root, manifest_path = _candidate_bundle(tmp_path)

    manifest = verify_copy_v2_candidate_artifacts(root, config)
    candidates = load_copy_v2_candidate_reservoir(root, config)

    assert manifest_path == root / "candidate_design/candidate_manifest.json"
    assert manifest["status"] == COPY_V2_CANDIDATE_STATUS
    assert manifest["data_version"] == COPY_V2_DATA_VERSION
    assert manifest["access_audit"]["model_forward_passes"] == 0
    assert manifest["access_audit"]["intervention_forward_passes"] == 0
    assert tuple(candidates.columns) == _PROMPTS_ALL_COLUMNS
    assert len(candidates) == 2 * 6 * 4
    assert set(candidates.groupby(["bank", "family_id"]).size()) == {2}
    assert (
        root / "candidate_design/copy_v1_token_exclusion.json"
    ).is_file()

    prompts_path = root / "candidate_design/prompts_all.csv"
    prompts_path.write_bytes(prompts_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_copy_v2_candidate_artifacts(root, config)


def test_candidate_reservoir_rejects_any_copy_v1_token_overlap(
    tmp_path: Path,
) -> None:
    config = _config()
    sequence = _sequence(config)
    first_tokens = {
        bank.bank: int(bank.token_ids[0]) for bank in sequence.token_banks
    }
    source = _write_copy_v1(
        tmp_path / "copy_v1_token_banks.json",
        _copy_v1_payload(first_tokens),
    )
    _bind_copy_v1(config, source)

    with pytest.raises(ValueError, match="reuse"):
        write_copy_v2_candidate_artifacts(
            sequence,
            config,
            tmp_path / "artifacts",
            copy_v1_token_banks_path=source,
        )


def test_passed_clean_result_freezes_existing_prompt_schemas_and_bindings(
    tmp_path: Path,
) -> None:
    config, _sequence_design, root, _manifest = _candidate_bundle(tmp_path)
    result = _passing_result(root, config)
    assert result.passed

    manifest_path = write_copy_v2_preselection_artifacts(
        result,
        config,
        root,
        runtime_record={
            "model": "Qwen/Qwen2.5-7B",
            "revision": config["model"]["revision"],
            "attention_implementation": "sdpa",
            "scientific_claim_allowed": True,
        },
        source_hashes={"copy_v2_artifacts.py": "a" * 64},
    )
    manifest = verify_copy_v2_preselection_artifacts(root, config)
    prompts_all = pd.read_csv(
        root / "design/prompts_all.csv", dtype=str, keep_default_na=False
    )
    prompts = pd.read_csv(
        root / "design/prompts.csv", dtype=str, keep_default_na=False
    )

    assert manifest_path == root / "design/preselection_manifest.json"
    assert manifest["status"] == COPY_V2_PRESELECTION_STATUS
    assert manifest["scientific_outcomes_included"] is False
    assert manifest["attention_outputs_included"] is False
    assert manifest["intervention_outcomes_included"] is False
    assert manifest["access_audit"]["intervention_forward_passes"] == 0
    assert tuple(prompts_all.columns) == _PROMPTS_ALL_COLUMNS
    assert tuple(prompts.columns) == _PROMPT_COLUMNS
    assert len(prompts_all) == 6 * 4
    assert len(prompts) == 2 * 4
    assert set(prompts["split"]) == {"train", "test"}
    assert set(prompts_all.groupby(["bank", "family_id"]).size()) == {1}
    assert (
        root / "design/eligibility/candidate_decisions.csv"
    ).is_file()
    assert (root / "design/eligibility/coverage.csv").is_file()
    assert (
        root / "design/eligibility/selected_prompt_ids.csv"
    ).is_file()

    coverage_path = root / "design/eligibility/coverage.csv"
    coverage_path.write_bytes(coverage_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_copy_v2_preselection_artifacts(root, config)


def test_preselection_round_trips_clean_score_floats_exactly(
    tmp_path: Path,
) -> None:
    config, _sequence_design, root, _manifest = _candidate_bundle(tmp_path)
    candidates = load_copy_v2_candidate_reservoir(root, config)
    scores = pd.DataFrame(
        {
            "prompt_id": candidates["prompt_id"],
            "candidate_correct": True,
            "candidate_margin": 12.38647747039795,
            "eager_sdpa_candidate_prediction_agreement": True,
            "finite_candidate_logits": True,
            "finite_target_nll": True,
            "target_nll": 2.0866472721099854,
        }
    )
    result = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts={bank: 1 for bank in SEQUENCE_BANKS},
    )
    write_copy_v2_preselection_artifacts(
        result,
        config,
        root,
        runtime_record={
            "runtime": "test",
            "scientific_claim_allowed": True,
        },
        source_hashes={"copy_v2_artifacts.py": "a" * 64},
    )

    manifest = verify_copy_v2_preselection_artifacts(root, config)
    assert (
        manifest["clean_eligibility"]["clean_scores_canonical_sha256"]
        == result.summary["input_hashes"]["clean_scores_sha256"]
    )


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    (
        ("target_nll", 9.0, "input_hashes"),
        ("eligibility_reason", "forged", "decisions"),
    ),
)
def test_preselection_rejects_rehashed_clean_decision_tampering(
    tmp_path: Path,
    column: str,
    replacement: object,
    message: str,
) -> None:
    config, _sequence_design, root, _manifest = _candidate_bundle(tmp_path)
    result = _passing_result(root, config)
    write_copy_v2_preselection_artifacts(
        result,
        config,
        root,
        runtime_record={
            "runtime": "test",
            "scientific_claim_allowed": True,
        },
        source_hashes={"copy_v2_artifacts.py": "a" * 64},
    )

    eligibility_dir = root / "design/eligibility"
    decisions_path = eligibility_dir / "candidate_decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[0, column] = replacement
    decisions.to_csv(decisions_path, index=False)

    eligibility_path = eligibility_dir / "eligibility_manifest.json"
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    eligibility["artifact_hashes"]["candidate_decisions.csv"] = file_sha256(
        decisions_path
    )
    write_json(eligibility_path, eligibility)

    preselection_path = root / "design/preselection_manifest.json"
    preselection = json.loads(preselection_path.read_text(encoding="utf-8"))
    preselection["clean_eligibility"]["manifest_sha256"] = file_sha256(
        eligibility_path
    )
    preselection["artifact_hashes"][
        "eligibility/candidate_decisions.csv"
    ] = file_sha256(decisions_path)
    preselection["artifact_hashes"][
        "eligibility/eligibility_manifest.json"
    ] = file_sha256(eligibility_path)
    write_json(preselection_path, preselection)

    with pytest.raises(ValueError, match=message):
        verify_copy_v2_preselection_artifacts(root, config)


def test_failed_or_forged_eligibility_cannot_emit_preselection(
    tmp_path: Path,
) -> None:
    config, _sequence_design, root, _manifest = _candidate_bundle(tmp_path)
    candidates = load_copy_v2_candidate_reservoir(root, config)
    scores = pd.DataFrame(
        {
            "prompt_id": candidates["prompt_id"],
            "candidate_correct": True,
            "candidate_margin": 2.0,
            "eager_sdpa_candidate_prediction_agreement": True,
            "finite_candidate_logits": True,
            "finite_target_nll": True,
        }
    )
    first_cell = (
        (candidates["bank"] == "reference")
        & (candidates["family_id"] == candidates.iloc[0]["family_id"])
    )
    scores.loc[first_cell.to_numpy(), "candidate_correct"] = False
    result = evaluate_copy_v2_clean_eligibility(
        candidates,
        scores,
        required_counts={bank: 1 for bank in SEQUENCE_BANKS},
    )
    assert not result.passed

    with pytest.raises(ValueError, match="passed clean eligibility"):
        write_copy_v2_preselection_artifacts(
            result,
            config,
            root,
            runtime_record={"runtime": "test"},
            source_hashes={"source.py": "b" * 64},
        )
