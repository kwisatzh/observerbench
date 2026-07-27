"""Orchestration tests for the prospective Qwen Copy-v2 clean boundary.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from observerbench.tasks.qwen_induction.artifacts import load_phase09_config
from observerbench.tasks.qwen_induction.copy_v2_runner import (
    COPY_V2_STAGES,
    CopyV2Runner,
)
from test_qwen_induction_runner import (
    FakePlant,
    FakeTokenizer,
    _fake_collateral_iterator,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COPY_V2_CONFIG = (
    REPO_ROOT / "configs/revision/phase10/qwen2_5_7b_induction_copy_v2.json"
)


def _config() -> dict:
    config = deepcopy(load_phase09_config(COPY_V2_CONFIG))
    config["status"] = "frozen_before_copy_v2_outcomes"
    config["token_pool"]["per_split_size"] = 64
    for bank in config["sequence_design"]["final_prompts_per_family"]:
        config["sequence_design"]["final_prompts_per_family"][bank] = 1
        config["sequence_design"]["reservoir_prompts_per_family"][bank] = 2
    config["head_discovery"]["bootstrap_repeats"] = 20
    config["uncertainty"]["bootstrap_repeats"] = 20
    config["runtime"]["discovery_batch_size"] = 2
    config["runtime"]["measurement_batch_size"] = 2
    return config


def _runner(
    config: dict,
    root: Path,
    *,
    plant_type: type[FakePlant] = FakePlant,
    resume: bool = False,
) -> CopyV2Runner:
    return CopyV2Runner(
        config,
        root,
        tokenizer_factory=lambda _config: FakeTokenizer(),
        plant_factory=lambda implementation, _config: plant_type(implementation),
        collateral_iterator=_fake_collateral_iterator,
        allow_injected_test_runtime=True,
        resume=resume,
    )


class OffsetFakePlant(FakePlant):
    implementations: list[str] = []
    events: list[str] = []
    clean_calls: list[tuple[str, tuple[str, ...], int]] = []

    def score_clean(self, records, *, batch_size):
        rows = super().score_clean(records, batch_size=batch_size)
        if self.attention_implementation == "eager":
            for row in rows:
                row.candidate_margin += 0.1
        return rows


def test_copy_v2_stops_attention_until_clean_selection_is_frozen(
    tmp_path: Path,
) -> None:
    FakePlant.implementations = []
    FakePlant.events = []
    runner = _runner(_config(), tmp_path)

    with pytest.raises(RuntimeError, match="out of order"):
        runner.discover()
    assert runner.prepare().status == "complete"
    assert FakePlant.implementations == []
    assert not (tmp_path / "design").exists()

    assert runner.eligibility().status == "complete"
    assert FakePlant.implementations == ["sdpa", "eager"]
    assert FakePlant.events == []
    preselection = json.loads(
        (tmp_path / "design/preselection_manifest.json").read_text()
    )
    assert preselection["clean_eligibility"]["passed"] is True
    assert preselection["access_audit"]["intervention_forward_passes"] == 0
    decisions = pd.read_csv(
        tmp_path / "design/eligibility/candidate_decisions.csv"
    )
    assert decisions["selected"].sum() == 24
    assert decisions["eligible"].all()

    assert runner.discover().status == "complete"
    assert FakePlant.events.index("noop:32") < FakePlant.events.index("effects:32")


def test_copy_v2_full_fake_chain_keeps_eligibility_in_the_frozen_design(
    tmp_path: Path,
) -> None:
    FakePlant.fail_after = None
    FakePlant.yielded_masks = []
    FakePlant.implementations = []
    FakePlant.events = []
    runner = _runner(_config(), tmp_path)

    results = runner.run("all")
    assert all(result.status == "complete" for result in results)
    assert tuple(result.stage for result in results) == COPY_V2_STAGES
    audit = json.loads((tmp_path / "work/stage_audit.json").read_text())
    assert tuple(audit["completed_stages"]) == COPY_V2_STAGES
    design = json.loads((tmp_path / "design/design_manifest.json").read_text())
    assert design["data_version"] == "copy-v2"
    assert "eligibility_gate" in design["gate_artifact_hashes"]
    effects = json.loads((tmp_path / "effects/effect_manifest.json").read_text())
    assert effects["data_version"] == "copy-v2"
    assert effects["scientific_claim_allowed"] is False
    assert (tmp_path / "design/gates/reference_discovery_gate.json").is_file()
    assert (tmp_path / "design/gates/reference_confirmation_gate.json").is_file()


def test_copy_v2_binds_each_rescore_to_its_actual_attention_implementation(
    tmp_path: Path,
) -> None:
    OffsetFakePlant.implementations = []
    OffsetFakePlant.events = []
    OffsetFakePlant.clean_calls = []
    config = _config()
    config["runtime"]["discovery_batch_size"] = 1
    config["runtime"]["measurement_batch_size"] = 2
    runner = _runner(config, tmp_path, plant_type=OffsetFakePlant)

    assert runner.prepare().status == "complete"
    assert runner.eligibility().status == "complete"
    assert runner.discover().status == "complete"

    decisions = pd.read_csv(tmp_path / "design/eligibility/candidate_decisions.csv")
    discovery = decisions.loc[decisions["bank"] == "discovery"]
    head_fit = decisions.loc[decisions["bank"] == "head_fit"]
    assert (discovery["stage_candidate_margin"] == 3.1).all()
    assert (head_fit["stage_candidate_margin"] == 3.0).all()
    eager_calls = [call for call in OffsetFakePlant.clean_calls if call[0] == "eager"]
    assert eager_calls
    assert all(call[1] == ("discovery",) for call in eager_calls)
    assert any(call[:2] == ("sdpa", ("reference",)) for call in OffsetFakePlant.clean_calls)


def test_copy_v2_revalidates_nested_preselection_before_discovery(
    tmp_path: Path,
) -> None:
    runner = _runner(_config(), tmp_path)
    runner.prepare()
    runner.eligibility()
    coverage = tmp_path / "design/eligibility/coverage.csv"
    coverage.write_bytes(coverage.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="artifact hash mismatch|completed stage artifact"):
        runner.discover()


def test_copy_v2_forbids_injected_factories_without_explicit_test_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="forbid injected factories"):
        CopyV2Runner(
            _config(),
            tmp_path,
            tokenizer_factory=lambda _config: FakeTokenizer(),
            plant_factory=lambda implementation, _config: FakePlant(implementation),
        )
