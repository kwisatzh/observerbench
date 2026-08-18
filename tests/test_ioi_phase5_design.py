"""Tests for the locked Phase 5 IOI prompt and mask design.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from observerbench.tasks.ioi.phase5_design import (
    PHASE5_TEMPLATES,
    PROMPT_SPLITS,
    build_mask_design,
    load_legacy_mask_bits,
    load_phase5_protocol,
    prepare_phase5_design,
    template_frame,
    validate_single_token_names,
    write_phase5_design,
)
from observerbench.tasks.ioi.phase5_effects import load_locked_ioi_design


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/revision/ioi_phase05_confirmatory_v2.json"
LEGACY_PATHS = (
    ROOT
    / "results/revision/phase02/inputs/stage2b_mean_end/ioi_stage2b_subset_design.csv",
    ROOT
    / "results/frozen/ioi/stage2c_primary_stratified_mean_end/ioi_stage2c_subset_design.csv",
)


@pytest.fixture(scope="module")
def frozen_design():
    protocol = load_phase5_protocol(PROTOCOL_PATH)
    names = list(protocol["name_candidates"])
    fake_tokens = {f" {name}": [index + 100] for index, name in enumerate(names)}
    return prepare_phase5_design(
        protocol,
        legacy_masks=load_legacy_mask_bits(LEGACY_PATHS),
        encode_name=lambda text: fake_tokens[text],
    )


def test_eight_literal_templates_encode_abba_and_baba() -> None:
    templates = template_frame()
    assert len(PHASE5_TEMPLATES) == 8
    assert templates.groupby("structure").size().to_dict() == {"ABBA": 4, "BABA": 4}
    assert templates["template_id"].is_unique
    assert templates["template_hash"].is_unique


def test_prompt_banks_are_disjoint_and_have_locked_counts(frozen_design) -> None:
    prompts = frozen_design.prompts
    names = frozen_design.names
    assert prompts.groupby("split").size().to_dict() == {
        "reference": 512,
        "train": 192,
        "validation": 64,
        "test": 256,
    }
    expected_per_template = {
        "reference": 64,
        "train": 24,
        "validation": 8,
        "test": 32,
    }
    for split in PROMPT_SPLITS:
        assert set(
            prompts.loc[prompts["split"] == split]
            .groupby("template_id")
            .size()
        ) == {expected_per_template[split]}
    banks = {
        split: set(names.loc[names["split"] == split, "name"])
        for split in PROMPT_SPLITS
    }
    assert all(len(bank) == 16 for bank in banks.values())
    assert sum(map(len, banks.values())) == len(set().union(*banks.values()))
    assert prompts["prompt_id"].is_unique
    assert prompts["prompt"].is_unique


def test_name_token_validation_rejects_multi_token_and_aliases() -> None:
    assert validate_single_token_names(
        ["Alice", "Bob"],
        lambda text: [1] if text == " Alice" else [2],
    ) == {"Alice": 1, "Bob": 2}
    with pytest.raises(ValueError, match="not one"):
        validate_single_token_names(["Alice"], lambda _text: [1, 2])
    with pytest.raises(ValueError, match="share token"):
        validate_single_token_names(["Alice", "Bob"], lambda _text: [1])


def test_masks_pass_leakage_balance_anchor_and_rank_gates(frozen_design) -> None:
    audit = frozen_design.leakage_audit
    assert audit["all_gates_pass"]
    assert all(audit["gates"].values())

    calibration = frozen_design.calibration_masks.sort_values("measurement_order")
    anchors = ["0" * 13]
    anchors.extend(
        "".join("1" if index == head else "0" for index in range(13))
        for head in range(13)
    )
    assert calibration.head(14)["mask_bits"].tolist() == anchors
    assert len(calibration) == 160
    assert len(frozen_design.candidate_masks) == 320
    assert frozen_design.candidate_masks.groupby("pool_id").size().eq(32).all()

    primary = frozen_design.rank_diagnostics.query("budget == 160")
    assert primary["full_attainable_rank"].all()
    budget_40 = frozen_design.rank_diagnostics.query("budget == 40")
    assert budget_40["full_attainable_rank"].all()


def test_small_mask_design_is_deterministic() -> None:
    kwargs = {
        "legacy_masks": set(),
        "seed": 19,
        "calibration_count": 40,
        "candidate_pool_count": 1,
        "candidate_head_counts": (4, 5, 6, 7, 8, 9, 10),
        "masks_per_stratum_by_head_count": {
            4: 2,
            5: 2,
            6: 3,
            7: 3,
            8: 2,
            9: 2,
            10: 2,
        },
    }
    calibration_a, candidates_a = build_mask_design(**kwargs)
    calibration_b, candidates_b = build_mask_design(**kwargs)
    pd.testing.assert_frame_equal(calibration_a, calibration_b)
    pd.testing.assert_frame_equal(candidates_a, candidates_b)


def test_written_design_matches_effect_loader_contract(tmp_path: Path, frozen_design) -> None:
    output = write_phase5_design(
        frozen_design,
        tmp_path / "design",
        protocol_path=PROTOCOL_PATH,
        legacy_paths=LEGACY_PATHS,
    )
    prompts, masks, manifest = load_locked_ioi_design(output)
    assert len(prompts) == 1024
    assert masks.groupby("bank").size().to_dict() == {
        "calibration": 160,
        "candidate": 320,
    }
    assert manifest["status"] == "frozen_before_outcomes"
    assert not manifest["contains_model_outcomes"]
    assert set(manifest["source_hashes"]) == {"prompts.csv", "masks.csv"}
