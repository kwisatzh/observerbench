"""Tests for the sealed prospective Phase-6 IOI design.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from observerbench.provenance import file_sha256
from observerbench.tasks.ioi.phase6_confirmatory import (
    DESIGN_SCHEMA,
    DESIGN_STATUS,
    PHASE6_STATUS,
    _head_quadratic_rank,
    build_test_pair_clusters,
    load_phase6_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/revision/ioi_phase06_fresh_confirmation_v1.json"
DESIGN = ROOT / "results/revision/phase06/ioi_fresh_confirmation/design"


def test_protocol_freezes_primary_multiplicity_and_access_boundary() -> None:
    protocol = load_phase6_protocol(CONFIG)
    assert protocol["status"] == PHASE6_STATUS
    assert protocol["primary_targets"] == [0.5, 1.0]
    assert protocol["stress_test_targets"] == [1.5]
    assert protocol["candidate_pool_count"] == 48
    assert "H1 alone defines primary success" in protocol["multiplicity_boundary"]
    assert "cannot rescue" in protocol["multiplicity_boundary"]
    assert "Only after that manifest" in protocol["access_sequence"][-1]
    assert protocol["bootstrap_interval"]["quantiles"] == [0.025, 0.975]
    assert protocol["bootstrap_repeats"] == 5000


def test_hash_pairing_is_disjoint_balanced_and_input_order_invariant() -> None:
    names = [f"Name{index:02d}" for index in range(64)]
    forward = build_test_pair_clusters(names, seed=26062)
    reverse = build_test_pair_clusters(list(reversed(names)), seed=26062)
    pd.testing.assert_frame_equal(forward, reverse)
    assert len(forward) == 32
    assert not forward.duplicated(["name_a", "name_b"]).any()
    degree = pd.concat(
        [forward["name_a"], forward["name_b"]], ignore_index=True
    ).value_counts()
    assert set(degree.to_numpy(int)) == {1}


def test_materialized_design_is_outcome_free_fresh_and_hash_locked() -> None:
    manifest = json.loads((DESIGN / "design_manifest.json").read_text())
    audit = json.loads((DESIGN / "leakage_audit.json").read_text())
    assert manifest["schema"] == DESIGN_SCHEMA
    assert manifest["status"] == DESIGN_STATUS
    assert manifest["contains_model_outcomes"] is False
    assert manifest["phase6_forward_passes_performed"] is False
    assert manifest["predictions_frozen"] is False
    assert manifest["all_design_gates_pass"] is True
    assert audit["all_gates_pass"] is True
    assert all(audit["gates"].values())
    for filename, digest in manifest["artifact_hashes"].items():
        assert file_sha256(DESIGN / filename) == digest


def test_materialized_splits_pairs_and_masks_match_frozen_counts() -> None:
    protocol = load_phase6_protocol(CONFIG)
    names = pd.read_csv(DESIGN / "names.csv")
    prompts = pd.read_csv(DESIGN / "prompts.csv")
    pairs = pd.read_csv(DESIGN / "pair_clusters.csv")
    calibration = pd.read_csv(
        DESIGN / "calibration_masks.csv", dtype={"mask_bits": str}
    )
    candidates = pd.read_csv(
        DESIGN / "candidate_masks.csv", dtype={"mask_bits": str}
    )
    phase5_masks = pd.read_csv(
        ROOT / "results/revision/phase05/design/masks.csv",
        dtype={"mask_bits": str},
    )

    assert names.groupby("split").size().to_dict() == {
        "reference": 16,
        "train": 12,
        "test": 64,
    }
    assert names["leading_space_token_id"].is_unique
    assert prompts.groupby("split").size().to_dict() == {
        "reference": 512,
        "train": 192,
        "test": 512,
    }
    test = prompts[prompts["split"] == "test"]
    assert len(pairs) == 32
    assert set(test.groupby("unordered_name_pair_id").size()) == {16}
    assert set(test.groupby(["unordered_name_pair_id", "template_id"]).size()) == {2}
    assert set(test["pair_orientation"]) == {"a_to_b", "b_to_a"}

    expected_calibration = phase5_masks[phase5_masks["bank"] == "calibration"]
    assert calibration.sort_values("measurement_order")["mask_bits"].tolist() == (
        expected_calibration.sort_values("measurement_order")["mask_bits"].tolist()
    )
    assert _head_quadratic_rank(calibration) == 92
    assert len(candidates) == 1536
    assert candidates["pool_id"].nunique() == 48
    assert set(candidates.groupby("pool_id").size()) == {32}
    assert not (
        set(candidates["mask_bits"]) & set(phase5_masks["mask_bits"])
    )
    assert set(candidates.groupby(["pool_id", "sampling_stratum"]).size()) == {16}
    assert protocol["new_candidate_mask_count"] == len(candidates)


def test_no_model_outcome_column_is_present_in_design_artifacts() -> None:
    forbidden = {"clean_ld", "ablated_ld", "drop_from_clean", "target_loss"}
    for filename in (
        "templates.csv",
        "names.csv",
        "pair_clusters.csv",
        "prompts.csv",
        "calibration_masks.csv",
        "candidate_masks.csv",
        "masks.csv",
    ):
        frame = pd.read_csv(DESIGN / filename)
        assert not (forbidden & set(frame.columns))
