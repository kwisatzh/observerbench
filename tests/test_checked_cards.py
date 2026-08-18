"""Tests for ObserverCards composed from checked paper results.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from observerbench.cards import (
    load_cards_from_results,
    validate_observer_card,
    validate_observer_card_bundle,
    write_observer_card_bundle,
)
from observerbench.paper_cards import build_phase4_cards, write_phase4_observer_cards


ROOT = Path(__file__).resolve().parents[1]


def _assert_finite(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite(nested)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_phase04_cards_are_checked_claim_cards(tmp_path: Path) -> None:
    cards = build_phase4_cards(ROOT)
    assert len(cards) == 5
    assert {card["task_name"] for card in cards} == {
        "ctl1_analytic",
        "trained_ctl1",
        "trained_ctl2",
        "ioi_stage2b",
        "ioi_stage2c",
    }
    assert all(card["observer_name"] != "task_summary" for card in cards)
    assert all(card["task_name"] != "unknown_task" for card in cards)
    for card in cards:
        validate_observer_card(card)
        _assert_finite(card)
        serialized = json.dumps(card)
        assert "/Users/" not in serialized

    ctl2 = next(card for card in cards if card["task_name"] == "trained_ctl2")
    assert ctl2["primary_metrics"]["positive_seed_level_fixed_direction_contrasts"] == 48
    assert ctl2["primary_metrics"]["total_seed_level_fixed_direction_contrasts"] == 48
    assert "Do not use the uncalibrated first-order estimator" in ctl2["recommendation"]

    stage2b = next(card for card in cards if card["task_name"] == "ioi_stage2b")
    stage2c = next(card for card in cards if card["task_name"] == "ioi_stage2c")
    assert stage2b["primary_metrics"]["rank_added_per_single_pair"] == 4
    assert stage2b["primary_metrics"]["all_pairs_delta_mae_vs_additive_q05"] > 0
    assert "strong baseline" in stage2b["recommendation"]
    assert stage2c["primary_metrics"]["PE_add_one_delta_mae_q05"] > 0
    assert stage2c["primary_metrics"]["PB_add_one_delta_mae_q05"] < 0
    assert stage2c["primary_metrics"]["PB_leave_one_out_delta_mae_q05"] > 0
    assert stage2c["primary_metrics"]["direct_PE_minus_PB_q05"] > 0

    json_path, markdown_path = write_phase4_observer_cards(ROOT, tmp_path)
    bundle = json.loads(json_path.read_text(encoding="utf-8"))
    validate_observer_card_bundle(bundle)
    assert bundle["schema_version"].endswith(".v2")
    assert "unknown_task" not in markdown_path.read_text(encoding="utf-8")


def test_make_card_understands_checked_capacity_directory(tmp_path: Path) -> None:
    results = ROOT / "results/revision/phase02/ioi_stage2c_capacity"
    cards = load_cards_from_results(results, tmp_path)
    assert len(cards) == 1
    assert cards[0]["task_name"] == "ioi_stage2c"
    assert cards[0]["observer_name"] == "capacity_matched_all_pairs"


def test_ctl2_card_preserves_sign_compatibility_metric(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame([
        {
            "observer": "sign_incompatible",
            "integrated_squared_error": 1.0,
            "cumulative_collateral_abs": 1.0,
            "observer_self_gain": -0.1,
            "observer_direction_sign_compatible": False,
            "large_error_trajectory_rate": 0.0,
            "target_error_worsened_rate": 0.0,
            "observer_bias_mae_path": 0.0,
        }
    ]).to_csv(results / "trained_transformer_ctl2_results.csv", index=False)
    cards_dir = tmp_path / "cards"
    write_observer_card_bundle(results, cards_dir)
    bundle = json.loads((cards_dir / "observer_card.json").read_text())
    card = bundle["cards"][0]
    assert card["primary_metrics"]["observer_direction_sign_compatible"] is False
    assert any("non-positive" in value for value in card["failure_modes_detected"])

