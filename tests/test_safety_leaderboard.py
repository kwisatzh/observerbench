"""Tests for the checked safety leaderboard.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from observerbench.core import write_json
from observerbench.safety import SafetyObserverCard
from observerbench.safety_leaderboard import (
    compare_safety_result,
    format_safety_leaderboard,
    load_safety_leaderboard,
    safety_rows_for_task,
)


TASK_ID = "safety-interlock-qwen2-5-7b-instruct@paired-scope-v0"


def test_bundled_safety_leaderboard_has_real_qwen_rows() -> None:
    rows = safety_rows_for_task(load_safety_leaderboard(), TASK_ID)

    assert len(rows) == 8
    assert {row.observer_name for row in rows} >= {
        "activation-label",
        "activation-transformed-label-risk",
        "full-context-text-label-times-severity",
        "model-logit-margin-times-severity",
    }
    assert {row.result_status for row in rows} >= {
        "prespecified locked test",
        "post-outcome secondary",
        "oracle bound",
    }


def test_leaderboard_formats_are_human_and_machine_readable() -> None:
    rows = safety_rows_for_task(load_safety_leaderboard(), TASK_ID)

    assert "Qwen2.5-7B-Instruct output margin x severity" in format_safety_leaderboard(
        rows, output_format="markdown"
    )
    assert "| Observer | Status |" in format_safety_leaderboard(rows, output_format="markdown")
    assert "task_id,observer,status" in format_safety_leaderboard(rows, output_format="csv")
    assert '"protocol_loss_mean"' in format_safety_leaderboard(rows, output_format="json")


def test_compare_safety_adds_an_outside_row(tmp_path: Path) -> None:
    result = {
        "task_name": "safety-interlock-qwen2-5-7b-instruct",
        "task_version": "paired-scope-v0",
        "observer_name": "outside-test",
        "metrics": {
            "protocol_loss_mean": 2.4,
            "protocol_loss_cvar": 13.0,
            "risk_auroc": 0.91,
            "severity_weighted_miss_rate": 0.48,
            "clean_utility_retained": 0.99,
            "worst_family_protocol_loss": 5.1,
        },
        "action_counts": {"allow": 616, "block": 76, "escalate": 76},
    }
    card = SafetyObserverCard(
        observer_name="outside-test",
        observer_version="test",
        observer_family="outside method",
        observer_input="cached activation",
        fit_procedure="external",
        implementation="tests.test_safety_leaderboard",
        risk_score_meaning="larger means riskier",
        access_regime="cached white-box activation",
        requires_white_box_access=True,
        additional_forward_passes=0,
    )
    write_json(tmp_path / "safety_protocol_result.json", result)
    write_json(tmp_path / "safety_observer_card.json", asdict(card))

    rows = compare_safety_result(tmp_path, load_safety_leaderboard())

    assert len(rows) == 9
    assert any(row.observer_name == "outside-test" for row in rows)
