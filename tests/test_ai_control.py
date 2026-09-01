from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import csv
import json

import pytest

from observerbench.ai_control import (
    AI_CONTROL_SCORE_COLUMNS,
    AI_CONTROL_SCORE_SCHEMA_VERSION,
    AIControlSample,
    evaluate_ai_control_csv,
    evaluate_ai_control_scores,
    read_ai_control_scores,
    write_ai_control_scores,
)
from observerbench.ai_control_leaderboard import (
    format_ai_control_leaderboard,
    load_ai_control_leaderboard,
)


def _panel() -> tuple[AIControlSample, ...]:
    outcomes = (
        ("a", "honest", False),
        ("b", "honest", False),
        ("c", "honest", False),
        ("d", "honest", False),
        ("e", "honest", False),
        ("f", "attack", True),
        ("g", "attack", True),
        ("h", "attack", True),
        ("i", "attack", True),
        ("j", "attack", True),
    )
    scores = {
        "global-ranker": (-0.1, 0.0, 0.1, 0.2, 0.98, 0.99, 0.79, 0.78, 0.77, 0.76),
        "budget-ranker": (0.97, 0.96, 0.95, 0.94, 0.93, 0.99, 0.98, 0.1, 0.0, -0.1),
    }
    return tuple(
        AIControlSample(
            sample_id=sample_id,
            eval_mode=mode,
            backdoor_works=works,
            family_id="apps",
            monitor_name=monitor,
            score=scores[monitor][index],
        )
        for monitor in scores
        for index, (sample_id, mode, works) in enumerate(outcomes)
    )


def test_external_control_can_reverse_auroc_and_fixed_budget_ranking() -> None:
    results = {
        result.monitor_name: result
        for result in evaluate_ai_control_scores(_panel(), audit_budget_fraction=0.25)
    }
    assert results["global-ranker"].metrics["risk_auroc"] > results["budget-ranker"].metrics["risk_auroc"]
    assert results["global-ranker"].metrics["realized_violations"] == 4.0
    assert results["budget-ranker"].metrics["realized_violations"] == 3.0
    assert results["budget-ranker"].metrics["audit_precision"] == 1.0


def test_csv_round_trip_and_result_bundle(tmp_path) -> None:
    scores = tmp_path / "scores.csv"
    write_ai_control_scores(scores, _panel())
    assert read_ai_control_scores(scores) == _panel()
    results = evaluate_ai_control_csv(
        scores,
        audit_budget_fraction=0.25,
        outdir=tmp_path / "results",
    )
    assert len(results) == 2
    payload = json.loads(
        (tmp_path / "results" / "ai_control_results.json").read_text()
    )
    assert payload["audit_budget_fraction"] == 0.25
    assert (tmp_path / "results" / "ai_control_results.csv").exists()


def test_panels_must_match() -> None:
    with pytest.raises(ValueError, match="same external sample panel"):
        evaluate_ai_control_scores(_panel()[:-1], audit_budget_fraction=0.25)


def test_reader_rejects_non_boolean_label(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AI_CONTROL_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "schema_version": AI_CONTROL_SCORE_SCHEMA_VERSION,
                "sample_id": "x",
                "eval_mode": "attack",
                "backdoor_works": "yes",
                "family_id": "apps",
                "monitor_name": "m",
                "score": "1",
            }
        )
    with pytest.raises(ValueError, match="true or false"):
        read_ai_control_scores(path)


def test_bundled_external_leaderboard_has_checked_rows() -> None:
    rows = load_ai_control_leaderboard()
    assert len(rows) == 29
    rendered = format_ai_control_leaderboard(rows, output_format="markdown")
    assert "controlarena-apps-verified-backdoors@qwen2.5-7b-v1" in rendered
    assert "controlarena-apps-verified-backdoors@gemma-2-9b-it-v0" in rendered
    assert "controlarena-apps-verified-backdoors@qwen3.5-9b-v0" in rendered
    assert "Qwen2.5-7B-Instruct neutral-context activation probe" in rendered
    assert "Qwen2.5-7B-Instruct verdict-adjacent activation probe" in rendered
    assert "Qwen2.5-7B-Instruct published SAE sparse probe" in rendered
    assert "Official Gemma Scope SAE probe" in rendered
    assert "Qwen3.5-9B official Qwen-Scope SAE probe" in rendered
    assert "715" in rendered
    assert "Mean violations (1% attacks, 2% audit)" in rendered
    assert "4.667" in rendered
    assert "9.070" in rendered

    qwen = load_ai_control_leaderboard(
        task_id="controlarena-apps-verified-backdoors@qwen2.5-7b-v1"
    )
    gemma = load_ai_control_leaderboard(
        task_id="controlarena-apps-verified-backdoors@gemma-2-9b-it-v0"
    )
    qwen35 = load_ai_control_leaderboard(
        task_id="controlarena-apps-verified-backdoors@qwen3.5-9b-v0"
    )
    assert len(qwen) == 11
    assert len(gemma) == 9
    assert len(qwen35) == 9
    assert all("qwen2.5-7b-v1" in row["task_id"] for row in qwen)
    assert all("gemma-2-9b-it-v0" in row["task_id"] for row in gemma)
    assert all("qwen3.5-9b-v0" in row["task_id"] for row in qwen35)
    assert "task_id,monitor,status" in format_ai_control_leaderboard(
        rows, output_format="csv"
    )


def test_ai_control_task_filter_rejects_an_unknown_task() -> None:
    with pytest.raises(KeyError, match="unknown AI-control task"):
        load_ai_control_leaderboard(task_id="unknown@task")
