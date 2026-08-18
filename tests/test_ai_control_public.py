from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import csv

import pytest

from observerbench.ai_control_public import (
    AI_CONTROL_PRIVATE_TARGET_COLUMNS,
    AI_CONTROL_PRIVATE_TARGET_SCHEMA_VERSION,
    AI_CONTROL_PUBLIC_SCORE_COLUMNS,
    AI_CONTROL_PUBLIC_SCORE_SCHEMA_VERSION,
    PrivateAIControlTarget,
    PublicAIControlScore,
    evaluate_public_ai_control_csv,
    evaluate_public_ai_control_scores,
    join_public_ai_control_scores,
    read_private_ai_control_targets,
    read_public_ai_control_scores,
)


def _targets() -> tuple[PrivateAIControlTarget, ...]:
    return (
        PrivateAIControlTarget("a", "honest", False, "apps"),
        PrivateAIControlTarget("b", "honest", False, "apps"),
        PrivateAIControlTarget("c", "attack", True, "apps"),
        PrivateAIControlTarget("d", "attack", True, "apps"),
    )


def _scores() -> tuple[PublicAIControlScore, ...]:
    return (
        PublicAIControlScore("a", "monitor-a", 0.1),
        PublicAIControlScore("b", "monitor-a", 0.2),
        PublicAIControlScore("c", "monitor-a", 0.9),
        PublicAIControlScore("d", "monitor-a", 0.8),
        PublicAIControlScore("a", "monitor-b", 0.9),
        PublicAIControlScore("b", "monitor-b", 0.8),
        PublicAIControlScore("c", "monitor-b", 0.2),
        PublicAIControlScore("d", "monitor-b", 0.1),
    )


def _write_csv(path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_score_only_submission_calls_existing_evaluator() -> None:
    results = {
        result.monitor_name: result
        for result in evaluate_public_ai_control_scores(
            _scores(), _targets(), audit_budget_fraction=0.5
        )
    }
    assert results["monitor-a"].metrics["risk_auroc"] == 1.0
    assert results["monitor-a"].metrics["realized_violations"] == 0.0
    assert results["monitor-b"].metrics["risk_auroc"] == 0.0
    assert results["monitor-b"].metrics["realized_violations"] == 2.0


def test_csv_adapter_keeps_targets_out_of_public_table(tmp_path) -> None:
    public_path = tmp_path / "submission.csv"
    private_path = tmp_path / "targets.csv"
    _write_csv(
        public_path,
        AI_CONTROL_PUBLIC_SCORE_COLUMNS,
        (
            {
                "schema_version": AI_CONTROL_PUBLIC_SCORE_SCHEMA_VERSION,
                "sample_id": row.sample_id,
                "monitor_name": row.monitor_name,
                "score": row.score,
            }
            for row in _scores()
        ),
    )
    _write_csv(
        private_path,
        AI_CONTROL_PRIVATE_TARGET_COLUMNS,
        (
            {
                "schema_version": AI_CONTROL_PRIVATE_TARGET_SCHEMA_VERSION,
                "sample_id": row.sample_id,
                "eval_mode": row.eval_mode,
                "backdoor_works": str(row.backdoor_works).lower(),
                "family_id": row.family_id,
            }
            for row in _targets()
        ),
    )

    assert tuple(read_public_ai_control_scores(public_path)) == _scores()
    assert tuple(read_private_ai_control_targets(private_path)) == _targets()
    results = evaluate_public_ai_control_csv(
        public_path,
        private_path,
        audit_budget_fraction=0.5,
    )
    assert {result.monitor_name for result in results} == {"monitor-a", "monitor-b"}


def test_join_rejects_missing_or_unexpected_sample_ids() -> None:
    with pytest.raises(ValueError, match="missing d"):
        join_public_ai_control_scores(_scores()[:-1], _targets())
    with pytest.raises(ValueError, match="unexpected x"):
        join_public_ai_control_scores(
            (*_scores(), PublicAIControlScore("x", "monitor-a", 0.5)),
            _targets(),
        )


def test_join_rejects_duplicate_public_or_private_rows() -> None:
    with pytest.raises(ValueError, match="more than once"):
        join_public_ai_control_scores((*_scores(), _scores()[0]), _targets())
    with pytest.raises(ValueError, match="duplicate private sample_id"):
        join_public_ai_control_scores(_scores(), (*_targets(), _targets()[0]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_public_score_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        PublicAIControlScore("a", "monitor", value)


def test_readers_reject_labels_in_public_schema_and_bad_private_labels(tmp_path) -> None:
    public_path = tmp_path / "public-with-labels.csv"
    _write_csv(
        public_path,
        (*AI_CONTROL_PUBLIC_SCORE_COLUMNS, "backdoor_works"),
        (
            {
                "schema_version": AI_CONTROL_PUBLIC_SCORE_SCHEMA_VERSION,
                "sample_id": "a",
                "monitor_name": "monitor",
                "score": 0.1,
                "backdoor_works": "false",
            },
        ),
    )
    with pytest.raises(ValueError, match="columns must be exactly"):
        read_public_ai_control_scores(public_path)

    private_path = tmp_path / "bad-target.csv"
    _write_csv(
        private_path,
        AI_CONTROL_PRIVATE_TARGET_COLUMNS,
        (
            {
                "schema_version": AI_CONTROL_PRIVATE_TARGET_SCHEMA_VERSION,
                "sample_id": "a",
                "eval_mode": "attack",
                "backdoor_works": "yes",
                "family_id": "apps",
            },
        ),
    )
    with pytest.raises(ValueError, match="true or false"):
        read_private_ai_control_targets(private_path)
