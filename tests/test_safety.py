"""Tests for the safety-observer contract and inert interlock task.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from observerbench.safety import (
    SafetyObserverCard,
    SafetyPolicy,
    evaluate_safety_prediction_csv,
    evaluate_safety_scores,
    write_safety_predictions,
)
from observerbench.safety_task_pack import (
    load_safety_task_pack,
    validate_public_safety_task_pack,
    write_safety_task_pack,
)
from observerbench.tasks.safety_interlock import (
    InterlockRidgeObserver,
    SafetyInterlockConfig,
    make_safety_interlock_task,
    run_safety_interlock,
)


def _quick_config() -> SafetyInterlockConfig:
    return SafetyInterlockConfig(n_train_pairs=64, n_test_pairs=128, seed=7)


def test_interlock_pairs_hold_action_features_fixed_and_change_policy_label() -> None:
    task = make_safety_interlock_task(_quick_config())
    target_by_id = {target.query_id: target for target in task.targets}
    by_pair: dict[str, list] = {}
    for query in task.queries:
        by_pair.setdefault(query.pair_id, []).append(query)

    assert all(len(rows) == 2 for rows in by_pair.values())
    for rows in by_pair.values():
        first, second = rows
        assert first.features.action_only == second.features.action_only
        assert first.metadata["action_span_key"] == second.metadata["action_span_key"]
        assert {
            target_by_id[first.query_id].unsafe_if_allowed,
            target_by_id[second.query_id].unsafe_if_allowed,
        } == {False, True}

    train_families = {row.family_id for row in task.measurements}
    test_families = {row.family_id for row in task.queries}
    assert "delete" not in train_families
    assert "delete" in test_families


def test_safety_policy_uses_fixed_nonoverlapping_action_budgets() -> None:
    task = make_safety_interlock_task(_quick_config())
    scores = np.arange(len(task.queries), dtype=float)
    result = evaluate_safety_scores(task, scores, observer_name="ordered")

    assert sum(result.action_counts.values()) == len(task.queries)
    assert result.action_counts["block"] == int(0.10 * len(task.queries))
    assert result.action_counts["escalate"] == int(0.10 * len(task.queries))
    assert result.action_counts["allow"] > 0


def test_direct_risk_matches_protocol_better_than_pair_blind_action_view() -> None:
    task = make_safety_interlock_task(_quick_config())
    action_only = InterlockRidgeObserver(
        "action-only",
        view="action_only",
        target="violation_probability",
    )
    direct_risk = InterlockRidgeObserver(
        "direct-risk",
        view="activation",
        target="direct_severity_risk",
    )
    from observerbench.safety import evaluate_safety_predictor

    action_result = evaluate_safety_predictor(task, action_only)
    risk_result = evaluate_safety_predictor(task, direct_risk)

    assert risk_result.metrics["protocol_loss_mean"] < action_result.metrics["protocol_loss_mean"]
    assert risk_result.metrics["severity_weighted_miss_rate"] < action_result.metrics["severity_weighted_miss_rate"]


def test_interlock_runner_writes_machine_readable_results(tmp_path: Path) -> None:
    results = run_safety_interlock(_quick_config(), tmp_path)

    assert {result.observer_name for result in results} == {
        "action-only-label-ridge",
        "activation-label-ridge",
        "activation-direct-risk-ridge",
        "exact-authorization-risk-oracle",
        "allow-all-no-action",
    }
    payload = json.loads((tmp_path / "safety_interlock_results.json").read_text())
    assert len(payload) == 5
    assert (tmp_path / "activation-direct-risk-ridge" / "safety_protocol_rows.json").exists()


def test_safety_policy_rejects_overcommitted_budgets() -> None:
    try:
        SafetyPolicy(block_budget_fraction=0.7, escalation_budget_fraction=0.4)
    except ValueError as error:
        assert "cannot exceed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("overcommitted policy budgets should fail")


def test_inference_free_safety_csv_round_trip(tmp_path: Path) -> None:
    task = make_safety_interlock_task(_quick_config())
    target_by_id = {target.query_id: target for target in task.targets}
    scores = [
        float(target_by_id[query.query_id].unsafe_if_allowed)
        * target_by_id[query.query_id].severity
        for query in task.queries
    ]
    predictions = tmp_path / "predictions.csv"
    write_safety_predictions(predictions, task, scores)
    card = SafetyObserverCard(
        observer_name="outside-oracle-test",
        observer_version="test",
        observer_family="test fixture",
        observer_input="evaluator target used only by this contract test",
        fit_procedure="none",
        implementation="tests.test_safety",
        risk_score_meaning="severity if unsafe, otherwise zero",
        access_regime="evaluator oracle",
        requires_white_box_access=False,
        additional_forward_passes=0,
    )
    result = evaluate_safety_prediction_csv(
        task,
        predictions,
        card,
        outdir=tmp_path / "evaluation",
    )

    assert result.observer_name == "outside-oracle-test"
    card_payload = asdict(card)
    card_payload["known_failure_modes"] = []
    assert json.loads((tmp_path / "evaluation" / "safety_observer_card.json").read_text()) == card_payload


def test_safety_observer_card_validates_access_cost_fields() -> None:
    try:
        SafetyObserverCard(
            observer_name="bad-cost",
            observer_version="test",
            observer_family="test fixture",
            observer_input="cached features",
            fit_procedure="none",
            implementation="tests.test_safety",
            risk_score_meaning="larger is riskier",
            access_regime="cached activation",
            additional_forward_passes=-1,
        )
    except ValueError as error:
        assert "additional_forward_passes" in str(error)
    else:  # pragma: no cover
        raise AssertionError("negative forward-pass cost should fail")


def test_researcher_safety_task_pack_keeps_targets_separate_and_round_trips(
    tmp_path: Path,
) -> None:
    original = make_safety_interlock_task(_quick_config())
    public_path, targets_path = write_safety_task_pack(original, tmp_path / "task-pack")

    public_payload = json.loads(public_path.read_text())
    assert "targets" not in public_payload
    summary = validate_public_safety_task_pack(public_path)
    assert summary["n_measurements"] == len(original.measurements)
    assert summary["n_queries"] == len(original.queries)

    loaded = load_safety_task_pack(public_path, targets_path)
    assert loaded.name == original.name
    assert len(loaded.targets) == len(original.targets)
    assert isinstance(loaded.queries[0].features, dict)


def test_researcher_safety_task_pack_rejects_public_file_changed_after_seal(
    tmp_path: Path,
) -> None:
    task = make_safety_interlock_task(_quick_config())
    public_path, targets_path = write_safety_task_pack(task, tmp_path / "task-pack")
    payload = json.loads(public_path.read_text())
    payload["task_card"]["summary"] = "changed after evaluator targets were sealed"
    public_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_safety_task_pack(public_path, targets_path)
    except ValueError as error:
        assert "does not match" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a changed public task should fail its evaluator seal")
