# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from dataclasses import replace

import numpy as np
import pytest

from observerbench.tasks.evidence_integrity import (
    EvidenceIntegrityConfig, VARIANTS, evidence_scores,
    make_evidence_integrity_task, reveal_measurements, run_evidence_integrity,
)


@pytest.mark.parametrize("variant", VARIANTS)
def test_only_purchased_facts_are_revealed_and_full_facts_recover_rule(variant):
    cfg = EvidenceIntegrityConfig(variant=variant, fit_examples=100, test_examples=100)
    task, bank = make_evidence_integrity_task(cfg)
    assert all(q.features["trusted_observation"] is None for q in task.queries)
    chosen = [task.queries[0].query_id]
    measured = reveal_measurements(task, bank, chosen, 1)
    assert sum(q.features["trusted_observation"] is not None for q in measured.queries) == 1
    assert "unsafe_if_allowed" not in measured.queries[0].features["trusted_observation"]
    complete = reveal_measurements(task, bank, list(bank), len(bank))
    scores = evidence_scores(complete.measurements, complete.queries)
    expected = [float(t.unsafe_if_allowed)*t.severity for t in complete.targets]
    np.testing.assert_equal(scores, expected)
    # The unmeasured task has not been modified in-place.
    assert all(q.features["trusted_observation"] is None for q in task.queries)


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "over_budget"])
def test_bad_measurement_plans_are_rejected(kind):
    task, bank = make_evidence_integrity_task(EvidenceIntegrityConfig(fit_examples=20, test_examples=20))
    ids = [q.query_id for q in task.queries]
    plan = [ids[0],ids[0]] if kind == "duplicate" else ["missing"] if kind == "unknown" else ids[:2]
    with pytest.raises(ValueError):
        reveal_measurements(task, bank, plan, 1)


def test_zero_measurement_methods_agree_and_cost_is_charged():
    cfg = EvidenceIntegrityConfig(fit_examples=100, test_examples=100, measurement_fraction=0)
    a = run_evidence_integrity(cfg, name="random")
    b = run_evidence_integrity(cfg, name="uncertainty-severity")
    assert a["protocol"]["metrics"] == b["protocol"]["metrics"]
    queried = run_evidence_integrity(replace(cfg, measurement_fraction=.1), name="random")
    assert queried["measurements_used"] == 10
    assert queried["total_loss_with_measurement"] == pytest.approx(queried["protocol"]["metrics"]["protocol_loss_mean"]+.001)


def test_same_report_can_hide_different_pending_operations():
    task, _ = make_evidence_integrity_task(EvidenceIntegrityConfig(fit_examples=100, test_examples=100))
    q = task.queries[0]
    twins = [q, replace(q, query_id="different-opaque-id")]
    scores = evidence_scores(task.measurements, twins)
    assert scores[0] == scores[1]
    # Facts are outside the free view: either operation is consistent with it.
    assert q.features["trusted_observation"] is None


def test_registry_and_task_identity_bind_the_whole_config():
    from observerbench.tasks.safety_registry import load_safety_task, safety_task_specs
    for spec in safety_task_specs():
        if spec.name.startswith("evidence_integrity_"):
            task = load_safety_task(spec.task_id)
            assert spec.task_id == f"{task.name}@{task.version}"
    cfg = EvidenceIntegrityConfig(fit_examples=20,test_examples=20)
    a,_=make_evidence_integrity_task(cfg)
    b,_=make_evidence_integrity_task(replace(cfg,measurement_cost=.02))
    assert a.version != b.version
