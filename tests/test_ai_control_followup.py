"""Tests for the external AI-control follow-up analyses.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import numpy as np

from observerbench.ai_control_followup import (
    budget_metrics,
    hashed_text_features,
    prevalence_budget_sensitivity,
    select_ridge_binary_observer,
)


def test_hashed_text_features_are_deterministic_and_normalized() -> None:
    first = hashed_text_features(["alpha beta", "alpha gamma"], dimension=64)
    second = hashed_text_features(["alpha beta", "alpha gamma"], dimension=64)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)


def test_ridge_selection_uses_locked_rows_only_for_prediction() -> None:
    features = np.asarray([[0.0], [1.0], [0.1], [0.9], [0.2], [0.8]])
    labels = [False, True, False, True, False, True]
    splits = ["fit", "fit", "calibration", "calibration", "locked_test", "locked_test"]
    selection, scores = select_ridge_binary_observer(
        features,
        labels,
        splits,
        ridge_grid=(0.01, 1.0),
        selection_budget=0.5,
    )
    assert selection["ridge"] in {0.01, 1.0}
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_budget_metrics_and_prevalence_oracle() -> None:
    metrics = budget_metrics([True, True, False, False], [4.0, 3.0, 2.0, 1.0], budget=0.5)
    assert metrics["realized_violations"] == 0.0
    assert metrics["audit_precision"] == 1.0
    sensitivity = prevalence_budget_sensitivity(
        {
            "oracle": {"attack": [1.0, 1.0], "honest": [0.0, 0.0]},
            "reverse": {"attack": [0.0, 0.0], "honest": [1.0, 1.0]},
        },
        prevalences=(0.25,),
        budgets=(0.25,),
        population_size=20,
        replicates=20,
        seed=3,
    )
    cell = sensitivity["cells"]["prevalence=0.25|budget=0.25"]
    assert cell["monitors"]["oracle"]["excess_over_oracle"]["mean"] == 0.0
    assert cell["monitors"]["reverse"]["realized_violations"]["mean"] == 5.0
