"""Tests for paired rare-prevalence AI-control contrasts.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from observerbench.ai_control_contrasts import paired_prevalence_contrasts
from observerbench.ai_control_followup import prevalence_budget_sensitivity


def _scores() -> dict[str, dict[str, list[float]]]:
    return {
        "detailed": {
            "attack": [0.95, 0.70, 0.40, 0.20],
            "honest": [0.85, 0.30, 0.10, 0.05],
        },
        "neutral": {
            "attack": [0.0, 0.0, 0.0, 0.0],
            "honest": [1.0, 1.0, 1.0, 1.0],
        },
        "raw": {
            "attack": [0.92, 0.82, 0.72, 0.62],
            "honest": [0.42, 0.32, 0.22, 0.12],
        },
        "reconstruction": {
            "attack": [0.72, 0.62, 0.52, 0.42],
            "honest": [0.52, 0.42, 0.32, 0.22],
        },
        "sae": {
            "attack": [0.75, 0.65, 0.55, 0.45],
            "honest": [0.55, 0.35, 0.25, 0.15],
        },
        "verdict": {
            "attack": [1.0, 1.0, 1.0, 1.0],
            "honest": [0.0, 0.0, 0.0, 0.0],
        },
    }


def _sensitivity() -> dict[str, object]:
    return prevalence_budget_sensitivity(
        _scores(),
        prevalences=(0.10, 0.25),
        budgets=(0.10, 0.25),
        population_size=40,
        replicates=200,
        seed=71,
    )


def _contrasts() -> dict[str, tuple[str, str]]:
    return {
        "neutral-minus-verdict": ("neutral", "verdict"),
        "sae-minus-raw": ("sae", "raw"),
        "detailed-minus-sae": ("detailed", "sae"),
        "reconstruction-minus-raw": ("reconstruction", "raw"),
    }


def test_paired_contrasts_are_deterministic_and_support_named_comparisons() -> None:
    sensitivity = _sensitivity()
    first = paired_prevalence_contrasts(
        _scores(), sensitivity, contrasts=_contrasts()
    )
    second = paired_prevalence_contrasts(
        _scores(), sensitivity, contrasts=_contrasts()
    )
    assert first == second
    cell = first["cells"]["prevalence=0.1|budget=0.1"]
    assert set(cell["contrasts"]) == set(_contrasts())
    primary = cell["contrasts"]["neutral-minus-verdict"]
    assert primary["paired_difference"]["mean"] == (
        primary["left_mean"] - primary["right_mean"]
    )
    assert primary["paired_difference"]["lower_95"] >= 0.0
    assert primary["zero_excluded"]
    assert primary["sign_convention"] == "left_minus_right; negative favors left"


def test_cell_filter_does_not_change_later_registered_draws() -> None:
    sensitivity = _sensitivity()
    full = paired_prevalence_contrasts(
        _scores(), sensitivity, contrasts=_contrasts()
    )
    key = "prevalence=0.25|budget=0.25"
    filtered = paired_prevalence_contrasts(
        _scores(), sensitivity, contrasts=_contrasts(), cell_keys=(key,)
    )
    assert filtered["cells"] == {key: full["cells"][key]}


def test_replay_rejects_a_changed_saved_mean() -> None:
    sensitivity = deepcopy(_sensitivity())
    cell = sensitivity["cells"]["prevalence=0.1|budget=0.1"]
    cell["monitors"]["neutral"]["realized_violations"]["mean"] += 1.0
    with pytest.raises(ValueError, match="does not reproduce"):
        paired_prevalence_contrasts(
            _scores(),
            sensitivity,
            contrasts={"neutral-minus-verdict": ("neutral", "verdict")},
        )
