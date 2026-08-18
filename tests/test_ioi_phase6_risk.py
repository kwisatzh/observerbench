"""Tests for the post-outcome IOI direct-risk exploration.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from observerbench.tasks.ioi.phase6_risk import (
    EXPLORATORY_STATUS,
    IOIRiskExploratoryConfig,
    POLICY_TARGET,
    SELECTOR_DIRECT_RISK,
    SELECTOR_MEAN_EFFECT,
    _comparison_pairs,
    _head_quadratic_design,
    decision_quality,
    evaluate_fixed_masks,
    fixed_action_oracle,
    select_fixed_masks,
)


def test_config_requires_explicit_post_outcome_status() -> None:
    mapping = {
        "schema": "observerbench.ioi_risk_exploratory.v1",
        "status": "confirmatory",
        "measurement_budgets": [20, 40, 80, 160],
        "primary_budget": 160,
        "targets": [0.5, 1.0, 1.5],
        "models": ["additive_head"],
        "ridge": 1e-6,
        "target_tolerance": 0.25,
        "head_cost_penalty": 0.02,
        "bootstrap_repeats": 20,
        "seed": 1,
    }
    with pytest.raises(ValueError, match="exploratory and post-outcome"):
        IOIRiskExploratoryConfig.from_mapping(mapping)


def test_direct_risk_selects_one_fixed_mask_per_pool_target() -> None:
    candidates = pd.DataFrame(
        {
            "mask_id": ["a", "b", "c", "d"],
            "pool_id": ["p0", "p0", "p1", "p1"],
            "n_heads": [5, 4, 6, 3],
            "size_match_cell": ["n5", "n4", "n6", "n3"],
        }
    )
    rows = []
    scores = {
        SELECTOR_DIRECT_RISK: [0.1, 0.3, 0.2, 0.4],
        SELECTOR_MEAN_EFFECT: [0.4, 0.2, 0.5, 0.1],
    }
    for family, values in scores.items():
        for mask, score in zip(candidates["mask_id"], values):
            rows.append(
                {
                    "analysis_status": EXPLORATORY_STATUS,
                    "selector_family": family,
                    "model": "additive_head",
                    "measurement_budget": 160,
                    "target": 1.0,
                    "mask_id": mask,
                    "predicted_target_loss": score,
                }
            )
    decisions = select_fixed_masks(
        pd.DataFrame(rows), candidates, head_cost_penalty=0.02
    )
    standard = decisions[decisions["policy"] == POLICY_TARGET]
    assert len(standard) == 4
    direct = standard[standard["selector_family"] == SELECTOR_DIRECT_RISK]
    assert direct.set_index("pool_id")["selected_mask_id"].to_dict() == {
        "p0": "a",
        "p1": "c",
    }

    effects = pd.DataFrame(
        [
            {"prompt_id": prompt, "mask_id": mask, "drop_from_clean": effect}
            for prompt, offset in (("x", 0.0), ("y", 0.1))
            for mask, effect in (("a", 0.9 + offset), ("b", 0.2), ("c", 1.1), ("d", 1.8))
        ]
    )
    outcomes = evaluate_fixed_masks(
        decisions,
        effects,
        target_tolerance=0.25,
        head_cost_penalty=0.02,
    )
    fixed = outcomes[
        (outcomes["selector_family"] == SELECTOR_DIRECT_RISK)
        & (outcomes["policy"] == POLICY_TARGET)
        & (outcomes["pool_id"] == "p0")
    ]
    assert fixed["selected_mask_id"].nunique() == 1
    assert set(fixed["prompt_id"]) == {"x", "y"}


def test_comparisons_include_same_basis_and_all_pairs_against_simpler_means() -> None:
    models = (
        "additive_head",
        "count_additive",
        "count_plus_PE_bin4",
        "count_plus_all_bin4",
    )
    pairs = _comparison_pairs(models)
    assert all((model, model) in pairs for model in models)
    assert ("count_plus_all_bin4", "additive_head") in pairs
    assert ("count_plus_all_bin4", "count_additive") in pairs
    assert len(set(pairs)) == len(pairs)


def test_full_head_quadratic_basis_has_every_distinct_pair_once() -> None:
    masks = np.zeros((3, 13), dtype=float)
    masks[1, 0] = 1.0
    masks[2, [0, 12]] = 1.0
    design, columns = _head_quadratic_design(masks)
    assert design.shape == (3, 92)
    assert len(set(columns)) == 92
    assert "head_0:head_12" in columns
    pair_column = columns.index("head_0:head_12")
    assert design[:, pair_column].tolist() == [0.0, 0.0, 1.0]


def test_fixed_action_oracle_yields_nonnegative_decision_regret() -> None:
    candidates = pd.DataFrame(
        {
            "mask_id": ["a", "b"],
            "pool_id": ["p0", "p0"],
            "n_heads": [1, 2],
        }
    )
    effects = pd.DataFrame(
        [
            {"prompt_id": "x", "mask_id": "a", "drop_from_clean": 0.9},
            {"prompt_id": "x", "mask_id": "b", "drop_from_clean": 0.2},
            {"prompt_id": "y", "mask_id": "a", "drop_from_clean": 1.1},
            {"prompt_id": "y", "mask_id": "b", "drop_from_clean": 0.3},
        ]
    )
    decision = pd.DataFrame(
        [
            {
                "analysis_status": EXPLORATORY_STATUS,
                "selector_family": SELECTOR_DIRECT_RISK,
                "model": "additive_head",
                "measurement_budget": 160,
                "pool_id": "p0",
                "target": 1.0,
                "policy": POLICY_TARGET,
                "selected_mask_id": "b",
                "selected_head_count": 2,
                "predicted_target_loss": 0.1,
                "predicted_objective": 0.1,
            }
        ]
    )
    outcomes = evaluate_fixed_masks(
        decision,
        effects,
        target_tolerance=0.25,
        head_cost_penalty=0.02,
    )
    oracle = fixed_action_oracle(
        candidates,
        effects,
        targets=(1.0,),
        head_cost_penalty=0.02,
    )
    quality = decision_quality(outcomes, oracle)
    assert quality["best_fixed_action_regret"].iloc[0] == pytest.approx(0.65)
    assert quality["best_fixed_action_regret"].iloc[0] >= 0.0
