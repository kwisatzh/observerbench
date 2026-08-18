"""Synthetic tests for the sealed Phase-6 prospective evaluator.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from observerbench.tasks.ioi.phase6_evaluation import (
    evaluate_phase6_tables,
    validate_complete_candidate_surface,
    validate_clustered_test_design,
    validate_frozen_evaluation_inputs,
)
from observerbench.tasks.ioi.phase6_freeze import (
    COST_POLICY,
    DIRECT_RISK,
    JENSEN_SCORE,
    NATURAL_MEAN,
    QUADRATIC_MODEL,
    TARGET_POLICY,
)


TARGETS = (0.5, 1.0, 1.5)


def _protocol() -> dict[str, object]:
    return {
        "targets": list(TARGETS),
        "primary_targets": [0.5, 1.0],
        "stress_test_targets": [1.5],
        "measurement_budget": 160,
        "direct_risk_models": [
            "additive_head",
            "count_additive",
            QUADRATIC_MODEL,
        ],
        "mean_effect_models": [QUADRATIC_MODEL],
        "jensen_score_sensitivity_models": [QUADRATIC_MODEL],
        "target_tolerance": 0.25,
        "head_cost_penalty": 0.02,
        "bootstrap_repeats": 200,
        "bootstrap_seed": 2606201,
        "bootstrap_interval": {"quantiles": [0.025, 0.975]},
        "test_unordered_pair_cluster_count": 2,
        "test_prompts_per_pair_cluster": 4,
        "candidate_pool_count": 2,
        "clean_template_validity": {
            "claim_gate": {
                "overall_IO_vs_subject_pairwise_accuracy_min": 0.90,
                "every_template_IO_vs_subject_pairwise_accuracy_min": 0.75,
                "every_template_mean_clean_logit_difference_strictly_positive": True,
            }
        },
        "hypotheses": {
            "H1_primary_estimand": {
                "minimum_loss_reduction_fraction": 0.05,
            },
            "H2_secondary_structure": {
                "minimum_loss_reduction_fraction_against_each": 0.10,
            },
        },
    }


def _prompts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("reference", "train"):
        for template in ("template_a", "template_b"):
            rows.append(
                {
                    "prompt_id": f"{split}_{template}",
                    "split": split,
                    "template_id": template,
                    "structure": "ABBA",
                    "unordered_name_pair_id": f"{split}_pair",
                    "pair_orientation": "not_applicable",
                    "io_name": "Alice",
                    "s_name": "Bob",
                }
            )
    for pair in range(2):
        for orientation in ("a_to_b", "b_to_a"):
            for template in ("template_a", "template_b"):
                rows.append(
                    {
                        "prompt_id": f"test_{pair}_{orientation}_{template}",
                        "split": "test",
                        "template_id": template,
                        "structure": "ABBA" if template == "template_a" else "BABA",
                        "unordered_name_pair_id": f"pair_{pair}",
                        "pair_orientation": orientation,
                        "io_name": "Alice" if orientation == "a_to_b" else "Bob",
                        "s_name": "Bob" if orientation == "a_to_b" else "Alice",
                    }
                )
    return pd.DataFrame(rows)


def _candidates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mask_number = 1
    for pool in range(2):
        for quality in ("good", "bad"):
            for target_index, _target in enumerate(TARGETS):
                bits = f"{mask_number:013b}"
                rows.append(
                    {
                        "mask_id": f"pool_{pool}_{quality}_{target_index}",
                        "mask_bits": bits,
                        "bank": "candidate",
                        "pool_id": f"pool_{pool}",
                        "n_heads": bits.count("1"),
                    }
                )
                mask_number += 1
    return pd.DataFrame(rows)


def _effects(prompts: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    test = prompts.loc[prompts["split"] == "test"]
    rows: list[dict[str, object]] = []
    for prompt in test.itertuples(index=False):
        for mask in candidates.itertuples(index=False):
            target_index = int(str(mask.mask_id).rsplit("_", 1)[1])
            effect = TARGETS[target_index]
            if "_bad_" in str(mask.mask_id):
                effect += 0.5
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "split": "test",
                    "template_id": prompt.template_id,
                    "structure": prompt.structure,
                    "mask_id": mask.mask_id,
                    "mask_bits": mask.mask_bits,
                    "bank": "candidate",
                    "pool_id": mask.pool_id,
                    "clean_ld": 2.0,
                    "ablated_ld": 2.0 - effect,
                    "drop_from_clean": effect,
                }
            )
    return pd.DataFrame(rows)


def _observers() -> tuple[tuple[str, str], ...]:
    return (
        (DIRECT_RISK, "additive_head"),
        (DIRECT_RISK, "count_additive"),
        (DIRECT_RISK, QUADRATIC_MODEL),
        (NATURAL_MEAN, QUADRATIC_MODEL),
        (JENSEN_SCORE, QUADRATIC_MODEL),
    )


def _predictions_actions(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    metadata = candidates.set_index("mask_id")
    for selector, model in _observers():
        quality = "good" if (selector, model) == (DIRECT_RISK, QUADRATIC_MODEL) else "bad"
        for target_index, target in enumerate(TARGETS):
            selected_by_pool = {
                f"pool_{pool}": f"pool_{pool}_{quality}_{target_index}"
                for pool in range(2)
            }
            for mask in candidates.itertuples(index=False):
                selected = selected_by_pool[str(mask.pool_id)] == str(mask.mask_id)
                mask_target_index = int(str(mask.mask_id).rsplit("_", 1)[1])
                mean_effect = TARGETS[mask_target_index] + (
                    0.5 if "_bad_" in str(mask.mask_id) else 0.0
                )
                prediction_rows.append(
                    {
                        "selector_family": selector,
                        "model": model,
                        "target": target,
                        "measurement_budget": 160,
                        "mask_id": mask.mask_id,
                        "predicted_target_loss": 0.0 if selected else 1.0,
                        "predicted_mean_effect": (
                            mean_effect if selector == NATURAL_MEAN else np.nan
                        ),
                    }
                )
            for pool_id, mask_id in selected_by_pool.items():
                count = int(metadata.loc[mask_id, "n_heads"])
                for policy in (TARGET_POLICY, COST_POLICY):
                    objective = 0.0 if policy == TARGET_POLICY else 0.02 * count
                    action_rows.append(
                        {
                            "selector_family": selector,
                            "model": model,
                            "target": target,
                            "pool_id": pool_id,
                            "measurement_budget": 160,
                            "policy": policy,
                            "selected_mask_id": mask_id,
                            "selected_head_count": count,
                            "predicted_target_loss": 0.0,
                            "predicted_objective": objective,
                        }
                    )
    return pd.DataFrame(prediction_rows), pd.DataFrame(action_rows)


def _fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    prompts = _prompts()
    candidates = _candidates()
    effects = _effects(prompts, candidates)
    predictions, actions = _predictions_actions(candidates)
    clean = prompts.loc[prompts["split"] == "test", ["prompt_id"]].copy()
    clean["clean_ld"] = 2.0
    return prompts, candidates, actions, predictions, effects, clean


def test_positive_fixture_passes_h1_h2_jensen_and_clean_gates() -> None:
    prompts, candidates, actions, predictions, effects, clean = _fixture()
    result = evaluate_phase6_tables(
        prompts,
        candidates,
        actions,
        predictions,
        effects,
        clean,
        protocol=_protocol(),
    )
    audit = result["hypothesis_audit"]
    assert audit["result_calibration"] == "positive"
    assert audit["H1_primary_estimand"]["passed"] is True
    assert audit["H2_secondary_structure"]["passed"] is True
    assert audit["Jensen_parameter_count_sensitivity"]["passed"] is True
    assert audit["clean_task_validity"]["passed"] is True
    assert len(result["fixed_action_prompt_losses"]) == len(actions) * 8
    assert len(result["leave_one_pair"]) == 3 * 2
    assert len(result["leave_one_template"]) == 3 * 2
    primary = result["prespecified_contrasts"].query(
        "comparison_id == 'H1_primary_estimand' and "
        "metric == 'absolute_target_loss_reduction' and "
        "target_scope == 'primary_pooled'"
    ).iloc[0]
    assert primary["relative_reduction_fraction"] == pytest.approx(1.0)
    assert primary["q025"] > 0.0
    assert primary["bootstrap_repeats"] == 200


def test_clean_failure_qualifies_language_but_never_changes_hypotheses() -> None:
    prompts, candidates, actions, predictions, effects, clean = _fixture()
    clean.loc[
        clean["prompt_id"].str.contains("template_b", regex=False), "clean_ld"
    ] = -1.0
    result = evaluate_phase6_tables(
        prompts,
        candidates,
        actions,
        predictions,
        effects,
        clean,
        protocol=_protocol(),
    )
    audit = result["hypothesis_audit"]
    assert audit["H1_primary_estimand"]["passed"] is True
    assert audit["H2_secondary_structure"]["passed"] is True
    assert audit["clean_task_validity"]["passed"] is False
    assert audit["ioi_language_allowed"] is False
    assert audit["clean_failure_never_changes_H1_or_H2"] is True


def test_primary_per_target_reversal_fails_frozen_direction_gate() -> None:
    prompts, candidates, actions, predictions, effects, clean = _fixture()
    target_one_good = effects["mask_id"].str.contains("_good_1", regex=False)
    effects.loc[target_one_good, "drop_from_clean"] = 2.0
    effects.loc[target_one_good, "ablated_ld"] = 0.0
    result = evaluate_phase6_tables(
        prompts,
        candidates,
        actions,
        predictions,
        effects,
        clean,
        protocol=_protocol(),
    )
    h1 = result["hypothesis_audit"]["H1_primary_estimand"]
    assert h1["passed"] is False
    assert h1["checks"]["target_1_nonnegative"] is False


def test_candidate_surface_rejects_missing_cells_and_action_drift() -> None:
    prompts, candidates, actions, predictions, effects, _clean = _fixture()
    with pytest.raises(ValueError, match="expected 96 held-out effect cells"):
        validate_complete_candidate_surface(effects.iloc[:-1], prompts, candidates)

    changed = actions.copy()
    changed.loc[0, "predicted_target_loss"] = 0.2
    with pytest.raises(ValueError, match="changed its sealed prediction"):
        validate_frozen_evaluation_inputs(
            changed,
            predictions,
            candidates,
            protocol=_protocol(),
        )


def test_frozen_input_audit_rejects_foreign_predictions_and_nonminimum_actions() -> None:
    _prompts_frame, candidates, actions, predictions, _effects_frame, _clean = _fixture()
    foreign = predictions.copy()
    foreign.loc[0, "mask_id"] = "foreign_mask"
    with pytest.raises(ValueError, match="differs from the candidate bank"):
        validate_frozen_evaluation_inputs(
            actions,
            foreign,
            candidates,
            protocol=_protocol(),
        )

    changed = actions.copy()
    row = changed.index[
        (changed["selector_family"] == DIRECT_RISK)
        & (changed["model"] == QUADRATIC_MODEL)
        & (changed["target"] == 0.5)
        & (changed["pool_id"] == "pool_0")
        & (changed["policy"] == TARGET_POLICY)
    ][0]
    replacement = "pool_0_bad_0"
    metadata = candidates.set_index("mask_id").loc[replacement]
    replacement_prediction = predictions.loc[
        (predictions["selector_family"] == DIRECT_RISK)
        & (predictions["model"] == QUADRATIC_MODEL)
        & (predictions["target"] == 0.5)
        & (predictions["mask_id"] == replacement),
        "predicted_target_loss",
    ].iloc[0]
    changed.loc[row, "selected_mask_id"] = replacement
    changed.loc[row, "selected_head_count"] = int(metadata["n_heads"])
    changed.loc[row, "predicted_target_loss"] = replacement_prediction
    changed.loc[row, "predicted_objective"] = replacement_prediction
    with pytest.raises(ValueError, match="deterministic selection rule"):
        validate_frozen_evaluation_inputs(
            changed,
            predictions,
            candidates,
            protocol=_protocol(),
        )


def test_cluster_axis_audit_rejects_lost_orientation_and_natural_mean_gap() -> None:
    prompts, candidates, actions, predictions, _effects_frame, _clean = _fixture()
    unbalanced = prompts.copy()
    test_rows = unbalanced.index[unbalanced["split"] == "test"]
    unbalanced.loc[test_rows[0], "pair_orientation"] = "b_to_a"
    with pytest.raises(ValueError, match="repeats a template-orientation"):
        validate_clustered_test_design(
            unbalanced,
            candidates,
            protocol=_protocol(),
        )

    missing_mean = predictions.copy()
    natural_row = missing_mean.index[
        missing_mean["selector_family"] == NATURAL_MEAN
    ][0]
    missing_mean.loc[natural_row, "predicted_mean_effect"] = np.nan
    with pytest.raises(ValueError, match="lacks its frozen mean prediction"):
        validate_frozen_evaluation_inputs(
            actions,
            missing_mean,
            candidates,
            protocol=_protocol(),
        )


def test_all_candidate_metrics_and_oracle_use_the_complete_surface() -> None:
    prompts, candidates, actions, predictions, effects, clean = _fixture()
    result = evaluate_phase6_tables(
        prompts,
        candidates,
        actions,
        predictions,
        effects,
        clean,
        protocol=_protocol(),
    )
    metrics = result["prediction_metrics"]
    assert set(metrics["candidate_count"]) == {len(candidates)}
    assert len(result["candidate_actual_risk"]) == len(candidates) * len(TARGETS)
    mean_metrics = result["natural_mean_estimand_metrics"]
    assert len(mean_metrics) == 1
    assert mean_metrics.iloc[0]["heldout_mean_effect_mae"] == pytest.approx(0.0)
    assert bool(mean_metrics.iloc[0]["descriptive_non_gating"]) is True
    assert len(result["best_fixed_oracle"]) == 2 * len(TARGETS) * 2
    assert np.allclose(
        result["best_fixed_oracle"]["oracle_mean_target_loss"], 0.0
    )
    assert (result["decision_quality"]["best_fixed_action_regret"] >= -1e-12).all()
