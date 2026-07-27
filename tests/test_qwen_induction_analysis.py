"""Tests for model-free Qwen induction response and action analysis.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from observerbench.tasks.qwen_induction.analysis import (
    ADDITIVE,
    DIRECT_RISK,
    EXACT_NOOP,
    NATURAL_MEAN,
    NO_EFFECT,
    QUADRATIC,
    TRANSFORMED_MEAN,
    bootstrap_aggregate_action_contrasts,
    bootstrap_action_contrasts,
    bootstrap_prediction_contrasts,
    effect_dispersion_decomposition,
    evaluate_fixed_actions,
    evaluate_mean_effect_predictions,
    freeze_actions,
    freeze_mean_effect_predictions,
    intervention_outcome_diagnostics,
    mask_design_matrix,
    prediction_error_diagnostics,
)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = np.arange(256)
    bits = [f"{value:08b}" for value in values]
    anchors = [0, *(1 << shift for shift in range(8))]
    pairs = [(1 << left) | (1 << right) for left, right in itertools.combinations(range(8), 2)]
    prefix = anchors + [value for value in pairs if value not in anchors]
    remaining = [value for value in values if value not in prefix]
    calibration_values = (prefix + remaining)[:128]
    locked_values = [value for value in values if value not in calibration_values]
    calibration = pd.DataFrame(
        {
            "measurement_order": np.arange(1, 129),
            "mask_id": [f"mask_{value:03d}" for value in calibration_values],
            "mask_bits": [bits[value] for value in calibration_values],
            "bank": "calibration",
        }
    )
    locked = pd.DataFrame(
        {
            "mask_id": [f"mask_{value:03d}" for value in locked_values],
            "mask_bits": [bits[value] for value in locked_values],
            "bank": "locked_test",
            "pool_id": [f"pool_{index // 8:02d}" for index in range(128)],
        }
    )

    def effect(mask_bits: str, prompt: int) -> float:
        x = np.asarray([int(bit) for bit in mask_bits], dtype=float)
        return float(
            0.08 * x.sum()
            + 0.16 * x[0] * x[1]
            + (0.03 if prompt % 2 == 0 and x[2] else -0.03 if x[2] else 0.0)
        )

    def cells(mask_table: pd.DataFrame, split: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "prompt_id": f"{split}_{prompt:02d}",
                    "family_id": f"family_{prompt % 4}",
                    "cluster_id": f"cluster_{prompt // 4}",
                    "split": split,
                    "mask_id": row.mask_id,
                    "mask_bits": row.mask_bits,
                    "effect": effect(row.mask_bits, prompt),
                    "clean_candidate_correct": True,
                    "ablated_candidate_correct": effect(row.mask_bits, prompt) < 0.6,
                    "clean_top1_correct": True,
                    "ablated_top1_correct": effect(row.mask_bits, prompt) < 0.5,
                    "clean_target_nll": 0.1,
                    "ablated_target_nll": 0.1 + effect(row.mask_bits, prompt),
                }
                for row in mask_table.itertuples(index=False)
                for prompt in range(8)
            ]
        )

    return calibration, locked, cells(calibration, "calibration"), cells(locked, "locked_test")


def test_fixed_design_has_nine_and_thirty_seven_columns() -> None:
    masks = np.asarray([[int(bit) for bit in f"{value:08b}"] for value in range(256)])
    additive, additive_terms = mask_design_matrix(masks, ADDITIVE)
    quadratic, quadratic_terms = mask_design_matrix(masks, QUADRATIC)

    assert additive.shape == (256, 9)
    assert quadratic.shape == (256, 37)
    assert len(additive_terms) == 9
    assert len(quadratic_terms) == 37
    assert np.linalg.matrix_rank(quadratic) == 37


def test_prediction_freeze_and_locked_evaluation_recover_planted_pair() -> None:
    calibration, locked, calibration_effects, locked_effects = _tables()
    frozen = freeze_mean_effect_predictions(
        calibration_effects,
        calibration,
        locked,
        ridge_grid=(1e-6,),
    )
    metrics = evaluate_mean_effect_predictions(
        frozen["predictions"], locked_effects, locked
    )
    full = metrics.loc[metrics["measurement_budget"] == 128].set_index("model")

    assert len(frozen["predictions"]) == 4 * 3 * 128
    assert full.loc[QUADRATIC, "mae"] < 1e-6
    assert full.loc[QUADRATIC, "mae"] < full.loc[ADDITIVE, "mae"]


def test_action_freeze_includes_noop_and_scores_every_policy() -> None:
    calibration, locked, calibration_effects, locked_effects = _tables()
    targets = (0.10, 0.25, 0.40)
    frozen = freeze_actions(
        calibration_effects,
        calibration,
        locked,
        targets,
        ridge_grid=(1e-6,),
    )
    actions = frozen["fixed_actions"]
    evaluation, oracles = evaluate_fixed_actions(
        actions, locked_effects, locked
    )

    assert set(actions["selector"]) == {
        NATURAL_MEAN,
        TRANSFORMED_MEAN,
        DIRECT_RISK,
        EXACT_NOOP,
    }
    assert len(actions) == 3 * 4 * 16
    assert len(evaluation) == len(actions)
    assert len(oracles) == 3 * 16
    assert (evaluation["regret"] >= -1e-12).all()
    assert actions.loc[actions["selector"] == EXACT_NOOP, "selected_is_noop"].all()


def test_action_freeze_rejects_pool_without_exact_eight_masks() -> None:
    calibration, locked, calibration_effects, _ = _tables()
    with pytest.raises(ValueError, match="complete exhaustive design"):
        freeze_actions(
            calibration_effects,
            calibration,
            locked.iloc[:-1],
            (0.2,),
            ridge_grid=(1e-6,),
        )


def test_two_way_bootstrap_preserves_positive_quadratic_contrast() -> None:
    calibration, locked, calibration_effects, locked_effects = _tables()
    frozen = freeze_mean_effect_predictions(
        calibration_effects,
        calibration,
        locked,
        ridge_grid=(1e-6,),
    )
    contrasts = bootstrap_prediction_contrasts(
        frozen["predictions"],
        locked_effects,
        locked,
        repeats=200,
        seed=3,
    )
    full = contrasts.loc[contrasts["measurement_budget"] == 128].iloc[0]
    assert full["estimate"] > 0.0
    assert full["ci_lower"] > 0.0
    assert bool(full["quadratic_better"])


def test_action_bootstrap_and_dispersion_decomposition_are_paired() -> None:
    calibration, locked, calibration_effects, locked_effects = _tables()
    targets = (0.10, 0.25, 0.40)
    frozen = freeze_actions(
        calibration_effects,
        calibration,
        locked,
        targets,
        ridge_grid=(1e-6,),
    )
    contrasts = bootstrap_action_contrasts(
        frozen["fixed_actions"],
        locked_effects,
        locked,
        repeats=100,
        seed=5,
    )
    decomposition = effect_dispersion_decomposition(
        locked_effects,
        locked,
        targets,
    )

    assert len(contrasts) == len(targets) * 3
    assert set(contrasts["reference"]) == {DIRECT_RISK}
    action_metrics, _ = evaluate_fixed_actions(
        frozen["fixed_actions"], locked_effects, locked
    )
    mean_loss = action_metrics.groupby(["target", "selector"])[
        "actual_target_loss"
    ].mean()
    for row in contrasts.itertuples(index=False):
        expected = (
            mean_loss.loc[(row.target, row.comparator)]
            - mean_loss.loc[(row.target, DIRECT_RISK)]
        )
        assert row.contrast_comparator_minus_reference == pytest.approx(expected)
    assert len(decomposition) == len(targets) * len(locked)
    assert (decomposition["dispersion_penalty"] >= 0.0).all()
    reconstructed = (
        decomposition["absolute_mean_error"]
        + decomposition["dispersion_penalty"]
    )
    assert np.allclose(reconstructed, decomposition["realized_target_loss"])


def test_aggregate_action_bootstrap_equal_weights_targets_with_common_draws() -> None:
    calibration, locked, calibration_effects, locked_effects = _tables()
    targets = (0.10, 0.25, 0.40)
    frozen = freeze_actions(
        calibration_effects,
        calibration,
        locked,
        targets,
        ridge_grid=(1e-6,),
    )
    target_specific = bootstrap_action_contrasts(
        frozen["fixed_actions"],
        locked_effects,
        locked,
        repeats=80,
        seed=11,
    )
    aggregate = bootstrap_aggregate_action_contrasts(
        frozen["fixed_actions"],
        locked_effects,
        locked,
        repeats=80,
        seed=11,
    )

    assert len(aggregate) == 3
    assert set(aggregate["aggregation"]) == {"equal_weight_across_targets"}
    assert set(aggregate["reference"]) == {DIRECT_RISK}
    assert set(aggregate["n_targets"]) == {3}
    for row in aggregate.itertuples(index=False):
        expected = target_specific.loc[
            target_specific["comparator"] == row.comparator,
            "contrast_comparator_minus_reference",
        ].mean()
        assert row.contrast_comparator_minus_reference == pytest.approx(expected)

    pool_ids = tuple(sorted(locked["pool_id"].astype(str).unique()))
    cluster_ids = tuple(sorted(locked_effects["cluster_id"].astype(str).unique()))
    differences: dict[str, list[np.ndarray]] = {
        NATURAL_MEAN: [],
        TRANSFORMED_MEAN: [],
        EXACT_NOOP: [],
    }
    for target in targets:
        actions = frozen["fixed_actions"].loc[
            frozen["fixed_actions"]["target"] == target
        ]
        losses: dict[str, np.ndarray] = {}
        for selector in (NATURAL_MEAN, TRANSFORMED_MEAN, DIRECT_RISK, EXACT_NOOP):
            values = np.empty((len(pool_ids), len(cluster_ids)), dtype=float)
            selected = actions.loc[actions["selector"] == selector].set_index(
                "pool_id"
            )
            for pool_index, pool_id in enumerate(pool_ids):
                action = selected.loc[pool_id]
                if bool(action["selected_is_noop"]):
                    values[pool_index] = abs(target)
                else:
                    cells = locked_effects.loc[
                        locked_effects["mask_id"] == action["selected_mask_id"]
                    ].copy()
                    cells["target_loss"] = np.abs(cells["effect"] - target)
                    values[pool_index] = np.asarray(
                        [
                            cells.loc[
                                cells["cluster_id"] == cluster_id, "target_loss"
                            ].mean()
                            for cluster_id in cluster_ids
                        ]
                    )
            losses[selector] = values
        for comparator in differences:
            differences[comparator].append(
                losses[comparator] - losses[DIRECT_RISK]
            )

    repeats = 80
    rng = np.random.default_rng(11)
    sampled_pools = rng.integers(
        0, len(pool_ids), size=(repeats, len(pool_ids))
    )
    sampled_clusters = rng.integers(
        0, len(cluster_ids), size=(repeats, len(cluster_ids))
    )
    for row in aggregate.itertuples(index=False):
        difference = np.stack(differences[row.comparator], axis=0)
        draws = np.asarray(
            [
                np.take(
                    np.take(difference, sampled_pools[repeat], axis=1),
                    sampled_clusters[repeat],
                    axis=2,
                ).mean()
                for repeat in range(repeats)
            ]
        )
        lower, upper = np.quantile(draws, (0.025, 0.975))
        assert row.ci_lower == pytest.approx(lower)
        assert row.ci_upper == pytest.approx(upper)


def test_prediction_and_intervention_diagnostics_keep_density_and_family_slices() -> None:
    calibration, locked, calibration_effects, locked_effects = _tables()
    frozen = freeze_mean_effect_predictions(
        calibration_effects,
        calibration,
        locked,
        ridge_grid=(1e-6,),
    )

    prediction = prediction_error_diagnostics(
        frozen["predictions"],
        locked_effects,
        locked,
    )
    outcomes = intervention_outcome_diagnostics(locked_effects, locked)

    assert len(prediction["mask_errors"]) == 4 * 3 * 128
    assert set(prediction["density_summary"]["n_heads"]) == set(
        locked["mask_bits"].str.count("1")
    )
    assert set(prediction["prompt_error_summary"]["family_id"]) == {
        "all",
        "family_0",
        "family_1",
        "family_2",
        "family_3",
    }
    assert len(outcomes["by_mask"]) == len(locked)
    assert set(outcomes["by_density"]["n_heads"]) == set(
        locked["mask_bits"].str.count("1")
    )
    assert set(outcomes["by_family"]["family_id"]) == {
        "family_0",
        "family_1",
        "family_2",
        "family_3",
    }
    assert (outcomes["by_density"]["mean_target_nll_increase"] >= 0.0).all()

    selected = prediction["mask_errors"].iloc[0]
    observed = locked_effects.loc[
        locked_effects["mask_id"] == selected["mask_id"], "effect"
    ].mean()
    assert selected["observed_mean_effect"] == pytest.approx(observed)
    assert selected["error"] == pytest.approx(
        selected["predicted_mean_effect"] - observed
    )
    family = outcomes["by_family"].set_index("family_id").loc["family_0"]
    raw_family = locked_effects.loc[locked_effects["family_id"] == "family_0"]
    assert family["mean_target_nll_increase"] == pytest.approx(
        (raw_family["ablated_target_nll"] - raw_family["clean_target_nll"]).mean()
    )


@pytest.mark.parametrize(
    "mutation, message",
    (
        ("duplicate", "duplicate"),
        ("missing", "cover every locked mask"),
        ("mask_bits", "mask bits"),
        ("unknown_model", "unexpected model"),
        ("nonfinite", "invalid numeric"),
    ),
)
def test_prediction_diagnostics_reject_invalid_frozen_rows(
    mutation: str,
    message: str,
) -> None:
    calibration, locked, calibration_effects, locked_effects = _tables()
    predictions = freeze_mean_effect_predictions(
        calibration_effects,
        calibration,
        locked,
        ridge_grid=(1e-6,),
    )["predictions"].copy()
    if mutation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif mutation == "missing":
        predictions = predictions.drop(index=0)
    elif mutation == "mask_bits":
        predictions.loc[0, "mask_bits"] = "00000000"
    elif mutation == "unknown_model":
        predictions.loc[predictions["model"] == NO_EFFECT, "model"] = "mystery"
    elif mutation == "nonfinite":
        predictions.loc[0, "predicted_mean_effect"] = np.inf
    with pytest.raises(ValueError, match=message):
        prediction_error_diagnostics(predictions, locked_effects, locked)


@pytest.mark.parametrize(
    "column, value, message",
    (
        ("ablated_candidate_correct", 2, "booleans or 0/1"),
        ("ablated_top1_correct", np.nan, "missing values"),
        ("ablated_target_nll", np.inf, "finite and nonnegative"),
        ("clean_target_nll", -0.1, "finite and nonnegative"),
    ),
)
def test_intervention_diagnostics_reject_invalid_retained_outcomes(
    column: str,
    value: object,
    message: str,
) -> None:
    _calibration, locked, _calibration_effects, locked_effects = _tables()
    invalid = locked_effects.copy()
    if "correct" in column:
        invalid[column] = invalid[column].astype(object)
    invalid.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        intervention_outcome_diagnostics(invalid, locked)


def test_intervention_diagnostics_reject_changed_clean_baseline() -> None:
    _calibration, locked, _calibration_effects, locked_effects = _tables()
    invalid = locked_effects.copy()
    prompt_id = invalid.iloc[0]["prompt_id"]
    row = invalid.index[invalid["prompt_id"] == prompt_id][1]
    invalid.loc[row, "clean_target_nll"] = 0.2
    with pytest.raises(ValueError, match="clean intervention baselines changed"):
        intervention_outcome_diagnostics(invalid, locked)
