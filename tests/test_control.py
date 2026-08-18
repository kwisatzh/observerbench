"""Tests for reusable ObserverBench control primitives.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import numpy as np
import pytest

from observerbench.control import (
    AffineStateEstimator,
    affine_subspace_adequacy,
    direction_support_metrics,
    fit_affine_support,
    loop_gain_diagnostics,
    positive_gain_matched_controller_gain,
    predict_unsaturated_affine_scalar_errors,
    project_direction_to_target_gain,
)


def test_affine_estimator_batch_and_calibration() -> None:
    estimator = AffineStateEstimator("base", 1.0, np.array([2.0, -1.0]))
    states = np.array([[0.0, 0.0], [1.0, 3.0]])

    assert estimator.estimate(states[1]) == 0.0
    assert np.allclose(estimator.estimate_batch(states), [1.0, 0.0])
    calibrated = estimator.affine_calibration(offset=-1.0, scale=0.5)
    assert np.allclose(calibrated.estimate_batch(states), [-0.5, -1.0])


def test_global_mismatch_can_be_adequate_on_the_intervention_subspace() -> None:
    operating_state = np.array([1.0, 2.0, 3.0])
    target = AffineStateEstimator("target", 0.4, np.array([2.0, -1.0, 5.0]))
    estimator_gradient = np.array([2.0, -1.0, -7.0])
    estimator = AffineStateEstimator(
        "estimator",
        target.estimate(operating_state) - float(estimator_gradient @ operating_state),
        estimator_gradient,
    )
    # Scaled and dependent columns should describe the same two-dimensional set.
    intervention_basis = np.array([[3.0, 0.0, 6.0], [0.0, -4.0, 0.0], [0.0, 0.0, 0.0]])

    diagnostics = affine_subspace_adequacy(
        target,
        estimator,
        operating_state,
        intervention_basis,
    )

    assert diagnostics.reachable_rank == 2
    assert diagnostics.adequate is True
    assert diagnostics.level_agreement is True
    assert diagnostics.response_agreement is True
    assert diagnostics.projected_gradient_mismatch_norm < 1e-12
    assert diagnostics.orthogonal_gradient_mismatch_norm == pytest.approx(12.0)
    assert diagnostics.global_gradient_mismatch_norm == pytest.approx(12.0)
    for coefficients in (np.array([0.2, -0.3, 0.0]), np.array([-1.0, 0.5, 2.0])):
        reachable_state = operating_state + intervention_basis @ coefficients
        assert target.estimate(reachable_state) == pytest.approx(estimator.estimate(reachable_state))
    assert target.estimate(operating_state + np.array([0.0, 0.0, 1.0])) != pytest.approx(
        estimator.estimate(operating_state + np.array([0.0, 0.0, 1.0]))
    )


def test_affine_subspace_diagnostic_separates_level_and_response_failures() -> None:
    state = np.array([2.0, -1.0])
    target = AffineStateEstimator("target", 0.0, np.array([1.0, 2.0]))
    basis = np.array([1.0, 0.0])

    level_failure = affine_subspace_adequacy(
        target,
        AffineStateEstimator("biased", 0.1, target.gradient),
        state,
        basis,
    )
    assert level_failure.level_agreement is False
    assert level_failure.response_agreement is True
    assert level_failure.adequate is False

    response_gradient = np.array([3.0, 2.0])
    response_failure = affine_subspace_adequacy(
        target,
        AffineStateEstimator(
            "wrong_response",
            target.estimate(state) - float(response_gradient @ state),
            response_gradient,
        ),
        state,
        basis,
    )
    assert response_failure.level_agreement is True
    assert response_failure.response_agreement is False
    assert response_failure.adequate is False


@pytest.mark.parametrize("horizon", [0, 1, 2, 7, 20])
def test_scalar_affine_error_prediction_matches_exact_state_rollout(horizon: int) -> None:
    target = AffineStateEstimator("target", 0.3, np.array([2.0, -1.0]))
    estimator = AffineStateEstimator("estimator", -0.2, np.array([0.5, 1.5]))
    direction = np.array([0.4, -0.2])
    controller_gain = 0.35
    reference = 1.7
    initial_state = np.array([-0.6, 0.9])
    initial_observer_error = reference - estimator.estimate(initial_state)
    initial_target_error = reference - target.estimate(initial_state)
    prediction = predict_unsaturated_affine_scalar_errors(
        initial_observer_error=initial_observer_error,
        initial_target_error=initial_target_error,
        observer_response_gain=float(estimator.gradient @ direction),
        target_response_gain=float(target.gradient @ direction),
        controller_gain=controller_gain,
        horizon=horizon,
    )

    state = initial_state.copy()
    cumulative_action = 0.0
    for _step in range(horizon):
        action = controller_gain * (reference - estimator.estimate(state))
        cumulative_action += action
        state = state + action * direction

    assert prediction.cumulative_action == pytest.approx(cumulative_action)
    assert prediction.final_observer_error == pytest.approx(reference - estimator.estimate(state))
    assert prediction.final_target_error == pytest.approx(reference - target.estimate(state))


def test_scalar_affine_prediction_handles_zero_observer_response() -> None:
    prediction = predict_unsaturated_affine_scalar_errors(
        initial_observer_error=2.0,
        initial_target_error=-1.0,
        observer_response_gain=0.0,
        target_response_gain=3.0,
        controller_gain=0.25,
        horizon=6,
    )

    assert prediction.observer_error_pole == 1.0
    assert prediction.final_observer_error == 2.0
    assert prediction.cumulative_action == 3.0
    assert prediction.final_target_error == -10.0


@pytest.mark.parametrize(
    ("self_gain", "expected_pole", "stable", "sign_compatible"),
    [
        (-1.0, 1.5, False, False),
        (0.0, 1.0, False, False),
        (1.0, 0.5, True, True),
        (4.0, -1.0, False, True),
    ],
)
def test_loop_gain_stability_boundaries(self_gain, expected_pole, stable, sign_compatible) -> None:
    diagnostics = loop_gain_diagnostics(
        target_gradient=np.array([1.0, 0.0]),
        estimator_gradient=np.array([self_gain, 0.0]),
        direction=np.array([1.0, 0.0]),
        controller_gain=0.5,
    )

    assert diagnostics.observer_error_pole == expected_pole
    assert diagnostics.locally_convergent_unsaturated is stable
    assert diagnostics.sign_compatible is sign_compatible


def test_positive_gain_matching_matches_reference_pole_and_rejects_wrong_sign() -> None:
    matched = positive_gain_matched_controller_gain(
        base_controller_gain=0.35,
        reference_self_gain=0.85,
        arm_self_gain=0.17,
    )
    assert matched == pytest.approx(1.75)
    assert 1.0 - matched * 0.17 == pytest.approx(1.0 - 0.35 * 0.85)
    assert positive_gain_matched_controller_gain(
        base_controller_gain=0.35,
        reference_self_gain=0.85,
        arm_self_gain=-0.01,
    ) is None


def test_affine_support_projection_is_translation_invariant_and_idempotent() -> None:
    states = np.array(
        [
            [4.0, -2.0, 9.0],
            [5.0, -2.0, 9.0],
            [4.0, -1.0, 9.0],
            [5.0, -1.0, 9.0],
        ]
    )
    support = fit_affine_support(states, relative_tolerance=1e-8)
    vector = np.array([2.0, 3.0, 7.0])
    projected = support.project_direction(vector)

    assert support.rank == 2
    assert np.allclose(projected, [2.0, 3.0, 0.0])
    assert np.allclose(support.project_direction(projected), projected)
    assert np.allclose(support.off_support_distance(states), 0.0)


def test_projected_direction_stays_in_support_and_restores_target_gain() -> None:
    states = np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0]])
    support = fit_affine_support(states, relative_tolerance=1e-8)
    projected, _scale = project_direction_to_target_gain(
        np.array([2.0, 1.0, 5.0]),
        support=support,
        target_gradient=np.array([1.0, 1.0, 1.0]),
        target_gain=0.85,
    )

    assert np.linalg.norm(support.off_support_component(projected)) < 1e-12
    assert np.dot(np.ones(3), projected) == pytest.approx(0.85)
    assert direction_support_metrics(projected, support)["direction_off_support_fraction"] < 1e-12


def test_projected_direction_without_target_authority_is_explicitly_infeasible() -> None:
    support = fit_affine_support(np.array([[0.0, 0.0], [1.0, 0.0]]), relative_tolerance=1e-8)
    with pytest.raises(ValueError, match="no usable target authority"):
        project_direction_to_target_gain(
            np.array([1.0, 0.0]),
            support=support,
            target_gradient=np.array([0.0, 1.0]),
            target_gain=0.85,
        )
