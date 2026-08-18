"""Tests for the Phase-5 nonlinear suffix-loop experiment.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from observerbench.tasks.trained_ctl1 import (
    generate_data,
    train_model,
)
from observerbench.tasks.trained_ctl2_suffix import (
    NonlinearSuffixConfig,
    _certificate,
    _clean_residual_distance_scale,
    _displacement_scale_diagnostics,
    _fit_prefix_observers,
    _natural_factorial_analysis,
    _residual_displacement_diagnostic,
    _rollout,
    _tokens_for_archetypes,
    extract_prefix_residuals,
    solve_controlled_direction,
)


def _quick_model():
    cfg = NonlinearSuffixConfig(
        seeds=(0,),
        n_train=256,
        n_test=64,
        train_steps=25,
        d_model=12,
        n_heads=2,
        n_layers=2,
        d_mlp=32,
        batch_size=64,
        bootstrap_repeats=20,
    )
    train, _test = generate_data(cfg)
    model, _info = train_model(cfg, train)
    return cfg, train, model


def test_split_encode_recomposes_forward() -> None:
    cfg, train, model = _quick_model()
    tokens = torch.as_tensor(train["tokens"][:7], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        direct_target, direct_features, direct_h = model(tokens, return_h=True)
        prefix = model.encode_prefix(tokens, completed_blocks=1)
        split_target, split_features, split_h = model.decode_suffix(
            prefix,
            completed_blocks=1,
            return_h=True,
        )
    assert torch.allclose(direct_target, split_target, atol=1e-6, rtol=1e-6)
    assert torch.allclose(direct_features, split_features, atol=1e-6, rtol=1e-6)
    assert torch.allclose(direct_h, split_h, atol=1e-6, rtol=1e-6)


def test_controlled_direction_hits_both_finite_gain_constraints() -> None:
    cfg, train, model = _quick_model()
    observers, _readout = _fit_prefix_observers(model, train, cfg)
    tokens, _archetypes = _tokens_for_archetypes(cfg.seq_len)
    prefix = extract_prefix_residuals(model, tokens[:1], cfg.completed_blocks)[0]
    solution = solve_controlled_direction(
        model,
        prefix,
        observers["first_order"].gradient,
        completed_blocks=cfg.completed_blocks,
        observer_gain=cfg.observer_self_gain,
        rho_target=0.75,
        delta=cfg.finite_delta,
        manipulability_tolerance=cfg.manipulability_tolerance,
        gain_tolerance=cfg.direction_gain_tolerance,
    )
    assert np.isclose(solution.observer_gain, cfg.observer_self_gain, atol=1e-7)
    assert np.isclose(solution.rho, 0.75, atol=cfg.direction_gain_tolerance / cfg.observer_self_gain)
    assert solution.manipulability_denominator > cfg.manipulability_tolerance

    all_prefixes = extract_prefix_residuals(model, tokens, cfg.completed_blocks)
    scale = _clean_residual_distance_scale(all_prefixes[:, -1, :], 1e-12)
    rollout = _rollout(
        model,
        prefix,
        observers["first_order"],
        solution.direction,
        observers["first_order"].estimate(prefix[-1]) + 0.25,
        scale,
        cfg,
    )
    assert rollout["clean_archetype_count"] == 4
    assert rollout["clean_pairwise_distance_count"] == 6
    assert np.isclose(
        rollout["residual_displacement_over_clean_pairwise_median"],
        rollout["residual_displacement_l2"] / scale.median,
    )
    assert np.isclose(
        rollout["residual_displacement_over_clean_pairwise_max"],
        rollout["residual_displacement_l2"] / scale.maximum,
    )


def test_certificate_is_exact_for_affine_shadow() -> None:
    ehat0 = 0.63
    mismatch = -0.12
    rho = 0.75
    gain = 0.85
    controller_gain = 0.35
    steps = 15
    pole = 1.0 - controller_gain * gain
    predicted = _certificate(ehat0, mismatch, rho, pole, steps)
    zhat = 0.4
    z = zhat + mismatch
    target = zhat + ehat0
    for _ in range(steps):
        action = controller_gain * (target - zhat)
        zhat += gain * action
        z += gain * (1.0 + rho) * action
    assert np.isclose(predicted, target - z, atol=1e-12)


def test_clean_residual_distance_scale_uses_all_six_archetype_pairs() -> None:
    residuals = np.asarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
        [1.0, 2.0],
    ])
    scale = _clean_residual_distance_scale(residuals, tolerance=1e-12)
    assert scale.archetype_count == 4
    assert scale.pair_count == 6
    assert scale.nonzero_pair_count == 6
    assert np.isclose(scale.median, 2.0)
    assert np.isclose(scale.maximum, np.sqrt(5.0))
    diagnostic = _displacement_scale_diagnostics(2.0, scale)
    assert np.isclose(diagnostic["residual_displacement_over_clean_pairwise_median"], 1.0)
    assert np.isclose(
        diagnostic["residual_displacement_over_clean_pairwise_max"],
        2.0 / np.sqrt(5.0),
    )


def test_collapsed_clean_residual_scale_marks_ratios_undefined() -> None:
    scale = _clean_residual_distance_scale(np.zeros((4, 3)), tolerance=1e-10)
    diagnostic = _displacement_scale_diagnostics(0.4, scale)
    assert scale.pair_count == 6
    assert scale.nonzero_pair_count == 0
    assert scale.median_degenerate
    assert scale.maximum_degenerate
    assert np.isnan(diagnostic["residual_displacement_over_clean_pairwise_median"])
    assert np.isnan(diagnostic["residual_displacement_over_clean_pairwise_max"])


def test_displacement_summary_uses_only_natural_diagonal_arms() -> None:
    scale = _clean_residual_distance_scale(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        tolerance=1e-12,
    )

    def row(seed: int, gamma: float, displacement: float) -> dict[str, object]:
        return {
            "seed": seed,
            "gamma": gamma,
            "residual_displacement_l2": displacement,
            **_displacement_scale_diagnostics(displacement, scale),
        }

    natural = pd.DataFrame([
        {
            **row(0, 0.0, 1.5),
            "estimator": "first_order",
            "direction_provider": "first_order",
        },
        {
            **row(1, 0.0, 3.0),
            "estimator": "first_order",
            "direction_provider": "first_order",
        },
        {
            **row(0, 0.0, 100.0),
            "estimator": "first_order",
            "direction_provider": "lifted_interaction",
        },
        {
            **row(0, 1.15, 0.75),
            "estimator": "lifted_interaction",
            "direction_provider": "lifted_interaction",
        },
    ])
    controlled = pd.DataFrame([
        {**row(0, 1.15, 1.5), "rho_target": -0.75},
        {**row(1, 1.15, 3.0), "rho_target": 0.75},
    ])
    summary = _residual_displacement_diagnostic(controlled, natural)
    assert summary["status"] == "diagnostic_only_not_a_success_gate"
    natural_groups = {
        (item["gamma"], item["estimator"]): item
        for item in summary["natural_diagonal_by_gamma_estimator"]
    }
    first_order = natural_groups[(0.0, "first_order")]
    assert first_order["condition_count"] == 2
    assert np.isclose(first_order["ratio_to_clean_pairwise_median_median"], 1.5)
    assert first_order["ratio_to_clean_pairwise_max_fraction_above_one"] == 0.0
    assert summary["controlled_overall"]["condition_count"] == 2
    assert len(summary["controlled_by_rho_target"]) == 2


def test_natural_factorial_analysis_flags_ill_conditioned_crossed_arms() -> None:
    rows = []
    cell_ise = {
        ("first_order", "first_order"): 2.6,
        ("first_order", "lifted_interaction"): 2.4,
        ("lifted_interaction", "first_order"): 3.7,
        ("lifted_interaction", "lifted_interaction"): 1.9,
    }
    for seed in (0, 1):
        for (estimator, direction), ise in cell_ise.items():
            crossed = estimator != direction
            rows.append({
                "seed": seed,
                "gamma": 1.15,
                "estimator": estimator,
                "direction_provider": direction,
                "actual_integrated_squared_error": ise + 0.1 * seed,
                "observer_gain": -0.2 if crossed else 0.85,
                "observer_pole": 1.1 if crossed else 0.7025,
                "would_clip_fraction": 0.2 if crossed else 0.0,
                "max_abs_action": 5.0 if crossed else 0.2,
                "residual_displacement_over_clean_pairwise_max": (
                    2.0 if crossed else 0.2
                ),
            })
    cfg = NonlinearSuffixConfig(
        seeds=(0, 1),
        gamma_values_natural=(1.15,),
        bootstrap_repeats=20,
    )
    analysis = _natural_factorial_analysis(pd.DataFrame(rows), cfg)
    gamma = analysis["by_gamma"]["1.15"]
    assert not gamma[
        "crossed_arms_pass_conditioning_and_clean_scale_diagnostics"
    ]
    assert gamma["attribution_status"].endswith(
        "diagonal_result_remains_system_level"
    )
    crossed = [
        cell for cell in gamma["cells"]
        if cell["estimator"] != cell["direction_provider"]
    ]
    assert all(cell["nonpositive_observer_gain_seed_count"] == 2 for cell in crossed)
    assert all(cell["unstable_observer_pole_seed_count"] == 2 for cell in crossed)
    assert all(cell["would_clip_condition_count"] == 2 for cell in crossed)
    assert all(
        cell["displacement_exceeds_clean_pairwise_max_condition_count"] == 2
        for cell in crossed
    )
    compatibility = gamma["contrasts"][
        "lifted_pair_compatibility_interaction"
    ]
    assert compatibility["mean_ise_difference"] > 0.0
