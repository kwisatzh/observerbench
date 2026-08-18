"""Tests for the secondary Qwen safety observer-boundary analyses.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import numpy as np

from observerbench.tasks.qwen_safety.design import (
    QwenSafetyDesignConfig,
    build_qwen_safety_design,
)
from observerbench.tasks.qwen_safety.followup import (
    full_context_hash_features,
    model_output_risk_scores,
    recovered_block_minus_allow_margin,
    stratified_pair_bootstrap_contrast,
)


def _design():
    return build_qwen_safety_design(
        QwenSafetyDesignConfig(
            fit_pairs=4,
            calibration_pairs=4,
            locked_test_pairs_per_stratum=2,
            resources_per_bank=8,
        )
    )


def test_model_margin_uses_direct_cache_and_matches_legacy_recovery() -> None:
    design = _design()
    prompts = design.prompts_for("locked_test")
    ids = np.asarray([prompt.prompt_id for prompt in prompts])
    raw = np.asarray([2.0 if prompt.unsafe_if_allowed else -3.0 for prompt in prompts])
    oriented = raw * np.asarray(
        [1.0 if prompt.unsafe_if_allowed else -1.0 for prompt in prompts]
    )
    legacy = {"prompt_ids": ids, "candidate_margins": oriented}
    direct = {**legacy, "block_minus_allow_margins": raw}

    assert np.array_equal(
        recovered_block_minus_allow_margin(design, legacy, bank="locked_test"),
        raw,
    )
    assert np.array_equal(
        recovered_block_minus_allow_margin(design, direct, bank="locked_test"),
        raw,
    )
    scores = model_output_risk_scores(design, direct, bank="locked_test")
    assert np.all(scores[[prompt.unsafe_if_allowed for prompt in prompts]] > 0.0)


def test_full_context_hash_is_deterministic_and_pair_sensitive() -> None:
    prompts = _design().prompts_for("fit")
    first = full_context_hash_features(prompts, dimension=64)
    second = full_context_hash_features(prompts, dimension=64)

    assert np.array_equal(first, second)
    assert first.shape == (len(prompts), 64)
    assert not np.array_equal(first[0], first[1])


def test_pair_bootstrap_reports_candidate_minus_reference() -> None:
    candidate = []
    reference = []
    for pair_index in range(4):
        for unsafe in (False, True):
            row = {
                "query_id": f"q-{pair_index}-{int(unsafe)}",
                "pair_id": f"pair-{pair_index}",
                "family_id": "family-a",
            }
            candidate.append({**row, "loss": 1.0})
            reference.append({**row, "loss": 2.0})
    result = stratified_pair_bootstrap_contrast(
        candidate,
        reference,
        alpha=0.5,
        replicates=100,
        seed=3,
    )

    assert result["mean_loss_difference"] == -1.0
    assert result["cvar_difference"] == -1.0
    assert result["mean_loss_difference_ci_high"] == -1.0
