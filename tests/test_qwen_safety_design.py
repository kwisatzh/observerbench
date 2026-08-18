"""Model-free tests for the Qwen safety design.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from collections import Counter, defaultdict

from observerbench.tasks.qwen_safety.design import (
    QwenSafetyDesignConfig,
    build_qwen_safety_design,
)


def _design():
    return build_qwen_safety_design(
        QwenSafetyDesignConfig(
            fit_pairs=12,
            calibration_pairs=8,
            locked_test_pairs_per_stratum=4,
            resources_per_bank=12,
        )
    )


def test_qwen_safety_pairs_hold_requested_action_fixed() -> None:
    design = _design()
    by_pair = defaultdict(list)
    for prompt in design.prompts:
        by_pair[prompt.pair_id].append(prompt)

    assert by_pair
    for rows in by_pair.values():
        assert len(rows) == 2
        safe, unsafe = sorted(rows, key=lambda row: row.unsafe_if_allowed)
        assert not safe.unsafe_if_allowed
        assert unsafe.unsafe_if_allowed
        assert safe.action_span == unsafe.action_span
        assert safe.resource == unsafe.resource
        assert safe.operation == unsafe.operation
        assert safe.granted_resource != unsafe.granted_resource


def test_qwen_safety_fit_excludes_both_heldout_dimensions() -> None:
    design = _design()
    fit = design.prompts_for("fit")
    calibration = design.prompts_for("calibration")

    assert all(prompt.operation != "delete" for prompt in (*fit, *calibration))
    assert all(prompt.template != "xml" for prompt in (*fit, *calibration))


def test_locked_test_balances_four_generalization_strata() -> None:
    design = _design()
    counts = Counter(prompt.family_id for prompt in design.prompts_for("locked_test"))

    assert set(counts) == {
        "seen_operation__seen_template",
        "heldout_operation__seen_template",
        "seen_operation__heldout_template",
        "heldout_operation__heldout_template",
    }
    assert set(counts.values()) == {8}


def test_resource_names_are_disjoint_across_banks() -> None:
    design = _design()
    banks = list(design.resource_banks.values())
    assert all(set(left).isdisjoint(right) for i, left in enumerate(banks) for right in banks[i + 1 :])


def test_design_hash_is_deterministic_and_seed_sensitive() -> None:
    first = _design()
    second = _design()
    changed = build_qwen_safety_design(
        QwenSafetyDesignConfig(
            seed=first.config.seed + 1,
            fit_pairs=12,
            calibration_pairs=8,
            locked_test_pairs_per_stratum=4,
            resources_per_bank=12,
        )
    )

    assert first.design_sha256 == second.design_sha256
    assert first.design_sha256 != changed.design_sha256
