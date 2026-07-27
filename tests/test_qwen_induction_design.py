"""Tests for the model-free Qwen induction design.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from observerbench.tasks.qwen_induction.design import (
    SEQUENCE_BANKS,
    SEQUENCE_FAMILIES,
    SUPPORTED_MASK_BUDGETS,
    SequenceBankCounts,
    build_mask_design,
    build_sequence_design,
    freeze_component_panel,
    validate_mask_design,
    validate_sequence_design,
)


COUNTS = SequenceBankCounts(
    reference=1,
    discovery=1,
    head_fit=1,
    head_test=1,
    calibration=1,
    locked_test=1,
)
TOKEN_POOL = range(10_000, 12_000)
HEADS = tuple(f"layer-{layer}.head-{head}" for layer in range(2) for head in range(4))


def _sequence_design(seed: int = 17):
    return build_sequence_design(
        TOKEN_POOL,
        bank_counts=COUNTS,
        per_split_size=128,
        seed=seed,
    )


def _mask_design(seed: int = 23):
    return build_mask_design(HEADS, seed=seed)


def test_sequence_design_is_deterministic_and_hashed() -> None:
    first = _sequence_design()
    second = _sequence_design()
    changed = _sequence_design(seed=18)

    assert first == second
    assert first.design_sha256 == second.design_sha256
    assert first.design_sha256 != changed.design_sha256
    assert first.token_pool_sha256 == second.token_pool_sha256
    validate_sequence_design(first, regular_token_pool=TOKEN_POOL)


def test_all_six_token_banks_are_disjoint() -> None:
    design = _sequence_design()
    bank_tokens: dict[str, set[int]] = {}
    for bank in SEQUENCE_BANKS:
        examples = design.examples_for(bank)
        assert len(examples) == len(SEQUENCE_FAMILIES)
        token_bank = next(item for item in design.token_banks if item.bank == bank)
        bank_tokens[bank] = set(token_bank.token_ids)
        assert all(
            set(example.tokens[:-1]).issubset(bank_tokens[bank])
            for example in examples
        )

    for index, left in enumerate(SEQUENCE_BANKS):
        for right in SEQUENCE_BANKS[index + 1 :]:
            assert bank_tokens[left].isdisjoint(bank_tokens[right])


def test_fixtures_may_reuse_tokens_only_inside_their_bank() -> None:
    counts = SequenceBankCounts(
        reference=2,
        discovery=2,
        head_fit=2,
        head_test=2,
        calibration=2,
        locked_test=2,
    )
    design = build_sequence_design(
        range(384),
        bank_counts=counts,
        per_split_size=64,
        seed=5,
    )

    for bank in SEQUENCE_BANKS:
        longest = design.examples_for(bank, SEQUENCE_FAMILIES[-1])
        assert set(longest[0].tokens[:-1]).intersection(longest[1].tokens[:-1])


def test_four_families_have_three_pairs_and_exact_successor_targets() -> None:
    design = _sequence_design()
    assert {
        (example.sequence_length, example.induction_gap)
        for example in design.examples
    } == {(32, 8), (32, 16), (64, 8), (64, 16)}

    for example in design.examples:
        assert len(example.tokens) == example.sequence_length
        assert len(set(example.tokens[:-1])) == example.sequence_length - 1
        assert example.final_key_position - example.target_key_position == (
            example.induction_gap
        )
        assert example.tokens[-1] == example.target_key_token
        assert example.target_value_token == example.tokens[
            example.target_key_position + 1
        ]
        assert example.candidate_value_tokens == example.value_tokens
        assert len(set(example.key_tokens + example.value_tokens)) == 6
        occupied: set[int] = set()
        for position in example.key_positions:
            assert not occupied.intersection({position, position + 1})
            occupied.update({position, position + 1})


def test_sequence_design_rejects_small_pool_and_hash_drift() -> None:
    with pytest.raises(ValueError, match="needs 378 unique"):
        build_sequence_design(
            range(377), bank_counts=COUNTS, per_split_size=63, seed=1
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_sequence_design(
            [*range(377), 376],
            bank_counts=COUNTS,
            per_split_size=63,
            seed=1,
        )
    with pytest.raises(ValueError, match="at least 63"):
        build_sequence_design(
            TOKEN_POOL, bank_counts=COUNTS, per_split_size=62, seed=1
        )
    with pytest.raises(ValueError, match="hash does not match"):
        validate_sequence_design(replace(_sequence_design(), design_sha256="0" * 64))


def test_mask_budgets_are_nested_with_no_op_and_singleton_anchors() -> None:
    design = _mask_design()
    max_bank = design.calibration_for(128)

    assert design.supported_budgets == SUPPORTED_MASK_BUDGETS
    for budget in SUPPORTED_MASK_BUDGETS:
        masks = design.calibration_for(budget)
        assert len(masks) == budget
        assert masks == max_bank[:budget]
        assert masks[0].bits == (0,) * 8
        assert {mask.bits for mask in masks if mask.cardinality == 1} >= {
            tuple(1 if index == selected else 0 for index in range(8))
            for selected in range(8)
        }
    assert design.no_op.cardinality == 0
    assert design.no_op.as_numpy().dtype == np.uint8
    assert not design.no_op.as_numpy().flags.writeable


def test_first_40_masks_identify_the_full_quadratic_basis() -> None:
    design = _mask_design()
    first_40 = design.calibration_for(40)
    expected_anchors = [(0,) * 8]
    expected_anchors.extend(
        tuple(1 if component == selected else 0 for component in range(8))
        for selected in range(8)
    )
    expected_anchors.extend(
        tuple(
            1 if component in selected else 0
            for component in range(8)
        )
        for selected in combinations(range(8), 2)
    )
    assert [mask.bits for mask in first_40[:37]] == expected_anchors
    assert [mask.cardinality for mask in first_40[37:40]] == [8, 3, 7]

    linear = np.asarray([mask.bits for mask in first_40], dtype=float)
    quadratic = np.column_stack(
        [
            np.ones(len(linear)),
            linear,
            *[
                linear[:, left] * linear[:, right]
                for left, right in combinations(range(8), 2)
            ],
        ]
    )
    assert quadratic.shape == (40, 37)
    assert np.linalg.matrix_rank(quadratic) == 37
    assert np.linalg.cond(quadratic) < 30.0


def test_full_calibration_covers_densities_and_stays_well_conditioned() -> None:
    masks = _mask_design().calibration_for(128)
    assert {mask.cardinality for mask in masks} == set(range(9))

    linear = np.asarray([mask.bits for mask in masks], dtype=float)
    quadratic = np.column_stack(
        [
            np.ones(len(linear)),
            linear,
            *[
                linear[:, left] * linear[:, right]
                for left, right in combinations(range(8), 2)
            ],
        ]
    )
    assert np.linalg.matrix_rank(quadratic) == 37
    assert np.linalg.cond(quadratic) < 25.0


def test_mask_partition_is_exhaustive_and_heldout_pools_include_no_op() -> None:
    design = _mask_design()
    calibration_bits = {mask.bits for mask in design.calibration_for(128)}
    heldout_bits = {mask.bits for mask in design.heldout_masks}

    assert len(calibration_bits) == len(heldout_bits) == 128
    assert calibration_bits.isdisjoint(heldout_bits)
    assert len(calibration_bits | heldout_bits) == 256
    assert len(design.action_pools) == 16
    pooled: list[str] = []
    for pool in design.action_pools:
        assert len(pool.mask_ids) == 9
        assert pool.mask_ids[0] == design.no_op.mask_id
        pooled.extend(pool.mask_ids[1:])
    assert pooled == [mask.mask_id for mask in design.heldout_masks]
    validate_mask_design(design)


def test_mask_design_is_deterministic_and_panel_order_is_frozen() -> None:
    first = _mask_design()
    assert first == _mask_design()
    assert first.design_sha256 != _mask_design(seed=24).design_sha256
    assert freeze_component_panel(HEADS).panel_sha256 != freeze_component_panel(
        tuple(reversed(HEADS))
    ).panel_sha256


def test_mask_design_requires_eight_heads_and_rejects_hash_drift() -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        build_mask_design(HEADS[:-1], seed=1)
    with pytest.raises(ValueError, match="exactly eight"):
        build_mask_design(HEADS + ("extra",), seed=1)
    with pytest.raises(ValueError, match="hash does not match"):
        validate_mask_design(replace(_mask_design(), design_sha256="f" * 64))
