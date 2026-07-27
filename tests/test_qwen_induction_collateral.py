"""Tests for the matched Qwen non-induction collateral diagnostic.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from observerbench.tasks.qwen_induction.collateral import (
    make_matched_non_induction_control,
    make_matched_non_induction_controls,
    measure_collateral_distribution_shifts,
    validate_matched_non_induction_control,
)
from observerbench.tasks.qwen_induction.plant import HeadRef, Qwen2InductionPlant


def _source_record(
    prompt_id: str = "source-0",
    tokens: tuple[int, ...] = (
        40,
        10,
        11,
        41,
        20,
        21,
        42,
        43,
        30,
        31,
        44,
        45,
        10,
    ),
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "family_id": "f0",
        "input_ids": tokens,
        "query_position": len(tokens) - 1,
        "target_token_id": tokens[2],
        "distractor_token_id_1": tokens[5],
        "distractor_token_id_2": tokens[9],
        "key_positions": (1, 4, 8),
    }


def test_matched_control_is_deterministic_exact_multiset_without_final_match() -> None:
    source = _source_record()
    first = make_matched_non_induction_control(source)
    second = make_matched_non_induction_control(source)

    assert first == second
    assert len(first.input_ids) == len(first.source_input_ids)
    assert Counter(first.input_ids) == Counter(first.source_input_ids)
    assert first.input_ids[-1] not in first.input_ids[:-1]
    changed = [
        index
        for index, values in enumerate(zip(first.source_input_ids, first.input_ids))
        if values[0] != values[1]
    ]
    assert changed == [first.swapped_position, first.query_position]
    assert first.input_ids[first.swapped_position] == first.source_input_ids[-1]
    assert first.input_ids[-1] == first.source_input_ids[first.swapped_position]
    kv_positions = {1, 2, 4, 5, 8, 9}
    assert first.swapped_position not in kv_positions
    validate_matched_non_induction_control(first)


def test_adapter_record_can_derive_distractor_key_positions() -> None:
    source = _source_record()
    source.pop("key_positions")
    source["source_key_position"] = 1

    control = make_matched_non_induction_control(source)

    assert control.key_positions == (1, 4, 8)


def test_matched_control_rejects_tampering_and_missing_filler() -> None:
    control = make_matched_non_induction_control(_source_record())
    changed = list(control.input_ids)
    changed[0] = 79
    with pytest.raises(ValueError, match="token multiset"):
        validate_matched_non_induction_control(
            replace(control, input_ids=tuple(changed))
        )

    no_filler = {
        "prompt_id": "no-filler",
        "family_id": "f0",
        "input_ids": (10, 11, 20, 21, 30, 31, 10),
        "query_position": 6,
        "key_positions": (0, 2, 4),
    }
    with pytest.raises(ValueError, match="no unique non-KV filler"):
        make_matched_non_induction_control(no_filler)


@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
def test_tiny_random_qwen_collateral_uses_final_logits_and_exact_noop(
    attention_implementation: str,
) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=80,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=79,
        eos_token_id=79,
    )
    config._attn_implementation = attention_implementation
    model = transformers.Qwen2ForCausalLM(config).eval()
    plant = Qwen2InductionPlant(model, tokenizer=SimpleNamespace(), device="cpu")

    sources = (
        _source_record("source-0"),
        _source_record(
            "source-1",
            (
                46,
                12,
                13,
                47,
                22,
                23,
                48,
                49,
                32,
                33,
                50,
                51,
                12,
            ),
        ),
    )
    heads = (HeadRef(0, 0, 0), HeadRef(0, 1, 0))
    means = plant.capture_reference_means(sources, heads, batch_size=2)
    controls = make_matched_non_induction_controls(sources)
    masks = (
        SimpleNamespace(mask_id="noop", bits=(0, 0)),
        SimpleNamespace(mask_id="head-0", bits=(1, 0)),
    )

    rows = measure_collateral_distribution_shifts(
        plant,
        controls,
        heads,
        masks,
        means,
        batch_size=2,
        include_total_variation=True,
    )

    assert len(rows) == 4
    noop = [row for row in rows if row.mask_id == "noop"]
    changed = [row for row in rows if row.mask_id == "head-0"]
    assert [row.prompt_id for row in noop] == [row.prompt_id for row in changed]
    assert all(row.kl_clean_to_intervened == 0.0 for row in noop)
    assert all(row.total_variation == 0.0 for row in noop)
    assert all(np.isfinite(row.kl_clean_to_intervened) for row in changed)
    assert all(row.kl_clean_to_intervened >= 0.0 for row in changed)
    assert all(
        row.total_variation is not None and 0.0 <= row.total_variation <= 1.0
        for row in changed
    )

    with pytest.raises(TypeError, match="only matched control"):
        measure_collateral_distribution_shifts(
            plant,
            [sources[0]],  # type: ignore[list-item]
            heads,
            masks[:1],
            means,
        )
