"""Matched non-induction collateral diagnostic for the Qwen copy task.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This module measures distribution shift on a deliberately unmatched control.
It is a narrow mechanism diagnostic, not a deployment-safety evaluation.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from observerbench.provenance import json_sha256
from observerbench.tasks.qwen_induction.plant import (
    HeadAblationMeans,
    HeadRef,
    _mask_record,
)


CONTROL_SCHEMA = "observerbench.qwen_induction.matched_non_induction.v1"


def _record_value(record: Any, *names: str) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    raise ValueError(f"record lacks one of: {', '.join(names)}")


def _integer_tuple(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, str):
        value = value.split()
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a sequence of integers") from error
    if not result or any(item < 0 for item in result):
        raise ValueError(f"{label} must be a nonempty nonnegative integer sequence")
    return result


def _source_key_positions(record: Any, tokens: tuple[int, ...]) -> tuple[int, int, int]:
    try:
        positions = _integer_tuple(_record_value(record, "key_positions"), "key positions")
    except ValueError as error:
        if "record lacks" not in str(error):
            raise
        query = int(_record_value(record, "query_position", "final_key_position"))
        target_position = int(
            _record_value(record, "source_key_position", "target_key_position")
        )
        distractors = (
            int(_record_value(record, "distractor_token_id_1")),
            int(_record_value(record, "distractor_token_id_2")),
        )
        derived = [target_position]
        for token in distractors:
            value_positions = [
                position
                for position, observed in enumerate(tokens[:query])
                if observed == token
            ]
            if len(value_positions) != 1 or value_positions[0] == 0:
                raise ValueError(
                    "cannot derive a unique distractor source bigram"
                ) from None
            derived.append(value_positions[0] - 1)
        positions = tuple(derived)
    if len(positions) != 3:
        raise ValueError("matched controls require exactly three key positions")
    return positions  # type: ignore[return-value]


def _validate_source_fixture(
    tokens: tuple[int, ...],
    query_position: int,
    key_positions: tuple[int, int, int],
) -> None:
    if query_position != len(tokens) - 1:
        raise ValueError("source induction query must be the final supplied token")
    prefix = tokens[:-1]
    if len(prefix) != len(set(prefix)):
        raise ValueError("source induction prefix must be token-unique")
    if len(set(key_positions)) != 3:
        raise ValueError("source key positions must be distinct")
    occupied: set[int] = set()
    for position in key_positions:
        if position < 0 or position + 1 >= query_position:
            raise ValueError("source key-value bigram lies outside the causal prefix")
        pair = {position, position + 1}
        if occupied.intersection(pair):
            raise ValueError("source key-value bigrams overlap")
        occupied.update(pair)
    key_value_tokens = tuple(
        tokens[position + offset]
        for position in key_positions
        for offset in (0, 1)
    )
    if len(set(key_value_tokens)) != 6:
        raise ValueError("source key-value tokens must be distinct")
    if tokens[query_position] != tokens[key_positions[0]]:
        raise ValueError("source final token does not repeat the target key")
    if tokens.count(tokens[query_position]) != 2:
        raise ValueError("source target key must occur exactly twice")


def _eligible_filler_positions(
    tokens: tuple[int, ...],
    query_position: int,
    key_positions: tuple[int, int, int],
) -> tuple[int, ...]:
    key_value_positions = {
        position + offset
        for position in key_positions
        for offset in (0, 1)
    }
    key_value_tokens = {tokens[position] for position in key_value_positions}
    counts = Counter(tokens)
    return tuple(
        position
        for position in range(query_position)
        if position not in key_value_positions
        and tokens[position] not in key_value_tokens
        and counts[tokens[position]] == 1
    )


def _deterministic_filler_position(
    *,
    source_prompt_id: str,
    family_id: str,
    tokens: tuple[int, ...],
    eligible: tuple[int, ...],
) -> int:
    if not eligible:
        raise ValueError("source fixture has no unique non-KV filler position")
    digest = json_sha256(
        {
            "schema_version": CONTROL_SCHEMA,
            "source_prompt_id": source_prompt_id,
            "family_id": family_id,
            "input_ids": tokens,
        }
    )
    return eligible[int(digest[:16], 16) % len(eligible)]


@dataclass(frozen=True, slots=True)
class MatchedNonInductionPrompt:
    """One exact-multiset control whose final token has no earlier match."""

    prompt_id: str
    source_prompt_id: str
    family_id: str
    source_input_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    query_position: int
    key_positions: tuple[int, int, int]
    swapped_position: int
    schema_version: str = CONTROL_SCHEMA

    @property
    def tokens(self) -> tuple[int, ...]:
        return self.input_ids


def validate_matched_non_induction_control(control: MatchedNonInductionPrompt) -> None:
    """Reject any control that is not the frozen two-position permutation."""

    if not isinstance(control, MatchedNonInductionPrompt):
        raise TypeError("collateral measurement accepts only matched control records")
    if control.schema_version != CONTROL_SCHEMA:
        raise ValueError("unexpected matched-control schema")
    if not control.prompt_id or not control.source_prompt_id or not control.family_id:
        raise ValueError("matched-control identifiers must be nonempty")
    source = _integer_tuple(control.source_input_ids, "source input IDs")
    matched = _integer_tuple(control.input_ids, "matched input IDs")
    if len(source) != len(matched):
        raise ValueError("matched control changed sequence length")
    _validate_source_fixture(source, control.query_position, control.key_positions)
    if control.query_position != len(matched) - 1:
        raise ValueError("matched-control query must remain at the final position")
    eligible = _eligible_filler_positions(
        source, control.query_position, control.key_positions
    )
    expected_swap = _deterministic_filler_position(
        source_prompt_id=control.source_prompt_id,
        family_id=control.family_id,
        tokens=source,
        eligible=eligible,
    )
    if control.swapped_position != expected_swap:
        raise ValueError("matched control did not use the frozen filler-selection rule")
    if Counter(source) != Counter(matched):
        raise ValueError("matched control changed the exact token multiset")
    expected = list(source)
    expected[control.swapped_position], expected[control.query_position] = (
        expected[control.query_position],
        expected[control.swapped_position],
    )
    if matched != tuple(expected):
        raise ValueError("matched control differs by more than the frozen swap")
    query_token = matched[control.query_position]
    if query_token in matched[: control.query_position]:
        raise ValueError("matched-control final token still has an earlier occurrence")
    changed = tuple(
        index for index, pair in enumerate(zip(source, matched)) if pair[0] != pair[1]
    )
    if changed != (control.swapped_position, control.query_position):
        raise ValueError("matched control must change exactly two token positions")
    expected_id = f"qcontrol_{json_sha256({'source_prompt_id': control.source_prompt_id, 'family_id': control.family_id, 'input_ids': matched, 'swapped_position': control.swapped_position})[:16]}"
    if control.prompt_id != expected_id:
        raise ValueError("matched-control prompt ID does not match its contents")


def make_matched_non_induction_control(record: Any) -> MatchedNonInductionPrompt:
    """Swap the repeated final key with one deterministically selected filler."""

    source_prompt_id = str(_record_value(record, "prompt_id", "example_id"))
    family_id = str(_record_value(record, "family_id"))
    tokens = _integer_tuple(_record_value(record, "input_ids", "tokens"), "input IDs")
    query_position = int(
        _record_value(record, "query_position", "final_key_position")
    )
    key_positions = _source_key_positions(record, tokens)
    _validate_source_fixture(tokens, query_position, key_positions)
    eligible = _eligible_filler_positions(tokens, query_position, key_positions)
    swapped_position = _deterministic_filler_position(
        source_prompt_id=source_prompt_id,
        family_id=family_id,
        tokens=tokens,
        eligible=eligible,
    )
    matched = list(tokens)
    matched[swapped_position], matched[query_position] = (
        matched[query_position],
        matched[swapped_position],
    )
    input_ids = tuple(matched)
    prompt_id = f"qcontrol_{json_sha256({'source_prompt_id': source_prompt_id, 'family_id': family_id, 'input_ids': input_ids, 'swapped_position': swapped_position})[:16]}"
    control = MatchedNonInductionPrompt(
        prompt_id=prompt_id,
        source_prompt_id=source_prompt_id,
        family_id=family_id,
        source_input_ids=tokens,
        input_ids=input_ids,
        query_position=query_position,
        key_positions=key_positions,
        swapped_position=swapped_position,
    )
    validate_matched_non_induction_control(control)
    return control


def make_matched_non_induction_controls(
    records: Sequence[Any],
) -> tuple[MatchedNonInductionPrompt, ...]:
    controls = tuple(make_matched_non_induction_control(record) for record in records)
    if len({control.prompt_id for control in controls}) != len(controls):
        raise ValueError("matched-control prompt IDs must be unique")
    return controls


@dataclass(frozen=True, slots=True)
class CollateralShiftRow:
    """Full-vocabulary next-token shift under one frozen head mask."""

    prompt_id: str
    source_prompt_id: str
    family_id: str
    mask_id: str
    mask_bits: str
    kl_clean_to_intervened: float
    total_variation: float | None


def _validated_controls(
    controls: Sequence[MatchedNonInductionPrompt],
) -> tuple[MatchedNonInductionPrompt, ...]:
    result = tuple(controls)
    if not result:
        raise ValueError("collateral measurement requires at least one control")
    for control in result:
        validate_matched_non_induction_control(control)
    if len({control.prompt_id for control in result}) != len(result):
        raise ValueError("collateral controls must have unique prompt IDs")
    return result


def _batches(
    controls: Sequence[MatchedNonInductionPrompt], batch_size: int
) -> Iterator[tuple[MatchedNonInductionPrompt, ...]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    by_length: dict[int, list[MatchedNonInductionPrompt]] = {}
    for control in controls:
        by_length.setdefault(len(control.input_ids), []).append(control)
    for length in sorted(by_length):
        rows = by_length[length]
        for start in range(0, len(rows), batch_size):
            yield tuple(rows[start : start + batch_size])


def _input_tensor(plant: Any, controls: Sequence[MatchedNonInductionPrompt]) -> Any:
    if not controls:
        raise ValueError("cannot construct an empty collateral batch")
    length = len(controls[0].input_ids)
    if any(
        control.query_position != len(control.input_ids) - 1
        or len(control.input_ids) != length
        for control in controls
    ):
        raise ValueError("collateral batch must have equal lengths and final queries")
    vocab_size = int(getattr(plant.model.config, "vocab_size"))
    if any(token >= vocab_size for control in controls for token in control.input_ids):
        raise ValueError("collateral input token lies outside the model vocabulary")
    tensor = plant.torch.as_tensor(
        [control.input_ids for control in controls],
        dtype=plant.torch.long,
        device=plant.device,
    )
    if tensor.ndim != 2 or tensor.shape != (len(controls), length):
        raise ValueError("collateral input tensor has the wrong shape")
    return tensor


def _forward_final_logits(
    plant: Any,
    controls: Sequence[MatchedNonInductionPrompt],
    heads: tuple[HeadRef, ...],
    means: HeadAblationMeans,
    bits: tuple[int, ...] | None,
) -> Any:
    context = nullcontext()
    if bits is not None:
        context = plant.head_intervention(
            heads,
            mask_rows=np.tile(np.asarray(bits, dtype=np.uint8), (len(controls), 1)),
            query_positions=[control.query_position for control in controls],
            family_ids=[control.family_id for control in controls],
            mode="mean",
            position_scope="final",
            means=means,
        )
    with context, plant.torch.inference_mode():
        output = plant.model(
            input_ids=_input_tensor(plant, controls),
            use_cache=False,
            logits_to_keep=1,
        )
    logits = output.logits
    vocab_size = int(getattr(plant.model.config, "vocab_size"))
    if (
        logits.ndim != 3
        or logits.shape[0] != len(controls)
        or logits.shape[1] != 1
        or logits.shape[2] != vocab_size
    ):
        raise ValueError("Qwen collateral scoring did not return final-position logits")
    final_logits = logits[:, 0].detach().float()
    if not bool(plant.torch.isfinite(final_logits).all().detach().cpu()):
        raise ValueError("Qwen collateral scoring returned non-finite logits")
    return final_logits


def iter_collateral_distribution_shifts(
    plant: Any,
    controls: Sequence[MatchedNonInductionPrompt],
    heads: Sequence[HeadRef],
    masks: Sequence[Any],
    means: HeadAblationMeans,
    *,
    batch_size: int = 16,
    include_total_variation: bool = True,
) -> Iterator[tuple[str, list[CollateralShiftRow]]]:
    """Yield one mask's matched-control KL rows for resumable serialization.

    This path intentionally does not call the induction candidate scorer: the
    final query has no previous match. It still validates the frozen swap,
    final query position, model inputs, intervention metadata, and logits.
    """

    controls = _validated_controls(controls)
    heads = tuple(heads)
    if not heads or means.heads != heads:
        raise ValueError("collateral measurement requires the frozen aligned head panel")
    if means.head_dim != int(plant.architecture.head_dim):
        raise ValueError("collateral reference means do not match the Qwen head dimension")
    for control in controls:
        means.family_index(control.family_id)
    parsed_masks = tuple(_mask_record(mask, len(heads)) for mask in masks)
    if not parsed_masks:
        raise ValueError("collateral measurement requires at least one mask")
    if len({mask_id for mask_id, _bits in parsed_masks}) != len(parsed_masks):
        raise ValueError("collateral mask IDs must be unique")

    clean_logits: dict[str, Any] = {}
    for batch in _batches(controls, batch_size):
        logits = _forward_final_logits(plant, batch, heads, means, None).cpu()
        for control, row in zip(batch, logits):
            clean_logits[control.prompt_id] = row.clone()

    for mask_id, bits in parsed_masks:
        bit_string = "".join(map(str, bits))
        if not any(bits):
            yield mask_id, [
                CollateralShiftRow(
                    prompt_id=control.prompt_id,
                    source_prompt_id=control.source_prompt_id,
                    family_id=control.family_id,
                    mask_id=mask_id,
                    mask_bits=bit_string,
                    kl_clean_to_intervened=0.0,
                    total_variation=0.0 if include_total_variation else None,
                )
                for control in controls
            ]
            continue

        metrics: dict[str, tuple[float, float | None]] = {}
        for batch in _batches(controls, batch_size):
            intervened = _forward_final_logits(
                plant, batch, heads, means, bits
            )
            clean = plant.torch.stack(
                [clean_logits[control.prompt_id] for control in batch]
            ).to(device=intervened.device, dtype=plant.torch.float32)
            clean_log_probability = plant.torch.log_softmax(clean, dim=-1)
            intervened_log_probability = plant.torch.log_softmax(
                intervened.float(), dim=-1
            )
            clean_probability = clean_log_probability.exp()
            kl = (
                clean_probability
                * (clean_log_probability - intervened_log_probability)
            ).sum(dim=-1).clamp_min(0.0)
            if include_total_variation:
                intervened_probability = intervened_log_probability.exp()
                tv = 0.5 * (
                    clean_probability - intervened_probability
                ).abs().sum(dim=-1)
            else:
                tv = None
            for index, control in enumerate(batch):
                value = float(kl[index].detach().cpu())
                variation = (
                    None
                    if tv is None
                    else float(tv[index].detach().cpu())
                )
                if not np.isfinite(value) or (
                    variation is not None and not np.isfinite(variation)
                ):
                    raise ValueError("collateral distribution shift is non-finite")
                metrics[control.prompt_id] = (value, variation)
        yield mask_id, [
            CollateralShiftRow(
                prompt_id=control.prompt_id,
                source_prompt_id=control.source_prompt_id,
                family_id=control.family_id,
                mask_id=mask_id,
                mask_bits=bit_string,
                kl_clean_to_intervened=metrics[control.prompt_id][0],
                total_variation=metrics[control.prompt_id][1],
            )
            for control in controls
        ]


def measure_collateral_distribution_shifts(
    plant: Any,
    controls: Sequence[MatchedNonInductionPrompt],
    heads: Sequence[HeadRef],
    masks: Sequence[Any],
    means: HeadAblationMeans,
    *,
    batch_size: int = 16,
    include_total_variation: bool = True,
) -> list[CollateralShiftRow]:
    """Return the flattened collateral table for supplied frozen masks."""

    return [
        row
        for _mask_id, rows in iter_collateral_distribution_shifts(
            plant,
            controls,
            heads,
            masks,
            means,
            batch_size=batch_size,
            include_total_variation=include_total_variation,
        )
        for row in rows
    ]
