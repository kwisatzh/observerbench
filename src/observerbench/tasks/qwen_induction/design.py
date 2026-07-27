"""Model-free sequence and mask design for the Qwen induction circuit.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from observerbench.provenance import json_sha256


SEQUENCE_BANKS: tuple[str, ...] = (
    "reference",
    "discovery",
    "head_fit",
    "head_test",
    "calibration",
    "locked_test",
)
SUPPORTED_MASK_BUDGETS: tuple[int, ...] = (16, 40, 64, 128)
SEQUENCE_DESIGN_SCHEMA = "observerbench.qwen_induction.sequence_design.v1"
MASK_DESIGN_SCHEMA = "observerbench.qwen_induction.mask_design.v1"
PANEL_SCHEMA = "observerbench.qwen_induction.component_panel.v1"


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _normalise_token_pool(token_pool: Sequence[int]) -> tuple[int, ...]:
    tokens = tuple(_integer(token, "token id") for token in token_pool)
    if len(tokens) != len(set(tokens)):
        raise ValueError("regular-token pool must not contain duplicate token ids")
    return tuple(sorted(tokens))


@dataclass(frozen=True, slots=True)
class SequenceFamily:
    """One frozen sequence-length and induction-gap condition."""

    sequence_length: int
    induction_gap: int

    @property
    def family_id(self) -> str:
        return f"length_{self.sequence_length}_gap_{self.induction_gap}"


SEQUENCE_FAMILIES: tuple[SequenceFamily, ...] = (
    SequenceFamily(32, 8),
    SequenceFamily(32, 16),
    SequenceFamily(64, 8),
    SequenceFamily(64, 16),
)


@dataclass(frozen=True, slots=True)
class SequenceBankCounts:
    """Examples per family in each sealed token bank."""

    reference: int
    discovery: int
    head_fit: int
    head_test: int
    calibration: int
    locked_test: int

    def __post_init__(self) -> None:
        for bank in SEQUENCE_BANKS:
            _integer(getattr(self, bank), f"{bank} count", minimum=1)

    def as_dict(self) -> dict[str, int]:
        return {bank: getattr(self, bank) for bank in SEQUENCE_BANKS}


@dataclass(frozen=True, slots=True)
class InductionSequence:
    """One collision-free three-pair induction fixture."""

    example_id: str
    bank: str
    family_id: str
    family_index: int
    tokens: tuple[int, ...]
    sequence_length: int
    induction_gap: int
    key_positions: tuple[int, int, int]
    key_tokens: tuple[int, int, int]
    value_tokens: tuple[int, int, int]
    final_key_position: int
    target_value_token: int
    distractor_value_tokens: tuple[int, int]

    @property
    def target_key_position(self) -> int:
        return self.key_positions[0]

    @property
    def target_key_token(self) -> int:
        return self.key_tokens[0]

    @property
    def candidate_value_tokens(self) -> tuple[int, int, int]:
        return (self.target_value_token,) + self.distractor_value_tokens


@dataclass(frozen=True, slots=True)
class TokenBank:
    """Hash and size of one bank's exclusive regular-token allocation."""

    bank: str
    token_ids: tuple[int, ...]
    token_count: int
    token_sha256: str


@dataclass(frozen=True, slots=True)
class SequenceDesign:
    """Six disjoint banks spanning all four frozen sequence families."""

    seed: int
    bank_counts: SequenceBankCounts
    per_split_size: int
    token_pool_size: int
    token_pool_sha256: str
    token_banks: tuple[TokenBank, ...]
    examples: tuple[InductionSequence, ...]
    design_sha256: str
    schema_version: str = SEQUENCE_DESIGN_SCHEMA

    def examples_for(
        self,
        bank: str,
        family: SequenceFamily | None = None,
    ) -> tuple[InductionSequence, ...]:
        if bank not in SEQUENCE_BANKS:
            raise ValueError(f"unknown sequence bank {bank!r}")
        family_id = None if family is None else family.family_id
        return tuple(
            example
            for example in self.examples
            if example.bank == bank
            and (family_id is None or example.family_id == family_id)
        )


def _bank_counts(
    counts: SequenceBankCounts | Mapping[str, int],
) -> SequenceBankCounts:
    if isinstance(counts, SequenceBankCounts):
        return counts
    if set(counts) != set(SEQUENCE_BANKS):
        raise ValueError(f"bank counts must define exactly {SEQUENCE_BANKS}")
    return SequenceBankCounts(**{bank: counts[bank] for bank in SEQUENCE_BANKS})


def _choose_distractor_positions(
    family: SequenceFamily,
    rng: np.random.Generator,
) -> tuple[int, int]:
    final_position = family.sequence_length - 1
    target_position = final_position - family.induction_gap
    occupied = {target_position, target_position + 1}
    selected: list[int] = []
    for raw_position in rng.permutation(final_position - 1):
        position = int(raw_position)
        pair_positions = {position, position + 1}
        if pair_positions.intersection(occupied):
            continue
        selected.append(position)
        occupied.update(pair_positions)
        if len(selected) == 2:
            return selected[0], selected[1]
    raise AssertionError("failed to place two non-overlapping distractor bigrams")


def _sequence_payload(example: InductionSequence) -> dict[str, object]:
    return {
        "bank": example.bank,
        "family_id": example.family_id,
        "family_index": example.family_index,
        "tokens": example.tokens,
        "sequence_length": example.sequence_length,
        "induction_gap": example.induction_gap,
        "key_positions": example.key_positions,
        "key_tokens": example.key_tokens,
        "value_tokens": example.value_tokens,
        "final_key_position": example.final_key_position,
        "target_value_token": example.target_value_token,
        "distractor_value_tokens": example.distractor_value_tokens,
    }


def _sequence_design_payload(design: SequenceDesign) -> dict[str, object]:
    return {
        "schema_version": design.schema_version,
        "seed": design.seed,
        "families": [
            {
                "family_id": family.family_id,
                "sequence_length": family.sequence_length,
                "induction_gap": family.induction_gap,
            }
            for family in SEQUENCE_FAMILIES
        ],
        "bank_counts": design.bank_counts.as_dict(),
        "per_split_size": design.per_split_size,
        "token_pool_size": design.token_pool_size,
        "token_pool_sha256": design.token_pool_sha256,
        "token_banks": [
            {
                "bank": bank.bank,
                "token_ids": bank.token_ids,
                "token_count": bank.token_count,
                "token_sha256": bank.token_sha256,
            }
            for bank in design.token_banks
        ],
        "examples": [
            {"example_id": example.example_id, **_sequence_payload(example)}
            for example in design.examples
        ],
    }


def _tokens_needed(per_split_size: int) -> int:
    return len(SEQUENCE_BANKS) * per_split_size


def build_sequence_design(
    regular_token_pool: Sequence[int],
    *,
    bank_counts: SequenceBankCounts | Mapping[str, int],
    per_split_size: int,
    seed: int,
) -> SequenceDesign:
    """Build the six leakage-proof token banks without loading a model.

    Each sequence contains three annotated, non-overlapping key-value bigrams.
    Its final token repeats the target key. The exact next-token target is the
    target key's earlier successor; the other two values are distractors. All
    non-final tokens are unique within each fixture. Token banks are disjoint;
    fixtures within the same bank may reuse that bank's tokens.
    """

    seed = _integer(seed, "seed")
    counts = _bank_counts(bank_counts)
    per_split_size = _integer(per_split_size, "per-split token count", minimum=1)
    minimum_size = max(
        family.sequence_length - 1 for family in SEQUENCE_FAMILIES
    )
    if per_split_size < minimum_size:
        raise ValueError(
            f"per-split token count must be at least {minimum_size}"
        )
    token_pool = _normalise_token_pool(regular_token_pool)
    required = _tokens_needed(per_split_size)
    if required > len(token_pool):
        raise ValueError(
            f"sequence design needs {required} unique regular tokens; "
            f"the pool has {len(token_pool)}"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(token_pool))[:required]
    selected_tokens = tuple(token_pool[int(index)] for index in order)
    offset = 0
    examples: list[InductionSequence] = []
    bank_tokens: dict[str, tuple[int, ...]] = {}
    for bank in SEQUENCE_BANKS:
        bank_pool = selected_tokens[offset : offset + per_split_size]
        offset += per_split_size
        bank_tokens[bank] = bank_pool
        for family in SEQUENCE_FAMILIES:
            for family_index in range(getattr(counts, bank)):
                prefix_length = family.sequence_length - 1
                bank_order = rng.permutation(per_split_size)[:prefix_length]
                prefix = tuple(bank_pool[int(index)] for index in bank_order)

                final_position = family.sequence_length - 1
                target_position = final_position - family.induction_gap
                distractors = _choose_distractor_positions(family, rng)
                key_positions = (target_position,) + distractors
                key_tokens = tuple(prefix[position] for position in key_positions)
                value_tokens = tuple(
                    prefix[position + 1] for position in key_positions
                )
                tokens = prefix + (key_tokens[0],)
                provisional = InductionSequence(
                    example_id="",
                    bank=bank,
                    family_id=family.family_id,
                    family_index=family_index,
                    tokens=tokens,
                    sequence_length=family.sequence_length,
                    induction_gap=family.induction_gap,
                    key_positions=key_positions,
                    key_tokens=key_tokens,
                    value_tokens=value_tokens,
                    final_key_position=final_position,
                    target_value_token=value_tokens[0],
                    distractor_value_tokens=(value_tokens[1], value_tokens[2]),
                )
                example_id = (
                    f"qseq_{json_sha256(_sequence_payload(provisional))[:16]}"
                )
                examples.append(
                    InductionSequence(
                        example_id=example_id,
                        bank=provisional.bank,
                        family_id=provisional.family_id,
                        family_index=provisional.family_index,
                        tokens=provisional.tokens,
                        sequence_length=provisional.sequence_length,
                        induction_gap=provisional.induction_gap,
                        key_positions=provisional.key_positions,
                        key_tokens=provisional.key_tokens,
                        value_tokens=provisional.value_tokens,
                        final_key_position=provisional.final_key_position,
                        target_value_token=provisional.target_value_token,
                        distractor_value_tokens=provisional.distractor_value_tokens,
                    )
                )

    token_banks = tuple(
        TokenBank(
            bank=bank,
            token_ids=tuple(sorted(bank_tokens[bank])),
            token_count=len(bank_tokens[bank]),
            token_sha256=json_sha256(tuple(sorted(bank_tokens[bank]))),
        )
        for bank in SEQUENCE_BANKS
    )
    provisional_design = SequenceDesign(
        seed=seed,
        bank_counts=counts,
        per_split_size=per_split_size,
        token_pool_size=len(token_pool),
        token_pool_sha256=json_sha256(token_pool),
        token_banks=token_banks,
        examples=tuple(examples),
        design_sha256="",
    )
    design = SequenceDesign(
        seed=provisional_design.seed,
        bank_counts=provisional_design.bank_counts,
        per_split_size=provisional_design.per_split_size,
        token_pool_size=provisional_design.token_pool_size,
        token_pool_sha256=provisional_design.token_pool_sha256,
        token_banks=provisional_design.token_banks,
        examples=provisional_design.examples,
        design_sha256=json_sha256(_sequence_design_payload(provisional_design)),
    )
    validate_sequence_design(design, regular_token_pool=token_pool)
    return design


def validate_sequence_design(
    design: SequenceDesign,
    *,
    regular_token_pool: Sequence[int] | None = None,
) -> None:
    """Reject token leakage, construction drift, and hash drift."""

    if design.schema_version != SEQUENCE_DESIGN_SCHEMA:
        raise ValueError("unexpected sequence design schema")
    _integer(design.seed, "seed")
    if not _is_sha256(design.token_pool_sha256):
        raise ValueError("invalid regular-token pool hash")
    if tuple(bank.bank for bank in design.token_banks) != SEQUENCE_BANKS:
        raise ValueError("token banks are missing or out of order")
    minimum_size = max(
        family.sequence_length - 1 for family in SEQUENCE_FAMILIES
    )
    if design.per_split_size < minimum_size:
        raise ValueError("per-split token count cannot support the longest family")

    family_by_id = {family.family_id: family for family in SEQUENCE_FAMILIES}
    seen_ids: set[str] = set()
    derived_bank_tokens: dict[str, set[int]] = {
        bank: set() for bank in SEQUENCE_BANKS
    }
    for bank in SEQUENCE_BANKS:
        expected_per_family = getattr(design.bank_counts, bank)
        for family in SEQUENCE_FAMILIES:
            examples = design.examples_for(bank, family)
            if len(examples) != expected_per_family:
                raise ValueError(
                    f"wrong number of {bank}/{family.family_id} examples"
                )
            if tuple(example.family_index for example in examples) != tuple(
                range(expected_per_family)
            ):
                raise ValueError("family indices are not contiguous")

    for example in design.examples:
        if example.bank not in SEQUENCE_BANKS:
            raise ValueError("sequence belongs to an unknown token bank")
        family = family_by_id.get(example.family_id)
        if family is None:
            raise ValueError("sequence belongs to an unknown length/gap family")
        if (
            example.sequence_length != family.sequence_length
            or example.induction_gap != family.induction_gap
        ):
            raise ValueError("sequence length/gap metadata changed")
        if len(example.tokens) != family.sequence_length:
            raise ValueError("sequence has the wrong length")
        final_position = family.sequence_length - 1
        target_position = final_position - family.induction_gap
        if example.final_key_position != final_position:
            raise ValueError("final repeated key is not at the final position")
        if example.key_positions[0] != target_position:
            raise ValueError("target key does not have the declared induction gap")
        occupied: set[int] = set()
        for position in example.key_positions:
            if position < 0 or position + 1 >= final_position:
                raise ValueError("key-value bigram lies outside the unique prefix")
            pair_positions = {position, position + 1}
            if occupied.intersection(pair_positions):
                raise ValueError("annotated key-value bigrams overlap")
            occupied.update(pair_positions)
        expected_keys = tuple(
            example.tokens[position] for position in example.key_positions
        )
        expected_values = tuple(
            example.tokens[position + 1] for position in example.key_positions
        )
        if example.key_tokens != expected_keys or example.value_tokens != expected_values:
            raise ValueError("key-value annotations do not match sequence tokens")
        if len(set(example.key_tokens + example.value_tokens)) != 6:
            raise ValueError("three key-value bigrams are not token-unique")
        if example.tokens[-1] != example.key_tokens[0]:
            raise ValueError("final token does not repeat the target key")
        if example.target_value_token != example.value_tokens[0]:
            raise ValueError("target is not the exact earlier successor")
        if example.distractor_value_tokens != example.value_tokens[1:]:
            raise ValueError("distractors are not the other two pair values")
        prefix = example.tokens[:-1]
        if len(set(prefix)) != len(prefix):
            raise ValueError("sequence prefix contains an accidental token collision")
        derived_bank_tokens[example.bank].update(prefix)
        expected_id = f"qseq_{json_sha256(_sequence_payload(example))[:16]}"
        if example.example_id != expected_id or example.example_id in seen_ids:
            raise ValueError("sequence example id is invalid or duplicated")
        seen_ids.add(example.example_id)

    allocated_bank_tokens: dict[str, set[int]] = {}
    for token_bank in design.token_banks:
        allocated = set(token_bank.token_ids)
        allocated_bank_tokens[token_bank.bank] = allocated
        if len(allocated) != len(token_bank.token_ids):
            raise ValueError("token-bank allocation contains duplicate ids")
        if tuple(sorted(token_bank.token_ids)) != token_bank.token_ids:
            raise ValueError("token-bank ids are not in canonical order")
        if token_bank.token_count != design.per_split_size:
            raise ValueError("token-bank allocation has the wrong size")
        if token_bank.token_count != len(allocated):
            raise ValueError("token-bank size changed")
        if token_bank.token_sha256 != json_sha256(token_bank.token_ids):
            raise ValueError("token-bank hash changed")
        if not _is_sha256(token_bank.token_sha256):
            raise ValueError("invalid token-bank hash")
        if not derived_bank_tokens[token_bank.bank].issubset(allocated):
            raise ValueError("fixture contains a token outside its sequence bank")
    bank_sets = tuple(
        allocated_bank_tokens[token_bank.bank] for token_bank in design.token_banks
    )
    for index, left in enumerate(bank_sets):
        for right in bank_sets[index + 1 :]:
            if left.intersection(right):
                raise ValueError("regular-token leakage between sequence banks")
    all_bank_tokens = set().union(*bank_sets)
    if len(all_bank_tokens) != _tokens_needed(design.per_split_size):
        raise ValueError("sequence token-bank accounting is inconsistent")
    if regular_token_pool is not None:
        token_pool = _normalise_token_pool(regular_token_pool)
        if len(token_pool) != design.token_pool_size:
            raise ValueError("regular-token pool size changed")
        if json_sha256(token_pool) != design.token_pool_sha256:
            raise ValueError("regular-token pool hash changed")
        if not all_bank_tokens.issubset(token_pool):
            raise ValueError("sequence contains a token outside the regular-token pool")
    if not _is_sha256(design.design_sha256):
        raise ValueError("invalid sequence design hash")
    if json_sha256(_sequence_design_payload(design)) != design.design_sha256:
        raise ValueError("sequence design hash does not match its contents")


@dataclass(frozen=True, slots=True)
class FrozenComponentPanel:
    """The exact ordered panel of eight heads fixed before measurement."""

    component_ids: tuple[str, ...]
    panel_sha256: str
    schema_version: str = PANEL_SCHEMA


@dataclass(frozen=True, slots=True)
class BinaryMask:
    """One immutable binary intervention over the eight-head panel."""

    mask_id: str
    bits: tuple[int, ...]
    cardinality: int

    def as_numpy(self) -> np.ndarray:
        array = np.asarray(self.bits, dtype=np.uint8)
        array.setflags(write=False)
        return array


@dataclass(frozen=True, slots=True)
class BudgetCalibration:
    """One nested prefix of the 128-mask calibration bank."""

    budget: int
    masks: tuple[BinaryMask, ...]


@dataclass(frozen=True, slots=True)
class ActionPool:
    """Eight held-out masks accompanied by the analytic no-op."""

    pool_id: str
    mask_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MaskDesign:
    """Exhaustive partition of the eight-head Boolean intervention cube."""

    panel: FrozenComponentPanel
    seed: int
    supported_budgets: tuple[int, ...]
    calibrations: tuple[BudgetCalibration, ...]
    heldout_masks: tuple[BinaryMask, ...]
    action_pools: tuple[ActionPool, ...]
    design_sha256: str
    schema_version: str = MASK_DESIGN_SCHEMA

    def calibration_for(self, budget: int) -> tuple[BinaryMask, ...]:
        for calibration in self.calibrations:
            if calibration.budget == budget:
                return calibration.masks
        supported = ", ".join(map(str, self.supported_budgets))
        raise ValueError(f"unsupported mask budget {budget}; choose {supported}")

    @property
    def no_op(self) -> BinaryMask:
        return self.calibrations[0].masks[0]


def freeze_component_panel(
    component_ids: Sequence[str],
) -> FrozenComponentPanel:
    """Freeze exactly eight component identities and their order."""

    components = tuple(component_ids)
    if len(components) != 8:
        raise ValueError("Qwen induction protocol requires exactly eight heads")
    if any(
        not isinstance(component, str)
        or not component
        or component.strip() != component
        for component in components
    ):
        raise ValueError("component ids must be non-empty, trimmed strings")
    if len(components) != len(set(components)):
        raise ValueError("component ids must be unique")
    payload = {"schema_version": PANEL_SCHEMA, "component_ids": components}
    return FrozenComponentPanel(
        component_ids=components,
        panel_sha256=json_sha256(payload),
    )


def _mask_id(panel: FrozenComponentPanel, bits: tuple[int, ...]) -> str:
    payload = {"panel_sha256": panel.panel_sha256, "bits": bits}
    return f"qmask_{json_sha256(payload)[:16]}"


def _binary_mask(
    panel: FrozenComponentPanel, bits: tuple[int, ...]
) -> BinaryMask:
    return BinaryMask(
        mask_id=_mask_id(panel, bits),
        bits=bits,
        cardinality=sum(bits),
    )


def _bits_from_integer(value: int) -> tuple[int, ...]:
    return tuple((value >> index) & 1 for index in range(8))


def _quadratic_mask_design(masks: Sequence[BinaryMask]) -> np.ndarray:
    linear = np.asarray([mask.bits for mask in masks], dtype=float)
    pairwise = np.column_stack(
        [
            linear[:, left] * linear[:, right]
            for left, right in combinations(range(8), 2)
        ]
    )
    return np.column_stack([np.ones(len(linear)), linear, pairwise])


def _mask_payload(mask: BinaryMask) -> dict[str, object]:
    return {
        "mask_id": mask.mask_id,
        "bits": mask.bits,
        "cardinality": mask.cardinality,
    }


def _mask_design_payload(design: MaskDesign) -> dict[str, object]:
    return {
        "schema_version": design.schema_version,
        "panel": {
            "schema_version": design.panel.schema_version,
            "component_ids": design.panel.component_ids,
            "panel_sha256": design.panel.panel_sha256,
        },
        "seed": design.seed,
        "supported_budgets": design.supported_budgets,
        "calibrations": [
            {
                "budget": calibration.budget,
                "masks": [_mask_payload(mask) for mask in calibration.masks],
            }
            for calibration in design.calibrations
        ],
        "heldout_masks": [_mask_payload(mask) for mask in design.heldout_masks],
        "action_pools": [
            {"pool_id": pool.pool_id, "mask_ids": pool.mask_ids}
            for pool in design.action_pools
        ],
    }


def build_mask_design(
    component_ids: Sequence[str] | FrozenComponentPanel,
    *,
    seed: int,
) -> MaskDesign:
    """Partition all 256 masks into 128 calibration and 128 held-out masks."""

    panel = (
        component_ids
        if isinstance(component_ids, FrozenComponentPanel)
        else freeze_component_panel(component_ids)
    )
    seed = _integer(seed, "seed")
    zero = (0,) * 8
    singleton_bits = tuple(
        tuple(1 if component == selected else 0 for component in range(8))
        for selected in range(8)
    )
    pair_bits = tuple(
        tuple(
            1 if component in selected else 0
            for component in range(8)
        )
        for selected in combinations(range(8), 2)
    )
    anchors = (zero,) + singleton_bits + pair_bits
    anchor_set = set(anchors)
    rng = np.random.default_rng(seed)
    random_bits = tuple(
        _bits_from_integer(int(encoded)) for encoded in rng.permutation(256)
    )
    remainder = tuple(bits for bits in random_bits if bits not in anchor_set)

    # The first three non-anchor rows span low, high, and full density. Add one
    # row at each remaining density before filling from the seeded order. This
    # keeps the registered 40-row prefix well-conditioned and guarantees that
    # the full calibration bank covers every Boolean-mask density.
    density_prefix = tuple(
        next(bits for bits in remainder if sum(bits) == cardinality)
        for cardinality in (8, 3, 7, 4, 5, 6)
    )
    density_prefix_set = set(density_prefix)
    ordered_non_anchors = density_prefix + tuple(
        bits for bits in remainder if bits not in density_prefix_set
    )
    calibration_bits = anchors + ordered_non_anchors[: 128 - len(anchors)]
    calibration_set = set(calibration_bits)
    heldout_bits = tuple(bits for bits in random_bits if bits not in calibration_set)

    calibration_bank = tuple(_binary_mask(panel, bits) for bits in calibration_bits)
    heldout_masks = tuple(_binary_mask(panel, bits) for bits in heldout_bits)
    calibrations = tuple(
        BudgetCalibration(budget=budget, masks=calibration_bank[:budget])
        for budget in SUPPORTED_MASK_BUDGETS
    )
    no_op_id = calibration_bank[0].mask_id
    action_pools: list[ActionPool] = []
    for start in range(0, 128, 8):
        chunk = heldout_masks[start : start + 8]
        mask_ids = (no_op_id,) + tuple(mask.mask_id for mask in chunk)
        pool_payload = {"panel_sha256": panel.panel_sha256, "mask_ids": mask_ids}
        action_pools.append(
            ActionPool(
                pool_id=f"qpool_{json_sha256(pool_payload)[:16]}",
                mask_ids=mask_ids,
            )
        )

    provisional = MaskDesign(
        panel=panel,
        seed=seed,
        supported_budgets=SUPPORTED_MASK_BUDGETS,
        calibrations=calibrations,
        heldout_masks=heldout_masks,
        action_pools=tuple(action_pools),
        design_sha256="",
    )
    design = MaskDesign(
        panel=provisional.panel,
        seed=provisional.seed,
        supported_budgets=provisional.supported_budgets,
        calibrations=provisional.calibrations,
        heldout_masks=provisional.heldout_masks,
        action_pools=provisional.action_pools,
        design_sha256=json_sha256(_mask_design_payload(provisional)),
    )
    validate_mask_design(design)
    return design


def validate_mask_design(design: MaskDesign) -> None:
    """Reject panel drift, partition leakage, missing no-ops, and hash drift."""

    if design.schema_version != MASK_DESIGN_SCHEMA:
        raise ValueError("unexpected mask design schema")
    if design.panel.schema_version != PANEL_SCHEMA:
        raise ValueError("unexpected component-panel schema")
    if len(design.panel.component_ids) != 8:
        raise ValueError("mask design does not contain exactly eight heads")
    panel_payload = {
        "schema_version": design.panel.schema_version,
        "component_ids": design.panel.component_ids,
    }
    if json_sha256(panel_payload) != design.panel.panel_sha256:
        raise ValueError("component-panel hash does not match its contents")
    if design.supported_budgets != SUPPORTED_MASK_BUDGETS:
        raise ValueError("mask design must support exactly 16/40/64/128 measurements")
    if tuple(item.budget for item in design.calibrations) != SUPPORTED_MASK_BUDGETS:
        raise ValueError("calibration budgets are missing or out of order")
    _integer(design.seed, "seed")

    def validate_mask(mask: BinaryMask) -> None:
        if len(mask.bits) != 8 or set(mask.bits) - {0, 1}:
            raise ValueError("mask does not match the frozen binary head panel")
        if mask.cardinality != sum(mask.bits):
            raise ValueError("mask cardinality does not match its bits")
        if mask.mask_id != _mask_id(design.panel, mask.bits):
            raise ValueError("mask id does not match its panel and bits")

    full_calibration = design.calibrations[-1].masks
    if len(full_calibration) != 128 or len(design.heldout_masks) != 128:
        raise ValueError("mask cube must be split into two banks of 128")
    for calibration in design.calibrations:
        if len(calibration.masks) != calibration.budget:
            raise ValueError("calibration mask count does not match its budget")
        if calibration.masks != full_calibration[: calibration.budget]:
            raise ValueError("calibration budgets are not nested prefixes")
    for mask in full_calibration + design.heldout_masks:
        validate_mask(mask)

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
    calibration_bits = tuple(mask.bits for mask in full_calibration)
    heldout_bits = tuple(mask.bits for mask in design.heldout_masks)
    if calibration_bits[:37] != tuple(expected_anchors):
        raise ValueError(
            "calibration bank must begin with no-op, singleton, and pair anchors"
        )
    if tuple(sum(bits) for bits in calibration_bits[37:40]) != (8, 3, 7):
        raise ValueError("first 40 calibration rows lost their density anchors")
    if {sum(bits) for bits in calibration_bits} != set(range(9)):
        raise ValueError("calibration bank does not cover every mask density")
    quadratic_40 = _quadratic_mask_design(full_calibration[:40])
    quadratic_128 = _quadratic_mask_design(full_calibration)
    if np.linalg.matrix_rank(quadratic_40) != 37:
        raise ValueError("first 40 masks do not identify the quadratic mask basis")
    if np.linalg.cond(quadratic_40) >= 30.0:
        raise ValueError("first 40 masks have poor quadratic conditioning")
    if np.linalg.cond(quadratic_128) >= 25.0:
        raise ValueError("full calibration bank has poor quadratic conditioning")
    if len(set(calibration_bits)) != 128 or len(set(heldout_bits)) != 128:
        raise ValueError("calibration or held-out bank contains duplicate masks")
    if set(calibration_bits).intersection(heldout_bits):
        raise ValueError("held-out mask leaked into the calibration bank")
    if set(calibration_bits).union(heldout_bits) != {
        _bits_from_integer(encoded) for encoded in range(256)
    }:
        raise ValueError("calibration and held-out banks do not exhaust the mask cube")

    no_op_id = design.no_op.mask_id
    heldout_ids = tuple(mask.mask_id for mask in design.heldout_masks)
    if len(design.action_pools) != 16:
        raise ValueError("held-out bank must form exactly sixteen action pools")
    pooled_ids: list[str] = []
    for pool in design.action_pools:
        if len(pool.mask_ids) != 9 or pool.mask_ids[0] != no_op_id:
            raise ValueError("each action pool needs no-op plus eight held-out masks")
        if len(set(pool.mask_ids)) != 9:
            raise ValueError("action pool contains duplicate masks")
        if not set(pool.mask_ids[1:]).issubset(heldout_ids):
            raise ValueError("action pool contains a non-held-out intervention")
        pool_payload = {
            "panel_sha256": design.panel.panel_sha256,
            "mask_ids": pool.mask_ids,
        }
        expected_pool_id = f"qpool_{json_sha256(pool_payload)[:16]}"
        if pool.pool_id != expected_pool_id:
            raise ValueError("action-pool id does not match its contents")
        pooled_ids.extend(pool.mask_ids[1:])
    if tuple(pooled_ids) != heldout_ids:
        raise ValueError("action pools do not partition held-out masks in order")
    if not _is_sha256(design.design_sha256):
        raise ValueError("invalid mask design hash")
    if json_sha256(_mask_design_payload(design)) != design.design_sha256:
        raise ValueError("mask design hash does not match its contents")
