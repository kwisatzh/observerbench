"""Tests for the Qwen Copy-v2 attribution-patching observers."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observerbench.effect_prediction import FiniteEffectMeasurement, FiniteEffectQuery
from observerbench.tasks.qwen_induction import (
    QWEN_CALIBRATED_ATP_BASELINE_NAME,
    QWEN_RAW_ATP_BASELINE_NAME,
    InductionMaskFeatures,
    QwenInductionAttributionMap,
    QwenInductionAttributionPatchingBaseline,
    load_qwen_reference_means,
    measure_qwen_induction_attribution_map,
    qwen_induction_attribution_patching_card,
)
from observerbench.tasks.qwen_induction.plant import (
    HeadAblationMeans,
    HeadRef,
    Qwen2InductionPlant,
)


HEADS = tuple(HeadRef(layer=0, head=index, kv_group=index // 2) for index in range(8))
LABELS = tuple(head.label for head in HEADS)


def _features(mask_id: str, bits: str) -> InductionMaskFeatures:
    mask = tuple(int(value) for value in bits)
    return InductionMaskFeatures(
        mask_id=mask_id,
        mask_bits=bits,
        head_mask=mask,
        head_labels=LABELS,
        head_layers=tuple(head.layer for head in HEADS),
        head_indices=tuple(head.head for head in HEADS),
        head_kv_groups=tuple(head.kv_group for head in HEADS),
        ablated_heads=tuple(
            label for label, included in zip(LABELS, mask) if included
        ),
        n_heads=sum(mask),
        candidate_pool="fixture",
    )


def _map(effects: tuple[float, ...] | None = None) -> QwenInductionAttributionMap:
    return QwenInductionAttributionMap(
        head_effects=effects or tuple(float(index) for index in range(1, 9)),
        head_labels=LABELS,
        n_prompts=256,
        prompt_split="train",
        model_revision="fixture-revision",
        intervention="fixture final-query family mean replacement",
    )


def test_raw_and_scalar_calibrated_qwen_predictions() -> None:
    first = _features("first", "10000000")
    first_two = _features("first-two", "11000000")
    last = _features("last", "00000001")
    measurements = (
        FiniteEffectMeasurement("first", first, 2.0),
        FiniteEffectMeasurement("first-two", first_two, 6.0),
    )
    queries = (FiniteEffectQuery("last", last),)

    raw = QwenInductionAttributionPatchingBaseline(_map(), calibrate_scalar=False)
    raw.fit(measurements)
    assert raw.name == QWEN_RAW_ATP_BASELINE_NAME
    assert raw.gain_ == 1.0
    assert raw.predict(queries) == pytest.approx((8.0,))

    calibrated = QwenInductionAttributionPatchingBaseline(
        _map(), calibrate_scalar=True
    )
    calibrated.fit(measurements)
    assert calibrated.name == QWEN_CALIBRATED_ATP_BASELINE_NAME
    assert calibrated.gain_ == pytest.approx(2.0)
    assert calibrated.predict(queries) == pytest.approx((16.0,))

    card = qwen_induction_attribution_patching_card(
        calibrated=True,
        attribution_map=_map(),
        fitted_gain=calibrated.gain_,
    )
    assert card.observer_name == QWEN_CALIBRATED_ATP_BASELINE_NAME
    assert card.metadata["result_status"] == "post-outcome published-method baseline"


def test_qwen_attribution_map_rejects_wrong_head_order() -> None:
    predictor = QwenInductionAttributionPatchingBaseline(
        _map(), calibrate_scalar=False
    )
    features = _features("first", "10000000")
    wrong = InductionMaskFeatures(
        **{
            **features.__dict__,
            "head_labels": tuple(reversed(features.head_labels)),
            "ablated_heads": (features.head_labels[-1],),
        }
    )
    measurement = FiniteEffectMeasurement("wrong", wrong, 1.0)
    predictor.fit((measurement,))
    with pytest.raises(ValueError, match="head order"):
        predictor.predict((FiniteEffectQuery("wrong", wrong),))


def test_qwen_reference_mean_loader_preserves_component_order(tmp_path: Path) -> None:
    path = tmp_path / "means.npz"
    values = np.arange(2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)
    np.savez_compressed(
        path,
        family_ids=np.asarray(("f0", "f1")),
        layers=np.asarray(tuple(head.layer for head in HEADS)),
        heads=np.asarray(tuple(head.head for head in HEADS)),
        kv_groups=np.asarray(tuple(head.kv_group for head in HEADS)),
        values=values,
        counts=np.asarray((3, 5)),
    )
    loaded = load_qwen_reference_means(path)
    assert loaded.heads == HEADS
    assert loaded.family_ids == ("f0", "f1")
    assert np.array_equal(loaded.values, values)


def test_tiny_qwen_attribution_has_registered_drop_sign() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=80,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).eval()
    plant = Qwen2InductionPlant(model, tokenizer=None, device="cpu")
    prompt = {
        "prompt_id": "p0",
        "family_id": "f0",
        "input_ids": (10, 11, 20, 21, 30, 31, 10),
        "query_position": 6,
        "target_token_id": 11,
        "distractor_token_id_1": 21,
        "distractor_token_id_2": 31,
        "key_positions": (0, 2, 4),
    }
    clean_means = plant.capture_reference_means((prompt,), HEADS, batch_size=1)
    direction = np.linspace(0.25, 1.0, clean_means.values.size).reshape(
        clean_means.values.shape
    )
    epsilon = 1e-3
    shifted = HeadAblationMeans(
        family_ids=clean_means.family_ids,
        heads=clean_means.heads,
        values=clean_means.values - epsilon * direction.astype(np.float32),
        counts=clean_means.counts,
    )
    attribution_map = measure_qwen_induction_attribution_map(
        plant,
        (prompt,),
        shifted,
        model_revision="tiny-fixture",
        batch_size=1,
    )
    finite = plant.measure_mask_effects(
        (prompt,),
        HEADS,
        ((1,) * 8,),
        shifted,
        batch_size=1,
    )[0].drop_from_clean
    predicted = sum(attribution_map.head_effects)
    assert np.sign(predicted) == np.sign(finite) or abs(finite) < 1e-8
    assert predicted == pytest.approx(finite, abs=2e-5, rel=0.08)
