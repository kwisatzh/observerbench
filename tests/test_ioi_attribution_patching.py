# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observerbench.effect_prediction import FiniteEffectMeasurement, FiniteEffectQuery
from observerbench.tasks.ioi import (
    IOI_CALIBRATED_ATP_BASELINE_NAME,
    IOI_RAW_ATP_BASELINE_NAME,
    IOIAttributionMap,
    IOIAttributionPatchingBaseline,
    IOIMaskFeatures,
    ioi_attribution_patching_card,
    load_template_head_means,
)
from observerbench.tasks.ioi.heads import head_records


def _features(mask_id: str, bits: str) -> IOIMaskFeatures:
    records = head_records()
    mask = tuple(int(value) for value in bits)
    selected = tuple(
        str(record["label"])
        for record, included in zip(records, mask)
        if included
    )
    groups = tuple(str(record["group"]) for record in records)
    return IOIMaskFeatures(
        mask_id=mask_id,
        mask_bits=bits,
        head_mask=mask,
        selected_heads=selected,
        n_heads=sum(mask),
        n_name_movers=sum(
            included for included, group in zip(mask, groups) if group == "P"
        ),
        n_backup_name_movers=sum(
            included for included, group in zip(mask, groups) if group == "B"
        ),
        n_negative_name_movers=sum(
            included for included, group in zip(mask, groups) if group == "E"
        ),
    )


def _map() -> IOIAttributionMap:
    return IOIAttributionMap(
        head_effects=tuple(float(index) for index in range(1, 14)),
        n_prompts=192,
        prompt_split="train",
        model_revision="fixture-revision",
        intervention="fixture final-position mean replacement",
    )


def test_raw_and_scalar_calibrated_predictions() -> None:
    attribution_map = _map()
    single_first = _features("first", "1000000000000")
    first_two = _features("first-two", "1100000000000")
    last = _features("last", "0000000000001")
    measurements = (
        FiniteEffectMeasurement("first", single_first, 2.0),
        FiniteEffectMeasurement("first-two", first_two, 6.0),
    )
    queries = (FiniteEffectQuery("last", last),)

    raw = IOIAttributionPatchingBaseline(attribution_map, calibrate_scalar=False)
    raw.fit(measurements)
    assert raw.name == IOI_RAW_ATP_BASELINE_NAME
    assert raw.gain_ == 1.0
    assert raw.predict(queries) == pytest.approx((13.0,))

    calibrated = IOIAttributionPatchingBaseline(
        attribution_map,
        calibrate_scalar=True,
    )
    calibrated.fit(measurements)
    assert calibrated.name == IOI_CALIBRATED_ATP_BASELINE_NAME
    assert calibrated.gain_ == pytest.approx(2.0)
    assert calibrated.predict(queries) == pytest.approx((26.0,))

    card = ioi_attribution_patching_card(
        calibrated=True,
        attribution_map=attribution_map,
        fitted_gain=calibrated.gain_,
    )
    assert card.observer_name == IOI_CALIBRATED_ATP_BASELINE_NAME
    assert card.metadata["fitted_scalar_gain"] == pytest.approx(2.0)


def test_invalid_map_and_unidentifiable_gain() -> None:
    with pytest.raises(ValueError, match="exactly 13"):
        IOIAttributionMap(
            head_effects=(1.0,),
            n_prompts=1,
            prompt_split="train",
            model_revision="fixture",
            intervention="fixture",
        )
    zeros = IOIAttributionMap(
        head_effects=(0.0,) * 13,
        n_prompts=1,
        prompt_split="train",
        model_revision="fixture",
        intervention="fixture",
    )
    predictor = IOIAttributionPatchingBaseline(zeros, calibrate_scalar=True)
    measurement = FiniteEffectMeasurement(
        "first",
        _features("first", "1000000000000"),
        1.0,
    )
    with pytest.raises(ValueError, match="cannot identify"):
        predictor.fit((measurement,))


def test_template_head_mean_loader(tmp_path: Path) -> None:
    path = tmp_path / "means.npz"
    expected = np.arange(2 * 13 * 4, dtype=np.float32).reshape(2, 13, 4)
    np.savez_compressed(path, means=expected, templates=np.asarray(("a", "b")))
    means, templates = load_template_head_means(path)
    assert np.array_equal(means, expected)
    assert templates == ("a", "b")
