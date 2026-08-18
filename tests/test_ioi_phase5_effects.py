"""Tests for Phase-5 IOI effect schemas and guards.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import pandas as pd
import pytest

from observerbench.tasks.ioi.phase5_effects import (
    GPT2_SMALL_REVISION,
    IOIPhase5EffectConfig,
    validate_effect_rows,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prompt_id": prompt,
                "mask_id": mask,
                "mask_bits": bits,
                "clean_ld": 2.0,
                "ablated_ld": score,
                "drop_from_clean": 2.0 - score,
            }
            for prompt in ("p0", "p1")
            for mask, bits, score in (("m0", "0" * 13, 2.0), ("m1", "1" + "0" * 12, 1.5))
        ]
    )


def test_effect_rows_require_complete_cartesian_product_and_sign() -> None:
    validate_effect_rows(_rows(), prompt_ids=("p0", "p1"), mask_ids=("m0", "m1"))

    missing = _rows().iloc[:-1]
    with pytest.raises(ValueError, match="expected 4"):
        validate_effect_rows(missing, prompt_ids=("p0", "p1"), mask_ids=("m0", "m1"))

    wrong_sign = _rows()
    wrong_sign.loc[1, "drop_from_clean"] *= -1
    with pytest.raises(ValueError, match="inconsistent sign"):
        validate_effect_rows(wrong_sign, prompt_ids=("p0", "p1"), mask_ids=("m0", "m1"))


def test_effect_rows_reject_nonzero_clean_mask() -> None:
    rows = _rows()
    rows.loc[rows["mask_id"] == "m0", "ablated_ld"] = 1.9
    rows.loc[rows["mask_id"] == "m0", "drop_from_clean"] = 0.1
    with pytest.raises(ValueError, match="clean mask"):
        validate_effect_rows(rows, prompt_ids=("p0", "p1"), mask_ids=("m0", "m1"))


def test_effect_config_pins_the_frozen_gpt2_revision() -> None:
    assert IOIPhase5EffectConfig().model_revision == GPT2_SMALL_REVISION
    with pytest.raises(ValueError, match="model_revision"):
        IOIPhase5EffectConfig(model_revision="")
