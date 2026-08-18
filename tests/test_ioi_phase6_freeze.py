"""Tests for Phase-6 calibration access and prediction/action sealing.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observerbench.tasks.ioi.phase6_confirmatory import load_phase6_protocol
from observerbench.tasks.ioi.phase6_freeze import (
    DIRECT_RISK,
    JENSEN_SCORE,
    NATURAL_MEAN,
    QUADRATIC_MODEL,
    Phase6FreezeConfig,
    fit_phase6_observers,
    select_phase6_actions,
)
from observerbench.tasks.ioi.phase6_measurement import _load_sealed_design


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/revision/ioi_phase06_fresh_confirmation_v1.json"
DESIGN = ROOT / "results/revision/phase06/ioi_fresh_confirmation/design"


def _synthetic_train_effects(calibration: pd.DataFrame) -> pd.DataFrame:
    ordered = calibration.sort_values("measurement_order").reset_index(drop=True)
    masks = np.asarray(
        [[int(bit) for bit in bits] for bits in ordered["mask_bits"].astype(str)],
        dtype=float,
    )
    weights = np.linspace(0.03, 0.17, 13)
    mean = masks @ weights + 0.25 * masks[:, 0] * masks[:, 11]
    prompt_offsets = 0.18 * np.sin(np.arange(192) * 0.37)
    mask_index = np.repeat(np.arange(len(ordered)), 192)
    prompt_index = np.tile(np.arange(192), len(ordered))
    effect = mean[mask_index] + prompt_offsets[prompt_index]
    return pd.DataFrame(
        {
            "prompt_id": [f"train_{index:03d}" for index in prompt_index],
            "split": "train",
            "mask_id": ordered.iloc[mask_index]["mask_id"].to_numpy(str),
            "mask_bits": ordered.iloc[mask_index]["mask_bits"].to_numpy(str),
            "bank": "calibration",
            "pool_id": "",
            "drop_from_clean": effect,
        }
    )


def test_calibration_loader_returns_only_reference_train_and_calibration() -> None:
    reference, train, calibration, manifest = _load_sealed_design(DESIGN)
    assert set(reference["split"]) == {"reference"}
    assert set(train["split"]) == {"train"}
    assert len(reference) == 512
    assert len(train) == 192
    assert set(calibration["bank"]) == {"calibration"}
    assert set(calibration["pool_id"]) == {""}
    assert manifest["contains_model_outcomes"] is False


def test_calibration_loader_does_not_parse_candidate_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    import observerbench.tasks.ioi.phase6_measurement as measurement

    original = measurement.pd.read_csv
    parsed: list[str] = []

    def recording_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        parsed.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(measurement.pd, "read_csv", recording_read_csv)
    _load_sealed_design(DESIGN)
    assert "prompts.csv" in parsed
    assert "calibration_masks.csv" in parsed
    assert "candidate_masks.csv" not in parsed


def test_train_only_fit_freezes_every_model_and_action() -> None:
    protocol = load_phase6_protocol(CONFIG)
    calibration = pd.read_csv(
        DESIGN / "calibration_masks.csv", dtype={"mask_id": str, "mask_bits": str}
    )
    candidates = pd.read_csv(
        DESIGN / "candidate_masks.csv", dtype={"mask_id": str, "mask_bits": str}
    )
    effects = _synthetic_train_effects(calibration)
    config = Phase6FreezeConfig()
    predictions, coefficients, diagnostics = fit_phase6_observers(
        calibration,
        candidates,
        effects,
        protocol=protocol,
        config=config,
    )
    assert len(predictions) == 36_864
    assert set(predictions["selector_family"]) == {
        DIRECT_RISK,
        NATURAL_MEAN,
        JENSEN_SCORE,
    }
    assert set(predictions["target"]) == {0.5, 1.0, 1.5}
    assert not coefficients.empty
    quadratic = diagnostics[diagnostics["model"] == QUADRATIC_MODEL]
    assert set(quadratic["design_rank"]) == {92}
    assert set(quadratic["n_columns"]) == {92}

    actions = select_phase6_actions(predictions, candidates, config=config)
    assert len(actions) == 2_304
    assert set(actions["policy"]) == {"target_loss", "cost_aware"}
    assert actions.groupby(
        ["selector_family", "model", "target", "pool_id", "policy"]
    ).size().eq(1).all()


def test_train_fit_rejects_candidate_or_nontrain_effects() -> None:
    protocol = load_phase6_protocol(CONFIG)
    calibration = pd.read_csv(
        DESIGN / "calibration_masks.csv", dtype={"mask_id": str, "mask_bits": str}
    )
    candidates = pd.read_csv(
        DESIGN / "candidate_masks.csv", dtype={"mask_id": str, "mask_bits": str}
    )
    effects = _synthetic_train_effects(calibration)
    contaminated = effects.copy()
    contaminated.loc[0, "bank"] = "candidate"
    with pytest.raises(ValueError, match="candidate-mask effects"):
        fit_phase6_observers(
            calibration,
            candidates,
            contaminated,
            protocol=protocol,
            config=Phase6FreezeConfig(),
        )
    contaminated = effects.copy()
    contaminated.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="non-train"):
        fit_phase6_observers(
            calibration,
            candidates,
            contaminated,
            protocol=protocol,
            config=Phase6FreezeConfig(),
        )


def test_negative_predictions_are_not_clipped_and_ties_are_deterministic() -> None:
    candidates = pd.DataFrame(
        {
            "mask_id": [f"mask_{index:02d}" for index in range(32)],
            "pool_id": "pool_00",
            "n_heads": [4] * 32,
            "size_match_cell": "n_heads_04",
        }
    )
    predictions = pd.DataFrame(
        {
            "selector_family": DIRECT_RISK,
            "model": "additive_head",
            "target": 0.5,
            "mask_id": candidates["mask_id"],
            "predicted_target_loss": [0.2] * 32,
        }
    )
    predictions.loc[0, "predicted_target_loss"] = -0.4
    actions = select_phase6_actions(
        predictions,
        candidates,
        config=Phase6FreezeConfig(),
    )
    assert set(actions["selected_mask_id"]) == {"mask_00"}
    assert actions["negative_prediction_selected"].all()
    assert set(actions["predicted_target_loss"]) == {-0.4}


def test_frozen_protocol_keeps_h1_h2_and_jensen_boundaries() -> None:
    protocol = json.loads(CONFIG.read_text())
    assert "H1 alone defines primary success" in protocol["multiplicity_boundary"]
    assert protocol["jensen_score_sensitivity_models"] == [QUADRATIC_MODEL]
    assert "same 92-column quadratic basis" in protocol["hypotheses"][
        "H1_primary_estimand"
    ]["reference"]
