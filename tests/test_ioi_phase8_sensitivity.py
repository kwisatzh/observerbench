"""Tests for the sealed Phase-8 IOI sensitivity package.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from observerbench.tasks.ioi.phase8_evaluation import secondary_contrasts
from observerbench.tasks.ioi.phase8_sensitivity import (
    DIRECT_RISK,
    EXACT_NOOP,
    NATURAL_MEAN,
    TRANSFORMED_MEAN,
    Phase8Paths,
    compute_phase8_freeze_tables,
    load_phase8_fit_inputs,
    load_phase8_protocol,
    validate_phase8_freeze,
    verify_phase8_protocol_sources,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results/revision/phase05"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"
PHASE8 = ROOT / "results/revision/phase08/ioi_target_sensitivity"
PROTOCOL = ROOT / "configs/revision/ioi_phase08_target_sensitivity_v1.json"


def _paths() -> Phase8Paths:
    return Phase8Paths(
        phase7_protocol=ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json",
        phase7_design=PHASE7 / "design",
        phase7_pretest=PHASE7 / "clean_pretest",
        phase7_freeze=PHASE7 / "prediction_freeze",
        phase7_audit=PHASE7 / "preoutcome_audit",
        phase7_measurement=PHASE7 / "selected_measurement",
        phase5_design=PHASE5 / "design",
        phase5_effects=PHASE5 / "ioi_effects",
    )


def test_phase8_protocol_and_inherited_hashes_are_fixed() -> None:
    protocol = load_phase8_protocol(PROTOCOL)
    verify_phase8_protocol_sources(protocol, ROOT)
    assert protocol["targets"] == [0.5, 1.0, 1.5]
    assert protocol["claim_boundary"].startswith("All results are labeled post-confirmatory")


def test_phase8_refit_reproduces_frozen_counts_and_phase7_t1_actions() -> None:
    raw_phase7_shard = (
        PHASE7
        / "selected_measurement/shards/test/effects_0000_0016.csv"
    )
    if raw_phase7_shard.is_file():
        _design, calibration, candidates, inherited, train = load_phase8_fit_inputs(
            _paths()
        )
        tables = compute_phase8_freeze_tables(
            calibration, candidates, inherited, train
        )
        predictions = tables["candidate_predictions.csv"]
        actions = tables["fixed_actions.csv"]
        all_selected = tables["all_selected_masks.csv"]
        reused = tables["reused_phase7_masks.csv"]
        new = tables["new_measurement_masks.csv"]
    else:
        # The public repository carries the frozen fit outputs but not the raw
        # prompt-level measurements.  Exercise the same count and action checks
        # against those hash-sealed public tables.
        freeze = PHASE8 / "prediction_freeze"
        predictions = pd.read_csv(freeze / "candidate_predictions.csv")
        actions = pd.read_csv(freeze / "fixed_actions.csv")
        all_selected = pd.read_csv(freeze / "all_selected_masks.csv")
        reused = pd.read_csv(freeze / "reused_phase7_masks.csv")
        new = pd.read_csv(freeze / "new_measurement_masks.csv")
    assert len(predictions) == 3 * 3 * 48 * 31
    assert len(actions) == 4 * 3 * 48
    assert len(all_selected) == 237
    assert len(reused) == 89
    assert len(new) == 148

    old = pd.read_csv(PHASE7 / "prediction_freeze/fixed_actions.csv")
    old = old.loc[old["selector"].isin([DIRECT_RISK, NATURAL_MEAN])]
    current = actions.loc[
        np.isclose(actions["target"].to_numpy(float), 1.0)
        & actions["selector"].isin([DIRECT_RISK, NATURAL_MEAN])
    ]
    columns = ["selector", "pool_id", "selected_mask_id"]
    assert old[columns].sort_values(columns).reset_index(drop=True).equals(
        current[columns].sort_values(columns).reset_index(drop=True)
    )

    direct_half = actions.loc[
        np.isclose(actions["target"].to_numpy(float), 0.5)
        & (actions["selector"] == DIRECT_RISK)
    ]
    transformed_one = actions.loc[
        np.isclose(actions["target"].to_numpy(float), 1.0)
        & (actions["selector"] == TRANSFORMED_MEAN)
    ]
    assert int(direct_half["selected_is_noop"].sum()) == 12
    assert int((transformed_one["predicted_target_loss"] < 0).sum()) == 44


def test_checked_phase8_freeze_is_preoutcome_and_complete() -> None:
    manifest = validate_phase8_freeze(
        PHASE8 / "prediction_freeze", protocol_path=PROTOCOL
    )
    assert manifest["new_candidate_outcomes_loaded"] is False
    assert manifest["phase7_outcome_values_loaded_during_fit"] is False
    assert manifest["counts"]["new_selected_masks_to_measure"] == 148
    assert manifest["counts"]["new_effect_cells_to_measure"] == 75_776


def test_secondary_bootstrap_reports_every_target_and_fixed_target_aggregate() -> None:
    selectors = (DIRECT_RISK, NATURAL_MEAN, TRANSFORMED_MEAN, EXACT_NOOP)
    targets = (0.5, 1.0, 1.5)
    offsets = {
        DIRECT_RISK: 0.0,
        NATURAL_MEAN: 0.10,
        TRANSFORMED_MEAN: 0.05,
        EXACT_NOOP: 0.20,
    }
    rows = []
    for target in targets:
        for selector in selectors:
            for pair in range(32):
                for pool in range(48):
                    for prompt in range(16):
                        rows.append(
                            {
                                "target": target,
                                "selector": selector,
                                "unordered_name_pair_id": f"pair_{pair:02d}",
                                "pool_id": f"pool_{pool:02d}",
                                "prompt_id": f"p_{pair:02d}_{prompt:02d}",
                                "actual_target_loss": target + offsets[selector],
                            }
                        )
    outcomes = pd.DataFrame(rows)
    protocol = {
        "targets": list(targets),
        "bootstrap": {"repeats": 5, "seed": 1},
    }
    contrasts, cells, signs = secondary_contrasts(outcomes, protocol=protocol)
    assert len(contrasts) == 3 * 4
    assert set(contrasts["target_scope"]) == {
        "target_0.5",
        "target_1",
        "target_1.5",
        "all_three_equal_weight",
    }
    assert contrasts["secondary_no_success_gate"].all()
    assert np.allclose(
        contrasts.loc[contrasts["reference"] == NATURAL_MEAN, "absolute_loss_reduction"],
        0.10,
    )
    assert len(cells) == 3 * 3 * 32 * 48
    assert len(signs) == 3 * 4
