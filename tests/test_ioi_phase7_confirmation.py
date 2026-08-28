"""Tests for the outcome-sealed Phase-7 IOI confirmation.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observerbench.provenance import file_sha256
from observerbench.tasks.ioi.phase7_confirmation import (
    DIRECT_RISK,
    EXACT_NOOP,
    NATURAL_MEAN,
    load_phase7_protocol,
    load_verified_phase7_design,
    validate_phase7_freeze,
)
from observerbench.tasks.ioi.phase7_evaluation import (
    apply_joint_primary_gate,
    paired_cluster_pool_contrasts,
    validate_phase7_measured_union,
)
from observerbench.tasks.ioi.phase7_freeze_audit import (
    independently_recompute_phase7_freeze,
    validate_phase7_preoutcome_audit,
)
from observerbench.tasks.ioi.phase7_measurement import load_phase7_measurement_inputs
from observerbench.tasks.ioi.phase7_pretest import clean_pretest_validity


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results/revision/phase05"
PHASE7_V1 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation"
PHASE7 = ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"
PROTOCOL = ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v2.json"


def test_v2_preserves_and_discloses_the_aborted_v1_attempt() -> None:
    protocol = load_phase7_protocol(PROTOCOL)
    repair = protocol["v1_aborted_attempt"]
    assert "only screened target" in protocol["target_choice"]["rule"]
    assert "largest pilot margin" not in protocol["target_choice"]["rule"]
    assert repair["partial_effect_cells"] == 8192
    assert repair["partial_selected_masks"] == 16
    assert repair["candidate_outcome_values_inspected"] is False
    assert repair["evaluation_run"] is False
    for relative, expected in repair["artifact_hashes"].items():
        path = ROOT / relative
        assert len(expected) == 64
        # Raw partial measurements are kept in the separately archived local
        # bundle, not the compact public repository.  Verify them whenever that
        # bundle is present while still checking the public disclosure itself.
        if path.is_file():
            assert file_sha256(path) == expected
    assert file_sha256(
        ROOT / "configs/revision/ioi_phase07_canonical_noop_confirmation_v1.json"
    ) == repair["v1_protocol_sha256"]
    # The v2 attempt has now passed its pre-outcome audit and completed the
    # separately sealed measurement and evaluation stages.  Their presence
    # must not overwrite or remove the disclosed v1 partial shard above.
    assert (PHASE7 / "selected_measurement/measurement_manifest.json").is_file()
    assert (PHASE7 / "evaluation/evaluation_manifest.json").is_file()


def test_frozen_phase7_design_has_new_prompts_and_explicit_noop() -> None:
    _manifest, protocol, prompts, calibration, actions = load_verified_phase7_design(
        PHASE7 / "design", PROTOCOL
    )
    assert len(prompts) == 512
    assert prompts["prompt"].is_unique
    assert prompts["unordered_name_pair_id"].nunique() == 32
    assert len(calibration) == 160
    assert len(actions) == 48 * 31
    assert actions.groupby("pool_id").size().eq(31).all()
    assert actions.groupby("pool_id")["is_noop"].sum().eq(1).all()
    assert int(protocol["target"]) == 1


def test_clean_pretest_gate_passes_frozen_scores_and_stops_on_bad_template() -> None:
    protocol = load_phase7_protocol(PROTOCOL)
    clean = pd.read_csv(PHASE7 / "clean_pretest/clean_scores_test.csv")
    table, gate = clean_pretest_validity(clean, protocol=protocol)
    assert gate["passed"]
    assert float(table.loc[table.scope == "overall", "io_vs_subject_pairwise_accuracy"].iloc[0]) >= 0.95

    broken = clean.copy()
    template = str(broken["template_id"].iloc[0])
    broken.loc[broken["template_id"] == template, "clean_ld"] = -1.0
    _table, failed = clean_pretest_validity(broken, protocol=protocol)
    assert not failed["passed"]
    assert not failed["checks"]["every_template_accuracy"]
    assert not failed["checks"]["every_template_positive_mean_clean_ld"]


def test_phase7_freeze_and_selected_only_inputs_validate_before_outcomes() -> None:
    manifest = validate_phase7_freeze(
        PHASE7 / "prediction_freeze",
        design_dir=PHASE7 / "design",
        pretest_dir=PHASE7 / "clean_pretest",
        phase5_design_dir=PHASE5 / "design",
        phase5_effects_dir=PHASE5 / "ioi_effects",
        protocol_path=PROTOCOL,
    )
    assert manifest["phase7_candidate_outcomes_loaded"] is False
    assert manifest["counts"]["selected_unique_nonnoop_masks"] == 89
    template_means = PHASE5 / "ioi_effects/template_head_means.npz"
    if not template_means.is_file():
        audit = json.loads(
            (PHASE7 / "preoutcome_audit/preoutcome_audit.json").read_text(
                encoding="utf-8"
            )
        )
        assert audit["all_checks_pass"]
        assert audit["counts"]["phase5_train_calibration_shards_read"] == 10
        assert audit["counts"]["phase7_candidate_outcome_rows_read"] == 0
        prompts = pd.read_csv(PHASE7 / "design/prompts.csv")
        selected = pd.read_csv(
            PHASE7 / "prediction_freeze/selected_measurement_masks.csv"
        )
        clean = pd.read_csv(PHASE7 / "clean_pretest/clean_scores_test.csv")
        assert len(prompts) == len(clean) == 512
        assert len(selected) == 89
        assert not selected["is_noop"].astype(bool).any()
        return
    audit = validate_phase7_preoutcome_audit(
        PHASE7 / "preoutcome_audit",
        design_dir=PHASE7 / "design",
        pretest_dir=PHASE7 / "clean_pretest",
        freeze_dir=PHASE7 / "prediction_freeze",
        phase5_effects_dir=PHASE5 / "ioi_effects",
        protocol_path=PROTOCOL,
    )
    assert audit["all_checks_pass"]
    assert audit["counts"]["phase5_train_calibration_shards_read"] == 10
    assert audit["counts"]["phase7_candidate_outcome_rows_read"] == 0
    freeze, _design, prompts, selected, clean, means, templates = (
        load_phase7_measurement_inputs(
            PHASE7 / "design",
            PHASE7 / "clean_pretest",
            PHASE7 / "prediction_freeze",
            PHASE7 / "preoutcome_audit",
            PHASE5 / "design",
            PHASE5 / "ioi_effects",
            protocol_path=PROTOCOL,
        )
    )
    assert freeze["phase7_candidate_mask_forward_passes"] == 0
    assert len(prompts) == len(clean) == 512
    assert len(selected) == 89
    assert not selected["is_noop"].any()
    assert means.shape == (8, 13, 64)
    assert len(templates) == 8


def test_independent_recomputation_rejects_a_tampered_coefficient(tmp_path: Path) -> None:
    if not (PHASE5 / "ioi_effects/template_head_means.npz").is_file():
        pytest.skip(
            "requires the separately archived Phase-5 raw measurement bundle"
        )
    source = PHASE7 / "prediction_freeze"
    target = tmp_path / "prediction_freeze"
    target.mkdir()
    for path in source.iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    coefficients = pd.read_csv(target / "observer_coefficients.csv")
    coefficients.loc[0, "coefficient"] += 0.01
    coefficients.to_csv(target / "observer_coefficients.csv", index=False)
    with pytest.raises(ValueError, match="artifact changed|coefficients"):
        independently_recompute_phase7_freeze(
            PHASE7 / "design",
            PHASE7 / "clean_pretest",
            target,
            PHASE5 / "design",
            PHASE5 / "ioi_effects",
            protocol_path=PROTOCOL,
        )


def test_preoutcome_audit_rejects_source_hash_drift(tmp_path: Path) -> None:
    source = PHASE7 / "preoutcome_audit/preoutcome_audit.json"
    audit = json.loads(source.read_text(encoding="utf-8"))
    label = next(iter(audit["source_code_hashes"]))
    audit["source_code_hashes"][label] = "0" * 64
    target = tmp_path / "audit"
    target.mkdir()
    (target / "preoutcome_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="audited source code changed"):
        validate_phase7_preoutcome_audit(
            target,
            design_dir=PHASE7 / "design",
            pretest_dir=PHASE7 / "clean_pretest",
            freeze_dir=PHASE7 / "prediction_freeze",
            phase5_effects_dir=PHASE5 / "ioi_effects",
            protocol_path=PROTOCOL,
        )


def test_measured_union_rejects_count_preserving_mask_substitution() -> None:
    prompts = pd.DataFrame({"prompt_id": ["p0", "p1"]})
    selected = pd.DataFrame(
        {
            "mask_id": ["m0", "m1"],
            "mask_bits": ["0000000000001", "0000000000010"],
            "pool_id": ["pool0", "pool1"],
        }
    )
    clean = pd.DataFrame({"prompt_id": ["p0", "p1"], "clean_ld": [2.0, 3.0]})
    effects = pd.DataFrame(
        [
            {
                "prompt_id": prompt,
                "mask_id": mask,
                "mask_bits": bits,
                "pool_id": pool,
                "split": "test",
                "bank": "candidate",
                "clean_ld": clean_ld,
                "ablated_ld": clean_ld - 0.5,
                "drop_from_clean": 0.5,
            }
            for prompt, clean_ld in (("p0", 2.0), ("p1", 3.0))
            for mask, bits, pool in (
                ("m0", "0000000000001", "pool0"),
                ("m1", "0000000000010", "pool1"),
            )
        ]
    )
    validate_phase7_measured_union(
        effects, prompts=prompts, selected=selected, clean=clean
    )
    substituted = effects.copy()
    substituted.loc[substituted["mask_id"] == "m1", "mask_id"] = "unselected"
    with pytest.raises(ValueError):
        validate_phase7_measured_union(
            substituted, prompts=prompts, selected=selected, clean=clean
        )


def _synthetic_outcomes() -> pd.DataFrame:
    rows = []
    selectors = {
        DIRECT_RISK: 0.70,
        NATURAL_MEAN: 0.90,
        EXACT_NOOP: 1.00,
    }
    for selector, loss in selectors.items():
        for pair in range(32):
            for pool in range(48):
                for prompt in range(16):
                    rows.append(
                        {
                            "selector": selector,
                            "unordered_name_pair_id": f"pair_{pair:02d}",
                            "pool_id": f"pool_{pool:02d}",
                            "prompt_id": f"prompt_{pair:02d}_{prompt:02d}",
                            "actual_target_loss": loss,
                        }
                    )
    return pd.DataFrame(rows)


def test_joint_primary_requires_mean_and_noop_comparisons() -> None:
    protocol = load_phase7_protocol(PROTOCOL)
    contrasts, cells, signs = paired_cluster_pool_contrasts(
        _synthetic_outcomes(), protocol=protocol
    )
    assert len(cells) == 2 * 32 * 48
    assert (signs["positive_pool_count"] == 48).all()
    audit = apply_joint_primary_gate(contrasts, protocol=protocol)
    assert audit["joint_primary_passed"]
    by_id = contrasts.set_index("comparison_id")
    assert np.isclose(by_id.loc["H1a_estimand", "absolute_loss_reduction"], 0.20)
    assert np.isclose(by_id.loc["H1b_intervention_value", "absolute_loss_reduction"], 0.30)

    failed = contrasts.copy()
    failed.loc[failed["comparison_id"] == "H1b_intervention_value", "q025"] = -0.01
    audit = apply_joint_primary_gate(failed, protocol=protocol)
    assert not audit["joint_primary_passed"]
    assert audit["comparisons"]["H1a_estimand"]["passed"]
    assert not audit["comparisons"]["H1b_intervention_value"]["passed"]
