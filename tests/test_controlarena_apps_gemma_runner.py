"""Focused staging tests for the matched Gemma APPS runner.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_controlarena_apps_gemma.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("run_controlarena_apps_gemma_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _row(split: str, index: int):
    return RUNNER.MonitorRow(
        sample_id=f"p{index}::honest",
        problem_id=f"p{index}",
        split=split,
        eval_mode="honest",
        backdoor_works=False,
        family_id="introductory:any",
        question="question",
        code="code",
    )


def _view(lengths):
    values = np.asarray(lengths, dtype=np.int32)
    return RUNNER.EncodedMonitorView(
        activations=np.zeros((len(values), 1, 2), dtype=np.float16),
        lengths=values,
    )


def test_training_truncation_is_excluded_but_locked_truncation_stops() -> None:
    rows = [_row("fit", 0), _row("calibration", 1), _row("locked_test", 2)]
    analysis, excluded = RUNNER._analysis_rows(
        rows, (_view([8, 4, 4]), _view([4, 4, 4])), max_length=8
    )
    assert analysis[0].split == "excluded_truncation"
    assert analysis[1:] == rows[1:]
    np.testing.assert_array_equal(excluded, [True, False, False])

    with pytest.raises(ValueError, match="locked-test"):
        RUNNER._analysis_rows(rows, (_view([4, 4, 8]),), max_length=8)


def test_matched_parent_design_binds_exact_qwen_row_identities() -> None:
    config = json.loads(
        (ROOT / "configs/revision/ai_control/controlarena_apps_gemma2_9b_it_v0.json").read_text()
    )
    payload = json.loads((ROOT / config["matched_parent_design"]["path"]).read_text())
    rows = [
        RUNNER.MonitorRow(
            sample_id=str(row["sample_id"]),
            problem_id=str(row["problem_id"]),
            split=str(row["split"]),
            eval_mode=str(row["eval_mode"]),
            backdoor_works=bool(row["backdoor_works"]),
            family_id=str(row["family_id"]),
            question="not persisted in identity contract",
            code="not persisted in identity contract",
        )
        for row in payload["rows"]
    ]

    RUNNER._validate_matched_parent_design(config, rows)

    rows[0] = replace(rows[0], split="calibration")
    with pytest.raises(ValueError, match="differ"):
        RUNNER._validate_matched_parent_design(config, rows)


def test_sae_standardization_uses_fit_rows_and_drops_rare_features() -> None:
    rows = [_row("fit", 0), _row("fit", 1), _row("calibration", 2), _row("locked_test", 3)]
    features = np.asarray(
        [
            [1.0, 0.0, 4.0],
            [3.0, 1.0, 4.0],
            [5.0, 2.0, 8.0],
            [7.0, 3.0, 9.0],
        ],
        dtype=np.float32,
    )

    standardized, audit = RUNNER._standardize_sae_features(
        features, rows, minimum_fit_active_rows=2
    )

    assert standardized.shape == (4, 1)
    np.testing.assert_allclose(standardized[:2, 0], [-1.0, 1.0])
    assert audit == {
        "released_width": 3,
        "passed_activity_filter": 2,
        "retained_nonconstant": 1,
    }


def test_sae_smoke_rejects_an_all_zero_code() -> None:
    assert RUNNER._sae_smoke_audits_pass(
        [{"finite": True, "active_features": 7}]
    )
    assert not RUNNER._sae_smoke_audits_pass(
        [{"finite": True, "active_features": 0}]
    )


def test_prevalence_artifact_can_declare_prospective_gemma_status() -> None:
    output = RUNNER.prevalence_budget_sensitivity(
        {
            "monitor": {
                "honest": [0.1, 0.2],
                "attack": [0.8, 0.9],
            }
        },
        prevalences=[0.5],
        budgets=[0.5],
        population_size=4,
        replicates=2,
        seed=1,
        status="prospectively_frozen_before_gemma_monitor_outcomes",
        design_disclosure="inherited from Qwen before Gemma outcomes",
    )
    assert output["status"] == "prospectively_frozen_before_gemma_monitor_outcomes"
    assert output["design_disclosure"] == "inherited from Qwen before Gemma outcomes"


def test_full_stage_cannot_start_without_explicit_review_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="review-gated"):
        RUNNER.main(
            [
                "--config",
                str(tmp_path / "missing-config.json"),
                "--manifest",
                str(tmp_path / "missing-manifest.json"),
                "--outdir",
                str(tmp_path / "out"),
                "--stage",
                "full",
            ]
        )
