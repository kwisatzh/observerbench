"""Focused tests for the checked Phase-6 IOI paper artifact.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from scripts.build_phase06_ioi_confirmation_artifact import build


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "results/revision/phase06/ioi_fresh_confirmation/evaluation"


def test_build_checks_frozen_values_and_writes_portable_artifacts(tmp_path: Path) -> None:
    outdir = tmp_path / "artifact"
    manifest = build(repo_root=ROOT, results_dir=EVALUATION, outdir=outdir)

    assert manifest["all_checks_pass"] is True
    assert manifest["contains_private_local_path"] is False
    assert manifest["claim_status"] == {
        "H1_primary_estimand": "pass",
        "H2_secondary_structure": "fail",
        "Jensen_parameter_count_sensitivity": "pass",
        "clean_task_validity": "fail",
    }
    assert manifest["headline_contrasts"][0][
        "relative_loss_reduction_fraction"
    ] == pytest.approx(0.22607799826902428)
    contrasts = {
        row["comparison_id"]: row for row in manifest["headline_contrasts"]
    }
    assert contrasts["H2_secondary_vs_additive"]["ci95_low"] == pytest.approx(
        -0.025486295092559876
    )
    assert contrasts["H2_secondary_vs_count"][
        "relative_loss_reduction_fraction"
    ] == pytest.approx(0.03540405154825488)
    assert contrasts["Jensen_parameter_count_sensitivity"][
        "ci95_low"
    ] == pytest.approx(0.11912255681357542)
    assert manifest["natural_mean_estimand_diagnostic"][
        "heldout_mean_effect_mae"
    ] == pytest.approx(0.12849992265111734)
    assert manifest["clean_task_gate"]["overall_accuracy"] == pytest.approx(0.8671875)
    assert manifest["clean_task_gate"]["worst_template"] == "p6_baba_museum"
    assert manifest["no_op_audit"]["mean_loss"] == pytest.approx(0.75)
    assert manifest["no_op_audit"]["direct_risk_mean_loss"] == pytest.approx(
        0.8325709196312042
    )
    assert manifest["selected_action_decomposition"]["natural_mean_effect"] == {
        "mean_to_target": pytest.approx(0.43326979352665755),
        "dispersion": pytest.approx(0.6425116389291361),
        "total_loss": pytest.approx(1.0757814324557937),
    }
    assert manifest["selected_action_decomposition"]["direct_risk"] == {
        "mean_to_target": pytest.approx(0.5499065573094413),
        "dispersion": pytest.approx(0.2826643623217630),
        "total_loss": pytest.approx(0.8325709196312042),
    }

    tex = (outdir / "ioi_confirmation_table.tex").read_text(encoding="utf-8")
    assert "H1 passes; H2 fails" in tex
    assert "clean-task gate fails" in tex
    assert "IOI-mechanism and fresh-template-generalization" in tex
    assert "No-action audit" in tex

    with (outdir / "ioi_confirmation_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert [row["status"] for row in rows[:4]] == ["pass", "fail", "fail", "pass"]

    persisted = json.loads(
        (outdir / "ioi_confirmation_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(persisted, sort_keys=True)
    assert "/Users/" not in serialized
    assert "observerbench-review-phase01" not in serialized


def test_build_rejects_any_drift_in_frozen_evaluation(tmp_path: Path) -> None:
    copied = tmp_path / "evaluation"
    shutil.copytree(EVALUATION, copied)
    audit_path = copied / "hypothesis_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["H1_primary_estimand"]["passed"] = False
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 for hypothesis_audit.json changed"):
        build(repo_root=ROOT, results_dir=copied, outdir=tmp_path / "artifact")
