"""Focused tests for the checked Phase-7 IOI paper artifact.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from scripts.build_phase07_ioi_confirmation_artifact import build


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT / "results/revision/phase07/ioi_canonical_noop_confirmation_v2"
)


def test_build_checks_frozen_values_and_writes_bounded_artifacts(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "artifact"
    manifest = build(repo_root=ROOT, result_root=RESULT_ROOT, outdir=outdir)

    assert manifest["all_checks_pass"] is True
    assert manifest["contains_private_local_path"] is False
    assert manifest["result_calibration"] == "good_positive_bounded"
    assert manifest["claim_status"] == {
        "joint_primary": "pass",
        "H1a_estimand": "pass",
        "H1b_target_tracking_vs_noop": "pass",
        "clean_task_validity": "pass",
    }
    assert manifest["measured_scope"] == {
        "selected_unique_nonnoop_masks": 89,
        "analytic_noop_measured": False,
    }
    assert manifest["chain_summary"]["effect_cells"] == 45_568
    assert manifest["clean_task_gate"]["overall_pairwise_accuracy"] == pytest.approx(
        0.9765625
    )
    assert manifest["clean_task_gate"][
        "worst_template_pairwise_accuracy"
    ] == pytest.approx(0.953125)

    contrasts = {
        row["comparison_id"]: row for row in manifest["headline_contrasts"]
    }
    assert contrasts["H1a_estimand"][
        "absolute_loss_reduction"
    ] == pytest.approx(0.2021368011677017)
    assert contrasts["H1a_estimand"]["ci95_low"] == pytest.approx(
        0.11496571685759895
    )
    assert contrasts["H1b_intervention_value"][
        "absolute_loss_reduction"
    ] == pytest.approx(0.10937722306698561)
    assert contrasts["H1b_intervention_value"]["ci95_low"] == pytest.approx(
        0.01246399262114816
    )
    assert manifest["template_directions"] == {
        "status": "descriptive",
        "H1a_estimand": "8/8",
        "H1b_target_tracking_vs_noop": "6/8",
    }

    decomposition = manifest["jensen_decomposition"]
    assert decomposition["direct_risk_head_pair_quadratic"] == {
        "mean_effect": pytest.approx(0.683923326122264),
        "total_loss": pytest.approx(0.8906227769330144),
        "mean_to_target": pytest.approx(0.3969548799796030),
        "dispersion": pytest.approx(0.4936678969534114),
    }
    assert decomposition["natural_mean_effect_head_pair_quadratic"] == {
        "mean_effect": pytest.approx(0.7708828156270707),
        "total_loss": pytest.approx(1.092759578100716),
        "mean_to_target": pytest.approx(0.2859615176372851),
        "dispersion": pytest.approx(0.8067980604634310),
    }
    assert decomposition["interpretation"] == {
        "status": "descriptive",
        "phase6_pattern_repeated": True,
        "direct_risk_accepts_larger_mean_to_target_term": True,
        "direct_risk_reduces_dispersion_more_than_the_mean_term_worsens": True,
    }

    tex = (outdir / "ioi_confirmation_table.tex").read_text(encoding="utf-8")
    assert "8/8" in tex
    assert "6/8" in tex
    assert "Intervals are absolute loss reductions" in tex
    assert "clean IO-versus-subject gate passed" in tex
    assert "uniform" not in tex.lower()

    with (outdir / "ioi_confirmation_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [row["status"] for row in rows] == ["pass", "pass"]

    persisted = json.loads(
        (outdir / "ioi_confirmation_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(persisted, sort_keys=True)
    assert "/Users/" not in serialized
    assert "observerbench-review-phase01" not in serialized


def test_build_rejects_any_drift_in_frozen_outcome(tmp_path: Path) -> None:
    copied = tmp_path / "ioi_canonical_noop_confirmation_v2"
    shutil.copytree(RESULT_ROOT, copied)
    audit_path = copied / "evaluation/hypothesis_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["joint_primary_passed"] = False
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 for hypothesis_audit changed"):
        build(repo_root=ROOT, result_root=copied, outdir=tmp_path / "artifact")
