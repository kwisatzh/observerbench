"""Focused tests for the checked Phase-8 IOI sensitivity paper artifact.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from scripts.build_phase08_ioi_sensitivity_artifact import build


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/revision/phase08/ioi_target_sensitivity"
CHECKED_ARTIFACT = ROOT / "paper/generated_phase08/ioi_sensitivity"


def test_build_checks_secondary_study_and_writes_bounded_artifacts(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "artifact"
    if (RESULT_ROOT / "evaluation_v2/fixed_action_prompt_losses.csv.gz").is_file():
        manifest = build(repo_root=ROOT, result_root=RESULT_ROOT, outdir=outdir)
        artifact_dir = outdir
    else:
        artifact_dir = CHECKED_ARTIFACT
        manifest = json.loads(
            (
                artifact_dir / "ioi_target_sensitivity_artifact_manifest.json"
            ).read_text(encoding="utf-8")
        )

    assert manifest["all_checks_pass"] is True
    assert manifest["contains_private_local_path"] is False
    assert manifest["result_calibration"] == (
        "good_positive_estimand_result_with_target_dependent_intervention_value"
    )
    assert manifest["claim_status"] == {
        "direct_vs_natural_mean_all_three_targets": (
            "positive_intervals_exclude_zero"
        ),
        "direct_vs_transformed_mean_all_three_targets": (
            "positive_intervals_exclude_zero"
        ),
        "direct_vs_noop_target_0.5": "negative_interval_excludes_zero",
        "direct_vs_noop_target_1.0": "positive_interval_excludes_zero",
        "direct_vs_noop_target_1.5": "positive_interval_excludes_zero",
        "equal_target_aggregate_vs_all_references": (
            "positive_intervals_exclude_zero"
        ),
        "success_gate": "none_post_confirmatory_secondary",
    }
    assert manifest["measurement_scope"] == {
        "selected_unique_nonnoop_masks": 237,
        "newly_measured_masks": 148,
        "hash_verified_reused_phase7_masks": 89,
        "noop_ablation_forward_passes": 0,
    }
    assert manifest["prediction_diagnostics"][
        "transformed_mean_selected_raw_negative_counts"
    ] == {
        "target_0.5": 44,
        "target_1.0": 44,
        "target_1.5": 16,
    }

    results = {row["target_scope"]: row for row in manifest["results"]}
    assert results["target_0.5"]["direct_risk_mean_loss"] == pytest.approx(
        0.731124513122874
    )
    assert results["target_0.5"]["natural_mean_relative_gain_fraction"] == (
        pytest.approx(0.2937500415927403)
    )
    assert results["target_0.5"]["transformed_mean_ci95_low"] == pytest.approx(
        0.1022460644482635
    )
    assert results["target_0.5"]["noop_relative_gain_fraction"] == pytest.approx(
        -0.462249026245748
    )
    assert results["target_0.5"]["noop_ci95_high"] == pytest.approx(
        -0.1614723555438104
    )
    assert results["target_1.5"]["noop_relative_gain_fraction"] == pytest.approx(
        0.2863755285507068
    )
    assert results["all_three_equal_weight"][
        "natural_mean_relative_gain_fraction"
    ] == pytest.approx(0.19065143951991775)

    tex = (artifact_dir / "ioi_target_sensitivity_table.tex").read_text(
        encoding="utf-8"
    )
    assert "Post-confirmatory target sensitivity" in tex
    assert "has no success gate" in tex
    assert "reference-minus-direct" in tex
    assert r"\textbf{$-$46.2\%" in tex
    assert "negative no-action result at $t=0.5$ is retained" in tex

    with (artifact_dir / "ioi_target_sensitivity_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[-1]["target_label"] == "Equal-target mean"

    persisted = json.loads(
        (artifact_dir / "ioi_target_sensitivity_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(persisted, sort_keys=True)
    assert "/Users/" not in serialized
    assert "observerbench-review-phase01" not in serialized


def test_build_rejects_drift_in_evaluated_contrasts(tmp_path: Path) -> None:
    copied = tmp_path / "ioi_target_sensitivity"
    shutil.copytree(RESULT_ROOT, copied)
    contrast_path = copied / "evaluation_v2/secondary_contrasts.csv"
    text = contrast_path.read_text(encoding="utf-8")
    contrast_path.unlink()
    contrast_path.write_text(
        text.replace("0.3040960973982389", "0.3040960973982388", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 for secondary_contrasts changed"):
        build(repo_root=ROOT, result_root=copied, outdir=tmp_path / "artifact")
