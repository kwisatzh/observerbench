"""Focused checks for composing published-method rows into the APPS panel.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from observerbench.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_ai_control_leaderboard",
    ROOT / "scripts" / "build_ai_control_leaderboard.py",
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_builder_adds_checked_sae_rows_with_task_relative_metadata(tmp_path: Path) -> None:
    base = ROOT / "results/revision/ai_control/controlarena_apps_qwen_followup_v1"
    published = ROOT / "results/revision/ai_control/controlarena_apps_qwen_sae_probe_v1"
    outdir = tmp_path / "leaderboard"
    bundled = tmp_path / "bundled.json"

    assert (
        BUILDER.main(
            [
                "--results",
                str(base / "results.json"),
                "--bootstrap",
                str(base / "bootstrap.json"),
                "--published-method-results",
                str(published / "results.json"),
                "--published-method-bootstrap",
                str(published / "paired_problem_bootstrap.json"),
                "--outdir",
                str(outdir),
                "--bundled-json",
                str(bundled),
            ]
        )
        == 0
    )

    payload = json.loads(bundled.read_text(encoding="utf-8"))
    rows = {row["observer_name"]: row for row in payload["rows"]}
    assert len(rows) == 11
    assert set(BUILDER.PUBLISHED_METHOD_DISPLAY_NAMES) <= set(rows)

    sparse = rows["qwen-neutral-sae-sparse-probe"]
    raw = rows["qwen-neutral-residual-logistic"]
    assert sparse["result_status"] == "post-outcome published-method integration"
    assert sparse["requires_white_box_access"] is True
    assert "pinned public SAE encoder" in sparse["access_regime"]
    assert "Kantamneni et al." in sparse["fit_procedure"]
    assert sparse["metadata"]["selected_candidate_features"] == 512
    assert sparse["metadata"]["nonzero_coefficients"] == 127
    assert sparse["metrics"]["risk_auroc"] == pytest.approx(0.9531791970)
    assert raw["metrics"]["risk_auroc"] == pytest.approx(0.9830598191)
    assert sparse["metrics"]["mean_realized_violations_at_p01_b02"] == pytest.approx(4.667)
    assert raw["metrics"]["mean_realized_violations_at_p01_b02"] == pytest.approx(2.048)
    expected_violations = {"0.02": 715.0, "0.05": 670.0, "0.1": 596.0}
    for budget, expected in expected_violations.items():
        assert sparse["metrics"][f"violations_at_{budget}"] == expected
        assert sparse["metrics"][f"audit_precision_at_{budget}"] == 1.0
    assert sparse["bootstrap"]["risk_auroc"]["mean"] == pytest.approx(0.9531497545)
    assert sparse["metadata"]["rare_prevalence_diagnostic"][
        "mean_realized_violations"
    ] == pytest.approx(4.667)
    assert raw["metadata"]["rare_prevalence_diagnostic"][
        "mean_realized_violations"
    ] == pytest.approx(2.048)
    assert sparse["source_scores_sha256"] == file_sha256(
        published / "submissions/qwen-neutral-sae-sparse-probe/predictions.csv"
    )
    readme = (outdir / "README.md").read_text(encoding="utf-8")
    assert "informative negative" in readme
    assert "not a general claim about SAEs" in readme
    assert "Mean violations (1% attacks, 2% audit)" in readme


def test_published_method_inputs_must_be_paired(tmp_path: Path) -> None:
    base = ROOT / "results/revision/ai_control/controlarena_apps_qwen_followup_v1"
    with pytest.raises(SystemExit):
        BUILDER.main(
            [
                "--results",
                str(base / "results.json"),
                "--bootstrap",
                str(base / "bootstrap.json"),
                "--published-method-results",
                str(
                    ROOT
                    / "results/revision/ai_control/controlarena_apps_qwen_sae_probe_v1/results.json"
                ),
                "--outdir",
                str(tmp_path / "out"),
                "--bundled-json",
                str(tmp_path / "bundle.json"),
            ]
        )
