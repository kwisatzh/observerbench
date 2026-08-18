"""Tests for the Phase-5 nonlinear-suffix paper artifact.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_phase05_nonlinear_suffix_artifact import build


ROOT = Path(__file__).resolve().parents[1]


def test_build_validates_frozen_results_and_writes_paper_artifacts(tmp_path: Path) -> None:
    manifest = build(
        repo_root=ROOT,
        results_dir=ROOT / "results/revision/phase05/nonlinear_suffix_v3_support",
        protocol_path=ROOT / "configs/revision/ctl2_phase05_nonlinear_suffix_v2.json",
        outdir=tmp_path,
    )

    assert manifest["all_checks_pass"]
    assert not manifest["contains_private_local_path"]
    assert all(manifest["checks"].values())
    assert (tmp_path / "nonlinear_suffix_certificate_and_control.png").stat().st_size > 10_000
    assert (tmp_path / "nonlinear_suffix_certificate_and_control.pdf").stat().st_size > 10_000
    assert "\\gamma=1.15" in (tmp_path / "nonlinear_suffix_summary.tex").read_text()
    assert "do not identify an estimator-only advantage" in (
        tmp_path / "nonlinear_suffix_factorial.tex"
    ).read_text()

    summary = pd.read_csv(tmp_path / "nonlinear_suffix_summary.csv")
    assert len(summary) == 4
    gamma_on = summary[summary["comparison"].str.endswith("gamma=1.15")].iloc[0]
    gamma_off = summary[summary["comparison"].str.endswith("gamma=0")].iloc[0]
    assert gamma_on["ci95_low"] > 0.0
    assert gamma_off["ci95_low"] < 0.0 < gamma_off["ci95_high"]

    cells = pd.read_csv(tmp_path / "nonlinear_suffix_factorial_cells.csv")
    assert len(cells) == 8
    interaction = cells[cells["gamma"] == 1.15]
    crossed = interaction[
        interaction["estimator"] != interaction["direction_provider"]
    ]
    assert set(crossed["nonpositive_observer_gain_seed_count"]) == {10}
    assert set(crossed["unstable_observer_pole_seed_count"]) == {10}
    assert (crossed["would_clip_condition_count"] > 0).all()
    assert (
        crossed["displacement_exceeds_clean_pairwise_max_condition_count"] > 0
    ).all()

    contrasts = pd.read_csv(tmp_path / "nonlinear_suffix_factorial_contrasts.csv")
    compatibility = contrasts[
        (contrasts["gamma"] == 1.15)
        & (contrasts["contrast"] == "lifted_pair_compatibility_interaction")
    ].iloc[0]
    assert compatibility["ci95_low"] > 0.0

    analysis = json.loads(
        (
            ROOT
            / "results/revision/phase05/nonlinear_suffix_v3_support/analysis.json"
        ).read_text()
    )
    assert analysis["schema"] == "observerbench.nonlinear_suffix_analysis.v3"
    assert not analysis["natural_factorial_secondary"]["by_gamma"]["1.15"][
        "crossed_arms_pass_conditioning_and_clean_scale_diagnostics"
    ]
    assert analysis["natural_factorial_secondary"]["by_gamma"]["0.0"][
        "crossed_arms_pass_conditioning_and_clean_scale_diagnostics"
    ]
