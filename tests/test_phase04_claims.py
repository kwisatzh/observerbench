"""Fast pins for the checked Phase-4 scientific claims.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_phase04_artifact import build


ROOT = Path(__file__).resolve().parents[1]


def test_checked_ctl2_factorial_null_and_gain_claims() -> None:
    audit = json.loads(
        (ROOT / "results/revision/phase04/ctl2_phase04_audit.json").read_text()
    )
    assert audit["all_required_checks_pass"] is True

    rows = audit["multiseed"]["fixed_direction_ise_effects"]
    nonzero = [
        row
        for row in rows
        if row["gamma"] > 0 and row["support_mode"] == "unprojected"
    ]
    assert len(nonzero) == 8
    assert all(row["n_seeds"] == 6 for row in nonzero)
    assert all(row["n_positive_fo_minus_lifted"] == 6 for row in nonzero)

    gamma_zero = [row for row in rows if row["gamma"] == 0]
    assert max(abs(row["mean"]) for row in gamma_zero) < 1e-5

    gain = audit["gain_match_small_setpoint"]
    assert gain["control_clip_fraction"] == 0
    assert gain["final_target_mse"] > gain["initial_target_mse"]
    assert 0 < 1 - gain["observer_error_pole_unsaturated"] < 2


def test_checked_ioi_capacity_ranks_and_model_comparisons() -> None:
    for stage in ("ioi_stage2b", "ioi_stage2c"):
        base = ROOT / f"results/revision/phase02/{stage}_capacity"
        capacity = pd.read_csv(base / "capacity_audit.csv").set_index("model")
        comparison = pd.read_csv(base / "model_comparison.csv").set_index("model")
        assert set(
            capacity.loc[
                [
                    "count_plus_PB_bin4",
                    "count_plus_PE_bin4",
                    "count_plus_BE_bin4",
                ],
                "rank_added_vs_count_additive",
            ]
        ) == {4}
        assert capacity.loc[
            "count_plus_all_bin4", "rank_added_vs_count_additive"
        ] == 12
        assert comparison.loc["count_plus_all_bin4", "mae"] < comparison.loc[
            "additive_head", "mae"
        ]
        assert comparison.loc["count_plus_all_bin4", "mae"] < comparison.loc[
            "count_additive", "mae"
        ]

    audit = json.loads(
        (ROOT / "results/revision/phase02/ioi_phase02_claim_audit.json").read_text()
    )
    assert audit["all_checks_pass"] is True
    assert all(audit["checks"].values())


def test_phase04_artifact_builds_from_checked_outputs(tmp_path: Path) -> None:
    outdir = tmp_path / "phase04"
    manifest = build(ROOT, outdir)

    assert manifest["all_checks_pass"] is True
    assert manifest["contains_private_local_path"] is False
    assert (outdir / "phase04_claim_audit.json").exists()
    assert (outdir / "PHASE04_INTEGRATION_AUDIT.md").exists()
    assert (outdir / "phase04_manifest.json").exists()
    assert (outdir / "observer_cards/observer_card.json").exists()
    assert all(not path.startswith("/") for path in manifest["input_hashes"])
    for path, digest in manifest["output_hashes"].items():
        assert len(digest) == 64, path


def test_checked_text_artifacts_do_not_expose_private_local_paths() -> None:
    banned = ("/Users/", "/Downloads/", "observerbench-review-phase01")
    paths = [
        ROOT / "results/revision/phase01/ctl2_phase01_audit.json",
        ROOT / "results/revision/phase01/ctl2_multiseed_sweep/phase01_sweep_manifest.json",
        ROOT / "results/revision/phase02/ioi_stage2b_capacity/run_manifest.json",
        ROOT / "results/revision/phase02/ioi_stage2b_capacity/report.md",
        ROOT / "results/revision/phase02/ioi_stage2c_capacity/run_manifest.json",
        ROOT / "results/revision/phase02/ioi_stage2c_capacity/report.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in banned), path
