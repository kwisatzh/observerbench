"""Checks for the model-free Qwen Copy-v2 open practice pack."""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK = REPO_ROOT / "practice" / "qwen_copy_v2_b040"
BUILDER = REPO_ROOT / "scripts" / "build_qwen_copy_v2_b040_practice_pack.py"
SOURCE = (
    REPO_ROOT
    / "results"
    / "revision"
    / "phase10"
    / "qwen_induction_copy_v2_complete"
    / "copy_v2"
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_open_pack_is_complete_and_noop_is_explicit() -> None:
    card = json.loads((PACK / "task_card.json").read_text(encoding="utf-8"))
    assert card["task_id"] == (
        "induction-qwen2-5-7b-finite-effects@copy-v2-b040"
    )
    assert card["participation_class"] == "open_practice"
    assert card["leaderboard_eligible"] is False
    assert card["sealed"] is False
    assert card["targets_public"] is True
    assert len(_rows(PACK / "calibration_measurements.csv")) == 40
    assert len(_rows(PACK / "queries.csv")) == 128
    assert len(_rows(PACK / "targets.csv")) == 128

    outcomes = _rows(PACK / "action_outcomes.csv")
    assert len(outcomes) == 3 * 16 * 9
    noops = [row for row in outcomes if row["is_noop"] == "true"]
    assert len(noops) == 3 * 16
    assert all(row["query_id"] == "analytic_noop" for row in noops)
    assert all(
        float(row["actual_target_loss"]) == pytest.approx(float(row["target"]))
        for row in noops
    )
    assert [item["value"] for item in card["action_contract"]["targets"]] == (
        pytest.approx(
            [0.8634306868771091, 1.7268613737542182, 2.5902920606313273]
        )
    )


def test_dependency_free_baseline_and_scorer(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    subprocess.run(
        [sys.executable, str(PACK / "baseline.py"), "--output", str(predictions)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [sys.executable, str(PACK / "score.py"), str(predictions), "--json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["participation_class"] == "open_practice"
    assert result["leaderboard_eligible"] is False
    assert result["prediction"]["mae"] == pytest.approx(0.2048538, abs=1e-6)
    assert result["prediction"]["rmse"] == pytest.approx(0.2294471, abs=1e-6)
    assert result["decision"]["aggregate"]["n_decisions"] == 48
    assert result["decision"]["aggregate"]["mean_regret"] == pytest.approx(
        0.0855571, abs=1e-6
    )
    assert len(result["decision"]["by_target"]) == 3


def test_builder_recreates_the_checked_aggregate_pack(tmp_path: Path) -> None:
    rebuilt = tmp_path / "pack"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--source-root",
            str(SOURCE),
            "--output",
            str(rebuilt),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for name in (
        "calibration_measurements.csv",
        "queries.csv",
        "targets.csv",
        "action_outcomes.csv",
        "task_card.json",
    ):
        assert _sha256(rebuilt / name) == _sha256(PACK / name)
