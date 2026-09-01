"""Tests for checked Qwen effect-prediction leaderboard panels."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_qwen_effect_leaderboards",
    ROOT / "scripts" / "build_qwen_effect_leaderboards.py",
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_qwen_builder_emits_budget_matched_published_method_panels(
    tmp_path: Path,
) -> None:
    finite_path = tmp_path / "finite.csv"
    atp_path = tmp_path / "atp.csv"
    finite_rows: list[dict[str, object]] = []
    atp_rows: list[dict[str, object]] = []
    for budget in BUILDER.BUDGETS:
        finite_rows.extend(
            (
                {
                    "measurement_budget": budget,
                    "model": "additive",
                    "mae": 0.3,
                    "rmse": 0.4,
                },
                {
                    "measurement_budget": budget,
                    "model": "quadratic",
                    "mae": 0.1,
                    "rmse": 0.2,
                },
            )
        )
        atp_rows.extend(
            (
                {
                    "observer": "qwen-induction-attribution-patching-raw",
                    "measurement_budget": budget,
                    "mae": 0.25,
                    "rmse": 0.35,
                },
                {
                    "observer": (
                        "qwen-induction-attribution-patching-scalar-calibrated"
                    ),
                    "measurement_budget": budget,
                    "mae": 0.15,
                    "rmse": 0.22,
                },
            )
        )
    _write_csv(
        finite_path,
        ("measurement_budget", "model", "mae", "rmse"),
        finite_rows,
    )
    _write_csv(
        atp_path,
        ("observer", "measurement_budget", "mae", "rmse"),
        atp_rows,
    )
    output = tmp_path / "leaderboards"
    BUILDER.build(output_root=output, finite_path=finite_path, atp_path=atp_path)

    panels = sorted(path.name for path in output.iterdir())
    assert panels == [
        "induction-qwen2-5-7b-finite-effects-b016",
        "induction-qwen2-5-7b-finite-effects-b040",
        "induction-qwen2-5-7b-finite-effects-b064",
        "induction-qwen2-5-7b-finite-effects-b128",
    ]
    payload = json.loads((output / panels[0] / "results.json").read_text())
    rows = {row["observer_name"]: row for row in payload["rows"]}
    calibrated = rows[
        "qwen-induction-attribution-patching-scalar-calibrated"
    ]
    assert calibrated["requires_white_box_access"]
    assert calibrated["result_status"] == "post-outcome published-method baseline"
    assert calibrated["metrics"]["mae"] < rows[
        "qwen-induction-attribution-patching-raw"
    ]["metrics"]["mae"]
