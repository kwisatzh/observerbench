"""Focused tests for the separate Gemma APPS checked-result panel.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from observerbench.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "results/revision/ai_control/controlarena_apps_gemma2_9b_it_v0/checked_summary.json"
)
SPEC = importlib.util.spec_from_file_location(
    "build_gemma_ai_control_leaderboard",
    ROOT / "scripts/build_gemma_ai_control_leaderboard.py",
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_gemma_builder_keeps_model_and_primary_regime_separate(tmp_path: Path) -> None:
    payload = BUILDER.build(SUMMARY, tmp_path)
    rows = {row["observer_name"]: row for row in payload["rows"]}

    assert payload["task"]["task_id"] == (
        "controlarena-apps-verified-backdoors@gemma-2-9b-it-v0"
    )
    assert payload["task"]["primary_metric"] == BUILDER.PRIMARY_METRIC
    assert payload["metric_direction"][BUILDER.PRIMARY_METRIC] == "lower"
    assert len(rows) == 9
    assert all("qwen" not in row["task_id"] for row in rows.values())

    expected_primary = {
        "exact-label-oracle": 0.0,
        "gemma-verdict-adjacent-residual-probe": 2.1775,
        "gemma-supervised-score-distribution": 4.423,
        "gemma-neutral-context-residual-probe": 4.9465,
        "supervised-text-ridge": 9.054,
        "gemma-neutral-context-gemma-scope-sae-probe": 9.07,
        "gemma-controlarena-detailed-score-only": 10.803,
        "gemma-controlarena-prompt-ensemble": 13.689,
        "gemma-controlarena-basic": 14.9265,
    }
    for name, expected in expected_primary.items():
        assert rows[name]["metrics"][BUILDER.PRIMARY_METRIC] == pytest.approx(expected)

    assert rows["gemma-verdict-adjacent-residual-probe"]["metrics"][
        "risk_auroc"
    ] == pytest.approx(0.9688511678)
    sae = rows["gemma-neutral-context-gemma-scope-sae-probe"]
    assert sae["metrics"]["risk_auroc"] == pytest.approx(0.8698819950)
    assert sae["metadata"]["selection"]["layer"] == 20
    assert sae["metadata"]["selection"]["sae_repository"] == (
        "google/gemma-scope-9b-it-res"
    )
    assert sae["result_status"] == "prospectively frozen matched replication"
    assert sae["source_summary_sha256"] == file_sha256(SUMMARY)


def test_gemma_builder_writes_cards_csv_and_plain_scope_text(tmp_path: Path) -> None:
    BUILDER.build(SUMMARY, tmp_path)

    cards = sorted((tmp_path / "cards").glob("*.json"))
    assert len(cards) == 9
    card = json.loads(
        (
            tmp_path
            / "cards/gemma-neutral-context-gemma-scope-sae-probe.json"
        ).read_text(encoding="utf-8")
    )
    assert card["schema_version"] == "observerbench.safety_observer_card.v0"
    assert card["requires_white_box_access"] is True
    assert card["metadata"]["registration_scope"] == (
        "local digest seal; not an independent timestamp"
    )
    assert (tmp_path / "results.csv").is_file()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "not an independent preregistration" in readme
    assert "positive cross-model result" in readme


def test_gemma_builder_rejects_wrong_task_id(tmp_path: Path) -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary["task_id"] = "wrong@task"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong task ID"):
        BUILDER.build(bad, tmp_path / "out")
