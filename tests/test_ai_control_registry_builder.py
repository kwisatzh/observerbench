"""Tests for composing task-separated AI-control panels into package data.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "leaderboards/safety/controlarena-apps-qwen-v1/results.json"
GEMMA = ROOT / "leaderboards/safety/controlarena-apps-gemma2-9b-it-v0/results.json"
QWEN35 = ROOT / "leaderboards/safety/controlarena-apps-qwen3-5-9b-v0/results.json"
SPEC = importlib.util.spec_from_file_location(
    "build_ai_control_registry", ROOT / "scripts/build_ai_control_registry.py"
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_registry_keeps_canonical_task_panels_separate(tmp_path: Path) -> None:
    output = tmp_path / "registry.json"
    payload = BUILDER.build((QWEN, GEMMA, QWEN35), output)

    assert len(payload["tasks"]) == 3
    assert len(payload["rows"]) == 29
    counts: dict[str, int] = {}
    for row in payload["rows"]:
        counts[row["task_id"]] = counts.get(row["task_id"], 0) + 1
    assert counts == {
        "controlarena-apps-verified-backdoors@qwen2.5-7b-v1": 11,
        "controlarena-apps-verified-backdoors@gemma-2-9b-it-v0": 9,
        "controlarena-apps-verified-backdoors@qwen3.5-9b-v0": 9,
    }
    assert all(task["panel_sha256"] for task in payload["tasks"])
    assert json.loads(output.read_text(encoding="utf-8"))["rows"] == payload["rows"]


def test_registry_rejects_duplicate_task_panels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate AI-control task panel"):
        BUILDER.build((QWEN, QWEN), tmp_path / "registry.json")
