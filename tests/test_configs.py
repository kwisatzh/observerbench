"""Configuration tests.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path

from observerbench.config import load_config
from observerbench.tasks.registry import TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_all_config_files_parse() -> None:
    config_paths = sorted((ROOT / "configs").glob("*.yaml"))

    assert {path.stem for path in config_paths} == set(TASKS)
    for path in config_paths:
        data = load_config(path)
        assert data["task"] == path.stem
        assert "mode" in data


def test_task_registry_contains_expected_task_names() -> None:
    assert set(TASKS) == {
        "ctl1_analytic",
        "trained_ctl1",
        "trained_ctl2",
        "ioi_stage1",
        "ioi_stage2b",
        "ioi_stage2c",
        "ioi_stage2d",
        "safety_interlock_analytic",
    }
