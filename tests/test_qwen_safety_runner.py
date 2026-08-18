"""Staging tests for the Qwen safety runner without loading a model.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from observerbench.tasks.qwen_safety.artifacts import write_activation_cache
from observerbench.tasks.qwen_safety.runner import QwenSafetyRunner


def _config() -> dict:
    return {
        "schema": "observerbench.qwen_safety.v0",
        "status": "test",
        "model": {
            "id": "fixture",
            "revision": "0" * 40,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "expected_layers": 2,
            "expected_query_heads": 2,
            "expected_kv_heads": 1,
        },
        "design": {
            "seed": 3,
            "fit_pairs": 24,
            "calibration_pairs": 16,
            "locked_test_pairs_per_stratum": 8,
            "resources_per_bank": 16,
        },
        "activations": {"layers": [1, 2], "projection_dim": 8, "projection_seed": 5},
        "observers": {"ridge_grid": [0.001, 0.1]},
        "policy": {
            "block_budget_fraction": 0.1,
            "escalation_budget_fraction": 0.1,
            "violation_cost": 1.0,
            "false_block_cost": 1.0,
            "escalation_cost": 0.05,
            "cvar_alpha": 0.9,
        },
        "clean_gate": {
            "minimum_overall_candidate_accuracy": 0.9,
            "minimum_family_candidate_accuracy": 0.8,
            "minimum_paired_candidate_accuracy": 0.8,
            "minimum_median_candidate_margin": 0.0,
        },
        "runtime": {"device": "cpu", "batch_size": 8},
    }


def _write_cache(runner: QwenSafetyRunner, bank: str, *, clean: bool = True) -> None:
    design = runner._design()
    prompts = design.prompts_for(bank)
    rng = np.random.default_rng({"fit": 1, "calibration": 2, "locked_test": 3}[bank])
    activations = rng.normal(0.0, 0.05, size=(len(prompts), 2, 16))
    for index, prompt in enumerate(prompts):
        activations[index, :, 0] += 3.0 * float(prompt.unsafe_if_allowed)
        activations[index, :, 1] += 3.0 * float(prompt.unsafe_if_allowed) * prompt.severity / 20.0
    correct = np.full(len(prompts), clean, dtype=bool)
    write_activation_cache(
        runner.cache_path(bank),
        prompt_ids=[prompt.prompt_id for prompt in prompts],
        layer_indices=(1, 2),
        activations=activations,
        candidate_margins=np.where(correct, 1.0, -1.0),
        candidate_correct=correct,
        top1_correct=correct,
        sequence_lengths=np.full(len(prompts), 64),
        metadata={"bank": bank},
    )


def test_runner_freezes_before_locked_test_and_evaluates(tmp_path: Path) -> None:
    runner = QwenSafetyRunner(_config(), tmp_path)
    runner.prepare()
    _write_cache(runner, "fit")
    _write_cache(runner, "calibration")
    runner.freeze_observers()

    assert runner.freeze_path.exists()
    assert runner.state_path.exists()
    _write_cache(runner, "locked_test")
    runner.evaluate()
    payload = json.loads((tmp_path / "evaluation" / "qwen_safety_results.json").read_text())
    assert payload["status"] == "complete"
    assert "activation-direct-risk" in payload["results"]


def test_runner_refuses_to_freeze_after_locked_cache_exists(tmp_path: Path) -> None:
    runner = QwenSafetyRunner(_config(), tmp_path)
    runner.prepare()
    _write_cache(runner, "fit")
    _write_cache(runner, "calibration")
    _write_cache(runner, "locked_test")

    with pytest.raises(RuntimeError, match="refusing to freeze"):
        runner.freeze_observers()


def test_failed_clean_gate_stops_before_observer_results(tmp_path: Path) -> None:
    runner = QwenSafetyRunner(_config(), tmp_path)
    runner.prepare()
    _write_cache(runner, "fit")
    _write_cache(runner, "calibration")
    runner.freeze_observers()
    _write_cache(runner, "locked_test", clean=False)
    runner.evaluate()

    payload = json.loads((tmp_path / "evaluation" / "qwen_safety_results.json").read_text())
    assert payload["status"] == "stopped_clean_gate_failed"
    assert payload["results"] == {}
