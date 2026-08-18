"""Artifact tests for the Qwen safety track.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from observerbench.tasks.qwen_safety.artifacts import (
    load_activation_cache,
    load_qwen_safety_design,
    write_activation_cache,
    write_qwen_safety_design,
)
from observerbench.tasks.qwen_safety.design import (
    QwenSafetyDesignConfig,
    build_qwen_safety_design,
)


def _design():
    return build_qwen_safety_design(
        QwenSafetyDesignConfig(
            fit_pairs=6,
            calibration_pairs=4,
            locked_test_pairs_per_stratum=2,
            resources_per_bank=8,
        )
    )


def test_qwen_safety_design_round_trip(tmp_path: Path) -> None:
    design = _design()
    design_path, prompt_path = write_qwen_safety_design(design, tmp_path)
    loaded = load_qwen_safety_design(design_path)

    assert loaded.design_sha256 == design.design_sha256
    assert loaded.prompts == design.prompts
    assert prompt_path.read_text().splitlines()[0].startswith("prompt_id,bank,pair_id")


def test_activation_cache_round_trip_and_hash_check(tmp_path: Path) -> None:
    path = tmp_path / "fit_activations.npz"
    rng = np.random.default_rng(0)
    write_activation_cache(
        path,
        prompt_ids=("a", "b", "c"),
        layer_indices=(4, 8),
        activations=rng.normal(size=(3, 2, 5)),
        candidate_margins=(1.0, -1.0, 2.0),
        block_minus_allow_margins=(-1.0, 1.0, 2.0),
        candidate_correct=(True, False, True),
        top1_correct=(False, False, True),
        sequence_lengths=(20, 22, 24),
        metadata={"model": "fixture"},
    )
    loaded = load_activation_cache(path)

    assert loaded["activations"].shape == (3, 2, 5)
    assert list(loaded["prompt_ids"]) == ["a", "b", "c"]
    assert list(loaded["block_minus_allow_margins"]) == [-1.0, 1.0, 2.0]
    assert loaded["manifest"]["metadata"]["model"] == "fixture"
