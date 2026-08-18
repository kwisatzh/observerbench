"""Tests for held-out IOI intervention selection.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from observerbench.tasks.ioi.phase2_capacity import LoadedIOIRun
from observerbench.tasks.ioi.selection import (
    IOISelectionConfig,
    evaluate_selection,
    paired_selection_contrasts,
    summarize_selection,
)


def _interaction_fixture() -> LoadedIOIRun:
    groups = ["P", "P", "P", "B", "B", "B", "B", "B", "B", "B", "B", "E", "E"]
    heads = pd.DataFrame(
        {
            "head_idx": range(13),
            "group": groups,
            "label": [f"{group}:{idx}" for idx, group in enumerate(groups)],
        }
    )
    rng = np.random.default_rng(42)
    masks = np.vstack([np.zeros(13, dtype=int), rng.integers(0, 2, size=(80, 13))])
    masks[1:, 0] = 1
    subset = pd.DataFrame(
        {
            "subset_idx": range(len(masks)),
            "mask_bits": ["".join(map(str, row)) for row in masks],
            "n_P": masks[:, :3].sum(axis=1),
            "n_B": masks[:, 3:11].sum(axis=1),
            "n_E": masks[:, 11:].sum(axis=1),
        }
    )
    p = masks[:, :3].sum(axis=1) / 3.0
    e = masks[:, 11:].sum(axis=1) / 2.0
    mean = 0.08 * masks.sum(axis=1) + 1.8 * p * e
    prompt_noise = rng.normal(0.0, 0.03, size=(40, len(masks)))
    drops = mean[None, :] + prompt_noise
    return LoadedIOIRun(
        prefix="fixture",
        source=Path("fixture"),
        heads=heads,
        subset=subset,
        masks=masks,
        prompt_drops=drops,
        mean_drops=drops.mean(axis=0),
        input_files=(),
    )


def test_interaction_observer_improves_held_out_selection() -> None:
    rows = evaluate_selection(
        _interaction_fixture(),
        config=IOISelectionConfig(
            repetitions=6,
            targets=(0.6, 1.2, 1.8),
            models=("additive_head", "count_plus_PE_bin4"),
            seed=7,
        ),
    )
    summary = summarize_selection(rows).set_index("model")
    assert summary.loc[
        "count_plus_PE_bin4", "oracle_regret_mean"
    ] < summary.loc["additive_head", "oracle_regret_mean"]

    contrasts = paired_selection_contrasts(
        rows,
        reference_models=("additive_head",),
        candidate_models=("count_plus_PE_bin4",),
    )
    assert contrasts["oracle_regret_reduction"].mean() > 0.0


def test_selection_splits_are_complete_and_cost_regret_nonnegative() -> None:
    config = IOISelectionConfig(
        repetitions=2,
        mask_folds=4,
        targets=(1.0,),
        models=("additive_head",),
        measurement_budgets=(10,),
    )
    rows = evaluate_selection(_interaction_fixture(), config=config)
    assert len(rows) == 2 * 4
    assert set(rows["n_fit_prompts"]) == {20}
    assert set(rows["n_eval_prompts"]) == {20}
    assert set(rows["n_measurements"]) == {10}
    assert (rows["oracle_regret"] >= -1e-12).all()
    assert (rows["same_size_oracle_regret"] >= -1e-12).all()
    assert (rows["cost_aware_regret"] >= -1e-12).all()
