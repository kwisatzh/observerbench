"""Tests for prospective Qwen induction gates and head selection.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from observerbench.tasks.qwen_induction.selection import (
    confirm_selected_vs_controls,
    evaluate_clean_gate,
    match_low_induction_controls,
    select_causal_heads,
    shortlist_attention_heads,
    summarize_singleton_effects,
)


def _discovery() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": layer,
                "head": head,
                "kv_group": head // 7,
                "attention_specificity": (
                    1.0 - 0.01 * (head % 7) - 0.0001 * (layer * 4 + head // 7)
                ),
                "direct_logit_attribution": 0.5 - 0.001 * head,
                "output_norm": 1.0 + 0.02 * head,
            }
            for layer in range(6)
            for head in range(28)
        ]
    )


def test_clean_gate_reports_pass_and_failure_reason() -> None:
    rows = pd.DataFrame(
        {
            "prompt_id": [f"p{index}" for index in range(40)],
            "family_id": [f"f{index % 4}" for index in range(40)],
            "candidate_correct": [True] * 39 + [False],
            "candidate_margin": np.linspace(0.2, 2.0, 40),
        }
    )
    assert evaluate_clean_gate(rows).passed
    rows.loc[rows["family_id"] == "f3", "candidate_correct"] = False
    result = evaluate_clean_gate(rows)
    assert not result.passed
    assert "worst_family_candidate_accuracy" in result.failures


def test_head_selection_obeys_layer_cap_and_control_matching_uses_no_outcomes() -> None:
    discovery = _discovery()
    shortlist = shortlist_attention_heads(discovery, count=32)
    effects = pd.DataFrame(
        [
            {
                "layer": int(head.layer),
                "head": int(head.head),
                "prompt_id": f"p{prompt:03d}",
                "family_id": f"f{prompt % 4}",
                "effect": 0.2 + 0.002 * (32 - int(head.attention_rank)) + 0.01 * (prompt % 2),
            }
            for head in shortlist.itertuples(index=False)
            for prompt in range(40)
        ]
    )
    summary = summarize_singleton_effects(effects, repeats=100, seed=3)
    selected = select_causal_heads(shortlist, summary)
    controls = match_low_induction_controls(discovery, shortlist, selected)

    assert len(selected) == len(controls) == 8
    assert selected["layer"].nunique() >= 4
    assert selected["layer"].value_counts().max() <= 2
    assert "effect" not in controls
    assert (
        controls["layer"].to_numpy(int) == selected.sort_values("component_index")["layer"].to_numpy(int)
    ).all()
    assert (
        controls["kv_group"].to_numpy(int)
        == selected.sort_values("component_index")["kv_group"].to_numpy(int)
    ).all()


def test_confirmation_gate_is_paired_and_freezes_three_targets() -> None:
    rows = pd.DataFrame(
        [
            {
                "prompt_id": f"p{prompt:03d}",
                "family_id": f"f{prompt % 4}",
                "arm": arm,
                "effect": (1.0 if arm == "selected" else 0.2) + 0.01 * (prompt % 3),
            }
            for prompt in range(80)
            for arm in ("selected", "control")
        ]
    )
    result = confirm_selected_vs_controls(rows, repeats=200, seed=8)

    assert result["passed"]
    assert result["contrast_ci_lower"] > 0.0
    assert result["target_025"] == pytest.approx(0.25 * result["selected_mean_effect"])
    assert result["target_075"] == pytest.approx(0.75 * result["selected_mean_effect"])
