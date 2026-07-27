"""Prospective gates and component selection for the Qwen induction study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanGateResult:
    passed: bool
    overall_candidate_accuracy: float
    worst_family_candidate_accuracy: float
    median_candidate_margin: float
    family_accuracy: dict[str, float]
    failures: tuple[str, ...]


def evaluate_clean_gate(
    clean_rows: pd.DataFrame,
    *,
    minimum_overall_accuracy: float = 0.95,
    minimum_family_accuracy: float = 0.90,
    minimum_median_margin: float = 0.0,
) -> CleanGateResult:
    """Apply the frozen exact-task gate before any intervention outcome."""

    required = {"prompt_id", "family_id", "candidate_correct", "candidate_margin"}
    missing = required - set(clean_rows)
    if missing:
        raise ValueError(f"clean rows lack columns: {sorted(missing)}")
    if clean_rows.empty or clean_rows["prompt_id"].astype(str).duplicated().any():
        raise ValueError("clean gate requires unique nonempty prompt rows")
    correct = clean_rows["candidate_correct"].astype(bool)
    margins = pd.to_numeric(clean_rows["candidate_margin"], errors="raise").to_numpy(float)
    if not np.isfinite(margins).all():
        raise ValueError("clean candidate margins must be finite")
    family_accuracy = {
        str(family): float(group["candidate_correct"].astype(bool).mean())
        for family, group in clean_rows.groupby("family_id", sort=True)
    }
    overall = float(correct.mean())
    worst = float(min(family_accuracy.values()))
    median = float(np.median(margins))
    failures: list[str] = []
    if overall < minimum_overall_accuracy:
        failures.append("overall_candidate_accuracy")
    if worst < minimum_family_accuracy:
        failures.append("worst_family_candidate_accuracy")
    if median <= minimum_median_margin:
        failures.append("median_candidate_margin")
    return CleanGateResult(
        passed=not failures,
        overall_candidate_accuracy=overall,
        worst_family_candidate_accuracy=worst,
        median_candidate_margin=median,
        family_accuracy=family_accuracy,
        failures=tuple(failures),
    )


def stratified_bootstrap_interval(
    values: Sequence[float],
    strata: Sequence[str],
    *,
    repeats: int = 5000,
    seed: int = 0,
    interval: float = 0.95,
) -> tuple[float, float, float]:
    """Return observed mean and a family-stratified percentile interval."""

    values_array = np.asarray(values, dtype=float)
    strata_array = np.asarray(tuple(map(str, strata)))
    if values_array.ndim != 1 or values_array.shape != strata_array.shape:
        raise ValueError("bootstrap values and strata must be aligned vectors")
    if len(values_array) == 0 or not np.isfinite(values_array).all():
        raise ValueError("bootstrap values must be nonempty and finite")
    if repeats <= 0 or not 0.0 < interval < 1.0:
        raise ValueError("bootstrap repeats and interval are invalid")
    rng = np.random.default_rng(seed)
    unique = tuple(sorted(set(strata_array.tolist())))
    indices = {value: np.flatnonzero(strata_array == value) for value in unique}
    draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in indices.values()]
        )
        draws[repeat] = values_array[sampled].mean()
    alpha = (1.0 - interval) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(values_array.mean()), float(lower), float(upper)


def shortlist_attention_heads(
    discovery: pd.DataFrame,
    *,
    count: int = 32,
) -> pd.DataFrame:
    """Freeze the top attention-specific heads with deterministic ties."""

    required = {
        "layer",
        "head",
        "kv_group",
        "attention_specificity",
        "direct_logit_attribution",
        "output_norm",
    }
    missing = required - set(discovery)
    if missing:
        raise ValueError(f"discovery rows lack columns: {sorted(missing)}")
    rows = discovery.copy()
    if rows[["layer", "head"]].duplicated().any():
        raise ValueError("discovery head coordinates must be unique")
    for column in ("attention_specificity", "direct_logit_attribution", "output_norm"):
        rows[column] = pd.to_numeric(rows[column], errors="raise").astype(float)
        if not np.isfinite(rows[column]).all():
            raise ValueError(f"discovery column {column} must be finite")
    if count <= 0 or count > len(rows):
        raise ValueError("attention shortlist size is invalid")
    rows = rows.sort_values(
        ["attention_specificity", "direct_logit_attribution", "layer", "head"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).head(count)
    rows = rows.reset_index(drop=True)
    rows.insert(0, "attention_rank", np.arange(1, len(rows) + 1))
    return rows


def summarize_singleton_effects(
    effect_cells: pd.DataFrame,
    *,
    repeats: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute one robust causal score per shortlisted head on head-fit prompts."""

    required = {"layer", "head", "prompt_id", "family_id", "effect"}
    missing = required - set(effect_cells)
    if missing:
        raise ValueError(f"singleton effects lack columns: {sorted(missing)}")
    if effect_cells[["layer", "head", "prompt_id"]].duplicated().any():
        raise ValueError("singleton head-fit cells must be unique")
    rows: list[dict[str, float | int]] = []
    for index, ((layer, head), group) in enumerate(
        effect_cells.groupby(["layer", "head"], sort=True)
    ):
        mean, lower, upper = stratified_bootstrap_interval(
            pd.to_numeric(group["effect"], errors="raise").to_numpy(float),
            group["family_id"].astype(str).to_numpy(),
            repeats=repeats,
            seed=seed + index,
        )
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "mean_effect": mean,
                "effect_ci_lower": lower,
                "effect_ci_upper": upper,
                "n_prompts": len(group),
            }
        )
    return pd.DataFrame(rows)


def select_causal_heads(
    shortlist: pd.DataFrame,
    singleton_summary: pd.DataFrame,
    *,
    count: int = 8,
    maximum_per_layer: int = 2,
    minimum_layers: int = 4,
) -> pd.DataFrame:
    """Select eight positive heads while enforcing the frozen layer constraints."""

    merged = shortlist.merge(
        singleton_summary,
        on=["layer", "head"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(shortlist):
        raise ValueError("head-fit effects do not cover the full attention shortlist")
    eligible = merged.loc[merged["effect_ci_lower"] > 0.0].sort_values(
        ["effect_ci_lower", "mean_effect", "attention_rank", "layer", "head"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    if len(eligible) < count or eligible["layer"].nunique() < minimum_layers:
        raise ValueError("fewer than eight robust heads or four eligible layers remain")

    # Reserve the best head from four distinct layers before filling the panel.
    layer_best = eligible.groupby("layer", sort=False).head(1).sort_values(
        ["effect_ci_lower", "mean_effect", "attention_rank"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    selected_indices = list(layer_best.head(minimum_layers).index)
    layer_counts = eligible.loc[selected_indices, "layer"].value_counts().to_dict()
    for row in eligible.itertuples():
        if row.Index in selected_indices:
            continue
        layer = int(row.layer)
        if layer_counts.get(layer, 0) >= maximum_per_layer:
            continue
        selected_indices.append(row.Index)
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if len(selected_indices) == count:
            break
    if len(selected_indices) != count:
        raise ValueError("layer cap leaves fewer than eight eligible heads")
    selected = eligible.loc[selected_indices].sort_values(
        ["effect_ci_lower", "mean_effect", "layer", "head"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if selected["layer"].value_counts().max() > maximum_per_layer:
        raise AssertionError("selected panel exceeds the frozen per-layer cap")
    if selected["layer"].nunique() < minimum_layers:
        raise AssertionError("selected panel misses the frozen layer-coverage gate")
    selected.insert(0, "component_index", np.arange(count))
    selected.insert(
        1,
        "head_label",
        [
            f"L{int(layer)}H{int(head)}"
            for layer, head in zip(selected["layer"], selected["head"])
        ],
    )
    return selected


def match_low_induction_controls(
    discovery: pd.DataFrame,
    shortlist: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Match controls by layer, KV group, and norm without causal outcomes."""

    excluded = set(zip(shortlist["layer"].astype(int), shortlist["head"].astype(int)))
    used: set[tuple[int, int]] = set()
    rows: list[pd.Series] = []
    for target in selected.sort_values("component_index").itertuples(index=False):
        layer = int(target.layer)
        kv_group = int(target.kv_group)
        layer_rows = discovery.loc[
            (discovery["layer"].astype(int) == layer)
            & (discovery["kv_group"].astype(int) == kv_group)
        ].copy()
        if layer_rows.empty:
            raise ValueError("selected head has no same-layer, same-KV control candidates")
        low_cutoff = float(layer_rows["attention_specificity"].median())
        candidates = layer_rows.loc[
            layer_rows["attention_specificity"] <= low_cutoff
        ].copy()
        candidates = candidates.loc[
            [
                (int(row.layer), int(row.head)) not in excluded
                and (int(row.layer), int(row.head)) not in used
                for row in candidates.itertuples(index=False)
            ]
        ]
        if candidates.empty:
            raise ValueError("no unused low-induction matched control remains")
        candidates["norm_distance"] = np.abs(
            candidates["output_norm"].to_numpy(float) - float(target.output_norm)
        )
        chosen = candidates.sort_values(
            ["norm_distance", "attention_specificity", "head"],
            kind="mergesort",
        ).iloc[0]
        used.add((int(chosen["layer"]), int(chosen["head"])))
        chosen = chosen.copy()
        chosen["matched_component_index"] = int(target.component_index)
        chosen["matched_head_label"] = str(target.head_label)
        rows.append(chosen)
    result = pd.DataFrame(rows).sort_values("matched_component_index").reset_index(drop=True)
    return result


def confirm_selected_vs_controls(
    confirmation_cells: pd.DataFrame,
    *,
    repeats: int = 5000,
    seed: int = 0,
) -> dict[str, float | bool]:
    """Apply the locked paired selected-minus-control causal gate."""

    required = {"prompt_id", "family_id", "arm", "effect"}
    missing = required - set(confirmation_cells)
    if missing:
        raise ValueError(f"confirmation cells lack columns: {sorted(missing)}")
    if set(confirmation_cells["arm"].astype(str)) != {"selected", "control"}:
        raise ValueError("confirmation requires selected and control arms")
    pivot = confirmation_cells.pivot(
        index=["prompt_id", "family_id"], columns="arm", values="effect"
    )
    if pivot.isna().any().any():
        raise ValueError("confirmation arms are not paired by prompt")
    contrast = pivot["selected"].to_numpy(float) - pivot["control"].to_numpy(float)
    mean, lower, upper = stratified_bootstrap_interval(
        contrast,
        pivot.index.get_level_values("family_id").astype(str).to_numpy(),
        repeats=repeats,
        seed=seed,
    )
    selected_effect = float(pivot["selected"].mean())
    fractions = (0.25, 0.5, 0.75)
    targets = tuple(selected_effect * fraction for fraction in fractions)
    passed = bool(lower > 0.0 and selected_effect > 0.0)
    return {
        "passed": passed,
        "selected_mean_effect": selected_effect,
        "control_mean_effect": float(pivot["control"].mean()),
        "contrast": mean,
        "contrast_ci_lower": lower,
        "contrast_ci_upper": upper,
        "target_025": targets[0],
        "target_050": targets[1],
        "target_075": targets[2],
    }


__all__ = [
    "CleanGateResult",
    "confirm_selected_vs_controls",
    "evaluate_clean_gate",
    "match_low_induction_controls",
    "select_causal_heads",
    "shortlist_attention_heads",
    "stratified_bootstrap_interval",
    "summarize_singleton_effects",
]
