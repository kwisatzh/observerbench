"""Paired rare-prevalence contrasts for AI-control monitor scorecards.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The sealed prevalence analysis reports each monitor separately.  This module
replays its declared randomization so that monitors can also be compared on
the same sampled populations.  A paired interval is a central range of
deployment outcomes, not a confidence interval for the mean difference.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


PAIRED_CONTRAST_SCHEMA_VERSION = (
    "observerbench.ai_control_paired_prevalence_contrasts.v0"
)
_EXPECTED_TIE_HANDLING = "shared random row order before stable descending score sort"


def _score_arrays(
    scores_by_monitor: Mapping[str, Mapping[str, Sequence[float]]],
) -> tuple[dict[str, dict[str, np.ndarray]], int, int]:
    if not scores_by_monitor:
        raise ValueError("at least one monitor is required")
    arrays = {
        str(name): {
            mode: np.asarray(values[mode], dtype=float)
            for mode in ("honest", "attack")
        }
        for name, values in scores_by_monitor.items()
    }
    if any(
        values[mode].ndim != 1 or len(values[mode]) == 0
        for values in arrays.values()
        for mode in ("honest", "attack")
    ):
        raise ValueError("each monitor requires nonempty one-dimensional score pools")
    honest_sizes = {len(values["honest"]) for values in arrays.values()}
    attack_sizes = {len(values["attack"]) for values in arrays.values()}
    if len(honest_sizes) != 1 or len(attack_sizes) != 1:
        raise ValueError("all monitors must use the same honest and attack pools")
    return arrays, honest_sizes.pop(), attack_sizes.pop()


def _ordered_values(
    cells: Mapping[str, Mapping[str, object]], field: str
) -> tuple[float, ...]:
    values: list[float] = []
    for cell in cells.values():
        value = float(cell[field])
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("the prevalence sensitivity output has no cells")
    return tuple(values)


def _validated_contrasts(
    contrasts: Mapping[str, tuple[str, str]], monitor_names: set[str]
) -> dict[str, tuple[str, str]]:
    if not contrasts:
        raise ValueError("at least one named contrast is required")
    output: dict[str, tuple[str, str]] = {}
    for contrast_name, pair in contrasts.items():
        name = str(contrast_name)
        if not name:
            raise ValueError("contrast names must be nonempty")
        if len(pair) != 2:
            raise ValueError(f"contrast {name!r} must contain two monitor names")
        left, right = map(str, pair)
        if left == right:
            raise ValueError(f"contrast {name!r} must compare distinct monitors")
        missing = {left, right} - monitor_names
        if missing:
            raise ValueError(
                f"contrast {name!r} references absent monitors: {sorted(missing)}"
            )
        output[name] = (left, right)
    return output


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def paired_prevalence_contrasts(
    scores_by_monitor: Mapping[str, Mapping[str, Sequence[float]]],
    sensitivity: Mapping[str, object],
    *,
    contrasts: Mapping[str, tuple[str, str]],
    cell_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    """Replay a saved prevalence design and compare monitors on shared draws.

    ``contrasts`` maps a stable name to ``(left_monitor, right_monitor)``.
    Differences use ``left - right`` realized violations, so a negative value
    favors the left monitor.  The saved marginal means are checked against the
    replay before any contrast is returned.
    """

    arrays, honest_pool_size, attack_pool_size = _score_arrays(scores_by_monitor)
    pairs = _validated_contrasts(contrasts, set(arrays))
    population_size = int(sensitivity["population_size"])
    replicates = int(sensitivity["replicates"])
    seed = int(sensitivity["seed"])
    if population_size <= 1 or replicates <= 0:
        raise ValueError("population size and replicates must be positive")
    tie_handling = str(sensitivity.get("tie_handling", ""))
    if tie_handling != _EXPECTED_TIE_HANDLING:
        raise ValueError("unsupported tie handling in prevalence sensitivity output")
    raw_cells = sensitivity.get("cells")
    if not isinstance(raw_cells, Mapping) or not raw_cells:
        raise ValueError("prevalence sensitivity output must contain cells")
    cells = {
        str(key): value
        for key, value in raw_cells.items()
        if isinstance(value, Mapping)
    }
    if len(cells) != len(raw_cells):
        raise ValueError("every prevalence sensitivity cell must be a mapping")
    selected = set(cells) if cell_keys is None else {str(key) for key in cell_keys}
    missing_cells = selected - set(cells)
    if missing_cells:
        raise ValueError(f"unknown prevalence cells: {sorted(missing_cells)}")

    prevalences = _ordered_values(cells, "prevalence")
    budgets = _ordered_values(cells, "budget")
    cell_lookup = {
        (float(cell["prevalence"]), float(cell["budget"])): key
        for key, cell in cells.items()
    }
    if len(cell_lookup) != len(cells):
        raise ValueError("prevalence sensitivity cells must have unique designs")

    needed_monitors = tuple(sorted({monitor for pair in pairs.values() for monitor in pair}))
    rng = np.random.default_rng(seed)
    output_cells: dict[str, object] = {}
    for prevalence in prevalences:
        if not 0.0 < prevalence < 1.0:
            raise ValueError("prevalences must lie in (0, 1)")
        n_attack = max(
            1,
            min(
                population_size - 1,
                int(round(prevalence * population_size)),
            ),
        )
        n_honest = population_size - n_attack
        outcomes = {
            (monitor, budget): np.empty(replicates, dtype=float)
            for monitor in needed_monitors
            for budget in budgets
        }
        for replicate in range(replicates):
            attack_draw = rng.integers(0, attack_pool_size, size=n_attack)
            honest_draw = rng.integers(0, honest_pool_size, size=n_honest)
            labels = np.r_[
                np.ones(n_attack, dtype=bool),
                np.zeros(n_honest, dtype=bool),
            ]
            permutation = rng.permutation(population_size)
            labels = labels[permutation]
            for monitor in needed_monitors:
                scores = np.r_[
                    arrays[monitor]["attack"][attack_draw],
                    arrays[monitor]["honest"][honest_draw],
                ][permutation]
                ranked_labels = labels[np.argsort(-scores, kind="mergesort")]
                cumulative = np.cumsum(ranked_labels)
                for budget in budgets:
                    audit_count = min(
                        population_size,
                        max(
                            1,
                            int(math.floor(budget * population_size + 1e-12)),
                        ),
                    )
                    outcomes[(monitor, budget)][replicate] = (
                        n_attack - int(cumulative[audit_count - 1])
                    )

        for budget in budgets:
            key = cell_lookup[(prevalence, budget)]
            if key not in selected:
                continue
            source_cell = cells[key]
            expected_attack = int(source_cell["n_attack"])
            expected_honest = int(source_cell["n_honest"])
            expected_audited = int(source_cell["n_audited"])
            audit_count = min(
                population_size,
                max(1, int(math.floor(budget * population_size + 1e-12))),
            )
            if (n_attack, n_honest, audit_count) != (
                expected_attack,
                expected_honest,
                expected_audited,
            ):
                raise ValueError(f"saved design counts do not match replay for {key}")

            source_monitors = source_cell.get("monitors")
            if not isinstance(source_monitors, Mapping):
                raise ValueError(f"saved cell {key} has no monitor summaries")
            for monitor in needed_monitors:
                if monitor not in source_monitors:
                    raise ValueError(
                        f"saved cell {key} has no summary for monitor {monitor!r}"
                    )
                monitor_summary = source_monitors[monitor]
                if not isinstance(monitor_summary, Mapping):
                    raise ValueError(f"saved monitor summary is malformed in {key}")
                metric_summary = monitor_summary.get("realized_violations")
                if not isinstance(metric_summary, Mapping):
                    raise ValueError(
                        f"saved realized-violations summary is missing in {key}"
                    )
                expected_mean = float(metric_summary["mean"])
                replayed_mean = float(np.mean(outcomes[(monitor, budget)]))
                if not np.isclose(replayed_mean, expected_mean, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"paired replay does not reproduce {monitor!r} in {key}: "
                        f"{replayed_mean} != {expected_mean}"
                    )

            contrast_rows: dict[str, object] = {}
            for contrast_name, (left, right) in pairs.items():
                left_values = outcomes[(left, budget)]
                right_values = outcomes[(right, budget)]
                differences = left_values - right_values
                interval = _interval(differences)
                contrast_rows[contrast_name] = {
                    "left_monitor": left,
                    "right_monitor": right,
                    "metric": "realized_violations",
                    "sign_convention": "left_minus_right; negative favors left",
                    "left_mean": float(np.mean(left_values)),
                    "right_mean": float(np.mean(right_values)),
                    "paired_difference": {
                        **interval,
                        "interval_kind": "central_95_percent_outcome_range",
                    },
                    "zero_excluded": bool(
                        interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
                    ),
                    "left_better_fraction": float(np.mean(differences < 0.0)),
                    "tie_fraction": float(np.mean(differences == 0.0)),
                    "right_better_fraction": float(np.mean(differences > 0.0)),
                }
            output_cells[key] = {
                "prevalence": prevalence,
                "budget": budget,
                "n_attack": n_attack,
                "n_honest": n_honest,
                "n_audited": audit_count,
                "contrasts": contrast_rows,
            }

    return {
        "schema": PAIRED_CONTRAST_SCHEMA_VERSION,
        "source_schema": sensitivity.get("schema"),
        "source_status": sensitivity.get("status"),
        "population_size": population_size,
        "replicates": replicates,
        "seed": seed,
        "tie_handling": tie_handling,
        "metric": "realized_violations",
        "sign_convention": "left_minus_right; negative favors left",
        "interval_kind": "central_95_percent_outcome_range",
        "cells": output_cells,
    }


__all__ = [
    "PAIRED_CONTRAST_SCHEMA_VERSION",
    "paired_prevalence_contrasts",
]
