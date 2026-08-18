"""Prespecified evaluation for the prospective Phase-6 IOI study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This stage consumes the immutable prediction/action seal and the complete
held-out candidate surface.  It never fits an observer, changes an action, or
filters a prompt, mask, pool, template, or target.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance
from observerbench.tasks.ioi.phase6_confirmatory import PHASE6_STATUS
from observerbench.tasks.ioi.phase6_freeze import (
    COST_POLICY,
    DIRECT_RISK,
    FREEZE_SCHEMA,
    FREEZE_STATUS,
    JENSEN_SCORE,
    NATURAL_MEAN,
    QUADRATIC_MODEL,
    TARGET_POLICY,
)
from observerbench.tasks.ioi.phase6_test_measurement import (
    POLICY_HIERARCHY,
    TEST_MEASUREMENT_PROGRESS_SCHEMA,
    TEST_MEASUREMENT_SCHEMA,
    TEST_MEASUREMENT_SPEC_SCHEMA,
    TEST_MEASUREMENT_SPEC_STATUS,
    TEST_MEASUREMENT_STATUS,
    Phase6TestMeasurementConfig,
    _source_hashes,
    _validate_complete_manifest,
    build_phase6_test_measurement_spec,
    load_phase6_test_inputs,
    validate_candidate_effect_shard,
    validate_clean_test_scores,
)


EVALUATION_SCHEMA = "observerbench.ioi_phase06_evaluation.v1"
EVALUATION_STATUS = "phase6_prespecified_evaluation_complete"


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    candidate_selector_family: str
    candidate_model: str
    reference_selector_family: str
    reference_model: str
    claim_role: str


CORE_COMPARISONS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "H1_primary_estimand",
        DIRECT_RISK,
        QUADRATIC_MODEL,
        NATURAL_MEAN,
        QUADRATIC_MODEL,
        "primary",
    ),
    ComparisonSpec(
        "H2_secondary_vs_additive",
        DIRECT_RISK,
        QUADRATIC_MODEL,
        DIRECT_RISK,
        "additive_head",
        "secondary_structure",
    ),
    ComparisonSpec(
        "H2_secondary_vs_count",
        DIRECT_RISK,
        QUADRATIC_MODEL,
        DIRECT_RISK,
        "count_additive",
        "secondary_structure",
    ),
    ComparisonSpec(
        "Jensen_parameter_count_sensitivity",
        DIRECT_RISK,
        QUADRATIC_MODEL,
        JENSEN_SCORE,
        QUADRATIC_MODEL,
        "prespecified_sensitivity",
    ),
)


def _required(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _target_label(target: float) -> str:
    return f"target_{float(target):g}"


def _scope_targets(protocol: Mapping[str, Any]) -> tuple[tuple[str, tuple[float, ...]], ...]:
    targets = tuple(float(value) for value in protocol["targets"])
    primary = tuple(float(value) for value in protocol["primary_targets"])
    stress = tuple(float(value) for value in protocol["stress_test_targets"])
    return (
        ("primary_pooled", primary),
        *((_target_label(target), (target,)) for target in primary),
        *((f"{_target_label(target)}_stress", (target,)) for target in stress),
        ("all_three_pooled", targets),
    )


def validate_complete_candidate_surface(
    effects: pd.DataFrame,
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Require the exact held-out prompt-by-candidate Cartesian surface."""

    _required(
        effects,
        (
            "prompt_id",
            "mask_id",
            "mask_bits",
            "bank",
            "pool_id",
            "split",
            "clean_ld",
            "ablated_ld",
            "drop_from_clean",
        ),
        label="held-out effects",
    )
    test = prompts.loc[prompts["split"].astype(str) == "test"].copy()
    expected = len(test) * len(candidates)
    if len(effects) != expected:
        raise ValueError(f"expected {expected} held-out effect cells, found {len(effects)}")
    if effects.duplicated(["prompt_id", "mask_id"]).any():
        raise ValueError("held-out effects contain duplicate prompt-mask cells")
    if set(effects["prompt_id"].astype(str)) != set(test["prompt_id"].astype(str)):
        raise ValueError("held-out effects do not contain the exact test prompts")
    if set(effects["mask_id"].astype(str)) != set(candidates["mask_id"].astype(str)):
        raise ValueError("held-out effects do not contain the exact candidate masks")
    if set(effects["split"].astype(str)) != {"test"}:
        raise ValueError("held-out effects contain a non-test prompt")
    if set(effects["bank"].astype(str)) != {"candidate"}:
        raise ValueError("held-out effects contain a non-candidate mask")
    numeric = effects[["clean_ld", "ablated_ld", "drop_from_clean"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("held-out effects contain non-finite values")
    residual = (
        effects["clean_ld"].to_numpy(float)
        - effects["ablated_ld"].to_numpy(float)
        - effects["drop_from_clean"].to_numpy(float)
    )
    if float(np.max(np.abs(residual))) > 1e-6:
        raise ValueError("held-out finite effects are internally inconsistent")

    observed = effects[["mask_id", "mask_bits", "pool_id"]].drop_duplicates().copy()
    expected_mapping = candidates[["mask_id", "mask_bits", "pool_id"]].copy()
    for frame in (observed, expected_mapping):
        frame["mask_id"] = frame["mask_id"].astype(str)
        frame["mask_bits"] = frame["mask_bits"].astype(str).str.zfill(13)
        frame["pool_id"] = frame["pool_id"].astype(str)
        frame.sort_values("mask_id", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not observed.equals(expected_mapping):
        raise ValueError("held-out mask metadata differs from the sealed candidate bank")


def validate_clustered_test_design(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> None:
    """Require balanced pair, orientation, template, and pool bootstrap axes."""

    test = prompts.loc[prompts["split"].astype(str) == "test"].copy()
    cluster_column = "unordered_name_pair_id"
    expected_clusters = int(protocol["test_unordered_pair_cluster_count"])
    expected_prompts = int(protocol["test_prompts_per_pair_cluster"])
    if test[cluster_column].astype(str).nunique() != expected_clusters:
        raise ValueError("the held-out name-pair cluster count changed")
    if not test.groupby(cluster_column).size().eq(expected_prompts).all():
        raise ValueError("a held-out name-pair cluster has the wrong prompt count")
    key = [cluster_column, "template_id", "pair_orientation"]
    if test.duplicated(key).any():
        raise ValueError("a name-pair cluster repeats a template-orientation cell")
    templates = set(test["template_id"].astype(str))
    for _cluster, frame in test.groupby(cluster_column, sort=False):
        if set(frame["template_id"].astype(str)) != templates:
            raise ValueError("a name-pair cluster omits a frozen template")
        if set(frame["pair_orientation"].astype(str)) != {"a_to_b", "b_to_a"}:
            raise ValueError("a name-pair cluster omits a frozen orientation")
        by_template = frame.groupby("template_id")["pair_orientation"].agg(
            lambda values: set(map(str, values))
        )
        if not by_template.map(lambda values: values == {"a_to_b", "b_to_a"}).all():
            raise ValueError("a template does not retain both name-pair orientations")

    expected_pools = int(protocol["candidate_pool_count"])
    if candidates["pool_id"].astype(str).nunique() != expected_pools:
        raise ValueError("the candidate-pool bootstrap axis changed")
    if "candidate_pool_size" in protocol and not candidates.groupby("pool_id").size().eq(
        int(protocol["candidate_pool_size"])
    ).all():
        raise ValueError("a candidate pool has the wrong number of masks")


def validate_frozen_evaluation_inputs(
    actions: pd.DataFrame,
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> None:
    """Check the sealed tables structurally without fitting or selecting again."""

    _required(
        predictions,
        (
            "selector_family",
            "model",
            "target",
            "measurement_budget",
            "mask_id",
            "predicted_target_loss",
            "predicted_mean_effect",
        ),
        label="frozen predictions",
    )
    _required(
        actions,
        (
            "selector_family",
            "model",
            "target",
            "pool_id",
            "measurement_budget",
            "policy",
            "selected_mask_id",
            "selected_head_count",
            "predicted_target_loss",
            "predicted_objective",
        ),
        label="frozen actions",
    )
    targets = set(map(float, protocol["targets"]))
    budget = int(protocol["measurement_budget"])
    pools = set(candidates["pool_id"].astype(str))
    candidate_ids = set(candidates["mask_id"].astype(str))
    if set(predictions["target"].astype(float)) != targets:
        raise ValueError("frozen predictions changed the target set")
    if set(actions["target"].astype(float)) != targets:
        raise ValueError("frozen actions changed the target set")
    if set(actions["policy"].astype(str)) != {TARGET_POLICY, COST_POLICY}:
        raise ValueError("frozen actions changed the policy hierarchy")
    if set(actions["pool_id"].astype(str)) != pools:
        raise ValueError("frozen actions do not cover every candidate pool")
    if not set(actions["selected_mask_id"].astype(str)) <= candidate_ids:
        raise ValueError("a frozen action names a mask outside the candidate bank")
    if set(actions["measurement_budget"].astype(int)) != {budget}:
        raise ValueError("frozen actions changed the measurement budget")
    if set(predictions["measurement_budget"].astype(int)) != {budget}:
        raise ValueError("frozen predictions changed the measurement budget")
    if not np.isfinite(predictions["predicted_target_loss"].to_numpy(float)).all():
        raise ValueError("frozen predictions contain a non-finite value")

    prediction_keys = ["selector_family", "model", "target", "mask_id"]
    if predictions.duplicated(prediction_keys).any():
        raise ValueError("frozen predictions contain duplicate observer-mask cells")
    action_keys = ["selector_family", "model", "target", "pool_id", "policy"]
    if actions.duplicated(action_keys).any():
        raise ValueError("frozen actions contain duplicate observer-pool cells")

    expected_observers = {
        *((DIRECT_RISK, str(model)) for model in protocol["direct_risk_models"]),
        *((NATURAL_MEAN, str(model)) for model in protocol["mean_effect_models"]),
        *((JENSEN_SCORE, str(model)) for model in protocol["jensen_score_sensitivity_models"]),
    }
    observed_prediction_observers = set(
        predictions[["selector_family", "model"]].itertuples(index=False, name=None)
    )
    observed_action_observers = set(
        actions[["selector_family", "model"]].itertuples(index=False, name=None)
    )
    if observed_prediction_observers != expected_observers:
        raise ValueError("frozen predictions changed the observer set")
    if observed_action_observers != expected_observers:
        raise ValueError("frozen actions changed the observer set")
    expected_prediction_rows = len(expected_observers) * len(targets) * len(candidates)
    expected_action_rows = (
        len(expected_observers) * len(targets) * len(pools) * 2
    )
    if len(predictions) != expected_prediction_rows:
        raise ValueError("frozen predictions do not cover every observer-target-mask cell")
    if len(actions) != expected_action_rows:
        raise ValueError("frozen actions do not cover every observer-target-pool-policy cell")
    prediction_group_sizes = predictions.groupby(
        ["selector_family", "model", "target"]
    ).size()
    if not prediction_group_sizes.eq(len(candidates)).all():
        raise ValueError("a frozen observer lacks candidate predictions")
    action_group_sizes = actions.groupby(
        ["selector_family", "model", "target", "policy"]
    ).size()
    if not action_group_sizes.eq(len(pools)).all():
        raise ValueError("a frozen observer lacks a pool-level action")
    for _keys, group in predictions.groupby(
        ["selector_family", "model", "target"], sort=False
    ):
        if set(group["mask_id"].astype(str)) != candidate_ids:
            raise ValueError(
                "a frozen observer prediction table differs from the candidate bank"
            )
    natural_predictions = predictions.loc[
        predictions["selector_family"].astype(str) == NATURAL_MEAN,
        "predicted_mean_effect",
    ]
    if not np.isfinite(natural_predictions.to_numpy(float)).all():
        raise ValueError("a natural mean observer lacks its frozen mean prediction")

    mapping = candidates[["mask_id", "pool_id", "n_heads"]].copy()
    mapping["mask_id"] = mapping["mask_id"].astype(str)
    mapping["pool_id"] = mapping["pool_id"].astype(str)
    selected = actions.merge(
        mapping,
        left_on="selected_mask_id",
        right_on="mask_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_candidate"),
    )
    if selected["mask_id"].isna().any():
        raise ValueError("a frozen action lacks candidate metadata")
    if not (selected["pool_id"] == selected["pool_id_candidate"]).all():
        raise ValueError("a frozen action selects from a different candidate pool")
    if not np.array_equal(
        selected["selected_head_count"].to_numpy(int),
        selected["n_heads"].to_numpy(int),
    ):
        raise ValueError("a frozen action records the wrong head count")

    joined = actions.merge(
        predictions,
        left_on=["selector_family", "model", "target", "selected_mask_id"],
        right_on=["selector_family", "model", "target", "mask_id"],
        how="left",
        validate="many_to_one",
        suffixes=("_action", "_prediction"),
    )
    if joined["mask_id"].isna().any():
        raise ValueError("a frozen action has no corresponding frozen prediction")
    if not np.allclose(
        joined["predicted_target_loss_action"].to_numpy(float),
        joined["predicted_target_loss_prediction"].to_numpy(float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("a frozen action changed its sealed prediction")
    penalty = float(protocol["head_cost_penalty"])
    expected_objective = joined["predicted_target_loss_action"].to_numpy(float) + np.where(
        joined["policy"].astype(str).to_numpy() == COST_POLICY,
        penalty * joined["selected_head_count"].to_numpy(int),
        0.0,
    )
    if not np.allclose(
        joined["predicted_objective"].to_numpy(float),
        expected_objective,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("a frozen action changed its sealed objective")

    # This is a seal-integrity assertion over frozen, outcome-free predictions.
    # It never supplies or changes an action used by the held-out evaluation.
    prediction_choices = predictions.merge(
        mapping,
        on="mask_id",
        how="left",
        validate="many_to_one",
    )
    penalty = float(protocol["head_cost_penalty"])
    action_index = actions.set_index(action_keys)
    for keys, group in prediction_choices.groupby(
        ["selector_family", "model", "target", "pool_id"], sort=False
    ):
        ids = group["mask_id"].astype(str).to_numpy()
        counts = group["n_heads"].to_numpy(int)
        predicted = group["predicted_target_loss"].to_numpy(float)
        for policy, objective in (
            (TARGET_POLICY, predicted),
            (COST_POLICY, predicted + penalty * counts),
        ):
            chosen = int(np.lexsort((ids, counts, objective))[0])
            action_key = (str(keys[0]), str(keys[1]), float(keys[2]), str(keys[3]), policy)
            recorded_mask = str(action_index.loc[action_key, "selected_mask_id"])
            if recorded_mask != str(ids[chosen]):
                raise ValueError(
                    "a frozen action violates the sealed deterministic selection rule"
                )


def fixed_action_prompt_losses(
    actions: pd.DataFrame,
    effects: pd.DataFrame,
    prompts: pd.DataFrame,
    *,
    target_tolerance: float,
    head_cost_penalty: float,
) -> pd.DataFrame:
    """Evaluate every already-frozen action on every held-out prompt."""

    test_columns = [
        "prompt_id",
        "template_id",
        "structure",
        "unordered_name_pair_id",
        "pair_orientation",
        "io_name",
        "s_name",
    ]
    test = prompts.loc[prompts["split"].astype(str) == "test", test_columns].copy()
    surface = effects[["prompt_id", "mask_id", "drop_from_clean"]].copy()
    surface["mask_id"] = surface["mask_id"].astype(str)
    rows = actions.merge(
        surface,
        left_on="selected_mask_id",
        right_on="mask_id",
        how="left",
        validate="many_to_many",
    ).drop(columns="mask_id")
    if rows["drop_from_clean"].isna().any():
        raise ValueError("a fixed action is missing held-out prompt effects")
    if len(rows) != len(actions) * len(test):
        raise ValueError("fixed-action outcomes are not a complete action-prompt table")
    rows = rows.merge(test, on="prompt_id", how="left", validate="many_to_one")
    if rows["template_id"].isna().any():
        raise ValueError("a fixed-action outcome lacks test-prompt metadata")
    rows["actual_target_loss"] = np.abs(
        rows["drop_from_clean"].to_numpy(float) - rows["target"].to_numpy(float)
    )
    rows["within_tolerance"] = (
        rows["actual_target_loss"].to_numpy(float) <= float(target_tolerance)
    ).astype(int)
    rows["actual_objective"] = rows["actual_target_loss"].to_numpy(float) + np.where(
        rows["policy"].astype(str).to_numpy() == COST_POLICY,
        float(head_cost_penalty) * rows["selected_head_count"].to_numpy(int),
        0.0,
    )
    rows.sort_values(
        ["selector_family", "model", "policy", "target", "pool_id", "prompt_id"],
        inplace=True,
    )
    rows.reset_index(drop=True, inplace=True)
    return rows


def _paired_outcomes(
    outcomes: pd.DataFrame,
    spec: ComparisonSpec,
    *,
    policy: str,
) -> pd.DataFrame:
    keys = ["prompt_id", "pool_id", "target"]
    shared = [
        *keys,
        "template_id",
        "unordered_name_pair_id",
        "pair_orientation",
        "actual_target_loss",
        "actual_objective",
        "within_tolerance",
        "selected_mask_id",
        "selected_head_count",
    ]
    candidate = outcomes.loc[
        (outcomes["selector_family"] == spec.candidate_selector_family)
        & (outcomes["model"] == spec.candidate_model)
        & (outcomes["policy"] == policy),
        shared,
    ]
    reference = outcomes.loc[
        (outcomes["selector_family"] == spec.reference_selector_family)
        & (outcomes["model"] == spec.reference_model)
        & (outcomes["policy"] == policy),
        shared,
    ]
    paired = reference.merge(
        candidate,
        on=keys,
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if paired.empty or len(paired) != len(reference) or len(paired) != len(candidate):
        raise ValueError(f"comparison {spec.comparison_id} is incomplete")
    for column in ("template_id", "unordered_name_pair_id", "pair_orientation"):
        if not (paired[f"{column}_reference"] == paired[f"{column}_candidate"]).all():
            raise ValueError(f"comparison {spec.comparison_id} changed prompt metadata")
        paired[column] = paired[f"{column}_reference"]
    paired["absolute_target_loss_reduction"] = (
        paired["actual_target_loss_reference"]
        - paired["actual_target_loss_candidate"]
    )
    paired["within_tolerance_improvement"] = (
        paired["within_tolerance_candidate"]
        - paired["within_tolerance_reference"]
    )
    paired["cost_aware_objective_reduction"] = (
        paired["actual_objective_reference"] - paired["actual_objective_candidate"]
    )
    return paired


def _bootstrap_weights(
    cluster_count: int,
    pool_count: int,
    *,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if min(cluster_count, pool_count, repeats) <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    cluster_draws = rng.integers(
        0, cluster_count, size=(repeats, cluster_count)
    )
    pool_draws = rng.integers(0, pool_count, size=(repeats, pool_count))
    cluster = np.zeros((repeats, cluster_count), dtype=float)
    pool = np.zeros((repeats, pool_count), dtype=float)
    repeat_rows = np.arange(repeats)[:, None]
    np.add.at(cluster, (repeat_rows, cluster_draws), 1.0)
    np.add.at(pool, (repeat_rows, pool_draws), 1.0)
    cluster /= cluster_count
    pool /= pool_count
    return cluster, pool


def _cluster_pool_matrix(
    rows: pd.DataFrame,
    value_column: str,
    *,
    targets: Sequence[float],
    expected_prompts_per_pair: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    scoped = rows.loc[rows["target"].astype(float).isin(tuple(map(float, targets)))].copy()
    cell_keys = ["unordered_name_pair_id", "pool_id"]
    counts = scoped.groupby(cell_keys).size()
    expected_per_cell = int(expected_prompts_per_pair) * len(tuple(targets))
    if counts.empty or not counts.eq(expected_per_cell).all():
        raise ValueError(
            "paired cluster-pool cells do not retain every target, orientation, and template"
        )
    cells = scoped.groupby(cell_keys, as_index=False)[value_column].mean()
    matrix = cells.pivot(
        index="unordered_name_pair_id", columns="pool_id", values=value_column
    ).sort_index().sort_index(axis=1)
    if not np.isfinite(matrix.to_numpy(float)).all():
        raise ValueError("paired cluster-pool contrast is incomplete")
    return (
        matrix.to_numpy(float),
        list(map(str, matrix.index)),
        list(map(str, matrix.columns)),
    )


def _contrast_row(
    paired: pd.DataFrame,
    spec: ComparisonSpec,
    *,
    policy: str,
    metric: str,
    target_scope: str,
    targets: Sequence[float],
    protocol: Mapping[str, Any],
    bootstrap_weights: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    scoped = paired.loc[paired["target"].astype(float).isin(tuple(map(float, targets)))].copy()
    if scoped.empty:
        raise ValueError(f"comparison {spec.comparison_id} lacks {target_scope}")
    if metric == "absolute_target_loss_reduction":
        reference_values = scoped["actual_target_loss_reference"].to_numpy(float)
        candidate_values = scoped["actual_target_loss_candidate"].to_numpy(float)
    elif metric == "cost_aware_objective_reduction":
        reference_values = scoped["actual_objective_reference"].to_numpy(float)
        candidate_values = scoped["actual_objective_candidate"].to_numpy(float)
    elif metric == "within_tolerance_improvement":
        reference_values = scoped["within_tolerance_reference"].to_numpy(float)
        candidate_values = scoped["within_tolerance_candidate"].to_numpy(float)
    else:
        raise ValueError(f"unknown contrast metric {metric}")
    difference = scoped[metric].to_numpy(float)
    row: dict[str, Any] = {
        "comparison_id": spec.comparison_id,
        "claim_role": spec.claim_role,
        "candidate_selector_family": spec.candidate_selector_family,
        "candidate_model": spec.candidate_model,
        "reference_selector_family": spec.reference_selector_family,
        "reference_model": spec.reference_model,
        "policy": policy,
        "metric": metric,
        "target_scope": target_scope,
        "targets": ",".join(f"{value:g}" for value in map(float, targets)),
        "row_count": len(scoped),
        "reference_mean": float(reference_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "mean": float(difference.mean()),
        "relative_reduction_fraction": (
            float(difference.mean() / max(reference_values.mean(), 1e-12))
            if metric in {
                "absolute_target_loss_reduction",
                "cost_aware_objective_reduction",
            }
            else np.nan
        ),
        "q025": np.nan,
        "q975": np.nan,
        "bootstrap_repeats": 0,
    }
    if bootstrap_weights is not None:
        matrix, clusters, pools = _cluster_pool_matrix(
            scoped,
            metric,
            targets=targets,
            expected_prompts_per_pair=int(protocol["test_prompts_per_pair_cluster"]),
        )
        cluster_weights, pool_weights = bootstrap_weights
        if cluster_weights.shape[1] != len(clusters) or pool_weights.shape[1] != len(pools):
            raise ValueError("bootstrap axes differ from the frozen pair and pool counts")
        draws = np.einsum(
            "bi,ij,bj->b", cluster_weights, matrix, pool_weights, optimize=True
        )
        quantiles = protocol.get("bootstrap_interval", {}).get("quantiles", [0.025, 0.975])
        if tuple(map(float, quantiles)) != (0.025, 0.975):
            raise ValueError("the frozen bootstrap quantiles changed")
        row["q025"] = float(np.quantile(draws, 0.025, method="linear"))
        row["q975"] = float(np.quantile(draws, 0.975, method="linear"))
        row["bootstrap_repeats"] = int(len(draws))
    return row


def prespecified_contrasts(
    outcomes: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute the frozen H1/H2/Jensen and non-gating secondary contrasts."""

    repeats = int(protocol["bootstrap_repeats"])
    if repeats <= 0:
        raise ValueError("the frozen bootstrap repeat count must be positive")
    cluster_count = int(protocol["test_unordered_pair_cluster_count"])
    pool_count = int(protocol["candidate_pool_count"])
    weights = _bootstrap_weights(
        cluster_count,
        pool_count,
        repeats=repeats,
        seed=int(protocol["bootstrap_seed"]),
    )
    scopes = _scope_targets(protocol)
    records: list[dict[str, Any]] = []
    paired_tables: dict[str, pd.DataFrame] = {}
    for spec in CORE_COMPARISONS:
        target_paired = _paired_outcomes(outcomes, spec, policy=TARGET_POLICY)
        cost_paired = _paired_outcomes(outcomes, spec, policy=COST_POLICY)
        paired_tables[spec.comparison_id] = target_paired
        for scope_name, targets in scopes:
            inferential = weights if scope_name == "primary_pooled" else None
            records.append(
                _contrast_row(
                    target_paired,
                    spec,
                    policy=TARGET_POLICY,
                    metric="absolute_target_loss_reduction",
                    target_scope=scope_name,
                    targets=targets,
                    protocol=protocol,
                    bootstrap_weights=inferential,
                )
            )
            records.append(
                _contrast_row(
                    target_paired,
                    spec,
                    policy=TARGET_POLICY,
                    metric="within_tolerance_improvement",
                    target_scope=scope_name,
                    targets=targets,
                    protocol=protocol,
                    bootstrap_weights=inferential,
                )
            )
            records.append(
                _contrast_row(
                    cost_paired,
                    spec,
                    policy=COST_POLICY,
                    metric="cost_aware_objective_reduction",
                    target_scope=scope_name,
                    targets=targets,
                    protocol=protocol,
                    bootstrap_weights=inferential,
                )
            )
    return pd.DataFrame(records), paired_tables


def pool_sign_diagnostics(
    paired_tables: Mapping[str, pd.DataFrame],
    *,
    primary_targets: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison_id, paired in paired_tables.items():
        primary = paired.loc[paired["target"].astype(float).isin(primary_targets)]
        by_pool = primary.groupby("pool_id")["absolute_target_loss_reduction"].mean()
        rows.extend(
            {
                "comparison_id": comparison_id,
                "pool_id": str(pool_id),
                "mean_absolute_target_loss_reduction": float(value),
                "positive_direction": bool(value > 0.0),
                "zero_direction": bool(value == 0.0),
            }
            for pool_id, value in by_pool.items()
        )
    return pd.DataFrame(rows)


def leave_one_out_diagnostics(
    paired_tables: Mapping[str, pd.DataFrame],
    *,
    primary_targets: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point diagnostics only; they never gate or alter the primary analysis."""

    pair_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    ids = (
        "H1_primary_estimand",
        "H2_secondary_vs_additive",
        "H2_secondary_vs_count",
    )
    for comparison_id in ids:
        paired = paired_tables[comparison_id]
        primary = paired.loc[paired["target"].astype(float).isin(primary_targets)].copy()
        for cluster in sorted(primary["unordered_name_pair_id"].astype(str).unique()):
            kept = primary.loc[primary["unordered_name_pair_id"].astype(str) != cluster]
            reference = float(kept["actual_target_loss_reference"].mean())
            reduction = float(kept["absolute_target_loss_reduction"].mean())
            pair_rows.append(
                {
                    "comparison_id": comparison_id,
                    "omitted_unordered_name_pair_id": cluster,
                    "leave_one_name_out_equivalent": True,
                    "mean_absolute_target_loss_reduction": reduction,
                    "relative_reduction_fraction": reduction / max(reference, 1e-12),
                }
            )
        for template in sorted(primary["template_id"].astype(str).unique()):
            kept = primary.loc[primary["template_id"].astype(str) != template]
            reference = float(kept["actual_target_loss_reference"].mean())
            reduction = float(kept["absolute_target_loss_reduction"].mean())
            template_rows.append(
                {
                    "comparison_id": comparison_id,
                    "omitted_template_id": template,
                    "mean_absolute_target_loss_reduction": reduction,
                    "relative_reduction_fraction": reduction / max(reference, 1e-12),
                }
            )
    return pd.DataFrame(pair_rows), pd.DataFrame(template_rows)


def clean_task_validity(
    clean_scores: pd.DataFrame,
    prompts: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the clean-task gate without dropping or replacing any row."""

    test = prompts.loc[prompts["split"].astype(str) == "test"].copy()
    clean = clean_scores[["prompt_id", "clean_ld"]].copy()
    rows = test.merge(clean, on="prompt_id", how="left", validate="one_to_one")
    if rows["clean_ld"].isna().any() or len(rows) != len(test):
        raise ValueError("clean-task validity lacks a frozen test prompt")
    counts = (
        prompts.groupby(["template_id", "split"]).size().unstack(fill_value=0)
    )

    records: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("overall", rows)]
    groups.extend(
        (str(template), frame)
        for template, frame in rows.groupby("template_id", sort=True)
    )
    for template, frame in groups:
        if template == "overall":
            split_counts = prompts.groupby("split").size().to_dict()
        else:
            split_counts = counts.loc[template].to_dict()
        records.append(
            {
                "scope": template,
                "test_prompt_count": len(frame),
                "reference_prompt_count": int(split_counts.get("reference", 0)),
                "train_prompt_count": int(split_counts.get("train", 0)),
                "mean_clean_logit_difference": float(frame["clean_ld"].mean()),
                "io_vs_subject_pairwise_accuracy": float(
                    (frame["clean_ld"].to_numpy(float) > 0.0).mean()
                ),
            }
        )
    table = pd.DataFrame(records)
    gate = protocol["clean_template_validity"]["claim_gate"]
    overall = table.loc[table["scope"] == "overall"].iloc[0]
    templates = table.loc[table["scope"] != "overall"]
    checks = {
        "overall_accuracy": bool(
            overall["io_vs_subject_pairwise_accuracy"]
            >= float(gate["overall_IO_vs_subject_pairwise_accuracy_min"])
        ),
        "every_template_accuracy": bool(
            (
                templates["io_vs_subject_pairwise_accuracy"]
                >= float(gate["every_template_IO_vs_subject_pairwise_accuracy_min"])
            ).all()
        ),
        "every_template_positive_mean_clean_ld": bool(
            (templates["mean_clean_logit_difference"] > 0.0).all()
        ),
        "no_rows_filtered": len(rows) == len(test),
    }
    audit = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failure_does_not_change_h1_or_h2": True,
        "ioi_mechanism_or_fresh_template_generalization_language_allowed": bool(
            all(checks.values())
        ),
    }
    return table, audit


def candidate_actual_risk(
    effects: pd.DataFrame,
    *,
    targets: Sequence[float],
) -> pd.DataFrame:
    means = effects.groupby("mask_id")["drop_from_clean"].mean()
    rows: list[pd.DataFrame] = []
    for target in targets:
        actual = (
            effects.assign(
                actual_target_loss=np.abs(
                    effects["drop_from_clean"].to_numpy(float) - float(target)
                )
            )
            .groupby("mask_id", as_index=False)["actual_target_loss"]
            .mean()
        )
        actual["target"] = float(target)
        actual["actual_mean_effect"] = actual["mask_id"].astype(str).map(means)
        rows.append(actual)
    return pd.concat(rows, ignore_index=True)


def all_candidate_prediction_metrics(
    predictions: pd.DataFrame,
    actual_risk: pd.DataFrame,
) -> pd.DataFrame:
    frame = predictions.merge(
        actual_risk[["mask_id", "target", "actual_target_loss"]],
        on=["mask_id", "target"],
        how="left",
        validate="many_to_one",
    )
    if frame["actual_target_loss"].isna().any():
        raise ValueError("prediction metrics lack a held-out candidate risk")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["selector_family", "model", "measurement_budget", "target"], sort=True
    ):
        predicted = group["predicted_target_loss"].to_numpy(float)
        actual = group["actual_target_loss"].to_numpy(float)
        residual = predicted - actual
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        ranks = pd.Series(predicted).rank().corr(pd.Series(actual).rank())
        rows.append(
            {
                "selector_family": str(keys[0]),
                "model": str(keys[1]),
                "measurement_budget": int(keys[2]),
                "target": float(keys[3]),
                "candidate_count": len(group),
                "test_candidate_mae": float(np.mean(np.abs(residual))),
                "test_candidate_rmse": float(np.sqrt(np.mean(residual**2))),
                "test_candidate_r2": (
                    float(1.0 - np.sum(residual**2) / denominator)
                    if denominator > 0.0
                    else np.nan
                ),
                "test_candidate_rank_correlation": float(ranks),
            }
        )
    return pd.DataFrame(rows)


def natural_mean_estimand_metrics(
    predictions: pd.DataFrame,
    actual_risk: pd.DataFrame,
) -> pd.DataFrame:
    """Non-gating check of whether mean observers recover their own estimand."""

    natural = predictions.loc[
        predictions["selector_family"].astype(str) == NATURAL_MEAN
    ].copy()
    if natural.empty:
        raise ValueError("the frozen predictions lack the natural mean observers")
    if not np.isfinite(natural["predicted_mean_effect"].to_numpy(float)).all():
        raise ValueError("a natural mean observer lacks its frozen mean prediction")
    consistency = natural.groupby(
        ["selector_family", "model", "measurement_budget", "mask_id"]
    )["predicted_mean_effect"].agg(["min", "max", "count"])
    if not np.allclose(
        consistency["min"].to_numpy(float),
        consistency["max"].to_numpy(float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("a shared natural mean fit changed across target rows")
    natural = natural.drop_duplicates(
        ["selector_family", "model", "measurement_budget", "mask_id"]
    )
    actual = actual_risk[["mask_id", "actual_mean_effect"]].drop_duplicates()
    frame = natural.merge(actual, on="mask_id", how="left", validate="many_to_one")
    if frame["actual_mean_effect"].isna().any():
        raise ValueError("natural mean metrics lack a held-out candidate effect")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["selector_family", "model", "measurement_budget"], sort=True
    ):
        predicted = group["predicted_mean_effect"].to_numpy(float)
        actual_values = group["actual_mean_effect"].to_numpy(float)
        residual = predicted - actual_values
        denominator = float(np.sum((actual_values - actual_values.mean()) ** 2))
        ranks = pd.Series(predicted).rank().corr(pd.Series(actual_values).rank())
        rows.append(
            {
                "selector_family": str(keys[0]),
                "model": str(keys[1]),
                "measurement_budget": int(keys[2]),
                "candidate_count": len(group),
                "heldout_mean_effect_mae": float(np.mean(np.abs(residual))),
                "heldout_mean_effect_rmse": float(np.sqrt(np.mean(residual**2))),
                "heldout_mean_effect_r2": (
                    float(1.0 - np.sum(residual**2) / denominator)
                    if denominator > 0.0
                    else np.nan
                ),
                "heldout_mean_effect_rank_correlation": float(ranks),
                "descriptive_non_gating": True,
            }
        )
    return pd.DataFrame(rows)


def best_fixed_oracle(
    candidates: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    targets: Sequence[float],
    head_cost_penalty: float,
) -> pd.DataFrame:
    """Descriptive held-out oracle; it never supplies an action to a comparison."""

    matrix = effects.pivot(index="prompt_id", columns="mask_id", values="drop_from_clean")
    rows: list[dict[str, Any]] = []
    for pool_id, pool in candidates.groupby("pool_id", sort=True):
        pool = pool.sort_values("mask_id").reset_index(drop=True)
        ids = pool["mask_id"].astype(str).to_numpy()
        counts = pool["n_heads"].to_numpy(int)
        values = matrix.reindex(columns=ids).to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"held-out oracle lacks candidates for {pool_id}")
        for target in targets:
            mean_loss = np.abs(values - float(target)).mean(axis=0)
            for policy, penalty in (
                (TARGET_POLICY, np.zeros(len(ids))),
                (COST_POLICY, float(head_cost_penalty) * counts),
            ):
                objective = mean_loss + penalty
                selected = int(np.lexsort((ids, counts, objective))[0])
                rows.append(
                    {
                        "pool_id": str(pool_id),
                        "target": float(target),
                        "policy": policy,
                        "oracle_mask_id": str(ids[selected]),
                        "oracle_head_count": int(counts[selected]),
                        "oracle_mean_target_loss": float(mean_loss[selected]),
                        "oracle_mean_objective": float(objective[selected]),
                        "post_outcome_descriptive_only": True,
                    }
                )
    return pd.DataFrame(rows)


def decision_quality(outcomes: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "selector_family",
        "model",
        "measurement_budget",
        "pool_id",
        "target",
        "policy",
        "selected_mask_id",
        "selected_head_count",
    ]
    quality = outcomes.groupby(group_columns, as_index=False).agg(
        selected_mean_target_loss=("actual_target_loss", "mean"),
        selected_mean_objective=("actual_objective", "mean"),
        selected_within_tolerance=("within_tolerance", "mean"),
    )
    quality = quality.merge(
        oracle,
        on=["pool_id", "target", "policy"],
        how="left",
        validate="many_to_one",
    )
    if quality["oracle_mean_objective"].isna().any():
        raise ValueError("decision quality lacks a held-out oracle cell")
    quality["best_fixed_action_regret"] = (
        quality["selected_mean_objective"] - quality["oracle_mean_objective"]
    )
    return quality


def observer_summary(outcomes: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    summary = outcomes.groupby(
        ["selector_family", "model", "measurement_budget", "target", "policy"],
        as_index=False,
    ).agg(
        mean_target_loss=("actual_target_loss", "mean"),
        mean_objective=("actual_objective", "mean"),
        within_tolerance_fraction=("within_tolerance", "mean"),
        mean_selected_head_count=("selected_head_count", "mean"),
    )
    regret = quality.groupby(
        ["selector_family", "model", "measurement_budget", "target", "policy"],
        as_index=False,
    )["best_fixed_action_regret"].mean()
    return summary.merge(
        regret,
        on=["selector_family", "model", "measurement_budget", "target", "policy"],
        validate="one_to_one",
    )


def _one_contrast(
    contrasts: pd.DataFrame,
    comparison_id: str,
    target_scope: str,
) -> pd.Series:
    row = contrasts.loc[
        (contrasts["comparison_id"] == comparison_id)
        & (contrasts["policy"] == TARGET_POLICY)
        & (contrasts["metric"] == "absolute_target_loss_reduction")
        & (contrasts["target_scope"] == target_scope)
    ]
    if len(row) != 1:
        raise ValueError(f"missing unique contrast {comparison_id}/{target_scope}")
    return row.iloc[0]


def hypothesis_classification(
    contrasts: pd.DataFrame,
    clean_audit: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the gates written in the frozen protocol."""

    primary_targets = tuple(map(float, protocol["primary_targets"]))

    def directions(comparison_id: str) -> dict[str, bool]:
        return {
            f"target_{target:g}_nonnegative": bool(
                _one_contrast(contrasts, comparison_id, _target_label(target))["mean"]
                >= 0.0
            )
            for target in primary_targets
        }

    h1 = _one_contrast(contrasts, "H1_primary_estimand", "primary_pooled")
    h1_threshold = float(
        protocol["hypotheses"]["H1_primary_estimand"][
            "minimum_loss_reduction_fraction"
        ]
    )
    h1_checks = {
        "relative_loss_reduction_at_least_frozen_threshold": bool(
            h1["relative_reduction_fraction"] >= h1_threshold
        ),
        "paired_cluster_interval_lower_strictly_positive": bool(h1["q025"] > 0.0),
        **directions("H1_primary_estimand"),
    }

    h2_threshold = float(
        protocol["hypotheses"]["H2_secondary_structure"][
            "minimum_loss_reduction_fraction_against_each"
        ]
    )
    h2_comparisons: dict[str, Any] = {}
    for comparison_id in ("H2_secondary_vs_additive", "H2_secondary_vs_count"):
        row = _one_contrast(contrasts, comparison_id, "primary_pooled")
        checks = {
            "relative_loss_reduction_at_least_frozen_threshold": bool(
                row["relative_reduction_fraction"] >= h2_threshold
            ),
            "paired_cluster_interval_lower_strictly_positive": bool(row["q025"] > 0.0),
            **directions(comparison_id),
        }
        h2_comparisons[comparison_id] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "relative_reduction_fraction": float(row["relative_reduction_fraction"]),
            "q025": float(row["q025"]),
            "q975": float(row["q975"]),
        }

    jensen = _one_contrast(
        contrasts, "Jensen_parameter_count_sensitivity", "primary_pooled"
    )
    jensen_target_directions = directions("Jensen_parameter_count_sensitivity")
    jensen_checks = {
        "pooled_point_direction_nonnegative": bool(jensen["mean"] >= 0.0),
        "paired_cluster_interval_lower_strictly_positive": bool(jensen["q025"] > 0.0),
    }
    h1_passed = bool(all(h1_checks.values()))
    h2_passed = bool(all(item["passed"] for item in h2_comparisons.values()))
    jensen_passed = bool(all(jensen_checks.values()))
    clean_passed = bool(clean_audit["passed"])

    licensed_claims: list[str] = []
    if h1_passed:
        licensed_claims.append(
            "The direct-risk estimand improves fixed-action selection over the natural "
            "mean-effect plug-in observer using the same quadratic basis."
        )
    if h2_passed:
        licensed_claims.append(
            "A pairwise risk basis improves fixed-action selection over additive and "
            "count-additive risk observers; this is not an estimand claim."
        )
    if h1_passed and jensen_passed:
        licensed_claims.append(
            "The direct-risk result survives the target-specific parameter-count "
            "sensitivity against the Jensen-score control."
        )

    if h1_passed and h2_passed:
        calibration = "positive"
    elif h1_passed or h2_passed:
        calibration = "mixed"
    else:
        calibration = "negative"
    return {
        "scientific_status": PHASE6_STATUS,
        "result_calibration": calibration,
        "H1_primary_estimand": {
            "passed": h1_passed,
            "checks": h1_checks,
            "relative_reduction_fraction": float(h1["relative_reduction_fraction"]),
            "q025": float(h1["q025"]),
            "q975": float(h1["q975"]),
            "alone_defines_primary_success": True,
        },
        "H2_secondary_structure": {
            "passed": h2_passed,
            "cannot_rescue_failed_H1": True,
            "comparisons": h2_comparisons,
        },
        "Jensen_parameter_count_sensitivity": {
            "passed": jensen_passed,
            "checks": jensen_checks,
            "per_primary_target_directions_reported_not_gated": jensen_target_directions,
            "q025": float(jensen["q025"]),
            "q975": float(jensen["q975"]),
            "qualifies_H1_only": True,
        },
        "clean_task_validity": dict(clean_audit),
        "licensed_claims": licensed_claims,
        "ioi_language_allowed": clean_passed,
        "clean_failure_never_changes_H1_or_H2": True,
    }


def evaluate_phase6_tables(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    actions: pd.DataFrame,
    predictions: pd.DataFrame,
    effects: pd.DataFrame,
    clean_scores: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure table evaluation used by the sealed runner and synthetic tests."""

    validate_complete_candidate_surface(effects, prompts, candidates)
    validate_clustered_test_design(prompts, candidates, protocol=protocol)
    validate_frozen_evaluation_inputs(
        actions, predictions, candidates, protocol=protocol
    )
    outcomes = fixed_action_prompt_losses(
        actions,
        effects,
        prompts,
        target_tolerance=float(protocol["target_tolerance"]),
        head_cost_penalty=float(protocol["head_cost_penalty"]),
    )
    contrasts, paired = prespecified_contrasts(outcomes, protocol=protocol)
    pool_signs = pool_sign_diagnostics(
        paired, primary_targets=tuple(map(float, protocol["primary_targets"]))
    )
    leave_pair, leave_template = leave_one_out_diagnostics(
        paired, primary_targets=tuple(map(float, protocol["primary_targets"]))
    )
    clean_table, clean_audit = clean_task_validity(
        clean_scores, prompts, protocol=protocol
    )
    actual_risk = candidate_actual_risk(
        effects, targets=tuple(map(float, protocol["targets"]))
    )
    prediction_metrics = all_candidate_prediction_metrics(predictions, actual_risk)
    mean_estimand_metrics = natural_mean_estimand_metrics(predictions, actual_risk)
    oracle = best_fixed_oracle(
        candidates,
        effects,
        targets=tuple(map(float, protocol["targets"])),
        head_cost_penalty=float(protocol["head_cost_penalty"]),
    )
    quality = decision_quality(outcomes, oracle)
    summary = observer_summary(outcomes, quality)
    audit = hypothesis_classification(contrasts, clean_audit, protocol=protocol)
    return {
        "fixed_action_prompt_losses": outcomes,
        "prespecified_contrasts": contrasts,
        "pool_signs": pool_signs,
        "leave_one_pair": leave_pair,
        "leave_one_template": leave_template,
        "clean_task_validity": clean_table,
        "candidate_actual_risk": actual_risk,
        "prediction_metrics": prediction_metrics,
        "natural_mean_estimand_metrics": mean_estimand_metrics,
        "best_fixed_oracle": oracle,
        "decision_quality": quality,
        "observer_summary": summary,
        "hypothesis_audit": audit,
    }


def _load_verified_test_surface(
    design_dir: str | Path,
    calibration_dir: str | Path,
    freeze_dir: str | Path,
    test_dir: str | Path,
    *,
    protocol_path: str | Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, Any],
]:
    """Verify every source and result seal, then load the complete test surface."""

    test_root = Path(test_dir)
    manifest_path = test_root / "test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TEST_MEASUREMENT_SCHEMA:
        raise ValueError("unexpected Phase-6 test manifest schema")
    if manifest.get("status") != TEST_MEASUREMENT_STATUS:
        raise ValueError("Phase-6 test measurement is not complete")
    config = Phase6TestMeasurementConfig(**manifest.get("config", {}))
    inputs = load_phase6_test_inputs(
        design_dir,
        calibration_dir,
        freeze_dir,
        protocol_path=protocol_path,
        config=config,
    )
    sources = _source_hashes(
        design_dir=design_dir,
        calibration_dir=calibration_dir,
        freeze_dir=freeze_dir,
        protocol_path=protocol_path,
    )
    expected_spec = build_phase6_test_measurement_spec(
        inputs, config=config, source_hashes=sources
    )
    _validate_complete_manifest(test_root, manifest, expected_spec=expected_spec)
    recorded_spec = json.loads(
        (test_root / "measurement_run_spec.json").read_text(encoding="utf-8")
    )
    if file_sha256(test_root / "measurement_run_spec.json") != str(
        manifest["artifact_hashes"]["measurement_run_spec.json"]
    ):
        raise ValueError("held-out measurement spec changed while it was read")
    if recorded_spec != expected_spec:
        raise ValueError("held-out measurement spec differs from its sealed sources")
    if recorded_spec.get("schema") != TEST_MEASUREMENT_SPEC_SCHEMA or recorded_spec.get(
        "status"
    ) != TEST_MEASUREMENT_SPEC_STATUS:
        raise ValueError("held-out measurement spec is not frozen")
    progress = json.loads(
        (test_root / "measurement_progress.json").read_text(encoding="utf-8")
    )
    if file_sha256(test_root / "measurement_progress.json") != str(
        manifest["artifact_hashes"]["measurement_progress.json"]
    ):
        raise ValueError("held-out progress manifest changed while it was read")
    expected_effect_cells = len(inputs.test_prompts) * len(inputs.candidate_masks)
    expected_shards = int(expected_spec["counts"]["mask_shards"])
    if progress.get("schema") != TEST_MEASUREMENT_PROGRESS_SCHEMA:
        raise ValueError("unexpected held-out measurement progress schema")
    if progress.get("status") != "complete":
        raise ValueError("held-out measurement progress is not complete")
    if progress.get("accessed_prompt_splits") != ["test"]:
        raise ValueError("held-out progress accessed a non-test prompt split")
    if progress.get("accessed_mask_banks") != ["candidate"]:
        raise ValueError("held-out progress accessed a non-candidate mask bank")
    if int(progress.get("completed_candidate_masks", -1)) != len(
        inputs.candidate_masks
    ):
        raise ValueError("held-out progress did not complete every candidate mask")
    if int(progress.get("completed_shards", -1)) != expected_shards:
        raise ValueError("held-out progress did not complete every deterministic shard")
    expected_logical = {
        "clean_test_prompts": len(inputs.test_prompts),
        "test_prompt_candidate_mask_pairs": expected_effect_cells,
    }
    if progress.get("logical_forward_evaluations") != expected_logical:
        raise ValueError("held-out progress forward accounting changed")
    progress_artifacts = progress.get("artifact_hashes", {})
    expected_progress_labels = {
        "clean_scores_test.csv",
        *(
            label
            for label in manifest["artifact_hashes"]
            if str(label).startswith("shards/test/effects_")
        ),
    }
    if set(map(str, progress_artifacts)) != expected_progress_labels:
        raise ValueError("held-out progress does not index the exact measured outcomes")
    for label, expected_hash in progress_artifacts.items():
        if file_sha256(test_root / str(label)) != str(expected_hash):
            raise ValueError(f"held-out progress artifact changed: {label}")

    if manifest.get("scientific_status") != PHASE6_STATUS:
        raise ValueError("held-out measurement scientific status changed")
    if manifest.get("design_id") != inputs.design_manifest.get("design_id"):
        raise ValueError("held-out measurement used a different design identifier")
    expected_counts = {
        "test_prompts": len(inputs.test_prompts),
        "candidate_masks": len(inputs.candidate_masks),
        "candidate_pools": int(inputs.candidate_masks["pool_id"].nunique()),
        "candidate_effect_cells": expected_effect_cells,
        "mask_shards": expected_shards,
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("held-out measurement counts differ from the sealed design")
    model_record = manifest.get("model", {})
    if model_record.get("requested_name") != config.model_name or model_record.get(
        "requested_revision"
    ) != config.model_revision:
        raise ValueError("held-out measurement changed the pinned model")
    if manifest.get("reused_calibration_artifacts") != ["template_head_means.npz"]:
        raise ValueError("held-out measurement changed the sealed reference cache")
    expected_changes = {
        "observers_refit": False,
        "predictions_recomputed": False,
        "actions_reselected": False,
        "prompts_filtered": False,
        "candidate_masks_filtered": False,
    }
    if manifest.get("fit_or_selection_changes") != expected_changes:
        raise ValueError("held-out measurement reports an adaptive fit, selection, or filter")
    if manifest.get("policy_hierarchy") != POLICY_HIERARCHY:
        raise ValueError("held-out measurement changed the policy hierarchy")
    if manifest.get("accessed_prompt_splits") != ["test"]:
        raise ValueError("held-out measurement accessed a non-test prompt split")
    if manifest.get("accessed_mask_banks") != ["candidate"]:
        raise ValueError("held-out measurement accessed a non-candidate mask bank")
    if int(manifest.get("test_prompt_forward_passes", -1)) != len(inputs.test_prompts):
        raise ValueError("held-out test-prompt accounting changed")
    if int(manifest.get("candidate_mask_forward_passes", -1)) != expected_effect_cells:
        raise ValueError("held-out candidate-prompt accounting changed")

    clean = pd.read_csv(test_root / "clean_scores_test.csv", dtype={"prompt_id": str})
    validate_clean_test_scores(clean, inputs.test_prompts)
    if file_sha256(test_root / "clean_scores_test.csv") != str(
        manifest["artifact_hashes"]["clean_scores_test.csv"]
    ):
        raise ValueError("held-out clean scores changed while they were read")
    shard_labels = sorted(
        label
        for label in manifest["artifact_hashes"]
        if str(label).startswith("shards/test/effects_")
    )
    shards: list[pd.DataFrame] = []
    for label in shard_labels:
        stem = Path(label).stem
        start, stop = map(int, stem.removeprefix("effects_").split("_"))
        masks = inputs.candidate_masks.iloc[start:stop].reset_index(drop=True)
        rows = pd.read_csv(
            test_root / label,
            dtype={"prompt_id": str, "mask_id": str, "mask_bits": str, "pool_id": str},
        )
        validate_candidate_effect_shard(
            rows,
            prompts=inputs.test_prompts,
            masks=masks,
            clean_scores=clean,
        )
        if file_sha256(test_root / label) != str(manifest["artifact_hashes"][label]):
            raise ValueError(f"held-out candidate shard changed while it was read: {label}")
        shards.append(rows)
    effects = pd.concat(shards, ignore_index=True)
    validate_complete_candidate_surface(
        effects, inputs.test_prompts, inputs.candidate_masks
    )
    return (
        inputs.test_prompts,
        inputs.candidate_masks,
        clean,
        effects,
        manifest,
    )


def evaluate_phase6_confirmation(
    design_dir: str | Path,
    calibration_dir: str | Path,
    freeze_dir: str | Path,
    test_dir: str | Path,
    outdir: str | Path,
    *,
    protocol_path: str | Path,
) -> Path:
    """Validate the seals and write the immutable prospective evaluation."""

    test_prompts, candidates, clean, effects, test_manifest = _load_verified_test_surface(
        design_dir,
        calibration_dir,
        freeze_dir,
        test_dir,
        protocol_path=protocol_path,
    )
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if file_sha256(protocol_path) != str(test_manifest["source_hashes"]["protocol"]):
        raise ValueError("Phase-6 protocol changed while it was read")
    all_prompts = pd.read_csv(
        Path(design_dir) / "prompts.csv", dtype={"prompt_id": str}
    )
    design_manifest = json.loads(
        (Path(design_dir) / "design_manifest.json").read_text(encoding="utf-8")
    )
    if file_sha256(Path(design_dir) / "prompts.csv") != str(
        design_manifest["artifact_hashes"]["prompts.csv"]
    ):
        raise ValueError("sealed prompts changed while they were read")
    freeze_root = Path(freeze_dir)
    freeze_manifest = json.loads(
        (freeze_root / "prediction_action_manifest.json").read_text(encoding="utf-8")
    )
    if freeze_manifest.get("schema") != FREEZE_SCHEMA or freeze_manifest.get(
        "status"
    ) != FREEZE_STATUS:
        raise ValueError("prediction/action seal changed before evaluation")
    if file_sha256(freeze_root / "prediction_action_manifest.json") != str(
        test_manifest["source_hashes"]["prediction_action_manifest"]
    ):
        raise ValueError("prediction/action manifest changed while it was read")
    actions = pd.read_csv(
        freeze_root / "fixed_actions.csv",
        dtype={"pool_id": str, "selected_mask_id": str},
    )
    predictions = pd.read_csv(
        freeze_root / "candidate_predictions.csv", dtype={"mask_id": str}
    )
    for filename in ("fixed_actions.csv", "candidate_predictions.csv"):
        if file_sha256(freeze_root / filename) != str(
            freeze_manifest["artifact_hashes"][filename]
        ):
            raise ValueError(f"frozen evaluation input changed while it was read: {filename}")
    results = evaluate_phase6_tables(
        all_prompts,
        candidates,
        actions,
        predictions,
        effects,
        clean,
        protocol=protocol,
    )

    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Phase-6 evaluation output is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            if name == "fixed_action_prompt_losses":
                path = output / f"{name}.csv.gz"
                value.to_csv(
                    path,
                    index=False,
                    compression={"method": "gzip", "mtime": 0},
                )
            else:
                path = output / f"{name}.csv"
                value.to_csv(path, index=False)
            artifacts[path.name] = path
    audit_path = output / "hypothesis_audit.json"
    write_json(audit_path, results["hypothesis_audit"])
    artifacts[audit_path.name] = audit_path
    digest_path = output / "result_digest.json"
    write_json(
        digest_path,
        {
            "scientific_status": PHASE6_STATUS,
            "claim_classification": results["hypothesis_audit"],
            "primary_target_policy": TARGET_POLICY,
            "secondary_cost_policy_cannot_rescue_primary": True,
            "stress_target_is_not_a_primary_gate": True,
            "all_candidate_outcomes_used": True,
            "no_confirmatory_fit_action_reselection_or_filtering": True,
            "post_outcome_best_fixed_oracle_is_descriptive_non_gating": True,
        },
    )
    artifacts[digest_path.name] = digest_path
    manifest = {
        "schema": EVALUATION_SCHEMA,
        "status": EVALUATION_STATUS,
        "scientific_status": PHASE6_STATUS,
        "protocol_sha256": file_sha256(protocol_path),
        "design_manifest_sha256": file_sha256(
            Path(design_dir) / "design_manifest.json"
        ),
        "calibration_manifest_sha256": file_sha256(
            Path(calibration_dir) / "calibration_manifest.json"
        ),
        "prediction_action_manifest_sha256": file_sha256(
            Path(freeze_dir) / "prediction_action_manifest.json"
        ),
        "test_manifest_sha256": file_sha256(Path(test_dir) / "test_manifest.json"),
        "test_manifest_schema": test_manifest["schema"],
        "bootstrap": {
            "repeats": int(protocol["bootstrap_repeats"]),
            "seed": int(protocol["bootstrap_seed"]),
            "axes": list(protocol["bootstrap_axes"]),
            "primary_targets_retained_together": True,
        },
        "adaptation": {
            "observers_refit": False,
            "confirmatory_actions_reselected": False,
            "post_outcome_descriptive_oracle_computed": True,
            "rows_filtered": False,
            "targets_changed": False,
            "gates_changed": False,
        },
        "counts": {
            "test_prompts": len(test_prompts),
            "candidate_masks": len(candidates),
            "candidate_effect_cells": len(effects),
            "fixed_action_prompt_loss_rows": len(
                results["fixed_action_prompt_losses"]
            ),
        },
        "artifact_hashes": {
            name: file_sha256(path) for name, path in artifacts.items()
        },
        "runtime": runtime_provenance(),
    }
    write_json(output / "evaluation_manifest.json", manifest)
    return output
