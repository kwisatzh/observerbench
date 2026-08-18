"""Capacity-matched IOI interaction re-analysis.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

from observerbench.core import write_json
from observerbench.provenance import (
    portable_artifact_path,
    runtime_provenance,
    source_hashes,
)
from observerbench.tasks.ioi.stage2d import (
    count_bin_features,
    eval_predictions,
    kfold_indices,
    parse_mask_bits,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PAIR_NAMES: tuple[str, ...] = ("PB", "PE", "BE")
REPO_ROOT = Path(__file__).resolve().parents[4]
COUNT_COLUMNS: tuple[str, ...] = (
    "P_count_eq_1",
    "P_count_eq_2",
    "P_count_eq_3",
    "B_count_eq_1",
    "B_count_2_3",
    "B_count_4_5",
    "B_count_6_8",
    "E_count_eq_1",
    "E_count_eq_2",
)
CAPACITY_MODELS: tuple[str, ...] = (
    "additive_head",
    "count_additive",
    "count_plus_PB_bin4",
    "count_plus_PE_bin4",
    "count_plus_BE_bin4",
    "count_plus_all_bin4",
    "count_plus_all_minus_PB_bin4",
    "count_plus_all_minus_PE_bin4",
    "count_plus_all_minus_BE_bin4",
    "count_plus_PB_poly4",
    "count_plus_PE_poly4",
    "count_plus_BE_poly4",
    "count_plus_all_poly4",
)


@dataclass(frozen=True)
class IOIPhase2Config:
    k_folds: int = 5
    cv_repeats: int = 10
    bootstrap_repeats: int = 1000
    ridge: float = 1e-6
    seed: int = 0
    eval_nonclean_only: bool = True
    whole_group_pb_interaction: float = 1.055


@dataclass(frozen=True)
class LoadedIOIRun:
    prefix: str
    source: Path
    heads: pd.DataFrame
    subset: pd.DataFrame
    masks: np.ndarray
    prompt_drops: np.ndarray
    mean_drops: np.ndarray
    input_files: tuple[Path, ...]


def _find_prefix(source: Path) -> str:
    for prefix in ("ioi_stage2b", "ioi_stage2c"):
        required = [
            source / f"{prefix}_head_records.csv",
            source / f"{prefix}_subset_design.csv",
            source / f"{prefix}_per_prompt_drops.csv",
        ]
        if all(path.exists() for path in required):
            return prefix
    raise FileNotFoundError(
        f"{source} does not contain Stage 2b or Stage 2c head, subset, and per-prompt files"
    )


def load_head_subset_run(input_run: str | Path) -> LoadedIOIRun:
    source = Path(input_run).resolve()
    prefix = _find_prefix(source)
    heads_path = source / f"{prefix}_head_records.csv"
    subset_path = source / f"{prefix}_subset_design.csv"
    drops_path = source / f"{prefix}_per_prompt_drops.csv"
    heads = pd.read_csv(heads_path)
    subset = (
        pd.read_csv(subset_path, dtype={"mask_bits": str})
        .sort_values("subset_idx")
        .reset_index(drop=True)
    )
    if subset["subset_idx"].tolist() != list(range(len(subset))):
        raise ValueError("subset_idx must be contiguous from zero")
    masks = np.stack(
        [parse_mask_bits(bits, len(heads)) for bits in subset["mask_bits"]],
        axis=0,
    )
    groups = heads["group"].astype(str).tolist()
    totals = {group: max(1, groups.count(group)) for group in ("P", "B", "E")}
    for group in ("P", "B", "E"):
        indices = [idx for idx, value in enumerate(groups) if value == group]
        subset[f"n_{group}"] = masks[:, indices].sum(axis=1)
        subset[f"has_{group}"] = (subset[f"n_{group}"] > 0).astype(int)
    for pair in PAIR_NAMES:
        left, right = pair
        subset[f"{left}_{right}_count"] = (
            subset[f"n_{left}"] / totals[left]
        ) * (subset[f"n_{right}"] / totals[right])
        subset[f"{left}_{right}"] = (
            (subset[f"n_{left}"] > 0) & (subset[f"n_{right}"] > 0)
        ).astype(int)
    long = pd.read_csv(drops_path)
    prompt_matrix = long.pivot_table(
        index="prompt_idx",
        columns="subset_idx",
        values="drop_from_clean",
        aggfunc="first",
    ).reindex(columns=list(range(len(subset))))
    if prompt_matrix.isnull().any().any():
        raise ValueError("per-prompt drop matrix contains missing values")
    drops = prompt_matrix.to_numpy(float)
    return LoadedIOIRun(
        prefix=prefix,
        source=source,
        heads=heads,
        subset=subset,
        masks=masks,
        prompt_drops=drops,
        mean_drops=drops.mean(axis=0),
        input_files=(heads_path, subset_path, drops_path),
    )


def _coarse_group_basis(group: str, counts: np.ndarray) -> np.ndarray:
    if group == "P":
        return np.column_stack([counts == 1, counts >= 2]).astype(float)
    if group == "B":
        return np.column_stack(
            [(counts >= 1) & (counts <= 3), counts >= 4]
        ).astype(float)
    if group == "E":
        return np.column_stack([counts == 1, counts == 2]).astype(float)
    raise ValueError(group)


def _polynomial_group_basis(
    group: str,
    counts: np.ndarray,
    totals: Mapping[str, int],
) -> np.ndarray:
    normalized = counts.astype(float) / float(totals[group])
    return np.column_stack([normalized, normalized**2])


def _pair_block(
    pair: str,
    subset: pd.DataFrame,
    totals: Mapping[str, int],
    *,
    family: str,
) -> tuple[np.ndarray, list[str]]:
    left, right = pair
    if family == "bin4":
        left_basis = _coarse_group_basis(left, subset[f"n_{left}"].to_numpy(int))
        right_basis = _coarse_group_basis(right, subset[f"n_{right}"].to_numpy(int))
    elif family == "poly4":
        left_basis = _polynomial_group_basis(
            left, subset[f"n_{left}"].to_numpy(int), totals
        )
        right_basis = _polynomial_group_basis(
            right, subset[f"n_{right}"].to_numpy(int), totals
        )
    else:
        raise ValueError(family)
    block = np.column_stack(
        [left_basis[:, i] * right_basis[:, j] for i in range(2) for j in range(2)]
    )
    columns = [f"{pair}_{family}_{i + 1}{j + 1}" for i in range(2) for j in range(2)]
    return block, columns


def _base_design(
    run: LoadedIOIRun,
    *,
    include_counts: bool,
) -> tuple[np.ndarray, list[str]]:
    head_columns = run.heads["label"].astype(str).tolist()
    columns = ["intercept", *head_columns]
    arrays: list[np.ndarray] = [
        np.ones((len(run.subset), 1), dtype=float),
        run.masks.astype(float),
    ]
    if include_counts:
        count_rows = []
        for row in run.subset.itertuples(index=False):
            features = count_bin_features(int(row.n_P), int(row.n_B), int(row.n_E))
            count_rows.append([features[column] for column in COUNT_COLUMNS])
        arrays.append(np.asarray(count_rows, dtype=float))
        columns.extend(COUNT_COLUMNS)
    return np.column_stack(arrays), columns


def build_capacity_design(
    run: LoadedIOIRun,
    model: str,
) -> tuple[np.ndarray, list[str]]:
    if model == "additive_head":
        return _base_design(run, include_counts=False)
    base, columns = _base_design(run, include_counts=True)
    if model == "count_additive":
        return base, columns
    family = "poly4" if model.endswith("_poly4") else "bin4"
    totals = {
        group: int((run.heads["group"].astype(str) == group).sum())
        for group in ("P", "B", "E")
    }
    if "_all_minus_" in model:
        omitted = model.split("_all_minus_", 1)[1].split("_", 1)[0]
        pairs = [pair for pair in PAIR_NAMES if pair != omitted]
    elif "_all_" in model:
        pairs = list(PAIR_NAMES)
    else:
        pairs = [next(pair for pair in PAIR_NAMES if f"_{pair}_" in model)]
    blocks = [base]
    all_columns = list(columns)
    for pair in pairs:
        block, block_columns = _pair_block(
            pair, run.subset, totals, family=family
        )
        blocks.append(block)
        all_columns.extend(block_columns)
    return np.column_stack(blocks), all_columns


def _ridge_projection(
    X: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ridge: float,
) -> np.ndarray:
    regularizer = ridge * np.eye(X.shape[1], dtype=float)
    regularizer[0, 0] = 0.0
    solution = np.linalg.solve(
        X[train].T @ X[train] + regularizer,
        X[train].T,
    )
    return X[test] @ solution


FoldPlan = tuple[np.ndarray, np.ndarray, np.ndarray]
RepeatedPlan = tuple[tuple[FoldPlan, ...], ...]


def make_repeated_plan(
    X: np.ndarray,
    *,
    n_rows: int,
    k_folds: int,
    cv_repeats: int,
    seed: int,
    ridge: float,
) -> RepeatedPlan:
    repetitions = []
    for repeat in range(cv_repeats):
        folds = kfold_indices(n_rows, k_folds, seed=seed + repeat)
        repetitions.append(
            tuple(
                (train, test, _ridge_projection(X, train, test, ridge))
                for train, test in folds
            )
        )
    return tuple(repetitions)


def _predict_once(plan: Sequence[FoldPlan], y: np.ndarray) -> np.ndarray:
    predictions = np.full(len(y), np.nan, dtype=float)
    for train, test, projection in plan:
        predictions[test] = projection @ y[train]
    return predictions


def repeated_metrics(
    plan: RepeatedPlan,
    y: np.ndarray,
    *,
    eval_nonclean_only: bool,
) -> dict[str, float]:
    rows = []
    for repetition in plan:
        prediction = _predict_once(repetition, y)
        mae, rmse, r2 = eval_predictions(
            y,
            prediction,
            eval_nonclean=eval_nonclean_only,
        )
        rows.append((mae, rmse, r2))
    values = np.asarray(rows, dtype=float)
    return {
        "mae": float(np.mean(values[:, 0])),
        "rmse": float(np.mean(values[:, 1])),
        "r2": float(np.mean(values[:, 2])),
        "mae_cv_std": float(np.std(values[:, 0], ddof=1)) if len(values) > 1 else 0.0,
    }


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n_bootstrap": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "p_gt_0": float(np.mean(values > 0)),
    }


def contrast_summary(
    bootstrap_mae: pd.DataFrame,
    contrasts: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    pivot = bootstrap_mae.pivot(
        index="bootstrap",
        columns="model",
        values="mae",
    )
    rows = []
    for label, (reference, candidate) in contrasts.items():
        values = (pivot[reference] - pivot[candidate]).to_numpy(float)
        row: dict[str, object] = {
            "contrast": label,
            "reference": reference,
            "candidate": candidate,
            "definition": "MAE(reference) - MAE(candidate)",
        }
        row.update(_summary(values))
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_audit(
    run: LoadedIOIRun,
    designs: Mapping[str, tuple[np.ndarray, list[str]]],
) -> pd.DataFrame:
    base_rank = int(np.linalg.matrix_rank(designs["count_additive"][0]))
    models = (
        "count_additive",
        "count_plus_PB_bin4",
        "count_plus_PE_bin4",
        "count_plus_BE_bin4",
        "count_plus_all_bin4",
        "count_plus_PB_poly4",
        "count_plus_PE_poly4",
        "count_plus_BE_poly4",
    )
    rows = []
    for model in models:
        matrix, columns = designs[model]
        rank = int(np.linalg.matrix_rank(matrix))
        rows.append(
            {
                "model": model,
                "n_columns": len(columns),
                "design_rank": rank,
                "rank_added_vs_count_additive": rank - base_rank,
                "n_subsets": len(run.subset),
            }
        )
    return pd.DataFrame(rows)


def _plot_pair_contrasts(
    add_one: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    outpath: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    panels = (
        (axes[0], add_one, "Add one pair block"),
        (axes[1], leave_one_out, "Remove one pair block"),
    )
    for ax, frame, panel_title in panels:
        labels = frame["contrast"].str.extract(r"(PB|PE|BE)", expand=False).tolist()
        means = frame["mean"].to_numpy(float)
        lower = means - frame["q05"].to_numpy(float)
        upper = frame["q95"].to_numpy(float) - means
        ax.bar(labels, means, color=["#4c78a8", "#f58518", "#54a24b"])
        ax.errorbar(
            labels,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            color="black",
            capsize=4,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(panel_title)
        ax.set_xlabel("capacity-matched pair block")
    axes[0].set_ylabel("paired MAE improvement")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _lookup_condition(
    run: LoadedIOIRun,
    n_p: int,
    n_b: int,
    n_e: int,
) -> int:
    hit = run.subset[
        (run.subset["n_P"] == n_p)
        & (run.subset["n_B"] == n_b)
        & (run.subset["n_E"] == n_e)
    ]
    if hit.empty:
        raise ValueError(f"missing whole-group condition {(n_p, n_b, n_e)}")
    return int(hit.iloc[0]["subset_idx"])


def bootstrap_mobius(
    run: LoadedIOIRun,
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = {
        group: int((run.heads["group"].astype(str) == group).sum())
        for group in ("P", "B", "E")
    }
    idx = {
        "clean": _lookup_condition(run, 0, 0, 0),
        "P": _lookup_condition(run, totals["P"], 0, 0),
        "B": _lookup_condition(run, 0, totals["B"], 0),
        "E": _lookup_condition(run, 0, 0, totals["E"]),
        "PB": _lookup_condition(run, totals["P"], totals["B"], 0),
        "PE": _lookup_condition(run, totals["P"], 0, totals["E"]),
        "BE": _lookup_condition(run, 0, totals["B"], totals["E"]),
        "PBE": _lookup_condition(run, totals["P"], totals["B"], totals["E"]),
    }

    def terms(prompt_mean: np.ndarray) -> dict[str, float]:
        d = {name: float(prompt_mean[column]) for name, column in idx.items()}
        pb = d["PB"] - d["P"] - d["B"] + d["clean"]
        pe = d["PE"] - d["P"] - d["E"] + d["clean"]
        be = d["BE"] - d["B"] - d["E"] + d["clean"]
        triple = (
            d["PBE"]
            - d["PB"]
            - d["PE"]
            - d["BE"]
            + d["P"]
            + d["B"]
            + d["E"]
            - d["clean"]
        )
        return {
            "PB": pb,
            "PE": pe,
            "BE": be,
            "PBE": triple,
            "PE_minus_PB": pe - pb,
        }

    point = terms(run.mean_drops)
    rng = np.random.default_rng(seed)
    rows = []
    for bootstrap_idx in range(repeats):
        prompt_indices = rng.integers(
            0,
            run.prompt_drops.shape[0],
            size=run.prompt_drops.shape[0],
        )
        draw = terms(run.prompt_drops[prompt_indices].mean(axis=0))
        rows.extend(
            {
                "bootstrap": bootstrap_idx,
                "term": term,
                "value": value,
            }
            for term, value in draw.items()
        )
    draws = pd.DataFrame(rows)
    summary_rows = []
    for term, group in draws.groupby("term", sort=False):
        values = group["value"].to_numpy(float)
        row: dict[str, object] = {"term": term, "point": point[term]}
        row.update(_summary(values))
        summary_rows.append(row)
    return draws, pd.DataFrame(summary_rows)


def _plot_mobius(summary: pd.DataFrame, outpath: Path) -> None:
    plot = summary[summary["term"].isin(["PB", "PE", "BE", "PBE"])].copy()
    means = plot["mean"].to_numpy(float)
    lower = means - plot["q05"].to_numpy(float)
    upper = plot["q95"].to_numpy(float) - means
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(
        plot["term"],
        means,
        color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"],
    )
    ax.errorbar(
        plot["term"],
        means,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_ylabel("direct non-additivity")
    ax.set_title("Whole-group IOI interactions")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def design_coverage(run: LoadedIOIRun) -> dict[str, float | int | str]:
    nonclean = run.subset.iloc[1:].copy()
    normalized_pb = (nonclean["n_P"] / 3.0) * (nonclean["n_B"] / 8.0)
    return {
        "design": "anchored broad random masks",
        "n_nonclean_subsets": int(len(nonclean)),
        "full_pb_corner_count": int(
            ((nonclean["n_P"] == 3) & (nonclean["n_B"] == 8)).sum()
        ),
        "full_pb_corner_note": "PB and PBE are forced whole-group anchors",
        "high_pb_coverage_count": int(
            ((nonclean["n_P"] >= 2) & (nonclean["n_B"] >= 6)).sum()
        ),
        "median_normalized_pb_exposure": float(normalized_pb.median()),
        "mean_normalized_pb_exposure": float(normalized_pb.mean()),
    }


def _augment(base: np.ndarray, feature: np.ndarray) -> np.ndarray:
    array = np.asarray(feature, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    return np.column_stack([base, array])


def design_sensitivity(
    run: LoadedIOIRun,
    *,
    config: IOIPhase2Config,
) -> pd.DataFrame:
    base, _ = _base_design(run, include_counts=False)
    p = run.subset["n_P"].to_numpy(float) / 3.0
    b = run.subset["n_B"].to_numpy(float) / 8.0
    shapes = {
        "bilinear": p * b,
        "full_corner": (
            (run.subset["n_P"] == 3) & (run.subset["n_B"] == 8)
        ).to_numpy(float),
        "high_coverage": (
            (run.subset["n_P"] >= 2) & (run.subset["n_B"] >= 6)
        ).to_numpy(float),
        "primary_full_x_backup_fraction": (
            run.subset["n_P"] == 3
        ).to_numpy(float)
        * b,
    }
    occupancy = (
        (run.subset["n_P"] > 0) & (run.subset["n_B"] > 0)
    ).to_numpy(float)
    bilinear = p * b
    bin4, _ = _pair_block(
        "PB",
        run.subset,
        {"P": 3, "B": 8, "E": 2},
        family="bin4",
    )
    rows = []
    for shape_name, shape in shapes.items():
        y = config.whole_group_pb_interaction * shape
        designs = {
            "additive_head": base,
            "legacy_occupancy": _augment(base, occupancy),
            "normalized_bilinear": _augment(base, bilinear),
            "balanced_bin4": _augment(base, bin4),
            "oracle_shape": _augment(base, shape),
        }
        metrics = {}
        for model, X in designs.items():
            plan = make_repeated_plan(
                X,
                n_rows=len(run.subset),
                k_folds=config.k_folds,
                cv_repeats=config.cv_repeats,
                seed=config.seed + 777,
                ridge=config.ridge,
            )
            metrics[model] = repeated_metrics(
                plan,
                y,
                eval_nonclean_only=True,
            )["mae"]
        rows.extend(
            {
                "shape": shape_name,
                "model": model,
                "base_additive_mae": metrics["additive_head"],
                "candidate_mae": mae,
                "mae_gain_vs_additive": metrics["additive_head"] - mae,
                "nonzero_subsets": int(np.count_nonzero(shape[1:])),
                "prevalence": float(np.mean(shape[1:] != 0)),
                "mean_absolute_signal": float(np.mean(np.abs(y[1:]))),
                "interaction_at_full": config.whole_group_pb_interaction,
            }
            for model, mae in metrics.items()
        )
    return pd.DataFrame(rows)


def _report_text(
    label: str,
    run: LoadedIOIRun,
    add_count: pd.DataFrame,
    add_additive: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> str:
    def line(frame: pd.DataFrame, pair: str) -> str:
        row = frame[frame["contrast"].str.contains(pair)].iloc[0]
        return f"{pair}: {row['mean']:.5f} [{row['q05']:.5f}, {row['q95']:.5f}]"

    return (
        f"# {label}: capacity-matched IOI pair analysis\n\n"
        f"Input: {portable_artifact_path(run.source, REPO_ROOT)}. "
        "The prompt bootstrap resamples prompts while holding "
        "the intervention masks and ten repeated five-fold splits fixed.\n\n"
        "Each pair receives the same four-column count-bin interaction block.\n\n"
        "## Add one, relative to count-additive\n\n"
        + "\n".join(f"- {line(add_count, pair)}" for pair in PAIR_NAMES)
        + "\n\n## Add one, relative to additive-head\n\n"
        + "\n".join(f"- {line(add_additive, pair)}" for pair in PAIR_NAMES)
        + "\n\n## Leave one pair out of the all-pairs model\n\n"
        + "\n".join(f"- {line(leave_one_out, pair)}" for pair in PAIR_NAMES)
        + "\n"
    )


def run_capacity_analysis(
    input_run: str | Path,
    outdir: str | Path,
    *,
    label: str,
    config: IOIPhase2Config,
    include_design_sensitivity: bool = False,
    include_mobius: bool = False,
) -> Path:
    run = load_head_subset_run(input_run)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    designs = {
        model: build_capacity_design(run, model)
        for model in CAPACITY_MODELS
    }
    plans = {
        model: make_repeated_plan(
            matrix,
            n_rows=len(run.subset),
            k_folds=config.k_folds,
            cv_repeats=config.cv_repeats,
            seed=config.seed + 777,
            ridge=config.ridge,
        )
        for model, (matrix, _columns) in designs.items()
    }

    point_rows = []
    for model, plan in plans.items():
        metrics = repeated_metrics(
            plan,
            run.mean_drops,
            eval_nonclean_only=config.eval_nonclean_only,
        )
        matrix, columns = designs[model]
        point_rows.append(
            {
                "model": model,
                **metrics,
                "n_columns": len(columns),
                "design_rank": int(np.linalg.matrix_rank(matrix)),
            }
        )
    point = pd.DataFrame(point_rows)

    rng = np.random.default_rng(config.seed + 999)
    bootstrap_rows = []
    for bootstrap_idx in range(config.bootstrap_repeats):
        prompt_indices = rng.integers(
            0,
            run.prompt_drops.shape[0],
            size=run.prompt_drops.shape[0],
        )
        y = run.prompt_drops[prompt_indices].mean(axis=0)
        for model, plan in plans.items():
            mae = repeated_metrics(
                plan,
                y,
                eval_nonclean_only=config.eval_nonclean_only,
            )["mae"]
            bootstrap_rows.append(
                {"bootstrap": bootstrap_idx, "model": model, "mae": mae}
            )
    bootstrap_mae = pd.DataFrame(bootstrap_rows)

    add_count = contrast_summary(
        bootstrap_mae,
        {
            f"add_{pair}_bin4_vs_count": (
                "count_additive",
                f"count_plus_{pair}_bin4",
            )
            for pair in PAIR_NAMES
        }
        | {
            "add_all_bin4_vs_count": (
                "count_additive",
                "count_plus_all_bin4",
            )
        },
    )
    add_additive = contrast_summary(
        bootstrap_mae,
        {
            f"add_{pair}_bin4_vs_additive": (
                "additive_head",
                f"count_plus_{pair}_bin4",
            )
            for pair in PAIR_NAMES
        }
        | {
            "add_all_bin4_vs_additive": (
                "additive_head",
                "count_plus_all_bin4",
            )
        },
    )
    leave_one_out = contrast_summary(
        bootstrap_mae,
        {
            f"remove_{pair}_bin4": (
                f"count_plus_all_minus_{pair}_bin4",
                "count_plus_all_bin4",
            )
            for pair in PAIR_NAMES
        },
    )
    polynomial = contrast_summary(
        bootstrap_mae,
        {
            f"add_{pair}_poly4_vs_count": (
                "count_additive",
                f"count_plus_{pair}_poly4",
            )
            for pair in PAIR_NAMES
        }
        | {
            "add_all_poly4_vs_count": (
                "count_additive",
                "count_plus_all_poly4",
            )
        },
    )

    point.to_csv(out / "model_comparison.csv", index=False)
    bootstrap_mae.to_csv(out / "bootstrap_model_mae.csv", index=False)
    add_count.to_csv(out / "add_one_vs_count_additive.csv", index=False)
    add_additive.to_csv(out / "add_one_vs_additive_head.csv", index=False)
    leave_one_out.to_csv(out / "leave_one_out_contrasts.csv", index=False)
    polynomial.to_csv(
        out / "polynomial_sensitivity_contrasts.csv",
        index=False,
    )
    capacity_audit(run, designs).to_csv(
        out / "capacity_audit.csv",
        index=False,
    )
    _plot_pair_contrasts(
        add_count[add_count["contrast"] != "add_all_bin4_vs_count"],
        leave_one_out,
        out / "pair_contrasts.png",
        f"{label}: capacity-matched interaction blocks",
    )

    if include_design_sensitivity:
        write_json(out / "design_coverage.json", design_coverage(run))
        design_sensitivity(run, config=config).to_csv(
            out / "design_sensitivity.csv",
            index=False,
        )
    if include_mobius:
        mobius_draws, mobius_summary = bootstrap_mobius(
            run,
            repeats=config.bootstrap_repeats,
            seed=config.seed + 313,
        )
        mobius_draws.to_csv(
            out / "mobius_bootstrap_draws.csv",
            index=False,
        )
        mobius_summary.to_csv(
            out / "mobius_bootstrap_summary.csv",
            index=False,
        )
        _plot_mobius(mobius_summary, out / "mobius_intervals.png")

    manifest = {
        "analysis_schema": "observerbench.ioi.phase2.capacity.v1",
        "label": label,
        "input_run": portable_artifact_path(run.source, REPO_ROOT),
        "input_hashes": source_hashes(run.input_files, REPO_ROOT),
        "n_heads": int(len(run.heads)),
        "n_subsets": int(len(run.subset)),
        "n_prompts": int(run.prompt_drops.shape[0]),
        "bootstrap_interpretation": (
            "prompt bootstrap conditional on fixed intervention masks and "
            "ten fixed repeated-CV partitions"
        ),
        "config": asdict(config),
        "runtime": runtime_provenance(REPO_ROOT),
    }
    write_json(out / "run_manifest.json", manifest)
    (out / "report.md").write_text(
        _report_text(
            label,
            run,
            add_count,
            add_additive,
            leave_one_out,
        ),
        encoding="utf-8",
    )
    return out


def _row(frame: pd.DataFrame, contrast: str) -> pd.Series:
    hit = frame[frame["contrast"] == contrast]
    if hit.empty:
        raise ValueError(f"missing contrast {contrast}")
    return hit.iloc[0]


def write_combined_claim_audit(
    stage2b_out: str | Path,
    stage2c_out: str | Path,
    outdir: str | Path,
) -> Path:
    stage2b_out = Path(stage2b_out)
    stage2c_out = Path(stage2c_out)
    out = Path(outdir)
    b_count = pd.read_csv(stage2b_out / "add_one_vs_count_additive.csv")
    b_add = pd.read_csv(stage2b_out / "add_one_vs_additive_head.csv")
    b_loo = pd.read_csv(stage2b_out / "leave_one_out_contrasts.csv")
    c_count = pd.read_csv(stage2c_out / "add_one_vs_count_additive.csv")
    c_add = pd.read_csv(stage2c_out / "add_one_vs_additive_head.csv")
    c_loo = pd.read_csv(stage2c_out / "leave_one_out_contrasts.csv")
    mobius = pd.read_csv(
        stage2c_out / "mobius_bootstrap_summary.csv"
    ).set_index("term")
    coverage = pd.read_json(
        stage2b_out / "design_coverage.json",
        typ="series",
    )
    checks = {
        "stage2b_pe_beats_both_baselines": bool(
            _row(b_count, "add_PE_bin4_vs_count")["q05"] > 0
            and _row(b_add, "add_PE_bin4_vs_additive")["q05"] > 0
        ),
        "stage2b_pe_largest_add_one": bool(
            _row(b_count, "add_PE_bin4_vs_count")["mean"]
            > max(
                _row(b_count, "add_PB_bin4_vs_count")["mean"],
                _row(b_count, "add_BE_bin4_vs_count")["mean"],
            )
        ),
        "stage2b_pe_largest_leave_one_out": bool(
            _row(b_loo, "remove_PE_bin4")["mean"]
            > max(
                _row(b_loo, "remove_PB_bin4")["mean"],
                _row(b_loo, "remove_BE_bin4")["mean"],
            )
        ),
        "stage2c_pe_beats_both_baselines": bool(
            _row(c_count, "add_PE_bin4_vs_count")["q05"] > 0
            and _row(c_add, "add_PE_bin4_vs_additive")["q05"] > 0
        ),
        "stage2c_pe_largest_leave_one_out": bool(
            _row(c_loo, "remove_PE_bin4")["mean"]
            > max(
                _row(c_loo, "remove_PB_bin4")["mean"],
                _row(c_loo, "remove_BE_bin4")["mean"],
            )
        ),
        "stage2c_pb_is_conditional": bool(
            _row(c_add, "add_PB_bin4_vs_additive")["q05"] <= 0
            and _row(c_loo, "remove_PB_bin4")["q05"] > 0
        ),
        "direct_pe_exceeds_pb": bool(
            mobius.loc["PE_minus_PB", "q05"] > 0
        ),
        "stage2b_has_two_forced_full_pb_anchors": bool(
            int(coverage["full_pb_corner_count"]) == 2
        ),
    }
    payload = {
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "claim": (
            "Per-head additivity remains a strong baseline, but capacity-matched "
            "count interactions improve held-out prediction under broad and "
            "primary-stratified masks. PE dominates add-one, leave-one-out, and "
            "direct group-mask analyses; PB contributes conditionally."
        ),
    }
    write_json(out / "ioi_phase02_claim_audit.json", payload)
    lines = [
        "# IOI Phase-2 claim audit",
        "",
        f"All required checks pass: **{payload['all_checks_pass']}**.",
        "",
        payload["claim"],
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}"
        for name, passed in checks.items()
    )
    (out / "IOI_PHASE02_CLAIM_AUDIT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return out
