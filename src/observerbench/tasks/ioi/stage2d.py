"""IOI Stage 2d per-pair decomposition with count-additive control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Sequence

import numpy as np
import pandas as pd


STAGE2D_MODELS: tuple[str, ...] = (
    "additive_head",
    "count_additive",
    "count_plus_PB_count",
    "count_plus_PE_count",
    "count_plus_BE_count",
    "count_plus_all_pairs",
)


@dataclass
class IOIStage2dConfig:
    task: str = "ioi_stage2d"
    mode: str = "quick"
    quick: bool = True
    input_run: str = ""
    k_folds: int = 5
    bootstrap_repeats: int = 50
    ridge: float = 1e-6
    seed: int = 0
    eval_nonclean_only: bool = True


def parse_mask_bits(bits, n_heads: int) -> np.ndarray:
    s = str(bits)
    if s.endswith(".0"):
        s = s[:-2]
    s = s.zfill(n_heads)
    if len(s) > n_heads:
        s = s[-n_heads:]
    return np.asarray([int(ch) for ch in s], dtype=int)


def load_stage2c(input_run: Path):
    subset_path = input_run / "ioi_stage2c_subset_design.csv"
    heads_path = input_run / "ioi_stage2c_head_records.csv"
    drops_path = input_run / "ioi_stage2c_per_prompt_drops.csv"
    missing = [p for p in [subset_path, heads_path, drops_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(str(p) for p in missing))
    heads = pd.read_csv(heads_path)
    n_heads = len(heads)
    subset = pd.read_csv(subset_path, dtype={"mask_bits": str}).sort_values("subset_idx").reset_index(drop=True)
    if subset["subset_idx"].tolist() != list(range(len(subset))):
        raise ValueError("subset_idx must be contiguous from 0")
    masks = np.stack([parse_mask_bits(x, n_heads) for x in subset["mask_bits"]], axis=0)

    groups = heads["group"].tolist()
    for group in ["P", "B", "E"]:
        idx = [i for i, value in enumerate(groups) if value == group]
        subset[f"n_{group}"] = masks[:, idx].sum(axis=1)
        subset[f"has_{group}"] = (subset[f"n_{group}"] > 0).astype(int)
    n_p_total = max(1, int((heads.group == "P").sum()))
    n_b_total = max(1, int((heads.group == "B").sum()))
    n_e_total = max(1, int((heads.group == "E").sum()))
    subset["P_B_count"] = (subset["n_P"] / n_p_total) * (subset["n_B"] / n_b_total)
    subset["P_E_count"] = (subset["n_P"] / n_p_total) * (subset["n_E"] / n_e_total)
    subset["B_E_count"] = (subset["n_B"] / n_b_total) * (subset["n_E"] / n_e_total)
    subset["P_B"] = ((subset["n_P"] > 0) & (subset["n_B"] > 0)).astype(int)
    subset["P_E"] = ((subset["n_P"] > 0) & (subset["n_E"] > 0)).astype(int)
    subset["B_E"] = ((subset["n_B"] > 0) & (subset["n_E"] > 0)).astype(int)

    long = pd.read_csv(drops_path)
    mat = long.pivot_table(index="prompt_idx", columns="subset_idx", values="drop_from_clean", aggfunc="first")
    mat = mat.reindex(columns=list(range(len(subset))))
    if mat.isnull().any().any():
        raise ValueError("per-prompt drop matrix contains NaNs after pivot")
    drops = mat.to_numpy(float)
    y = drops.mean(axis=0)
    return heads, subset, masks, drops, y


def count_bin_features(n_p: int, n_b: int, n_e: int) -> dict[str, float]:
    feats: dict[str, float] = {}
    for k in [1, 2, 3]:
        feats[f"P_count_eq_{k}"] = float(n_p == k)
    feats.update(
        {
            "B_count_eq_1": float(n_b == 1),
            "B_count_2_3": float(2 <= n_b <= 3),
            "B_count_4_5": float(4 <= n_b <= 5),
            "B_count_6_8": float(6 <= n_b <= 8),
        }
    )
    for k in [1, 2]:
        feats[f"E_count_eq_{k}"] = float(n_e == k)
    return feats


def build_design(masks: np.ndarray, heads: pd.DataFrame, subset: pd.DataFrame, model: str):
    head_labels = heads["label"].tolist()
    count_cols = [
        "P_count_eq_1",
        "P_count_eq_2",
        "P_count_eq_3",
        "B_count_eq_1",
        "B_count_2_3",
        "B_count_4_5",
        "B_count_6_8",
        "E_count_eq_1",
        "E_count_eq_2",
    ]
    cols = ["intercept"] + head_labels
    if model == "additive_head":
        pass
    elif model == "count_additive":
        cols += count_cols
    elif model == "count_plus_PB_count":
        cols += count_cols + ["P_B_count"]
    elif model == "count_plus_PE_count":
        cols += count_cols + ["P_E_count"]
    elif model == "count_plus_BE_count":
        cols += count_cols + ["B_E_count"]
    elif model == "count_plus_all_pairs":
        cols += count_cols + ["P_B_count", "P_E_count", "B_E_count"]
    else:
        raise ValueError(model)

    rows = []
    for row_idx, mask in enumerate(masks):
        n_p = int(subset.loc[row_idx, "n_P"])
        n_b = int(subset.loc[row_idx, "n_B"])
        n_e = int(subset.loc[row_idx, "n_E"])
        feats: dict[str, float] = {"intercept": 1.0}
        for bit, label in zip(mask, head_labels):
            feats[label] = float(bit)
        feats.update(count_bin_features(n_p, n_b, n_e))
        feats["P_B_count"] = float(subset.loc[row_idx, "P_B_count"])
        feats["P_E_count"] = float(subset.loc[row_idx, "P_E_count"])
        feats["B_E_count"] = float(subset.loc[row_idx, "B_E_count"])
        rows.append([feats.get(col, 0.0) for col in cols])
    return np.asarray(rows, dtype=float), cols


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    reg = ridge * np.eye(X.shape[1], dtype=float)
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if n <= 2:
        return [(idx, idx)]
    nonclean = idx[1:]
    rng.shuffle(nonclean)
    k = max(2, min(k, len(nonclean)))
    folds = np.array_split(nonclean, k)
    return [(np.setdiff1d(idx, test), test) for test in folds if len(test)]


def eval_predictions(y: np.ndarray, preds: np.ndarray, eval_nonclean: bool = True):
    mask = np.isfinite(preds)
    if eval_nonclean and len(mask):
        mask[0] = False
    yy = y[mask]
    pp = preds[mask]
    mae = float(np.mean(np.abs(pp - yy)))
    rmse = float(np.sqrt(np.mean((pp - yy) ** 2)))
    denom = float(np.sum((yy - yy.mean()) ** 2))
    r2 = float(1.0 - np.sum((pp - yy) ** 2) / denom) if denom > 1e-12 else float("nan")
    return mae, rmse, r2


def kfold_predict(masks, heads, subset, y, model, ridge=1e-6, k_folds=5, seed=0, eval_nonclean=True):
    X, cols = build_design(masks, heads, subset, model)
    preds = np.full(len(y), np.nan, dtype=float)
    rows = []
    for fold, (train, test) in enumerate(kfold_indices(len(y), k_folds, seed=seed)):
        beta = ridge_fit(X[train], y[train], ridge)
        pred = X[test] @ beta
        preds[test] = pred
        for idx, value in zip(test, pred):
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "subset_idx": int(idx),
                    "observed": float(y[idx]),
                    "predicted": float(value),
                    "error": float(value - y[idx]),
                }
            )
    mae, rmse, r2 = eval_predictions(y, preds, eval_nonclean=eval_nonclean)
    return pd.DataFrame(rows), {
        "model": model,
        "n_rows": int(np.isfinite(preds[1:] if eval_nonclean else preds).sum()),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "columns": ",".join(cols),
        "n_params": len(cols),
    }


def fit_all(masks, heads, subset, y, model, ridge=1e-6):
    X, cols = build_design(masks, heads, subset, model)
    beta = ridge_fit(X, y, ridge)
    return pd.DataFrame({"model": model, "term": cols, "coef": beta})


def annotate(pred: pd.DataFrame, subset: pd.DataFrame):
    meta_cols = [
        "subset_idx",
        "subset_name",
        "n_heads",
        "n_P",
        "n_B",
        "n_E",
        "has_P",
        "has_B",
        "has_E",
        "P_B",
        "P_E",
        "B_E",
        "P_B_count",
        "P_E_count",
        "B_E_count",
    ]
    return pred.merge(subset[meta_cols].copy(), on="subset_idx", how="left")


def bootstrap(drops, masks, heads, subset, models, repeats, k_folds, ridge, seed, eval_nonclean=True):
    rng = np.random.default_rng(seed)
    rows = []
    for bootstrap_idx in range(repeats):
        prompt_idx = rng.integers(0, drops.shape[0], size=drops.shape[0])
        yb = drops[prompt_idx].mean(axis=0)
        for model in models:
            _pred, metrics = kfold_predict(
                masks,
                heads,
                subset,
                yb,
                model,
                ridge=ridge,
                k_folds=k_folds,
                seed=seed + 17,
                eval_nonclean=eval_nonclean,
            )
            metrics["bootstrap"] = bootstrap_idx
            rows.append(metrics)
    boot = pd.DataFrame(rows)
    summary_rows = []
    for model, group in boot.groupby("model"):
        row = {"model": model, "n_bootstrap": len(group)}
        for col in ["mae", "rmse", "r2"]:
            values = group[col].to_numpy(float)
            row[f"{col}_mean"] = float(np.nanmean(values))
            row[f"{col}_median"] = float(np.nanmedian(values))
            row[f"{col}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_q05"] = float(np.nanquantile(values, 0.05))
            row[f"{col}_q95"] = float(np.nanquantile(values, 0.95))
        summary_rows.append(row)
    return boot, pd.DataFrame(summary_rows)


def paired_delta(boot: pd.DataFrame, baseline: str) -> pd.DataFrame:
    piv = boot.pivot_table(index="bootstrap", columns="model", values="mae", aggfunc="first")
    rows = []
    for model in STAGE2D_MODELS:
        if baseline not in piv.columns or model not in piv.columns:
            continue
        delta = (piv[baseline] - piv[model]).dropna().to_numpy(float)
        rows.append(
            {
                "baseline": baseline,
                "model": model,
                "n_bootstrap": int(len(delta)),
                "delta_mae_mean": float(np.mean(delta)),
                "delta_mae_median": float(np.median(delta)),
                "delta_mae_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "delta_mae_q05": float(np.quantile(delta, 0.05)),
                "delta_mae_q95": float(np.quantile(delta, 0.95)),
                "p_delta_gt_0": float(np.mean(delta > 0)),
                "strict_success_q05_gt_0": bool(np.quantile(delta, 0.05) > 0),
                "weak_success_p_gt_0_ge_0.95": bool(np.mean(delta > 0) >= 0.95),
            }
        )
    return pd.DataFrame(rows)


def model_comparison(boot_summary: pd.DataFrame, delta_vs_additive: pd.DataFrame, delta_vs_count: pd.DataFrame) -> pd.DataFrame:
    rows = []
    add = delta_vs_additive.set_index("model")
    count = delta_vs_count.set_index("model")
    summary = boot_summary.set_index("model")
    for model in STAGE2D_MODELS:
        row = {"model": model}
        if model in summary.index:
            row.update(summary.loc[model].to_dict())
        for source, prefix in [
            (add, "delta_mae_vs_additive"),
            (count, "delta_mae_vs_count_additive"),
        ]:
            if model in source.index:
                item = source.loc[model]
                row[f"{prefix}_mean"] = float(item["delta_mae_mean"])
                row[f"{prefix}_q05"] = float(item["delta_mae_q05"])
                row[f"{prefix}_q95"] = float(item["delta_mae_q95"])
                p_name = "p_delta_vs_additive_gt_0" if prefix == "delta_mae_vs_additive" else "p_delta_vs_count_additive_gt_0"
                row[p_name] = float(item["p_delta_gt_0"])
            else:
                row[f"{prefix}_mean"] = 0.0
                row[f"{prefix}_q05"] = 0.0
                row[f"{prefix}_q95"] = 0.0
                p_name = "p_delta_vs_additive_gt_0" if prefix == "delta_mae_vs_additive" else "p_delta_vs_count_additive_gt_0"
                row[p_name] = 0.0
        row["beats_additive_head"] = bool(row["delta_mae_vs_additive_q05"] > 0.0 and row["p_delta_vs_additive_gt_0"] >= 0.95)
        row["beats_count_additive"] = bool(row["delta_mae_vs_count_additive_q05"] > 0.0 and row["p_delta_vs_count_additive_gt_0"] >= 0.95)
        # A model is not a main success merely for beating count_additive.
        row["main_success"] = bool(row["beats_additive_head"] and row["beats_count_additive"])
        rows.append(row)
    return pd.DataFrame(rows)


def group_interactions(subset: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    totals = {"P": int(subset["n_P"].max()), "B": int(subset["n_B"].max()), "E": int(subset["n_E"].max())}

    def lookup(n_p: int, n_b: int, n_e: int) -> float:
        hit = subset[(subset["n_P"] == n_p) & (subset["n_B"] == n_b) & (subset["n_E"] == n_e)]
        if hit.empty:
            return float("nan")
        return float(y[int(hit.iloc[0]["subset_idx"])])

    clean = lookup(0, 0, 0)
    drops = {
        "P": lookup(totals["P"], 0, 0),
        "B": lookup(0, totals["B"], 0),
        "E": lookup(0, 0, totals["E"]),
        "PB": lookup(totals["P"], totals["B"], 0),
        "PE": lookup(totals["P"], 0, totals["E"]),
        "BE": lookup(0, totals["B"], totals["E"]),
    }
    rows = []
    for pair, left, right in [("PB", "P", "B"), ("PE", "P", "E"), ("BE", "B", "E")]:
        interaction = drops[pair] - drops[left] - drops[right] + clean
        rows.append(
            {
                "pair": pair,
                "drop_pair": drops[pair],
                f"drop_{left}": drops[left],
                f"drop_{right}": drops[right],
                "interaction": interaction,
            }
        )
    return pd.DataFrame(rows)


def _plot_bar(df: pd.DataFrame, outpath: Path, y_col: str, title: str) -> None:
    import matplotlib.pyplot as plt

    ordered = df.sort_values(y_col)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ordered["model"], ordered[y_col])
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def run_ioi_stage2d(cfg: IOIStage2dConfig, outdir: str | Path, input_run: str | Path | None = None) -> Path:
    source = Path(input_run or cfg.input_run)
    if not str(source):
        raise ValueError("ioi_stage2d requires --input-run or config input_run")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    heads, subset, masks, drops, y = load_stage2c(source)
    heads.to_csv(out / "ioi_stage2d_head_records.csv", index=False)
    subset.to_csv(out / "ioi_stage2d_subset_design.csv", index=False)

    pred_parts = []
    metrics = []
    coef_parts = []
    for model in STAGE2D_MODELS:
        pred, met = kfold_predict(
            masks,
            heads,
            subset,
            y,
            model,
            ridge=cfg.ridge,
            k_folds=cfg.k_folds,
            seed=cfg.seed + 777,
            eval_nonclean=cfg.eval_nonclean_only,
        )
        pred_parts.append(annotate(pred, subset))
        metrics.append(met)
        coef_parts.append(fit_all(masks, heads, subset, y, model, ridge=cfg.ridge))

    pred = pd.concat(pred_parts, ignore_index=True)
    fit_summary = pd.DataFrame(metrics)
    coefs = pd.concat(coef_parts, ignore_index=True)
    boot, boot_summary = bootstrap(
        drops,
        masks,
        heads,
        subset,
        STAGE2D_MODELS,
        repeats=cfg.bootstrap_repeats,
        k_folds=cfg.k_folds,
        ridge=cfg.ridge,
        seed=cfg.seed + 999,
        eval_nonclean=cfg.eval_nonclean_only,
    )
    delta_vs_additive = paired_delta(boot, "additive_head")
    delta_vs_count = paired_delta(boot, "count_additive")
    comparison = model_comparison(boot_summary, delta_vs_additive, delta_vs_count)
    group_df = group_interactions(subset, y)

    pred.to_csv(out / "ioi_stage2d_kfold_predictions.csv", index=False)
    fit_summary.merge(comparison, on="model", how="left").to_csv(out / "ioi_stage2d_fit_summary.csv", index=False)
    coefs.to_csv(out / "ioi_stage2d_coefficients.csv", index=False)
    boot.to_csv(out / "ioi_stage2d_bootstrap_metrics.csv", index=False)
    boot_summary.to_csv(out / "ioi_stage2d_bootstrap_summary.csv", index=False)
    delta_vs_count.to_csv(out / "ioi_stage2d_paired_delta_vs_count_additive.csv", index=False)
    delta_vs_additive.to_csv(out / "ioi_stage2d_paired_delta_vs_additive.csv", index=False)
    comparison.to_csv(out / "ioi_stage2d_model_comparison.csv", index=False)
    group_df.to_csv(out / "ioi_stage2d_group_interactions.csv", index=False)

    diag = {
        "input_run": str(source),
        "n_subsets": int(len(subset)),
        "n_heads": int(len(heads)),
        "n_prompts": int(drops.shape[0]),
        "models": list(STAGE2D_MODELS),
        "baseline_primary": "additive_head",
        "success_rule": "main_success requires beating additive_head and count_additive",
        "config": asdict(cfg),
    }
    for _, row in comparison.iterrows():
        diag[f"{row['model']}_main_success"] = bool(row["main_success"])
    for term in ["P_B_count", "P_E_count", "B_E_count"]:
        hit = coefs[(coefs.model == "count_plus_all_pairs") & (coefs.term == term)]
        if not hit.empty:
            diag[f"coef_all_pairs_{term}"] = float(hit.coef.iloc[0])
    (out / "ioi_stage2d_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")

    _plot_bar(fit_summary, out / "ioi_stage2d_mae_bar.png", "mae", "IOI Stage 2d held-out MAE")
    _plot_bar(comparison, out / "ioi_stage2d_delta_vs_additive.png", "delta_mae_vs_additive_mean", "Delta MAE vs additive_head")
    (out / "ioi_stage2d_report.md").write_text(
        "# IOI Stage 2d per-pair decomposition\n\n"
        "This postprocesses a Stage 2c run without rerunning GPT-2. "
        "A model is not labeled a main success unless it beats both additive_head and count_additive.\n",
        encoding="utf-8",
    )
    return out


def available_design_columns() -> dict[str, Sequence[str]]:
    heads = pd.DataFrame(
        [
            {"label": "P:9.9", "group": "P"},
            {"label": "B:9.0", "group": "B"},
            {"label": "E:10.7", "group": "E"},
        ]
    )
    subset = pd.DataFrame(
        [
            {"n_P": 0, "n_B": 0, "n_E": 0, "P_B_count": 0.0, "P_E_count": 0.0, "B_E_count": 0.0},
            {"n_P": 1, "n_B": 1, "n_E": 1, "P_B_count": 1.0, "P_E_count": 1.0, "B_E_count": 1.0},
        ]
    )
    masks = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=int)
    return {model: build_design(masks, heads, subset, model)[1] for model in STAGE2D_MODELS}
