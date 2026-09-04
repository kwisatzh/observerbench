"""Source-problem uncertainty for a frozen rare-attack audit comparison.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

This composes the existing deployment evaluator with a paired outer bootstrap.
It does not refit observers, change their scores, or select a deployment cell.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from analyze_controlarena_apps_bootstrap import _auroc
from observerbench.ai_control import read_ai_control_scores
from observerbench.ai_control_followup import prevalence_budget_sensitivity
from observerbench.core import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("LEFT", "RIGHT"))
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--inner-replicates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9042026)
    args = parser.parse_args()
    if args.replicates < 2 or args.inner_replicates < 2:
        parser.error("Both replicate counts must be at least two")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = read_ai_control_scores(args.scores)
    names = sorted({name for pair in args.pair for name in pair})
    index = {(r.monitor_name, r.sample_id): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate monitor/sample rows")
    problems = sorted({r.sample_id.rsplit("::", 1)[0] for r in rows})
    scores = {}
    for name in names:
        scores[name] = {}
        for mode in ("honest", "attack"):
            selected = [index[name, f"{problem}::{mode}"] for problem in problems]
            if any(r.backdoor_works != (mode == "attack") for r in selected):
                raise ValueError("This comparison requires verified working attacks")
            scores[name][mode] = np.asarray([r.score for r in selected])
    n = len(problems)
    labels = np.r_[np.zeros(n, dtype=bool), np.ones(n, dtype=bool)]
    rng = np.random.default_rng(args.seed)
    draws = {name: {metric: [] for metric in ("auroc", "violations")} for name in names}
    original = {}

    for replicate in range(args.replicates + 1):
        chosen = np.arange(n) if replicate == 0 else rng.integers(0, n, n)
        panel = {name: {mode: scores[name][mode][chosen]
                        for mode in ("honest", "attack")} for name in names}
        # All monitors see identical deployment draws and shared random tie order.
        result = prevalence_budget_sensitivity(
            panel, prevalences=(0.01,), budgets=(0.02,), population_size=1488,
            replicates=2000 if replicate == 0 else args.inner_replicates,
            seed=7411 if replicate == 0 else int(rng.integers(0, 2**31)),
            status="post_review_source_problem_bootstrap",
        )["cells"]["prevalence=0.01|budget=0.02"]["monitors"]
        for name in names:
            metrics = {
                "auroc": _auroc(labels, np.r_[panel[name]["honest"], panel[name]["attack"]]),
                "violations": result[name]["realized_violations"]["mean"],
            }
            if replicate == 0:
                original[name] = metrics
            else:
                for metric, value in metrics.items():
                    draws[name][metric].append(value)
        if replicate % 50 == 0:
            print(f"{replicate}/{args.replicates} problem bootstraps", flush=True)

    contrasts = []
    for left, right in args.pair:
        contrast = {"left": left, "right": right, "sign": "left minus right"}
        for metric in ("auroc", "violations"):
            values = np.asarray(draws[left][metric]) - np.asarray(draws[right][metric])
            contrast[metric] = {
                "original_difference": original[left][metric] - original[right][metric],
                "bootstrap_mean_difference": float(values.mean()),
                "percentile_interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
            }
        contrasts.append(contrast)

    write_json(args.out, {
        "schema": "observerbench.apps_source_uncertainty.v1",
        "status": "secondary analysis specified after observing the original rankings",
        "score_file_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        "n_source_problem_pairs": n,
        "outer_resampling_unit": "APPS problem with paired honest and attack rows; same draw for all monitors",
        "replicates": args.replicates, "inner_replicates": args.inner_replicates,
        "seed": args.seed,
        "deployment": {"population": 1488, "attacks": 15, "audits": 29},
        "interval_scope": "Approximate percentile intervals over source problems, conditional on frozen fits and splits. Inner Monte Carlo adds numerical uncertainty; intervals exclude refitting, model selection, and adaptive attacks.",
        "original_panel": original, "contrasts": contrasts,
    })


if __name__ == "__main__":
    main()
