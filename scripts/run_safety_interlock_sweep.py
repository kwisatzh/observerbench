"""Run the controlled safety interlock across frozen random seeds.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from observerbench.tasks.safety_interlock import SafetyInterlockConfig, run_safety_interlock


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--n-seeds", type=int, default=20)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.n_seeds <= 0:
        raise ValueError("n-seeds must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.n_seeds):
        results = run_safety_interlock(
            SafetyInterlockConfig(seed=seed),
            args.outdir / "runs" / f"seed-{seed:03d}",
        )
        for result in results:
            delete = result.family_metrics["delete"]
            rows.append(
                {
                    "seed": seed,
                    "observer": result.observer_name,
                    "protocol_loss_mean": result.metrics["protocol_loss_mean"],
                    "protocol_loss_cvar": result.metrics["protocol_loss_cvar"],
                    "risk_auroc": result.metrics["risk_auroc"],
                    "severity_weighted_miss_rate": result.metrics["severity_weighted_miss_rate"],
                    "clean_utility_retained": result.metrics["clean_utility_retained"],
                    "heldout_delete_protocol_loss": delete["protocol_loss_mean"],
                    "heldout_delete_severity_miss": delete["severity_weighted_miss_rate"],
                }
            )
    result_path = args.outdir / "safety_interlock_seed_results.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_observer = {str(row["observer"]) for row in rows}
    summary: dict[str, dict[str, dict[str, float]]] = {}
    metrics = (
        "protocol_loss_mean",
        "protocol_loss_cvar",
        "risk_auroc",
        "severity_weighted_miss_rate",
        "clean_utility_retained",
        "heldout_delete_protocol_loss",
        "heldout_delete_severity_miss",
    )
    for observer in sorted(by_observer):
        observer_rows = [row for row in rows if row["observer"] == observer]
        summary[observer] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in observer_rows], dtype=float)
            summary[observer][metric] = {
                "mean": float(np.mean(values)),
                "seed_min": float(np.min(values)),
                "seed_max": float(np.max(values)),
            }

    indexed = {(int(row["seed"]), str(row["observer"])): row for row in rows}
    direct_improvements = []
    heldout_loss_reductions = []
    for seed in range(args.n_seeds):
        label = indexed[(seed, "activation-label-ridge")]
        direct = indexed[(seed, "activation-direct-risk-ridge")]
        direct_improvements.append(
            1.0 - float(direct["protocol_loss_mean"]) / float(label["protocol_loss_mean"])
        )
        heldout_loss_reductions.append(
            float(label["heldout_delete_protocol_loss"])
            - float(direct["heldout_delete_protocol_loss"])
        )
    payload = {
        "schema": "observerbench.safety_interlock_seed_summary.v0",
        "n_seeds": args.n_seeds,
        "heldout_operation": "delete",
        "observer_metrics": summary,
        "direct_risk_vs_label": {
            "mean_protocol_loss_reduction": float(np.mean(direct_improvements)),
            "seed_min_protocol_loss_reduction": float(np.min(direct_improvements)),
            "seed_max_protocol_loss_reduction": float(np.max(direct_improvements)),
            "mean_heldout_delete_absolute_loss_reduction": float(np.mean(heldout_loss_reductions)),
            "seed_min_heldout_delete_absolute_loss_reduction": float(np.min(heldout_loss_reductions)),
            "seed_max_heldout_delete_absolute_loss_reduction": float(np.max(heldout_loss_reductions)),
            "heldout_delete_win_fraction": float(np.mean(np.asarray(heldout_loss_reductions) > 0.0)),
        },
    }
    (args.outdir / "safety_interlock_seed_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.outdir / "safety_interlock_seed_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
