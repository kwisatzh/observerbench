"""Measure and score raw and scalar-calibrated IOI attribution patching.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from observerbench import run_effect_prediction_task
from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance, source_hashes
from observerbench.tasks.ioi import (
    DEFAULT_IOI_EFFECT_ARTIFACT_ROOT,
    IOI_EFFECT_MEASUREMENT_BUDGETS,
    IOI_EFFECT_MODEL_REVISION,
    IOIAttributionPatchingBaseline,
    ioi_attribution_patching_card,
    load_ioi_effect_prediction_task,
    load_template_head_means,
    measure_ioi_attribution_map,
)


ROOT = Path(__file__).resolve().parents[1]
MAP_SCHEMA = "observerbench.ioi_attribution_patching_run.v1"
SUMMARY_FIELDS = (
    "observer",
    "measurement_budget",
    "fitted_scalar_gain",
    "n_queries",
    "mae",
    "rmse",
    "mean_error",
    "max_absolute_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_IOI_EFFECT_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/revision/published_baselines/ioi_attribution_patching_v1",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--measurement-budgets",
        type=int,
        nargs="+",
        default=IOI_EFFECT_MEASUREMENT_BUDGETS,
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Engineering smoke only: measure a prefix of train prompts.",
    )
    return parser.parse_args()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    unsupported = sorted(set(args.measurement_budgets) - set(IOI_EFFECT_MEASUREMENT_BUDGETS))
    if unsupported:
        raise ValueError(f"unsupported measurement budgets: {unsupported}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        revision=IOI_EFFECT_MODEL_REVISION,
        device=args.device,
    )
    model.eval()
    means_path = args.artifacts_root / "ioi_effects/template_head_means.npz"
    means, templates = load_template_head_means(means_path)
    attribution_map = measure_ioi_attribution_map(
        model,
        args.artifacts_root / "design",
        means,
        templates,
        model_revision=IOI_EFFECT_MODEL_REVISION,
        batch_size=args.batch_size,
        max_prompts=args.max_prompts,
    )
    map_path = args.outdir / "attribution_map.json"
    write_json(map_path, attribution_map.to_dict())

    summary_rows: list[dict[str, Any]] = []
    result_payloads: list[dict[str, Any]] = []
    for budget in args.measurement_budgets:
        task = load_ioi_effect_prediction_task(
            args.artifacts_root,
            measurement_budget=budget,
        )
        for calibrated in (False, True):
            predictor = IOIAttributionPatchingBaseline(
                attribution_map,
                calibrate_scalar=calibrated,
            )
            predictor.fit(tuple(task.measurements))
            card = ioi_attribution_patching_card(
                calibrated=calibrated,
                attribution_map=attribution_map,
                fitted_gain=predictor.gain_,
            )
            observer_dir = "scalar_calibrated" if calibrated else "raw"
            row_outdir = args.outdir / observer_dir / f"b{budget:03d}"
            result = run_effect_prediction_task(
                task,
                predictor,
                card,
                outdir=row_outdir,
            )
            row = {
                "observer": predictor.name,
                "measurement_budget": budget,
                "fitted_scalar_gain": predictor.gain_,
                "n_queries": result.n_queries,
                **result.metrics,
            }
            summary_rows.append(row)
            result_payloads.append(result.to_dict())

    summary_path = args.outdir / "summary.csv"
    _write_summary(summary_path, summary_rows)
    write_json(args.outdir / "summary.json", summary_rows)
    source_paths = (
        Path(__file__),
        ROOT / "src/observerbench/tasks/ioi/attribution_patching.py",
        ROOT / "src/observerbench/tasks/ioi/effect_task.py",
        ROOT / "src/observerbench/effect_prediction.py",
    )
    manifest = {
        "schema": MAP_SCHEMA,
        "status": "engineering_smoke" if args.max_prompts is not None else "complete",
        "claim_eligible": args.max_prompts is None,
        "model": {
            "name": "gpt2-small",
            "revision": IOI_EFFECT_MODEL_REVISION,
            "device": args.device,
        },
        "measurement": {
            "prompt_split": attribution_map.prompt_split,
            "n_gradient_prompts": attribution_map.n_prompts,
            "batch_size": args.batch_size,
            "intervention": attribution_map.intervention,
            "template_head_means": {
                "path": means_path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(means_path),
            },
        },
        "budgets": list(args.measurement_budgets),
        "results": result_payloads,
        "artifacts": {
            "attribution_map.json": file_sha256(map_path),
            "summary.csv": file_sha256(summary_path),
        },
        "sources": source_hashes(source_paths, ROOT),
        "runtime": runtime_provenance(ROOT),
    }
    write_json(args.outdir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
