#!/usr/bin/env python3
"""Measure and score raw and scalar-calibrated Qwen attribution patching."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from observerbench import run_effect_prediction_task
from observerbench.core import write_json
from observerbench.provenance import file_sha256, runtime_provenance, source_hashes
from observerbench.tasks.qwen_induction import (
    QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS,
    QWEN_INDUCTION_MODEL_NAME,
    QWEN_INDUCTION_MODEL_REVISION,
    QwenInductionAttributionPatchingBaseline,
    load_qwen_induction_effect_prediction_task,
    load_qwen_reference_means,
    load_qwen_train_prompts,
    measure_qwen_induction_attribution_map,
    qwen_induction_attribution_patching_card,
)
from observerbench.tasks.qwen_induction.plant import Qwen2InductionPlant


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COPY_V2_ROOT = (
    ROOT
    / "results"
    / "revision"
    / "phase10"
    / "qwen_induction_copy_v2_complete"
    / "copy_v2"
)
DEFAULT_OUTDIR = (
    ROOT
    / "results"
    / "revision"
    / "published_baselines"
    / "qwen_induction_attribution_patching_v1"
)
RUN_SCHEMA = "observerbench.qwen_induction_attribution_patching_run.v1"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_COPY_V2_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--measurement-budgets",
        type=int,
        nargs="+",
        default=QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS,
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Engineering smoke only: measure a prefix of public train prompts.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network model downloads.",
    )
    return parser.parse_args()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    unsupported = sorted(
        set(args.measurement_budgets)
        - set(QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS)
    )
    if unsupported:
        raise ValueError(f"unsupported measurement budgets: {unsupported}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("max-prompts must be positive when supplied")

    design_dir = args.artifacts_root / "design"
    prompts_path = design_dir / "prompts.csv"
    means_path = (
        args.artifacts_root
        / "work"
        / "confirmation"
        / "reference_selected_means.npz"
    )
    required = (
        design_dir / "design_manifest.json",
        prompts_path,
        design_dir / "selected_heads.csv",
        means_path,
        args.artifacts_root / "effects" / "effect_manifest.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required frozen Copy-v2 artifact is missing: {path}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    prompts = load_qwen_train_prompts(prompts_path, max_prompts=args.max_prompts)
    means = load_qwen_reference_means(means_path)
    plant = Qwen2InductionPlant.from_pretrained(
        QWEN_INDUCTION_MODEL_NAME,
        QWEN_INDUCTION_MODEL_REVISION,
        device=args.device,
        dtype="bfloat16",
        attention_implementation="sdpa",
        local_files_only=args.local_files_only,
    )
    plant_audit = plant.audit_runtime(
        expected_model_id=QWEN_INDUCTION_MODEL_NAME,
        expected_revision=QWEN_INDUCTION_MODEL_REVISION,
        expected_layers=28,
        expected_query_heads=28,
        expected_kv_heads=4,
        expected_dtype="bfloat16",
        expected_attention_implementation="sdpa",
    )
    attribution_map = measure_qwen_induction_attribution_map(
        plant,
        prompts,
        means,
        model_revision=QWEN_INDUCTION_MODEL_REVISION,
        batch_size=args.batch_size,
    )
    map_path = args.outdir / "attribution_map.json"
    write_json(map_path, attribution_map.to_dict())

    # The local gradient map is written before the cached evaluator opens any
    # held-out effects. The run remains retrospective because Copy-v2 outcomes
    # existed before this published-method baseline was designed.
    summary_rows: list[dict[str, Any]] = []
    result_payloads: list[dict[str, Any]] = []
    for budget in args.measurement_budgets:
        task = load_qwen_induction_effect_prediction_task(
            args.artifacts_root,
            measurement_budget=budget,
            expected_data_version="copy-v2",
        )
        for calibrated in (False, True):
            predictor = QwenInductionAttributionPatchingBaseline(
                attribution_map,
                calibrate_scalar=calibrated,
            )
            predictor.fit(tuple(task.measurements))
            card = qwen_induction_attribution_patching_card(
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
        ROOT
        / "src"
        / "observerbench"
        / "tasks"
        / "qwen_induction"
        / "attribution_patching.py",
        ROOT
        / "src"
        / "observerbench"
        / "tasks"
        / "qwen_induction"
        / "effect_task.py",
        ROOT
        / "src"
        / "observerbench"
        / "tasks"
        / "qwen_induction"
        / "plant.py",
        ROOT / "src" / "observerbench" / "effect_prediction.py",
    )
    manifest = {
        "schema": RUN_SCHEMA,
        "status": "engineering_smoke" if args.max_prompts is not None else "complete",
        "claim_eligible": args.max_prompts is None,
        "result_status": "post-outcome published-method baseline",
        "model": {
            "name": QWEN_INDUCTION_MODEL_NAME,
            "revision": QWEN_INDUCTION_MODEL_REVISION,
            "device": str(plant.device),
            "audit": plant_audit,
        },
        "measurement": {
            "prompt_split": attribution_map.prompt_split,
            "n_gradient_prompts": attribution_map.n_prompts,
            "batch_size": args.batch_size,
            "intervention": attribution_map.intervention,
            "autograd_scope": "frozen weights; suffix beginning at earliest selected layer",
            "reference_means": {
                "path": means_path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(means_path),
            },
            "prompts": {
                "path": prompts_path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(prompts_path),
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
