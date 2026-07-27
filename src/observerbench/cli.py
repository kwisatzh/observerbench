"""Command line interface for the ObserverBench reproduction workbench."""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from observerbench.cards import write_observer_card_bundle
from observerbench.config import load_config
from observerbench.effect_prediction import EffectObserverCard, evaluate_effect_prediction_csv
from observerbench.tasks.effect_registry import (
    finite_effect_task_specs,
    load_finite_effect_task,
)
from observerbench.tasks.registry import run_registered_task, task_names, task_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="observerbench",
        description="ObserverBench paper reproduction CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tasks", help="List paper reproduction tasks.")
    subparsers.add_parser(
        "list-effect-tasks",
        help="List exact inference-free finite-effect task versions.",
    )

    effect_parser = subparsers.add_parser(
        "evaluate-effect-csv",
        help="Evaluate an outside finite-effect prediction table.",
    )
    effect_parser.add_argument("task_id")
    effect_parser.add_argument("--artifacts-root", required=True, type=Path)
    effect_parser.add_argument("--predictions", required=True, type=Path)
    effect_parser.add_argument("--observer-card", required=True, type=Path)
    effect_parser.add_argument("--outdir", required=True, type=Path)
    effect_parser.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help="Check required files and schemas but skip SHA-256 verification.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a registered paper task.",
    )
    run_parser.add_argument("task_name", choices=task_names())
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--outdir", required=True, type=Path)
    run_parser.add_argument("--quick", action="store_true", help="Force the task's quick/smoke mode.")
    run_parser.add_argument("--input-run", type=Path, default=None, help="Existing run directory for postprocess tasks.")

    card_parser = subparsers.add_parser(
        "make-card",
        help="Generate ObserverCard JSON and Markdown from an existing result path.",
    )
    card_parser.add_argument("--results", required=True, type=Path)
    card_parser.add_argument("--outdir", required=True, type=Path)

    figure_parser = subparsers.add_parser(
        "make-figures",
        help="Generate a placeholder figure manifest from an existing result directory.",
    )
    figure_parser.add_argument("--results", required=True, type=Path)
    figure_parser.add_argument("--outdir", required=True, type=Path)

    return parser


def list_tasks() -> int:
    for spec in task_specs():
        capability = "\texternal-observer:v0" if spec.supports_external_observer else ""
        print(f"{spec.name}\t{spec.summary}{capability}")
    return 0


def list_effect_tasks() -> int:
    for spec in finite_effect_task_specs():
        print(
            f"{spec.task_id}\tbudget:{spec.measurement_budget}\t{spec.summary}"
        )
    return 0


def evaluate_effect_csv(
    task_id: str,
    artifacts_root: Path,
    predictions: Path,
    observer_card_path: Path,
    outdir: Path,
    *,
    verify_hashes: bool,
) -> int:
    payload = json.loads(observer_card_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observer card must be a JSON object")
    payload.pop("schema_version", None)
    observer_card = EffectObserverCard(**payload)
    task = load_finite_effect_task(
        task_id,
        artifacts_root=artifacts_root,
        verify_hashes=verify_hashes,
    )
    result = evaluate_effect_prediction_csv(
        task,
        predictions,
        observer_card,
        outdir=outdir,
    )
    print(outdir / "effect_evaluation.json")
    return 0 if result.n_queries == len(task.queries) else 1


def make_card(results: Path, outdir: Path) -> int:
    json_path, md_path = write_observer_card_bundle(results, outdir)
    print(json_path)
    print(md_path)
    return 0


def make_figures(results: Path, outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "observerbench.figures.placeholder.v0",
        "source_results": str(results),
        "status": "placeholder",
        "note": (
            "Figure generation is scaffolded only; paper figure rendering will "
            "be wired to frozen outputs during migration."
        ),
    }
    manifest_path = outdir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


def run_task(task_name: str, config_path: Path, outdir: Path, quick: bool = False, input_run: Path | None = None) -> int:
    config = load_config(config_path)
    if quick:
        config["quick"] = True
        config["mode"] = "quick"
    if input_run is not None:
        config["input_run"] = str(input_run)
    run_registered_task(task_name, config, config_path, outdir)
    print(outdir)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-tasks":
        return list_tasks()
    if args.command == "list-effect-tasks":
        return list_effect_tasks()
    if args.command == "evaluate-effect-csv":
        return evaluate_effect_csv(
            args.task_id,
            args.artifacts_root,
            args.predictions,
            args.observer_card,
            args.outdir,
            verify_hashes=not args.no_verify_hashes,
        )
    if args.command == "run":
        return run_task(args.task_name, args.config, args.outdir, quick=args.quick, input_run=args.input_run)
    if args.command == "make-card":
        return make_card(args.results, args.outdir)
    if args.command == "make-figures":
        return make_figures(args.results, args.outdir)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
