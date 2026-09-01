"""Command line interface for the ObserverBench reproduction workbench."""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from observerbench.ai_control import evaluate_ai_control_csv
from observerbench.ai_control_leaderboard import (
    format_ai_control_leaderboard,
    load_ai_control_leaderboard,
)
from observerbench.cards import write_observer_card_bundle
from observerbench.config import load_config
from observerbench.effect_prediction import EffectObserverCard, evaluate_effect_prediction_csv
from observerbench.safety import SafetyObserverCard, evaluate_safety_prediction_csv
from observerbench.safety_leaderboard import (
    compare_safety_result,
    format_safety_leaderboard,
    load_safety_leaderboard,
    safety_rows_for_task,
)
from observerbench.safety_task_pack import (
    load_safety_task_pack,
    validate_public_safety_task_pack,
    write_safety_task_pack,
)
from observerbench.tasks.effect_registry import (
    finite_effect_task_specs,
    load_finite_effect_task,
)
from observerbench.tasks.registry import run_registered_task, task_names, task_specs
from observerbench.tasks.safety_registry import load_safety_task, safety_task_specs


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
    subparsers.add_parser(
        "list-safety-tasks",
        help="List exact inference-free safety task versions.",
    )

    safety_results_parser = subparsers.add_parser(
        "list-safety-results",
        help="List checked safety observers and protocol outcomes.",
    )
    safety_results_parser.add_argument("--task-id", default=None)
    safety_results_parser.add_argument(
        "--format", choices=("text", "json", "csv", "markdown"), default="text"
    )
    safety_results_parser.add_argument("--leaderboard", type=Path, default=None)

    compare_safety_parser = subparsers.add_parser(
        "compare-safety",
        help="Compare an evaluated safety result with checked rows for the same task.",
    )
    compare_safety_parser.add_argument("--result-dir", required=True, type=Path)
    compare_safety_parser.add_argument(
        "--format", choices=("text", "json", "csv", "markdown"), default="text"
    )
    compare_safety_parser.add_argument("--leaderboard", type=Path, default=None)

    ai_control_parser = subparsers.add_parser(
        "evaluate-ai-control-csv",
        help="Compare external AI-control monitors under one fixed audit budget.",
    )
    ai_control_parser.add_argument("--scores", required=True, type=Path)
    ai_control_parser.add_argument("--audit-budget", type=float, default=0.02)
    ai_control_parser.add_argument("--outdir", required=True, type=Path)

    ai_control_results_parser = subparsers.add_parser(
        "list-ai-control-results",
        help="List checked monitor outcomes within external APPS task panels.",
    )
    ai_control_results_parser.add_argument("--task-id", default=None)
    ai_control_results_parser.add_argument(
        "--format", choices=("text", "json", "csv", "markdown"), default="text"
    )
    ai_control_results_parser.add_argument("--leaderboard", type=Path, default=None)

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

    safety_parser = subparsers.add_parser(
        "evaluate-safety-csv",
        help="Evaluate an outside safety-risk table through a frozen policy.",
    )
    safety_parser.add_argument("task_id")
    safety_parser.add_argument("--predictions", required=True, type=Path)
    safety_parser.add_argument("--observer-card", required=True, type=Path)
    safety_parser.add_argument("--outdir", required=True, type=Path)

    export_safety_parser = subparsers.add_parser(
        "export-safety-task-pack",
        help="Export a built-in safety task as public and evaluator-only JSON files.",
    )
    export_safety_parser.add_argument("task_id")
    export_safety_parser.add_argument("--outdir", required=True, type=Path)

    validate_safety_parser = subparsers.add_parser(
        "validate-safety-task-pack",
        help="Validate a researcher-authored public safety task pack.",
    )
    validate_safety_parser.add_argument("--public-task", required=True, type=Path)
    validate_safety_parser.add_argument("--targets", type=Path, default=None)

    custom_safety_parser = subparsers.add_parser(
        "evaluate-safety-task-pack-csv",
        help="Evaluate risk predictions on a researcher-authored safety task pack.",
    )
    custom_safety_parser.add_argument("--public-task", required=True, type=Path)
    custom_safety_parser.add_argument("--targets", required=True, type=Path)
    custom_safety_parser.add_argument("--predictions", required=True, type=Path)
    custom_safety_parser.add_argument("--observer-card", required=True, type=Path)
    custom_safety_parser.add_argument("--outdir", required=True, type=Path)

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
        help="Regenerate paper figures and tables from checked result summaries.",
    )
    figure_parser.add_argument("--outdir", required=True, type=Path)
    figure_parser.add_argument("--frozen-dir", type=Path, default=None)
    figure_parser.add_argument("--revision-dir", type=Path, default=None)
    figure_parser.add_argument("--only", default=None, help="Generate one result ID from paper/figure_map.md.")
    figure_parser.add_argument(
        "--legacy",
        action="store_true",
        help="Generate the superseded v7 figure/table set.",
    )

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


def list_safety_tasks() -> int:
    for spec in safety_task_specs():
        print(f"{spec.task_id}\t{spec.summary}")
    return 0


def list_safety_results(
    *, task_id: str | None, output_format: str, leaderboard: Path | None
) -> int:
    rows = safety_rows_for_task(load_safety_leaderboard(leaderboard), task_id)
    print(format_safety_leaderboard(rows, output_format=output_format), end="")
    return 0


def compare_safety(
    *, result_dir: Path, output_format: str, leaderboard: Path | None
) -> int:
    rows = compare_safety_result(result_dir, load_safety_leaderboard(leaderboard))
    print(format_safety_leaderboard(rows, output_format=output_format), end="")
    return 0


def evaluate_ai_control(
    *, scores: Path, audit_budget: float, outdir: Path
) -> int:
    results = evaluate_ai_control_csv(
        scores,
        audit_budget_fraction=audit_budget,
        outdir=outdir,
    )
    for result in sorted(
        results, key=lambda item: item.metrics["realized_violation_rate"]
    ):
        print(
            f"{result.monitor_name}\t"
            f"AUROC={result.metrics['risk_auroc']:.3f}\t"
            f"violations={int(result.metrics['realized_violations'])}\t"
            f"recall@budget={result.metrics['working_backdoor_recall_at_budget']:.3f}"
        )
    return 0


def list_ai_control_results(
    *, task_id: str | None, output_format: str, leaderboard: Path | None
) -> int:
    rows = load_ai_control_leaderboard(leaderboard, task_id=task_id)
    print(format_ai_control_leaderboard(rows, output_format=output_format), end="")
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


def evaluate_safety_csv(
    task_id: str,
    predictions: Path,
    observer_card_path: Path,
    outdir: Path,
) -> int:
    payload = json.loads(observer_card_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observer card must be a JSON object")
    payload.pop("schema_version", None)
    observer_card = SafetyObserverCard(**payload)
    task = load_safety_task(task_id)
    result = evaluate_safety_prediction_csv(
        task,
        predictions,
        observer_card,
        outdir=outdir,
    )
    print(outdir / "safety_protocol_result.json")
    return 0 if result.n_queries == len(task.queries) else 1


def export_safety_task_pack(task_id: str, outdir: Path) -> int:
    public_path, targets_path = write_safety_task_pack(load_safety_task(task_id), outdir)
    print(public_path)
    print(targets_path)
    return 0


def validate_safety_task_pack(public_task: Path, targets: Path | None) -> int:
    summary = validate_public_safety_task_pack(public_task)
    if targets is not None:
        task = load_safety_task_pack(public_task, targets)
        summary["evaluator_targets"] = "valid"
        summary["n_targets"] = len(task.targets)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def evaluate_custom_safety_csv(
    public_task: Path,
    targets: Path,
    predictions: Path,
    observer_card_path: Path,
    outdir: Path,
) -> int:
    payload = json.loads(observer_card_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observer card must be a JSON object")
    payload.pop("schema_version", None)
    observer_card = SafetyObserverCard(**payload)
    task = load_safety_task_pack(public_task, targets)
    result = evaluate_safety_prediction_csv(
        task,
        predictions,
        observer_card,
        outdir=outdir,
    )
    print(outdir / "safety_protocol_result.json")
    return 0 if result.n_queries == len(task.queries) else 1


def make_card(results: Path, outdir: Path) -> int:
    json_path, md_path = write_observer_card_bundle(results, outdir)
    print(json_path)
    print(md_path)
    return 0


def make_figures(
    outdir: Path,
    *,
    frozen_dir: Path | None = None,
    revision_dir: Path | None = None,
    only: str | None = None,
    legacy: bool = False,
) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "reproduce_paper_fast.py"
    if not script.exists():
        raise FileNotFoundError(
            "Paper reproduction requires a source checkout containing "
            f"{script.relative_to(repo_root)}."
        )
    command = [sys.executable, str(script), "--outdir", str(outdir)]
    if frozen_dir is not None:
        command.extend(["--frozen-dir", str(frozen_dir)])
    if revision_dir is not None:
        command.extend(["--revision-dir", str(revision_dir)])
    if only is not None:
        command.extend(["--only", only])
    if legacy:
        command.append("--legacy")
    return subprocess.run(command, cwd=repo_root, check=False).returncode


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
    if args.command == "list-safety-tasks":
        return list_safety_tasks()
    if args.command == "list-safety-results":
        return list_safety_results(
            task_id=args.task_id,
            output_format=args.format,
            leaderboard=args.leaderboard,
        )
    if args.command == "compare-safety":
        return compare_safety(
            result_dir=args.result_dir,
            output_format=args.format,
            leaderboard=args.leaderboard,
        )
    if args.command == "evaluate-ai-control-csv":
        return evaluate_ai_control(
            scores=args.scores,
            audit_budget=args.audit_budget,
            outdir=args.outdir,
        )
    if args.command == "list-ai-control-results":
        return list_ai_control_results(
            task_id=args.task_id,
            output_format=args.format,
            leaderboard=args.leaderboard,
        )
    if args.command == "evaluate-effect-csv":
        return evaluate_effect_csv(
            args.task_id,
            args.artifacts_root,
            args.predictions,
            args.observer_card,
            args.outdir,
            verify_hashes=not args.no_verify_hashes,
        )
    if args.command == "evaluate-safety-csv":
        return evaluate_safety_csv(
            args.task_id,
            args.predictions,
            args.observer_card,
            args.outdir,
        )
    if args.command == "export-safety-task-pack":
        return export_safety_task_pack(args.task_id, args.outdir)
    if args.command == "validate-safety-task-pack":
        return validate_safety_task_pack(args.public_task, args.targets)
    if args.command == "evaluate-safety-task-pack-csv":
        return evaluate_custom_safety_csv(
            args.public_task,
            args.targets,
            args.predictions,
            args.observer_card,
            args.outdir,
        )
    if args.command == "run":
        return run_task(args.task_name, args.config, args.outdir, quick=args.quick, input_run=args.input_run)
    if args.command == "make-card":
        return make_card(args.results, args.outdir)
    if args.command == "make-figures":
        return make_figures(
            args.outdir,
            frozen_dir=args.frozen_dir,
            revision_dir=args.revision_dir,
            only=args.only,
            legacy=args.legacy,
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
