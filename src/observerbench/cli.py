"""Command line interface for the ObserverBench reproduction workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from observerbench.cards import write_observer_card_bundle
from observerbench.config import load_config
from observerbench.tasks.registry import TASKS, run_registered_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="observerbench",
        description="ObserverBench paper reproduction CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tasks", help="List paper reproduction tasks.")

    run_parser = subparsers.add_parser(
        "run",
        help="Run a registered paper task.",
    )
    run_parser.add_argument("task_name", choices=sorted(TASKS))
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
    for name in sorted(TASKS):
        spec = TASKS[name]
        print(f"{spec.name}\t{spec.summary}")
    return 0


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
