"""Compose canonical task panels into the bundled AI-control result registry.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from observerbench.core import write_json
from observerbench.provenance import file_sha256


SCHEMA_VERSION = "observerbench.ai_control_leaderboard.v0"


def build(panel_paths: Sequence[Path], output: Path) -> dict[str, Any]:
    if not panel_paths:
        raise ValueError("at least one canonical AI-control panel is required")
    rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    seen_rows: set[tuple[str, str]] = set()

    for path in panel_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported AI-control panel schema: {path}")
        panel_rows = payload.get("rows")
        if not isinstance(panel_rows, list) or not panel_rows:
            raise ValueError(f"AI-control panel has no rows: {path}")
        task_ids = {str(row.get("task_id", "")) for row in panel_rows}
        if len(task_ids) != 1 or "" in task_ids:
            raise ValueError(f"AI-control panel must contain exactly one task: {path}")
        task_id = next(iter(task_ids))
        if task_id in seen_tasks:
            raise ValueError(f"duplicate AI-control task panel: {task_id}")
        seen_tasks.add(task_id)

        for row in panel_rows:
            identity = (task_id, str(row.get("observer_name", "")))
            if not identity[1] or identity in seen_rows:
                raise ValueError(f"duplicate or unnamed AI-control row: {identity}")
            seen_rows.add(identity)
            rows.append(dict(row))

        task = dict(payload.get("task", {}))
        task["task_id"] = task_id
        task["panel_source"] = path.as_posix()
        task["panel_sha256"] = file_sha256(path)
        task["metric_direction"] = dict(payload.get("metric_direction", {}))
        tasks.append(task)

    registry = {
        "schema_version": SCHEMA_VERSION,
        "tasks": tasks,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, registry)
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.panel, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
