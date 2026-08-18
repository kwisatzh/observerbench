#!/usr/bin/env python3
"""Build the static ObserverBench site from checked leaderboard JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable


CATALOG_SCHEMA = "observerbench.site_catalog.v0"
ARCHIVED_PANELS = {"controlarena-apps-qwen-v0"}
PUBLIC_TASK_PACK_FILES = {"README.md", "queries.jsonl", "safety_task.json", "task.json"}
PRIVATE_NAME_TOKENS = {"private", "secret", "target", "id_map", "key"}


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "leaderboard"


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty rows list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every leaderboard row must be an object")
    return payload


def _task_id(row: dict[str, Any], fallback: str) -> str:
    if row.get("task_id"):
        return str(row["task_id"])
    if row.get("task_name") and row.get("task_version"):
        return f"{row['task_name']}@{row['task_version']}"
    return fallback


def _title(task_id: str, panel: str) -> str:
    known = {
        "qwen-paired-scope-v0": "Qwen authorization triage",
        "qwen-paired-scope-v1": "Qwen authorization triage — public v1",
        "controlarena-apps-qwen-v1": "APPS verified-backdoor monitoring",
        "controlarena-apps-qwen-v0": "APPS monitoring — initial panel",
    }
    if panel in known:
        return known[panel]
    return task_id.replace("@", " · ").replace("-", " ").title()


def _metric_directions(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    declared = payload.get("metric_direction")
    if isinstance(declared, dict) and declared:
        return {str(k): str(v) for k, v in declared.items()}
    metrics = rows[0].get("metrics", {})
    directions: dict[str, str] = {}
    for name in metrics:
        directions[name] = "higher" if any(
            token in name for token in ("auroc", "precision", "utility", "recall")
        ) else "lower"
    return directions


def _primary_metric(metrics: dict[str, str]) -> str:
    for candidate in (
        "protocol_loss_mean",
        "violations_at_0.02",
        "closed_loop_loss",
        "action_loss",
        "risk_auroc",
    ):
        if candidate in metrics:
            return candidate
    return next(iter(metrics))


def _candidate_files(repo_root: Path) -> Iterable[Path]:
    canonical_root = repo_root / "leaderboards"
    if canonical_root.is_dir():
        yield from sorted(canonical_root.glob("**/results.json"))
        return

    # Installed-package fallback. These are mirrors of the checked canonical
    # panels and are only used when a source checkout has no leaderboards tree.
    data_root = repo_root / "src" / "observerbench" / "data"
    for name in ("safety_leaderboard_v0.json", "ai_control_leaderboard_v0.json"):
        path = data_root / name
        if path.is_file():
            yield path


def _copy_leaderboards(repo_root: Path, output: Path) -> list[dict[str, Any]]:
    data_dir = output / "data" / "leaderboards"
    data_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    paths = list(_candidate_files(repo_root))
    has_v1 = any("controlarena-apps-qwen-v1" in path.parts for path in paths)

    for path in paths:
        panel = path.parent.name if path.name == "results.json" else path.stem
        if has_v1 and panel in ARCHIVED_PANELS:
            continue
        payload = _load_payload(path)
        rows = payload["rows"]
        task_id = _task_id(rows[0], panel)
        if task_id in seen_tasks:
            continue
        seen_tasks.add(task_id)

        filename = f"{_slug(panel)}.json"
        destination = data_dir / filename
        shutil.copy2(path, destination)
        directions = _metric_directions(payload, rows)
        primary = _primary_metric(directions)
        statuses = sorted({str(row.get("result_status", "unspecified")) for row in rows})
        access_regimes = sorted({str(row.get("access_regime", "unspecified")) for row in rows})
        catalog.append(
            {
                "id": _slug(panel),
                "title": _title(task_id, panel),
                "track": "safety" if "safety" in path.parts else "observer",
                "task_id": task_id,
                "panel": panel,
                "row_count": len(rows),
                "status": "checked baseline panel",
                "primary_metric": primary,
                "metric_directions": directions,
                "available_metrics": list(directions),
                "result_statuses": statuses,
                "access_regimes": access_regimes,
                "data_url": f"data/leaderboards/{filename}",
                "source": str(path.relative_to(repo_root)),
            }
        )
    return sorted(catalog, key=lambda item: (item["track"], item["title"]))


def _copy_public_task_packs(repo_root: Path, output: Path) -> None:
    source_root = repo_root / "task-packs"
    if not source_root.is_dir():
        return
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise RuntimeError(f"task-pack symlinks are not publishable: {source}")
        if not source.is_file():
            continue
        lowered = source.name.lower()
        if any(token in lowered for token in PRIVATE_NAME_TOKENS):
            raise RuntimeError(f"refusing to publish sensitive-looking task-pack file: {source}")
        if source.name not in PUBLIC_TASK_PACK_FILES:
            continue
        relative = source.relative_to(source_root)
        destination = output / "task-packs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build(repo_root: Path, source: Path, output: Path) -> None:
    repo_root = repo_root.resolve()
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"site source not found: {source}")
    if not repo_root.is_dir():
        raise FileNotFoundError(f"ObserverBench repository not found: {repo_root}")

    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    _copy_public_task_packs(repo_root, output)
    catalog = _copy_leaderboards(repo_root, output)
    if not catalog:
        raise RuntimeError(f"no checked leaderboard JSON found under {repo_root}")

    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": CATALOG_SCHEMA,
                "leaderboards": catalog,
                "ranking_policy": {
                    "global_rank": False,
                    "rule": (
                        "Ranks are only issued inside the same task version, "
                        "comparison track, budget, controller, loss, and access boundary."
                    ),
                    "reference_rows_ranked": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("OBSERVERBENCH_REPO_ROOT", root)),
        help="ObserverBench source checkout containing leaderboards/",
    )
    parser.add_argument("--source", type=Path, default=root / "site")
    parser.add_argument("--output", type=Path, default=root / "build" / "pages")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.repo_root, args.source, args.output)
    print(f"Built ObserverBench site at {args.output.resolve()}")
