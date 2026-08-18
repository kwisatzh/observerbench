#!/usr/bin/env python3
"""Validate ObserverBench data-only submission bundles.

This script intentionally uses only the Python standard library.  A public
submission is data, never executable code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any


MAX_PREDICTIONS_BYTES = 5 * 1024 * 1024
MAX_CARD_BYTES = 64 * 1024
MAX_ROWS = 250_000
PATH_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}$")

TRACKS = {
    "safety": {
        "columns": ("schema_version", "query_id", "predicted_risk"),
        "schema": "observerbench.safety_predictions.v0",
        "id_column": "query_id",
        "value_column": "predicted_risk",
        "card_schema": "observerbench.safety_observer_card.v0",
        "required_card_fields": (
            "observer_name",
            "observer_version",
            "observer_family",
            "observer_input",
            "fit_procedure",
            "implementation",
            "risk_score_meaning",
            "access_regime",
        ),
    },
    "effect": {
        "columns": ("schema_version", "query_id", "predicted_effect"),
        "schema": "observerbench.effect_predictions.v0",
        "id_column": "query_id",
        "value_column": "predicted_effect",
        "card_schema": "observerbench.effect_observer_card.v0",
        "required_card_fields": (
            "observer_name",
            "observer_version",
            "observer_family",
            "access_regime",
            "measurement_basis",
            "fit_procedure",
            "implementation",
        ),
    },
    "ai-control": {
        "columns": ("schema_version", "sample_id", "monitor_name", "score"),
        "schema": "observerbench.ai_control_public_scores.v0",
        "id_column": "sample_id",
        "value_column": "score",
        "card_schema": "observerbench.safety_observer_card.v0",
        "required_card_fields": (
            "observer_name",
            "observer_version",
            "observer_family",
            "observer_input",
            "fit_procedure",
            "implementation",
            "risk_score_meaning",
            "access_regime",
        ),
    },
}


class SubmissionError(ValueError):
    """A public submission violates the data-only contract."""


def _regular_file(path: Path, *, maximum_bytes: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise SubmissionError(f"{path.name} must be a regular file")
    size = path.stat().st_size
    if size <= 0:
        raise SubmissionError(f"{path.name} cannot be empty")
    if size > maximum_bytes:
        raise SubmissionError(
            f"{path.name} is {size} bytes; the limit is {maximum_bytes}"
        )


def _nonempty_string(payload: dict[str, Any], field: str) -> None:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SubmissionError(f"observer_card.json field {field!r} must be non-empty")


def _validate_card(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    _regular_file(path, maximum_bytes=MAX_CARD_BYTES)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SubmissionError("observer_card.json must contain valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise SubmissionError("observer_card.json must contain one JSON object")
    if payload.get("schema_version") != spec["card_schema"]:
        raise SubmissionError(
            "observer_card.json has the wrong schema_version; expected "
            f"{spec['card_schema']}"
        )
    for field in spec["required_card_fields"]:
        _nonempty_string(payload, field)
    failures = payload.get("known_failure_modes", [])
    if not isinstance(failures, list) or any(
        not isinstance(value, str) or not value.strip() for value in failures
    ):
        raise SubmissionError("known_failure_modes must be a list of non-empty strings")
    white_box = payload.get("requires_white_box_access")
    if white_box is not None and not isinstance(white_box, bool):
        raise SubmissionError("requires_white_box_access must be true, false, or null")
    passes = payload.get("additional_forward_passes")
    if passes is not None and (
        isinstance(passes, bool) or not isinstance(passes, int) or passes < 0
    ):
        raise SubmissionError("additional_forward_passes must be non-negative or null")
    return payload


def _validate_predictions(
    path: Path, spec: dict[str, Any], *, expected_observer_name: str
) -> int:
    _regular_file(path, maximum_bytes=MAX_PREDICTIONS_BYTES)
    identities: set[str] = set()
    monitor_names: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8", newline="")
        with handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != spec["columns"]:
                raise SubmissionError(
                    "predictions.csv columns must be exactly "
                    + ",".join(spec["columns"])
                )
            for row_number, row in enumerate(reader, start=2):
                if row_number > MAX_ROWS + 1:
                    raise SubmissionError(f"predictions.csv exceeds {MAX_ROWS} rows")
                if row["schema_version"] != spec["schema"]:
                    raise SubmissionError(
                        f"unsupported prediction schema at row {row_number}; "
                        f"expected {spec['schema']}"
                    )
                identity = row[spec["id_column"]].strip()
                if not identity:
                    raise SubmissionError(f"empty ID at row {row_number}")
                if identity in identities:
                    raise SubmissionError(f"duplicate prediction ID: {identity}")
                identities.add(identity)
                try:
                    value = float(row[spec["value_column"]])
                except (TypeError, ValueError) as error:
                    raise SubmissionError(
                        f"prediction at row {row_number} must be finite"
                    ) from error
                if not math.isfinite(value):
                    raise SubmissionError(f"prediction at row {row_number} must be finite")
                if "monitor_name" in row:
                    monitor_name = row["monitor_name"].strip()
                    if not monitor_name:
                        raise SubmissionError(f"empty monitor_name at row {row_number}")
                    monitor_names.add(monitor_name)
    except UnicodeDecodeError as error:
        raise SubmissionError("predictions.csv must be UTF-8") from error
    if not identities:
        raise SubmissionError("predictions.csv must contain at least one prediction")
    if monitor_names and monitor_names != {expected_observer_name}:
        raise SubmissionError(
            "AI-control monitor_name must match observer_name on every row"
        )
    return len(identities)


def validate_submission(path: Path, *, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise SubmissionError(f"submission path must be a directory: {path}")
    relative = path.relative_to(root)
    if len(relative.parts) != 3:
        raise SubmissionError(
            "submission directories must be track/task-id/submission-id"
        )
    track, task_id, submission_id = relative.parts
    if track not in TRACKS:
        raise SubmissionError(f"unknown submission track: {track}")
    for label, value in (("task ID", task_id), ("submission ID", submission_id)):
        if not PATH_PART.fullmatch(value):
            raise SubmissionError(f"invalid {label}: {value!r}")

    children = {child.name for child in path.iterdir()}
    expected = {"predictions.csv", "observer_card.json"}
    if children != expected:
        raise SubmissionError(
            f"{relative}: expected only {sorted(expected)}, found {sorted(children)}"
        )
    spec = TRACKS[track]
    card = _validate_card(path / "observer_card.json", spec)
    count = _validate_predictions(
        path / "predictions.csv",
        spec,
        expected_observer_name=card["observer_name"],
    )
    return {
        "track": track,
        "task_id": task_id,
        "submission_id": submission_id,
        "observer_name": card["observer_name"],
        "observer_version": card["observer_version"],
        "prediction_count": count,
        "status": "preflight-passed",
    }


def validate_root(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise SubmissionError(f"submission root does not exist: {root}")
    candidates = sorted(
        path
        for path in root.glob("*/*/*")
        if path.is_dir() or path.is_symlink()
    )
    if not candidates:
        raise SubmissionError("no submission directory was found")
    if len(candidates) != 1:
        raise SubmissionError("one pull request must contain exactly one submission")
    return [validate_submission(candidates[0], root=root)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = {
            "schema_version": "observerbench.public_submission_preflight.v0",
            "submissions": validate_root(args.root),
        }
    except SubmissionError as error:
        parser.error(str(error))
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
