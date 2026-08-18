#!/usr/bin/env python3
"""Turn a structured GitHub issue into a validated data-only submission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
FIELDS = (
    "Task and version",
    "Observer name and version",
    "Prediction-table URL",
    "ObserverCard JSON",
)
TASK_TRACKS = {
    "safety-interlock-qwen2-5-7b-instruct@paired-scope-v1": "safety",
}

VALIDATOR_PATH = Path(__file__).with_name("validate_public_submission.py")
SPEC = importlib.util.spec_from_file_location("observerbench_submission_validator", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - installation failure
    raise RuntimeError(f"cannot load submission validator from {VALIDATOR_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class IssueSubmissionError(ValueError):
    """A structured observer-job issue cannot be accepted."""


def parse_form_body(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^### ([^\n]+)\n", body))
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[name] = body[start:end].strip()
    missing = [name for name in FIELDS if not sections.get(name)]
    if missing:
        raise IssueSubmissionError(f"issue form is missing fields: {', '.join(missing)}")
    return sections


def parse_card(value: str) -> dict[str, Any]:
    fenced = re.fullmatch(r"```(?:json)?\s*\n(?P<body>.*)\n```", value, re.DOTALL)
    raw = fenced.group("body") if fenced else value
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise IssueSubmissionError("ObserverCard JSON is invalid") from error
    if not isinstance(payload, dict):
        raise IssueSubmissionError("ObserverCard must be one JSON object")
    return payload


def validate_prediction_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname != "raw.githubusercontent.com"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
    ):
        raise IssueSubmissionError(
            "Prediction URL must be an HTTPS raw.githubusercontent.com URL"
        )
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 4 or not re.fullmatch(r"[0-9a-fA-F]{40,64}", parts[2]):
        raise IssueSubmissionError(
            "Prediction URL must pin an immutable 40- or 64-character commit SHA"
        )
    return value.strip()


def download_predictions(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ObserverBench-submission-preflight/1"})
    with urlopen(request, timeout=30) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > MAX_DOWNLOAD_BYTES:
            raise IssueSubmissionError("prediction table exceeds the 5 MiB limit")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise IssueSubmissionError("prediction table exceeds the 5 MiB limit")
    if not payload:
        raise IssueSubmissionError("prediction table is empty")
    return payload


def prepare_issue(
    event: dict[str, Any], output_root: Path, *, prediction_bytes: bytes | None = None
) -> dict[str, Any]:
    issue = event.get("issue")
    repository = event.get("repository")
    if not isinstance(issue, dict) or not isinstance(repository, dict):
        raise IssueSubmissionError("event must contain issue and repository objects")
    body = issue.get("body")
    if not isinstance(body, str):
        raise IssueSubmissionError("issue body is missing")
    sections = parse_form_body(body)
    task_id = sections["Task and version"].strip()
    track = TASK_TRACKS.get(task_id)
    if track is None:
        raise IssueSubmissionError(f"task is not open for automatic submissions: {task_id}")
    observer_identity = sections["Observer name and version"].strip()
    card = parse_card(sections["ObserverCard JSON"])
    card_identity = f"{card.get('observer_name', '')}@{card.get('observer_version', '')}"
    if observer_identity != card_identity:
        raise IssueSubmissionError(
            "Observer name and version must match observer_name@observer_version in the card"
        )
    prediction_url = validate_prediction_url(sections["Prediction-table URL"])
    payload = prediction_bytes if prediction_bytes is not None else download_predictions(prediction_url)

    issue_number = issue.get("number")
    if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
        raise IssueSubmissionError("issue number is invalid")
    submission_id = f"issue-{issue_number}-{card['observer_name']}-{card['observer_version']}"
    submission_id = re.sub(r"[^A-Za-z0-9._@+-]", "-", submission_id)[:128]
    root = output_root / "pending"
    bundle = root / track / task_id / submission_id
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "predictions.csv").write_bytes(payload)
    (bundle / "observer_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight = VALIDATOR.validate_submission(bundle, root=root)
    result = {
        "schema_version": "observerbench.issue_submission_request.v0",
        "source_repository": repository.get("full_name"),
        "issue_number": issue_number,
        "issue_url": issue.get("html_url"),
        "submitter": (issue.get("user") or {}).get("login"),
        "prediction_url": prediction_url,
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        **preflight,
    }
    (output_root / "request.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        event = json.loads(args.event.read_text(encoding="utf-8"))
        result = prepare_issue(event, args.output)
    except (IssueSubmissionError, VALIDATOR.SubmissionError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
