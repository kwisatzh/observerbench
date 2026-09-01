"""Build the checked safety leaderboard from sealed Qwen result artifacts.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from observerbench.core import write_json
from observerbench.provenance import file_sha256
from observerbench.safety import SafetyObserverCard
from observerbench.safety_leaderboard import (
    SAFETY_LEADERBOARD_METRICS,
    SAFETY_LEADERBOARD_SCHEMA_VERSION,
    SafetyLeaderboardRow,
    format_safety_leaderboard,
)


OBSERVER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "observer_name": "allow-all-no-action",
        "display_name": "Allow all",
        "observer_family": "no-action reference",
        "result_status": "reference bound",
        "observer_input": "none",
        "access_regime": "none",
        "requires_white_box_access": False,
        "additional_forward_passes": 0,
        "fit_procedure": "none",
        "risk_score_meaning": "constant zero risk",
        "known_failure_modes": ("Allows every unsafe request.",),
        "source": "primary",
    },
    {
        "observer_name": "action-only-direct-risk",
        "display_name": "Action-only direct risk",
        "observer_family": "action-feature ridge",
        "result_status": "prespecified locked test",
        "observer_input": "requested operation and declared severity",
        "access_regime": "action fields",
        "requires_white_box_access": False,
        "additional_forward_passes": 0,
        "fit_procedure": "ridge fit to severity-weighted violation risk",
        "risk_score_meaning": "larger values indicate larger loss if allowed",
        "known_failure_modes": ("Cannot distinguish paired prompts with identical requested actions.",),
        "source": "primary",
    },
    {
        "observer_name": "activation-label",
        "display_name": "Activation label",
        "observer_family": "linear activation probe",
        "result_status": "prespecified locked test",
        "observer_input": "projected layer-8 residual plus action fields",
        "access_regime": "white-box residual plus action fields",
        "requires_white_box_access": True,
        "additional_forward_passes": 0,
        "fit_procedure": "ridge fit to the authorization label",
        "risk_score_meaning": "larger values indicate greater probability of exceeding scope",
        "known_failure_modes": ("Ignores consequence severity when allocating the action budget.",),
        "source": "primary",
    },
    {
        "observer_name": "activation-direct-risk",
        "display_name": "Activation direct risk",
        "observer_family": "linear activation risk probe",
        "result_status": "prespecified locked test",
        "observer_input": "projected layer-8 residual plus action fields",
        "access_regime": "white-box residual plus action fields",
        "requires_white_box_access": True,
        "additional_forward_passes": 0,
        "fit_procedure": "ridge fit to severity-weighted violation risk",
        "risk_score_meaning": "larger values indicate larger loss if allowed",
        "known_failure_modes": ("Tail loss does not improve over the action-only reference.",),
        "source": "primary",
    },
    {
        "observer_name": "activation-transformed-label-risk",
        "display_name": "Activation label x severity",
        "observer_family": "consequence-weighted activation probe",
        "result_status": "prespecified locked test",
        "observer_input": "projected layer-8 residual plus action fields and severity",
        "access_regime": "white-box residual plus action fields",
        "requires_white_box_access": True,
        "additional_forward_passes": 0,
        "fit_procedure": "ridge label fit multiplied by known consequence severity",
        "risk_score_meaning": "estimated authorization violation probability times severity",
        "known_failure_modes": ("Score scale shifts across held-out operation-format strata.",),
        "source": "primary",
    },
    {
        "observer_name": "full-context-text-label-times-severity",
        "display_name": "Full-context text label x severity",
        "observer_family": "hashed text ridge",
        "result_status": "post-outcome secondary",
        "observer_input": "complete system and user prompt plus severity",
        "access_regime": "full prompt text plus action fields",
        "requires_white_box_access": False,
        "additional_forward_passes": 0,
        "fit_procedure": "fixed text hash and ridge label fit multiplied by severity",
        "risk_score_meaning": "estimated authorization violation probability times severity",
        "known_failure_modes": ("Designed after the original locked-test outcome was known.",),
        "source": "followup",
    },
    {
        "observer_name": "model-logit-margin-times-severity",
        "display_name": "Qwen2.5-7B-Instruct output margin x severity",
        "observer_family": "model-output observer",
        "result_status": "post-outcome secondary",
        "observer_input": "block-minus-allow output-logit margin plus severity",
        "access_regime": "output logits plus action fields",
        "requires_white_box_access": False,
        "additional_forward_passes": 0,
        "fit_procedure": "no fit; multiply the frozen output margin by severity",
        "risk_score_meaning": "model decision margin times consequence severity",
        "known_failure_modes": ("Only defined when the model is explicitly asked for a verdict.",),
        "source": "followup",
    },
    {
        "observer_name": "exact-authorization-risk-oracle",
        "display_name": "Exact authorization-risk oracle",
        "observer_family": "evaluator oracle",
        "result_status": "oracle bound",
        "observer_input": "evaluator-only authorization label and severity",
        "access_regime": "evaluator oracle",
        "requires_white_box_access": False,
        "additional_forward_passes": 0,
        "fit_procedure": "none",
        "risk_score_meaning": "exact violation indicator times severity",
        "known_failure_modes": ("Not available to a deployed observer.",),
        "source": "primary",
    },
)


def _load_results(path: Path) -> Mapping[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["results"]


def _row(
    spec: Mapping[str, Any],
    result: Mapping[str, Any],
    source_path: Path,
    repo_root: Path,
) -> SafetyLeaderboardRow:
    return SafetyLeaderboardRow(
        task_name=str(result["task_name"]),
        task_version=str(result["task_version"]),
        observer_name=str(spec["observer_name"]),
        display_name=str(spec["display_name"]),
        observer_family=str(spec["observer_family"]),
        result_status=str(spec["result_status"]),
        access_regime=str(spec["access_regime"]),
        requires_white_box_access=spec["requires_white_box_access"],
        additional_forward_passes=spec["additional_forward_passes"],
        metrics={name: float(result["metrics"][name]) for name in SAFETY_LEADERBOARD_METRICS},
        action_counts={name: int(result["action_counts"][name]) for name in ("allow", "block", "escalate")},
        source_result=source_path.relative_to(repo_root).as_posix(),
        source_sha256=file_sha256(source_path),
        known_failure_modes=tuple(spec["known_failure_modes"]),
        metadata={
            "observer_input": spec["observer_input"],
            "risk_score_meaning": spec["risk_score_meaning"],
        },
    )


def build(repo_root: Path) -> None:
    artifacts = repo_root / "results/revision/qwen_safety/paired_scope_v0"
    primary_path = artifacts / "evaluation/qwen_safety_results.json"
    followup_path = artifacts / "followup_v0/qwen_safety_followup_results.json"
    primary = _load_results(primary_path)
    followup = _load_results(followup_path)
    rows: list[SafetyLeaderboardRow] = []

    output = repo_root / "leaderboards/safety/qwen-paired-scope-v0"
    cards = output / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    for spec in OBSERVER_SPECS:
        source_path = primary_path if spec["source"] == "primary" else followup_path
        result = (primary if spec["source"] == "primary" else followup)[spec["observer_name"]]
        row = _row(spec, result, source_path, repo_root)
        rows.append(row)
        card = SafetyObserverCard(
            observer_name=str(spec["observer_name"]),
            observer_version="qwen-paired-scope-v0",
            observer_family=str(spec["observer_family"]),
            observer_input=str(spec["observer_input"]),
            fit_procedure=str(spec["fit_procedure"]),
            implementation="ObserverBench checked baseline",
            risk_score_meaning=str(spec["risk_score_meaning"]),
            known_failure_modes=tuple(spec["known_failure_modes"]),
            access_regime=str(spec["access_regime"]),
            requires_white_box_access=spec["requires_white_box_access"],
            additional_forward_passes=spec["additional_forward_passes"],
            metadata={"result_status": spec["result_status"]},
        )
        write_json(cards / f"{spec['observer_name']}.json", asdict(card))

    rows = sorted(rows, key=lambda item: item.metrics["protocol_loss_mean"])
    payload = {
        "schema_version": SAFETY_LEADERBOARD_SCHEMA_VERSION,
        "task_id": rows[0].task_id,
        "metric_direction": {
            "protocol_loss_mean": "lower",
            "protocol_loss_cvar": "lower",
            "risk_auroc": "higher",
            "severity_weighted_miss_rate": "lower",
            "clean_utility_retained": "higher",
            "worst_family_protocol_loss": "lower",
        },
        "rows": [row.to_dict() for row in rows],
    }
    write_json(output / "results.json", payload)
    (output / "results.csv").write_text(
        format_safety_leaderboard(rows, output_format="csv"), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Qwen2.5-7B-Instruct paired-scope safety leaderboard\n\n"
        "Rows are generated from checked result artifacts. Status distinguishes "
        "prespecified comparisons, post-outcome controls, and reference bounds.\n\n"
        + format_safety_leaderboard(rows, output_format="markdown"),
        encoding="utf-8",
    )
    package_path = repo_root / "src/observerbench/data/safety_leaderboard_v0.json"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(package_path, payload)


def main() -> int:
    build(Path(__file__).resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
