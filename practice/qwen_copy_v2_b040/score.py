#!/usr/bin/env python3
"""Score an open Qwen Copy-v2 practice submission without model inference."""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


PREDICTION_SCHEMA = "observerbench.effect_predictions.v0"
PREDICTION_FIELDS = ("schema_version", "query_id", "predicted_effect")
PACK_ROOT = Path(__file__).resolve().parent


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def _mean(values: Iterable[float]) -> float:
    data = tuple(values)
    if not data:
        raise ValueError("cannot average an empty collection")
    return math.fsum(data) / len(data)


def _checked_pack(pack_root: Path) -> dict[str, object]:
    card = _read_json(pack_root / "task_card.json")
    if card.get("schema") != "observerbench.open_practice_pack.v1":
        raise ValueError("unsupported practice-pack schema")
    if card.get("participation_class") != "open_practice":
        raise ValueError("this scorer only accepts an open practice pack")
    if card.get("leaderboard_eligible") is not False or card.get("sealed") is not False:
        raise ValueError("open practice pack must not claim sealed leaderboard status")
    if card.get("targets_public") is not True:
        raise ValueError("open practice targets must be public")
    files = card.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("task card lacks file fingerprints")
    for name in (
        "calibration_measurements.csv",
        "queries.csv",
        "targets.csv",
        "action_outcomes.csv",
    ):
        record = files.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"task card lacks fingerprint for {name}")
        path = pack_root / name
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"practice-pack file hash mismatch: {name}")
    return card


def _predictions(path: Path, expected_ids: set[str]) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
            raise ValueError(
                "prediction columns must be exactly: " + ", ".join(PREDICTION_FIELDS)
            )
        predictions: dict[str, float] = {}
        for row_number, row in enumerate(reader, start=2):
            if row["schema_version"] != PREDICTION_SCHEMA:
                raise ValueError(f"unsupported schema_version at row {row_number}")
            query_id = row["query_id"].strip()
            if not query_id or query_id in predictions:
                raise ValueError(f"empty or duplicate query_id at row {row_number}")
            predictions[query_id] = _finite(
                row["predicted_effect"], f"predicted_effect at row {row_number}"
            )
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        unexpected = sorted(set(predictions) - expected_ids)
        raise ValueError(
            f"prediction IDs do not match the 128 queries; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return predictions


def _prediction_metrics(
    targets: Mapping[str, float], predictions: Mapping[str, float]
) -> dict[str, float]:
    errors = [predictions[key] - targets[key] for key in sorted(targets)]
    absolute = [abs(error) for error in errors]
    squared = [error * error for error in errors]
    return {
        "mae": _mean(absolute),
        "rmse": math.sqrt(_mean(squared)),
        "mean_error": _mean(errors),
        "max_absolute_error": max(absolute),
    }


def _action_metrics(
    queries: Mapping[str, Mapping[str, str]],
    predictions: Mapping[str, float],
    outcomes: list[dict[str, str]],
) -> tuple[dict[str, float | int], list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in outcomes:
        grouped.setdefault((row["target_fraction"], row["pool_id"]), []).append(row)

    decisions: list[dict[str, object]] = []
    for (fraction, pool_id), candidates in sorted(grouped.items()):
        if len(candidates) != 9:
            raise ValueError(f"action pool {fraction}/{pool_id} must contain 8 masks and no-op")
        target_values = {_finite(row["target"], "target") for row in candidates}
        if len(target_values) != 1:
            raise ValueError(f"target changes within action pool {fraction}/{pool_id}")
        target = target_values.pop()
        actual: dict[str, float] = {}
        head_counts: dict[str, int] = {}
        for row in candidates:
            query_id = row["query_id"]
            if query_id in actual:
                raise ValueError(f"duplicate action candidate {query_id} in {pool_id}")
            is_noop = row["is_noop"].lower() == "true"
            if is_noop != (query_id == "analytic_noop"):
                raise ValueError("analytic no-op marker disagrees with query_id")
            n_heads = int(row["n_heads"])
            if not is_noop:
                query = queries.get(query_id)
                if query is None or query["pool_id"] != pool_id:
                    raise ValueError(f"action outcome does not match query pool: {query_id}")
                if int(query["n_heads"]) != n_heads:
                    raise ValueError(f"action outcome changes head count: {query_id}")
            actual[query_id] = _finite(row["actual_target_loss"], "actual_target_loss")
            head_counts[query_id] = n_heads
        if "analytic_noop" not in actual or not math.isclose(
            actual["analytic_noop"], target, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("analytic no-op must have exact realized loss equal to target")

        predicted_risk = {
            query_id: (
                target
                if query_id == "analytic_noop"
                else abs(predictions[query_id] - target)
            )
            for query_id in actual
        }
        selected = min(
            actual,
            key=lambda query_id: (
                predicted_risk[query_id],
                head_counts[query_id],
                query_id,
            ),
        )
        oracle = min(
            actual,
            key=lambda query_id: (
                actual[query_id],
                head_counts[query_id],
                query_id,
            ),
        )
        decisions.append(
            {
                "target_fraction": fraction,
                "target": target,
                "pool_id": pool_id,
                "selected_query_id": selected,
                "selected_is_noop": selected == "analytic_noop",
                "predicted_target_loss": predicted_risk[selected],
                "actual_target_loss": actual[selected],
                "oracle_query_id": oracle,
                "oracle_is_noop": oracle == "analytic_noop",
                "oracle_target_loss": actual[oracle],
                "regret": actual[selected] - actual[oracle],
                "noop_target_loss": actual["analytic_noop"],
            }
        )

    if len(decisions) != 48:
        raise ValueError("action evaluation must contain 3 targets by 16 pools")
    aggregate: dict[str, float | int] = {
        "n_decisions": len(decisions),
        "mean_selected_action_loss": _mean(
            float(row["actual_target_loss"]) for row in decisions
        ),
        "mean_oracle_action_loss": _mean(
            float(row["oracle_target_loss"]) for row in decisions
        ),
        "mean_regret": _mean(float(row["regret"]) for row in decisions),
        "mean_noop_action_loss": _mean(
            float(row["noop_target_loss"]) for row in decisions
        ),
        "mean_loss_reduction_vs_noop": _mean(
            float(row["noop_target_loss"]) - float(row["actual_target_loss"])
            for row in decisions
        ),
        "oracle_action_match_rate": _mean(
            float(row["selected_query_id"] == row["oracle_query_id"])
            for row in decisions
        ),
        "noop_selection_rate": _mean(
            float(bool(row["selected_is_noop"])) for row in decisions
        ),
    }
    by_target: list[dict[str, object]] = []
    for fraction in sorted({str(row["target_fraction"]) for row in decisions}):
        rows = [row for row in decisions if row["target_fraction"] == fraction]
        by_target.append(
            {
                "target_fraction": fraction,
                "target": rows[0]["target"],
                "n_decisions": len(rows),
                "mean_selected_action_loss": _mean(
                    float(row["actual_target_loss"]) for row in rows
                ),
                "mean_oracle_action_loss": _mean(
                    float(row["oracle_target_loss"]) for row in rows
                ),
                "mean_regret": _mean(float(row["regret"]) for row in rows),
                "mean_noop_action_loss": _mean(
                    float(row["noop_target_loss"]) for row in rows
                ),
                "noop_selection_rate": _mean(
                    float(bool(row["selected_is_noop"])) for row in rows
                ),
            }
        )
    return aggregate, by_target, decisions


def score_submission(
    prediction_path: Path,
    *,
    pack_root: Path = PACK_ROOT,
) -> dict[str, object]:
    """Return prediction and downstream-decision metrics for one CSV."""

    card = _checked_pack(pack_root)
    query_rows = _read_csv(pack_root / "queries.csv")
    target_rows = _read_csv(pack_root / "targets.csv")
    outcome_rows = _read_csv(pack_root / "action_outcomes.csv")
    queries = {row["query_id"]: row for row in query_rows}
    targets = {
        row["query_id"]: _finite(row["observed_mean_effect"], "observed_mean_effect")
        for row in target_rows
    }
    if len(queries) != 128 or set(queries) != set(targets):
        raise ValueError("practice queries and mean-effect targets do not align")
    predictions = _predictions(prediction_path, set(queries))
    action, by_target, decisions = _action_metrics(
        queries, predictions, outcome_rows
    )
    return {
        "task_id": card["task_id"],
        "participation_class": "open_practice",
        "leaderboard_eligible": False,
        "warning": "Public-target practice result; do not report as a sealed leaderboard score.",
        "prediction": _prediction_metrics(targets, predictions),
        "decision": {
            "aggregate": action,
            "by_target": by_target,
            "selections": decisions,
        },
    }


def _print_summary(result: Mapping[str, object]) -> None:
    prediction = result["prediction"]
    decision = result["decision"]
    if not isinstance(prediction, Mapping) or not isinstance(decision, Mapping):
        raise ValueError("invalid result payload")
    aggregate = decision["aggregate"]
    by_target = decision["by_target"]
    if not isinstance(aggregate, Mapping) or not isinstance(by_target, list):
        raise ValueError("invalid decision payload")
    print(f"Task: {result['task_id']}")
    print("Status: OPEN PRACTICE — public targets, not leaderboard eligible")
    print()
    print("Held-out effect prediction")
    print(f"  MAE:  {float(prediction['mae']):.6f}")
    print(f"  RMSE: {float(prediction['rmse']):.6f}")
    print()
    print("Downstream action (48 target/pool decisions, exact no-op included)")
    print(
        f"  Mean realized loss: {float(aggregate['mean_selected_action_loss']):.6f}"
    )
    print(f"  Mean oracle loss:   {float(aggregate['mean_oracle_action_loss']):.6f}")
    print(f"  Mean regret:        {float(aggregate['mean_regret']):.6f}")
    print(
        f"  Gain over no-op:    {float(aggregate['mean_loss_reduction_vs_noop']):.6f}"
    )
    print(f"  No-op selected:     {100 * float(aggregate['noop_selection_rate']):.1f}%")
    print()
    print("By registered target")
    for row in by_target:
        print(
            f"  {row['target_fraction']}× full-panel effect "
            f"(t={float(row['target']):.6f}): "
            f"loss={float(row['mean_selected_action_loss']):.6f}, "
            f"regret={float(row['mean_regret']):.6f}, "
            f"no-op={100 * float(row['noop_selection_rate']):.1f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--pack-root", type=Path, default=PACK_ROOT)
    parser.add_argument("--json", action="store_true", help="print the complete JSON result")
    parser.add_argument("--output", type=Path, help="also write the complete JSON result")
    args = parser.parse_args()
    result = score_submission(
        args.predictions.resolve(), pack_root=args.pack_root.resolve()
    )
    if args.output:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
