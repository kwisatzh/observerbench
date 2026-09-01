#!/usr/bin/env python3
"""Fit a small additive observer and write a practice submission CSV."""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import argparse
import csv
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parent
FIELDS = ("schema_version", "query_id", "predicted_effect")
SCHEMA = "observerbench.effect_predictions.v0"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve one small dense system with partial-pivot Gaussian elimination."""

    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("additive baseline design is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _feature(row: dict[str, str]) -> list[float]:
    return [1.0, *(float(row[f"h{index}"]) for index in range(8))]


def fit_predict(pack_root: Path) -> list[dict[str, object]]:
    calibration = _read(pack_root / "calibration_measurements.csv")
    queries = _read(pack_root / "queries.csv")
    size = 9
    normal = [[0.0 for _ in range(size)] for _ in range(size)]
    response = [0.0 for _ in range(size)]
    for row in calibration:
        x = _feature(row)
        y = float(row["observed_mean_effect"])
        for left in range(size):
            response[left] += x[left] * y
            for right in range(size):
                normal[left][right] += x[left] * x[right]
    for index in range(1, size):
        normal[index][index] += 1e-6
    coefficients = _solve(normal, response)
    return [
        {
            "schema_version": SCHEMA,
            "query_id": row["query_id"],
            "predicted_effect": repr(
                sum(value * weight for value, weight in zip(_feature(row), coefficients))
            ),
        }
        for row in queries
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, default=PACK_ROOT)
    parser.add_argument("--output", type=Path, default=PACK_ROOT / "predictions.csv")
    args = parser.parse_args()
    rows = fit_predict(args.pack_root.resolve())
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
