# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from observerbench import (
    EFFECT_OBSERVER_CARD_SCHEMA_VERSION,
    EFFECT_PREDICTION_CONTRACT_VERSION,
    EFFECT_PREDICTION_CSV_SCHEMA_VERSION,
    EFFECT_PREDICTION_RESULT_SCHEMA_VERSION,
    EFFECT_TASK_CARD_SCHEMA_VERSION,
    EffectObserverCard,
    EffectTaskCard,
    FiniteEffectMeasurement,
    FiniteEffectPrediction,
    FiniteEffectPredictionTask,
    FiniteEffectPredictor,
    FiniteEffectQuery,
    FiniteEffectTarget,
    evaluate_effect_prediction_csv,
    evaluate_effect_predictions,
    read_effect_predictions,
    run_effect_prediction_task,
    write_effect_predictions,
)


class OffsetPredictor:
    name = "offset-predictor"

    def __init__(self) -> None:
        self.offset = 0.0
        self.fit_ids: tuple[str, ...] = ()

    def fit(self, measurements) -> None:
        self.fit_ids = tuple(row.measurement_id for row in measurements)
        self.offset = sum(
            row.observed_effect - row.features for row in measurements
        ) / len(measurements)

    def predict(self, queries):
        return [row.features + self.offset for row in queries]


class ShortPredictor(OffsetPredictor):
    name = "short-predictor"

    def predict(self, queries):
        del queries
        return [1.0]


def make_task() -> FiniteEffectPredictionTask[float]:
    card = EffectTaskCard(
        task_name="finite-offset",
        task_version="1.2.0",
        summary="Predict the held-out effect of a scalar offset.",
        model_or_substrate="scalar fixture",
        access_regime="forward measurements",
        estimand="finite scalar response",
        intervention_family="scalar offsets",
        measurement_design="two fixed calibration points",
        validation_target="held-out finite-effect prediction",
        train_split="train-v1",
        evaluation_split="test-v1",
        known_scope_limits=("Test fixture only.",),
        metadata={"split_sha256": "fixture"},
    )
    return FiniteEffectPredictionTask(
        name="finite-offset",
        version="1.2.0",
        measurements=(
            FiniteEffectMeasurement("m0", 0.0, 2.0),
            FiniteEffectMeasurement("m1", 1.0, 3.0),
        ),
        queries=(
            FiniteEffectQuery("q1", 3.0),
            FiniteEffectQuery("q2", -1.0),
        ),
        targets=(
            FiniteEffectTarget("q1", 5.0),
            FiniteEffectTarget("q2", 1.0),
        ),
        card=card,
    )


def make_observer_card(name: str = "offset-predictor") -> EffectObserverCard:
    return EffectObserverCard(
        observer_name=name,
        observer_version="0.3.0",
        observer_family="affine finite-effect predictor",
        access_regime="forward measurements",
        measurement_basis="scalar intervention",
        fit_procedure="mean observed offset",
        implementation="outside_package.OffsetPredictor",
        known_failure_modes=("Assumes a constant offset.",),
        metadata={"source_revision": "fixture"},
    )


def test_external_predictor_composes_and_writes_versioned_bundle(tmp_path: Path) -> None:
    task = make_task()
    predictor = OffsetPredictor()

    result = run_effect_prediction_task(
        task,
        predictor,
        make_observer_card(),
        outdir=tmp_path,
    )

    assert isinstance(predictor, FiniteEffectPredictor)
    assert predictor.fit_ids == ("m0", "m1")
    assert result.measurement_budget == 2
    assert result.n_queries == 2
    assert result.metrics == {
        "mae": 0.0,
        "rmse": 0.0,
        "mean_error": 0.0,
        "max_absolute_error": 0.0,
    }
    assert result.contract_version == EFFECT_PREDICTION_CONTRACT_VERSION
    assert result.schema_version == EFFECT_PREDICTION_RESULT_SCHEMA_VERSION
    assert set(path.name for path in tmp_path.iterdir()) == {
        "effect_predictions.csv",
        "effect_evaluation.json",
        "task_card.json",
        "observer_card.json",
    }

    task_card = json.loads((tmp_path / "task_card.json").read_text())
    observer_card = json.loads((tmp_path / "observer_card.json").read_text())
    evaluation = json.loads((tmp_path / "effect_evaluation.json").read_text())
    assert task_card["schema_version"] == EFFECT_TASK_CARD_SCHEMA_VERSION
    assert observer_card["schema_version"] == EFFECT_OBSERVER_CARD_SCHEMA_VERSION
    assert evaluation["measurement_budget"] == 2
    assert evaluation["observer_name"] == "offset-predictor"


def test_csv_interchange_round_trips_and_evaluates_by_id(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    predictions = [
        FiniteEffectPrediction("q2", 1.0),
        FiniteEffectPrediction("q1", 5.0),
    ]

    write_effect_predictions(path, predictions)

    assert read_effect_predictions(path) == predictions
    assert path.read_text().splitlines()[0] == (
        "schema_version,query_id,predicted_effect"
    )
    assert EFFECT_PREDICTION_CSV_SCHEMA_VERSION in path.read_text()
    result = evaluate_effect_prediction_csv(
        make_task(),
        path,
        make_observer_card(),
        outdir=tmp_path / "evaluated",
    )
    assert result.metrics["mae"] == 0.0


def test_evaluator_reports_generic_finite_effect_errors() -> None:
    result = evaluate_effect_predictions(
        make_task(),
        [
            FiniteEffectPrediction("q1", 6.0),
            FiniteEffectPrediction("q2", -1.0),
        ],
        make_observer_card(),
    )

    assert result.metrics["mae"] == pytest.approx(1.5)
    assert result.metrics["rmse"] == pytest.approx(2.5**0.5)
    assert result.metrics["mean_error"] == pytest.approx(-0.5)
    assert result.metrics["max_absolute_error"] == pytest.approx(2.0)


def test_csv_reader_rejects_schema_duplicates_and_nonfinite_values(tmp_path: Path) -> None:
    wrong_schema = tmp_path / "wrong-schema.csv"
    wrong_schema.write_text(
        "schema_version,query_id,predicted_effect\nold,q1,1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported prediction CSV schema"):
        read_effect_predictions(wrong_schema)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "schema_version,query_id,predicted_effect\n"
        f"{EFFECT_PREDICTION_CSV_SCHEMA_VERSION},q1,1.0\n"
        f"{EFFECT_PREDICTION_CSV_SCHEMA_VERSION},q1,2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prediction ID"):
        read_effect_predictions(duplicate)

    nonfinite = tmp_path / "nonfinite.csv"
    nonfinite.write_text(
        "schema_version,query_id,predicted_effect\n"
        f"{EFFECT_PREDICTION_CSV_SCHEMA_VERSION},q1,nan\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite real number"):
        read_effect_predictions(nonfinite)


def test_task_and_prediction_boundaries_are_enforced(tmp_path: Path) -> None:
    task = make_task()
    with pytest.raises(ValueError, match=r"missing=\['q2'\]"):
        evaluate_effect_predictions(
            task,
            [FiniteEffectPrediction("q1", 5.0)],
            make_observer_card(),
        )

    with pytest.raises(ValueError, match="predictor name must match"):
        run_effect_prediction_task(
            task,
            OffsetPredictor(),
            make_observer_card("different-name"),
            outdir=tmp_path / "wrong-name",
        )

    with pytest.raises(ValueError, match="exactly one prediction per query"):
        run_effect_prediction_task(
            task,
            ShortPredictor(),
            make_observer_card("short-predictor"),
            outdir=tmp_path / "short",
        )

    with pytest.raises(ValueError, match="task name and version must match"):
        FiniteEffectPredictionTask(
            name="renamed-task",
            version=task.version,
            measurements=task.measurements,
            queries=task.queries,
            targets=task.targets,
            card=task.card,
        )


def test_documented_effect_prediction_example_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = (
        Path(__file__).parents[1]
        / "docs"
        / "EFFECT_PREDICTION_CONTRACT.md"
    ).read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", docs, flags=re.DOTALL)
    assert len(blocks) == 1
    monkeypatch.chdir(tmp_path)

    exec(compile(blocks[0], "docs/EFFECT_PREDICTION_CONTRACT.md", "exec"), {})

    output = tmp_path / "runs" / "scalar-effects" / "outside-linear"
    assert (output / "effect_predictions.csv").exists()
    evaluation = json.loads((output / "effect_evaluation.json").read_text())
    assert evaluation["metrics"]["mae"] == 0.0
