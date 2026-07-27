"""Versioned composition contract for held-out finite-effect prediction."""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import csv
from dataclasses import asdict, dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import (
    Any,
    Generic,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from .core import write_json


FeatureT = TypeVar("FeatureT")

EFFECT_PREDICTION_CONTRACT_VERSION = "observerbench.effect_prediction.v0"
EFFECT_PREDICTION_CSV_SCHEMA_VERSION = "observerbench.effect_predictions.v0"
EFFECT_PREDICTION_RESULT_SCHEMA_VERSION = "observerbench.effect_prediction_result.v0"
EFFECT_TASK_CARD_SCHEMA_VERSION = "observerbench.effect_task_card.v0"
EFFECT_OBSERVER_CARD_SCHEMA_VERSION = "observerbench.effect_observer_card.v0"

EFFECT_PREDICTION_CSV_COLUMNS = (
    "schema_version",
    "query_id",
    "predicted_effect",
)


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite real number")
    return result


@dataclass(frozen=True)
class FiniteEffectMeasurement(Generic[FeatureT]):
    """One task-supplied measurement available when fitting an observer."""

    measurement_id: str
    features: FeatureT
    observed_effect: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.measurement_id, "measurement_id")
        object.__setattr__(
            self,
            "observed_effect",
            _finite_float(self.observed_effect, "observed_effect"),
        )


@dataclass(frozen=True)
class FiniteEffectQuery(Generic[FeatureT]):
    """One held-out intervention for which an observer predicts an effect."""

    query_id: str
    features: FeatureT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.query_id, "query_id")


@dataclass(frozen=True)
class FiniteEffectTarget:
    """Benchmark-owned finite response for a held-out query."""

    query_id: str
    observed_effect: float

    def __post_init__(self) -> None:
        _nonempty(self.query_id, "query_id")
        object.__setattr__(
            self,
            "observed_effect",
            _finite_float(self.observed_effect, "observed_effect"),
        )


@dataclass(frozen=True)
class FiniteEffectPrediction:
    """Observer-supplied finite-effect prediction for one query."""

    query_id: str
    predicted_effect: float

    def __post_init__(self) -> None:
        _nonempty(self.query_id, "query_id")
        object.__setattr__(
            self,
            "predicted_effect",
            _finite_float(self.predicted_effect, "predicted_effect"),
        )


@runtime_checkable
class FiniteEffectPredictor(Protocol[FeatureT]):
    """Structural interface for an externally implemented effect observer."""

    name: str

    def fit(self, measurements: Sequence[FiniteEffectMeasurement[FeatureT]]) -> None:
        """Fit using only the measurements supplied by the task."""

    def predict(self, queries: Sequence[FiniteEffectQuery[FeatureT]]) -> Sequence[float]:
        """Predict one finite effect per held-out query, in query order."""


@dataclass(frozen=True)
class EffectTaskCard:
    """Machine-readable metadata for a finite-effect prediction task."""

    task_name: str
    task_version: str
    summary: str
    model_or_substrate: str
    access_regime: str
    estimand: str
    intervention_family: str
    measurement_design: str
    validation_target: str
    train_split: str
    evaluation_split: str
    primary_metrics: tuple[str, ...] = ("mae", "rmse")
    known_scope_limits: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(
        default=EFFECT_TASK_CARD_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "task_name",
            "task_version",
            "summary",
            "model_or_substrate",
            "access_regime",
            "estimand",
            "intervention_family",
            "measurement_design",
            "validation_target",
            "train_split",
            "evaluation_split",
        ):
            _nonempty(getattr(self, field_name), field_name)
        if not self.primary_metrics or any(
            not isinstance(metric, str) or not metric.strip()
            for metric in self.primary_metrics
        ):
            raise ValueError("primary_metrics must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectObserverCard:
    """Machine-readable method metadata paired with a task evaluation."""

    observer_name: str
    observer_version: str
    observer_family: str
    access_regime: str
    measurement_basis: str
    fit_procedure: str
    implementation: str
    known_failure_modes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(
        default=EFFECT_OBSERVER_CARD_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "observer_name",
            "observer_version",
            "observer_family",
            "access_regime",
            "measurement_basis",
            "fit_procedure",
            "implementation",
        ):
            _nonempty(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FiniteEffectPredictionTask(Generic[FeatureT]):
    """A fixed measurement split and held-out finite-effect evaluation."""

    name: str
    version: str
    measurements: Sequence[FiniteEffectMeasurement[FeatureT]]
    queries: Sequence[FiniteEffectQuery[FeatureT]]
    targets: Sequence[FiniteEffectTarget]
    card: EffectTaskCard

    def __post_init__(self) -> None:
        _nonempty(self.name, "name")
        _nonempty(self.version, "version")
        object.__setattr__(self, "measurements", tuple(self.measurements))
        object.__setattr__(self, "queries", tuple(self.queries))
        object.__setattr__(self, "targets", tuple(self.targets))
        if self.card.task_name != self.name or self.card.task_version != self.version:
            raise ValueError("task name and version must match the task card")
        _require_unique_ids(
            (measurement.measurement_id for measurement in self.measurements),
            "measurement",
        )
        query_ids = _require_unique_ids(
            (query.query_id for query in self.queries),
            "query",
        )
        target_ids = _require_unique_ids(
            (target.query_id for target in self.targets),
            "target",
        )
        if query_ids != target_ids:
            raise ValueError("queries and targets must have exactly the same IDs")

    @property
    def measurement_budget(self) -> int:
        return len(self.measurements)


@dataclass(frozen=True)
class EffectPredictionResult:
    """Generic held-out prediction metrics plus evaluation provenance."""

    task_name: str
    task_version: str
    observer_name: str
    observer_version: str
    measurement_budget: int
    n_queries: int
    metrics: Mapping[str, float]
    contract_version: str = EFFECT_PREDICTION_CONTRACT_VERSION
    schema_version: str = EFFECT_PREDICTION_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_unique_ids(values: Iterable[str], kind: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        _nonempty(value, f"{kind}_id")
        if value in seen:
            raise ValueError(f"duplicate {kind} ID: {value}")
        seen.add(value)
    if not seen:
        raise ValueError(f"at least one {kind} is required")
    return seen


def write_effect_predictions(
    path: str | Path,
    predictions: Sequence[FiniteEffectPrediction],
) -> None:
    """Write the exact v0 CSV interchange format."""

    _require_unique_ids(
        (prediction.query_id for prediction in predictions),
        "prediction",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EFFECT_PREDICTION_CSV_COLUMNS)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "schema_version": EFFECT_PREDICTION_CSV_SCHEMA_VERSION,
                    "query_id": prediction.query_id,
                    "predicted_effect": repr(prediction.predicted_effect),
                }
            )


def read_effect_predictions(path: str | Path) -> list[FiniteEffectPrediction]:
    """Read and validate the exact v0 CSV interchange format."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EFFECT_PREDICTION_CSV_COLUMNS:
            raise ValueError(
                "prediction CSV columns must be exactly "
                + ", ".join(EFFECT_PREDICTION_CSV_COLUMNS)
            )
        predictions: list[FiniteEffectPrediction] = []
        for row_number, row in enumerate(reader, start=2):
            if row["schema_version"] != EFFECT_PREDICTION_CSV_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported prediction CSV schema at row {row_number}: "
                    f"{row['schema_version']}"
                )
            try:
                value = float(row["predicted_effect"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"predicted_effect at row {row_number} must be a finite real number"
                ) from error
            predictions.append(
                FiniteEffectPrediction(
                    query_id=row["query_id"],
                    predicted_effect=value,
                )
            )
    _require_unique_ids(
        (prediction.query_id for prediction in predictions),
        "prediction",
    )
    return predictions


def _prediction_metrics(
    targets: Sequence[FiniteEffectTarget],
    predictions: Sequence[FiniteEffectPrediction],
) -> dict[str, float]:
    target_by_id = {target.query_id: target.observed_effect for target in targets}
    prediction_by_id = {
        prediction.query_id: prediction.predicted_effect
        for prediction in predictions
    }
    target_ids = set(target_by_id)
    prediction_ids = set(prediction_by_id)
    if prediction_ids != target_ids:
        missing = sorted(target_ids - prediction_ids)
        unexpected = sorted(prediction_ids - target_ids)
        raise ValueError(
            f"prediction IDs do not match task queries; missing={missing}, "
            f"unexpected={unexpected}"
        )

    errors = [prediction_by_id[key] - target_by_id[key] for key in sorted(target_ids)]
    absolute = [abs(error) for error in errors]
    squared = [error * error for error in errors]
    count = len(errors)
    return {
        "mae": sum(absolute) / count,
        "rmse": math.sqrt(sum(squared) / count),
        "mean_error": sum(errors) / count,
        "max_absolute_error": max(absolute),
    }


def evaluate_effect_predictions(
    task: FiniteEffectPredictionTask[Any],
    predictions: Sequence[FiniteEffectPrediction],
    observer_card: EffectObserverCard,
    *,
    outdir: str | Path | None = None,
) -> EffectPredictionResult:
    """Evaluate predictions and optionally write the portable result bundle."""

    _require_unique_ids(
        (prediction.query_id for prediction in predictions),
        "prediction",
    )
    result = EffectPredictionResult(
        task_name=task.name,
        task_version=task.version,
        observer_name=observer_card.observer_name,
        observer_version=observer_card.observer_version,
        measurement_budget=task.measurement_budget,
        n_queries=len(task.queries),
        metrics=_prediction_metrics(task.targets, predictions),
    )
    if outdir is not None:
        output = Path(outdir)
        output.mkdir(parents=True, exist_ok=True)
        write_effect_predictions(output / "effect_predictions.csv", predictions)
        write_json(output / "effect_evaluation.json", result.to_dict())
        write_json(output / "task_card.json", task.card.to_dict())
        write_json(output / "observer_card.json", observer_card.to_dict())
    return result


def evaluate_effect_prediction_csv(
    task: FiniteEffectPredictionTask[Any],
    prediction_csv: str | Path,
    observer_card: EffectObserverCard,
    *,
    outdir: str | Path | None = None,
) -> EffectPredictionResult:
    """Evaluate a CSV submission without importing its implementation."""

    return evaluate_effect_predictions(
        task,
        read_effect_predictions(prediction_csv),
        observer_card,
        outdir=outdir,
    )


def run_effect_prediction_task(
    task: FiniteEffectPredictionTask[FeatureT],
    predictor: FiniteEffectPredictor[FeatureT],
    observer_card: EffectObserverCard,
    *,
    outdir: str | Path,
) -> EffectPredictionResult:
    """Fit an external predictor on the task budget and evaluate held-out effects."""

    _nonempty(predictor.name, "predictor.name")
    if observer_card.observer_name != predictor.name:
        raise ValueError("predictor name must match observer card")
    predictor.fit(tuple(task.measurements))
    raw_predictions = list(predictor.predict(tuple(task.queries)))
    if len(raw_predictions) != len(task.queries):
        raise ValueError(
            "predictor must return exactly one prediction per query: "
            f"expected {len(task.queries)}, got {len(raw_predictions)}"
        )
    predictions = [
        FiniteEffectPrediction(
            query_id=query.query_id,
            predicted_effect=value,
        )
        for query, value in zip(task.queries, raw_predictions)
    ]
    return evaluate_effect_predictions(
        task,
        predictions,
        observer_card,
        outdir=outdir,
    )
