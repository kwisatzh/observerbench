"""ObserverBench reproduction CLI.

v0 is a paper reproduction workbench, not a general benchmark platform.
"""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import os
import tempfile
from pathlib import Path

from observerbench.core import (
    API_CONTRACT_VERSION,
    ActuationDirectionProvider,
    Controller,
    ControlRequest,
    Observation,
    Observer,
    ObserverTask,
    StateEstimator,
    run_observer_task,
)
from observerbench.effect_prediction import (
    EFFECT_OBSERVER_CARD_SCHEMA_VERSION,
    EFFECT_PREDICTION_CONTRACT_VERSION,
    EFFECT_PREDICTION_CSV_SCHEMA_VERSION,
    EFFECT_PREDICTION_RESULT_SCHEMA_VERSION,
    EFFECT_TASK_CARD_SCHEMA_VERSION,
    EffectObserverCard,
    EffectPredictionResult,
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

__all__ = [
    "API_CONTRACT_VERSION",
    "ActuationDirectionProvider",
    "Controller",
    "ControlRequest",
    "EFFECT_OBSERVER_CARD_SCHEMA_VERSION",
    "EFFECT_PREDICTION_CONTRACT_VERSION",
    "EFFECT_PREDICTION_CSV_SCHEMA_VERSION",
    "EFFECT_PREDICTION_RESULT_SCHEMA_VERSION",
    "EFFECT_TASK_CARD_SCHEMA_VERSION",
    "EffectObserverCard",
    "EffectPredictionResult",
    "EffectTaskCard",
    "FiniteEffectMeasurement",
    "FiniteEffectPrediction",
    "FiniteEffectPredictionTask",
    "FiniteEffectPredictor",
    "FiniteEffectQuery",
    "FiniteEffectTarget",
    "Observation",
    "Observer",
    "ObserverTask",
    "StateEstimator",
    "__version__",
    "evaluate_effect_prediction_csv",
    "evaluate_effect_predictions",
    "read_effect_predictions",
    "run_observer_task",
    "run_effect_prediction_task",
    "write_effect_predictions",
]

__version__ = "0.0.0"


def _ensure_matplotlib_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "observerbench-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


_ensure_matplotlib_cache()
