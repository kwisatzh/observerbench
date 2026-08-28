"""ObserverBench reproduction CLI.

v0 is a paper reproduction workbench, not a general benchmark platform.
"""

from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

import os
import tempfile
from pathlib import Path

from observerbench.ai_control import (
    AI_CONTROL_RESULT_SCHEMA_VERSION,
    AI_CONTROL_SCORE_SCHEMA_VERSION,
    AIControlMonitorResult,
    AIControlSample,
    evaluate_ai_control_csv,
    evaluate_ai_control_scores,
    read_ai_control_scores,
    write_ai_control_scores,
)
from observerbench.ai_control_public import (
    AI_CONTROL_PRIVATE_TARGET_SCHEMA_VERSION,
    AI_CONTROL_PUBLIC_SCORE_SCHEMA_VERSION,
    PrivateAIControlTarget,
    PublicAIControlScore,
    evaluate_public_ai_control_csv,
    evaluate_public_ai_control_scores,
    join_public_ai_control_scores,
    read_private_ai_control_targets,
    read_public_ai_control_scores,
)
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
from observerbench.safety import (
    SAFETY_PROTOCOL_CONTRACT_VERSION,
    SafetyAction,
    SafetyMeasurement,
    SafetyObserverCard,
    SafetyPolicy,
    SafetyProtocolResult,
    SafetyQuery,
    SafetyRiskPredictor,
    SafetyTarget,
    SafetyTask,
    SafetyTaskCard,
    evaluate_safety_prediction_csv,
    evaluate_safety_predictor,
    evaluate_safety_scores,
    read_safety_predictions,
    write_safety_predictions,
)
from observerbench.safety_task_pack import (
    SAFETY_TASK_PACK_SCHEMA_VERSION,
    SAFETY_TASK_TARGETS_SCHEMA_VERSION,
    load_safety_task_pack,
    validate_public_safety_task_pack,
    write_safety_task_pack,
)
from observerbench.safety_leaderboard import (
    SAFETY_LEADERBOARD_SCHEMA_VERSION,
    SafetyLeaderboardRow,
    compare_safety_result,
    format_safety_leaderboard,
    load_safety_leaderboard,
)

__all__ = [
    "API_CONTRACT_VERSION",
    "AI_CONTROL_RESULT_SCHEMA_VERSION",
    "AI_CONTROL_SCORE_SCHEMA_VERSION",
    "AI_CONTROL_PRIVATE_TARGET_SCHEMA_VERSION",
    "AI_CONTROL_PUBLIC_SCORE_SCHEMA_VERSION",
    "AIControlMonitorResult",
    "AIControlSample",
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
    "PrivateAIControlTarget",
    "PublicAIControlScore",
    "SAFETY_PROTOCOL_CONTRACT_VERSION",
    "SAFETY_LEADERBOARD_SCHEMA_VERSION",
    "SAFETY_TASK_PACK_SCHEMA_VERSION",
    "SAFETY_TASK_TARGETS_SCHEMA_VERSION",
    "SafetyAction",
    "SafetyLeaderboardRow",
    "SafetyMeasurement",
    "SafetyObserverCard",
    "SafetyPolicy",
    "SafetyProtocolResult",
    "SafetyQuery",
    "SafetyRiskPredictor",
    "SafetyTarget",
    "SafetyTask",
    "SafetyTaskCard",
    "StateEstimator",
    "__version__",
    "evaluate_effect_prediction_csv",
    "evaluate_effect_predictions",
    "evaluate_ai_control_csv",
    "evaluate_ai_control_scores",
    "evaluate_public_ai_control_csv",
    "evaluate_public_ai_control_scores",
    "evaluate_safety_prediction_csv",
    "evaluate_safety_predictor",
    "evaluate_safety_scores",
    "compare_safety_result",
    "format_safety_leaderboard",
    "load_safety_leaderboard",
    "join_public_ai_control_scores",
    "read_effect_predictions",
    "read_ai_control_scores",
    "read_private_ai_control_targets",
    "read_public_ai_control_scores",
    "read_safety_predictions",
    "load_safety_task_pack",
    "run_observer_task",
    "run_effect_prediction_task",
    "write_effect_predictions",
    "write_ai_control_scores",
    "write_safety_predictions",
    "validate_public_safety_task_pack",
    "write_safety_task_pack",
]

__version__ = "0.1.0"


def _ensure_matplotlib_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "observerbench-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


_ensure_matplotlib_cache()
