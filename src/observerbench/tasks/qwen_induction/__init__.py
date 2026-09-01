"""Qwen induction-copy finite-effect task and optional experiment producer.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from observerbench.tasks.qwen_induction.effect_task import (
    DEFAULT_QWEN_INDUCTION_EFFECT_ARTIFACT_ROOT,
    QWEN_INDUCTION_ADDITIVE_BASELINE_NAME,
    QWEN_INDUCTION_ADDITIVE_BASELINE_VERSION,
    QWEN_INDUCTION_EFFECT_DATA_VERSION,
    QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS,
    QWEN_INDUCTION_EFFECT_TASK_NAME,
    QWEN_INDUCTION_MASK_FEATURE_SCHEMA,
    QWEN_INDUCTION_MODEL_NAME,
    QWEN_INDUCTION_MODEL_REVISION,
    InductionMaskFeatures,
    QwenInductionAdditiveRidgeBaseline,
    load_qwen_induction_effect_prediction_task,
    qwen_induction_additive_baseline_card,
    qwen_induction_effect_task_version,
)
from observerbench.tasks.qwen_induction.attribution_patching import (
    QWEN_ATP_BASELINE_VERSION,
    QWEN_CALIBRATED_ATP_BASELINE_NAME,
    QWEN_RAW_ATP_BASELINE_NAME,
    QwenInductionAttributionMap,
    QwenInductionAttributionPatchingBaseline,
    load_qwen_reference_means,
    load_qwen_train_prompts,
    measure_qwen_induction_attribution_map,
    qwen_induction_attribution_patching_card,
)

__all__ = [
    "DEFAULT_QWEN_INDUCTION_EFFECT_ARTIFACT_ROOT",
    "InductionMaskFeatures",
    "QWEN_ATP_BASELINE_VERSION",
    "QWEN_INDUCTION_ADDITIVE_BASELINE_NAME",
    "QWEN_INDUCTION_ADDITIVE_BASELINE_VERSION",
    "QWEN_INDUCTION_EFFECT_DATA_VERSION",
    "QWEN_INDUCTION_EFFECT_MEASUREMENT_BUDGETS",
    "QWEN_INDUCTION_EFFECT_TASK_NAME",
    "QWEN_INDUCTION_MASK_FEATURE_SCHEMA",
    "QWEN_INDUCTION_MODEL_NAME",
    "QWEN_INDUCTION_MODEL_REVISION",
    "QWEN_CALIBRATED_ATP_BASELINE_NAME",
    "QWEN_RAW_ATP_BASELINE_NAME",
    "QwenInductionAdditiveRidgeBaseline",
    "QwenInductionAttributionMap",
    "QwenInductionAttributionPatchingBaseline",
    "load_qwen_reference_means",
    "load_qwen_train_prompts",
    "load_qwen_induction_effect_prediction_task",
    "measure_qwen_induction_attribution_map",
    "qwen_induction_additive_baseline_card",
    "qwen_induction_attribution_patching_card",
    "qwen_induction_effect_task_version",
]
