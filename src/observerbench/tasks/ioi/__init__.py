"""IOI diagnostics for the ObserverBench paper reproduction."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from observerbench.tasks.ioi.effect_task import (
    DEFAULT_IOI_EFFECT_ARTIFACT_ROOT,
    IOI_ADDITIVE_BASELINE_NAME,
    IOI_ADDITIVE_BASELINE_VERSION,
    IOI_EFFECT_DATA_VERSION,
    IOI_EFFECT_MEASUREMENT_BUDGETS,
    IOI_EFFECT_MODEL_REVISION,
    IOI_EFFECT_TASK_NAME,
    IOI_MASK_FEATURE_SCHEMA,
    IOIAdditiveRidgeBaseline,
    IOIMaskFeatures,
    ioi_additive_baseline_card,
    ioi_effect_task_version,
    load_ioi_effect_prediction_task,
)
from observerbench.tasks.ioi.attribution_patching import (
    IOI_ATP_BASELINE_VERSION,
    IOI_CALIBRATED_ATP_BASELINE_NAME,
    IOI_RAW_ATP_BASELINE_NAME,
    IOIAttributionMap,
    IOIAttributionPatchingBaseline,
    ioi_attribution_patching_card,
    load_template_head_means,
    measure_ioi_attribution_map,
)
from observerbench.tasks.ioi.heads import BACKUP_NAME_MOVERS, NAME_MOVERS, NEGATIVE_NAME_MOVERS

__all__ = [
    "BACKUP_NAME_MOVERS",
    "DEFAULT_IOI_EFFECT_ARTIFACT_ROOT",
    "IOI_ADDITIVE_BASELINE_NAME",
    "IOI_ADDITIVE_BASELINE_VERSION",
    "IOI_ATP_BASELINE_VERSION",
    "IOI_CALIBRATED_ATP_BASELINE_NAME",
    "IOI_EFFECT_DATA_VERSION",
    "IOI_EFFECT_MEASUREMENT_BUDGETS",
    "IOI_EFFECT_MODEL_REVISION",
    "IOI_EFFECT_TASK_NAME",
    "IOI_MASK_FEATURE_SCHEMA",
    "IOI_RAW_ATP_BASELINE_NAME",
    "IOIAdditiveRidgeBaseline",
    "IOIAttributionMap",
    "IOIAttributionPatchingBaseline",
    "IOIMaskFeatures",
    "NAME_MOVERS",
    "NEGATIVE_NAME_MOVERS",
    "ioi_additive_baseline_card",
    "ioi_attribution_patching_card",
    "ioi_effect_task_version",
    "load_ioi_effect_prediction_task",
    "load_template_head_means",
    "measure_ioi_attribution_map",
]
