"""Matched Gemma APPS replication utilities.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from observerbench.tasks.gemma_apps.hf_monitor import (
    EncodedMonitorView,
    HFCausalMonitor,
    integer_score_token_ids,
    validate_contextual_score_token_ids,
)
from observerbench.tasks.gemma_apps.registration import (
    load_gemma_apps_config,
    seal_gemma_apps_sources,
    verify_gemma_apps_source_manifest,
)
from observerbench.tasks.gemma_apps.sae import (
    GemmaScopeParameters,
    encode_gemma_scope_features,
    load_gemma_scope_parameters,
)

__all__ = [
    "EncodedMonitorView",
    "GemmaScopeParameters",
    "HFCausalMonitor",
    "encode_gemma_scope_features",
    "integer_score_token_ids",
    "validate_contextual_score_token_ids",
    "load_gemma_apps_config",
    "load_gemma_scope_parameters",
    "seal_gemma_apps_sources",
    "verify_gemma_apps_source_manifest",
]
