"""ControlArena APPS observer compositions.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from observerbench.tasks.controlarena_apps.sae_probe import (
    BatchTopKSAE,
    FrozenAppsRow,
    SAEEncoding,
    SparseFeatureMatrix,
    context_length_diagnostics,
    encode_batch_topk_sae,
    load_frozen_apps_design,
    load_neutral_activation_cache,
    load_sae_encoding,
    reconstruction_diagnostics,
    select_dense_logistic_observer,
    select_sparse_logistic_observer,
    write_sae_encoding,
)

__all__ = [
    "BatchTopKSAE",
    "FrozenAppsRow",
    "SAEEncoding",
    "SparseFeatureMatrix",
    "context_length_diagnostics",
    "encode_batch_topk_sae",
    "load_frozen_apps_design",
    "load_neutral_activation_cache",
    "load_sae_encoding",
    "reconstruction_diagnostics",
    "select_dense_logistic_observer",
    "select_sparse_logistic_observer",
    "write_sae_encoding",
]
