"""Focused tests for the official Gemma Scope NPZ adapter.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observerbench.tasks.gemma_apps.sae import (
    GemmaScopeParameters,
    encode_gemma_scope_features,
    load_gemma_scope_parameters,
)


def _parameters() -> GemmaScopeParameters:
    return GemmaScopeParameters(
        w_enc=np.asarray([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]], dtype=np.float32),
        b_enc=np.asarray([0.0, 0.5, -0.5], dtype=np.float32),
        w_dec=np.zeros((3, 2), dtype=np.float32),
        b_dec=np.asarray([1.0, -1.0], dtype=np.float32),
        threshold=np.asarray([0.25, 0.75, 0.0], dtype=np.float32),
    )


def test_jumprelu_uses_folded_encoder_bias_and_applies_threshold() -> None:
    parameters = _parameters()
    residual = np.asarray([[2.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    observed = parameters.encode_numpy(residual)

    expected_pre = residual @ parameters.w_enc + parameters.b_enc
    expected = np.where(expected_pre > parameters.threshold, expected_pre, 0.0)
    np.testing.assert_allclose(observed, expected)
    assert parameters.d_in == 2
    assert parameters.width == 3


def test_accelerator_encoder_matches_reference_on_cpu() -> None:
    parameters = _parameters()
    residual = np.asarray([[2.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    observed = encode_gemma_scope_features(
        residual, parameters, device="cpu", dtype="float32", batch_size=1
    )

    np.testing.assert_allclose(observed, parameters.encode_numpy(residual))
    assert observed.dtype == np.float32


def test_npz_loader_uses_official_five_array_contract(tmp_path: Path) -> None:
    parameters = _parameters()
    path = tmp_path / "params.npz"
    np.savez(
        path,
        W_enc=parameters.w_enc,
        b_enc=parameters.b_enc,
        W_dec=parameters.w_dec,
        b_dec=parameters.b_dec,
        threshold=parameters.threshold,
        unrelated_metadata=np.asarray([7]),
    )

    loaded = load_gemma_scope_parameters(path)

    np.testing.assert_array_equal(loaded.w_enc, parameters.w_enc)
    np.testing.assert_array_equal(loaded.threshold, parameters.threshold)


def test_loader_rejects_missing_or_incompatible_arrays(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    np.savez(missing, W_enc=np.eye(2))
    with pytest.raises(ValueError, match="missing"):
        load_gemma_scope_parameters(missing)

    with pytest.raises(ValueError, match="decoder shape"):
        GemmaScopeParameters(
            w_enc=np.zeros((2, 3)),
            b_enc=np.zeros(3),
            w_dec=np.zeros((2, 3)),
            b_dec=np.zeros(2),
            threshold=np.zeros(3),
        )
