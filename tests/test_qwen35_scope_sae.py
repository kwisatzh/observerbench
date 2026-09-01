"""Focused tests for the official Qwen-Scope TopK SAE adapter.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from observerbench.tasks.qwen35_apps.qwen_scope import (
    QwenScopeTopKParameters,
    decode_qwen_scope_features,
    encode_qwen_scope_features,
    load_qwen_scope_parameters,
)


def _parameters() -> QwenScopeTopKParameters:
    return QwenScopeTopKParameters(
        w_enc=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        w_dec=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        b_enc=np.zeros(5, dtype=np.float32),
        b_dec=np.asarray([0.5, -0.5, 1.0], dtype=np.float32),
        top_k=2,
    )


def test_numpy_encoder_keeps_exact_top_k_and_decoder_uses_released_orientation() -> None:
    parameters = _parameters()
    residuals = np.asarray([[1.0, 2.0, 0.0], [1.0, -1.0, 2.0]], dtype=np.float32)

    features = parameters.encode_numpy(residuals)
    decoded = parameters.decode_numpy(features)

    np.testing.assert_allclose(features[0], [0.0, 2.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(features[1], [1.0, 0.0, 2.0, 0.0, 0.0])
    assert np.all(np.count_nonzero(features, axis=1) == 2)
    np.testing.assert_allclose(decoded[0], [0.5, 1.5, 1.0])
    np.testing.assert_allclose(decoded[1], [1.5, -0.5, 3.0])
    np.testing.assert_allclose(parameters.reconstruct_numpy(residuals), decoded)


def test_accelerated_cpu_path_matches_numpy_reference() -> None:
    parameters = _parameters()
    residuals = np.asarray([[1.0, 2.0, 0.0], [1.0, -1.0, 2.0]], dtype=np.float32)

    features = encode_qwen_scope_features(
        residuals, parameters, device="cpu", batch_size=1
    )
    decoded = decode_qwen_scope_features(
        features, parameters, device="cpu", batch_size=1
    )

    np.testing.assert_allclose(features, parameters.encode_numpy(residuals))
    np.testing.assert_allclose(decoded, parameters.decode_numpy(features))
    assert features.dtype == np.float32
    assert decoded.dtype == np.float32


def test_loader_accepts_only_the_four_tensor_contract(tmp_path) -> None:
    parameters = _parameters()
    path = tmp_path / "layer12.sae.pt"
    torch.save(
        {
            "W_enc": torch.from_numpy(parameters.w_enc),
            "W_dec": torch.from_numpy(parameters.w_dec),
            "b_enc": torch.from_numpy(parameters.b_enc),
            "b_dec": torch.from_numpy(parameters.b_dec),
        },
        path,
    )

    loaded = load_qwen_scope_parameters(path, top_k=2)

    assert loaded.d_in == 3
    assert loaded.width == 5
    assert loaded.top_k == 2
    np.testing.assert_allclose(loaded.w_enc, parameters.w_enc)
    np.testing.assert_allclose(loaded.w_dec, parameters.w_dec)


def test_loader_rejects_a_missing_released_tensor(tmp_path) -> None:
    path = tmp_path / "bad.sae.pt"
    torch.save(
        {
            "W_enc": torch.zeros((5, 3)),
            "W_dec": torch.zeros((3, 5)),
            "b_enc": torch.zeros(5),
        },
        path,
    )

    with pytest.raises(ValueError, match="b_dec"):
        load_qwen_scope_parameters(path, top_k=2)


def test_parameter_contract_rejects_shape_and_finiteness_drift() -> None:
    parameters = _parameters()

    with pytest.raises(ValueError, match="decoder shape"):
        QwenScopeTopKParameters(
            w_enc=parameters.w_enc,
            w_dec=np.zeros((5, 3), dtype=np.float32),
            b_enc=parameters.b_enc,
            b_dec=parameters.b_dec,
            top_k=2,
        )
    bad_encoder = parameters.w_enc.copy()
    bad_encoder[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and numeric"):
        QwenScopeTopKParameters(
            w_enc=bad_encoder,
            w_dec=parameters.w_dec,
            b_enc=parameters.b_enc,
            b_dec=parameters.b_dec,
            top_k=2,
        )
