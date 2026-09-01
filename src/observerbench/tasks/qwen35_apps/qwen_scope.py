"""Official Qwen-Scope TopK SAE adapter for the Qwen3.5 APPS study.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

The public Qwen3.5 checkpoints contain four tensors per residual-stream SAE.
This module implements only that declared contract and the released TopK
activation rule. It does not interpret or label individual features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _numeric_finite(value: np.ndarray, name: str) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"Qwen-Scope {name} must be finite and numeric")


@dataclass(frozen=True)
class QwenScopeTopKParameters:
    """The four released tensors needed to encode and decode one residual."""

    w_enc: np.ndarray
    w_dec: np.ndarray
    b_enc: np.ndarray
    b_dec: np.ndarray
    top_k: int = 50

    def __post_init__(self) -> None:
        w_enc = np.asarray(self.w_enc)
        w_dec = np.asarray(self.w_dec)
        b_enc = np.asarray(self.b_enc).reshape(-1)
        b_dec = np.asarray(self.b_dec).reshape(-1)
        if w_enc.ndim != 2 or w_dec.ndim != 2:
            raise ValueError("Qwen-Scope encoder and decoder weights must be matrices")
        width, d_in = w_enc.shape
        if w_dec.shape != (d_in, width):
            raise ValueError("Qwen-Scope decoder shape must transpose the encoder")
        if b_enc.shape != (width,) or b_dec.shape != (d_in,):
            raise ValueError("Qwen-Scope bias shapes differ from the released contract")
        if isinstance(self.top_k, bool) or not 0 < int(self.top_k) <= width:
            raise ValueError("Qwen-Scope top_k must lie between one and the SAE width")
        for name, value in (
            ("W_enc", w_enc),
            ("W_dec", w_dec),
            ("b_enc", b_enc),
            ("b_dec", b_dec),
        ):
            _numeric_finite(value, name)

    @property
    def d_in(self) -> int:
        return int(np.asarray(self.w_enc).shape[1])

    @property
    def width(self) -> int:
        return int(np.asarray(self.w_enc).shape[0])

    def encode_numpy(self, residuals: np.ndarray) -> np.ndarray:
        """Reference implementation of the released affine-then-TopK encoder."""

        values = np.asarray(residuals, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.d_in:
            raise ValueError("residuals must have row-by-d_in shape")
        pre = (
            values @ np.asarray(self.w_enc, dtype=np.float32).T
            + np.asarray(self.b_enc, dtype=np.float32).reshape(1, -1)
        )
        indices = np.argsort(-pre, axis=1, kind="stable")[:, : int(self.top_k)]
        features = np.zeros_like(pre, dtype=np.float32)
        np.put_along_axis(features, indices, np.take_along_axis(pre, indices, axis=1), axis=1)
        return features

    def decode_numpy(self, features: np.ndarray) -> np.ndarray:
        """Decode dense feature rows using the released decoder orientation."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError("features must have row-by-width shape")
        return (
            values @ np.asarray(self.w_dec, dtype=np.float32).T
            + np.asarray(self.b_dec, dtype=np.float32).reshape(1, -1)
        ).astype(np.float32)

    def reconstruct_numpy(self, residuals: np.ndarray) -> np.ndarray:
        return self.decode_numpy(self.encode_numpy(residuals))


def _as_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value).copy()
    except Exception as error:
        raise ValueError(f"Qwen-Scope {name} is not tensor-like") from error


def load_qwen_scope_parameters(
    path: str | Path, *, top_k: int = 50
) -> QwenScopeTopKParameters:
    """Load the four-tensor public checkpoint without enabling pickle objects."""

    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required to load Qwen-Scope checkpoints") from error
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Qwen-Scope checkpoint must contain a tensor mapping")
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Qwen-Scope checkpoint is missing: {', '.join(missing)}")
    return QwenScopeTopKParameters(
        w_enc=_as_numpy(payload["W_enc"], "W_enc"),
        w_dec=_as_numpy(payload["W_dec"], "W_dec"),
        b_enc=_as_numpy(payload["b_enc"], "b_enc"),
        b_dec=_as_numpy(payload["b_dec"], "b_dec"),
        top_k=int(top_k),
    )


def _torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[str(name)]
    except KeyError as error:
        raise ValueError(f"unsupported Qwen-Scope compute dtype {name!r}") from error


def encode_qwen_scope_features(
    residuals: np.ndarray,
    parameters: QwenScopeTopKParameters,
    *,
    device: str,
    dtype: str = "float32",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode residual rows on one device and return float32 feature rows."""

    if batch_size <= 0:
        raise ValueError("Qwen-Scope batch size must be positive")
    values = np.asarray(residuals)
    if values.ndim != 2 or values.shape[1] != parameters.d_in:
        raise ValueError("residuals must have row-by-d_in shape")
    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required for Qwen-Scope encoding") from error
    compute_dtype = _torch_dtype(torch, dtype)
    w_enc = torch.as_tensor(parameters.w_enc, dtype=compute_dtype, device=device)
    b_enc = torch.as_tensor(parameters.b_enc, dtype=compute_dtype, device=device)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size],
                dtype=compute_dtype,
                device=device,
            )
            pre = batch @ w_enc.T + b_enc
            top_values, top_indices = torch.topk(
                pre, k=int(parameters.top_k), dim=-1
            )
            features = torch.zeros_like(pre)
            features.scatter_(1, top_indices, top_values)
            chunks.append(features.to(dtype=torch.float32).cpu().numpy())
            del batch, pre, top_values, top_indices, features
    del w_enc, b_enc
    return (
        np.concatenate(chunks)
        if chunks
        else np.empty((0, parameters.width), dtype=np.float32)
    )


def decode_qwen_scope_features(
    features: np.ndarray,
    parameters: QwenScopeTopKParameters,
    *,
    device: str,
    dtype: str = "float32",
    batch_size: int = 64,
) -> np.ndarray:
    """Decode feature rows on one device and return float32 residual rows."""

    if batch_size <= 0:
        raise ValueError("Qwen-Scope batch size must be positive")
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != parameters.width:
        raise ValueError("features must have row-by-width shape")
    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required for Qwen-Scope decoding") from error
    compute_dtype = _torch_dtype(torch, dtype)
    w_dec = torch.as_tensor(parameters.w_dec, dtype=compute_dtype, device=device)
    b_dec = torch.as_tensor(parameters.b_dec, dtype=compute_dtype, device=device)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size],
                dtype=compute_dtype,
                device=device,
            )
            decoded = batch @ w_dec.T + b_dec
            chunks.append(decoded.to(dtype=torch.float32).cpu().numpy())
            del batch, decoded
    del w_dec, b_dec
    return (
        np.concatenate(chunks)
        if chunks
        else np.empty((0, parameters.d_in), dtype=np.float32)
    )


__all__ = [
    "QwenScopeTopKParameters",
    "decode_qwen_scope_features",
    "encode_qwen_scope_features",
    "load_qwen_scope_parameters",
]
