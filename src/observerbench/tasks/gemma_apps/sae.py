"""Minimal loader and encoder for official Gemma Scope residual SAEs.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GemmaScopeParameters:
    """The five arrays needed by an official Gemma Scope JumpReLU SAE."""

    w_enc: np.ndarray
    b_enc: np.ndarray
    w_dec: np.ndarray
    b_dec: np.ndarray
    threshold: np.ndarray

    def __post_init__(self) -> None:
        w_enc = np.asarray(self.w_enc)
        w_dec = np.asarray(self.w_dec)
        b_enc = np.asarray(self.b_enc).reshape(-1)
        b_dec = np.asarray(self.b_dec).reshape(-1)
        threshold = np.asarray(self.threshold).reshape(-1)
        if w_enc.ndim != 2 or w_dec.ndim != 2:
            raise ValueError("Gemma Scope encoder and decoder weights must be matrices")
        d_in, width = w_enc.shape
        if w_dec.shape != (width, d_in):
            raise ValueError("Gemma Scope decoder shape must transpose the encoder shape")
        if b_enc.shape != (width,) or threshold.shape not in ((1,), (width,)):
            raise ValueError("Gemma Scope feature bias or threshold has the wrong width")
        if b_dec.shape != (d_in,):
            raise ValueError("Gemma Scope decoder bias has the wrong input width")
        arrays = (w_enc, w_dec, b_enc, b_dec, threshold)
        if any(not np.issubdtype(value.dtype, np.number) for value in arrays):
            raise ValueError("Gemma Scope arrays must be numeric")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("Gemma Scope arrays must be finite")

    @property
    def d_in(self) -> int:
        return int(np.asarray(self.w_enc).shape[0])

    @property
    def width(self) -> int:
        return int(np.asarray(self.w_enc).shape[1])

    def encode_numpy(self, residuals: np.ndarray) -> np.ndarray:
        """Reference implementation used by tests and small CPU fixtures."""

        values = np.asarray(residuals, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.d_in:
            raise ValueError("residuals must have row-by-d_in shape")
        # Gemma Scope release files fold the training-time input-centering term
        # into b_enc. b_dec belongs to decoding and must not be subtracted again.
        pre = (
            values @ np.asarray(self.w_enc, dtype=np.float32)
            + np.asarray(self.b_enc, dtype=np.float32).reshape(1, -1)
        )
        threshold = np.asarray(self.threshold, dtype=np.float32).reshape(1, -1)
        return np.where(pre > threshold, pre, 0.0).astype(np.float32)


def load_gemma_scope_parameters(path: str | Path) -> GemmaScopeParameters:
    """Load only the stable five-array public Gemma Scope NPZ contract."""

    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"W_enc", "b_enc", "W_dec", "b_dec", "threshold"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Gemma Scope archive is missing: {', '.join(missing)}")
        return GemmaScopeParameters(
            w_enc=archive["W_enc"].copy(),
            b_enc=archive["b_enc"].copy(),
            w_dec=archive["W_dec"].copy(),
            b_dec=archive["b_dec"].copy(),
            threshold=archive["threshold"].copy(),
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
        raise ValueError(f"unsupported SAE compute dtype {name!r}") from error


def encode_gemma_scope_features(
    residuals: np.ndarray,
    parameters: GemmaScopeParameters,
    *,
    device: str,
    dtype: str = "float32",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode residual rows on the selected accelerator, returning float32 features."""

    if batch_size <= 0:
        raise ValueError("SAE encoding batch size must be positive")
    values = np.asarray(residuals)
    if values.ndim != 2 or values.shape[1] != parameters.d_in:
        raise ValueError("residuals must have row-by-d_in shape")
    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required for accelerator SAE encoding") from error
    compute_dtype = _torch_dtype(torch, dtype)
    w_enc = torch.as_tensor(parameters.w_enc, dtype=compute_dtype, device=device)
    b_enc = torch.as_tensor(parameters.b_enc, dtype=compute_dtype, device=device)
    threshold = torch.as_tensor(parameters.threshold, dtype=compute_dtype, device=device)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size],
                dtype=compute_dtype,
                device=device,
            )
            pre = batch @ w_enc + b_enc
            features = torch.where(pre > threshold, pre, torch.zeros_like(pre))
            chunks.append(features.to(dtype=torch.float32).cpu().numpy())
            del batch, pre, features
    del w_enc, b_enc, threshold
    return np.concatenate(chunks) if chunks else np.empty((0, parameters.width), np.float32)


__all__ = [
    "GemmaScopeParameters",
    "encode_gemma_scope_features",
    "load_gemma_scope_parameters",
]
