"""Baseline observer implementations.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


class LinearObserver:
    """Small ridge-regression observer used by the MVP tasks."""

    def __init__(self, name: str, feature_fn: Callable[[dict], np.ndarray], ridge: float = 1e-6):
        self.name = name
        self.feature_fn = feature_fn
        self.ridge = ridge
        self.coef_: Optional[np.ndarray] = None

    def _design(self, data: dict) -> np.ndarray:
        X = np.asarray(self.feature_fn(data), dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return np.c_[np.ones(len(X)), X]

    def fit(self, data: dict, y: np.ndarray) -> "LinearObserver":
        X = self._design(data)
        y = np.asarray(y, dtype=float)
        reg = self.ridge * np.eye(X.shape[1])
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.solve(X.T @ X + reg, X.T @ y)
        return self

    def predict(self, data: dict) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Observer must be fit before predict().")
        return self._design(data) @ self.coef_


def first_order_features(data: dict) -> np.ndarray:
    return np.c_[data["x1"], data["x2"]]


def lifted_pair_features(data: dict) -> np.ndarray:
    return np.c_[data["x1"], data["x2"], data["x1"] * data["x2"]]


def interaction_only_features(data: dict) -> np.ndarray:
    return (data["x1"] * data["x2"])[:, None]
