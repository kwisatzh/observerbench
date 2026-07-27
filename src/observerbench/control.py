"""Reusable control and residual-support primitives for ObserverBench.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from observerbench.core import ControlRequest


def _finite_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


@dataclass(frozen=True)
class AffineStateEstimator:
    """A fitted scalar estimator ``intercept + gradient @ state``."""

    name: str
    intercept: float
    gradient: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "gradient", _finite_vector(self.gradient, name="gradient"))

    def estimate(self, state: np.ndarray) -> float:
        value = _finite_vector(state, name="state")
        if value.shape != self.gradient.shape:
            raise ValueError("state and estimator gradient must have the same shape")
        return float(self.intercept + self.gradient @ value)

    def estimate_batch(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.gradient):
            raise ValueError("states must have shape (n, state_dimension)")
        if not np.isfinite(values).all():
            raise ValueError("states must contain only finite values")
        return self.intercept + values @ self.gradient

    def affine_calibration(self, offset: float, scale: float, *, name: str | None = None) -> "AffineStateEstimator":
        """Return ``offset + scale * estimate(state)`` as another estimator."""

        if not np.isfinite([offset, scale]).all():
            raise ValueError("calibration offset and scale must be finite")
        return AffineStateEstimator(
            name=name or f"{self.name}_affine_calibrated",
            intercept=float(offset + scale * self.intercept),
            gradient=float(scale) * self.gradient,
        )


@dataclass(frozen=True)
class AffineSubspaceAdequacyDiagnostics:
    """Agreement of two affine readouts on an intervention-reachable set.

    The reachable set is ``operating_state + span(intervention_basis)``.
    ``operating_point_mismatch`` and ``gradient_mismatch`` use the convention
    ``target - estimator``.  For affine readouts, ``adequate`` is an exact
    certificate (up to the declared tolerances): the readouts agree everywhere
    on that set if and only if they agree at the operating point and their
    projected gradient mismatch is zero.
    """

    reachable_rank: int
    operating_point_mismatch: float
    projected_gradient_mismatch: np.ndarray
    projected_gradient_mismatch_norm: float
    orthogonal_gradient_mismatch_norm: float
    global_gradient_mismatch_norm: float
    level_agreement: bool
    response_agreement: bool
    adequate: bool


def affine_subspace_adequacy(
    target: AffineStateEstimator,
    estimator: AffineStateEstimator,
    operating_state: np.ndarray,
    intervention_basis: np.ndarray,
    *,
    level_tolerance: float = 1e-9,
    response_tolerance: float = 1e-9,
    rank_tolerance: float = 1e-12,
) -> AffineSubspaceAdequacyDiagnostics:
    """Test whether an affine estimator is adequate on a reachable subspace.

    Columns of ``intervention_basis`` span the allowed intervention directions;
    they need not be orthogonal, normalized, or linearly independent.  The
    diagnostic deliberately distinguishes task-specific adequacy from global
    readout equality: mismatch orthogonal to the reachable subspace is reported
    but does not make the estimator inadequate for this intervention family.
    """

    state = _finite_vector(operating_state, name="operating_state")
    if target.gradient.shape != estimator.gradient.shape or target.gradient.shape != state.shape:
        raise ValueError("target, estimator, and operating state must have the same dimension")

    basis = np.asarray(intervention_basis, dtype=float)
    if basis.ndim == 1:
        basis = basis[:, None]
    if basis.ndim != 2 or basis.shape[0] != len(state):
        raise ValueError("intervention_basis must have shape (state_dimension, n_directions)")
    if not np.isfinite(basis).all():
        raise ValueError("intervention_basis must contain only finite values")
    tolerances = np.asarray([level_tolerance, response_tolerance, rank_tolerance], dtype=float)
    if not np.isfinite(tolerances).all() or np.any(tolerances < 0.0):
        raise ValueError("adequacy tolerances must be finite and non-negative")

    if basis.shape[1] == 0:
        orthonormal_basis = np.zeros((len(state), 0), dtype=float)
    else:
        left, singular_values, _right = np.linalg.svd(basis, full_matrices=False)
        leading = float(singular_values[0]) if len(singular_values) else 0.0
        absolute_rank_tolerance = float(rank_tolerance * max(leading, 1.0))
        rank = int(np.sum(singular_values > absolute_rank_tolerance))
        orthonormal_basis = left[:, :rank]

    gradient_mismatch = target.gradient - estimator.gradient
    if orthonormal_basis.shape[1]:
        projected = orthonormal_basis @ (orthonormal_basis.T @ gradient_mismatch)
    else:
        projected = np.zeros_like(gradient_mismatch)
    orthogonal = gradient_mismatch - projected
    operating_point_mismatch = float(target.estimate(state) - estimator.estimate(state))
    projected_norm = float(np.linalg.norm(projected))
    level_agreement = bool(abs(operating_point_mismatch) <= level_tolerance)
    response_agreement = bool(projected_norm <= response_tolerance)

    return AffineSubspaceAdequacyDiagnostics(
        reachable_rank=int(orthonormal_basis.shape[1]),
        operating_point_mismatch=operating_point_mismatch,
        projected_gradient_mismatch=projected,
        projected_gradient_mismatch_norm=projected_norm,
        orthogonal_gradient_mismatch_norm=float(np.linalg.norm(orthogonal)),
        global_gradient_mismatch_norm=float(np.linalg.norm(gradient_mismatch)),
        level_agreement=level_agreement,
        response_agreement=response_agreement,
        adequate=bool(level_agreement and response_agreement),
    )


@dataclass(frozen=True)
class ScalarAffineErrorPrediction:
    """Exact finite-horizon errors for an unsaturated fixed-direction loop."""

    horizon: int
    observer_error_pole: float
    cumulative_action: float
    final_observer_error: float
    final_target_error: float


def predict_unsaturated_affine_scalar_errors(
    *,
    initial_observer_error: float,
    initial_target_error: float,
    observer_response_gain: float,
    target_response_gain: float,
    controller_gain: float,
    horizon: int,
) -> ScalarAffineErrorPrediction:
    """Predict exact errors for ``h <- h + K (y* - zhat(h)) d``.

    Errors use ``reference - readout`` and response gains are the corresponding
    affine gradients dotted with the fixed direction.  The formula includes the
    zero-observer-gain case, where the observer error stays constant while the
    target can drift linearly.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    values = np.asarray(
        [
            initial_observer_error,
            initial_target_error,
            observer_response_gain,
            target_response_gain,
            controller_gain,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("scalar affine loop inputs must be finite")

    pole = float(1.0 - controller_gain * observer_response_gain)
    pole_power = float(pole ** int(horizon))
    loop_gain = float(controller_gain * observer_response_gain)
    if loop_gain == 0.0:
        geometric_sum = float(horizon)
    else:
        geometric_sum = float((1.0 - pole_power) / loop_gain)
    cumulative_action = float(controller_gain * initial_observer_error * geometric_sum)

    return ScalarAffineErrorPrediction(
        horizon=int(horizon),
        observer_error_pole=pole,
        cumulative_action=cumulative_action,
        final_observer_error=float(pole_power * initial_observer_error),
        final_target_error=float(initial_target_error - target_response_gain * cumulative_action),
    )


@dataclass(frozen=True)
class FixedDirectionProvider:
    """A state-independent direction in a task's intervention space."""

    name: str
    vector: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", _finite_vector(self.vector, name="direction"))

    def direction(self, state: np.ndarray) -> np.ndarray:
        del state
        return self.vector.copy()


@dataclass(frozen=True)
class ClippedProportionalController:
    """A scalar proportional controller with symmetric clipping."""

    gain: float
    max_action: float
    name: str = "clipped_proportional"

    def __post_init__(self) -> None:
        if not np.isfinite(self.gain):
            raise ValueError("controller gain must be finite")
        if not np.isfinite(self.max_action) or self.max_action <= 0:
            raise ValueError("max_action must be positive and finite")

    def raw_action(self, request: ControlRequest[np.ndarray, float]) -> float:
        return float(self.gain * (request.target - request.estimate))

    def action(self, request: ControlRequest[np.ndarray, float]) -> float:
        return float(np.clip(self.raw_action(request), -self.max_action, self.max_action))


@dataclass(frozen=True)
class LoopGainDiagnostics:
    """Local unsaturated diagnostics for one estimator--direction pair."""

    target_gain: float
    observer_self_gain: float
    mismatch_projection: float
    omitted_response_gain: float
    normalized_omitted_response: float
    true_to_observer_response_ratio: float
    true_response_direction_compatible: bool
    observer_error_pole: float
    sign_compatible: bool
    locally_convergent_unsaturated: bool


def loop_gain_diagnostics(
    target_gradient: np.ndarray,
    estimator_gradient: np.ndarray,
    direction: np.ndarray,
    controller_gain: float,
    *,
    tolerance: float = 1e-12,
) -> LoopGainDiagnostics:
    """Compute gains for ``h <- h + K(y* - zhat) d`` before clipping."""

    target = _finite_vector(target_gradient, name="target_gradient")
    estimator = _finite_vector(estimator_gradient, name="estimator_gradient")
    actuator = _finite_vector(direction, name="direction")
    if target.shape != estimator.shape or target.shape != actuator.shape:
        raise ValueError("target, estimator, and direction vectors must have the same shape")
    if not np.isfinite(controller_gain):
        raise ValueError("controller_gain must be finite")

    target_gain = float(target @ actuator)
    self_gain = float(estimator @ actuator)
    omitted_gain = float(target_gain - self_gain)
    if abs(self_gain) > tolerance:
        normalized_omitted = float(omitted_gain / self_gain)
        true_to_observer = float(target_gain / self_gain)
        true_response_compatible = bool(true_to_observer > 0.0)
    else:
        normalized_omitted = float("nan")
        true_to_observer = float("nan")
        true_response_compatible = False
    pole = float(1.0 - controller_gain * self_gain)
    return LoopGainDiagnostics(
        target_gain=target_gain,
        observer_self_gain=self_gain,
        mismatch_projection=float(self_gain - target_gain),
        omitted_response_gain=omitted_gain,
        normalized_omitted_response=normalized_omitted,
        true_to_observer_response_ratio=true_to_observer,
        true_response_direction_compatible=true_response_compatible,
        observer_error_pole=pole,
        sign_compatible=bool(controller_gain * self_gain > tolerance),
        locally_convergent_unsaturated=bool(abs(pole) < 1.0),
    )


def positive_gain_matched_controller_gain(
    *,
    base_controller_gain: float,
    reference_self_gain: float,
    arm_self_gain: float,
    tolerance: float = 1e-12,
) -> float | None:
    """Match the reference unsaturated pole when the arm has positive gain.

    ``None`` marks a zero- or negative-self-gain arm as ineligible.  A negative
    controller gain would reverse target authority and is not a calibration.
    """

    reference_loop_gain = float(base_controller_gain * reference_self_gain)
    if not 0.0 < reference_loop_gain < 2.0:
        raise ValueError("reference K*self_gain must lie strictly between 0 and 2")
    if arm_self_gain <= tolerance:
        return None
    return float(reference_loop_gain / arm_self_gain)


@dataclass(frozen=True)
class AffineSupport:
    """Affine support estimated from distinct clean residual-state centroids."""

    center: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    absolute_tolerance: float
    clean_rms_radius: float
    n_centroids: int

    def __post_init__(self) -> None:
        center = _finite_vector(self.center, name="support center")
        basis = np.asarray(self.basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != len(center):
            raise ValueError("support basis must have shape (state_dimension, rank)")
        if not np.isfinite(basis).all():
            raise ValueError("support basis must be finite")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "basis", basis.copy())
        object.__setattr__(self, "singular_values", _finite_vector(self.singular_values, name="singular_values"))

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def project_direction(self, direction: np.ndarray) -> np.ndarray:
        vector = _finite_vector(direction, name="direction")
        if vector.shape != self.center.shape:
            raise ValueError("direction dimension does not match support")
        if self.rank == 0:
            return np.zeros_like(vector)
        return self.basis @ (self.basis.T @ vector)

    def off_support_component(self, direction: np.ndarray) -> np.ndarray:
        vector = _finite_vector(direction, name="direction")
        return vector - self.project_direction(vector)

    def off_support_distance(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        one = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != len(self.center):
            raise ValueError("state dimension does not match support")
        centered = values - self.center
        if self.rank:
            centered = centered - (centered @ self.basis) @ self.basis.T
        distances = np.linalg.norm(centered, axis=1)
        return distances[0] if one else distances

    def normalized_off_support_distance(self, states: np.ndarray) -> np.ndarray:
        scale = max(float(self.clean_rms_radius), 1e-12)
        return self.off_support_distance(states) / scale


def fit_affine_support(
    states: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    relative_tolerance: float = 1e-6,
) -> AffineSupport:
    """Fit an affine span, optionally after averaging repeated clean states."""

    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("states must be a nonempty finite matrix")
    if relative_tolerance <= 0 or not np.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be positive and finite")

    centroids = values
    if labels is not None:
        keys = np.asarray(labels)
        if keys.ndim == 1:
            keys = keys[:, None]
        if keys.shape[0] != values.shape[0]:
            raise ValueError("labels and states must have the same number of rows")
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        centroids = np.stack([values[inverse == idx].mean(axis=0) for idx in range(inverse.max() + 1)])

    center = centroids.mean(axis=0)
    centered = centroids - center
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    absolute_tolerance = float(relative_tolerance * max(leading, 1.0))
    rank = int(np.sum(singular_values > absolute_tolerance))
    basis = vh[:rank].T if rank else np.zeros((values.shape[1], 0), dtype=float)
    projected = centered @ basis if rank else np.zeros((len(centered), 0), dtype=float)
    radii = np.linalg.norm(projected, axis=1)
    clean_rms_radius = float(np.sqrt(np.mean(radii ** 2)))
    return AffineSupport(
        center=center,
        basis=basis,
        singular_values=singular_values,
        absolute_tolerance=absolute_tolerance,
        clean_rms_radius=clean_rms_radius,
        n_centroids=int(len(centroids)),
    )


def project_direction_to_target_gain(
    direction: np.ndarray,
    *,
    support: AffineSupport,
    target_gradient: np.ndarray,
    target_gain: float,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Project a final actuator into support and restore its target authority."""

    projected = support.project_direction(direction)
    target = _finite_vector(target_gradient, name="target_gradient")
    denominator = float(target @ projected)
    if abs(denominator) <= tolerance:
        raise ValueError("projected direction has no usable target authority")
    scale = float(target_gain / denominator)
    return scale * projected, scale


def direction_support_metrics(direction: np.ndarray, support: AffineSupport) -> dict[str, float]:
    """Return on/off-support norms and gain-independent geometry diagnostics."""

    vector = _finite_vector(direction, name="direction")
    on = support.project_direction(vector)
    off = vector - on
    norm = float(np.linalg.norm(vector))
    return {
        "direction_on_support_norm": float(np.linalg.norm(on)),
        "direction_off_support_norm": float(np.linalg.norm(off)),
        "direction_off_support_fraction": float(np.linalg.norm(off) / max(norm, 1e-12)),
    }
