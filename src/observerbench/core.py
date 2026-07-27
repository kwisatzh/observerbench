from __future__ import annotations

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from dataclasses import dataclass, field, asdict
import math
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Generic, List, Mapping, Optional, Protocol, TypeVar, runtime_checkable
import json


StateT = TypeVar("StateT")
EstimateT = TypeVar("EstimateT")
DirectionT = TypeVar("DirectionT")
RunOutputT = TypeVar("RunOutputT")

API_CONTRACT_VERSION = "observerbench.api.v0"


@runtime_checkable
class StateEstimator(Protocol[StateT, EstimateT]):
    """Estimate a task-defined quantity from the current plant state."""

    name: str

    def estimate(self, state: StateT) -> EstimateT:
        """Return the state estimate used by the controller."""


@runtime_checkable
class ActuationDirectionProvider(Protocol[StateT, DirectionT]):
    """Provide the direction along which a task applies controller output."""

    name: str

    def direction(self, state: StateT) -> DirectionT:
        """Return a task-compatible direction without consuming the estimate."""


@dataclass(frozen=True)
class Observation(Generic[EstimateT, DirectionT]):
    """The estimate and actuation direction exposed at one control step."""

    estimate: EstimateT
    direction: DirectionT


@dataclass(frozen=True)
class Observer(Generic[StateT, EstimateT, DirectionT]):
    """Compose state estimation and actuation geometry without coupling them."""

    name: str
    estimator: StateEstimator[StateT, EstimateT]
    direction_provider: ActuationDirectionProvider[StateT, DirectionT]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def observe(self, state: StateT) -> Observation[EstimateT, DirectionT]:
        # Resolve direction first so an estimator cannot change direction by
        # mutating a shared state object. Components must still treat task state
        # as read-only; the generic contract cannot copy every possible type.
        direction = self.direction_provider.direction(state)
        estimate = self.estimator.estimate(state)
        return Observation(estimate=estimate, direction=direction)


@dataclass(frozen=True)
class ControlRequest(Generic[StateT, EstimateT]):
    """Inputs available to a controller at one task-defined step."""

    state: StateT
    target: EstimateT
    estimate: EstimateT
    step: int


@runtime_checkable
class Controller(Protocol[StateT, EstimateT]):
    """Choose action magnitude without choosing the actuation direction."""

    name: str

    def action(self, request: ControlRequest[StateT, EstimateT]) -> float:
        """Return the scalar action applied along the observer direction."""


class TaskRunner(Protocol[StateT, EstimateT, DirectionT, RunOutputT]):
    """Callable implemented by a task that owns the plant and evaluation."""

    def __call__(
        self,
        *,
        observer: Observer[StateT, EstimateT, DirectionT],
        controller: Controller[StateT, EstimateT],
        config: Mapping[str, Any],
        outdir: Path,
    ) -> RunOutputT:
        """Run one observer-controller composition on the task."""


@dataclass(frozen=True)
class ObserverTask(Generic[StateT, EstimateT, DirectionT, RunOutputT]):
    """A named task plus its runner, composed without global registration."""

    name: str
    summary: str
    runner: TaskRunner[StateT, EstimateT, DirectionT, RunOutputT]


def run_observer_task(
    task: ObserverTask[StateT, EstimateT, DirectionT, RunOutputT],
    observer: Observer[StateT, EstimateT, DirectionT],
    controller: Controller[StateT, EstimateT],
    *,
    config: Mapping[str, Any],
    outdir: str | Path,
) -> RunOutputT:
    """Run a composed task using an externally supplied observer."""

    output_path = Path(outdir)
    output_path.mkdir(parents=True, exist_ok=True)
    return task.runner(
        observer=observer,
        controller=controller,
        config=config,
        outdir=output_path,
    )


@dataclass
class AccessRegime:
    name: str
    gradients: bool = False
    hvp: bool = False
    forward_only: bool = True
    differentiable_readout: bool = False
    notes: str = ""


@dataclass
class ObserverResult:
    task: str
    observer: str
    access_regime: str
    observer_family: str
    metrics: Dict[str, float]
    known_failure_modes: List[str] = field(default_factory=list)
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObserverCard:
    observer: str
    task: str
    access_regime: str
    estimand: str
    measurement_design: str
    validation_target: str
    metrics: Dict[str, float]
    known_failure_modes: List[str]
    recommendation: str
    notes: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# ObserverCard: `{self.observer}` on `{self.task}`",
            "",
            f"**Access regime.** {self.access_regime}",
            f"**Estimand.** {self.estimand}",
            f"**Measurement design.** {self.measurement_design}",
            f"**Validation target.** {self.validation_target}",
            "",
            "## Metrics",
        ]
        for k, v in sorted(self.metrics.items()):
            if isinstance(v, float):
                lines.append(f"- `{k}`: {v:.6g}")
            else:
                lines.append(f"- `{k}`: {v}")
        lines += ["", "## Known failure modes"]
        if self.known_failure_modes:
            lines += [f"- {m}" for m in self.known_failure_modes]
        else:
            lines.append("- None observed under this benchmark configuration.")
        lines += ["", "## Recommendation", self.recommendation]
        if self.notes:
            lines += ["", "## Notes", self.notes]
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_markdown(), encoding="utf-8")


def _strict_json_value(value: Any) -> Any:
    """Replace non-finite real values with JSON null, recursively."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _strict_json_value(value.tolist())
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
    return value


def write_json(path: str | Path, obj: Any) -> None:
    """Write strict, interoperable JSON; non-finite metrics become ``null``."""

    payload = _strict_json_value(obj)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
