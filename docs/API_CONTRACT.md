# ObserverBench Python Contract

ObserverBench v0 exposes a small Python contract for comparing an externally
supplied observer under a task-defined intervention. It is ordinary Python
composition, not a plugin system: there is no module discovery, entry-point
loading, global registration, or configuration-driven import.

Methods that predict held-out finite intervention responses use the separate
contract in `docs/EFFECT_PREDICTION_CONTRACT.md`. That contract leaves the
measurement budget, split, targets, and evaluation with the benchmark task.

Code that records contract provenance can import
`API_CONTRACT_VERSION`, whose v0 value is `observerbench.api.v0`.

The public contract separates four responsibilities:

| Component | Owns | Does not own |
| --- | --- | --- |
| `StateEstimator` | Task-defined estimate from the current state | Actuation geometry or control gain |
| `ActuationDirectionProvider` | Direction in the task's intervention space | Action magnitude |
| `Controller` | Scalar action from target and estimate | State transition or direction |
| `ObserverTask` runner | Plant, intervention, measurement design, and metrics | Observer implementation |

`Observer` composes an estimator and direction provider. This separation is
deliberate: changing the estimate and changing the actuation direction are
different experimental factors. The direction provider receives the current
state but not the estimate, and `Observer.observe` resolves direction before
estimation. Components must treat the supplied state as read-only. The v0
interface separates dataflow; it does not promise isolation from arbitrary
component side effects.

## Implement an observer

The following complete example uses only the public package imports:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from observerbench import (
    ControlRequest,
    Observer,
    ObserverTask,
    run_observer_task,
)


@dataclass(frozen=True)
class State:
    value: float


class MyEstimator:
    name = "my-estimator"

    def estimate(self, state: State) -> float:
        return state.value


class MyDirection:
    name = "positive-axis"

    def direction(self, state: State) -> float:
        return 1.0


class ProportionalController:
    name = "proportional"

    def action(self, request: ControlRequest[State, float]) -> float:
        return 0.2 * (request.target - request.estimate)


def run_scalar_task(
    *,
    observer,
    controller,
    config: Mapping[str, Any],
    outdir: Path,
):
    state = State(float(config["initial_state"]))
    observation = observer.observe(state)
    action = controller.action(
        ControlRequest(
            state=state,
            target=float(config["target"]),
            estimate=observation.estimate,
            step=0,
        )
    )
    return state.value + action * observation.direction


observer = Observer(
    name="my-observer",
    estimator=MyEstimator(),
    direction_provider=MyDirection(),
)
task = ObserverTask(
    name="scalar-task",
    summary="One-step scalar control.",
    runner=run_scalar_task,
)
result = run_observer_task(
    task,
    observer,
    ProportionalController(),
    config={"initial_state": 0.0, "target": 1.0},
    outdir="runs/my-observer",
)
```

External code uses structural typing. It does not subclass ObserverBench
classes. An estimator needs `name` and `estimate(state)`. A direction provider
needs `name` and `direction(state)`. A controller needs `name` and
`action(request)`.

An observer may be fitted before composition. Training data and fitting
procedures remain task-specific in v0; the runtime contract begins when the
task supplies the current state.

The estimate type is task-defined. The paper's control tasks use a scalar
`float`, but the contract also permits vector or structured estimates when a
task and controller agree on that type. Controller output remains scalar
because v0 applies an action magnitude along the separately supplied
direction.

## Run an outside observer on bundled Ctl-1

The analytic Ctl-1 fixture is the first bundled task with an external
composition point. Its state is the three-vector `[x1, x2, x1*x2]`; its
estimate must be scalar; and its direction must be a fixed finite three-vector.

This example supplies both estimator and direction from outside ObserverBench:

```python
import numpy as np

from observerbench import Observer
from observerbench.tasks import evaluate_ctl1_analytic_observer


class OutsideEstimator:
    name = "outside-estimator"

    def __init__(self, readout):
        self.readout = np.asarray(readout, dtype=float)

    def estimate(self, state):
        return float(self.readout @ state)


class OutsideDirection:
    name = "outside-direction"

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)

    def direction(self, state):
        return self.vector.copy()


config = {
    "task": "ctl1_analytic",
    "n_train": 512,
    "n_test": 512,
    "seed": 0,
    "beta1": 0.35,
    "beta2": 0.25,
    "gamma": 1.15,
}
readout = np.array([config["beta1"], config["beta2"], config["gamma"]])
observer = Observer(
    name="outside-oracle",
    estimator=OutsideEstimator(readout),
    direction_provider=OutsideDirection(readout),
    provenance={"implementation": "my_package.outside_oracle", "version": "0.1.0"},
)
result = evaluate_ctl1_analytic_observer(
    observer,
    config=config,
    outdir="runs/outside-oracle-ctl1",
)
print(result.metrics)
```

The benchmark-facing wrapper reuses the bundled data generator, target and
nuisance readouts, direction normalization, metrics, and proportional
controller. It writes strict JSON to `observer_result.json` and
`run_metadata.json`, including contract/result schema versions, package and Git
revision when available, component classes and parameters, and the caller's
observer provenance.

The lower-level `make_ctl1_analytic_task()` composition point also accepts a
custom controller through `run_observer_task`. Such a run is marked
`custom_controller` and is not controller-comparable to the bundled benchmark
arm.

The adapter deliberately leaves training data and fitting policy outside the
v0 runtime contract. The external observer must be fitted before composition.

## Task registry

The built-in paper tasks have deterministic discovery and lookup:

```python
from observerbench.tasks import (
    get_task_spec,
    make_observer_task,
    observer_task_names,
    task_names,
    task_specs,
)

print(task_names())
print(observer_task_names())  # currently ("ctl1_analytic",)
spec = get_task_spec("ctl1_analytic")
task = make_observer_task("ctl1_analytic")
```

The CLI uses the same registry:

```bash
observerbench list-tasks
observerbench run ctl1_analytic \
  --config configs/ctl1_analytic.yaml \
  --outdir runs/ctl1
```

`TaskSpec.supports_external_observer` distinguishes CLI-only paper tasks from
tested observer adapters. The built-in registry is closed in v0. External tasks
should create an `ObserverTask` locally and pass it to `run_observer_task`; they
should not mutate `observerbench.tasks.TASKS`. This keeps paper reproduction
deterministic without imposing a plugin framework.

## Runner obligations

A task runner must:

- interpret the state and direction types;
- hold the plant, controller settings, initial conditions, and intervention
  protocol fixed for a comparison;
- call `observer.observe(state)` at each measurement step;
- call `controller.action(ControlRequest(...))` for action magnitude;
- apply the returned scalar action along the returned direction;
- compute and return task-specific metrics;
- record the observer, estimator, direction provider, controller, task
  configuration, result schema, package/source version, and observer
  implementation provenance with persisted results.

For paper-facing evaluations, return `ObserverResult` objects or write the
equivalent provenance beside the task metrics. `run_observer_task` creates the
output directory but does not prescribe a result format.

## v0 stability boundary

The names and call signatures documented here are the v0 external contract:

- `API_CONTRACT_VERSION`
- `StateEstimator`
- `ActuationDirectionProvider`
- `Observation`
- `Observer`
- `ControlRequest`
- `Controller`
- `ObserverTask`
- `run_observer_task`
- `task_names`, `task_specs`, `get_task_spec`, `observer_task_names`, and
  `make_observer_task`
- `evaluate_ctl1_analytic_observer`

The scientific contents and configuration fields of individual paper tasks
are versioned separately. v0 does not promise remote execution, dynamic task
loading, automatic model downloads, or compatibility between arbitrary
direction types and arbitrary task runners.
