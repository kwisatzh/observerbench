# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pytest

from observerbench import (
    API_CONTRACT_VERSION,
    ActuationDirectionProvider,
    Controller,
    ControlRequest,
    Observer,
    ObserverTask,
    StateEstimator,
    run_observer_task,
)
from observerbench.core import write_json
from observerbench.tasks import (
    get_task_spec,
    make_observer_task,
    observer_task_names,
    task_names,
    task_specs,
)


@dataclass(frozen=True)
class ScalarState:
    value: float


class OffsetEstimator:
    name = "offset-estimator"

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def estimate(self, state: ScalarState) -> float:
        return state.value + self.offset


class PositiveDirection:
    name = "positive-direction"

    def direction(self, state: ScalarState) -> float:
        del state
        return 2.0


class NegativeDirection:
    name = "negative-direction"

    def direction(self, state: ScalarState) -> float:
        del state
        return -2.0


class PairEstimator:
    name = "pair-estimator"

    def estimate(self, state: ScalarState) -> tuple[float, float]:
        return (state.value, state.value**2)


class ProportionalController:
    name = "proportional-controller"

    def __init__(self, gain: float) -> None:
        self.gain = gain

    def action(self, request: ControlRequest[ScalarState, float]) -> float:
        return self.gain * (request.target - request.estimate)


class MutableState:
    def __init__(self, value: float) -> None:
        self.value = value


class MutatingEstimator:
    name = "mutating-estimator"

    def estimate(self, state: MutableState) -> float:
        state.value = -100.0
        return state.value


class StateDirection:
    name = "state-direction"

    def direction(self, state: MutableState) -> float:
        return state.value


def scalar_runner(
    *,
    observer: Observer[ScalarState, float, float],
    controller: Controller[ScalarState, float],
    config: Mapping[str, Any],
    outdir: Path,
) -> dict[str, float | str]:
    state = ScalarState(float(config["initial_state"]))
    observation = observer.observe(state)
    action = controller.action(
        ControlRequest(
            state=state,
            target=float(config["target"]),
            estimate=observation.estimate,
            step=0,
        )
    )
    return {
        "observer": observer.name,
        "estimate": observation.estimate,
        "direction": observation.direction,
        "action": action,
        "next_state": state.value + action * observation.direction,
        "outdir": str(outdir),
    }


def test_external_components_compose_without_registration(tmp_path: Path) -> None:
    estimator = OffsetEstimator(offset=0.25)
    direction_provider = PositiveDirection()
    controller = ProportionalController(gain=0.5)
    observer = Observer(
        name="external-observer",
        estimator=estimator,
        direction_provider=direction_provider,
    )
    task = ObserverTask(
        name="scalar-control",
        summary="Small contract test.",
        runner=scalar_runner,
    )

    result = run_observer_task(
        task,
        observer,
        controller,
        config={"initial_state": 1.0, "target": 2.0},
        outdir=tmp_path / "external-run",
    )

    assert isinstance(estimator, StateEstimator)
    assert isinstance(direction_provider, ActuationDirectionProvider)
    assert isinstance(controller, Controller)
    assert result == {
        "observer": "external-observer",
        "estimate": 1.25,
        "direction": 2.0,
        "action": 0.375,
        "next_state": 1.75,
        "outdir": str(tmp_path / "external-run"),
    }
    assert (tmp_path / "external-run").is_dir()


def test_estimator_and_direction_provider_can_be_swapped_independently() -> None:
    positive = Observer(
        name="positive-estimate",
        estimator=OffsetEstimator(offset=0.0),
        direction_provider=PositiveDirection(),
    )
    negative = Observer(
        name="negative-estimate",
        estimator=OffsetEstimator(offset=-2.0),
        direction_provider=positive.direction_provider,
    )
    negative_direction = Observer(
        name="negative-direction",
        estimator=positive.estimator,
        direction_provider=NegativeDirection(),
    )

    assert positive.observe(ScalarState(1.0)).direction == 2.0
    assert negative.observe(ScalarState(1.0)).estimate == -1.0
    assert negative.observe(ScalarState(1.0)).direction == 2.0
    assert negative_direction.observe(ScalarState(1.0)).estimate == 1.0
    assert negative_direction.observe(ScalarState(1.0)).direction == -2.0


def test_estimate_type_is_task_defined() -> None:
    observer = Observer(
        name="structured-estimate",
        estimator=PairEstimator(),
        direction_provider=PositiveDirection(),
    )

    assert observer.observe(ScalarState(3.0)).estimate == (3.0, 9.0)


def test_direction_is_resolved_before_estimator_side_effect() -> None:
    observer = Observer(
        name="side-effect-check",
        estimator=MutatingEstimator(),
        direction_provider=StateDirection(),
    )

    observation = observer.observe(MutableState(2.0))

    assert observation.direction == 2.0
    assert observation.estimate == -100.0


def test_builtin_registry_has_stable_lookup_helpers() -> None:
    names = task_names()

    assert names == tuple(sorted(names))
    assert tuple(spec.name for spec in task_specs()) == names
    ctl1 = get_task_spec("ctl1_analytic")
    assert ctl1.name == "ctl1_analytic"
    assert ctl1.supports_external_observer
    assert observer_task_names() == ("ctl1_analytic",)
    assert make_observer_task("ctl1_analytic").name == "ctl1_analytic"
    with pytest.raises(KeyError, match="Unknown ObserverBench task"):
        get_task_spec("not-a-task")
    with pytest.raises(KeyError, match="no external-observer adapter"):
        make_observer_task("trained_ctl2")


def test_contract_has_machine_readable_version() -> None:
    assert API_CONTRACT_VERSION == "observerbench.api.v0"


def test_write_json_uses_null_for_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"

    write_json(path, {"finite": 1.0, "missing": float("nan")})

    assert "NaN" not in path.read_text()
    assert json.loads(path.read_text()) == {"finite": 1.0, "missing": None}


def test_documented_python_contract_examples_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = (Path(__file__).parents[1] / "docs" / "API_CONTRACT.md").read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", docs, flags=re.DOTALL)
    assert len(blocks) == 3
    monkeypatch.chdir(tmp_path)

    for index, block in enumerate(blocks):
        exec(compile(block, f"docs/API_CONTRACT.md:block-{index}", "exec"), {})

    assert (tmp_path / "runs" / "my-observer").is_dir()
    assert (tmp_path / "runs" / "outside-oracle-ctl1" / "run_metadata.json").exists()
