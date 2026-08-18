"""Minimal external safety-observer submission.

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""

from __future__ import annotations

from observerbench import evaluate_safety_predictor
from observerbench.tasks import load_safety_task
from observerbench.tasks.safety_interlock import InterlockRidgeObserver


task = load_safety_task("safety_interlock_analytic@paired-scope-v0")
observer = InterlockRidgeObserver(
    "example-direct-risk-observer",
    view="activation",
    target="direct_severity_risk",
)
result = evaluate_safety_predictor(
    task,
    observer,
    outdir="runs/example-direct-risk-observer",
)
print(result.metrics)
