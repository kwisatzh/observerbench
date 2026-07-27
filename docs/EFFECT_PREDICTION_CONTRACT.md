# External Finite-Effect Prediction Contract

ObserverBench exposes a second composition point for methods that predict the
finite response to an intervention. It is separate from the scalar control
contract: an effect predictor does not choose a controller, an actuation
direction, or a plant transition.

The contract version is `observerbench.effect_prediction.v0`.

| Component | Owns | Does not own |
| --- | --- | --- |
| Benchmark task | measurement budget, split, held-out interventions, targets, and evaluation | observer fitting procedure |
| Effect predictor | fitting on supplied measurements and predicting held-out finite effects | query selection, targets, or metrics |
| Task card | estimand, intervention family, access regime, splits, and scope | method claims |
| Observer card | basis, fit procedure, implementation, and known failure modes | benchmark definition |

This boundary supports the comparison needed by intervention-facing tasks:
observers receive the same measurements and predict the same held-out finite
responses. The task can then use those predictions in a fixed downstream
selection or control rule. The generic v0 evaluator reports prediction error;
task-specific evaluations such as intervention-selection regret can compose
that fixed rule on top of the same prediction rows.

## Python composition

External methods use structural typing. They implement `name`, `fit`, and
`predict`; they do not subclass an ObserverBench class or register a plugin.

```python
from observerbench import (
    EffectObserverCard,
    EffectTaskCard,
    FiniteEffectMeasurement,
    FiniteEffectPredictionTask,
    FiniteEffectQuery,
    FiniteEffectTarget,
    run_effect_prediction_task,
)


# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
class LinearEffectPredictor:
    name = "outside-linear"

    def fit(self, measurements):
        numerator = sum(row.features * row.observed_effect for row in measurements)
        denominator = sum(row.features**2 for row in measurements)
        self.slope = numerator / denominator

    def predict(self, queries):
        return [self.slope * row.features for row in queries]


task_card = EffectTaskCard(
    task_name="scalar-effects",
    task_version="1.0",
    summary="Predict held-out effects of scalar interventions.",
    model_or_substrate="documented scalar fixture",
    access_regime="forward measurements",
    estimand="finite change in the scalar readout",
    intervention_family="scalar offsets",
    measurement_design="two fixed calibration interventions",
    validation_target="held-out finite-effect error",
    train_split="calibration-v1",
    evaluation_split="heldout-v1",
    known_scope_limits=("Documentation fixture only.",),
)
task = FiniteEffectPredictionTask(
    name="scalar-effects",
    version="1.0",
    measurements=(
        FiniteEffectMeasurement("m-negative", -1.0, -2.0),
        FiniteEffectMeasurement("m-positive", 1.0, 2.0),
    ),
    queries=(
        FiniteEffectQuery("q-small", 0.5),
        FiniteEffectQuery("q-large", 2.0),
    ),
    targets=(
        FiniteEffectTarget("q-small", 1.0),
        FiniteEffectTarget("q-large", 4.0),
    ),
    card=task_card,
)
observer_card = EffectObserverCard(
    observer_name="outside-linear",
    observer_version="0.1.0",
    observer_family="linear finite-effect predictor",
    access_regime="forward measurements",
    measurement_basis="scalar intervention magnitude",
    fit_procedure="least squares through the origin",
    implementation="my_package.LinearEffectPredictor",
    known_failure_modes=("Cannot represent nonlinear response.",),
)
result = run_effect_prediction_task(
    task,
    LinearEffectPredictor(),
    observer_card,
    outdir="runs/scalar-effects/outside-linear",
)
assert result.metrics["mae"] == 0.0
```

The output bundle contains:

```text
effect_predictions.csv
effect_evaluation.json
task_card.json
observer_card.json
```

The measurement objects carry task-defined `features`. They may be scalars,
arrays, mask records, or another structure understood by both the task and
the predictor. The task owns their meaning. The generic contract only fixes
IDs, observed finite effects, and the fit/predict boundary.
Predictors must treat the supplied records and their feature objects as
read-only; v0 cannot copy every task-defined feature type.

## CSV interchange

A method can run outside the ObserverBench process and submit a CSV. The v0
header is exact:

```csv
schema_version,query_id,predicted_effect
observerbench.effect_predictions.v0,q-small,1.0
observerbench.effect_predictions.v0,q-large,4.0
```

Use `evaluate_effect_prediction_csv(task, path, observer_card, outdir=...)` to
evaluate it. The reader rejects unknown schemas, duplicate IDs, missing or
unexpected query IDs, and non-finite predictions. Row order does not affect
the metrics.

## Card boundary

Every evaluation writes separate task and observer cards. This prevents method
metadata from silently redefining the task:

- `EffectTaskCard` records the estimand, intervention family, measurement
  design, fixed splits, primary metrics, and known scope limits.
- `EffectObserverCard` records the method version, family, access regime,
  measurement basis, fit procedure, implementation, and known failure modes.

The result records both versions, the exact measurement budget, query count,
contract version, and generic held-out metrics. Task-specific provenance such
as data hashes and candidate-pool versions belongs in the task card's metadata;
method-specific source hashes belong in the observer card's metadata.

## Bundled finite-effect task registry

ObserverBench includes a closed, deterministic registry for table-backed
finite-effect tasks. Import the discovery and loader functions from
`observerbench.tasks`:

- `finite_effect_task_ids()` returns exact `name@version` identifiers;
- `finite_effect_task_specs()` returns their versions and measurement budgets;
- `finite_effect_task_versions(name)` and
  `finite_effect_measurement_budgets(name)` expose the available 20, 40, 80,
  task-specific budget variants;
- `get_finite_effect_task_spec(task_id)` performs exact lookup;
- `load_finite_effect_task(task_id, artifacts_root=...)` verifies the frozen
  manifests and hashes, then loads the task.

The checked GPT-2-small IOI task exposes 20, 40, 80, and 160 measurements. Each
feature is an `IOIMaskFeatures` record with the 13-bit mask,
selected head labels, group counts, and candidate-pool metadata. Measurement
labels average effects over the frozen train prompts. Each held-out target is
the corresponding candidate-mask effect averaged over the disjoint test
prompts, matching the paper's finite-effect prediction metric.

`IOIAdditiveRidgeBaseline` is the bundled first-order predictor; pair it with
`ioi_additive_baseline_card()`. The task loader reads CSV and JSON artifacts
only. It does not import TransformerLens, download a checkpoint, or run model
inference. The inference scripts remain a separate reproduction path.

The prospective Qwen2.5-7B induction-copy task exposes 16, 40, 64, and 128
measurements over eight frozen heads. Its `InductionMaskFeatures` record includes
the eight-bit mask plus each head's layer, query-head index, and shared KV-group
index. The 128 held-out masks are the exact complement of the 128-mask
calibration bank, not a sampled subset. `QwenInductionAdditiveRidgeBaseline` and
`qwen_induction_additive_baseline_card()` supply the first-order reference.
The 128-measurement result is the preregistered primary additive-versus-quadratic
comparison; smaller budgets form a secondary sample-efficiency curve.

The Qwen loader remains inference-free. The separate producer owns tokenizer
checks, head discovery, causal confirmation, mean ablation, and locked model
measurement. Registering the table schema does not assert that those prospective
scientific gates have passed; a loader succeeds only for a complete,
hash-verified artifact.

The CLI exposes the same closed registry and CSV boundary:

```bash
observerbench list-effect-tasks
observerbench evaluate-effect-csv <name@version> \
  --artifacts-root <frozen-task-root> \
  --predictions <effect_predictions.csv> \
  --observer-card <observer_card.json> \
  --outdir <evaluation-output>
```

The observer-card JSON uses the fields of `EffectObserverCard`. The command
writes the same four-file evaluation bundle as the Python composition path.

## v0 stability boundary

The public names and signatures documented here are the v0 contract. The
finite-effect registry is closed and deterministic; v0 does not provide dynamic discovery,
remote execution, automatic model loading, general feature serialization, or a
mutable task registry. A benchmark may also construct a
`FiniteEffectPredictionTask` locally and call the runner or CSV evaluator.
