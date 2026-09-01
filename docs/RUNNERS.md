# Run an observer locally

ObserverBench does not host your model or execute uploaded research code. You
run the observer in your own environment and pass its estimates through a
small, declared boundary: a Python interface or a prediction table.

This separation lets activation probes, attribution methods, direct logit
scores, SAE features, text baselines, and other methods face the same task,
controller, and loss without giving the evaluator control of their runtime.

For a model-free first run, use the
[open Qwen2.5-7B practice pack](../practice/qwen_copy_v2_b040/). It fits an
observer on 40 cached interventions and reports both effect-prediction error
and selected-action loss.

## Start with a small CPU run

Install the package from a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the example safety observer:

```bash
python examples/safety_interlock_submission.py
```

The example fits an observer on the task's permitted rows, evaluates it under
the fixed safety policy, and writes a local result bundle under `runs/`.

For a short closed-loop control run:

```bash
observerbench run trained_ctl2 \
  --config configs/trained_ctl2.yaml \
  --outdir runs/smoke-ctl2 \
  --quick
```

## Pick the boundary that matches your method

### I already have scores

You do not need a runner. Produce one score per frozen query and an
ObserverCard that describes how the score was obtained. Use the
[submission page](https://kwisatzh.github.io/observerbench/submit/) for the
current public task.

### I have a finite-effect predictor

Use the table-backed effect contract. The task supplies permitted calibration
measurements and held-out interventions; your method supplies predictions.

```bash
observerbench list-effect-tasks
observerbench evaluate-effect-csv <name@version> \
  --artifacts-root <frozen-task-root> \
  --predictions effect_predictions.csv \
  --observer-card observer_card.json \
  --outdir runs/my-effect-observer
```

The full schema and Python `fit`/`predict` boundary are in
[EFFECT_PREDICTION_CONTRACT.md](EFFECT_PREDICTION_CONTRACT.md).

### I have a safety monitor

A safety observer returns a risk score. ObserverBench converts that score into
allow, block, escalate, or audit actions under the same budgets used for every
observer.

```bash
observerbench evaluate-safety-csv \
  safety_interlock_analytic@paired-scope-v0 \
  --predictions predictions.csv \
  --observer-card observer_card.json \
  --outdir runs/my-safety-observer

observerbench compare-safety \
  --result-dir runs/my-safety-observer \
  --format markdown
```

See [SAFETY_API.md](SAFETY_API.md) for the Python interface, CSV schema,
researcher-authored task packs, and APPS monitor comparisons.

### I have a state estimator for control

The control contract keeps the estimator, actuation direction, controller, and
task separate. An external implementation supplies a `name` and an
`estimate(state)` method. It may also supply its own direction provider.

The smallest complete composition example is in
[API_CONTRACT.md](API_CONTRACT.md#run-an-outside-observer-on-bundled-ctl-1).

## What a local run produces

Depending on the task, a run writes:

- predictions or state estimates tied to fixed query IDs;
- an ObserverCard describing access, fitting, cost, and known failures;
- task-specific prediction and action metrics;
- versions and hashes needed to repeat the comparison.

These outputs are local evidence. They do not become a public score
automatically.

## Colab notebooks

The supplied notebooks reproduce named studies. They are useful starting
points for model access and measurement hooks, but they are not one generic
hosted runner.

| Study | Notebook |
| --- | --- |
| Qwen2.5 induction-copy | [`qwen_induction_copy_v2_colab.ipynb`](../notebooks/qwen_induction_copy_v2_colab.ipynb) |
| Qwen2.5 attribution patching | [`qwen_attribution_patching_colab.ipynb`](../notebooks/qwen_attribution_patching_colab.ipynb) |
| Qwen2.5 APPS SAE monitor | [`qwen_controlarena_apps_sae_observer_colab.ipynb`](../notebooks/qwen_controlarena_apps_sae_observer_colab.ipynb) |
| Gemma Scope APPS monitor | [`controlarena_apps_gemma_scope_colab.ipynb`](../notebooks/controlarena_apps_gemma_scope_colab.ipynb) |
| Qwen3.5 Qwen-Scope APPS monitor | [`controlarena_apps_qwen35_scope_colab.ipynb`](../notebooks/controlarena_apps_qwen35_scope_colab.ipynb) |
| Qwen2.5 paired-scope safety | [`qwen_safety_paired_scope_colab.ipynb`](../notebooks/qwen_safety_paired_scope_colab.ipynb) |

## Public submission boundary

Automatic public preflight currently accepts only prediction tables for the
target-free Qwen2.5 `paired-scope-v1` task pack. It checks plain CSV and JSON
data and never executes the contributor's code. Sealed scoring is not active
today, so a preflight pass is not a score or leaderboard rank.

Read the [public workflow](PUBLIC_SUBMISSION_WORKFLOW.md) before submitting.
The exact file requirements are in [`submissions/README.md`](../submissions/README.md).

## Current limit

There is no hosted bring-your-own-model service, general dynamic plugin system,
or sandbox for arbitrary submitted code. Model-specific notebooks reproduce
the checked studies; the stable outside-user boundary is the prediction table
or the documented local Python contract.
