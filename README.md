# ObserverBench

**Test an internal estimate by the action it causes.**

[Website](https://kwisatzh.github.io/observerbench/) ·
[Try it](https://kwisatzh.github.io/observerbench/try/) ·
[Leaderboards](https://kwisatzh.github.io/observerbench/leaderboards/) ·
[Run an observer](https://kwisatzh.github.io/observerbench/runners/) ·
[Submit predictions](https://kwisatzh.github.io/observerbench/submit/) ·
[Paper](https://kwisatzh.github.io/observerbench/downloads/observerbench.pdf) ·
[Archived release](https://doi.org/10.5281/zenodo.22136091)

ObserverBench evaluates an internal estimator—an **observer**—through the
intervention, control loop, or safety decision it guides. It keeps the task,
allowed actions, controller, and loss fixed, then asks what changes when the
observer changes.

This is the evaluation companion to
[Mechanistic Tomography](https://arxiv.org/abs/2608.19338). Mechanistic
tomography asks how to measure a hidden internal quantity. ObserverBench asks
whether the resulting estimate is good enough for the action we want to take.

The current software release is `0.1.0`.

## Sixty-second demo

No package, model download, GPU, or API key is required:

```bash
git clone https://github.com/kwisatzh/observerbench.git
cd observerbench
make demo
```

The command runs two small, open examples:

- A safety monitor with better classification AUROC spends its limited audit
  budget on likely but low-impact violations and produces worse action loss.
  Edit the short `score(row)` function in
  [`demo/safety_tutorial.py`](demo/safety_tutorial.py), then run it again.
- A nine-parameter observer learns from 40 cached Qwen2.5-7B interventions,
  predicts 128 new effects, and is scored both on prediction error and on the
  interventions it selects. Start with the
  [Qwen practice task](practice/qwen_copy_v2_b040/README.md).

Prefer a browser? Use the
[open safety tutorial](https://kwisatzh.github.io/observerbench/try/). It runs
locally in the page and accepts a CSV containing one score per request. Its
targets are public for immediate feedback, so tutorial results are not eligible
for the leaderboard.

## Why ObserverBench exists

A good prediction does not always produce a good decision.

Suppose a safety monitor ranks risky requests accurately. A controller can
still spend its limited audit budget on minor violations and let more costly
ones pass. In the same way, an observer can predict an intervention's average
effect well but choose an edit that behaves poorly on individual prompts.

ObserverBench reports both sides:

- **Statistical quality:** Did the observer estimate or rank the target well?
- **Decision quality:** What happened after a fixed controller used that
  estimate?

Neither number replaces the other.

## What an ObserverBench task fixes

Every task declares five things before observers are compared:

1. **Target:** what the observer must estimate.
2. **Information boundary:** what the observer is allowed to read.
3. **Actions:** which interventions or policy choices are allowed.
4. **Decision rule and loss:** how an estimate becomes an action and how that
   action is scored.
5. **Deployment setting:** the prompt distribution, budget, prevalence, and
   other operating conditions.

Results are ranked only inside the same task version and comparison track.
ObserverBench does not issue a global rank across unlike tasks or access
regimes.

## What is included

| Mode | Main question | Checked examples |
| --- | --- | --- |
| Closed-loop control | Does observer error reach the feedback loop or move collateral state? | Analytic and learned control fixtures |
| Effect prediction | Does the estimated effect map predict held-out interventions and choose a good action? | GPT-2 IOI and Qwen2.5-7B induction-copy panels |
| Safety triage | Does a monitor use a fixed audit or intervention budget well? | Qwen2.5-7B-Instruct, Gemma-2-9B-it, and Qwen3.5-9B APPS panels |

Three findings summarize why the distinction matters:

- On GPT-2 and Qwen2.5, a model can predict mean intervention effects better
  without choosing a better edit. Observers trained for the downstream loss
  choose better actions.
- On Gemma, a prompted monitor has higher AUROC than an official Gemma Scope
  SAE probe (`0.893` versus `0.870`) but causes more mean realized violations
  (`10.80` versus `9.07`). Classification is not enough.
- The lowest-loss monitoring context changes across Qwen2.5, Gemma, and
  Qwen3.5. A monitor must be tested on the model and operating setting where it
  will be used.

See the [live leaderboards](https://kwisatzh.github.io/observerbench/leaderboards/)
for the checked rows, access boundaries, and task-specific metrics.

## Open practice tasks

Open practice tasks reveal their targets so that a new observer gets immediate
feedback. They are useful for learning the contract and testing code, but their
scores do not enter the sealed leaderboards.

| Task | What you can change | What it reports |
| --- | --- | --- |
| [Safety tutorial](site/try/) | One risk score per request | AUROC, missed harm, clean audits, action loss |
| [Qwen2.5-7B Copy-v2, budget 40](practice/qwen_copy_v2_b040/README.md) | Any predictor trained on 40 cached interventions | MAE, RMSE, selected-action loss, regret, gain over no-op |

## Choose a path

### Explore the evidence

Open the [leaderboards](https://kwisatzh.github.io/observerbench/leaderboards/)
and choose one task panel. The table ranks observers only within that panel.

### Run an observer locally

Keep your model, weights, and method in your own environment. ObserverBench
reads only the resulting predictions and a description of how you produced
them.

Start with the [runner guide](https://kwisatzh.github.io/observerbench/runners/)
or jump to the relevant contract:

| Your method | Start here |
| --- | --- |
| State estimator and actuation direction | [Control observer contract](docs/API_CONTRACT.md#run-an-outside-observer-on-bundled-ctl-1) |
| Finite-effect predictor or prediction table | [Effect-prediction contract](docs/EFFECT_PREDICTION_CONTRACT.md) |
| Safety risk monitor or score table | [Safety observer contract](docs/SAFETY_API.md) |
| APPS or AI-control monitor | [External monitor comparison](docs/SAFETY_API.md#external-ai-control-monitor-comparison) |

The notebooks under [`notebooks/`](notebooks/) reproduce specific checked
studies. They are not generic hosted runners.

### Submit predictions

The public path accepts plain CSV and JSON files; it never executes contributor
code. Automatic preflight checks the schema, query IDs, finite values,
source link, and declared access.

In the current public release, automatic submission is enabled only for the blinded
`paired-scope-v1` safety pack. Sealed scoring is not yet active, so a passing
submission receives a preflight result—not a benchmark score or rank.

- [Start a submission](https://github.com/kwisatzh/observerbench/issues/new?template=observer-submission.yml)
- [Public submission workflow](docs/PUBLIC_SUBMISSION_WORKFLOW.md)
- [Submission file contract](submissions/README.md)
- [Blinded public task pack](task-packs/safety/safety-interlock-qwen2-5-7b-instruct/paired-scope-v1/)

### Reproduce the paper

Build the paper from the checked local artifacts:

```bash
make -C paper/observerbench_v15_source
```

This path does not download GPT-2 or Qwen and does not rerun model inference.
The [reproduction map](paper/figure_map.md) links each claim to its checked
artifact and, where available, the full scientific rerun.

## Quick start

```bash
git clone https://github.com/kwisatzh/observerbench.git
cd observerbench
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

See what is available:

```bash
observerbench list-tasks
observerbench list-effect-tasks
observerbench list-safety-tasks
observerbench list-safety-results
```

Run a small CPU task:

```bash
observerbench run safety_interlock_analytic \
  --config configs/safety_interlock_analytic.yaml \
  --outdir runs/safety-interlock
```

Generate an ObserverCard:

```bash
observerbench make-card \
  --results runs/safety-interlock \
  --outdir runs/safety-interlock/cards
```

An ObserverCard records what the method could read, what it estimated, how it
was fitted, its main metrics, its known failures, and the exact task version.

## Repository map

| Path | Contents |
| --- | --- |
| [`src/observerbench/`](src/observerbench/) | Python package and task contracts |
| [`leaderboards/`](leaderboards/) | Checked task-specific result panels |
| [`practice/`](practice/) | Open tasks with public targets and immediate local scoring |
| [`task-packs/`](task-packs/) | Public task definitions and target-free queries |
| [`notebooks/`](notebooks/) | Colab reproductions for named studies |
| [`paper/`](paper/) | Manuscript source and claim-to-artifact map |
| [`docs/`](docs/) | Detailed contracts, registrations, and protocols |
| [`submissions/`](submissions/) | Prediction and ObserverCard templates |

## Current boundaries

ObserverBench currently provides fixed, versioned tasks. It does not provide:

- hosted model inference or execution of uploaded code;
- a general bring-your-own-model plugin system;
- a universal rank across different tasks and information boundaries;
- a complete safety evaluation against adaptive or monitor-aware attackers;
- open-ended circuit discovery.

The IOI panels use documented circuit groups as a controlled diagnostic. The
Qwen induction-copy panel tests one confirmed intervention surface; it does not
claim to describe the model's complete circuit.

## Credit

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex.

## Citation

```bibtex
@software{erramilli2026observerbench,
  author  = {Vijay Erramilli},
  title   = {ObserverBench: Testing Mechanistic Estimates for Intervention and Control},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.22136091},
  url     = {https://doi.org/10.5281/zenodo.22136091}
}
```

## License

ObserverBench is released under the [Apache License 2.0](LICENSE).
