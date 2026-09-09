# ObserverBench

**Test an internal estimate by the action it causes.**

[Website](https://kwisatzh.github.io/observerbench/) ·
[Browser walkthrough](https://kwisatzh.github.io/observerbench/try/) ·
[Leaderboards](https://kwisatzh.github.io/observerbench/leaderboards/) ·
[Run an observer](https://kwisatzh.github.io/observerbench/runners/) ·
[Submit predictions](https://kwisatzh.github.io/observerbench/submit/) ·
[Paper](https://kwisatzh.github.io/observerbench/downloads/observerbench.pdf) ·
[Archived release](https://doi.org/10.5281/zenodo.22136091)

An **observer** uses available measurements to estimate something we cannot
read directly—for example, whether a model's proposed action is unsafe or what
an edit inside the model will do. ObserverBench tests both the estimate and the
decision made from it. The task and decision rule stay fixed, so different
observers face the same test.

[Mechanistic Tomography](https://arxiv.org/abs/2608.19338) asks how to measure a
hidden internal quantity. ObserverBench asks whether that estimate helps us act.

<a id="sixty-second-demo"></a>

## Run two examples—no GPU required

You need Git, Python 3, and make. No model download, extra Python packages, or
API key is needed.

```bash
git clone https://github.com/kwisatzh/observerbench.git
cd observerbench
make demo
```

One command runs both examples and prints their scores:

- **Safety: which requests should we check?** In a made-up set of 16 requests,
  only four can be checked. The observer that ranks violations better leaves
  24 harm points unchecked; the other leaves 6. You get both its ranking score
  (AUROC) and the harm its choices miss. Change `score(row)` in
  [`demo/safety_tutorial.py`](demo/safety_tutorial.py) and rerun.
- **Qwen: which internal edit should we choose?** Using saved measurements from
  **Qwen2.5-7B base**, a small observer fits 40 interventions and predicts 128
  more. The scorer reports prediction error, how far the selected edits miss
  their targets, and a comparison with doing nothing. Change the observer in
  the [Qwen practice task](practice/qwen_copy_v2_b040/README.md) and compare.

These examples score locally and show their answers: no upload or maintainer
review is needed. Their practice scores do not enter the research rankings.

Prefer no setup? [Try the browser walkthrough](https://kwisatzh.github.io/observerbench/try/).
Choose an observer, change the checking budget, and watch which decisions improve.

## Why test both predictions and decisions?

ObserverBench reports:

- **Prediction quality:** How well did the observer estimate or rank the target?
- **Decision quality:** What happened when the decision rule used that estimate?

Neither number replaces the other. Results are compared within the same task
version and information access—not in one global ranking.

To make the comparison fair, every task declares five things:

1. **Target:** what the observer must estimate.
2. **Information boundary:** what the observer is allowed to read.
3. **Actions:** which interventions or policy choices are allowed.
4. **Decision rule and loss:** how an estimate becomes an action and how that
   action is scored.
5. **Deployment setting:** the prompt distribution, budget, prevalence, and
   other operating conditions.

## What is included

Tasks cover closed-loop control, choosing internal edits, and safety triage.
Three findings illustrate what they test:

- **A better effect prediction need not choose a better edit.** On the GPT-2-small
  and Qwen2.5-7B base tasks, predicting average effects more accurately does not
  automatically improve the action. Observers fitted to predict action loss
  choose better edits.
- **Classification and safety outcomes can rank observers differently.** On the
  recorded Gemma-2-9B-it panel, the detailed prompted monitor has higher AUROC
  than the Gemma Scope sparse-autoencoder (SAE) probe (`0.893` vs. `0.870`), but
  more mean missed violations (`10.80` vs. `9.07`). Resampling the source
  problems leaves the action ranking uncertain.
- **Where we read a model matters.** The best monitoring context changes across
  Qwen2.5-7B-Instruct, Gemma-2-9B-it, and Qwen3.5-9B. Test the observer on the
  model and operating setting where it will be used.

The [leaderboards](https://kwisatzh.github.io/observerbench/leaderboards/) show
the results and comparison conditions for each task.

## Open practice tasks

After the two examples, try another observer on one of these tasks. All return
local feedback using public answers.

| Task | What you can change | What it reports |
| --- | --- | --- |
| [Safety tutorial](https://kwisatzh.github.io/observerbench/try/) | One risk score per request | Ranking accuracy, missed harm, unnecessary checks, action loss |
| [Qwen2.5-7B base Copy-v2](practice/qwen_copy_v2_b040/README.md) | A predictor fitted on 40 saved interventions | Prediction error, selected-action loss, regret, gain over no-op |
| [GPT-2-small IOI decision replay](practice/ioi_decision_v1/README.md) | A mean-effect prediction CSV; includes attribution patching | Chosen masks, action loss, paired comparisons with four observers and no-op |
| [Evidence integrity and cross-agent origin](docs/evidence_integrity.md) | Which trusted observations to buy, then how to score pending operations | Missed severity, clean work retained, measurement cost, total loss |

The separate [cross-model reuse diagnostic](docs/monitor_transfer.md) tests
unchanged observer coefficients across architectures. It shows no transfer
advantage; only aggregate results are public, not a runnable practice pack.

## Try your own observer

Run your method locally. For effect and safety tasks, provide a prediction
table and an **ObserverCard**: a short record of what your method could read,
how you built it, its cost, and the task version. The control task also accepts
a Python implementation.

Start with the [runner guide](https://kwisatzh.github.io/observerbench/runners/)
or jump to the relevant contract:

| Your method | Start here |
| --- | --- |
| State estimator and actuation direction | [Control observer contract](docs/API_CONTRACT.md#run-an-outside-observer-on-bundled-ctl-1) |
| Finite-effect predictor or prediction table | [Effect-prediction contract](docs/EFFECT_PREDICTION_CONTRACT.md) |
| Safety risk monitor or score table | [Safety observer contract](docs/SAFETY_API.md) |
| APPS or AI-control monitor | [External monitor comparison](docs/SAFETY_API.md#external-ai-control-monitor-comparison) |

For an existing-method example, run `python examples/ioi_decision_submission.py`
after installing `numpy pandas matplotlib`. It scores attribution patching's
saved predictions through the action-selection rule, without loading GPT-2.
[See the IOI guide](practice/ioi_decision_v1/README.md).

## Submit predictions

Public submissions use CSV and JSON files, not uploaded code. Automatic checks
verify the file format, query IDs, finite values, source link, and declared
access.

This route currently supports only the blinded `paired-scope-v1` safety pack.
**Sealed scoring is not active:** a passing submission receives a file-check
result, not a score or rank. Use the open practice tasks for immediate scoring.

- [Start a submission](https://github.com/kwisatzh/observerbench/issues/new?template=observer-submission.yml)
- [Public submission workflow](docs/PUBLIC_SUBMISSION_WORKFLOW.md)
- [Submission file contract](submissions/README.md)
- [Blinded public task pack](task-packs/safety/safety-interlock-qwen2-5-7b-instruct/paired-scope-v1/)

<a id="quick-start"></a>

## Install the full workbench

The two examples above need no installation. For the full Python command-line
tools, run these from your downloaded repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

See what is available:

```bash
observerbench list-tasks
observerbench list-effect-tasks
observerbench list-safety-tasks
observerbench list-safety-results
```

Run a small CPU task and generate its ObserverCard:

```bash
observerbench run safety_interlock_analytic \
  --config configs/safety_interlock_analytic.yaml \
  --outdir runs/safety-interlock
observerbench make-card \
  --results runs/safety-interlock \
  --outdir runs/safety-interlock/cards
```

## Reproduce the paper

The [reproduction map](paper/figure_map.md) links claims to saved results and
instructions for rerunning the experiments. The [notebooks](notebooks/) cover
named studies; they are not generic hosted runners.

To rebuild the paper from saved results, with LaTeX installed:

```bash
make -C paper/observerbench_v15_source
```

This rebuild does not download model weights or rerun model inference.

## Repository map

| Path | Contents |
| --- | --- |
| [`src/observerbench/`](src/observerbench/) | Python package and task contracts |
| [`notebooks/`](notebooks/) | Colab reproductions for named studies |
| [`paper/`](paper/) | Manuscript source and claim-to-artifact map |
| [`docs/`](docs/) | Detailed contracts, registrations, and protocols |

## Current boundaries

These are fixed tasks, not a hosted inference service or a general
bring-your-own-model system. IOI uses documented circuit groups; the Qwen copy
task tests a selected intervention surface, not the complete circuit. The
safety tasks do not establish robustness against adaptive attackers.

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

Software release `0.1.0` uses the [Apache License 2.0](LICENSE).
