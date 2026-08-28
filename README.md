# ObserverBench

**Public site:** <https://kwisatzh.github.io/observerbench/>

**Archived release:** <https://doi.org/10.5281/zenodo.22136091>

ObserverBench is a research artifact for evaluating observers in control-oriented interpretability.

It is the evaluation companion to [Mechanistic Tomography](https://arxiv.org/abs/2608.19338):
that framework designs measurements for recovering internal quantities and
intervention effects, while ObserverBench tests whether a proposed estimate is
adequate for a declared action.

ObserverBench asks when an observer is adequate for an intervention. A more
complex observer is not automatically better: a simpler or misspecified
observer may still serve the target task while changing collateral state,
prediction error, or feedback behavior.

This repository is the v0 reproduction/workbench for the ObserverBench paper. It is not a finished universal benchmark or a plugin platform.

## What is ObserverBench?

An ObserverBench task fixes:

- a substrate or model state,
- a target state to estimate, control, or predict,
- a set of candidate observers,
- an intervention operator, admissible direction space, and normalization rule,
- a controller for benchmark-comparable runs,
- a validation target that checks whether the observer remains useful under intervention.

The estimator and actuation direction are separate experimental factors. A
fixed-direction comparison changes only the estimator; a factorial comparison
varies both factors and reports each effect separately.

The v0 repo includes the paper tasks:

- Ctl-1 analytic collateral geometry.
- Trained-transformer Ctl-1 one-shot observer geometry.
- Trained-transformer Ctl-2 closed-loop observer-control.
- A non-executing paired-scope safety interlock with exact evaluator-owned policy
  labels and fixed allow/block/escalate budgets, plus a sealed
  Qwen2.5-7B-Instruct run using the same protocol. The best activation observer
  lowers aggregate protocol loss by 32.1% relative to the strongest action-only
  observer; the doubly-held-out stratum remains a documented failure case.
- An external ControlArena APPS monitor comparison on 744 locked
  honest/verified-backdoor pairs. A Qwen residual probe read from a neutral
  problem-and-code context reaches AUROC 0.979, compared with 0.953 for a
  verdict-adjacent probe, 0.868 for supervised text, and 0.830 for a supervised
  observer over the prompted score distribution. It matches the exact-label
  oracle's audit loss on the balanced panel; a checked prevalence sweep shows
  that the tie with the oracle disappears when attacks are rare. The portable
  score table ships with the repo.
- IOI known-circuit diagnostics on GPT-2-small outputs:
  Stage 1 whole-group self-repair, Stage 2b anchored broad-random head subsets,
  Stage 2c primary-stratified subsets, and capacity-matched pair analyses.
- A prospectively frozen Qwen2.5-7B induction-copy task with exact next-token
  labels and an exhaustive eight-head intervention cube. On 128 held-out masks,
  pairwise effect prediction lowers MAE from 0.121 to 0.040. Direct-loss
  observers lower aggregate action loss by 8.9% against same-feature
  natural-mean observers and by 49.6% against exact no-op. The claim is bounded
  to eligible prompts, the confirmed panel, and the declared mask pools.

The repo includes commands to regenerate the paper figures and ObserverCards.
It also provides a minimal CLI for the current tasks and a small Python
contract for external observers.

The public site builds from the checked leaderboard JSON in this repository.
Routine public jobs use a structured GitHub issue: the submitter links an
immutable prediction table and supplies an ObserverCard, automatic preflight
checks the files as data, and a private evaluator can score them against sealed
targets. Maintainer review is reserved for new task contracts, executable-code
requests, and flagged submissions. See `docs/PUBLIC_SUBMISSION_WORKFLOW.md` and
`submissions/README.md`.

The first automatic task is the blinded Qwen authorization pack at
`task-packs/safety/safety-interlock-qwen2-5-7b-instruct/paired-scope-v1/`.
It provides 1,152 labeled fit/calibration measurements and 768 target-free
queries. The matching targets remain evaluator-only; the evaluator returns
aggregate prediction and action metrics. The APPS panel remains visible for
comparison, but its public examples are matchable to the labelled upstream
dataset and are therefore described as an open adoption panel rather than a
sealed test.

The external contract has four checked extension paths. `ctl1_analytic` accepts
an outside state estimator and direction provider. The inference-free
finite-effect registry accepts an outside predictor through `fit`/`predict` or
a result table. The safety path accepts either an outside risk-score table for a
built-in task or a researcher-authored public task pack with separately sealed
evaluator targets. An AI-control adapter compares outside monitor scores on a
common verified-backdoor panel under a fixed audit budget. These paths do not
load GPT-2, Qwen, or an interpretability
runtime. The trained control
tasks remain reproduction tasks. The v0 contract does not cover arbitrary
model attachment, bring-your-own-model workflows, or dynamic plugins.

For safety monitors, the artifact uses four checks: inspect the richest output
distribution, compare it with a surface baseline under matched supervision,
remove the monitoring request, and evaluate frozen scores at the intended
prevalence, budget, strata, and loss. Together, these checks show where useful
information is available. They do not by themselves establish deception or
hidden knowledge.

## Why Observers Matter For Control

Control-oriented interpretability uses internal measurements as inputs to action: activation steering, representation engineering, patching workflows, constrained decoding, or feedback controllers. In all of these cases, the controller does not consume an explanation in the abstract. It consumes an observer: a procedure that maps measurements to a state estimate.

That observer can fail in ways that ordinary prediction metrics do not show.
It can move the target while dragging collateral state. It can look adequate
in one-shot control and fail in a feedback loop. An interaction can also look
unimportant when the intervention design gives it little leverage or when
candidate models receive unequal capacity.

ObserverBench v0 isolates those cases in small, reproducible settings. The IOI component uses GPT-2-small as a known-circuit diagnostic, not as a claim that this repo is a general LLM safety evaluation.

## Quickstart

The 5-minute path is the CPU-safe path: install the package, run the default tests, list tasks, and run quick smoke jobs. It does not download GPT-2 and does not run training in CI.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

List the available paper tasks:

```bash
observerbench list-tasks
```

List exact inference-free finite-effect task versions:

```bash
observerbench list-effect-tasks
```

List exact inference-free safety task versions:

```bash
observerbench list-safety-tasks
```

List the checked safety observers and the external APPS monitor baselines:

```bash
observerbench list-safety-results
observerbench list-ai-control-results
```

Run the controlled authorization interlock:

```bash
observerbench run safety_interlock_analytic \
  --config configs/safety_interlock_analytic.yaml \
  --outdir runs/safety_interlock
```

Within each matched pair, the requested action is fixed while the granted scope
changes. The task reports classification AUROC, realized protocol loss,
tail loss, severity-weighted misses, clean utility, and escalation cost. The
safety-track contract and claim boundary are documented in
`docs/SAFETY_TRACK.md`.

Export the task-pack format for a new researcher-authored safety study:

```bash
observerbench export-safety-task-pack \
  safety_interlock_analytic@paired-scope-v0 \
  --outdir task-pack

observerbench validate-safety-task-pack \
  --public-task task-pack/safety_task.json \
  --targets task-pack/evaluator_targets.json
```

The public file contains the threat model, oracle declaration, fixed policy,
labeled fit measurements, and target-free queries. The second file stays with
the evaluator. See `docs/SAFETY_API.md` for observer and task-author workflows,
and `results/revision/qwen_safety/RESULTS.md` for the learned Qwen baseline.

Run a tiny Ctl-2 closed-loop smoke test:

```bash
observerbench run trained_ctl2 \
  --config configs/trained_ctl2.yaml \
  --outdir runs/smoke_ctl2 \
  --quick
```

Run the IOI Stage 1 quick diagnostic fixture:

```bash
observerbench run ioi_stage1 \
  --config configs/ioi_stage1.yaml \
  --outdir runs/smoke_ioi_stage1 \
  --quick
```

Generate an ObserverCard from a run directory:

```bash
observerbench make-card \
  --results runs/smoke_ctl2 \
  --outdir runs/smoke_ctl2/cards
```

## Reproduce Paper Figures

Fast reproduction reads the paper summaries under `results/frozen/` and the
checked review outputs under `results/revision/`. It regenerates the current
paper figures, tables, and ObserverCards without changing either input tree.
The active manuscript also uses the checked nonlinear-suffix artifact and the
three checked IOI follow-up artifacts.

```bash
python scripts/build_phase04_artifact.py
python scripts/build_phase05_nonlinear_suffix_artifact.py
python scripts/build_phase06_ioi_confirmation_artifact.py
python scripts/build_phase07_ioi_confirmation_artifact.py
python scripts/build_phase08_ioi_sensitivity_artifact.py
python scripts/reproduce_paper_fast.py
```

From the repository root, the complete local manuscript build is:

```bash
make -C paper/observerbench_v15_source
```

That command runs all six steps above and writes
`paper/observerbench_v15_source/build/observerbench.pdf`. Use
`make -C paper/observerbench_v15_source artifacts` to validate and regenerate
the checked artifacts without compiling the PDF. Both commands use only local
checked results; they do not run GPT-2 inference or produce a public repo URL.

The CLI invokes the same renderer from a source checkout:

```bash
observerbench make-figures --outdir paper/generated_current
```

This path is CPU-only. It does not download GPT-2, import TransformerLens, or
train models. The default set matches the current manuscript. Use
`--legacy` only to regenerate the superseded v7 figure/table set.

To regenerate one checked result:

```bash
python scripts/reproduce_paper_fast.py --only revision_ctl2_factorial
python scripts/reproduce_paper_fast.py --only revision_ioi_stage2c
observerbench make-figures \
  --only revision_ioi_mobius \
  --outdir paper/generated_current
python scripts/reproduce_paper_fast.py --legacy
```

The command-to-figure map is in:

```text
paper/figure_map.md
```

That file maps checked claims to:

- the checked result path,
- the fast reproduction command,
- the scientific rerun or audit command,
- expected output files,
- and the scope of legacy frozen outputs.

Full reruns are separate from fast reproduction:

```bash
python scripts/reproduce_paper_full.py
```

By default, the full script will not start expensive runs. Model training and
runs that may require TransformerLens/GPT-2 require explicit acknowledgement:

```bash
python scripts/reproduce_paper_full.py --yes-run-expensive
```

This distinction is intentional. Fast reproduction should verify the paper artifact quickly from frozen outputs. Full reruns are for local scientific auditing and can be slow.

The accepted post-confirmatory target-sensitivity result uses explicit `v2`
artifact roots because an earlier measurement copy was discarded before its
values were inspected or evaluated. The exact audit, inference, and evaluation
commands are in:

```text
docs/ioi_phase08_checked_reproduction.md
```

## Task List

Current task names:

```text
ctl1_analytic
trained_ctl1
trained_ctl2
safety_interlock_analytic
ioi_stage1
ioi_stage2b
ioi_stage2c
ioi_stage2d
```

The main CLI forms are:

```bash
observerbench list-tasks
observerbench run <task-name> --config <yaml> --outdir <dir>
observerbench make-card --results <csv-or-dir> --outdir <dir>
observerbench make-figures --outdir <dir> [--only <result-id>]
```

IOI full reruns are intentionally outside the default test path. They may require the optional IOI dependency scope, local model access, and explicit expensive-run acknowledgement.

## ObserverCards

ObserverCards are compact reports generated from result metrics. They include:

- task name,
- model or substrate,
- access regime,
- observer family and observer name,
- estimand,
- measurement design,
- validation target,
- primary metrics,
- thresholds,
- detected failure modes,
- recommendation,
- scope limits,
- commands to reproduce,
- whether the result is frozen, checked, or a local run,
- source-file hashes and revision provenance.

Example:

```bash
observerbench make-card \
  --results runs/smoke_ctl2 \
  --outdir runs/smoke_ctl2/cards
```

This writes:

```text
observer_card.json
observer_card.md
```

Recommendations are computed from metrics, not observer names. The same name
with different metrics can receive different recommendations, while different
names with identical metrics receive the same recommendation.

The checked paper bundle is under
`results/revision/phase04/observer_cards/`. It contains aggregate cards for
analytic and trained Ctl-1, the six-seed Ctl-2 factorial, and both
capacity-matched IOI measurement designs. The manuscript shows the Ctl-2 card;
the JSON and Markdown artifact contain all five.

## Observer and Task Contract

The v0 Python contract separates the state estimator, actuation-direction provider, controller, and task runner. An external observer can implement the small structural interfaces without subclassing ObserverBench code.

The complete contract and a runnable composition example are in:

```text
docs/API_CONTRACT.md
```

External finite-effect observers use a separate versioned fit/predict and CSV
interchange contract:

```text
docs/EFFECT_PREDICTION_CONTRACT.md
```

That task owns the measurement budget, held-out intervention IDs, targets, and
evaluation. A submitted method supplies only finite-effect predictions. The
generic evaluator writes separate TaskCard and ObserverCard metadata, so an
observer cannot silently redefine the benchmark split or estimand.

The checked Phase-5 IOI tables provide the first built-in finite-effect task at
20, 40, 80, and 160 masks. The frozen Qwen induction-copy adapter adds 16, 40,
64, and 128-mask versions over a complete 256-mask universe. Its checked
artifacts come from the sealed Copy-v2 run described in the paper.
`finite_effect_task_ids()` lists exact `name@version` identifiers, and
`load_finite_effect_task(...)` verifies the task-specific manifests and hashes.
Both loaders return prompt-averaged measurements and held-out targets without
model inference. Each task supplies its own additive ridge reference.

An external implementation can submit the three-column prediction CSV from
`docs/EFFECT_PREDICTION_CONTRACT.md` without importing ObserverBench in its own
process:

```bash
observerbench evaluate-effect-csv <name@version> \
  --artifacts-root <frozen-task-root> \
  --predictions <effect_predictions.csv> \
  --observer-card <observer_card.json> \
  --outdir <evaluation-output>
```

Built-in paper tasks remain in a closed, deterministic registry. External tasks are composed locally with `ObserverTask` and `run_observer_task`; v0 does not perform dynamic plugin discovery.

The bundled analytic Ctl-1 task remains the composition point for running an
outside state estimator and direction provider. The table-backed IOI and Qwen
tasks share a separate composition point for outside finite-effect predictors.
Neither is a
general bring-your-own-model API: model inference remains in the reproduction
scripts. See `docs/API_CONTRACT.md` and
`docs/EFFECT_PREDICTION_CONTRACT.md`.

Current implementations live under:

```text
src/observerbench/tasks/
src/observerbench/tasks/ioi/
src/observerbench/tasks/qwen_induction/
```

Configs live under:

```text
configs/
```

Legacy script entry points under `scripts/` are kept as wrappers or reproduction helpers where possible.

The documented public names and signatures in `docs/API_CONTRACT.md` are the v0 stability boundary. Individual paper-task configuration fields and scientific fixtures are versioned separately.

### Qwen induction reproduction

The checked release includes the completed Copy-v2 run. The producer keeps
locked outcomes unavailable until predictions and actions are frozen. The full
configuration and producer-source manifest are sealed, and the runner rejects
a configuration or source bundle that differs from either seal. Install the
optional runtime dependencies:

```bash
pip install -e '.[qwen]'
```

The 0.5B smoke configuration checks the Qwen hooks and artifact path but does
not reproduce a paper claim:

```bash
python scripts/run_qwen_induction_phase09.py \
  --config configs/revision/phase09/qwen2_5_0_5b_induction_smoke.json \
  --artifacts-root results/revision/phase09/qwen_induction_smoke \
  --stage engineering-smoke \
  --resume
```

The scientific 7B reproduction uses the frozen config and this exact stage
order:
`prepare`, `discover`, `confirm`, `freeze-design`, `measure-calibration`,
`freeze-observers`, `measure-locked-test`, `evaluate`, and `measure-collateral`.
The collateral stage measures only masks already selected by the sealed actions
on deterministic matched non-induction controls; it does not gate or alter the
primary analysis. A failed clean, hook-parity, discovery, or confirmation gate
stops the run. The Colab driver is
`notebooks/qwen_induction_phase09_colab.ipynb`; it keeps all experiment logic in
the package and writes resumable artifacts to a user-selected Drive directory.
Resume accepts only checkpoints produced by the same source, model, hardware,
and dependency identity; use a fresh artifact root after an environment change.

## Known Limitations

ObserverBench v0 does not include:

- a comprehensive InterpBench sweep,
- adversarial oversight,
- a general LLM safety evaluation,
- fully unsupervised discovery of mechanisms,
- dynamic plugin loading or remote task execution,
- hosted model inference or a sandbox for uploaded executable code.

The IOI tasks are known-circuit diagnostics. They use published IOI head groups
as borrowed structure for testing observer methodology. They should not be read
as a new IOI circuit discovery claim. The Qwen task is described more narrowly
as an induction-copy mechanism surface; it does not claim circuit completeness.

The frozen and checked outputs are summaries needed for reproduction. They do
not include model weights, GPT-2 caches, virtual environments, or private local
paths.

Slow tests are skipped by default:

```bash
pytest -q
```

To include slow local smoke tests:

```bash
pytest -q --runslow
```

CI runs the default suite only. It must not download GPT-2 or train models.

## Citation

Please cite the archived software release:

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

The conceptual framework is described in
[Mechanistic Tomography](https://arxiv.org/abs/2608.19338).
