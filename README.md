# ObserverBench

ObserverBench is a research artifact for evaluating observers in control-oriented interpretability.

The central question is not whether a richer observer can be made to win. The question is when an observer is good enough to support intervention, and when a cheap or misspecified observer stays target-useful while silently changing collateral state, prediction error, or feedback behavior.

This repository is the v0 reproduction/workbench for the ObserverBench paper. It is not a finished universal benchmark.

## What is ObserverBench?

An ObserverBench task fixes:

- a substrate or model state,
- a target state to estimate, control, or predict,
- a set of candidate observers,
- an actuator or intervention distribution,
- a validation target that checks whether the observer remains useful under intervention.

The v0 repo includes the paper tasks:

- Ctl-1 analytic collateral geometry.
- Trained-transformer Ctl-1 one-shot observer geometry.
- Trained-transformer Ctl-2 closed-loop observer-control.
- IOI known-circuit diagnostics on GPT-2-small outputs:
  Stage 1 whole-group self-repair, Stage 2b random head subsets, Stage 2c primary-stratified subsets, and Stage 2d per-pair decomposition with a count-additive control.

The repo is meant to reproduce the paper figures, generate ObserverCards, and expose a minimal CLI for the current tasks.

## Why Observers Matter For Control

Control-oriented interpretability uses internal measurements as inputs to action: activation steering, representation engineering, patching workflows, constrained decoding, or feedback controllers. In all of these cases, the controller does not consume an explanation in the abstract. It consumes an observer: a procedure that maps measurements to a state estimate.

That observer can fail in ways that ordinary prediction metrics do not show. It can move the target while dragging collateral state. It can look adequate in one-shot control and fail in a feedback loop. It can be sufficient under broad random interventions and insufficient under mechanism-targeted interventions.

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

Fast reproduction reads frozen CSV/JSON summaries under `results/frozen/` and regenerates the paper figures and tables into `paper/generated_figures/`.

```bash
python scripts/reproduce_paper_fast.py
```

This path is CPU-only. It must not download GPT-2, import TransformerLens, or train models.

To regenerate one result:

```bash
python scripts/reproduce_paper_fast.py --only figure_05
python scripts/reproduce_paper_fast.py --only table_02
```

The command-to-figure map is in:

```text
paper/figure_map.md
```

That file maps every paper figure/table to:

- the frozen result path,
- the fast reproduction command,
- the full rerun command,
- expected output files,
- approximate runtime,
- GPU/MPS and TransformerLens/GPT-2 requirements.

Full reruns are separate from fast reproduction:

```bash
python scripts/reproduce_paper_full.py
```

By default, the full script refuses expensive runs. Anything that trains a model or may require TransformerLens/GPT-2 needs an explicit acknowledgement:

```bash
python scripts/reproduce_paper_full.py --yes-run-expensive
```

This distinction is intentional. Fast reproduction should verify the paper artifact quickly from frozen outputs. Full reruns are for local scientific auditing and can be slow.

## Task List

Current task names:

```text
ctl1_analytic
trained_ctl1
trained_ctl2
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
observerbench make-figures --results <dir> --outdir <dir>
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
- commands to reproduce.

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

Recommendations are metric-derived. They are not stamped from observer names. For example, the same observer name with opposite metrics should receive different recommendations, and different observer names with identical metrics should receive the same recommendation.

## Internal Task Structure

The task API in v0 is minimal and internal-facing. The docs may explain how the current tasks are structured, but they do not define a stable external task/plugin interface.

Current implementations live under:

```text
src/observerbench/tasks/
src/observerbench/tasks/ioi/
```

Configs live under:

```text
configs/
```

Legacy script entry points under `scripts/` are kept as wrappers or reproduction helpers where possible.

You can inspect or adapt the current task implementations, but v0 makes no API stability guarantees.

## Known Limitations

ObserverBench v0 does not include:

- a comprehensive InterpBench sweep,
- adversarial oversight,
- a general LLM safety evaluation,
- fully unsupervised discovery of mechanisms,
- a stable external plugin/task API,
- a hosted demo or product surface.

The IOI tasks are known-circuit diagnostics. They use published IOI head groups as borrowed structure for testing observer methodology. They should not be read as a new IOI circuit discovery claim.

The frozen outputs are summaries needed for reproduction. They do not include model weights, GPT-2 caches, virtual environments, or private local paths.

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

If you use this repository, cite the ObserverBench paper version associated with this release and include the repository commit hash. The public citation block will be updated when final bibliographic metadata is available.
