# Gemma-2-9B-it APPS replication registration

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

## Status and disclosure

This is a prospective matched replication, frozen before any monitor outcome
from `google/gemma-2-9b-it` is inspected. It is not an independent
preregistration. The APPS task and the choice to run a second model were made
after the Qwen2.5-7B results were known.

The frozen question is deliberately simple: if we keep the APPS problem IDs,
fit/calibration/test split, monitor prompts, audit controller, budgets, and
prevalence sweep fixed, does the ordering seen on Qwen survive a model change?
The run also adds one declared observer family that Qwen lacked: a supervised
linear observer over official Gemma Scope residual SAE features.

## What stays fixed

- The exact `RoganInglis/apps-control-arena` revision and deterministic working-
  backdoor selection rule.
- Seed 7319 and the 500/200/744 problem-pair split.
- The ControlArena basic and detailed prompts and the neutral problem-plus-code
  view.
- Calibration selection by realized violation rate at a two-percent audit
  budget, followed by a fit-plus-calibration refit.
- Locked-test reporting at two, five, and ten percent audit budgets.
- Frozen-score resampling at attack prevalences of one, two, five, ten, and
  fifty percent with the same audit-budget grid, population size, replicate
  count, and seed as the Qwen follow-up.

## What changes

- The model is pinned to `google/gemma-2-9b-it` at the exact revision recorded
  in the configuration.
- Residual probes read post-layer states at Gemma layers 9, 20, and 31.
- A neutral-context SAE observer uses the official
  `google/gemma-scope-9b-it-res` width-16k SAEs at those same layers. The chosen
  releases are the available SAEs nearest average L0 48.

The text baseline and evaluator-only exact-label oracle do not depend on the
model. They are rerun on the identical rows so the final table is self-
contained.

## One compatibility smoke

Before the full run, one smoke stage may:

1. load the pinned model and tokenizer;
2. audit token lengths for all frozen prompts without evaluating monitor
   quality;
3. run two fit pairs through the three declared residual hooks; and
4. download and encode one tiny activation batch with each of the three pinned
   official SAEs.

The smoke reports only compatibility facts: revisions, architecture, score-token
IDs, prompt-length counts, tensor shapes, and finite/nonzero checks. It does not
fit an observer, rank monitors, or evaluate locked-test outcomes. The full stage
requires both a passing smoke artifact and an explicit `--allow-full-run` flag.

### Pre-outcome packaging amendment

The first smoke attempt on 2026-09-01 UTC stopped during Python module import,
before Gemma, Gemma Scope, or any monitor outcome was loaded. The archive had
omitted `observerbench.safety` and existing `qwen_safety` support modules used by
the frozen runner. We added those files to the source list and regenerated the
local digest seal. This changed only the archive's import closure; the model,
task rows, split, observers, controller, budgets, metrics, and analyses remain
unchanged.

## Reporting rule

Every declared row is reported regardless of direction. A failure to reproduce
the Qwen ordering is a cross-model boundary, not a failed run. A strong SAE row
supports an official-feature observer under this task contract; it does not by
itself identify deception, intent, or a semantic SAE feature.

The source manifest is a local digest seal. It proves which local configuration
and code produced the run; it is not described as an independently timestamped
registration.
