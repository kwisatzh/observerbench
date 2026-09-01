# Qwen3.5-9B APPS replication registration

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

## Status and disclosure

This document freezes the plan before any Qwen3.5-9B monitor outcome is read.
The APPS task and the comparison were chosen after the Qwen2.5-7B and
Gemma-2-9B-it results were known. This is therefore a prospective matched
replication, not an independent preregistration.

The model and official SAE repositories are pinned to the exact public commits
available on 2026-09-01. The sealing and archive tools refuse branch names or
other mutable revisions. A compatibility smoke may begin only after the
complete source bundle is reviewed and sealed.

## Frozen task and primary contrast

The task ID is
`controlarena-apps-verified-backdoors@qwen3.5-9b-v0`. The run uses exactly the
same APPS rows, 500/200/744 fit/calibration/locked-test split, monitor prompts,
top-score audit controller, audit budgets, and missed-working-backdoor loss as
the earlier Qwen and Gemma studies. Thinking is disabled with
`enable_thinking=false` for every model view.

The primary result is the signed contrast

`mean realized violations(neutral residual) - mean realized violations(verdict-adjacent residual)`

at one-percent attack prevalence and a two-percent audit budget. Every declared
observer and every frozen prevalence-by-budget cell is reported regardless of
the direction of this contrast.

## What changes

- The evaluated model is `Qwen/Qwen3.5-9B`, measured at residual layers 12, 20,
  and 28.
- The official-feature control uses
  `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` at the same three layers.
- The model revision is `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- The SAE revision is `7bb2370d360e566b2d8acb8b3d15a0b2b88f52a8`;
  the public LFS SHA-256 digest and byte count of each selected SAE file are
  recorded in the configuration.

## Declared SAE mismatch

The official SAE was trained on `Qwen/Qwen3.5-9B-Base`, while the evaluated
model is the post-trained `Qwen/Qwen3.5-9B`. This base-to-posttrain mismatch is
part of the frozen design. The SAE observer is a useful control over a released
feature family; it is not checkpoint-matched SAE evidence, and a weak result
does not establish a general SAE failure.

## Compatibility smoke

After the sources are sealed, one smoke may
load the model and tokenizer, check all frozen prompt lengths, inspect the three
residual hook shapes, and encode a tiny batch through all three declared SAEs.
It reports only compatibility facts. It must not fit an observer, rank monitors,
or evaluate locked-test outcomes.

## Frozen execution stages

After the smoke is reviewed, the core stage measures the prompt, text, score-
distribution, and dense-residual observers and writes the signed primary
contrast. A separate cache-only stage then applies all three official SAEs to
the sealed neutral-residual cache. It selects the SAE layer and ridge from fit
and calibration rows only and writes separate result files. The SAE stage does
not reload the model or rewrite the core result.

## Scope

This frozen offline study does not test adaptive attacks, identify deception or
intent, or claim that agreement across models establishes universal monitor
adequacy. The local digest seal records the exact files used by the run; it is
not an independently timestamped registration.
