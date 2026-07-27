# Frozen preregistration: Qwen induction-copy mechanism surface, Copy-v2

Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

Status: frozen before any Copy-v2 clean scores or other model outcomes. The
Phase-10 configuration, this document, and the producer source are digest-sealed
before the first Copy-v2 model forward pass.

Data version: `copy-v2`.

Frozen configuration SHA-256:
`d55b416eda85087da7ec4ef9e1249333c954e2576fb8749719f8bb7d7be55f4f`.

Fresh artifact root:
`results/revision/phase10/qwen_induction_copy_v2`.

## Why this is a new study

Copy-v1 stopped at its frozen clean gate. On its discovery bank,
Qwen2.5-7B reached candidate accuracy 0.9297 rather than the registered 0.95;
the four family rates ranged from 0.9063 to 0.9688. No attention scan or
intervention outcome was opened. Copy-v1 remains a negative result and its
artifacts remain immutable.

Copy-v2 does not lower the Copy-v1 gate and does not continue that run. It
defines a new conditional population before inspecting any new prompts:
synthetic induction-copy fixtures on which the clean pinned model confidently
identifies the planted continuation among the three declared candidates.
Conditioning on clean competence is common in causal intervention studies, but
it narrows the claim. Copy-v2 cannot establish performance or mechanism claims
for the unfiltered prompt distribution.

## Model, task, and claim boundary

The model remains base `Qwen/Qwen2.5-7B` at revision
`d149729398750b98c0af14eb82c78cfe92750796`, loaded unquantized in bfloat16.
The task remains unchanged: each prompt contains three distinct key-value
bigrams and ends with a repeat of one key. The planted continuation is the
value that followed that key; the other two values are declared distractors.

For prompt `i`, define

    M_i = logit_i(target) - logmeanexp(logit_i(distractor_1),
                                      logit_i(distractor_2)).

For intervention mask `m`, define

    Y_i(m) = M_i(no-op) - M_i(m).

Positive effects mean that the intervention impaired candidate-constrained
copy discrimination. Full-vocabulary top-1 accuracy and target NLL remain
secondary diagnostics. They do not define eligibility, head selection, or the
primary response-surface claim. In particular, Copy-v2 must not be described
as evidence that Qwen copies the planted token in unconstrained generation
unless the separately reported top-1 results support that statement.

If the registered coverage and causal gates pass, the study may establish a
Qwen **clean-eligible induction-copy mechanism surface**: a causally validated
eight-head panel whose finite joint interventions are measured exhaustively in
the registered basis. It does not claim circuit completeness, general
in-context learning, or transport to prompts that fail the clean eligibility
rule.

## Frozen execution environment

The Colab run uses Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8, and the
exact package versions in `configs/revision/phase10/colab_constraints.txt`.
The producer sets the registered matmul and cuDNN TF32 policy inside every
stage process and records CUDA, cuDNN, package, device, and precision settings
in the runtime artifacts. A version or precision mismatch stops the run.

Resume is explicit. Completed stages and their prerequisites are hash-checked.
An interrupted clean, attention, or reference-mean computation is recomputed;
only per-mask intervention checkpoints may be reused, and each such checkpoint
must already appear in its hash-bound manifest. Candidate and preselection
directories are written atomically, so a crash cannot expose a partial bundle
as a frozen design.

## Fresh, preassigned reservoirs

Copy-v2 uses new token-pool and sequence seeds. Before candidate generation,
the producer excludes the union of all token IDs allocated to the six Copy-v1
token banks. The preserved Copy-v1 `token_banks.json` is an input to this
exclusion. Its frozen file hash is
`f599a958e310fdf9cedd52f92c77555cfd57b58fc117536781c21101f5786c91`,
and its declared 24,576-token pool hash is
`34f3ef26314647c4df54e4c8a8a80888f5663b1dbce64a7234d35a9c1f90cf33`.
Copy-v2 therefore reuses neither Copy-v1 prompts nor Copy-v1 allocated tokens.

Before any clean scoring, the producer assigns every candidate to one of six
disjoint banks and one of four fixed length-gap families. Each bank receives a
separate token allocation. Prompt IDs are globally unique, and token IDs do
not cross banks.

The four families remain `(32, 8)`, `(32, 16)`, `(64, 8)`, and `(64, 16)`,
where the pair denotes sequence length and repeat gap. Reservoir sizes are
exactly twice the final counts:

| Bank | Final prompts per family | Reservoir prompts per family |
|---|---:|---:|
| Reference | 32 | 64 |
| Discovery | 32 | 64 |
| Head fit | 64 | 128 |
| Head confirmation | 64 | 128 |
| Calibration | 64 | 128 |
| Locked test | 128 | 256 |

The complete reservoirs, prompt contents, bank assignments, token exclusions,
and hashes are written before Qwen inference. A failed reservoir is not
expanded or regenerated.

## Clean-only eligibility

The producer scores every reservoir prompt without hooks or interventions
using the pinned model and SDPA implementation. A prompt is eligible if and
only if:

1. the target is the highest-logit declared candidate;
2. `M_i >= ln(4) = 1.3862943611198906`; and
3. the candidate logits and target NLL are finite.

The threshold has a fixed interpretation. Under the three-candidate softmax,
`M_i >= ln(4)` implies target probability at least two thirds. It is not a
Copy-v2 quantile and cannot be changed after clean scores are seen.

For the discovery reservoir, the clean candidate prediction must also agree
between eager attention and SDPA, and the margin must meet `ln(4)` under both
implementations. This bank is the only one used for eager attention screening.
Head fitting and every causal measurement use SDPA.

Full-vocabulary top-1 correctness is recorded but cannot include or exclude a
prompt. Once a prompt passes the threshold, its margin, NLL, top-1 status,
decoded text, and token IDs cannot affect selection.

## Coverage gate

The complete preassigned reservoirs form the denominator. Before selecting
final prompts, Copy-v2 must satisfy all of the following:

- eligible fraction at least 0.80 overall;
- eligible fraction at least 0.75 in every family, pooling banks;
- eligible fraction at least 0.70 in every bank-by-family cell; and
- enough eligible prompts to fill every registered final cell.

Any failure terminates Copy-v2. We do not lower a threshold, add candidates,
regenerate tokens, remove a bank or family, or substitute Copy-v1 prompts.

The coverage artifact reports counts, fractions, and descriptive 95-percent
Wilson intervals overall, by family, and by bank-by-family cell. It also
reports, for the full reservoir and separately for included and excluded
prompts:

- unfiltered candidate accuracy;
- candidate-margin quantiles;
- full-vocabulary top-1 accuracy;
- target-NLL summaries; and
- token-ID, decoded-length, and character-class distributions.

These summaries expose how much of the generated task survives the clean-only
condition and whether eligibility selects a narrow token class.

## Deterministic inclusion and freeze

Within each eligible bank-by-family cell, the producer orders prompts by
`SHA-256("12119:" || prompt_id)`, breaking an impossible hash tie by
`prompt_id`, and takes the registered final count. It never ranks prompts by a
model score or token property.

The inclusion artifact retains every candidate's clean scores, eligibility
decision, exclusion reason, and final-selection flag. The producer then seals
the final prompt tables, inclusion artifact, coverage report, Copy-v1 token
exclusion, model/runtime audit, and source hashes. No attention map, direct
logit attribution, reference mean, head ablation, or other intervention may be
computed before this clean-only freeze completes.

When a selected bank is later used, the producer rescoring must reproduce the
stored candidate prediction and retain margin at least `ln(4)`. We report
margin differences but do not gate on bit-level equality: batch order can
change the last digits under the same model and implementation. The existing
primary-hook no-op parity gate also remains in force.

## Disjoint scientific stages

Clean scoring is observed for all reservoirs because it defines the target
population. Intervention effects remain split and sealed as follows:

1. The selected discovery bank screens every query head by target-value
   attention minus mean distractor-value attention.
2. Only the top 32 screened heads receive singleton mean ablations on head-fit.
   Before either reference-mean capture, the selected reference prompts must
   pass their SDPA clean rescore; both reference gates enter the frozen design.
3. Eight positive-lower-bound heads are selected with at most two per layer and
   at least four layers represented. Matched controls use discovery geometry,
   never control causal outcomes.
4. On head-confirmation, the selected full panel must impair copying more than
   its matched-control panel under the registered paired 95-percent interval.
5. Only a passed causal gate freezes the head panel and mask design.
6. Calibration effects are measured, then all predictors, coefficients,
   targets, actions, and selected-mask unions are sealed.
7. Only then may the locked-test intervention effects be opened and the
   primary evaluation run.
8. The secondary matched non-induction collateral diagnostic runs after the
   primary evaluation and cannot change or block it.

The six selected prompt banks retain disjoint prompt IDs and token banks.
Head-fit cannot read head-confirmation effects; calibration cannot read locked
effects; and locked clean margins cannot enter any predictor or action rule.

## Downstream protocol retained from Copy-v1

Copy-v1 stopped before head discovery, so its downstream choices have no Qwen
intervention outcome conditioning. Copy-v2 retains them unchanged:

- top-32 attention shortlist; eight selected heads; eight matched controls;
- family-conditioned mean replacement at final-query attention `z` before the
  output projection;
- final-query zeroing as robustness only;
- exhaustive 256-mask Boolean cube over the eight selected heads;
- calibration/test partition of 128 masks each and nested budgets 16, 40, 64,
  and 128;
- no-effect, additive, and all-pairs quadratic mean-effect models;
- primary additive-minus-quadratic held-out MAE at budget 128;
- natural-mean, transformed-mean, direct-risk, and exact-no-op selectors using
  the same 37-column quadratic basis;
- targets at 0.25, 0.50, and 0.75 of the confirmed selected-full effect; and
- target-specific and equal-weight aggregate action contrasts with common
  bootstrap draws.

The matched non-induction collateral diagnostic also remains secondary and
non-gating.

## Interpretation and stopping rules

- Coverage failure: negative evidence that the registered conditional task is
  not broad enough; no intervention run and no second-mechanism claim.
- Causal confirmation failure: no second-mechanism claim and no response
  surface.
- Coverage and causal gates pass: Copy-v2 may support a second model and a
  second, clean-eligible mechanism surface.
- Quadratic prediction wins the registered budget-128 contrast: positive
  transfer of interaction-aware effect prediction.
- The contrast is null: valid second mechanism task, but no positive
  interaction-aware transfer claim.
- Direct risk beats both same-basis controls and exact no-op on the registered
  aggregate: positive transfer of the observer-estimand result.
- A learned action loses to no-op: negative result for acting at that target.

Copy-v1's clean-gate failure must be reported alongside Copy-v2. Copy-v2 does
not retroactively rescue the broader Copy-v1 population.

## Registered risks

Clean-margin conditioning changes the prompt and finite-effect distributions
and can enlarge apparent ablation effects. The fixed threshold, two-times
reservoir, hash selection, disjoint effect splits, and explicit coverage report
limit researcher choice but do not remove that population shift. The resulting
task is model-specific rather than a model-neutral prompt benchmark. Low
full-vocabulary top-1 performance would further restrict the result to
candidate-constrained copying. Any redesign after viewing Copy-v2 clean scores
must receive a new data version and cannot be folded into this study.
