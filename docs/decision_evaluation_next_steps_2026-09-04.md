# Decision evaluation: next steps

Date: September 4, 2026. This is an implementation checklist, not a claim that
the capabilities below are already available.

## Make an outside submission a complete decision test

The first implementation is the existing GPT-2 IOI task, not a new task
family. Its local artifact contains all 320 candidate masks on all 256
held-out prompts (81,920 rows, ten pools; no missing masks or duplicate
prompt/mask pairs). No additional model measurements are needed. Exact no-op
has an analytic response of zero. A new decision-evaluation contract can add
that baseline without rewriting the historical experiment or prediction task.

- [ ] Connect one public effect-prediction task to the existing frozen
  action-selection runners. The generic CSV evaluator currently reports
  prediction errors; the paper's action-loss experiments use separate runners.
- [ ] Declare the submitted quantity: mean effect or expected prompt-level
  action loss. A mean prediction cannot be silently treated as a loss prediction.
  Start with the existing effect CSV and the frozen mean-selection rule; defer
  a separately typed loss-prediction route until that path works.
- [ ] Freeze candidate pools, target, exact no-op, selection rule, and tie-breaking
  in the task version. Do not add a threshold after seeing evaluation outcomes.
- [ ] Return selected action IDs, held-out action loss, excess loss over no-op,
  differences from named reference observers, prediction metrics, and uncertainty.
- [ ] Score one published observer (AtP) through that complete path. Preserve
  its mean-effect estimand; label any fitted loss adapter separately.
- [ ] Supply a CPU-only example: download the task, submit a table, and obtain
  a decision scorecard in one command. Test it from a clean installation.
- [ ] Separate prediction-only and decision-tested rows on the website. Do not
  rank methods from different targets, access boundaries, or task versions together.

## Make uncertainty and cost part of the returned score

- [ ] Add paired source-problem bootstrap intervals to safety scorecards.
  Keep deployment outcome ranges separate; repeated simulated populations are
  not new independent APPS problems.
- [ ] Test stability across independently refit observers and split seeds.
  Fixed-score bootstrap intervals cannot answer that question.
- [ ] Count the whole observer path: model inference, residual extraction,
  SAE encoding, readout, calibration measurements, time, and memory. A sparse
  readout coefficient count is not a runtime or total-compute measurement.
- [ ] Add failure tests for missing/duplicate action IDs, target mismatch,
  malformed values, no-op handling, fit/test separation, and tie handling.

## Extend scope only after the submission path is complete

- [ ] Add closed-loop token-generation control on a pretrained model, with a
  fixed actuator and an independent target check.
- [ ] Add an adaptive safety task with a declared attacker budget and trusted
  versus corruptible telemetry. Stored-backdoor rankings do not establish this.
- [ ] Add a checkpoint-matched posttrained Qwen3.5 SAE when available or train
  one under a separate, documented study. Preserve the existing transfer result.
- [ ] Add prompt/task transfer with frozen observers before claiming broad
  language-model decision improvements.

## Manuscript work in this revision

- [x] Align interface claims with what the current evaluator returns.
- [x] Make initial-state calibration explicit in the reachable-subspace example.
- [x] Distinguish action-ranking evidence, sparse-readout results, and theory's
  role as a diagnostic rather than a new general decision-learning theorem.
- [x] Add source-problem uncertainty from stored scores and full protocol details.
- [x] Rebuild both papers and check the rendered pages. The ICLR main text ends
  on page 9, with statements beginning on page 10. Both ICLR build locations
  hold the same PDF and remain private; only the long paper is for publication.

Finishing these checks would remove concrete objections. No checklist can
guarantee a reviewer score; broader scientific evidence still requires the
experiments listed above.

### What the new uncertainty check found

The secondary source-problem bootstrap supports verdict-over-neutral action
advantages on Gemma and Qwen3.5. It does not establish both sides of either
highlighted AUROC/action ranking inversion: Gemma's action-loss difference
and Qwen3.5's AUROC difference include zero. Both papers report this outcome,
including the conditional nature of the intervals. The original scorecards
and frozen archives are unchanged.
