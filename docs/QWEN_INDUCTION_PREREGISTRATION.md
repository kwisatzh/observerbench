# Preregistration: Qwen induction-copy mechanism surface

Status: frozen before Qwen model outcomes.

The production runner accepts only the canonical frozen-config SHA-256
`ec2835be61a85c6f963cab901c1de17f512e9fa08d5983c79bd14874000bdb77`.
Scaled configs are test fixtures and cannot enter the scientific runner.
The producer source is separately sealed in
`configs/revision/phase09/qwen_phase09_source_manifest.json`; its bundle digest
is `5a36270b977dd5cf11da9048e641c8e05eec375b4b7a502b55416ed4d8bcc3ca`.
The production runner refuses source drift. Record a private commit or tag for
the sealed checkout before launching the A100 run.

This is ObserverBench's prospective second-model, second-mechanism study. It
does not reuse the refusal-margin endpoint in the separate Mechanistic
Tomography follow-up. That endpoint has exact behavioral measurements but no
mechanistic ground truth. Here the next-token target is fixed by construction,
and component selection and validation use disjoint prompt splits.

## Claim boundary

The primary model is the base `Qwen/Qwen2.5-7B` checkpoint at revision
`d149729398750b98c0af14eb82c78cfe92750796`. The base checkpoint avoids chat
templates and post-training as extra variables. `Qwen/Qwen2.5-0.5B` at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` is an engineering smoke only. It
cannot support a paper claim.

The study may establish an **induction-copy mechanism surface**: a causally
validated set of heads whose finite joint interventions are measured
exhaustively in a declared basis. It does not claim that the eight heads form a
complete Qwen induction circuit, that induction explains general in-context
learning, or that the result transfers beyond the pinned model and task.

## Exact task

Each prompt contains three distinct key-value bigrams and ends with a repeat of
one key. The correct next token is the value that followed that key earlier.
The other two values are matched distractors. Tokens are direct tokenizer IDs;
every token is regular, printable, non-special, and round-trips through the
pinned tokenizer. A prompt contains no collision except the deliberate final
key repeat.

Four families cross sequence lengths 32 and 64 with repeat gaps 8 and 16. The
target margin for prompt `i` is

    M_i = logit_i(target) - logmeanexp(logit_i(distractor_1),
                                      logit_i(distractor_2)).

The finite effect of mask `m` is

    Y_i(m) = M_i(no-op) - M_i(m).

Positive effects therefore mean that the intervention impaired exact copying.
Candidate accuracy, full-vocabulary top-1 accuracy, target NLL, and the margin
are all retained. The candidate margin is primary because the construction
declares exactly three matched continuations; full-vocabulary accuracy remains
a stricter secondary diagnostic.

Reference, attention-discovery, head-fit, head-confirmation, calibration, and
locked-test prompts use disjoint prompt IDs and disjoint token banks. Token IDs,
family assignments, prompt rows, and hashes are frozen before model inference.

## Clean and runtime gates

Before intervention outcomes on a split are used:

- candidate accuracy must be at least 0.95 overall and 0.90 in every family;
- the median clean candidate margin must be positive;
- an all-false intervention hook must reproduce hook-free candidate margins
  within `1e-5` and reproduce candidate predictions exactly;
- the loaded model must report 28 layers, 28 query heads, four KV heads, and
  Q/K/V biases, as specified by the official checkpoint;
- the run must use native Hugging Face Qwen2 weights without quantization.

Failure stops the affected scientific stage. Thresholds are not weakened after
outcomes are seen.

## Component discovery and confirmation

The discovery split screens every query head by attention from the final key to
the correct earlier value minus the arithmetic mean of its attention to the two
distractor values. Direct logit
attribution and output norms are recorded as diagnostics. Attention alone does
not earn the label "induction head."

Only the top 32 attention-screened heads receive singleton mean ablations on
the head-fit split. The eight heads with the largest positive lower confidence
bounds are frozen, subject to at most two heads per layer and coverage of at
least four layers. If eight eligible heads do not remain, the study stops.

Each selected head is matched to a low-induction control in the same layer and
KV group with the closest clean output norm. Control matching never reads a
control's causal outcome. On the separate head-confirmation split, the selected
eight-head intervention must impair copying more than the matched-control
intervention under a paired 95-percent interval that excludes zero. Failure is
a negative result and stops the expensive response-surface run.

The primary intervention replaces a selected head's pre-output-projection `z`
vector at the final query position with its family-conditioned mean from the
reference split. Head zeroing is a robustness analysis only. Every selected
query head records its layer, query-head index, and shared KV-group index.
After the primary selected-versus-control confirmation gate passes, we repeat
that full-panel comparison with final-query head zeroing. This robustness result
does not affect selection, stopping, targets, or the primary response surface.

## Exhaustive mask design

Eight frozen components yield exactly 256 binary masks. The full universe is
partitioned before calibration or locked-test mask outcomes:

- calibration: no-op plus 127 nonempty masks;
- locked test: the remaining 128 masks;
- nested calibration budgets: 16, 40, 64, and 128 masks;
- the first 40 rows include the no-op, all eight singletons, and all 28 pairs,
  so the 37-column quadratic design has full rank;
- the remaining calibration rows are selected deterministically for
  conditioning and density coverage;
- the 128 test masks form 16 fixed pools of eight masks; an exact analytic
  no-op is added to every pool.

The locked set is therefore the complete complement of the calibration set,
not a sampled or pilot-selected subset.

## Prediction and action questions

Mean-effect prediction compares three fixed models:

1. a no-effect diagnostic;
2. an additive ridge model with an intercept and eight head indicators;
3. a quadratic ridge model with the same terms plus all 28 head pairs.

The primary prediction contrast is additive minus quadratic held-out mask-level
MAE at the full preregistered budget of 128 calibration masks. The 16-, 40-,
and 64-mask results are budget curves, not alternative primary tests. RMSE,
R-squared, density slices, and prompt-cell residuals of each mask-level mean
prediction are secondary. A paired bootstrap resamples prompt clusters and
masks with common draws for both models.

Three targets equal 0.25, 0.50, and 0.75 times the selected full-mask effect on
the head-confirmation split. All targets and their equal-weight aggregate are
reported. Every policy chooses from the same test pool plus exact no-op:

- natural mean fits mean effect, then applies absolute target loss;
- transformed mean fits the absolute loss of the calibration-mask mean;
- direct risk fits mean prompt-level absolute target loss;
- exact no-op is reported as its own policy.

All three learned policies use the identical 37-column quadratic basis. Thus a
direct-risk advantage cannot be attributed to greater feature capacity.
Predictions, coefficients, target values, actions, selected-mask unions, source
hashes, and environment metadata are sealed before locked outcomes are opened.

Primary action metrics are realized absolute target loss, regret to the pool
oracle, and paired contrasts of direct risk against natural mean, transformed
mean, and no-op. We report each target and an equal-weight aggregate across all
three. The aggregate bootstrap uses the same resampled action pools and prompt
clusters for every target and selector. The decomposition

    E|Y-t| = |E[Y]-t| + J_t

is reported per mask, where `J_t` is the prompt-dispersion penalty.

## Secondary copy-control diagnostic

After the observers and actions are frozen, every locked-test prompt receives a
deterministic matched non-induction control. We swap the repeated final key with
a unique filler outside all three key-value bigrams. This changes exactly two
positions, preserves sequence length and the exact token multiset, and leaves a
final token that never appeared earlier. The transform and its prompt IDs are
content-hashed.

We measure every unique mask selected by any frozen action, plus analytic
no-op. The mask set depends only on the sealed actions, not on locked outcomes.
For each mask and control prompt, we report
`KL(p_clean || p_intervened)` over the full next-token vocabulary and total
variation, defined as one half of the L1 distance. We aggregate by mask,
family, and frozen action. Measurement occurs after the locked-test intervention
and primary evaluation stages because it uses the same sealed actions. A
collateral runtime failure cannot block or change the primary evaluation. This
asks whether an action that
suppresses exact propagation also distorts a matched ordinary continuation. It
is a narrow collateral diagnostic, not an AI-safety benchmark or deployment
claim.

## Artifact and stopping rules

Expensive stages checkpoint after each head or mask. The final public task is a
hash-checked table adapter under ObserverBench's existing
`FiniteEffectPredictionTask` contract. External users supply predictions or a
predictor and never need Qwen, Transformers, a GPU, or the prompt token IDs.

Interpretation is fixed:

- causal gate passes and quadratic prediction wins: positive transfer of
  interaction-aware effect prediction to a second model and mechanism;
- causal gate passes and prediction is null: useful negative result plus a
  valid second inference-free task;
- direct risk beats both same-basis controls: strongest positive transfer of
  the observer-estimand result beyond IOI;
- a learned policy loses to no-op: negative result for acting at that target;
- causal gate fails: no second-mechanism claim and no expensive surface run.
