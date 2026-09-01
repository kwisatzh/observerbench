# ObserverBench safety track

ObserverBench treats a safety monitor as an observer only after the task fixes
what it estimates, what actions consume the estimate, which states an attacker
or controller may reach, and what the resulting protocol counts as loss.
Classification quality is reported, but it is not the acceptance condition.

## Release requirement

Safety is a first-class release claim only when all of the following are true:

1. At least one runnable task has exact evaluator-owned policy labels.
2. The task exposes `allow`, `block`, and `escalate` through a frozen controller.
3. Every observer is compared at the same block and escalation budgets.
4. Results include realized violations, tail loss, benign utility, and audit cost.
5. A no-action reference and an exact policy-engine reference are present.
6. Text or action-only baselines are run before an activation result is accepted.
7. Held-out attack families are evaluated before adaptive attacks are introduced.
8. The TaskCard states who can alter, omit, delay, or spoof each measurement
   and whether requests are independent or share episode-level attacker state.
9. The threat model states what the attacker knows about the monitor, what
   feedback it receives, and whether later attacks may adapt to earlier policy
   decisions.

## Phase S0: protocol contract

`observerbench.safety` defines the initial contract. A task supplies labeled fit
measurements, held-out queries, evaluator-only targets, a task card, and a frozen
triage policy. An outside observer supplies risk scores. The benchmark—not the
observer—turns those scores into block, escalation, and allow decisions.

The v1 TaskCard also declares measurement integrity and episode structure. A
fixed offline task may state that its views are trusted and its requests are
independent. A trajectory task must instead state which telemetry is
authenticated, what an attacker can corrupt, and which rows belong to one
shared episode.

The primary quantities are mean protocol loss, tail protocol loss, severity-
weighted missed risk, and clean utility retained. AUROC remains a diagnostic.

## Phase S1: paired authorization interlock

The first controlled task is inert. A benign requested tool action is held fixed
within each pair while the granted resource scope changes. The authorization
graph is evaluator-owned. This prevents the action text from stating the label
and directly implements the stop rules learned from the rejected cyber fixture.

The analytic fixture is a contract and consequence test, not evidence about a
pretrained model. It compares:

- an action-only label observer;
- an activation label observer;
- an activation observer fit to severity-weighted risk;
- an exact authorization-risk oracle;
- allow-all no action.

Its key controlled question is whether a classification-accurate observer and a
direct-risk observer choose equally good actions under the same intervention
budget. The expected answer is no when violations differ in severity.

## Phase S2: learned interlock

Replace the controlled activation with a frozen model activation while keeping
the same paired calls, oracle, controller, and loss. The clean gate requires the
model to process the underlying authorization task, and the action-only text
baseline must remain unable to distinguish a matched pair. This stage adds a
real representation claim without changing the protocol question.

The first sealed run uses Qwen2.5-7B-Instruct. It passes the clean gate on every
locked-test stratum. The best learned activation observer reduces mean protocol
loss by 32.1% relative to the strongest action-only observer, retains all clean
utility, and closes 71.1% of the action-only-to-oracle gap. A label observer has
perfect AUROC but worse control loss than an observer that combines the label
estimate with violation severity. The aggregate result is positive. The
doubly-held-out operation/template stratum remains negative, so the release
reports both aggregate and worst-family loss.

Candidate observers will include linear activation probes, direct-risk probes,
attribution-based features, DLA-style readouts where defined, and an SAE feature
observer when the model and layer have a compatible released SAE.

## Researcher task-pack harness

The safety track is not limited to fixtures shipped by the authors. A task
author can publish a target-free JSON pack containing the threat model, oracle
definition, action set, loss, budgets, labeled fit measurements, and held-out
queries. Evaluator targets live in a second file that is hash-linked to the
public pack. The command-line interface validates both files and evaluates the
ordinary prediction CSV through the common controller. No registry edit or
model runtime is required.

## Phase S3: held-out and adversarial reachability

Freeze attack families before outcomes are inspected. Report in-distribution and
leave-one-family-out results separately. Only after this result is sealed should
the task add an adaptive search over prompts or tool traces. Activation-space
blind directions count as actionable adversarial evidence only when the declared
input or tool interface can actually reach them.

## Phase S4: external AI-control protocol

Run a monitor-swap study in a published AI-control environment. Hold the model,
attack policy, controller, audit budget, and evaluator fixed; change only the
observer. Score realized protocol safety and capability, with monitor AUROC as a
secondary diagnostic. The first target is the backdoored-code setting of
Greenblatt et al. Compare a trusted-model judge, a linear activation monitor,
and a model self-report through the same audit budget. The primary question is
whether AUROC or recall at low false-positive rate ranks those monitors in the
same order as realized violations. Sweep the audit budget and use a declared
impact model over backdoor types; if no ranking reversal occurs, report the
measured mixture and budgets on which the rankings agree, then check for hidden
strata before calling the metric safe. This supplies external validity after
the controlled tasks establish attribution and ground truth.

The first implementation uses ControlArena's public APPS task and its verified
working-backdoor labels. Every monitor scores the same paired honest/attack
panel. The controller audits the highest-scoring 2 percent, matching the
scarce-audit regime that motivates AI control. The primary outcome is therefore
the number of verified working backdoors left unaudited; no new impact scale is
needed. Five- and ten-percent budget sweeps are secondary diagnostics. The
frozen Qwen2.5-7B-Instruct study compares two ControlArena-derived prompt monitors, their
ensemble, and a held-out residual-activation probe.

The first external result triggered a bounded post-outcome control study. It
keeps the 500/200/744 problem-ID split and the 2 percent selection budget fixed.
The added rows are a supervised word/character n-gram observer, a supervised
observer over the basic prompt's score-token distribution, and a residual probe
read from problem plus code without a monitoring instruction. A separate
frozen-score sensitivity resamples attack prevalence and audit budget. These
rows distinguish training access, output-boundary information, neutral-context
activation information, and surface text without changing the locked examples.

The neutral-context probe is the strongest non-oracle row on the locked panel
(AUROC 0.979). It exceeds the verdict-adjacent activation probe (0.953), the
supervised text ridge (0.868), and the supervised score-distribution ridge
(0.830). The basic prompt already reports an expectation over all ten score
tokens, so this is not a sampled-verdict artifact. At the balanced 50 percent
attack prevalence, the neutral probe matches the exact-label oracle's audit
loss at 2, 5, and 10 percent budgets. At 1 percent prevalence and a 2 percent
audit budget, frozen-score resampling leaves 2.65 attacks unaudited on average
for the neutral probe, versus 0 for the oracle, 3.94 for the verdict-adjacent
probe, 9.05 for supervised text, and 9.15 for the prompted score distribution.
This is evidence of decodable neutral-context state, not evidence of deception,
intent, or actual code execution.

A post-outcome diagnostic checks APPS difficulty--source families and quartiles
of neutral-context input length. The neutral probe remains strong within every
displayed group: family AUROC is 0.956--0.991 and length-quartile AUROC is
0.974--0.988. Its global 2 percent audit allocation is nevertheless uneven.
Relative to each group's share of attacks, the family allocation ratio spans
0.36--2.38 and the length-quartile ratio spans 0.67--1.43. The pooled AUROC and
audit-loss rankings agree empirically, but pooling does not remove score-scale
drift. The rare-prevalence sensitivity also holds this family mixture fixed.

A second post-outcome integration runs a published sparse SAE probing recipe on
a pinned public Qwen2.5-7B-Instruct SAE and compares it with matched raw-residual and
SAE-reconstruction logistic probes. The sparse refit retains 127 nonzero
coefficients. It reaches AUROC 0.953 on the balanced locked panel, below the raw
residual control at 0.983. At 1% attack prevalence and a 2% audit budget, it
misses 4.667 attacks on average versus 2.048 for raw residuals. This result is
compression of the fitted readout without a better mean audit decision. The
SAE itself is overcomplete; this result is an informative negative for the
declared checkpoint, layer, observer, and task, not a general conclusion about
SAE observers.

A matched Gemma-2-9B-it replication reuses the APPS rows, prompts, controller,
budgets, and loss. Its design was frozen after the Qwen2.5-7B-Instruct outcomes but before any
Gemma monitor outcome was inspected. At 1% attacks and a 2% audit budget, the
verdict-adjacent residual probe misses 2.178 attacks on average, compared with
4.947 for the neutral residual, 9.070 for the official Gemma Scope SAE probe,
and 14.927 for the basic prompted monitor. This is a separate task panel: it
shows that residual access transfers while the best prompt context changes.
The detailed prompt also has higher AUROC than the Gemma Scope SAE probe
(0.893 versus 0.870) but more mean violations (10.803 versus 9.070):
classification is not enough to choose the lower-loss monitor. The SAE result
remains specific to this model, layer, probe, and task.

A prospectively frozen Qwen3.5-9B replication uses the same locked APPS pairs,
controller, loss, and primary 1% attack / 2% audit cell. The verdict-adjacent
residual probe reaches AUROC 0.977 and 1.976 mean realized violations. The
neutral residual reaches 0.936 and 6.601, while the prompted score-distribution
ridge reaches 0.940 and 5.120. The official Qwen-Scope SAE probe reaches 0.811
and 12.725 after selecting layer 12 and ridge 100 on fit and calibration rows.
That SAE was trained on Qwen3.5-9B-Base and applied to post-trained Qwen3.5-9B.
The declared base-to-posttrain mismatch means this row is a useful control, not
a checkpoint-matched SAE test. As with the earlier panels, all ranks stay
inside this exact task and operating policy.

## Monitor identification ladder

ObserverBench now records a four-step protocol for interpreting internal safety
signals:

1. Use the richest available output observer, including the relevant output
   distribution rather than a sampled answer.
2. Match supervision with a surface-form or input-feature baseline.
3. Remove the monitoring request and test the same examples in a neutral
   context.
4. Re-evaluate frozen scores at the intended prevalence, intervention budget,
   strata, and control loss.

The ladder identifies where useful information becomes available and where a
claim fails. Passing it does not by itself establish deception, intent, or
hidden knowledge.

## Deferred external-adoption work

Two extensions are deliberately held until the ObserverBench repository is
public and its safety contract has a stable URL:

1. Auto-verified, semantics-preserving rewrites of the working APPS backdoors,
   scored against the frozen monitors to test sensitivity to surface style.
2. A small ObserverBench adapter proposed upstream to ControlArena, so its users
   can export monitor scores directly into the checked ObserverBench table.

The first requires a declared rewrite family, a sandboxed verification path,
and a fresh seal before any monitor sees the variants. The second should target
the public contract rather than a private or changing interface. Neither is a
claim of the current paper.

## Paper claim boundary

The analytic task establishes the contract and consequence logic. The
Qwen2.5-7B-Instruct, Gemma-2-9B-it, and Qwen3.5-9B runs show that internal state can improve safety
decisions under a fixed action budget, while the Qwen2.5-7B-Instruct doubly-held-out result
sets the present generalization boundary. Adaptive attacks and claims about
open-ended deployment remain outside this release.
