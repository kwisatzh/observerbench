# Monitoring extensions: September 4, 2026

These additions ask what happens when a monitor's evidence changes, rather
than treating a fixed score table as the whole evaluation problem.

## Completed in this round

- [x] Add a CPU task with potentially false reports, a limited trusted
  observation budget, and a fixed allow/block/escalate policy.
- [x] Add a scripted reader-to-relay origin variant. It shares the first
  task's generator; it is not a second independent empirical replication.
- [x] Let users supply both measurement selection and risk prediction.
- [x] Run the full integrity grid and retain the simple missing-receipt baseline.
- [x] Run an exploratory Qwen2.5-to-Qwen3.5 output-observer reuse diagnostic,
  with all target-only fitting budgets and seeds reported. Different
  architectures make failed reuse unsurprising; no source-assisted
  adaptation was tested.
- [x] Keep per-example APPS caches and fitted transfer parameters private.

## Next experiments, not findings of this release

1. **Sequential evidence acquisition.** Let an observer request a second
   trusted observation after seeing the first. Compare with the one-round
   policy at equal observation and action budgets. Measure when the second
   observation changes the decision, not just classification accuracy.
2. **Monitor-aware evasion.** In an inert authorization fixture, let a
   bounded adversary edit reports after seeing a limited number of monitor
   responses. It cannot change broker facts, labels, or evaluator code.
   Freeze the adversary's query allowance and evaluate on new episodes.
3. **Live relay histories.** Replace scripted actor histories with sandboxed
   agents. Retain a trusted event record and an exact authorization rule.
   Score complete episodes, since messages and shared state make requests
   dependent. Do not execute untrusted APPS code as part of this extension.
4. **Measurement reuse after model changes (MT).** Compare source-assisted
   adaptation, target-only fitting, and alternative measurement views at
   equal target-measurement budgets. Include a same-architecture update,
   then multiple transitions and tasks. Keep the update sequence separate
   from test outcomes. This substantial experiment remains to be done.
   It can inform successor-model evaluation; it does not simulate RSI.

## Paper boundary

The misleading-evidence task and scripted relay variant belong to monitoring
integrity. They remain in the longer paper and an ICLR appendix. The Qwen
reuse diagnostic stays in the workbench, outside both papers. Its learning
curve must not be described as source-assisted adaptation or as a new MT result.
Neither paper should claim that these studies reproduce the Hugging Face
incident, establish security against adaptive agents, or validate an
autonomous self-improvement process.
