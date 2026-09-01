# Qwen2.5-7B-Instruct paired-scope-v1 submission panel

This panel seeds the automatic public task with three rows: the exact evaluator
oracle, a constant-risk observer passed through the fixed triage controller,
and the allow-all/no-action reference. The first two were recomputed on the
blinded v1 task. The no-action row is shown separately because it does not spend
the controller's block or escalation budget.

Model-output, text, and activation observers from paired-scope-v0 are not copied
here. Rekeying prompt entities can change their predictions, so they require a
fresh v1 run. Automatic data-only submissions will appear as `community
evaluated` rows. Reference rows remain unranked.
