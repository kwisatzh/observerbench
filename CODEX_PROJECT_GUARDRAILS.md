Project guardrails for ObserverBench:

This repo is a reproduction and diagnostic workbench for the ObserverBench paper. It is not a general benchmark platform yet.

## Reviewer-driven Phase 0--1 revision

The `observerbench-review-phase01` branch is authorized to repair the paper's
existing Ctl-1/Ctl-2 experiments and to expose the minimal public contracts
needed to run an outside observer on an existing task.  This is a bounded
scientific revision, not permission to add unrelated tasks or broaden the
paper's scope.

The frozen v7 results under `results/frozen/` are provenance and must never be
overwritten.  Revised outputs belong under a versioned revision directory and
must record the source commit, configuration, and result schema.

Do not:
- add experiments outside the reviewer-requested Ctl-1/Ctl-2 repairs,
- add new tasks beyond the current paper,
- change scientific claims without a corresponding checked result,
- invent missing results,
- broaden the scope into a hosted demo, product, general plugin platform, or InterpBench sweep,
- relabel weak/null results as positive wins.

Allowed in Phase 0--1:
- factor Ctl-2 into independent estimator and direction components,
- add factorial, calibration, gain, clipping, and residual-support diagnostics,
- add a manifold-respecting analytic Ctl-1 check,
- add a small, documented observer/task contract by composing current core
  types and registry behavior,
- add tests and versioned frozen outputs for those changes.

## Reviewer-driven Phase 2--3 revision

The subsequent review authorizes a bounded IOI re-analysis and artifact
cleanup. Phase 2 may reuse the saved prompt-level Stage 2b/2c effects to compare
equal-capacity pair bases, add-one and leave-one-out contrasts, prompt-bootstrap
direct effects, and measurement-design leverage. It must not turn into a new
model, circuit, or behavioral benchmark sweep. Phase 3 may update the
manuscript, checked reproduction map, CLI renderer, package metadata, and the
bounded v0 contract. The rejected cyber fixture belongs in an appendix and is
not a benchmark task.

The repository remains private. Do not add a repository URL, publish a release,
or run the deferred cyber redesign or weak-ground-truth observer-control study
as part of this revision.

Definition of done for v0:
- reproduce the paper figures/tables from frozen outputs,
- run quick smoke tests on CPU,
- generate ObserverCards from metrics,
- expose the current tasks through a stable-enough internal CLI,
- document exact command-to-figure mapping,
- pass tests that pin the scientific claims.

Stop when the repo matches the paper. Do not expand the paper.
