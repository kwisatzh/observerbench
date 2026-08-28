# Public observer submissions

ObserverBench v1 accepts prediction tables, not executable code. The recommended
path is the repository's **Observer evaluation job** issue form. It asks for an
immutable GitHub URL to the prediction table and the ObserverCard JSON. Public
automation validates the request without a pull request or maintainer review.

The public launch enables this preflight step, but not sealed scoring. Scoring
runs only after a separate trusted evaluator has been configured and the public
repository reports that it is enabled. Before that point, a passing issue
receives a preflight result only; it does not receive a benchmark score or
leaderboard rank.

For bulk or development use, the equivalent repository bundle has this shape:

```text
submissions/pending/<track>/<task-id>/<submission-id>/
  predictions.csv
  observer_card.json
```

The valid tracks are `safety`, `effect`, and `ai-control`. One pull request may
contain one submission. The automatic preflight rejects extra files, symbolic
links, unexpected columns, invalid schemas, duplicate IDs, non-finite values,
oversized files, and incomplete ObserverCards.

When sealed scoring is enabled, a separately trusted evaluator joins the
predictions to evaluator-held targets, applies the fixed action rule, and
produces a task-specific scorecard. Contributor code is never checked out or
executed by that evaluator. Results are compared only with rows using the same
task version, observation boundary, action rule, and budget.

Templates are under `submissions/templates/`. Copy the two files for the chosen
track and replace the example IDs and values. Do not add
held-out labels, cached model activations, model weights, credentials, or raw
private data.

Issue-form preflight is automatic. After sealed scoring is enabled, human review
is reserved for:

- new tasks, losses, controllers, or target definitions;
- requests to execute or verify an implementation;
- submissions stopped by abuse, integrity, or metadata checks.

The initial release may display a result as `community evaluated` before it is
eligible for an `implementation verified` badge. Oracle and reference rows are
shown for context but are never assigned a public rank.
