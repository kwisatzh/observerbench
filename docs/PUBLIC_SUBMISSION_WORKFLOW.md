# Local submission and offline evaluation

Nothing runs when an issue is opened, reopened, edited, or labelled. There is no
hosted scoring service. CI and Pages builds are maintainer-dispatched only.

## Submitters

1. Run `python scripts/authorization_starter.py --output my-observer` on CPU,
   or prepare your own predictions using the public task pack.
2. Run `python scripts/score_authorization_validation.py my-observer/validation_predictions.csv`.
   Report the entire validation scorecard and how hyperparameters were chosen.
   Public validation scores are not test leaderboard scores.
3. Attach a ZIP containing only `predictions.csv` (test IDs) and
   `observer_card.json` to an issue, or link immutable copies. Include the task
   version and validation scorecard. The maintainer computes hashes on receipt.
4. The maintainer checks inert data locally, then scores test predictions offline
   in monthly batches, scored within 30 days. Revisions count as new submissions.
   Limit: at most three scored submissions per person and task per rolling
   30 days; the maintainer checks the local log before scoring.
   No per-example feedback is returned.

APPS submissions and the open-replay export are on hold pending comment-artifact
review. Existing historical panels remain visible. Do not submit APPS jobs yet.

## Maintainer: trusted checkout, no submitted code

Download the two files as ordinary files; inspect archive paths before extraction.
Do not install or run a submitter's code. Record the issue, submitter, date,
task, and input hashes in a local log; check the submission limit first.
Compute SHA-256 hashes from the received files (for example,
`shasum -a 256 predictions.csv observer_card.json`); submitters need not supply them.
Keep protected targets, that log, and scoring outputs outside this checkout.
The script rejects symlinks, oversized inputs, wrong schemas, duplicate or missing
IDs, mismatched hashes, and targets not bound to the public task.

```sh
python scripts/score_submission_offline.py \
  --predictions /private/submission/predictions.csv --predictions-sha256 PREDICTION_SHA256 \
  --card /private/submission/observer_card.json --card-sha256 CARD_SHA256 \
  --task task-packs/safety/safety-interlock-qwen2-5-7b-instruct/paired-scope-v1/safety_task.json --task-sha256 TASK_SHA256 \
  --targets /private/evaluation/targets.json --targets-sha256 APPROVED_TARGET_SHA256 \
  --output /private/evaluation/draft-row.json
```

Use hashes from reviewed intake and the approved target manifest, not unchecked
submission values. The target's internal public-task hash is also verified.
The output is an aggregate draft, never an automatic leaderboard edit. Review
access claims, validation report, limits, and draft before approving an addendum.
No labels, private paths, or per-query outcomes are copied to the draft.

## Preflight options

**Recommended and implemented: local only.** The offline script composes the
existing validator and scorer. Standalone format checking also works:
`python scripts/validate_public_submission.py --root submissions`, using the
layout `safety/TASK/SUBMISSION/{predictions.csv,observer_card.json}`.

**Alternative, not implemented:** a workflow could require `run-preflight` plus
an explicit maintainer-permission check, trusted validator checkout, read-only
contents permission, no secrets, and no submitted code. It still allocates an
Actions runner for the label event and adds a billing and trust boundary.
Local-only is simpler and matches the policy.

There are no evaluator dispatch tokens or hosted-evaluator activation steps.
