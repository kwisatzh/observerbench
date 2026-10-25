# Public observer submissions — local-only policy, 24 October 2026

Self-score public authorization validation data first. Attach test predictions
and an ObserverCard to an issue, with the validation scorecard and file hashes.
Opening an issue or PR runs nothing. The maintainer validates and scores inert
data offline in batches, then approves any public aggregate row separately.
There is no hosted scoring service, planned or active. APPS submissions are held.

See [the complete local workflow](../docs/PUBLIC_SUBMISSION_WORKFLOW.md).
For local format validation, use this directory layout:

```text
submissions/pending/safety/TASK/SUBMISSION/
  predictions.csv
  observer_card.json
```

Run `python scripts/validate_public_submission.py --root submissions/pending`.
This checks format only; the offline maintainer scorer also checks hashes and
exact task coverage before opening protected targets. Never include hidden
labels, activations, model weights, credentials, or executable code. Existing
templates show the data schemas. Historical records describing automatic
scoring are superseded by this dated policy; their hashes remain unchanged.
