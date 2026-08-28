# GitHub-native public evaluation

The public repository and GitHub Pages site are the visible half of
ObserverBench. A separate private evaluator holds sealed targets. This keeps
routine preflight automatic without giving untrusted code access to labels or
repository credentials.

## Public-release status

The public repository automatically performs data-only preflight. The sealed
evaluator is a separate deployment and is inactive at public launch. It can be
enabled after the repository variable `SEALED_EVALUATOR_ENABLED` is set to
`true` and the evaluator settings below are present. With that switch off, a
passing submission receives a preflight comment and artifact only. It has not
been scored and is not eligible for a leaderboard rank.

## Routine path

1. A researcher opens the structured Observer evaluation job issue, links a
   prediction table at an immutable GitHub commit, and pastes an ObserverCard.
2. `issue-submission.yml` automatically downloads and checks the table as
   untrusted data. It executes no contributor code.
3. If the optional sealed evaluator is enabled, the checked request and its
   SHA-256 digest are sent to the private evaluator automatically.
4. The private evaluator reads only the allowed data files, joins prediction IDs
   to sealed targets, applies the frozen controller and budget, and emits a
   redacted scorecard.
5. An approved result JSON updates the matching task leaderboard. GitHub Pages
   rebuilds from checked result JSON. The initial public release does not
   automate this publication step.

The issue is a job record, not a request for line-by-line human review. A pull
request remains available as a bulk or development path, but routine jobs do
not need one.

## Trust boundary

The public CI workflow may validate untrusted files but receives no target or
publication secret. The private evaluator uses trusted code from its own default
branch and never installs, imports, or executes content supplied by the
researcher. It should fetch the prediction and card blobs by the exact commit
SHA recorded by preflight, enforce the same size and schema limits again, and
publish only aggregate or otherwise approved outputs.

Configure these public-repository settings when the evaluator repository is
ready:

- repository variable `SEALED_EVALUATOR_ENABLED=true`;
- repository variable `SEALED_EVALUATOR_REPOSITORY=<owner>/<private-repo>`;
- repository secret `SEALED_EVALUATOR_TOKEN` with permission only to dispatch
  the private evaluator workflow.

Keep evaluator targets, per-query labels, and private result rows out of the
public repository and out of public workflow artifacts.

## Human review

Data-only preflight needs no manual approval. Once sealed scoring is enabled,
add a moderation or verification step only for new scientific contracts,
executable submissions, abuse flags, disputes, or an `implementation verified`
badge. Rate limits and a provisional `community evaluated` status can protect
the sealed test set without creating a review queue.

## Public task-pack integrity

The first preflight-enabled task is the Qwen safety `paired-scope-v1` pack.
Its public query IDs and synthetic account/resource names are re-keyed; the
source mapping and targets stay with the evaluator. This prevents the direct
label leakage in the checked v0 identifiers.

The existing APPS reproduction table includes `eval_mode`, `backdoor_works`,
and `family_id`. Public submissions omit all three, and the evaluator joins them
privately. The exact question/code examples remain matchable to the public
labelled upstream dataset, however, so this is an open adoption panel rather
than a sealed generalization test. Its checked panel remains visible, but
routine APPS jobs stay closed until the normalized public query export is
available. The result page must preserve the open-panel label.
