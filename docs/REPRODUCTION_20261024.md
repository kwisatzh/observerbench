# Reproducing existing leaderboard rows

The [per-row index](REPRODUCTION_RECIPES_20261024.json) lists every one of the
81 existing rows across 15 panels, with its exact expected metrics, input
records and seals, configuration, and CPU command. It does not change the rows.

Distinguish three operations:

- **Reconstructing a leaderboard from saved metrics:** the effect, APPS, and
  original authorization builders do this. It checks faithful packaging, not
  fresh scientific replication. Use a disposable checkout, set `PYTHONPATH=src`,
  and install the project's CPU dependencies before running the listed command.
- **Replaying a decision from saved measurements:** IOI's per-row commands run
  the fixed action rule on the public response matrix. No new model inference.
  The exact no-op row has loss equal to the target by construction.
- **Recomputing private authorization scores:** only the maintainer can do this
  with protected targets. Use `score_submission_offline.py` with reviewed hashes
  as described in PUBLIC_SUBMISSION_WORKFLOW.md. The v1 constant-risk row uses
  zero scores; the oracle is an evaluator-only bound and its predictions must
  never be shared; allow-all uses zero action budgets and is not the fixed-policy
  constant-risk row. Public users cannot reproduce hidden-target scores alone.

Some historical rows point to acquisition-time paths such as `/content/...`.
Use the file with the matching recorded SHA-256 in the retained result archive,
not a newly generated substitute. The index preserves these historical paths
instead of pretending all raw acquisition data are publicly bundled. Missing
raw/model caches block a fresh fit, not reconstruction from checked summaries.
New GPU acquisition is not part of these instructions or this patch's checks.

For APPS, read the current submission hold before proposing any replay export.
Original panels remain historical records; post-panel controls will be separate
dated addenda if approved. The local keyword diagnostic is not a published row.
