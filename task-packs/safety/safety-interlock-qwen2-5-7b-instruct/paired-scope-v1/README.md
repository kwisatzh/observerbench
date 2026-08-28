# Qwen authorization triage: paired-scope-v1

This is the public, target-free ObserverBench task pack.

- `safety_task.json` is the complete inference-free task contract. It contains
  1,152 labeled fit/calibration measurements, 768 held-out queries, the fixed
  controller, and the task card. It contains no held-out targets.
- `queries.jsonl` is a lighter query-only view for runners that do not use the
  Python API.
- `task.json` records the public schema, prompt format, and file hash.

Identifiers and synthetic account/resource names are re-keyed for this release.
Submit one finite `predicted_risk` for every held-out `query_id`. The matching
targets remain evaluator-only; the evaluator reports aggregate prediction and
action metrics.

The authorization rule itself is not secret: an exact symbolic checker remains
the legitimate oracle reference. The seal prevents direct lookup from the
checked v0 files and keeps held-out consequence rows out of submissions.
