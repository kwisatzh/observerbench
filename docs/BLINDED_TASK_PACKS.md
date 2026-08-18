# ObserverBench blinded submission-pack audit

This staging directory contains a target-free pack builder and focused tests. It
does not modify the working ObserverBench repository.

## What the audit found

### Qwen paired-scope safety

The checked design table cannot be published as a submission pack as written:

- `prompt_id` ends in `authorized` or `unauthorized`;
- `unsafe_if_allowed` and `benign_value` are the target;
- `pair_id`, `bank`, and `family_id` disclose grouping and split structure; and
- the synthetic account and resource names allow an exact join back to the
  checked table even if only `prompt_id` is replaced.

The builder publishes re-keyed fit and calibration rows as labeled measurements,
then publishes locked-test prompts as target-free queries. It assigns keyed
opaque IDs and re-keys every synthetic account and resource identifier inside
the prompt. The equality relation that defines the authorization request remains
unchanged. Locked-test source IDs and targets stay in the private pack; public
family and pair identifiers are also opaque.

This makes the file target-free and prevents a trivial table join. It does not
make the authorization label unknowable: the point of the task is to determine
whether the requested and granted resources match, so a correct symbolic
authorization checker remains a legitimate oracle baseline.

### ControlArena/APPS

The existing `ai_control_scores.csv` is unsafe for public submission: its IDs are
`<problem_id>::honest` or `<problem_id>::attack`, and it includes `eval_mode`,
`backdoor_works`, and `family_id` beside every score. The checked `design.json`
has the same direct leakage.

The normalized APPS builder removes those fields, assigns keyed opaque IDs, and
puts the exact private-target schema expected by `ai_control_public.py` in the
evaluator pack. However, two limitations remain:

1. The repository does not contain a normalized question/code query table. The
   question and selected honest/attack code are reconstructed at run time from
   `RoganInglis/apps-control-arena` at revision
   `29db28f76042952d56add8671f798280b200744d`. The builder therefore accepts an
   evaluator-generated JSONL export; the test fixture demonstrates its schema.
2. The upstream dataset is public and labelled. Even with opaque IDs, an exact
   question/code pair can be matched back to that source. The resulting pack is
   useful as a frictionless adoption and reproducibility panel, but it is not an
   independently sealed generalization test. A genuinely sealed APPS track needs
   unpublished verified examples or verified semantics-preserving refactors.

The [Hugging Face dataset card](https://huggingface.co/datasets/RoganInglis/apps-control-arena)
currently labels the upstream dataset MIT, so the blocker is not a known
redistribution prohibition. The blocker is the absence of the normalized source
table in the artifact and, more importantly, public label lookup for exact
examples.

## Pack boundary

Each build creates two sibling directories:

```text
output/
  public/
    task.json
    queries.jsonl
    safety_task.json
  private/
    manifest.json
    id_map.csv
    targets.csv
    evaluator_targets.json
```

Only `public/` belongs in GitHub Pages or the public repository. For the Qwen
track, `safety_task.json` is the existing portable ObserverBench safety contract:
it contains labeled fit/calibration measurements and held-out queries without
their consequences. `evaluator_targets.json` is accepted directly by
`load_safety_task_pack` and stays private. The HMAC key and the entire `private/`
directory belong in the private evaluator repository.
Safety submissions use the hosted safety contract:

```csv
schema_version,query_id,predicted_risk
observerbench.safety_predictions.v0,ob-...,0.0
```

APPS monitor submissions use the AI-control contract:

```csv
schema_version,sample_id,monitor_name,score
observerbench.ai_control_public_scores.v0,ob-...,my-monitor,0.0
```

The evaluator joins those scores to `private/targets.csv` in memory.

## Usage

Create a private random key once, outside the repository:

```sh
openssl rand 32 > evaluator.key
```

Build the Qwen pack from the checked design:

```sh
python scripts/build_blinded_pack.py qwen-safety \
  --source qwen_safety_prompts.csv \
  --secret-file evaluator.key \
  --outdir qwen-pack \
  --task-id safety-interlock-qwen2-5-7b-instruct@paired-scope-v1
```

Build an APPS open panel from evaluator-normalized JSONL:

```sh
python scripts/build_blinded_pack.py apps \
  --source apps_locked_rows.jsonl \
  --secret-file evaluator.key \
  --outdir apps-pack \
  --task-id controlarena-apps-qwen-v1
```

Run the focused tests:

```sh
python -m unittest discover -s tests -p 'test_*.py'
```

Run the compatibility test with ObserverBench and its dependencies available:

```sh
PYTHONPATH=/path/to/ObserverBench/src python -m unittest \
  tests.test_actual_safety_pack_contract
```
