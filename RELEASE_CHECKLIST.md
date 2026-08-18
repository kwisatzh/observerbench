# Release Checklist

## Hygiene Completed

- Removed local Python bytecode and test caches.
- Removed local Matplotlib cache.
- Removed local virtual environment.
- Removed local smoke run outputs.
- Removed private migration notes.
- Removed the untracked local paper PDF artifact.
- Removed generated `egg-info` packaging state.
- Confirmed no notebooks, bytecode files, or macOS metadata files remain in the working tree.
- Added/updated `.gitignore` for local envs, caches, run outputs, env files, logs, and local paper PDFs.
- Added `LICENSE`.
- Added `CITATION.cff`.

## Tests Run

- `pytest -q`
  - Result: `20 passed, 4 skipped`.
  - The skipped tests are marked slow and are skipped by default.
- `python scripts/reproduce_paper_fast.py`
  - Result: passed.
  - Regenerated all paper figures and paper tables from frozen outputs.

## Grep Checks Run

The required release greps from the prompt were run after local cleanup.

- Local absolute user path check: no matches.
- Author email prefix check:
  - Source tree excluding Git internals: only `CITATION.cff`.
  - Raw recursive grep also reports Git reflog author metadata under `.git/logs`; that is repository metadata, not release file content.
- Hugging Face home/cache environment variable checks: no matches.
- Lowercase model-hub vendor-name check over configs, frozen results, scripts, source, and README: no matches.
- Experiment-tracker name check: no matches.
- API key string check: no matches.
- Generic credential-word check:
  - Reviewed manually.
  - Remaining source-tree hits are code variables for synthetic sequence inputs, one frozen sequence-length metadata field, and migration-plan prose.
  - No credential, access secret, or model-download credential was found.

## Known Expensive Tests Not Run

- `pytest -q --runslow`
- `python scripts/reproduce_paper_full.py --yes-run-expensive`
- Full trained Ctl-1 paper-scale reruns.
- Full trained Ctl-2 paper-scale reruns and sweeps.
- Full IOI GPT-2/TransformerLens reruns.

These are intentionally outside the default release/CI path.

## Required Manual Checks Before Release

- Confirm final bibliographic metadata and author names in `CITATION.cff`.
- Confirm the selected license is intended for public release.
- Confirm the public remote URL, if any, before adding it to citation metadata.
- Review generated paper figures visually after any frozen-output refresh.
- Confirm frozen outputs are approved for public distribution.
- Confirm no additional private notes, local PDFs, model caches, or raw model weights are staged.
- Confirm the release commit hash is recorded alongside the paper version.
