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

- A clean archive of the release commit: `301 passed, 11 skipped`.
  The skipped checks require expensive model runs or raw experiment bundles
  that are archived separately; checked public artifacts remain covered.
- Copy-v2 producer, registry, and CLI checks: `47 passed`.
- Public submission, issue-form, and Pages checks: `12 passed`.
- Static-site build: passed, including the private-file stop test.
- Paper build: passed (`latexmk`, 25 pages), followed by rendered-page review.

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

## Manual Checks Completed

- Confirmed bibliographic metadata and author name in `CITATION.cff`.
- Confirmed the MIT license and public remote URL.
- Reviewed the final paper render after the release edits.
- Confirmed that frozen public outputs are approved for distribution.
- Confirmed that evaluator targets, keys, private mappings, model caches, and
  model weights are absent from the tracked tree and history.
- Recorded the release in Git and Zenodo.
