# ObserverBench Reproduction Repo Migration Plan

## Scope

v0 is a reproduction artifact and diagnostic workbench for the ObserverBench paper. It is not a general benchmark platform, not a plugin system, and not a place to add new experiments or revise scientific claims. The migration should make the existing paper results reproducible, generate ObserverCards from the frozen and rerun metrics, and expose the minimal task interface needed by the current tasks.

The current source files are outside this repo, under:

`<LOCAL_SOURCE_ROOT>`

The latest ObserverBench package code is in:

`<LOCAL_SOURCE_ROOT>/observerbench_mvp_v7_ctl2`

The IOI code is currently split across:

- `<LOCAL_SOURCE_ROOT>/ioi_stage1_v0`
- `<LOCAL_SOURCE_ROOT>/ioi_stage2b_v0`
- `<LOCAL_SOURCE_ROOT>/ioi_stage2c_v0`
- `<LOCAL_SOURCE_ROOT>/ioi_stage2d_v0`

Concrete local path mappings belong only in `PRIVATE_MIGRATION_NOTES.md`, which must be gitignored.

## Guardrails Read

`CODEX_PROJECT_GUARDRAILS.md` says v0 should stop at paper reproduction. The repo should:

- reproduce figures and tables from frozen outputs,
- support CPU smoke tests,
- generate ObserverCards from metrics,
- expose current tasks through a stable-enough internal CLI,
- include exact command-to-figure mapping,
- pin scientific claims with tests.

The repo should not:

- add new tasks or experiments,
- broaden into a benchmark platform,
- alter or relabel paper claims,
- invent missing results,
- migrate stale future-facing placeholders as public API.

## Current Code Tree Audit

The source tree is mostly a sequence of MVP snapshots and generated run directories. The public repo should migrate only the latest source plus selected frozen outputs.

Keep as source:

- `observerbench_mvp_v7_ctl2/observerbench/core.py`
- `observerbench_mvp_v7_ctl2/observerbench/metrics.py`
- `observerbench_mvp_v7_ctl2/observerbench/observers.py`
- `observerbench_mvp_v7_ctl2/observerbench/cards.py`
- `observerbench_mvp_v7_ctl2/observerbench/tasks/collateral_interaction.py`
- `observerbench_mvp_v7_ctl2/observerbench/tasks/trained_transformer_ctl1.py`
- `observerbench_mvp_v7_ctl2/observerbench/tasks/trained_transformer_ctl2.py`
- the current runner scripts, rewritten later as thin CLI entry points.

Keep as IOI source:

- `ioi_stage1_v0/scripts/ioi_stage1_self_repair.py`
- `ioi_stage2b_v0/scripts/ioi_stage2b_head_subset_prediction.py`
- `ioi_stage2c_v0/scripts/ioi_stage2c_primary_stratified.py`
- `ioi_stage2d_v0/scripts/ioi_stage2d_per_pair_decomposition.py`

Do not migrate into v0 public API:

- older `observerbench_mvp*` source snapshots except as provenance references,
- `.zip` archives,
- `.venv`, `__pycache__`, `.DS_Store`,
- exploratory or stale runs not tied to paper figures/tables,
- `observerbench/tasks/interpbench_adapter.py`, because v0 is not a general adapter/plugin platform,
- Tracr, belief-tomography, or Claim 3 experiments unless they are explicitly needed for a paper figure/table in this reproduction artifact.

## Main Duplications And Fragile Paths

Duplicated code to consolidate:

- IOI Stage 1, 2b, and 2c duplicate model loading, device choice, head parsing, prompt construction, token helpers, mean-ablation cache construction, ablation hooks, logit-difference scoring, batching, bootstrap utilities, and plotting.
- IOI Stage 2b and 2c duplicate subset-design logic, ridge/k-fold evaluation, coefficient export, and prediction plots.
- IOI Stage 2d duplicates ridge/k-fold/bootstrap helpers already present in Stage 2b/2c postprocessing.
- Ctl-1 and Ctl-2 runner scripts duplicate output-directory creation, JSON/CSV writes, plot style, sweep summarization, and card generation hooks.
- Several scripts manually inject `sys.path` with `Path(__file__).resolve().parents[1]`, which will break when moved or installed differently.
- `run_trained_transformer_ctl1.py` exits with `os._exit(0)`, which is brittle and should not survive migration.
- README text in the v7 source tree still says "MVP v2" and contains stale version-history notes.
- `pyproject.toml` and `requirements.txt` disagree about torch; IOI also needs TransformerLens.

Migration principle: compose existing `core.*`, metrics, observers, and task functions into a small CLI. Do not introduce new abstractions unless they remove real duplicated behavior.

## Proposed Package Layout

```text
src/observerbench/
  __init__.py
  core.py
  metrics.py
  observers.py
  cards.py
  io.py
  cli.py
  frozen.py
  plotting/
    __init__.py
    common.py
    ctl1.py
    ctl2.py
    ioi.py
  tasks/
    __init__.py
    ctl1_analytic.py
    ctl1_trained.py
    ctl2_trained.py
    ioi/
      __init__.py
      heads.py
      prompts.py
      ablation.py
      scoring.py
      subsets.py
      regression.py
      stage1.py
      stage2b.py
      stage2c.py
      stage2d.py
scripts/
  observerbench-legacy/
    run_collateral_sweeps.py
    run_trained_transformer_sweeps.py
    run_trained_transformer_ctl2.py
    run_trained_transformer_ctl2_sweeps.py
    ioi_stage1_self_repair.py
    ioi_stage2b_head_subset_prediction.py
    ioi_stage2c_primary_stratified.py
    ioi_stage2d_per_pair_decomposition.py
artifacts/
  frozen/
  generated/
tests/
  fixtures/
  test_core_cards.py
  test_metrics.py
  test_frozen_manifest.py
  test_ctl1_analytic_smoke.py
  test_paper_claims_from_frozen.py
  test_ioi_stage2d_postprocess_fixture.py
```

The public command surface should be small:

- `observerbench reproduce --frozen artifacts/frozen --out artifacts/generated --cards`
- `observerbench figures <figure-id|all> --frozen artifacts/frozen --out artifacts/generated`
- `observerbench tables <table-id|all> --frozen artifacts/frozen --out artifacts/generated`
- `observerbench cards --frozen artifacts/frozen --out artifacts/generated/cards`
- `observerbench validate --frozen artifacts/frozen`
- `observerbench run <task> --smoke ...`
- `observerbench run <task> --full ...`

The CLI should call task modules directly and preserve the legacy scripts as wrappers only during transition.

## Current Task Structure

ObserverBench v0 exposes an internal task interface for the paper tasks only. It does not promise a stable plugin API, and this migration should not include "adding a new task" documentation beyond explaining how the current paper tasks are wired.

Each migrated task should compose existing primitives and expose only:

- a frozen-output reader,
- a fast figure/table/card generator,
- a CPU-safe smoke run when feasible,
- an opt-in full rerun command,
- manifest metadata for frozen files and dependencies.

## Frozen Output Layout

Planned frozen-output paths in the repo:

```text
artifacts/frozen/
  ctl1_analytic/collateral_sweeps_v2/
  ctl1_trained/trained_transformer_sweeps_v5_clean/
  ctl2/default_v7_clean/
  ctl2/sweeps_v7/
  ioi/stage1_both_end/
  ioi/stage2b_mean_end/
  ioi/stage2c_primary_stratified_mean_end/
  ioi/stage2d_per_pair/
```

Frozen outputs should include only the files needed to regenerate paper figures/tables, validate claims, and generate ObserverCards:

- summary CSVs,
- pairwise CSVs,
- metadata/config JSON,
- diagnostics JSON,
- reports used as provenance,
- figure PNGs if they are canonical paper outputs,
- generated ObserverCards.

Do not vendor model weights, `.venv`, caches, zip archives, or raw transient debug outputs.

## Frozen Output Manifest Requirement

`artifacts/frozen/manifest.yaml` is required for public release. It is the authoritative index from paper result to frozen provenance, generated artifacts, integrity hashes, and full-rerun dependency scope.

Every paper result in the command map below must have one manifest entry with:

- `id`: stable result id such as `fig1_ctl1_analytic_gamma`.
- `paper_result`: paper figure/table or named section result.
- `frozen_files`: repo-relative paths for every CSV, JSON, report, and canonical PNG needed for fast reproduction.
- `expected_generated`: repo-relative figure, table, and card outputs that fast reproduction should create.
- `legacy_command`: the source command used to create the frozen output.
- `full_rerun_command`: the public rerun command after migration.
- `sha256`: one hash per frozen file.
- `requires`: booleans for `torch`, `transformerlens`, `gpu_or_mps`, and `training`.
- `notes`: short provenance or caveat text, with no private local paths.

Example shape:

```yaml
results:
  - id: fig1_ctl1_analytic_gamma
    paper_result: Figure 1
    frozen_files:
      - path: artifacts/frozen/ctl1_analytic/collateral_sweeps_v2/gamma_sweep_summary.csv
        sha256: <sha256>
      - path: artifacts/frozen/ctl1_analytic/collateral_sweeps_v2/gamma_sweep_pairwise_summary.csv
        sha256: <sha256>
    expected_generated:
      figures:
        - artifacts/generated/figures/fig1_ctl1_analytic_gamma.png
      tables: []
      cards:
        - artifacts/generated/cards/ctl1_analytic_observer_cards.json
    legacy_command: "PYTHONPATH=. python scripts/run_collateral_sweeps.py --outdir runs/collateral_sweeps_v2 --seeds 0,1,2,3,4,5,6,7,8,9"
    full_rerun_command: "observerbench run ctl1-analytic --full --out runs/collateral_sweeps_v2 --seeds 0,1,2,3,4,5,6,7,8,9"
    requires:
      torch: false
      transformerlens: false
      gpu_or_mps: false
      training: false
    notes: "Analytic sweep; no model download or training."
```

## Paper Result To Command Map

The "frozen output path" column gives the planned repo path after migration. The current source path can be copied from the source roots listed above.

| Paper result | Paper figure/table | Existing script | Expected output files | Frozen output path | Smoke-test command | Full-rerun command | Approx. runtime | GPU/MPS/TransformerLens required |
|---|---|---|---|---|---|---|---|---|
| Analytic Ctl-1 null sanity | Sec. 4.1 null check | `observerbench_mvp_v7_ctl2/scripts/run_collateral_sweeps.py` | `null_gamma0_summary.csv`, `null_gamma0_target_vs_collateral.png`, `collateral_sweeps_configs.json` | `artifacts/frozen/ctl1_analytic/collateral_sweeps_v2/` | `PYTHONPATH=. python scripts/run_collateral_sweeps.py --outdir runs/smoke/collateral --seeds 0 --gamma-values 0,0.5 --nuisance-weights 0,1 --n-train 256 --n-test 256` | `PYTHONPATH=. python scripts/run_collateral_sweeps.py --outdir runs/collateral_sweeps_v2 --seeds 0,1,2,3,4,5,6,7,8,9` | smoke <5s; full <1m CPU | No GPU, no MPS, no TransformerLens |
| Analytic Ctl-1 interaction-strength sweep | Figure 1 | `observerbench_mvp_v7_ctl2/scripts/run_collateral_sweeps.py` | `gamma_sweep_summary.csv`, `gamma_sweep_pairwise_summary.csv`, `gamma_sweep_collateral_ratio.png`, `gamma_sweep_target_fraction.png`, `gamma_sweep_target_mse.png`, `collateral_sweeps_all_observers.csv`, `collateral_sweeps_pairwise.csv` | `artifacts/frozen/ctl1_analytic/collateral_sweeps_v2/` | same smoke as above | same full command as above with default gamma values `0,0.25,0.5,0.75,1.0,1.15,1.5` | smoke <5s; full <1m CPU | No GPU, no MPS, no TransformerLens |
| Analytic Ctl-1 nuisance-placement sweep | Figure 2 | `observerbench_mvp_v7_ctl2/scripts/run_collateral_sweeps.py` | `nuisance_weight_sweep_summary.csv`, `nuisance_weight_pairwise_summary.csv`, `nuisance_weight_collateral_ratio.png`, `nuisance_weight_effective_collateral_gain.png` | `artifacts/frozen/ctl1_analytic/collateral_sweeps_v2/` | same smoke as above | same full command as above with default nuisance weights `0,0.25,0.5,0.75,0.9,1.0` | smoke <5s; full <1m CPU | No GPU, no MPS, no TransformerLens |
| Trained-transformer Ctl-1 null sanity | Sec. 5.1 null check | `observerbench_mvp_v7_ctl2/scripts/run_trained_transformer_sweeps.py` | `null_gamma0_summary.csv`, `trained_transformer_sweeps_configs.json` | `artifacts/frozen/ctl1_trained/trained_transformer_sweeps_v5_clean/` | `PYTHONPATH=. python scripts/run_trained_transformer_sweeps.py --outdir runs/smoke/trained_ctl1 --seeds 0 --gamma-values 0,1.15 --nuisance-weights 0,1 --quick --device cpu` | `PYTHONPATH=. python scripts/run_trained_transformer_sweeps.py --outdir runs/trained_transformer_sweeps_v5_clean --seeds 0,1,2,3,4,5 --gamma-values 0,0.5,1.0,1.15,1.5 --nuisance-weights 0,0.25,0.5,0.75,0.9,1.0 --train-steps 600 --device auto` | smoke 1-5m CPU; full 1-3h MPS/GPU or longer CPU | Torch required; MPS/GPU recommended; no TransformerLens |
| Trained-transformer Ctl-1 interaction-strength sweep | Figure 3 | `observerbench_mvp_v7_ctl2/scripts/run_trained_transformer_sweeps.py` | `gamma_sweep_summary.csv`, `gamma_sweep_pairwise_summary.csv`, `trained_gamma_sweep_collateral_ratio.png`, `trained_gamma_sweep_target_fraction.png`, `trained_transformer_sweeps_all_observers.csv`, `trained_transformer_sweeps_pairwise.csv` | `artifacts/frozen/ctl1_trained/trained_transformer_sweeps_v5_clean/` | same trained Ctl-1 smoke as above | same trained Ctl-1 full command as above | smoke 1-5m CPU; full 1-3h MPS/GPU or longer CPU | Torch required; MPS/GPU recommended; no TransformerLens |
| Trained-transformer Ctl-1 nuisance-placement sweep | Figure 4 | `observerbench_mvp_v7_ctl2/scripts/run_trained_transformer_sweeps.py` | `nuisance_weight_sweep_summary.csv`, `nuisance_weight_pairwise_summary.csv`, `trained_nuisance_weight_collateral_ratio.png` | `artifacts/frozen/ctl1_trained/trained_transformer_sweeps_v5_clean/` | same trained Ctl-1 smoke as above | same trained Ctl-1 full command as above | smoke 1-5m CPU; full 1-3h MPS/GPU or longer CPU | Torch required; MPS/GPU recommended; no TransformerLens |
| Trained-transformer Ctl-2 closed-loop default trajectories | Figures 5 and 7 | `observerbench_mvp_v7_ctl2/scripts/run_trained_transformer_ctl2.py`; `observerbench_mvp_v7_ctl2/scripts/combine_trained_transformer_ctl2_runs.py` | per seed: `trained_transformer_ctl2_results.csv`, `trained_transformer_ctl2_per_step_examples.csv`, `ctl2_target_mse_trajectory.png`, `ctl2_collateral_trajectory.png`, `ctl2_observer_bias_trajectory.png`; combined: `trained_transformer_ctl2_all_runs.csv`, `trained_transformer_ctl2_summary.csv`, `trained_transformer_ctl2_pairwise_summary.csv`, combined fan plots | `artifacts/frozen/ctl2/default_v7_clean/` | `PYTHONPATH=. python scripts/run_trained_transformer_ctl2.py --outdir runs/smoke/ctl2 --seed 0 --quick --device cpu` | `PYTHONPATH=. python scripts/run_trained_transformer_ctl2.py --outdir runs/trained_transformer_ctl2_v7_clean/seed0 --seed 0 --device auto --train-steps 600 --loop-steps 15` then repeat seeds `0..5` and run `PYTHONPATH=. python scripts/combine_trained_transformer_ctl2_runs.py --inputs 'runs/trained_transformer_ctl2_v7_clean/seed*/trained_transformer_ctl2_results.csv' --outdir runs/trained_transformer_ctl2_v7_clean/combined` | smoke 1-5m CPU; full default seeds 30-90m MPS/GPU or longer CPU | Torch required; MPS/GPU recommended; no TransformerLens |
| Trained-transformer Ctl-2 interaction sweep and divergence-rate delta | Table 1 and Figure 6 | `observerbench_mvp_v7_ctl2/scripts/run_trained_transformer_ctl2_sweeps.py` | `gamma_sweep_pairwise_summary.csv`, `trained_transformer_ctl2_sweeps_all_observers.csv`, `trained_transformer_ctl2_sweeps_configs.json`, `trained_transformer_ctl2_sweeps_report.md`, `ctl2_gamma_ise_ratio.png`, `ctl2_gamma_ise_delta.png`, `ctl2_gamma_divergence_delta.png`, `ctl2_gamma_collateral_ratio.png`, `ctl2_gamma_collateral_delta.png` | `artifacts/frozen/ctl2/sweeps_v7/` | `PYTHONPATH=. python scripts/run_trained_transformer_ctl2_sweeps.py --outdir runs/smoke/ctl2_sweeps --seeds 0 --gamma-values 0,1.15 --nuisance-weights 0,1 --quick --device cpu` | `PYTHONPATH=. python scripts/run_trained_transformer_ctl2_sweeps.py --outdir runs/trained_transformer_ctl2_v7_sweeps --seeds 0,1,2,3,4,5 --gamma-values 0,0.5,1.0,1.15,1.5 --nuisance-weights 0,0.25,0.5,0.75,0.9,1.0 --train-steps 600 --loop-steps 15 --device auto` | smoke 2-8m CPU; full 2-5h MPS/GPU or longer CPU | Torch required; MPS/GPU recommended; no TransformerLens |
| Trained-transformer Ctl-2 nuisance-placement collateral sweep | Figure 7 support/caveat | `observerbench_mvp_v7_ctl2/scripts/run_trained_transformer_ctl2_sweeps.py` | `nuisance_weight_pairwise_summary.csv`, `ctl2_nuisance_collateral_ratio.png`, `trained_transformer_ctl2_sweeps_report.md` | `artifacts/frozen/ctl2/sweeps_v7/` | same Ctl-2 sweep smoke as above | same Ctl-2 sweep full command as above | smoke 2-8m CPU; full 2-5h MPS/GPU or longer CPU | Torch required; MPS/GPU recommended; no TransformerLens |
| IOI Stage 1 whole-group self-repair diagnostic | Figure 8 | `ioi_stage1_v0/scripts/ioi_stage1_self_repair.py` | `ioi_stage1_condition_results.csv`, `ioi_stage1_summary.csv`, `ioi_stage1_metadata.json`, `ioi_stage1_prompts.csv`, `ioi_stage1_bar.png` | `artifacts/frozen/ioi/stage1_both_end/` | `PYTHONPATH=. python scripts/ioi_stage1_self_repair.py --outdir runs/smoke/ioi_stage1 --device cpu --n-prompts 16 --n-reference 32 --batch-size 8 --ablation mean --positions end` | `PYTHONPATH=. python scripts/ioi_stage1_self_repair.py --outdir runs/ioi_stage1_both_end --device auto --n-prompts 256 --n-reference 512 --batch-size 32 --ablation both --positions end` | smoke 5-20m CPU after model cache; full 10-30m MPS/GPU after model cache | TransformerLens and GPT-2-small required; MPS/GPU recommended |
| IOI Stage 2b random head-subset prediction | Figure 9 | `ioi_stage2b_v0/scripts/ioi_stage2b_head_subset_prediction.py` | `ioi_stage2b_report.md`, `ioi_stage2b_fit_summary.csv`, `ioi_stage2b_kfold_predictions.csv`, `ioi_stage2b_subset_measurements.csv`, `ioi_stage2b_coefficients.csv`, `ioi_stage2b_bootstrap_summary.csv`, `ioi_stage2b_prediction_scatter.png`, `ioi_stage2b_mae_bar.png`, `ioi_stage2b_group_occupancy_errors.png`, metadata/diagnostics/per-prompt CSVs | `artifacts/frozen/ioi/stage2b_mean_end/` | `PYTHONPATH=. python scripts/ioi_stage2b_head_subset_prediction.py --outdir runs/smoke/ioi_stage2b --quick --device cpu` | `PYTHONPATH=. python scripts/ioi_stage2b_head_subset_prediction.py --outdir runs/ioi_stage2b_mean_end --device auto --n-prompts 256 --n-reference 512 --n-subsets 160 --batch-size 32 --ablation mean --positions end --k-folds 5 --bootstrap-repeats 200` | smoke 10-30m CPU after model cache; full 1-3h MPS/GPU | TransformerLens and GPT-2-small required; MPS/GPU recommended |
| IOI Stage 2c primary-stratified head-subset prediction | Figure 10 | `ioi_stage2c_v0/scripts/ioi_stage2c_primary_stratified.py` | `ioi_stage2c_report.md`, `ioi_stage2c_fit_summary.csv`, `ioi_stage2c_bootstrap_summary.csv`, `ioi_stage2c_paired_delta_mae.csv`, `ioi_stage2c_primary_count_errors.csv`, `ioi_stage2c_prediction_scatter.png`, `ioi_stage2c_bootstrap_mae.png`, `ioi_stage2c_paired_delta_mae.png`, `ioi_stage2c_primary_coverage_errors.png`, metadata/diagnostics/per-prompt CSVs | `artifacts/frozen/ioi/stage2c_primary_stratified_mean_end/` | `PYTHONPATH=. python scripts/ioi_stage2c_primary_stratified.py --outdir runs/smoke/ioi_stage2c --quick --device cpu` | `PYTHONPATH=. python scripts/ioi_stage2c_primary_stratified.py --outdir runs/ioi_stage2c_primary_stratified_mean_end --device auto --n-prompts 256 --n-reference 512 --n-subsets 240 --sampling-mode primary_stratified --batch-size 32 --ablation mean --positions end --k-folds 5 --bootstrap-repeats 200` | smoke 10-30m CPU after model cache; full 2-4h MPS/GPU | TransformerLens and GPT-2-small required; MPS/GPU recommended |
| IOI Stage 2d per-pair decomposition with count-additive control | Table 2 | `ioi_stage2d_v0/scripts/ioi_stage2d_per_pair_decomposition.py` | `ioi_stage2d_report.md`, `ioi_stage2d_fit_summary.csv`, `ioi_stage2d_bootstrap_summary.csv`, `ioi_stage2d_paired_delta_vs_count_additive.csv`, `ioi_stage2d_paired_delta_vs_additive.csv`, `ioi_stage2d_coefficients.csv`, `ioi_stage2d_kfold_predictions.csv`, `ioi_stage2d_diagnostics.json`, plots | `artifacts/frozen/ioi/stage2d_per_pair/` | `PYTHONPATH=. python scripts/ioi_stage2d_per_pair_decomposition.py --input-run artifacts/frozen/ioi/stage2c_primary_stratified_mean_end --outdir runs/smoke/ioi_stage2d --k-folds 5 --bootstrap-repeats 20` | `PYTHONPATH=. python scripts/ioi_stage2d_per_pair_decomposition.py --input-run runs/ioi_stage2c_primary_stratified_mean_end --outdir runs/ioi_stage2d_per_pair --k-folds 5 --bootstrap-repeats 300` | smoke <30s CPU; full postprocess <2m CPU if Stage 2c data already exists | No GPU and no TransformerLens for Stage 2d postprocess; requires Stage 2c frozen or rerun outputs |

## Fast Reproduction Commands

Fast reproduction should never reload GPT-2 or retrain transformers by default. It should read `artifacts/frozen`, regenerate paper-facing figures/tables, regenerate ObserverCards, and run frozen-claim validation.

Target commands:

```bash
observerbench reproduce --frozen artifacts/frozen --out artifacts/generated --cards
observerbench figures all --frozen artifacts/frozen --out artifacts/generated/figures
observerbench tables all --frozen artifacts/frozen --out artifacts/generated/tables
observerbench cards --frozen artifacts/frozen --out artifacts/generated/cards
observerbench validate --frozen artifacts/frozen
```

Expected outputs:

```text
artifacts/generated/
  figures/
    fig1_ctl1_analytic_gamma.png
    fig2_ctl1_analytic_nuisance.png
    fig3_ctl1_trained_gamma.png
    fig4_ctl1_trained_nuisance.png
    fig5_ctl2_trajectories.png
    fig6_ctl2_divergence_delta.png
    fig7_ctl2_collateral_trajectories.png
    fig8_ioi_stage1.png
    fig9_ioi_stage2b.png
    fig10_ioi_stage2c.png
  tables/
    table1_ctl2_gamma_sweep.csv
    table2_ioi_stage2d.csv
  cards/
    ctl1_analytic_observer_cards.json
    ctl1_trained_observer_cards.json
    ctl2_observer_cards.json
    ioi_stage1_observer_cards.json
    ioi_stage2b_observer_cards.json
    ioi_stage2c_observer_cards.json
    ioi_stage2d_observer_cards.json
```

## Expensive Full Reruns

Full reruns should be opt-in and documented separately from fast reproduction.

Ctl-1 analytic:

```bash
observerbench run ctl1-analytic --full --out runs/collateral_sweeps_v2 --seeds 0,1,2,3,4,5,6,7,8,9
```

Ctl-1 trained transformer:

```bash
observerbench run ctl1-trained --full --out runs/trained_transformer_sweeps_v5_clean --seeds 0,1,2,3,4,5 --device auto
```

Ctl-2 closed loop:

```bash
observerbench run ctl2-default --full --out runs/trained_transformer_ctl2_v7_clean --seeds 0,1,2,3,4,5 --device auto
observerbench run ctl2-sweeps --full --out runs/trained_transformer_ctl2_v7_sweeps --seeds 0,1,2,3,4,5 --device auto
```

IOI:

```bash
observerbench run ioi-stage1 --full --out runs/ioi_stage1_both_end --device auto
observerbench run ioi-stage2b --full --out runs/ioi_stage2b_mean_end --device auto
observerbench run ioi-stage2c --full --out runs/ioi_stage2c_primary_stratified_mean_end --device auto
observerbench run ioi-stage2d --full --input-run runs/ioi_stage2c_primary_stratified_mean_end --out runs/ioi_stage2d_per_pair
```

The IOI Stage 2d full command is computationally cheap if Stage 2c outputs exist. The expensive part is Stage 2c, which reruns GPT-2-small interventions.

## Dependency Scope

The base install must support fast reproduction from frozen outputs without TransformerLens, without GPT-2 downloads, and without training models. Base dependencies should cover CSV/JSON loading, plotting, validation, and ObserverCard generation only.

Optional full-rerun dependencies:

- Ctl-1 trained and Ctl-2 full reruns may require `torch`.
- IOI full reruns require `observerbench[ioi]`, which may include TransformerLens and its model-loading dependencies.
- Full IOI reruns may require GPT-2-small access, but this must never happen in default tests or fast reproduction.

CI rules:

- CI must not download GPT-2.
- CI must not train models.
- CI must run `pytest -q` against base install plus frozen fixtures.
- Optional TransformerLens tests must be skipped unless explicitly enabled by an environment flag and model cache is already present.

## Minimal Tests

Default tests must not download models, require TransformerLens, or run long training jobs.

Core tests:

- `ObserverResult` and `ObserverCard` JSON serialization round-trips.
- `cards.py` recommendations preserve null/sufficient/collateral caution behavior.
- `metrics.py` handles degenerate `r2_score` and `safe_spearman` cases.
- frozen-output manifest resolves every required file in `artifacts/frozen`.

Analytic Ctl-1 smoke tests:

- one tiny seed runs on CPU and writes expected CSV/JSON/PNG files.
- gamma-zero null sanity: first-order and lifted target/collateral metrics are approximately tied.
- positive-gamma sanity: first-order keeps high target fraction while collateral ratio is greater than lifted.
- nuisance placement changes collateral preference in the expected direction.

Frozen paper-claim tests:

- Figure 1/2 CSVs contain the analytic collateral tradeoff and nuisance crossover described in the paper.
- Figure 3/4 CSVs contain the trained Ctl-1 directionality and null sanity without asserting stronger claims than the paper.
- Table 1/Figure 6 CSVs show Ctl-2 first-order minus lifted medians positive at the paper's default interaction strengths, with null near zero.
- Figure 8 validates only the IOI Stage 1 feasibility/self-repair diagnostic.
- Figure 9 validates that IOI Stage 2b random subsets make additive behavior mostly sufficient/null.
- Figure 10 validates that IOI Stage 2c primary-stratified subsets expose count interactions.
- Table 2 validates that Stage 2d's largest improvement is not primarily P x B, and that P x E/count-additive structure dominates as stated in the paper.

Ctl-2 genuine feedback-loop tests:

- residual state changes across loop steps in the quick configuration.
- observer features are read from the current edited residual at each step, not from a cached initial state.
- `oracle_target` is stable under the quick configuration and does not diverge from its expected control behavior.
- `gamma=0` first-order and lifted-interaction observers match within tolerance for target and collateral metrics.

IOI Stage 2d claim tests:

- paired delta MAE is reported against both `additive_head` and `count_additive`.
- the frozen fixture asserts P x E is the dominant single-pair interaction.
- P x B is smaller than P x E.
- B x E is not labeled successful unless it beats `additive_head`.
- `count_additive` alone does not explain the all-pairs win.

ObserverCards claim tests:

- recommendations are metric-derived, not observer-name-derived.
- the same observer name with opposite metrics produces different recommendations.
- different observer names with identical metrics produce the same recommendation.

IOI fixture tests:

- Stage 2d postprocess runs on a small frozen Stage 2c fixture without TransformerLens.
- IOI tests that need TransformerLens are marked optional, skipped unless `OBSERVERBENCH_RUN_TL=1` is set and the model is already available.

CLI tests:

- `observerbench reproduce --frozen tests/fixtures/frozen --out tmp` writes all expected figure/table/card paths.
- `observerbench validate --frozen tests/fixtures/frozen` fails clearly if a required frozen file is missing.
- smoke CLI commands use repo-relative output paths and do not rely on manually mutating `sys.path`.

## ObserverCards Plan

Reuse the existing `ObserverResult` and `ObserverCard` primitives. Add only small composition functions that translate frozen summary rows into cards:

- analytic Ctl-1 cards from `gamma_sweep_pairwise_summary.csv` and `nuisance_weight_pairwise_summary.csv`,
- trained Ctl-1 cards from trained sweep summaries,
- Ctl-2 cards from default closed-loop summaries and sweep summaries,
- IOI cards from Stage 1/2b/2c/2d fit summaries and diagnostics.

Cards should record:

- task name,
- observer/access regime,
- primary metric,
- target-vs-collateral behavior,
- reproducibility provenance,
- whether the result is frozen or freshly rerun,
- failure mode/recommendation text already supported by `cards.py`.

Do not make cards a general benchmark registry.

## Public Release Definition Of Done

- `pip install -e .` works from a fresh checkout.
- `pytest -q` passes without downloading GPT-2 or training.
- `python scripts/reproduce_paper_fast.py` regenerates all paper figures from frozen outputs.
- ObserverCards generate valid JSON and Markdown.
- `paper/figure_map.md` maps every paper result to frozen outputs and full-rerun commands.
- `artifacts/frozen/manifest.yaml` lists every paper result, frozen file, expected generated artifact, command, sha256, and dependency requirement.
- No unintended local paths, credentials, cache paths, or private information remain.
- Stop at reproduction. Do not add experiments.

## Migration Steps

1. Create package skeleton under `src/observerbench/`.
2. Move latest v7 core modules into `src/observerbench/` without behavioral changes.
3. Move Ctl task modules into `src/observerbench/tasks/` and keep legacy scripts as wrappers.
4. Extract duplicated IOI helpers into `tasks/ioi/` modules by composition, preserving stage commands and outputs.
5. Add `artifacts/frozen/manifest.yaml` and copy only paper-required frozen outputs.
6. Add figure/table reproduction commands that read frozen outputs.
7. Add ObserverCard generation from frozen metrics.
8. Add CPU-safe smoke tests and frozen-claim tests.
9. Document fast reproduction separately from expensive full reruns.
10. Remove stale MVP wording, placeholder adapters, and obsolete source snapshots from the public surface.

## Open Provenance Notes

- The latest code for trained Ctl-1 is in `observerbench_mvp_v7_ctl2`, but the clean frozen trained Ctl-1 paper outputs are under `observerbench_mvp_v5/runs/trained_transformer_sweeps_v5_clean`. Preserve that provenance in the frozen manifest until a controlled v7 rerun replaces it.
- Ctl-2 paper outputs come from `observerbench_mvp_v7_ctl2/runs/trained_transformer_ctl2_v7_clean` and `observerbench_mvp_v7_ctl2/runs/trained_transformer_ctl2_v7_sweeps`.
- IOI Stage 2d is a postprocess over Stage 2c outputs; fast reproduction should use frozen Stage 2c data and should not rerun GPT-2.
- The IOI results are diagnostics for a known-answer setting, not new IOI scientific claims.
