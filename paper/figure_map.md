# ObserverBench Paper Reproduction Map

ObserverBench v0 is a reproduction artifact and diagnostic workbench for the paper. It is not a general benchmark platform, not a plugin system, and not a place to add new experiments or revise scientific claims.

Paper source used for this map: `observerbench_paper_v9.pdf` (13 pages, generated June 29, 2026).

Fast reproduction is CPU-only:

```bash
python scripts/reproduce_paper_fast.py
```

It reads frozen CSV/JSON summaries under `results/frozen/`, writes `paper/generated_figures/`, does not download GPT-2, does not import TransformerLens, and does not train models.

Full reruns are routed through:

```bash
python scripts/reproduce_paper_full.py
```

The full script refuses selected training, TransformerLens, or GPT-2 runs unless `--yes-run-expensive` is passed. The IOI full GPT-2/TransformerLens runners are documented as expensive optional reruns; the migrated base package currently supports CPU smoke fixtures and Stage 2d CPU postprocessing.

| Figure/Table | Paper section | Result claim | Command to regenerate from frozen outputs | Command to rerun full experiment | Expected output files | Approx runtime | Requires GPU/MPS? | Requires TransformerLens/GPT-2? | Frozen result path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 1 | Sec. 4.2, Analytic Ctl-1 interaction strength | First-order remains target-competitive while paying extra collateral as interaction strength grows. | `python scripts/reproduce_paper_fast.py --only figure_01` | `python scripts/reproduce_paper_full.py --only ctl1_analytic` | `paper/generated_figures/figure_01_ctl1_analytic_interaction_sweep.png` | fast <5s; full analytic <1m CPU | No | No | `results/frozen/ctl1_analytic/collateral_sweeps_v2/` |
| Figure 2 | Sec. 4.3, Analytic nuisance placement | Collateral burden is geometry-dependent: first-order pays more when nuisance lies in the main-effect steering subspace, lifted pays more as nuisance moves toward the interaction coordinate. | `python scripts/reproduce_paper_fast.py --only figure_02` | `python scripts/reproduce_paper_full.py --only ctl1_analytic` | `paper/generated_figures/figure_02_ctl1_analytic_nuisance_sweep.png` | fast <5s; full analytic <1m CPU | No | No | `results/frozen/ctl1_analytic/collateral_sweeps_v2/` |
| Figure 3 | Sec. 5, Trained-transformer Ctl-1 interaction strength | The learned residual-space task preserves the Ctl-1 directionality: first-order is target-useful but increasingly worse than lifted as interaction strength grows. | `python scripts/reproduce_paper_fast.py --only figure_03` | `python scripts/reproduce_paper_full.py --only trained_ctl1 --yes-run-expensive` | `paper/generated_figures/figure_03_trained_ctl1_interaction_sweep.png` | fast <5s; full 1-3h paper-scale, longer on CPU | Recommended | No | `results/frozen/trained_ctl1/trained_transformer_sweeps_v5_clean/` |
| Figure 4 | Sec. 5, Trained-transformer Ctl-1 nuisance placement | Learned residual geometry preserves the qualitative nuisance-placement crossover; observer name alone does not determine collateral. | `python scripts/reproduce_paper_fast.py --only figure_04` | `python scripts/reproduce_paper_full.py --only trained_ctl1 --yes-run-expensive` | `paper/generated_figures/figure_04_trained_ctl1_nuisance_sweep.png` | fast <5s; full 1-3h paper-scale, longer on CPU | Recommended | No | `results/frozen/trained_ctl1/trained_transformer_sweeps_v5_clean/` |
| Figure 5 | Sec. 6, Ctl-2 closed-loop default trajectories | With controller and plant fixed, first-order observer bias compounds through the loop; lifted attenuates the instability and oracle_target verifies the gain is stable for an accurate observer. | `python scripts/reproduce_paper_fast.py --only figure_05` | `python scripts/reproduce_paper_full.py --only trained_ctl2 --yes-run-expensive` | `paper/generated_figures/figure_05_ctl2_closed_loop_trajectories.png` | fast <5s; full 30-90m paper-scale MPS/GPU, longer on CPU | Recommended | No | `results/frozen/ctl2/default_v7_clean/` |
| Table 1 | Sec. 6, Ctl-2 interaction-strength sweep | At gamma=0 first-order and lifted tie; in the interactional regime first-order has higher integrated error, collateral, divergence rate, and final target MSE than lifted. | `python scripts/reproduce_paper_fast.py --only table_01` | `python scripts/reproduce_paper_full.py --only trained_ctl2 --yes-run-expensive` | `paper/generated_figures/table_01_ctl2_interaction_sweep.csv`; `paper/generated_figures/table_01_ctl2_interaction_sweep.md` | fast <5s; full 2-5h sweep paper-scale MPS/GPU, longer on CPU | Recommended | No | `results/frozen/ctl2/sweeps_v7/` |
| Figure 6 | Sec. 6, Ctl-2 divergence-rate delta | Divergence-rate delta is near zero at the null and positive in the interactional regime, exposing observer-induced feedback failure. | `python scripts/reproduce_paper_fast.py --only figure_06` | `python scripts/reproduce_paper_full.py --only trained_ctl2 --yes-run-expensive` | `paper/generated_figures/figure_06_ctl2_divergence_rate_delta.png` | fast <5s; full 2-5h sweep paper-scale MPS/GPU, longer on CPU | Recommended | No | `results/frozen/ctl2/sweeps_v7/` |
| Figure 7 | Sec. 6, Ctl-2 collateral trajectories | First-order accumulates more collateral at the default nuisance placement; the nuisance sweep keeps the geometric caveat explicit. | `python scripts/reproduce_paper_fast.py --only figure_07` | `python scripts/reproduce_paper_full.py --only trained_ctl2 --yes-run-expensive` | `paper/generated_figures/figure_07_ctl2_collateral_trajectories.png` | fast <5s; full 30-90m paper-scale MPS/GPU, longer on CPU | Recommended | No | `results/frozen/ctl2/default_v7_clean/`; `results/frozen/ctl2/sweeps_v7/` |
| Figure 8 | Sec. 7.1, IOI Stage 1 whole-group self-repair diagnostic | Mean ablation reproduces a large conditional self-repair diagnostic: the joint primary+backup drop exceeds the singleton-additive prediction. | `python scripts/reproduce_paper_fast.py --only figure_08` | `python scripts/reproduce_paper_full.py --only ioi_stage1 --yes-run-expensive` | `paper/generated_figures/figure_08_ioi_stage1_self_repair.png` | fast <5s; full 10-30m after GPT-2 is cached | Recommended | Yes | `results/frozen/ioi/stage1_both_end/` |
| Figure 9 | Sec. 7.3, IOI Stage 2b random head-level subsets | Over broad random head subsets, the per-head additive observer is already strong; the cheap observer is mostly sufficient under this intervention distribution. | `python scripts/reproduce_paper_fast.py --only figure_09` | `python scripts/reproduce_paper_full.py --only ioi_stage2b --yes-run-expensive` | `paper/generated_figures/figure_09_ioi_stage2b_random_subsets.png` | fast <5s; full 1-3h after GPT-2 is cached | Recommended | Yes | `results/frozen/ioi/stage2b_mean_end/` |
| Figure 10 | Sec. 7.4, IOI Stage 2c primary-stratified subsets | Under primary-stratified interventions, count interactions matter; the bundled win should not be assigned to P x B alone. | `python scripts/reproduce_paper_fast.py --only figure_10` | `python scripts/reproduce_paper_full.py --only ioi_stage2c --yes-run-expensive` | `paper/generated_figures/figure_10_ioi_stage2c_primary_stratified.png` | fast <5s; full 2-4h after GPT-2 is cached | Recommended | Yes | `results/frozen/ioi/stage2c_primary_stratified_mean_end/` |
| Table 2 | Sec. 7.5, IOI Stage 2d per-pair decomposition | The Stage 2c win is real cross-group signal dominated by P x E rather than P x B; count_additive alone does not recover the all-pairs gain. | `python scripts/reproduce_paper_fast.py --only table_02` | `python scripts/reproduce_paper_full.py --only ioi_stage2d` | `paper/generated_figures/table_02_ioi_stage2d_per_pair.csv`; `paper/generated_figures/table_02_ioi_stage2d_per_pair.md` | fast <5s; full postprocess <2m CPU if Stage 2c outputs exist | No | No for Stage 2d postprocess; yes for full Stage 2c rerun | `results/frozen/ioi/stage2d_per_pair/` |

## Frozen Inputs

Fast reproduction expects these frozen summaries:

- `results/frozen/ctl1_analytic/collateral_sweeps_v2/gamma_sweep_pairwise_summary.csv`
- `results/frozen/ctl1_analytic/collateral_sweeps_v2/nuisance_weight_pairwise_summary.csv`
- `results/frozen/trained_ctl1/trained_transformer_sweeps_v5_clean/gamma_sweep_pairwise_summary.csv`
- `results/frozen/trained_ctl1/trained_transformer_sweeps_v5_clean/nuisance_weight_pairwise_summary.csv`
- `results/frozen/ctl2/default_v7_clean/ctl2_trajectory_quantiles.csv`
- `results/frozen/ctl2/sweeps_v7/gamma_sweep_pairwise_summary.csv`
- `results/frozen/ioi/stage1_both_end/ioi_stage1_summary.csv`
- `results/frozen/ioi/stage2b_mean_end/ioi_stage2b_bootstrap_summary.csv`
- `results/frozen/ioi/stage2b_mean_end/ioi_stage2b_kfold_predictions.csv`
- `results/frozen/ioi/stage2c_primary_stratified_mean_end/ioi_stage2c_paired_delta_mae.csv`
- `results/frozen/ioi/stage2c_primary_stratified_mean_end/ioi_stage2c_primary_count_errors.csv`
- `results/frozen/ioi/stage2d_per_pair/ioi_stage2d_bootstrap_summary.csv`
- `results/frozen/ioi/stage2d_per_pair/ioi_stage2d_model_comparison.csv`

The frozen files are summaries and diagnostics only. They do not include model weights, GPT-2 caches, Python virtual environments, or private local paths.
