# ObserverBench Paper Reproduction Map

ObserverBench v0 is a bounded paper-reproduction artifact and observer-control
workbench. It is not a plugin system or a bring-your-own-model API. The tested
external composition points are the analytic Ctl-1 adapter and the
inference-free IOI finite-effect registry. Trained Ctl-1 and Ctl-2 remain
registered reproduction tasks.

The current manuscript source is under `paper/observerbench_v15_source/`. This
map names results by stable reproduction ID rather than by typeset figure
number, because figure numbers can move during revision.

## Fast Path

Both commands below run the same CPU-only renderer:

```bash
python scripts/reproduce_paper_fast.py
observerbench make-figures --outdir paper/generated_current
```

The renderer reads legacy summaries from `results/frozen/` and checked review
outputs from `results/revision/`. It writes only to the requested output
directory. It does not change either input tree, download GPT-2, import
TransformerLens, or train a model.

The default IDs match the current manuscript:

```text
figure_01  figure_02  figure_03  figure_04  figure_08
revision_ctl2_factorial  revision_ctl2_calibration  revision_ctl2_table
revision_ioi_stage2b  revision_ioi_stage2c  revision_ioi_mobius
revision_ioi_table  revision_observer_cards
```

Generate one checked result with:

```bash
python scripts/reproduce_paper_fast.py --only revision_ctl2_factorial
observerbench make-figures \
  --only revision_ioi_stage2c \
  --outdir paper/generated_current
```

## Current checked manuscript artifacts

| Study | Claim supported | Paper artifact builder | Checked inputs |
| --- | --- | --- | --- |
| Study 1 | The affine certificate predicts error after each edit traverses a learned nonlinear Transformer suffix. The diagonal lifted pair helps for the interactional target; the crossed pairs fail conditioning and support only a compatibility claim. | `python scripts/build_phase05_nonlinear_suffix_artifact.py` | `results/revision/phase05/nonlinear_suffix_v3_support/` |
| Study 2 | Interaction-aware models improve held-out mean-effect prediction, but that improvement does not reliably lower fixed-action loss. | Included in the checked IOI registry and manuscript source | `results/revision/phase05/ioi_confirmatory/` |
| Study 3 | Direct-risk fitting beats same-basis mean controls in forced choice, while the missing no-op and failed clean gate bound the result. | `python scripts/build_phase06_ioi_confirmation_artifact.py` | `results/revision/phase06/ioi_fresh_confirmation/` |
| Study 4 | On clean-gated canonical prompts at the pilot-informed target, direct risk beats the natural-mean plug-in and exact no action. | `python scripts/build_phase07_ioi_confirmation_artifact.py` | `results/revision/phase07/ioi_canonical_noop_confirmation_v2/` |
| Study-4 sensitivity | Direct risk beats natural-mean and transformed-mean controls at all three targets; its value against no action is target-dependent. | `python scripts/build_phase08_ioi_sensitivity_artifact.py` | `results/revision/phase08/ioi_target_sensitivity/` |
| External APPS safety monitor | A neutral-context activation probe outperforms the declared output-distribution and supervised-text controls; rare-prevalence resampling removes the balanced-panel audit floor. | `python scripts/plot_controlarena_prevalence_sensitivity.py` | `results/revision/ai_control/controlarena_apps_qwen_followup_v1/` |
| APPS post-outcome diagnostics | Within-group separation remains high, but global audit allocation drifts across dataset families and neutral-input-length quartiles. Cached verdict layers do not explain the neutral-context gap. | `python scripts/analyze_controlarena_apps_strata.py`; `python scripts/analyze_controlarena_verdict_layers.py` | `results/revision/ai_control/controlarena_apps_qwen_followup_v1/` |

The full manuscript command validates all of these before compiling the PDF:

```bash
make -C paper/observerbench_v15_source
```

## Earlier checked revision results

| Reproduction ID | Claim supported | Fast output | Scientific rerun or audit | Checked inputs |
| --- | --- | --- | --- | --- |
| `revision_ctl2_factorial` | The 2x2 estimator-by-direction design separates estimator error from direction geometry. For every nonzero interaction strength, first-order ISE exceeds lifted ISE in all six training seeds with either direction held fixed. Bands are seed min--max, not percentile intervals. | `revision_ctl2_factorial_ise_by_gamma.png` | Current-code 30-cell sweep followed by `scripts/audit_ctl2_phase01.py` with the Phase-4 paths | `results/revision/phase04/ctl2_multiseed_sweep_current/phase01_sweep_results.csv`; `results/revision/phase04/ctl2_phase04_audit.json` |
| `revision_ctl2_calibration` | Held-out scalar response calibration corrects local response gain in the affine fixture, but does not remove estimation bias and can be ill-conditioned. | `revision_ctl2_response_calibration_by_gamma.png` | Same current-code sweep and audit as above | Same checked Phase-4 Ctl-2 inputs as above |
| `revision_ctl2_table` | Fixed-direction Ctl-2 effects used in the manuscript table. | `revision_ctl2_factorial_table.csv`; `.md`; `.tex` | `python scripts/build_phase04_artifact.py` | `results/revision/phase04/ctl2_phase04_audit.json` |
| `revision_ioi_stage2b` | Per-head additivity remains strong, but equal-capacity count interactions improve held-out prediction under the anchored broad-random mask design. PE is largest in both add-one and leave-one-out analyses; PB contributes conditionally. | `revision_ioi_stage2b_capacity_matched.png` | `python scripts/run_ioi_phase2_capacity.py` | `results/revision/phase02/ioi_stage2b_capacity/` |
| `revision_ioi_stage2c` | Primary-stratified masks increase the value of interaction-aware prediction. With four added degrees of freedom per pair, PE dominates; PB is weak in add-one analysis but positive when removed from the full pair model. | `revision_ioi_stage2c_capacity_matched.png` | `python scripts/run_ioi_phase2_capacity.py` | `results/revision/phase02/ioi_stage2c_capacity/` |
| `revision_ioi_mobius` | Direct group-mask interactions corroborate the predictive ranking: PE exceeds PB, with a prompt-bootstrap interval for PE-minus-PB that stays above zero. | `revision_ioi_mobius_intervals.png` | `python scripts/run_ioi_phase2_capacity.py` | `results/revision/phase02/ioi_stage2c_capacity/mobius_bootstrap_summary.csv` |
| `revision_ioi_table` | Capacity-matched add-one, leave-one-out, and direct-interaction estimates with prompt-bootstrap 5--95% intervals. | `revision_ioi_capacity_contrasts.csv`; `.md`; `.tex`; `revision_ioi_mobius_intervals.csv`; `.md` | `python scripts/run_ioi_phase2_capacity.py` | `results/revision/phase02/ioi_stage2b_capacity/`; `results/revision/phase02/ioi_stage2c_capacity/` |
| `revision_observer_cards` | Five checked cards record the estimand, measurement design, metrics, thresholds, recommendation, scope, status, and source hashes. The Ctl-2 card is typeset in the paper. | `observer_cards/observer_card.json`; `.md`; `ctl2_observer_card.tex` | `python scripts/build_phase04_artifact.py` | `results/revision/phase04/observer_cards/`; `results/revision/phase04/phase04_manifest.json` |

The combined IOI claim gate is
`results/revision/phase02/ioi_phase02_claim_audit.json`. The current Ctl-2
claim gate is `results/revision/phase04/ctl2_phase04_audit.json`. The combined
Phase-4 gate is `results/revision/phase04/phase04_claim_audit.json`.

## Legacy Frozen Results

The following superseded IDs remain available through `--only` or as part of
`--legacy`:

```text
figure_05  figure_06  figure_07  figure_09  figure_10
table_01   table_02
```

They read only from `results/frozen/`. Their pre-review Ctl-2 attribution and
single-product IOI pair ranking are superseded by the checked revision results
above. Frozen IDs `figure_01`--`figure_04` and `figure_08` remain in the current
paper.

## Input Boundary

Frozen and checked inputs contain result summaries, diagnostics, and the
prompt-level values required for the checked bootstrap. They do not contain
model weights, GPT-2 caches, virtual environments, or private local paths.

Expensive model and training reruns remain outside the fast renderer:

```bash
python scripts/reproduce_paper_full.py --yes-run-expensive
```

The full dispatcher requires explicit acknowledgement before selected training
or TransformerLens/GPT-2 tasks.
