# ObserverCards

## ctl1_analytic / first_order_vs_lifted_analytic_sweep

**Task name.** ctl1_analytic
**Observer name.** first_order_vs_lifted_analytic_sweep
**Model or substrate.** analytic three-coordinate activation geometry
**Access regime.** white-box latent state
**Observer family.** first-order and lifted estimator--direction pairs

**Estimand.** one-shot target response and fixed-readout collateral movement
**Measurement design.** sweep interaction strength and nuisance placement while matching initial target authority
**Validation target.** target usefulness with low collateral movement, including a gamma-zero null
**Result status.** frozen

### Primary Metrics
- `null_gamma`: 0.0
- `null_collateral_ratio_first_order_over_lifted`: 1.0001235748986246
- `reported_gamma`: 0.5
- `collateral_ratio_first_order_over_lifted`: 2.111432516227684
- `target_improvement_fraction_of_lifted`: 0.9802459316109742
- `target_mse_ratio_first_order_over_lifted`: 1.1708426004609085
- `n_seeds`: 10

### Thresholds
- `null_collateral_ratio_tolerance`: 0.01
- `target_competitive_fraction_min`: 0.8
- `collateral_ratio_caution`: 1.2

### Failure Modes Detected
- The first-order estimator--direction pair pays more collateral than the lifted pair at the reported interaction setting.
- The collateral ranking changes when the nuisance readout moves, so the observer name alone does not determine collateral cost.

### Recommendation
Use the lifted pair when collateral movement matters in this fixture, and report direction--nuisance overlap rather than attributing collateral to the observer name.

### Scope limits
- This is a coordinate-level one-shot intervention.
- The base-coordinate robustness check remains a continuous edit, not a realizable token intervention.
- Ten random seeds are the uncertainty unit.

### Commands to reproduce
- `python scripts/reproduce_paper_fast.py --only figure_01`

### Reproducibility provenance
- `source_files`: {'results/frozen/ctl1_analytic/collateral_sweeps_v2/gamma_sweep_pairwise_summary.csv': '2e77e6ae53aa8f2f5957332705e01a579f022b080403b64f9deff0d556bc018c', 'results/frozen/ctl1_analytic/collateral_sweeps_v2/nuisance_weight_pairwise_summary.csv': '24391579895b02157d0c61689739a68bdc9e888290472c72ac19d5d90186a8cd'}
- `source_revision`: 0cec2c4ef17d9c3b0ef3fea8fdf15347ee4e28cb
- `package_version`: 0.0.0

## trained_ctl1 / first_order_vs_lifted_trained_sweep

**Task name.** trained_ctl1
**Observer name.** first_order_vs_lifted_trained_sweep
**Model or substrate.** tiny trained Transformer residual stream
**Access regime.** white-box learned residual representation
**Observer family.** first-order and lifted estimator--direction pairs

**Estimand.** one-shot target response and fixed-readout collateral movement
**Measurement design.** sweep interaction strength and nuisance placement while matching initial target authority
**Validation target.** target usefulness with low collateral movement, including a gamma-zero null
**Result status.** frozen

### Primary Metrics
- `null_gamma`: 0.0
- `null_collateral_ratio_first_order_over_lifted`: 1.000000044688213
- `reported_gamma`: 1.15
- `collateral_ratio_first_order_over_lifted`: 1.6279949383922765
- `target_improvement_fraction_of_lifted`: 0.9060205803454612
- `target_mse_ratio_first_order_over_lifted`: 1.8345507455975063
- `n_seeds`: 6

### Thresholds
- `null_collateral_ratio_tolerance`: 0.01
- `target_competitive_fraction_min`: 0.8
- `collateral_ratio_caution`: 1.2

### Failure Modes Detected
- The first-order estimator--direction pair pays more collateral than the lifted pair at the reported interaction setting.
- The collateral ranking changes when the nuisance readout moves, so the observer name alone does not determine collateral cost.

### Recommendation
Use the lifted pair when collateral movement matters in this fixture, and report direction--nuisance overlap rather than attributing collateral to the observer name.

### Scope limits
- The auxiliary feature heads are supervised.
- Six trained models are the uncertainty unit.
- Small denominators make some collateral ratios sensitive.

### Commands to reproduce
- `python scripts/reproduce_paper_fast.py --only figure_03`

### Reproducibility provenance
- `source_files`: {'results/frozen/trained_ctl1/trained_transformer_sweeps_v5_clean/gamma_sweep_pairwise_summary.csv': '02a03eef10de48fc4bc9bdf84e56d5303c75f79feda0aef5c19012880f3f3210', 'results/frozen/trained_ctl1/trained_transformer_sweeps_v5_clean/nuisance_weight_pairwise_summary.csv': '9aaf45ce22fb749cc52704bddfe73353c912412a3f423ea2e6604fdb1a57d19b'}
- `source_revision`: 0cec2c4ef17d9c3b0ef3fea8fdf15347ee4e28cb
- `package_version`: 0.0.0

## trained_ctl2 / first_order_estimator_factorial_evaluation

**Task name.** trained_ctl2
**Observer name.** first_order_estimator_factorial_evaluation
**Model or substrate.** tiny trained Transformer final-residual representation
**Access regime.** white-box affine residual readouts with additive residual updates
**Observer family.** factorial first-order and interaction-aware estimators and directions

**Estimand.** target-head state along a repeated additive residual intervention
**Measurement design.** hold direction fixed while comparing estimators, then hold estimator fixed while comparing directions across interaction strengths and model seeds
**Validation target.** integrated true target error, observer self-gain, clipping, fitted nuisance movement, and distance from clean residual states
**Result status.** checked_revision

### Primary Metrics
- `nonzero_gamma_settings`: 4
- `gamma_settings`: [0.0, 0.5, 1.0, 1.15, 1.5]
- `fixed_direction_contrasts_per_gamma`: 2
- `training_seeds`: 6
- `positive_seed_level_fixed_direction_contrasts`: 48
- `total_seed_level_fixed_direction_contrasts`: 48
- `gamma_zero_max_absolute_factorial_effect`: 9.5526928885666e-07
- `gamma_0p5_diagonal_ise_delta_mean`: 6.712665514828281
- `gamma_0p5_threshold_flag_delta_mean`: -0.15077777777777782
- `gamma_1p15_first_order_direction_ise_delta_mean`: 60.73993009925025
- `gamma_1p15_first_order_direction_ise_delta_seed_min`: 11.713859219587771
- `gamma_1p15_first_order_direction_ise_delta_seed_max`: 132.96104985099248
- `gamma_1p15_lifted_direction_ise_delta_mean`: 50.02418721161393
- `gamma_1p15_lifted_direction_ise_delta_seed_min`: 18.97668480530213
- `gamma_1p15_lifted_direction_ise_delta_seed_max`: 102.31698466629268
- `gain_matched_initial_target_mse`: 9.99998881171672e-05
- `gain_matched_final_target_mse`: 0.1695972899752609
- `gain_matched_observer_error_pole`: 0.7025
- `gain_matched_control_clip_fraction`: 0.0
- `response_calibrated_initial_target_mse`: 0.7200662986178054
- `response_calibrated_final_target_mse`: 0.0689253850595846
- `response_calibrated_observer_self_gain`: 0.8499999999999999
- `projected_direction_norm_ratio`: 7.47036619498702
- `projected_cumulative_collateral_abs`: 13.361957980279035
- `projected_mean_nearest_clean_normalized_path`: 1.239753502607438
- `max_absolute_response_calibration_scale`: 1146.0019858890057

### Thresholds
- `integrated_squared_error_ratio_vs_reference_max`: 1.25
- `cumulative_collateral_ratio_vs_reference_max`: 2.0
- `divergence_rate_max`: 0.05
- `divergence_rate_mse_growth_max`: 0.05
- `target_error_worsened_rate_max`: 0.25
- `observer_bias_mae_path_max`: 0.5
- `direction_norm_ratio_vs_unprojected_max`: 5.0
- `mean_nearest_clean_residual_normalized_path_max`: 1.0
- `control_clip_fraction_max`: 0.25
- `gamma_zero_max_absolute_factorial_effect`: 1e-05
- `all_nonzero_seed_level_estimator_contrasts_must_be_positive`: True
- `unsaturated_observer_error_convergence`: 0 < K*g_E < 2

### Failure Modes Detected
- The uncalibrated first-order estimator has greater ISE than the lifted estimator with either direction fixed at every nonzero interaction strength in all six seeds.
- Matching the observer-error pole does not repair true-target tracking.
- Affine-span projection is ill-conditioned and does not establish an on-manifold actuator.
- Some scalar response calibrations require very large gains.

### Recommendation
Do not use the uncalibrated first-order estimator for this control target. If response calibration is used, report its conditioning and validate true-target tracking; treat affine-span projection as a diagnostic only.

### Scope limits
- The plant is an additive final-residual loop, not a rerun Transformer.
- The target is the model target head; collateral is a fitted nuisance probe.
- The local convergence condition applies only to the unsaturated scalar observer-error mode.
- Four binary inputs yield four trajectories per trained model, so training seed is the uncertainty unit.

### Commands to reproduce
- `python scripts/build_phase04_artifact.py`
- `python scripts/reproduce_paper_fast.py --only revision_ctl2_factorial`

### Reproducibility provenance
- `source_files`: {'results/revision/phase04/ctl2_phase04_audit.json': 'b58dd09efd4bbc48c82cc6c29d8e117eaecce5ea2fa91e42a629a7b47c16a347', 'results/revision/phase04/ctl2_multiseed_sweep_current/phase01_sweep_results.csv': 'a900f52ba2e1d92171b1c25a54bf05be24e1ff4b530a5ff9e4e255c909011cf4'}
- `source_revision`: 0cec2c4ef17d9c3b0ef3fea8fdf15347ee4e28cb
- `package_version`: 0.0.0

## ioi_stage2b / capacity_matched_all_pairs

**Task name.** ioi_stage2b
**Observer name.** capacity_matched_all_pairs
**Model or substrate.** GPT-2-small IOI head-subset intervention outputs
**Access regime.** frozen prompt-level head-ablation effects
**Observer family.** capacity-matched count-bin interaction predictors

**Estimand.** held-out drop in IOI logit difference for a head-ablation subset
**Measurement design.** ten repeated five-fold splits over anchored broad-random head masks; each single pair receives four added design ranks
**Validation target.** held-out MAE against both per-head and count-additive baselines, with add-one and leave-one-out pair contrasts
**Result status.** checked_revision

### Primary Metrics
- `n_subsets`: 160
- `n_evaluated_nonclean_subsets`: 159
- `n_prompts`: 256
- `cross_validation_folds`: 5
- `cross_validation_repeats`: 10
- `prompt_bootstrap_repeats`: 1000
- `count_additive_design_rank`: 21
- `rank_added_per_single_pair`: 4
- `additive_head_mae`: 0.3216232953780197
- `additive_head_r2`: 0.7533491967694076
- `count_additive_mae`: 0.3248973071905573
- `capacity_matched_all_pairs_mae`: 0.258671793147741
- `capacity_matched_all_pairs_r2`: 0.8361217543987266
- `all_pairs_delta_mae_vs_additive_mean`: 0.062993453361166
- `all_pairs_delta_mae_vs_additive_q05`: 0.0587918579233343
- `all_pairs_delta_mae_vs_count_mean`: 0.066145407138419
- `all_pairs_delta_mae_vs_count_q05`: 0.0629301579139464
- `PE_add_one_delta_mae_mean`: 0.0597727595889404
- `PE_add_one_delta_mae_q05`: 0.056692250543868
- `PE_add_one_delta_mae_q95`: 0.0629446219579399
- `PB_add_one_delta_mae_mean`: 0.0035466455644394
- `PB_add_one_delta_mae_q05`: 0.0028267796986535
- `PB_add_one_delta_mae_q95`: 0.0043321912973337
- `PB_add_one_vs_additive_delta_mae_mean`: 0.0003946917871865
- `PB_add_one_vs_additive_delta_mae_q05`: -0.0024140157692874
- `PB_add_one_vs_additive_delta_mae_q95`: 0.0031374184614876
- `PE_leave_one_out_delta_mae_mean`: 0.0734915017848547
- `PE_leave_one_out_delta_mae_q05`: 0.0701291660344524
- `PE_leave_one_out_delta_mae_q95`: 0.0768087183199516
- `PB_leave_one_out_delta_mae_mean`: 0.0086528455805933
- `PB_leave_one_out_delta_mae_q05`: 0.0058878663753258
- `PB_leave_one_out_delta_mae_q95`: 0.0113603527097453
- `full_pb_corner_count`: 2
- `high_pb_coverage_count`: 9
- `median_normalized_pb_exposure`: 0.08333333333333333

### Thresholds
- `main_success_requires_beating_additive_head`: True
- `main_success_requires_beating_count_additive`: True
- `paired_delta_q05_success_min`: 0.0
- `p_delta_gt_0_success_min`: 0.95
- `rank_added_per_single_pair_required`: 4

### Failure Modes Detected
- A per-head-only observer leaves held-out error captured by the capacity-matched interaction family.
- A single add-one contrast understates conditional PB contribution; leave-one-out analysis is also required.

### Recommendation
Use the capacity-matched all-pairs predictor for this intervention design. Keep per-head additivity as a strong baseline, and report PE as dominant while describing PB as conditional.

### Scope limits
- GPT-2-small IOI is a known-circuit diagnostic, not a general LLM benchmark.
- Prompt-bootstrap intervals condition on fixed masks, repeated cross-validation splits, prompt templates, and interaction bases.
- The pair ranking is conditional: add-one and leave-one-out contrasts answer different questions.
- The broad design includes two forced full-PB anchors and has limited leverage near the full PB corner.

### Commands to reproduce
- `python scripts/run_ioi_phase2_capacity.py`
- `python scripts/reproduce_paper_fast.py --only revision_ioi_stage2b`

### Reproducibility provenance
- `source_files`: {'results/revision/phase02/ioi_stage2b_capacity/model_comparison.csv': '6458a69b98b9e631e6e7f75d06c8262c46b0cfc0f28e517547f89053bd80a322', 'results/revision/phase02/ioi_stage2b_capacity/capacity_audit.csv': '7e1efbf2fc6d7371fe9da7e5df55270c08304d14b235a4b2d1b960efda32a1e4', 'results/revision/phase02/ioi_stage2b_capacity/add_one_vs_additive_head.csv': '355ea8aef080c3e5bd68421709eee2016113621b4bf29833b5c82ae348c3c318', 'results/revision/phase02/ioi_stage2b_capacity/add_one_vs_count_additive.csv': 'de7ad9c5f40209ca11488569132842af04ee8056dc2b5e51b3c8cc79254c03c2', 'results/revision/phase02/ioi_stage2b_capacity/leave_one_out_contrasts.csv': '5b726a581bf16cfff65c9a24af732b326d9e71803e25855d569f0079f328c7f3', 'results/revision/phase02/ioi_stage2b_capacity/run_manifest.json': '85680b29537c616f30461f625200d1bb719126ed2c11a91a99ceb84e4d33bceb', 'results/revision/phase02/ioi_stage2b_capacity/design_coverage.json': 'ee39fb126cfebf3d2880927fe147564b1d40a86774fb00e93f9c07e084dcec9f'}
- `source_revision`: 0cec2c4ef17d9c3b0ef3fea8fdf15347ee4e28cb
- `package_version`: 0.0.0

## ioi_stage2c / capacity_matched_all_pairs

**Task name.** ioi_stage2c
**Observer name.** capacity_matched_all_pairs
**Model or substrate.** GPT-2-small IOI head-subset intervention outputs
**Access regime.** frozen prompt-level head-ablation effects
**Observer family.** capacity-matched count-bin interaction predictors

**Estimand.** held-out drop in IOI logit difference for a head-ablation subset
**Measurement design.** ten repeated five-fold splits over primary-stratified head masks; each single pair receives four added design ranks
**Validation target.** held-out MAE against both per-head and count-additive baselines, with add-one and leave-one-out pair contrasts
**Result status.** checked_revision

### Primary Metrics
- `n_subsets`: 240
- `n_evaluated_nonclean_subsets`: 239
- `n_prompts`: 256
- `cross_validation_folds`: 5
- `cross_validation_repeats`: 10
- `prompt_bootstrap_repeats`: 1000
- `count_additive_design_rank`: 21
- `rank_added_per_single_pair`: 4
- `additive_head_mae`: 0.3836517067254424
- `additive_head_r2`: 0.7587036787914485
- `count_additive_mae`: 0.3900386920822848
- `capacity_matched_all_pairs_mae`: 0.2589825807558233
- `capacity_matched_all_pairs_r2`: 0.8763775632532121
- `all_pairs_delta_mae_vs_additive_mean`: 0.1245971554447058
- `all_pairs_delta_mae_vs_additive_q05`: 0.1189048361140709
- `all_pairs_delta_mae_vs_count_mean`: 0.1310055178087394
- `all_pairs_delta_mae_vs_count_q05`: 0.1247960658785361
- `PE_add_one_delta_mae_mean`: 0.1231590696142023
- `PE_add_one_delta_mae_q05`: 0.1168215556686451
- `PE_add_one_delta_mae_q95`: 0.1296990099775229
- `PB_add_one_delta_mae_mean`: 0.0001881919953857
- `PB_add_one_delta_mae_q05`: -0.0010558825906456
- `PB_add_one_delta_mae_q95`: 0.0014141226778053
- `PB_add_one_vs_additive_delta_mae_mean`: -0.0062201703686478
- `PB_add_one_vs_additive_delta_mae_q05`: -0.0079887836055324
- `PB_add_one_vs_additive_delta_mae_q95`: -0.0045820499336166
- `PE_leave_one_out_delta_mae_mean`: 0.1329401707400224
- `PE_leave_one_out_delta_mae_q05`: 0.1257137237050375
- `PE_leave_one_out_delta_mae_q95`: 0.1405332144336528
- `PB_leave_one_out_delta_mae_mean`: 0.0096413904559045
- `PB_leave_one_out_delta_mae_q05`: 0.008110589885892
- `PB_leave_one_out_delta_mae_q95`: 0.0110334316510107
- `direct_PE_point`: 1.848310153931379
- `direct_PB_point`: 1.0552864745259285
- `direct_PE_minus_PB_q05`: 0.7042970446869731

### Thresholds
- `main_success_requires_beating_additive_head`: True
- `main_success_requires_beating_count_additive`: True
- `paired_delta_q05_success_min`: 0.0
- `p_delta_gt_0_success_min`: 0.95
- `rank_added_per_single_pair_required`: 4

### Failure Modes Detected
- A per-head-only observer leaves held-out error captured by the capacity-matched interaction family.
- A single add-one contrast understates conditional PB contribution; leave-one-out analysis is also required.

### Recommendation
Use the capacity-matched all-pairs predictor for this intervention design. Keep per-head additivity as a strong baseline, and report PE as dominant while describing PB as conditional.

### Scope limits
- GPT-2-small IOI is a known-circuit diagnostic, not a general LLM benchmark.
- Prompt-bootstrap intervals condition on fixed masks, repeated cross-validation splits, prompt templates, and interaction bases.
- The pair ranking is conditional: add-one and leave-one-out contrasts answer different questions.

### Commands to reproduce
- `python scripts/run_ioi_phase2_capacity.py`
- `python scripts/reproduce_paper_fast.py --only revision_ioi_stage2c`

### Reproducibility provenance
- `source_files`: {'results/revision/phase02/ioi_stage2c_capacity/model_comparison.csv': '4a5d048bf3edd6bdabcea801e8aa25492b47c6c43d0cfa69634e85b37501bb19', 'results/revision/phase02/ioi_stage2c_capacity/capacity_audit.csv': '6859a7517ed59fd74f21da25e1b1c22fb28b75a20d09de1fa0c601c1abfd4481', 'results/revision/phase02/ioi_stage2c_capacity/add_one_vs_additive_head.csv': '499f1085b1316807d1475be76aab9b8b75be533a906b955fdcd91da5da03514b', 'results/revision/phase02/ioi_stage2c_capacity/add_one_vs_count_additive.csv': '35a7c6af3744db8d35793a921858207881872d580a4329277963d33db9ce7695', 'results/revision/phase02/ioi_stage2c_capacity/leave_one_out_contrasts.csv': 'a9418d4cae7c3d22e5bb8cc8ad3a0e77d14934a280354381594be244def5a469', 'results/revision/phase02/ioi_stage2c_capacity/run_manifest.json': '4bc1ef75fa1021d8b3fac728b93ceeb42ef1866f2ccb54af45dbeab3585b9bae', 'results/revision/phase02/ioi_stage2c_capacity/mobius_bootstrap_summary.csv': '2a204012e1d2acc5c356854520cf9f333867ae1909b4b6c66e75389c152daabe'}
- `source_revision`: 0cec2c4ef17d9c3b0ef3fea8fdf15347ee4e28cb
- `package_version`: 0.0.0
