# ObserverBench Phase-4 integration audit

All checks pass: **True**.

Checked Ctl-2 factorial, null, gain, rank, and IOI model-comparison results agree with the Phase-4 cards and current manuscript claims.

## Checked gates

- [x] `ctl2_phase01_audit_passes`
- [x] `ctl2_nonzero_fixed_direction_effects_are_6_of_6`
- [x] `ctl2_gamma_zero_factorial_null_within_1e_5`
- [x] `ctl2_gain_match_is_unclipped_but_target_worsens`
- [x] `ioi_phase02_audit_passes`
- [x] `ioi_stage2b_single_pairs_add_four_ranks`
- [x] `ioi_stage2b_all_pairs_add_twelve_ranks`
- [x] `ioi_stage2c_single_pairs_add_four_ranks`
- [x] `ioi_stage2c_all_pairs_add_twelve_ranks`
- [x] `ioi_stage2b_all_pairs_beats_both_baselines`
- [x] `ioi_stage2c_all_pairs_beats_both_baselines`
- [x] `checked_cards_cover_ctl2_and_both_ioi_designs`
- [x] `checked_cards_have_no_unknown_task`
- [x] `ctl2_card_pins_48_of_48_seed_level_contrasts`
- [x] `stage2b_card_retracts_additive_sufficiency`
- [x] `stage2c_card_records_conditional_pb`

No model is trained and no GPT-2 data are downloaded by this integration step.
