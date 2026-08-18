# Ctl-2 Phase-1 audit

All required checks pass: **True**.

## Checked conclusions

- The 2x2 factorial separates estimator error from direction geometry.
- For every nonzero interaction strength, first-order ISE exceeds lifted ISE in all six seeds with either direction held fixed.
- Direction geometry changes the size of the failure, especially for the first-order estimator.
- A scalar held-out response calibration fixes the local response gain in this affine fixture, but it leaves initial estimation bias and can be ill-conditioned.
- Affine-span projection repairs target tracking at gamma=0 but uses a much larger direction and moves far from the observed clean residual states.
- The unclipped small-setpoint gain-matched arm still worsens true target error, so the result is not only a controller-gain artifact.
- At gamma=0.5, the thresholded large-error flag can reverse while continuous ISE remains worse in every seed; continuous errors should be primary.

## Scope

The plant is an additive final-residual loop with an affine target head. Collateral is a fitted nuisance-probe displacement. The four binary inputs yield four distinct trajectories per training seed, so seed—not repeated test row—is the uncertainty unit.

## Required checks

- [x] `gamma0_projection_target_gate`
- [x] `gamma0_projection_not_global_repair`
- [x] `gamma0_response_calibration_gate`
- [x] `default_seed0_fixed_direction_attribution`
- [x] `gain_match_realized_without_clipping`
- [x] `gain_match_does_not_repair_target`
- [x] `multiseed_estimator_effect_for_gamma_above_zero`
- [x] `gamma0p5_continuous_metric_resolves_sign_flip`

## Fixed-direction ISE effects across seeds

| gamma | support | fixed direction | positive seeds | mean FO-lifted | seed min | seed max |
|---:|---|---|---:|---:|---:|---:|
| 0.50 | unprojected | first_order | 6/6 | 5.348 | 0.01344 | 24.26 |
| 0.50 | unprojected | lifted_interaction | 6/6 | 7.632 | 2.79 | 21.24 |
| 0.50 | projected | first_order | 6/6 | 46.82 | 0.1549 | 275.8 |
| 0.50 | projected | lifted_interaction | 6/6 | 31.98 | 2.587 | 142 |
| 1.00 | unprojected | first_order | 6/6 | 43.03 | 9.089 | 78.16 |
| 1.00 | unprojected | lifted_interaction | 6/6 | 36.05 | 14.92 | 65.45 |
| 1.00 | projected | first_order | 6/6 | 184.7 | 10.74 | 337.5 |
| 1.00 | projected | lifted_interaction | 6/6 | 109.2 | 26.36 | 245.9 |
| 1.15 | unprojected | first_order | 6/6 | 60.74 | 11.71 | 133 |
| 1.15 | unprojected | lifted_interaction | 6/6 | 50.02 | 18.98 | 102.3 |
| 1.15 | projected | first_order | 6/6 | 188.1 | 13.33 | 338.4 |
| 1.15 | projected | lifted_interaction | 6/6 | 120.2 | 31.41 | 269.2 |
| 1.50 | unprojected | first_order | 6/6 | 95.73 | 15.75 | 211 |
| 1.50 | unprojected | lifted_interaction | 6/6 | 77.09 | 25.02 | 175.9 |
| 1.50 | projected | first_order | 6/6 | 190.7 | 14.04 | 322.5 |
| 1.50 | projected | lifted_interaction | 6/6 | 128.9 | 31.67 | 265.8 |
