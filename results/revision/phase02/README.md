# Phase 2: capacity-matched IOI interaction audit

This directory contains the Phase 2 re-analysis requested after review of the
IOI result. It reuses the saved GPT-2-small prompt-level intervention effects;
it does not rerun the model.

## Question

The earlier analysis gave each group pair one normalized-product parameter.
That basis favored pairs involving the two-head Negative Name Mover group over
pairs involving the eight-head Backup Name Mover group. Phase 2 gives every
pair the same four effective degrees of freedom.

For each group, two count indicators distinguish low and high occupancy. The
outer product of the indicators for two groups gives a four-column pair block.
The capacity audit confirms that each block adds four columns and four design
ranks to the same count-additive baseline.

## Evaluation

- Inputs: 256 prompt-level effects for each intervention mask.
- Designs: 160 anchored broad-random masks in Stage 2b and 240
  primary-stratified masks in Stage 2c.
- Prediction: ten repeated five-way subset cross-validations with identical
  splits for every model.
- Uncertainty: 1,000 joint prompt bootstrap draws. Intervention masks, splits,
  prompt template, and the count-bin basis remain fixed, so the intervals are
  conditional on those choices.
- Contrasts: add one pair block to the count-additive model, and remove one
  pair block from the all-pairs model.

## Result

Per-head additivity remains a strong baseline, but the earlier broad-random
null does not survive capacity matching. The all-pairs model improves paired
MAE over the count-additive baseline by 0.0661 [0.0629, 0.0693] in Stage 2b and
0.1310 [0.1248, 0.1373] in Stage 2c.

The Name Mover x Negative Name Mover block dominates in both designs. Its
add-one improvement is 0.0598 [0.0567, 0.0629] in Stage 2b and 0.1232 [0.1168,
0.1297] in Stage 2c. Its leave-one-out contribution is 0.0735 [0.0701, 0.0768]
and 0.1329 [0.1257, 0.1405], respectively. The Name Mover x Backup Name Mover
block contributes conditionally: it has a small or null add-one contrast but a
positive leave-one-out contrast in both designs.

Direct whole-group Mobius effects preserve the same ranking. The Name Mover x
Negative Name Mover term is 1.849 [1.774, 1.926], compared with 1.055 [0.967,
1.139] for Name Mover x Backup Name Mover; their paired difference is 0.794
[0.704, 0.881].

The broad-random design is not uniform. It contains clean, singleton, and
whole-group anchors followed by variable-density masks. Only two of 159
evaluated masks occupy the full Name Mover--Backup corner, and only nine have
at least two of three primary heads and six of eight backup heads ablated.
Primary stratification increases the predictive value of interaction terms,
but the matched basis shows that interaction signal is already present in the
broad design.

## Reproduce

From the repository root:

```sh
PYTHONPATH=src MPLBACKEND=Agg python scripts/run_ioi_phase2_capacity.py
PYTHONPATH=src MPLBACKEND=Agg python -m pytest -q tests/test_ioi_phase2_capacity.py
```

The checked summary is `IOI_PHASE02_CLAIM_AUDIT.md`. Each result subdirectory
also contains a manifest, model table, paired contrasts, capacity audit, plots,
and the underlying bootstrap summaries.
