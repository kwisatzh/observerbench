# Does changing the fitting data change the decision comparison?

This follow-up refits four IOI observers on 40 training halves. It keeps name
pairs together, uses both halves of each of 20 random splits, and leaves the
measurement masks, held-out prompts, action pools, targets, and selector fixed.
Each half has 48 name pairs and 89--103 prompts.

The interaction model improves mean prediction in every fit. At target 1,
however, it chooses a worse action than the additive model in 28 of 40 fits.
Scalar calibration also improves AtP prediction in every fit, while action
quality depends on the requested target. At target 1.5 it worsens AtP action
loss in all 40 fits; at target 0.5 it improves action loss in 31 of 40.

These are repeated splits of **one population**, not 40 independent datasets.
The original Phase-5 clean-task limitation remains. This analysis is separate
from the later clean-task confirmation, and its percentile ranges are
descriptive rather than confidence intervals over independent experiments.

## Reproduce

From the repository root:

```bash
python scripts/run_ioi_refit_sensitivity.py
```

The included training effects and per-prompt gradient cache allow CPU-only
refitting with NumPy, pandas, and Matplotlib. No model is loaded when both
caches are present. The script saves all predictions and selected actions
before opening the held-out responses in the separate decision replay pack.

To remeasure gradients, use a separate output directory, the original Phase-5
measurement archive, and the existing TransformerLens environment. The pinned
model revision, device, and software versions are recorded. Do not delete or
overwrite the original archive to rerun this analysis.

## Files

- `protocol.json`: the post-outcome analysis plan and split seeds.
- `training_effects.npz`: original training responses for the 160 masks.
- `training_input_provenance.json`: their source-shard hashes.
- `per_prompt_attribution.npz`: one 13-head gradient map per training prompt.
- `predictions.csv` and `selected_actions.csv`: all frozen refits and choices.
- `refit_scores.csv` and `results.json`: scores and paired descriptive changes.
- `fits.csv`: population counts, gains, and cached fit/selection times.
- `gradient_measurement_cost.json`: actual local gradient measurement time,
  **excluding** finite-effect acquisition and reference-mean construction.

The separate Gemma A100 experiment measures larger-model construction and use.
Neither this cached refit time nor its coefficient count measures that cost.
