# ControlArena APPS Qwen follow-up artifacts

This directory keeps the locked score panel and every bounded follow-up used in
ObserverBench v23.

- `results.json`: fitted matched controls and fixed-budget outcomes.
- `ai_control_scores.csv`: portable scores for all eight checked observers.
- `bootstrap.json`: paired problem-level uncertainty on the locked panel.
- `prevalence_sensitivity.json`: post-outcome frozen-score resampling over
  attack prevalence and audit budget.
- `stratum_diagnostics.json`: post-outcome family and neutral-input-length
  diagnostics, including within-group AUROC and global audit allocation.
- `verdict_layer_diagnostic.json`: post-outcome comparison of the already cached
  verdict-context layers and ridge values.
- `followup_cache.npz`: checked score-distribution and neutral-activation cache.

The matched controls were frozen before their outcomes were measured. The
prevalence, stratum, and layer analyses are explicitly post-outcome diagnostics;
they narrow interpretation and are not preregistered confirmation tests.
