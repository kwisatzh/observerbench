# Phase 8: canonical IOI target sensitivity

This is a post-review, post-confirmatory sensitivity study. It does not change
the Phase-7 primary result.

The frozen package applies the same 92-column quadratic basis to three
estimands at targets 0.5, 1.0, and 1.5:

- direct expected absolute target loss;
- the natural mean-effect plug-in score;
- the target-specific transformed-mean score used as the Phase-6 parameter-
  count control.

Exact no-op is present at every target. The study reports direct risk against
all three references, target by target and as an equally weighted fixed-target
aggregate. It has no success gate.

The pre-outcome freeze selects 237 distinct nonempty masks. Hash-verified
Phase-7 rows already cover 89. The resumable run therefore measures exactly
148 new masks on 512 prompts, or 75,776 prompt-mask cells. It never remeasures
the 89 inherited masks and has no path to an unselected candidate mask.

Run the measurement from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_ioi_phase08_selected_measurement.py --device mps --pair-batch-size 128
```

Resume with the same command. After it completes, evaluate all frozen
comparisons with:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_ioi_phase08_sensitivity.py
```

The MPS command is intended for the local M4 Max. `--device cuda` uses an
available CUDA GPU with the same frozen scientific design.

