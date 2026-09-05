# What the Gemma observers cost

This September 4, 2026 follow-up measures the existing Gemma-2-9B-it observers
on an A100-SXM4-80GB. It measures this implementation, not the fastest possible
implementation and not a new accuracy ranking.

## Main finding

Model inference dominates. For the same 64 held-out examples, the neutral dense
probe takes 5.983 seconds and the neutral SAE probe 6.038 seconds (medians of
three warm repeats). The basic and detailed prompted monitors take 8.207 and
7.631 seconds. Their prompts are longer, so this is not an equal-token test.
The SAE does not save a transformer forward pass; its sparse readout is not
evidence of lower total runtime.

## Construction and use are different costs

- Acquisition covers all 500 original fit pairs, 200 calibration pairs and
  32 held-out timing pairs: 1,464 examples. One at-limit training/calibration
  row is excluded under the original fitting rule; no held-out row is excluded.
- Collecting neutral residuals at three candidate layers takes 136.689 seconds.
  Dense layer/ridge selection and refitting take another 0.330 seconds.
- With the same residuals cached, SAE encoding, layer/ridge selection and
  refitting take 11.702 seconds. Do not count shared extraction twice.
- Warm inference uses the selected layer (20 for both probes), includes
  tokenization, the full model forward, extraction, and the encoder/readout.
  SAE encoding includes weight transfer. There is no early-exit optimization.
- Choosing an audit from 64 cached scores takes less than 0.1 ms. This is a
  separate CPU decision cost, not an alternative to measuring the scores.
- Cold model download/loading takes 65.734 seconds; three SAE downloads take
  15.385 seconds in total. Model and SAE pretraining are excluded.

Batch size is 2, the context cap is 4,096, and inference uses bfloat16/SDPA.
Full-panel median lengths are 984.5 tokens (basic), 917.5 (detailed), and 692.5
(neutral). Peak allocated GPU memory during warm measurement is 17.55--17.61
GiB including model weights. CPU memory records are process-lifetime peaks,
not independently measured per-stage peaks.

## Reproduce or inspect

`results.json` records every timing, the model revision, source hashes and run
scope. `software_versions.json`, `lengths.json` and `timings.csv` give the
environment, workload lengths and individual stages. The cost runner was
uploaded alongside checkout `a930093`; its separate source hash is authoritative
because the checkout's clean-status field does not include an untracked runner.

Rebuild the summary without model access:

```bash
python scripts/summarize_gemma_observer_costs.py \
  results/revision/gemma_observer_costs_20260904/results.json \
  --out /tmp/gemma-cost-summary.json
```

To remeasure, use [the Colab notebook](../../../notebooks/gemma_observer_costs_colab.ipynb).
Its setup and runner were executed on the A100 through Colab's command interface;
the notebook wrapper itself has not been run top-to-bottom. Gemma access requires
the user's Hugging Face license acceptance and login. APPS solutions are read
as text, never executed. The original raw archive is preserved privately;
this directory contains no activation caches, fitted readout weights or keys.
