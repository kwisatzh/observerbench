# Cross-model reuse diagnostic and target-only learning curve

This is a workbench diagnostic, not a paper contribution or an RSI result.
It checks one fitted output-score rule across two different architectures,
then measures how a new rule learns from target-model data.

Qwen2.5 and Qwen3.5 differ in architecture, training, and prompt handling.
We should not expect the same regression coefficients to work unchanged.
The shared ten-bin output format makes reuse possible to test; it does not
make it statistically justified. This comparison cannot isolate the cause
of failure among those differences.

## The experiment

We reuse measurements from Qwen2.5-7B-Instruct and Qwen3.5-9B on identical
APPS examples. Both measurements are the basic monitor prompt's ten
candidate-normalized log probabilities for scores 0 through 9. The models
use their own prompt adapters. No hidden-state coordinate mapping is assumed.

We fit a ridge observer on Qwen2.5 measurements and apply it unchanged to
Qwen3.5 measurements. Then we train from scratch with 8, 32, 128, 512, or 700 target-model
problem pairs. A pair contains one honest and one stored-backdoor solution.
Ten seeds choose nested subsets of the 700 available training problems.
Held-out problems never enter fitting or ridge selection.

The controller audits the highest-scored 2% of 1,488 requests. Each observer
faces the same 2,000 population draws at 1% and 5% attack prevalence. These
contain 15 and 74 attacks and permit 29 audits. An oracle therefore cannot
do better than 0 and 45 missed attacks, respectively.

## Results on Qwen3.5-9B

| Observer | Target training pairs | AUROC | Missed attacks at 1% | Missed attacks at 5% |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 fit, unchanged | 0 | 0.293 | 15.000 | 73.994 |
| Raw Qwen3.5 prompted score | 0 | 0.876 | 12.154 | 62.831 |
| Target refit | 8 | 0.920 | 7.655 | 52.319 |
| Target refit | 32 | 0.932 | 6.043 | 48.635 |
| Target refit | 128 | 0.938 | 5.166 | 46.081 |
| Target refit | 512 | 0.940 | 5.120 | 45.282 |
| Target refit | 700 | 0.940 | 5.121 | 45.245 |
| Target fit, original calibration split | 700 | 0.940 | 5.088 | 45.226 |

Refit rows average ten fitting seeds. With 128 pairs, mean violations at
1% range from 5.044 to 5.521 across seeds; with 8 pairs, the range is
5.207 to 10.774. These are descriptive ranges, not confidence intervals
over new task populations. The unchanged source observer has AUROC 0.830
on its own model, so the transfer failure is not an already-useless source fit.

The result has two separate parts: this fitted score rule did not transfer,
and a target-only learning curve approached the full-data reference on this
held-out population. The target refits do not reuse source coefficients or
source measurements, so they establish no benefit from transfer. We did not
establish a universal 128-pair requirement. Nor can a positive affine
rescaling repair the failed ranking: it
preserves score order, so a fixed top-budget audit selects the same requests.

ObserverBench's evaluation contract works across these models without
requiring an individual fitted observer to transfer. A stronger MT experiment
would compare source-assisted adaptation, target-only fitting, and alternative
measurement views at equal target-measurement budgets, including a
same-architecture update. That experiment remains to be done.

## Reproduce or extend

[`configs/revision/qwen_monitor_transfer_v0.json`](../configs/revision/qwen_monitor_transfer_v0.json)
records the model views, adaptation budgets, seeds, and fitting rules.
[`scripts/run_qwen_monitor_transfer.py`](../scripts/run_qwen_monitor_transfer.py)
composes the existing safety evaluator with a ten-feature ridge fit.

The script needs the original measurement caches, which contain private
per-example evaluation data and are not bundled in the public repository.
It is an analysis runner, not a self-contained public demo. The
[integrity task](evidence_integrity.md) is the new fully local demo.

```bash
python -m pip install -e '.[sae]'
python scripts/run_qwen_monitor_transfer.py \
  --outdir runs/monitor-transfer \
  --private-outdir /path/outside/public-repo/transfer-parameters
```

Public aggregate results and hashes are under
[`results/revision/qwen_monitor_transfer_v0`](../results/revision/qwen_monitor_transfer_v0).
Fitted parameters and per-example predictions stay outside the public results.
The pair budget counts target measurements reused for adaptation, not fresh
GPU runs or newly acquired labels. This is a secondary analysis of existing
checkpoints on one stored task, not a self-improvement or adaptive-attack study.
