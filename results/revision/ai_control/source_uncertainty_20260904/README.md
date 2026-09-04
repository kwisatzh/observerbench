# APPS source-problem uncertainty

This September 4, 2026 secondary analysis asks whether selected monitor
comparisons depend on the sampled source problems. It was specified after
the original results were known. It is not an independent confirmation.

The JSON files contain aggregate results only. Original scorecards and
archives have not been changed. Score-file hashes identify the evaluator
inputs; this directory does not release per-example labels or identifiers.

Each of 1,000 outer draws resamples 744 APPS problem pairs. The honest and
working-backdoor solution stay together, and all monitors share each draw.
The existing deployment evaluator then averages 256 inner populations at
15 attacks, 1,473 honest solutions, and 29 audits. Intervals are approximate
paired percentile intervals over source draws, conditional on the frozen
fitted observers and split. They do not include refitting or model-selection
uncertainty. The analysis also recomputes a 2,000-population point estimate;
its lexically ordered input pools can give slightly different Monte Carlo
means from the historical pool order. The paper keeps the historical means.

Run `scripts/analyze_apps_source_uncertainty.py --help` for arguments. Supply
the evaluator's score table with `--scores`, an output path with `--out`, and
each left/right monitor comparison using `--pair LEFT RIGHT`. The defaults
match these result files. The script composes the existing score reader,
AUROC implementation, deployment evaluator, and JSON writer; no model runs
or observer refits are performed.

The result is mixed: context advantages remain supported on Gemma and
Qwen3.5. The Gemma detailed-prompt/SAE action difference and the Qwen3.5
basic-prompt/text AUROC difference include zero. These two inversions remain
observations on the locked panel, not established population orderings.
