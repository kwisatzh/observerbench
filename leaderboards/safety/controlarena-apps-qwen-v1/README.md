# Checked external AI-control monitor results

All rows score the same 744 paired honest/verified-backdoor APPS problems. The controller audits the highest-scoring fraction. Rows are ordered by 2% audit loss, then 10% audit loss, then AUROC; close rows are not a claim of statistical separation.

| Monitor | Status | Access | AUROC | Violations @ 2% | Audit precision @ 2% | Violations @ 10% |
|---|---|---|---:|---:|---:|---:|
| Exact working-backdoor oracle | evaluator-only bound | evaluator-only verified-backdoor label | 1.000 | 715 | 1.000 | 596 |
| Qwen neutral-context activation probe | bounded post-outcome control | cached white-box residual after problem plus code; no monitoring instruction | 0.979 | 715 | 1.000 | 596 |
| Qwen verdict-adjacent activation probe | prespecified external check | cached white-box residual under the basic monitoring prompt | 0.953 | 715 | 1.000 | 596 |
| Qwen supervised score-distribution ridge | bounded post-outcome control | ten score-token log probabilities under the basic monitoring prompt | 0.830 | 715 | 1.000 | 597 |
| Supervised APPS text ridge | bounded post-outcome control | hashed word and character n-grams over problem plus code | 0.868 | 715 | 1.000 | 598 |
| Qwen prompt-score ensemble | prespecified external check | mean of two expected score-token values; two forward measurements | 0.817 | 716 | 0.966 | 598 |
| Qwen + ControlArena detailed prompt | prespecified external check | expected value over ten score-token logits; one forward measurement | 0.810 | 716 | 0.966 | 598 |
| Qwen + ControlArena basic prompt | prespecified external check | expected value over ten score-token logits; one forward measurement | 0.786 | 717 | 0.931 | 607 |

Interpretation follows the monitor identification ladder: check the output distribution, match supervision with a surface baseline, remove the monitoring request, and test the intended prevalence, budget, strata, and loss. The neutral-context result shows decodable signal; it does not by itself establish deception or hidden knowledge. The bundled post-outcome stratum diagnostic also finds audit-allocation drift across APPS families and input lengths.
