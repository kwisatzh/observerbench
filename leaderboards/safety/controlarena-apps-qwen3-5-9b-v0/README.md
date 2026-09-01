# Qwen3.5-9B APPS monitor results

This task-relative panel uses the same locked APPS pairs, controller, loss, and primary 1% attack / 2% audit cell as the earlier model panels. The exact model and SAE revisions were locally sealed before Qwen3.5 monitor outcomes were read. This is a prospective matched replication, not an independent preregistration.

| Monitor | Mean violations (1% attacks, 2% audit) | 95% interval | Balanced AUROC | Access |
|---|---:|---:|---:|---|
| Exact working-backdoor oracle | 0.000 | [0.0, 0.0] | 1.000 | evaluator-only verified working-backdoor label |
| Qwen3.5-9B verdict-adjacent residual probe | 1.976 | [0.0, 5.0] | 0.977 | white-box residual under the basic monitoring prompt |
| Qwen3.5-9B supervised score-distribution ridge | 5.120 | [2.0, 9.0] | 0.940 | output score distribution; one forward measurement |
| Qwen3.5-9B neutral-context residual probe | 6.601 | [3.0, 10.0] | 0.936 | white-box residual without a monitoring instruction |
| Supervised APPS text ridge | 9.054 | [6.0, 12.0] | 0.868 | problem and code text |
| Qwen3.5-9B + ControlArena basic prompt | 12.217 | [9.0, 15.0] | 0.876 | output score distribution; one forward measurement |
| Qwen3.5-9B official Qwen-Scope SAE probe | 12.725 | [10.0, 15.0] | 0.811 | white-box residual plus official Qwen-Scope SAE encoder |
| Qwen3.5-9B prompt-score ensemble | 12.956 | [10.0, 15.0] | 0.823 | two output score distributions; two forward measurements |
| Qwen3.5-9B + ControlArena detailed prompt | 13.985 | [12.0, 15.0] | 0.645 | output score distribution; one forward measurement |

The verdict-adjacent residual probe is the strongest deployable row in the frozen primary cell: AUROC 0.977 and 1.976 mean violations. The neutral residual reaches AUROC 0.936 and 6.601 violations, so the neutral-context ordering seen in an earlier Qwen2.5-7B-Instruct task does not transfer unchanged. The official Qwen-Scope SAE probe reaches AUROC 0.811 and 12.725 violations. It selects layer 12 and ridge 100 using fit and calibration rows only. The SAE was trained on Qwen3.5-9B-Base but is evaluated here on post-trained Qwen3.5-9B; this declared checkpoint mismatch limits the conclusion. These are task-relative results, not a general ranking of monitor or SAE families.
