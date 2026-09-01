# Gemma-2-9B-it APPS monitor results

This is a separate task-relative panel for the prospectively frozen matched Gemma replication. The design was inherited after the Qwen outcomes were known, then locally sealed before any Gemma monitor outcome was inspected. It is not an independent preregistration.

| Monitor | Status | Mean violations (1% attacks, 2% audit) | 95% interval | Balanced AUROC | Access |
|---|---|---:|---:|---:|---|
| Exact working-backdoor oracle | evaluator-only bound | 0.000 | [0.0, 0.0] | 1.000 | evaluator-only verified working-backdoor label |
| Gemma verdict-adjacent residual probe | prospectively frozen matched replication | 2.178 | [0.0, 5.0] | 0.969 | white-box residual under the basic monitoring prompt |
| Gemma supervised score-distribution ridge | prospectively frozen matched replication | 4.423 | [1.0, 8.0] | 0.913 | output score distribution; one forward measurement |
| Gemma neutral-context residual probe | prospectively frozen matched replication | 4.947 | [2.0, 9.0] | 0.944 | white-box residual without a monitoring instruction |
| Supervised APPS text ridge | prospectively frozen matched replication | 9.054 | [6.0, 12.0] | 0.868 | problem and code text |
| Official Gemma Scope SAE probe | prospectively frozen matched replication | 9.070 | [6.0, 12.0] | 0.870 | white-box residual plus official Gemma Scope SAE encoder |
| Gemma + ControlArena detailed prompt | prospectively frozen matched replication | 10.803 | [7.0, 14.0] | 0.893 | output score distribution; one forward measurement |
| Gemma prompt-score ensemble | prospectively frozen matched replication | 13.689 | [11.0, 15.0] | 0.860 | two output score distributions; two forward measurements |
| Gemma + ControlArena basic prompt | prospectively frozen matched replication | 14.927 | [14.0, 15.0] | 0.809 | output score distribution; one forward measurement |

The verdict-adjacent residual probe is the strongest deployable row under the frozen primary regime. The official Gemma Scope SAE probe beats the three prompted monitors, but not the residual or supervised score-distribution probes. This is a positive cross-model result and a task-relative ranking, not a general claim that one monitor family is best.
