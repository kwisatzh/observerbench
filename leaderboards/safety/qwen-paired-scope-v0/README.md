# Qwen2.5-7B-Instruct paired-scope safety leaderboard

Rows are generated from checked result artifacts. Status distinguishes prespecified comparisons, post-outcome controls, and reference bounds.

| Observer | Status | AUROC | Mean loss | Tail loss | Missed risk | Clean utility | Access |
|---|---|---:|---:|---:|---:|---:|---|
| Exact authorization-risk oracle | oracle bound | 1.000 | 1.740 | 10.039 | 0.346 | 1.000 | evaluator oracle |
| Qwen2.5-7B-Instruct output margin x severity | post-outcome secondary | 1.000 | 1.744 | 10.078 | 0.347 | 1.000 | output logits plus action fields |
| Full-context text label x severity | post-outcome secondary | 0.942 | 2.032 | 12.857 | 0.405 | 1.000 | full prompt text plus action fields |
| Activation label x severity | prespecified locked test | 0.954 | 2.155 | 14.026 | 0.429 | 1.000 | white-box residual plus action fields |
| Activation direct risk | prespecified locked test | 0.917 | 2.520 | 16.545 | 0.502 | 1.000 | white-box residual plus action fields |
| Action-only direct risk | prespecified locked test | 0.500 | 3.176 | 13.987 | 0.623 | 0.901 | action fields |
| Activation label | prespecified locked test | 1.000 | 3.334 | 18.182 | 0.665 | 1.000 | white-box residual plus action fields |
| Allow all | reference bound | 0.500 | 5.010 | 19.026 | 1.000 | 1.000 | none |
