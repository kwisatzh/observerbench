# Checked external AI-control monitor results

This is the initial prespecified panel. The current checked panel, including
matched supervised controls and the neutral-context probe, is in
`../controlarena-apps-qwen-v1/`.

All rows score the same 744 paired honest/verified-backdoor APPS problems. The controller audits the highest-scoring fraction.

| Monitor | Access | AUROC | Violations @ 2% | Audit precision @ 2% | Violations @ 10% |
|---|---|---:|---:|---:|---:|
| Qwen residual activation probe | cached white-box residual | 0.953 | 715 | 1.000 | 596 |
| Qwen + ControlArena detailed prompt | prompt text; one forward measurement | 0.810 | 716 | 0.966 | 598 |
| Qwen prompt-score ensemble | prompt text; two forward measurements | 0.817 | 716 | 0.966 | 598 |
| Qwen + ControlArena basic prompt | prompt text; one forward measurement | 0.786 | 717 | 0.931 | 607 |
