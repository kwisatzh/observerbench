# Qwen2.5-7B paired-scope safety interlock

This sealed run asks whether an internal-state observer can allocate a fixed
block-and-escalation budget better than an observer that sees only the requested
action. Each safe/unsafe prompt pair requests the same operation on the same
resource. Only the granted resource scope changes. All workspaces and actions
are inert.

The clean gate passed: Qwen chose the correct allow/block token on every locked
test prompt, including every held-out operation/template stratum. Candidate and
top-1 accuracy were both 1.000; the median correct-token margin was 22.125.

| Observer | Mean loss | Tail loss | Risk AUROC | Severity-weighted miss | Clean utility | Worst-family loss |
|---|---:|---:|---:|---:|---:|---:|
| Allow all | 5.010 | 19.026 | 0.500 | 1.000 | 1.000 | 7.708 |
| Action-only direct risk | 3.176 | 13.987 | 0.500 | 0.623 | 0.901 | 4.528 |
| Activation label | 3.334 | 18.182 | 1.000 | 0.665 | 1.000 | 7.553 |
| Activation direct risk | 2.520 | 16.545 | 0.917 | 0.502 | 1.000 | 7.241 |
| Activation label transformed to risk | **2.155** | 14.026 | 0.954 | **0.429** | **1.000** | 5.007 |
| Exact authorization-risk oracle | 1.740 | **10.039** | 1.000 | 0.346 | 1.000 | **1.951** |

The best learned observer reduces mean protocol loss by 32.1% relative to the
stronger action-only baseline and by 57.0% relative to allowing every request.
It closes 71.1% of the action-only-to-oracle gap, keeps all benign utility, and
causes no false blocks. Directly fitting the intervention-facing risk target is
not the only route: transforming the activation label estimate with the known
severity reduces loss by another 14.5% relative to the activation direct-risk
fit.

The result also gives a useful warning. The activation label observer attains
perfect AUROC but has higher protocol loss than the two activation observers
that encode severity. Classification is therefore not enough to rank observers
for this controller. Generalization is also uneven: the best learned observer
is weaker than the action-only baseline on the doubly held-out operation and
template stratum. The aggregate control result is positive; the worst-family
result is negative and should remain visible.

The complete design, activation caches, frozen observer coefficients, locked
test results, runtime record, and hashes are under `paired_scope_v0/`. The
compressed Colab artifact is `qwen_safety_artifacts.tar.gz`.

## Post-outcome observer-boundary checks

These checks were designed after the locked-test results above were known. They
are secondary analyses, not independent confirmation.

| Observer | Mean loss | Tail loss | Risk AUROC | Severity-weighted miss | Clean utility |
|---|---:|---:|---:|---:|---:|
| Action-only direct risk | 3.176 | 13.987 | 0.500 | 0.623 | 0.901 |
| Activation label transformed to risk | 2.155 | 14.026 | 0.954 | 0.429 | 1.000 |
| Full-context text label transformed to risk | 2.032 | 12.857 | 0.942 | 0.405 | 1.000 |
| Qwen logit margin times severity | 1.744 | 10.078 | 1.000 | 0.347 | 1.000 |
| Qwen decision times severity | 1.740 | 10.039 | 1.000 | 0.346 | 1.000 |
| Exact authorization-risk oracle | 1.740 | 10.039 | 1.000 | 0.346 | 1.000 |

The Qwen logit row comes from a direct post-outcome A100 extraction of
`logit(block) - logit(allow)`, not from test-label orientation. It lowers mean
loss by 0.411 relative to the activation observer (paired 95% interval
[-0.689, -0.138]) and tail loss by 3.948 ([-5.026, -2.935]). Because the clean
decision is perfect, the binary Qwen decision times known severity exactly
matches the policy oracle.

CVaR-based calibration selects the same layer and ridge as mean-loss selection.
The activation observer's tail loss exceeds action-only by 0.039, but the paired
interval [-1.013, 1.156] spans zero. The activation observer's underlying label
score has AUROC 1.000 within every stratum. Its doubly-held-out loss comes from
cross-stratum score calibration and global budget allocation, not from failure
to separate safe and unsafe requests within that stratum.

The complete secondary artifact, paired bootstrap intervals, stratum AUROCs,
allocation diagnostics, text-baseline specification, and hashes are under
`paired_scope_v0/followup_v0/`. The direct decision-margin extraction and its
runtime manifest are under `paired_scope_v0/decision_margins/`.
