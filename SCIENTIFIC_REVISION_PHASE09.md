# Phase 9: Qwen second-model / induction-copy mechanism study

This checklist separates code readiness from scientific outcomes. No Qwen
outcome has been inspected at the time the protocol and configs are frozen.

- [x] Keep the existing Qwen refusal-margin experiment outside the
  second-circuit claim.
- [x] Choose Qwen2.5-7B base and pin the official checkpoint revision.
- [x] Define an exact induction-copy task with disjoint token and prompt banks.
- [x] Freeze clean, architecture, hook-parity, and causal confirmation gates.
- [x] Freeze eight components and the exhaustive 256-mask design.
- [x] Put exact no-op in every action pool and report all three targets.
- [x] Freeze additive, quadratic, natural-mean, transformed-mean, and direct-risk
  comparisons before outcomes.
- [x] Implement and test deterministic prompt and mask generation.
- [x] Implement and test native-Hugging-Face discovery and mean-ablation hooks.
- [x] Implement reference means, head screening, causal confirmation, and stop
  gates.
- [x] Implement resumable calibration and locked-test measurement.
- [x] Seal predictions and actions before locked outcomes.
- [x] Fix the 128-mask additive-minus-quadratic contrast as primary; keep lower
  budgets as secondary budget curves.
- [x] Add the common-draw, equal-weight action aggregate across all targets.
- [x] Retain and validate intervention-side accuracy and target-NLL outcomes,
  plus density and prompt-cell residual diagnostics.
- [x] Add the deterministic exact-multiset matched non-induction collateral
  stage over masks selected by frozen actions; keep it secondary and non-gating.
- [x] Add the inference-free table loader and four versioned registry entries.
- [x] Add a tested external additive baseline and CSV-only evaluation path.
- [x] Run the Qwen2.5-0.5B engineering smoke; do not use it for claims.
- [x] Freeze the exact producer-source manifest and make the production runner
  reject source drift.
- [ ] Run the Qwen2.5-7B clean and causal gates on the A100.
- [ ] If and only if the causal gate passes, run the exhaustive 7B surface.
- [ ] Classify each result explicitly as positive, null, or negative before
  revising the manuscript.

## Engineering smoke record

- Checkpoint: `Qwen/Qwen2.5-0.5B` at the pinned smoke revision.
- Execution: fresh `qwen_induction_smoke_local_v3` root with cached weights and
  CPU fallback after the local PyTorch build rejected MPS at runtime; this does
  not affect the CUDA/A100 plan.
- Result: complete engineering pass. The eager scan covered 336 query heads;
  the SDPA primary zero-mask hook had exact prediction parity and zero margin
  error; all 32 measured finite-effect cells were finite.
- Tiny-bank diagnostic only: three-candidate accuracy was 1.0 on both discovery
  and head-fit banks. This is encouraging pipeline evidence, not a scientific
  claim or a substitute for the frozen 7B gates.
- Smoke-config SHA-256:
  `4f6f24970be784f04d6fb4f710694591a5504ff1c00cec8c27cb604b4fbf0f82`.
- Frozen producer-source bundle SHA-256:
  `5a36270b977dd5cf11da9048e641c8e05eec375b4b7a502b55416ed4d8bcc3ca`.
- The fresh runtime audit records the exact model revision, eager and SDPA
  implementations, dependency versions, and the original producer source
  hashes. The final 27-file source seal also covers the package import path.
- Summary SHA-256:
  `96193529a7068b74c706c1ff38ece964fce5dfadf686b7d561a1d5fc8e300aa6`.
