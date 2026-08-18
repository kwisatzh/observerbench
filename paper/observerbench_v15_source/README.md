# ObserverBench manuscript source

This folder reconstructs the editable ObserverBench manuscript from
`observerbench_paper_v14_cybergate.pdf` and the repository's frozen figures.
The original TeX source was not present in the repository, its Git history, or
the handoff bundle.

`observerbench.tex` is the current manuscript. It includes the reachable-
subspace and finite-horizon results, the Phase 5 nonlinear-suffix experiment,
the fresh held-out IOI prediction and fixed-action analysis, the clean-gated
no-op-inclusive IOI confirmation, and the prospectively frozen Qwen2.5-7B
Copy-v2 study. The latter adds a second checked finite-effect registry and a
second pretrained-model intervention surface.
`observerbench_phase01.tex` preserves the earlier Phase 0--1 checkpoint.

Build from this directory with:

```sh
make
```

The default build regenerates and validates the checked Phase 4 artifacts, the
Phase 5 nonlinear-suffix artifact under `paper/generated_phase05/`, and the
Phase 6 and Phase 7 IOI confirmation artifacts under `paper/generated_phase06/`
and `paper/generated_phase07/`. It then writes the PDF to
`build/observerbench.pdf`. Activate the repository
environment before running `make`. Use `make artifacts` to refresh the checked
inputs without compiling the PDF, or `make phase01` to rebuild the checkpoint
as `build/observerbench_phase01.pdf` without changing its source.

The `sections/*_phase01.tex` files preserve earlier checkpoints where they are
not used by the current manuscript. The unqualified section files contain the
current text.
