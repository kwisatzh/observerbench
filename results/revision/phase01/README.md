# Phase-1 checked results

These outputs revise the Ctl-1/Ctl-2 controls without changing
`results/frozen/`. Each Ctl-2 run contains its effective configuration, source
hashes, Git revision, dependency versions, aggregate results, four-state
archetype summaries, and a run manifest.

Checked outputs:

- `ctl1_analytic_manifold_seed0/`: manifold-respecting analytic Ctl-1 check.
- `ctl2_gamma0_seed0_v3/`: gamma-zero 3x3 factorial, response calibration,
  gain matching, and affine-span projection diagnostics.
- `ctl2_gamma115_seed0_v3/`: the same diagnostic design at the paper's default
  interaction strength.
- `ctl2_gamma0_gainmatch_small_offset_seed0_v2/`: small relative-setpoint
  gain-matching check with no clipping.
- `ctl2_multiseed_sweep/`: six training seeds at gamma 0, 0.5, 1.0, 1.15,
  and 1.5. It uses the primary 2x2 estimator--direction factorial, both
  direction modes, and finite-response calibration. Repeated test rows are not
  treated as independent observations.
- `CTL2_PHASE01_AUDIT.md` and `ctl2_phase01_audit.json`: claim-facing audit of
  the checked runs.

Reproduce the checked seed-zero runs:

```bash
observerbench run trained_ctl2 \
  --config configs/revision/trained_ctl2_phase01_gate_v3.yaml \
  --outdir runs/phase01/ctl2_gamma0_seed0_v3

observerbench run trained_ctl2 \
  --config configs/revision/trained_ctl2_phase01_default_v3.yaml \
  --outdir runs/phase01/ctl2_gamma115_seed0_v3

observerbench run trained_ctl2 \
  --config configs/revision/trained_ctl2_phase01_gainmatch_unsaturated_v2.yaml \
  --outdir runs/phase01/ctl2_gamma0_gainmatch_small_offset_seed0_v2
```

Reproduce or resume the compact sweep and rerun its audit:

```bash
python scripts/run_ctl2_phase01_sweep.py
python scripts/run_ctl2_phase01_sweep.py --resume
python scripts/audit_ctl2_phase01.py
```

The seed bands in the sweep figures are min--max ranges across six model
training seeds. They are not 5--95 percentile intervals.

The locally retained pre-v3 exploratory directories are ignored because two
per-step CSVs total roughly 1.2 GB. They are superseded by the compact checked
outputs above.
