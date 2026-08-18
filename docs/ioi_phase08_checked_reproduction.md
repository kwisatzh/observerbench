# Checked Study-4 target-sensitivity path

The accepted post-confirmatory sensitivity artifacts use the `v2` audit and
measurement roots:

- `results/revision/phase08/ioi_target_sensitivity/preoutcome_audit_v2`
- `results/revision/phase08/ioi_target_sensitivity/new_measurement_v2`
- `results/revision/phase08/ioi_target_sensitivity/evaluation_v2`

The command-line defaults retain the unused `v1` path because the accepted
pre-outcome audit hashes those source files. Use the explicit paths below when
checking or reproducing the accepted result.

```sh
PYTHONPATH=src .venv-ioi/bin/python scripts/audit_ioi_phase08_preoutcome.py \
  --outdir results/revision/phase08/ioi_target_sensitivity/preoutcome_audit_v2
```

The next command runs model inference for the 148-mask frozen set difference.
On an NVIDIA runtime, use `--device cuda`; on Apple Silicon, use `--device mps`.

```sh
PYTHONPATH=src .venv-ioi/bin/python scripts/run_ioi_phase08_selected_measurement.py \
  --audit-dir results/revision/phase08/ioi_target_sensitivity/preoutcome_audit_v2 \
  --outdir results/revision/phase08/ioi_target_sensitivity/new_measurement_v2 \
  --device cuda \
  --pair-batch-size 128
```

Evaluation uses the accepted audit and measurement roots explicitly:

```sh
PYTHONPATH=src .venv-ioi/bin/python scripts/evaluate_ioi_phase08_sensitivity.py \
  --audit-dir results/revision/phase08/ioi_target_sensitivity/preoutcome_audit_v2 \
  --measurement-dir results/revision/phase08/ioi_target_sensitivity/new_measurement_v2 \
  --outdir results/revision/phase08/ioi_target_sensitivity/evaluation_v2
```

The earlier `preoutcome_audit`, `new_measurement`, and aborted-run record remain
as provenance. They do not contribute outcomes to `evaluation_v2` or to the
paper artifact.
