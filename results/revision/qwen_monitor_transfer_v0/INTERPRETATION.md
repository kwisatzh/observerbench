# Interpretation after the architecture review

These results are an exploratory workbench diagnostic, not an ICLR or MT
paper result. The original configuration and result files remain unchanged
to preserve the run record.

Qwen2.5 and Qwen3.5 have different architectures and training. Their shared
ten-bin output format does not imply that regression coefficients should
transfer. Failed reuse is therefore not surprising, and this experiment
does not identify architecture as its cause.

Rows called `refit` train from scratch on target-model measurements. They
form a target-only learning curve, not evidence of a transfer advantage.
The comparison with source-assisted adaptation at equal target-measurement
cost remains to be run. See [the diagnostic guide](../../../docs/monitor_transfer.md).
