# Causal Crossover-Tau Results

This directory contains only the current paper's DJ30 causal experiment:
completed 2019 history initializes the controller, and 2020 is the sole
evaluation period. The threshold is selected before each 63-session block and
is then frozen throughout that block.

- `guarded/`: positive-LCB admission rule. All 450 paths fall back to the causal
  rolling-validation single-RL baseline.
- `mechanism_only/`: crossover rule without the LCB gate. It beats the causal
  baseline in 12/15 configuration means and 279/450 refit paths; it beats the
  retrospectively stronger fixed component in 7/15 configuration means.
- `statistics/`: final 15-configuration and 450-path diagnostic summaries.
- `attribution/`: inputs, table data, and Figure 15 success/failure attribution.

The compact CSVs retain final path metrics and block audits. Full daily traces,
feedback tensors, candidate holdings, and checkpoints are regenerated under
`work/` and are not publication artifacts.
