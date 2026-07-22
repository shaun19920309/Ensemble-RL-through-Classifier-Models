# Reproduction Package Manifest

This repository is a compact, publication-facing reproduction package. It
contains the code, bottom-level input panels, final run-level evidence needed
for reported uncertainty intervals, final summaries, audits, and the exact
figures used by the current paper.

## Included

- `external_data/`: DJ30, 2019 DJ30 warm-up, SSE50, and HSTech10 input panels.
- `finrl/reproduction/`: classifier voting, disagreement metrics, forecasting
  primitives, causal crossover-tau logic, and performance metrics.
- `examples/`: runners and independent audit scripts for current paper
  experiments only.
- `results/main_dj30/`, `results/external_sse50/`, and
  `results/external_hstech10/`: complete 30-refit fixed-global-tau results.
- `results/disagreement/`: final ablation tables; its exact paper figure is in
  `results/paper_figures/disagreement_metric_ablation/`.
- `results/model_family/`: final traditional/deep forecasting and DRL extension
  outputs for all three markets.
- `results/causal_tau/`: 2020 guarded and mechanism-only causal results,
  statistical summaries, and success/failure attribution.
- `results/paper_figures/`: exact figures included in the current manuscript.
- `results/data_quality/`: split audits and SHA-256 hashes for every input file.
- `unit_tests/`: focused tests for the published decision mechanisms.

## Intentionally Excluded

- The LaTeX source and compiled paper.
- Trained model checkpoints and optimizer state.
- Candidate-holding and forecast caches.
- Per-day full-grid decision traces and compressed feedback tensors.
- Smoke runs, partial-window runs, exploratory 2021 tests, multiseed pilots,
  nested-threshold experiments, and retired dynamic-threshold controllers.
- Legacy FinRL demonstration notebooks unrelated to the paper experiments.
- Local environments, machine-specific paths, and download caches.

Excluded work products are regenerated beneath `work/` by the commands in
`EXPERIMENTS.md`. Their omission does not change the final tables or figures;
it avoids publishing redundant intermediate artifacts and large checkpoints.

## Provenance

The data inventory is machine-readable in
`results/data_quality/data_file_manifest.csv`. The result directories retain
run-level metrics where they are required to reconstruct means, confidence
intervals, min-max envelopes, and paired comparisons. Reports and manifests
use repository-relative paths.
