# Experiment Commands

Run every command from the repository root after installing the environment in
`README.md`. Intermediate checkpoints and daily traces are written under
`work/`, which is deliberately excluded from version control.

## Shared Protocol

- RL candidates are trained independently for each dataset.
- RL training history expands at each 63-session window.
- The best checkpoint is chosen on the immediately preceding 63 sessions.
- Policy inference is deterministic.
- Classifiers use only the immediately preceding block and are refitted 30
  times with fixed model definitions and voting membership.
- Each candidate threshold in `0.01, ..., 0.89` is fixed over the full path.
- Fixed-threshold tables report the threshold with the largest mean Sharpe.

## 1. Main A2C/PPO/SAC Experiment

Generate expanding-window RL candidates for DJ30:

```bash
PYTHONPATH=. python examples/reproduce_classifier_ensemble.py \
  --mode rolling --data-source trademaster \
  --trademaster-data-dir external_data/trademaster_dj30 \
  --trademaster-trade-split valid \
  --output-dir work/core_dj30_candidates \
  --timesteps 100000 --rl-eval-interval 20000 \
  --groups 1,2,3,4,5 --pairs a2c_sac,ppo_sac,a2c_ppo \
  --tau-start 0.01 --tau-stop 0.89 --tau-step 0.01
```

Run the 30 classifier-refit paths and regenerate Figures 3--5:

```bash
PYTHONPATH=. python examples/run_fixed_rl_30_backtests.py \
  --data-dir external_data/trademaster_dj30 \
  --fixed-run-dir work/core_dj30_candidates \
  --output-dir work/main_dj30_rebuild \
  --trade-split valid --dataset-label DJ30 \
  --repetitions 30 \
  --rl-inference-mode deterministic --timesteps 100000
```

Repeat both commands for SSE50 and HSTech10, changing the dataset, label, and
output directory and using `--trademaster-trade-split test` in the first
command and `--trade-split test` in the second. The published compact outputs
are in `results/external_sse50/` and `results/external_hstech10/`.

## 2. Holding-Disagreement Ablation

After generating a dataset's candidates and fixed-RL result above:

```bash
PYTHONPATH=. python examples/run_disagreement_metric_ablation.py \
  --data-dir external_data/trademaster_dj30 \
  --fixed-run-dir work/core_dj30_candidates \
  --v1-result-dir work/main_dj30_rebuild \
  --output-dir work/disagreement_dj30 \
  --dataset-label DJ30 --trade-split valid \
  --repetitions 30
```

Use the corresponding external candidate/result directories, `test` split,
and labels for SSE50 and HSTech10. Combine the three runs with:

```bash
PYTHONPATH=. python examples/summarize_disagreement_metric_ablation.py \
  --dj30-dir work/disagreement_dj30 \
  --sse50-dir work/disagreement_sse50 \
  --hstech10-dir work/disagreement_hstech10 \
  --output-dir work/disagreement_summary
```

## 3. Forecasting Model Families

Representative forecasting models (ARIMA, XGBoost, LSTM):

```bash
PYTHONPATH=. python examples/run_forecasting_group1.py \
  --experiment-group group1 \
  --data-dir external_data/trademaster_dj30 --trade-split valid \
  --dataset-label DJ30 --output-dir work/model_family/dj30/forecasting_group1 \
  --repetitions 30
```

Deep forecasting models (PatchTST and iTransformer):

```bash
PYTHONPATH=. python examples/run_forecasting_group2.py \
  --data-dir external_data/trademaster_dj30 --trade-split valid \
  --dataset-label DJ30 --output-dir work/model_family/dj30/forecasting_group2 \
  --repetitions 30
```

Repeat with the SSE50 and HSTech10 datasets and the `test` split. The full
hyperparameters are recorded by each runner's `--help` output and experiment
manifest.

## 4. Deep-RL Extensions

PPO+TQC is the main extension; TD3+TQC is retained as a stress test:

```bash
PYTHONPATH=. python examples/run_rl_group3.py \
  --models ppo tqc --experiment-role main \
  --data-dir external_data/trademaster_dj30 --trade-split valid \
  --dataset-label DJ30 --output-dir work/model_family/dj30/ppo_tqc \
  --timesteps 100000 --eval-interval 20000 --repetitions 30

PYTHONPATH=. python examples/run_rl_group3.py \
  --models td3 tqc --experiment-role stress_test \
  --data-dir external_data/trademaster_dj30 --trade-split valid \
  --dataset-label DJ30 --output-dir work/model_family/dj30/td3_tqc \
  --timesteps 100000 --eval-interval 20000 --repetitions 30
```

Repeat for SSE50 and HSTech10 with the `test` split. Audit all final roots:

```bash
PYTHONPATH=. python examples/audit_model_family_results.py \
  work/model_family/dj30/forecasting_group1 \
  work/model_family/dj30/forecasting_group2 \
  work/model_family/dj30/ppo_tqc \
  work/model_family/dj30/td3_tqc
```

## 5. Causal 2020 Crossover Tau

Generate the 2019 collection paths with independently trained, expanding-window
agents:

```bash
while read -r left right; do
  PYTHONPATH=. python examples/run_rl_group3.py \
    --models "$left" "$right" --experiment-role main \
    --data-dir external_data/trademaster_dj30_exante_2019 \
    --trade-split valid --dataset-label DJ30-2019 \
    --output-dir "work/causal_candidates/2019/${left}_${right}" \
    --timesteps 100000 --eval-interval 20000 --repetitions 30
done <<'PAIRS'
a2c ppo
a2c sac
ppo sac
PAIRS
```

The 2020 candidate paths reuse the audited DJ30 core checkpoints generated in
Section 1. Runtime reserialization preserves policy tensors while normalizing
checkpoint metadata:

```bash
while read -r left right; do
  PYTHONPATH=. python examples/run_rl_group3.py \
    --models "$left" "$right" --experiment-role main \
    --model-source "$left=work/core_dj30_candidates" \
    --model-source "$right=work/core_dj30_candidates" \
    --normalize-imported-checkpoints \
    --data-dir external_data/trademaster_dj30 \
    --trade-split valid --dataset-label DJ30-2020 \
    --output-dir "work/causal_candidates/2020/${left}_${right}" \
    --timesteps 100000 --eval-interval 20000 --repetitions 30
done <<'PAIRS'
a2c ppo
a2c sac
ppo sac
PAIRS
```

Run the guarded and mechanism-only variants:

```bash
PYTHONPATH=. python examples/run_rl_crossover_tau.py \
  --candidate-root work/causal_candidates \
  --output-dir work/causal_tau_guarded --repetitions 30

PYTHONPATH=. python examples/run_rl_crossover_tau.py \
  --candidate-root work/causal_candidates \
  --output-dir work/causal_tau_mechanism_only \
  --disable-lcb-guardrail --repetitions 30
```

Only completed 2019 history is used to initialize the first threshold; every
2020 block threshold is frozen before that block begins. No 2021 observation
enters the published experiment.

Generate the comparison and retrospective attribution tables:

```bash
PYTHONPATH=. python examples/analyze_rl_crossover_tau_results.py \
  --guarded-results work/causal_tau_guarded \
  --mechanism-only-results work/causal_tau_mechanism_only \
  --output-dir work/causal_tau_comparison

PYTHONPATH=. python examples/analyze_2020_crossover_statistics.py \
  --results-dir work/causal_tau_mechanism_only \
  --candidate-root work/causal_candidates/2020 \
  --mechanism-validation \
    work/causal_tau_comparison/next_block_mechanism_validation.csv \
  --output-dir work/causal_tau_statistics

PYTHONPATH=. python examples/analyze_2020_success_failure_causes.py \
  --daily-decisions \
    work/causal_tau_mechanism_only/all_daily_crossover_tau_decisions.csv \
  --block-audit work/causal_tau_mechanism_only/all_block_tau_audit.csv \
  --run-metrics work/causal_tau_mechanism_only/all_crossover_tau_metrics.csv \
  --config-features work/causal_tau_statistics/configuration_features_15.csv \
  --candidate-root work/causal_candidates/2020 \
  --output-dir work/causal_tau_attribution
```

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q \
  unit_tests/test_classifier_ensemble.py \
  unit_tests/test_disagreement.py \
  unit_tests/test_forecasting_ensemble.py \
  unit_tests/test_causal_crossover_tau.py

PYTHONPATH=. python examples/build_data_manifest.py

for result in results/main_dj30 results/external_sse50 results/external_hstech10; do
  PYTHONPATH=. python examples/audit_fixed_rl_30_results.py \
    --result-dir "$result" --allow-omitted-checkpoints
done
```

The exact manuscript figures are preserved under `results/paper_figures/` so
they can be compared byte-for-byte with a rebuilt artifact where PDF metadata
is deterministic.
