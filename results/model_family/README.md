# Model-Family Extension Results

Each market contains four complete experiment families:

- `forecasting_group1`: ARIMA, XGBoost, and LSTM, evaluated in all three pairs.
- `forecasting_group2`: PatchTST+iTransformer.
- `ppo_tqc`: main deep-RL extension.
- `td3_tqc`: deep-RL stress test.

Every configuration uses five fixed classifier groups, 89 fixed global
thresholds, and 30 rolling classifier refits. Models are fit separately on
each market with expanding history. Final summaries, run-level metrics,
classifier audits, reports, and experiment manifests are included; forecast
caches, candidate holdings, and trained checkpoints are excluded.

Across DJ30, SSE50, and HSTech10, the ensemble beats the stronger component in
24/45 Group-1 forecasting configurations, 10/15 Group-2 configurations, 11/15
PPO+TQC configurations, and 7/15 TD3+TQC stress configurations. The combined
tables and figures are in `summary/`.
