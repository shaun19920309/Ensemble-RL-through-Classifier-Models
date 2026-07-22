# DJ30 Fixed-RL 30-Backtest Experiment

## Protocol

- Universe: 29 aligned stocks.
- Evaluation split: `valid`, 2020-01-02 through 2020-12-31.
- Repetitions: 30; only rolling classifier fits vary across repetitions.
- RL: one validation-selected A2C/PPO/SAC checkpoint per expanding window, trained with seed 42 and reused unchanged in every repetition.
- Inference: `deterministic=True` for validation, classifier-decision data, and trading.
- Classifiers: five fixed groups, no grid search, refitted on the immediately prior observed block.
- Thresholds: 0.01 through 0.89 by 0.01; each candidate tau is fixed across the complete evaluation path.
- Selection: the reported tau maximizes mean Sharpe across the 30 refits; ties use the smaller tau.

## Rolling Windows

| window | train_start | train_end | calibration_start | calibration_end | calibration_source | trade_start | trade_end | trade_dates |
|---|---|---|---|---|---|---|---|---|
| 1 | 2012-01-04 | 2019-10-01 | 2019-10-02 | 2019-12-31 | train_tail | 2020-01-02 | 2020-04-01 | 63 |
| 2 | 2012-01-04 | 2019-12-31 | 2020-01-02 | 2020-04-01 | previous_trade | 2020-04-02 | 2020-07-01 | 63 |
| 3 | 2012-01-04 | 2020-04-01 | 2020-04-02 | 2020-07-01 | previous_trade | 2020-07-02 | 2020-09-30 | 63 |
| 4 | 2012-01-04 | 2020-07-01 | 2020-07-02 | 2020-09-30 | previous_trade | 2020-10-01 | 2020-12-31 | 64 |

## Selected RL Checkpoints

| window | model | selected_validation_step | training_seed | sha256 |
|---|---|---|---|---|
| 1 | a2c | 100000 | 42 | 365b243e7ac3 |
| 1 | ppo | 20000 | 42 | beca5e37dc36 |
| 1 | sac | 40000 | 42 | f35ea23df4ad |
| 2 | a2c | 80000 | 42 | 14c75a841856 |
| 2 | ppo | 80000 | 42 | b8a085506e98 |
| 2 | sac | 60000 | 42 | 90bf9b91e05e |
| 3 | a2c | 100000 | 42 | 5910dc7d728c |
| 3 | ppo | 40000 | 42 | 90dc595e0dee |
| 3 | sac | 80000 | 42 | e4578ba32baa |
| 4 | a2c | 80000 | 42 | e25b43e3ea5f |
| 4 | ppo | 80000 | 42 | 8e0e44a8e857 |
| 4 | sac | 100000 | 42 | e7af22a1eec6 |

## Single-RL Baselines

| model | cumulative_return_mean | cumulative_return_sd | sharpe_mean | sharpe_sd | calmar_mean | max_drawdown_mean |
|---|---|---|---|---|---|---|
| a2c | 0.0308 | 0.0000 | 0.2611 | 0.0000 | 0.0971 | -0.3171 |
| ppo | 0.1325 | 0.0000 | 0.5877 | 0.0000 | 0.6489 | -0.2041 |
| sac | 0.0304 | 0.0000 | 0.2810 | 0.0000 | 0.0789 | -0.3859 |

## Main Ensemble Results

| pair | classifier_group | selected_global_tau | ensemble_return_mean | ensemble_return_sd | ensemble_sharpe_mean | ensemble_sharpe_sd | ensemble_calmar_mean | ensemble_mdd_mean | stronger_component | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | wins_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2c_ppo | 1 | 0.3300 | 0.1214 | 0.0151 | 0.5517 | 0.0493 | 0.5277 | -0.2301 | ppo | -0.0360 | -0.0544 | -0.0176 | 8 |
| a2c_ppo | 2 | 0.3300 | 0.1628 | 0.0000 | 0.6617 | 0.0000 | 0.6988 | -0.2329 | ppo | 0.0740 | 0.0740 | 0.0740 | 30 |
| a2c_ppo | 3 | 0.2800 | 0.0401 | 0.0245 | 0.2865 | 0.0653 | 0.1264 | -0.3171 | ppo | -0.3012 | -0.3256 | -0.2768 | 0 |
| a2c_ppo | 4 | 0.3300 | 0.1698 | 0.0070 | 0.6892 | 0.0195 | 0.7287 | -0.2330 | ppo | 0.1014 | 0.0942 | 0.1087 | 30 |
| a2c_ppo | 5 | 0.3300 | 0.1317 | 0.0102 | 0.5717 | 0.0349 | 0.5488 | -0.2403 | ppo | -0.0161 | -0.0291 | -0.0030 | 7 |
| a2c_sac | 1 | 0.3700 | 0.1816 | 0.0401 | 0.6285 | 0.0922 | 0.5661 | -0.3225 | sac | 0.3475 | 0.3130 | 0.3819 | 30 |
| a2c_sac | 2 | 0.3900 | 0.1999 | 0.0040 | 0.6710 | 0.0100 | 0.6346 | -0.3150 | sac | 0.3900 | 0.3863 | 0.3937 | 30 |
| a2c_sac | 3 | 0.2800 | 0.0583 | 0.0599 | 0.3321 | 0.1543 | 0.1838 | -0.3171 | sac | 0.0511 | -0.0065 | 0.1087 | 8 |
| a2c_sac | 4 | 0.3700 | 0.1929 | 0.0221 | 0.6543 | 0.0518 | 0.6037 | -0.3216 | sac | 0.3733 | 0.3539 | 0.3926 | 30 |
| a2c_sac | 5 | 0.3700 | 0.1958 | 0.0039 | 0.6606 | 0.0096 | 0.6157 | -0.3180 | sac | 0.3796 | 0.3760 | 0.3832 | 30 |
| ppo_sac | 1 | 0.0100 | 0.1769 | 0.0607 | 0.7129 | 0.1833 | 0.8464 | -0.2125 | ppo | 0.1251 | 0.0567 | 0.1936 | 16 |
| ppo_sac | 2 | 0.1600 | 0.1664 | 0.0000 | 0.6933 | 0.0000 | 0.7801 | -0.2134 | ppo | 0.1055 | 0.1055 | 0.1055 | 30 |
| ppo_sac | 3 | 0.0100 | 0.1421 | 0.0302 | 0.6115 | 0.0856 | 0.6959 | -0.2041 | ppo | 0.0238 | -0.0082 | 0.0558 | 5 |
| ppo_sac | 4 | 0.1600 | 0.2112 | 0.0429 | 0.8198 | 0.1218 | 0.9842 | -0.2145 | ppo | 0.2321 | 0.1866 | 0.2776 | 30 |
| ppo_sac | 5 | 0.1600 | 0.1555 | 0.0263 | 0.6603 | 0.0791 | 0.7637 | -0.2035 | ppo | 0.0726 | 0.0431 | 0.1022 | 29 |

## Configuration-Level Conclusion

| criterion | result |
|---|---|
| Mean cumulative return is higher | 12/15 |
| Mean Sharpe ratio is higher | 12/15 |
| Paired Sharpe 95% interval is entirely positive | 10/15 |
| Sharpe win rate exceeds 50% across classifier refits | 10/15 |
| Mean Sharpe exceeds the globally strongest single model | 11/15 |
| Mean Calmar ratio is higher | 12/15 |
| Mean maximum drawdown is better (closer to zero) | 7/15 |

The highest mean Sharpe is PPO + SAC Group 4 at tau=0.16: 0.8198.

The 30 repetitions estimate sensitivity to classifier refitting conditional on fixed RL checkpoints and one realized market path. They are not 30 independent RL trainings. Tau is selected on the same completed evaluation span, so these results are a full-path sensitivity analysis rather than a deployable out-of-sample estimate.

## Audit

- Fixed checkpoints: 12 files with 12 unique SHA256 hashes.
- Classifier refits: 1800 recorded fits at the group level.
- Configuration rows: 15; Group-1 range rows: 3.
- Run-level metrics and curves are retained under `runs/` for direct recomputation.
- Figure 3 uses the observed pointwise minimum and maximum over 30 classifier refits; deterministic single-RL bounds collapse to their curves.

## Artifacts

- `base_model_30_backtest_summary.csv`: deterministic single-RL table.
- `selected_tau_summary.csv`: all 15 pair-group results at selected global tau.
- `paired_sharpe_stability.csv`: paired confidence intervals and wins over the stronger component.
- `configuration_comparison.csv` and `outperformance_summary.csv`: detailed and aggregate conclusions.
- `figure3_range_coverage.csv`: observed ensemble-envelope coverage.
- `figures/`: paper-corresponding Figures 3, 4, and 5 in PNG and PDF.
