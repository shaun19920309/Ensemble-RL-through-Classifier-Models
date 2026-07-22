# SSE50 Fixed-RL 30-Backtest Experiment

## Protocol

- Universe: 26 aligned stocks.
- Evaluation split: `test`, 2019-10-28 through 2020-08-31.
- Repetitions: 30; only rolling classifier fits vary across repetitions.
- RL: one validation-selected A2C/PPO/SAC checkpoint per expanding window, trained with seed 42 and reused unchanged in every repetition.
- Inference: `deterministic=True` for validation, classifier-decision data, and trading.
- Classifiers: five fixed groups, no grid search, refitted on the immediately prior observed block.
- Thresholds: 0.01 through 0.89 by 0.01; each candidate tau is fixed across the complete evaluation path.
- Selection: the reported tau maximizes mean Sharpe across the 30 refits; ties use the smaller tau.

## Rolling Windows

| window | train_start | train_end | calibration_start | calibration_end | calibration_source | trade_start | trade_end | trade_dates |
|---|---|---|---|---|---|---|---|---|
| 1 | 2016-06-01 | 2019-07-22 | 2019-07-23 | 2019-10-25 | train_tail | 2019-10-28 | 2020-01-23 | 63 |
| 2 | 2016-06-01 | 2019-10-25 | 2019-10-28 | 2020-01-23 | previous_trade | 2020-02-03 | 2020-04-30 | 63 |
| 3 | 2016-06-01 | 2020-01-23 | 2020-02-03 | 2020-04-30 | previous_trade | 2020-05-06 | 2020-08-04 | 63 |
| 4 | 2016-06-01 | 2020-04-30 | 2020-05-06 | 2020-08-04 | previous_trade | 2020-08-05 | 2020-08-31 | 19 |

## Selected RL Checkpoints

| window | model | selected_validation_step | training_seed | sha256 |
|---|---|---|---|---|
| 1 | a2c | 20000 | 42 | fc65874a337e |
| 1 | ppo | 100000 | 42 | 35aba109b2aa |
| 1 | sac | 100000 | 42 | 8205778719de |
| 2 | a2c | 80000 | 42 | 3378b66262fa |
| 2 | ppo | 100000 | 42 | 7354dd07c1e7 |
| 2 | sac | 80000 | 42 | 0a9e9e98a22b |
| 3 | a2c | 80000 | 42 | 9875ef4dacef |
| 3 | ppo | 20000 | 42 | 2e44caa0deaa |
| 3 | sac | 100000 | 42 | 631c88b5ae58 |
| 4 | a2c | 80000 | 42 | 2eed8499a1c1 |
| 4 | ppo | 100000 | 42 | d4a5b10f4fd0 |
| 4 | sac | 100000 | 42 | c4b6566cca15 |

## Single-RL Baselines

| model | cumulative_return_mean | cumulative_return_sd | sharpe_mean | sharpe_sd | calmar_mean | max_drawdown_mean |
|---|---|---|---|---|---|---|
| a2c | 0.2873 | 0.0000 | 1.2204 | 0.0000 | 1.3701 | -0.2627 |
| ppo | 0.4919 | 0.0000 | 1.8413 | 0.0000 | 3.2957 | -0.1904 |
| sac | 0.4781 | 0.0000 | 1.8335 | 0.0000 | 3.2351 | -0.1883 |

## Main Ensemble Results

| pair | classifier_group | selected_global_tau | ensemble_return_mean | ensemble_return_sd | ensemble_sharpe_mean | ensemble_sharpe_sd | ensemble_calmar_mean | ensemble_mdd_mean | stronger_component | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | wins_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2c_ppo | 1 | 0.4000 | 0.5895 | 0.0020 | 2.2247 | 0.0059 | 4.1057 | -0.1846 | ppo | 0.3834 | 0.3812 | 0.3856 | 30 |
| a2c_ppo | 2 | 0.2300 | 0.6381 | 0.0000 | 2.4574 | 0.0000 | 4.1108 | -0.2004 | ppo | 0.6161 | 0.6161 | 0.6161 | 30 |
| a2c_ppo | 3 | 0.2900 | 0.4069 | 0.1645 | 1.6448 | 0.5616 | 2.4749 | -0.2271 | ppo | -0.1965 | -0.4062 | 0.0132 | 12 |
| a2c_ppo | 4 | 0.4000 | 0.5803 | 0.0003 | 2.1983 | 0.0015 | 4.0508 | -0.1841 | ppo | 0.3570 | 0.3564 | 0.3575 | 30 |
| a2c_ppo | 5 | 0.4000 | 0.5873 | 0.0118 | 2.2189 | 0.0336 | 4.0898 | -0.1846 | ppo | 0.3776 | 0.3650 | 0.3901 | 30 |
| a2c_sac | 1 | 0.0100 | 0.5820 | 0.0022 | 2.2303 | 0.0070 | 4.1582 | -0.1799 | sac | 0.3967 | 0.3941 | 0.3994 | 30 |
| a2c_sac | 2 | 0.0100 | 0.4969 | 0.0024 | 2.0036 | 0.0068 | 3.5218 | -0.1800 | sac | 0.1701 | 0.1675 | 0.1726 | 30 |
| a2c_sac | 3 | 0.3500 | 0.3638 | 0.1225 | 1.5078 | 0.4231 | 2.2291 | -0.2268 | sac | -0.3257 | -0.4837 | -0.1677 | 7 |
| a2c_sac | 4 | 0.0100 | 0.5836 | 0.0003 | 2.2352 | 0.0010 | 4.1672 | -0.1800 | sac | 0.4017 | 0.4013 | 0.4021 | 30 |
| a2c_sac | 5 | 0.0100 | 0.5433 | 0.0486 | 2.1256 | 0.1303 | 3.8708 | -0.1799 | sac | 0.2921 | 0.2434 | 0.3408 | 30 |
| ppo_sac | 1 | 0.0100 | 0.4948 | 0.0008 | 1.8950 | 0.0025 | 3.4884 | -0.1810 | ppo | 0.0536 | 0.0527 | 0.0545 | 30 |
| ppo_sac | 2 | 0.2300 | 0.4978 | 0.0000 | 1.9095 | 0.0000 | 3.4803 | -0.1825 | ppo | 0.0682 | 0.0682 | 0.0682 | 30 |
| ppo_sac | 3 | 0.0100 | 0.4996 | 0.0154 | 1.8956 | 0.0602 | 3.4876 | -0.1833 | ppo | 0.0542 | 0.0318 | 0.0767 | 23 |
| ppo_sac | 4 | 0.2300 | 0.4910 | 0.0064 | 1.8823 | 0.0240 | 3.3891 | -0.1848 | ppo | 0.0410 | 0.0320 | 0.0499 | 30 |
| ppo_sac | 5 | 0.2300 | 0.4891 | 0.0069 | 1.8750 | 0.0262 | 3.3956 | -0.1837 | ppo | 0.0337 | 0.0239 | 0.0435 | 29 |

## Configuration-Level Conclusion

| criterion | result |
|---|---|
| Mean cumulative return is higher | 11/15 |
| Mean Sharpe ratio is higher | 13/15 |
| Paired Sharpe 95% interval is entirely positive | 13/15 |
| Sharpe win rate exceeds 50% across classifier refits | 13/15 |
| Mean Sharpe exceeds the globally strongest single model | 13/15 |
| Mean Calmar ratio is higher | 13/15 |
| Mean maximum drawdown is better (closer to zero) | 12/15 |

The highest mean Sharpe is A2C + PPO Group 2 at tau=0.23: 2.4574.

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
