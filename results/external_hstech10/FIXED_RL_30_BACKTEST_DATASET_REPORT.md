# HSTech10 Fixed-RL 30-Backtest Experiment

## Protocol

- Universe: 10 aligned stocks.
- Evaluation split: `test`, 2019-11-01 through 2020-08-31.
- Repetitions: 30; only rolling classifier fits vary across repetitions.
- RL: one validation-selected A2C/PPO/SAC checkpoint per expanding window, trained with seed 42 and reused unchanged in every repetition.
- Inference: `deterministic=True` for validation, classifier-decision data, and trading.
- Classifiers: five fixed groups, no grid search, refitted on the immediately prior observed block.
- Thresholds: 0.01 through 0.89 by 0.01; each candidate tau is fixed across the complete evaluation path.
- Selection: the reported tau maximizes mean Sharpe across the 30 refits; ties use the smaller tau.

## Rolling Windows

| window | train_start | train_end | calibration_start | calibration_end | calibration_source | trade_start | trade_end | trade_dates |
|---|---|---|---|---|---|---|---|---|
| 1 | 2016-06-01 | 2019-08-01 | 2019-08-02 | 2019-10-31 | train_tail | 2019-11-01 | 2020-02-04 | 63 |
| 2 | 2016-06-01 | 2019-10-31 | 2019-11-01 | 2020-02-04 | previous_trade | 2020-02-05 | 2020-05-07 | 63 |
| 3 | 2016-06-01 | 2020-02-04 | 2020-02-05 | 2020-05-07 | previous_trade | 2020-05-08 | 2020-08-06 | 63 |
| 4 | 2016-06-01 | 2020-05-07 | 2020-05-08 | 2020-08-06 | previous_trade | 2020-08-07 | 2020-08-31 | 17 |

## Selected RL Checkpoints

| window | model | selected_validation_step | training_seed | sha256 |
|---|---|---|---|---|
| 1 | a2c | 100000 | 42 | fe34cdbf5629 |
| 1 | ppo | 20000 | 42 | 71f9af0a1020 |
| 1 | sac | 20000 | 42 | 2f414590089c |
| 2 | a2c | 100000 | 42 | 9d0765bd87f8 |
| 2 | ppo | 80000 | 42 | 636ee498d79d |
| 2 | sac | 20000 | 42 | 568ed5e134d3 |
| 3 | a2c | 20000 | 42 | 23fcf285a459 |
| 3 | ppo | 60000 | 42 | 7214ccf13523 |
| 3 | sac | 20000 | 42 | 8042b17b8edc |
| 4 | a2c | 60000 | 42 | 9fb2bc7cf7ee |
| 4 | ppo | 40000 | 42 | f2f432e30f4c |
| 4 | sac | 20000 | 42 | 6d2b9dc66784 |

## Single-RL Baselines

| model | cumulative_return_mean | cumulative_return_sd | sharpe_mean | sharpe_sd | calmar_mean | max_drawdown_mean |
|---|---|---|---|---|---|---|
| a2c | 0.3345 | 0.0000 | 1.1603 | 0.0000 | 1.4988 | -0.2841 |
| ppo | 0.0532 | 0.0000 | 0.7239 | 0.0000 | 1.2611 | -0.0522 |
| sac | 0.6732 | 0.0000 | 1.9675 | 0.0000 | 4.0306 | -0.2190 |

## Main Ensemble Results

| pair | classifier_group | selected_global_tau | ensemble_return_mean | ensemble_return_sd | ensemble_sharpe_mean | ensemble_sharpe_sd | ensemble_calmar_mean | ensemble_mdd_mean | stronger_component | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | wins_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2c_ppo | 1 | 0.2700 | 0.3564 | 0.0004 | 1.6768 | 0.0018 | 4.2077 | -0.1080 | a2c | 0.5164 | 0.5158 | 0.5171 | 30 |
| a2c_ppo | 2 | 0.2500 | 0.3989 | 0.0000 | 1.8918 | 0.0000 | 5.3645 | -0.0952 | a2c | 0.7314 | 0.7314 | 0.7314 | 30 |
| a2c_ppo | 3 | 0.2500 | 0.2581 | 0.1062 | 1.0344 | 0.3399 | 1.3530 | -0.2524 | a2c | -0.1260 | -0.2529 | 0.0009 | 19 |
| a2c_ppo | 4 | 0.2500 | 0.4025 | 0.0007 | 1.9825 | 0.0033 | 5.4530 | -0.0945 | a2c | 0.8222 | 0.8209 | 0.8234 | 30 |
| a2c_ppo | 5 | 0.2500 | 0.3890 | 0.0131 | 1.9060 | 0.0267 | 5.2249 | -0.0952 | a2c | 0.7456 | 0.7357 | 0.7556 | 30 |
| a2c_sac | 1 | 0.5300 | 0.5610 | 0.0216 | 1.9286 | 0.1022 | 3.3727 | -0.2161 | sac | -0.0389 | -0.0770 | -0.0007 | 18 |
| a2c_sac | 2 | 0.5300 | 0.5262 | 0.0003 | 1.8196 | 0.0014 | 2.8134 | -0.2422 | sac | -0.1479 | -0.1484 | -0.1474 | 0 |
| a2c_sac | 3 | 0.4000 | 0.3551 | 0.0462 | 1.2827 | 0.1936 | 1.8087 | -0.2570 | sac | -0.6848 | -0.7571 | -0.6125 | 0 |
| a2c_sac | 4 | 0.5300 | 0.5437 | 0.0113 | 1.8800 | 0.0395 | 3.0865 | -0.2287 | sac | -0.0875 | -0.1023 | -0.0728 | 2 |
| a2c_sac | 5 | 0.5300 | 0.5188 | 0.0033 | 1.7952 | 0.0114 | 2.7785 | -0.2417 | sac | -0.1723 | -0.1766 | -0.1681 | 0 |
| ppo_sac | 1 | 0.6100 | 0.7366 | 0.0030 | 2.4046 | 0.0126 | 9.6174 | -0.1010 | sac | 0.4370 | 0.4324 | 0.4417 | 30 |
| ppo_sac | 2 | 0.6100 | 0.4033 | 0.0069 | 2.6013 | 0.0277 | 6.4403 | -0.0802 | sac | 0.6338 | 0.6235 | 0.6442 | 30 |
| ppo_sac | 3 | 0.4800 | 0.0576 | 0.0287 | 0.7336 | 0.3144 | 1.4180 | -0.0660 | sac | -1.2339 | -1.3513 | -1.1165 | 0 |
| ppo_sac | 4 | 0.6100 | 0.6647 | 0.0167 | 2.3267 | 0.0227 | 8.4760 | -0.1028 | sac | 0.3592 | 0.3507 | 0.3677 | 30 |
| ppo_sac | 5 | 0.6100 | 0.6076 | 0.0063 | 2.2330 | 0.0074 | 7.9447 | -0.0997 | sac | 0.2655 | 0.2627 | 0.2682 | 30 |

## Configuration-Level Conclusion

| criterion | result |
|---|---|
| Mean cumulative return is higher | 5/15 |
| Mean Sharpe ratio is higher | 8/15 |
| Paired Sharpe 95% interval is entirely positive | 8/15 |
| Sharpe win rate exceeds 50% across classifier refits | 10/15 |
| Mean Sharpe exceeds the globally strongest single model | 5/15 |
| Mean Calmar ratio is higher | 8/15 |
| Mean maximum drawdown is better (closer to zero) | 11/15 |

The highest mean Sharpe is PPO + SAC Group 2 at tau=0.61: 2.6013.

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
