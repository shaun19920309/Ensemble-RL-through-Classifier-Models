# Expanded Deep-RL Pair: TD3 + TQC

Experiment role: **stress test**.

## Protocol

- Dataset: SSE50; `external_data/trademaster_sse50_daily` `test` split; 26 aligned assets.
- Evaluation: 2019-10-28 to 2020-08-31, 208 sessions in 4 blocks.
- Base agents: Twin Delayed DDPG (TD3) and Truncated Quantile Critics (TQC).
- Each selected checkpoint comes from seed 42 expanding-window training for 100,000 steps per agent and window.
- Calibration Sharpe is evaluated every 20,000 steps; the best prior-block checkpoint supplies both calibration and trade candidates.
- Reused checkpoint source hashes are recorded. If serialization normalization is required, policy-tensor fingerprints must remain identical before and after migration.
- Candidate generation and trading inference use `deterministic=True`.
- The first classifier block is the training tail; each later classifier is fitted only on the immediately previous traded block.
- Five fixed classifier groups, no classifier grid search, and the original voting mechanism are used.
- Every tau is fixed across the complete path; grid 0.01-0.89 by 0.01.
- 30 repetitions refit only rolling classifiers; selected RL checkpoints and deterministic candidates remain fixed.

## Rolling Windows

| window | train_start | train_end | calibration_start | calibration_end | trade_start | trade_end | trade_dates |
|---|---|---|---|---|---|---|---|
| 1 | 2016-06-01 | 2019-07-22 | 2019-07-23 | 2019-10-25 | 2019-10-28 | 2020-01-23 | 63 |
| 2 | 2016-06-01 | 2019-10-25 | 2019-10-28 | 2020-01-23 | 2020-02-03 | 2020-04-30 | 63 |
| 3 | 2016-06-01 | 2020-01-23 | 2020-02-03 | 2020-04-30 | 2020-05-06 | 2020-08-04 | 63 |
| 4 | 2016-06-01 | 2020-04-30 | 2020-05-06 | 2020-08-04 | 2020-08-05 | 2020-08-31 | 19 |

## Selected RL Checkpoints

| window | model | selected_validation_step | training_seed |
|---|---|---|---|
| 1 | td3 | 20000 | 42 |
| 1 | tqc | 20000 | 42 |
| 2 | td3 | 20000 | 42 |
| 2 | tqc | 60000 | 42 |
| 3 | td3 | 20000 | 42 |
| 3 | tqc | 20000 | 42 |
| 4 | td3 | 20000 | 42 |
| 4 | tqc | 100000 | 42 |

## Single Models

| model | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| td3 | 0.3388 | 1.5523 | 2.5996 | -0.1641 |
| tqc | 0.5006 | 1.9048 | 3.4562 | -0.1849 |

## Simple Average Control

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| td3_tqc | 0.4191 | 1.8574 | 3.3236 | -0.1598 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| td3_tqc | 1 | 0.3900 | 0.4742 | 1.9526 | tqc | 1.9048 | 1.8574 | 0.0952 | 0.0478 | 0.0240 | 0.0716 | 0.8667 |
| td3_tqc | 2 | 0.3900 | 0.5472 | 2.1926 | tqc | 1.9048 | 1.8574 | 0.3351 | 0.2877 | 0.2838 | 0.2916 | 1.0000 |
| td3_tqc | 3 | 0.3900 | 0.3779 | 1.7204 | tqc | 1.9048 | 1.8574 | -0.1370 | -0.1844 | -0.3070 | -0.0618 | 0.3333 |
| td3_tqc | 4 | 0.3900 | 0.5459 | 2.1865 | tqc | 1.9048 | 1.8574 | 0.3290 | 0.2816 | 0.2773 | 0.2860 | 1.0000 |
| td3_tqc | 5 | 0.3900 | 0.4985 | 2.0605 | tqc | 1.9048 | 1.8574 | 0.2030 | 0.1556 | 0.1293 | 0.1819 | 1.0000 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| td3_tqc | 1 | 0.3900 | 1 | 1 | 0.39 | 0.39 | 1 |
| td3_tqc | 2 | 0.3900 | 1 | 1 | 0.39 | 0.39 | 1 |
| td3_tqc | 3 | 0.3900 | 0 | 0 | not applicable | not applicable | 1 |
| td3_tqc | 4 | 0.3900 | 1 | 1 | 0.39 | 0.39 | 1 |
| td3_tqc | 5 | 0.3900 | 1 | 1 | 0.39 | 0.39 | 1 |

## Common Tau Across Classifier Groups

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| td3_tqc | 0.3900 | 2.0225 | 1.7204 | -0.1844 | -0.1370 | 4 | 4 |

## Candidate Diversity

| window | segment | samples | mean_holding_l1 | mean_dispersion | identical_holding_rate |
|---|---|---|---|---|---|
| 1 | calibration | 62 | 24705.8387 | 0.3887 | 0.0000 |
| 1 | trade | 62 | 23439.5806 | 0.3848 | 0.0000 |
| 2 | calibration | 62 | 25802.2581 | 0.4375 | 0.0000 |
| 2 | trade | 62 | 31595.0806 | 0.3920 | 0.0000 |
| 3 | calibration | 62 | 26848.5484 | 0.4430 | 0.0000 |
| 3 | trade | 62 | 32360.5806 | 0.4015 | 0.0000 |
| 4 | calibration | 62 | 26305.6452 | 0.4457 | 0.0000 |
| 4 | trade | 18 | 33269.6667 | 0.4128 | 0.0000 |

## Paired Distribution Audit

| classifier_group | wins_vs_stronger | win_rate_vs_stronger | delta_sharpe_mean | delta_sharpe_q25 | delta_sharpe_median | delta_sharpe_q75 | delta_sharpe_min | delta_sharpe_max | one_sided_sign_test_p |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 26 | 0.8667 | 0.0478 | 0.0239 | 0.0262 | 0.0308 | -0.0151 | 0.2271 | 0.0000 |
| 2 | 30 | 1.0000 | 0.2877 | 0.2843 | 0.2843 | 0.2843 | 0.2843 | 0.3184 | 0.0000 |
| 3 | 10 | 0.3333 | -0.1844 | -0.3525 | -0.3525 | 0.2324 | -0.6059 | 0.2931 | 0.9786 |
| 4 | 30 | 1.0000 | 0.2816 | 0.2843 | 0.2843 | 0.2843 | 0.2502 | 0.3184 | 0.0000 |
| 5 | 30 | 1.0000 | 0.1556 | 0.1197 | 0.1197 | 0.1534 | 0.1172 | 0.3303 | 0.0000 |

Group 1: mean paired Sharpe delta 0.047800, median 0.026192, wins 26/30, one-sided sign-test p=0.0000.
Group 2: mean paired Sharpe delta 0.287732, median 0.284323, wins 30/30, one-sided sign-test p=0.0000.
Group 3: mean paired Sharpe delta -0.184401, median -0.352498, wins 10/30, one-sided sign-test p=0.9786.
Group 4: mean paired Sharpe delta 0.281635, median 0.284323, wins 30/30, one-sided sign-test p=0.0000.
Group 5: mean paired Sharpe delta 0.155613, median 0.119720, wins 30/30, one-sided sign-test p=0.0000.

## Main Finding

At each classifier group's mean-Sharpe-maximizing global tau, TD3 + TQC beats its stronger component in 4/5 groups and the simple holding average in 4/5; 4/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected after comparing complete evaluation paths, so this is a full-path sensitivity analysis rather than an out-of-sample tau-selection claim. The 30 repetitions measure classifier-refit variation conditional on one fixed checkpoint set and one realized market path; they do not measure variation across RL training seeds.
