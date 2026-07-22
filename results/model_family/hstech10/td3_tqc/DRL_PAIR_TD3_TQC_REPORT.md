# Expanded Deep-RL Pair: TD3 + TQC

Experiment role: **stress test**.

## Protocol

- Dataset: HSTech10; `external_data/trademaster_hstech10` `test` split; 10 aligned assets.
- Evaluation: 2019-11-01 to 2020-08-31, 206 sessions in 4 blocks.
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
| 1 | 2016-06-01 | 2019-08-01 | 2019-08-02 | 2019-10-31 | 2019-11-01 | 2020-02-04 | 63 |
| 2 | 2016-06-01 | 2019-10-31 | 2019-11-01 | 2020-02-04 | 2020-02-05 | 2020-05-07 | 63 |
| 3 | 2016-06-01 | 2020-02-04 | 2020-02-05 | 2020-05-07 | 2020-05-08 | 2020-08-06 | 63 |
| 4 | 2016-06-01 | 2020-05-07 | 2020-05-08 | 2020-08-06 | 2020-08-07 | 2020-08-31 | 17 |

## Selected RL Checkpoints

| window | model | selected_validation_step | training_seed |
|---|---|---|---|
| 1 | td3 | 20000 | 42 |
| 1 | tqc | 100000 | 42 |
| 2 | td3 | 20000 | 42 |
| 2 | tqc | 20000 | 42 |
| 3 | td3 | 20000 | 42 |
| 3 | tqc | 20000 | 42 |
| 4 | td3 | 20000 | 42 |
| 4 | tqc | 20000 | 42 |

## Single Models

| model | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| td3 | 0.5060 | 1.6077 | 2.8859 | -0.2267 |
| tqc | 0.4756 | 1.2111 | 1.9624 | -0.3125 |

## Simple Average Control

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| td3_tqc | 0.4907 | 1.4303 | 2.4355 | -0.2601 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| td3_tqc | 1 | 0.7000 | 0.4532 | 1.5048 | td3 | 1.6077 | 1.4303 | 0.0746 | -0.1029 | -0.1069 | -0.0990 | 0.0000 |
| td3_tqc | 2 | 0.7000 | 0.5768 | 1.5160 | td3 | 1.6077 | 1.4303 | 0.0857 | -0.0918 | -0.0918 | -0.0918 | 0.0000 |
| td3_tqc | 3 | 0.4100 | 0.5065 | 1.6090 | td3 | 1.6077 | 1.4303 | 0.1788 | 0.0013 | 0.0005 | 0.0021 | 0.2667 |
| td3_tqc | 4 | 0.7000 | 0.5753 | 1.5157 | td3 | 1.6077 | 1.4303 | 0.0854 | -0.0921 | -0.0924 | -0.0917 | 0.0000 |
| td3_tqc | 5 | 0.7000 | 0.5733 | 1.5182 | td3 | 1.6077 | 1.4303 | 0.0879 | -0.0896 | -0.0907 | -0.0885 | 0.0000 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| td3_tqc | 1 | 0.7000 | 0 | 20 | not applicable | not applicable | 20 |
| td3_tqc | 2 | 0.7000 | 0 | 20 | not applicable | not applicable | 20 |
| td3_tqc | 3 | 0.4100 | 1 | 89 | 0.41 | 0.41 | 50 |
| td3_tqc | 4 | 0.7000 | 0 | 20 | not applicable | not applicable | 20 |
| td3_tqc | 5 | 0.7000 | 0 | 20 | not applicable | not applicable | 20 |

## Common Tau Across Classifier Groups

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| td3_tqc | 0.7000 | 1.5325 | 1.5048 | -0.1029 | 0.0746 | 0 | 5 |

## Candidate Diversity

| window | segment | samples | mean_holding_l1 | mean_dispersion | identical_holding_rate |
|---|---|---|---|---|---|
| 1 | calibration | 62 | 20057.1935 | 0.6584 | 0.0000 |
| 1 | trade | 62 | 19390.8065 | 0.6543 | 0.0000 |
| 2 | calibration | 62 | 7654.9355 | 0.3646 | 0.0000 |
| 2 | trade | 62 | 37218.0645 | 0.4614 | 0.0000 |
| 3 | calibration | 62 | 6457.8226 | 0.3403 | 0.0000 |
| 3 | trade | 62 | 52640.8065 | 0.4124 | 0.0000 |
| 4 | calibration | 62 | 19174.0806 | 0.6358 | 0.0000 |
| 4 | trade | 16 | 55484.3125 | 0.3897 | 0.0000 |

## Paired Distribution Audit

| classifier_group | wins_vs_stronger | win_rate_vs_stronger | delta_sharpe_mean | delta_sharpe_q25 | delta_sharpe_median | delta_sharpe_q75 | delta_sharpe_min | delta_sharpe_max | one_sided_sign_test_p |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.0000 | -0.1029 | -0.1113 | -0.1053 | -0.0932 | -0.1155 | -0.0776 | 1.0000 |
| 2 | 0 | 0.0000 | -0.0918 | -0.0918 | -0.0918 | -0.0918 | -0.0918 | -0.0918 | 1.0000 |
| 3 | 8 | 0.2667 | 0.0013 | 0.0000 | 0.0000 | 0.0036 | 0.0000 | 0.0048 | 0.0039 |
| 4 | 0 | 0.0000 | -0.0921 | -0.0918 | -0.0918 | -0.0918 | -0.0942 | -0.0886 | 1.0000 |
| 5 | 0 | 0.0000 | -0.0896 | -0.0918 | -0.0888 | -0.0888 | -0.0928 | -0.0764 | 1.0000 |

Group 1: mean paired Sharpe delta -0.102908, median -0.105325, wins 0/30, one-sided sign-test p=1.0000.
Group 2: mean paired Sharpe delta -0.091763, median -0.091763, wins 0/30, one-sided sign-test p=1.0000.
Group 3: mean paired Sharpe delta 0.001292, median 0.000000, wins 8/30, one-sided sign-test p=0.0039.
Group 4: mean paired Sharpe delta -0.092051, median -0.091789, wins 0/30, one-sided sign-test p=1.0000.
Group 5: mean paired Sharpe delta -0.089587, median -0.088843, wins 0/30, one-sided sign-test p=1.0000.

## Main Finding

At each classifier group's mean-Sharpe-maximizing global tau, TD3 + TQC beats its stronger component in 1/5 groups and the simple holding average in 5/5; 1/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected after comparing complete evaluation paths, so this is a full-path sensitivity analysis rather than an out-of-sample tau-selection claim. The 30 repetitions measure classifier-refit variation conditional on one fixed checkpoint set and one realized market path; they do not measure variation across RL training seeds.
