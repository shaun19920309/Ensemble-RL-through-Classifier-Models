# Deep Reinforcement Learning Group 3: TD3 + TQC

## Protocol

- Dataset: DJ30; `external_data/trademaster_dj30` `valid` split; 29 aligned assets.
- Evaluation: 2020-01-02 to 2020-12-31, 253 sessions in 4 blocks.
- Base agents: Twin Delayed DDPG (TD3) and Truncated Quantile Critics (TQC).
- Each agent is freshly trained from seed 42 on the expanding history in every window for 100,000 steps.
- Calibration Sharpe is evaluated every 20,000 steps; the best prior-block checkpoint supplies both calibration and trade candidates.
- Candidate generation and trading inference use `deterministic=True`.
- The first classifier block is the training tail; each later classifier is fitted only on the immediately previous traded block.
- Five fixed classifier groups, no classifier grid search, and the original voting mechanism are used.
- Every tau is fixed across the complete path; grid 0.01-0.89 by 0.01.
- 30 repetitions refit only rolling classifiers; selected RL checkpoints and deterministic candidates remain fixed.

## Rolling Windows

| window | train_start | train_end | calibration_start | calibration_end | trade_start | trade_end | trade_dates |
|---|---|---|---|---|---|---|---|
| 1 | 2012-01-04 | 2019-10-01 | 2019-10-02 | 2019-12-31 | 2020-01-02 | 2020-04-01 | 63 |
| 2 | 2012-01-04 | 2019-12-31 | 2020-01-02 | 2020-04-01 | 2020-04-02 | 2020-07-01 | 63 |
| 3 | 2012-01-04 | 2020-04-01 | 2020-04-02 | 2020-07-01 | 2020-07-02 | 2020-09-30 | 63 |
| 4 | 2012-01-04 | 2020-07-01 | 2020-07-02 | 2020-09-30 | 2020-10-01 | 2020-12-31 | 64 |

## Selected RL Checkpoints

| window | model | selected_validation_step | training_seed |
|---|---|---|---|
| 1 | td3 | 20000 | 42 |
| 1 | tqc | 20000 | 42 |
| 2 | td3 | 20000 | 42 |
| 2 | tqc | 80000 | 42 |
| 3 | td3 | 20000 | 42 |
| 3 | tqc | 20000 | 42 |
| 4 | td3 | 20000 | 42 |
| 4 | tqc | 80000 | 42 |

## Single Models

| model | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| td3 | 0.0477 | 0.3117 | 0.1273 | -0.3748 |
| tqc | 0.1674 | 0.5761 | 0.4370 | -0.3831 |

## Simple Average Control

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| td3_tqc | 0.1076 | 0.4580 | 0.2847 | -0.3780 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| td3_tqc | 1 | 0.2000 | 0.1796 | 0.6008 | tqc | 0.5761 | 0.4580 | 0.1428 | 0.0247 | 0.0031 | 0.0463 | 0.2333 |
| td3_tqc | 2 | 0.6300 | 0.1658 | 0.5766 | tqc | 0.5761 | 0.4580 | 0.1187 | 0.0006 | 0.0001 | 0.0011 | 0.7667 |
| td3_tqc | 3 | 0.0100 | 0.0799 | 0.3800 | tqc | 0.5761 | 0.4580 | -0.0780 | -0.1961 | -0.2871 | -0.1051 | 0.3000 |
| td3_tqc | 4 | 0.6300 | 0.1521 | 0.5575 | tqc | 0.5761 | 0.4580 | 0.0995 | -0.0186 | -0.0209 | -0.0163 | 0.0000 |
| td3_tqc | 5 | 0.2000 | 0.1556 | 0.5677 | tqc | 0.5761 | 0.4580 | 0.1097 | -0.0084 | -0.0389 | 0.0221 | 0.2667 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| td3_tqc | 1 | 0.2000 | 4 | 70 | 0.2000 | 0.2300 | 4 |
| td3_tqc | 2 | 0.6300 | 27 | 69 | 0.6300 | 0.8900 | 31 |
| td3_tqc | 3 | 0.0100 | 0 | 0 | nan | nan | 19 |
| td3_tqc | 4 | 0.6300 | 0 | 68 | nan | nan | 31 |
| td3_tqc | 5 | 0.2000 | 0 | 70 | nan | nan | 29 |

## Common Tau Across Classifier Groups

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| td3_tqc | 0.2300 | 0.4627 | 0.3198 | -0.2563 | -0.1382 | 1 | 2 |

## Candidate Diversity

| window | segment | samples | mean_holding_l1 | mean_dispersion | identical_holding_rate |
|---|---|---|---|---|---|
| 1 | calibration | 62 | 10063.6290 | 0.2771 | 0.0000 |
| 1 | trade | 62 | 9429.0161 | 0.2817 | 0.0000 |
| 2 | calibration | 62 | 7514.8387 | 0.2570 | 0.0000 |
| 2 | trade | 62 | 10319.7097 | 0.2270 | 0.0000 |
| 3 | calibration | 62 | 10992.6935 | 0.3505 | 0.0000 |
| 3 | trade | 62 | 11412.1935 | 0.1945 | 0.0000 |
| 4 | calibration | 62 | 9341.8710 | 0.2544 | 0.0000 |
| 4 | trade | 63 | 11659.6349 | 0.1925 | 0.0000 |

## Paired Distribution Audit

| classifier_group | wins_vs_stronger | win_rate_vs_stronger | delta_sharpe_mean | delta_sharpe_q25 | delta_sharpe_median | delta_sharpe_q75 | delta_sharpe_min | delta_sharpe_max | one_sided_sign_test_p |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 | 0.2333 | 0.0247 | -0.0049 | -0.0036 | -0.0009 | -0.0077 | 0.1383 | 0.9993 |
| 2 | 23 | 0.7667 | 0.0006 | 0.0013 | 0.0013 | 0.0013 | -0.0018 | 0.0013 | 0.0026 |
| 3 | 9 | 0.3000 | -0.1961 | -0.3063 | -0.2644 | 0.1118 | -0.5582 | 0.1408 | 0.9919 |
| 4 | 0 | 0.0000 | -0.0186 | -0.0234 | -0.0184 | -0.0184 | -0.0235 | -0.0018 | 1.0000 |
| 5 | 8 | 0.2667 | -0.0084 | -0.0615 | -0.0532 | 0.0747 | -0.0735 | 0.1412 | 0.9974 |

Group 1 has a positive mean and Student-t interval but only 7/30 positive paired differences; its median difference is negative and a small number of high-Sharpe paths drive the mean. Group 2 has 23/30 wins, but its mean improvement over TQC is only about 0.0006 Sharpe. These two results should therefore be described as distributionally fragile and practically small, respectively.

## Main Finding

At each classifier group's mean-Sharpe-maximizing global tau, TD3+TQC beats its stronger component in 2/5 groups and the simple holding average in 4/5; 2/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected after comparing complete evaluation paths, so this is a full-path sensitivity analysis rather than an out-of-sample tau-selection claim. The 30 repetitions measure classifier-refit variation conditional on one fixed checkpoint set and one realized market path; they do not measure variation across RL training seeds.
