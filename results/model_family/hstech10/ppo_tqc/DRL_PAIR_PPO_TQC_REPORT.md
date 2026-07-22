# Expanded Deep-RL Pair: PPO + TQC

Experiment role: **main**.

## Protocol

- Dataset: HSTech10; `external_data/trademaster_hstech10` `test` split; 10 aligned assets.
- Evaluation: 2019-11-01 to 2020-08-31, 206 sessions in 4 blocks.
- Base agents: Proximal Policy Optimization (PPO) and Truncated Quantile Critics (TQC).
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
| 1 | ppo | 20000 | 42 |
| 1 | tqc | 100000 | 42 |
| 2 | ppo | 80000 | 42 |
| 2 | tqc | 20000 | 42 |
| 3 | ppo | 60000 | 42 |
| 3 | tqc | 20000 | 42 |
| 4 | ppo | 40000 | 42 |
| 4 | tqc | 20000 | 42 |

## Single Models

| model | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| ppo | 0.0532 | 0.7239 | 1.2611 | -0.0522 |
| tqc | 0.4756 | 1.2111 | 1.9624 | -0.3125 |

## Simple Average Control

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| ppo_tqc | 0.2644 | 1.0799 | 1.7597 | -0.1900 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_tqc | 1 | 0.4600 | 0.3379 | 1.2814 | tqc | 1.2111 | 1.0799 | 0.2015 | 0.0703 | 0.0410 | 0.0997 | 0.9667 |
| ppo_tqc | 2 | 0.4600 | 0.3625 | 1.3653 | tqc | 1.2111 | 1.0799 | 0.2854 | 0.1542 | 0.1542 | 0.1542 | 1.0000 |
| ppo_tqc | 3 | 0.0100 | 0.0716 | 0.7659 | tqc | 1.2111 | 1.0799 | -0.3140 | -0.4452 | -0.4871 | -0.4033 | 0.0000 |
| ppo_tqc | 4 | 0.4600 | 0.3439 | 1.2963 | tqc | 1.2111 | 1.0799 | 0.2163 | 0.0852 | 0.0849 | 0.0854 | 1.0000 |
| ppo_tqc | 5 | 0.4600 | 0.3533 | 1.3310 | tqc | 1.2111 | 1.0799 | 0.2511 | 0.1199 | 0.1069 | 0.1330 | 1.0000 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| ppo_tqc | 1 | 0.4600 | 3 | 45 | 0.45 | 0.47 | 1 |
| ppo_tqc | 2 | 0.4600 | 8 | 48 | 0.42 | 0.49 | 1 |
| ppo_tqc | 3 | 0.0100 | 0 | 0 | not applicable | not applicable | 38 |
| ppo_tqc | 4 | 0.4600 | 3 | 48 | 0.45 | 0.47 | 1 |
| ppo_tqc | 5 | 0.4600 | 5 | 48 | 0.44 | 0.48 | 1 |

## Common Tau Across Classifier Groups

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| ppo_tqc | 0.4100 | 0.7101 | 0.5901 | -0.6210 | -0.4898 | 0 | 0 |

## Candidate Diversity

| window | segment | samples | mean_holding_l1 | mean_dispersion | identical_holding_rate |
|---|---|---|---|---|---|
| 1 | calibration | 62 | 18144.0000 | 0.5760 | 0.0000 |
| 1 | trade | 62 | 17620.2903 | 0.5757 | 0.0000 |
| 2 | calibration | 62 | 11952.4032 | 0.5565 | 0.0000 |
| 2 | trade | 62 | 37408.8710 | 0.4649 | 0.0000 |
| 3 | calibration | 62 | 8939.0000 | 0.4902 | 0.0000 |
| 3 | trade | 62 | 52185.6774 | 0.4138 | 0.0000 |
| 4 | calibration | 62 | 17652.3871 | 0.5616 | 0.0000 |
| 4 | trade | 16 | 55184.6875 | 0.3951 | 0.0000 |

## Paired Distribution Audit

| classifier_group | wins_vs_stronger | win_rate_vs_stronger | delta_sharpe_mean | delta_sharpe_q25 | delta_sharpe_median | delta_sharpe_q75 | delta_sharpe_min | delta_sharpe_max | one_sided_sign_test_p |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 29 | 0.9667 | 0.0703 | 0.0843 | 0.0843 | 0.0853 | -0.3463 | 0.0857 | 0.0000 |
| 2 | 30 | 1.0000 | 0.1542 | 0.1542 | 0.1542 | 0.1542 | 0.1542 | 0.1542 | 0.0000 |
| 3 | 0 | 0.0000 | -0.4452 | -0.4872 | -0.4872 | -0.4872 | -0.4980 | -0.1641 | 1.0000 |
| 4 | 30 | 1.0000 | 0.0852 | 0.0843 | 0.0857 | 0.0857 | 0.0843 | 0.0857 | 0.0000 |
| 5 | 30 | 1.0000 | 0.1199 | 0.0857 | 0.1199 | 0.1542 | 0.0857 | 0.1542 | 0.0000 |

Group 1: mean paired Sharpe delta 0.070339, median 0.084339, wins 29/30, one-sided sign-test p=0.0000.
Group 2: mean paired Sharpe delta 0.154209, median 0.154209, wins 30/30, one-sided sign-test p=0.0000.
Group 3: mean paired Sharpe delta -0.445160, median -0.487154, wins 0/30, one-sided sign-test p=1.0000.
Group 4: mean paired Sharpe delta 0.085180, median 0.085666, wins 30/30, one-sided sign-test p=0.0000.
Group 5: mean paired Sharpe delta 0.119937, median 0.119937, wins 30/30, one-sided sign-test p=0.0000.

## Main Finding

At each classifier group's mean-Sharpe-maximizing global tau, PPO + TQC beats its stronger component in 4/5 groups and the simple holding average in 4/5; 4/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected after comparing complete evaluation paths, so this is a full-path sensitivity analysis rather than an out-of-sample tau-selection claim. The 30 repetitions measure classifier-refit variation conditional on one fixed checkpoint set and one realized market path; they do not measure variation across RL training seeds.
