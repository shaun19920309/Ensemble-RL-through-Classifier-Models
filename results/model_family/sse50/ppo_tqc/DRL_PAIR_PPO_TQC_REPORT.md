# Expanded Deep-RL Pair: PPO + TQC

Experiment role: **main**.

## Protocol

- Dataset: SSE50; `external_data/trademaster_sse50_daily` `test` split; 26 aligned assets.
- Evaluation: 2019-10-28 to 2020-08-31, 208 sessions in 4 blocks.
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
| 1 | 2016-06-01 | 2019-07-22 | 2019-07-23 | 2019-10-25 | 2019-10-28 | 2020-01-23 | 63 |
| 2 | 2016-06-01 | 2019-10-25 | 2019-10-28 | 2020-01-23 | 2020-02-03 | 2020-04-30 | 63 |
| 3 | 2016-06-01 | 2020-01-23 | 2020-02-03 | 2020-04-30 | 2020-05-06 | 2020-08-04 | 63 |
| 4 | 2016-06-01 | 2020-04-30 | 2020-05-06 | 2020-08-04 | 2020-08-05 | 2020-08-31 | 19 |

## Selected RL Checkpoints

| window | model | selected_validation_step | training_seed |
|---|---|---|---|
| 1 | ppo | 100000 | 42 |
| 1 | tqc | 20000 | 42 |
| 2 | ppo | 100000 | 42 |
| 2 | tqc | 60000 | 42 |
| 3 | ppo | 20000 | 42 |
| 3 | tqc | 20000 | 42 |
| 4 | ppo | 100000 | 42 |
| 4 | tqc | 100000 | 42 |

## Single Models

| model | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| ppo | 0.4919 | 1.8413 | 3.2957 | -0.1904 |
| tqc | 0.5006 | 1.9048 | 3.4562 | -0.1849 |

## Simple Average Control

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| ppo_tqc | 0.4961 | 1.8741 | 3.3757 | -0.1875 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_tqc | 1 | 0.2600 | 0.5108 | 1.9355 | tqc | 1.9048 | 1.8741 | 0.0614 | 0.0306 | 0.0296 | 0.0316 | 1.0000 |
| ppo_tqc | 2 | 0.2700 | 0.4981 | 1.8773 | tqc | 1.9048 | 1.8741 | 0.0033 | -0.0275 | -0.0292 | -0.0258 | 0.0000 |
| ppo_tqc | 3 | 0.1700 | 0.4982 | 1.8687 | tqc | 1.9048 | 1.8741 | -0.0053 | -0.0361 | -0.0509 | -0.0213 | 0.1333 |
| ppo_tqc | 4 | 0.2700 | 0.5150 | 1.9541 | tqc | 1.9048 | 1.8741 | 0.0801 | 0.0493 | 0.0469 | 0.0516 | 1.0000 |
| ppo_tqc | 5 | 0.2600 | 0.5080 | 1.9253 | tqc | 1.9048 | 1.8741 | 0.0513 | 0.0205 | 0.0126 | 0.0284 | 0.8000 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| ppo_tqc | 1 | 0.2600 | 64 | 66 | 0.26 | 0.89 | 64 |
| ppo_tqc | 2 | 0.2700 | 0 | 63 | not applicable | not applicable | 64 |
| ppo_tqc | 3 | 0.1700 | 0 | 0 | not applicable | not applicable | 3 |
| ppo_tqc | 4 | 0.2700 | 64 | 67 | 0.26 | 0.89 | 64 |
| ppo_tqc | 5 | 0.2600 | 64 | 65 | 0.26 | 0.89 | 64 |

## Common Tau Across Classifier Groups

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| ppo_tqc | 0.2300 | 1.8602 | 1.8508 | -0.0540 | -0.0232 | 0 | 1 |

## Candidate Diversity

| window | segment | samples | mean_holding_l1 | mean_dispersion | identical_holding_rate |
|---|---|---|---|---|---|
| 1 | calibration | 62 | 5264.1129 | 0.2844 | 0.0000 |
| 1 | trade | 62 | 4397.4032 | 0.2550 | 0.0000 |
| 2 | calibration | 62 | 6687.9355 | 0.3046 | 0.0000 |
| 2 | trade | 62 | 4020.0968 | 0.2008 | 0.0000 |
| 3 | calibration | 62 | 4804.9839 | 0.2724 | 0.0000 |
| 3 | trade | 62 | 3804.6129 | 0.1678 | 0.0000 |
| 4 | calibration | 62 | 4980.7097 | 0.2002 | 0.0000 |
| 4 | trade | 18 | 3825.3889 | 0.1937 | 0.0000 |

## Paired Distribution Audit

| classifier_group | wins_vs_stronger | win_rate_vs_stronger | delta_sharpe_mean | delta_sharpe_q25 | delta_sharpe_median | delta_sharpe_q75 | delta_sharpe_min | delta_sharpe_max | one_sided_sign_test_p |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 30 | 1.0000 | 0.0306 | 0.0297 | 0.0298 | 0.0314 | 0.0258 | 0.0385 | 0.0000 |
| 2 | 0 | 0.0000 | -0.0275 | -0.0339 | -0.0243 | -0.0243 | -0.0339 | -0.0243 | 1.0000 |
| 3 | 4 | 0.1333 | -0.0361 | -0.0635 | -0.0389 | -0.0143 | -0.1144 | 0.0386 | 1.0000 |
| 4 | 30 | 1.0000 | 0.0493 | 0.0454 | 0.0497 | 0.0560 | 0.0335 | 0.0560 | 0.0000 |
| 5 | 24 | 0.8000 | 0.0205 | 0.0025 | 0.0075 | 0.0408 | -0.0040 | 0.0521 | 0.0007 |

Group 1: mean paired Sharpe delta 0.030639, median 0.029807, wins 30/30, one-sided sign-test p=0.0000.
Group 2: mean paired Sharpe delta -0.027494, median -0.024287, wins 0/30, one-sided sign-test p=1.0000.
Group 3: mean paired Sharpe delta -0.036096, median -0.038896, wins 4/30, one-sided sign-test p=1.0000.
Group 4: mean paired Sharpe delta 0.049270, median 0.049688, wins 30/30, one-sided sign-test p=0.0000.
Group 5: mean paired Sharpe delta 0.020502, median 0.007468, wins 24/30, one-sided sign-test p=0.0007.

## Main Finding

At each classifier group's mean-Sharpe-maximizing global tau, PPO + TQC beats its stronger component in 3/5 groups and the simple holding average in 4/5; 3/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected after comparing complete evaluation paths, so this is a full-path sensitivity analysis rather than an out-of-sample tau-selection claim. The 30 repetitions measure classifier-refit variation conditional on one fixed checkpoint set and one realized market path; they do not measure variation across RL training seeds.
