# Expanded Deep-RL Pair: PPO + TQC

Experiment role: **main**.

## Protocol

- Dataset: DJ30; `external_data/trademaster_dj30` `valid` split; 29 aligned assets.
- Evaluation: 2020-01-02 to 2020-12-31, 253 sessions in 4 blocks.
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
| 1 | 2012-01-04 | 2019-10-01 | 2019-10-02 | 2019-12-31 | 2020-01-02 | 2020-04-01 | 63 |
| 2 | 2012-01-04 | 2019-12-31 | 2020-01-02 | 2020-04-01 | 2020-04-02 | 2020-07-01 | 63 |
| 3 | 2012-01-04 | 2020-04-01 | 2020-04-02 | 2020-07-01 | 2020-07-02 | 2020-09-30 | 63 |
| 4 | 2012-01-04 | 2020-07-01 | 2020-07-02 | 2020-09-30 | 2020-10-01 | 2020-12-31 | 64 |

## Selected RL Checkpoints

| window | model | selected_validation_step | training_seed |
|---|---|---|---|
| 1 | ppo | 20000 | 42 |
| 1 | tqc | 20000 | 42 |
| 2 | ppo | 80000 | 42 |
| 2 | tqc | 80000 | 42 |
| 3 | ppo | 40000 | 42 |
| 3 | tqc | 20000 | 42 |
| 4 | ppo | 80000 | 42 |
| 4 | tqc | 80000 | 42 |

## Single Models

| model | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| ppo | 0.1325 | 0.5877 | 0.6489 | -0.2041 |
| tqc | 0.1674 | 0.5761 | 0.4370 | -0.3831 |

## Simple Average Control

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| ppo_tqc | 0.1501 | 0.5848 | 0.5140 | -0.2920 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_tqc | 1 | 0.0100 | 0.1303 | 0.5809 | ppo | 0.5877 | 0.5848 | -0.0040 | -0.0069 | -0.0098 | -0.0039 | 0.1333 |
| ppo_tqc | 2 | 0.0100 | 0.3555 | 1.0311 | ppo | 0.5877 | 0.5848 | 0.4462 | 0.4433 | 0.4422 | 0.4445 | 1.0000 |
| ppo_tqc | 3 | 0.1700 | 0.2101 | 0.7959 | ppo | 0.5877 | 0.5848 | 0.2110 | 0.2081 | 0.0920 | 0.3242 | 0.5667 |
| ppo_tqc | 4 | 0.0100 | 0.1533 | 0.6061 | ppo | 0.5877 | 0.5848 | 0.0212 | 0.0183 | 0.0180 | 0.0186 | 1.0000 |
| ppo_tqc | 5 | 0.2700 | 0.2064 | 0.7347 | ppo | 0.5877 | 0.5848 | 0.1499 | 0.1470 | 0.0987 | 0.1953 | 1.0000 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| ppo_tqc | 1 | 0.0100 | 0 | 0 | not applicable | not applicable | 16 |
| ppo_tqc | 2 | 0.0100 | 21 | 21 | 0.01 | 0.21 | 16 |
| ppo_tqc | 3 | 0.1700 | 23 | 23 | 0.01 | 0.23 | 18 |
| ppo_tqc | 4 | 0.0100 | 16 | 16 | 0.01 | 0.16 | 16 |
| ppo_tqc | 5 | 0.2700 | 67 | 67 | 0.23 | 0.89 | 62 |

## Common Tau Across Classifier Groups

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| ppo_tqc | 0.0100 | 0.7130 | 0.5527 | -0.0351 | -0.0322 | 3 | 3 |

## Candidate Diversity

| window | segment | samples | mean_holding_l1 | mean_dispersion | identical_holding_rate |
|---|---|---|---|---|---|
| 1 | calibration | 62 | 7163.0806 | 0.1742 | 0.0000 |
| 1 | trade | 62 | 6749.7419 | 0.1795 | 0.0000 |
| 2 | calibration | 62 | 7729.3226 | 0.2444 | 0.0000 |
| 2 | trade | 62 | 13108.3710 | 0.2775 | 0.0000 |
| 3 | calibration | 62 | 8069.9032 | 0.3488 | 0.0000 |
| 3 | trade | 62 | 14470.6452 | 0.2380 | 0.0000 |
| 4 | calibration | 62 | 9693.2581 | 0.2539 | 0.0000 |
| 4 | trade | 63 | 14027.8413 | 0.2125 | 0.0000 |

## Paired Distribution Audit

| classifier_group | wins_vs_stronger | win_rate_vs_stronger | delta_sharpe_mean | delta_sharpe_q25 | delta_sharpe_median | delta_sharpe_q75 | delta_sharpe_min | delta_sharpe_max | one_sided_sign_test_p |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.1333 | -0.0069 | -0.0079 | -0.0077 | -0.0077 | -0.0243 | 0.0097 | 1.0000 |
| 2 | 30 | 1.0000 | 0.4433 | 0.4439 | 0.4439 | 0.4439 | 0.4265 | 0.4439 | 0.0000 |
| 3 | 17 | 0.5667 | 0.2081 | -0.0000 | 0.0035 | 0.5226 | -0.3783 | 0.6356 | 0.2923 |
| 4 | 30 | 1.0000 | 0.0183 | 0.0172 | 0.0189 | 0.0189 | 0.0172 | 0.0189 | 0.0000 |
| 5 | 30 | 1.0000 | 0.1470 | 0.0162 | 0.1436 | 0.2845 | 0.0066 | 0.3074 | 0.0000 |

Group 1: mean paired Sharpe delta -0.006863, median -0.007684, wins 4/30, one-sided sign-test p=1.0000.
Group 2: mean paired Sharpe delta 0.443348, median 0.443929, wins 30/30, one-sided sign-test p=0.0000.
Group 3: mean paired Sharpe delta 0.208147, median 0.003477, wins 17/30, one-sided sign-test p=0.2923.
Group 4: mean paired Sharpe delta 0.018335, median 0.018880, wins 30/30, one-sided sign-test p=0.0000.
Group 5: mean paired Sharpe delta 0.146989, median 0.143590, wins 30/30, one-sided sign-test p=0.0000.

## Main Finding

At each classifier group's mean-Sharpe-maximizing global tau, PPO + TQC beats its stronger component in 4/5 groups and the simple holding average in 4/5; 4/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected after comparing complete evaluation paths, so this is a full-path sensitivity analysis rather than an out-of-sample tau-selection claim. The 30 repetitions measure classifier-refit variation conditional on one fixed checkpoint set and one realized market path; they do not measure variation across RL training seeds.
