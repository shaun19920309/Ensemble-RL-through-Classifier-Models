# Modern Deep Forecasting Model Ensemble Experiment

## Protocol

- Dataset: DJ30; source `external_data/trademaster_dj30`; trade split `valid`; 29 assets are present in the supplied files.
- Evaluation span: 2020-01-02 to 2020-12-31, 253 sessions in 4 rolling blocks (63, 63, 63, 64 sessions).
- Base models: one-step supervised adaptations of PatchTST with channel-independent patch tokens and iTransformer with variate tokens.
- All models predict the next-session cross-section of log returns from close-price history only.
- Common portfolio map: trailing-20-session volatility scaling, cross-sectional softmax, maximum weight 0.20, gross exposure 0.95.
- Forecast-model fitting uses an expanding historical window. Classifiers use only the immediately preceding rolling calibration block.
- Every tau is fixed for a complete path; grid 0.01-0.89 by 0.01.
- Repetitions: 30; candidate forecasts and holdings remain fixed while rolling classifiers are refitted.
- Classifier groups and voting are unchanged from the RL experiment; no classifier grid search is used.

## Single Models

| model | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| itransformer | 0.0530 | 0.3224 | 0.2014 | -0.2631 |
| patchtst | 0.0318 | 0.2503 | 0.1243 | -0.2560 |

## Simple Average Controls

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| patchtst_itransformer | 0.0668 | 0.3742 | 0.2624 | -0.2544 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 1 | 0.2400 | 0.1108 | 0.5526 | itransformer | 0.3224 | 0.3742 | 0.1784 | 0.2302 | 0.2257 | 0.2346 | 1.0000 |
| patchtst_itransformer | 2 | 0.2400 | 0.0810 | 0.4412 | itransformer | 0.3224 | 0.3742 | 0.0670 | 0.1188 | 0.1188 | 0.1188 | 1.0000 |
| patchtst_itransformer | 3 | 0.2300 | 0.0657 | 0.3821 | itransformer | 0.3224 | 0.3742 | 0.0079 | 0.0597 | 0.0460 | 0.0734 | 1.0000 |
| patchtst_itransformer | 4 | 0.2400 | 0.0874 | 0.4661 | itransformer | 0.3224 | 0.3742 | 0.0918 | 0.1436 | 0.1436 | 0.1436 | 1.0000 |
| patchtst_itransformer | 5 | 0.2400 | 0.0977 | 0.5041 | itransformer | 0.3224 | 0.3742 | 0.1299 | 0.1817 | 0.1758 | 0.1876 | 1.0000 |

## Pair-Level Outcomes

| pair | beats_stronger | positive_ci | beats_simple_average |
|---|---|---|---|
| patchtst_itransformer | 5 | 5 | 5 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 1 | 0.2400 | 29 | 21 | 0.0100 | 0.2900 | 2 |
| patchtst_itransformer | 2 | 0.2400 | 24 | 6 | 0.0100 | 0.3000 | 2 |
| patchtst_itransformer | 3 | 0.2300 | 24 | 3 | 0.0100 | 0.2900 | 3 |
| patchtst_itransformer | 4 | 0.2400 | 10 | 9 | 0.2000 | 0.2900 | 2 |
| patchtst_itransformer | 5 | 0.2400 | 28 | 9 | 0.0100 | 0.3000 | 2 |

## Common Tau Across Classifier Groups

The selected common tau maximizes the worst classifier group's Sharpe advantage over the simple holding average.

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 0.2300 | 0.4599 | 0.3821 | 0.0597 | 0.0079 | 5 | 5 |

## Forecast Diagnostics

| model | segment | mae | rmse | directional_accuracy |
|---|---|---|---|---|
| itransformer | calibration | 0.0183 | 0.0272 | 0.4965 |
| itransformer | trade | 0.0191 | 0.0282 | 0.5228 |
| patchtst | calibration | 0.0175 | 0.0259 | 0.4989 |
| patchtst | trade | 0.0186 | 0.0277 | 0.5211 |

## Main Finding

At each pair-group configuration's mean-Sharpe-maximizing global tau, the classifier-assisted ensemble exceeds its stronger component in 5/5 configurations and the simple holding average in 5/5; 5/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected on the completed evaluation span, so this is a sensitivity experiment rather than an out-of-sample tau-selection claim. Confidence intervals are conditional on the selected tau and fixed candidate forecasts and therefore do not correct for tau-selection uncertainty.
