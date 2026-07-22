# Modern Deep Forecasting Model Ensemble Experiment

## Protocol

- Dataset: SSE50; source `external_data/trademaster_sse50_daily`; trade split `test`; 26 assets are present in the supplied files.
- Evaluation span: 2019-10-28 to 2020-08-31, 208 sessions in 4 rolling blocks (63, 63, 63, 19 sessions).
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
| itransformer | 0.2656 | 1.4762 | 2.2886 | -0.1451 |
| patchtst | 0.0987 | 0.8305 | 1.0134 | -0.1198 |

## Simple Average Controls

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| patchtst_itransformer | 0.1908 | 1.2877 | 1.8186 | -0.1302 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 1 | 0.1500 | 0.2753 | 1.6526 | itransformer | 1.4762 | 1.2877 | 0.3649 | 0.1765 | 0.1681 | 0.1849 | 1.0000 |
| patchtst_itransformer | 2 | 0.1700 | 0.2840 | 1.6519 | itransformer | 1.4762 | 1.2877 | 0.3642 | 0.1758 | 0.1758 | 0.1758 | 1.0000 |
| patchtst_itransformer | 3 | 0.0100 | 0.2164 | 1.4779 | itransformer | 1.4762 | 1.2877 | 0.1902 | 0.0018 | -0.0200 | 0.0235 | 0.5333 |
| patchtst_itransformer | 4 | 0.1700 | 0.3039 | 1.8059 | itransformer | 1.4762 | 1.2877 | 0.5181 | 0.3297 | 0.3260 | 0.3334 | 1.0000 |
| patchtst_itransformer | 5 | 0.0400 | 0.2808 | 1.6802 | itransformer | 1.4762 | 1.2877 | 0.3925 | 0.2040 | 0.1860 | 0.2221 | 1.0000 |

## Pair-Level Outcomes

| pair | beats_stronger | positive_ci | beats_simple_average |
|---|---|---|---|
| patchtst_itransformer | 5 | 4 | 5 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 1 | 0.1500 | 22 | 26 | 0.0100 | 0.2200 | 16 |
| patchtst_itransformer | 2 | 0.1700 | 22 | 26 | 0.0100 | 0.2200 | 1 |
| patchtst_itransformer | 3 | 0.0100 | 4 | 25 | 0.0100 | 0.1500 | 16 |
| patchtst_itransformer | 4 | 0.1700 | 23 | 26 | 0.0100 | 0.2300 | 1 |
| patchtst_itransformer | 5 | 0.0400 | 22 | 26 | 0.0100 | 0.2200 | 17 |

## Common Tau Across Classifier Groups

The selected common tau maximizes the worst classifier group's Sharpe advantage over the simple holding average.

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 0.0100 | 1.6411 | 1.4779 | 0.0018 | 0.1902 | 5 | 5 |

## Forecast Diagnostics

| model | segment | mae | rmse | directional_accuracy |
|---|---|---|---|---|
| itransformer | calibration | 0.0143 | 0.0200 | 0.4846 |
| itransformer | trade | 0.0154 | 0.0218 | 0.4669 |
| patchtst | calibration | 0.0133 | 0.0191 | 0.4854 |
| patchtst | trade | 0.0147 | 0.0211 | 0.4686 |

## Main Finding

At each pair-group configuration's mean-Sharpe-maximizing global tau, the classifier-assisted ensemble exceeds its stronger component in 5/5 configurations and the simple holding average in 5/5; 4/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected on the completed evaluation span, so this is a sensitivity experiment rather than an out-of-sample tau-selection claim. Confidence intervals are conditional on the selected tau and fixed candidate forecasts and therefore do not correct for tau-selection uncertainty.
