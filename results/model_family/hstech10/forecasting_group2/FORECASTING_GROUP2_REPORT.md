# Modern Deep Forecasting Model Ensemble Experiment

## Protocol

- Dataset: HSTech10; source `external_data/trademaster_hstech10`; trade split `test`; 10 assets are present in the supplied files.
- Evaluation span: 2019-11-01 to 2020-08-31, 206 sessions in 4 rolling blocks (63, 63, 63, 17 sessions).
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
| itransformer | 0.4937 | 1.5383 | 3.0068 | -0.2120 |
| patchtst | 0.5169 | 1.6196 | 3.1861 | -0.2099 |

## Simple Average Controls

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| patchtst_itransformer | 0.5094 | 1.5960 | 3.1320 | -0.2103 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 1 | 0.6600 | 0.5118 | 1.6088 | patchtst | 1.6196 | 1.5960 | 0.0128 | -0.0108 | -0.0147 | -0.0069 | 0.0000 |
| patchtst_itransformer | 2 | 0.6600 | 0.5086 | 1.6105 | patchtst | 1.6196 | 1.5960 | 0.0145 | -0.0091 | -0.0091 | -0.0091 | 0.0000 |
| patchtst_itransformer | 3 | 0.6600 | 0.5067 | 1.5937 | patchtst | 1.6196 | 1.5960 | -0.0023 | -0.0259 | -0.0282 | -0.0236 | 0.0000 |
| patchtst_itransformer | 4 | 0.6600 | 0.5084 | 1.6104 | patchtst | 1.6196 | 1.5960 | 0.0144 | -0.0092 | -0.0096 | -0.0088 | 0.0000 |
| patchtst_itransformer | 5 | 0.6600 | 0.5083 | 1.6097 | patchtst | 1.6196 | 1.5960 | 0.0136 | -0.0099 | -0.0103 | -0.0095 | 0.0000 |

## Pair-Level Outcomes

| pair | beats_stronger | positive_ci | beats_simple_average |
|---|---|---|---|
| patchtst_itransformer | 0 | 0 | 4 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 1 | 0.6600 | 0 | 26 | nan | nan | 24 |
| patchtst_itransformer | 2 | 0.6600 | 0 | 26 | nan | nan | 24 |
| patchtst_itransformer | 3 | 0.6600 | 0 | 0 | nan | nan | 24 |
| patchtst_itransformer | 4 | 0.6600 | 0 | 26 | nan | nan | 24 |
| patchtst_itransformer | 5 | 0.6600 | 0 | 26 | nan | nan | 24 |

## Common Tau Across Classifier Groups

The selected common tau maximizes the worst classifier group's Sharpe advantage over the simple holding average.

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| patchtst_itransformer | 0.6600 | 1.6066 | 1.5937 | -0.0259 | -0.0023 | 0 | 4 |

## Forecast Diagnostics

| model | segment | mae | rmse | directional_accuracy |
|---|---|---|---|---|
| itransformer | calibration | 0.0264 | 0.0362 | 0.4996 |
| itransformer | trade | 0.0291 | 0.0394 | 0.4906 |
| patchtst | calibration | 0.0257 | 0.0353 | 0.4810 |
| patchtst | trade | 0.0278 | 0.0377 | 0.5005 |

## Main Finding

At each pair-group configuration's mean-Sharpe-maximizing global tau, the classifier-assisted ensemble exceeds its stronger component in 0/5 configurations and the simple holding average in 4/5; 0/5 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected on the completed evaluation span, so this is a sensitivity experiment rather than an out-of-sample tau-selection claim. Confidence intervals are conditional on the selected tau and fixed candidate forecasts and therefore do not correct for tau-selection uncertainty.
