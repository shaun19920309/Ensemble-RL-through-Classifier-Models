# Representative Forecasting Model Ensemble Experiment

## Protocol

- Dataset: HSTech10; source `external_data/trademaster_hstech10`; trade split `test`; 10 assets are present in the supplied files.
- Evaluation span: 2019-11-01 to 2020-08-31, 206 sessions in 4 rolling blocks (63, 63, 63, 17 sessions).
- Base models: ARIMA(1,0,1) with innovations MLE, XGBRegressor, and a two-layer LSTM.
- All models predict the next-session cross-section of log returns from close-price history only.
- Common portfolio map: trailing-20-session volatility scaling, cross-sectional softmax, maximum weight 0.20, gross exposure 0.95.
- Forecast-model fitting uses an expanding historical window. Classifiers use only the immediately preceding rolling calibration block.
- Every tau is fixed for a complete path; grid 0.01-0.89 by 0.01.
- Repetitions: 30; candidate forecasts and holdings remain fixed while rolling classifiers are refitted.
- Classifier groups and voting are unchanged from the RL experiment; no classifier grid search is used.

## Single Models

| model | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| arima | 0.5272 | 1.6137 | 3.3014 | -0.2068 |
| lstm | 0.6182 | 1.7657 | 3.6883 | -0.2188 |
| xgboost | 0.4891 | 1.6371 | 3.0233 | -0.2088 |

## Simple Average Controls

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| arima_lstm | 0.5751 | 1.7025 | 3.5221 | -0.2124 |
| arima_xgboost | 0.5130 | 1.6436 | 3.2037 | -0.2072 |
| xgboost_lstm | 0.5561 | 1.7185 | 3.3838 | -0.2134 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arima_lstm | 1 | 0.3000 | 0.5868 | 1.7175 | lstm | 1.7657 | 1.7025 | 0.0151 | -0.0482 | -0.0517 | -0.0446 | 0.0000 |
| arima_lstm | 2 | 0.2900 | 0.6282 | 1.8578 | lstm | 1.7657 | 1.7025 | 0.1553 | 0.0920 | 0.0920 | 0.0920 | 1.0000 |
| arima_lstm | 3 | 0.3600 | 0.5337 | 1.6269 | lstm | 1.7657 | 1.7025 | -0.0755 | -0.1388 | -0.1389 | -0.1387 | 0.0000 |
| arima_lstm | 4 | 0.3000 | 0.6186 | 1.8210 | lstm | 1.7657 | 1.7025 | 0.1185 | 0.0553 | 0.0516 | 0.0590 | 1.0000 |
| arima_lstm | 5 | 0.3100 | 0.5873 | 1.7306 | lstm | 1.7657 | 1.7025 | 0.0281 | -0.0351 | -0.0382 | -0.0320 | 0.0000 |
| arima_xgboost | 1 | 0.4300 | 0.4874 | 1.6231 | xgboost | 1.6371 | 1.6436 | -0.0205 | -0.0140 | -0.0175 | -0.0106 | 0.0000 |
| arima_xgboost | 2 | 0.1900 | 0.5125 | 1.6754 | xgboost | 1.6371 | 1.6436 | 0.0318 | 0.0383 | 0.0374 | 0.0392 | 1.0000 |
| arima_xgboost | 3 | 0.3000 | 0.5008 | 1.6356 | xgboost | 1.6371 | 1.6436 | -0.0080 | -0.0015 | -0.0060 | 0.0029 | 0.4667 |
| arima_xgboost | 4 | 0.3000 | 0.4855 | 1.6291 | xgboost | 1.6371 | 1.6436 | -0.0144 | -0.0080 | -0.0087 | -0.0072 | 0.0000 |
| arima_xgboost | 5 | 0.2900 | 0.4969 | 1.6240 | xgboost | 1.6371 | 1.6436 | -0.0195 | -0.0131 | -0.0168 | -0.0094 | 0.0667 |
| xgboost_lstm | 1 | 0.5200 | 0.6306 | 1.8803 | lstm | 1.7657 | 1.7185 | 0.1619 | 0.1146 | 0.1077 | 0.1216 | 1.0000 |
| xgboost_lstm | 2 | 0.3300 | 0.6240 | 1.8442 | lstm | 1.7657 | 1.7185 | 0.1257 | 0.0785 | 0.0785 | 0.0785 | 1.0000 |
| xgboost_lstm | 3 | 0.3000 | 0.4872 | 1.6246 | lstm | 1.7657 | 1.7185 | -0.0939 | -0.1411 | -0.1447 | -0.1375 | 0.0000 |
| xgboost_lstm | 4 | 0.3300 | 0.5977 | 1.8031 | lstm | 1.7657 | 1.7185 | 0.0846 | 0.0374 | 0.0278 | 0.0470 | 0.9000 |
| xgboost_lstm | 5 | 0.5500 | 0.5713 | 1.7671 | lstm | 1.7657 | 1.7185 | 0.0486 | 0.0014 | -0.0060 | 0.0089 | 0.8333 |

## Pair-Level Outcomes

| pair | beats_stronger | positive_ci | beats_simple_average |
|---|---|---|---|
| arima_lstm | 2 | 2 | 4 |
| arima_xgboost | 1 | 1 | 1 |
| xgboost_lstm | 4 | 3 | 4 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| arima_lstm | 1 | 0.3000 | 0 | 5 | nan | nan | 5 |
| arima_lstm | 2 | 0.2900 | 32 | 32 | 0.0100 | 0.3200 | 5 |
| arima_lstm | 3 | 0.3600 | 0 | 0 | nan | nan | 4 |
| arima_lstm | 4 | 0.3000 | 12 | 32 | 0.2000 | 0.3100 | 1 |
| arima_lstm | 5 | 0.3100 | 0 | 5 | nan | nan | 2 |
| arima_xgboost | 1 | 0.4300 | 0 | 0 | nan | nan | 16 |
| arima_xgboost | 2 | 0.1900 | 22 | 22 | 0.0100 | 0.2200 | 22 |
| arima_xgboost | 3 | 0.3000 | 0 | 0 | nan | nan | 1 |
| arima_xgboost | 4 | 0.3000 | 0 | 0 | nan | nan | 2 |
| arima_xgboost | 5 | 0.2900 | 0 | 0 | nan | nan | 3 |
| xgboost_lstm | 1 | 0.5200 | 55 | 56 | 0.3500 | 0.8900 | 3 |
| xgboost_lstm | 2 | 0.3300 | 42 | 53 | 0.3200 | 0.8900 | 1 |
| xgboost_lstm | 3 | 0.3000 | 0 | 0 | nan | nan | 74 |
| xgboost_lstm | 4 | 0.3300 | 48 | 57 | 0.3300 | 0.8900 | 4 |
| xgboost_lstm | 5 | 0.5500 | 35 | 54 | 0.5500 | 0.8900 | 37 |

## Common Tau Across Classifier Groups

The selected common tau maximizes the worst classifier group's Sharpe advantage over the simple holding average.

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| arima_lstm | 0.3200 | 1.6996 | 1.6080 | -0.1577 | -0.0945 | 1 | 2 |
| arima_xgboost | 0.3000 | 1.6209 | 1.6036 | -0.0336 | -0.0400 | 0 | 0 |
| xgboost_lstm | 0.5600 | 1.7598 | 1.6199 | -0.1458 | -0.0986 | 4 | 4 |

## Forecast Diagnostics

| model | segment | mae | rmse | directional_accuracy |
|---|---|---|---|---|
| arima | calibration | 0.0253 | 0.0348 | 0.5109 |
| arima | trade | 0.0277 | 0.0376 | 0.4924 |
| lstm | calibration | 0.0252 | 0.0347 | 0.5230 |
| lstm | trade | 0.0277 | 0.0376 | 0.4967 |
| xgboost | calibration | 0.0269 | 0.0369 | 0.4883 |
| xgboost | trade | 0.0297 | 0.0413 | 0.4819 |

## Main Finding

At each pair-group configuration's mean-Sharpe-maximizing global tau, the classifier-assisted ensemble exceeds its stronger component in 7/15 configurations and the simple holding average in 9/15; 6/15 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected on the completed evaluation span, so this is a sensitivity experiment rather than an out-of-sample tau-selection claim. Confidence intervals are conditional on the selected tau and fixed candidate forecasts and therefore do not correct for tau-selection uncertainty.
