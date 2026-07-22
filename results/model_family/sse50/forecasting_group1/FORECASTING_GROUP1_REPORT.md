# Representative Forecasting Model Ensemble Experiment

## Protocol

- Dataset: SSE50; source `external_data/trademaster_sse50_daily`; trade split `test`; 26 assets are present in the supplied files.
- Evaluation span: 2019-10-28 to 2020-08-31, 208 sessions in 4 rolling blocks (63, 63, 63, 19 sessions).
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
| arima | 0.1358 | 0.9094 | 1.1337 | -0.1479 |
| lstm | 0.1851 | 1.1266 | 1.3721 | -0.1674 |
| xgboost | 0.0968 | 0.7744 | 0.9293 | -0.1281 |

## Simple Average Controls

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| arima_lstm | 0.1669 | 1.0628 | 1.3266 | -0.1558 |
| arima_xgboost | 0.1253 | 0.9114 | 1.1392 | -0.1356 |
| xgboost_lstm | 0.1478 | 1.0226 | 1.2527 | -0.1459 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arima_lstm | 1 | 0.3800 | 0.1750 | 1.1238 | lstm | 1.1266 | 1.0628 | 0.0610 | -0.0027 | -0.0048 | -0.0007 | 0.3000 |
| arima_lstm | 2 | 0.2100 | 0.1796 | 1.1211 | lstm | 1.1266 | 1.0628 | 0.0583 | -0.0055 | -0.0058 | -0.0051 | 0.0000 |
| arima_lstm | 3 | 0.1700 | 0.1561 | 1.0148 | lstm | 1.1266 | 1.0628 | -0.0480 | -0.1117 | -0.1174 | -0.1061 | 0.0000 |
| arima_lstm | 4 | 0.2100 | 0.1725 | 1.0895 | lstm | 1.1266 | 1.0628 | 0.0266 | -0.0371 | -0.0418 | -0.0324 | 0.0000 |
| arima_lstm | 5 | 0.2100 | 0.1730 | 1.0917 | lstm | 1.1266 | 1.0628 | 0.0289 | -0.0348 | -0.0402 | -0.0294 | 0.0000 |
| arima_xgboost | 1 | 0.2900 | 0.1331 | 0.9551 | arima | 0.9094 | 0.9114 | 0.0437 | 0.0457 | 0.0272 | 0.0642 | 0.8333 |
| arima_xgboost | 2 | 0.2800 | 0.1335 | 0.9703 | arima | 0.9094 | 0.9114 | 0.0589 | 0.0609 | 0.0609 | 0.0609 | 1.0000 |
| arima_xgboost | 3 | 0.2300 | 0.1364 | 0.9272 | arima | 0.9094 | 0.9114 | 0.0158 | 0.0178 | 0.0061 | 0.0295 | 0.5333 |
| arima_xgboost | 4 | 0.2700 | 0.1274 | 0.9353 | arima | 0.9094 | 0.9114 | 0.0239 | 0.0259 | 0.0238 | 0.0280 | 1.0000 |
| arima_xgboost | 5 | 0.2700 | 0.1310 | 0.9510 | arima | 0.9094 | 0.9114 | 0.0395 | 0.0415 | 0.0311 | 0.0520 | 1.0000 |
| xgboost_lstm | 1 | 0.1800 | 0.1695 | 1.1219 | lstm | 1.1266 | 1.0226 | 0.0993 | -0.0046 | -0.0309 | 0.0216 | 0.4333 |
| xgboost_lstm | 2 | 0.1800 | 0.1889 | 1.2337 | lstm | 1.1266 | 1.0226 | 0.2111 | 0.1071 | 0.1064 | 0.1079 | 1.0000 |
| xgboost_lstm | 3 | 0.2000 | 0.1299 | 0.9364 | lstm | 1.1266 | 1.0226 | -0.0861 | -0.1901 | -0.2204 | -0.1598 | 0.0000 |
| xgboost_lstm | 4 | 0.1800 | 0.1815 | 1.1902 | lstm | 1.1266 | 1.0226 | 0.1676 | 0.0636 | 0.0590 | 0.0683 | 1.0000 |
| xgboost_lstm | 5 | 0.1800 | 0.1796 | 1.1908 | lstm | 1.1266 | 1.0226 | 0.1682 | 0.0642 | 0.0615 | 0.0670 | 1.0000 |

## Pair-Level Outcomes

| pair | beats_stronger | positive_ci | beats_simple_average |
|---|---|---|---|
| arima_lstm | 0 | 0 | 4 |
| arima_xgboost | 5 | 5 | 5 |
| xgboost_lstm | 3 | 3 | 4 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| arima_lstm | 1 | 0.3800 | 0 | 61 | nan | nan | 55 |
| arima_lstm | 2 | 0.2100 | 0 | 6 | nan | nan | 1 |
| arima_lstm | 3 | 0.1700 | 0 | 0 | nan | nan | 2 |
| arima_lstm | 4 | 0.2100 | 0 | 60 | nan | nan | 56 |
| arima_lstm | 5 | 0.2100 | 0 | 4 | nan | nan | 2 |
| arima_xgboost | 1 | 0.2900 | 56 | 55 | 0.2600 | 0.8900 | 4 |
| arima_xgboost | 2 | 0.2800 | 8 | 8 | 0.2200 | 0.2900 | 2 |
| arima_xgboost | 3 | 0.2300 | 2 | 2 | 0.2200 | 0.2300 | 1 |
| arima_xgboost | 4 | 0.2700 | 5 | 5 | 0.2300 | 0.2800 | 2 |
| arima_xgboost | 5 | 0.2700 | 6 | 6 | 0.2300 | 0.2900 | 3 |
| xgboost_lstm | 1 | 0.1800 | 0 | 19 | nan | nan | 1 |
| xgboost_lstm | 2 | 0.1800 | 20 | 23 | 0.0100 | 0.2000 | 1 |
| xgboost_lstm | 3 | 0.2000 | 0 | 0 | nan | nan | 20 |
| xgboost_lstm | 4 | 0.1800 | 19 | 21 | 0.0100 | 0.1900 | 1 |
| xgboost_lstm | 5 | 0.1800 | 19 | 22 | 0.0100 | 0.1900 | 1 |

## Common Tau Across Classifier Groups

The selected common tau maximizes the worst classifier group's Sharpe advantage over the simple holding average.

| pair | tau | sharpe_across_groups_mean | sharpe_across_groups_min | min_delta_vs_stronger | min_delta_vs_simple_average | groups_beating_stronger | groups_beating_simple_average |
|---|---|---|---|---|---|---|---|
| arima_lstm | 0.1800 | 1.0119 | 0.9932 | -0.1334 | -0.0697 | 0 | 0 |
| arima_xgboost | 0.2800 | 0.9344 | 0.8884 | -0.0210 | -0.0230 | 4 | 4 |
| xgboost_lstm | 0.2000 | 1.0550 | 0.9364 | -0.1901 | -0.0861 | 1 | 3 |

## Forecast Diagnostics

| model | segment | mae | rmse | directional_accuracy |
|---|---|---|---|---|
| arima | calibration | 0.0132 | 0.0189 | 0.4941 |
| arima | trade | 0.0144 | 0.0206 | 0.4889 |
| lstm | calibration | 0.0132 | 0.0189 | 0.4819 |
| lstm | trade | 0.0143 | 0.0206 | 0.4849 |
| xgboost | calibration | 0.0138 | 0.0195 | 0.4881 |
| xgboost | trade | 0.0152 | 0.0214 | 0.4827 |

## Main Finding

At each pair-group configuration's mean-Sharpe-maximizing global tau, the classifier-assisted ensemble exceeds its stronger component in 8/15 configurations and the simple holding average in 13/15; 8/15 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected on the completed evaluation span, so this is a sensitivity experiment rather than an out-of-sample tau-selection claim. Confidence intervals are conditional on the selected tau and fixed candidate forecasts and therefore do not correct for tau-selection uncertainty.
