# Representative Forecasting Model Ensemble Experiment

## Protocol

- Dataset: DJ30; source `external_data/trademaster_dj30`; trade split `valid`; 29 assets are present in the supplied files.
- Evaluation span: 2020-01-02 to 2020-12-31, 253 sessions in 4 rolling blocks (63, 63, 63, 64 sessions).
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
| arima | 0.0892 | 0.4385 | 0.3195 | -0.2791 |
| lstm | 0.1084 | 0.4871 | 0.3712 | -0.2920 |
| xgboost | 0.0143 | 0.1854 | 0.0530 | -0.2701 |

## Simple Average Controls

| pair | cumulative_return | sharpe | calmar | max_drawdown |
|---|---|---|---|---|
| arima_lstm | 0.1149 | 0.5141 | 0.4046 | -0.2839 |
| arima_xgboost | 0.0757 | 0.4032 | 0.2796 | -0.2707 |
| xgboost_lstm | 0.0796 | 0.4125 | 0.2868 | -0.2777 |

## Selected Global Tau

| pair | classifier_group | selected_global_tau | ensemble_cumulative_return_mean | ensemble_sharpe_mean | stronger_model | stronger_sharpe | simple_average_sharpe | delta_sharpe_vs_average | delta_sharpe_mean | delta_sharpe_ci_low | delta_sharpe_ci_high | win_rate_vs_stronger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arima_lstm | 1 | 0.2200 | 0.1458 | 0.6181 | lstm | 0.4871 | 0.5141 | 0.1040 | 0.1309 | 0.1272 | 0.1346 | 1.0000 |
| arima_lstm | 2 | 0.3800 | 0.1555 | 0.6430 | lstm | 0.4871 | 0.5141 | 0.1289 | 0.1559 | 0.1559 | 0.1559 | 1.0000 |
| arima_lstm | 3 | 0.3000 | 0.1166 | 0.5180 | lstm | 0.4871 | 0.5141 | 0.0040 | 0.0309 | 0.0169 | 0.0449 | 0.7333 |
| arima_lstm | 4 | 0.3800 | 0.1558 | 0.6446 | lstm | 0.4871 | 0.5141 | 0.1305 | 0.1575 | 0.1567 | 0.1583 | 1.0000 |
| arima_lstm | 5 | 0.3800 | 0.1550 | 0.6420 | lstm | 0.4871 | 0.5141 | 0.1279 | 0.1548 | 0.1536 | 0.1561 | 1.0000 |
| arima_xgboost | 1 | 0.4800 | 0.0657 | 0.3726 | arima | 0.4385 | 0.4032 | -0.0306 | -0.0658 | -0.0701 | -0.0616 | 0.0000 |
| arima_xgboost | 2 | 0.4800 | 0.0697 | 0.3865 | arima | 0.4385 | 0.4032 | -0.0168 | -0.0520 | -0.0520 | -0.0520 | 0.0000 |
| arima_xgboost | 3 | 0.3000 | 0.0799 | 0.4253 | arima | 0.4385 | 0.4032 | 0.0221 | -0.0132 | -0.0281 | 0.0018 | 0.5000 |
| arima_xgboost | 4 | 0.4800 | 0.0632 | 0.3635 | arima | 0.4385 | 0.4032 | -0.0397 | -0.0749 | -0.0791 | -0.0708 | 0.0000 |
| arima_xgboost | 5 | 0.4800 | 0.0670 | 0.3772 | arima | 0.4385 | 0.4032 | -0.0260 | -0.0612 | -0.0666 | -0.0559 | 0.0000 |
| xgboost_lstm | 1 | 0.2700 | 0.1085 | 0.5213 | lstm | 0.4871 | 0.4125 | 0.1088 | 0.0342 | 0.0245 | 0.0439 | 0.8667 |
| xgboost_lstm | 2 | 0.2400 | 0.1018 | 0.4981 | lstm | 0.4871 | 0.4125 | 0.0856 | 0.0110 | 0.0110 | 0.0110 | 1.0000 |
| xgboost_lstm | 3 | 0.2000 | 0.0835 | 0.4314 | lstm | 0.4871 | 0.4125 | 0.0188 | -0.0557 | -0.0687 | -0.0428 | 0.0333 |
| xgboost_lstm | 4 | 0.2400 | 0.0995 | 0.4901 | lstm | 0.4871 | 0.4125 | 0.0775 | 0.0030 | 0.0013 | 0.0047 | 0.5667 |
| xgboost_lstm | 5 | 0.2400 | 0.1041 | 0.5078 | lstm | 0.4871 | 0.4125 | 0.0953 | 0.0207 | 0.0130 | 0.0284 | 0.9000 |

## Pair-Level Outcomes

| pair | beats_stronger | positive_ci | beats_simple_average |
|---|---|---|---|
| arima_lstm | 5 | 5 | 5 |
| arima_xgboost | 0 | 0 | 1 |
| xgboost_lstm | 4 | 4 | 5 |

## Tau Robustness

| pair | classifier_group | selected_global_tau | tau_beating_stronger | tau_beating_simple_average | tau_beating_stronger_min | tau_beating_stronger_max | tau_within_0.01_sharpe_of_best |
|---|---|---|---|---|---|---|---|
| arima_lstm | 1 | 0.2200 | 70 | 69 | 0.2000 | 0.8900 | 6 |
| arima_lstm | 2 | 0.3800 | 70 | 70 | 0.2000 | 0.8900 | 7 |
| arima_lstm | 3 | 0.3000 | 11 | 1 | 0.2100 | 0.3800 | 1 |
| arima_lstm | 4 | 0.3800 | 70 | 69 | 0.2000 | 0.8900 | 5 |
| arima_lstm | 5 | 0.3800 | 70 | 70 | 0.2000 | 0.8900 | 3 |
| arima_xgboost | 1 | 0.4800 | 0 | 0 | nan | nan | 47 |
| arima_xgboost | 2 | 0.4800 | 0 | 0 | nan | nan | 47 |
| arima_xgboost | 3 | 0.3000 | 0 | 51 | nan | nan | 1 |
| arima_xgboost | 4 | 0.4800 | 0 | 0 | nan | nan | 47 |
| arima_xgboost | 5 | 0.4800 | 0 | 0 | nan | nan | 47 |
| xgboost_lstm | 1 | 0.2700 | 2 | 9 | 0.2700 | 0.2800 | 1 |
| xgboost_lstm | 2 | 0.2400 | 3 | 14 | 0.2400 | 0.2900 | 3 |
| xgboost_lstm | 3 | 0.2000 | 0 | 3 | nan | nan | 1 |
| xgboost_lstm | 4 | 0.2400 | 1 | 13 | 0.2400 | 0.2400 | 2 |
| xgboost_lstm | 5 | 0.2400 | 2 | 13 | 0.2400 | 0.2500 | 2 |

## Forecast Diagnostics

| model | segment | mae | rmse | directional_accuracy |
|---|---|---|---|---|
| arima | calibration | 0.0173 | 0.0257 | 0.5225 |
| arima | trade | 0.0186 | 0.0277 | 0.5119 |
| lstm | calibration | 0.0173 | 0.0257 | 0.5122 |
| lstm | trade | 0.0187 | 0.0280 | 0.4900 |
| xgboost | calibration | 0.0181 | 0.0266 | 0.5033 |
| xgboost | trade | 0.0191 | 0.0284 | 0.4958 |

## Main Finding

At each pair-group configuration's mean-Sharpe-maximizing global tau, the classifier-assisted ensemble exceeds its stronger component in 9/15 configurations and the simple holding average in 11/15; 9/15 paired 95% intervals versus the stronger component are entirely positive.

Tau is selected on the completed evaluation span, so this is a sensitivity experiment rather than an out-of-sample tau-selection claim. Confidence intervals are conditional on the selected tau and fixed candidate forecasts and therefore do not correct for tau-selection uncertainty.
