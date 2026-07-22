# Ensemble RL through Classifier Models

Code, input data, and compact final results for the experiments in:

**Ensemble RL through Classifier Models: Enhancing Risk-Return Trade-offs in Trading Strategies**

- Paper: https://arxiv.org/abs/2502.17518
- Base implementation: FinRL with Stable-Baselines3
- Markets: DJ30, SSE50, and HSTech10
- Python: 3.10

The paper source and compiled manuscript are intentionally excluded. This
repository contains only the current paper experiments. Development-only tau
controllers, smoke runs, 2021 extensions, obsolete parameterizations, model
checkpoints, candidate-holding caches, and full daily trace files are not
published.

## What Is Reproduced

The package covers five experiment families:

1. The A2C, PPO, and SAC main experiment on DJ30.
2. The complete SSE50 and HSTech10 external-market repetitions.
3. The original, L1, and risk-weighted holding-disagreement ablation.
4. Forecasting and deep-RL model-family extensions.
5. The 2020 causal ex-ante tau experiment and its success-failure attribution.

All core fixed-threshold paths use the paper-aligned protocol:

- RL training expands through time and is independent for every dataset.
- The immediately preceding 63-session block selects the RL checkpoint.
- Policy inference is deterministic (`deterministic=True`).
- Classifiers are refitted on the immediately preceding block only.
- Classifier group and hyperparameters are fixed; there is no classifier grid search.
- A candidate global tau is fixed for the complete evaluation path.
- Each tau is evaluated with 30 rolling classifier refits.
- The reported fixed-global-tau result selects the mean-Sharpe-maximizing tau
  after evaluating the full path, as a sensitivity result rather than an
  ex-ante deployment rule.

The causal extension is separate: it uses 2019 only as completed-history
warm-up, selects tau before each 2020 block, and freezes it for that block.

## Published Evidence Snapshot

At each configuration's selected fixed global tau, the core ensemble beats its
stronger component by mean Sharpe in 12/15 DJ30, 13/15 SSE50, and 8/15 HSTech10
configurations. The paired classifier-refit interval is entirely positive in
10/15, 13/15, and 8/15 configurations, respectively.

Across all three markets, the model-family extensions beat the stronger
component in 24/45 representative-forecasting configurations, 10/15
PatchTST+iTransformer configurations, 11/15 PPO+TQC configurations, and 7/15
TD3+TQC stress configurations.

For the 2020 causal mechanism-only strategy, 12/15 configurations beat the
causal rolling-validation single-RL baseline. Across 450 classifier-refit paths,
it records 279 wins, 115 ties, and 56 losses against that baseline; 7/15
configuration means also exceed the retrospectively stronger fixed component.

These are conditional results on fixed RL candidates and the observed market
paths. The 30 repetitions measure classifier-refit variation, not independent
market samples or independent RL training seeds.

## Repository Layout

```text
.
|-- external_data/
|   |-- trademaster_dj30/
|   |-- trademaster_dj30_exante_2019/
|   |-- trademaster_sse50_daily/
|   `-- trademaster_hstech10/
|-- finrl/reproduction/             # Decision rules and experiment primitives
|-- examples/                       # Only current-paper runners and audits
|-- results/
|   |-- main_dj30/
|   |-- external_sse50/
|   |-- external_hstech10/
|   |-- disagreement/
|   |-- model_family/
|   |-- causal_tau/
|   |-- paper_figures/              # Exact figures used by the current paper
|   `-- data_quality/
|-- unit_tests/
|-- EXPERIMENTS.md                  # End-to-end commands
|-- MANIFEST.md                     # Inclusion and exclusion policy
|-- environment.yml
`-- requirements-reproduction.txt
```

`results/` contains final metrics, selected-threshold tables, audits, reports,
and paper figures. Large work products are deliberately omitted. Full runs
write those products under `work/`, which is ignored by Git.

## Environment

Conda:

```bash
conda env create -f environment.yml
conda activate ensemble-rl-paper
pip install -e .
```

Or install into an existing Python 3.10 environment:

```bash
pip install -r requirements-reproduction.txt
pip install -e .
```

On macOS, XGBoost may additionally require:

```bash
conda install -c conda-forge llvm-openmp
```

## Quick Verification

```bash
pytest -q unit_tests/test_classifier_ensemble.py \
  unit_tests/test_disagreement.py \
  unit_tests/test_forecasting_ensemble.py \
  unit_tests/test_causal_crossover_tau.py
```

Audit the published input panels and regenerate their checksums:

```bash
python examples/audit_portfolio_data.py \
  --dataset DJ30=external_data/trademaster_dj30 \
  --dataset DJ30Warmup2019=external_data/trademaster_dj30_exante_2019 \
  --dataset SSE50=external_data/trademaster_sse50_daily \
  --dataset HSTech10=external_data/trademaster_hstech10 \
  --output results/data_quality/portfolio_panel_audit.csv

python examples/build_data_manifest.py
```

The complete training and analysis commands are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Data Scope

| Dataset | Train span | Evaluation split | Assets | Evaluation sessions |
| --- | --- | --- | ---: | ---: |
| DJ30 | 2012-01-04 to 2019-12-31 | 2020-01-02 to 2020-12-31 | 29 | 253 |
| DJ30 warm-up | 2012-01-04 to 2018-12-31 | 2019-01-02 to 2019-12-31 | 29 | 252 |
| SSE50 | 2016-06-01 to 2018-12-14 | 2019-10-28 to 2020-08-31 | 26 | 208 |
| HSTech10 | 2016-06-01 to 2018-12-31 | 2019-11-01 to 2020-08-31 | 10 | 206 |

The aligned panels have no duplicate date-ticker keys, missing cells, or
non-positive closes. They are complete-history panels rather than historical
point-in-time index reconstructions; survivorship and universe-selection bias
therefore remain relevant when interpreting the results.

The source panels come from the portfolio-management datasets in
[fork-TradeMaster](https://github.com/shaun19920309/fork-TradeMaster). SSE50 is
aggregated from the available intraday rows to daily OHLCV; HSTech10 is the
ten-symbol complete-history subset documented in its metadata. The original
DJ30 `test.csv` split is retained for data provenance, but the published DJ30
experiments evaluate the 2020 `valid.csv` path only.

## Result Interpretation

The repository supports a conditional claim: classifier-assisted switching can
improve risk-adjusted performance when the candidate strategies provide usable
blockwise complementarity and the classifier vote creates stable, correctly
timed branch divergence. It does not establish universal superiority over the
best constituent or live-trading readiness.

The execution model uses daily closing prices and a fixed proportional
transaction cost. It does not model bid-ask spreads, market impact, partial
fills, latency, taxes, liquidity, or capacity.

## Citation

See [CITATION.cff](CITATION.cff). The source paper is available at
https://arxiv.org/abs/2502.17518.
