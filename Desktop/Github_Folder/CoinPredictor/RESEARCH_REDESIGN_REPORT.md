# CoinPredictor Research Redesign Report

Date: 2026-07-10

## Scope

This report documents the research review and implementation pass performed against the current local `CoinPredictor/` repository and the current local dataset/cache artifacts. The implementation in this pass is deliberately compatibility-preserving: it does not require raw shard regeneration or external data access, and it works from the existing aggregate cache and prediction artifacts.

The code changes in this pass improve production observability, robustness gates, and add a first-class economic ranking architecture. This report does not claim a fresh 80M-row profitability result from the local machine; it documents the architecture now available for that run and the diagnostics required to trust or reject it.

## Current Dataset Verified

The active dataset is represented by `CoinPredictor/shard_dataset_30_volatile_from_existing/kline_growth_dataset.meta.json` plus aggregate cache files under `CoinPredictor/.gbdt_cache_full30_volatile/`.

Verified manifest/cache facts:

| Item | Verified value |
| --- | --- |
| Aggregate rows | 80,321,284 |
| Current engineered features | 76 |
| Symbols | 30 |
| Monthly coverage | 2017-08 through 2026-07 |
| Source shard entries | 1,866 |
| Label mode | `target_stop` |
| Target exit mode | `first_decline` |
| Prediction window | 5 minutes |
| Upside target | 0.02 |
| Downside stop | 0.02 |
| Growth threshold | 0.05 |
| Fee | 0.001 |
| Slippage | 0.0005 |
| Market regime features | enabled |
| Market breadth features | enabled |

Compatibility constraint: source shard CSVs are not guaranteed to be locally complete, but the aggregate cache is present and compatible. Any redesign must keep `load_sharded_rows()` and aggregate-cache attachment working.

## Original Architecture Map

Data generation starts in `DataScraper.cpp`. It constructs one-minute kline samples, 37 base OHLCV/microstructure-style features, 27 BTC/ETH market-regime features, and 12 market-breadth features. Labels and realized returns are path-derived. In the current manifest, `target_stop` with `first_decline` creates a label from whether the 2% target or 2% stop is reached in the 5-minute future path, while `trade_return` records the first-decline exit return for target hits and stop loss for stop hits.

`RiskAnalyzer.cpp` is a symbol-universe helper. It scrapes Binance trade archives, computes recent buy/sell ratios, and filters unusual symbols. It is not the training objective.

`merge_shard_dataset_cache.py` reconciles shard inventories and aggregate caches. It is important because it preserves operation when raw shards are unavailable.

`gbdt_pipeline.py` is the research and execution monolith. It handles cache loading, chronological splits, embargo, row sampling, LightGBM/internal fallback models, classification, regression, hybrid scoring, calibration, uncertainty heuristics, meta filters, symbol filters, threshold tuning, walk-forward evaluation, execution simulation, dynamic exits, position sizing, report generation, and experiment summaries.

`run_experiments.py` orchestrates large heuristic sweeps over this monolith. Most recent profiles are variants of hybrid LightGBM plus validation retuning, symbol filters, and threshold penalties.

`test_gbdt_pipeline.py` is broad and valuable. It covers cache behavior, splitting, validation, execution, reports, and several edge cases.

## What The Original System Optimizes

The base model primarily optimizes tabular ML losses: binary classification loss, return-regression loss, or a hybrid of probability and predicted return.

The trading system then optimizes validation-time decision rules: score thresholds, top-k selection, meta filters, symbol filters, dynamic thresholds, concentration penalties, and execution heuristics. The effective objective is therefore not a clean statistical target. It is:

1. Learn a noisy row-wise proxy.
2. Search validation thresholds and filters until a portfolio simulator looks acceptable.
3. Evaluate with walk-forward summaries that can still mark permissive runs as accepted.

This can be useful, but it is fragile. A strong trading strategy should show stable ranking quality and tail economics before thresholds and filters are added.

## What It Pretends To Optimize But Does Not Reliably Optimize

The existing system appears to optimize real-world PnL, Sharpe-like stability, and robust cross-regime profitability. In practice, historical results show that those properties are often downstream of threshold/filter choices and can be driven by a small number of months or symbols.

High classification accuracy is especially uninformative here because the class is imbalanced and only the extreme score tail is traded. A run can have high accuracy, no trades, or one-fold-dominated profit and still look superficially good.

## Bottleneck Ranking

| Rank | Bottleneck | Class | Expected impact if fixed | Evidence and effect |
| --- | --- | --- | --- | --- |
| 1 | Target/economic mismatch | First-order structural | High EV, trade quality, Sharpe, Sortino, drawdown, calibration | `target_stop` is binary and path-thresholded, while trading value depends on conditional net payoff, adverse excursion, rank, concentration, and capital allocation. |
| 2 | Threshold and filter overfit | First-order structural | High fold/month stability, overfit resistance, trade consistency | Recent results show sparse runs can be marked accepted with `acceptance_tier=none` despite gate failures. Threshold logic does too much work after model fitting. |
| 3 | Tail ranking blind spot | First-order structural | High precision in traded region, EV, allocation, calibration | Existing calibration uses coarse probability buckets and reports executed aggregates, not top-score monotonicity or concentration. |
| 4 | Concentration fragility | First-order structural | High drawdown, regime robustness, symbol selection | Historical profitable runs are often concentrated in one symbol or month. This is portfolio risk, not just reporting noise. |
| 5 | Data utilization bottleneck | First-order structural | Medium/high recall and generalization | The cache has 80.3M rows, while common profiles train on 1.5M rows and retune thresholds on samples. Sampling may be necessary, but it must be justified by stability diagnostics. |
| 6 | Panel/time structure underuse | First-order structural | Medium/high rank quality and drift adaptation | The model is row-wise; market breadth is present but there is no true groupwise time ranking or hierarchical regime/symbol family layer. |
| 7 | Tail calibration weakness | Second-order but important | Medium/high precision and sizing | Platt/global calibration can be adequate globally while wrong in the top 1%-10% score region where trades happen. |
| 8 | Monolithic heuristic accretion | Second-order engineering | Medium reproducibility and research speed | A giant pipeline makes it easy to add knobs and hard to isolate causal gains. |
| 9 | Execution realism limits | Second-order data-bound | Medium drawdown and live gap | Fees/slippage/latency/volume caps exist, but the dataset lacks full order book and spread data. |

## Research-Backed Redesign

The strongest architecture compatible with this dataset is not a pure classifier. It should be a candidate-generation plus economic-ranking system.

Recommended target formulation:

1. Candidate generation: broad, high-recall filter using calibrated probability of clearing a fee/slippage-adjusted hurdle.
2. Multi-task payoff estimation: estimate expected net return, hit probability, downside/MAE proxy, and uncertainty from the same panel features.
3. Groupwise ranking: rank candidates within time buckets or fold-local decision groups by expected utility, not just probability. LightGBM officially supports ranking objectives such as `lambdarank` and `rank_xendcg`, which fit this use case better than a thresholded binary objective when the portfolio trades the top tail.
4. Tail calibration: calibrate and validate only in the decision region, including top 1%, 5%, 10%, and score-decile monotonicity.
5. Portfolio allocation: allocate capital by expected utility subject to concentration, active-day, and regime gates.
6. Nested chronological validation: keep model selection, threshold selection, and final reporting separated with embargo and fold/month/symbol diagnostics.

Why not lead with transformers, reinforcement learning, or graph models:

- The dataset is tabular, engineered, one-minute panel data with no full order book state. GBDT and ranking objectives are likely higher-signal and easier to validate.
- RL would mostly learn the simulator and threshold/filter artifacts unless raw state/action/reward coverage is much richer.
- Sequence models may help later, but they first need leakage-safe sequence windows and a benchmark showing that tabular ranking has been exhausted.

External references used:

- LightGBM official documentation lists binary, regression, and ranking objectives including `lambdarank` and `rank_xendcg`: https://lightgbm.readthedocs.io/en/stable/Parameters.html
- scikit-learn calibration documentation emphasizes that well-calibrated probabilities should match observed event frequency and that Brier/log loss mix reliability and discrimination: https://scikit-learn.org/stable/modules/calibration.html
- Carr and Lopez de Prado warn that calibrating trading rules through repeated historical simulation contributes to backtest overfitting: https://arxiv.org/abs/1408.1159
- Rej, Seager, and Bouchaud discuss how iterative strategy tweaking can improve noise realizations rather than true expected performance: https://arxiv.org/abs/1902.01802

## Implementation Delivered

### 1. Memory profiling correctness

Fixed `current_rss_gib()` so it actually reads process RSS when `psutil` is available and updates peak RSS by stage. The previous function returned `None` when `psutil` existed, while the real RSS code was accidentally unreachable inside `record_profile_stage()` with an undefined `stage` variable.

Expected impact:

- Direct PnL: none.
- Production robustness: high for large 80M-row runs.
- Research reliability: high, because memory budget and profiling reports are now meaningful.

### 2. Walk-forward gate status

Added explicit `walkforward_gate_status` and `walkforward_gate_failed` fields. This preserves existing permissive `accepted` behavior when `acceptance_tier=none`, but prevents a run from looking clean when the walk-forward robustness gate failed.

Expected impact:

- Precision/recall: none directly.
- Overfit resistance: medium/high.
- Fold/month stability interpretation: high.
- Strategy acceptance clarity: high.

### 3. Ranking and tail diagnostics

Added a first-class `kline_growth_ranking_report.csv` output and `--ranking-report-out` parser option. The report is generated from existing prediction files and is included in:

- Run summary JSON.
- Experiment summary CSV.
- Experiment markdown report.
- Results-directory output path mapping.

The report evaluates each available score source:

- `trade_score`
- `hybrid_score`
- `expected_value`
- `predicted_net_return`
- `calibrated_probability`
- `probability`

For each source it reports:

- Top 1%, 5%, 10%, 25%, and 100% buckets.
- Score deciles.
- Executed-trade subset.
- Label positive rate.
- Average realized trade return.
- Average net return after configured fee, slippage, and latency penalty.
- Profit factor.
- Estimated portfolio profit if selected at configured max position fraction.
- Symbol, month, and active-day coverage.
- Top symbol and top month concentration.
- Net-return monotonicity across score deciles.

Expected impact:

- Direct PnL: indirect but high leverage.
- EV/trade quality: high, because bad top-tail ranking is now visible.
- Threshold adaptation: high, because thresholds can be rejected when ranking is non-monotonic.
- Concentration control: high, because symbol/month dominance is now measured in the traded tail.
- Overfit resistance: high, because sparse/no-trade and one-month solutions become easier to reject.

### 4. Economic ranker

Implemented a new `economic_ranking` objective path in `gbdt_pipeline.py`.

What changed:

- Added train-only utility labels from realized `trade_return - fee - slippage - latency`, with optional adverse-excursion penalty.
- Quantized positive utility into relevance grades using training-fold quantiles only.
- Added fold-local LightGBM ranking groups by decision-time bucket.
- Added LightGBM ranking support with `rank_xendcg` and automatic fallback to `lambdarank`.
- Added `ranker_score` to prediction bundles, prediction caches, ensemble averaging, prediction CSVs, feature importance, metrics, ranking reports, experiment summaries, and markdown reports.
- Added economic-ranker selection with `topk_score`, `top_percent_score`, and `top_utility`.
- Made threshold search opt-in for the ranker path; the default ranker path is top-tail allocation rather than validation threshold mining.
- Disabled binary AUC for the economic-ranker path to avoid presenting the old label metric as the main success criterion.

Expected impact:

- Precision: medium/high in the traded tail if relevance grades preserve net utility ordering.
- Recall: medium; the ranker is intentionally selective, but top-percent allocation prevents collapse into a single threshold.
- Expected value: high, because the training target is net utility rather than raw binary target hit.
- Sharpe/Sortino/drawdown: medium/high through better tail ranking and concentration diagnostics.
- Stability across folds/months: medium/high if groupwise ranking generalizes; this must be verified with walk-forward reports.
- Calibration quality: indirect; ranker scores are not probabilities, so calibration moves to tail economics and monotonicity.
- Entry timing and symbol selection: high potential because candidates compete inside decision-time groups.

### 5. Experiment runner profile

Added an `economic-ranker` profile to `run_experiments.py` targeting the current aggregate-cache dataset:

- `--objective-mode economic_ranking`
- `--trade-score ranker_score`
- `--ranker-objective rank_xendcg`
- top-k and top-percent variants
- current `shard_dataset_30_volatile_from_existing` input and `.gbdt_cache_full30_volatile` cache

### 6. Tests

Added tests for:

- `current_rss_gib()` with a fake `psutil` process.
- Ranking report top-tail math and cost adjustment.
- Summary/CSV wiring for ranking and gate fields.
- Markdown report inclusion.

Full Python suite result:

`python3 -m unittest CoinPredictor.test_gbdt_pipeline`

Result: 172 tests passed, 25 skipped.

## Historical Old vs New Observability

No fresh 80M-row retrain was run in this pass. To validate the new diagnostic against real repository artifacts, I ran it on two existing walk-forward prediction files.

Important limitation: these historical files were written in `prediction-output-mode=trades`, so the ranking report evaluates executed rows, not the full candidate universe. Future production comparisons should run with `--prediction-output-mode candidates` to audit candidate ranking before execution filters without writing every scored row.

| Existing run | Rows available | Trade-score top decile net return | Trade-score top 1% net return | Top decile label rate | Decile monotonicity | Top symbol share | Top month share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `classify_walk_24_4_4_sampling_fix` | 58,157 | 0.000380 | -0.000167 | 0.1279 | 0.7778 | 0.7831 | 0.2619 |
| `hybrid_10fold_full30_q_late_recent_negfloor_07` | 10,283 | 0.017276 | -0.010807 | 0.3599 | 0.7778 | 0.1012 | 0.9543 |

Interpretation:

- The broad run has positive top-decile economics but a negative top 1% and extreme top-symbol concentration. That is a warning against blindly increasing score selectivity.
- The sparse high-profit-looking run has a strong top decile but is overwhelmingly concentrated in one month, and its top 1% is negative. That is exactly the kind of fragility the original summaries could hide.

## Recommended Production Evaluation Standard

Every serious future experiment should include:

1. Strict chronological fixed split.
2. Walk-forward analysis with embargo.
3. Full candidate prediction output for at least representative folds.
4. Ranking report over full candidates, not only executed trades.
5. Calibration report plus top-tail ranking report.
6. Symbol and month concentration diagnostics.
7. Regime diagnostics.
8. Baseline comparison.
9. Explicit rejection if profit comes from one symbol, one month, or a non-monotonic top tail.

## Next Empirical Work

The next step is empirical, not architectural:

1. Run the `economic-ranker` profile over the current aggregate cache with walk-forward enabled.
2. Write full candidate predictions for representative folds, not only executed trades.
3. Compare against the strongest existing hybrid profile using the same months, fees, slippage, and memory budget.
4. Reject the ranker if top-decile utility is not positive, decile monotonicity is weak, or profit is dominated by a small number of symbols/months.
5. Only after that, test stacking the ranker with the hybrid expected-return score.

## Bottom Line

The original system is powerful but too easy to fool: it can produce attractive aggregate numbers while hiding top-tail inversion, sparse trading, and month/symbol concentration. The redesigned path changes the production standard from "did the thresholded simulator make money?" to "does the model rank economically useful, diversified trades in the out-of-sample tail?"

That is the necessary foundation for any genuinely stronger trading architecture built from the current dataset.
