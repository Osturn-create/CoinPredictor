# CoinPredictor Experiment Report

## Dataset Summary
- Input: `C:\Users\alexa\OneDrive\Desktop\GitHub_Files\Desktop\Github_Folder\CoinPredictor\shard_dataset_30_volatile_from_existing`
- Rows: `80321284`
- Features: `76`
- Symbols: `AAVEUSDT, ACTUSDT, ADAUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DEXEUSDT, DOGEUSDT, ETHUSDT, HAEDALUSDT, HEIUSDT, INJUSDT, LINKUSDT, LTCUSDT, MBOXUSDT, NEARUSDT, NMRUSDT, NXPCUSDT, ONDOUSDT, ONGUSDT, RESOLVUSDT, SOLUSDT, SUIUSDT, SYSUSDT, TAOUSDT, TONUSDT, WLDUSDT, XRPUSDT, ZECUSDT`
- Months: `2017-08 to 2026-07`

## Split Settings
- Split mode: `fixed`
- Walk-forward enabled: `0`
- Embargo minutes: `0`

## Model Settings
- Model kind: `lightgbm`
- Objective mode: `classification`
- Threshold objective: `profit_balanced`
- Position sizing mode: `fixed_fraction`

## Execution Settings
- Fee mode: `fixed`
- Fee: `0.001000`
- Slippage: `0.000500`
- Latency penalty bps: `0.00`
- Max open positions: `0`
- Trade regime filter: `none`
- Trade regime breadth threshold: `0.5000`

## Chronological Execution Audit
- Audit status: `already_chronological`
- Candidate input rows: `3`
- Candidate input time decreases: `0`
- Max backward jump minutes: `0`
- Execution buckets: `3`
- Max bucket candidates: `1`
- Max same-timestamp executed trades: `1`
- Max top-k bucket selected count: `0`
- Top-k bucket limit violations: `0`
- Max concurrent positions: `1`
- Max simultaneous capital: `1000.0000`
- Max capital usage vs equity: `0.1000`
- Capital over-allocation checks: `0`

## Cost / Score Diagnostics
- Raw signals before regime filter: `3`
- Raw signals blocked by regime filter: `0`
- Executable candidates blocked by regime filter: `0`
- Fold-mean gross return before costs: `0.007028`
- Fold-mean net return after costs: `0.005528`
- Trade-weighted gross return before costs: `0.000000`
- Trade-weighted net return after costs: `0.000000`
- Active-fold net return after costs: `0.000000`
- Avg cost drag per trade: `0.001500`
- Total cost drag: `4.0869`
- Executed winner score avg: `0.990770`
- Executed loser score avg: `0.000000`
- Winner-minus-loser score gap: `0.000000`

## Artifact Coverage
- Candidate artifact: ``
- Candidate artifact rows: `0`
- Candidate artifact full coverage: `0`
- Portfolio ledger: `C:\Users\alexa\OneDrive\Desktop\GitHub_Files\Desktop\Github_Folder\CoinPredictor\runs\portfolio_ledger_smoke\kline_growth_portfolio_events.csv.gz`
- Portfolio events: `8`
- Reconciliation status: `passed`
- Drawdown precision: `exact_realized`
- Exact realized drawdown: `0.000000`
- Mark-to-market drawdown available: `0`
- Maximum capital utilization: `0.100000`
- Maximum per-symbol exposure: `0.100000`

## Walk-Forward Summary
- Folds total: `0`
- Folds active: `0`
- Profitable fold rate: `0.0000`
- Median fold return: `0.0000`
- Worst fold drawdown: `0.0000`

## Calibration Summary
- Brier score: `0.051046`
- Expected calibration error: `0.106459`
- Max calibration error: `0.802250`
- Calibration skipped reason: `prediction_output_empty`

## Ranking / Tail Diagnostics
- Ranking rows available: `0`
- Score sources: ``
- Trade-score top decile net return after costs: `0.000000`
- Trade-score top 1% net return after costs: `0.000000`
- Trade-score top 5% net return after costs: `0.000000`
- Trade-score top decile label rate: `0.0000`
- Trade-score net-return monotonicity: `0.0000`
- Trade-score top decile symbol concentration: `0.0000`
- Trade-score top decile month concentration: `0.0000`
- Ranker-score top decile net return after costs: `0.000000`
- Ranker-score top 1% net return after costs: `0.000000`
- Ranker-score top 5% net return after costs: `0.000000`
- Ranker-score net-return monotonicity: `0.0000`
- Ranker-score tail warning: ``

## Threshold Rejection Diagnostics
- Candidate rows available: `15`
- Primary rejection: `rejected_over_max_trades`
- Primary rejection count: `7`
- Best candidate avg net return: `0.034766`
- Best candidate threshold: `0.992525`
- Best candidate trades: `10`
- Best top-decile net return: `0.021475`
- Best candidate positive-utility recall: `1.0000`
- Max missed positive utility: `0.212418`
- Max high-score loser share: `0.6667`
- Best raw-candidate top-decile utility: `0.034766`
- Candidate utility diagnostics skipped: `0`
- Near-miss candidates blocked only by avg-net floor: `0`
- Best near-miss split/fold: `` / `0`
- Best near-miss threshold: `0.000000`
- Best near-miss trades: `0`
- Best near-miss avg net return: `0.000000`
- Best near-miss top 1% net return: `0.000000`
- Best near-miss top-decile net return: `0.000000`
- Best near-miss top-symbol concentration: `0.0000`
- Best near-miss top-3 concentration: `0.0000`
- Best near-miss rejection flags: ``

## Robustness Gates
- Gate mode: `warn`
- Gate action: `warn`
- Gate status: `failed`
- Failed checks: `trade_count 0 < 10; top_symbol_share 0.9631 > 0.7000`
- Strength label: `robustness_rejected`

## Regime Summary
- Best regime by avg trade return: ``
- Worst regime by avg trade return: ``
- Best regime by profit: ``
- Worst regime by profit: ``

## Symbol Robustness Summary
- Profitable symbols: `2`
- Losing symbols: `0`
- Top-1 concentration: `0.9631`
- Top-3 concentration: `1.0000`
- Best symbol: `BNBUSDT`
- Worst symbol: `LTCUSDT`

## Feature Stability Summary
- Top stable features: ``

## Baseline Comparison
- Best baseline: `simple_momentum`
- Model vs best baseline return delta: `-0.0055`
- Model vs best baseline profit delta: `-0.0530`

## Main Risks / Limitations
- Buy-and-hold style baselines are skipped unless a continuous price path is available in the evaluation dataset.
- Regime summaries are derived only from columns available at prediction time.
- Execution realism is still approximate because the dataset does not include full exchange microstructure.

## Conclusion
- Overall accepted: `1`
- Rejection reason: ``
- Robustness gate status: `failed`
- Profitable but fragile: `0`
- Walk-forward acceptance: `n/a (disabled)`
