# CoinPredictor Experiment Report

## Dataset Summary
- Input: `/tmp/tmprra_u7fb/synthetic.csv`
- Rows: `1`
- Features: `1`
- Symbols: `TESTUSDT`
- Months: `2020-01 to 2020-01`

## Split Settings
- Split mode: `ratio`
- Walk-forward enabled: `0`
- Embargo minutes: `0`

## Model Settings
- Model kind: `lightgbm`
- Objective mode: `classification`
- Threshold objective: `ev`
- Position sizing mode: `fixed_fraction`

## Execution Settings
- Fee mode: `fixed`
- Fee: `0.000000`
- Slippage: `0.000000`
- Latency penalty bps: `0.00`
- Max open positions: `0`

## Walk-Forward Summary
- Folds total: `1`
- Folds active: `1`
- Profitable fold rate: `1.0000`
- Median fold return: `0.0100`
- Worst fold drawdown: `0.2000`

## Calibration Summary
- Brier score: `0.000000`
- Expected calibration error: `0.000000`
- Max calibration error: `0.000000`

## Regime Summary
- Best regime by avg trade return: ``
- Worst regime by avg trade return: ``
- Best regime by profit: ``
- Worst regime by profit: ``

## Symbol Robustness Summary
- Profitable symbols: `0`
- Losing symbols: `0`
- Top-1 concentration: `0.0000`
- Top-3 concentration: `0.0000`
- Best symbol: ``
- Worst symbol: ``

## Feature Stability Summary
- Top stable features: ``

## Baseline Comparison
- Best baseline: `no_trade`
- Model vs best baseline return delta: `0.0100`
- Model vs best baseline profit delta: `10.0000`

## Main Risks / Limitations
- Buy-and-hold style baselines are skipped unless a continuous price path is available in the evaluation dataset.
- Regime summaries are derived only from columns available at prediction time.
- Execution realism is still approximate because the dataset does not include full exchange microstructure.

## Conclusion
- Walk-forward acceptance passed: `1`
- Acceptance reasons: ``
