# CoinPredict

Build the C++ scraper/logistic baseline:

```bash
cd Desktop/Github_Folder/CoinPredictor
g++ -std=c++11 main.cpp DataScraper.cpp MonthSevenTester.cpp -o coin_predictor
g++ -std=c++11 -DDATASCRAPER_STANDALONE DataScraper.cpp MonthSevenTester.cpp -o data_scraper
```

Generate Binance 1m kline samples and run the logistic baseline:

```bash
./coin_predictor train BTCUSDT --months 8 --label-mode target_stop --tie-policy stop_first
```

To use each symbol's full available Binance history and split it by that
symbol's own month count, use ratio mode:

```bash
./coin_predictor train BTCUSDT ETHUSDT SOLUSDT --months all --split-mode ratio --train-ratio 0.70 --validation-ratio 0.15 --test-ratio 0.15 --label-mode target_stop --tie-policy stop_first --min-net-return 0 --generate-only
```

In ratio mode, the earliest months are training, the next months are validation,
and the latest months are test for each symbol separately.
Use `--generate-only` for large LightGBM runs; it streams
`kline_growth_training.csv` to disk and skips the in-memory C++ logistic
baseline.

`target_stop` is the default label mode. Useful options include
`--label-mode future_high|target_stop`, `--tie-policy stop_first|target_first|skip`,
`--min-net-return`,
`--growth-threshold`, `--upside-target`, `--downside-stop`,
`--prediction-window`, `--learning-rate`, `--epochs`, `--l2`,
`--split-mode`, `--train-ratio`, `--validation-ratio`, `--test-ratio`,
`--generate-only`,
`--positive-weight-cap`, `--initial-capital`, `--max-position-fraction`,
`--max-volume-fraction`, `--max-trades-per-period`, `--trade-period-minutes`,
`--holding-period-minutes`, `--min-validation-trades`,
`--max-validation-trades`, `--min-validation-precision`,
`--threshold-objective`, `--profit-safety`, `--fee`, and `--slippage`.

Run the boosted-tree path after `kline_growth_training.csv` is generated:

```bash
python3 gbdt_pipeline.py --walk-forward
```

`gbdt_pipeline.py` uses LightGBM when it is installed and otherwise uses the
internal boosted-stump fallback. The fallback is also available explicitly with
`--model internal`.

For profit-based runs, threshold tuning now has two safety modes. The default
`--profit-safety explore` picks the best available validation threshold even if
the validation month is slightly negative, which is useful for diagnosing whether
the old `1.01` no-trade result was too strict. Use `--profit-safety strict` to
restore the conservative behavior where every losing validation threshold
selects `1.01` and reports zero trades.

The Python runner also adds adaptive validation thresholds from the model's
actual probability distribution, so compressed probabilities are no longer
missed by a fixed grid. Disable that with `--disable-adaptive-thresholds` when
you want an exact fixed-grid comparison.

Recommended laptop-safe rerun after generating the full training CSV:

```bash
.venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --threshold-objective avg_profit --profit-safety explore --min-validation-trades 5 --max-validation-trades 250 --min-validation-precision 0.25 --min-selected-threshold 0.90 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 10 --trade-period-minutes 60 --holding-period-minutes 5 --feature-storage memmap32 --cache-dir .gbdt_cache --max-train-rows 2000000 --max-validation-rows 1000000 --max-final-train-rows 2000000 --prediction-batch-rows 200000 --prediction-output-mode trades --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 2>&1 | tee gbdt_run.log
```

The Python runner defaults to disk-backed `memmap32` feature storage for
full-database LightGBM runs. This keeps the full feature table out of RAM while
preserving the same rows, features, labels, threshold tuning, and walk-forward
logic. Use `--feature-storage matrix32` when you have plenty of RAM and want a
faster in-memory run, `--feature-storage memmap64` or `matrix64` when you want
more numeric precision, and `--feature-storage list` only for debugging older
Python sequence behavior. You can place the temporary memmap file on a specific
drive with `--memmap-dir /path/to/fast/disk`.
LightGBM defaults to `--max-bin 63`, `--lightgbm-histogram-pool-mb 128`,
`--subsample-for-bin 100000`, and `--n-jobs 2` to keep laptop RAM predictable;
raise those when you have more
memory and want to trade RAM for speed or slightly finer tree bins.
For very large CSVs, the runner now also caps the in-memory LightGBM fit
matrices with deterministic samples: `--max-train-rows 2000000`,
`--max-validation-rows 1000000`, and `--max-final-train-rows 2000000` by
default. The default `--train-sample-mode stratified` keeps all positive
examples when they fit, caps positives at `--max-positive-sample-fraction`
otherwise, and fills the rest with chronological negatives. That keeps rare
event coverage without training on a fake 50/50 market. Thresholds are still
retuned on the full validation period with batched prediction unless
`--skip-full-validation-retune` is set. Test and walk-forward prediction also
run in batches via `--prediction-batch-rows`, and prediction CSVs default to
`--prediction-output-mode trades` to avoid writing tens of millions of no-trade
rows. Use `--prediction-output-mode all` when you need a probability row for
every test sample.

Disk-backed runs cache parsed features and compact metadata. The first run
builds the cache; later experiments skip the full CSV parse when its path,
timestamp, size, and feature dtype still match. Use `--rebuild-cache` to force a
refresh, `--disable-cache` for a one-off transient memmap, and `--cache-dir` to
put the reusable cache on a fast local drive.

Progress lines include elapsed time and, when the optional `psutil` package is
installed, RSS memory usage. With `python -u` and `tee`, they appear immediately.
Watch an existing run from another WSL terminal with:

```bash
tail -f gbdt_run.log
```

LightGBM model selection uses chronological validation early stopping. Tune it
with `--early-stopping-rounds`, `--eval-metric binary_logloss|auc`, and
`--log-evaluation-period`.

When `target_stop` labels are used, generated CSVs include `trade_return`, so
profit metrics use the target/stop exit return rather than only the
end-of-window close return.

The evaluator simulates a shared cash portfolio instead of applying a cooldown.
Each entry is limited to the smallest of the available cash, a fixed percentage
of starting capital (`--max-position-fraction`, default `0.10`), and a
percentage of that candle's quote volume (`--max-volume-fraction`, default
`0.01`). The starting-capital cap stays fixed as capital changes. Open trades
lock cash for `--holding-period-minutes`, and at most
`--max-trades-per-period` entries are allowed globally during each
rolling `--trade-period-minutes` interval. Generated prediction CSVs include the
selected `position_size`.

Thresholds must also produce at least `--min-validation-trades` validation
trades, no more than `--max-validation-trades` validation trades when that
value is nonzero, and at least `--min-validation-precision` before they can be
selected. The default Python threshold objective is `avg_profit`, which
optimizes average portfolio profit per validation trade. Use
`--min-selected-threshold 0` when you want to disable the probability-floor
guardrail for experiments.

## Windows PowerShell With WSL

Build:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && g++ -std=c++11 main.cpp DataScraper.cpp MonthSevenTester.cpp -o coin_predictor && g++ -std=c++11 -DDATASCRAPER_STANDALONE DataScraper.cpp MonthSevenTester.cpp -o data_scraper'
```

Generate a large CSV with all available months:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && ./coin_predictor train BTCUSDT ETHUSDT SOLUSDT --months all --split-mode ratio --label-mode target_stop --tie-policy stop_first --generate-only'
```

Run the laptop-safe LightGBM command and watch progress:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --feature-storage memmap32 --cache-dir .gbdt_cache --threshold-objective avg_profit --profit-safety explore --max-train-rows 2000000 --max-validation-rows 1000000 --max-final-train-rows 2000000 --prediction-batch-rows 200000 --prediction-output-mode trades --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 2>&1 | tee gbdt_run.log'
```

For a machine with more RAM, try `--max-bin 127`,
`--lightgbm-histogram-pool-mb 256`, `--n-jobs 4`, and larger capped fit
samples. Keep `memmap32` unless profiling shows that an in-memory matrix is
worth the extra RAM.

## Outputs

The boosted-tree runner writes:

- `kline_growth_predictions_gbdt.csv`: fixed-split executed trades by default.
- `kline_growth_metrics_gbdt.csv`: fixed-split classification and portfolio metrics.
- `kline_growth_walkforward_metrics.csv`: fold metrics plus a walk-forward average.
- `kline_growth_predictions_gbdt_walkforward.csv`: walk-forward executed trades.
- `kline_growth_feature_importance.csv`: model feature importance.
- `kline_growth_run_summary.json`: current run arguments, cache details, parameters, and metrics.
- `kline_growth_experiment_summary.csv`: one appended comparison row per experiment.

Use the portfolio metrics for strategy comparisons: `ending_capital`,
`portfolio_profit`, `portfolio_return`, `max_capital_drawdown`,
`average_profit_per_trade`, and `worst_trade`. Raw classification fields begin
with `raw_`; executed-trade precision and recall reflect portfolio constraints.

## Offline Checks

These checks do not download Binance data:

```bash
./data_scraper --self-test
.venv/bin/python -m unittest -v test_gbdt_pipeline.py
```
