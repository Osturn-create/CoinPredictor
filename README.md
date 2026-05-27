# CoinPredict

Build the C++ scraper/logistic baseline:

```bash
cd Desktop/Github_Folder/CoinPredictor
g++ -std=c++11 main.cpp DataScraper.cpp MonthSevenTester.cpp -o coin_predictor
g++ -std=c++11 -DDATASCRAPER_STANDALONE DataScraper.cpp MonthSevenTester.cpp -o data_scraper
```

Generate Binance 1m kline samples and run the logistic baseline:

```bash
./coin_predictor train BTCUSDT --months 8
```

To use each symbol's full available Binance history and split it by that
symbol's own month count, use ratio mode:

```bash
./coin_predictor train BTCUSDT ETHUSDT SOLUSDT --months all --split-mode ratio --train-ratio 0.70 --validation-ratio 0.15 --test-ratio 0.15 --generate-only
```

In ratio mode, the earliest months are training, the next months are validation,
and the latest months are test for each symbol separately.
Use `--generate-only` for large LightGBM runs; it streams
`kline_growth_training.csv` to disk and skips the in-memory C++ logistic
baseline.

Useful options include `--label-mode future_high|target_stop`,
`--growth-threshold`, `--upside-target`, `--downside-stop`,
`--prediction-window`, `--learning-rate`, `--epochs`, `--l2`,
`--split-mode`, `--train-ratio`, `--validation-ratio`, `--test-ratio`,
`--generate-only`,
`--positive-weight-cap`, `--cooldown-minutes`, `--min-validation-trades`,
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

Recommended diagnostic rerun after generating the full training CSV:

```bash
.venv/bin/python gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --threshold-objective avg_profit --profit-safety explore --cooldown-minutes 60 --min-validation-trades 5 --max-validation-trades 250 --min-validation-precision 0.25 --min-selected-threshold 0.90 --max-trades-per-symbol-month 50 --positive-weight-cap 10
```

The Python runner defaults to disk-backed `memmap32` feature storage for
full-database LightGBM runs. This keeps the full feature table out of RAM while
preserving the same rows, features, labels, threshold tuning, and walk-forward
logic. Use `--feature-storage matrix32` when you have plenty of RAM and want a
faster in-memory run, `--feature-storage memmap64` or `matrix64` when you want
more numeric precision, and `--feature-storage list` only for debugging older
Python sequence behavior. You can place the temporary memmap file on a specific
drive with `--memmap-dir /path/to/fast/disk`.
LightGBM also defaults to `--max-bin 127`, `--lightgbm-histogram-pool-mb 256`,
and `--n-jobs 4` to keep RAM more predictable; raise those when you have more
memory and want to trade RAM for speed or slightly finer tree bins.

When `target_stop` labels are used, generated CSVs include `trade_return`, so
profit metrics use the target/stop exit return rather than only the
end-of-window close return.

The evaluator also supports a per-symbol signal cooldown. This prevents a bad
minute-by-minute score from becoming thousands of overlapping trades. The C++
baseline defaults cooldown to the prediction window; the Python boosted-tree
runner defaults to `--cooldown-minutes 10`. Thresholds must also produce at
least `--min-validation-trades` validation trades, no more than
`--max-validation-trades` validation trades when that value is nonzero, and at
least `--min-validation-precision` before they can be selected. The default
Python threshold objective is `avg_profit`, which optimizes average net return
per validation trade instead of letting a high-volume threshold win only because
it produced more total validation profit.

The Python runner also has two execution guardrails aimed at walk-forward
stability: `--min-selected-threshold` rejects loose probability thresholds, and
`--max-trades-per-symbol-month` prevents one symbol/month from dominating the
backtest. Set either to `0` when you want to disable that guardrail for
experiments.
