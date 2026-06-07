# CoinPredict

Build the C++ scraper/logistic baseline:

```bash
cd Desktop/Github_Folder/CoinPredictor
g++ -std=c++11 main.cpp DataScraper.cpp -o coin_predictor
g++ -std=c++11 -DDATASCRAPER_STANDALONE DataScraper.cpp -o data_scraper
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

If you want to add new coins later without reparsing every old CSV into cache
again, use shard output instead of one giant combined CSV:

```bash
./coin_predictor train BTCUSDT ETHUSDT SOLUSDT \
  --months all \
  --split-mode ratio \
  --train-ratio 0.70 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --label-mode target_stop \
  --target-exit-mode first_decline \
  --upside-target 0.02 \
  --downside-stop 0.02 \
  --market-regime-features \
  --market-breadth-features \
  --generate-only \
  --shard-output-dir shard_dataset \
  --skip-combined-output
```

That writes:

- `shard_dataset/kline_growth_dataset.meta.json`
- `shard_dataset/shards/SYMBOL/YYYY-MM.csv`
- `shard_dataset/shards/SYMBOL/YYYY-MM.meta.json`

Later, add only new coins to that same directory:

```bash
./coin_predictor train AAVEUSDT AVAXUSDT LINKUSDT \
  --months all \
  --split-mode ratio \
  --train-ratio 0.70 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --label-mode target_stop \
  --target-exit-mode first_decline \
  --upside-target 0.02 \
  --downside-stop 0.02 \
  --market-regime-features \
  --market-breadth-features \
  --generate-only \
  --shard-output-dir shard_dataset \
  --skip-combined-output
```

Then run the Python pipeline against the shard directory:

```bash
.venv/bin/python -u gbdt_pipeline.py --input shard_dataset --feature-storage memmap32 --cache-dir .gbdt_cache --walk-forward
```

Existing shard caches are reused when the shard CSV and shard settings did not
change. The aggregate dataset cache refreshes when shard inventory changes, so
new symbol-month shards can be added without rebuilding every old shard cache.

For future full-history rebuilds, the recommended generator path now enables
BTC/ETH market-regime features:

```bash
./coin_predictor train BTCUSDT ETHUSDT SOLUSDT --months all --split-mode ratio --train-ratio 0.70 --validation-ratio 0.15 --test-ratio 0.15 --label-mode target_stop --target-exit-mode first_decline --upside-target 0.02 --downside-stop 0.02 --market-regime-features --generate-only
```

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
of total account equity (`cash + invested`, via `--max-position-fraction`,
default `0.10`), and a percentage of that candle's quote volume
(`--max-volume-fraction`, default `0.01`). Open trades
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
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && g++ -std=c++11 main.cpp DataScraper.cpp -o coin_predictor && g++ -std=c++11 -DDATASCRAPER_STANDALONE DataScraper.cpp -o data_scraper'
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

## 7.8GB WSL Same-Spec EV Walk-Forward Run

These commands keep the same working WSL memory and performance profile instead
of shrinking the run to a much smaller job. They add EV scoring, top-K trade
selection, optional Platt calibration, and RSS guardrails on top of the
existing memmap/cache pipeline.

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --threshold-objective ev --trade-selection topk_ev --top-k-per-minute 3 --calibration platt --calibration-max-rows 500000 --ev-safety-margin 0.002 --profit-safety explore --min-validation-trades 5 --max-validation-trades 250 --min-validation-precision 0.25 --min-selected-threshold 0.90 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 10 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 2>&1 | tee gbdt_7_8gb_same_specs_ev.log'
```

The stricter version keeps the same specs and rejects runs that fail the
walk-forward stability checks:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --require-positive-walkforward --min-profitable-fold-rate 0.55 --min-median-fold-return 0.0 --min-mean-fold-return 0.0 --feature-storage memmap32 --cache-dir .gbdt_cache --threshold-objective ev --trade-selection topk_ev --top-k-per-minute 3 --calibration platt --calibration-max-rows 500000 --ev-safety-margin 0.002 --profit-safety explore --min-validation-trades 5 --max-validation-trades 250 --min-validation-precision 0.25 --min-selected-threshold 0.90 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 10 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 2>&1 | tee gbdt_7_8gb_same_specs_ev_strict.log'
```

## Next algorithm upgrades: regime features and return scoring

The earlier EV + calibration + top-K work made the strategy safer, but the
latest walk-forward results showed the edge was still sparse and
regime-dependent. The next step is to let the model see BTC/ETH regime context
and to score trades by expected return as well as by binary classification.

Build the scraper and generator:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && g++ -std=c++11 main.cpp DataScraper.cpp -o coin_predictor && g++ -std=c++11 -DDATASCRAPER_STANDALONE DataScraper.cpp -o data_scraper'
```

Generate a regime-aware dataset:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && ./coin_predictor train BTCUSDT ETHUSDT SOLUSDT --months all --split-mode ratio --label-mode target_stop --target-exit-mode first_decline --upside-target 0.02 --downside-stop 0.02 --market-regime-features --generate-only'
```

Classification + EV scoring:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --objective-mode classification --threshold-objective ev --trade-selection topk_score --trade-score ev --top-k-per-minute 3 --calibration platt --calibration-max-rows 500000 --ev-safety-margin 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 250 --min-validation-precision 0.20 --min-selected-threshold 0.80 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 10 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 --acceptance-tier exploration 2>&1 | tee gbdt_regime_ev_7_8gb.log'
```

Return-regression scoring:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --objective-mode return_regression --trade-selection topk_score --trade-score predicted_return --top-k-per-minute 3 --min-predicted-net-return 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 250 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 10 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 --acceptance-tier exploration 2>&1 | tee gbdt_regime_return_regression_7_8gb.log'
```

Hybrid probability × return scoring:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --objective-mode hybrid --trade-selection topk_score --trade-score hybrid --top-k-per-minute 3 --calibration platt --calibration-max-rows 500000 --hybrid-min-score 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 250 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 10 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 --acceptance-tier exploration 2>&1 | tee gbdt_regime_hybrid_7_8gb.log'
```

To sweep a small same-spec grid without loading the dataset inside the runner,
use:

```bash
.venv/bin/python run_experiments.py --profile 7.8gb --max-runs 6
```

## Overtrade-control follow-up after classification EV

The last classification EV walk-forward run improved total walk-forward profit,
but it still had many inactive folds and a few ugly high-trade losing folds.
This next round keeps the same 7.8GB WSL profile and existing `.gbdt_cache`,
and focuses on reducing bad active overtrading instead of making the model
larger.

Quick checks before a new run:

```bash
python3 -m py_compile gbdt_pipeline.py
python3 -m py_compile run_experiments.py
python3 -m unittest discover -v
```

These runs are cache-aware. They should load the existing `.gbdt_cache` unless
you explicitly remove it.

## Cache-only optimized runs

Use `--cache-only` when `.gbdt_cache` is already built. It prevents accidental
fallback to parsing the full dataset, keeps the current cached dataset exactly
unchanged, and is the recommended mode for repeated experiments on the same
cache.

Recommended cache-only classification run:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --objective-mode classification --threshold-objective ev --trade-selection topk_score --trade-score ev --top-k-per-minute 1 --calibration platt --calibration-max-rows 500000 --ev-safety-margin 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --min-validation-precision 0.20 --min-selected-threshold 0.80 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --holding-period-minutes 5 --max-trades-per-day 20 --max-trades-per-fold 100 --max-losing-trades-per-day 5 --max-daily-drawdown 0.015 --pause-after-drawdown-minutes 240 --threshold-drawdown-penalty 0.25 --threshold-trade-count-penalty 0.0005 --target-validation-trades 75 --overactive-trade-threshold 100 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 --acceptance-tier exploration 2>&1 | tee gbdt_cache_only_optimized.log'
```

Fast smoke test using the existing cache metadata only:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --smoke-test-cache --feature-storage memmap32 --cache-dir .gbdt_cache'
```

## Pipeline profiling, cache inspection, and cleanup

The Python runner now writes a stage profile CSV by default:

- `kline_growth_pipeline_profile.csv`

Each row includes stage timing, RSS memory at the start/end, row counts when
available, throughput when available, and a short extra-info field. Override
the location with `--profile-out`, or disable it with `--disable-profile`.

Inspect an existing cache without training:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --inspect-cache --feature-storage memmap32 --cache-dir .gbdt_cache'
```

Review stale cache files without deleting them:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-cleanup --dry-run --feature-storage memmap32 --cache-dir .gbdt_cache'
```

Delete cache files that do not belong to the current input only after you have
reviewed the dry run:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-cleanup --confirm-delete --feature-storage memmap32 --cache-dir .gbdt_cache'
```

Useful disk guardrails:

- `--min-free-disk-gb 20`
- `--abort-on-low-disk`
- `--max-cache-size-gb 80`

These do not change results; they just make long runs less likely to die
late for boring reasons.

## Results directories and compressed outputs

Use `--results-dir` to keep one run's outputs together instead of overwriting
top-level CSVs:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --model lightgbm --split-mode ratio --walk-forward --feature-storage memmap32 --cache-dir .gbdt_cache --results-dir results/manual_run --profile-out results/manual_run/kline_growth_pipeline_profile.csv'
```

To compress the result CSVs after writing them:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --model lightgbm --split-mode ratio --walk-forward --feature-storage memmap32 --cache-dir .gbdt_cache --results-dir results/compressed_run --compress-outputs'
```

That leaves the JSON summary uncompressed and writes the large CSV artifacts as
`.gz` files.

`gbdt_pipeline.py` also accepts `.csv.gz` as `--input`, so future datasets can
be compressed on disk and still be streamed into cache builds without fully
decompressing them first.

`run_experiments.py` now creates a timestamped results root by default, places
each experiment in its own subdirectory with `--results-dir`, and can skip
completed runs with `--resume`:

```bash
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u run_experiments.py --profile 7.8gb --max-runs 6 --results-root results/experiments_same_spec --resume'
```

## Optimization plan reference

The implementation-oriented optimization note for the current pipeline lives at:

- `Desktop/Github_Folder/CoinPredictor/PIPELINE_OPTIMIZATION_PLAN.md`

Classification top-1 less-overtrade run:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --objective-mode classification --threshold-objective ev --trade-selection topk_score --trade-score ev --top-k-per-minute 1 --calibration platt --calibration-max-rows 500000 --ev-safety-margin 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --min-validation-precision 0.20 --min-selected-threshold 0.80 --threshold-drawdown-penalty 0.05 --threshold-trade-count-penalty 0.0005 --target-validation-trades 100 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --max-trades-per-day 0 --max-trades-per-fold 0 --max-losing-trades-per-day 0 --max-daily-drawdown 0 --pause-after-drawdown-minutes 0 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 --acceptance-tier exploration 2>&1 | tee gbdt_overtrade_classification.log'
```

Optional hybrid follow-up after the classification run:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --objective-mode hybrid --trade-selection topk_score --trade-score hybrid --top-k-per-minute 1 --calibration platt --calibration-max-rows 500000 --hybrid-min-score 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --threshold-drawdown-penalty 0.05 --threshold-trade-count-penalty 0.0005 --target-validation-trades 100 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 --acceptance-tier exploration 2>&1 | tee gbdt_overtrade_hybrid.log'
```

To compare the four quick top-K / overtrade-control experiments without
rebuilding the cache:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u run_experiments.py --profile 7.8gb-overtrade-check --max-runs 4 2>&1 | tee gbdt_overtrade_experiments.log'
```

## EV target auto-detection from training manifest

`gbdt_pipeline.py` reads `upside_target` and `downside_stop` from
`kline_growth_training.meta.json` unless the command explicitly passes
`--upside-target` or `--downside-stop`. This keeps EV scoring aligned with the
dataset, so a CSV generated with `0.02` targets is no longer accidentally scored
with the Python default of `0.05`.

This cached run relies on the manifest targets and does not pass explicit EV
target values:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --model lightgbm --split-mode ratio --walk-forward --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --objective-mode classification --threshold-objective ev --trade-selection topk_score --trade-score ev --top-k-per-minute 1 --calibration platt --calibration-max-rows 500000 --ev-safety-margin 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --min-validation-precision 0.20 --min-selected-threshold 0.80 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --holding-period-minutes 5 --max-trades-per-day 20 --max-trades-per-fold 100 --max-losing-trades-per-day 5 --max-daily-drawdown 0.015 --pause-after-drawdown-minutes 240 --threshold-drawdown-penalty 0.25 --threshold-trade-count-penalty 0.0005 --target-validation-trades 75 --overactive-trade-threshold 100 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 --acceptance-tier exploration 2>&1 | tee gbdt_manifest_ev_targets.log'
```

## Optional market-breadth features

The C++ generator supports `--market-breadth-features`. It is off by default.
When enabled, it adds leakage-safe breadth features computed from symbols in the
generated dataset, using only candles available at or before each timestamp.
Use `--market-breadth-min-symbols N` to require more or fewer aligned symbols
before breadth values are considered valid. The default is `5`.

Example regeneration command:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && ./coin_predictor train BTCUSDT ETHUSDT SOLUSDT --months all --split-mode ratio --label-mode target_stop --target-exit-mode first_decline --upside-target 0.02 --downside-stop 0.02 --market-regime-features --market-breadth-features --market-breadth-min-symbols 5 --generate-only'
```

Breadth columns change the CSV feature set, so regenerating with this flag will
need a compatible new cache. Existing cached runs without breadth features still
work.

Recommended sharded 25-coin rebuild:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && ./coin_predictor train BTCUSDT ETHUSDT BNBUSDT SOLUSDT SUIUSDT DOGEUSDT XRPUSDT TONUSDT NEARUSDT ONDOUSDT TAOUSDT INJUSDT WLDUSDT AVAXUSDT LTCUSDT LINKUSDT AAVEUSDT ADAUSDT BCHUSDT TIAUSDT UNIUSDT HBARUSDT SEIUSDT FETUSDT DOTUSDT --months all --split-mode ratio --train-ratio 0.70 --validation-ratio 0.15 --test-ratio 0.15 --label-mode target_stop --target-exit-mode first_decline --upside-target 0.02 --downside-stop 0.02 --market-regime-features --market-breadth-features --market-breadth-min-symbols 5 --generate-only --shard-output-dir shard_dataset_25 --skip-combined-output'
```

## Hybrid Upgrades

Hybrid mode now supports:

- regression target variants: `trade_return`, `net_return`, `clipped_trade_return`, `clipped_net_return`, `risk_adjusted_return`
- regression calibration: `none`, `linear`, `isotonic-lite`
- risk-adjusted hybrid scoring with residual uncertainty penalties
- dynamic BTC / volatility regime thresholds
- optional second-stage meta filter
- optional fixed-split ensemble windows

Recommended less-sparse hybrid run on the 7.8GB profile:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --input shard_dataset_25 --cache-dir .gbdt_cache_shard25 --model lightgbm --split-mode ratio --walk-forward --objective-mode hybrid --trade-selection topk_score --trade-score hybrid --top-k-per-minute 1 --regression-calibration linear --regression-target clipped_net_return --hybrid-score-mode risk_adjusted --hybrid-uncertainty-method bucket_residual --hybrid-uncertainty-penalty 0.25 --ev-payoff-mode predicted_return --symbol-filter-stage candidate_blend --threshold-tiebreaker balanced --threshold-target-trades 75 --calibration platt --calibration-max-rows 500000 --hybrid-min-score 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --min-validation-precision 0.20 --min-selected-threshold 0.80 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --holding-period-minutes 5 --max-trades-per-day 20 --max-trades-per-fold 100 --max-losing-trades-per-day 5 --max-daily-drawdown 0.015 --pause-after-drawdown-minutes 240 --threshold-drawdown-penalty 0.25 --threshold-trade-count-penalty 0.0005 --target-validation-trades 75 --overactive-trade-threshold 100 --fee 0.001 --slippage 0.0005 --test-slippage-multiplier 1.0 --validation-slippage-multiplier 1.0 --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 --acceptance-tier exploration 2>&1 | tee gbdt_hybrid_upgraded.log'
```

Optional fixed-split ensemble example:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --model lightgbm --split-mode ratio --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --cache-dir .gbdt_cache --objective-mode hybrid --trade-selection topk_score --trade-score hybrid --calibration platt --calibration-max-rows 500000 --regression-calibration linear --regression-target clipped_net_return --hybrid-score-mode risk_adjusted --hybrid-uncertainty-method bucket_residual --hybrid-uncertainty-penalty 0.25 --dynamic-hybrid-thresholds btc_volatility_regime --meta-filter logistic --meta-filter-min-probability 0.55 --ensemble-windows 6,9 --hybrid-min-score 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --threshold-drawdown-penalty 0.05 --threshold-trade-count-penalty 0.0005 --target-validation-trades 100 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --prediction-output-mode trades --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 2>&1 | tee gbdt_hybrid_ensemble_fixed.log'
```

## Hybrid return-combination modes

Use `--hybrid-return-combination` to choose how hybrid mode combines the classifier and regression heads without changing the dataset or rebuilding the cache.

Legacy conservative mode:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --input shard_dataset_25 --cache-dir .gbdt_cache_shard25 --model lightgbm --split-mode ratio --walk-forward --objective-mode hybrid --trade-selection topk_score --trade-score hybrid --top-k-per-minute 1 --regression-calibration linear --regression-target clipped_net_return --hybrid-return-combination probability_times_return --hybrid-score-mode risk_adjusted --hybrid-uncertainty-method bucket_residual --hybrid-uncertainty-penalty 0.25 --calibration platt --calibration-max-rows 500000 --hybrid-min-score 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --min-validation-precision 0.20 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 2>&1 | tee gbdt_hybrid_probability_times_return.log'
```

Expected-return mode with a probability gate:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --input shard_dataset_25 --cache-dir .gbdt_cache_shard25 --model lightgbm --split-mode ratio --walk-forward --objective-mode hybrid --trade-selection topk_score --trade-score hybrid --top-k-per-minute 1 --regression-calibration linear --regression-target clipped_net_return --hybrid-return-combination expected_return --hybrid-min-probability 0.10 --hybrid-score-mode risk_adjusted --hybrid-uncertainty-method bucket_residual --hybrid-uncertainty-penalty 0.25 --calibration platt --calibration-max-rows 500000 --hybrid-min-score 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --min-validation-precision 0.20 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 2>&1 | tee gbdt_hybrid_expected_return.log'
```

Conditional-payoff mode from validation statistics:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u gbdt_pipeline.py --cache-only --input shard_dataset_25 --cache-dir .gbdt_cache_shard25 --model lightgbm --split-mode ratio --walk-forward --objective-mode hybrid --trade-selection topk_score --trade-score hybrid --top-k-per-minute 1 --hybrid-return-combination conditional_payoff --conditional-payoff-min-positive-rows 25 --conditional-payoff-min-negative-rows 25 --conditional-payoff-max-rows 500000 --hybrid-score-mode risk_adjusted --hybrid-uncertainty-method bucket_residual --hybrid-uncertainty-penalty 0.25 --calibration platt --calibration-max-rows 500000 --hybrid-min-score 0.001 --profit-safety explore --min-validation-trades 5 --max-validation-trades 150 --min-validation-precision 0.20 --positive-weight-cap 10 --initial-capital 10000 --max-position-fraction 0.10 --max-volume-fraction 0.01 --max-trades-per-period 5 --trade-period-minutes 60 --holding-period-minutes 5 --fee 0.001 --slippage 0.0005 --memory-budget-gb 7.8 --max-rss-gb 7.8 --feature-storage memmap32 --max-train-rows 1500000 --max-validation-rows 750000 --max-final-train-rows 1500000 --prediction-batch-rows 200000 --auc-sample-rows 1000000 --adaptive-threshold-sample-rows 1000000 --max-bin 63 --lightgbm-histogram-pool-mb 128 --subsample-for-bin 100000 --n-jobs 2 2>&1 | tee gbdt_hybrid_conditional_payoff.log'
```

Experiment runner profiles:

```powershell
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u run_experiments.py --profile hybrid-calibration --max-runs 6'
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u run_experiments.py --profile hybrid-risk-adjusted --max-runs 4'
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u run_experiments.py --profile hybrid-meta-filter --max-runs 4'
wsl bash -lc 'cd "/mnt/c/Users/alexa/OneDrive/Desktop/GitHub_Files/Desktop/Github_Folder/CoinPredictor" && .venv/bin/python -u run_experiments.py --profile hybrid-ensemble-small --max-runs 2'
```

The main files to save and compare after each run are:

- `kline_growth_metrics_gbdt.csv`
- `kline_growth_walkforward_metrics.csv`
- `kline_growth_walkforward_diagnostics.csv`
- `kline_growth_run_summary.json`
- `kline_growth_experiment_summary.csv`

## Outputs

The boosted-tree runner writes:

- `kline_growth_predictions_gbdt.csv`: fixed-split executed trades by default.
- `kline_growth_metrics_gbdt.csv`: fixed-split classification and portfolio metrics.
- `kline_growth_walkforward_metrics.csv`: fold metrics plus a walk-forward average.
- `kline_growth_walkforward_diagnostics.csv`: one row per walk-forward fold with fold activity, calibration, threshold, and portfolio diagnostics.
- `kline_growth_predictions_gbdt_walkforward.csv`: walk-forward executed trades.
- `kline_growth_feature_importance.csv`: model feature importance.
- `kline_growth_run_summary.json`: current run arguments, cache details, parameters, and metrics.
- `kline_growth_experiment_summary.csv`: one appended comparison row per experiment.
- `kline_growth_experiment_grid_results.csv`: experiment-runner comparison table across the same-spec grid.

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
