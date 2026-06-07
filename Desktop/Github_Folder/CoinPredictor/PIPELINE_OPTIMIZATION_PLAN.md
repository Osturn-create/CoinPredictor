# Pipeline Optimization Plan

## Current Data Flow

1. `DataScraper.cpp` downloads Binance 1m kline archives and generates either:
   - a combined `kline_growth_training.csv`, or
   - a sharded dataset directory with one CSV per `symbol/month`
2. `gbdt_pipeline.py` loads the dataset, validates the training manifest, and either:
   - parses CSV input into compact arrays / memmaps, or
   - reuses `.gbdt_cache` in `--cache-only` mode
3. The Python runner builds chronological train / validation / test views, fits LightGBM or fallback models, tunes thresholds, and runs fixed-split plus walk-forward evaluation.
4. Predictions, metrics, diagnostics, summaries, and experiment results are written to CSV / JSON outputs.

## Slowest Stages

- First-time CSV parse and cache build on very large datasets
- LightGBM fit / retune cycles during walk-forward folds
- Writing very large prediction tables when non-trade rows are included
- Repeated experiment runs when cache reuse is not enforced

## Largest Disk Users

- `kline_growth_training.csv` or shard CSVs
- `.gbdt_cache` feature memmaps and metadata archives
- prediction CSVs from large walk-forward runs
- repeated experiment result folders and logs

## Memory-Heavy Operations

- Fit matrices for LightGBM train / validation / final-train samples
- Large validation / test prediction arrays
- Temporary aggregate views while building monolithic caches from many shards
- Full-dataset metadata arrays when cache is attached

## Safe Optimizations

These are safe because they do not change features, labels, thresholds, or trading logic:

- Keep `memmap32` as the default large-run feature store
- Reuse compatible cache in `--cache-only` mode and fail fast on mismatches
- Add pipeline profiling CSV output and cache inspection commands
- Use `--results-dir` to isolate outputs per run
- Compress result CSVs after writing
- Support sharded dataset input so new symbol-month data can be added without rebuilding every old shard cache
- Add cache cleanup review tools and disk guardrails

## Backward-Compatible Storage Extensions

- Cache manifests can carry richer metadata while still loading older manifests
- Gzip-compressed dataset input (`.csv.gz`) is acceptable because cache build still streams rows
- Sharded dataset directories are opt-in; the combined CSV path remains valid
- Output compression is post-processing only; existing filenames and commands still work when compression is off

## Deferred Changes

These are useful, but larger or riskier than the current pass:

- A true raw-download reuse layer in `DataScraper.cpp` with `--raw-data-dir`, download manifests, and retry / worker controls
- C++ pipeline profiling parity with the Python profile CSV
- Native generator support for writing `kline_growth_training.csv.gz`
- Full per-month cache shards that let fold loading skip unrelated months inside one aggregate run
- A dedicated incremental update wrapper (`pipeline_update.py` or similar)

## Practical Next Steps

1. Keep using shard dataset generation for future symbol additions.
2. Use `--cache-only`, `--inspect-cache`, and the pipeline profile CSV for repeated experiments.
3. Add raw-download reuse and generator-side profiling only after the current modeling / backtesting workflow stabilizes.
