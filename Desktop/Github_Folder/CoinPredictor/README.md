# CoinPredictor

## Execution-time dynamic exits without rebuilding shards/cache

The pipeline supports an execution-only trailing-stop overlay:

```bash
--exit-policy trailing_stop \
--trailing-activation-return 0.01 \
--trailing-drawdown 0.003 \
--stop-loss 0.02 \
--max-holding-period-minutes 60
```

This means:

- activate trailing only after `+1%`
- sell after a `0.3%` pullback from the best reached return
- hard stop at `-2%`
- never hold more than `60` minutes

Important:

- This does not modify the shard dataset.
- This does not modify `.gbdt_cache`.
- It is an execution/backtest overlay.
- Exact trailing stop requires ordered future candle path data.
- If ordered future path data is unavailable, the run fails instead of silently approximating.

Default behavior remains unchanged:

```bash
--exit-policy fixed_horizon
```

`--holding-period-minutes` is still supported and still controls the legacy fixed-horizon exit. If `--max-holding-period-minutes` is omitted, it defaults to the same value as `--holding-period-minutes`.

Future price path lookup order:

1. `--dynamic-exit-price-source existing_rows`
2. `--dynamic-exit-price-source raw_klines`
3. `--dynamic-exit-price-source auto`

With `auto`, the pipeline first checks whether the runtime rows already expose an ordered future minute-by-minute return path. If they do not, it falls back to raw Binance-style 1-minute kline files provided through:

```bash
--raw-kline-dir <path>
```

Supported raw kline inputs are execution-time only and may be plain `.csv`, `.csv.gz`, or `.zip` files containing Binance 1-minute candles.
