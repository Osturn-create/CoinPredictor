#!/usr/bin/env python3
import csv
import gzip
import io
import json
import os
import tempfile
import unittest
import warnings
from unittest import mock
from types import SimpleNamespace

import gbdt_pipeline as pipeline
import run_experiments


HEADER = [
    "symbol",
    "month",
    "month_index",
    "open_time",
    "label",
    "forward_return",
    "trade_return",
    "max_future_high_return",
    "max_future_low_return",
    "quote_volume",
    "log_quote_volume",
    "ret_1m",
]


def write_training_manifest(path, label_mode="target_stop", growth_threshold=0.05,
                            upside_target=0.05, downside_stop=0.02,
                            target_exit_mode="fixed_target",
                            include_targets=True):
    manifest_path = os.path.splitext(path)[0] + ".meta.json"
    target_lines = ""
    if include_targets:
        target_lines = (
            '  "upside_target": %.12g,\n'
            '  "downside_stop": %.12g\n'
        ) % (upside_target, downside_stop)
    else:
        target_lines = '  "market_breadth_features": false\n'
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(
            "{\n"
            '  "version": 1,\n'
            '  "training_csv": "kline_growth_training.csv",\n'
            '  "label_mode": "%s",\n'
            '  "target_exit_mode": "%s",\n'
            '  "growth_threshold": %.12g,\n'
            "%s"
            "}\n" % (label_mode, target_exit_mode, growth_threshold, target_lines)
        )
    return manifest_path


def shard_dataset_manifest(feature_names=None, upside_target=0.02, downside_stop=0.02,
                           market_breadth_min_symbols=5, shards=None):
    names = feature_names or ["log_quote_volume", "ret_1m"]
    return {
        "version": 1,
        "kind": "symbol_month_shards",
        "feature_count": len(names),
        "feature_names": names,
        "label_mode": "target_stop",
        "target_exit_mode": "first_decline",
        "prediction_window_minutes": 5,
        "growth_threshold": 0.05,
        "upside_target": upside_target,
        "downside_stop": downside_stop,
        "tie_policy": "stop_first",
        "fee": 0.001,
        "slippage": 0.0005,
        "min_net_return": 0.0,
        "split_mode": "ratio",
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "training_months": 6,
        "validation_months": 1,
        "test_months": 1,
        "market_regime_features": False,
        "market_breadth_features": False,
        "market_breadth_min_symbols": market_breadth_min_symbols,
        "shards": list(shards or []),
    }


def write_sharded_dataset(dataset_dir, shards, compression="none"):
    os.makedirs(os.path.join(dataset_dir, "shards"), exist_ok=True)
    manifest = shard_dataset_manifest(shards=[
        {
            "symbol": shard["symbol"],
            "month": shard["month"],
            "csv_path": "shards/{}/{}{}".format(
                shard["symbol"],
                shard["month"],
                ".csv.gz" if compression == "gzip" else ".csv",
            ),
            "compression": compression,
            "row_count": len(shard["rows"]),
        }
        for shard in shards
    ])
    with open(os.path.join(dataset_dir, "kline_growth_dataset.meta.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    for shard in shards:
        symbol = shard["symbol"]
        month = shard["month"]
        rows = shard["rows"]
        symbol_dir = os.path.join(dataset_dir, "shards", symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        csv_name = "{}{}".format(month, ".csv.gz" if compression == "gzip" else ".csv")
        csv_path = os.path.join(symbol_dir, csv_name)
        if compression == "gzip":
            with gzip.open(csv_path, "wt", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                for item in rows:
                    writer.writerow(item)
        else:
            with open(csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                for item in rows:
                    writer.writerow(item)
        shard_manifest = dict(manifest)
        shard_manifest.update({
            "version": 1,
            "kind": "symbol_month_shard",
            "symbol": symbol,
            "month": month,
            "csv_path": "shards/{}/{}".format(symbol, csv_name),
            "compression": compression,
            "row_count": len(rows),
        })
        with open(os.path.join(symbol_dir, "{}.meta.json".format(month)), "w", encoding="utf-8") as handle:
            json.dump(shard_manifest, handle, indent=2, sort_keys=True)


def row(open_time, label=0, quote_volume=1000000.0, trade_return=0.01):
    return pipeline.DataRow(
        "TESTUSDT",
        "2020-01",
        0,
        open_time,
        label,
        trade_return,
        trade_return,
        max(0.0, trade_return),
        min(0.0, trade_return),
        quote_volume,
        [0.0, 0.0],
    )


def symbol_row(symbol, open_time, label=0, quote_volume=1000000.0, trade_return=0.01):
    return pipeline.DataRow(
        symbol,
        "2020-01",
        0,
        open_time,
        label,
        trade_return,
        trade_return,
        max(0.0, trade_return),
        min(0.0, trade_return),
        quote_volume,
        [0.0, 0.0],
    )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp.name, "synthetic.csv")
        with open(self.csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            base = 1600000000000
            for month in range(8):
                for minute in range(4):
                    quote_volume = 100000.0 + month * 1000.0 + minute
                    writer.writerow([
                        "TESTUSDT",
                        "2020-{:02d}".format(month + 1),
                        month,
                        base + (month * 100 + minute) * 60000,
                        1 if minute == 0 else 0,
                        0.01 if minute == 0 else -0.005,
                        0.01 if minute == 0 else -0.005,
                        0.02,
                        -0.01,
                        quote_volume,
                        pipeline.math.log1p(quote_volume),
                        0.001 * minute,
                    ])

    def tearDown(self):
        self.temp.cleanup()

    def shard_row(self, symbol, month, month_index, open_time, label, trade_return, quote_volume):
        return [
            symbol,
            month,
            month_index,
            open_time,
            label,
            trade_return,
            trade_return,
            max(0.0, trade_return),
            min(0.0, trade_return),
            quote_volume,
            pipeline.math.log1p(quote_volume),
            0.001,
        ]

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_compact_cache_reuse_and_persistent_cleanup(self):
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, features, has_returns = pipeline.load_rows(
            self.csv_path, "memmap32", cache_dir=cache_dir
        )
        self.assertEqual(len(rows), 32)
        self.assertEqual(features, ["log_quote_volume", "ret_1m"])
        self.assertTrue(has_returns)
        feature_path = rows.table.memmap_path
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "rebuilt")
        rows.cleanup()
        self.assertTrue(os.path.exists(feature_path))

        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "hit")
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_temporary_memmap_cleanup(self):
        rows, _, _ = pipeline.load_rows(
            self.csv_path,
            "memmap32",
            memmap_dir=self.temp.name,
            disable_cache=True,
        )
        feature_path = rows.table.memmap_path
        self.assertTrue(os.path.exists(feature_path))
        rows.cleanup()
        self.assertFalse(os.path.exists(feature_path))

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_gzip_training_input_loads_without_decompressing_first(self):
        gz_path = self.csv_path + ".gz"
        with open(self.csv_path, "rb") as source:
            with gzip.open(gz_path, "wb") as target:
                target.write(source.read())
        rows, features, has_returns = pipeline.load_rows(
            gz_path,
            "memmap32",
            disable_cache=True,
            memmap_dir=self.temp.name,
        )
        self.assertEqual(len(rows), 32)
        self.assertEqual(features, ["log_quote_volume", "ret_1m"])
        self.assertTrue(has_returns)
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_gzip_sharded_dataset_loads_without_decompressing_first(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        write_sharded_dataset(dataset_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000060000, 0, -0.01, 100500.0),
                ],
            },
        ], compression="gzip")
        rows, features, has_returns = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=os.path.join(self.temp.name, "cache"))
        self.assertEqual(len(rows), 2)
        self.assertEqual(features, ["log_quote_volume", "ret_1m"])
        self.assertTrue(has_returns)
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_cache_rebuilds_when_csv_timestamp_changes(self):
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        rows.cleanup()
        stat = os.stat(self.csv_path)
        os.utime(self.csv_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "rebuilt")
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_sharded_dataset_cache_reuses_existing_shards_when_adding_new_one(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        existing_shards = [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000060000, 0, -0.01, 100500.0),
                ],
            },
            {
                "symbol": "BBBUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("BBBUSDT", "2020-01", 0, 1600000000000, 0, -0.005, 200000.0),
                ],
            },
        ]
        write_sharded_dataset(dataset_dir, existing_shards)
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, features, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=cache_dir)
        self.assertEqual(len(rows), 3)
        self.assertEqual(features, ["log_quote_volume", "ret_1m"])
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "rebuilt")
        shard_cache_dir = pipeline.CACHE_LOAD_INFO["paths"]["shard_cache_dir"]
        shard_manifests = sorted(
            os.path.join(root, name)
            for root, _, files in os.walk(shard_cache_dir)
            for name in files
            if name == "shard_cache_manifest.json"
        )
        self.assertEqual(len(shard_manifests), 2)
        first_manifest = shard_manifests[0]
        first_stat = os.stat(first_manifest)
        rows.cleanup()

        write_sharded_dataset(dataset_dir, existing_shards + [
            {
                "symbol": "CCCUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("CCCUSDT", "2020-01", 0, 1600000120000, 1, 0.03, 150000.0),
                ],
            },
        ])
        rows, _, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=cache_dir)
        self.assertEqual(len(rows), 4)
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "rebuilt")
        second_stat = os.stat(first_manifest)
        self.assertEqual(first_stat.st_mtime_ns, second_stat.st_mtime_ns)
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_sharded_dataset_manifest_inventory_ignores_stale_unlisted_shards(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        write_sharded_dataset(dataset_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        stale_symbol_dir = os.path.join(dataset_dir, "shards", "STALEUSDT")
        os.makedirs(stale_symbol_dir, exist_ok=True)
        with open(os.path.join(stale_symbol_dir, "2020-01.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow(self.shard_row("STALEUSDT", "2020-01", 0, 1600000060000, 0, -0.01, 50000.0))
        stale_manifest = shard_dataset_manifest()
        stale_manifest.update({
            "version": 1,
            "kind": "symbol_month_shard",
            "symbol": "STALEUSDT",
            "month": "2020-01",
            "row_count": 1,
        })
        with open(os.path.join(stale_symbol_dir, "2020-01.meta.json"), "w", encoding="utf-8") as handle:
            json.dump(stale_manifest, handle, indent=2, sort_keys=True)
        rows, _, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=os.path.join(self.temp.name, "cache"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.table.symbols, ["AAAUSDT"])
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_sharded_dataset_cache_only_hits_after_aggregate_build(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        write_sharded_dataset(dataset_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=cache_dir)
        rows.cleanup()
        rows, _, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=cache_dir, cache_only=True)
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "hit")
        self.assertTrue(pipeline.CACHE_LOAD_INFO.get("sharded_dataset"))
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_sharded_dataset_aggregate_remaps_month_indices_globally(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        write_sharded_dataset(dataset_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
            {
                "symbol": "AAAUSDT",
                "month": "2020-03",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-03", 1, 1600000060000, 1, 0.02, 100000.0),
                ],
            },
            {
                "symbol": "BBBUSDT",
                "month": "2020-02",
                "rows": [
                    self.shard_row("BBBUSDT", "2020-02", 0, 1600000120000, 1, 0.02, 100000.0),
                ],
            },
        ])
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=cache_dir)
        table = rows.table
        observed = {}
        for position in range(len(table.labels)):
            month_name = table.months[int(table.month_codes[position])]
            month_index = int(table.month_indices[position])
            observed.setdefault(month_name, set()).add(month_index)
        self.assertEqual(observed["2020-01"], {0})
        self.assertEqual(observed["2020-02"], {1})
        self.assertEqual(observed["2020-03"], {2})
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_market_breadth_sidecar_augmentation_uses_existing_sharded_cache(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        os.makedirs(os.path.join(dataset_dir, "shards"), exist_ok=True)
        feature_names = [
            "log_quote_volume",
            "ret_1m",
            "ret_5m",
            "ret_15m",
            "ret_60m",
            "rolling_quote_volume_zscore_60m",
        ]
        dataset_manifest = shard_dataset_manifest(feature_names=feature_names)
        with open(os.path.join(dataset_dir, "kline_growth_dataset.meta.json"), "w", encoding="utf-8") as handle:
            json.dump(dataset_manifest, handle, indent=2, sort_keys=True)
        symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]
        base_time = 1600000000000
        values = [
            (0.01, 0.02, 0.03, 0.10),
            (-0.02, -0.01, 0.01, -0.20),
            (0.03, 0.01, -0.02, 0.30),
            (0.00, 0.02, 0.02, -0.10),
            (0.04, 0.03, 0.05, 0.00),
        ]
        header = [
            "symbol", "month", "month_index", "open_time", "label", "forward_return", "trade_return",
            "max_future_high_return", "max_future_low_return", "quote_volume",
            "log_quote_volume", "ret_1m", "ret_5m", "ret_15m", "ret_60m", "rolling_quote_volume_zscore_60m",
        ]
        for symbol, (ret5, ret15, ret60, quote_z) in zip(symbols, values):
            symbol_dir = os.path.join(dataset_dir, "shards", symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            csv_path = os.path.join(symbol_dir, "2020-01.csv")
            with open(csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                for offset in range(2):
                    quote_volume = 100000.0 + offset * 1000.0
                    writer.writerow([
                        symbol, "2020-01", 0, base_time + offset * 60000, 1 if ret5 > 0 else 0,
                        ret5, ret5, max(ret5, 0.0), min(ret5, 0.0), quote_volume,
                        pipeline.math.log1p(quote_volume), 0.001 * offset, ret5, ret15, ret60, quote_z,
                    ])
            shard_manifest = dict(dataset_manifest)
            shard_manifest.update({
                "version": 1,
                "kind": "symbol_month_shard",
                "symbol": symbol,
                "month": "2020-01",
                "row_count": 2,
            })
            with open(os.path.join(symbol_dir, "2020-01.meta.json"), "w", encoding="utf-8") as handle:
                json.dump(shard_manifest, handle, indent=2, sort_keys=True)
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, feature_columns, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=cache_dir)
        args = SimpleNamespace(
            augment_market_breadth_features=True,
            market_breadth_min_symbols=5,
            cache_only=False,
            market_breadth_features=False,
        )
        rows, feature_columns = pipeline.maybe_augment_market_breadth_rows(
            rows,
            feature_columns,
            args,
            dataset_dir,
            cache_dir,
        )
        self.assertTrue(all(name in feature_columns for name in pipeline.MARKET_BREADTH_FEATURE_COLUMNS))
        breadth_up_5m = pipeline.row_feature_array(rows, "market_breadth_up_5m")
        self.assertTrue(any(abs(float(value) - 0.6) < 1e-6 for value in breadth_up_5m))
        missing = pipeline.row_feature_array(rows, "market_breadth_missing")
        self.assertTrue(all(abs(float(value)) < 1e-6 for value in missing))
        sidecar_manifests = [
            name for name in os.listdir(cache_dir)
            if name.startswith("breadth-") and name.endswith(".manifest.json")
        ]
        self.assertTrue(sidecar_manifests)
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_cache_only_fails_when_cache_is_missing(self):
        with self.assertRaisesRegex(ValueError, "--cache-only was set"):
            pipeline.main([
                "--input", self.csv_path,
                "--feature-storage", "memmap32",
                "--cache-dir", os.path.join(self.temp.name, "missing-cache"),
                "--cache-only",
                "--smoke-test-cache",
            ])

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_cache_only_reuses_existing_cache_without_csv_parse(self):
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        rows.cleanup()
        with mock.patch.object(pipeline, "load_compact_rows", side_effect=AssertionError("CSV parse path should not run")):
            rows, _, _ = pipeline.load_rows(
                self.csv_path,
                "memmap32",
                cache_dir=cache_dir,
                cache_only=True,
            )
            self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "hit")
            rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_smoke_test_cache_exits_before_training(self):
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        rows.cleanup()
        with mock.patch.object(pipeline, "run_fixed_split", side_effect=AssertionError("training should not run")):
            status = pipeline.main([
                "--input", self.csv_path,
                "--feature-storage", "memmap32",
                "--cache-dir", cache_dir,
                "--cache-only",
                "--smoke-test-cache",
            ])
        self.assertEqual(status, 0)

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_cache_report_lists_split_metadata_arrays_for_sharded_dataset(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        write_sharded_dataset(dataset_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=cache_dir)
        rows.cleanup()
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--input", dataset_dir,
            "--feature-storage", "memmap32",
            "--cache-dir", cache_dir,
        ])
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            status = pipeline.cache_report(
                args,
                *pipeline.load_training_manifest(dataset_dir)
            )
        self.assertEqual(status, 0)
        text = output.getvalue()
        self.assertIn("dataset_type=sharded", text)
        self.assertIn("shard_cache_hits=1", text)
        self.assertIn("metadata_arrays.symbol_codes", text)

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_cache_cleanup_preserves_active_monolithic_cache_files(self):
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        paths = dict(pipeline.CACHE_LOAD_INFO["paths"])
        rows.cleanup()
        status = pipeline.cache_cleanup(SimpleNamespace(
            input=self.csv_path,
            cache_dir=cache_dir,
            dry_run=False,
            confirm_delete=True,
        ))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(paths["features"]))
        self.assertTrue(all(os.path.exists(path) for path in paths["metadata_arrays"].values()))
        self.assertTrue(os.path.exists(paths["manifest"]))

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_cache_cleanup_preserves_active_sharded_cache_files(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        write_sharded_dataset(dataset_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=cache_dir)
        paths = dict(pipeline.CACHE_LOAD_INFO["paths"])
        shard_cache_dir = paths["shard_cache_dir"]
        shard_cache_files = sorted(
            os.path.join(shard_cache_dir, name)
            for name in os.listdir(shard_cache_dir)
        )
        rows.cleanup()
        status = pipeline.cache_cleanup(SimpleNamespace(
            input=dataset_dir,
            cache_dir=cache_dir,
            dry_run=False,
            confirm_delete=True,
        ))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(paths["features"]))
        self.assertTrue(all(os.path.exists(path) for path in paths["metadata_arrays"].values()))
        self.assertTrue(os.path.exists(paths["manifest"]))
        self.assertTrue(os.path.isdir(shard_cache_dir))
        self.assertTrue(all(os.path.exists(path) for path in shard_cache_files))

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_monolithic_cache_rebuild_clears_stale_sharded_state(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        write_sharded_dataset(dataset_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        sharded_cache_dir = os.path.join(self.temp.name, "sharded-cache")
        rows, _, _ = pipeline.load_rows(dataset_dir, "memmap32", cache_dir=sharded_cache_dir)
        rows.cleanup()
        self.assertTrue(pipeline.CACHE_LOAD_INFO.get("sharded_dataset"))
        monolithic_cache_dir = os.path.join(self.temp.name, "monolithic-cache")
        rows, _, _ = pipeline.load_rows(
            self.csv_path,
            "memmap32",
            cache_dir=monolithic_cache_dir,
            rebuild_cache=True,
        )
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "rebuilt")
        self.assertNotIn("sharded_dataset", pipeline.CACHE_LOAD_INFO)
        rows.cleanup()

    def test_missing_liquidity_column_fails_clearly(self):
        path = os.path.join(self.temp.name, "missing-liquidity.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["symbol", "month", "month_index", "open_time", "label", "ret_1m"])
            writer.writerow(["TESTUSDT", "2020-01", 0, 1600000000000, 0, 0.0])
        with self.assertRaisesRegex(ValueError, "quote_volume or log_quote_volume"):
            pipeline.load_rows(path, "float32", disable_cache=True)

    def test_canonical_training_csv_requires_manifest(self):
        path = os.path.join(self.temp.name, "kline_growth_training.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow([
                "TESTUSDT", "2020-01", 0, 1600000000000, 0,
                -0.005, -0.005, 0.01, -0.01, 100000.0, pipeline.math.log1p(100000.0), 0.0,
            ])
        with self.assertRaisesRegex(ValueError, "missing kline_growth_training.meta.json"):
            pipeline.load_rows(path, "float32", disable_cache=True)

    def test_default_input_recovery_prefers_single_cached_sharded_dataset(self):
        requested_path = os.path.join(self.temp.name, "kline_growth_training.csv")
        dataset_dir = os.path.join(self.temp.name, "shard_dataset_25")
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_manifest_path = os.path.join(dataset_dir, pipeline.SHARDED_DATASET_MANIFEST)
        with open(dataset_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(shard_dataset_manifest(), handle)
        cache_dir = os.path.join(self.temp.name, ".gbdt_cache_shard25")
        os.makedirs(cache_dir, exist_ok=True)
        aggregate_manifest_path = os.path.join(cache_dir, "synthetic.aggregate.manifest.json")
        with open(aggregate_manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "dataset_path": dataset_dir,
                "dataset_manifest_path": dataset_manifest_path,
            }, handle)
        with mock.patch.object(pipeline.os, "getcwd", return_value=self.temp.name):
            recovered = pipeline.discover_recoverable_default_dataset(requested_path)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["input_path"], os.path.abspath(dataset_dir))
        self.assertEqual(recovered["cache_dir"], os.path.abspath(cache_dir))
        self.assertEqual(recovered["source"], "cache_manifest")

    def test_default_input_recovery_returns_none_for_ambiguous_local_shard_datasets(self):
        requested_path = os.path.join(self.temp.name, "kline_growth_training.csv")
        for name in ("shard_dataset_a", "shard_dataset_b"):
            dataset_dir = os.path.join(self.temp.name, name)
            os.makedirs(dataset_dir, exist_ok=True)
            with open(os.path.join(dataset_dir, pipeline.SHARDED_DATASET_MANIFEST), "w", encoding="utf-8") as handle:
                json.dump(shard_dataset_manifest(), handle)
        with mock.patch.object(pipeline.os, "getcwd", return_value=self.temp.name):
            recovered = pipeline.discover_recoverable_default_dataset(requested_path)
        self.assertIsNone(recovered)

    def test_manifest_compatibility_signature_includes_market_breadth_min_symbols(self):
        left = shard_dataset_manifest(market_breadth_min_symbols=5)
        right = shard_dataset_manifest(market_breadth_min_symbols=10)
        self.assertNotEqual(
            pipeline.manifest_compatibility_signature(left),
            pipeline.manifest_compatibility_signature(right),
        )

    def test_default_input_recovery_skips_invalid_local_shard_manifest(self):
        requested_path = os.path.join(self.temp.name, "kline_growth_training.csv")
        invalid_dir = os.path.join(self.temp.name, "invalid_dataset")
        os.makedirs(invalid_dir, exist_ok=True)
        with open(os.path.join(invalid_dir, pipeline.SHARDED_DATASET_MANIFEST), "w", encoding="utf-8") as handle:
            json.dump({"version": 999, "kind": "old_layout"}, handle)
        valid_dir = os.path.join(self.temp.name, "valid_dataset")
        os.makedirs(valid_dir, exist_ok=True)
        with open(os.path.join(valid_dir, pipeline.SHARDED_DATASET_MANIFEST), "w", encoding="utf-8") as handle:
            json.dump(shard_dataset_manifest(), handle)
        with mock.patch.object(pipeline.os, "getcwd", return_value=self.temp.name):
            recovered = pipeline.discover_recoverable_default_dataset(requested_path)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["input_path"], os.path.abspath(valid_dir))

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact rows")
    def test_monolithic_csv_reindexes_month_indices_globally(self):
        path = os.path.join(self.temp.name, "mixed.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow([
                "AAAUSDT", "2020-01", 0, 1600000000000, 1,
                0.01, 0.01, 0.02, -0.01, 100000.0, pipeline.math.log1p(100000.0), 0.0,
            ])
            writer.writerow([
                "BBBUSDT", "2020-03", 0, 1605000000000, 0,
                -0.01, -0.01, 0.01, -0.02, 120000.0, pipeline.math.log1p(120000.0), 0.0,
            ])
        rows, _, _ = pipeline.load_rows(path, "memmap32", disable_cache=True, memmap_dir=self.temp.name)
        observed = {}
        for position in range(len(rows.table.labels)):
            month_name = rows.table.months[int(rows.table.month_codes[position])]
            observed[month_name] = int(rows.table.month_indices[position])
        self.assertEqual(observed["2020-01"], 0)
        self.assertEqual(observed["2020-03"], 1)
        rows.cleanup()

    def test_sharded_object_rows_reindex_months_globally_without_cache(self):
        dataset_dir = os.path.join(self.temp.name, "dataset")
        write_sharded_dataset(dataset_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
            {
                "symbol": "BBBUSDT",
                "month": "2020-03",
                "rows": [
                    self.shard_row("BBBUSDT", "2020-03", 0, 1605000000000, 0, -0.01, 120000.0),
                ],
            },
        ])
        rows, _, _ = pipeline.load_rows(dataset_dir, "float32", disable_cache=True)
        observed = {
            row.month: row.month_index
            for row in rows
        }
        self.assertEqual(observed["2020-01"], 0)
        self.assertEqual(observed["2020-03"], 1)

    def test_target_stop_manifest_rejects_uncapped_trade_returns(self):
        path = os.path.join(self.temp.name, "kline_growth_training.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow([
                "TESTUSDT", "2020-01", 0, 1600000000000, 0,
                -0.50, -0.50, 0.01, -0.55, 100000.0, pipeline.math.log1p(100000.0), 0.0,
            ])
        write_training_manifest(path, label_mode="target_stop", upside_target=0.05, downside_stop=0.02)
        with self.assertRaisesRegex(ValueError, "incompatible with kline_growth_training.meta.json"):
            pipeline.load_rows(path, "float32", disable_cache=True)

    def test_first_decline_manifest_allows_positive_returns_above_target(self):
        path = os.path.join(self.temp.name, "kline_growth_training.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow([
                "TESTUSDT", "2020-01", 0, 1600000000000, 1,
                0.035, 0.035, 0.04, -0.01, 100000.0, pipeline.math.log1p(100000.0), 0.0,
            ])
        write_training_manifest(
            path,
            label_mode="target_stop",
            upside_target=0.02,
            downside_stop=0.02,
            target_exit_mode="first_decline",
        )
        rows, _, _ = pipeline.load_rows(path, "float32", disable_cache=True)
        self.assertEqual(len(rows), 1)
        if hasattr(rows, "cleanup"):
            rows.cleanup()

    def test_manifest_ev_targets_are_used_without_cli_override(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([])
        manifest = {
            "upside_target": 0.02,
            "downside_stop": 0.015,
            "market_regime_features": True,
            "market_breadth_features": False,
        }
        pipeline.apply_manifest_ev_targets(args, manifest, set())
        self.assertAlmostEqual(args.upside_target, 0.02)
        self.assertAlmostEqual(args.downside_stop, 0.015)
        self.assertEqual(args.ev_upside_target_source, "manifest")
        self.assertEqual(args.ev_downside_stop_source, "manifest")
        self.assertTrue(args.market_regime_features)
        self.assertFalse(args.market_breadth_features)

    def test_cli_ev_target_overrides_manifest(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["--upside-target", "0.03", "--downside-stop", "0.01"])
        manifest = {"upside_target": 0.02, "downside_stop": 0.02}
        pipeline.apply_manifest_ev_targets(args, manifest, {"upside_target", "downside_stop"})
        self.assertAlmostEqual(args.upside_target, 0.03)
        self.assertAlmostEqual(args.downside_stop, 0.01)
        self.assertEqual(args.ev_upside_target_source, "cli")
        self.assertEqual(args.ev_downside_stop_source, "cli")

    def test_walk_forward_final_model_defaults_to_selected(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.walk_forward_final_model, "selected")
        self.assertEqual(args.walk_validation_months, 1)

    def test_walk_forward_split_bounds_use_full_train_window(self):
        train_start, train_end, validation_start, validation_end, test_month = (
            pipeline.walk_forward_split_bounds(0, 6, 1)
        )
        self.assertEqual((train_start, train_end), (0, 6))
        self.assertEqual((validation_start, validation_end), (6, 7))
        self.assertEqual(test_month, 7)

    def test_inactive_fold_blocker_check_identifies_probability_threshold(self):
        rows = [
            symbol_row("ALPHAUSDT", 1600000000000, 0, trade_return=0.01),
            symbol_row("BETAUSDT", 1600000060000, 0, trade_return=0.01),
        ]
        predictions = pipeline.build_prediction_bundle(
            probability=[0.32, 0.41],
            calibrated_probability=[0.32, 0.41],
            predicted_trade_return=[0.01, 0.01],
        )
        parser = pipeline.build_parser()
        args = parser.parse_args([])
        args.objective_mode = "classification"
        args.threshold_objective = "avg_profit"
        args.ev_safety_margin = 0.0
        args.validation_slippage_multiplier = 1.0
        metrics = {"predicted_trades": 0, "raw_signal_trades": 0}
        validation_metrics = {
            "selected_validation_trade_count": 8,
            "selected_validation_portfolio_profit": 125.0,
            "selected_score_threshold": 0.5,
        }
        info = pipeline.inactive_fold_blocker_check(
            rows,
            predictions,
            0.5,
            metrics,
            validation_metrics,
            args,
            "probability",
        )
        self.assertEqual(info["inactive_blocker_source"], "test_threshold")
        self.assertEqual(info["inactive_blocker_metric"], "probability")
        self.assertAlmostEqual(info["inactive_blocker_threshold"], 0.5)
        self.assertAlmostEqual(info["inactive_blocker_best_score"], 0.41)
        self.assertEqual(info["inactive_closest_symbol"], "BETAUSDT")
        self.assertEqual(info["inactive_promising_fold"], 1)

    def test_invalid_manifest_ev_target_falls_back(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([])
        manifest = {"upside_target": "not-a-number", "downside_stop": -0.5}
        pipeline.apply_manifest_ev_targets(args, manifest, set())
        self.assertAlmostEqual(args.upside_target, 0.05)
        self.assertAlmostEqual(args.downside_stop, 0.02)
        self.assertEqual(args.ev_upside_target_source, "invalid_manifest_fallback")
        self.assertEqual(args.ev_downside_stop_source, "invalid_manifest_fallback")

    def test_symbol_validation_filter_drops_negative_symbol_and_improves_profit(self):
        validation_rows = [
            symbol_row("GOODUSDT", 1600000000000, 1, trade_return=0.03),
            symbol_row("GOODUSDT", 1600000060000, 1, trade_return=0.02),
            symbol_row("BADUSDT", 1600000120000, 0, trade_return=-0.03),
            symbol_row("BADUSDT", 1600000180000, 0, trade_return=-0.02),
        ]
        test_rows = [
            symbol_row("GOODUSDT", 1600000240000, 1, trade_return=0.02),
            symbol_row("GOODUSDT", 1600000300000, 1, trade_return=0.01),
            symbol_row("BADUSDT", 1600000360000, 0, trade_return=-0.02),
            symbol_row("BADUSDT", 1600000420000, 0, trade_return=-0.01),
        ]
        predictions = pipeline.build_prediction_bundle(
            probability=[0.99, 0.99, 0.99, 0.99],
            calibrated_probability=[0.99, 0.99, 0.99, 0.99],
            predicted_trade_return=[0.02, 0.02, 0.02, 0.02],
        )
        args = SimpleNamespace(
            symbol_validation_filter="positive_avg_profit",
            min_symbol_validation_trades=2,
            min_symbol_validation_average_profit=0.0,
            min_symbol_validation_total_profit=0.0,
            fee=0.0,
            slippage=0.0,
            validation_slippage_multiplier=1.0,
            initial_capital=10000.0,
            max_position_fraction=0.25,
            max_volume_fraction=1.0,
            max_trades_per_period=0,
            trade_period_minutes=60,
            holding_period_minutes=5,
            threshold_objective="avg_profit",
            trade_selection="threshold",
            top_k_per_minute=1,
            upside_target=0.02,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="classification",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.0,
            max_trades_per_day=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            hybrid_score_mode="basic",
            hybrid_uncertainty_penalty=0.0,
            dynamic_hybrid_thresholds="none",
            meta_filter="none",
            meta_filter_min_probability=0.0,
        )
        symbol_filter = pipeline.fit_symbol_validation_filter(
            validation_rows,
            predictions,
            0.5,
            args,
            "probability",
        )
        self.assertTrue(symbol_filter["enabled"])
        self.assertEqual(symbol_filter["allowed_symbols"], ["GOODUSDT"])
        unfiltered = pipeline.evaluate(
            test_rows,
            predictions,
            0.5,
            0.0,
            0.0,
            compute_auc=False,
            initial_capital=10000.0,
            max_position_fraction=0.25,
            max_volume_fraction=1.0,
            max_trades_per_period=0,
            trade_period_minutes=60,
            holding_period_minutes=5,
            threshold_objective="avg_profit",
            trade_selection="threshold",
            top_k_per_minute=1,
            objective_mode="classification",
            trade_score_name="probability",
        )
        filtered = pipeline.evaluate(
            test_rows,
            predictions,
            0.5,
            0.0,
            0.0,
            compute_auc=False,
            initial_capital=10000.0,
            max_position_fraction=0.25,
            max_volume_fraction=1.0,
            max_trades_per_period=0,
            trade_period_minutes=60,
            holding_period_minutes=5,
            threshold_objective="avg_profit",
            trade_selection="threshold",
            top_k_per_minute=1,
            objective_mode="classification",
            trade_score_name="probability",
            symbol_filter_info=symbol_filter,
        )
        self.assertEqual(unfiltered["predicted_trades"], 4)
        self.assertEqual(filtered["predicted_trades"], 2)
        self.assertGreater(filtered["portfolio_profit"], unfiltered["portfolio_profit"])

    def test_old_manifest_without_targets_still_loads(self):
        path = os.path.join(self.temp.name, "kline_growth_training.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow([
                "TESTUSDT", "2020-01", 0, 1600000000000, 0,
                -0.005, -0.005, 0.01, -0.01, 100000.0, pipeline.math.log1p(100000.0), 0.0,
            ])
        write_training_manifest(path, label_mode="target_stop", include_targets=False)
        rows, _, _ = pipeline.load_rows(path, "float32", disable_cache=True)
        self.assertEqual(len(rows), 1)
        if hasattr(rows, "cleanup"):
            rows.cleanup()

    def test_ratio_and_walk_forward_views(self):
        rows, _, _ = pipeline.load_rows(self.csv_path, "float32", disable_cache=True)
        args = SimpleNamespace(train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15)
        train, validation, test = pipeline.select_ratio_split(rows, args)
        self.assertEqual((len(train), len(validation), len(test)), (24, 4, 4))
        self.assertEqual(len(pipeline.select_month_range(rows, 0, 6)), 24)
        self.assertEqual(len(pipeline.select_month_range(rows, 6, 7)), 4)
        if hasattr(rows, "cleanup"):
            rows.cleanup()

    def test_auc_sampling_uses_sampled_row_count(self):
        rows, _, _ = pipeline.load_rows(self.csv_path, "float32", disable_cache=True)
        previous_limit = pipeline.AUC_SAMPLE_ROWS
        try:
            pipeline.AUC_SAMPLE_ROWS = 3
            if pipeline.np is not None:
                probabilities = pipeline.np.linspace(0.0, 1.0, len(rows), dtype=pipeline.np.float32)
            else:
                probabilities = [index / float(max(1, len(rows) - 1)) for index in range(len(rows))]
            score = pipeline.auc_score_from_rows(probabilities, rows)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        finally:
            pipeline.AUC_SAMPLE_ROWS = previous_limit
            if hasattr(rows, "cleanup"):
                rows.cleanup()

    def test_threshold_constraints(self):
        base = 1600000000000
        rows = [row(base + index * 60000, label=1 if index < 2 else 0) for index in range(4)]
        selection = pipeline.tune_threshold(
            rows,
            [0.9, 0.8, 0.7, 0.6],
            [0.5, 0.75],
            "precision",
            0.0,
            0.0,
            2,
            2,
            0.0,
            "explore",
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
        )
        threshold = selection["threshold"]
        metrics = selection["validation_metrics"]
        self.assertEqual(threshold, 0.75)
        self.assertEqual(metrics["predicted_trades"], 2)

    def test_threshold_selection_matches_simple_reference(self):
        base = 1600000000000
        rows = [row(base + index * 60000, label=1 if index < 2 else 0, trade_return=0.01 if index < 2 else -0.01) for index in range(4)]
        predictions = [0.9, 0.8, 0.7, 0.6]
        thresholds = [0.5, 0.75, 0.75]
        selection = pipeline.tune_threshold(
            rows,
            predictions,
            thresholds,
            "precision",
            0.0,
            0.0,
            2,
            0,
            0.0,
            "explore",
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
        )
        threshold = selection["threshold"]
        metrics = selection["validation_metrics"]
        reference_threshold = None
        reference_rank = (-float("inf"),) * 5
        for candidate in sorted(set(float(value) for value in thresholds)):
            candidate_metrics = pipeline.evaluate(
                rows, predictions, candidate, 0.0, 0.0,
                compute_auc=False,
                initial_capital=10000.0,
                max_position_fraction=0.10,
                max_volume_fraction=0.01,
                max_trades_per_period=10,
                trade_period_minutes=60,
                holding_period_minutes=5,
            )
            if candidate_metrics["predicted_trades"] < 2:
                continue
            rank = pipeline.threshold_rank(candidate_metrics, "precision", -float("inf"))
            if rank > reference_rank:
                reference_rank = rank
                reference_threshold = candidate
        self.assertEqual(threshold, reference_threshold)
        self.assertEqual(metrics["predicted_trades"], 2)

    def test_hybrid_selected_diagnostics_are_consistent(self):
        base = 1600000000000
        rows = [
            row(base, label=1, trade_return=0.03, quote_volume=1000000000.0),
            row(base + 60000, label=0, trade_return=-0.01, quote_volume=1000000000.0),
            row(base + 120000, label=1, trade_return=0.025, quote_volume=1000000000.0),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.8, 0.3, 0.75],
            calibrated_probability=[0.8, 0.3, 0.75],
            predicted_trade_return=[0.03, -0.01, 0.025],
        )
        selection = pipeline.tune_threshold(
            rows,
            bundle,
            [0.001],
            "avg_profit",
            0.0,
            0.0,
            1,
            0,
            0.0,
            "explore",
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            objective_mode="hybrid",
            trade_score_name="hybrid",
            hybrid_min_score=0.001,
        )
        metrics = selection["validation_metrics"]
        self.assertEqual(selection["selected_score_name"], "hybrid")
        self.assertGreater(metrics["predicted_trades"], 0)
        self.assertEqual(metrics["selected_validation_trade_count"], metrics["predicted_trades"])
        self.assertEqual(metrics["selected_validation_portfolio_profit"], metrics["portfolio_profit"])
        self.assertTrue(pipeline.math.isfinite(metrics["selected_objective_score"]))

    def test_hybrid_no_trade_fallback_is_explicit(self):
        base = 1600000000000
        rows = [
            row(base, label=0, trade_return=-0.01, quote_volume=1000000000.0),
            row(base + 60000, label=0, trade_return=-0.02, quote_volume=1000000000.0),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.2, 0.2],
            calibrated_probability=[0.2, 0.2],
            predicted_trade_return=[0.0, 0.0],
        )
        selection = pipeline.tune_threshold(
            rows,
            bundle,
            [0.001],
            "avg_profit",
            0.0,
            0.0,
            1,
            0,
            0.0,
            "explore",
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            objective_mode="hybrid",
            trade_score_name="hybrid",
            hybrid_min_score=0.001,
        )
        metrics = selection["validation_metrics"]
        self.assertEqual(selection["threshold"], 1.01)
        self.assertEqual(metrics["selected_validation_trade_count"], 0)
        self.assertEqual(metrics["predicted_trades"], 0)
        self.assertFalse(pipeline.math.isfinite(metrics["selected_objective_score"]))

    def test_penalties_can_change_selected_candidate(self):
        low_trade_metrics = {
            "predicted_trades": 20,
            "portfolio_profit": 20.0,
            "max_capital_drawdown": 0.05,
            "precision": 0.4,
            "raw_signal_trades": 20,
            "portfolio_return": 0.002,
            "recall": 0.2,
            "average_profit_after_fee_and_slippage": 1.0,
            "total_profit_after_fee_and_slippage": 20.0,
        }
        high_trade_metrics = {
            "predicted_trades": 120,
            "portfolio_profit": 180.0,
            "max_capital_drawdown": 0.20,
            "precision": 0.35,
            "raw_signal_trades": 120,
            "portfolio_return": 0.018,
            "recall": 0.25,
            "average_profit_after_fee_and_slippage": 1.5,
            "total_profit_after_fee_and_slippage": 180.0,
        }
        low_result = pipeline.build_selected_threshold_result(
            0.001, dict(low_trade_metrics), "avg_profit", -float("inf"), 0.5, 0.05, 50, "hybrid"
        )
        high_result = pipeline.build_selected_threshold_result(
            0.002, dict(high_trade_metrics), "avg_profit", -float("inf"), 0.5, 0.05, 50, "hybrid"
        )
        self.assertGreater(
            pipeline.threshold_rank(high_result["validation_metrics"], "avg_profit", -float("inf"), high_result["base_objective_score"]),
            pipeline.threshold_rank(low_result["validation_metrics"], "avg_profit", -float("inf"), low_result["base_objective_score"]),
        )
        self.assertGreater(
            pipeline.threshold_rank(low_result["validation_metrics"], "avg_profit", -float("inf"), low_result["penalized_objective_score"]),
            pipeline.threshold_rank(high_result["validation_metrics"], "avg_profit", -float("inf"), high_result["penalized_objective_score"]),
        )

    def test_normalize_open_time_ms(self):
        self.assertEqual(pipeline.normalize_open_time_ms(1600000000000), 1600000000000)
        self.assertEqual(pipeline.normalize_open_time_ms(1600000000000000), 1600000000000)

    def test_portfolio_cash_locking_and_volume_cap(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0),
            row(base + 60000, quote_volume=1000000000.0),
        ]
        execution = pipeline.portfolio_execution(
            rows, [1.0, 1.0], 0.5, 0.0, 0.0, 10000.0, 0.60, 0.01, 10, 60, 120
        )
        self.assertEqual(list(execution["executed"].values()), [6000.0, 4000.0])
        volume = pipeline.portfolio_execution(
            [row(base, quote_volume=25000.0)],
            [1.0],
            0.5,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
        )
        self.assertEqual(list(volume["executed"].values()), [250.0])

    def test_zero_trade_cap_disables_period_limit(self):
        base = 1600000000000
        rows = [
            row(base + index * 60000, quote_volume=1000000000.0)
            for index in range(3)
        ]
        execution = pipeline.portfolio_execution(
            rows, [1.0, 1.0, 1.0], 0.5, 0.0, 0.0, 10000.0, 0.50, 0.01, 0, 60, 120
        )
        self.assertEqual(list(execution["executed"].values()), [5000.0, 5000.0])

    def test_equity_fraction_cap_limits_simultaneous_trades_to_ten(self):
        base = 1600000000000
        rows = [
            row(base + index * 60000, quote_volume=1000000000.0)
            for index in range(11)
        ]
        execution = pipeline.portfolio_execution(
            rows, [1.0] * len(rows), 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 600
        )
        self.assertEqual(list(execution["executed"].values()), [1000.0] * 10)

    def test_prediction_output_modes(self):
        base = 1600000000000
        rows = [row(base + index * 60000) for index in range(3)]
        common = dict(
            rows=rows,
            predictions=[0.9, 0.1, 0.1],
            threshold=0.5,
            model_name="test",
            fee=0.0,
            slippage=0.0,
        )
        trades_path = os.path.join(self.temp.name, "trades.csv")
        pipeline.write_predictions(trades_path, output_mode="trades", **common)
        with open(trades_path, newline="") as handle:
            self.assertEqual(len(list(csv.reader(handle))), 2)

        none_path = os.path.join(self.temp.name, "none.csv")
        pipeline.write_predictions(none_path, output_mode="none", **common)
        with open(none_path, newline="") as handle:
            self.assertEqual(len(list(csv.reader(handle))), 1)

        all_path = os.path.join(self.temp.name, "all.csv")
        pipeline.write_predictions(all_path, output_mode="all", **common)
        with open(all_path, newline="") as handle:
            self.assertEqual(len(list(csv.reader(handle))), 4)

    def test_expected_value_calculation(self):
        value = pipeline.expected_value_from_probability(0.75, 0.05, 0.02, 0.001, 0.0005)
        self.assertAlmostEqual(value, 0.75 * 0.05 - 0.25 * 0.02 - 0.0015)

    def test_platt_calibration_improves_brier(self):
        if pipeline.np is not None:
            probabilities = pipeline.np.asarray([0.2, 0.3, 0.7, 0.8], dtype=pipeline.np.float32)
            labels_values = pipeline.np.asarray([0, 0, 1, 1], dtype=pipeline.np.float32)
        else:
            probabilities = [0.2, 0.3, 0.7, 0.8]
            labels_values = [0, 0, 1, 1]
        calibration = pipeline.fit_platt_calibration(probabilities, labels_values, 100)
        calibrated = pipeline.apply_platt_calibration(probabilities.copy() if hasattr(probabilities, "copy") else list(probabilities), calibration["a"], calibration["b"])
        self.assertEqual(calibration["mode"], "platt")
        self.assertGreater(calibration["rows"], 0)
        self.assertLessEqual(pipeline.brier_score(calibrated, labels_values), calibration["validation_brier_before"] + 1e-6)

    def test_topk_ev_selection_one_minute(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base, quote_volume=1000000000.0, trade_return=0.01),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.95, 0.90, 0.85],
            0.5,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            threshold_objective="ev",
            trade_selection="topk_ev",
            top_k_per_minute=2,
            upside_target=0.05,
            downside_stop=0.02,
            ev_safety_margin=0.0,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 1])
        self.assertEqual(execution["executed_selection_ranks"][0], 1)
        self.assertEqual(execution["executed_selection_ranks"][1], 2)

    def test_topk_ev_selection_streams_multiple_minutes(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base + 60000, quote_volume=1000000000.0, trade_return=0.01),
            row(base + 60000, quote_volume=1000000000.0, trade_return=0.01),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.95, 0.80, 0.92, 0.70],
            0.5,
            0.0,
            0.0,
            10000.0,
            0.20,
            0.01,
            10,
            60,
            5,
            threshold_objective="ev",
            trade_selection="topk_ev",
            top_k_per_minute=1,
            upside_target=0.05,
            downside_stop=0.02,
            ev_safety_margin=0.0,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 2])
        self.assertEqual(execution["executed_selected_by_topk"][0], 1)
        self.assertEqual(execution["executed_selected_by_topk"][2], 1)

    def test_topk_fast_path_matches_reference_for_k1(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base + 60000, quote_volume=1000000000.0, trade_return=0.01),
            row(base + 60000, quote_volume=1000000000.0, trade_return=0.01),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.95, 0.80, 0.92, 0.70],
            0.5,
            0.0,
            0.0,
            10000.0,
            0.20,
            0.01,
            10,
            60,
            5,
            threshold_objective="ev",
            trade_selection="topk_ev",
            top_k_per_minute=1,
            upside_target=0.05,
            downside_stop=0.02,
            ev_safety_margin=0.0,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 2])
        self.assertEqual(execution["executed_selection_ranks"][0], 1)
        self.assertEqual(execution["executed_selection_ranks"][2], 1)

    def test_topk_grouping_normalizes_mixed_timestamp_units(self):
        base_ms = 1600000000000
        base_us = base_ms * 1000
        rows = [
            row(base_ms, quote_volume=1000000000.0, trade_return=0.01),
            row(base_us, quote_volume=1000000000.0, trade_return=0.01),
            row(base_ms + 60000, quote_volume=1000000000.0, trade_return=0.01),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.95, 0.90, 0.85],
            0.5,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="topk_ev",
            top_k_per_minute=1,
            threshold_objective="ev",
            upside_target=0.05,
            downside_stop=0.02,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 2])

    def test_trade_score_predicted_return(self):
        bundle = pipeline.build_prediction_bundle(predicted_trade_return=[0.01])
        score = pipeline.trade_score_value(bundle, 0, "predicted_return", 0.05, 0.02, 0.001, 0.0005)
        self.assertAlmostEqual(score, 0.0085)

    def test_hybrid_score_calculation(self):
        bundle = pipeline.build_prediction_bundle(
            probability=[0.5],
            calibrated_probability=[0.5],
            predicted_trade_return=[0.04],
        )
        score = pipeline.hybrid_score_value(bundle, 0, 0.001, 0.0005)
        self.assertAlmostEqual(score, 0.0185)

    def test_hybrid_expected_return_does_not_multiply_by_probability(self):
        bundle = pipeline.build_prediction_bundle(
            probability=[0.1],
            calibrated_probability=[0.1],
            predicted_trade_return=[0.04],
        )
        args = SimpleNamespace(
            hybrid_return_combination="expected_return",
            hybrid_min_probability=0.0,
        )
        score = pipeline.hybrid_score_value(bundle, 0, 0.001, 0.0005, args=args)
        self.assertAlmostEqual(score, 0.0385)

    def test_hybrid_expected_return_probability_gate_blocks_low_probability_candidate(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base + 60000, quote_volume=1000000000.0, trade_return=0.01),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.40, 0.80],
            calibrated_probability=[0.40, 0.80],
            predicted_trade_return=[0.03, 0.02],
        )
        args = SimpleNamespace(
            hybrid_return_combination="expected_return",
            hybrid_min_probability=0.50,
            hybrid_score_mode="basic",
            hybrid_uncertainty_penalty=0.0,
        )
        execution = pipeline.portfolio_execution(
            rows,
            bundle,
            0.0,
            0.001,
            0.0005,
            10000.0,
            0.10,
            0.01,
            0,
            60,
            5,
            objective_mode="hybrid",
            trade_score_name="hybrid",
            hybrid_runtime_args=args,
            hybrid_min_score=0.0,
        )
        self.assertEqual(sorted(execution["executed"]), [1])

    def test_hybrid_conditional_payoff_uses_empirical_validation_payoffs(self):
        rows = [
            row(1600000000000, label=1, trade_return=0.03),
            row(1600000060000, label=1, trade_return=0.05),
            row(1600000120000, label=0, trade_return=-0.01),
            row(1600000180000, label=0, trade_return=-0.03),
        ]
        bundle = pipeline.build_prediction_bundle(calibrated_probability=[0.75])
        args = SimpleNamespace(
            hybrid_return_combination="conditional_payoff",
            hybrid_min_probability=0.0,
            conditional_payoff_min_positive_rows=1,
            conditional_payoff_min_negative_rows=1,
            conditional_payoff_max_rows=100,
            effective_upside_target=0.02,
            effective_downside_stop=0.02,
            upside_target=0.02,
            downside_stop=0.02,
        )
        bundle["hybrid_return_context"] = pipeline.fit_hybrid_return_context(rows, bundle, args)
        details = pipeline.hybrid_score_details_for_bundle(bundle, 0, 0.001, 0.0005, args=args)
        self.assertAlmostEqual(details["conditional_expected_win_return"], 0.04)
        self.assertAlmostEqual(details["conditional_expected_loss_return"], -0.02)
        self.assertEqual(details["conditional_payoff_source"], "empirical_validation")
        self.assertAlmostEqual(details["base_hybrid_score"], 0.75 * 0.04 + 0.25 * -0.02 - 0.0015)

    def test_hybrid_conditional_payoff_falls_back_when_validation_rows_are_thin(self):
        rows = [row(1600000000000, label=1, trade_return=0.08)]
        bundle = pipeline.build_prediction_bundle(calibrated_probability=[0.6])
        args = SimpleNamespace(
            hybrid_return_combination="conditional_payoff",
            hybrid_min_probability=0.0,
            conditional_payoff_min_positive_rows=2,
            conditional_payoff_min_negative_rows=2,
            conditional_payoff_max_rows=100,
            effective_upside_target=0.02,
            effective_downside_stop=0.03,
            upside_target=0.02,
            downside_stop=0.03,
        )
        bundle["hybrid_return_context"] = pipeline.fit_hybrid_return_context(rows, bundle, args)
        details = pipeline.hybrid_score_details_for_bundle(bundle, 0, 0.001, 0.0005, args=args)
        self.assertAlmostEqual(details["conditional_expected_win_return"], 0.02)
        self.assertAlmostEqual(details["conditional_expected_loss_return"], -0.03)
        self.assertEqual(details["conditional_payoff_source"], "fixed_fallback")

    def test_hybrid_risk_adjusted_subtracts_uncertainty_after_base_score(self):
        bundle = pipeline.build_prediction_bundle(
            probability=[0.2],
            calibrated_probability=[0.2],
            predicted_trade_return=[0.04],
            predicted_return_uncertainty=[0.1],
        )
        args = SimpleNamespace(
            hybrid_return_combination="expected_return",
            hybrid_min_probability=0.0,
        )
        score = pipeline.hybrid_score_value(
            bundle,
            0,
            0.001,
            0.0005,
            "risk_adjusted",
            0.5,
            args,
        )
        self.assertAlmostEqual(score, 0.0385 - 0.05)

    def test_score_values_for_bundle_hybrid_uses_selected_final_hybrid_score(self):
        rows = [row(1600000000000), row(1600000060000)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.90, 0.10],
            calibrated_probability=[0.90, 0.10],
            predicted_trade_return=[0.01, 0.04],
        )
        args = SimpleNamespace(
            objective_mode="hybrid",
            fee=0.001,
            slippage=0.0005,
            validation_slippage_multiplier=1.0,
            hybrid_score_mode="basic",
            hybrid_uncertainty_penalty=0.0,
            hybrid_return_combination="expected_return",
            hybrid_min_probability=0.0,
            upside_target=0.05,
            downside_stop=0.02,
        )
        scores = pipeline.score_values_for_bundle(rows, bundle, args)
        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_topk_score_selection_by_predicted_return(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base, quote_volume=1000000000.0, trade_return=0.03),
            row(base, quote_volume=1000000000.0, trade_return=0.02),
        ]
        bundle = pipeline.build_prediction_bundle(predicted_trade_return=[0.01, 0.03, 0.02])
        execution = pipeline.portfolio_execution(
            rows,
            bundle,
            0.0,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="topk_score",
            top_k_per_minute=2,
            objective_mode="return_regression",
            trade_score_name="predicted_return",
            min_predicted_net_return=0.0,
        )
        self.assertEqual(sorted(execution["executed"]), [1, 2])
        self.assertEqual(execution["executed_selection_ranks"][1], 1)
        self.assertEqual(execution["executed_selection_ranks"][2], 2)

    def test_fold_status_classification(self):
        self.assertEqual(pipeline.fold_status_from_metrics(0, 0.0), "inactive")
        self.assertEqual(pipeline.fold_status_from_metrics(1, 1.0), "active_profitable")
        self.assertEqual(pipeline.fold_status_from_metrics(1, -1.0), "active_losing")

    def test_daily_trade_limit_blocks_extra_entries(self):
        base = 1600000000000
        rows = [row(base + index * 60000, trade_return=0.01, quote_volume=1000000000.0) for index in range(3)]
        execution = pipeline.portfolio_execution(
            rows, [1.0, 1.0, 1.0], 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 5,
            max_trades_per_day=2,
        )
        self.assertEqual(len(execution["executed"]), 2)
        self.assertEqual(execution["daily_trade_limit_blocked"], 1)

    def test_fold_trade_limit_blocks_extra_entries(self):
        base = 1600000000000
        rows = [row(base + index * 60000, trade_return=0.01, quote_volume=1000000000.0) for index in range(4)]
        execution = pipeline.portfolio_execution(
            rows, [1.0, 1.0, 1.0, 1.0], 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 5,
            max_trades_per_fold=2,
        )
        self.assertEqual(len(execution["executed"]), 2)
        self.assertEqual(execution["fold_trade_limit_blocked"], 2)

    def test_daily_loss_limit_blocks_after_realized_losses(self):
        base = 1600000000000
        rows = [
            row(base, trade_return=-0.02, quote_volume=1000000000.0),
            row(base + 60000, trade_return=0.01, quote_volume=1000000000.0),
        ]
        execution = pipeline.portfolio_execution(
            rows, [1.0, 1.0], 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 1,
            max_losing_trades_per_day=1,
        )
        self.assertEqual(list(execution["executed"]), [0])
        self.assertEqual(execution["daily_loss_limit_blocked"], 1)

    def test_daily_drawdown_pause_blocks_same_day_entries(self):
        base = 1600000000000
        rows = [
            row(base, trade_return=-0.02, quote_volume=1000000000.0),
            row(base + 60000, trade_return=0.01, quote_volume=1000000000.0),
            row(base + 120000, trade_return=0.01, quote_volume=1000000000.0),
        ]
        execution = pipeline.portfolio_execution(
            rows, [1.0, 1.0, 1.0], 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 1,
            max_daily_drawdown=0.001,
            pause_after_drawdown_minutes=120,
        )
        self.assertEqual(list(execution["executed"]), [0])
        self.assertEqual(execution["daily_drawdown_limit_blocked"], 2)

    def test_threshold_penalized_score(self):
        metrics = {"predicted_trades": 120, "portfolio_profit": 12.0, "max_capital_drawdown": 0.10, "precision": 0.4}
        base_score, penalized_score = pipeline.threshold_penalized_score(
            metrics,
            "avg_profit",
            drawdown_penalty=0.5,
            trade_count_penalty=0.1,
            target_validation_trades=100,
        )
        self.assertAlmostEqual(base_score, 0.1)
        self.assertAlmostEqual(penalized_score, 0.1 - 0.5 * 0.10 - 0.1 * 20)

    def test_threshold_score_ev_uses_expected_value(self):
        metrics = {
            "predicted_trades": 10,
            "portfolio_profit": 5.0,
            "average_expected_value": 0.0125,
        }
        self.assertAlmostEqual(pipeline.threshold_score(metrics, "avg_profit"), 0.5)
        self.assertAlmostEqual(pipeline.threshold_score(metrics, "ev"), 0.0125)

    def test_parse_ensemble_windows(self):
        self.assertEqual(pipeline.parse_ensemble_windows("6,9,12,9"), [6, 9, 12])
        self.assertEqual(pipeline.parse_ensemble_windows(""), [])
        with self.assertRaises(ValueError):
            pipeline.parse_ensemble_windows("6,-1")

    def test_linear_regression_calibration_improves_simple_rmse(self):
        predictions = [-0.01, 0.0, 0.01, 0.02, 0.03]
        actuals = [0.0, 0.01, 0.02, 0.03, 0.04]
        calibration = pipeline.fit_linear_regression_calibration(predictions, actuals, 100)
        calibrated = pipeline.apply_regression_calibration(list(predictions), calibration)
        self.assertLessEqual(
            pipeline.rmse_score(calibrated, actuals),
            pipeline.rmse_score(predictions, actuals) + 1e-12,
        )
        self.assertEqual(calibration["mode"], "linear")

    def test_risk_adjusted_return_target_round_trip(self):
        rows = [
            pipeline.DataRow("TEST", "2020-01", 0, 1, 1, 0.02, 0.02, 0.02, -0.01, 1000.0, [0.01], {"rolling_volatility_60m": 0}),
            pipeline.DataRow("TEST", "2020-01", 0, 2, 0, -0.01, -0.01, 0.01, -0.02, 1000.0, [0.02], {"rolling_volatility_60m": 0}),
        ]
        args = SimpleNamespace(
            regression_target="risk_adjusted_return",
            risk_adjusted_return_epsilon=1e-6,
            fee=0.001,
            slippage=0.0005,
        )
        targets = pipeline.regression_targets_for_rows(rows, args)
        restored = pipeline.regression_predictions_to_trade_return(targets, rows, args)
        self.assertAlmostEqual(float(restored[0]), 0.02, places=6)
        self.assertAlmostEqual(float(restored[1]), -0.01, places=6)

    def test_dynamic_hybrid_thresholds_fallback_when_features_missing(self):
        args = SimpleNamespace(
            dynamic_hybrid_thresholds="btc_regime",
            btc_bullish_threshold=0.01,
            btc_bearish_threshold=-0.01,
            hybrid_min_score_bullish=0.001,
            hybrid_min_score_neutral=0.0015,
            hybrid_min_score_bearish=0.0025,
            volatility_high_threshold=0.02,
            hybrid_min_score_high_vol=0.0025,
            hybrid_min_score_normal_vol=0.001,
        )
        thresholds, buckets = pipeline.compute_dynamic_hybrid_thresholds([row(1600000000000)], args, 0.001)
        self.assertIsNone(thresholds)
        self.assertIsNone(buckets)

    def test_meta_filter_trains_on_candidate_rows_and_scores_them(self):
        rows = []
        probabilities = []
        predicted_returns = []
        for index in range(24):
            winner = index % 2 == 0
            rows.append(
                pipeline.DataRow(
                    "TEST",
                    "2020-01",
                    0,
                    1600000000000 + index * 60000,
                    1 if winner else 0,
                    0.02 if winner else -0.02,
                    0.02 if winner else -0.02,
                    0.03,
                    -0.03,
                    1000000.0,
                    [],
                )
            )
            probabilities.append(0.9 if winner else 0.55)
            predicted_returns.append(0.03 if winner else 0.015)
        bundle = pipeline.build_prediction_bundle(
            probability=probabilities,
            calibrated_probability=list(probabilities),
            predicted_trade_return=predicted_returns,
            raw_predicted_trade_return=list(predicted_returns),
            predicted_return_uncertainty=[0.001] * len(rows),
        )
        args = SimpleNamespace(
            fee=0.001,
            slippage=0.0005,
            validation_slippage_multiplier=1.0,
            test_slippage_multiplier=1.0,
            initial_capital=10000.0,
            max_position_fraction=0.10,
            max_volume_fraction=0.01,
            max_trades_per_period=0,
            trade_period_minutes=60,
            holding_period_minutes=5,
            threshold_objective="avg_profit",
            trade_selection="threshold",
            top_k_per_minute=1,
            upside_target=0.02,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="hybrid",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.001,
            max_trades_per_day=0,
            max_trades_per_fold=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            meta_filter="logistic",
            meta_filter_min_probability=0.55,
            meta_filter_max_rows=500000,
            dynamic_hybrid_thresholds="none",
            hybrid_score_mode="risk_adjusted",
            hybrid_uncertainty_penalty=0.25,
            hybrid_uncertainty_method="bucket_residual",
            n_jobs=1,
        )
        meta_info = pipeline.fit_meta_filter(rows, bundle, 0.001, args)
        self.assertTrue(meta_info["enabled"])
        scored = pipeline.apply_meta_filter(rows, bundle, 0.001, args, meta_info)
        self.assertGreater(float(scored[0]), float(scored[1]))

    def test_recalibrate_meta_filter_validation_disables_filter_when_validation_collapses(self):
        rows = [row(1600000000000 + index * 60000, label=1, trade_return=0.02) for index in range(12)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.9] * len(rows),
            calibrated_probability=[0.9] * len(rows),
            predicted_trade_return=[0.03] * len(rows),
            raw_predicted_trade_return=[0.03] * len(rows),
            predicted_return_uncertainty=[0.001] * len(rows),
        )
        args = SimpleNamespace(
            fee=0.001,
            slippage=0.0005,
            validation_slippage_multiplier=1.0,
            test_slippage_multiplier=1.0,
            initial_capital=10000.0,
            max_position_fraction=0.10,
            max_volume_fraction=0.01,
            max_trades_per_period=0,
            trade_period_minutes=60,
            holding_period_minutes=5,
            threshold_objective="avg_profit",
            trade_selection="threshold",
            top_k_per_minute=1,
            upside_target=0.02,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="hybrid",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.001,
            max_trades_per_day=0,
            max_trades_per_fold=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            meta_filter="logistic",
            meta_filter_min_probability=0.55,
            meta_filter_max_rows=500000,
            dynamic_hybrid_thresholds="none",
            hybrid_score_mode="risk_adjusted",
            hybrid_uncertainty_penalty=0.10,
            hybrid_uncertainty_method="bucket_residual",
            min_validation_trades=5,
            max_validation_trades=150,
            min_validation_precision=0.25,
            threshold_drawdown_penalty=0.0,
            threshold_trade_count_penalty=0.0,
            target_validation_trades=0,
            profit_safety="explore",
            n_jobs=1,
        )
        baseline_metrics = pipeline.evaluate(
            rows,
            bundle,
            0.001,
            args.fee,
            args.slippage * args.validation_slippage_multiplier,
            compute_auc=False,
            initial_capital=args.initial_capital,
            max_position_fraction=args.max_position_fraction,
            max_volume_fraction=args.max_volume_fraction,
            max_trades_per_period=args.max_trades_per_period,
            trade_period_minutes=args.trade_period_minutes,
            holding_period_minutes=args.holding_period_minutes,
            threshold_objective=args.threshold_objective,
            trade_selection=args.trade_selection,
            top_k_per_minute=args.top_k_per_minute,
            upside_target=args.upside_target,
            downside_stop=args.downside_stop,
            ev_safety_margin=args.ev_safety_margin,
            objective_mode=args.objective_mode,
            trade_score_name="hybrid",
            min_predicted_net_return=args.min_predicted_net_return,
            hybrid_min_score=args.hybrid_min_score,
            max_trades_per_day=args.max_trades_per_day,
            max_trades_per_fold=0,
            max_losing_trades_per_day=args.max_losing_trades_per_day,
            max_daily_drawdown=args.max_daily_drawdown,
            pause_after_drawdown_minutes=args.pause_after_drawdown_minutes,
            hybrid_runtime_args=args,
            symbol_filter_info=None,
        )
        baseline_selection = pipeline.build_selected_threshold_result(
            0.001,
            baseline_metrics,
            args.threshold_objective,
            -float("inf"),
            args.threshold_drawdown_penalty,
            args.threshold_trade_count_penalty,
            args.target_validation_trades,
            "hybrid",
        )
        meta_info = {
            "enabled": True,
            "mode": "logistic",
            "rows": 12,
            "positive_rate": 0.5,
            "accuracy": 0.7,
            "auc": 0.0,
            "model": object(),
        }
        with mock.patch.object(pipeline, "apply_meta_filter", return_value=pipeline.np.zeros(len(rows), dtype=pipeline.np.float32) if pipeline.np is not None else [0.0] * len(rows)):
            recalibrated_info, recalibrated_selection = pipeline.recalibrate_meta_filter_validation(
                rows,
                bundle,
                0.001,
                args,
                baseline_selection,
                meta_info,
                None,
                "hybrid",
            )
        self.assertFalse(recalibrated_info["enabled"])
        self.assertEqual(recalibrated_info["disabled_reason"], "validation_under_min_trades")
        self.assertEqual(recalibrated_selection["validation_metrics"]["predicted_trades"], baseline_selection["validation_metrics"]["predicted_trades"])

    def test_recalibrate_symbol_filter_validation_disables_underperforming_filter(self):
        rows = [symbol_row("GOOD", 1600000000000 + index * 60000, label=1, trade_return=0.02) for index in range(8)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.9] * len(rows),
            calibrated_probability=[0.9] * len(rows),
            predicted_trade_return=[0.03] * len(rows),
            raw_predicted_trade_return=[0.03] * len(rows),
            predicted_return_uncertainty=[0.001] * len(rows),
        )
        args = SimpleNamespace(
            fee=0.001,
            slippage=0.0005,
            validation_slippage_multiplier=1.0,
            initial_capital=10000.0,
            max_position_fraction=0.10,
            max_volume_fraction=0.01,
            max_trades_per_period=0,
            trade_period_minutes=60,
            holding_period_minutes=5,
            threshold_objective="avg_profit",
            trade_selection="threshold",
            top_k_per_minute=1,
            upside_target=0.02,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="hybrid",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.001,
            max_trades_per_day=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            threshold_drawdown_penalty=0.0,
            threshold_trade_count_penalty=0.0,
            target_validation_trades=0,
            min_validation_trades=5,
            max_validation_trades=150,
            min_validation_precision=0.25,
            profit_safety="explore",
            hybrid_score_mode="risk_adjusted",
            hybrid_uncertainty_penalty=0.10,
        )
        baseline_metrics = {"predicted_trades": 8, "selected_objective_score": 10.0, "precision": 0.5}
        selection = {"objective_score": 10.0, "validation_metrics": dict(baseline_metrics)}
        symbol_filter_info = {
            "mode": "positive_avg_profit",
            "enabled": True,
            "allowed_symbols": ["GOOD"],
            "total_symbols": 2,
        }
        with mock.patch.object(
            pipeline,
            "evaluate",
            return_value={"predicted_trades": 3, "precision": 0.5, "portfolio_profit": 4.0, "raw_signal_trades": 3, "max_capital_drawdown": 0.0},
        ):
            recalibrated_info, recalibrated_selection = pipeline.recalibrate_symbol_filter_validation(
                rows,
                bundle,
                0.001,
                args,
                selection,
                symbol_filter_info,
                "hybrid",
            )
        self.assertFalse(recalibrated_info["enabled"])
        self.assertEqual(recalibrated_info["disabled_reason"], "validation_under_min_trades")
        self.assertIs(recalibrated_selection, selection)

    def test_selected_thresholds_for_bundle_hybrid_uses_score_adaptive_grid(self):
        rows = [row(1600000000000 + index * 60000, label=1 if index % 2 == 0 else 0, trade_return=0.02 if index % 2 == 0 else -0.01) for index in range(12)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7],
            calibrated_probability=[0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7],
            predicted_trade_return=[-0.005, -0.003, -0.001, 0.0, 0.002, 0.003, 0.005, 0.007, 0.009, 0.012, 0.015, 0.02],
            raw_predicted_trade_return=[-0.005, -0.003, -0.001, 0.0, 0.002, 0.003, 0.005, 0.007, 0.009, 0.012, 0.015, 0.02],
            predicted_return_uncertainty=[0.001] * 12,
        )
        args = SimpleNamespace(
            objective_mode="hybrid",
            disable_adaptive_thresholds=False,
            adaptive_threshold_sample_rows=1000000,
            fee=0.001,
            slippage=0.0005,
            validation_slippage_multiplier=1.0,
            hybrid_score_mode="risk_adjusted",
            hybrid_uncertainty_penalty=0.10,
            hybrid_min_score=0.001,
        )
        thresholds, fixed_threshold, score_values = pipeline.selected_thresholds_for_bundle(bundle, rows, args)
        self.assertEqual(fixed_threshold, 0.001)
        self.assertIsNotNone(score_values)
        self.assertGreater(len(thresholds), 1)
        self.assertIn(0.001, thresholds)
        self.assertTrue(all(value >= 0.001 for value in thresholds))

    def test_adaptive_score_thresholds_respect_base_floor(self):
        thresholds = pipeline.adaptive_score_thresholds(
            [-0.02, -0.01, -0.005, 0.0, 0.0008, 0.0012, 0.004],
            0.001,
        )
        self.assertEqual(thresholds[0], 0.001)
        self.assertTrue(all(value >= 0.001 for value in thresholds))

    def test_select_fixed_split_model_retries_with_backfill_after_no_trade(self):
        rows = [
            pipeline.DataRow(
                "TESTUSDT",
                "2020-{:02d}".format(month_index + 1),
                month_index,
                1600000000000 + month_index * 60000,
                1 if month_index % 2 == 0 else 0,
                0.01,
                0.01,
                0.02,
                -0.01,
                1000000.0,
                [0.0, 0.0],
            )
            for month_index in range(8)
        ]
        train_rows = pipeline.select_month_range(rows, 0, 6)
        validation_rows = pipeline.select_month_range(rows, 6, 7)
        args = SimpleNamespace(
            fixed_validation_backfill_months=2,
        )
        with mock.patch.object(
            pipeline,
            "fit_select_model",
            side_effect=[
                {"threshold": 1.01, "validation_metrics": {"predicted_trades": 0}},
                {"threshold": 0.002, "validation_metrics": {"predicted_trades": 8}},
            ],
        ) as mocked_fit:
            selected = pipeline.select_fixed_split_model(
                train_rows,
                validation_rows,
                ["ret_1m"],
                args,
                "internal",
            )
        self.assertEqual(mocked_fit.call_count, 2)
        fallback_train_rows = mocked_fit.call_args_list[1].args[0]
        fallback_validation_rows = mocked_fit.call_args_list[1].args[1]
        self.assertLess(len(fallback_train_rows), len(train_rows))
        self.assertGreater(len(fallback_validation_rows), len(validation_rows))
        self.assertEqual(selected["fixed_validation_backfill_used"], 1)
        self.assertEqual(selected["fixed_validation_backfill_months"], 2)

    def test_threshold_score_profit_balanced_discourages_sparse_thresholds(self):
        sparse_metrics = {
            "predicted_trades": 5,
            "portfolio_profit": 100.0,
            "precision": 0.8,
        }
        dense_metrics = {
            "predicted_trades": 75,
            "portfolio_profit": 95.0,
            "precision": 0.6,
        }
        sparse_score = pipeline.threshold_score(
            sparse_metrics,
            "profit_balanced",
            target_validation_trades=75,
            min_validation_trades=5,
            max_validation_trades=150,
        )
        dense_score = pipeline.threshold_score(
            dense_metrics,
            "profit_balanced",
            target_validation_trades=75,
            min_validation_trades=5,
            max_validation_trades=150,
        )
        self.assertLess(sparse_score, dense_score)
        self.assertLess(sparse_score, sparse_metrics["portfolio_profit"])
        self.assertAlmostEqual(dense_score, dense_metrics["portfolio_profit"])

    def test_hybrid_relative_uncertainty_penalty_scales_by_typical_return(self):
        bundle = pipeline.build_prediction_bundle(
            probability=[0.1],
            calibrated_probability=[0.1],
            predicted_trade_return=[0.04],
            predicted_return_uncertainty=[0.02],
            uncertainty_context={
                "mode": "bucket_residual",
                "global_std": 0.02,
                "penalty_reference_scale": 0.01,
                "penalty_mode": "relative_return",
            },
        )
        args = SimpleNamespace(
            hybrid_return_combination="expected_return",
            hybrid_min_probability=0.0,
            hybrid_uncertainty_penalty_mode="relative_return",
        )
        score = pipeline.hybrid_score_value(
            bundle,
            0,
            0.001,
            0.0005,
            "risk_adjusted",
            0.5,
            args,
        )
        self.assertAlmostEqual(score, 0.0385 - 0.005)

    def test_empirical_ev_payoff_uses_validation_returns(self):
        rows = [
            row(1600000000000, label=1, trade_return=0.04),
            row(1600000060000, label=1, trade_return=0.02),
            row(1600000120000, label=0, trade_return=-0.01),
            row(1600000180000, label=0, trade_return=-0.03),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.6] * len(rows),
            calibrated_probability=[0.6] * len(rows),
        )
        args = SimpleNamespace(
            ev_payoff_mode="empirical_validation",
            ev_payoff_calibration_max_rows=0,
            ev_payoff_min_positive_rows=1,
            ev_payoff_min_negative_rows=1,
            effective_upside_target=0.02,
            effective_downside_stop=0.02,
            objective_mode="classification",
        )
        context = pipeline.fit_ev_payoff_context(rows, bundle, args)
        bundle["ev_context"] = context
        details = pipeline.expected_value_details_for_bundle(bundle, 0, 0.02, 0.02, 0.001, 0.0005, args)
        self.assertEqual(context["ev_payoff_source"], "empirical_validation")
        self.assertAlmostEqual(context["ev_expected_win_return"], 0.03)
        self.assertAlmostEqual(context["ev_expected_loss_return"], -0.02)
        self.assertAlmostEqual(details["expected_value"], 0.6 * 0.03 + 0.4 * -0.02 - 0.0015)

    def test_predicted_return_ev_uses_predicted_return_source(self):
        rows = [row(1600000000000, label=1, trade_return=0.04)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.5],
            calibrated_probability=[0.5],
            predicted_trade_return=[0.04],
            raw_predicted_trade_return=[0.04],
        )
        args = SimpleNamespace(
            ev_payoff_mode="predicted_return",
            ev_payoff_calibration_max_rows=0,
            ev_payoff_min_positive_rows=1,
            ev_payoff_min_negative_rows=1,
            effective_upside_target=0.02,
            effective_downside_stop=0.02,
            objective_mode="hybrid",
        )
        context = pipeline.fit_ev_payoff_context(rows, bundle, args)
        bundle["ev_context"] = context
        details = pipeline.expected_value_details_for_bundle(bundle, 0, 0.02, 0.02, 0.001, 0.0005, args)
        self.assertEqual(context["ev_payoff_source"], "predicted_return")
        self.assertAlmostEqual(details["expected_value_predicted_return"], 0.04 - 0.0015)
        self.assertAlmostEqual(details["expected_value"], details["expected_value_predicted_return"])

    def test_symbol_filter_candidate_blend_keeps_crowded_out_good_symbol(self):
        rows = [
            symbol_row("LEADER", 1600000000000, label=1, trade_return=0.02),
            symbol_row("GOOD", 1600000000000, label=1, trade_return=0.015),
            symbol_row("BAD", 1600000000000, label=0, trade_return=-0.02),
            symbol_row("LEADER", 1600000060000, label=1, trade_return=0.02),
            symbol_row("GOOD", 1600000060000, label=1, trade_return=0.015),
            symbol_row("BAD", 1600000060000, label=0, trade_return=-0.02),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.96, 0.95, 0.94, 0.96, 0.95, 0.94],
            calibrated_probability=[0.96, 0.95, 0.94, 0.96, 0.95, 0.94],
        )
        args = SimpleNamespace(
            symbol_validation_filter="positive_avg_profit",
            symbol_filter_stage="candidate_blend",
            symbol_filter_min_candidates=2,
            symbol_filter_min_executed=5,
            symbol_filter_candidate_weight=0.5,
            symbol_filter_executed_weight=0.5,
            symbol_filter_shrinkage=0.0,
            min_symbol_validation_trades=1,
            min_symbol_validation_average_profit=0.0,
            min_symbol_validation_total_profit=0.0,
            fee=0.0,
            slippage=0.0,
            validation_slippage_multiplier=1.0,
            initial_capital=10000.0,
            max_position_fraction=0.25,
            max_volume_fraction=1.0,
            max_trades_per_period=0,
            trade_period_minutes=60,
            holding_period_minutes=5,
            threshold_objective="avg_profit",
            trade_selection="topk_score",
            top_k_per_minute=1,
            upside_target=0.02,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="classification",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.0,
            max_trades_per_day=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            hybrid_score_mode="basic",
            hybrid_uncertainty_penalty=0.0,
            dynamic_hybrid_thresholds="none",
            meta_filter="none",
            meta_filter_min_probability=0.0,
        )
        symbol_filter = pipeline.fit_symbol_validation_filter(rows, bundle, 0.5, args, "probability")
        self.assertTrue(symbol_filter["enabled"])
        self.assertIn("GOOD", symbol_filter["allowed_symbols"])
        self.assertIn("BAD", symbol_filter["filtered_symbols"])

    def test_symbol_filter_candidate_blend_does_not_double_count_executed_support(self):
        rows = [
            symbol_row("LEADER", 1600000000000, label=1, trade_return=0.02),
            symbol_row("GOOD", 1600000000000, label=1, trade_return=0.015),
            symbol_row("LEADER", 1600000060000, label=1, trade_return=0.02),
            symbol_row("GOOD", 1600000060000, label=1, trade_return=0.015),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.96, 0.95, 0.96, 0.95],
            calibrated_probability=[0.96, 0.95, 0.96, 0.95],
        )
        args = SimpleNamespace(
            symbol_validation_filter="positive_avg_profit",
            symbol_filter_stage="candidate_blend",
            symbol_filter_min_candidates=1,
            symbol_filter_min_executed=1,
            symbol_filter_candidate_weight=0.5,
            symbol_filter_executed_weight=0.5,
            symbol_filter_shrinkage=2.0,
            min_symbol_validation_trades=1,
            min_symbol_validation_average_profit=0.0,
            min_symbol_validation_total_profit=0.0,
            fee=0.0,
            slippage=0.0,
            validation_slippage_multiplier=1.0,
            initial_capital=10000.0,
            max_position_fraction=0.25,
            max_volume_fraction=1.0,
            max_trades_per_period=0,
            trade_period_minutes=60,
            holding_period_minutes=5,
            threshold_objective="avg_profit",
            trade_selection="topk_score",
            top_k_per_minute=1,
            upside_target=0.02,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="classification",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.0,
            max_trades_per_day=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            hybrid_score_mode="basic",
            hybrid_uncertainty_penalty=0.0,
            dynamic_hybrid_thresholds="none",
            meta_filter="none",
            meta_filter_min_probability=0.0,
        )
        symbol_filter = pipeline.fit_symbol_validation_filter(rows, bundle, 0.5, args, "probability")
        leader_diag = next(record for record in symbol_filter["diagnostics"] if record["symbol"] == "LEADER")
        self.assertAlmostEqual(leader_diag["candidate_quality"], 0.02, places=6)
        self.assertAlmostEqual(leader_diag["executed_quality"], 0.02, places=6)
        self.assertAlmostEqual(leader_diag["symbol_score"], 0.01875, places=6)

    def test_threshold_tiebreaker_balanced_prefers_more_active_days(self):
        args = SimpleNamespace(
            threshold_tiebreaker="balanced",
            threshold_tie_epsilon=1e-9,
            threshold_target_trades=75,
            threshold_target_active_days=0,
            target_validation_trades=75,
            threshold_objective="avg_profit",
        )
        quieter = pipeline.build_selected_threshold_result(
            0.001,
            {
                "predicted_trades": 75,
                "portfolio_profit": 30.0,
                "max_capital_drawdown": 0.10,
                "precision": 0.4,
                "raw_signal_trades": 75,
                "portfolio_return": 0.003,
                "recall": 0.2,
                "average_profit_after_fee_and_slippage": 0.4,
                "total_profit_after_fee_and_slippage": 30.0,
                "active_days": 2,
            },
            "avg_profit",
            -float("inf"),
            0.0,
            0.0,
            75,
            "hybrid",
        )
        broader = pipeline.build_selected_threshold_result(
            0.002,
            {
                "predicted_trades": 75,
                "portfolio_profit": 30.0,
                "max_capital_drawdown": 0.10,
                "precision": 0.4,
                "raw_signal_trades": 75,
                "portfolio_return": 0.003,
                "recall": 0.2,
                "average_profit_after_fee_and_slippage": 0.4,
                "total_profit_after_fee_and_slippage": 30.0,
                "active_days": 5,
            },
            "avg_profit",
            -float("inf"),
            0.0,
            0.0,
            75,
            "hybrid",
        )
        better, reason = pipeline.compare_threshold_results(
            broader,
            quieter,
            args,
            5,
            150,
        )
        self.assertTrue(better)
        self.assertEqual(reason, "balanced")

    def test_meta_filter_inactive_does_not_mark_trades_as_selected_by_meta_filter(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base + 60000, quote_volume=1000000000.0, trade_return=0.01),
        ]
        args = SimpleNamespace(
            meta_filter="logistic",
            meta_filter_min_probability=0.55,
            hybrid_score_mode="risk_adjusted",
            hybrid_uncertainty_penalty=0.10,
        )
        execution = pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(probability=[0.9, 0.9], calibrated_probability=[0.9, 0.9]),
            0.5,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            0,
            60,
            5,
            threshold_objective="avg_profit",
            trade_selection="threshold",
            top_k_per_minute=1,
            objective_mode="classification",
            trade_score_name="probability",
            hybrid_runtime_args=args,
        )
        self.assertEqual(execution["executed_selected_by_meta_filter"].get(0, 0), 0)

    def test_meta_feature_matrix_uses_stage_slippage(self):
        rows = [row(1600000000000, label=1, trade_return=0.02)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.5],
            calibrated_probability=[0.5],
            predicted_trade_return=[0.02],
            raw_predicted_trade_return=[0.02],
            predicted_return_uncertainty=[0.001],
        )
        args = SimpleNamespace(
            fee=0.001,
            slippage=0.0005,
            upside_target=0.02,
            downside_stop=0.02,
            objective_mode="hybrid",
            hybrid_score_mode="risk_adjusted",
            hybrid_uncertainty_penalty=0.10,
        )
        low = pipeline.meta_feature_matrix(rows, bundle, [0], args, fee=0.001, slippage=0.0005)
        high = pipeline.meta_feature_matrix(rows, bundle, [0], args, fee=0.001, slippage=0.005)
        self.assertGreater(float(low[0][3]), float(high[0][3]))
        self.assertGreater(float(low[0][4]), float(high[0][4]))

    def test_internal_meta_logistic_accepts_numpy_arrays(self):
        if pipeline.np is None:
            self.skipTest("requires numpy")
        model = pipeline.InternalMetaLogistic(epochs=4, learning_rate=0.05, l2=0.001)
        x_rows = pipeline.np.asarray([
            [0.1, 0.0],
            [0.2, 0.1],
            [-0.1, -0.2],
            [-0.2, -0.1],
        ], dtype=pipeline.np.float32)
        y_rows = pipeline.np.asarray([1, 1, 0, 0], dtype=pipeline.np.float32)
        model.fit(x_rows, y_rows)
        probabilities = model.predict_proba(x_rows)
        self.assertEqual(len(probabilities), 4)

    def test_internal_mean_regressor_accepts_numpy_targets(self):
        if pipeline.np is None:
            self.skipTest("requires numpy")
        model = pipeline.InternalMeanRegressor()
        x_rows = pipeline.np.asarray([[0.0], [1.0]], dtype=pipeline.np.float32)
        y_rows = pipeline.np.asarray([0.25, -0.25], dtype=pipeline.np.float32)
        model.fit(x_rows, y_rows, ["feature"])
        predictions = model.predict_values(x_rows)
        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(float(predictions[0]), 0.0, places=6)

    def test_walkforward_active_fold_metrics(self):
        args = SimpleNamespace(
            acceptance_tier="none",
            require_positive_walkforward=False,
            min_profitable_fold_rate=0.55,
            min_median_fold_return=0.0,
            min_mean_fold_return=0.0,
            max_worst_fold_drawdown=1.0,
            overactive_trade_threshold=150,
        )
        summary = pipeline.walkforward_acceptance_summary([
            {"split": "walkforward_fold_1", "portfolio_profit": 10.0, "portfolio_return": 0.02, "precision": 0.5, "max_capital_drawdown": 0.1, "predicted_trades": 2},
            {"split": "walkforward_fold_2", "portfolio_profit": 0.0, "portfolio_return": 0.0, "precision": 0.0, "max_capital_drawdown": 0.0, "predicted_trades": 0},
            {"split": "walkforward_fold_3", "portfolio_profit": -5.0, "portfolio_return": -0.01, "precision": 0.2, "max_capital_drawdown": 0.3, "predicted_trades": 1},
        ], args)
        self.assertEqual(summary["active_fold_count"], 2)
        self.assertEqual(summary["inactive_fold_count"], 1)
        self.assertAlmostEqual(summary["active_fold_rate"], 2.0 / 3.0)
        self.assertEqual(summary["active_profitable_fold_count"], 1)
        self.assertEqual(summary["active_losing_fold_count"], 1)
        self.assertAlmostEqual(summary["profit_per_active_fold"], 2.5)
        self.assertEqual(summary["overactive_losing_folds"], 0)

    def test_walkforward_diagnostic_handles_none_calibration_info(self):
        args = SimpleNamespace(
            ev_payoff_mode="fixed_targets",
            effective_upside_target=0.02,
            upside_target=0.02,
            effective_downside_stop=0.02,
            downside_stop=0.02,
            dynamic_hybrid_thresholds="none",
            meta_filter="none",
            symbol_validation_filter="none",
            symbol_filter_stage="executed",
            symbol_filter_min_candidates=25,
            symbol_filter_min_executed=5,
            symbol_filter_candidate_weight=0.5,
            symbol_filter_executed_weight=0.5,
            symbol_filter_shrinkage=50.0,
            ensemble_window_list=[],
            hybrid_score_mode="basic",
            hybrid_uncertainty_method="none",
            hybrid_uncertainty_penalty=0.0,
            regression_target="trade_return",
            threshold_tiebreaker="fewer_trades",
            threshold_tie_epsilon=1e-9,
            threshold_target_trades=0,
            threshold_target_active_days=0,
        )
        record = pipeline.build_walkforward_diagnostic_record(
            1,
            "walkforward_fold_1",
            [row(1600000000000)],
            [row(1600000060000)],
            [row(1600000120000)],
            0.5,
            0.1,
            {"predicted_trades": 0, "portfolio_profit": 0.0},
            None,
            {},
            args,
        )
        self.assertEqual(record["calibration"], "none")
        self.assertEqual(record["ev_payoff_mode"], "fixed_targets")

    def test_aggregate_walkforward_selected_objective_ignores_nonfinite_values(self):
        records = [
            {
                "model": "gbdt",
                "split": "walkforward_fold_1",
                "split_mode": "ratio",
                "objective_mode": "hybrid",
                "trade_score": "hybrid",
                "trade_score_name": "hybrid",
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
                "threshold_objective": "avg_profit",
                "selected_threshold": 0.001,
                "selected_score_name": "hybrid",
                "selected_score_threshold": 0.001,
                "selected_objective_score": 0.5,
                "selected_base_objective_score": 0.6,
                "selected_penalized_objective_score": 0.5,
                "selected_validation_trade_count": 10,
                "selected_validation_max_drawdown": 0.1,
                "selected_validation_raw_signal_count": 12,
                "train_rows": 10,
                "validation_rows": 5,
                "test_rows": 5,
                "predicted_trades": 2,
                "precision": 0.5,
                "recall": 0.2,
                "portfolio_profit": 10.0,
                "portfolio_return": 0.01,
            },
            {
                "model": "gbdt",
                "split": "walkforward_fold_2",
                "split_mode": "ratio",
                "objective_mode": "hybrid",
                "trade_score": "hybrid",
                "trade_score_name": "hybrid",
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
                "threshold_objective": "avg_profit",
                "selected_threshold": 1.01,
                "selected_score_name": "hybrid",
                "selected_score_threshold": 1.01,
                "selected_objective_score": float("-inf"),
                "selected_base_objective_score": float("-inf"),
                "selected_penalized_objective_score": float("-inf"),
                "selected_validation_trade_count": 0,
                "selected_validation_max_drawdown": 0.0,
                "selected_validation_raw_signal_count": 0,
                "train_rows": 10,
                "validation_rows": 5,
                "test_rows": 5,
                "predicted_trades": 0,
                "precision": 0.0,
                "recall": 0.0,
                "portfolio_profit": 0.0,
                "portfolio_return": 0.0,
            },
        ]
        aggregate = pipeline.aggregate_fold_records(records, "gbdt", "avg_profit")
        self.assertEqual(aggregate["selected_objective_finite_folds"], 1)
        self.assertEqual(aggregate["selected_objective_nonfinite_folds"], 1)
        self.assertEqual(aggregate["selected_no_trade_folds"], 1)
        self.assertAlmostEqual(aggregate["selected_objective_score"], 0.5)
        self.assertAlmostEqual(aggregate["mean_selected_validation_trade_count"], 5.0)

    def test_acceptance_tier_logic(self):
        args = SimpleNamespace(
            acceptance_tier="research",
            require_positive_walkforward=False,
            min_profitable_fold_rate=0.55,
            min_median_fold_return=0.0,
            min_mean_fold_return=0.0,
            max_worst_fold_drawdown=1.0,
            overactive_trade_threshold=150,
        )
        good = pipeline.walkforward_acceptance_summary([
            {"split": "walkforward_fold_1", "portfolio_profit": 10.0, "portfolio_return": 0.02, "precision": 0.5, "max_capital_drawdown": 0.1, "predicted_trades": 2},
            {"split": "walkforward_fold_2", "portfolio_profit": 5.0, "portfolio_return": 0.01, "precision": 0.4, "max_capital_drawdown": 0.2, "predicted_trades": 2},
            {"split": "walkforward_fold_3", "portfolio_profit": 2.0, "portfolio_return": 0.0, "precision": 0.3, "max_capital_drawdown": 0.1, "predicted_trades": 1},
        ], args)
        bad = pipeline.walkforward_acceptance_summary([
            {"split": "walkforward_fold_1", "portfolio_profit": -10.0, "portfolio_return": -0.02, "precision": 0.1, "max_capital_drawdown": 1.5, "predicted_trades": 2},
            {"split": "walkforward_fold_2", "portfolio_profit": 1.0, "portfolio_return": 0.0, "precision": 0.1, "max_capital_drawdown": 0.5, "predicted_trades": 1},
        ], args)
        self.assertEqual(good["accepted"], 1)
        self.assertEqual(good["strategy_strength"], "research_pass")
        self.assertEqual(bad["accepted"], 0)
        self.assertEqual(bad["strategy_strength"], "rejected")
        self.assertTrue(
            "mean_portfolio_return" in bad["failed_acceptance_checks"]
            or "profitable_fold_rate" in bad["failed_acceptance_checks"]
        )

    def test_experiment_runner_command_generation(self):
        args = run_experiments.parse_args(["--profile", "7.8gb-overtrade-check"])
        experiment = run_experiments.build_experiment_grid_for_profile("7.8gb-overtrade-check", False, 4)[0]
        command = run_experiments.build_command(args, experiment)
        self.assertIn("--memory-budget-gb", command)
        self.assertIn("7.8", command)
        self.assertIn("--objective-mode", command)
        self.assertIn("classification", command)
        self.assertIn("--trade-score", command)
        self.assertIn("ev", command)
        self.assertIn("--threshold-drawdown-penalty", command)
        self.assertIn("--cache-dir", command)
        self.assertIn(".gbdt_cache", command)
        self.assertIn("--cache-only", command)

    def test_configure_output_paths_maps_into_results_dir(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["--results-dir", os.path.join(self.temp.name, "results")])
        pipeline.configure_output_paths(args)
        self.assertTrue(args.predictions_out.startswith(os.path.join(self.temp.name, "results")))
        self.assertTrue(args.run_summary_out.startswith(os.path.join(self.temp.name, "results")))
        self.assertTrue(args.profile_out.startswith(os.path.join(self.temp.name, "results")))

    def test_compress_outputs_gzip_replaces_csv(self):
        parser = pipeline.build_parser()
        path = os.path.join(self.temp.name, "metrics.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("a,b\n1,2\n")
        args = parser.parse_args(["--compress-outputs"])
        args.output_compression = "gzip"
        args.metrics_out = path
        args.predictions_out = ""
        args.walkforward_metrics_out = ""
        args.walkforward_diagnostics_out = ""
        args.walk_predictions_out = ""
        args.feature_importance_out = ""
        args.experiment_summary_out = ""
        args.profile_out = ""
        pipeline.postprocess_output_files(args)
        self.assertFalse(os.path.exists(path))
        self.assertTrue(os.path.exists(path + ".gz"))

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_inspect_cache_does_not_train(self):
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        rows.cleanup()
        profile_path = os.path.join(self.temp.name, "profile.csv")
        with mock.patch.object(pipeline, "run_fixed_split", side_effect=AssertionError("training should not run")):
            status = pipeline.main([
                "--input", self.csv_path,
                "--feature-storage", "memmap32",
                "--cache-dir", cache_dir,
                "--inspect-cache",
                "--profile-out", profile_path,
            ])
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(profile_path))

    def test_walkforward_diagnostic_columns_include_overtrade_fields(self):
        for field in (
            "max_trades_in_day",
            "trades_per_active_day",
            "blocked_trades_total",
            "selected_penalized_objective_score",
            "normalized_microsecond_open_times",
            "selected_score_name",
            "selected_score_threshold",
            "selected_validation_raw_signal_count",
        ):
            self.assertIn(field, pipeline.WALKFORWARD_DIAGNOSTIC_COLUMNS)

    def test_memory_budget_defaults_and_explicit_override(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["--memory-budget-gb", "7.8"])
        explicit = pipeline.parse_explicit_flags(["--memory-budget-gb", "7.8"])
        applied = pipeline.apply_memory_budget_defaults(args, explicit)
        self.assertTrue(applied)
        self.assertEqual(args.feature_storage, "memmap32")
        self.assertEqual(args.max_train_rows, 1500000)
        self.assertEqual(args.max_rss_gb, 7.8)

        args = parser.parse_args(["--memory-budget-gb", "7.8", "--n-jobs", "4"])
        explicit = pipeline.parse_explicit_flags(["--memory-budget-gb", "7.8", "--n-jobs", "4"])
        pipeline.apply_memory_budget_defaults(args, explicit)
        self.assertEqual(args.n_jobs, 4)

    def test_memory_guard_without_psutil(self):
        args = SimpleNamespace(max_rss_gb=7.8, abort_on_memory_limit=True)
        previous_psutil = pipeline.psutil
        try:
            pipeline.psutil = None
            self.assertFalse(pipeline.check_memory_limit("test", args))
            pipeline.log_memory("no-psutil stage")
            self.assertIsNone(pipeline.current_rss_gib("no-psutil stage"))
        finally:
            pipeline.psutil = previous_psutil

    def test_atomic_write_path_produces_valid_json(self):
        output_path = os.path.join(self.temp.name, "atomic.json")
        def write_one(path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"ok": True, "value": 3}, handle)
        pipeline.atomic_write_path(output_path, write_one)
        with open(output_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["value"], 3)

    def test_lightgbm_feature_name_warning_filter_covers_regressor(self):
        matched = False
        for entry in warnings.filters:
            action, message, category, module, lineno = entry
            if action != "ignore" or category is not UserWarning or message is None:
                continue
            text = getattr(message, "pattern", str(message))
            if "LGBM(Classifier|Regressor)" in text:
                matched = True
                break
        self.assertTrue(matched)

    def test_apply_split_embargo_removes_boundary_rows(self):
        train_rows = [row(0), row(4 * 60 * 1000), row(8 * 60 * 1000)]
        validation_rows = [row(20 * 60 * 1000), row(24 * 60 * 1000), row(28 * 60 * 1000)]
        test_rows = [row(40 * 60 * 1000), row(44 * 60 * 1000), row(48 * 60 * 1000)]
        train_rows, validation_rows, test_rows, summary = pipeline.apply_split_embargo(
            train_rows,
            validation_rows,
            test_rows,
            5,
        )
        self.assertEqual(len(train_rows), 3)
        self.assertEqual(len(validation_rows), 1)
        self.assertEqual(len(test_rows), 1)
        self.assertEqual(summary["embargo_minutes"], 5)
        self.assertEqual(summary["embargo_validation_rows_removed"], 2)
        self.assertEqual(summary["embargo_test_rows_removed"], 2)

    def test_calibration_report_from_predictions_prefers_calibrated_probability(self):
        path = os.path.join(self.temp.name, "predictions.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "symbol", "month", "month_index", "open_time", "label",
                "probability", "calibrated_probability", "selected_threshold",
                "forward_return", "trade_return", "predicted",
            ])
            writer.writerow(["BTCUSDT", "2020-01", 0, 0, 1, 0.9, 0.8, 0.5, 0.02, 0.01, 1])
            writer.writerow(["BTCUSDT", "2020-01", 0, 60000, 0, 0.2, 0.3, 0.5, -0.01, -0.02, 0])
        rows, summary = pipeline.calibration_report_from_predictions(path)
        self.assertEqual(len(rows), 20)
        self.assertIn("brier_score", summary)
        self.assertGreaterEqual(summary["expected_calibration_error"], 0.0)
        self.assertTrue(any(row["probability_source"] == "calibrated_probability" for row in rows))

    def test_portfolio_execution_blocks_symbol_during_loss_cooldown(self):
        base_time = 1600000000000
        rows = [
            symbol_row("BTCUSDT", base_time, label=0, trade_return=-0.10),
            symbol_row("BTCUSDT", base_time + 2 * 60 * 1000, label=1, trade_return=0.05),
            symbol_row("BTCUSDT", base_time + 20 * 60 * 1000, label=1, trade_return=0.05),
        ]
        args = SimpleNamespace(
            fee_mode="fixed",
            position_sizing_mode="fixed_fraction",
            min_order_notional=0.0,
            lot_size_step=0.0,
            tick_size=0.0,
            latency_penalty_bps=0.0,
            partial_fill_mode="none",
            max_open_positions=0,
            max_daily_loss_fraction=0.0,
            cooldown_after_loss_minutes=10,
            meta_filter="none",
            volatility_high_threshold=0.02,
            upside_target=0.05,
        )
        execution = pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(probability=[1.0, 1.0, 1.0], calibrated_probability=[1.0, 1.0, 1.0]),
            0.5,
            0.0,
            0.0,
            100.0,
            1.0,
            1.0,
            0,
            60,
            1,
            hybrid_runtime_args=args,
        )
        self.assertEqual(len(execution["executed"]), 2)
        self.assertEqual(execution["cooldown_after_loss_blocked"], 1)

    def test_summary_output_includes_new_fields(self):
        args = SimpleNamespace(
            split_mode="ratio",
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            initial_capital=10000.0,
            max_position_fraction=0.1,
            max_volume_fraction=0.01,
            max_trades_per_period=10,
            trade_period_minutes=60,
            holding_period_minutes=5,
            min_validation_trades=5,
            max_validation_trades=250,
            min_validation_precision=0.25,
            min_selected_threshold=0.9,
            min_predicted_net_return=0.0,
            hybrid_min_score=0.0,
            hybrid_return_combination="expected_return",
            hybrid_min_probability=0.1,
            conditional_payoff_min_positive_rows=25,
            conditional_payoff_min_negative_rows=25,
            conditional_payoff_max_rows=500000,
            profit_safety="explore",
            disable_adaptive_thresholds=False,
            trade_selection="topk_ev",
            top_k_per_minute=3,
            trade_score="auto",
            objective_mode="classification",
            ev_safety_margin=0.002,
            threshold_objective="ev",
            ev_upside_target_source="manifest",
            ev_downside_stop_source="manifest",
            manifest_upside_target=0.02,
            manifest_downside_stop=0.02,
            effective_upside_target=0.02,
            effective_downside_stop=0.02,
            market_regime_features=True,
            market_breadth_features=False,
            max_trades_per_day=0,
            max_trades_per_fold=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            threshold_drawdown_penalty=0.0,
            threshold_trade_count_penalty=0.0,
            target_validation_trades=0,
            overactive_trade_threshold=150,
            require_positive_walkforward=True,
            min_profitable_fold_rate=0.55,
            min_median_fold_return=0.0,
            min_mean_fold_return=0.0,
            max_worst_fold_drawdown=1.0,
            acceptance_tier="exploration",
            feature_storage="memmap32",
            memmap_dir="",
            cache_dir=".gbdt_cache",
            max_train_rows=1500000,
            max_validation_rows=750000,
            max_final_train_rows=1500000,
            prediction_batch_rows=200000,
            max_bin=63,
            subsample_for_bin=100000,
            lightgbm_histogram_pool_mb=128.0,
            n_jobs=2,
            memory_budget_gb=7.8,
            max_rss_gb=7.8,
            input=self.csv_path,
            run_summary_out=os.path.join(self.temp.name, "summary.json"),
            experiment_summary_out=os.path.join(self.temp.name, "summary.csv"),
            _explicit_flags={"memory_budget_gb", "max_rss_gb"},
        )
        fixed_record = {
            "selected_threshold": 0.95,
            "portfolio_profit": 10.0,
            "portfolio_return": 0.01,
            "precision": 0.5,
            "recall": 0.25,
            "raw_precision": 0.5,
            "raw_recall": 0.25,
            "hybrid_return_combination": "expected_return",
            "hybrid_min_probability": 0.1,
            "conditional_expected_win_return": 0.03,
            "conditional_expected_loss_return": -0.02,
            "conditional_payoff_rows": 100,
            "conditional_payoff_positive_rows": 25,
            "conditional_payoff_negative_rows": 30,
            "conditional_payoff_source": "empirical_validation",
        }
        walk_records = [{
            "split": "walkforward_fold_1",
            "portfolio_profit": 10.0,
            "portfolio_return": 0.01,
            "precision": 0.5,
            "max_capital_drawdown": 0.2,
            "predicted_trades": 3,
        }]
        pipeline.write_run_summaries(
            args,
            [row(1600000000000)],
            ["ret_1m"],
            "lightgbm",
            {"params": {}, "best_iteration": None},
            fixed_record,
            walk_records,
        )
        with open(args.run_summary_out, encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertIn("walk_forward_summary", summary)
        self.assertIn("accepted", summary["walk_forward_summary"])
        self.assertIn("active_fold_rate", summary["walk_forward_summary"])
        self.assertIn("strategy_strength", summary["walk_forward_summary"])
        self.assertIsInstance(summary["args"]["_explicit_flags"], list)
        self.assertIn("normalized_microsecond_open_times", summary)
        self.assertIn("max_rss_stage", summary)
        self.assertEqual(summary["ev_upside_target_source"], "manifest")
        self.assertAlmostEqual(summary["effective_upside_target"], 0.02)
        self.assertTrue(summary["market_regime_features"])
        with open(args.experiment_summary_out, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn("accepted", rows[0])
        self.assertIn("rejection_reason", rows[0])
        self.assertIn("active_fold_rate", rows[0])
        self.assertIn("strategy_strength", rows[0])
        self.assertIn("normalized_microsecond_open_times", rows[0])
        self.assertIn("ev_upside_target_source", rows[0])
        self.assertIn("effective_upside_target", rows[0])
        self.assertIn("max_rss_stage", rows[0])
        self.assertIn("selected_score_name", rows[0])
        self.assertIn("selected_score_threshold", rows[0])
        self.assertIn("hybrid_return_combination", rows[0])
        self.assertIn("hybrid_min_probability", rows[0])
        self.assertIn("conditional_payoff_source", rows[0])


if __name__ == "__main__":
    unittest.main()
