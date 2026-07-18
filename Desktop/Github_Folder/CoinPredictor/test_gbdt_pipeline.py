#!/usr/bin/env python3
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock
from types import SimpleNamespace

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

import gbdt_pipeline as pipeline
import candidate_artifacts
import created_file_inventory
import merge_shard_dataset_cache as merge_shard_dataset_cache
import offline_candidate_research
import portfolio_ledger
import research_diagnostics
import run_experiments
from tools import verify_cpp_build


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


def write_text_if_changed(path, content):
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as handle:
            if handle.read() == content:
                return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def write_gzip_text_if_changed(path, content):
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            if handle.read() == content:
                return
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(content)


def shard_manifest_content(dataset_manifest, symbol, month, row_count, csv_path=None, compression="none"):
    shard_manifest = dict(dataset_manifest)
    shard_manifest.pop("shards", None)
    shard_manifest.update({
        "version": 1,
        "kind": "symbol_month_shard",
        "symbol": symbol,
        "month": month,
        "row_count": int(row_count),
    })
    if csv_path is not None:
        shard_manifest["csv_path"] = csv_path
    if compression:
        shard_manifest["compression"] = compression
    return json.dumps(shard_manifest, indent=2, sort_keys=True)


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
    dataset_manifest_content = json.dumps(manifest, indent=2, sort_keys=True)
    write_text_if_changed(
        os.path.join(dataset_dir, "kline_growth_dataset.meta.json"),
        dataset_manifest_content,
    )
    for shard in shards:
        symbol = shard["symbol"]
        month = shard["month"]
        rows = shard["rows"]
        symbol_dir = os.path.join(dataset_dir, "shards", symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        csv_name = "{}{}".format(month, ".csv.gz" if compression == "gzip" else ".csv")
        csv_path = os.path.join(symbol_dir, csv_name)
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(HEADER)
        for item in rows:
            writer.writerow(item)
        csv_content = csv_buffer.getvalue()
        if compression == "gzip":
            write_gzip_text_if_changed(csv_path, csv_content)
        else:
            write_text_if_changed(csv_path, csv_content)
        write_text_if_changed(
            os.path.join(symbol_dir, "{}.meta.json".format(month)),
            shard_manifest_content(
                manifest,
                symbol,
                month,
                len(rows),
                csv_path="shards/{}/{}".format(symbol, csv_name),
                compression=compression,
            ),
        )


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


def future_path_row(symbol, open_time, future_returns, label=1, quote_volume=1000000.0, trade_return=None):
    realized_trade_return = float(future_returns[-1]) if trade_return is None else float(trade_return)
    return SimpleNamespace(
        symbol=symbol,
        month="2020-01",
        month_index=0,
        open_time=open_time,
        label=label,
        forward_return=realized_trade_return,
        trade_return=realized_trade_return,
        max_future_high_return=max(max(float(value) for value in future_returns), 0.0),
        max_future_low_return=min(min(float(value) for value in future_returns), 0.0),
        quote_volume=quote_volume,
        features=[0.0, 0.0],
        feature_lookup={},
        future_candle_returns=[float(value) for value in future_returns],
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

    def test_discover_sharded_dataset_shards_supports_external_meta_path_inventory(self):
        source_dir = os.path.join(self.temp.name, "source_dataset")
        write_sharded_dataset(source_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        destination_dir = os.path.join(self.temp.name, "destination_dataset")
        os.makedirs(os.path.join(destination_dir, "shards"), exist_ok=True)
        with open(os.path.join(source_dir, "kline_growth_dataset.meta.json"), encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        source_csv = os.path.join(source_dir, "shards", "AAAUSDT", "2020-01.csv")
        source_meta = os.path.join(source_dir, "shards", "AAAUSDT", "2020-01.meta.json")
        destination_manifest = dict(source_manifest)
        destination_manifest["shards"] = [{
            "symbol": "AAAUSDT",
            "month": "2020-01",
            "csv_path": os.path.relpath(source_csv, destination_dir),
            "meta_path": os.path.relpath(source_meta, destination_dir),
            "compression": "none",
            "row_count": 1,
        }]
        with open(os.path.join(destination_dir, "kline_growth_dataset.meta.json"), "w", encoding="utf-8") as handle:
            json.dump(destination_manifest, handle, indent=2, sort_keys=True)
        shards = pipeline.discover_sharded_dataset_shards(destination_dir, destination_manifest)
        self.assertEqual(len(shards), 1)
        self.assertEqual(os.path.abspath(source_csv), shards[0]["csv_path"])
        self.assertEqual(os.path.abspath(source_meta), shards[0]["meta_path"])

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed cache validation")
    def test_cache_manifest_matches_allows_equivalent_manifest_signature_with_different_manifest_path(self):
        csv_path = os.path.join(self.temp.name, "synthetic.csv")
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("symbol,month,month_index,open_time,label,forward_return,trade_return,max_future_high_return,max_future_low_return,quote_volume,log_quote_volume,ret_1m\n")
        manifest_path_one = os.path.join(self.temp.name, "first.meta.json")
        manifest_path_two = os.path.join(self.temp.name, "second.meta.json")
        manifest = {"version": 1, "label_mode": "target_stop", "target_exit_mode": "first_decline"}
        for path in (manifest_path_one, manifest_path_two):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
        paths = pipeline.cache_paths(csv_path, self.temp.name, pipeline.np.float32)
        os.makedirs(os.path.dirname(paths["features"]), exist_ok=True)
        with open(paths["features"], "wb") as handle:
            handle.write(b"")
        for metadata_path in paths["metadata_arrays"].values():
            with open(metadata_path, "wb") as handle:
                handle.write(b"")
        info = pipeline.source_csv_info(
            csv_path,
            manifest,
            manifest_path_one,
            manifest_signature_override="same-signature",
        )
        cache_manifest = dict(info)
        cache_manifest.update({
            "version": pipeline.CACHE_VERSION,
            "feature_dtype": "float32",
            "feature_columns": ["ret_1m"],
            "row_count": 0,
            "training_manifest_path": manifest_path_two,
        })
        self.assertTrue(
            pipeline.cache_manifest_matches(
                cache_manifest,
                csv_path,
                pipeline.np.float32,
                paths,
                manifest,
                manifest_path_one,
                manifest_signature_override="same-signature",
            )
        )

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
    def test_sharded_dataset_cache_hit_survives_missing_external_source_shards(self):
        source_dir = os.path.join(self.temp.name, "source_dataset")
        write_sharded_dataset(source_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        destination_dir = os.path.join(self.temp.name, "destination_dataset")
        os.makedirs(os.path.join(destination_dir, "shards"), exist_ok=True)
        with open(os.path.join(source_dir, "kline_growth_dataset.meta.json"), encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        source_csv = os.path.join(source_dir, "shards", "AAAUSDT", "2020-01.csv")
        source_meta = os.path.join(source_dir, "shards", "AAAUSDT", "2020-01.meta.json")
        destination_manifest = dict(source_manifest)
        destination_manifest["shards"] = [{
            "symbol": "AAAUSDT",
            "month": "2020-01",
            "csv_path": os.path.relpath(source_csv, destination_dir),
            "meta_path": os.path.relpath(source_meta, destination_dir),
            "compression": "none",
            "row_count": 1,
        }]
        with open(os.path.join(destination_dir, "kline_growth_dataset.meta.json"), "w", encoding="utf-8") as handle:
            json.dump(destination_manifest, handle, indent=2, sort_keys=True)

        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(destination_dir, "memmap32", cache_dir=cache_dir)
        rows.cleanup()

        os.remove(source_meta)
        os.remove(source_csv)

        rows, _, _ = pipeline.load_rows(destination_dir, "memmap32", cache_dir=cache_dir, cache_only=True)
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "hit")
        self.assertTrue(pipeline.CACHE_LOAD_INFO.get("sharded_dataset"))
        self.assertEqual(len(rows), 1)
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_sharded_dataset_cache_hit_survives_dataset_path_change_with_same_basename(self):
        source_root = os.path.join(self.temp.name, "source_root")
        destination_root = os.path.join(self.temp.name, "destination_root")
        source_dir = os.path.join(source_root, "shard_dataset_recent")
        destination_dir = os.path.join(destination_root, "shard_dataset_recent")
        write_sharded_dataset(source_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        os.makedirs(os.path.join(destination_dir, "shards"), exist_ok=True)
        with open(os.path.join(source_dir, "kline_growth_dataset.meta.json"), encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        source_csv = os.path.join(source_dir, "shards", "AAAUSDT", "2020-01.csv")
        source_meta = os.path.join(source_dir, "shards", "AAAUSDT", "2020-01.meta.json")
        destination_manifest = dict(source_manifest)
        destination_manifest["shards"] = [{
            "symbol": "AAAUSDT",
            "month": "2020-01",
            "csv_path": os.path.relpath(source_csv, destination_dir),
            "meta_path": os.path.relpath(source_meta, destination_dir),
            "compression": "none",
            "row_count": 1,
        }]
        with open(os.path.join(destination_dir, "kline_growth_dataset.meta.json"), "w", encoding="utf-8") as handle:
            json.dump(destination_manifest, handle, indent=2, sort_keys=True)

        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(source_dir, "memmap32", cache_dir=cache_dir)
        rows.cleanup()

        os.remove(source_meta)
        os.remove(source_csv)

        rows, _, _ = pipeline.load_rows(destination_dir, "memmap32", cache_dir=cache_dir, cache_only=True)
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "hit")
        self.assertTrue(pipeline.CACHE_LOAD_INFO.get("sharded_dataset"))
        self.assertEqual(len(rows), 1)
        rows.cleanup()

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_inspect_cache_reports_loaded_compatible_sharded_paths(self):
        source_root = os.path.join(self.temp.name, "source_root")
        destination_root = os.path.join(self.temp.name, "destination_root")
        source_dir = os.path.join(source_root, "shard_dataset_recent")
        destination_dir = os.path.join(destination_root, "shard_dataset_recent")
        write_sharded_dataset(source_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        os.makedirs(os.path.join(destination_dir, "shards"), exist_ok=True)
        with open(os.path.join(source_dir, "kline_growth_dataset.meta.json"), encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        source_csv = os.path.join(source_dir, "shards", "AAAUSDT", "2020-01.csv")
        source_meta = os.path.join(source_dir, "shards", "AAAUSDT", "2020-01.meta.json")
        destination_manifest = dict(source_manifest)
        destination_manifest["shards"] = [{
            "symbol": "AAAUSDT",
            "month": "2020-01",
            "csv_path": os.path.relpath(source_csv, destination_dir),
            "meta_path": os.path.relpath(source_meta, destination_dir),
            "compression": "none",
            "row_count": 1,
        }]
        with open(os.path.join(destination_dir, "kline_growth_dataset.meta.json"), "w", encoding="utf-8") as handle:
            json.dump(destination_manifest, handle, indent=2, sort_keys=True)

        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(source_dir, "memmap32", cache_dir=cache_dir)
        rows.cleanup()
        os.remove(source_meta)
        os.remove(source_csv)
        expected_destination_paths = pipeline.sharded_dataset_cache_paths(
            destination_dir,
            cache_dir,
            pipeline.np.float32,
        )
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--input", destination_dir,
            "--feature-storage", "memmap32",
            "--cache-dir", cache_dir,
        ])

        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            status = pipeline.inspect_cache(
                args,
                *pipeline.load_training_manifest(destination_dir),
            )

        self.assertEqual(status, 0)
        text = output.getvalue()
        loaded_paths = pipeline.CACHE_LOAD_INFO["paths"]
        self.assertIn("cache_status=hit", text)
        self.assertIn("cache_file features exists=1", text)
        self.assertIn(loaded_paths["features"], text)
        self.assertNotIn(expected_destination_paths["features"], text)

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

    def test_merge_shard_dataset_cache_deduplicates_identical_symbol_month_shards(self):
        dataset_a = os.path.join(self.temp.name, "dataset_a")
        dataset_b = os.path.join(self.temp.name, "dataset_b")
        output_dir = os.path.join(self.temp.name, "merged_dataset")
        shard_rows = [
            self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
            self.shard_row("AAAUSDT", "2020-01", 0, 1600000060000, 0, -0.01, 100500.0),
        ]
        for dataset_dir in (dataset_a, dataset_b):
            write_sharded_dataset(dataset_dir, [
                {
                    "symbol": "AAAUSDT",
                    "month": "2020-01",
                    "rows": shard_rows,
                },
            ])

        result = merge_shard_dataset_cache.create_combined_dataset(
            output_dir,
            [dataset_a, dataset_b],
        )

        self.assertEqual(result["merged_shard_count"], 1)
        self.assertEqual(result["source_added_counts"], [(dataset_a, 1), (dataset_b, 0)])
        with open(os.path.join(output_dir, "kline_growth_dataset.meta.json"), encoding="utf-8") as handle:
            merged_manifest = json.load(handle)
        self.assertEqual(len(merged_manifest["shards"]), 1)

    def test_merge_shard_dataset_cache_rejects_conflicting_duplicate_symbol_month_shards(self):
        dataset_a = os.path.join(self.temp.name, "dataset_a")
        dataset_b = os.path.join(self.temp.name, "dataset_b")
        output_dir = os.path.join(self.temp.name, "merged_dataset")
        write_sharded_dataset(dataset_a, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        write_sharded_dataset(dataset_b, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 0, -0.03, 100000.0),
                ],
            },
        ])

        with self.assertRaisesRegex(ValueError, "conflicting duplicate shard"):
            merge_shard_dataset_cache.create_combined_dataset(
                output_dir,
                [dataset_a, dataset_b],
            )

    def test_merge_shard_dataset_cache_failed_manifest_update_keeps_original_manifest(self):
        destination_dir = os.path.join(self.temp.name, "destination_dataset")
        source_dir = os.path.join(self.temp.name, "source_dataset")
        write_sharded_dataset(destination_dir, [
            {
                "symbol": "AAAUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("AAAUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        write_sharded_dataset(source_dir, [
            {
                "symbol": "BBBUSDT",
                "month": "2020-01",
                "rows": [
                    self.shard_row("BBBUSDT", "2020-01", 0, 1600000000000, 1, 0.02, 100000.0),
                ],
            },
        ])
        manifest_path = os.path.join(destination_dir, "kline_growth_dataset.meta.json")
        with open(manifest_path, encoding="utf-8") as handle:
            original_manifest = json.load(handle)

        def write_partial_then_fail(payload, handle, *args, **kwargs):
            del payload, args, kwargs
            handle.write('{"partial":')
            raise RuntimeError("simulated manifest write failure")

        with mock.patch.object(merge_shard_dataset_cache.json, "dump", side_effect=write_partial_then_fail):
            with self.assertRaisesRegex(RuntimeError, "simulated manifest write failure"):
                merge_shard_dataset_cache.update_destination_manifest(destination_dir, [source_dir])

        with open(manifest_path, encoding="utf-8") as handle:
            preserved_manifest = json.load(handle)
        self.assertEqual(preserved_manifest, original_manifest)

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
        dataset_manifest = shard_dataset_manifest(feature_names=feature_names, upside_target=0.01)
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
            with open(os.path.join(symbol_dir, "2020-01.meta.json"), "w", encoding="utf-8") as handle:
                handle.write(
                    shard_manifest_content(
                        dataset_manifest,
                        symbol,
                        "2020-01",
                        2,
                    )
                )
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

    def test_invalid_open_time_fails_in_object_row_loader(self):
        path = os.path.join(self.temp.name, "invalid-open-time.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow([
                "TESTUSDT", "2020-01", 0, "not-a-time", 0,
                -0.005, -0.005, 0.01, -0.01, 100000.0, pipeline.math.log1p(100000.0), 0.0,
            ])
        write_training_manifest(path)
        with self.assertRaisesRegex(ValueError, "open_time must be numeric"):
            pipeline.load_rows(path, "float32", disable_cache=True)

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed compact cache")
    def test_invalid_open_time_fails_in_compact_loader(self):
        path = os.path.join(self.temp.name, "invalid-open-time-compact.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerow([
                "TESTUSDT", "2020-01", 0, "not-a-time", 0,
                -0.005, -0.005, 0.01, -0.01, 100000.0, pipeline.math.log1p(100000.0), 0.0,
            ])
        write_training_manifest(path)
        with self.assertRaisesRegex(ValueError, "open_time must be numeric"):
            pipeline.load_rows(
                path,
                "memmap32",
                cache_dir=os.path.join(self.temp.name, "invalid-open-time-cache"),
            )

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
        self.assertEqual(args.walk_forward_max_folds, 0)
        self.assertEqual(args.walk_validation_months, 1)
        self.assertEqual(args.walk_test_months, 1)

    def test_walk_forward_split_bounds_use_full_train_window(self):
        train_start, train_end, validation_start, validation_end, test_start, test_end = (
            pipeline.walk_forward_split_bounds(0, 6, 1)
        )
        self.assertEqual((train_start, train_end), (0, 6))
        self.assertEqual((validation_start, validation_end), (6, 7))
        self.assertEqual((test_start, test_end), (7, 8))

    def test_walk_forward_split_bounds_support_multi_month_test_windows(self):
        train_start, train_end, validation_start, validation_end, test_start, test_end = (
            pipeline.walk_forward_split_bounds(0, 24, 4, 4)
        )
        self.assertEqual((train_start, train_end), (0, 24))
        self.assertEqual((validation_start, validation_end), (24, 28))
        self.assertEqual((test_start, test_end), (28, 32))

    def test_single_month_label_uses_range_for_multi_month_windows(self):
        rows = [
            symbol_row("TESTUSDT", 1600000000000),
            pipeline.DataRow("TESTUSDT", "2020-02", 1, 1602688400000, 0, 0.01, 0.01, 0.01, 0.0, 1000000.0, [0.0, 0.0]),
        ]
        self.assertEqual(pipeline.single_month_label(rows), "2020-01..2020-02")

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

    def test_threshold_diagnostics_capture_rejection_reasons(self):
        base = 1600000000000
        rows = [row(base + index * 60000, label=1 if index < 2 else 0) for index in range(4)]
        selection = pipeline.tune_threshold(
            rows,
            [0.9, 0.8, 0.7, 0.6],
            [0.5, 0.75],
            "precision",
            0.0,
            0.0,
            1,
            1,
            0.75,
            "explore",
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
        )
        metrics = selection["validation_metrics"]
        self.assertEqual(metrics["candidate_threshold_count"], 2)
        self.assertEqual(metrics["rejected_over_max_trades_count"], 2)
        self.assertEqual(metrics["rejected_under_min_precision_count"], 1)
        self.assertEqual(metrics["admissible_candidate_count"], 0)
        self.assertEqual(metrics["selected_threshold_tie_rank_reason"], "no_trade_fallback")

    def test_threshold_diagnostics_summarize_avg_net_near_misses(self):
        fixed_record = {
            "threshold_rejection_diagnostics": [
                {
                    "threshold": 0.10,
                    "predicted_trades": 1,
                    "average_net_return_after_costs": 0.010,
                    "executed_score_top_1pct_avg_net_return": 0.010,
                    "executed_score_top_decile_avg_net_return": 0.010,
                    "rejected_under_min_trades": 1,
                    "rejected_under_min_avg_net_return": 1,
                },
                {
                    "threshold": 0.20,
                    "predicted_trades": 39,
                    "raw_signal_share": 0.004,
                    "average_net_return_after_costs": 0.00047,
                    "executed_score_top_1pct_avg_net_return": 0.00086,
                    "executed_score_top_decile_avg_net_return": 0.00122,
                    "symbol_profit_concentration_top1": 0.31,
                    "symbol_profit_concentration_top3": 0.71,
                    "symbol_trade_concentration_top1": 0.10,
                    "rejected_under_min_avg_net_return": 1,
                },
                {
                    "threshold": 0.30,
                    "predicted_trades": 42,
                    "average_net_return_after_costs": 0.00080,
                    "executed_score_top_1pct_avg_net_return": 0.00100,
                    "executed_score_top_decile_avg_net_return": 0.00200,
                    "symbol_profit_concentration_top1": 0.90,
                    "rejected_over_top1_concentration": 1,
                },
            ],
        }
        _, summary = pipeline.threshold_diagnostics_report_from_records(fixed_record, [])
        self.assertAlmostEqual(summary["threshold_diagnostics_best_avg_net_return"], 0.010)
        self.assertEqual(summary["threshold_diagnostics_best_avg_net_return_trades"], 1)
        self.assertEqual(summary["threshold_diagnostics_near_miss_count"], 1)
        self.assertEqual(summary["threshold_diagnostics_best_near_miss_source_split"], "fixed")
        self.assertEqual(summary["threshold_diagnostics_best_near_miss_fold_index"], 0)
        self.assertAlmostEqual(summary["threshold_diagnostics_best_near_miss_threshold"], 0.20)
        self.assertEqual(summary["threshold_diagnostics_best_near_miss_trades"], 39)
        self.assertAlmostEqual(summary["threshold_diagnostics_best_near_miss_avg_net_return"], 0.00047)
        self.assertAlmostEqual(summary["threshold_diagnostics_best_near_miss_top_decile_net_return"], 0.00122)
        self.assertAlmostEqual(summary["threshold_diagnostics_best_near_miss_top1_concentration"], 0.31)
        self.assertEqual(
            summary["threshold_diagnostics_best_near_miss_rejection_flags"],
            "rejected_under_min_avg_net_return",
        )

    def test_trade_regime_filter_reduces_raw_signals_and_execution_candidates(self):
        feature_lookup = {
            "btc_return_240m": 0,
            "market_average_return_60m": 1,
            "market_breadth_up_60m": 2,
        }

        def regime_row(open_time, btc_return, market_return, trade_return):
            return pipeline.DataRow(
                "TESTUSDT",
                "2020-01",
                0,
                open_time,
                1 if trade_return > 0.0 else 0,
                trade_return,
                trade_return,
                max(0.0, trade_return),
                min(0.0, trade_return),
                1000000.0,
                [btc_return, market_return, 0.6],
                feature_lookup,
            )

        rows = [
            regime_row(1600000000000, 0.01, 0.01, 0.01),
            regime_row(1600000060000, -0.01, -0.01, -0.01),
            regime_row(1600000120000, -0.01, 0.02, 0.02),
        ]
        runtime_args = SimpleNamespace(
            trade_regime_filter="market_positive",
            trade_regime_breadth_threshold=0.5,
        )
        raw_metrics = pipeline.raw_classification_metrics(
            rows,
            [0.9, 0.9, 0.9],
            0.5,
            hybrid_runtime_args=runtime_args,
        )
        self.assertEqual(raw_metrics["raw_signal_trades_before_regime_filter"], 3)
        self.assertEqual(raw_metrics["raw_signal_regime_filter_blocked"], 1)
        self.assertEqual(raw_metrics["raw_signal_trades"], 2)
        metrics = pipeline.evaluate(
            rows,
            [0.9, 0.9, 0.9],
            0.5,
            0.0,
            0.0,
            compute_auc=False,
            trade_selection="threshold",
            top_k_per_minute=3,
            hybrid_runtime_args=runtime_args,
        )
        self.assertEqual(metrics["predicted_trades"], 2)
        self.assertEqual(metrics["regime_filter_blocked"], 1)
        self.assertEqual(metrics["raw_signal_trades_before_regime_filter"], 3)
        self.assertEqual(metrics["raw_signal_regime_filter_blocked"], 1)
        self.assertEqual(metrics["raw_signal_trades"], 2)

    def test_threshold_diagnostics_carry_regime_filter_counts(self):
        fixed_record = {
            "threshold_rejection_diagnostics": [
                {
                    "threshold": 0.42,
                    "predicted_trades": 2,
                    "raw_signal_trades": 2,
                    "raw_signal_trades_before_regime_filter": 5,
                    "raw_signal_regime_filter_blocked": 3,
                    "raw_signal_share": 0.02,
                    "trade_regime_filter": "market_positive",
                },
            ],
        }
        rows, _ = pipeline.threshold_diagnostics_report_from_records(fixed_record, [])
        self.assertEqual(rows[0]["raw_signal_trades"], 2)
        self.assertEqual(rows[0]["raw_signal_trades_before_regime_filter"], 5)
        self.assertEqual(rows[0]["raw_signal_regime_filter_blocked"], 3)
        self.assertEqual(rows[0]["trade_regime_filter"], "market_positive")

    def test_threshold_diagnostics_carry_candidate_utility_fields(self):
        fixed_record = {
            "threshold_rejection_diagnostics": [
                {
                    "threshold": 0.30,
                    "predicted_trades": 1,
                    "candidate_utility_count": 4,
                    "candidate_utility_diagnostic_rows": 4,
                    "candidate_positive_utility_count": 2,
                    "candidate_positive_utility_recall": 0.5,
                    "candidate_missed_best_net_utility": 0.021,
                    "candidate_high_score_loss_share": 0.25,
                    "candidate_top_decile_net_utility": 0.003,
                    "candidate_score_positive_utility_gap": -0.12,
                },
                {
                    "threshold": 0.40,
                    "predicted_trades": 1,
                    "candidate_utility_count": 3,
                    "candidate_utility_diagnostic_rows": 3,
                    "candidate_positive_utility_count": 1,
                    "candidate_positive_utility_recall": 1.0,
                    "candidate_missed_best_net_utility": 0.0,
                    "candidate_high_score_loss_share": 1.0,
                    "candidate_top_decile_net_utility": -0.010,
                    "candidate_utility_diagnostics_skipped": 1,
                },
            ],
        }
        rows, summary = pipeline.threshold_diagnostics_report_from_records(fixed_record, [])
        self.assertEqual(rows[0]["candidate_utility_count"], 4)
        self.assertAlmostEqual(rows[0]["candidate_positive_utility_recall"], 0.5)
        self.assertAlmostEqual(rows[0]["candidate_score_positive_utility_gap"], -0.12)
        self.assertAlmostEqual(summary["threshold_diagnostics_best_candidate_positive_utility_recall"], 1.0)
        self.assertAlmostEqual(summary["threshold_diagnostics_max_candidate_missed_best_net_utility"], 0.021)
        self.assertAlmostEqual(summary["threshold_diagnostics_max_candidate_high_score_loss_share"], 1.0)
        self.assertAlmostEqual(summary["threshold_diagnostics_best_candidate_top_decile_net_utility"], 0.003)
        self.assertEqual(summary["threshold_diagnostics_candidate_utility_skipped_count"], 1)

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
        self.assertEqual(selection["threshold_selection_status"], "no_valid_threshold")
        self.assertEqual(selection["threshold_is_valid"], 0)
        self.assertEqual(selection["threshold_fallback_reason"], "no_valid_threshold")
        self.assertEqual(metrics["threshold_selection_status"], "no_valid_threshold")
        self.assertEqual(metrics["threshold_is_valid"], 0)
        self.assertEqual(metrics["threshold_fallback_reason"], "no_valid_threshold")
        self.assertEqual(metrics["execution_threshold"], 1.01)
        self.assertEqual(metrics["selected_validation_trade_count"], 0)
        self.assertEqual(metrics["predicted_trades"], 0)
        self.assertFalse(pipeline.math.isfinite(metrics["selected_objective_score"]))

    def test_select_threshold_for_bundle_retries_hybrid_expected_return_after_no_trade_fallback(self):
        rows = [row(1600000000000, trade_return=0.01, quote_volume=1000000000.0)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.6],
            calibrated_probability=[0.6],
            predicted_trade_return=[0.02],
        )
        args = SimpleNamespace(
            threshold_objective="avg_profit",
            fee=0.0,
            slippage=0.0,
            validation_slippage_multiplier=1.0,
            min_validation_trades=1,
            max_validation_trades=0,
            min_validation_precision=0.0,
            profit_safety="explore",
            initial_capital=10000.0,
            max_position_fraction=0.10,
            max_volume_fraction=0.01,
            max_trades_per_period=10,
            trade_period_minutes=60,
            holding_period_minutes=5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            upside_target=0.05,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="hybrid",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.0,
            max_trades_per_day=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            threshold_drawdown_penalty=0.0,
            threshold_trade_count_penalty=0.0,
            target_validation_trades=0,
            threshold_tiebreaker="fewer_trades",
            threshold_tie_epsilon=1e-9,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            hybrid_return_combination="expected_return",
            hybrid_min_probability=0.5,
        )
        bundle["hybrid_return_context"] = pipeline.fit_hybrid_return_context(rows, bundle, args)
        initial_selection = {
            "threshold": 1.01,
            "objective_score": -float("inf"),
            "penalized_objective_score": -float("inf"),
            "tie_rank_reason": "no_trade_fallback",
            "validation_metrics": {
                "predicted_trades": 0,
                "portfolio_profit": 0.0,
                "precision": 0.0,
                "active_days": 0,
                "selected_threshold_tie_rank_reason": "no_trade_fallback",
            },
        }
        retry_selection = {
            "threshold": 0.01,
            "objective_score": 0.25,
            "penalized_objective_score": 0.25,
            "tie_rank_reason": "higher_objective_score",
            "validation_metrics": {
                "predicted_trades": 3,
                "portfolio_profit": 0.75,
                "precision": 0.5,
                "active_days": 1,
                "selected_threshold_tie_rank_reason": "higher_objective_score",
            },
        }
        with mock.patch.object(
            pipeline,
            "selected_thresholds_for_bundle",
            return_value=([0.1], None, None),
        ), mock.patch.object(
            pipeline,
            "tune_threshold",
            side_effect=[initial_selection, retry_selection],
        ) as mocked_tune:
            selection = pipeline.select_threshold_for_bundle(rows, bundle, args, "hybrid")
        self.assertEqual(selection["validation_metrics"]["predicted_trades"], 3)
        self.assertTrue(selection["hybrid_return_selection_fallback_used"])
        self.assertEqual(bundle["hybrid_return_context"]["hybrid_return_combination"], "probability_times_return")
        self.assertEqual(bundle["hybrid_return_context"]["hybrid_return_combination_fallback_from"], "expected_return")
        self.assertEqual(selection["validation_metrics"]["hybrid_return_combination_requested"], "expected_return")
        self.assertEqual(selection["validation_metrics"]["hybrid_return_combination_selected"], "probability_times_return")
        self.assertEqual(mocked_tune.call_count, 2)
        self.assertEqual(mocked_tune.call_args_list[1].args[-1].hybrid_return_combination, "probability_times_return")

    def test_select_threshold_for_bundle_keeps_expected_return_when_retry_does_not_improve(self):
        rows = [row(1600000000000, trade_return=0.01, quote_volume=1000000000.0)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.6],
            calibrated_probability=[0.6],
            predicted_trade_return=[0.02],
        )
        args = SimpleNamespace(
            threshold_objective="avg_profit",
            fee=0.0,
            slippage=0.0,
            validation_slippage_multiplier=1.0,
            min_validation_trades=1,
            max_validation_trades=0,
            min_validation_precision=0.0,
            profit_safety="explore",
            initial_capital=10000.0,
            max_position_fraction=0.10,
            max_volume_fraction=0.01,
            max_trades_per_period=10,
            trade_period_minutes=60,
            holding_period_minutes=5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            upside_target=0.05,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="hybrid",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.0,
            max_trades_per_day=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            threshold_drawdown_penalty=0.0,
            threshold_trade_count_penalty=0.0,
            target_validation_trades=0,
            threshold_tiebreaker="fewer_trades",
            threshold_tie_epsilon=1e-9,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            hybrid_return_combination="expected_return",
            hybrid_min_probability=0.5,
        )
        bundle["hybrid_return_context"] = pipeline.fit_hybrid_return_context(rows, bundle, args)
        original_context = dict(bundle["hybrid_return_context"])
        initial_selection = {
            "threshold": 1.01,
            "objective_score": -float("inf"),
            "penalized_objective_score": -float("inf"),
            "tie_rank_reason": "no_trade_fallback",
            "validation_metrics": {
                "predicted_trades": 0,
                "portfolio_profit": 0.0,
                "precision": 0.0,
                "active_days": 0,
                "selected_threshold_tie_rank_reason": "no_trade_fallback",
            },
        }
        retry_selection = {
            "threshold": 1.01,
            "objective_score": -float("inf"),
            "penalized_objective_score": -float("inf"),
            "tie_rank_reason": "no_trade_fallback",
            "validation_metrics": {
                "predicted_trades": 0,
                "portfolio_profit": 0.0,
                "precision": 0.0,
                "active_days": 0,
                "selected_threshold_tie_rank_reason": "no_trade_fallback",
            },
        }
        with mock.patch.object(
            pipeline,
            "selected_thresholds_for_bundle",
            return_value=([0.1], None, None),
        ), mock.patch.object(
            pipeline,
            "tune_threshold",
            side_effect=[initial_selection, retry_selection],
        ):
            selection = pipeline.select_threshold_for_bundle(rows, bundle, args, "hybrid")
        self.assertIs(selection, initial_selection)
        self.assertEqual(bundle["hybrid_return_context"], original_context)

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

    def test_burst_penalty_can_change_selected_candidate(self):
        calm_metrics = {
            "predicted_trades": 20,
            "portfolio_profit": 40.0,
            "max_capital_drawdown": 0.05,
            "precision": 0.45,
            "raw_signal_trades": 20,
            "portfolio_return": 0.004,
            "recall": 0.2,
            "average_profit_after_fee_and_slippage": 2.0,
            "total_profit_after_fee_and_slippage": 40.0,
            "trades_per_day": 2.0,
            "max_trades_in_any_day": 3,
        }
        bursty_metrics = {
            "predicted_trades": 60,
            "portfolio_profit": 60.0,
            "max_capital_drawdown": 0.05,
            "precision": 0.45,
            "raw_signal_trades": 60,
            "portfolio_return": 0.006,
            "recall": 0.25,
            "average_profit_after_fee_and_slippage": 1.0,
            "total_profit_after_fee_and_slippage": 60.0,
            "trades_per_day": 12.0,
            "max_trades_in_any_day": 20,
        }
        calm_result = pipeline.build_selected_threshold_result(
            0.001, dict(calm_metrics), "profit", -float("inf"), 0.0, 0.0, 0, "hybrid",
            burst_trades_per_day_penalty=0.1,
            burst_max_trades_in_day_penalty=0.05,
            target_trades_per_day=4.0,
            target_max_trades_in_day=6,
        )
        bursty_result = pipeline.build_selected_threshold_result(
            0.002, dict(bursty_metrics), "profit", -float("inf"), 0.0, 0.0, 0, "hybrid",
            burst_trades_per_day_penalty=0.1,
            burst_max_trades_in_day_penalty=0.05,
            target_trades_per_day=4.0,
            target_max_trades_in_day=6,
        )
        self.assertGreater(bursty_result["base_objective_score"], calm_result["base_objective_score"])
        self.assertGreater(calm_result["penalized_objective_score"], bursty_result["penalized_objective_score"])

    def test_short_history_metrics_and_penalty_are_recorded(self):
        base = 1600000000000
        rows = [
            symbol_row("OLDUSDT", base, label=1, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("OLDUSDT", base + 24 * 60 * 60 * 1000, label=1, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("NEWUSDT", base + 24 * 60 * 60 * 1000, label=1, trade_return=0.01, quote_volume=1000000000.0),
        ]
        metrics = pipeline.evaluate(
            rows,
            [0.9, 0.85, 0.8],
            0.5,
            0.0,
            0.0,
            compute_auc=False,
            initial_capital=10000.0,
            max_position_fraction=0.10,
            max_volume_fraction=0.01,
            max_trades_per_period=10,
            trade_period_minutes=60,
            holding_period_minutes=5,
            hybrid_runtime_args=SimpleNamespace(threshold_short_history_days=0.5),
        )
        self.assertAlmostEqual(metrics["short_history_symbol_trade_share"], 2.0 / 3.0)
        penalized = pipeline.build_selected_threshold_result(
            0.5, dict(metrics), "avg_profit", -float("inf"), 0.0, 0.0, 0, "probability",
            short_history_penalty=1.0,
        )
        self.assertLess(penalized["penalized_objective_score"], penalized["base_objective_score"])

    def test_normalize_open_time_ms(self):
        self.assertEqual(pipeline.normalize_open_time_ms(1600000000000), 1600000000000)
        self.assertEqual(pipeline.normalize_open_time_ms(1600000000000000), 1600000000000)

    def test_cpp_train_help_succeeds_and_invalid_option_fails(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("C++ compiler not available")
        output_name = "coin_predictor_test.exe" if os.name == "nt" else "coin_predictor_test"
        output_path = os.path.join(self.temp.name, output_name)
        compile_command = [
            compiler,
            "-std=c++11",
            os.path.join(TEST_DIR, "main.cpp"),
            os.path.join(TEST_DIR, "DataScraper.cpp"),
            "-o",
            output_path,
            "-lz",
        ]
        subprocess.run(compile_command, cwd=TEST_DIR, check=True, capture_output=True, text=True)

        help_result = subprocess.run(
            [output_path, "train", "--help"],
            cwd=self.temp.name,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("Usage: coin_predictor train", help_result.stdout)

        invalid_result = subprocess.run(
            [output_path, "train", "--definitely-invalid-option"],
            cwd=self.temp.name,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(invalid_result.returncode, 0)
        self.assertIn("Unknown option", invalid_result.stderr)

    def test_cpp_self_test_validates_labels_and_portfolio_drawdown(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("C++ compiler not available")
        output_name = "coin_predictor_self_test.exe" if os.name == "nt" else "coin_predictor_self_test"
        output_path = os.path.join(self.temp.name, output_name)
        compile_command = [
            compiler,
            "-std=c++11",
            os.path.join(TEST_DIR, "main.cpp"),
            os.path.join(TEST_DIR, "DataScraper.cpp"),
            "-o",
            output_path,
            "-lz",
        ]
        subprocess.run(compile_command, cwd=TEST_DIR, check=True, capture_output=True, text=True)

        result = subprocess.run(
            [output_path, "train", "--self-test"],
            cwd=self.temp.name,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("C++ offline self-tests passed", result.stdout)

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

    def test_execution_cost_drag_and_score_edge_diagnostics(self):
        base = 1600000000000
        rows = [
            row(base, label=1, quote_volume=1000000000.0, trade_return=0.010),
            row(base + 60000, label=0, quote_volume=1000000000.0, trade_return=-0.005),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.90, 0.80],
            0.5,
            0.001,
            0.0005,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
        )
        self.assertAlmostEqual(execution["average_gross_return_before_costs"], 0.0025)
        self.assertAlmostEqual(execution["average_net_return_after_costs"], 0.0010)
        self.assertAlmostEqual(execution["average_cost_drag_per_trade"], 0.0015)
        self.assertAlmostEqual(execution["total_cost_drag"], 3.0)
        self.assertEqual(execution["executed_winning_trade_count"], 1)
        self.assertEqual(execution["executed_losing_trade_count"], 1)
        self.assertAlmostEqual(execution["executed_winner_score_avg"], 0.90)
        self.assertAlmostEqual(execution["executed_loser_score_avg"], 0.80)
        self.assertAlmostEqual(execution["executed_score_win_loss_gap"], 0.10)

        metrics = pipeline.evaluate(
            rows,
            [0.90, 0.80],
            0.5,
            0.001,
            0.0005,
            compute_auc=False,
        )
        self.assertAlmostEqual(metrics["average_net_return_after_costs"], 0.0010)
        self.assertAlmostEqual(metrics["executed_score_win_loss_gap"], 0.10)

    def test_execution_reports_trade_weighted_edge_when_position_sizes_differ(self):
        base = 1600000000000
        rows = [
            row(base, label=1, quote_volume=10000.0, trade_return=0.10),
            row(base + 60000, label=0, quote_volume=1000000000.0, trade_return=-0.02),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.90, 0.80],
            0.5,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            0,
            60,
            5,
        )
        self.assertAlmostEqual(execution["average_gross_return_before_costs"], 0.04)
        self.assertAlmostEqual(execution["trade_weighted_average_gross_return_before_costs"], -10.0 / 1100.0)
        self.assertAlmostEqual(execution["trade_weighted_average_net_return_after_costs"], -10.0 / 1100.0)

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

        candidates_path = os.path.join(self.temp.name, "candidates.csv")
        pipeline.write_predictions(candidates_path, output_mode="candidates", **common)
        with open(candidates_path, newline="") as handle:
            candidate_rows = list(csv.DictReader(handle))
        self.assertEqual(len(candidate_rows), 1)
        self.assertEqual(candidate_rows[0]["raw_signal"], "1")
        self.assertEqual(candidate_rows[0]["predicted"], "1")

    def test_candidate_prediction_indices_exclude_non_candidates(self):
        indices = pipeline.prediction_output_indices(
            "candidates",
            5,
            raw_selected={1: 1, 3: 1},
            executed={3: 100.0},
            executed_selected_by_topk={2: 1, 4: 0},
        )
        self.assertEqual(indices, [1, 2, 3])

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

    def test_recent_only_calibration_uses_tail_rows(self):
        rows = [row(1600000000000 + index * 60000, label=0 if index < 3 else 1) for index in range(6)]
        probabilities = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        args = SimpleNamespace(
            calibration="platt",
            calibration_max_rows=0,
            calibration_window_mode="recent",
            calibration_recent_ratio=0.5,
            calibration_recent_rows=0,
        )
        calibration = pipeline.fit_calibration(probabilities, rows, args)
        self.assertEqual(calibration["rows"], 3)
        self.assertEqual(calibration["window_mode"], "recent")

    def test_recent_calibration_subset_uses_latest_timestamps_not_row_order(self):
        base = 1600000000000
        rows = [
            row(base + 10 * 60000),
            row(base + 40 * 60000),
            row(base + 20 * 60000),
            row(base + 30 * 60000),
        ]
        probabilities = [0.10, 0.40, 0.20, 0.30]
        args = SimpleNamespace(
            calibration_window_mode="recent",
            calibration_recent_ratio=0.5,
            calibration_recent_rows=0,
        )
        subset_probabilities, subset_rows = pipeline.recent_calibration_subset(probabilities, rows, args)

        self.assertEqual(subset_probabilities, [0.40, 0.30])
        self.assertEqual([item.open_time for item in subset_rows], [base + 40 * 60000, base + 30 * 60000])

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

    def test_topk_symbol_minute_cap_diversifies_ranked_selection(self):
        base = 1600000000000
        rows = [
            symbol_row("AAVEUSDT", base, quote_volume=1000000000.0, trade_return=0.03),
            symbol_row("AAVEUSDT", base, quote_volume=1000000000.0, trade_return=0.02),
            symbol_row("AAVEUSDT", base, quote_volume=1000000000.0, trade_return=0.01),
            symbol_row("BTCUSDT", base, quote_volume=1000000000.0, trade_return=0.015),
        ]
        args = SimpleNamespace(top_k_per_symbol_minute=1)
        execution = pipeline.portfolio_execution(
            rows,
            [0.99, 0.98, 0.97, 0.96],
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
            hybrid_runtime_args=args,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 3])
        self.assertEqual(execution["symbol_minute_cap_blocked"], 2)

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

    def test_symbol_period_cap_blocks_repeat_symbol_entries(self):
        base = 1600000000000
        rows = [
            symbol_row("AAVEUSDT", base, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 30 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 61 * 60000, trade_return=0.01, quote_volume=1000000000.0),
        ]
        args = SimpleNamespace(max_trades_per_symbol_period=1)
        execution = pipeline.portfolio_execution(
            rows, [1.0, 1.0, 1.0], 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 1,
            hybrid_runtime_args=args,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 2])
        self.assertEqual(execution["symbol_trade_limit_blocked"], 1)

    def test_symbol_reentry_cooldown_blocks_nearby_repeat_entries(self):
        base = 1600000000000
        rows = [
            symbol_row("AAVEUSDT", base, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 10 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 16 * 60000, trade_return=0.01, quote_volume=1000000000.0),
        ]
        args = SimpleNamespace(symbol_reentry_cooldown_minutes=15)
        execution = pipeline.portfolio_execution(
            rows, [1.0, 1.0, 1.0], 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 1,
            hybrid_runtime_args=args,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 2])
        self.assertEqual(execution["symbol_reentry_cooldown_blocked"], 1)

    def test_same_symbol_streak_cap_blocks_repeat_entries(self):
        base = 1600000000000
        rows = [
            symbol_row("AAVEUSDT", base, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 1 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 2 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("BTCUSDT", base + 3 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 4 * 60000, trade_return=0.01, quote_volume=1000000000.0),
        ]
        args = SimpleNamespace(max_same_symbol_streak=2)
        execution = pipeline.portfolio_execution(
            rows, [1.0] * len(rows), 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 1,
            hybrid_runtime_args=args,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 1, 3, 4])
        self.assertEqual(execution["same_symbol_streak_blocked"], 1)

    def test_symbol_fold_share_cap_blocks_dominant_symbol_after_grace(self):
        base = 1600000000000
        rows = [
            symbol_row("AAVEUSDT", base, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 1 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 2 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("BTCUSDT", base + 3 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 4 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("BTCUSDT", base + 5 * 60000, trade_return=0.01, quote_volume=1000000000.0),
        ]
        args = SimpleNamespace(
            max_symbol_fold_trade_share=0.50,
            max_symbol_fold_trade_share_min_trades=2,
        )
        execution = pipeline.portfolio_execution(
            rows, [1.0] * len(rows), 0.5, 0.0, 0.0, 10000.0, 0.10, 0.01, 0, 60, 1,
            hybrid_runtime_args=args,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 1, 3, 5])
        self.assertEqual(execution["symbol_fold_share_blocked"], 2)

    def test_topk_backfills_when_concentration_cap_blocks_leader(self):
        base = 1600000000000
        rows = [
            symbol_row("AAVEUSDT", base, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 1 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("AAVEUSDT", base + 2 * 60000, trade_return=0.01, quote_volume=1000000000.0),
            symbol_row("BTCUSDT", base + 2 * 60000, trade_return=0.01, quote_volume=1000000000.0),
        ]
        args = SimpleNamespace(max_same_symbol_streak=2)
        execution = pipeline.portfolio_execution(
            rows,
            [1.0, 1.0, 1.0, 0.9],
            0.5,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            0,
            60,
            1,
            trade_selection="topk_score",
            top_k_per_minute=1,
            hybrid_runtime_args=args,
        )
        self.assertEqual(sorted(execution["executed"]), [0, 1, 3])
        self.assertEqual(execution["same_symbol_streak_blocked"], 1)
        self.assertEqual(execution["executed_selection_ranks"][3], 2)

    def test_ranker_threshold_rejects_negative_top_1pct_bucket(self):
        base = 1600000000000
        rows = []
        ranker_scores = []
        for index in range(100):
            trade_return = -0.20 if index == 0 else (0.05 if index < 10 else 0.001)
            rows.append(symbol_row(
                "SYM{:03d}USDT".format(index),
                base + index * 60000,
                label=1 if trade_return > 0.0 else 0,
                trade_return=trade_return,
                quote_volume=1000000000.0,
            ))
            ranker_scores.append(float(100 - index))
        bundle = pipeline.build_prediction_bundle(ranker_score=ranker_scores)
        args = SimpleNamespace(threshold_require_positive_top_1pct=True)
        selection = pipeline.tune_threshold(
            rows,
            bundle,
            [0.0, 50.0],
            "avg_profit",
            0.0,
            0.0,
            1,
            0,
            0.0,
            "explore",
            10000.0,
            0.01,
            1.0,
            0,
            60,
            1,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=args,
        )
        metrics = selection["validation_metrics"]
        self.assertEqual(metrics["rejected_negative_top_1pct_count"], 2)
        self.assertGreater(metrics["threshold_rejection_diagnostics"][0]["executed_score_top_decile_avg_net_return"], 0.0)
        self.assertEqual(metrics["predicted_trades"], 0)
        self.assertEqual(selection["tie_rank_reason"], "no_trade_fallback")
        self.assertEqual(selection["threshold"], pipeline.no_trade_threshold_for_mode("economic_ranking"))

    def test_ranker_threshold_rejects_overbroad_raw_signal_share(self):
        base = 1600000000000
        rows = []
        ranker_scores = []
        for index in range(100):
            rows.append(symbol_row(
                "SYM{:03d}USDT".format(index),
                base + index * 60000,
                label=1,
                trade_return=0.01,
                quote_volume=1000000000.0,
            ))
            ranker_scores.append(float(100 - index))
        bundle = pipeline.build_prediction_bundle(ranker_score=ranker_scores)
        args = SimpleNamespace(threshold_max_raw_signal_share=0.20)
        selection = pipeline.tune_threshold(
            rows,
            bundle,
            [0.0, 90.0],
            "avg_profit",
            0.0,
            0.0,
            1,
            0,
            0.0,
            "explore",
            10000.0,
            0.01,
            1.0,
            0,
            60,
            1,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=args,
        )
        metrics = selection["validation_metrics"]
        self.assertEqual(selection["threshold"], 90.0)
        self.assertEqual(metrics["rejected_over_raw_signal_share_count"], 1)
        diagnostics = {item["threshold"]: item for item in metrics["threshold_rejection_diagnostics"]}
        self.assertEqual(diagnostics[0.0]["rejected_over_raw_signal_share"], 1)
        self.assertGreater(diagnostics[0.0]["raw_signal_share"], 0.20)
        self.assertLessEqual(diagnostics[90.0]["raw_signal_share"], 0.20)

    def test_ranker_threshold_rejects_thin_average_net_return(self):
        base = 1600000000000
        rows = [
            symbol_row("AUSDT", base, label=1, trade_return=0.006, quote_volume=1000000000.0),
            symbol_row("BUSDT", base + 60000, label=0, trade_return=-0.001, quote_volume=1000000000.0),
            symbol_row("CUSDT", base + 2 * 60000, label=0, trade_return=-0.001, quote_volume=1000000000.0),
        ]
        bundle = pipeline.build_prediction_bundle(ranker_score=[0.95, 0.60, 0.55])
        args = SimpleNamespace(
            threshold_min_avg_net_return=0.003,
            threshold_min_top_decile_net_return=-999.0,
        )
        selection = pipeline.tune_threshold(
            rows,
            bundle,
            [0.50, 0.90],
            "avg_profit",
            0.001,
            0.0005,
            1,
            0,
            0.0,
            "explore",
            10000.0,
            0.10,
            1.0,
            0,
            60,
            1,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=args,
        )
        metrics = selection["validation_metrics"]
        self.assertEqual(selection["threshold"], 0.90)
        self.assertEqual(metrics["rejected_under_min_avg_net_return_count"], 1)
        diagnostics = {item["threshold"]: item for item in metrics["threshold_rejection_diagnostics"]}
        self.assertEqual(diagnostics[0.50]["rejected_under_min_avg_net_return"], 1)
        self.assertGreaterEqual(diagnostics[0.90]["average_net_return_after_costs"], 0.003)

    def test_ranker_threshold_rejects_inverted_score_gap(self):
        base = 1600000000000
        rows = [
            symbol_row("AUSDT", base, label=0, trade_return=-0.005, quote_volume=1000000000.0),
            symbol_row("BUSDT", base + 60000, label=1, trade_return=0.010, quote_volume=1000000000.0),
            symbol_row("CUSDT", base + 2 * 60000, label=1, trade_return=0.012, quote_volume=1000000000.0),
        ]
        bundle = pipeline.build_prediction_bundle(ranker_score=[0.95, 0.70, 0.69])
        args = SimpleNamespace(threshold_min_score_win_loss_gap=0.0)
        selection = pipeline.tune_threshold(
            rows,
            bundle,
            [0.60, 0.90],
            "avg_profit",
            0.0,
            0.0,
            1,
            0,
            0.0,
            "explore",
            10000.0,
            0.10,
            1.0,
            0,
            60,
            1,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=args,
        )
        metrics = selection["validation_metrics"]
        self.assertEqual(selection["threshold"], 0.90)
        self.assertEqual(metrics["rejected_under_min_score_win_loss_gap_count"], 1)
        diagnostics = {item["threshold"]: item for item in metrics["threshold_rejection_diagnostics"]}
        self.assertEqual(diagnostics[0.60]["rejected_under_min_score_win_loss_gap"], 1)
        self.assertLess(diagnostics[0.60]["executed_score_win_loss_gap"], 0.0)

    def test_candidate_rank_applies_symbol_dominance_penalty_before_selection(self):
        bucket = [
            (0, 100, 1, "BNBUSDT", 1.0, 0.01, 0.01, 1.00, 0.95, 0.0, 0.0),
            (1, 100, 1, "ETHUSDT", 1.0, 0.01, 0.01, 0.92, 0.90, 0.0, 0.0),
        ]
        runtime_args = SimpleNamespace(
            symbol_dominance_penalty_validation_weight=0.8,
            symbol_dominance_penalty_recent_weight=0.0,
            symbol_dominance_penalty_grace=0.0,
        )
        chosen = pipeline.candidate_rank(
            bucket,
            "topk_score",
            False,
            1,
            0,
            lambda *args, **kwargs: None,
            trade_period_minutes=60,
            recent_entry_minutes=[],
            recent_entry_minutes_by_symbol={},
            symbol_executed_counts={},
            fold_trade_count=0,
            validation_dominance_shares={"BNBUSDT": 0.9, "ETHUSDT": 0.1},
            runtime_args=runtime_args,
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0][3], "ETHUSDT")

    def test_candidate_rank_applies_recent_symbol_dominance_penalty_across_minutes(self):
        bucket = [
            (0, 180, 1, "BNBUSDT", 1.0, 0.01, 0.01, 1.00, 0.95, 0.0, 0.0),
            (1, 180, 1, "ETHUSDT", 1.0, 0.01, 0.01, 0.92, 0.90, 0.0, 0.0),
        ]
        runtime_args = SimpleNamespace(
            symbol_dominance_penalty_validation_weight=0.0,
            symbol_dominance_penalty_recent_weight=0.8,
            symbol_dominance_penalty_grace=0.0,
            symbol_reentry_cooldown_minutes=30,
            holding_period_minutes=5,
        )
        chosen = pipeline.candidate_rank(
            bucket,
            "topk_score",
            False,
            1,
            0,
            lambda *args, **kwargs: None,
            trade_period_minutes=60,
            recent_entry_minutes=[170, 175],
            recent_entry_minutes_by_symbol={"BNBUSDT": [170, 175]},
            symbol_executed_counts={"BNBUSDT": 2},
            fold_trade_count=2,
            validation_dominance_shares={},
            runtime_args=runtime_args,
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0][3], "ETHUSDT")

    def test_candidate_rank_prefers_unique_symbols_before_duplicate_fill(self):
        bucket = [
            (0, 100, 1, "BNBUSDT", 1.0, 0.01, 0.01, 0.99, 0.99, 0.0, 0.0),
            (1, 100, 1, "BNBUSDT", 1.0, 0.01, 0.01, 0.98, 0.98, 0.0, 0.0),
            (2, 100, 1, "ETHUSDT", 1.0, 0.01, 0.01, 0.97, 0.97, 0.0, 0.0),
        ]
        runtime_args = SimpleNamespace(prefer_unique_symbols=True)
        chosen = pipeline.candidate_rank(
            bucket,
            "topk_score",
            False,
            2,
            0,
            lambda *args, **kwargs: None,
            runtime_args=runtime_args,
        )
        self.assertEqual([item[3] for item in chosen], ["BNBUSDT", "ETHUSDT"])

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

    def test_portfolio_drawdown_is_reported_as_fraction_of_initial_capital(self):
        rows = [
            row(1600000000000, trade_return=-0.10, quote_volume=1000000000.0),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [1.0],
            0.5,
            0.0,
            0.0,
            10000.0,
            0.10,
            1.0,
            0,
            60,
            1,
        )
        self.assertAlmostEqual(execution["portfolio_profit"], -100.0)
        self.assertAlmostEqual(execution["max_capital_drawdown"], 0.01)
        self.assertAlmostEqual(execution["max_capital_drawdown_amount"], 100.0)

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

    def test_threshold_penalized_score_applies_symbol_concentration_penalty(self):
        metrics = {
            "predicted_trades": 10,
            "portfolio_profit": 5.0,
            "symbol_profit_concentration_top1": 0.8,
            "symbol_profit_concentration_top3": 0.9,
        }
        base_score, penalized_score = pipeline.threshold_penalized_score(
            metrics,
            "profit",
            top1_concentration_penalty=2.0,
            top3_concentration_penalty=1.0,
        )
        self.assertAlmostEqual(base_score, 5.0)
        self.assertAlmostEqual(penalized_score, 5.0 - 2.0 * 0.8 - 1.0 * 0.9)

    def test_threshold_penalized_score_applies_soft_cap_overflow_penalty(self):
        metrics = {
            "predicted_trades": 10,
            "portfolio_profit": 5.0,
            "symbol_trade_concentration_top1": 0.8,
        }
        base_score, penalized_score = pipeline.threshold_penalized_score(
            metrics,
            "profit",
            trade_top1_concentration_penalty=0.5,
            trade_top1_concentration_cap=0.75,
            concentration_cap_mode="soft",
        )
        self.assertAlmostEqual(base_score, 5.0)
        self.assertAlmostEqual(penalized_score, 5.0 - 0.5 * 0.8 - ((0.8 - 0.75) / (1.0 - 0.75)))

    def test_threshold_penalized_score_applies_diversity_penalty_weight(self):
        metrics = {
            "predicted_trades": 10,
            "portfolio_profit": 5.0,
            "symbol_profit_concentration_top1": 0.8,
            "symbol_trade_concentration_top1": 0.8,
            "symbol_execution_rows": [{"symbol": "BNBUSDT"}, {"symbol": "ETHUSDT"}],
        }
        base_score, penalized_score = pipeline.threshold_penalized_score(
            metrics,
            "profit",
            diversity_penalty_weight=0.01,
        )
        self.assertAlmostEqual(base_score, 5.0)
        self.assertLess(penalized_score, base_score)

    def test_threshold_penalized_score_can_blend_double_cost_objective(self):
        metrics = {
            "predicted_trades": 10,
            "portfolio_profit": 100.0,
            "precision": 0.4,
            "total_cost_drag": 120.0,
        }
        base_score, penalized_score = pipeline.threshold_penalized_score(
            metrics,
            "profit",
            threshold_cost_stress_multiplier=2.0,
            threshold_cost_stress_weight=0.5,
        )
        self.assertAlmostEqual(base_score, 100.0)
        self.assertAlmostEqual(penalized_score, 40.0)

    def test_cost_stressed_threshold_selection_prefers_cost_tolerant_candidate(self):
        fragile = pipeline.build_selected_threshold_result(
            0.001,
            {
                "predicted_trades": 20,
                "portfolio_profit": 100.0,
                "max_capital_drawdown": 0.05,
                "precision": 0.5,
                "raw_signal_trades": 20,
                "portfolio_return": 0.01,
                "recall": 0.3,
                "average_profit_after_fee_and_slippage": 5.0,
                "total_profit_after_fee_and_slippage": 100.0,
                "total_cost_drag": 150.0,
                "active_days": 2,
            },
            "profit",
            -float("inf"),
            0.0,
            0.0,
            20,
            "hybrid",
            threshold_cost_stress_multiplier=2.0,
            threshold_cost_stress_weight=1.0,
        )
        tolerant = pipeline.build_selected_threshold_result(
            0.002,
            {
                "predicted_trades": 20,
                "portfolio_profit": 70.0,
                "max_capital_drawdown": 0.05,
                "precision": 0.5,
                "raw_signal_trades": 20,
                "portfolio_return": 0.007,
                "recall": 0.3,
                "average_profit_after_fee_and_slippage": 3.5,
                "total_profit_after_fee_and_slippage": 70.0,
                "total_cost_drag": 10.0,
                "active_days": 2,
            },
            "profit",
            -float("inf"),
            0.0,
            0.0,
            20,
            "hybrid",
            threshold_cost_stress_multiplier=2.0,
            threshold_cost_stress_weight=1.0,
        )
        better, reason = pipeline.compare_threshold_results(
            tolerant,
            fragile,
            SimpleNamespace(
                threshold_tiebreaker="fewer_trades",
                threshold_tie_epsilon=1e-9,
                threshold_target_trades=20,
                threshold_target_active_days=0,
                target_validation_trades=20,
                threshold_objective="profit",
            ),
            5,
            100,
        )
        self.assertTrue(better)
        self.assertEqual(reason, "higher_objective_score")
        self.assertLess(fragile["penalized_objective_score"], tolerant["penalized_objective_score"])
        self.assertLess(fragile["validation_metrics"]["selected_validation_cost_stress_portfolio_profit"], 0.0)
        self.assertGreater(tolerant["validation_metrics"]["selected_validation_cost_stress_portfolio_profit"], 0.0)

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

    def test_selected_thresholds_for_bundle_hybrid_recent_calibration_can_lower_score_floor(self):
        rows = [row(1600000000000 + index * 60000, label=1 if index % 2 == 0 else 0, trade_return=0.02 if index % 2 == 0 else -0.01) for index in range(12)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7],
            calibrated_probability=[0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7],
            predicted_trade_return=[-0.0008, -0.0004, -0.0001, 0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0009, 0.0010, 0.0011, 0.0012],
            raw_predicted_trade_return=[-0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.010, 0.011, 0.012],
            predicted_return_uncertainty=[0.0] * 12,
        )
        bundle["regression_calibration"] = {"mode": "linear", "a": 0.03, "b": 0.0}
        args = SimpleNamespace(
            objective_mode="hybrid",
            disable_adaptive_thresholds=False,
            adaptive_threshold_sample_rows=1000000,
            fee=0.001,
            slippage=0.0005,
            validation_slippage_multiplier=1.0,
            hybrid_score_mode="risk_adjusted",
            hybrid_uncertainty_penalty=0.0,
            hybrid_min_score=0.001,
            hybrid_min_score_calibration_aware=True,
            hybrid_min_score_calibration_reference_scale=0.20,
            hybrid_min_score_calibration_min_ratio=0.25,
            hybrid_min_score_calibration_floor_min=0.00025,
            hybrid_min_score_calibration_floor_max=0.0,
            calibration_window_mode="recent",
        )
        thresholds, fixed_threshold, score_values = pipeline.selected_thresholds_for_bundle(bundle, rows, args)
        self.assertEqual(fixed_threshold, 0.00025)
        self.assertIsNotNone(score_values)
        self.assertIn(0.00025, thresholds)
        self.assertEqual(bundle["hybrid_gate_diagnostics"]["configured_hybrid_min_score"], 0.001)
        self.assertEqual(bundle["hybrid_gate_diagnostics"]["effective_hybrid_min_score"], 0.00025)

    def test_selected_thresholds_for_bundle_hybrid_recent_calibration_can_cross_below_zero(self):
        rows = [row(1600000000000 + index * 60000, label=1 if index % 2 == 0 else 0, trade_return=0.02 if index % 2 == 0 else -0.01) for index in range(12)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7],
            calibrated_probability=[0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7],
            predicted_trade_return=[-0.0008, -0.0004, -0.0001, 0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0009, 0.0010, 0.0011, 0.0012],
            raw_predicted_trade_return=[-0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.010, 0.011, 0.012],
            predicted_return_uncertainty=[0.0] * 12,
        )
        bundle["regression_calibration"] = {"mode": "linear", "a": 0.03, "b": 0.0}
        args = SimpleNamespace(
            objective_mode="hybrid",
            disable_adaptive_thresholds=False,
            adaptive_threshold_sample_rows=1000000,
            fee=0.001,
            slippage=0.0005,
            validation_slippage_multiplier=1.0,
            hybrid_score_mode="risk_adjusted",
            hybrid_uncertainty_penalty=0.0,
            hybrid_min_score=0.001,
            hybrid_min_score_calibration_aware=True,
            hybrid_min_score_calibration_reference_scale=0.20,
            hybrid_min_score_calibration_min_ratio=0.25,
            hybrid_min_score_calibration_floor_min=-0.0012,
            hybrid_min_score_calibration_floor_max=-0.0007,
            calibration_window_mode="recent",
        )
        thresholds, fixed_threshold, score_values = pipeline.selected_thresholds_for_bundle(bundle, rows, args)
        self.assertEqual(fixed_threshold, -0.0007)
        self.assertIsNotNone(score_values)
        self.assertIn(-0.0007, thresholds)
        self.assertEqual(bundle["hybrid_gate_diagnostics"]["configured_hybrid_min_score"], 0.001)
        self.assertEqual(bundle["hybrid_gate_diagnostics"]["effective_hybrid_min_score"], -0.0007)

    def test_selected_thresholds_for_bundle_topk_classification_falls_back_to_adaptive_grid(self):
        rows = [row(1600000000000 + index * 60000, label=1 if index % 2 == 0 else 0) for index in range(12)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.021, 0.024, 0.03],
            calibrated_probability=[0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.021, 0.024, 0.03],
        )
        args = SimpleNamespace(
            objective_mode="classification",
            thresholds=[0.001, 0.005, 0.01, 0.02, 0.05],
            disable_adaptive_thresholds=False,
            min_validation_trades=5,
            adaptive_threshold_sample_rows=1000000,
            min_selected_threshold=0.05,
            trade_selection="topk_score",
        )
        thresholds, fixed_threshold, score_values = pipeline.selected_thresholds_for_bundle(bundle, rows, args)
        self.assertIsNone(fixed_threshold)
        self.assertIs(score_values, bundle["calibrated_probability"])
        self.assertGreaterEqual(len(thresholds), 1)
        self.assertNotEqual(thresholds, [1.01])
        self.assertLessEqual(max(thresholds), 0.03)

    def test_selected_thresholds_for_bundle_threshold_classification_still_uses_no_trade_fallback(self):
        rows = [row(1600000000000 + index * 60000, label=1 if index % 2 == 0 else 0) for index in range(12)]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.021, 0.024, 0.03],
            calibrated_probability=[0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.021, 0.024, 0.03],
        )
        args = SimpleNamespace(
            objective_mode="classification",
            thresholds=[0.001, 0.005, 0.01, 0.02, 0.05],
            disable_adaptive_thresholds=False,
            min_validation_trades=5,
            adaptive_threshold_sample_rows=1000000,
            min_selected_threshold=0.05,
            trade_selection="threshold",
        )
        thresholds, fixed_threshold, score_values = pipeline.selected_thresholds_for_bundle(bundle, rows, args)
        self.assertIsNone(fixed_threshold)
        self.assertIs(score_values, bundle["calibrated_probability"])
        self.assertEqual(thresholds, [0.05])

    def test_adaptive_score_thresholds_respect_base_floor(self):
        thresholds = pipeline.adaptive_score_thresholds(
            [-0.02, -0.01, -0.005, 0.0, 0.0008, 0.0012, 0.004],
            0.001,
        )
        self.assertEqual(thresholds[0], 0.001)
        self.assertTrue(all(value >= 0.001 for value in thresholds))

    def test_adaptive_score_thresholds_include_target_count_cutoffs(self):
        thresholds = pipeline.adaptive_score_thresholds(
            [float(value) for value in range(100)],
            0.0,
            target_counts=[3, 10],
        )
        self.assertIn(97.0, thresholds)
        self.assertIn(90.0, thresholds)

    def test_ranker_threshold_search_uses_validation_trade_targets(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--trade-score", "ranker_score",
            "--ranker-threshold-search",
            "--min-validation-trades", "5",
            "--max-validation-trades", "10",
            "--target-validation-trades", "7",
        ])
        bundle = pipeline.build_prediction_bundle(ranker_score=[float(value) for value in range(100)])

        thresholds, fixed_threshold, score_values = pipeline.selected_thresholds_for_bundle(bundle, [], args)

        self.assertEqual(fixed_threshold, args.ranker_min_score)
        self.assertIs(score_values, bundle["ranker_score"])
        self.assertIn(95.0, thresholds)
        self.assertIn(93.0, thresholds)
        self.assertIn(90.0, thresholds)

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

    def test_symbol_filter_dominance_support_floor_blocks_thin_leader(self):
        rows = [
            symbol_row("AAVEUSDT", 1600000000000, label=1, trade_return=0.02),
            symbol_row("BTCUSDT", 1600000000000, label=1, trade_return=0.015),
            symbol_row("AAVEUSDT", 1600000060000, label=1, trade_return=0.02),
            symbol_row("BTCUSDT", 1600000060000, label=1, trade_return=0.015),
            symbol_row("AAVEUSDT", 1600000120000, label=1, trade_return=0.02),
            symbol_row("BTCUSDT", 1600000120000, label=1, trade_return=0.015),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.99, 0.70, 0.99, 0.70, 0.99, 0.70],
            calibrated_probability=[0.99, 0.70, 0.99, 0.70, 0.99, 0.70],
        )
        args = SimpleNamespace(
            symbol_validation_filter="positive_avg_profit",
            symbol_filter_stage="executed",
            symbol_filter_min_candidates=1,
            symbol_filter_min_executed=5,
            symbol_filter_candidate_weight=0.5,
            symbol_filter_executed_weight=0.5,
            symbol_filter_shrinkage=0.0,
            symbol_filter_min_active_days=2,
            symbol_filter_max_executed_trade_share=0.6,
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
            fee_mode="fixed",
            position_sizing_mode="fixed_fraction",
            min_order_notional=0.0,
            lot_size_step=0.0,
            tick_size=0.0,
            latency_penalty_bps=0.0,
            partial_fill_mode="none",
            max_open_positions=0,
            max_daily_loss_fraction=0.0,
            cooldown_after_loss_minutes=0,
        )
        symbol_filter = pipeline.fit_symbol_validation_filter(rows, bundle, 0.5, args, "probability")
        diagnostics = {item["symbol"]: item for item in symbol_filter["diagnostics"]}
        self.assertEqual(diagnostics["AAVEUSDT"]["symbol_filter_reason"], "dominance_support_floor")
        self.assertEqual(diagnostics["AAVEUSDT"]["dominance_gate_triggered"], 1)
        self.assertNotIn("AAVEUSDT", symbol_filter["allowed_symbols"])

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
                "average_gross_return_before_costs": 0.004,
                "average_net_return_after_costs": 0.002,
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
                "average_gross_return_before_costs": 0.0,
                "average_net_return_after_costs": 0.0,
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
        self.assertAlmostEqual(aggregate["average_net_return_after_costs"], 0.001)
        self.assertAlmostEqual(aggregate["active_fold_average_net_return_after_costs"], 0.002)
        self.assertAlmostEqual(aggregate["trade_weighted_average_net_return_after_costs"], 0.002)

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

    def test_experiment_runner_hybrid_late_recent_includes_calibration_aware_hybrid_floor_flags(self):
        args = run_experiments.parse_args(["--profile", "hybrid-late-recent"])
        experiment = run_experiments.build_experiment_grid_for_profile("hybrid-late-recent", False, 1)[0]
        command = run_experiments.build_command(args, experiment)
        self.assertIn("--hybrid-min-score-calibration-aware", command)
        self.assertIn("--hybrid-min-score-calibration-reference-scale", command)
        self.assertIn("--hybrid-min-score-calibration-min-ratio", command)
        self.assertIn("--hybrid-min-score-calibration-floor-min", command)

    def test_experiment_runner_summary_record_exposes_calibration_report_status(self):
        experiment = run_experiments.build_experiment_grid_for_profile("hybrid-late-recent", False, 1)[0]
        summary = {
            "fixed_metrics": {"predicted_trades": 0},
            "brier_score": 0.21,
            "expected_calibration_error": 0.08,
            "max_calibration_error": 0.17,
            "calibration_report_skipped_reason": "prediction_output_missing",
        }
        record = run_experiments.summary_record(experiment, summary, 0)
        self.assertIn("calibration_report_skipped_reason", run_experiments.RESULT_COLUMNS)
        self.assertAlmostEqual(record["brier_score"], 0.21)
        self.assertAlmostEqual(record["expected_calibration_error"], 0.08)
        self.assertAlmostEqual(record["max_calibration_error"], 0.17)
        self.assertEqual(record["calibration_report_skipped_reason"], "prediction_output_missing")

    def test_experiment_runner_summary_record_carries_execution_audit_metrics(self):
        experiment = run_experiments.build_experiment_grid_for_profile("hybrid-late-recent", False, 1)[0]
        audit_metrics = {
            key: index + 1
            for index, key in enumerate(pipeline.EXECUTION_AUDIT_METRIC_KEYS)
        }
        audit_metrics["execution_chronological_audit_status"] = "sorted_nonmonotonic_input"
        audit_metrics["portfolio_profit"] = 12.5
        audit_metrics["portfolio_return"] = 0.0125
        audit_metrics["average_profit_per_trade"] = 1.25
        summary = {
            "walk_forward_aggregate_metrics": audit_metrics,
            "walk_forward_summary": {
                "walkforward_total_predicted_trades": 10,
            },
        }
        record = run_experiments.summary_record(experiment, summary, 0)
        for key in pipeline.EXECUTION_AUDIT_METRIC_KEYS:
            self.assertIn(key, run_experiments.RESULT_COLUMNS)
            self.assertIn(key, record)
            self.assertEqual(record[key], audit_metrics[key])

    def test_experiment_runner_hybrid_late_recent_tuned_uses_start_fold_88_and_walk_months(self):
        args = run_experiments.parse_args(["--profile", "hybrid-late-recent-tuned"])
        experiment = run_experiments.build_experiment_grid_for_profile("hybrid-late-recent-tuned", False, 1)[0]
        command = run_experiments.build_command(args, experiment)
        self.assertIn("--walk-forward-start-fold", command)
        self.assertIn("88", command)
        self.assertIn("--walk-train-months", command)
        self.assertIn("6", command)
        self.assertIn("--threshold-floor-snap-penalty-weight", command)
        self.assertIn("--hybrid-return-combination", command)

    def test_experiment_runner_economic_ranker_profile_emits_ranking_flags(self):
        args = run_experiments.parse_args(["--profile", "economic-ranker"])
        experiments = run_experiments.build_experiment_grid_for_profile("economic-ranker", False, 3)
        self.assertEqual([item["trade_regime_filter"] for item in experiments], [
            "none",
            "market_positive",
            "btc_or_market_positive",
        ])
        self.assertEqual(len(set(run_experiments.experiment_name(item) for item in experiments)), 3)
        experiment = experiments[0]
        command = run_experiments.build_command(args, experiment)
        self.assertIn("--objective-mode", command)
        self.assertIn("economic_ranking", command)
        self.assertIn("--trade-score", command)
        self.assertIn("ranker_score", command)
        self.assertIn("--ranker-objective", command)
        self.assertIn("rank_xendcg", command)
        self.assertIn("--ranker-threshold-search", command)
        self.assertIn("--ranker-group-minutes", command)
        self.assertIn("--top-percent-per-period", command)
        def effective_flag_value(flag):
            positions = [index for index, value in enumerate(command) if value == flag]
            return command[positions[-1] + 1]
        self.assertEqual(effective_flag_value("--max-train-rows"), "750000")
        self.assertEqual(effective_flag_value("--max-validation-rows"), "250000")
        self.assertEqual(effective_flag_value("--max-final-train-rows"), "750000")
        self.assertEqual(effective_flag_value("--prediction-batch-rows"), "500000")
        self.assertEqual(effective_flag_value("--auc-sample-rows"), "250000")
        self.assertEqual(effective_flag_value("--adaptive-threshold-sample-rows"), "250000")
        self.assertEqual(effective_flag_value("--n-estimators"), "120")
        self.assertEqual(effective_flag_value("--model-candidate-count"), "1")
        self.assertEqual(effective_flag_value("--walk-validation-months"), "2")
        self.assertEqual(effective_flag_value("--threshold-objective"), "avg_profit")
        self.assertEqual(effective_flag_value("--top-k-per-minute"), "1")
        self.assertEqual(effective_flag_value("--max-trades-per-period"), "6")
        self.assertEqual(effective_flag_value("--max-trades-per-symbol-period"), "1")
        self.assertEqual(effective_flag_value("--symbol-reentry-cooldown-minutes"), "240")
        self.assertEqual(effective_flag_value("--max-same-symbol-streak"), "4")
        self.assertEqual(effective_flag_value("--max-symbol-fold-trade-share"), "0.3")
        self.assertEqual(effective_flag_value("--max-symbol-fold-trade-share-min-trades"), "20")
        self.assertEqual(effective_flag_value("--min-validation-trades"), "20")
        self.assertEqual(effective_flag_value("--max-validation-trades"), "120")
        self.assertEqual(effective_flag_value("--ranker-score-upper-quantile"), "0.9")
        self.assertEqual(effective_flag_value("--ranker-group-minutes"), "5")
        self.assertEqual(effective_flag_value("--ranker-adverse-penalty"), "0.1")
        self.assertEqual(effective_flag_value("--ranker-relevance-mode"), "utility_tail")
        self.assertEqual(effective_flag_value("--ranker-relevance-q1"), "0.5")
        self.assertEqual(effective_flag_value("--ranker-relevance-q2"), "0.8")
        self.assertEqual(effective_flag_value("--ranker-relevance-q3"), "0.93")
        self.assertEqual(effective_flag_value("--ranker-utility-regression-blend"), "0.65")
        self.assertEqual(effective_flag_value("--ranker-utility-regression-clip-min"), "-0.03")
        self.assertEqual(effective_flag_value("--ranker-utility-regression-clip-max"), "0.05")
        self.assertEqual(effective_flag_value("--ranker-selection-calibration"), "score_bucket")
        self.assertEqual(effective_flag_value("--ranker-selection-calibration-bins"), "20")
        self.assertEqual(effective_flag_value("--ranker-selection-calibration-min-rows"), "200")
        self.assertEqual(effective_flag_value("--ranker-selection-calibration-shrinkage"), "50.0")
        self.assertEqual(effective_flag_value("--ranker-threshold-score"), "selection")
        self.assertEqual(effective_flag_value("--trade-regime-filter"), "none")
        self.assertEqual(effective_flag_value("--trade-regime-breadth-threshold"), "0.5")
        self.assertEqual(effective_flag_value("--threshold-max-raw-signal-share"), "0.01")
        self.assertEqual(effective_flag_value("--threshold-min-avg-net-return"), "0.0")
        self.assertEqual(effective_flag_value("--threshold-min-top-decile-net-return"), "0.0")
        self.assertEqual(effective_flag_value("--threshold-min-score-win-loss-gap"), "0.0")
        self.assertEqual(effective_flag_value("--threshold-max-top1-concentration"), "0.7")
        self.assertEqual(effective_flag_value("--threshold-max-top3-concentration"), "0.9")
        self.assertEqual(effective_flag_value("--threshold-max-trade-top1-concentration"), "0.35")
        self.assertEqual(effective_flag_value("--threshold-concentration-cap-mode"), "hard")
        self.assertEqual(effective_flag_value("--threshold-profit-concentration-cap-mode"), "soft")
        self.assertEqual(effective_flag_value("--target-validation-trades"), "40")
        self.assertEqual(effective_flag_value("--min-profitable-fold-rate"), "0.5")
        self.assertEqual(effective_flag_value("--min-median-fold-return"), "0.0")
        self.assertEqual(effective_flag_value("--robustness-gate-action"), "reject")
        self.assertEqual(effective_flag_value("--robust-min-executed-score-gap"), "0.0")
        self.assertIn("--threshold-require-positive-top-1pct", command)
        self.assertIn("--robust-require-positive-top-1pct", command)
        self.assertIn("--robust-require-positive-top-5pct", command)
        self.assertIn("--robust-require-positive-top-decile", command)
        self.assertIn("--skip-full-validation-retune", command)
        self.assertIn(".gbdt_cache_full30_volatile", command)

    def test_experiment_runner_cost_aware_ranker_profile_emits_utility_target_flags(self):
        args = run_experiments.parse_args(["--profile", "economic-ranker-cost-aware"])
        experiments = run_experiments.build_experiment_grid_for_profile("economic-ranker-cost-aware", False, 3)
        self.assertEqual(len(experiments), 3)
        self.assertEqual({item["profile"] for item in experiments}, {"economic-ranker-cost-aware"})
        command = run_experiments.build_command(args, experiments[0])
        def effective_flag_value(flag):
            positions = [index for index, value in enumerate(command) if value == flag]
            return command[positions[-1] + 1]
        self.assertEqual(effective_flag_value("--ranker-utility-target-margin"), "0.001")
        self.assertEqual(effective_flag_value("--ranker-min-calibration-top-bin-utility"), "0.0")
        self.assertEqual(effective_flag_value("--trade-selection"), "top_utility")
        self.assertIn("--ranker-reject-negative-calibration-top-bin", command)
        self.assertIn("cost", run_experiments.experiment_name(experiments[0]))

    def test_experiment_runner_summary_record_carries_near_miss_diagnostics(self):
        experiment = {
            "profile": "economic-ranker",
            "objective_mode": "economic_ranking",
            "trade_score": "ranker_score",
            "trade_selection": "topk_score",
            "top_k_per_minute": 1,
            "trade_regime_filter": "market_positive",
            "trade_regime_breadth_threshold": 0.5,
        }
        summary = {
            "walk_forward_aggregate_metrics": {
                "regime_filter_blocked": 11,
                "raw_signal_trades_before_regime_filter": 30,
                "raw_signal_regime_filter_blocked": 19,
            },
            "threshold_diagnostics_near_miss_count": 2,
            "threshold_diagnostics_near_miss_ignored_flags": "rejected_under_min_avg_net_return",
            "threshold_diagnostics_best_near_miss_source_split": "walkforward_fold_8_test_months_106_106",
            "threshold_diagnostics_best_near_miss_fold_index": 8,
            "threshold_diagnostics_best_near_miss_threshold": 0.290485,
            "threshold_diagnostics_best_near_miss_trades": 39,
            "threshold_diagnostics_best_near_miss_avg_net_return": 0.0004706,
            "threshold_diagnostics_best_near_miss_top_1pct_net_return": 0.0008631,
            "threshold_diagnostics_best_near_miss_top_decile_net_return": 0.0012177,
            "threshold_diagnostics_best_near_miss_top1_concentration": 0.306,
            "threshold_diagnostics_best_near_miss_top3_concentration": 0.711,
            "threshold_diagnostics_best_near_miss_trade_top1_concentration": 0.103,
            "threshold_diagnostics_best_near_miss_rejection_flags": "rejected_under_min_avg_net_return",
        }
        record = run_experiments.summary_record(experiment, summary, 0)
        self.assertEqual(record["trade_regime_filter"], "market_positive")
        self.assertAlmostEqual(record["trade_regime_breadth_threshold"], 0.5)
        self.assertEqual(record["regime_filter_blocked"], 11)
        self.assertEqual(record["raw_signal_trades_before_regime_filter"], 30)
        self.assertEqual(record["raw_signal_regime_filter_blocked"], 19)
        self.assertEqual(record["threshold_diagnostics_near_miss_count"], 2)
        self.assertEqual(record["threshold_diagnostics_near_miss_ignored_flags"], "rejected_under_min_avg_net_return")
        self.assertEqual(record["threshold_diagnostics_best_near_miss_fold_index"], 8)
        self.assertEqual(record["threshold_diagnostics_best_near_miss_trades"], 39)
        self.assertAlmostEqual(record["threshold_diagnostics_best_near_miss_avg_net_return"], 0.0004706)
        self.assertAlmostEqual(record["threshold_diagnostics_best_near_miss_trade_top1_concentration"], 0.103)
        self.assertEqual(
            record["threshold_diagnostics_best_near_miss_rejection_flags"],
            "rejected_under_min_avg_net_return",
        )

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

    def test_parser_accepts_symbol_concentration_controls(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--max-trades-per-symbol-period", "2",
            "--top-k-per-symbol-minute", "1",
            "--prefer-unique-symbols",
            "--symbol-reentry-cooldown-minutes", "15",
            "--threshold-top1-concentration-penalty", "2.5",
            "--threshold-max-top1-concentration", "0.7",
            "--threshold-trade-top1-concentration-penalty", "1.25",
            "--threshold-max-trade-top1-concentration", "0.65",
            "--threshold-concentration-cap-mode", "hard",
            "--threshold-diversity-profit-tolerance-ratio", "0.2",
            "--symbol-dominance-penalty-validation-weight", "0.3",
            "--symbol-dominance-penalty-recent-weight", "0.4",
            "--symbol-dominance-penalty-grace", "0.25",
            "--symbol-filter-min-active-days", "3",
            "--symbol-filter-max-executed-trade-share", "0.55",
            "--prediction-bundle-cache", "disk",
            "--fast-diagnostics",
        ])
        self.assertEqual(args.max_trades_per_symbol_period, 2)
        self.assertEqual(args.top_k_per_symbol_minute, 1)
        self.assertTrue(args.prefer_unique_symbols)
        self.assertEqual(args.symbol_reentry_cooldown_minutes, 15)
        self.assertAlmostEqual(args.threshold_top1_concentration_penalty, 2.5)
        self.assertAlmostEqual(args.threshold_max_top1_concentration, 0.7)
        self.assertAlmostEqual(args.threshold_trade_top1_concentration_penalty, 1.25)
        self.assertAlmostEqual(args.threshold_max_trade_top1_concentration, 0.65)
        self.assertEqual(args.threshold_concentration_cap_mode, "hard")
        self.assertAlmostEqual(args.threshold_diversity_profit_tolerance_ratio, 0.2)
        self.assertAlmostEqual(args.symbol_dominance_penalty_validation_weight, 0.3)
        self.assertAlmostEqual(args.symbol_dominance_penalty_recent_weight, 0.4)
        self.assertAlmostEqual(args.symbol_dominance_penalty_grace, 0.25)
        self.assertEqual(args.symbol_filter_min_active_days, 3)
        self.assertAlmostEqual(args.symbol_filter_max_executed_trade_share, 0.55)
        self.assertEqual(args.prediction_bundle_cache, "disk")
        self.assertTrue(args.fast_diagnostics)

    def test_apply_diversity_selection_defaults_keeps_symbol_filter_explicit(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--threshold-max-top1-concentration", "0.8",
        ])
        changed = pipeline.apply_diversity_selection_defaults(args, set())
        self.assertTrue(changed)
        self.assertEqual(args.threshold_tiebreaker, "diversified")
        self.assertEqual(args.symbol_validation_filter, "none")
        self.assertEqual(args.symbol_filter_stage, "executed")
        self.assertEqual(args.symbol_filter_min_active_days, 0)
        self.assertAlmostEqual(args.symbol_filter_max_executed_trade_share, 0.0)
        self.assertEqual(args.threshold_concentration_cap_mode, "soft")
        self.assertAlmostEqual(args.threshold_diversity_penalty_weight, 0.01)
        self.assertAlmostEqual(args.threshold_diversity_profit_tolerance_ratio, 0.25)
        self.assertAlmostEqual(args.symbol_dominance_penalty_validation_weight, 0.35)
        self.assertAlmostEqual(args.symbol_dominance_penalty_recent_weight, 0.50)
        self.assertAlmostEqual(args.symbol_dominance_penalty_grace, 0.35)
        self.assertTrue(args.prefer_unique_symbols)
        self.assertEqual(args.symbol_reentry_cooldown_minutes, 15)

    def test_symbol_filter_stage_does_not_activate_validation_filter_without_opt_in(self):
        parser = pipeline.build_parser()
        argv = ["--symbol-filter-stage", "candidate_blend"]
        args = parser.parse_args(argv)
        explicit_flags = pipeline.parse_explicit_flags(argv)
        changed = pipeline.apply_symbol_filter_activation_defaults(args, explicit_flags)
        self.assertFalse(changed)
        self.assertEqual(args.symbol_validation_filter, "none")
        self.assertEqual(args.symbol_filter_stage, "candidate_blend")

    def test_symbol_filter_activation_defaults_require_explicit_enable_flag(self):
        parser = pipeline.build_parser()
        argv = [
            "--enable-symbol-filter-activation-defaults",
            "--symbol-filter-stage", "candidate_blend",
        ]
        args = parser.parse_args(argv)
        explicit_flags = pipeline.parse_explicit_flags(argv)
        changed = pipeline.apply_symbol_filter_activation_defaults(args, explicit_flags)
        self.assertTrue(changed)
        self.assertEqual(args.symbol_validation_filter, "positive_avg_profit")
        self.assertEqual(args.symbol_filter_stage, "candidate_blend")

    def test_explicit_symbol_validation_none_keeps_symbol_filter_disabled(self):
        parser = pipeline.build_parser()
        argv = [
            "--symbol-filter-stage", "candidate_blend",
            "--symbol-validation-filter", "none",
        ]
        args = parser.parse_args(argv)
        explicit_flags = pipeline.parse_explicit_flags(argv)
        changed = pipeline.apply_symbol_filter_activation_defaults(args, explicit_flags)
        self.assertFalse(changed)
        self.assertEqual(args.symbol_validation_filter, "none")

    def test_threshold_score_edge_gate_enables_validation_health_floors(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["--threshold-score-edge-gate"])
        changed = pipeline.apply_threshold_score_edge_gate(args, {"threshold_score_edge_gate"})
        self.assertTrue(changed)
        self.assertAlmostEqual(args.threshold_min_top_decile_net_return, 0.0)
        self.assertAlmostEqual(args.threshold_min_score_win_loss_gap, 0.0)

    def test_threshold_score_edge_gate_respects_explicit_floor_overrides(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--threshold-score-edge-gate",
            "--threshold-min-top-decile-net-return", "0.002",
            "--threshold-min-score-win-loss-gap", "0.0001",
        ])
        changed = pipeline.apply_threshold_score_edge_gate(
            args,
            {
                "threshold_score_edge_gate",
                "threshold_min_top_decile_net_return",
                "threshold_min_score_win_loss_gap",
            },
        )
        self.assertFalse(changed)
        self.assertAlmostEqual(args.threshold_min_top_decile_net_return, 0.002)
        self.assertAlmostEqual(args.threshold_min_score_win_loss_gap, 0.0001)

    def test_tune_threshold_soft_concentration_cap_keeps_near_cap_candidate(self):
        rows = [row(1600000000000)]
        candidate_metrics = {
            "predicted_trades": 31,
            "portfolio_profit": 215.15,
            "portfolio_return": 0.0215,
            "precision": 0.4516,
            "recall": 0.12,
            "active_days": 8,
            "raw_signal_trades": 31,
            "max_capital_drawdown": 0.10,
            "average_profit_after_fee_and_slippage": 6.94,
            "total_profit_after_fee_and_slippage": 215.15,
            "symbol_trade_concentration_top1": 24.0 / 31.0,
            "symbol_profit_concentration_top1": 0.93,
            "symbol_profit_concentration_top3": 1.0,
        }
        runtime_args = SimpleNamespace(
            threshold_max_trade_top1_concentration=0.77375,
            threshold_trade_top1_concentration_penalty=0.0,
            threshold_max_top1_concentration=0.0,
            threshold_max_top3_concentration=0.0,
            threshold_top1_concentration_penalty=0.0,
            threshold_top3_concentration_penalty=0.0,
            threshold_concentration_cap_mode="soft",
            threshold_tiebreaker="fewer_trades",
            threshold_tie_epsilon=1e-9,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            target_validation_trades=0,
            threshold_objective="profit",
        )
        with mock.patch.object(pipeline, "evaluate", return_value=dict(candidate_metrics)):
            selection = pipeline.tune_threshold(
                rows,
                [0.8],
                [0.77],
                "profit",
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
                hybrid_runtime_args=runtime_args,
            )
        self.assertEqual(selection["threshold"], 0.77)
        self.assertGreater(selection["validation_metrics"]["predicted_trades"], 0)

    def test_tune_threshold_hard_concentration_cap_rejects_near_cap_candidate(self):
        rows = [row(1600000000000)]
        candidate_metrics = {
            "predicted_trades": 31,
            "portfolio_profit": 215.15,
            "portfolio_return": 0.0215,
            "precision": 0.4516,
            "recall": 0.12,
            "active_days": 8,
            "raw_signal_trades": 31,
            "max_capital_drawdown": 0.10,
            "average_profit_after_fee_and_slippage": 6.94,
            "total_profit_after_fee_and_slippage": 215.15,
            "symbol_trade_concentration_top1": 24.0 / 31.0,
            "symbol_profit_concentration_top1": 0.93,
            "symbol_profit_concentration_top3": 1.0,
        }
        fallback_metrics = {
            "predicted_trades": 0,
            "portfolio_profit": 0.0,
            "portfolio_return": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "active_days": 0,
            "raw_signal_trades": 0,
            "max_capital_drawdown": 0.0,
            "average_profit_after_fee_and_slippage": 0.0,
            "total_profit_after_fee_and_slippage": 0.0,
            "symbol_trade_concentration_top1": 0.0,
            "symbol_profit_concentration_top1": 0.0,
            "symbol_profit_concentration_top3": 0.0,
        }
        runtime_args = SimpleNamespace(
            threshold_max_trade_top1_concentration=0.77375,
            threshold_trade_top1_concentration_penalty=0.0,
            threshold_max_top1_concentration=0.0,
            threshold_max_top3_concentration=0.0,
            threshold_top1_concentration_penalty=0.0,
            threshold_top3_concentration_penalty=0.0,
            threshold_concentration_cap_mode="hard",
            threshold_tiebreaker="fewer_trades",
            threshold_tie_epsilon=1e-9,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            target_validation_trades=0,
            threshold_objective="profit",
        )
        with mock.patch.object(pipeline, "evaluate", side_effect=[dict(candidate_metrics), dict(fallback_metrics)]):
            selection = pipeline.tune_threshold(
                rows,
                [0.8],
                [0.77],
                "profit",
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
                hybrid_runtime_args=runtime_args,
            )
        self.assertEqual(selection["threshold"], 1.01)
        self.assertEqual(selection["validation_metrics"]["selected_threshold_tie_rank_reason"], "no_trade_fallback")

    def test_tune_threshold_hard_profit_concentration_cap_rejects_fragile_candidate(self):
        rows = [row(1600000000000)]
        candidate_metrics = {
            "predicted_trades": 22,
            "portfolio_profit": 48.0,
            "portfolio_return": 0.0048,
            "precision": 0.0,
            "recall": 0.0,
            "active_days": 9,
            "raw_signal_trades": 28,
            "max_capital_drawdown": 0.10,
            "average_profit_after_fee_and_slippage": 0.0012,
            "total_profit_after_fee_and_slippage": 0.0264,
            "executed_score_top_1pct_rows": 1,
            "executed_score_top_1pct_avg_net_return": 0.003,
            "executed_score_top_decile_rows": 3,
            "executed_score_top_decile_avg_net_return": 0.002,
            "executed_winning_trade_count": 12,
            "executed_losing_trade_count": 10,
            "executed_score_win_loss_gap": 0.01,
            "symbol_trade_concentration_top1": 0.25,
            "symbol_profit_concentration_top1": 0.989,
            "symbol_profit_concentration_top3": 0.999,
        }
        fallback_metrics = {
            "predicted_trades": 0,
            "portfolio_profit": 0.0,
            "portfolio_return": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "active_days": 0,
            "raw_signal_trades": 0,
            "max_capital_drawdown": 0.0,
            "average_profit_after_fee_and_slippage": 0.0,
            "total_profit_after_fee_and_slippage": 0.0,
            "symbol_trade_concentration_top1": 0.0,
            "symbol_profit_concentration_top1": 0.0,
            "symbol_profit_concentration_top3": 0.0,
        }
        runtime_args = SimpleNamespace(
            threshold_max_trade_top1_concentration=0.35,
            threshold_trade_top1_concentration_penalty=0.0,
            threshold_max_top1_concentration=0.70,
            threshold_max_top3_concentration=0.90,
            threshold_top1_concentration_penalty=0.0,
            threshold_top3_concentration_penalty=0.0,
            threshold_concentration_cap_mode="hard",
            threshold_require_positive_top_1pct=True,
            threshold_max_raw_signal_share=0.0,
            threshold_min_avg_net_return=0.0010,
            threshold_min_top_decile_net_return=0.0,
            threshold_min_score_win_loss_gap=0.0,
            threshold_tiebreaker="fewer_trades",
            threshold_tie_epsilon=1e-9,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            target_validation_trades=0,
            threshold_objective="profit",
        )
        with mock.patch.object(pipeline, "evaluate", side_effect=[dict(candidate_metrics), dict(fallback_metrics)]):
            selection = pipeline.tune_threshold(
                rows,
                [0.8],
                [0.77],
                "profit",
                0.0,
                0.0,
                20,
                120,
                0.0,
                "explore",
                10000.0,
                0.10,
                0.01,
                10,
                60,
                5,
                hybrid_runtime_args=runtime_args,
            )
        self.assertEqual(selection["threshold"], 1.01)
        self.assertEqual(selection["validation_metrics"]["rejected_over_top1_concentration_count"], 1)
        self.assertEqual(selection["validation_metrics"]["rejected_over_top3_concentration_count"], 1)
        diagnostics = selection["validation_metrics"]["threshold_rejection_diagnostics"]
        self.assertEqual(diagnostics[0]["rejected_over_top1_concentration"], 1)
        self.assertEqual(diagnostics[0]["rejected_over_top3_concentration"], 1)

    def test_tune_threshold_soft_profit_concentration_cap_keeps_trade_diversified_candidate(self):
        rows = [row(1600000000000)]
        candidate_metrics = {
            "predicted_trades": 23,
            "portfolio_profit": 14.0,
            "portfolio_return": 0.0014,
            "precision": 0.0,
            "recall": 0.0,
            "active_days": 8,
            "raw_signal_trades": 30,
            "max_capital_drawdown": 0.04,
            "average_profit_after_fee_and_slippage": 0.00061,
            "total_profit_after_fee_and_slippage": 0.0140,
            "executed_score_top_1pct_rows": 1,
            "executed_score_top_1pct_avg_net_return": 0.003,
            "executed_score_top_decile_rows": 3,
            "executed_score_top_decile_avg_net_return": 0.002,
            "executed_winning_trade_count": 12,
            "executed_losing_trade_count": 11,
            "executed_score_win_loss_gap": 0.01,
            "symbol_trade_concentration_top1": 0.18,
            "symbol_profit_concentration_top1": 0.46,
            "symbol_profit_concentration_top3": 0.994,
        }
        fallback_metrics = {
            "predicted_trades": 0,
            "portfolio_profit": 0.0,
            "portfolio_return": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "active_days": 0,
            "raw_signal_trades": 0,
            "max_capital_drawdown": 0.0,
            "average_profit_after_fee_and_slippage": 0.0,
            "total_profit_after_fee_and_slippage": 0.0,
            "symbol_trade_concentration_top1": 0.0,
            "symbol_profit_concentration_top1": 0.0,
            "symbol_profit_concentration_top3": 0.0,
        }
        runtime_args = SimpleNamespace(
            threshold_max_trade_top1_concentration=0.35,
            threshold_trade_top1_concentration_penalty=0.0,
            threshold_max_top1_concentration=0.70,
            threshold_max_top3_concentration=0.90,
            threshold_top1_concentration_penalty=0.0,
            threshold_top3_concentration_penalty=0.0,
            threshold_concentration_cap_mode="hard",
            threshold_profit_concentration_cap_mode="soft",
            threshold_require_positive_top_1pct=True,
            threshold_max_raw_signal_share=0.0,
            threshold_min_avg_net_return=0.0,
            threshold_min_top_decile_net_return=0.0,
            threshold_min_score_win_loss_gap=0.0,
            threshold_tiebreaker="fewer_trades",
            threshold_tie_epsilon=1e-9,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            target_validation_trades=0,
            threshold_objective="profit",
        )
        with mock.patch.object(pipeline, "evaluate", side_effect=[dict(candidate_metrics), dict(fallback_metrics)]):
            selection = pipeline.tune_threshold(
                rows,
                [0.8],
                [0.77],
                "profit",
                0.0,
                0.0,
                20,
                120,
                0.0,
                "explore",
                10000.0,
                0.10,
                0.01,
                10,
                60,
                5,
                hybrid_runtime_args=runtime_args,
            )
        self.assertEqual(selection["threshold"], 0.77)
        self.assertEqual(selection["validation_metrics"]["rejected_over_top3_concentration_count"], 0)

    def test_metrics_record_captures_diversity_runtime_settings(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--prefer-unique-symbols",
            "--symbol-reentry-cooldown-minutes", "15",
            "--symbol-dominance-penalty-validation-weight", "0.6",
            "--symbol-dominance-penalty-recent-weight", "0.9",
            "--symbol-dominance-penalty-grace", "0.2",
        ])
        record = pipeline.metrics_record(
            "gbdt",
            "fixed",
            "profit_balanced",
            0.05,
            [],
            [],
            [],
            {},
            args,
        )
        self.assertEqual(record["prefer_unique_symbols"], 1)
        self.assertEqual(record["symbol_reentry_cooldown_minutes"], 15)
        self.assertAlmostEqual(record["symbol_dominance_penalty_validation_weight"], 0.6)
        self.assertAlmostEqual(record["symbol_dominance_penalty_recent_weight"], 0.9)
        self.assertAlmostEqual(record["symbol_dominance_penalty_grace"], 0.2)
        self.assertEqual(record["threshold_concentration_cap_mode"], "soft")

    def test_metrics_record_hides_no_valid_threshold_sentinel(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([])
        record = pipeline.metrics_record(
            "gbdt",
            "fixed",
            "profit_balanced",
            1.01,
            [],
            [],
            [],
            {},
            args,
            {
                "threshold_selection_status": pipeline.THRESHOLD_STATUS_NO_VALID,
                "threshold_is_valid": 0,
                "threshold_fallback_reason": pipeline.THRESHOLD_STATUS_NO_VALID,
                "execution_threshold": 1.01,
                "selected_score_threshold": 1.01,
                "predicted_trades": 0,
                "precision": 0.0,
                "recall": 0.0,
                "average_profit_after_fee_and_slippage": 0.0,
                "total_profit_after_fee_and_slippage": 0.0,
                "portfolio_profit": 0.0,
                "portfolio_return": 0.0,
            },
        )
        self.assertIsNone(record["selected_threshold"])
        self.assertIsNone(record["selected_score_threshold"])
        self.assertEqual(record["execution_threshold"], 1.01)
        self.assertEqual(record["threshold_selection_status"], pipeline.THRESHOLD_STATUS_NO_VALID)

    def test_compare_threshold_results_diversified_prefers_lower_concentration_within_tolerance(self):
        args = SimpleNamespace(
            threshold_tiebreaker="diversified",
            threshold_tie_epsilon=1e-9,
            threshold_diversity_profit_tolerance_ratio=0.25,
            threshold_diversity_min_profit_top1_improvement=0.05,
            threshold_diversity_min_trade_top1_improvement=0.05,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            target_validation_trades=0,
            threshold_objective="profit_balanced",
        )
        best_result = {
            "objective_score": 314.0,
            "validation_metrics": {
                "predicted_trades": 54,
                "portfolio_profit": 314.0,
                "precision": 0.37,
                "active_days": 8,
                "max_capital_drawdown": 0.1,
                "symbol_profit_concentration_top1": 0.75,
                "symbol_trade_concentration_top1": 0.85,
            },
        }
        candidate_result = {
            "objective_score": 249.0,
            "validation_metrics": {
                "predicted_trades": 70,
                "portfolio_profit": 249.0,
                "precision": 0.31,
                "active_days": 8,
                "max_capital_drawdown": 0.1,
                "symbol_profit_concentration_top1": 0.46,
                "symbol_trade_concentration_top1": 0.55,
            },
        }
        better, reason = pipeline.compare_threshold_results(candidate_result, best_result, args, 20, 8000)
        self.assertTrue(better)
        self.assertEqual(reason, "diversified_tolerance")

    def test_compare_threshold_results_diversified_preserves_more_diverse_best_within_tolerance(self):
        args = SimpleNamespace(
            threshold_tiebreaker="diversified",
            threshold_tie_epsilon=1e-9,
            threshold_diversity_profit_tolerance_ratio=0.25,
            threshold_diversity_min_profit_top1_improvement=0.05,
            threshold_diversity_min_trade_top1_improvement=0.05,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            target_validation_trades=0,
            threshold_objective="profit_balanced",
        )
        best_result = {
            "objective_score": 249.0,
            "validation_metrics": {
                "predicted_trades": 70,
                "portfolio_profit": 249.0,
                "precision": 0.31,
                "active_days": 8,
                "max_capital_drawdown": 0.1,
                "symbol_profit_concentration_top1": 0.46,
                "symbol_trade_concentration_top1": 0.55,
            },
        }
        candidate_result = {
            "objective_score": 314.0,
            "validation_metrics": {
                "predicted_trades": 54,
                "portfolio_profit": 314.0,
                "precision": 0.37,
                "active_days": 8,
                "max_capital_drawdown": 0.1,
                "symbol_profit_concentration_top1": 0.75,
                "symbol_trade_concentration_top1": 0.85,
            },
        }
        better, reason = pipeline.compare_threshold_results(candidate_result, best_result, args, 20, 8000)
        self.assertFalse(better)
        self.assertEqual(reason, "diversified")

    def test_compare_threshold_results_prefers_non_floor_snap_when_scores_are_close(self):
        args = SimpleNamespace(
            threshold_tiebreaker="balanced",
            threshold_tie_epsilon=1e-9,
            threshold_floor_snap_score_tolerance_ratio=0.10,
            threshold_target_trades=0,
            threshold_target_active_days=0,
            target_validation_trades=0,
            threshold_objective="profit_balanced",
        )
        best_result = {
            "objective_score": 100.0,
            "penalized_objective_score": 100.0,
            "validation_metrics": {
                "predicted_trades": 40,
                "portfolio_profit": 100.0,
                "precision": 0.4,
                "active_days": 8,
                "max_capital_drawdown": 0.1,
                "symbol_profit_concentration_top1": 0.4,
                "symbol_trade_concentration_top1": 0.4,
                "threshold_floor_snap_applied": 1,
            },
        }
        candidate_result = {
            "objective_score": 94.0,
            "penalized_objective_score": 94.0,
            "validation_metrics": {
                "predicted_trades": 34,
                "portfolio_profit": 94.0,
                "precision": 0.41,
                "active_days": 8,
                "max_capital_drawdown": 0.1,
                "symbol_profit_concentration_top1": 0.4,
                "symbol_trade_concentration_top1": 0.4,
                "threshold_floor_snap_applied": 0,
            },
        }
        better, reason = pipeline.compare_threshold_results(candidate_result, best_result, args, 20, 8000)
        self.assertTrue(better)
        self.assertEqual(reason, "avoid_floor_snap")

    def test_metrics_record_carries_validation_concentration_controls(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["--input", self.csv_path])
        metrics = {
            "portfolio_profit": -10.0,
            "portfolio_return": -0.001,
            "precision": 0.1,
            "recall": 0.02,
            "predicted_trades": 5,
            "symbol_profit_concentration_top1": 0.9,
            "symbol_profit_concentration_top3": 1.0,
            "symbol_trade_concentration_top1": 0.8,
        }
        validation_metrics = {
            "predicted_trades": 12,
            "precision": 0.4,
            "recall": 0.08,
            "average_profit_after_fee_and_slippage": 0.01,
            "total_profit_after_fee_and_slippage": 0.12,
            "portfolio_profit": 25.0,
            "portfolio_return": 0.0025,
            "selected_validation_trade_count": 12,
            "selected_validation_portfolio_profit": 25.0,
            "selected_validation_portfolio_return": 0.0025,
            "selected_validation_precision": 0.4,
            "selected_validation_recall": 0.08,
            "selected_validation_active_days": 4,
            "selected_validation_profit_per_active_day": 6.25,
            "symbol_profit_concentration_top1": 0.62,
            "symbol_profit_concentration_top3": 0.88,
            "symbol_trade_concentration_top1": 0.5,
            "rejected_over_top1_concentration_count": 3,
            "rejected_over_top3_concentration_count": 2,
            "closest_top1_concentration": 0.81,
            "closest_top1_concentration_threshold": 0.04,
            "closest_top3_concentration": 0.97,
            "closest_top3_concentration_threshold": 0.03,
            "threshold_rejection_diagnostics": [
                {"threshold": 0.04, "rejected_over_top1_concentration": 1},
                {"threshold": 0.05, "rejected_over_top1_concentration": 1},
                {
                    "threshold": 0.06,
                    "predicted_trades": 24,
                    "raw_signal_share": 0.004,
                    "average_net_return_after_costs": 0.00041,
                    "executed_score_top_1pct_avg_net_return": 0.00050,
                    "executed_score_top_decile_avg_net_return": 0.00080,
                    "symbol_profit_concentration_top1": 0.40,
                    "symbol_profit_concentration_top3": 0.65,
                    "symbol_trade_concentration_top1": 0.20,
                    "rejected_under_min_avg_net_return": 1,
                },
            ],
        }
        record = pipeline.metrics_record(
            "gbdt_lightgbm",
            "fixed",
            "profit_balanced",
            0.05,
            [row(1600000000000)],
            [row(1600000060000)],
            [row(1600000120000)],
            metrics,
            args,
            validation_metrics,
            1.23,
        )
        self.assertAlmostEqual(record["selected_validation_symbol_profit_concentration_top1"], 0.62)
        self.assertAlmostEqual(record["selected_validation_symbol_profit_concentration_top3"], 0.88)
        self.assertAlmostEqual(record["selected_validation_symbol_trade_concentration_top1"], 0.5)
        self.assertEqual(record["selected_validation_symbol_count"], 0)
        self.assertEqual(record["rejected_over_top1_concentration_count"], 3)
        self.assertEqual(record["rejected_over_top3_concentration_count"], 2)
        self.assertEqual(record["threshold_rejection_diagnostics"][0]["threshold"], 0.04)

    def test_portfolio_execution_emits_ranking_and_score_diagnostics(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=0.01),
            row(base, quote_volume=1000000000.0, trade_return=0.02),
            row(base, quote_volume=1000000000.0, trade_return=-0.01),
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
            trade_selection="topk_score",
            top_k_per_minute=1,
            objective_mode="classification",
            trade_score_name="probability",
        )
        self.assertEqual(execution["ranking_candidate_count"], 3)
        self.assertEqual(execution["ranking_selected_candidate_count"], 1)
        self.assertEqual(execution["ranking_rejected_candidate_count"], 2)
        self.assertEqual(execution["execution_rejected_candidate_count"], 0)
        self.assertEqual(execution["raw_candidate_score_count"], 3)
        self.assertEqual(execution["executed_trade_score_count"], 1)
        self.assertEqual(execution["rejected_trade_score_count"], 2)
        self.assertEqual(execution["candidate_positive_utility_count"], 2)
        self.assertEqual(execution["candidate_executed_positive_utility_count"], 1)
        self.assertAlmostEqual(execution["candidate_positive_utility_recall"], 0.5)
        self.assertEqual(execution["candidate_rank_rejected_positive_utility_count"], 1)
        self.assertAlmostEqual(execution["candidate_missed_best_net_utility"], 0.02)
        self.assertAlmostEqual(execution["candidate_top_decile_net_utility"], 0.01)
        self.assertEqual(execution["candidate_high_score_loss_count"], 0)
        self.assertAlmostEqual(execution["candidate_score_positive_utility_gap"], 0.075)

    def test_portfolio_execution_candidate_diagnostics_detect_high_score_losers(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=-0.02),
            row(base, quote_volume=1000000000.0, trade_return=0.02),
            row(base, quote_volume=1000000000.0, trade_return=0.01),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.99, 0.20, 0.10],
            0.05,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            objective_mode="classification",
            trade_score_name="probability",
        )
        self.assertEqual(list(execution["executed"].keys()), [0])
        self.assertEqual(execution["candidate_positive_utility_count"], 2)
        self.assertEqual(execution["candidate_executed_positive_utility_count"], 0)
        self.assertAlmostEqual(execution["candidate_positive_utility_recall"], 0.0)
        self.assertEqual(execution["candidate_missed_positive_utility_count"], 2)
        self.assertAlmostEqual(execution["candidate_missed_best_net_utility"], 0.02)
        self.assertAlmostEqual(execution["candidate_high_score_loss_share"], 1.0)
        self.assertAlmostEqual(execution["candidate_high_score_loss_avg_net_utility"], -0.02)
        self.assertLess(execution["candidate_score_positive_utility_gap"], 0.0)

    def test_portfolio_execution_uses_ranker_selection_score_for_topk_ordering(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=-0.02),
            row(base, quote_volume=1000000000.0, trade_return=0.02),
            row(base, quote_volume=1000000000.0, trade_return=0.01),
        ]
        bundle = pipeline.build_prediction_bundle(
            ranker_score=[0.99, 0.20, 0.10],
            ranker_selection_score=[-0.020, 0.020, 0.010],
        )
        execution = pipeline.portfolio_execution(
            rows,
            bundle,
            0.05,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
        )
        self.assertEqual(list(execution["executed"].keys()), [1])
        self.assertEqual(execution["candidate_positive_utility_count"], 2)
        self.assertEqual(execution["candidate_executed_positive_utility_count"], 1)
        self.assertAlmostEqual(execution["candidate_positive_utility_recall"], 0.5)
        self.assertAlmostEqual(execution["candidate_high_score_loss_share"], 1.0)

    def test_portfolio_execution_groups_topk_candidates_by_clock_minute(self):
        base = 1600000000000
        rows = [
            symbol_row("AAAUSDT", base, quote_volume=1000000000.0, trade_return=0.01),
            symbol_row("BBBUSDT", base + 60000, quote_volume=1000000000.0, trade_return=0.01),
            symbol_row("CCCUSDT", base, quote_volume=1000000000.0, trade_return=0.02),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.90, 0.10, 0.95],
            0.0,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            0,
            60,
            5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            objective_mode="classification",
            trade_score_name="probability",
        )
        self.assertEqual(set(execution["executed"].keys()), {1, 2})
        self.assertNotIn(0, execution["executed"])

    def test_portfolio_execution_reports_chronological_audit_diagnostics(self):
        base = 1600000000000
        rows = [
            symbol_row("AAAUSDT", base, quote_volume=1000000000.0, trade_return=0.01),
            symbol_row("BBBUSDT", base + 60000, quote_volume=1000000000.0, trade_return=0.01),
            symbol_row("CCCUSDT", base, quote_volume=1000000000.0, trade_return=0.02),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            [0.90, 0.10, 0.95],
            0.0,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            0,
            60,
            5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            objective_mode="classification",
            trade_score_name="probability",
        )
        self.assertEqual(execution["execution_chronological_processing"], 1)
        self.assertEqual(execution["execution_chronological_sort_applied"], 1)
        self.assertEqual(execution["execution_chronological_audit_status"], "sorted_nonmonotonic_input")
        self.assertEqual(execution["execution_input_candidate_count"], 3)
        self.assertEqual(execution["execution_input_time_decrease_count"], 1)
        self.assertEqual(execution["execution_input_max_backward_minutes"], 1)
        self.assertEqual(execution["execution_bucket_count"], 2)
        self.assertEqual(execution["max_topk_bucket_size"], 2)
        self.assertEqual(execution["max_topk_bucket_ranked_count"], 2)
        self.assertEqual(execution["max_topk_bucket_selected_count"], 1)
        self.assertEqual(execution["topk_bucket_limit_violation_count"], 0)
        self.assertEqual(execution["max_same_timestamp_executed_trades"], 1)
        self.assertEqual(execution["max_concurrent_positions"], 2)
        self.assertAlmostEqual(execution["max_simultaneous_capital"], 2000.0)
        self.assertAlmostEqual(execution["max_capital_usage_fraction"], 0.2)
        self.assertAlmostEqual(execution["max_capital_usage_pct_initial"], 0.2)
        self.assertEqual(execution["capital_overallocated_count"], 0)

    def test_execution_audit_metrics_are_exported(self):
        for field in pipeline.EXECUTION_AUDIT_METRIC_KEYS:
            self.assertIn(field, pipeline.METRIC_COLUMNS)
            self.assertIn(field, pipeline.WALKFORWARD_DIAGNOSTIC_COLUMNS)

    def test_top_utility_ranker_requires_positive_selection_utility(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=-0.02),
            row(base, quote_volume=1000000000.0, trade_return=0.02),
            row(base, quote_volume=1000000000.0, trade_return=0.01),
        ]
        bundle = pipeline.build_prediction_bundle(
            ranker_score=[0.99, 0.20, 0.10],
            ranker_selection_score=[-0.020, 0.020, -0.001],
        )
        execution = pipeline.portfolio_execution(
            rows,
            bundle,
            -1000000000.0,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="top_utility",
            top_k_per_minute=3,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=SimpleNamespace(ranker_min_calibration_top_bin_utility=0.0),
        )
        self.assertEqual(list(execution["executed"].keys()), [1])
        self.assertEqual(execution["ranking_candidate_count"], 3)
        self.assertEqual(execution["ranking_selected_candidate_count"], 1)
        self.assertEqual(execution["ranking_rejected_candidate_count"], 2)

    def test_top_utility_ranker_abstains_without_selection_utility(self):
        base = 1600000000000
        rows = [
            row(base, quote_volume=1000000000.0, trade_return=0.02),
            row(base, quote_volume=1000000000.0, trade_return=0.01),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(ranker_score=[0.99, 0.20]),
            -1000000000.0,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="top_utility",
            top_k_per_minute=2,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=SimpleNamespace(ranker_min_calibration_top_bin_utility=0.0),
        )
        self.assertEqual(execution["ranking_candidate_count"], 2)
        self.assertEqual(execution["ranking_selected_candidate_count"], 0)
        self.assertEqual(len(execution["executed"]), 0)

    def test_ranker_selection_calibration_demotes_negative_high_score_bucket(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--ranker-selection-calibration", "score_bucket",
            "--ranker-selection-calibration-bins", "2",
            "--ranker-selection-calibration-min-rows", "4",
            "--ranker-selection-calibration-shrinkage", "0",
            "--fee", "0",
            "--slippage", "0",
        ])
        base = 1600000000000
        validation_rows = [
            row(base, trade_return=0.020),
            row(base + 60000, trade_return=0.015),
            row(base + 120000, trade_return=-0.020),
            row(base + 180000, trade_return=-0.030),
        ]
        bundle = pipeline.build_prediction_bundle(ranker_score=[0.10, 0.20, 0.90, 0.99])
        calibration = pipeline.fit_ranker_selection_calibration(validation_rows, bundle, args)
        self.assertTrue(calibration["enabled"])
        calibrated = pipeline.apply_ranker_selection_calibration([0.95, 0.15], calibration)
        self.assertLess(float(calibrated[0]), float(calibrated[1]))

    def test_ranker_relevance_context_uses_train_only_utility_quantiles(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--fee", "0",
            "--slippage", "0",
            "--ranker-relevance-q1", "0.5",
            "--ranker-relevance-q2", "0.75",
            "--ranker-relevance-q3", "1.0",
        ])
        train_rows = [
            row(1600000000000, trade_return=0.001),
            row(1600000060000, trade_return=0.002),
            row(1600000120000, trade_return=0.004),
            row(1600000180000, trade_return=-0.010),
        ]
        context = pipeline.fit_ranker_relevance_context(train_rows, args)
        self.assertEqual(context["utility_rows"], 4)
        self.assertEqual(context["positive_utility_rows"], 3)
        self.assertAlmostEqual(context["weak_positive_threshold"], 0.002)
        self.assertAlmostEqual(context["useful_positive_threshold"], 0.003)
        self.assertAlmostEqual(context["strong_positive_threshold"], 0.004)
        validation_labels = pipeline.ranker_relevance_labels([
            row(1600000240000, trade_return=0.100),
            row(1600000300000, trade_return=0.000),
        ], args, context)
        self.assertEqual(list(validation_labels), [4, 0])
        self.assertAlmostEqual(context["strong_positive_threshold"], 0.004)

    def test_ranker_utility_target_margin_requires_cost_plus_margin(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--fee", "0.001",
            "--slippage", "0.0005",
            "--ranker-utility-target-margin", "0.001",
        ])
        rows_for_context = [
            row(1600000000000, trade_return=0.0030),
            row(1600000060000, trade_return=0.0020),
        ]
        utilities = pipeline.ranker_net_utility_values(rows_for_context, args)
        self.assertAlmostEqual(float(utilities[0]), 0.0005, places=7)
        self.assertAlmostEqual(float(utilities[1]), -0.0005, places=7)
        context = pipeline.fit_ranker_relevance_context(rows_for_context, args)
        self.assertEqual(context["positive_utility_rows"], 1)
        self.assertAlmostEqual(context["ranker_utility_target_margin"], 0.001)

    def test_ranker_utility_tail_relevance_ignores_tiny_positive_utility(self):
        context = {
            "mode": "utility_tail",
            "weak_positive_threshold": 0.003,
            "useful_positive_threshold": 0.006,
            "strong_positive_threshold": 0.008,
        }
        labels = pipeline.ranker_relevance_labels_from_utilities(
            [-0.010, 0.001, 0.004, 0.007, 0.009],
            context,
        )
        self.assertEqual(list(labels), [0, 0, 1, 3, 4])

    def test_ranker_relevance_context_reports_label_distribution(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--ranker-relevance-mode", "utility_tail",
            "--ranker-relevance-q1", "0.25",
            "--ranker-relevance-q2", "0.5",
            "--ranker-relevance-q3", "0.75",
            "--fee", "0",
            "--slippage", "0",
        ])
        rows_for_context = [
            row(1600000000000, trade_return=-0.010),
            row(1600000060000, trade_return=0.001),
            row(1600000120000, trade_return=0.004),
            row(1600000180000, trade_return=0.008),
            row(1600000240000, trade_return=0.010),
        ]
        context = pipeline.fit_ranker_relevance_context(rows_for_context, args)
        self.assertEqual(context["ranker_label_0_rows"], 2)
        self.assertEqual(context["ranker_label_1_rows"], 1)
        self.assertEqual(context["ranker_label_3_rows"], 1)
        self.assertEqual(context["ranker_label_4_rows"], 1)
        self.assertEqual(context["ranker_relevant_rows"], 3)
        self.assertAlmostEqual(context["ranker_relevant_row_share"], 0.6)

    def test_ranker_selection_score_blends_bucket_utility_and_regression_utility(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--ranker-utility-regression-blend", "0.5",
        ])
        calibration = {
            "enabled": True,
            "global_avg_utility": 0.0,
            "bins": [
                {"upper_score": 0.5, "calibrated_utility": -0.002},
                {"upper_score": 1.0, "calibrated_utility": 0.002},
            ],
        }
        blended = pipeline.rank_selection_score_with_utility_blend(
            [0.25, 0.75],
            calibration,
            [0.006, -0.006],
            args,
        )
        self.assertGreater(float(blended[0]), float(blended[1]))

    def test_negative_calibration_top_bin_rejects_ranker_selection_scores(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--fee", "0",
            "--slippage", "0",
            "--ranker-selection-calibration", "score_bucket",
            "--ranker-selection-calibration-bins", "2",
            "--ranker-selection-calibration-min-rows", "4",
            "--ranker-selection-calibration-shrinkage", "0",
            "--ranker-reject-negative-calibration-top-bin",
        ])
        validation_rows = [
            row(1600000000000, trade_return=0.010),
            row(1600000060000, trade_return=0.008),
            row(1600000120000, trade_return=-0.020),
            row(1600000180000, trade_return=-0.030),
        ]
        raw_scores = [0.10, 0.20, 0.90, 0.99]
        calibration = pipeline.fit_ranker_selection_calibration(
            validation_rows,
            pipeline.build_prediction_bundle(ranker_score=raw_scores),
            args,
        )
        self.assertTrue(calibration["top_bin_rejected"])
        selection_scores = pipeline.rank_selection_score_with_calibration_gate(
            raw_scores,
            calibration,
            None,
            args,
        )
        self.assertTrue(all(math.isinf(float(value)) and float(value) < 0.0 for value in selection_scores))

    def test_ranker_grouped_rows_drops_singleton_decision_groups(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--ranker-group-minutes", "1",
            "--ranker-min-group-size", "2",
        ])
        base = 1600000000000
        grouped_rows, groups, info = pipeline.ranker_grouped_rows([
            symbol_row("AAAUSDT", base, trade_return=0.01),
            symbol_row("BBBUSDT", base + 10 * 1000, trade_return=0.02),
            symbol_row("CCCUSDT", base + 2 * 60 * 1000, trade_return=0.03),
        ], args)
        self.assertEqual(groups, [2])
        self.assertEqual(len(grouped_rows), 2)
        self.assertEqual(info["ranker_group_count"], 1)
        self.assertEqual(info["ranker_group_rows"], 2)

    def test_prediction_bundle_exposes_ranker_trade_score(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["--objective-mode", "economic_ranking"])
        bundle = pipeline.build_prediction_bundle(ranker_score=[0.1, 0.25])
        self.assertAlmostEqual(pipeline.trade_score_value(bundle, 1, "ranker_score", 0.05, 0.02, 0.0, 0.0), 0.25)
        values = list(pipeline.score_values_for_bundle([], bundle, args))
        self.assertAlmostEqual(float(values[0]), 0.1)
        self.assertAlmostEqual(float(values[1]), 0.25)

    def test_ranker_threshold_search_can_use_selection_score(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--ranker-threshold-score", "selection",
        ])
        bundle = pipeline.build_prediction_bundle(
            ranker_score=[0.90, 0.10],
            ranker_selection_score=[-0.02, 0.03],
        )
        values = list(pipeline.score_values_for_bundle([], bundle, args))
        self.assertAlmostEqual(float(values[0]), -0.02)
        self.assertAlmostEqual(float(values[1]), 0.03)

    def test_ranker_selection_threshold_reports_selected_score_name(self):
        base = 1600000000000
        rows = [
            symbol_row("AAAUSDT", base, quote_volume=1000000000.0, trade_return=0.02),
            symbol_row("BBBUSDT", base + 60000, quote_volume=1000000000.0, trade_return=0.03),
        ]
        runtime_args = SimpleNamespace(
            ranker_threshold_score="selection",
            ranker_score_upper_quantile=1.0,
        )
        selection = pipeline.tune_threshold(
            rows,
            pipeline.build_prediction_bundle(
                ranker_score=[0.90, 0.10],
                ranker_selection_score=[-0.02, 0.03],
            ),
            [0.0],
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
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=runtime_args,
        )
        self.assertEqual(selection["selected_score_name"], "ranker_selection_score")
        self.assertEqual(selection["validation_metrics"]["selected_score_name"], "ranker_selection_score")

    def test_portfolio_execution_economic_threshold_can_use_selection_score(self):
        base = 1600000000000
        rows = [
            symbol_row("AAAUSDT", base, quote_volume=1000000000.0, trade_return=0.02),
            symbol_row("BBBUSDT", base + 60000, quote_volume=1000000000.0, trade_return=0.03),
        ]
        runtime_args = SimpleNamespace(
            ranker_score_upper_quantile=1.0,
            ranker_threshold_score="selection",
            top_k_per_symbol_minute=0,
            max_trades_per_symbol_period=0,
            max_same_symbol_streak=0,
            max_symbol_fold_trade_share=0.0,
            max_symbol_fold_trade_share_min_trades=0,
            prefer_unique_symbols=False,
        )
        execution = pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(
                ranker_score=[0.90, 0.10],
                ranker_selection_score=[-0.02, 0.03],
            ),
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
            top_k_per_minute=1,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=runtime_args,
        )
        self.assertEqual(list(execution["executed"].keys()), [1])
        self.assertAlmostEqual(execution["executed_trade_scores"][1], 0.10)

    def test_ranker_threshold_search_uses_scores_below_upper_cap(self):
        args = SimpleNamespace(
            objective_mode="economic_ranking",
            ranker_score_upper_quantile=0.90,
        )
        bundle = pipeline.build_prediction_bundle(ranker_score=[float(value) for value in range(1, 101)])
        search_scores = pipeline.score_values_for_bundle([], bundle, args)
        finite_scores = [float(value) for value in search_scores if math.isfinite(float(value))]
        self.assertEqual(len(finite_scores), 90)
        self.assertAlmostEqual(max(finite_scores), 90.0)
        thresholds = pipeline.adaptive_score_thresholds(
            search_scores,
            -1000000000.0,
            target_counts=[5],
        )
        self.assertIn(86.0, thresholds)
        self.assertNotIn(100.0, thresholds)

    def test_portfolio_execution_top_percent_ranker_selection(self):
        base = 1600000000000
        rows = [
            symbol_row("AAAUSDT", base, quote_volume=1000000000.0, trade_return=0.01),
            symbol_row("BBBUSDT", base, quote_volume=1000000000.0, trade_return=0.02),
            symbol_row("CCCUSDT", base, quote_volume=1000000000.0, trade_return=0.03),
            symbol_row("DDDUSDT", base, quote_volume=1000000000.0, trade_return=0.04),
        ]
        execution = pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(ranker_score=[4.0, 3.0, 2.0, 1.0]),
            -1000000000.0,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="top_percent_score",
            top_k_per_minute=0,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=SimpleNamespace(top_percent_per_period=0.5, prefer_unique_symbols=False),
        )
        self.assertEqual(execution["ranking_candidate_count"], 4)
        self.assertEqual(execution["ranking_selected_candidate_count"], 2)
        self.assertEqual(len(execution["executed"]), 2)
        self.assertEqual(execution["executed_selection_ranks"], {0: 1, 1: 2})

    def test_ranker_upper_quantile_blocks_extreme_score_tail(self):
        base = 1600000000000
        rows = [
            symbol_row("AAAUSDT", base, quote_volume=1000000000.0, trade_return=-0.03),
            symbol_row("BBBUSDT", base, quote_volume=1000000000.0, trade_return=-0.02),
            symbol_row("CCCUSDT", base, quote_volume=1000000000.0, trade_return=0.03),
            symbol_row("DDDUSDT", base, quote_volume=1000000000.0, trade_return=0.01),
        ]
        runtime_args = SimpleNamespace(ranker_score_upper_quantile=0.5)
        execution = pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(ranker_score=[4.0, 3.0, 2.0, 1.0]),
            -1000000000.0,
            0.0,
            0.0,
            10000.0,
            0.10,
            0.01,
            10,
            60,
            5,
            trade_selection="topk_score",
            top_k_per_minute=1,
            objective_mode="economic_ranking",
            trade_score_name="ranker_score",
            hybrid_runtime_args=runtime_args,
        )
        self.assertAlmostEqual(execution["ranker_score_upper_quantile"], 0.5)
        self.assertAlmostEqual(execution["ranker_score_upper_cap"], 2.5)
        self.assertEqual(execution["ranker_score_upper_cap_blocked"], 2)
        self.assertEqual(execution["blocked_trades_total"], 2)
        self.assertEqual(list(execution["executed"].keys()), [2])
        self.assertAlmostEqual(execution["executed_trade_scores"][2], 2.0)

    def test_metrics_record_carries_ranker_training_diagnostics(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--trade-score", "ranker_score",
            "--ranker-adverse-penalty", "0.1",
        ])
        validation_metrics = {
            "predicted_trades": 1,
            "precision": 1.0,
            "recall": 0.5,
            "average_profit_after_fee_and_slippage": 0.01,
            "total_profit_after_fee_and_slippage": 0.01,
            "portfolio_profit": 10.0,
            "portfolio_return": 0.001,
            "calibration_info": {
                "ranker_objective": "rank_xendcg",
                "ranker_relevance_mode": "train_quantiles",
                "ranker_utility_rows": 100,
                "ranker_positive_utility_rows": 20,
                "ranker_weak_positive_threshold": 0.001,
                "ranker_useful_positive_threshold": 0.002,
                "ranker_strong_positive_threshold": 0.004,
                "ranker_train_group_count": 10,
                "ranker_validation_group_count": 3,
                "ranker_train_average_group_size": 8.0,
                "ranker_validation_average_group_size": 5.0,
            },
        }
        record = pipeline.metrics_record(
            "gbdt_lightgbm",
            "fixed",
            "profit_balanced",
            -1000000000.0,
            [],
            [],
            [],
            {},
            args,
            validation_metrics,
        )
        self.assertEqual(record["trade_score"], "ranker_score")
        self.assertEqual(record["ranker_relevance_mode"], "train_quantiles")
        self.assertEqual(record["ranker_positive_utility_rows"], 20)
        self.assertAlmostEqual(record["ranker_adverse_penalty"], 0.1)

    def test_feature_usage_summary_reports_nonzero_features(self):
        class DummyModel(object):
            def feature_importance(self, feature_names):
                return [
                    (feature_names[0], 6.0, 0.60),
                    (feature_names[1], 4.0, 0.40),
                    (feature_names[2], 0.0, 0.0),
                ]

        summary = pipeline.feature_usage_summary(DummyModel(), ["ret_1m", "ret_5m", "ret_15m"])
        self.assertEqual(summary["nonzero_feature_count"], 2)
        self.assertEqual(summary["zero_importance_feature_count"], 1)
        self.assertIn("ret_1m", summary["top_nonzero_features"])

    def test_prediction_bundle_for_models_reuses_memory_cache(self):
        class DummyModel(object):
            def __init__(self):
                self.calls = 0

            def predict_values(self, x_rows):
                self.calls += 1
                return [0.75] * len(x_rows)

        pipeline.PREDICTION_BUNDLE_MEMORY_CACHE.clear()
        model = DummyModel()
        rows = [row(1600000000000), row(1600000060000)]
        args = SimpleNamespace(
            prediction_batch_rows=100,
            memmap_dir=self.temp.name,
            prediction_bundle_cache="memory",
            prediction_bundle_cache_dir=self.temp.name,
            input=self.csv_path,
            cache_dir=self.temp.name,
            meta_filter="none",
            threshold_objective="profit_balanced",
            trade_selection="threshold",
            top_k_per_minute=1,
            upside_target=0.05,
            downside_stop=0.02,
            ev_safety_margin=0.0,
            objective_mode="classification",
            min_predicted_net_return=0.0,
            hybrid_min_score=0.0,
            max_trades_per_day=0,
            max_trades_per_fold=0,
            max_losing_trades_per_day=0,
            max_daily_drawdown=0.0,
            pause_after_drawdown_minutes=0,
            hybrid_score_mode="basic",
            hybrid_uncertainty_penalty=0.0,
            regression_target="trade_return",
            n_jobs=1,
            max_rss_gb=0.0,
        )
        selected = {
            "classification_model": model,
            "regression_model": None,
            "threshold": 0.5,
            "params": {"n_estimators": 10},
            "calibration": None,
            "regression_calibration": None,
            "uncertainty_model": None,
            "meta_filter_info": None,
            "symbol_filter_info": None,
            "ev_payoff_info": None,
            "hybrid_return_info": None,
        }
        first = pipeline.prediction_bundle_for_models(selected, rows, "internal", args, "unit test")
        second = pipeline.prediction_bundle_for_models(selected, rows, "internal", args, "unit test")
        self.assertEqual(model.calls, 1)
        self.assertEqual(list(first["probability"]), list(second["probability"]))

    def test_prediction_bundle_cache_key_changes_when_unsampled_row_changes(self):
        args = SimpleNamespace(
            prediction_batch_rows=100,
            prediction_bundle_cache_dir=self.temp.name,
            input=self.csv_path,
            cache_dir=self.temp.name,
        )
        selected = {
            "params": {},
            "threshold": 0.5,
            "classification_model": object(),
            "regression_model": None,
            "ranker_model": None,
            "ranker_utility_model": None,
        }
        base = 1600000000000
        left_rows = [
            symbol_row("AAAUSDT", base),
            symbol_row("BBBUSDT", base + 60000),
            symbol_row("CCCUSDT", base + 120000),
            symbol_row("DDDUSDT", base + 180000),
        ]
        right_rows = [
            symbol_row("AAAUSDT", base),
            symbol_row("ZZZUSDT", base + 60000),
            symbol_row("CCCUSDT", base + 120000),
            symbol_row("DDDUSDT", base + 180000),
        ]

        left_key = pipeline.prediction_bundle_cache_key(selected, left_rows, "internal", args, "unit test")
        right_key = pipeline.prediction_bundle_cache_key(selected, right_rows, "internal", args, "unit test")

        self.assertNotEqual(left_key, right_key)

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

    def test_current_rss_gib_uses_psutil_and_updates_peak(self):
        class FakeMemoryInfo:
            rss = 2 * 1024 ** 3

        class FakeProcess:
            def memory_info(self):
                return FakeMemoryInfo()

        class FakePsutil:
            @staticmethod
            def Process(pid):
                del pid
                return FakeProcess()

        previous_psutil = pipeline.psutil
        previous_peak = pipeline.MAX_RSS_GIB_OBSERVED
        previous_stage = pipeline.MAX_RSS_STAGE
        previous_profile_enabled = pipeline.PROFILE_ENABLED
        previous_records = list(pipeline.PROFILE_RECORDS)
        try:
            pipeline.psutil = FakePsutil()
            pipeline.MAX_RSS_GIB_OBSERVED = 0.0
            pipeline.MAX_RSS_STAGE = ""
            pipeline.profile_reset(True)
            self.assertAlmostEqual(pipeline.current_rss_gib("fake-psutil"), 2.0)
            self.assertAlmostEqual(pipeline.MAX_RSS_GIB_OBSERVED, 2.0)
            self.assertEqual(pipeline.MAX_RSS_STAGE, "fake-psutil")
            pipeline.record_profile_stage("manual-stage", 0.5, rows_processed=10)
            self.assertEqual(pipeline.PROFILE_RECORDS[-1]["stage_name"], "manual-stage")
            self.assertAlmostEqual(pipeline.PROFILE_RECORDS[-1]["rss_gb_end"], 2.0)
        finally:
            pipeline.psutil = previous_psutil
            pipeline.MAX_RSS_GIB_OBSERVED = previous_peak
            pipeline.MAX_RSS_STAGE = previous_stage
            pipeline.PROFILE_ENABLED = previous_profile_enabled
            del pipeline.PROFILE_RECORDS[:]
            pipeline.PROFILE_RECORDS.extend(previous_records)

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

    def test_make_model_uses_native_lightgbm_for_classification(self):
        params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "min_child_samples": 20,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "max_bin": 63,
            "subsample_for_bin": 1000,
            "histogram_pool_size": 128.0,
            "n_jobs": 2,
        }
        model = pipeline.make_model("lightgbm", params, 3.0, objective_mode="classification")
        self.assertIsInstance(model, pipeline.NativeLightGBMModel)
        self.assertEqual(model.task, "classification")
        self.assertAlmostEqual(model.positive_weight, 3.0)

    def test_make_model_uses_native_lightgbm_for_regression(self):
        params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "min_child_samples": 20,
            "min_split_gain": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "max_bin": 63,
            "subsample_for_bin": 1000,
            "histogram_pool_size": 128.0,
            "n_jobs": 2,
        }
        model = pipeline.make_model("lightgbm", params, 1.0, objective_mode="regression")
        self.assertIsInstance(model, pipeline.NativeLightGBMModel)
        self.assertEqual(model.task, "regression")

    def test_model_candidate_count_limits_candidate_grid(self):
        parser = pipeline.build_parser()
        full_args = parser.parse_args([])
        self.assertEqual(len(pipeline.candidate_params("lightgbm", full_args)), 3)

        limited_args = parser.parse_args(["--model-candidate-count", "2"])
        candidates = pipeline.candidate_params("lightgbm", limited_args)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[-1]["num_leaves"], 31)

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

    def test_calibration_report_missing_predictions_marks_summary_unavailable(self):
        rows, summary = pipeline.calibration_report_from_predictions("")
        self.assertEqual(rows[0]["skipped_reason"], "prediction_output_missing")
        self.assertEqual(summary["calibration_report_skipped_reason"], "prediction_output_missing")
        resolved = pipeline.resolved_calibration_summary(rows, summary, {"brier_score": 0.25})
        self.assertEqual(resolved["calibration_report_skipped_reason"], "prediction_output_missing")
        self.assertAlmostEqual(resolved["brier_score"], 0.25)

    def test_ranking_report_from_predictions_scores_top_tail(self):
        path = os.path.join(self.temp.name, "rank_predictions.csv")
        fieldnames = [
            "symbol",
            "month",
            "month_index",
            "open_time",
            "label",
            "probability",
            "calibrated_probability",
            "hybrid_score",
            "ranker_score",
            "expected_value",
            "predicted_net_return",
            "trade_score",
            "predicted",
            "raw_signal",
            "position_size",
            "forward_return",
            "trade_return",
        ]
        records = []
        for index in range(20):
            records.append({
                "symbol": "AAAUSDT" if index < 2 else "BBBUSDT",
                "month": "2026-01" if index < 2 else "2026-02",
                "month_index": index,
                "open_time": 1704067200000 + index * 60000,
                "label": 1 if index < 2 else 0,
                "probability": 0.9 - index * 0.01,
                "calibrated_probability": 0.85 - index * 0.01,
                "hybrid_score": 1.0 - index * 0.01,
                "ranker_score": 2.0 - index * 0.01,
                "expected_value": 0.05 - index * 0.002,
                "predicted_net_return": 0.04 - index * 0.002,
                "trade_score": 1.0 - index * 0.01,
                "predicted": 1 if index < 3 else 0,
                "raw_signal": 1 if index < 5 else 0,
                "position_size": 1000.0 if index == 0 else 0.0,
                "forward_return": 0.01 if index < 2 else -0.005,
                "trade_return": 0.02 if index == 0 else (0.03 if index == 1 else -0.01),
            })
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        args = SimpleNamespace(
            fee=0.001,
            slippage=0.0005,
            test_slippage_multiplier=1.0,
            latency_penalty_bps=0.0,
            fee_mode="fixed",
            initial_capital=10000.0,
            max_position_fraction=0.10,
        )
        report_rows, summary = pipeline.ranking_report_from_predictions(path, args)
        top_decile = next(
            row
            for row in report_rows
            if row["score_source"] == "trade_score"
            and row["bucket_type"] == "score_decile"
            and row["bucket"] == "score_decile_01"
        )
        self.assertEqual(top_decile["rows"], 2)
        self.assertEqual(top_decile["predicted_trades"], 2)
        self.assertAlmostEqual(top_decile["avg_net_return_after_costs"], 0.0235)
        self.assertAlmostEqual(top_decile["portfolio_profit_if_selected"], 47.0)
        self.assertEqual(top_decile["top_symbol"], "AAAUSDT")
        self.assertAlmostEqual(top_decile["top_symbol_share"], 1.0)
        self.assertEqual(summary["ranking_report_rows_available"], 20)
        self.assertIn("trade_score", summary["ranking_report_score_sources"])
        self.assertIn("ranker_score", summary["ranking_report_score_sources"])
        self.assertAlmostEqual(summary["ranking_trade_score_top_decile_avg_net_return"], 0.0235)
        self.assertAlmostEqual(summary["ranking_trade_score_top_5pct_avg_net_return"], 0.0185)
        self.assertAlmostEqual(summary["ranking_ranker_score_top_decile_avg_net_return"], 0.0235)
        self.assertAlmostEqual(summary["ranking_ranker_score_top_5pct_avg_net_return"], 0.0185)
        raw_candidates = next(
            row
            for row in report_rows
            if row["score_source"] == "ranker_score"
            and row["bucket_type"] == "selection"
            and row["bucket"] == "raw_candidates"
        )
        self.assertEqual(raw_candidates["rows"], 5)

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

    def test_fixed_horizon_exit_policy_reproduces_trade_return(self):
        row_item = future_path_row("BTCUSDT", 1600000000000, [0.003, 0.006, 0.01], trade_return=0.01)
        args = SimpleNamespace(
            exit_policy="fixed_horizon",
            holding_period_minutes=3,
            max_holding_period_minutes=0,
            fee_mode="fixed",
            position_sizing_mode="fixed_fraction",
            min_order_notional=0.0,
            lot_size_step=0.0,
            tick_size=0.0,
            latency_penalty_bps=0.0,
            partial_fill_mode="none",
            max_open_positions=0,
            max_daily_loss_fraction=0.0,
            cooldown_after_loss_minutes=0,
            meta_filter="none",
            volatility_high_threshold=0.02,
            upside_target=0.05,
        )
        execution = pipeline.portfolio_execution(
            [row_item],
            pipeline.build_prediction_bundle(probability=[1.0], calibrated_probability=[1.0]),
            0.5,
            0.0,
            0.0,
            100.0,
            1.0,
            1.0,
            0,
            60,
            3,
            hybrid_runtime_args=args,
        )
        self.assertEqual(execution["exit_policy"], "fixed_horizon")
        self.assertEqual(execution["executed_exit_minutes"][0], 3)
        self.assertEqual(execution["executed_exit_reasons"][0], "fixed_horizon")
        self.assertAlmostEqual(execution["executed_dynamic_trade_returns"][0], 0.01)
        self.assertAlmostEqual(execution["portfolio_profit"], 1.0)

    def test_trailing_stop_does_not_activate_before_activation_return(self):
        details = pipeline.trailing_stop_exit_details(
            [0.005, 0.009, 0.0085, 0.008],
            0.008,
            0.01,
            0.003,
            0.02,
            4,
        )
        self.assertEqual(details["exit_reason"], "max_holding_time")
        self.assertEqual(details["exit_minutes"], 4)

    def test_trailing_stop_loss_fires_before_activation(self):
        details = pipeline.trailing_stop_exit_details(
            [-0.021, -0.015, 0.01],
            0.01,
            0.01,
            0.003,
            0.02,
            3,
        )
        self.assertEqual(details["exit_reason"], "stop_loss")
        self.assertEqual(details["exit_minutes"], 1)
        self.assertAlmostEqual(details["dynamic_trade_return"], -0.021)

    def test_trailing_stop_activates_after_positive_one_percent(self):
        details = pipeline.trailing_stop_exit_details(
            [0.011, 0.0115, 0.0112],
            0.0112,
            0.01,
            0.003,
            0.02,
            3,
        )
        self.assertEqual(details["exit_reason"], "max_holding_time")
        self.assertEqual(details["exit_minutes"], 3)
        self.assertGreater(details["max_favorable_excursion_before_exit"], 0.01)

    def test_trailing_stop_exits_after_drawdown_from_best_return(self):
        details = pipeline.trailing_stop_exit_details(
            [0.011, 0.015, 0.012],
            0.012,
            0.01,
            0.003,
            0.02,
            3,
        )
        self.assertEqual(details["exit_reason"], "trailing_stop")
        self.assertEqual(details["exit_minutes"], 3)
        self.assertAlmostEqual(details["dynamic_trade_return"], 0.012)

    def test_trailing_stop_exits_at_max_holding_when_no_event_occurs(self):
        details = pipeline.trailing_stop_exit_details(
            [0.001, 0.002, 0.0015, 0.0025],
            0.0025,
            0.01,
            0.003,
            0.02,
            4,
        )
        self.assertEqual(details["exit_reason"], "max_holding_time")
        self.assertEqual(details["exit_minutes"], 4)
        self.assertAlmostEqual(details["dynamic_trade_return"], 0.0025)

    def test_trailing_stop_releases_capital_at_dynamic_exit_time(self):
        base_time = 1600000000000
        rows = [
            future_path_row("AAAUSDT", base_time, [0.011, 0.007, 0.006, 0.005], trade_return=0.005),
            future_path_row("BBBUSDT", base_time + 3 * 60 * 1000, [0.002, 0.002, 0.002, 0.002], trade_return=0.002),
        ]
        args = SimpleNamespace(
            exit_policy="trailing_stop",
            holding_period_minutes=5,
            max_holding_period_minutes=4,
            trailing_activation_return=0.01,
            trailing_drawdown=0.003,
            stop_loss=0.02,
            dynamic_exit_price_source="existing_rows",
            fee_mode="fixed",
            position_sizing_mode="fixed_fraction",
            min_order_notional=0.0,
            lot_size_step=0.0,
            tick_size=0.0,
            latency_penalty_bps=0.0,
            partial_fill_mode="none",
            max_open_positions=1,
            max_daily_loss_fraction=0.0,
            cooldown_after_loss_minutes=0,
            meta_filter="none",
            volatility_high_threshold=0.02,
            upside_target=0.05,
        )
        execution = pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(probability=[1.0, 1.0], calibrated_probability=[1.0, 1.0]),
            0.5,
            0.0,
            0.0,
            100.0,
            1.0,
            1.0,
            0,
            60,
            5,
            hybrid_runtime_args=args,
        )
        self.assertEqual(len(execution["executed"]), 2)
        self.assertEqual(execution["executed_exit_minutes"][0], 2)
        self.assertIn(1, execution["executed"])

    def test_trailing_stop_fails_clearly_without_ordered_future_path(self):
        rows = [symbol_row("BTCUSDT", 1600000000000, trade_return=0.01)]
        args = SimpleNamespace(
            exit_policy="trailing_stop",
            holding_period_minutes=5,
            max_holding_period_minutes=5,
            trailing_activation_return=0.01,
            trailing_drawdown=0.003,
            stop_loss=0.02,
            dynamic_exit_price_source="existing_rows",
            fee_mode="fixed",
            position_sizing_mode="fixed_fraction",
            min_order_notional=0.0,
            lot_size_step=0.0,
            tick_size=0.0,
            latency_penalty_bps=0.0,
            partial_fill_mode="none",
            max_open_positions=0,
            max_daily_loss_fraction=0.0,
            cooldown_after_loss_minutes=0,
            meta_filter="none",
            volatility_high_threshold=0.02,
            upside_target=0.05,
        )
        with self.assertRaisesRegex(ValueError, "ordered future candle path"):
            pipeline.portfolio_execution(
                rows,
                pipeline.build_prediction_bundle(probability=[1.0], calibrated_probability=[1.0]),
                0.5,
                0.0,
                0.0,
                100.0,
                1.0,
                1.0,
                0,
                60,
                5,
                hybrid_runtime_args=args,
            )

    def test_dynamic_exit_logic_does_not_write_cache_files(self):
        cache_dir = os.path.join(self.temp.name, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        sentinel = os.path.join(cache_dir, "sentinel.txt")
        with open(sentinel, "w", encoding="utf-8") as handle:
            handle.write("keep")
        before = sorted(os.listdir(cache_dir))
        rows = [future_path_row("BTCUSDT", 1600000000000, [0.011, 0.007, 0.006], trade_return=0.006)]
        args = SimpleNamespace(
            exit_policy="trailing_stop",
            holding_period_minutes=5,
            max_holding_period_minutes=3,
            trailing_activation_return=0.01,
            trailing_drawdown=0.003,
            stop_loss=0.02,
            dynamic_exit_price_source="existing_rows",
            fee_mode="fixed",
            position_sizing_mode="fixed_fraction",
            min_order_notional=0.0,
            lot_size_step=0.0,
            tick_size=0.0,
            latency_penalty_bps=0.0,
            partial_fill_mode="none",
            max_open_positions=0,
            max_daily_loss_fraction=0.0,
            cooldown_after_loss_minutes=0,
            meta_filter="none",
            volatility_high_threshold=0.02,
            upside_target=0.05,
            cache_dir=cache_dir,
        )
        pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(probability=[1.0], calibrated_probability=[1.0]),
            0.5,
            0.0,
            0.0,
            100.0,
            1.0,
            1.0,
            0,
            60,
            5,
            hybrid_runtime_args=args,
        )
        self.assertEqual(before, sorted(os.listdir(cache_dir)))

    def test_raw_kline_dynamic_exit_uses_decision_candle_close_as_entry_price(self):
        raw_dir = os.path.join(self.temp.name, "raw_klines")
        os.makedirs(raw_dir, exist_ok=True)
        base_time = 1600000000000
        raw_path = os.path.join(raw_dir, "BTCUSDT-1m-2020-09-13.csv")
        with open(raw_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("{},100,116,99,110,10\n".format(base_time))
            handle.write("{},110,117,109,115,10\n".format(base_time + 60000))
            handle.write("{},115,118,114,116,10\n".format(base_time + 120000))

        source = pipeline.RawKlineExecutionPriceSource(raw_dir)
        path = source.return_path("BTCUSDT", base_time, 2)

        self.assertAlmostEqual(path[0], 115.0 / 110.0 - 1.0)
        self.assertAlmostEqual(path[1], 116.0 / 110.0 - 1.0)

    def test_raw_kline_dynamic_exit_rejects_conflicting_duplicate_candles(self):
        raw_dir = os.path.join(self.temp.name, "raw_klines_duplicate")
        os.makedirs(raw_dir, exist_ok=True)
        base_time = 1600000000000
        raw_path = os.path.join(raw_dir, "BTCUSDT-1m-2020-09-13.csv")
        with open(raw_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("{},100,111,99,110,10\n".format(base_time))
            handle.write("{},100,112,99,111,10\n".format(base_time))
            handle.write("{},111,114,110,113,10\n".format(base_time + 60000))

        source = pipeline.RawKlineExecutionPriceSource(raw_dir)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate raw kline candle"):
            source.return_path("BTCUSDT", base_time, 1)

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
            exit_policy="trailing_stop",
            trailing_activation_return=0.01,
            trailing_drawdown=0.003,
            stop_loss=0.02,
            max_holding_period_minutes=60,
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
            calibration_report_out=os.path.join(self.temp.name, "calibration.csv"),
            regime_report_out=os.path.join(self.temp.name, "regime.csv"),
            symbol_report_out=os.path.join(self.temp.name, "symbol.csv"),
            feature_stability_out=os.path.join(self.temp.name, "feature_stability.csv"),
            baseline_report_out=os.path.join(self.temp.name, "baseline.csv"),
            ranking_report_out=os.path.join(self.temp.name, "ranking.csv"),
            threshold_diagnostics_out=os.path.join(self.temp.name, "threshold_diagnostics.csv"),
            experiment_report_out=os.path.join(self.temp.name, "experiment.md"),
            _explicit_flags={"memory_budget_gb", "max_rss_gb"},
        )
        fixed_record = {
            "selected_threshold": 0.95,
            "exit_policy": "trailing_stop",
            "trailing_activation_return": 0.01,
            "trailing_drawdown": 0.003,
            "stop_loss": 0.02,
            "max_holding_period_minutes": 60,
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
            "rejected_over_top1_concentration_count": 2,
            "rejected_over_top3_concentration_count": 1,
            "closest_top1_concentration": 0.82,
            "closest_top1_concentration_threshold": 0.04,
            "closest_top3_concentration": 0.99,
            "closest_top3_concentration_threshold": 0.03,
            "selected_validation_symbol_profit_concentration_top1": 0.61,
            "selected_validation_symbol_profit_concentration_top3": 0.90,
            "selected_validation_symbol_trade_concentration_top1": 0.58,
            "average_exit_minutes": 7.5,
            "trailing_stop_exit_count": 2,
            "stop_loss_exit_count": 1,
            "max_holding_exit_count": 1,
            "average_dynamic_trade_return": 0.012,
            "average_fixed_horizon_trade_return": 0.01,
            "dynamic_minus_fixed_avg_return": 0.002,
            "trade_regime_filter": "market_positive",
            "trade_regime_breadth_threshold": 0.5,
            "regime_filter_blocked": 7,
            "raw_signal_trades_before_regime_filter": 30,
            "raw_signal_regime_filter_blocked": 6,
            "threshold_rejection_diagnostics": [
                {"threshold": 0.04, "rejected_over_top1_concentration": 1},
                {"threshold": 0.05, "rejected_over_top1_concentration": 1},
                {
                    "threshold": 0.06,
                    "predicted_trades": 24,
                    "raw_signal_share": 0.004,
                    "average_net_return_after_costs": 0.00041,
                    "executed_score_top_1pct_avg_net_return": 0.00050,
                    "executed_score_top_decile_avg_net_return": 0.00080,
                    "symbol_profit_concentration_top1": 0.40,
                    "symbol_profit_concentration_top3": 0.65,
                    "symbol_trade_concentration_top1": 0.20,
                    "rejected_under_min_avg_net_return": 1,
                },
            ],
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
        self.assertIn("walkforward_gate_status", summary["walk_forward_summary"])
        self.assertIsInstance(summary["args"]["_explicit_flags"], list)
        self.assertIn("normalized_microsecond_open_times", summary)
        self.assertIn("max_rss_stage", summary)
        self.assertIn("ranking_summary", summary)
        self.assertIn("ranking_report_rows_available", summary)
        self.assertIn("threshold_diagnostic_summary", summary)
        self.assertEqual(summary["threshold_diagnostics_rows_available"], 3)
        self.assertEqual(summary["trade_regime_filter"], "market_positive")
        self.assertEqual(summary["regime_filter_blocked"], 7)
        self.assertEqual(summary["raw_signal_trades_before_regime_filter"], 30)
        self.assertEqual(summary["raw_signal_regime_filter_blocked"], 6)
        self.assertEqual(summary["threshold_diagnostics_primary_rejection"], "rejected_over_top1_concentration")
        self.assertEqual(summary["threshold_diagnostics_near_miss_count"], 1)
        self.assertAlmostEqual(summary["threshold_diagnostics_best_near_miss_threshold"], 0.06)
        self.assertEqual(summary["threshold_diagnostics_best_near_miss_trades"], 24)
        self.assertAlmostEqual(summary["threshold_diagnostics_best_near_miss_top1_concentration"], 0.40)
        self.assertEqual(summary["ev_upside_target_source"], "manifest")
        self.assertAlmostEqual(summary["effective_upside_target"], 0.02)
        self.assertTrue(summary["market_regime_features"])
        self.assertAlmostEqual(summary["selected_threshold"], 0.95)
        self.assertEqual(summary["rejected_over_top1_concentration_count"], 2)
        self.assertAlmostEqual(summary["selected_validation_symbol_profit_concentration_top1"], 0.61)
        self.assertEqual(summary["threshold_rejection_diagnostics"][0]["threshold"], 0.04)
        self.assertEqual(summary["exit_policy"], "trailing_stop")
        self.assertAlmostEqual(summary["average_exit_minutes"], 7.5)
        self.assertIn("nonzero_feature_count", summary)
        with open(args.experiment_summary_out, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn("accepted", rows[0])
        self.assertIn("rejection_reason", rows[0])
        self.assertIn("active_fold_rate", rows[0])
        self.assertIn("strategy_strength", rows[0])
        self.assertIn("walkforward_gate_status", rows[0])
        self.assertIn("normalized_microsecond_open_times", rows[0])
        self.assertIn("ev_upside_target_source", rows[0])
        self.assertIn("effective_upside_target", rows[0])
        self.assertIn("ranking_report_rows_available", rows[0])
        self.assertIn("ranking_trade_score_top_decile_avg_net_return", rows[0])
        self.assertIn("threshold_diagnostics_near_miss_count", rows[0])
        self.assertEqual(rows[0]["trade_regime_filter"], "market_positive")
        self.assertEqual(rows[0]["regime_filter_blocked"], "7")
        self.assertEqual(rows[0]["raw_signal_trades_before_regime_filter"], "30")
        self.assertEqual(rows[0]["raw_signal_regime_filter_blocked"], "6")
        self.assertEqual(rows[0]["threshold_diagnostics_best_near_miss_trades"], "24")
        self.assertIn("max_rss_stage", rows[0])
        self.assertIn("selected_score_name", rows[0])
        self.assertIn("selected_score_threshold", rows[0])
        self.assertIn("exit_policy", rows[0])
        self.assertIn("average_exit_minutes", rows[0])
        self.assertIn("hybrid_return_combination", rows[0])
        self.assertIn("hybrid_min_probability", rows[0])
        self.assertIn("conditional_payoff_source", rows[0])
        self.assertIn("selected_validation_symbol_trade_concentration_top1", rows[0])
        self.assertIn("nonzero_feature_count", rows[0])
        with open(args.threshold_diagnostics_out, newline="") as handle:
            threshold_rows = list(csv.DictReader(handle))
        self.assertEqual(len(threshold_rows), 3)
        self.assertEqual(threshold_rows[0]["source_split"], "fixed")
        self.assertIn("trade_regime_filter", threshold_rows[0])
        self.assertIn("raw_signal_regime_filter_blocked", threshold_rows[0])
        self.assertEqual(threshold_rows[0]["rejected_over_top1_concentration"], "1")
        with open(args.experiment_report_out, encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn("Best near-miss threshold", report)
        self.assertIn("Best near-miss trades", report)

    def test_fixed_split_summary_does_not_inherit_walkforward_rejection(self):
        args = SimpleNamespace(
            split_mode="fixed",
            walk_forward=False,
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            initial_capital=10000.0,
            max_position_fraction=0.1,
            max_volume_fraction=0.01,
            max_trades_per_period=0,
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
            top_k_per_minute=0,
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
            run_summary_out=os.path.join(self.temp.name, "fixed_summary.json"),
            experiment_summary_out=os.path.join(self.temp.name, "fixed_summary.csv"),
            experiment_report_out=os.path.join(self.temp.name, "fixed_report.md"),
            calibration_report_out=os.path.join(self.temp.name, "fixed_calibration.csv"),
            regime_report_out=os.path.join(self.temp.name, "fixed_regime.csv"),
            symbol_report_out=os.path.join(self.temp.name, "fixed_symbol.csv"),
            feature_stability_out=os.path.join(self.temp.name, "fixed_feature_stability.csv"),
            baseline_report_out=os.path.join(self.temp.name, "fixed_baseline.csv"),
            ranking_report_out=os.path.join(self.temp.name, "fixed_ranking.csv"),
            threshold_diagnostics_out=os.path.join(self.temp.name, "fixed_threshold_diagnostics.csv"),
            results_dir=self.temp.name,
            _explicit_flags={"memory_budget_gb", "max_rss_gb"},
        )
        fixed_record = {
            "selected_threshold": 0.5,
            "threshold_selection_status": "valid_threshold",
            "threshold_is_valid": 1,
            "selected_validation_trade_count": 5,
            "validation_predicted_trades": 5,
            "predicted_trades": 5,
            "portfolio_profit": 1.0,
            "portfolio_return": 0.0001,
            "precision": 1.0,
            "recall": 1.0,
            "raw_precision": 0.0,
            "raw_recall": 0.0,
            "accepted": 1,
            "failed_acceptance_checks": "",
            "rejection_reason": "",
            "strategy_strength": "not_checked",
            "brier_score": 0.125,
            "expected_calibration_error": 0.05,
            "max_calibration_error": 0.08,
        }
        pipeline.write_run_summaries(
            args,
            [row(1600000000000)],
            ["ret_1m"],
            "lightgbm",
            {"params": {}, "best_iteration": None},
            fixed_record,
            [],
        )
        with open(args.run_summary_out, encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejection_reason"], "")
        self.assertEqual(summary["walk_forward_summary"]["rejection_reason"], "")
        self.assertAlmostEqual(summary["brier_score"], 0.125)
        with open(args.experiment_summary_out, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["accepted"], "1")
        self.assertEqual(rows[0]["rejection_reason"], "")
        self.assertEqual(rows[0]["brier_score"], "0.125")
        with open(args.experiment_report_out, encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn("Walk-forward acceptance: `n/a (disabled)`", report)
        self.assertIn("Ranking / Tail Diagnostics", report)
        self.assertNotIn("no_walkforward_folds", report)

    def test_print_comparison_omits_missing_logistic_file_warning_when_not_requested(self):
        args = SimpleNamespace(
            logistic_metrics_in=os.path.join(self.temp.name, "missing_logistic.csv"),
            profit_safety="explore",
            acceptance_tier="none",
            _explicit_flags=set(),
        )
        gbdt_record = {
            "model": "gbdt_lightgbm",
            "selected_threshold": 0.5,
            "precision": 0.3,
            "recall": 0.1,
            "portfolio_profit": 10.0,
            "portfolio_return": 0.01,
            "validation_predicted_trades": 5,
            "validation_precision": 0.4,
            "validation_recall": 0.2,
            "validation_portfolio_profit": 12.0,
            "validation_portfolio_return": 0.012,
        }
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            pipeline.print_comparison(gbdt_record, [], args)
        output = stdout.getvalue()
        self.assertIn("Logistic: baseline unavailable for this run", output)
        self.assertNotIn("metrics file not found", output)

    def test_baseline_report_treats_unreadable_logistic_metrics_as_unavailable(self):
        class UnreadableCsv:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                return self

            def __next__(self):
                raise OSError(22, "Invalid argument")

        args = SimpleNamespace(logistic_metrics_in="cloud_placeholder_logistic.csv")

        with mock.patch("gbdt_pipeline.os.path.exists", return_value=True):
            with mock.patch("builtins.open", return_value=UnreadableCsv()):
                rows, summary = pipeline.baseline_report(args)

        logistic_rows = [row for row in rows if row["baseline_name"] == "logistic_baseline"]
        self.assertEqual(len(logistic_rows), 1)
        self.assertEqual(logistic_rows[0]["available"], 0)
        self.assertEqual(logistic_rows[0]["skipped_reason"], "logistic_metrics_unavailable")
        self.assertIn("best_baseline_name", summary)

    def test_configure_output_paths_places_disk_prediction_bundle_cache_under_results_dir(self):
        args = SimpleNamespace(
            results_dir=os.path.join(self.temp.name, "results"),
            prediction_bundle_cache="disk",
            prediction_bundle_cache_dir="",
            predictions_out="kline_growth_predictions_gbdt.csv",
            metrics_out="kline_growth_metrics_gbdt.csv",
            baseline_report_out="kline_growth_baseline_report.csv",
            calibration_report_out="kline_growth_calibration_report.csv",
            experiment_summary_out="kline_growth_experiment_summary.csv",
            experiment_report_out="kline_growth_experiment_report.md",
            feature_importance_out="kline_growth_feature_importance.csv",
            feature_stability_out="kline_growth_feature_stability.csv",
            regime_report_out="kline_growth_regime_report.csv",
            run_summary_out="kline_growth_run_summary.json",
            symbol_filter_diagnostics_out="kline_growth_symbol_filter_diagnostics.csv",
            symbol_report_out="kline_growth_symbol_report.csv",
            walkforward_diagnostics_out="kline_growth_walkforward_diagnostics.csv",
            walkforward_metrics_out="kline_growth_walkforward_metrics.csv",
        )
        pipeline.configure_output_paths(args)
        self.assertEqual(
            args.prediction_bundle_cache_dir,
            os.path.join(os.path.abspath(args.results_dir), "prediction_bundles"),
        )

    @unittest.skipUnless(pipeline.np is not None, "requires numpy-backed prediction cache")
    def test_save_prediction_bundle_cache_ignores_cache_write_permission_errors(self):
        args = SimpleNamespace(
            prediction_bundle_cache="disk",
            prediction_bundle_cache_dir=os.path.join(self.temp.name, "prediction_bundles"),
            prediction_batch_rows=10,
        )
        selected = {}
        rows = [row(1600000000000), row(1600000060000)]
        bundle = pipeline.build_prediction_bundle(probability=[0.1, 0.2])
        with mock.patch("gbdt_pipeline.atomic_write_path", side_effect=PermissionError("denied")):
            pipeline.save_prediction_bundle_cache(selected, rows, "lightgbm", args, "test", bundle)

    def test_inactive_fold_blocker_uses_ranker_score_for_economic_ranking(self):
        rows = [
            symbol_row("AAAUSDT", 1600000000000, trade_return=0.01),
            symbol_row("BBBUSDT", 1600000060000, trade_return=0.01),
        ]
        bundle = pipeline.build_prediction_bundle(ranker_score=[1.2, 0.9])
        result = pipeline.inactive_fold_blocker_check(
            rows,
            bundle,
            pipeline.no_trade_threshold_for_mode("economic_ranking"),
            {"predicted_trades": 0, "raw_signal_trades": 0},
            {"selected_score_threshold": pipeline.no_trade_threshold_for_mode("economic_ranking")},
            SimpleNamespace(objective_mode="economic_ranking"),
            "ranker_score",
        )
        self.assertEqual(result["inactive_blocker_source"], "validation_no_trade_fallback")
        self.assertEqual(result["inactive_blocker_metric"], "ranker_score")
        self.assertEqual(result["inactive_closest_symbol"], "AAAUSDT")

    def test_robustness_summary_flags_profitable_but_fragile_ranker(self):
        args = SimpleNamespace(
            objective_mode="economic_ranking",
            trade_score="ranker_score",
            robustness_gates="warn",
            robustness_gate_action="warn",
            robust_min_trades=10,
            robust_min_active_days=3,
            robust_min_active_symbols=3,
            robust_min_active_months=2,
            robust_max_top_symbol_share=0.70,
            robust_max_top_month_share=0.70,
            robust_min_profit_factor=1.05,
            robust_min_tail_monotonicity=0.60,
            robust_require_positive_top_1pct=True,
            robust_require_positive_top_5pct=True,
            robust_require_positive_top_decile=True,
            robust_min_executed_score_gap=0.0,
            robust_require_positive_total_profit=False,
        )
        summary = {
            "objective_mode": "economic_ranking",
            "trade_score": "ranker_score",
            "walkforward_total_portfolio_profit": 120.0,
            "walkforward_total_predicted_trades": 25,
            "ranking_report_score_sources": "ranker_score",
            "ranking_ranker_score_executed_profit_factor": 1.20,
            "ranking_ranker_score_executed_symbol_count": 1,
            "ranking_ranker_score_executed_month_count": 1,
            "ranking_ranker_score_executed_active_days": 2,
            "ranking_ranker_score_executed_top_symbol_share": 0.92,
            "ranking_ranker_score_executed_top_month_share": 0.95,
            "ranking_ranker_score_net_return_monotonicity": 0.50,
            "ranking_ranker_score_top_1pct_rows": 5,
            "ranking_ranker_score_top_1pct_avg_net_return": -0.001,
            "ranking_ranker_score_top_5pct_rows": 8,
            "ranking_ranker_score_top_5pct_avg_net_return": -0.002,
            "ranking_ranker_score_top_decile_rows": 10,
            "ranking_ranker_score_top_decile_avg_net_return": -0.003,
            "walk_forward_aggregate_metrics": {
                "executed_score_win_loss_gap": -0.01,
            },
        }
        robust = pipeline.robustness_summary(summary, args)
        self.assertEqual(robust["robustness_gate_status"], "failed")
        self.assertEqual(robust["profitable_but_fragile"], 1)
        self.assertIn("top_symbol_share", robust["robustness_failed_checks"])
        self.assertIn("top_1pct_net_return", robust["robustness_failed_checks"])
        self.assertIn("top_5pct_net_return", robust["robustness_failed_checks"])
        self.assertIn("top_decile_net_return", robust["robustness_failed_checks"])
        self.assertIn("executed_score_gap", robust["robustness_failed_checks"])

    def test_parser_accepts_robustness_gate_controls(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--robustness-gate-action", "reject",
            "--robust-min-trades", "20",
            "--robust-min-active-days", "5",
            "--robust-min-active-symbols", "4",
            "--robust-min-active-months", "3",
            "--robust-max-top-symbol-share", "0.55",
            "--robust-max-top-month-share", "0.60",
            "--robust-min-profit-factor", "1.2",
            "--robust-min-tail-monotonicity", "0.7",
            "--robust-require-positive-top-1pct",
            "--robust-require-positive-top-5pct",
            "--robust-require-positive-top-decile",
            "--robust-min-executed-score-gap", "0.0",
        ])
        self.assertEqual(args.robustness_gate_action, "reject")
        self.assertEqual(args.robust_min_trades, 20)
        self.assertAlmostEqual(args.robust_max_top_symbol_share, 0.55)
        self.assertTrue(args.robust_require_positive_top_1pct)
        self.assertTrue(args.robust_require_positive_top_5pct)
        self.assertAlmostEqual(args.robust_min_executed_score_gap, 0.0)

    def test_parser_accepts_ranker_upper_quantile_control(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--objective-mode", "economic_ranking",
            "--trade-score", "ranker_score",
            "--ranker-score-upper-quantile", "0.9",
        ])
        self.assertAlmostEqual(args.ranker_score_upper_quantile, 0.9)

    def test_output_column_schemas_are_unique(self):
        self.assertEqual(len(pipeline.METRIC_COLUMNS), len(set(pipeline.METRIC_COLUMNS)))
        self.assertEqual(len(pipeline.THRESHOLD_DIAGNOSTIC_COLUMNS), len(set(pipeline.THRESHOLD_DIAGNOSTIC_COLUMNS)))
        self.assertEqual(len(pipeline.WALKFORWARD_DIAGNOSTIC_COLUMNS), len(set(pipeline.WALKFORWARD_DIAGNOSTIC_COLUMNS)))
        self.assertEqual(len(run_experiments.RESULT_COLUMNS), len(set(run_experiments.RESULT_COLUMNS)))

    def test_experiment_report_spells_out_profitable_but_fragile(self):
        report_path = os.path.join(self.temp.name, "fragile_report.md")
        pipeline.write_experiment_report(report_path, {
            "accepted": 1,
            "rejection_reason": "Profitable but fragile.",
            "robustness_gate_status": "failed",
            "profitable_but_fragile": 1,
            "robustness_gates": "warn",
            "robustness_gate_action": "warn",
            "robustness_failed_checks": "top_month_share 0.9500 > 0.7000",
            "robustness_strength": "profitable_but_fragile",
        })
        with open(report_path, encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn("Profitable but fragile.", report)
        self.assertIn("Robustness Gates", report)


class ResearchDiagnosticsTests(unittest.TestCase):
    def trade_row(self, symbol, open_time, trade_return, score, position_size=1000.0, label=1, day="2026-01-01"):
        return {
            "symbol": symbol,
            "open_time": str(open_time),
            "trade_day": day,
            "dynamic_trade_return": str(trade_return),
            "trade_return": str(trade_return),
            "position_size": str(position_size),
            "trade_score": str(score),
            "expected_value": str(score),
            "predicted_net_return": str(score),
            "probability": "0.6" if label else "0.4",
            "calibrated_probability": "0.55" if label else "0.45",
            "label": str(label),
        }

    def test_symbol_exposure_reports_trade_capital_profit_and_loss_shares(self):
        rows = [
            self.trade_row("AAA", 1, 0.0315, 0.9),
            self.trade_row("AAA", 2, -0.0185, 0.8, label=0),
            self.trade_row("BBB", 3, 0.0115, 0.7),
        ]
        exposure = research_diagnostics.symbol_exposure(rows)
        by_symbol = {row["symbol"]: row for row in exposure}
        self.assertAlmostEqual(by_symbol["AAA"]["trade_share"], 2 / 3)
        self.assertAlmostEqual(by_symbol["AAA"]["capital_share"], 2 / 3)
        self.assertAlmostEqual(by_symbol["AAA"]["net_profit"], 10.0)
        self.assertAlmostEqual(by_symbol["BBB"]["net_profit"], 10.0)
        self.assertGreater(by_symbol["AAA"]["loss_share"], 0.99)

    def test_leave_one_symbol_out_recomputes_return_and_trade_count(self):
        rows = [
            self.trade_row("AAA", 1, 0.0315, 0.9),
            self.trade_row("BBB", 2, 0.0115, 0.8),
        ]
        result = research_diagnostics.leave_one_symbol_out_rows(rows, ["AAA"], "fixed_test")
        self.assertEqual(result[0]["removed_trade_count"], 1)
        self.assertAlmostEqual(result[0]["portfolio_profit"], 10.0)
        self.assertAlmostEqual(result[0]["portfolio_return"], 0.001)
        self.assertAlmostEqual(result[0]["baseline_portfolio_return"], 0.004)

    def test_leave_one_fold_out_uses_remaining_folds_only(self):
        rows = [
            {"split": "walkforward_fold_1", "portfolio_profit": "100", "portfolio_return": "0.01", "predicted_trades": "2"},
            {"split": "walkforward_fold_2", "portfolio_profit": "-50", "portfolio_return": "-0.005", "predicted_trades": "1"},
            {"split": "walkforward_fold_3", "portfolio_profit": "0", "portfolio_return": "0", "predicted_trades": "0"},
        ]
        result = {
            row["removed_fold"]: row
            for row in research_diagnostics.leave_one_fold_out_rows(rows)
        }
        self.assertAlmostEqual(result["walkforward_fold_1"]["remaining_mean_fold_return"], -0.0025)
        self.assertEqual(result["walkforward_fold_2"]["remaining_active_folds"], 1)
        self.assertEqual(result["walkforward_fold_2"]["remaining_profitable_folds"], 1)

    def test_score_deciles_report_monotonicity_and_top_symbol_concentration(self):
        rows = [
            self.trade_row("AAA", i, -0.0185 if i < 5 else 0.0315, i / 10.0, label=int(i >= 5))
            for i in range(10)
        ]
        deciles, monotonicity = research_diagnostics.score_decile_rows(rows, "walkforward", ["trade_score"])
        self.assertEqual(len(deciles), 10)
        self.assertAlmostEqual(monotonicity[("walkforward", "trade_score")], 1.0)
        self.assertEqual(deciles[-1]["top_symbol"], "AAA")
        self.assertAlmostEqual(deciles[-1]["symbol_concentration_top1"], 1.0)

    def test_threshold_stability_uses_neighbor_performance_and_trade_variation(self):
        rows = [
            {
                "available": "1",
                "source_split": "fixed",
                "fold_index": "0",
                "candidate_index": str(index),
                "threshold": str(0.1 * index),
                "predicted_trades": str(10 + index),
                "average_net_return_after_costs": str(0.01 - 0.001 * abs(index - 3)),
                "symbol_trade_concentration_top1": str(0.2 + 0.01 * index),
                "rejection_reason_flags": "",
            }
            for index in range(1, 6)
        ]
        result = research_diagnostics.threshold_stability_rows(rows, {("fixed", 0): 0.3})
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["closest_candidate_threshold"], 0.3)
        self.assertGreater(result[0]["threshold_stability_score"], 0.5)
        self.assertEqual(result[0]["neighbor_min_trades"], 11)
        self.assertEqual(result[0]["neighbor_max_trades"], 15)

    def test_cost_stress_reduces_return_by_extra_round_trip_cost(self):
        rows = [
            self.trade_row("AAA", 1, 0.0315, 0.9),
            self.trade_row("BBB", 2, 0.0315, 0.8),
        ]
        stress = {
            row["cost_scenario"]: row
            for row in research_diagnostics.cost_stress_rows(rows, "fixed_test")
        }
        self.assertAlmostEqual(stress["baseline_cost"]["portfolio_profit"], 60.0)
        self.assertAlmostEqual(stress["double_cost"]["portfolio_profit"], 57.0)
        self.assertLess(stress["double_cost"]["portfolio_return"], stress["baseline_cost"]["portfolio_return"])

    def test_composite_score_penalizes_concentration_and_fragility(self):
        summary = {
            "profit_factor": 1.1,
            "symbol_profit_concentration_top1": 0.8,
            "ranking_trade_score_net_return_monotonicity": 0.25,
        }
        walk_metrics = [
            {"split": "walkforward_fold_1", "portfolio_return": "0.02", "portfolio_profit": "200", "predicted_trades": "2"},
            {"split": "walkforward_fold_2", "portfolio_return": "-0.08", "portfolio_profit": "-800", "predicted_trades": "3"},
        ]
        threshold = [{"threshold_stability_score": 0.4}]
        leave_symbols = [{"source": "walkforward", "portfolio_return": "-0.01"}]
        leave_folds = [{"remaining_mean_fold_return": "-0.08"}]
        costs = [{"source": "walkforward", "cost_scenario": "double_cost", "portfolio_return": "-0.02"}]
        components = research_diagnostics.composite_robustness_score(
            summary,
            walk_metrics,
            threshold,
            leave_symbols,
            leave_folds,
            costs,
            {("walkforward", "trade_score"): 0.25},
        )
        score = components[-1]["component_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 0.6)

    def test_candidate_rich_command_switches_output_mode_and_directory(self):
        profile = {
            "exact_command": (
                r"python gbdt_pipeline.py --prediction-output-mode trades "
                r"--results-dir results\chrono_fix_rerun_20260714_manual\hyb_late_tuned_6m_sf0p04_f88 "
                r"--run-summary-out results\chrono_fix_rerun_20260714_manual\hyb_late_tuned_6m_sf0p04_f88\kline_growth_run_summary.json"
            )
        }
        command = research_diagnostics.candidate_rich_command_from_profile(profile)
        self.assertIn("--prediction-output-mode candidates", command)
        self.assertIn(r"results\candidate_rich_20260714_baseline_candidates", command)
        self.assertIn("--predictions-out kline_growth_predictions_gbdt_candidates.csv", command)
        self.assertIn("--walk-predictions-out kline_growth_predictions_gbdt_walkforward_candidates.csv", command)
        self.assertNotIn("--prediction-output-mode all", command)

    def test_machine_readable_run_status_distinguishes_rejected_strategy(self):
        summary = {"accepted": 0, "rejection_reason": "worst_fold_return -0.0877 < -0.0800"}
        args = SimpleNamespace(acceptance_tier="exploration")
        pipeline.apply_machine_readable_run_status(summary, args)
        self.assertEqual(summary["pipeline_execution_status"], "completed")
        self.assertEqual(summary["strategy_acceptance_status"], "rejected")
        self.assertEqual(summary["strategy_rejected"], 1)
        self.assertEqual(summary["rejection_reasons"], "worst_fold_return -0.0877 < -0.0800")

    def test_machine_readable_run_status_rejects_failed_robustness_gate(self):
        summary = {
            "accepted": 1,
            "walkforward_gate_status": "passed",
            "robustness_gate_failed": 1,
            "robustness_failed_checks": "tail_monotonicity 0.5556 < 0.6000",
        }
        args = SimpleNamespace(acceptance_tier="exploration")
        pipeline.apply_machine_readable_run_status(summary, args)
        self.assertEqual(summary["pipeline_execution_status"], "completed")
        self.assertEqual(summary["strategy_acceptance_status"], "rejected")
        self.assertEqual(summary["strategy_rejected"], 1)
        self.assertIn("tail_monotonicity", summary["rejection_reasons"])

    def test_machine_readable_run_status_can_mark_not_checked(self):
        summary = {"accepted": 1, "walkforward_gate_status": "not_applicable"}
        args = SimpleNamespace(acceptance_tier="none")
        pipeline.apply_machine_readable_run_status(summary, args)
        self.assertEqual(summary["strategy_acceptance_status"], "not_checked")
        self.assertEqual(summary["strategy_rejected"], 0)

    def test_exploration_acceptance_rejects_fragile_positive_mean_walkforward(self):
        returns = [
            -0.00205,
            0.0,
            -0.02046,
            -0.01317,
            -0.00022,
            -0.00198,
            0.0,
            0.06309,
            -0.00958,
            0.01451,
        ]
        trades = [6, 0, 226, 249, 2, 2, 0, 154, 661, 38]
        records = []
        for index, (portfolio_return, trade_count) in enumerate(zip(returns, trades), 1):
            records.append({
                "split": "walkforward_fold_{}".format(index),
                "portfolio_profit": portfolio_return * 10000.0,
                "portfolio_return": portfolio_return,
                "precision": 0.0,
                "max_capital_drawdown": 0.01,
                "predicted_trades": trade_count,
                "meta_filter_enabled": 0,
            })
        args = SimpleNamespace(
            overactive_trade_threshold=150,
            acceptance_tier="exploration",
            min_profitable_fold_rate=0.0,
            min_median_fold_return=-999.0,
            max_worst_fold_drawdown=1.0,
            min_active_profitable_fold_rate=0.5,
            min_walkforward_total_trades=10,
        )

        summary = pipeline.walkforward_acceptance_summary(records, args)

        self.assertEqual(summary["accepted"], 0)
        self.assertEqual(summary["strategy_strength"], "rejected")
        self.assertIn("active_profitable_fold_rate", summary["failed_acceptance_checks"])
        self.assertIn("median_portfolio_return", summary["failed_acceptance_checks"])

    def test_resolved_run_acceptance_rejects_no_valid_fixed_threshold(self):
        fixed_record = {
            "threshold_selection_status": "no_valid_threshold",
            "threshold_is_valid": 0,
            "selected_validation_trade_count": 0,
            "validation_predicted_trades": 0,
            "predicted_trades": 0,
        }
        walkforward_summary = {
            "accepted": 1,
            "acceptance_tier": "exploration",
            "failed_acceptance_checks": "",
            "rejection_reason": "",
            "strategy_strength": "exploration_pass",
        }
        args = SimpleNamespace(
            walk_forward=True,
            acceptance_tier="exploration",
            min_fixed_test_trades=1,
        )

        acceptance = pipeline.resolved_run_acceptance(fixed_record, walkforward_summary, args)

        self.assertEqual(acceptance["accepted"], 0)
        self.assertEqual(acceptance["strategy_strength"], "rejected")
        self.assertIn("no_valid_threshold", acceptance["failed_acceptance_checks"])
        self.assertIn("zero_fixed_test_trades", acceptance["failed_acceptance_checks"])

    def test_research_diagnostics_candidate_artifacts_use_selected_rows_for_accounting(self):
        rows = [
            self.trade_row("AAA", 1, 0.0315, 0.9),
            self.trade_row("BBB", 2, -0.0185, 0.8, label=0),
        ]
        rows[0]["selected_by_topk"] = "1"
        rows[0]["predicted"] = "1"
        rows[1]["selected_by_topk"] = "0"
        rows[1]["predicted"] = "1"
        selected = research_diagnostics.selected_prediction_rows(rows)
        metrics = research_diagnostics.portfolio_metrics(selected)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["symbol"], "AAA")
        self.assertAlmostEqual(metrics["portfolio_profit"], 30.0)

    def test_offline_candidate_research_prefers_candidate_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_path = root / "kline_growth_predictions_gbdt_candidates.csv"
            all_path = root / "kline_growth_predictions_gbdt_all.csv"
            all_path.write_text("all\n", encoding="utf-8")
            self.assertEqual(
                offline_candidate_research.first_existing_path([candidate_path, all_path]),
                all_path,
            )
            candidate_path.write_text("candidates\n", encoding="utf-8")
            self.assertEqual(
                offline_candidate_research.first_existing_path([candidate_path, all_path]),
                candidate_path,
            )

    def test_offline_candidate_research_preserves_artifact_fold_identity(self):
        first = offline_candidate_research.compact_artifact_candidate(
            {
                "candidate_id": "candidate-fold-1",
                "stable_row_id": "walkforward:fold_1:1173355",
                "source": "walkforward",
                "symbol": "AAA",
                "month": "2025-07",
                "fold_id": "fold_1",
                "row_position": "1173355",
                "selected_by_score_before_execution": "1",
                "executed": "1",
                "position_size": "100",
                "trade_return": "0.01",
            },
            "walkforward",
            {},
        )
        second = offline_candidate_research.compact_artifact_candidate(
            {
                "candidate_id": "candidate-fold-3",
                "stable_row_id": "walkforward:fold_3:1173355",
                "source": "walkforward",
                "symbol": "BBB",
                "month": "2025-09",
                "fold_id": "fold_3",
                "row_position": "1173355",
                "selected_by_score_before_execution": "1",
                "executed": "1",
                "position_size": "100",
                "trade_return": "0.02",
            },
            "walkforward",
            {},
        )

        self.assertEqual(first["_fold_index"], "1")
        self.assertEqual(first["_fold_split"], "walkforward_fold_1")
        self.assertEqual(second["_fold_index"], "3")
        self.assertEqual(second["_fold_split"], "walkforward_fold_3")
        self.assertNotEqual(
            offline_candidate_research.row_identity(first),
            offline_candidate_research.row_identity(second),
        )
        overlap = offline_candidate_research.selection_overlap(
            "walkforward",
            "baseline_selected_by_topk",
            [first, second],
            [first, second],
        )
        self.assertEqual(overlap["selected_trades"], 2)
        self.assertEqual(overlap["overlap_trades"], 2)
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp) / "candidate_snapshot_walkforward.csv"
            offline_candidate_research.write_csv(
                snapshot,
                [first, second],
                offline_candidate_research.CANDIDATE_FIELDS,
            )
            reloaded = offline_candidate_research.read_csv_rows(snapshot)
        self.assertEqual(reloaded[0]["_candidate_id"], "candidate-fold-1")
        self.assertEqual(reloaded[1]["_stable_row_id"], "walkforward:fold_3:1173355")
        self.assertNotEqual(
            offline_candidate_research.row_identity(reloaded[0]),
            offline_candidate_research.row_identity(reloaded[1]),
        )

    def test_offline_candidate_research_refreshes_stale_artifact_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "kline_growth_walkforward_candidates.csv.gz"
            snapshot = root / "candidate_snapshot_walkforward.csv"
            candidate_artifacts.write_candidate_artifact(
                artifact,
                [
                    {
                        "candidate_id": "candidate-fold-1",
                        "stable_row_id": "walkforward:fold_1:1173355",
                        "source": "walkforward",
                        "symbol": "AAA",
                        "month": "2025-07",
                        "month_index": 1,
                        "open_time": 1752247680000,
                        "fold_id": "fold_1",
                        "row_position": 1173355,
                        "label": 1,
                        "trade_return": 0.01,
                        "final_preselection_score": 0.9,
                        "selected_by_score_before_execution": 1,
                        "executed": 1,
                        "position_size": 100.0,
                        "rank_within_decision_bucket": 1,
                    }
                ],
                metadata={"source_run_id": "stale-snapshot-test"},
            )
            snapshot.write_text(
                "_source,_row_index,_fold_index,_fold_split,symbol,selected_by_topk,predicted\n"
                "walkforward,1173355,0,,AAA,1,1\n",
                encoding="utf-8",
            )

            rows, stats = offline_candidate_research.extract_candidate_artifact(
                artifact,
                snapshot,
                "walkforward",
                force=False,
            )

            self.assertEqual(stats["loaded_from_snapshot"], 0)
            self.assertEqual(rows[0]["_fold_index"], "1")
            reloaded = offline_candidate_research.read_csv_rows(snapshot)
            self.assertEqual(reloaded[0]["_candidate_id"], "candidate-fold-1")
            self.assertEqual(reloaded[0]["_stable_row_id"], "walkforward:fold_1:1173355")

    def test_offline_candidate_research_rejects_late_artifact_for_prefilter_ablation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "late_candidates.csv.gz"
            snapshot = root / "candidate_snapshot_fixed.csv"
            candidate_artifacts.write_candidate_artifact(
                artifact,
                [
                    {
                        "candidate_id": "late-1",
                        "source": "fixed",
                        "symbol": "AAA",
                        "month": "2025-07",
                        "open_time": 1752247680000,
                        "fold_id": "fixed",
                        "row_position": 1,
                        "selected_by_score_before_execution": 1,
                        "executed": 1,
                        "position_size": 100.0,
                        "trade_return": 0.01,
                    }
                ],
                metadata={
                    "source_run_id": "late-stage-test",
                    "candidate_serialization_stage": "post_selection",
                },
            )

            with self.assertRaises(RuntimeError) as raised:
                offline_candidate_research.extract_candidate_artifact(
                    artifact,
                    snapshot,
                    "fixed",
                    require_pre_filter=True,
                )

            self.assertIn("pre-filter", str(raised.exception))

    def test_offline_candidate_pool_uses_threshold_qualified_prefilter_rows(self):
        rows = [
            {
                "_candidate_id": "current",
                "symbol": "AAA",
                "open_time": "1600000000000",
                "raw_signal": "0",
                "selected_by_threshold": "1",
                "selected_by_score_edge": "1",
                "selected_by_symbol_filter": "1",
                "candidate_serialization_stage": "pre_score_edge_pre_symbol_filter",
                "trade_score": "0.9",
            },
            {
                "_candidate_id": "threshold_blocked",
                "symbol": "BBB",
                "open_time": "1600000000000",
                "raw_signal": "0",
                "selected_by_threshold": "0",
                "selected_by_score_edge": "1",
                "selected_by_symbol_filter": "1",
                "candidate_serialization_stage": "pre_score_edge_pre_symbol_filter",
                "trade_score": "1.0",
            },
            {
                "_candidate_id": "score_edge_blocked",
                "symbol": "CCC",
                "open_time": "1600000000000",
                "raw_signal": "0",
                "selected_by_threshold": "1",
                "selected_by_score_edge": "0",
                "selected_by_symbol_filter": "1",
                "candidate_serialization_stage": "pre_score_edge_pre_symbol_filter",
                "trade_score": "0.8",
            },
            {
                "_candidate_id": "symbol_blocked",
                "symbol": "DDD",
                "open_time": "1600000000000",
                "raw_signal": "0",
                "selected_by_threshold": "1",
                "selected_by_score_edge": "1",
                "selected_by_symbol_filter": "0",
                "candidate_serialization_stage": "pre_score_edge_pre_symbol_filter",
                "trade_score": "0.7",
            },
        ]

        current = offline_candidate_research.candidate_pool(rows)
        no_score_edge = offline_candidate_research.candidate_pool(rows, score_edge_filter=False)
        no_symbol_filter = offline_candidate_research.candidate_pool(rows, symbol_filter=False)
        no_filters = offline_candidate_research.candidate_pool(
            rows,
            score_edge_filter=False,
            symbol_filter=False,
        )

        self.assertEqual([row["_candidate_id"] for row in current], ["current"])
        self.assertEqual(
            {row["_candidate_id"] for row in no_score_edge},
            {"current", "score_edge_blocked"},
        )
        self.assertEqual(
            {row["_candidate_id"] for row in no_symbol_filter},
            {"current", "symbol_blocked"},
        )
        self.assertEqual(
            {row["_candidate_id"] for row in no_filters},
            {"current", "score_edge_blocked", "symbol_blocked"},
        )

    def test_offline_candidate_artifact_research_scope_streams_relevant_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "kline_growth_walkforward_candidates.csv.gz"
            snapshot = root / "candidate_snapshot_walkforward.csv"
            candidate_artifacts.write_candidate_artifact(
                artifact,
                [
                    {
                        "candidate_id": "threshold-qualified",
                        "stable_row_id": "walkforward:fold_1:1",
                        "source": "walkforward",
                        "symbol": "AAA",
                        "month": "2025-07",
                        "open_time": 1752247680000,
                        "fold_id": "fold_1",
                        "row_position": 1,
                        "selected_by_threshold": 1,
                        "selected_by_score_edge": 1,
                        "selected_by_symbol_filter": 1,
                        "raw_signal": 0,
                        "selected_by_score_before_execution": 0,
                        "executed": 0,
                        "trade_return": 0.01,
                        "final_preselection_score": 0.9,
                    },
                    {
                        "candidate_id": "not-research-relevant",
                        "stable_row_id": "walkforward:fold_1:2",
                        "source": "walkforward",
                        "symbol": "BBB",
                        "month": "2025-07",
                        "open_time": 1752247680000,
                        "fold_id": "fold_1",
                        "row_position": 2,
                        "selected_by_threshold": 0,
                        "selected_by_score_edge": 0,
                        "selected_by_symbol_filter": 1,
                        "raw_signal": 0,
                        "selected_by_score_before_execution": 0,
                        "executed": 0,
                        "trade_return": 0.02,
                        "final_preselection_score": 1.0,
                    },
                    {
                        "candidate_id": "executed",
                        "stable_row_id": "walkforward:fold_1:3",
                        "source": "walkforward",
                        "symbol": "CCC",
                        "month": "2025-07",
                        "open_time": 1752247680000,
                        "fold_id": "fold_1",
                        "row_position": 3,
                        "selected_by_threshold": 0,
                        "selected_by_score_edge": 0,
                        "selected_by_symbol_filter": 1,
                        "raw_signal": 1,
                        "selected_by_score_before_execution": 1,
                        "executed": 1,
                        "position_size": 100.0,
                        "trade_return": 0.03,
                        "final_preselection_score": 0.8,
                    },
                ],
                metadata={
                    "source_run_id": "research-scope-test",
                    "candidate_serialization_stage": "pre_score_edge_pre_symbol_filter",
                    "full_candidate_coverage": True,
                },
            )

            rows, stats = offline_candidate_research.extract_candidate_artifact(
                artifact,
                snapshot,
                "walkforward",
                force=False,
                require_pre_filter=True,
                row_scope="research_relevant",
            )

            self.assertEqual(stats["rows_scanned"], 3)
            self.assertEqual(stats["candidate_rows"], 2)
            self.assertEqual(stats["artifact_row_scope"], "research_relevant")
            self.assertEqual(
                {row["_candidate_id"] for row in rows},
                {"threshold-qualified", "executed"},
            )
            reloaded = offline_candidate_research.read_csv_rows(snapshot)
            self.assertEqual(len(reloaded), 2)
            self.assertTrue(all(row["_artifact_row_scope"] == "research_relevant" for row in reloaded))

    def test_offline_research_rejects_overlapping_confirmation_folds(self):
        args = SimpleNamespace(discovery_folds="1-2", selection_folds="3-4", confirmation_folds="4-5")
        with self.assertRaises(ValueError):
            offline_candidate_research.build_research_fold_sets(args, 10)

    def test_offline_research_split_matrix_keeps_confirmation_separate(self):
        fold_rows = [
            {"experiment": "baseline", "fold_index": "1", "portfolio_profit": "100", "portfolio_return": "0.01", "predicted_trades": "10"},
            {"experiment": "baseline", "fold_index": "2", "portfolio_profit": "-50", "portfolio_return": "-0.005", "predicted_trades": "5"},
            {"experiment": "baseline", "fold_index": "3", "portfolio_profit": "300", "portfolio_return": "0.03", "predicted_trades": "7"},
        ]
        split_rows = offline_candidate_research.research_split_matrix_rows(
            fold_rows,
            {"selection": {1, 2}, "confirmation": {3}},
        )
        by_phase = {row["phase"]: row for row in split_rows}
        self.assertAlmostEqual(by_phase["selection"]["portfolio_return"], 0.0025)
        self.assertAlmostEqual(by_phase["confirmation"]["portfolio_return"], 0.03)
        self.assertEqual(by_phase["confirmation"]["folds"], "3")

    def test_offline_research_decision_rejects_underpowered_tuning_split(self):
        experiment_rows = [
            {"source": "walkforward", "experiment": "baseline_executed_trades", "portfolio_return": "0.0087", "trade_count": "1755"},
            {"source": "walkforward", "experiment": "score_replay_trade_score", "portfolio_return": "0.0081", "trade_count": "1755"},
        ]
        split_rows = [
            {"experiment": "baseline_executed_trades", "phase": "selection", "trade_count": "4", "active_folds": "2"},
            {"experiment": "score_replay_trade_score", "phase": "selection", "trade_count": "4", "active_folds": "2"},
            {"experiment": "baseline_executed_trades", "phase": "confirmation", "trade_count": "853", "active_folds": "3"},
            {"experiment": "score_replay_trade_score", "phase": "confirmation", "trade_count": "853", "active_folds": "3"},
        ]
        bucket_rows = [
            {
                "source": "walkforward",
                "bucket_scope": "execution_open_time",
                "selected_replaceable_share": "0.0547",
            }
        ]
        cost_rows = [
            {
                "source": "walkforward",
                "experiment": "baseline_executed_trades",
                "cost_scenario": "double_cost",
                "portfolio_return": "-0.0098",
            }
        ]

        decision = offline_candidate_research.research_decision_rows(
            experiment_rows,
            split_rows,
            bucket_rows,
            cost_rows,
        )[0]

        self.assertEqual(decision["decision"], "do_not_promote_offline_tuning")
        self.assertEqual(decision["best_walkforward_experiment"], "baseline_executed_trades")
        self.assertIn("selection_underpowered", decision["reasons"])
        self.assertIn("candidate_reselection_sparse", decision["reasons"])
        self.assertIn("baseline_best", decision["reasons"])
        self.assertIn("cost_stress_failed", decision["reasons"])

    def test_offline_candidate_bucket_audit_separates_execution_and_artifact_buckets(self):
        base_minute = 26666640
        rows = [
            {
                "_candidate_id": "a",
                "symbol": "AAA",
                "open_time": str(base_minute * 60000),
                "decision_bucket_id": str(base_minute),
                "configured_top_k": "1",
                "raw_signal": "1",
                "selected_by_topk": "1",
                "predicted": "1",
                "trade_return": "0.01",
                "trade_score": "0.90",
            },
            {
                "_candidate_id": "b",
                "symbol": "BBB",
                "open_time": str((base_minute + 10) * 60000),
                "decision_bucket_id": str(base_minute),
                "configured_top_k": "1",
                "raw_signal": "1",
                "selected_by_topk": "1",
                "predicted": "1",
                "trade_return": "0.01",
                "trade_score": "0.80",
            },
        ]
        audit = {
            row["bucket_scope"]: row
            for row in offline_candidate_research.candidate_bucket_audit_rows("walkforward", rows)
        }

        self.assertEqual(audit["execution_open_time"]["bucket_count"], 2)
        self.assertEqual(audit["execution_open_time"]["bucket_candidate_count_max"], 1)
        self.assertAlmostEqual(audit["execution_open_time"]["selected_replaceable_share"], 0.0)
        self.assertEqual(audit["artifact_decision_bucket"]["bucket_count"], 1)
        self.assertEqual(audit["artifact_decision_bucket"]["bucket_candidate_count_max"], 2)
        self.assertAlmostEqual(audit["artifact_decision_bucket"]["selected_replaceable_share"], 1.0)

    def test_offline_candidate_research_keeps_executed_baseline_separate(self):
        rows = [
            {
                "_candidate_id": "executed",
                "symbol": "AAA",
                "open_time": "1600000000000",
                "raw_signal": "1",
                "selected_by_topk": "1",
                "predicted": "1",
                "position_size": "100",
                "trade_return": "0.01",
                "trade_score": "0.9",
            },
            {
                "_candidate_id": "ranked_not_executed",
                "symbol": "BBB",
                "open_time": "1600000060000",
                "raw_signal": "1",
                "selected_by_topk": "1",
                "predicted": "0",
                "position_size": "",
                "trade_return": "0.02",
                "trade_score": "0.8",
            },
        ]
        experiments = offline_candidate_research.selected_sets(rows, {})

        self.assertEqual(len(experiments["baseline_executed_trades"]), 1)
        self.assertEqual(len(experiments["baseline_selected_by_topk"]), 2)
        self.assertEqual(experiments["baseline_executed_trades"][0]["symbol"], "AAA")
        self.assertEqual(experiments["baseline_selected_by_topk"][1]["_position_size_fallback"], "1")

    def test_created_file_inventory_writes_absolute_paths_and_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "created_files.csv"
            artifact = root / "artifact.csv"
            artifact.write_text("abc\n", encoding="utf-8")
            summary = created_file_inventory.write_inventory(
                output,
                [("artifact", artifact), ("missing", root / "missing.csv")],
            )
            self.assertEqual(summary["rows"], 1)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["label"], "artifact")
            self.assertEqual(rows[0]["path"], str(artifact.resolve()))
            self.assertEqual(int(rows[0]["size_bytes"]), artifact.stat().st_size)

    def test_pipeline_candidate_artifact_inventory_prints_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "candidates.csv.gz"
            candidate_artifacts.write_candidate_artifact(
                artifact,
                [{"candidate_id": "abc", "executed": 1}],
                metadata={"source_run_id": "test"},
            )
            args = SimpleNamespace(
                candidate_artifact_out=str(artifact),
                walk_candidate_artifact_out="",
                results_dir="",
            )
            with mock.patch("builtins.print") as printed:
                summary = pipeline.write_candidate_artifact_created_files_inventory(args)
            output = root / "candidate_artifact_created_files.csv"
            self.assertEqual(summary["path"], str(output.resolve()))
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["label"] for row in rows}, {"fixed_candidate_artifact", "fixed_candidate_artifact_manifest"})
            printed.assert_called()
            self.assertIn(str(output.resolve()), printed.call_args.args[0])

    def test_offline_research_inventory_prints_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in (
                "candidate_snapshot_fixed.csv",
                "candidate_snapshot_walkforward.csv",
                "candidate_snapshot_summary.csv",
                "offline_experiment_matrix.csv",
                "offline_cost_stress.csv",
                "offline_leave_one_symbol_out.csv",
                "offline_leave_one_fold_out.csv",
                "offline_walkforward_fold_metrics.csv",
                "offline_research_split_matrix.csv",
                "offline_selection_overlap.csv",
                "offline_candidate_bucket_audit.csv",
                "offline_selected_trades.csv",
                "position_size_fallbacks.json",
                "research_report.md",
            ):
                (root / name).write_text("x\n", encoding="utf-8")
            with mock.patch("builtins.print") as printed:
                summary = offline_candidate_research.write_offline_research_created_files_inventory(root)
            output = root / "offline_research_created_files.csv"
            self.assertEqual(summary["rows"], 14)
            self.assertEqual(summary["path"], str(output.resolve()))
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn(str((root / "research_report.md").resolve()), {row["path"] for row in rows})
            printed.assert_called()
            self.assertIn(str(output.resolve()), printed.call_args.args[0])

    def test_offline_research_accepts_fixed_only_candidate_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            out_dir = root / "offline"
            run_dir.mkdir()
            (run_dir / "kline_growth_metrics_gbdt.csv").write_text(
                "portfolio_profit,portfolio_return,predicted_trades\n1.0,0.0001,1\n",
                encoding="utf-8",
            )
            artifact = run_dir / "kline_growth_candidates.csv.gz"
            candidate_artifacts.write_candidate_artifact(
                artifact,
                [
                    {
                        "candidate_id": "fixed-1",
                        "source": "fixed",
                        "stable_row_id": "fixed:0",
                        "symbol": "AAA",
                        "month": "2020-01",
                        "month_index": 0,
                        "open_time": 1600000000000,
                        "fold_id": "fixed",
                        "decision_bucket_id": 26666666,
                        "row_position": 0,
                        "label": 1,
                        "trade_return": 0.01,
                        "final_preselection_score": 0.9,
                        "selected_by_score_before_execution": 1,
                        "executed": 1,
                        "position_size": 100.0,
                        "rank_within_decision_bucket": 1,
                        "candidate_count_within_decision_bucket": 1,
                    }
                ],
                metadata={"source_run_id": "fixed-only-test"},
            )
            argv = [
                "offline_candidate_research.py",
                "--run-dir",
                str(run_dir),
                "--out-dir",
                str(out_dir),
                "--fixed-candidate-artifact",
                str(artifact),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("builtins.print"):
                    self.assertEqual(offline_candidate_research.main(), 0)
            self.assertTrue((out_dir / "offline_research_created_files.csv").exists())
            self.assertFalse((out_dir / "candidate_snapshot_walkforward.csv").exists())

    def test_candidate_artifact_round_trip_preserves_rejected_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "candidates.csv.gz"
            records = [
                {
                    "candidate_id": candidate_artifacts.deterministic_candidate_id("run", 1),
                    "source": "fixed",
                    "stable_row_id": "fixed:0",
                    "symbol": "AAA",
                    "month": "2020-01",
                    "month_index": 0,
                    "open_time": 1600000000000,
                    "fold_id": "fixed",
                    "decision_bucket_id": 26666666,
                    "row_position": 0,
                    "label": 1,
                    "trade_return": 0.02,
                    "final_preselection_score": 0.9,
                    "selected_by_score_before_execution": 1,
                    "executed": 1,
                    "position_size": 100.0,
                    "rank_within_decision_bucket": 1,
                    "candidate_count_within_decision_bucket": 2,
                },
                {
                    "candidate_id": candidate_artifacts.deterministic_candidate_id("run", 2),
                    "source": "fixed",
                    "stable_row_id": "fixed:1",
                    "symbol": "BBB",
                    "month": "2020-01",
                    "month_index": 0,
                    "open_time": 1600000000000,
                    "fold_id": "fixed",
                    "decision_bucket_id": 26666666,
                    "row_position": 1,
                    "label": 0,
                    "trade_return": -0.01,
                    "final_preselection_score": 0.8,
                    "selected_by_score_before_execution": 0,
                    "executed": 0,
                    "rejection_stage": "ranking",
                    "rejection_reason": "rank_rejected",
                    "rank_within_decision_bucket": 2,
                    "candidate_count_within_decision_bucket": 2,
                },
            ]
            manifest = candidate_artifacts.write_candidate_artifact(
                path,
                records,
                metadata={"source_run_id": "test"},
                artifact_format="csv_gzip",
            )
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["row_count"], 2)
            rows = list(candidate_artifacts.read_candidate_rows(path))
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(1 for item in rows if item["executed"] == "1"), 1)
            self.assertEqual(rows[1]["rejection_reason"], "rank_rejected")

    def test_candidate_artifact_rejects_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "incomplete.csv"
            path.write_text("candidate_id\nabc\n", encoding="utf-8")
            manifest_path = Path(str(path) + ".manifest.json")
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "complete": False,
                "checksum": "",
            }), encoding="utf-8")
            with self.assertRaises(candidate_artifacts.CandidateArtifactError):
                candidate_artifacts.load_manifest(path)

    def test_pipeline_candidate_artifact_includes_topk_rejected_candidate(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--trade-selection", "topk_score",
            "--trade-score", "probability",
            "--top-k-per-minute", "1",
            "--max-trades-per-period", "0",
        ])
        rows = [
            symbol_row("AAA", 1600000000000, label=1, trade_return=0.02),
            symbol_row("BBB", 1600000000000, label=0, trade_return=-0.01),
        ]
        bundle = pipeline.build_prediction_bundle(probability=[0.9, 0.8], calibrated_probability=[0.9, 0.8])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pipeline_candidates.csv.gz"
            manifest = pipeline.write_candidate_artifact_for_split(
                str(path),
                rows,
                bundle,
                0.5,
                args,
                "fixed",
                "fixed",
                "gbdt_internal",
                ["ret_1m"],
                runtime_args=args,
                trade_score_name="probability",
            )
            self.assertEqual(manifest["row_count"], 2)
            artifact_rows = list(candidate_artifacts.read_candidate_rows(path))
            executed = [row for row in artifact_rows if row["executed"] == "1"]
            rejected = [row for row in artifact_rows if row["executed"] != "1"]
            self.assertEqual(len(executed), 1)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["rejection_stage"], "ranking")
            self.assertEqual(rejected[0]["candidate_count_within_decision_bucket"], "2")

    def test_pipeline_candidate_artifact_decision_bucket_matches_execution_minute(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--trade-selection", "topk_score",
            "--trade-score", "probability",
            "--top-k-per-minute", "1",
            "--max-trades-per-period", "0",
            "--trade-period-minutes", "60",
        ])
        base_minute = 26666640
        rows = [
            symbol_row("AAA", base_minute * 60000, label=1, trade_return=0.02),
            symbol_row("BBB", (base_minute + 10) * 60000, label=1, trade_return=0.02),
        ]
        bundle = pipeline.build_prediction_bundle(probability=[0.9, 0.8], calibrated_probability=[0.9, 0.8])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pipeline_candidates.csv.gz"
            manifest = pipeline.write_candidate_artifact_for_split(
                str(path),
                rows,
                bundle,
                0.5,
                args,
                "fixed",
                "fixed",
                "gbdt_internal",
                ["ret_1m"],
                runtime_args=args,
                trade_score_name="probability",
            )
            self.assertEqual(manifest["row_count"], 2)
            artifact_rows = list(candidate_artifacts.read_candidate_rows(path))

        self.assertEqual([row["candidate_count_within_decision_bucket"] for row in artifact_rows], ["1", "1"])
        self.assertEqual([row["candidate_count_within_execution_bucket"] for row in artifact_rows], ["1", "1"])
        self.assertEqual([row["candidate_count_within_period_bucket"] for row in artifact_rows], ["2", "2"])
        self.assertNotEqual(artifact_rows[0]["decision_bucket_id"], artifact_rows[1]["decision_bucket_id"])
        self.assertEqual(artifact_rows[0]["period_bucket_id"], artifact_rows[1]["period_bucket_id"])

    def test_pipeline_candidate_artifact_prefilter_includes_threshold_and_symbol_rejections(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--candidate-serialization-stage", "pre_score_edge_pre_symbol_filter",
            "--trade-selection", "topk_score",
            "--trade-score", "probability",
            "--top-k-per-minute", "1",
            "--max-trades-per-period", "0",
        ])
        rows = [
            symbol_row("AAA", 1600000000000, label=1, trade_return=0.02),
            symbol_row("BBB", 1600000000000, label=0, trade_return=-0.01),
            symbol_row("CCC", 1600000000000, label=1, trade_return=0.03),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.90, 0.80, 0.40],
            calibrated_probability=[0.90, 0.80, 0.40],
        )
        symbol_filter_info = {
            "enabled": True,
            "mode": "positive_avg_profit",
            "allowed_symbols": ["AAA", "CCC"],
            "filtered_symbols": ["BBB"],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pipeline_candidates.csv.gz"
            manifest = pipeline.write_candidate_artifact_for_split(
                str(path),
                rows,
                bundle,
                0.5,
                args,
                "fixed",
                "fixed",
                "gbdt_internal",
                ["ret_1m"],
                runtime_args=args,
                trade_score_name="probability",
                symbol_filter_info=symbol_filter_info,
            )
            artifact_rows = list(candidate_artifacts.read_candidate_rows(path))

        by_symbol = {row["symbol"]: row for row in artifact_rows}
        self.assertEqual(manifest["candidate_serialization_stage"], "pre_score_edge_pre_symbol_filter")
        self.assertEqual(manifest["row_count"], 3)
        self.assertEqual(by_symbol["AAA"]["raw_signal"], "1")
        self.assertEqual(by_symbol["AAA"]["selected_by_topk"], "1")
        self.assertEqual(by_symbol["AAA"]["executed"], "1")
        self.assertEqual(by_symbol["BBB"]["selected_by_symbol_filter"], "0")
        self.assertEqual(by_symbol["BBB"]["raw_signal"], "0")
        self.assertEqual(by_symbol["BBB"]["rejection_stage"], "symbol_filter")
        self.assertEqual(by_symbol["CCC"]["selected_by_threshold"], "0")
        self.assertEqual(by_symbol["CCC"]["raw_signal"], "0")
        self.assertEqual(by_symbol["CCC"]["rejection_stage"], "threshold")

    def test_pipeline_candidate_artifact_hides_no_valid_threshold_sentinel(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--candidate-serialization-stage", "pre_score_edge_pre_symbol_filter",
            "--trade-selection", "topk_score",
            "--trade-score", "probability",
            "--top-k-per-minute", "1",
            "--max-trades-per-period", "0",
        ])
        rows = [
            symbol_row("AAA", 1600000000000, label=1, trade_return=0.02),
            symbol_row("BBB", 1600000000000, label=0, trade_return=-0.01),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[0.90, 0.80],
            calibrated_probability=[0.90, 0.80],
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pipeline_candidates.csv.gz"
            pipeline.write_candidate_artifact_for_split(
                str(path),
                rows,
                bundle,
                1.01,
                args,
                "fixed",
                "fixed",
                "gbdt_internal",
                ["ret_1m"],
                runtime_args=args,
                trade_score_name="probability",
                threshold_selection_status=pipeline.THRESHOLD_STATUS_NO_VALID,
            )
            artifact_rows = list(candidate_artifacts.read_candidate_rows(path))

        self.assertEqual([row["selected_threshold"] for row in artifact_rows], ["", ""])
        self.assertEqual([row["selected_by_threshold"] for row in artifact_rows], ["0", "0"])
        self.assertEqual({row["rejection_stage"] for row in artifact_rows}, {"threshold"})

    def test_portfolio_ledger_reconciles_overlapping_positions(self):
        rows = [
            symbol_row("AAA", 1600000000000, label=1, trade_return=0.10),
            symbol_row("BBB", 1600000000000, label=0, trade_return=-0.02),
        ]
        collector = portfolio_ledger.PortfolioEventCollector()
        execution = pipeline.portfolio_execution(
            rows,
            pipeline.build_prediction_bundle(probability=[1.0, 1.0], calibrated_probability=[1.0, 1.0]),
            0.5,
            0.001,
            0.0005,
            100.0,
            0.5,
            1.0,
            0,
            60,
            5,
            trade_selection="threshold",
            portfolio_event_recorder=collector,
            portfolio_fold_id="fixed",
        )
        reconciliation = portfolio_ledger.reconcile_events(
            collector.events,
            starting_capital=100.0,
            reported_total_fees=execution["total_fee_amount"],
            reported_total_slippage=execution["total_slippage_amount"],
        )
        self.assertEqual(reconciliation["portfolio_reconciliation_status"], "passed")
        self.assertEqual(reconciliation["max_concurrent_positions"], 2)
        self.assertAlmostEqual(reconciliation["ending_capital"], execution["ending_capital"])
        self.assertEqual(reconciliation["drawdown_precision"], "exact_realized")
        self.assertEqual(reconciliation["mark_to_market_drawdown_available"], 0)

    def test_portfolio_ledger_candidate_ids_match_candidate_artifact(self):
        parser = pipeline.build_parser()
        args = parser.parse_args([
            "--portfolio-timeline-mode", "events",
            "--trade-selection", "topk_score",
            "--trade-score", "probability",
            "--top-k-per-minute", "1",
            "--max-trades-per-period", "0",
        ])
        args.test_slippage_multiplier = 1.0
        rows = [
            symbol_row("AAA", 1600000000000, label=1, trade_return=0.10),
            symbol_row("BBB", 1600000000000, label=0, trade_return=-0.02),
        ]
        bundle = pipeline.build_prediction_bundle(
            probability=[1.0, 0.9],
            calibrated_probability=[1.0, 0.9],
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_path = root / "candidates.csv.gz"
            ledger_path = root / "portfolio_events.csv.gz"
            pipeline.write_candidate_artifact_for_split(
                str(candidate_path),
                rows,
                bundle,
                0.5,
                args,
                "walkforward",
                "fold_1",
                "gbdt_internal",
                ["ret_1m"],
                runtime_args=args,
                trade_score_name="probability",
            )
            pipeline.write_portfolio_ledger_for_split(
                str(ledger_path),
                rows,
                bundle,
                0.5,
                args,
                "walkforward",
                "fold_1",
                runtime_args=args,
                trade_score_name="probability",
            )
            artifact_rows = list(candidate_artifacts.read_candidate_rows(candidate_path))
            ledger_events = list(portfolio_ledger.read_events(ledger_path))

        artifact_executed_ids = {
            row["candidate_id"]
            for row in artifact_rows
            if row["executed"] == "1"
        }
        ledger_open_ids = {
            row["candidate_id"]
            for row in ledger_events
            if row["event_type"] == "position_open"
        }
        self.assertEqual(ledger_open_ids, artifact_executed_ids)

    def test_portfolio_ledger_streams_multiple_walk_forward_folds(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["--portfolio-timeline-mode", "events"])
        args.test_slippage_multiplier = 1.0
        rows = [
            symbol_row("AAA", 1600000000000, label=1, trade_return=0.10),
            symbol_row("BBB", 1600000060000, label=0, trade_return=-0.02),
        ]
        bundle = pipeline.build_prediction_bundle(probability=[1.0, 1.0], calibrated_probability=[1.0, 1.0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "walk_portfolio_events.csv.gz"
            with portfolio_ledger.PortfolioLedgerWriter(path, {"source": "walkforward"}) as writer:
                first = pipeline.write_portfolio_ledger_for_split(
                    "",
                    rows,
                    bundle,
                    0.5,
                    args,
                    "walkforward",
                    "fold_1",
                    writer=writer,
                )
                second = pipeline.write_portfolio_ledger_for_split(
                    "",
                    rows,
                    bundle,
                    0.5,
                    args,
                    "walkforward",
                    "fold_2",
                    writer=writer,
                )
            self.assertEqual(first["portfolio_reconciliation_status"], "passed")
            self.assertEqual(second["portfolio_reconciliation_status"], "passed")
            events = list(portfolio_ledger.read_events(path))
            self.assertEqual({row["fold_id"] for row in events}, {"fold_1", "fold_2"})
            self.assertEqual([int(row["event_sequence"]) for row in events], list(range(len(events))))
            summary = portfolio_ledger.reconcile_ledger_file_by_fold(path, starting_capital=args.initial_capital)
            self.assertEqual(summary["portfolio_reconciliation_status"], "passed")
            self.assertEqual(summary["fold_count"], 2)

    def test_cpp_discovery_reports_path_compiler_candidate(self):
        with mock.patch.object(verify_cpp_build.shutil, "which", side_effect=lambda name, path=None: r"C:\Tools\g++.exe" if name in ("g++.exe", "g++") else None):
            with mock.patch.object(verify_cpp_build, "discover_visual_studio_with_vswhere", return_value=[]):
                with mock.patch.object(verify_cpp_build, "discover_common_windows_compilers", return_value=[]):
                    candidates = verify_cpp_build.discover_compilers(system="Windows")
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].compiler_type, "mingw")

    def test_cpp_verify_no_compiler_is_unavailable_not_passed(self):
        with mock.patch.object(verify_cpp_build, "discover_compilers", return_value=[]):
            status = verify_cpp_build.verify(TEST_DIR, build=True)
        self.assertFalse(status["compiler_found"])
        self.assertEqual(status["verification_status"], "unavailable_no_compiler")
        self.assertFalse(status["build_passed"])
        self.assertTrue(status["installation_guidance"])

    def test_cpp_build_failure_classifies_missing_zlib(self):
        missing = verify_cpp_build.classify_build_failure("fatal error: zlib.h: No such file or directory")
        self.assertIn("zlib", missing)

    def test_powershell_cpp_wrapper_exists_and_invokes_python_helper(self):
        wrapper = Path(TEST_DIR) / "tools" / "verify_cpp_build.ps1"
        self.assertTrue(wrapper.exists())
        self.assertIn("verify_cpp_build.py", wrapper.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
