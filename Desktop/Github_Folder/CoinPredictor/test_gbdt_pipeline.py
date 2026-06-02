#!/usr/bin/env python3
import csv
import os
import tempfile
import unittest
from types import SimpleNamespace

import gbdt_pipeline as pipeline


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
                            target_exit_mode="fixed_target"):
    manifest_path = os.path.splitext(path)[0] + ".meta.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(
            "{\n"
            '  "version": 1,\n'
            '  "training_csv": "kline_growth_training.csv",\n'
            '  "label_mode": "%s",\n'
            '  "target_exit_mode": "%s",\n'
            '  "growth_threshold": %.12g,\n'
            '  "upside_target": %.12g,\n'
            '  "downside_stop": %.12g\n'
            "}\n" % (label_mode, target_exit_mode, growth_threshold, upside_target, downside_stop)
        )
    return manifest_path


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

    def test_cache_rebuilds_when_csv_timestamp_changes(self):
        cache_dir = os.path.join(self.temp.name, "cache")
        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        rows.cleanup()
        stat = os.stat(self.csv_path)
        os.utime(self.csv_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        rows, _, _ = pipeline.load_rows(self.csv_path, "memmap32", cache_dir=cache_dir)
        self.assertEqual(pipeline.CACHE_LOAD_INFO["status"], "rebuilt")
        rows.cleanup()

    def test_missing_liquidity_column_fails_clearly(self):
        path = os.path.join(self.temp.name, "missing-liquidity.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["symbol", "month", "month_index", "open_time", "label", "ret_1m"])
            writer.writerow(["TESTUSDT", "2020-01", 0, 1600000000000, 0, 0.0])
        with self.assertRaisesRegex(ValueError, "log_quote_volume"):
            pipeline.load_rows(path, "matrix32", disable_cache=True)

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
            pipeline.load_rows(path, "matrix32", disable_cache=True)

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
            pipeline.load_rows(path, "matrix32", disable_cache=True)

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
        rows, _, _ = pipeline.load_rows(path, "matrix32", disable_cache=True)
        self.assertEqual(len(rows), 1)
        rows.cleanup()

    def test_ratio_and_walk_forward_views(self):
        rows, _, _ = pipeline.load_rows(self.csv_path, "matrix32", disable_cache=True)
        args = SimpleNamespace(train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15)
        train, validation, test = pipeline.select_ratio_split(rows, args)
        self.assertEqual((len(train), len(validation), len(test)), (24, 4, 4))
        self.assertEqual(len(pipeline.select_month_range(rows, 0, 6)), 24)
        self.assertEqual(len(pipeline.select_month_range(rows, 6, 7)), 4)
        rows.cleanup()

    def test_auc_sampling_uses_sampled_row_count(self):
        rows, _, _ = pipeline.load_rows(self.csv_path, "matrix32", disable_cache=True)
        previous_limit = pipeline.AUC_SAMPLE_ROWS
        try:
            pipeline.AUC_SAMPLE_ROWS = 3
            probabilities = pipeline.np.linspace(0.0, 1.0, len(rows), dtype=pipeline.np.float32)
            score = pipeline.auc_score_from_rows(probabilities, rows)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        finally:
            pipeline.AUC_SAMPLE_ROWS = previous_limit
            rows.cleanup()

    def test_threshold_constraints(self):
        base = 1600000000000
        rows = [row(base + index * 60000, label=1 if index < 2 else 0) for index in range(4)]
        threshold, metrics = pipeline.tune_threshold(
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
        self.assertEqual(threshold, 0.75)
        self.assertEqual(metrics["predicted_trades"], 2)

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
            probabilities=[0.9, 0.1, 0.1],
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


if __name__ == "__main__":
    unittest.main()
