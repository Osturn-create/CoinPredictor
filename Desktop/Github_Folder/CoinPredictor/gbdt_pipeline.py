#!/usr/bin/env python3
"""Train/evaluate a boosted-tree model from kline_growth_training.csv.

The script uses LightGBM when it is installed and otherwise falls back to a
small standard-library boosted-stump model. Splits are chronological by each
symbol's month_index.
"""

import argparse
from array import array
import bisect
from collections import deque
import csv
import datetime
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import gzip

try:
    import numpy as np
except ImportError:
    np = None

try:
    import psutil
except ImportError:
    psutil = None

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBM(Classifier|Regressor) was fitted with feature names",
    category=UserWarning,
)


METADATA_COLUMNS = set([
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
])

RECONSTRUCTED_QUOTE_VOLUME_DISCOUNT = 0.99999
CACHE_VERSION = 2
SHARDED_AGGREGATE_CACHE_VERSION = 3
START_TIME = time.time()
CACHE_LOAD_INFO = {}
TEMP_PREDICTION_PATHS = set()
AUC_SAMPLE_ROWS = 1000000
CANONICAL_TRAINING_CSV = "kline_growth_training.csv"
TRAINING_MANIFEST_VERSION = 1
SHARDED_DATASET_MANIFEST = "kline_growth_dataset.meta.json"
SHARDED_DATASET_MANIFEST_VERSION = 1
SHARD_MANIFEST_VERSION = 1
MARKET_BREADTH_AUGMENTATION_VERSION = 1
MARKET_BREADTH_FEATURE_COLUMNS = [
    "market_breadth_up_5m",
    "market_breadth_up_15m",
    "market_breadth_up_60m",
    "market_average_return_5m",
    "market_average_return_15m",
    "market_average_return_60m",
    "market_median_return_15m",
    "market_quote_volume_zscore_60m",
    "symbol_return_minus_market_5m",
    "symbol_return_minus_market_15m",
    "symbol_return_minus_market_60m",
    "market_breadth_missing",
]
MAX_RSS_GIB_OBSERVED = 0.0
MAX_RSS_STAGE = ""
NORMALIZED_MICROSECOND_OPEN_TIMES = 0
DEFAULT_MEMORY_BUDGET_GB = 7.8
PROFILE_RECORDS = []
PROFILE_ENABLED = True
WARNED_MESSAGES = set()
PROFILE_COLUMNS = [
    "stage_name",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "rss_gb_start",
    "rss_gb_end",
    "rss_gb_peak_if_available",
    "rows_processed",
    "rows_per_second_if_available",
    "extra_info",
]
MEMORY_BUDGET_DEFAULTS = {
    "feature_storage": "memmap32",
    "cache_dir": ".gbdt_cache",
    "max_train_rows": 1500000,
    "max_validation_rows": 750000,
    "max_final_train_rows": 1500000,
    "prediction_batch_rows": 200000,
    "prediction_output_mode": "trades",
    "auc_sample_rows": 1000000,
    "adaptive_threshold_sample_rows": 1000000,
    "max_bin": 63,
    "lightgbm_histogram_pool_mb": 128.0,
    "subsample_for_bin": 100000,
    "n_jobs": 2,
    "calibration_max_rows": 500000,
}


class MemoryLimitExceeded(RuntimeError):
    def __init__(self, stage, rss_gib, limit_gib):
        RuntimeError.__init__(self, "memory limit exceeded")
        self.stage = stage
        self.rss_gib = rss_gib
        self.limit_gib = limit_gib


def update_peak_rss(rss_gib, stage=None):
    global MAX_RSS_GIB_OBSERVED, MAX_RSS_STAGE
    if rss_gib is None:
        return
    if rss_gib >= MAX_RSS_GIB_OBSERVED:
        MAX_RSS_GIB_OBSERVED = rss_gib
        if stage:
            MAX_RSS_STAGE = stage


def log_memory(stage):
    elapsed = time.time() - START_TIME
    if psutil is None:
        print("[progress {:8.1f}s] {}".format(elapsed, stage), flush=True)
        return
    rss_gib = psutil.Process(os.getpid()).memory_info().rss / float(1024 ** 3)
    update_peak_rss(rss_gib, stage)
    print("[progress {:8.1f}s rss={:.2f} GiB] {}".format(elapsed, rss_gib, stage), flush=True)


def close_memmap(values):
    if values is None:
        return
    try:
        values.flush()
    except Exception:
        pass
    mmap = getattr(values, "_mmap", None)
    if mmap is not None:
        try:
            mmap.close()
        except Exception:
            pass


def sigmoid(value):
    if value < -40.0:
        return 0.0
    if value > 40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except ValueError:
        return default


def open_csv_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")


def warn_once(key, message):
    if key in WARNED_MESSAGES:
        return
    WARNED_MESSAGES.add(key)
    print(message, file=sys.stderr, flush=True)


def normalize_open_time_ms(open_time):
    value = int(open_time)
    if value > 10_000_000_000_000 or value < -10_000_000_000_000:
        return value // 1000
    return value


def normalize_open_times_array(open_times):
    global NORMALIZED_MICROSECOND_OPEN_TIMES
    if np is None:
        count = 0
        normalized = []
        for value in open_times:
            raw = int(value)
            normalized_value = normalize_open_time_ms(raw)
            if normalized_value != raw:
                count += 1
            normalized.append(normalized_value)
        NORMALIZED_MICROSECOND_OPEN_TIMES += count
        return normalized, count
    values = np.asarray(open_times, dtype=np.int64)
    if values.size == 0:
        return values, 0
    mask = np.abs(values) > 10_000_000_000_000
    count = int(np.count_nonzero(mask))
    if count:
        values[mask] = values[mask] // 1000
    NORMALIZED_MICROSECOND_OPEN_TIMES += count
    return values, count


def parse_threshold_grid(text):
    values = [safe_float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("threshold grid cannot be empty")
    return sorted(values)


def parse_explicit_flags(argv):
    explicit = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:]
        if "=" in name:
            name = name.split("=", 1)[0]
            explicit.add(name.replace("-", "_"))
            index += 1
            continue
        explicit.add(name.replace("-", "_"))
        index += 1
    return explicit


def apply_memory_budget_defaults(args, explicit_flags):
    if args.memory_budget_gb <= 0.0:
        return False
    if abs(args.memory_budget_gb - DEFAULT_MEMORY_BUDGET_GB) > 1e-9:
        print(
            "Warning: only the 7.8GB same-spec profile has built-in defaults; keeping explicit arguments unchanged.",
            file=sys.stderr,
            flush=True,
        )
        return False
    for key, value in MEMORY_BUDGET_DEFAULTS.items():
        if key not in explicit_flags:
            setattr(args, key, value)
    if "max_rss_gb" not in explicit_flags and args.max_rss_gb <= 0.0:
        args.max_rss_gb = args.memory_budget_gb
    return True


def print_memory_budget_summary(args, budget_applied):
    if not budget_applied:
        return
    print("Memory budget mode: 7.8GB same-spec profile", flush=True)
    print("Feature storage: {}".format(args.feature_storage), flush=True)
    print("Max train rows: {}".format(args.max_train_rows), flush=True)
    print("Max validation rows: {}".format(args.max_validation_rows), flush=True)
    print("Max final train rows: {}".format(args.max_final_train_rows), flush=True)
    print("Prediction batch rows: {}".format(args.prediction_batch_rows), flush=True)
    print("AUC sample rows: {}".format(args.auc_sample_rows), flush=True)
    print("Adaptive threshold sample rows: {}".format(args.adaptive_threshold_sample_rows), flush=True)
    print("Max bin: {}".format(args.max_bin), flush=True)
    print("Histogram pool MB: {}".format(args.lightgbm_histogram_pool_mb), flush=True)
    print("Subsample for bin: {}".format(args.subsample_for_bin), flush=True)
    print("LightGBM n_jobs: {}".format(args.n_jobs), flush=True)
    print("Max RSS guard: {:.1f}GB".format(args.max_rss_gb), flush=True)


def current_rss_gib(stage=None):
    if psutil is None:
        return None


def utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def profile_reset(enabled=True):
    global PROFILE_ENABLED
    PROFILE_ENABLED = bool(enabled)
    del PROFILE_RECORDS[:]


def start_profile_stage(stage_name, extra_info=""):
    if not PROFILE_ENABLED:
        return None
    return {
        "stage_name": stage_name,
        "start_monotonic": time.time(),
        "start_time": utc_now_iso(),
        "rss_gb_start": current_rss_gib(stage_name),
        "extra_info": extra_info,
    }


def finish_profile_stage(token, rows_processed=0, extra_info=None, rss_peak_if_available=None):
    if token is None or not PROFILE_ENABLED:
        return
    end_monotonic = time.time()
    elapsed_seconds = max(0.0, end_monotonic - token["start_monotonic"])
    rss_end = current_rss_gib(token["stage_name"])
    peak = rss_peak_if_available
    if peak is None:
        candidates = [value for value in (token.get("rss_gb_start"), rss_end) if value is not None]
        peak = max(candidates) if candidates else None
    rows_per_second = 0.0
    if rows_processed and elapsed_seconds > 0.0:
        rows_per_second = float(rows_processed) / elapsed_seconds
    PROFILE_RECORDS.append({
        "stage_name": token["stage_name"],
        "start_time": token["start_time"],
        "end_time": utc_now_iso(),
        "elapsed_seconds": elapsed_seconds,
        "rss_gb_start": token.get("rss_gb_start"),
        "rss_gb_end": rss_end,
        "rss_gb_peak_if_available": peak,
        "rows_processed": rows_processed,
        "rows_per_second_if_available": rows_per_second,
        "extra_info": extra_info if extra_info is not None else token.get("extra_info", ""),
    })


def record_profile_stage(stage_name, elapsed_seconds, rows_processed=0, extra_info="",
                         rss_gb_start=None, rss_gb_end=None, rss_gb_peak_if_available=None):
    if not PROFILE_ENABLED:
        return
    if rss_gb_start is None:
        rss_gb_start = current_rss_gib(stage_name)
    if rss_gb_end is None:
        rss_gb_end = current_rss_gib(stage_name)
    if rss_gb_peak_if_available is None:
        candidates = [value for value in (rss_gb_start, rss_gb_end) if value is not None]
        rss_gb_peak_if_available = max(candidates) if candidates else None
    rows_per_second = 0.0
    if rows_processed and elapsed_seconds > 0.0:
        rows_per_second = float(rows_processed) / float(elapsed_seconds)
    PROFILE_RECORDS.append({
        "stage_name": stage_name,
        "start_time": "",
        "end_time": "",
        "elapsed_seconds": float(elapsed_seconds),
        "rss_gb_start": rss_gb_start,
        "rss_gb_end": rss_gb_end,
        "rss_gb_peak_if_available": rss_gb_peak_if_available,
        "rows_processed": int(rows_processed) if rows_processed else 0,
        "rows_per_second_if_available": rows_per_second,
        "extra_info": extra_info,
    })
    try:
        rss_gib = psutil.Process(os.getpid()).memory_info().rss / float(1024 ** 3)
        update_peak_rss(rss_gib, stage)
        return rss_gib
    except Exception:
        return None


def check_memory_limit(stage, args):
    if args is None or args.max_rss_gb <= 0.0:
        return False
    rss_gib = current_rss_gib(stage)
    if rss_gib is None or rss_gib <= args.max_rss_gb:
        return False
    print(
        "MEMORY WARNING at {}: RSS={:.2f} GiB exceeds max_rss_gb={:.2f}".format(
            stage,
            rss_gib,
            args.max_rss_gb,
        ),
        file=sys.stderr,
        flush=True,
    )
    if args.abort_on_memory_limit:
        raise MemoryLimitExceeded(stage, rss_gib, args.max_rss_gb)
    return True


def memory_checkpoint(stage, args=None):
    log_memory(stage)
    check_memory_limit(stage, args)


def atomic_write_path(path, writer):
    target_path = os.path.abspath(path)
    directory = os.path.dirname(target_path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(target_path) + ".",
        suffix=".tmp",
        dir=directory,
    )
    os.close(descriptor)
    try:
        writer(temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def output_path_args():
    return [
        "predictions_out",
        "metrics_out",
        "walkforward_metrics_out",
        "walkforward_diagnostics_out",
        "symbol_filter_diagnostics_out",
        "walk_predictions_out",
        "feature_importance_out",
        "run_summary_out",
        "experiment_summary_out",
        "profile_out",
    ]


def configure_output_paths(args):
    if not getattr(args, "results_dir", ""):
        return
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    for name in output_path_args():
        value = getattr(args, name, "")
        if not value:
            continue
        setattr(args, name, os.path.join(results_dir, os.path.basename(value)))


def safe_free_disk_gb(path):
    target = path
    if not os.path.exists(target):
        target = os.path.dirname(os.path.abspath(target)) or "."
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return None
    return usage.free / float(1024 ** 3)


def directory_size_gb(path):
    if not path or not os.path.exists(path):
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                pass
    return total / float(1024 ** 3)


def check_free_disk(path, args, stage_name):
    if getattr(args, "min_free_disk_gb", 0.0) <= 0.0:
        return
    free_gb = safe_free_disk_gb(path)
    if free_gb is None:
        return
    if free_gb >= args.min_free_disk_gb:
        return
    message = "LOW DISK at {}: free={:.2f}GB below min_free_disk_gb={:.2f}".format(
        stage_name,
        free_gb,
        args.min_free_disk_gb,
    )
    print(message, file=sys.stderr, flush=True)
    if getattr(args, "abort_on_low_disk", False):
        raise RuntimeError(message)


def warn_if_cache_dir_large(cache_dir, args):
    if getattr(args, "max_cache_size_gb", 0.0) <= 0.0:
        return
    size_gb = directory_size_gb(cache_dir)
    if size_gb > args.max_cache_size_gb:
        print(
            "Warning: cache dir {} is {:.2f}GB, above --max-cache-size-gb {:.2f}. "
            "Run --cache-cleanup to review old cache files.".format(
                cache_dir,
                size_gb,
                args.max_cache_size_gb,
            ),
            file=sys.stderr,
            flush=True,
        )


def maybe_compress_file(path, compression):
    if compression != "gzip" or not os.path.exists(path) or path.endswith(".gz"):
        return path
    gz_path = path + ".gz"
    with open(path, "rb") as source:
        with gzip.open(gz_path, "wb") as target:
            shutil.copyfileobj(source, target)
    os.remove(path)
    return gz_path


def postprocess_output_files(args):
    if not getattr(args, "compress_outputs", False) or args.output_compression == "none":
        return
    for name in (
        "predictions_out",
        "metrics_out",
        "walkforward_metrics_out",
        "walkforward_diagnostics_out",
        "symbol_filter_diagnostics_out",
        "walk_predictions_out",
        "feature_importance_out",
        "experiment_summary_out",
        "profile_out",
    ):
        value = getattr(args, name, "")
        if value:
            setattr(args, name, maybe_compress_file(value, args.output_compression))


def resolve_cache_dir(path, cache_dir):
    if cache_dir:
        return cache_dir
    return os.path.join(os.path.dirname(os.path.abspath(path)), ".gbdt_cache")


def sample_positions_for_count(count, max_rows):
    if count <= 0:
        return []
    if max_rows and count > max_rows:
        if np is not None:
            return np.linspace(0, count - 1, num=max_rows, dtype=np.int64)
        if max_rows == 1:
            return [0]
        return [int(round(index * (count - 1) / float(max_rows - 1))) for index in range(max_rows)]
    if np is not None:
        return np.arange(count, dtype=np.int64)
    return list(range(count))


def expected_value_from_probability(probability, upside_target, downside_stop, fee, slippage):
    return (
        float(probability) * upside_target
        - (1.0 - float(probability)) * downside_stop
        - fee
        - slippage
    )


def empirical_payoff_statistics(rows, args):
    fixed_win = float(getattr(args, "effective_upside_target", getattr(args, "upside_target", 0.05)))
    fixed_loss = -float(getattr(args, "effective_downside_stop", getattr(args, "downside_stop", 0.02)))
    labels = rows.labels_array() if is_compact_rows(rows) else [row.label for row in rows]
    trade_returns = actual_trade_returns(rows)
    positions = sample_positions_for_count(len(labels), getattr(args, "ev_payoff_calibration_max_rows", 0))
    positive_returns = []
    negative_returns = []
    if np is not None and isinstance(labels, np.ndarray):
        sampled_labels = labels[positions]
        sampled_returns = trade_returns[positions]
        positive_mask = sampled_labels == 1
        negative_mask = (sampled_labels == 0) & (sampled_returns < 0.0)
        positive_returns = sampled_returns[positive_mask]
        negative_returns = sampled_returns[negative_mask]
        positive_rows = int(positive_returns.size)
        negative_rows = int(negative_returns.size)
        sampled_rows = int(len(sampled_labels))
        empirical_win = float(np.mean(positive_returns)) if positive_rows else fixed_win
        empirical_loss = float(np.mean(negative_returns)) if negative_rows else fixed_loss
    else:
        sampled_rows = len(positions)
        positive_rows = 0
        negative_rows = 0
        positive_sum = 0.0
        negative_sum = 0.0
        for position in positions:
            label = int(labels[position])
            trade_return = float(trade_returns[position])
            if label == 1:
                positive_rows += 1
                positive_sum += trade_return
            elif trade_return < 0.0:
                negative_rows += 1
                negative_sum += trade_return
        empirical_win = positive_sum / float(positive_rows) if positive_rows else fixed_win
        empirical_loss = negative_sum / float(negative_rows) if negative_rows else fixed_loss
    min_positive = int(getattr(args, "ev_payoff_min_positive_rows", 25))
    min_negative = int(getattr(args, "ev_payoff_min_negative_rows", 25))
    used_positive_fallback = positive_rows < min_positive
    used_negative_fallback = negative_rows < min_negative
    if used_positive_fallback:
        empirical_win = fixed_win
    if used_negative_fallback:
        empirical_loss = fixed_loss
    source = "empirical_validation"
    if used_positive_fallback or used_negative_fallback:
        source = "empirical_validation_with_fallback"
    return {
        "fixed_expected_win_return": fixed_win,
        "fixed_expected_loss_return": fixed_loss,
        "empirical_expected_win_return": float(empirical_win),
        "empirical_expected_loss_return": float(empirical_loss),
        "ev_payoff_rows": int(sampled_rows),
        "ev_payoff_positive_rows": int(positive_rows),
        "ev_payoff_negative_rows": int(negative_rows),
        "empirical_ev_payoff_source": source,
    }


def fit_ev_payoff_context(rows, bundle, args):
    stats = empirical_payoff_statistics(rows, args)
    requested_mode = getattr(args, "ev_payoff_mode", "fixed_targets")
    predicted_available = bundle.get("predicted_trade_return") is not None
    actual_mode = requested_mode
    source = "fixed_targets"
    if requested_mode == "predicted_return":
        if predicted_available and getattr(args, "objective_mode", "classification") in ("hybrid", "return_regression"):
            source = "predicted_return"
        else:
            warn_once(
                "ev-payoff-predicted-return-fallback",
                "Warning: --ev-payoff-mode predicted_return requested without predicted returns; falling back to empirical/fixed EV.",
            )
            if stats["empirical_ev_payoff_source"].startswith("empirical_validation"):
                actual_mode = "empirical_validation"
                source = stats["empirical_ev_payoff_source"]
            else:
                actual_mode = "fixed_targets"
                source = "fixed_targets"
    elif requested_mode == "empirical_validation":
        source = stats["empirical_ev_payoff_source"]
    else:
        source = "fixed_targets"
    if actual_mode == "fixed_targets":
        expected_win = stats["fixed_expected_win_return"]
        expected_loss = stats["fixed_expected_loss_return"]
    else:
        expected_win = stats["empirical_expected_win_return"]
        expected_loss = stats["empirical_expected_loss_return"]
    context = dict(stats)
    context.update({
        "ev_payoff_mode": requested_mode,
        "ev_payoff_actual_mode": actual_mode,
        "ev_payoff_source": source,
        "predicted_return_available": bool(predicted_available),
        "ev_expected_win_return": float(expected_win),
        "ev_expected_loss_return": float(expected_loss),
    })
    return context


def default_ev_context(args, upside_target, downside_stop):
    fixed_win = float(getattr(args, "effective_upside_target", upside_target)) if args is not None else float(upside_target)
    fixed_loss = -float(getattr(args, "effective_downside_stop", downside_stop)) if args is not None else -float(downside_stop)
    mode = getattr(args, "ev_payoff_mode", "fixed_targets") if args is not None else "fixed_targets"
    return {
        "ev_payoff_mode": mode,
        "ev_payoff_actual_mode": "fixed_targets",
        "ev_payoff_source": "fixed_targets",
        "ev_expected_win_return": fixed_win,
        "ev_expected_loss_return": fixed_loss,
        "fixed_expected_win_return": fixed_win,
        "fixed_expected_loss_return": fixed_loss,
        "empirical_expected_win_return": fixed_win,
        "empirical_expected_loss_return": fixed_loss,
        "ev_payoff_rows": 0,
        "ev_payoff_positive_rows": 0,
        "ev_payoff_negative_rows": 0,
        "predicted_return_available": False,
    }


def resolve_ev_context(bundle, args, upside_target, downside_stop):
    context = bundle.get("ev_context")
    if context:
        return context
    return default_ev_context(args, upside_target, downside_stop)


def conditional_payoff_statistics(rows, args):
    fixed_win = float(getattr(args, "effective_upside_target", getattr(args, "upside_target", 0.05))) if args is not None else 0.05
    fixed_loss = -float(getattr(args, "effective_downside_stop", getattr(args, "downside_stop", 0.02))) if args is not None else -0.02
    sampled_rows = len(rows)
    positive_rows = 0
    negative_rows = 0
    positive_sum = 0.0
    negative_sum = 0.0
    max_rows = int(getattr(args, "conditional_payoff_max_rows", 500000)) if args is not None else 500000
    if is_compact_rows(rows):
        labels = rows.labels_array()
        trade_returns = rows.trade_returns_array()
        row_count = len(labels)
        if max_rows and row_count > max_rows:
            positions = np.linspace(0, row_count - 1, num=max_rows, dtype=np.int64) if np is not None else [
                int(round(index * (row_count - 1) / float(max_rows - 1))) for index in range(max_rows)
            ]
            labels_iter = labels[positions] if np is not None else [labels[position] for position in positions]
            returns_iter = trade_returns[positions] if np is not None else [trade_returns[position] for position in positions]
            sampled_rows = int(max_rows)
        else:
            labels_iter = labels
            returns_iter = trade_returns
        for label, trade_return in zip(labels_iter, returns_iter):
            trade_return = float(trade_return)
            if int(label) == 1:
                positive_rows += 1
                positive_sum += trade_return
            elif trade_return < 0.0:
                negative_rows += 1
                negative_sum += trade_return
    else:
        if max_rows and len(rows) > max_rows:
            sampled_rows = int(max_rows)
            positions = np.linspace(0, len(rows) - 1, num=max_rows, dtype=np.int64) if np is not None else [
                int(round(index * (len(rows) - 1) / float(max_rows - 1))) for index in range(max_rows)
            ]
            iterable = [rows[int(position)] for position in positions]
        else:
            iterable = rows
        for row in iterable:
            trade_return = float(row.trade_return)
            if int(row.label) == 1:
                positive_rows += 1
                positive_sum += trade_return
            elif trade_return < 0.0:
                negative_rows += 1
                negative_sum += trade_return
    expected_win = positive_sum / float(positive_rows) if positive_rows else fixed_win
    expected_loss = negative_sum / float(negative_rows) if negative_rows else fixed_loss
    min_positive = int(getattr(args, "conditional_payoff_min_positive_rows", 25)) if args is not None else 25
    min_negative = int(getattr(args, "conditional_payoff_min_negative_rows", 25)) if args is not None else 25
    used_positive_fallback = positive_rows < min_positive
    used_negative_fallback = negative_rows < min_negative
    if used_positive_fallback:
        expected_win = fixed_win
    if used_negative_fallback:
        expected_loss = fixed_loss
    if used_positive_fallback and used_negative_fallback:
        source = "fixed_fallback"
    elif used_positive_fallback or used_negative_fallback:
        source = "empirical_validation_with_fallback"
    else:
        source = "empirical_validation"
    return {
        "conditional_expected_win_return": float(expected_win),
        "conditional_expected_loss_return": float(expected_loss),
        "conditional_payoff_rows": int(sampled_rows),
        "conditional_payoff_positive_rows": int(positive_rows),
        "conditional_payoff_negative_rows": int(negative_rows),
        "conditional_payoff_source": source,
    }


def default_hybrid_return_context(args, upside_target, downside_stop):
    del upside_target, downside_stop
    return {
        "hybrid_return_combination": getattr(args, "hybrid_return_combination", "probability_times_return") if args is not None else "probability_times_return",
        "hybrid_min_probability": float(getattr(args, "hybrid_min_probability", 0.0)) if args is not None else 0.0,
        "conditional_expected_win_return": 0.0,
        "conditional_expected_loss_return": 0.0,
        "conditional_payoff_rows": 0,
        "conditional_payoff_positive_rows": 0,
        "conditional_payoff_negative_rows": 0,
        "conditional_payoff_source": "not_used",
    }


def fit_hybrid_return_context(rows, bundle, args):
    requested_mode = getattr(args, "hybrid_return_combination", "probability_times_return")
    context = default_hybrid_return_context(args, getattr(args, "upside_target", 0.05), getattr(args, "downside_stop", 0.02))
    context["hybrid_return_combination"] = requested_mode
    if requested_mode == "conditional_payoff":
        context.update(conditional_payoff_statistics(rows, args))
    return context


def resolve_hybrid_return_context(bundle, args, upside_target, downside_stop):
    context = bundle.get("hybrid_return_context")
    if context:
        return context
    return default_hybrid_return_context(args, upside_target, downside_stop)


def hybrid_probability_gate_threshold(bundle, args, upside_target, downside_stop):
    context = resolve_hybrid_return_context(bundle, args, upside_target, downside_stop)
    if context.get("hybrid_return_combination", "probability_times_return") != "expected_return":
        return 0.0
    return max(0.0, float(context.get("hybrid_min_probability", 0.0)))


def resolve_hybrid_predicted_returns(bundle):
    predicted_returns = bundle.get("predicted_trade_return")
    if predicted_returns is not None:
        return predicted_returns
    raw_predicted_returns = bundle.get("raw_predicted_trade_return")
    if raw_predicted_returns is not None:
        warn_once(
            "hybrid-raw-predicted-return-fallback",
            "Warning: calibrated predicted returns unavailable for hybrid scoring; using raw predicted returns.",
        )
        return raw_predicted_returns
    return None


def hybrid_base_score_components(probability, predicted_trade_return, fee, slippage, hybrid_context):
    combination = hybrid_context.get("hybrid_return_combination", "probability_times_return")
    if combination == "expected_return":
        if predicted_trade_return is None:
            return 0.0
        return float(predicted_trade_return) - fee - slippage
    if combination == "conditional_payoff":
        expected_win_return = float(hybrid_context.get("conditional_expected_win_return", 0.0))
        expected_loss_return = float(hybrid_context.get("conditional_expected_loss_return", 0.0))
        return float(probability) * expected_win_return + (1.0 - float(probability)) * expected_loss_return - fee - slippage
    if predicted_trade_return is None:
        return 0.0
    return float(probability) * float(predicted_trade_return) - fee - slippage


def uncertainty_penalty_mode(args=None, uncertainty_context=None):
    if uncertainty_context and uncertainty_context.get("penalty_mode"):
        return uncertainty_context.get("penalty_mode")
    if args is None:
        return "raw"
    return getattr(args, "hybrid_uncertainty_penalty_mode", "raw")


def uncertainty_penalty_amount(uncertainty, hybrid_uncertainty_penalty,
                               args=None, uncertainty_context=None):
    if uncertainty is None or hybrid_uncertainty_penalty <= 0.0:
        return 0.0
    mode = uncertainty_penalty_mode(args, uncertainty_context)
    uncertainty_value = float(uncertainty)
    if mode == "relative_return":
        global_std = max(float((uncertainty_context or {}).get("global_std", 0.0)), 1e-12)
        reference_scale = max(float((uncertainty_context or {}).get("penalty_reference_scale", 0.0)), 0.0)
        if reference_scale > 0.0:
            return float(hybrid_uncertainty_penalty) * (uncertainty_value / global_std) * reference_scale
    return float(hybrid_uncertainty_penalty) * uncertainty_value


def uncertainty_penalty_values_array(uncertainty_values, hybrid_uncertainty_penalty,
                                     args=None, uncertainty_context=None):
    values = np.asarray(uncertainty_values, dtype=np.float32)
    if hybrid_uncertainty_penalty <= 0.0:
        return np.zeros(len(values), dtype=np.float32)
    mode = uncertainty_penalty_mode(args, uncertainty_context)
    if mode == "relative_return":
        global_std = max(float((uncertainty_context or {}).get("global_std", 0.0)), 1e-12)
        reference_scale = max(float((uncertainty_context or {}).get("penalty_reference_scale", 0.0)), 0.0)
        if reference_scale > 0.0:
            return (
                values / np.float32(global_std)
            ) * np.float32(reference_scale * float(hybrid_uncertainty_penalty))
    return values * np.float32(hybrid_uncertainty_penalty)


def hybrid_score_details_for_bundle(bundle, index, fee, slippage,
                                    hybrid_score_mode="basic",
                                    hybrid_uncertainty_penalty=0.0,
                                    args=None,
                                    upside_target=0.05,
                                    downside_stop=0.02):
    hybrid_context = resolve_hybrid_return_context(bundle, args, upside_target, downside_stop)
    probability = calibrated_probability_value(bundle, index)
    predicted_returns = resolve_hybrid_predicted_returns(bundle)
    predicted_trade_return = float(predicted_returns[index]) if predicted_returns is not None else None
    raw_predicted_trade_return = raw_predicted_trade_return_value(bundle, index)
    base_score = hybrid_base_score_components(probability, predicted_trade_return, fee, slippage, hybrid_context)
    uncertainty = predicted_return_uncertainty_value(bundle, index)
    uncertainty_context = bundle.get("uncertainty_context") or {}
    hybrid_score = base_score
    if hybrid_score_mode == "risk_adjusted":
        hybrid_score = base_score - uncertainty_penalty_amount(
            uncertainty,
            hybrid_uncertainty_penalty,
            args,
            uncertainty_context,
        )
    return {
        "hybrid_return_combination": hybrid_context.get("hybrid_return_combination", "probability_times_return"),
        "hybrid_min_probability": float(hybrid_context.get("hybrid_min_probability", 0.0)),
        "base_hybrid_score": float(base_score),
        "hybrid_score": float(hybrid_score),
        "calibrated_probability": float(probability),
        "raw_predicted_trade_return": float(raw_predicted_trade_return),
        "calibrated_predicted_trade_return": float(predicted_trade_return) if predicted_trade_return is not None else 0.0,
        "conditional_expected_win_return": float(hybrid_context.get("conditional_expected_win_return", 0.0)),
        "conditional_expected_loss_return": float(hybrid_context.get("conditional_expected_loss_return", 0.0)),
        "conditional_payoff_rows": int(hybrid_context.get("conditional_payoff_rows", 0)),
        "conditional_payoff_positive_rows": int(hybrid_context.get("conditional_payoff_positive_rows", 0)),
        "conditional_payoff_negative_rows": int(hybrid_context.get("conditional_payoff_negative_rows", 0)),
        "conditional_payoff_source": hybrid_context.get("conditional_payoff_source", "not_used"),
    }


def expected_value_components(probability, predicted_trade_return, fee, slippage, objective_mode, ev_context):
    fixed_value = expected_value_from_probability(
        probability,
        float(ev_context.get("fixed_expected_win_return", 0.0)),
        -float(ev_context.get("fixed_expected_loss_return", 0.0)),
        fee,
        slippage,
    )
    empirical_value = (
        float(probability) * float(ev_context.get("empirical_expected_win_return", ev_context.get("fixed_expected_win_return", 0.0)))
        + (1.0 - float(probability)) * float(ev_context.get("empirical_expected_loss_return", ev_context.get("fixed_expected_loss_return", 0.0)))
        - fee
        - slippage
    )
    predicted_value = empirical_value
    if predicted_trade_return is not None:
        if objective_mode in ("return_regression", "hybrid"):
            predicted_value = float(predicted_trade_return) - fee - slippage
        else:
            predicted_value = float(probability) * float(predicted_trade_return) - fee - slippage
    source = ev_context.get("ev_payoff_source", ev_context.get("ev_payoff_actual_mode", "fixed_targets"))
    if source == "predicted_return" and predicted_trade_return is not None:
        expected_value = predicted_value
    elif source.startswith("empirical_validation"):
        expected_value = empirical_value
    else:
        expected_value = fixed_value
    return {
        "expected_value_fixed_targets": float(fixed_value),
        "expected_value_empirical": float(empirical_value),
        "expected_value_predicted_return": float(predicted_value),
        "expected_value": float(expected_value),
    }


def clipped_probability(probability):
    return min(max(float(probability), 1e-6), 1.0 - 1e-6)


def logit(probability):
    clipped = clipped_probability(probability)
    return math.log(clipped / (1.0 - clipped))


def sample_probability_label_pairs(probabilities, labels_values, max_rows):
    count = len(labels_values)
    if max_rows and count > max_rows:
        positions = np.linspace(0, count - 1, num=max_rows, dtype=np.int64) if np is not None else [
            int(round(index * (count - 1) / float(max_rows - 1))) for index in range(max_rows)
        ]
        if np is not None:
            return np.asarray(probabilities)[positions], np.asarray(labels_values)[positions]
        return [probabilities[position] for position in positions], [labels_values[position] for position in positions]
    return probabilities, labels_values


def brier_score(probabilities, labels_values):
    if np is not None and isinstance(probabilities, np.ndarray):
        diff = probabilities.astype(np.float32, copy=False) - np.asarray(labels_values, dtype=np.float32)
        return float(np.mean(diff * diff)) if len(diff) else 0.0
    if not labels_values:
        return 0.0
    total = 0.0
    for probability, label in zip(probabilities, labels_values):
        diff = float(probability) - float(label)
        total += diff * diff
    return total / float(len(labels_values))


def sample_regression_pairs(predictions, actuals, max_rows):
    count = len(actuals)
    if max_rows and count > max_rows:
        positions = np.linspace(0, count - 1, num=max_rows, dtype=np.int64) if np is not None else [
            int(round(index * (count - 1) / float(max_rows - 1))) for index in range(max_rows)
        ]
        if np is not None:
            return np.asarray(predictions)[positions], np.asarray(actuals)[positions]
        return [predictions[position] for position in positions], [actuals[position] for position in positions]
    return predictions, actuals


def rmse_score(predictions, actuals):
    if np is not None and isinstance(predictions, np.ndarray):
        diff = predictions.astype(np.float32, copy=False) - np.asarray(actuals, dtype=np.float32)
        return float(np.sqrt(np.mean(diff * diff))) if len(diff) else 0.0
    if not actuals:
        return 0.0
    total = 0.0
    for prediction, actual in zip(predictions, actuals):
        delta = float(prediction) - float(actual)
        total += delta * delta
    return math.sqrt(total / float(len(actuals)))


def mae_score(predictions, actuals):
    if np is not None and isinstance(predictions, np.ndarray):
        diff = np.abs(predictions.astype(np.float32, copy=False) - np.asarray(actuals, dtype=np.float32))
        return float(np.mean(diff)) if len(diff) else 0.0
    if not actuals:
        return 0.0
    return sum(abs(float(prediction) - float(actual)) for prediction, actual in zip(predictions, actuals)) / float(len(actuals))


def apply_linear_regression_calibration(values, a_value, b_value, batch_size=200000):
    if np is not None and isinstance(values, np.ndarray):
        for start in range(0, len(values), batch_size):
            end = min(len(values), start + batch_size)
            values[start:end] = values[start:end].astype(np.float32, copy=False) * np.float32(a_value) + np.float32(b_value)
        return values
    return [float(a_value) * float(value) + float(b_value) for value in values]


def fit_linear_regression_calibration(predictions, actuals, max_rows):
    sampled_predictions, sampled_actuals = sample_regression_pairs(predictions, actuals, max_rows)
    if np is not None:
        sampled_predictions = np.asarray(sampled_predictions, dtype=np.float32)
        sampled_actuals = np.asarray(sampled_actuals, dtype=np.float32)
        rows = int(len(sampled_actuals))
        before_rmse = rmse_score(sampled_predictions, sampled_actuals)
        before_mae = mae_score(sampled_predictions, sampled_actuals)
        if rows < 8:
            return {
                "mode": "linear",
                "a": 1.0,
                "b": 0.0,
                "rows": rows,
                "regression_calibration_rmse_before": before_rmse,
                "regression_calibration_rmse_after": before_rmse,
                "regression_calibration_mae_before": before_mae,
                "regression_calibration_mae_after": before_mae,
                "fallback": "identity",
            }
        centered = sampled_predictions - np.mean(sampled_predictions)
        variance = float(np.mean(centered * centered))
        if variance <= 1e-12:
            a_value = 1.0
            b_value = 0.0
        else:
            covariance = float(np.mean(centered * (sampled_actuals - np.mean(sampled_actuals))))
            a_value = covariance / variance
            b_value = float(np.mean(sampled_actuals) - a_value * np.mean(sampled_predictions))
        after_predictions = apply_linear_regression_calibration(sampled_predictions.copy(), a_value, b_value)
        return {
            "mode": "linear",
            "a": float(a_value),
            "b": float(b_value),
            "rows": rows,
            "regression_calibration_rmse_before": before_rmse,
            "regression_calibration_rmse_after": rmse_score(after_predictions, sampled_actuals),
            "regression_calibration_mae_before": before_mae,
            "regression_calibration_mae_after": mae_score(after_predictions, sampled_actuals),
        }

    sampled_predictions = [float(value) for value in sampled_predictions]
    sampled_actuals = [float(value) for value in sampled_actuals]
    rows = len(sampled_actuals)
    before_rmse = rmse_score(sampled_predictions, sampled_actuals)
    before_mae = mae_score(sampled_predictions, sampled_actuals)
    if rows < 8:
        return {
            "mode": "linear",
            "a": 1.0,
            "b": 0.0,
            "rows": rows,
            "regression_calibration_rmse_before": before_rmse,
            "regression_calibration_rmse_after": before_rmse,
            "regression_calibration_mae_before": before_mae,
            "regression_calibration_mae_after": before_mae,
            "fallback": "identity",
        }
    mean_prediction = sum(sampled_predictions) / float(rows)
    mean_actual = sum(sampled_actuals) / float(rows)
    variance = sum((value - mean_prediction) ** 2 for value in sampled_predictions) / float(rows)
    if variance <= 1e-12:
        a_value = 1.0
        b_value = 0.0
    else:
        covariance = sum(
            (prediction - mean_prediction) * (actual - mean_actual)
            for prediction, actual in zip(sampled_predictions, sampled_actuals)
        ) / float(rows)
        a_value = covariance / variance
        b_value = mean_actual - a_value * mean_prediction
    after_predictions = [a_value * value + b_value for value in sampled_predictions]
    return {
        "mode": "linear",
        "a": float(a_value),
        "b": float(b_value),
        "rows": rows,
        "regression_calibration_rmse_before": before_rmse,
        "regression_calibration_rmse_after": rmse_score(after_predictions, sampled_actuals),
        "regression_calibration_mae_before": before_mae,
        "regression_calibration_mae_after": mae_score(after_predictions, sampled_actuals),
    }


def fit_isotonic_lite_regression_calibration(predictions, actuals, max_rows, buckets):
    sampled_predictions, sampled_actuals = sample_regression_pairs(predictions, actuals, max_rows)
    if np is not None:
        sampled_predictions = np.asarray(sampled_predictions, dtype=np.float32)
        sampled_actuals = np.asarray(sampled_actuals, dtype=np.float32)
        rows = int(len(sampled_actuals))
        before_rmse = rmse_score(sampled_predictions, sampled_actuals)
        before_mae = mae_score(sampled_predictions, sampled_actuals)
        if rows < max(8, buckets):
            return {
                "mode": "isotonic-lite",
                "rows": rows,
                "bucket_edges": [],
                "bucket_values": [],
                "regression_calibration_rmse_before": before_rmse,
                "regression_calibration_rmse_after": before_rmse,
                "regression_calibration_mae_before": before_mae,
                "regression_calibration_mae_after": before_mae,
                "fallback": "identity",
            }
        order = np.argsort(sampled_predictions, kind="mergesort")
        sorted_predictions = sampled_predictions[order]
        sorted_actuals = sampled_actuals[order]
        edges = []
        bucket_values = []
        bucket_count = max(2, int(buckets))
        for bucket_index in range(bucket_count):
            start = int(bucket_index * rows / float(bucket_count))
            end = int((bucket_index + 1) * rows / float(bucket_count))
            if end <= start:
                continue
            edges.append(float(sorted_predictions[end - 1]))
            bucket_values.append(float(np.mean(sorted_actuals[start:end])))
        if bucket_values:
            bucket_values = list(np.maximum.accumulate(np.asarray(bucket_values, dtype=np.float32)))
        calibrated = apply_regression_calibration(sampled_predictions.copy(), {
            "mode": "isotonic-lite",
            "bucket_edges": edges,
            "bucket_values": bucket_values,
        })
        return {
            "mode": "isotonic-lite",
            "rows": rows,
            "bucket_edges": edges,
            "bucket_values": bucket_values,
            "regression_calibration_rmse_before": before_rmse,
            "regression_calibration_rmse_after": rmse_score(calibrated, sampled_actuals),
            "regression_calibration_mae_before": before_mae,
            "regression_calibration_mae_after": mae_score(calibrated, sampled_actuals),
        }

    sampled_predictions = [float(value) for value in sampled_predictions]
    sampled_actuals = [float(value) for value in sampled_actuals]
    rows = len(sampled_actuals)
    before_rmse = rmse_score(sampled_predictions, sampled_actuals)
    before_mae = mae_score(sampled_predictions, sampled_actuals)
    if rows < max(8, buckets):
        return {
            "mode": "isotonic-lite",
            "rows": rows,
            "bucket_edges": [],
            "bucket_values": [],
            "regression_calibration_rmse_before": before_rmse,
            "regression_calibration_rmse_after": before_rmse,
            "regression_calibration_mae_before": before_mae,
            "regression_calibration_mae_after": before_mae,
            "fallback": "identity",
        }
    ordered = sorted(zip(sampled_predictions, sampled_actuals), key=lambda pair: pair[0])
    edges = []
    bucket_values = []
    bucket_count = max(2, int(buckets))
    for bucket_index in range(bucket_count):
        start = int(bucket_index * rows / float(bucket_count))
        end = int((bucket_index + 1) * rows / float(bucket_count))
        if end <= start:
            continue
        chunk = ordered[start:end]
        edges.append(float(chunk[-1][0]))
        bucket_values.append(sum(actual for _, actual in chunk) / float(len(chunk)))
    monotonic_values = []
    running = -float("inf")
    for value in bucket_values:
        running = max(running, value)
        monotonic_values.append(running)
    calibrated = apply_regression_calibration(sampled_predictions[:], {
        "mode": "isotonic-lite",
        "bucket_edges": edges,
        "bucket_values": monotonic_values,
    })
    return {
        "mode": "isotonic-lite",
        "rows": rows,
        "bucket_edges": edges,
        "bucket_values": monotonic_values,
        "regression_calibration_rmse_before": before_rmse,
        "regression_calibration_rmse_after": rmse_score(calibrated, sampled_actuals),
        "regression_calibration_mae_before": before_mae,
        "regression_calibration_mae_after": mae_score(calibrated, sampled_actuals),
    }


def fit_regression_calibration(predicted_trade_returns, rows, args):
    mode = getattr(args, "regression_calibration", "none")
    if mode == "none" or predicted_trade_returns is None:
        return None
    actuals = actual_trade_returns(rows)
    if mode == "linear":
        return fit_linear_regression_calibration(predicted_trade_returns, actuals, args.regression_calibration_max_rows)
    if mode == "isotonic-lite":
        return fit_isotonic_lite_regression_calibration(
            predicted_trade_returns,
            actuals,
            args.regression_calibration_max_rows,
            args.regression_calibration_buckets,
        )
    return None


def apply_regression_calibration(values, calibration, batch_size=200000):
    if not calibration:
        return values
    mode = calibration.get("mode", "none")
    if mode == "linear":
        return apply_linear_regression_calibration(values, calibration.get("a", 1.0), calibration.get("b", 0.0), batch_size)
    if mode == "isotonic-lite":
        edges = calibration.get("bucket_edges", [])
        bucket_values = calibration.get("bucket_values", [])
        if not edges or not bucket_values:
            return values
        if np is not None and isinstance(values, np.ndarray):
            edge_array = np.asarray(edges, dtype=np.float32)
            value_array = np.asarray(bucket_values, dtype=np.float32)
            for start in range(0, len(values), batch_size):
                end = min(len(values), start + batch_size)
                indices = np.searchsorted(edge_array, values[start:end], side="left")
                indices = np.clip(indices, 0, len(value_array) - 1)
                values[start:end] = value_array[indices]
            return values
        calibrated = []
        for value in values:
            index = bisect.bisect_left(edges, float(value))
            if index >= len(bucket_values):
                index = len(bucket_values) - 1
            calibrated.append(float(bucket_values[index]))
        return calibrated
    return values


def fit_platt_calibration(probabilities, labels_values, max_rows):
    sampled_probabilities, sampled_labels = sample_probability_label_pairs(probabilities, labels_values, max_rows)
    if np is not None:
        sampled_probabilities = np.asarray(sampled_probabilities, dtype=np.float32)
        sampled_labels = np.asarray(sampled_labels, dtype=np.float32)
        logits = np.clip(sampled_probabilities, 1e-6, 1.0 - 1e-6)
        logits = np.log(logits / (1.0 - logits)).astype(np.float32, copy=False)
        before = brier_score(sampled_probabilities, sampled_labels)
        a = 1.0
        b = 0.0
        learning_rate = 0.05
        for _ in range(200):
            z_value = np.clip(a * logits + b, -40.0, 40.0)
            calibrated = (1.0 / (1.0 + np.exp(-z_value))).astype(np.float32, copy=False)
            error = calibrated - sampled_labels
            gradient_a = float(np.mean(error * logits))
            gradient_b = float(np.mean(error))
            a -= learning_rate * gradient_a
            b -= learning_rate * gradient_b
        after_probabilities = apply_platt_calibration(sampled_probabilities, a, b)
        after = brier_score(after_probabilities, sampled_labels)
        del sampled_probabilities
        del sampled_labels
        del logits
        del after_probabilities
        gc.collect()
        return {
            "mode": "platt",
            "a": float(a),
            "b": float(b),
            "rows": int(len(labels_values) if not max_rows else min(len(labels_values), max_rows)),
            "validation_brier_before": before,
            "validation_brier_after": after,
        }

    sampled_probabilities = [clipped_probability(value) for value in sampled_probabilities]
    logits = [math.log(value / (1.0 - value)) for value in sampled_probabilities]
    before = brier_score(sampled_probabilities, sampled_labels)
    a = 1.0
    b = 0.0
    learning_rate = 0.05
    for _ in range(200):
        gradient_a = 0.0
        gradient_b = 0.0
        for logit_value, label in zip(logits, sampled_labels):
            calibrated = sigmoid(a * logit_value + b)
            error = calibrated - float(label)
            gradient_a += error * logit_value
            gradient_b += error
        gradient_a /= float(len(logits))
        gradient_b /= float(len(logits))
        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b
    after = brier_score([sigmoid(a * value + b) for value in logits], sampled_labels)
    return {
        "mode": "platt",
        "a": float(a),
        "b": float(b),
        "rows": int(len(sampled_labels)),
        "validation_brier_before": before,
        "validation_brier_after": after,
    }


def apply_platt_calibration(probabilities, a_value, b_value, batch_size=200000):
    if np is not None and isinstance(probabilities, np.ndarray):
        for start in range(0, len(probabilities), batch_size):
            end = min(len(probabilities), start + batch_size)
            clipped = np.clip(np.asarray(probabilities[start:end], dtype=np.float32), 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)
            z_value = np.clip(a_value * logits + b_value, -40.0, 40.0)
            probabilities[start:end] = (1.0 / (1.0 + np.exp(-z_value))).astype(np.float32, copy=False)
        return probabilities
    return [sigmoid(a_value * logit(probability) + b_value) for probability in probabilities]


def fit_calibration(probabilities, rows, args):
    if args.calibration != "platt":
        return None
    labels_values = rows.labels_array() if is_compact_rows(rows) else [row.label for row in rows]
    return fit_platt_calibration(probabilities, labels_values, args.calibration_max_rows)


def calibrate_probabilities(probabilities, calibration):
    if not calibration or calibration.get("mode") != "platt":
        return probabilities
    return apply_platt_calibration(probabilities, calibration["a"], calibration["b"])


def score_name_for_args(args):
    explicit = getattr(args, "_explicit_flags", set())
    if "trade_score" in explicit and args.trade_score != "auto":
        return args.trade_score
    if args.objective_mode == "return_regression":
        return "predicted_return"
    if args.objective_mode == "hybrid":
        return "hybrid"
    return "ev" if args.threshold_objective == "ev" else "probability"


def calibrated_probability_value(bundle, index):
    values = bundle.get("calibrated_probability")
    if values is None:
        values = bundle.get("probability")
    if values is None:
        return 0.0
    return float(values[index])


def raw_probability_value(bundle, index):
    values = bundle.get("probability")
    if values is None:
        return 0.0
    return float(values[index])


def predicted_trade_return_value(bundle, index):
    values = bundle.get("predicted_trade_return")
    if values is None:
        return 0.0
    return float(values[index])


def raw_predicted_trade_return_value(bundle, index):
    values = bundle.get("raw_predicted_trade_return")
    if values is None:
        return predicted_trade_return_value(bundle, index)
    return float(values[index])


def predicted_net_return_value(bundle, index, fee, slippage):
    return predicted_trade_return_value(bundle, index) - fee - slippage


def predicted_return_uncertainty_value(bundle, index):
    values = bundle.get("predicted_return_uncertainty")
    if values is None:
        return 0.0
    return float(values[index])


def meta_probability_value(bundle, index):
    values = bundle.get("meta_probability")
    if values is None:
        return 1.0
    return float(values[index])


def hybrid_score_basic_value(bundle, index, fee, slippage, args=None,
                             upside_target=0.05, downside_stop=0.02):
    return hybrid_score_details_for_bundle(
        bundle,
        index,
        fee,
        slippage,
        "basic",
        0.0,
        args,
        upside_target,
        downside_stop,
    )["base_hybrid_score"]


def hybrid_score_value(bundle, index, fee, slippage,
                       hybrid_score_mode="basic",
                       hybrid_uncertainty_penalty=0.0,
                       args=None,
                       upside_target=0.05,
                       downside_stop=0.02):
    return hybrid_score_details_for_bundle(
        bundle,
        index,
        fee,
        slippage,
        hybrid_score_mode,
        hybrid_uncertainty_penalty,
        args,
        upside_target,
        downside_stop,
    )["hybrid_score"]


def expected_value_for_bundle(bundle, index, upside_target, downside_stop, fee, slippage, args=None):
    probability = calibrated_probability_value(bundle, index)
    if probability <= 0.0 and bundle.get("probability") is None and bundle.get("calibrated_probability") is None:
        return 0.0
    ev_context = resolve_ev_context(bundle, args, upside_target, downside_stop)
    predicted_trade_return = (
        predicted_trade_return_value(bundle, index)
        if bundle.get("predicted_trade_return") is not None
        else None
    )
    return expected_value_components(
        probability,
        predicted_trade_return,
        fee,
        slippage,
        getattr(args, "objective_mode", "classification") if args is not None else "classification",
        ev_context,
    )["expected_value"]


def expected_value_details_for_bundle(bundle, index, upside_target, downside_stop, fee, slippage, args=None):
    probability = calibrated_probability_value(bundle, index)
    ev_context = resolve_ev_context(bundle, args, upside_target, downside_stop)
    predicted_trade_return = (
        predicted_trade_return_value(bundle, index)
        if bundle.get("predicted_trade_return") is not None
        else None
    )
    details = expected_value_components(
        probability,
        predicted_trade_return,
        fee,
        slippage,
        getattr(args, "objective_mode", "classification") if args is not None else "classification",
        ev_context,
    )
    details.update({
        "ev_payoff_mode": ev_context.get("ev_payoff_mode", "fixed_targets"),
        "ev_payoff_source": ev_context.get("ev_payoff_source", "fixed_targets"),
        "ev_expected_win_return": float(ev_context.get("ev_expected_win_return", 0.0)),
        "ev_expected_loss_return": float(ev_context.get("ev_expected_loss_return", 0.0)),
    })
    return details


def trade_score_value(bundle, index, score_name, upside_target, downside_stop, fee, slippage,
                      hybrid_score_mode="basic", hybrid_uncertainty_penalty=0.0,
                      args=None):
    if score_name == "probability":
        return calibrated_probability_value(bundle, index)
    if score_name == "ev":
        return expected_value_for_bundle(bundle, index, upside_target, downside_stop, fee, slippage, args)
    if score_name == "predicted_return":
        return predicted_net_return_value(bundle, index, fee, slippage)
    if score_name == "hybrid":
        return hybrid_score_value(
            bundle,
            index,
            fee,
            slippage,
            hybrid_score_mode,
            hybrid_uncertainty_penalty,
            args,
            upside_target,
            downside_stop,
        )
    raise ValueError("unknown trade score: {}".format(score_name))


def maybe_float32_array(values):
    if values is None:
        return None
    if np is None:
        return values
    return np.asarray(values, dtype=np.float32)


def score_batch_ev(probabilities, upside_target, downside_stop, fee, slippage,
                   predicted_trade_returns=None,
                   objective_mode="classification",
                   ev_context=None):
    context = ev_context or default_ev_context(None, upside_target, downside_stop)
    source = context.get("ev_payoff_source", context.get("ev_payoff_actual_mode", "fixed_targets"))
    if source == "predicted_return" and predicted_trade_returns is not None:
        if np is not None and isinstance(probabilities, np.ndarray):
            if objective_mode == "return_regression":
                return predicted_trade_returns.astype(np.float32, copy=False) - np.float32(fee + slippage)
            return probabilities.astype(np.float32, copy=False) * predicted_trade_returns.astype(np.float32, copy=False) - np.float32(fee + slippage)
        if objective_mode == "return_regression":
            return [float(value) - fee - slippage for value in predicted_trade_returns]
        return [
            float(probability) * float(predicted_trade_return) - fee - slippage
            for probability, predicted_trade_return in zip(probabilities, predicted_trade_returns)
        ]
    if source.startswith("empirical_validation"):
        win_return = float(context.get("empirical_expected_win_return", upside_target))
        loss_return = float(context.get("empirical_expected_loss_return", -downside_stop))
        if np is not None and isinstance(probabilities, np.ndarray):
            values = probabilities.astype(np.float32, copy=False)
            return values * np.float32(win_return) + (1.0 - values) * np.float32(loss_return) - np.float32(fee + slippage)
        return [
            float(probability) * win_return + (1.0 - float(probability)) * loss_return - fee - slippage
            for probability in probabilities
        ]
    if np is not None and isinstance(probabilities, np.ndarray):
        values = probabilities.astype(np.float32, copy=False)
        return values * np.float32(context.get("fixed_expected_win_return", upside_target)) - (
            1.0 - values
        ) * np.float32(-context.get("fixed_expected_loss_return", -downside_stop)) - np.float32(fee + slippage)
    return [
        expected_value_from_probability(
            probability,
            float(context.get("fixed_expected_win_return", upside_target)),
            float(-context.get("fixed_expected_loss_return", -downside_stop)),
            fee,
            slippage,
        )
        for probability in probabilities
    ]


def score_batch_hybrid(probabilities, predicted_trade_returns, fee, slippage,
                       predicted_return_uncertainty=None,
                       hybrid_score_mode="basic",
                       hybrid_uncertainty_penalty=0.0,
                       hybrid_context=None,
                       uncertainty_context=None,
                       hybrid_runtime_args=None):
    context = hybrid_context or default_hybrid_return_context(None, 0.05, 0.02)
    combination = context.get("hybrid_return_combination", "probability_times_return")
    if np is not None and isinstance(probabilities, np.ndarray):
        probability_values = probabilities.astype(np.float32, copy=False)
        if combination == "expected_return":
            if predicted_trade_returns is None:
                values = np.zeros(len(probability_values), dtype=np.float32)
            else:
                values = np.asarray(predicted_trade_returns, dtype=np.float32) - np.float32(fee + slippage)
        elif combination == "conditional_payoff":
            expected_win = np.float32(context.get("conditional_expected_win_return", 0.0))
            expected_loss = np.float32(context.get("conditional_expected_loss_return", 0.0))
            values = probability_values * expected_win + (1.0 - probability_values) * expected_loss - np.float32(fee + slippage)
        else:
            if predicted_trade_returns is None:
                values = np.zeros(len(probability_values), dtype=np.float32)
            else:
                values = probability_values * np.asarray(predicted_trade_returns, dtype=np.float32) - np.float32(fee + slippage)
        if hybrid_score_mode == "risk_adjusted" and predicted_return_uncertainty is not None and hybrid_uncertainty_penalty > 0.0:
            values = values - uncertainty_penalty_values_array(
                predicted_return_uncertainty,
                hybrid_uncertainty_penalty,
                hybrid_runtime_args,
                uncertainty_context,
            )
        return values
    scored = []
    for index, probability in enumerate(probabilities):
        predicted_trade_return = None if predicted_trade_returns is None else float(predicted_trade_returns[index])
        scored.append(
            hybrid_base_score_components(
                probability,
                predicted_trade_return,
                fee,
                slippage,
                context,
            )
        )
    if hybrid_score_mode == "risk_adjusted" and predicted_return_uncertainty is not None and hybrid_uncertainty_penalty > 0.0:
        return [
            score - uncertainty_penalty_amount(
                predicted_return_uncertainty[index],
                hybrid_uncertainty_penalty,
                hybrid_runtime_args,
                uncertainty_context,
            )
            for index, score in enumerate(scored)
        ]
    return scored


def fit_uncertainty_model(predicted_trade_returns, rows, args):
    mode = getattr(args, "hybrid_uncertainty_method", "none")
    penalty_mode = getattr(args, "hybrid_uncertainty_penalty_mode", "raw")
    if mode == "none" or predicted_trade_returns is None:
        return None
    sampled_predictions, sampled_actuals = sample_regression_pairs(
        predicted_trade_returns,
        actual_trade_returns(rows),
        args.regression_calibration_max_rows,
    )
    if np is not None:
        sampled_predictions = np.asarray(sampled_predictions, dtype=np.float32)
        sampled_actuals = np.asarray(sampled_actuals, dtype=np.float32)
        residuals = sampled_actuals - sampled_predictions
        rows_count = int(len(residuals))
        penalty_reference_scale = float(
            max(
                np.percentile(np.abs(sampled_predictions), 75.0) if rows_count else 0.0,
                np.percentile(np.abs(sampled_actuals), 75.0) if rows_count else 0.0,
                1e-6,
            )
        )
        if rows_count < 8:
            return {
                "mode": mode,
                "rows": rows_count,
                "global_std": 0.0,
                "bucket_edges": [],
                "bucket_stds": [],
                "fallback": "insufficient_rows",
                "penalty_mode": penalty_mode,
                "penalty_reference_scale": penalty_reference_scale,
            }
        if mode == "global_residual":
            return {
                "mode": mode,
                "rows": rows_count,
                "global_std": float(np.std(residuals)),
                "penalty_mode": penalty_mode,
                "penalty_reference_scale": penalty_reference_scale,
            }
        order = np.argsort(sampled_predictions, kind="mergesort")
        sorted_predictions = sampled_predictions[order]
        sorted_residuals = residuals[order]
        bucket_edges = []
        bucket_stds = []
        bucket_count = max(2, int(args.hybrid_uncertainty_buckets))
        for bucket_index in range(bucket_count):
            start = int(bucket_index * rows_count / float(bucket_count))
            end = int((bucket_index + 1) * rows_count / float(bucket_count))
            if end <= start:
                continue
            bucket_edges.append(float(sorted_predictions[end - 1]))
            bucket_stds.append(float(np.std(sorted_residuals[start:end])))
        return {
            "mode": mode,
            "rows": rows_count,
            "global_std": float(np.std(residuals)),
            "bucket_edges": bucket_edges,
            "bucket_stds": bucket_stds,
            "penalty_mode": penalty_mode,
            "penalty_reference_scale": penalty_reference_scale,
        }
    sampled_predictions = [float(value) for value in sampled_predictions]
    sampled_actuals = [float(value) for value in sampled_actuals]
    residuals = [actual - prediction for prediction, actual in zip(sampled_predictions, sampled_actuals)]
    rows_count = len(residuals)
    abs_predictions = sorted(abs(value) for value in sampled_predictions)
    abs_actuals = sorted(abs(value) for value in sampled_actuals)
    percentile_index = int(round((rows_count - 1) * 0.75)) if rows_count else 0
    penalty_reference_scale = max(
        abs_predictions[percentile_index] if abs_predictions else 0.0,
        abs_actuals[percentile_index] if abs_actuals else 0.0,
        1e-6,
    )
    if rows_count < 8:
        return {
            "mode": mode,
            "rows": rows_count,
            "global_std": 0.0,
            "bucket_edges": [],
            "bucket_stds": [],
            "fallback": "insufficient_rows",
            "penalty_mode": penalty_mode,
            "penalty_reference_scale": penalty_reference_scale,
        }
    global_std = math.sqrt(sum(value * value for value in residuals) / float(rows_count))
    if mode == "global_residual":
        return {
            "mode": mode,
            "rows": rows_count,
            "global_std": global_std,
            "penalty_mode": penalty_mode,
            "penalty_reference_scale": penalty_reference_scale,
        }
    ordered = sorted(zip(sampled_predictions, residuals), key=lambda pair: pair[0])
    bucket_edges = []
    bucket_stds = []
    bucket_count = max(2, int(args.hybrid_uncertainty_buckets))
    for bucket_index in range(bucket_count):
        start = int(bucket_index * rows_count / float(bucket_count))
        end = int((bucket_index + 1) * rows_count / float(bucket_count))
        if end <= start:
            continue
        chunk = ordered[start:end]
        bucket_edges.append(float(chunk[-1][0]))
        mean_value = sum(residual for _, residual in chunk) / float(len(chunk))
        variance = sum((residual - mean_value) ** 2 for _, residual in chunk) / float(len(chunk))
        bucket_stds.append(math.sqrt(variance))
    return {
        "mode": mode,
        "rows": rows_count,
        "global_std": global_std,
        "bucket_edges": bucket_edges,
        "bucket_stds": bucket_stds,
        "penalty_mode": penalty_mode,
        "penalty_reference_scale": penalty_reference_scale,
    }


def apply_uncertainty_model(predicted_trade_returns, uncertainty_model, batch_size=200000):
    if predicted_trade_returns is None or not uncertainty_model:
        return None
    mode = uncertainty_model.get("mode", "none")
    if mode == "none":
        return None
    if mode == "global_residual":
        value = float(uncertainty_model.get("global_std", 0.0))
        if np is not None and isinstance(predicted_trade_returns, np.ndarray):
            return np.full(len(predicted_trade_returns), value, dtype=np.float32)
        return [value] * len(predicted_trade_returns)
    bucket_edges = uncertainty_model.get("bucket_edges", [])
    bucket_stds = uncertainty_model.get("bucket_stds", [])
    if not bucket_edges or not bucket_stds:
        value = float(uncertainty_model.get("global_std", 0.0))
        if np is not None and isinstance(predicted_trade_returns, np.ndarray):
            return np.full(len(predicted_trade_returns), value, dtype=np.float32)
        return [value] * len(predicted_trade_returns)
    if np is not None and isinstance(predicted_trade_returns, np.ndarray):
        result = np.empty(len(predicted_trade_returns), dtype=np.float32)
        edge_array = np.asarray(bucket_edges, dtype=np.float32)
        std_array = np.asarray(bucket_stds, dtype=np.float32)
        for start in range(0, len(predicted_trade_returns), batch_size):
            end = min(len(predicted_trade_returns), start + batch_size)
            indices = np.searchsorted(edge_array, predicted_trade_returns[start:end], side="left")
            indices = np.clip(indices, 0, len(std_array) - 1)
            result[start:end] = std_array[indices]
        return result
    values = []
    for predicted in predicted_trade_returns:
        index = bisect.bisect_left(bucket_edges, float(predicted))
        if index >= len(bucket_stds):
            index = len(bucket_stds) - 1
        values.append(float(bucket_stds[index]))
    return values


def compute_dynamic_hybrid_thresholds(rows, args, base_threshold):
    mode = getattr(args, "dynamic_hybrid_thresholds", "none")
    if mode == "none":
        return None, None
    btc_return = row_feature_array(rows, "btc_return_240m")
    if btc_return is None:
        btc_return = row_feature_array(rows, "btc_return_60m")
    volatility = row_feature_array(rows, "btc_volatility_60m")
    if volatility is None:
        volatility = row_feature_array(rows, "rolling_volatility_60m")
    if mode in ("btc_regime", "btc_volatility_regime") and btc_return is None:
        warn_once(
            "dynamic-hybrid-threshold-btc-missing",
            "Warning: dynamic hybrid thresholds requested but BTC regime features are unavailable; falling back to static hybrid_min_score.",
        )
        return None, None
    if mode in ("volatility_regime", "btc_volatility_regime") and volatility is None:
        warn_once(
            "dynamic-hybrid-threshold-vol-missing",
            "Warning: dynamic hybrid thresholds requested but volatility features are unavailable; falling back to static hybrid_min_score.",
        )
        return None, None
    count = len(rows)
    if np is not None:
        thresholds = np.empty(count, dtype=np.float32)
    else:
        thresholds = [0.0] * count
    buckets = ["" for _ in range(count)]
    for index in range(count):
        threshold_value = float(base_threshold)
        bucket = "static"
        btc_value = float(btc_return[index]) if btc_return is not None else 0.0
        volatility_value = float(volatility[index]) if volatility is not None else 0.0
        bullish = btc_value > args.btc_bullish_threshold
        bearish = btc_value < args.btc_bearish_threshold
        high_vol = volatility_value > args.volatility_high_threshold
        if mode == "btc_regime":
            if bullish:
                threshold_value = args.hybrid_min_score_bullish
                bucket = "bullish"
            elif bearish:
                threshold_value = args.hybrid_min_score_bearish
                bucket = "bearish"
            else:
                threshold_value = args.hybrid_min_score_neutral
                bucket = "neutral"
        elif mode == "volatility_regime":
            if high_vol:
                threshold_value = args.hybrid_min_score_high_vol
                bucket = "high_vol"
            else:
                threshold_value = args.hybrid_min_score_normal_vol
                bucket = "normal_vol"
        elif mode == "btc_volatility_regime":
            if bearish or high_vol:
                threshold_value = max(args.hybrid_min_score_bearish, args.hybrid_min_score_high_vol)
                bucket = "bearish_or_high_vol"
            elif bullish and not high_vol:
                threshold_value = min(args.hybrid_min_score_bullish, args.hybrid_min_score_normal_vol)
                bucket = "bullish_normal_vol"
            else:
                threshold_value = args.hybrid_min_score_neutral
                bucket = "neutral"
        thresholds[index] = np.float32(threshold_value) if np is not None else threshold_value
        buckets[index] = bucket
    return thresholds, buckets


def parse_ensemble_windows(text):
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    values = []
    seen = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("--ensemble-windows values must be positive integers")
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def month_index_bounds(rows):
    if not rows:
        return 0, 0
    if is_compact_rows(rows):
        month_indices = rows.table.month_indices if rows.indices is None else rows.table.month_indices[rows.indices]
        if len(month_indices) == 0:
            return 0, 0
        if np is not None:
            return int(np.min(month_indices)), int(np.max(month_indices))
        return int(min(month_indices)), int(max(month_indices))
    values = [int(row.month_index) for row in rows]
    return min(values), max(values)


def recent_month_window_rows(rows, month_count):
    if month_count <= 0 or not rows:
        return rows
    start_month, end_month = month_index_bounds(rows)
    end_exclusive = end_month + 1
    start = max(start_month, end_exclusive - month_count)
    return select_month_range(rows, start, end_exclusive)


def concatenate_row_views(*parts):
    non_empty = [part for part in parts if part is not None and len(part) > 0]
    if not non_empty:
        return [] if not parts else parts[0]
    if len(non_empty) == 1:
        return non_empty[0]
    compact = is_compact_rows(non_empty[0])
    if any(is_compact_rows(part) != compact for part in non_empty[1:]):
        raise ValueError("cannot concatenate mixed row container types")
    if compact:
        table = non_empty[0].table
        merged_parts = []
        for part in non_empty:
            if part.table is not table:
                raise ValueError("cannot concatenate compact rows from different tables")
            if part.indices is None:
                merged_parts.append(np.arange(len(table.labels), dtype=np.int32))
            else:
                merged_parts.append(np.asarray(part.indices, dtype=np.int32))
        merged = np.concatenate(merged_parts).astype(np.int32, copy=False)
        if merged.size > 1 and np.any(merged[1:] < merged[:-1]):
            merged = np.sort(merged, kind="mergesort")
        return non_empty[0].subset(merged)
    merged = []
    for part in non_empty:
        merged.extend(part)
    return merged


def prepare_fixed_split_backfill(train_rows, validation_rows, month_count):
    if month_count <= 0 or not train_rows or not validation_rows:
        return None
    backfill_rows = recent_month_window_rows(train_rows, month_count)
    if not backfill_rows or len(backfill_rows) >= len(train_rows):
        return None
    train_start_month, _ = month_index_bounds(train_rows)
    backfill_start_month, backfill_end_month = month_index_bounds(backfill_rows)
    if backfill_start_month <= train_start_month:
        return None
    reduced_train_rows = select_month_range(train_rows, train_start_month, backfill_start_month)
    if not reduced_train_rows:
        return None
    combined_validation_rows = concatenate_row_views(backfill_rows, validation_rows)
    return {
        "train_rows": reduced_train_rows,
        "validation_rows": combined_validation_rows,
        "backfill_rows": backfill_rows,
        "month_count": int(backfill_end_month - backfill_start_month + 1),
        "backfill_start_month": int(backfill_start_month),
        "backfill_end_month": int(backfill_end_month),
    }


class InternalMetaLogistic(object):
    def __init__(self, epochs=120, learning_rate=0.05, l2=0.001):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.mean = None
        self.scale = None
        self.weights = None
        self.bias = 0.0

    def fit(self, x_rows, y_rows):
        row_count = len(x_rows)
        if row_count == 0 or len(y_rows) != row_count:
            raise ValueError("meta filter requires non-empty aligned features and labels")
        if np is not None:
            x = np.asarray(x_rows, dtype=np.float32)
            y = np.asarray(y_rows, dtype=np.float32)
            self.mean = np.mean(x, axis=0, dtype=np.float32)
            self.scale = np.std(x, axis=0, dtype=np.float32)
            self.scale[self.scale < 1e-6] = 1.0
            z = (x - self.mean) / self.scale
            self.weights = np.zeros(z.shape[1], dtype=np.float32)
            self.bias = 0.0
            for _ in range(self.epochs):
                logits = np.clip(np.dot(z, self.weights) + self.bias, -40.0, 40.0)
                predictions = 1.0 / (1.0 + np.exp(-logits))
                error = predictions - y
                gradient_w = np.dot(z.T, error) / float(len(y)) + self.l2 * self.weights
                gradient_b = float(np.mean(error))
                self.weights -= np.float32(self.learning_rate) * gradient_w.astype(np.float32, copy=False)
                self.bias -= self.learning_rate * gradient_b
            return self

        feature_count = len(x_rows[0])
        self.mean = [0.0] * feature_count
        self.scale = [1.0] * feature_count
        for feature_index in range(feature_count):
            column = [float(row[feature_index]) for row in x_rows]
            column_mean = sum(column) / float(len(column))
            variance = sum((value - column_mean) ** 2 for value in column) / float(len(column))
            self.mean[feature_index] = column_mean
            self.scale[feature_index] = math.sqrt(variance) if variance > 1e-12 else 1.0
        self.weights = [0.0] * feature_count
        self.bias = 0.0
        for _ in range(self.epochs):
            gradient_w = [0.0] * feature_count
            gradient_b = 0.0
            for row_index, row in enumerate(x_rows):
                z_row = [
                    (float(row[feature_index]) - self.mean[feature_index]) / self.scale[feature_index]
                    for feature_index in range(feature_count)
                ]
                logit_value = max(-40.0, min(40.0, sum(self.weights[i] * z_row[i] for i in range(feature_count)) + self.bias))
                prediction = sigmoid(logit_value)
                error = prediction - float(y_rows[row_index])
                for feature_index in range(feature_count):
                    gradient_w[feature_index] += error * z_row[feature_index]
                gradient_b += error
            for feature_index in range(feature_count):
                gradient = gradient_w[feature_index] / float(len(x_rows)) + self.l2 * self.weights[feature_index]
                self.weights[feature_index] -= self.learning_rate * gradient
            self.bias -= self.learning_rate * (gradient_b / float(len(x_rows)))
        return self

    def predict_proba(self, x_rows):
        if np is not None:
            x = np.asarray(x_rows, dtype=np.float32)
            z = (x - self.mean) / self.scale
            logits = np.clip(np.dot(z, self.weights) + self.bias, -40.0, 40.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            return np.asarray(probabilities, dtype=np.float32)
        probabilities = []
        for row in x_rows:
            z_row = [
                (float(row[feature_index]) - self.mean[feature_index]) / self.scale[feature_index]
                for feature_index in range(len(self.weights))
            ]
            probabilities.append(sigmoid(sum(self.weights[i] * z_row[i] for i in range(len(self.weights))) + self.bias))
        return probabilities


def meta_feature_matrix(rows, bundle, candidate_indices, args, effective_thresholds=None,
                        fee=None, slippage=None):
    if not candidate_indices:
        return np.empty((0, 0), dtype=np.float32) if np is not None else []
    fee = args.fee if fee is None else fee
    slippage = args.slippage if slippage is None else slippage
    btc_return_60m = feature_values_or_default(rows, "btc_return_60m", 0.0)
    btc_return_240m = feature_values_or_default(rows, "btc_return_240m", 0.0)
    btc_volatility_60m = feature_values_or_default(rows, "btc_volatility_60m", 0.0)
    eth_return_60m = feature_values_or_default(rows, "eth_return_60m", 0.0)
    rolling_volatility_60m = feature_values_or_default(rows, "rolling_volatility_60m", 0.0)
    rolling_quote_volume_zscore_60m = feature_values_or_default(rows, "rolling_quote_volume_zscore_60m", 0.0)
    taker_buy_ratio_zscore_60m = feature_values_or_default(rows, "taker_buy_ratio_zscore_60m", 0.0)
    distance_from_recent_high_60m = feature_values_or_default(rows, "distance_from_recent_high_60m", 0.0)
    distance_from_recent_low_60m = feature_values_or_default(rows, "distance_from_recent_low_60m", 0.0)

    matrix = []
    for selection_rank, local_index in enumerate(candidate_indices, 1):
        effective_threshold = (
            float(effective_thresholds[local_index])
            if effective_thresholds is not None
            else 0.0
        )
        row_values = [
            calibrated_probability_value(bundle, local_index),
            raw_predicted_trade_return_value(bundle, local_index),
            predicted_trade_return_value(bundle, local_index),
            predicted_net_return_value(bundle, local_index, fee, slippage),
            hybrid_score_basic_value(bundle, local_index, fee, slippage, args, args.upside_target, args.downside_stop),
            hybrid_score_value(bundle, local_index, fee, slippage, "risk_adjusted", 1.0, args, args.upside_target, args.downside_stop),
            predicted_return_uncertainty_value(bundle, local_index),
            expected_value_for_bundle(bundle, local_index, args.upside_target, args.downside_stop, fee, slippage, args),
            trade_score_value(
                bundle,
                local_index,
                score_name_for_args(args),
                args.upside_target,
                args.downside_stop,
                fee,
                slippage,
                args.hybrid_score_mode,
                args.hybrid_uncertainty_penalty,
                args,
            ),
            float(btc_return_60m[local_index]),
            float(btc_return_240m[local_index]),
            float(btc_volatility_60m[local_index]),
            float(eth_return_60m[local_index]),
            float(rolling_volatility_60m[local_index]),
            float(rolling_quote_volume_zscore_60m[local_index]),
            float(taker_buy_ratio_zscore_60m[local_index]),
            float(distance_from_recent_high_60m[local_index]),
            float(distance_from_recent_low_60m[local_index]),
            float(selection_rank),
            float(effective_threshold),
        ]
        matrix.append(row_values)
    if np is not None:
        return np.asarray(matrix, dtype=np.float32)
    return matrix


def meta_labels_for_indices(rows, candidate_indices, fee, slippage):
    trade_returns = actual_trade_returns(rows)
    labels_values = []
    for local_index in candidate_indices:
        trade_return = float(trade_returns[local_index])
        labels_values.append(1 if trade_return - fee - slippage > 0.0 else 0)
    if np is not None:
        return np.asarray(labels_values, dtype=np.int8)
    return labels_values


def meta_accuracy(probabilities, labels_values, threshold=0.5):
    if len(labels_values) == 0:
        return 0.0
    correct = 0
    for index, label in enumerate(labels_values):
        correct += 1 if ((float(probabilities[index]) >= threshold) == bool(label)) else 0
    return correct / float(len(labels_values))


def fit_meta_filter(rows, bundle, threshold, args, symbol_filter_info=None):
    mode = getattr(args, "meta_filter", "none")
    if mode == "none":
        return None
    validation_slippage = args.slippage * args.validation_slippage_multiplier
    execution = portfolio_execution(
        rows,
        bundle,
        threshold,
        args.fee,
        validation_slippage,
        args.initial_capital,
        args.max_position_fraction,
        args.max_volume_fraction,
        args.max_trades_per_period,
        args.trade_period_minutes,
        args.holding_period_minutes,
        args.threshold_objective,
        args.trade_selection,
        args.top_k_per_minute,
        args.upside_target,
        args.downside_stop,
        args.ev_safety_margin,
        args.objective_mode,
        score_name_for_args(args),
        args.min_predicted_net_return,
        args.hybrid_min_score,
        args.max_trades_per_day,
        args.max_trades_per_fold,
        args.max_losing_trades_per_day,
        args.max_daily_drawdown,
        args.pause_after_drawdown_minutes,
        capture_blocked_details=False,
        hybrid_runtime_args=args,
        symbol_filter_info=symbol_filter_info,
    )
    candidate_indices = sorted(int(index) for index in execution["raw_selected"].keys())
    if args.meta_filter_max_rows > 0 and len(candidate_indices) > args.meta_filter_max_rows:
        step = max(1, int(math.ceil(len(candidate_indices) / float(args.meta_filter_max_rows))))
        candidate_indices = candidate_indices[::step][:args.meta_filter_max_rows]
    effective_thresholds, _ = compute_dynamic_hybrid_thresholds(
        rows,
        args,
        max(float(threshold), float(args.hybrid_min_score)),
    ) if args.objective_mode == "hybrid" else (None, None)
    if len(candidate_indices) < 20:
        return {
            "mode": mode,
            "enabled": False,
            "rows": len(candidate_indices),
            "positive_rate": 0.0,
            "accuracy": 0.0,
            "auc": 0.0,
            "disabled_reason": "insufficient_candidates",
        }
    x_all = meta_feature_matrix(
        rows,
        bundle,
        candidate_indices,
        args,
        effective_thresholds,
        fee=args.fee,
        slippage=validation_slippage,
    )
    y_all = meta_labels_for_indices(rows, candidate_indices, args.fee, validation_slippage)
    positive_rate = float(np.mean(y_all)) if np is not None else (sum(y_all) / float(len(y_all)))
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return {
            "mode": mode,
            "enabled": False,
            "rows": len(candidate_indices),
            "positive_rate": positive_rate,
            "accuracy": max(positive_rate, 1.0 - positive_rate),
            "auc": 0.0,
            "disabled_reason": "single_class_candidates",
        }
    split_at = max(8, int(len(candidate_indices) * 0.8))
    split_at = min(split_at, len(candidate_indices) - 1)
    x_train = x_all[:split_at]
    y_train = y_all[:split_at]
    x_eval = x_all[split_at:]
    y_eval = y_all[split_at:]
    if len(y_eval) == 0:
        x_eval = x_train
        y_eval = y_train
    if mode == "lightgbm":
        if not external_available("lightgbm"):
            warn_once(
                "meta-filter-lightgbm-missing",
                "Warning: --meta-filter lightgbm requested but LightGBM is unavailable; falling back to logistic meta filter.",
            )
            mode = "logistic"
        else:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=80,
                learning_rate=0.05,
                num_leaves=15,
                max_depth=4,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.85,
                min_child_samples=10,
                reg_alpha=0.25,
                reg_lambda=1.0,
                n_jobs=max(1, int(getattr(args, "n_jobs", 1))),
                objective="binary",
                random_state=17,
                verbosity=-1,
                force_col_wise=True,
            )
            model.fit(x_train, y_train)
            eval_probabilities = np.asarray(model.predict_proba(x_eval)[:, 1], dtype=np.float32) if np is not None else [float(row[1]) for row in model.predict_proba(x_eval)]
            return {
                "mode": "lightgbm",
                "enabled": True,
                "rows": len(candidate_indices),
                "positive_rate": positive_rate,
                "accuracy": meta_accuracy(eval_probabilities, y_eval, args.meta_filter_min_probability),
                "auc": 0.0,
                "model": model,
            }
    model = InternalMetaLogistic()
    model.fit(x_train, y_train)
    eval_probabilities = model.predict_proba(x_eval)
    return {
        "mode": "logistic",
        "enabled": True,
        "rows": len(candidate_indices),
        "positive_rate": positive_rate,
        "accuracy": meta_accuracy(eval_probabilities, y_eval, args.meta_filter_min_probability),
        "auc": 0.0,
        "model": model,
    }


def apply_meta_filter(rows, bundle, threshold, args, meta_filter_info, symbol_filter_info=None):
    if not meta_filter_info or not meta_filter_info.get("enabled") or meta_filter_info.get("model") is None:
        return None
    test_slippage = args.slippage * args.test_slippage_multiplier
    execution = portfolio_execution(
        rows,
        bundle,
        threshold,
        args.fee,
        test_slippage,
        args.initial_capital,
        args.max_position_fraction,
        args.max_volume_fraction,
        args.max_trades_per_period,
        args.trade_period_minutes,
        args.holding_period_minutes,
        args.threshold_objective,
        args.trade_selection,
        args.top_k_per_minute,
        args.upside_target,
        args.downside_stop,
        args.ev_safety_margin,
        args.objective_mode,
        score_name_for_args(args),
        args.min_predicted_net_return,
        args.hybrid_min_score,
        args.max_trades_per_day,
        args.max_trades_per_fold,
        args.max_losing_trades_per_day,
        args.max_daily_drawdown,
        args.pause_after_drawdown_minutes,
        capture_blocked_details=False,
        hybrid_runtime_args=args,
        symbol_filter_info=symbol_filter_info,
    )
    candidate_indices = sorted(int(index) for index in execution["raw_selected"].keys())
    if not candidate_indices:
        return None
    effective_thresholds, _ = compute_dynamic_hybrid_thresholds(
        rows,
        args,
        max(float(threshold), float(args.hybrid_min_score)),
    ) if args.objective_mode == "hybrid" else (None, None)
    x_all = meta_feature_matrix(
        rows,
        bundle,
        candidate_indices,
        args,
        effective_thresholds,
        fee=args.fee,
        slippage=test_slippage,
    )
    probabilities = meta_filter_info["model"].predict_proba(x_all)
    if np is not None:
        output = np.zeros(len(rows), dtype=np.float32)
        output[np.asarray(candidate_indices, dtype=np.int64)] = np.asarray(probabilities, dtype=np.float32)
        return output
    output = [0.0] * len(rows)
    for offset, local_index in enumerate(candidate_indices):
        output[local_index] = float(probabilities[offset])
    return output


def disabled_meta_filter_info(meta_filter_info, reason, extra=None):
    if meta_filter_info is None:
        return None
    result = dict(meta_filter_info)
    result["enabled"] = False
    result["disabled_reason"] = reason
    if extra:
        result.update(extra)
    return result


def recalibrate_meta_filter_validation(rows, bundle, threshold, args, selection,
                                       meta_filter_info, symbol_filter_info,
                                       selected_score_name):
    if not meta_filter_info or not meta_filter_info.get("enabled"):
        return meta_filter_info, selection
    baseline_metrics = selection.get("validation_metrics", {})
    baseline_score = float(
        selection.get(
            "objective_score",
            baseline_metrics.get("selected_objective_score", -float("inf")),
        )
    )
    meta_probability = apply_meta_filter(
        rows,
        bundle,
        threshold,
        args,
        meta_filter_info,
        symbol_filter_info,
    )
    if meta_probability is None:
        return disabled_meta_filter_info(meta_filter_info, "no_meta_candidates"), selection
    validation_bundle = dict(bundle)
    validation_bundle["meta_probability"] = meta_probability
    meta_metrics = evaluate(
        rows,
        validation_bundle,
        threshold,
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
        trade_score_name=score_name_for_args(args),
        min_predicted_net_return=args.min_predicted_net_return,
        hybrid_min_score=args.hybrid_min_score,
        max_trades_per_day=args.max_trades_per_day,
        max_trades_per_fold=0,
        max_losing_trades_per_day=args.max_losing_trades_per_day,
        max_daily_drawdown=args.max_daily_drawdown,
        pause_after_drawdown_minutes=args.pause_after_drawdown_minutes,
        hybrid_runtime_args=args,
        symbol_filter_info=symbol_filter_info,
    )
    meta_selection = build_selected_threshold_result(
        threshold,
        meta_metrics,
        args.threshold_objective,
        0.0 if args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev") and args.profit_safety == "strict" else -float("inf"),
        args.threshold_drawdown_penalty,
        args.threshold_trade_count_penalty,
        args.target_validation_trades,
        selected_score_name,
        args.min_validation_trades,
        args.max_validation_trades,
    )
    trade_count = int(meta_selection["validation_metrics"].get("predicted_trades", 0))
    raw_signal_count = int(meta_selection["validation_metrics"].get("raw_signal_trades", 0))
    precision = float(meta_selection["validation_metrics"].get("precision", 0.0))
    meta_info = dict(meta_filter_info)
    meta_info["validation_trade_count"] = trade_count
    meta_info["validation_raw_signal_count"] = raw_signal_count
    meta_info["validation_trade_retention"] = float(trade_count) / float(raw_signal_count) if raw_signal_count else 0.0
    meta_info["validation_portfolio_profit"] = float(meta_selection["validation_metrics"].get("portfolio_profit", 0.0))
    meta_info["validation_objective_score"] = float(meta_selection.get("objective_score", baseline_score))
    if trade_count < int(args.min_validation_trades):
        return disabled_meta_filter_info(meta_info, "validation_under_min_trades"), selection
    if args.max_validation_trades > 0 and trade_count > int(args.max_validation_trades):
        return disabled_meta_filter_info(meta_info, "validation_over_max_trades"), selection
    if precision + 1e-12 < float(args.min_validation_precision):
        return disabled_meta_filter_info(meta_info, "validation_under_min_precision"), selection
    if float(meta_selection.get("objective_score", -float("inf"))) + 1e-12 < baseline_score:
        return disabled_meta_filter_info(meta_info, "validation_underperformed"), selection
    return meta_info, meta_selection


def compact_signal_indices_for_bundle(rows, bundle, threshold, objective_mode="classification",
                                      threshold_objective="avg_profit", trade_score_name="probability",
                                      upside_target=0.05, downside_stop=0.02, fee=0.0, slippage=0.0,
                                      ev_safety_margin=0.0, min_predicted_net_return=0.0,
                                      hybrid_min_score=0.0, batch_size=1000000,
                                      classification_ev_from_trade_score=False,
                                      hybrid_score_mode="basic",
                                      hybrid_uncertainty_penalty=0.0,
                                      effective_hybrid_thresholds=None,
                                      ev_context=None,
                                      hybrid_context=None):
    if np is None:
        raise ValueError("compact signal selection requires numpy")

    threshold_array = None
    scalar_threshold = None
    if isinstance(threshold, list) or isinstance(threshold, tuple) or isinstance(threshold, np.ndarray):
        threshold_array = np.asarray(threshold, dtype=np.float32)
    else:
        scalar_threshold = float(threshold)

    calibrated_values = maybe_float32_array(bundle.get("calibrated_probability"))
    if calibrated_values is None:
        calibrated_values = maybe_float32_array(bundle.get("probability"))
    predicted_returns = maybe_float32_array(bundle.get("predicted_trade_return"))
    if predicted_returns is None:
        predicted_returns = maybe_float32_array(bundle.get("raw_predicted_trade_return"))
    predicted_uncertainty = maybe_float32_array(bundle.get("predicted_return_uncertainty"))
    minimum_predicted = max(float(threshold if scalar_threshold is not None else 0.0), float(min_predicted_net_return))
    minimum_hybrid = max(float(threshold if scalar_threshold is not None else 0.0), float(hybrid_min_score))
    hybrid_probability_gate = 0.0
    hybrid_requires_predicted_return = True
    if hybrid_context is not None and hybrid_context.get("hybrid_return_combination", "probability_times_return") == "expected_return":
        hybrid_probability_gate = max(0.0, float(hybrid_context.get("hybrid_min_probability", 0.0)))
    if hybrid_context is not None and hybrid_context.get("hybrid_return_combination", "probability_times_return") == "conditional_payoff":
        hybrid_requires_predicted_return = False
    chunks = []

    for start in range(0, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        if objective_mode == "classification":
            if calibrated_values is None:
                continue
            probabilities = calibrated_values[start:end]
            if threshold_array is not None:
                mask = probabilities >= threshold_array[start:end]
            else:
                mask = probabilities >= scalar_threshold
            if not np.any(mask):
                continue
            if threshold_objective == "ev" or (classification_ev_from_trade_score and trade_score_name == "ev"):
                ev_values = score_batch_ev(
                    probabilities,
                    upside_target,
                    downside_stop,
                    fee,
                    slippage,
                    predicted_trade_returns=predicted_returns[start:end] if predicted_returns is not None else None,
                    objective_mode=objective_mode,
                    ev_context=ev_context,
                )
                mask &= ev_values > ev_safety_margin
        elif objective_mode == "return_regression":
            if predicted_returns is None:
                continue
            mask = (predicted_returns[start:end] - fee - slippage) >= minimum_predicted
        else:
            if calibrated_values is None or (predicted_returns is None and hybrid_requires_predicted_return):
                continue
            mask = score_batch_hybrid(
                calibrated_values[start:end],
                predicted_returns[start:end] if predicted_returns is not None else None,
                fee,
                slippage,
                predicted_uncertainty[start:end] if predicted_uncertainty is not None else None,
                hybrid_score_mode,
                hybrid_uncertainty_penalty,
                hybrid_context=hybrid_context,
                uncertainty_context=bundle.get("uncertainty_context"),
                hybrid_runtime_args=hybrid_runtime_args,
            ) >= (
                effective_hybrid_thresholds[start:end]
                if effective_hybrid_thresholds is not None
                else minimum_hybrid
            )
            if hybrid_probability_gate > 0.0:
                mask &= calibrated_values[start:end] >= hybrid_probability_gate
        selected = np.nonzero(mask)[0]
        if selected.size:
            chunks.append((selected + start).astype(np.int32, copy=False))
    if not chunks:
        return np.asarray([], dtype=np.int32)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks).astype(np.int32, copy=False)


def threshold_for_mode(args):
    if args.objective_mode == "return_regression":
        return args.min_predicted_net_return
    if args.objective_mode == "hybrid":
        return args.hybrid_min_score
    return None


def build_prediction_bundle(probability=None, calibrated_probability=None, predicted_trade_return=None,
                            raw_predicted_trade_return=None, predicted_return_uncertainty=None,
                            meta_probability=None, ev_context=None, hybrid_return_context=None,
                            uncertainty_context=None):
    return {
        "probability": probability,
        "calibrated_probability": calibrated_probability,
        "predicted_trade_return": predicted_trade_return,
        "raw_predicted_trade_return": raw_predicted_trade_return,
        "predicted_return_uncertainty": predicted_return_uncertainty,
        "meta_probability": meta_probability,
        "ev_context": ev_context,
        "hybrid_return_context": hybrid_return_context,
        "uncertainty_context": uncertainty_context,
    }


def bundle_length(bundle):
    for key in ("probability", "calibrated_probability", "predicted_trade_return",
                "raw_predicted_trade_return", "predicted_return_uncertainty", "meta_probability"):
        values = bundle.get(key)
        if values is not None:
            return len(values)
    return 0


def training_manifest_path(path):
    if os.path.isdir(path):
        return os.path.join(os.path.abspath(path), SHARDED_DATASET_MANIFEST)
    root, _ = os.path.splitext(os.path.abspath(path))
    return root + ".meta.json"


def canonical_training_csv(path):
    return os.path.basename(os.path.abspath(path)) == CANONICAL_TRAINING_CSV


def aggregate_cache_manifest_paths(cache_dir):
    if not cache_dir or not os.path.isdir(cache_dir):
        return []
    return sorted(
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.endswith(".aggregate.manifest.json")
    )


def discover_recoverable_default_dataset(requested_path, cache_dir=None):
    if not canonical_training_csv(requested_path):
        return None
    requested_abs = os.path.abspath(requested_path)
    if os.path.exists(training_manifest_path(requested_abs)):
        return None

    search_roots = []
    for root in (os.path.dirname(requested_abs) or os.getcwd(), os.getcwd()):
        absolute_root = os.path.abspath(root)
        if absolute_root not in search_roots:
            search_roots.append(absolute_root)

    cache_dirs = []
    if cache_dir:
        absolute_cache_dir = os.path.abspath(cache_dir)
        if os.path.isdir(absolute_cache_dir):
            cache_dirs.append(absolute_cache_dir)
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.startswith(".gbdt_cache"):
                continue
            candidate_dir = os.path.join(root, name)
            if os.path.isdir(candidate_dir) and candidate_dir not in cache_dirs:
                cache_dirs.append(candidate_dir)

    dataset_candidates = {}
    for candidate_cache_dir in cache_dirs:
        for manifest_path in aggregate_cache_manifest_paths(candidate_cache_dir):
            try:
                manifest = load_json_file(manifest_path, "aggregate cache manifest")
            except ValueError:
                continue
            dataset_path = manifest.get("dataset_path")
            dataset_manifest_path = manifest.get("dataset_manifest_path")
            if not dataset_path or not dataset_manifest_path:
                continue
            dataset_path = os.path.abspath(dataset_path)
            dataset_manifest_path = os.path.abspath(dataset_manifest_path)
            if not os.path.isdir(dataset_path) or not os.path.exists(dataset_manifest_path):
                continue
            candidate = dataset_candidates.setdefault(dataset_path, {
                "input_path": dataset_path,
                "manifest_path": dataset_manifest_path,
                "cache_dirs": [],
                "source": "cache_manifest",
            })
            if candidate_cache_dir not in candidate["cache_dirs"]:
                candidate["cache_dirs"].append(candidate_cache_dir)

    if len(dataset_candidates) == 1:
        only_candidate = list(dataset_candidates.values())[0]
        return {
            "input_path": only_candidate["input_path"],
            "manifest_path": only_candidate["manifest_path"],
            "cache_dir": only_candidate["cache_dirs"][0] if only_candidate["cache_dirs"] else "",
            "source": "cache_manifest",
        }

    local_candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            candidate_dir = os.path.join(root, name)
            if not os.path.isdir(candidate_dir):
                continue
            manifest_path = os.path.join(candidate_dir, SHARDED_DATASET_MANIFEST)
            if not os.path.exists(manifest_path):
                continue
            local_candidates.append({
                "input_path": os.path.abspath(candidate_dir),
                "manifest_path": os.path.abspath(manifest_path),
                "cache_dir": "",
                "source": "local_shard_dataset",
            })

    deduped_local = []
    seen_local_paths = set()
    for candidate in local_candidates:
        if candidate["input_path"] in seen_local_paths:
            continue
        seen_local_paths.add(candidate["input_path"])
        deduped_local.append(candidate)
    if len(deduped_local) == 1:
        return deduped_local[0]
    return None


def has_market_breadth_columns(feature_columns):
    return all(name in feature_columns for name in MARKET_BREADTH_FEATURE_COLUMNS)


def market_breadth_required_columns_present(feature_columns):
    required = (
        "ret_5m",
        "ret_15m",
        "ret_60m",
        "rolling_quote_volume_zscore_60m",
    )
    return all(name in feature_columns for name in required)


def market_breadth_sidecar_paths(path, cache_dir, dtype, min_symbols):
    resolved_cache_dir = resolve_cache_dir(path, cache_dir)
    source_path = os.path.abspath(path)
    source_hash = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]
    prefix = os.path.join(
        resolved_cache_dir,
        "breadth-{}-{}-min{}".format(source_hash, np.dtype(dtype).name, int(min_symbols)),
    )
    return {
        "features": prefix + ".features.dat",
        "manifest": prefix + ".manifest.json",
    }


def base_feature_signature(feature_columns):
    payload = {
        "feature_columns": list(feature_columns),
    }
    return manifest_signature(payload)


def market_breadth_sidecar_manifest_matches(manifest, path, dtype, feature_columns,
                                            row_count, min_symbols, paths,
                                            base_cache_manifest_signature):
    return (
        manifest.get("version") == MARKET_BREADTH_AUGMENTATION_VERSION
        and manifest.get("dataset_path") == os.path.abspath(path)
        and manifest.get("feature_dtype") == np.dtype(dtype).name
        and manifest.get("row_count") == int(row_count)
        and manifest.get("market_breadth_min_symbols") == int(min_symbols)
        and manifest.get("base_feature_signature") == base_feature_signature(feature_columns)
        and manifest.get("base_cache_manifest_signature") == base_cache_manifest_signature
        and manifest.get("feature_columns") == MARKET_BREADTH_FEATURE_COLUMNS
        and os.path.exists(paths["features"])
    )


def inspect_market_breadth_sidecar(path, rows, feature_columns, cache_dir, min_symbols):
    if np is None or not is_compact_rows(rows):
        return None
    dtype = rows.table.features.dtype
    paths = market_breadth_sidecar_paths(path, cache_dir, dtype, min_symbols)
    manifest, manifest_load_seconds = load_cache_manifest_file(paths["manifest"])
    print("Breadth sidecar manifest load time: {:.3f}s".format(manifest_load_seconds), flush=True)
    base_manifest = CACHE_LOAD_INFO.get("manifest", {})
    base_cache_manifest_signature = manifest_signature(base_manifest) if isinstance(base_manifest, dict) and base_manifest else ""
    if not manifest or not market_breadth_sidecar_manifest_matches(
            manifest,
            path,
            dtype,
            feature_columns,
            len(rows.table.labels),
            min_symbols,
            paths,
            base_cache_manifest_signature):
        return None
    memmap_started = time.time()
    extra_features = np.memmap(
        paths["features"],
        dtype=dtype,
        mode="r",
        shape=(int(manifest["row_count"]), len(MARKET_BREADTH_FEATURE_COLUMNS)),
    )
    memmap_attach_seconds = time.time() - memmap_started
    print("Breadth sidecar memmap attach time: {:.3f}s".format(memmap_attach_seconds), flush=True)
    return {
        "features": extra_features,
        "manifest": manifest,
        "paths": paths,
        "manifest_load_seconds": manifest_load_seconds,
        "memmap_attach_seconds": memmap_attach_seconds,
    }


def median_small_group(values):
    if len(values) == 0:
        return 0.0
    ordered = np.sort(values.astype(np.float32, copy=False))
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) * 0.5)


def build_market_breadth_sidecar(path, rows, feature_columns, cache_dir, min_symbols):
    if np is None or not is_compact_rows(rows):
        raise ValueError("market breadth augmentation requires numpy-backed compact rows")
    if not market_breadth_required_columns_present(feature_columns):
        raise ValueError("market breadth augmentation requires ret_5m, ret_15m, ret_60m, and rolling_quote_volume_zscore_60m")
    dtype = rows.table.features.dtype
    paths = market_breadth_sidecar_paths(path, cache_dir, dtype, min_symbols)
    os.makedirs(os.path.dirname(paths["features"]), exist_ok=True)
    for sidecar_path in paths.values():
        try:
            os.remove(sidecar_path)
        except OSError:
            pass

    table = rows.table
    row_count = len(table.labels)
    feature_lookup = table.feature_lookup
    ret5 = np.asarray(table.feature_column(feature_lookup["ret_5m"]), dtype=np.float32)
    ret15 = np.asarray(table.feature_column(feature_lookup["ret_15m"]), dtype=np.float32)
    ret60 = np.asarray(table.feature_column(feature_lookup["ret_60m"]), dtype=np.float32)
    quote_z = np.asarray(table.feature_column(feature_lookup["rolling_quote_volume_zscore_60m"]), dtype=np.float32)
    month_indices = np.asarray(table.month_indices, dtype=np.int32)
    open_times = np.asarray(table.open_times, dtype=np.int64)
    breadth_values = np.memmap(
        paths["features"],
        dtype=dtype,
        mode="w+",
        shape=(row_count, len(MARKET_BREADTH_FEATURE_COLUMNS)),
    )
    breadth_values[:] = 0.0

    token = start_profile_stage("market_breadth_sidecar_build", os.path.abspath(path))
    unique_months = np.unique(month_indices)
    for month_value in unique_months:
        month_positions = np.nonzero(month_indices == month_value)[0]
        if month_positions.size == 0:
            continue
        month_order = np.argsort(open_times[month_positions], kind="mergesort")
        ordered_positions = month_positions[month_order]
        ordered_times = open_times[ordered_positions]
        unique_times, group_starts, group_counts = np.unique(
            ordered_times,
            return_index=True,
            return_counts=True,
        )
        del unique_times
        ordered_ret5 = ret5[ordered_positions]
        ordered_ret15 = ret15[ordered_positions]
        ordered_ret60 = ret60[ordered_positions]
        ordered_quote_z = quote_z[ordered_positions]
        count_values = group_counts.astype(np.float32, copy=False)
        sum_ret5 = np.add.reduceat(ordered_ret5, group_starts).astype(np.float32, copy=False)
        sum_ret15 = np.add.reduceat(ordered_ret15, group_starts).astype(np.float32, copy=False)
        sum_ret60 = np.add.reduceat(ordered_ret60, group_starts).astype(np.float32, copy=False)
        sum_quote_z = np.add.reduceat(ordered_quote_z, group_starts).astype(np.float32, copy=False)
        up5 = np.add.reduceat((ordered_ret5 > 0.0).astype(np.float32), group_starts).astype(np.float32, copy=False)
        up15 = np.add.reduceat((ordered_ret15 > 0.0).astype(np.float32), group_starts).astype(np.float32, copy=False)
        up60 = np.add.reduceat((ordered_ret60 > 0.0).astype(np.float32), group_starts).astype(np.float32, copy=False)

        average_ret5 = sum_ret5 / count_values
        average_ret15 = sum_ret15 / count_values
        average_ret60 = sum_ret60 / count_values
        average_quote_z = sum_quote_z / count_values
        breadth_up5 = up5 / count_values
        breadth_up15 = up15 / count_values
        breadth_up60 = up60 / count_values
        median_ret15 = np.zeros(len(group_starts), dtype=np.float32)
        sufficient = group_counts >= int(min_symbols)
        for group_index, start in enumerate(group_starts):
            if not sufficient[group_index]:
                continue
            count = int(group_counts[group_index])
            median_ret15[group_index] = np.float32(median_small_group(ordered_ret15[start:start + count]))

        average_ret5[~sufficient] = 0.0
        average_ret15[~sufficient] = 0.0
        average_ret60[~sufficient] = 0.0
        average_quote_z[~sufficient] = 0.0
        breadth_up5[~sufficient] = 0.0
        breadth_up15[~sufficient] = 0.0
        breadth_up60[~sufficient] = 0.0
        missing_values = (~sufficient).astype(np.float32)

        repeated_avg5 = np.repeat(average_ret5, group_counts)
        repeated_avg15 = np.repeat(average_ret15, group_counts)
        repeated_avg60 = np.repeat(average_ret60, group_counts)
        repeated_quote_z = np.repeat(average_quote_z, group_counts)
        repeated_breadth5 = np.repeat(breadth_up5, group_counts)
        repeated_breadth15 = np.repeat(breadth_up15, group_counts)
        repeated_breadth60 = np.repeat(breadth_up60, group_counts)
        repeated_median15 = np.repeat(median_ret15, group_counts)
        repeated_missing = np.repeat(missing_values, group_counts)

        breadth_values[ordered_positions, 0] = repeated_breadth5.astype(dtype, copy=False)
        breadth_values[ordered_positions, 1] = repeated_breadth15.astype(dtype, copy=False)
        breadth_values[ordered_positions, 2] = repeated_breadth60.astype(dtype, copy=False)
        breadth_values[ordered_positions, 3] = repeated_avg5.astype(dtype, copy=False)
        breadth_values[ordered_positions, 4] = repeated_avg15.astype(dtype, copy=False)
        breadth_values[ordered_positions, 5] = repeated_avg60.astype(dtype, copy=False)
        breadth_values[ordered_positions, 6] = repeated_median15.astype(dtype, copy=False)
        breadth_values[ordered_positions, 7] = repeated_quote_z.astype(dtype, copy=False)
        breadth_values[ordered_positions, 8] = (ordered_ret5 - repeated_avg5).astype(dtype, copy=False)
        breadth_values[ordered_positions, 9] = (ordered_ret15 - repeated_avg15).astype(dtype, copy=False)
        breadth_values[ordered_positions, 10] = (ordered_ret60 - repeated_avg60).astype(dtype, copy=False)
        breadth_values[ordered_positions, 11] = repeated_missing.astype(dtype, copy=False)

    breadth_values.flush()
    base_manifest = CACHE_LOAD_INFO.get("manifest", {})
    manifest = {
        "version": MARKET_BREADTH_AUGMENTATION_VERSION,
        "dataset_path": os.path.abspath(path),
        "feature_dtype": np.dtype(dtype).name,
        "row_count": int(row_count),
        "market_breadth_min_symbols": int(min_symbols),
        "base_feature_signature": base_feature_signature(feature_columns),
        "base_cache_manifest_signature": manifest_signature(base_manifest) if isinstance(base_manifest, dict) and base_manifest else "",
        "feature_columns": MARKET_BREADTH_FEATURE_COLUMNS,
    }
    with open(paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    finish_profile_stage(token, rows_processed=row_count, extra_info=paths["manifest"])
    log_memory("Built market breadth sidecar cache")
    return {
        "features": breadth_values,
        "manifest": manifest,
        "paths": paths,
    }


def attach_market_breadth_sidecar(rows, feature_columns, sidecar):
    if not is_compact_rows(rows):
        return rows, feature_columns
    if has_market_breadth_columns(feature_columns):
        return rows, feature_columns
    table = rows.table
    combined_lookup = dict(table.feature_lookup)
    base_count = table.base_feature_count
    for offset, name in enumerate(MARKET_BREADTH_FEATURE_COLUMNS):
        combined_lookup[name] = base_count + offset
    table.extra_features = sidecar["features"]
    table.extra_memmap_path = sidecar["paths"]["features"]
    table.remove_extra_memmap_on_cleanup = False
    table.feature_lookup = combined_lookup
    augmented_columns = list(feature_columns) + [name for name in MARKET_BREADTH_FEATURE_COLUMNS if name not in feature_columns]
    return rows, augmented_columns


def maybe_augment_market_breadth_rows(rows, feature_columns, args, dataset_path, cache_dir):
    if not getattr(args, "augment_market_breadth_features", False):
        return rows, feature_columns
    if has_market_breadth_columns(feature_columns):
        args.market_breadth_features = True
        return rows, feature_columns
    if np is None or not is_compact_rows(rows):
        print(
            "Warning: --augment-market-breadth-features requires numpy-backed compact rows; continuing without synthetic breadth.",
            file=sys.stderr,
            flush=True,
        )
        return rows, feature_columns
    if not market_breadth_required_columns_present(feature_columns):
        print(
            "Warning: synthetic market breadth requires ret_5m, ret_15m, ret_60m, and rolling_quote_volume_zscore_60m; continuing without breadth augmentation.",
            file=sys.stderr,
            flush=True,
        )
        return rows, feature_columns
    min_symbols = max(1, int(getattr(args, "market_breadth_min_symbols", 5)))
    sidecar = inspect_market_breadth_sidecar(dataset_path, rows, feature_columns, cache_dir, min_symbols)
    if sidecar is None:
        if getattr(args, "cache_only", False):
            print(
                "Cache-only base load succeeded; building market breadth sidecar once on top of the existing cache.",
                file=sys.stderr,
                flush=True,
            )
        sidecar = build_market_breadth_sidecar(dataset_path, rows, feature_columns, cache_dir, min_symbols)
    else:
        log_memory("Loaded compatible market breadth sidecar cache")
    args.market_breadth_features = True
    return attach_market_breadth_sidecar(rows, feature_columns, sidecar)


def manifest_signature_subset(manifest):
    if not manifest:
        return {}
    keys = [
        "version",
        "feature_count",
        "feature_names",
        "label_mode",
        "target_exit_mode",
        "prediction_window_minutes",
        "growth_threshold",
        "upside_target",
        "downside_stop",
        "tie_policy",
        "fee",
        "slippage",
        "min_net_return",
        "split_mode",
        "train_ratio",
        "validation_ratio",
        "test_ratio",
        "training_months",
        "validation_months",
        "test_months",
        "market_regime_features",
        "market_breadth_features",
    ]
    return {key: manifest.get(key) for key in keys}


def manifest_compatibility_signature(manifest):
    return manifest_signature(manifest_signature_subset(manifest))


def shard_manifest_path(path):
    root, _ = os.path.splitext(os.path.abspath(path))
    return root + ".meta.json"


def load_training_manifest(path):
    token = start_profile_stage("manifest_load", os.path.abspath(path))
    manifest_path = training_manifest_path(path)
    if not os.path.exists(manifest_path):
        if os.path.isdir(path):
            raise ValueError(
                "{} is missing {}. Generate shard output with the current C++ generator before running gbdt_pipeline.py.".format(
                    path,
                    os.path.basename(manifest_path),
                )
            )
        if canonical_training_csv(path):
            raise ValueError(
                "{} is missing {}. This usually means the training CSV was generated by an older build. "
                "Rebuild the dataset with the current C++ generator before running gbdt_pipeline.py.".format(
                    path,
                    os.path.basename(manifest_path),
                )
            )
        return None, manifest_path

    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("unable to read training manifest {}: {}".format(manifest_path, error))

    if os.path.isdir(path):
        if manifest.get("version") != SHARDED_DATASET_MANIFEST_VERSION or manifest.get("kind") != "symbol_month_shards":
            raise ValueError(
                "{} has unsupported sharded dataset manifest version/kind. Regenerate the shard dataset with the current generator.".format(
                    manifest_path,
                )
            )
        finish_profile_stage(token, extra_info=manifest_path)
        return manifest, manifest_path

    if manifest.get("version") != TRAINING_MANIFEST_VERSION:
        raise ValueError(
            "{} has unsupported version {} (expected {}). Rebuild the dataset with the current generator.".format(
                manifest_path,
                manifest.get("version"),
                TRAINING_MANIFEST_VERSION,
            )
        )
    finish_profile_stage(token, extra_info=manifest_path)
    return manifest, manifest_path


def manifest_positive_float(manifest, key):
    if not manifest or key not in manifest:
        return None, "missing"
    value = safe_float(manifest.get(key), 0.0)
    if value <= 0.0:
        return None, "invalid"
    return value, "valid"


def apply_manifest_ev_targets(args, manifest, explicit_flags):
    def apply_one(cli_name, manifest_key):
        source_attr = "ev_{}_source".format(cli_name)
        manifest_attr = "manifest_{}".format(cli_name)
        effective_attr = "effective_{}".format(cli_name)
        current_value = float(getattr(args, cli_name))
        manifest_value, status = manifest_positive_float(manifest, manifest_key)
        setattr(args, manifest_attr, manifest_value if manifest_value is not None else 0.0)

        if cli_name in explicit_flags:
            setattr(args, source_attr, "cli")
            setattr(args, effective_attr, current_value)
            print("Using CLI {} override: {:.12g}".format(cli_name, current_value), flush=True)
            return

        if status == "valid":
            setattr(args, cli_name, manifest_value)
            setattr(args, source_attr, "manifest")
            setattr(args, effective_attr, manifest_value)
            print(
                "Using EV {} from training manifest: {:.12g}".format(cli_name, manifest_value),
                flush=True,
            )
            return

        if status == "invalid":
            print(
                "Warning: invalid {} in training manifest; keeping default {:.12g}".format(
                    manifest_key,
                    current_value,
                ),
                file=sys.stderr,
                flush=True,
            )
            setattr(args, source_attr, "invalid_manifest_fallback")
        else:
            setattr(args, source_attr, "missing_manifest_fallback")
        setattr(args, effective_attr, current_value)

    apply_one("upside_target", "upside_target")
    apply_one("downside_stop", "downside_stop")
    args.market_regime_features = bool(manifest.get("market_regime_features", False)) if manifest else False
    args.market_breadth_features = bool(manifest.get("market_breadth_features", False)) if manifest else False
    print("Training manifest market_regime_features: {}".format(str(args.market_regime_features).lower()), flush=True)
    print("Training manifest market_breadth_features: {}".format(str(args.market_breadth_features).lower()), flush=True)


def load_json_file(path, description):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("unable to read {} {}: {}".format(description, path, error))


def discover_sharded_dataset_shards(path, dataset_manifest):
    shard_root = os.path.join(os.path.abspath(path), "shards")
    if not os.path.isdir(shard_root):
        raise ValueError("{} does not contain a shards/ directory".format(path))
    dataset_signature = manifest_compatibility_signature(dataset_manifest)
    shards = []
    for root, _, files in os.walk(shard_root):
        for name in files:
            if not name.endswith(".csv"):
                continue
            csv_path = os.path.join(root, name)
            meta_path = shard_manifest_path(csv_path)
            if not os.path.exists(meta_path):
                raise ValueError("shard {} is missing {}".format(csv_path, os.path.basename(meta_path)))
            shard_manifest = load_json_file(meta_path, "shard manifest")
            if shard_manifest.get("version") != SHARD_MANIFEST_VERSION or shard_manifest.get("kind") != "symbol_month_shard":
                raise ValueError("shard manifest {} has unsupported version/kind".format(meta_path))
            shard_signature = manifest_compatibility_signature(shard_manifest)
            if shard_signature != dataset_signature:
                raise ValueError(
                    "shard {} is incompatible with {}. Its feature/label configuration does not match the dataset manifest. "
                    "Use a separate shard directory for different settings or regenerate the mismatched shards.".format(
                        csv_path,
                        os.path.basename(training_manifest_path(path)),
                    )
                )
            csv_stat = os.stat(csv_path)
            meta_stat = os.stat(meta_path)
            shards.append({
                "csv_path": os.path.abspath(csv_path),
                "meta_path": os.path.abspath(meta_path),
                "symbol": str(shard_manifest.get("symbol", "")),
                "month": str(shard_manifest.get("month", "")),
                "row_count": int(shard_manifest.get("row_count", 0)),
                "manifest": shard_manifest,
                "manifest_signature": shard_signature,
                "csv_mtime_ns": getattr(csv_stat, "st_mtime_ns", int(csv_stat.st_mtime * 1000000000)),
                "csv_size": int(csv_stat.st_size),
                "meta_mtime_ns": getattr(meta_stat, "st_mtime_ns", int(meta_stat.st_mtime * 1000000000)),
                "meta_size": int(meta_stat.st_size),
            })
    shards.sort(key=lambda item: (item["symbol"], item["month"], item["csv_path"]))
    if not shards:
        raise ValueError("{} does not contain any shard CSV files under shards/".format(path))
    return shards


def sharded_inventory_signature(shards):
    payload = [
        {
            "csv_path": item["csv_path"],
            "csv_mtime_ns": item["csv_mtime_ns"],
            "csv_size": item["csv_size"],
            "meta_path": item["meta_path"],
            "meta_mtime_ns": item["meta_mtime_ns"],
            "meta_size": item["meta_size"],
        }
        for item in shards
    ]
    return manifest_signature(payload)


def cached_text(cache, value):
    if value in cache:
        return cache[value]
    cache[value] = value
    return value


def build_features(item, feature_columns, storage):
    if storage == "float32":
        return array("f", (safe_float(item.get(name), 0.0) for name in feature_columns))
    if storage == "float64":
        return array("d", (safe_float(item.get(name), 0.0) for name in feature_columns))
    return [safe_float(item.get(name), 0.0) for name in feature_columns]


def adaptive_thresholds(probabilities, base_thresholds, min_validation_trades, max_sample_rows=1000000):
    thresholds = set(value for value in base_thresholds if 0.0 <= value <= 1.0)
    if np is not None and isinstance(probabilities, np.ndarray):
        values = probabilities
        if max_sample_rows and len(values) > max_sample_rows:
            sample_positions = np.linspace(0, len(values) - 1, num=max_sample_rows, dtype=np.int64)
            values = values[sample_positions]
        valid = values[(values >= 0.0) & (values <= 1.0)]
        if valid.size == 0:
            return sorted(thresholds)
        ordered = np.sort(valid)
        ordered_count = int(ordered.size)
        max_probability = float(ordered[-1])
        quantiles = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
        for quantile in quantiles:
            index = int((ordered_count - 1) * quantile)
            thresholds.add(float(ordered[index]))
        target_counts = [max(1, min_validation_trades), max(1, min_validation_trades * 2), 10, 25, 50, 100, 250, 500, 1000]
        for count in target_counts:
            if count <= ordered_count:
                thresholds.add(float(ordered[-count]))
        thresholds.add(max(0.0, max_probability - 1e-12))
        return sorted(thresholds)

    values = probabilities
    if max_sample_rows and len(values) > max_sample_rows:
        values = [values[int(index * (len(values) - 1) / float(max_sample_rows - 1))] for index in range(max_sample_rows)]
    ordered = sorted(value for value in values if 0.0 <= value <= 1.0)
    if not ordered:
        return sorted(thresholds)

    quantiles = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
    for quantile in quantiles:
        index = int((len(ordered) - 1) * quantile)
        thresholds.add(ordered[index])

    # Include thresholds that roughly target small, medium, and large numbers of
    # raw validation signals. Portfolio entry limits are applied during evaluation.
    target_counts = [max(1, min_validation_trades), max(1, min_validation_trades * 2), 10, 25, 50, 100, 250, 500, 1000]
    for count in target_counts:
        if count <= len(ordered):
            thresholds.add(ordered[-count])

    thresholds.add(max(0.0, ordered[-1] - 1e-12))
    return sorted(thresholds)


def adaptive_score_thresholds(scores, base_threshold, max_sample_rows=1000000):
    base = float(base_threshold)
    def finalize_thresholds(values):
        filtered = sorted(
            float(round(value, 6))
            for value in values
            if math.isfinite(float(value)) and float(value) >= base
        )
        if not filtered:
            return [base]
        deduped = []
        for value in filtered:
            if deduped and abs(deduped[-1] - value) <= 1e-12:
                continue
            deduped.append(value)
        return deduped

    if scores is None:
        return [base]
    if np is not None and isinstance(scores, np.ndarray):
        values = scores
        if max_sample_rows and len(values) > max_sample_rows:
            sample_positions = np.linspace(0, len(values) - 1, num=max_sample_rows, dtype=np.int64)
            values = values[sample_positions]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return [base]
        ordered = np.sort(finite.astype(np.float32, copy=False))
        quantiles = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
        thresholds = {base, float(ordered[-1])}
        ordered_count = int(ordered.size)
        for quantile in quantiles:
            index = int((ordered_count - 1) * quantile)
            thresholds.add(float(ordered[index]))
        return finalize_thresholds(thresholds)
    values = list(scores)
    if max_sample_rows and len(values) > max_sample_rows:
        values = [values[int(index * (len(values) - 1) / float(max_sample_rows - 1))] for index in range(max_sample_rows)]
    ordered = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not ordered:
        return [base]
    quantiles = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
    thresholds = {base, float(ordered[-1])}
    for quantile in quantiles:
        index = int((len(ordered) - 1) * quantile)
        thresholds.add(float(ordered[index]))
    return finalize_thresholds(thresholds)


def score_values_for_bundle(rows, bundle, args):
    del rows
    if args.objective_mode == "classification":
        return bundle["calibrated_probability"] if bundle.get("calibrated_probability") is not None else bundle.get("probability")
    if args.objective_mode == "return_regression":
        predicted = bundle.get("predicted_trade_return")
        if predicted is None:
            return None
        fee_slippage = args.fee + args.slippage * args.validation_slippage_multiplier
        if np is not None and isinstance(predicted, np.ndarray):
            return predicted.astype(np.float32, copy=False) - np.float32(fee_slippage)
        return [float(value) - fee_slippage for value in predicted]
    probabilities = bundle["calibrated_probability"] if bundle.get("calibrated_probability") is not None else bundle.get("probability")
    predicted = bundle.get("predicted_trade_return")
    if predicted is None:
        predicted = bundle.get("raw_predicted_trade_return")
    upside_target = getattr(args, "upside_target", 0.05)
    downside_stop = getattr(args, "downside_stop", 0.02)
    hybrid_context = resolve_hybrid_return_context(bundle, args, upside_target, downside_stop)
    hybrid_requires_predicted_return = hybrid_context.get("hybrid_return_combination", "probability_times_return") != "conditional_payoff"
    if probabilities is None or (predicted is None and hybrid_requires_predicted_return):
        return None
    uncertainty = bundle.get("predicted_return_uncertainty")
    return score_batch_hybrid(
        maybe_float32_array(probabilities),
        maybe_float32_array(predicted) if predicted is not None else None,
        args.fee,
        args.slippage * args.validation_slippage_multiplier,
        maybe_float32_array(uncertainty) if uncertainty is not None else None,
        args.hybrid_score_mode,
        args.hybrid_uncertainty_penalty,
        hybrid_context=hybrid_context,
        uncertainty_context=bundle.get("uncertainty_context"),
        hybrid_runtime_args=args,
    )


class DataRow(object):
    __slots__ = (
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
        "features",
        "feature_lookup",
    )

    def __init__(self, symbol, month, month_index, open_time, label, forward_return, trade_return,
                 max_future_high_return, max_future_low_return, quote_volume, features, feature_lookup=None):
        self.symbol = symbol
        self.month = month
        self.month_index = month_index
        self.open_time = open_time
        self.label = label
        self.forward_return = forward_return
        self.trade_return = trade_return
        self.max_future_high_return = max_future_high_return
        self.max_future_low_return = max_future_low_return
        self.quote_volume = quote_volume
        self.features = features
        self.feature_lookup = feature_lookup or {}


class CompactTable(object):
    __slots__ = (
        "symbols",
        "months",
        "symbol_codes",
        "month_codes",
        "month_indices",
        "open_times",
        "labels",
        "forward_returns",
        "trade_returns",
        "max_future_high_returns",
        "max_future_low_returns",
        "quote_volumes",
        "features",
        "extra_features",
        "feature_lookup",
        "base_feature_count",
        "memmap_path",
        "remove_memmap_on_cleanup",
        "extra_memmap_path",
        "remove_extra_memmap_on_cleanup",
    )

    def __init__(self, symbols, months, symbol_codes, month_codes, month_indices, open_times,
                 labels, forward_returns, trade_returns, max_future_high_returns,
                 max_future_low_returns, quote_volumes, features, feature_lookup, memmap_path=None,
                 remove_memmap_on_cleanup=True, extra_features=None,
                 extra_memmap_path=None, remove_extra_memmap_on_cleanup=True):
        self.symbols = symbols
        self.months = months
        self.symbol_codes = symbol_codes
        self.month_codes = month_codes
        self.month_indices = month_indices
        self.open_times = open_times
        self.labels = labels
        self.forward_returns = forward_returns
        self.trade_returns = trade_returns
        self.max_future_high_returns = max_future_high_returns
        self.max_future_low_returns = max_future_low_returns
        self.quote_volumes = quote_volumes
        self.features = features
        self.extra_features = extra_features
        self.feature_lookup = feature_lookup
        self.base_feature_count = int(features.shape[1]) if features is not None and hasattr(features, "shape") else len(feature_lookup)
        self.memmap_path = memmap_path
        self.remove_memmap_on_cleanup = remove_memmap_on_cleanup
        self.extra_memmap_path = extra_memmap_path
        self.remove_extra_memmap_on_cleanup = remove_extra_memmap_on_cleanup

    def feature_column(self, feature_index, indices=None):
        if feature_index < self.base_feature_count:
            if indices is None:
                return self.features[:, feature_index]
            return self.features[indices, feature_index]
        if self.extra_features is None:
            raise IndexError("feature index {} is out of range".format(feature_index))
        extra_index = feature_index - self.base_feature_count
        if indices is None:
            return self.extra_features[:, extra_index]
        return self.extra_features[indices, extra_index]

    def feature_value_at(self, position, feature_index):
        if feature_index < self.base_feature_count:
            return float(self.features[position, feature_index])
        if self.extra_features is None:
            raise IndexError("feature index {} is out of range".format(feature_index))
        return float(self.extra_features[position, feature_index - self.base_feature_count])

    def feature_matrix(self, indices=None):
        if indices is None:
            base_values = self.features
            extra_values = self.extra_features
        else:
            base_values = self.features[indices, :]
            extra_values = self.extra_features[indices, :] if self.extra_features is not None else None
        if extra_values is None:
            return base_values
        return np.concatenate((base_values, extra_values), axis=1)

    def cleanup(self):
        if self.memmap_path:
            close_memmap(self.features)
            self.features = None
            if self.remove_memmap_on_cleanup:
                try:
                    os.remove(self.memmap_path)
                except OSError:
                    pass
            self.memmap_path = None
        if self.extra_memmap_path:
            close_memmap(self.extra_features)
            self.extra_features = None
            if self.remove_extra_memmap_on_cleanup:
                try:
                    os.remove(self.extra_memmap_path)
                except OSError:
                    pass
            self.extra_memmap_path = None


class CompactRows(object):
    """A memory-light row view backed by one NumPy feature matrix.

    The large LightGBM path uses this instead of one Python object plus one
    feature array per candle. Subsets store integer row positions only, so split
    logic remains chronological without duplicating metadata.
    """

    __slots__ = ("table", "indices")

    def __init__(self, table, indices=None):
        self.table = table
        self.indices = indices

    def __len__(self):
        if self.indices is None:
            return len(self.table.labels)
        return len(self.indices)

    def positions(self):
        if self.indices is None:
            return range(len(self.table.labels))
        return self.indices

    def subset(self, indices):
        return CompactRows(self.table, indices)

    def feature_matrix(self):
        return self.table.feature_matrix(self.indices)

    def labels_array(self):
        if self.indices is None:
            return self.table.labels
        return self.table.labels[self.indices]

    def cleanup(self):
        self.table.cleanup()


def is_compact_rows(rows):
    return isinstance(rows, CompactRows)


def manifest_signature(manifest):
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def compact_trade_return_arrays(rows):
    if rows.indices is None:
        return rows.table.trade_returns, rows.table.labels
    return rows.table.trade_returns[rows.indices], rows.table.labels[rows.indices]


def validate_rows_against_training_manifest(path, rows, manifest):
    if not manifest:
        return

    label_mode = str(manifest.get("label_mode", "")).strip()
    target_exit_mode = str(manifest.get("target_exit_mode", "fixed_target")).strip()
    tolerance = 1e-6

    if is_compact_rows(rows):
        trade_returns, labels_array = compact_trade_return_arrays(rows)
    else:
        trade_returns = [row.trade_return for row in rows]
        labels_array = [row.label for row in rows]

    if label_mode == "target_stop":
        upside_target = safe_float(manifest.get("upside_target"), 0.0)
        downside_stop = safe_float(manifest.get("downside_stop"), 0.0)
        if upside_target <= 0.0 or downside_stop <= 0.0:
            print(
                "Warning: {} has missing or invalid target_stop parameters in {}; skipping strict target/stop "
                "return validation for compatibility with older manifests.".format(
                    path,
                    os.path.basename(training_manifest_path(path)),
                ),
                file=sys.stderr,
                flush=True,
            )
            return
        if target_exit_mode not in ("fixed_target", "first_decline"):
            raise ValueError(
                "{} has unsupported target_exit_mode={} in {}. Rebuild the dataset with a supported generator.".format(
                    path,
                    target_exit_mode,
                    os.path.basename(training_manifest_path(path)),
                )
            )
        if is_compact_rows(rows):
            min_trade_return = float(np.min(trade_returns)) if len(trade_returns) else 0.0
            max_trade_return = float(np.max(trade_returns)) if len(trade_returns) else 0.0
            positive_returns = trade_returns[labels_array == 1]
            if target_exit_mode == "fixed_target":
                positive_mismatch = bool(len(positive_returns)) and bool(
                    np.any(np.abs(positive_returns - upside_target) > tolerance)
                )
            else:
                positive_mismatch = bool(len(positive_returns)) and bool(
                    np.any(positive_returns < upside_target - tolerance)
                )
        else:
            min_trade_return = min(trade_returns) if trade_returns else 0.0
            max_trade_return = max(trade_returns) if trade_returns else 0.0
            if target_exit_mode == "fixed_target":
                positive_mismatch = any(
                    label == 1 and abs(trade_return - upside_target) > tolerance
                    for trade_return, label in zip(trade_returns, labels_array)
                )
            else:
                positive_mismatch = any(
                    label == 1 and trade_return < upside_target - tolerance
                    for trade_return, label in zip(trade_returns, labels_array)
                )
        if target_exit_mode == "fixed_target":
            max_allowed_trade_return = upside_target + tolerance
        else:
            max_allowed_trade_return = float("inf")
        if min_trade_return < -downside_stop - tolerance or max_trade_return > max_allowed_trade_return or positive_mismatch:
            raise ValueError(
                "{} is incompatible with {}. target_stop datasets using target_exit_mode={} must keep trade_return "
                "above {:.6f}, and positive labels must exit {} the upside target; observed min={:.6f} max={:.6f}. "
                "This CSV was likely generated by an older build or with different label settings. Rebuild it.".format(
                    path,
                    os.path.basename(training_manifest_path(path)),
                    target_exit_mode,
                    -downside_stop,
                    "at" if target_exit_mode == "fixed_target" else "at or above",
                    min_trade_return,
                    max_trade_return,
                )
            )
    elif label_mode == "future_high":
        growth_threshold = safe_float(manifest.get("growth_threshold"), 0.0)
        if growth_threshold <= 0.0:
            raise ValueError(
                "{} has invalid future_high parameters in {}. Rebuild the dataset.".format(
                    path,
                    os.path.basename(training_manifest_path(path)),
                )
            )
        if is_compact_rows(rows):
            positive_returns = trade_returns[labels_array == 1]
            positive_mismatch = bool(len(positive_returns)) and bool(
                np.any(np.abs(positive_returns - growth_threshold) > tolerance)
            )
        else:
            positive_mismatch = any(
                label == 1 and abs(trade_return - growth_threshold) > tolerance
                for trade_return, label in zip(trade_returns, labels_array)
            )
        if positive_mismatch:
            raise ValueError(
                "{} is incompatible with {}. future_high datasets require positive labels to use trade_return={:.6f}. "
                "Rebuild the dataset.".format(
                    path,
                    os.path.basename(training_manifest_path(path)),
                    growth_threshold,
                )
            )


def make_row(item, feature_columns, month_index_lookup, text_cache, feature_storage):
    global NORMALIZED_MICROSECOND_OPEN_TIMES
    symbol = cached_text(text_cache, item.get("symbol", ""))
    month = cached_text(text_cache, item.get("month", ""))
    month_index = int(safe_float(item.get("month_index"), month_index_lookup.get((symbol, month), 0)))
    raw_open_time = int(safe_float(item.get("open_time"), 0.0))
    open_time = normalize_open_time_ms(raw_open_time)
    if open_time != raw_open_time:
        NORMALIZED_MICROSECOND_OPEN_TIMES += 1
    label = 1 if str(item.get("label", "0")).strip() == "1" else 0
    forward_return = safe_float(item.get("forward_return"), 0.0)
    trade_return = safe_float(item.get("trade_return"), forward_return)
    max_future_high_return = safe_float(item.get("max_future_high_return"), forward_return)
    max_future_low_return = safe_float(item.get("max_future_low_return"), forward_return)
    features = build_features(item, feature_columns, feature_storage)
    quote_volume = safe_float(item.get("quote_volume"), 0.0)
    if quote_volume <= 0.0 and "log_quote_volume" in feature_columns:
        quote_volume = max(
            0.0,
            math.expm1(float(features[feature_columns.index("log_quote_volume")]))
            * RECONSTRUCTED_QUOTE_VOLUME_DISCOUNT,
        )
    return DataRow(
        symbol,
        month,
        month_index,
        open_time,
        label,
        forward_return,
        trade_return,
        max_future_high_return,
        max_future_low_return,
        quote_volume,
        features,
        {name: index for index, name in enumerate(feature_columns)},
    )


def csv_value(fields, positions, name, default=""):
    position = positions.get(name)
    if position is None or position >= len(fields):
        return default
    return fields[position]


def count_csv_rows(path):
    with open_csv_text(path) as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def load_compact_rows(path, feature_storage="matrix32", memmap_dir=None, feature_path=None,
                      remove_memmap_on_cleanup=True):
    global NORMALIZED_MICROSECOND_OPEN_TIMES
    if np is None:
        raise ValueError("compact matrix storage requires numpy")

    dtype = np.float32 if feature_storage in ("auto", "matrix32", "memmap32") else np.float64
    use_memmap = feature_storage in ("memmap32", "memmap64")
    count_token = start_profile_stage("csv_row_count", os.path.abspath(path))
    row_count = count_csv_rows(path)
    finish_profile_stage(count_token, rows_processed=row_count, extra_info=os.path.abspath(path))
    parse_token = start_profile_stage("csv_parse", os.path.abspath(path))
    with open_csv_text(path) as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = next(reader)
        except StopIteration:
            raise ValueError("empty CSV: {}".format(path))

        positions = {name: index for index, name in enumerate(fieldnames)}
        feature_columns = [name for name in fieldnames if name not in METADATA_COLUMNS]
        if "label" not in positions:
            raise ValueError("{} must contain a label column".format(path))
        if "month_index" not in positions:
            raise ValueError("compact storage requires a month_index column")
        if not feature_columns:
            raise ValueError("{} does not contain feature columns".format(path))
        if "log_quote_volume" not in feature_columns:
            raise ValueError("{} must contain log_quote_volume for compact portfolio sizing".format(path))

        feature_positions = [positions[name] for name in feature_columns]
        memmap_path = None
        if use_memmap:
            directory = memmap_dir or tempfile.gettempdir()
            os.makedirs(directory, exist_ok=True)
            if feature_path:
                memmap_path = feature_path
            else:
                descriptor, memmap_path = tempfile.mkstemp(
                    prefix="gbdt_features_",
                    suffix=".dat",
                    dir=directory,
                )
                os.close(descriptor)
            features = np.memmap(
                memmap_path,
                dtype=dtype,
                mode="w+",
                shape=(row_count, len(feature_columns)),
            )
        else:
            features = np.empty((row_count, len(feature_columns)), dtype=dtype)
        symbol_codes = np.empty(row_count, dtype=np.int32)
        month_codes = np.empty(row_count, dtype=np.int32)
        month_indices = np.empty(row_count, dtype=np.int16)
        open_times = np.empty(row_count, dtype=np.int64)
        labels_values = np.empty(row_count, dtype=np.int8)
        forward_returns = np.empty(row_count, dtype=np.float32)
        trade_returns = np.empty(row_count, dtype=np.float32)
        max_future_high_returns = np.empty(row_count, dtype=np.float32)
        max_future_low_returns = np.empty(row_count, dtype=np.float32)
        quote_volumes = np.empty(row_count, dtype=np.float32) if "quote_volume" in positions else None
        symbol_lookup = {}
        month_lookup = {}
        symbols = []
        months = []

        for row_index, fields in enumerate(reader):
            symbol = csv_value(fields, positions, "symbol")
            month = csv_value(fields, positions, "month")
            symbol_code = symbol_lookup.get(symbol)
            if symbol_code is None:
                symbol_code = len(symbols)
                symbol_lookup[symbol] = symbol_code
                symbols.append(symbol)
            month_code = month_lookup.get(month)
            if month_code is None:
                month_code = len(months)
                month_lookup[month] = month_code
                months.append(month)

            symbol_codes[row_index] = symbol_code
            month_codes[row_index] = month_code
            month_indices[row_index] = int(safe_float(csv_value(fields, positions, "month_index"), 0.0))
            raw_open_time = int(safe_float(csv_value(fields, positions, "open_time"), 0.0))
            normalized_open_time = normalize_open_time_ms(raw_open_time)
            if normalized_open_time != raw_open_time:
                NORMALIZED_MICROSECOND_OPEN_TIMES += 1
            open_times[row_index] = normalized_open_time
            labels_values[row_index] = 1 if str(csv_value(fields, positions, "label", "0")).strip() == "1" else 0
            forward_return = safe_float(csv_value(fields, positions, "forward_return"), 0.0)
            trade_return = safe_float(csv_value(fields, positions, "trade_return"), forward_return)
            forward_returns[row_index] = forward_return
            trade_returns[row_index] = trade_return
            max_future_high_returns[row_index] = safe_float(
                csv_value(fields, positions, "max_future_high_return"),
                forward_return,
            )
            max_future_low_returns[row_index] = safe_float(
                csv_value(fields, positions, "max_future_low_return"),
                forward_return,
            )
            if quote_volumes is not None:
                quote_volumes[row_index] = safe_float(csv_value(fields, positions, "quote_volume"), 0.0)
            for feature_index, field_index in enumerate(feature_positions):
                features[row_index, feature_index] = safe_float(
                    fields[field_index] if field_index < len(fields) else "",
                    0.0,
                )
            if row_index and row_index % 1000000 == 0:
                log_memory("CSV parse progress: {:,}/{:,} rows".format(row_index, row_count))
        if use_memmap:
            features.flush()
    finish_profile_stage(parse_token, rows_processed=row_count, extra_info=os.path.abspath(path))

    table = CompactTable(
        symbols,
        months,
        symbol_codes,
        month_codes,
        month_indices,
        open_times,
        labels_values,
        forward_returns,
        trade_returns,
        max_future_high_returns,
        max_future_low_returns,
        quote_volumes,
        features,
        {name: index for index, name in enumerate(feature_columns)},
        memmap_path,
        remove_memmap_on_cleanup,
    )
    return CompactRows(table), feature_columns, "forward_return" in positions


def load_object_rows(path, feature_storage="float32"):
    with open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("empty CSV: {}".format(path))

        feature_columns = [
            name for name in reader.fieldnames
            if name not in METADATA_COLUMNS
        ]
        if "label" not in reader.fieldnames:
            raise ValueError("{} must contain a label column".format(path))
        if not feature_columns:
            raise ValueError("{} does not contain feature columns".format(path))
        if "quote_volume" not in reader.fieldnames and "log_quote_volume" not in feature_columns:
            raise ValueError("{} must contain quote_volume or log_quote_volume".format(path))

        raw_rows = []
        symbol_months = {}
        has_month_index = "month_index" in reader.fieldnames
        has_returns = "forward_return" in reader.fieldnames
        text_cache = {}

        for item in reader:
            if has_month_index:
                raw_rows.append(make_row(item, feature_columns, {}, text_cache, feature_storage))
            else:
                symbol = item.get("symbol", "")
                month = item.get("month", "")
                if symbol and month:
                    symbol_months.setdefault(symbol, set()).add(month)
                raw_rows.append(item)

    if has_month_index:
        rows = raw_rows
        rows.sort(key=lambda row: (row.month_index, row.open_time, row.symbol))
        return rows, feature_columns, has_returns

    month_index_lookup = {}
    for symbol, months in symbol_months.items():
        for index, month in enumerate(sorted(months)):
            month_index_lookup[(symbol, month)] = index

    rows = []
    for item in raw_rows:
        rows.append(make_row(item, feature_columns, month_index_lookup, text_cache, feature_storage))

    rows.sort(key=lambda row: (row.month_index, row.open_time, row.symbol))
    return rows, feature_columns, has_returns


def cache_paths(path, cache_dir, dtype):
    source_path = os.path.abspath(path)
    source_hash = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = os.path.join(cache_dir, "{}-{}-{}".format(stem, source_hash, np.dtype(dtype).name))
    return {
        "features": prefix + ".features.dat",
        "metadata": prefix + ".metadata.npz",
        "manifest": prefix + ".manifest.json",
    }


def load_cache_manifest_file(path):
    token = start_profile_stage("cache_manifest_load", path)
    started = time.time()
    manifest = None
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            manifest = None
    elapsed = time.time() - started
    finish_profile_stage(token, extra_info=path)
    return manifest, elapsed


def clear_cache_files(paths):
    for cache_path in paths.values():
        try:
            os.remove(cache_path)
        except OSError:
            pass


def cache_only_missing_message(cache_dir):
    return (
        "ERROR: --cache-only was set, but no compatible cache was found in {}.\n"
        "Run once without --cache-only or rebuild the dataset/cache intentionally."
    ).format(cache_dir)


def source_csv_info(path, training_manifest=None, manifest_path=None, manifest_signature_override=None):
    stat = os.stat(path)
    info = {
        "source_csv_path": os.path.abspath(path),
        "source_csv_mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000)),
        "source_csv_size": stat.st_size,
    }
    if training_manifest is not None and manifest_path and os.path.exists(manifest_path):
        manifest_stat = os.stat(manifest_path)
        manifest_mtime_ns = 0 if manifest_signature_override is not None else getattr(
            manifest_stat,
            "st_mtime_ns",
            int(manifest_stat.st_mtime * 1000000000),
        )
        info.update({
            "training_manifest_path": os.path.abspath(manifest_path),
            "training_manifest_mtime_ns": manifest_mtime_ns,
            "training_manifest_signature": manifest_signature_override or manifest_signature(training_manifest),
        })
    return info


def cache_manifest_matches(manifest, path, dtype, paths, training_manifest=None, manifest_path=None,
                           manifest_signature_override=None):
    source = source_csv_info(path, training_manifest, manifest_path, manifest_signature_override)
    return (
        manifest.get("version") == CACHE_VERSION
        and manifest.get("source_csv_path") == source["source_csv_path"]
        and manifest.get("source_csv_mtime_ns") == source["source_csv_mtime_ns"]
        and manifest.get("source_csv_size") == source["source_csv_size"]
        and manifest.get("training_manifest_path") == source.get("training_manifest_path")
        and manifest.get("training_manifest_mtime_ns") == source.get("training_manifest_mtime_ns")
        and manifest.get("training_manifest_signature") == source.get("training_manifest_signature")
        and manifest.get("feature_dtype") == np.dtype(dtype).name
        and manifest.get("row_count", 0) >= 0
        and bool(manifest.get("feature_columns"))
        and os.path.exists(paths["features"])
        and os.path.exists(paths["metadata"])
    )


def inspect_binary_cache(path, feature_storage, cache_dir, training_manifest=None,
                         manifest_path=None, require_cache_hit=False,
                         manifest_signature_override=None):
    if np is None:
        raise ValueError("binary cache inspection requires numpy")
    if feature_storage not in ("memmap32", "memmap64"):
        raise ValueError("--cache-only/--smoke-test-cache require --feature-storage memmap32 or memmap64")
    dtype = np.float32 if feature_storage == "memmap32" else np.float64
    resolved_cache_dir = resolve_cache_dir(path, cache_dir)
    paths = cache_paths(path, resolved_cache_dir, dtype)
    manifest, manifest_load_seconds = load_cache_manifest_file(paths["manifest"])
    print("Cache manifest load time: {:.3f}s".format(manifest_load_seconds), flush=True)
    if not manifest or not cache_manifest_matches(
            manifest, path, dtype, paths, training_manifest, manifest_path, manifest_signature_override):
        if require_cache_hit:
            raise ValueError(cache_only_missing_message(resolved_cache_dir))
        return None

    metadata_started = time.time()
    with np.load(paths["metadata"], allow_pickle=False) as metadata:
        metadata_info = {
            "symbols": [str(value) for value in metadata["symbols"]],
            "months": [str(value) for value in metadata["months"]],
            "symbol_codes": metadata["symbol_codes"],
            "month_codes": metadata["month_codes"],
            "month_indices": metadata["month_indices"],
            "labels": metadata["labels"],
            "forward_returns": metadata["forward_returns"],
            "trade_returns": metadata["trade_returns"],
            "max_future_high_returns": metadata["max_future_high_returns"],
            "max_future_low_returns": metadata["max_future_low_returns"],
            "quote_volumes": metadata["quote_volumes"],
            "open_times": metadata["open_times"],
        }
    metadata_load_seconds = time.time() - metadata_started
    print("Metadata load time: {:.3f}s".format(metadata_load_seconds), flush=True)
    record_profile_stage(
        "metadata_load",
        metadata_load_seconds,
        rows_processed=len(metadata_info["labels"]),
        extra_info=paths["metadata"],
    )

    row_count = int(manifest["row_count"])
    feature_count = len(manifest["feature_columns"])
    if len(metadata_info["labels"]) != row_count:
        raise ValueError("cache metadata row count mismatch for {}".format(paths["metadata"]))

    open_times, normalized_count = normalize_open_times_array(metadata_info["open_times"])
    metadata_info["open_times"] = open_times
    quote_volumes = metadata_info["quote_volumes"]
    if quote_volumes.size == 0:
        quote_volumes = None
    metadata_info["quote_volumes"] = quote_volumes

    memmap_started = time.time()
    feature_values = np.memmap(
        paths["features"],
        dtype=dtype,
        mode="r",
        shape=(row_count, feature_count),
    )
    memmap_attach_seconds = time.time() - memmap_started
    print("Feature memmap attach time: {:.3f}s".format(memmap_attach_seconds), flush=True)
    record_profile_stage(
        "feature_memmap_attach",
        memmap_attach_seconds,
        rows_processed=row_count,
        extra_info=paths["features"],
    )

    CACHE_LOAD_INFO.clear()
    CACHE_LOAD_INFO.update({
        "status": "hit",
        "paths": paths,
        "manifest": manifest,
        "cache_dir": resolved_cache_dir,
        "feature_dtype": np.dtype(dtype).name,
        "manifest_load_seconds": manifest_load_seconds,
        "metadata_load_seconds": metadata_load_seconds,
        "memmap_attach_seconds": memmap_attach_seconds,
        "normalized_microsecond_open_times": normalized_count,
    })
    return {
        "manifest": manifest,
        "metadata": metadata_info,
        "features": feature_values,
        "dtype": np.dtype(dtype).name,
        "paths": paths,
        "cache_dir": resolved_cache_dir,
        "manifest_load_seconds": manifest_load_seconds,
        "metadata_load_seconds": metadata_load_seconds,
        "memmap_attach_seconds": memmap_attach_seconds,
        "normalized_microsecond_open_times": normalized_count,
    }


def load_cached_compact_rows(path, feature_storage, cache_dir, rebuild_cache=False,
                             training_manifest=None, manifest_path=None,
                             cache_only=False, manifest_signature_override=None):
    dtype = np.float32 if feature_storage == "memmap32" else np.float64
    resolved_cache_dir = resolve_cache_dir(path, cache_dir)
    if rebuild_cache and cache_only:
        raise ValueError("--cache-only cannot be combined with --rebuild-cache")
    cache_hit = None if rebuild_cache else inspect_binary_cache(
        path,
        feature_storage,
        resolved_cache_dir,
        training_manifest,
        manifest_path,
        require_cache_hit=cache_only,
        manifest_signature_override=manifest_signature_override,
    )

    if cache_hit:
        manifest = cache_hit["manifest"]
        metadata = cache_hit["metadata"]
        table = CompactTable(
            metadata["symbols"],
            metadata["months"],
            metadata["symbol_codes"],
            metadata["month_codes"],
            metadata["month_indices"],
            metadata["open_times"],
            metadata["labels"],
            metadata["forward_returns"],
            metadata["trade_returns"],
            metadata["max_future_high_returns"],
            metadata["max_future_low_returns"],
            metadata["quote_volumes"],
            cache_hit["features"],
            {name: index for index, name in enumerate(manifest["feature_columns"])},
            cache_hit["paths"]["features"],
            False,
        )
        log_memory("Loaded compatible binary cache")
        return CompactRows(table), list(manifest["feature_columns"]), bool(manifest.get("has_returns"))

    if cache_only:
        raise ValueError(cache_only_missing_message(resolved_cache_dir))

    os.makedirs(resolved_cache_dir, exist_ok=True)
    paths = cache_paths(path, resolved_cache_dir, dtype)
    clear_cache_files(paths)

    log_memory("Building binary cache from CSV")
    cache_build_token = start_profile_stage("cache_build", os.path.abspath(path))
    rows, feature_columns, has_returns = load_compact_rows(
        path,
        feature_storage,
        resolved_cache_dir,
        paths["features"],
        remove_memmap_on_cleanup=False,
    )
    table = rows.table
    np.savez_compressed(
        paths["metadata"],
        symbols=np.asarray(table.symbols),
        months=np.asarray(table.months),
        symbol_codes=table.symbol_codes,
        month_codes=table.month_codes,
        month_indices=table.month_indices,
        open_times=table.open_times,
        labels=table.labels,
        forward_returns=table.forward_returns,
        trade_returns=table.trade_returns,
        max_future_high_returns=table.max_future_high_returns,
        max_future_low_returns=table.max_future_low_returns,
        quote_volumes=table.quote_volumes if table.quote_volumes is not None else np.asarray([], dtype=np.float32),
    )
    manifest = source_csv_info(path, training_manifest, manifest_path, manifest_signature_override)
    manifest.update({
        "version": CACHE_VERSION,
        "feature_dtype": np.dtype(dtype).name,
        "feature_columns": feature_columns,
        "row_count": len(rows),
        "has_returns": has_returns,
    })
    with open(paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    CACHE_LOAD_INFO.update({
        "status": "rebuilt",
        "paths": paths,
        "manifest": manifest,
        "cache_dir": resolved_cache_dir,
        "feature_dtype": np.dtype(dtype).name,
        "normalized_microsecond_open_times": NORMALIZED_MICROSECOND_OPEN_TIMES,
    })
    finish_profile_stage(cache_build_token, rows_processed=len(rows), extra_info=paths["manifest"])
    log_memory("Finished binary cache build")
    return rows, feature_columns, has_returns


def sharded_dataset_cache_paths(path, cache_dir, dtype):
    source_path = os.path.abspath(path)
    source_hash = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]
    prefix = os.path.join(cache_dir, "sharded-{}-{}".format(source_hash, np.dtype(dtype).name))
    return {
        "features": prefix + ".aggregate.features.dat",
        "metadata": prefix + ".aggregate.metadata.npz",
        "manifest": prefix + ".aggregate.manifest.json",
        "shard_cache_dir": prefix + ".shards",
    }


def sharded_aggregate_manifest_matches(manifest, path, dtype, paths, dataset_manifest, shards):
    return (
        manifest.get("version") == SHARDED_AGGREGATE_CACHE_VERSION
        and manifest.get("dataset_path") == os.path.abspath(path)
        and manifest.get("dataset_manifest_path") == os.path.abspath(training_manifest_path(path))
        and manifest.get("dataset_manifest_signature") == manifest_compatibility_signature(dataset_manifest)
        and manifest.get("inventory_signature") == sharded_inventory_signature(shards)
        and manifest.get("feature_dtype") == np.dtype(dtype).name
        and manifest.get("row_count", 0) >= 0
        and bool(manifest.get("feature_columns"))
        and os.path.exists(paths["features"])
        and os.path.exists(paths["metadata"])
    )


def inspect_sharded_binary_cache(path, feature_storage, cache_dir, dataset_manifest, shards, require_cache_hit=False):
    if np is None:
        raise ValueError("binary cache inspection requires numpy")
    if feature_storage not in ("memmap32", "memmap64"):
        raise ValueError("--cache-only/--smoke-test-cache require --feature-storage memmap32 or memmap64")
    dtype = np.float32 if feature_storage == "memmap32" else np.float64
    resolved_cache_dir = resolve_cache_dir(path, cache_dir)
    paths = sharded_dataset_cache_paths(path, resolved_cache_dir, dtype)
    manifest, manifest_load_seconds = load_cache_manifest_file(paths["manifest"])
    print("Cache manifest load time: {:.3f}s".format(manifest_load_seconds), flush=True)
    if not manifest or not sharded_aggregate_manifest_matches(manifest, path, dtype, paths, dataset_manifest, shards):
        if require_cache_hit:
            raise ValueError(cache_only_missing_message(resolved_cache_dir))
        return None

    metadata_started = time.time()
    with np.load(paths["metadata"], allow_pickle=False) as metadata:
        metadata_info = {
            "symbols": [str(value) for value in metadata["symbols"]],
            "months": [str(value) for value in metadata["months"]],
            "symbol_codes": metadata["symbol_codes"],
            "month_codes": metadata["month_codes"],
            "month_indices": metadata["month_indices"],
            "labels": metadata["labels"],
            "forward_returns": metadata["forward_returns"],
            "trade_returns": metadata["trade_returns"],
            "max_future_high_returns": metadata["max_future_high_returns"],
            "max_future_low_returns": metadata["max_future_low_returns"],
            "quote_volumes": metadata["quote_volumes"],
            "open_times": metadata["open_times"],
        }
    metadata_load_seconds = time.time() - metadata_started
    print("Metadata load time: {:.3f}s".format(metadata_load_seconds), flush=True)

    row_count = int(manifest["row_count"])
    feature_count = len(manifest["feature_columns"])
    if len(metadata_info["labels"]) != row_count:
        raise ValueError("cache metadata row count mismatch for {}".format(paths["metadata"]))

    open_times, normalized_count = normalize_open_times_array(metadata_info["open_times"])
    metadata_info["open_times"] = open_times
    quote_volumes = metadata_info["quote_volumes"]
    if quote_volumes.size == 0:
        quote_volumes = None
    metadata_info["quote_volumes"] = quote_volumes

    memmap_started = time.time()
    feature_values = np.memmap(
        paths["features"],
        dtype=dtype,
        mode="r",
        shape=(row_count, feature_count),
    )
    memmap_attach_seconds = time.time() - memmap_started
    print("Feature memmap attach time: {:.3f}s".format(memmap_attach_seconds), flush=True)

    CACHE_LOAD_INFO.clear()
    CACHE_LOAD_INFO.update({
        "status": "hit",
        "paths": paths,
        "manifest": manifest,
        "cache_dir": resolved_cache_dir,
        "feature_dtype": np.dtype(dtype).name,
        "manifest_load_seconds": manifest_load_seconds,
        "metadata_load_seconds": metadata_load_seconds,
        "memmap_attach_seconds": memmap_attach_seconds,
        "normalized_microsecond_open_times": normalized_count,
        "sharded_dataset": True,
    })
    return {
        "manifest": manifest,
        "metadata": metadata_info,
        "features": feature_values,
        "dtype": np.dtype(dtype).name,
        "paths": paths,
        "cache_dir": resolved_cache_dir,
        "manifest_load_seconds": manifest_load_seconds,
        "metadata_load_seconds": metadata_load_seconds,
        "memmap_attach_seconds": memmap_attach_seconds,
        "normalized_microsecond_open_times": normalized_count,
    }


def build_sharded_aggregate_cache(path, feature_storage, cache_dir, dataset_manifest, shards):
    dtype = np.float32 if feature_storage == "memmap32" else np.float64
    resolved_cache_dir = resolve_cache_dir(path, cache_dir)
    os.makedirs(resolved_cache_dir, exist_ok=True)
    paths = sharded_dataset_cache_paths(path, resolved_cache_dir, dtype)
    clear_cache_files({
        "features": paths["features"],
        "metadata": paths["metadata"],
        "manifest": paths["manifest"],
    })
    os.makedirs(paths["shard_cache_dir"], exist_ok=True)
    shard_signature_override = manifest_compatibility_signature(dataset_manifest)
    cache_build_token = start_profile_stage("cache_build", os.path.abspath(path))

    shard_rows = []
    feature_columns = None
    has_returns = False
    total_rows = 0
    all_month_names = set()
    for shard in shards:
        rows, shard_feature_columns, shard_has_returns = load_cached_compact_rows(
            shard["csv_path"],
            feature_storage,
            paths["shard_cache_dir"],
            rebuild_cache=False,
            training_manifest=dataset_manifest,
            manifest_path=training_manifest_path(path),
            cache_only=False,
            manifest_signature_override=shard_signature_override,
        )
        if feature_columns is None:
            feature_columns = list(shard_feature_columns)
        elif list(shard_feature_columns) != feature_columns:
            raise ValueError("shard {} has incompatible feature columns".format(shard["csv_path"]))
        has_returns = has_returns or shard_has_returns
        total_rows += len(rows)
        all_month_names.update(rows.table.months)
        shard_rows.append(rows)

    if feature_columns is None:
        raise ValueError("{} does not contain any readable shard rows".format(path))

    symbol_lookup = {}
    symbols = []
    months = sorted(all_month_names)
    month_lookup = {month_name: index for index, month_name in enumerate(months)}
    feature_values = np.memmap(
        paths["features"],
        dtype=dtype,
        mode="w+",
        shape=(total_rows, len(feature_columns)),
    )
    symbol_codes = np.empty(total_rows, dtype=np.int32)
    month_codes = np.empty(total_rows, dtype=np.int32)
    month_indices = np.empty(total_rows, dtype=np.int16)
    open_times = np.empty(total_rows, dtype=np.int64)
    labels_values = np.empty(total_rows, dtype=np.int8)
    forward_returns = np.empty(total_rows, dtype=np.float32)
    trade_returns = np.empty(total_rows, dtype=np.float32)
    max_future_high_returns = np.empty(total_rows, dtype=np.float32)
    max_future_low_returns = np.empty(total_rows, dtype=np.float32)
    quote_volumes = np.empty(total_rows, dtype=np.float32)

    offset = 0
    try:
        for rows in shard_rows:
            table = rows.table
            count = len(table.labels)
            for symbol_name in table.symbols:
                if symbol_name not in symbol_lookup:
                    symbol_lookup[symbol_name] = len(symbols)
                    symbols.append(symbol_name)
            feature_values[offset:offset + count, :] = table.features
            symbol_codes[offset:offset + count] = np.asarray(
                [symbol_lookup[table.symbols[int(code)]] for code in table.symbol_codes],
                dtype=np.int32,
            )
            global_month_codes = np.asarray(
                [month_lookup[table.months[int(code)]] for code in table.month_codes],
                dtype=np.int32,
            )
            month_codes[offset:offset + count] = global_month_codes
            month_indices[offset:offset + count] = global_month_codes.astype(np.int16, copy=False)
            open_times[offset:offset + count] = table.open_times
            labels_values[offset:offset + count] = table.labels
            forward_returns[offset:offset + count] = table.forward_returns
            trade_returns[offset:offset + count] = table.trade_returns
            max_future_high_returns[offset:offset + count] = table.max_future_high_returns
            max_future_low_returns[offset:offset + count] = table.max_future_low_returns
            if table.quote_volumes is None:
                quote_volumes[offset:offset + count] = 0.0
            else:
                quote_volumes[offset:offset + count] = table.quote_volumes
            offset += count
            rows.cleanup()
        feature_values.flush()
    except Exception:
        close_memmap(feature_values)
        raise

    np.savez_compressed(
        paths["metadata"],
        symbols=np.asarray(symbols),
        months=np.asarray(months),
        symbol_codes=symbol_codes,
        month_codes=month_codes,
        month_indices=month_indices,
        open_times=open_times,
        labels=labels_values,
        forward_returns=forward_returns,
        trade_returns=trade_returns,
        max_future_high_returns=max_future_high_returns,
        max_future_low_returns=max_future_low_returns,
        quote_volumes=quote_volumes,
    )
    manifest = {
        "version": SHARDED_AGGREGATE_CACHE_VERSION,
        "dataset_path": os.path.abspath(path),
        "dataset_manifest_path": os.path.abspath(training_manifest_path(path)),
        "dataset_manifest_signature": manifest_compatibility_signature(dataset_manifest),
        "inventory_signature": sharded_inventory_signature(shards),
        "feature_dtype": np.dtype(dtype).name,
        "feature_columns": feature_columns,
        "row_count": int(total_rows),
        "has_returns": bool(has_returns),
        "shard_count": len(shards),
    }
    with open(paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    CACHE_LOAD_INFO.clear()
    CACHE_LOAD_INFO.update({
        "status": "rebuilt",
        "paths": paths,
        "manifest": manifest,
        "cache_dir": resolved_cache_dir,
        "feature_dtype": np.dtype(dtype).name,
        "normalized_microsecond_open_times": NORMALIZED_MICROSECOND_OPEN_TIMES,
        "sharded_dataset": True,
    })
    finish_profile_stage(cache_build_token, rows_processed=total_rows, extra_info=paths["manifest"])
    metadata = {
        "symbols": symbols,
        "months": months,
        "symbol_codes": symbol_codes,
        "month_codes": month_codes,
        "month_indices": month_indices,
        "labels": labels_values,
        "forward_returns": forward_returns,
        "trade_returns": trade_returns,
        "max_future_high_returns": max_future_high_returns,
        "max_future_low_returns": max_future_low_returns,
        "quote_volumes": quote_volumes,
        "open_times": open_times,
    }
    table = CompactTable(
        symbols,
        months,
        symbol_codes,
        month_codes,
        month_indices,
        open_times,
        labels_values,
        forward_returns,
        trade_returns,
        max_future_high_returns,
        max_future_low_returns,
        quote_volumes,
        feature_values,
        {name: index for index, name in enumerate(feature_columns)},
        paths["features"],
        False,
    )
    return CompactRows(table), feature_columns, has_returns


def load_sharded_rows(path, feature_storage="float32", memmap_dir=None, cache_dir=None,
                      rebuild_cache=False, disable_cache=False,
                      training_manifest=None, manifest_path=None, cache_only=False):
    if training_manifest is None and manifest_path is None:
        training_manifest, manifest_path = load_training_manifest(path)
    elif manifest_path is None:
        manifest_path = training_manifest_path(path)
    shards = discover_sharded_dataset_shards(path, training_manifest)
    if feature_storage in ("memmap32", "memmap64") and not disable_cache:
        resolved_cache_dir = resolve_cache_dir(path, cache_dir)
        if rebuild_cache and cache_only:
            raise ValueError("--cache-only cannot be combined with --rebuild-cache")
        cache_hit = None if rebuild_cache else inspect_sharded_binary_cache(
            path,
            feature_storage,
            resolved_cache_dir,
            training_manifest,
            shards,
            require_cache_hit=cache_only,
        )
        if cache_hit:
            manifest = cache_hit["manifest"]
            metadata = cache_hit["metadata"]
            table = CompactTable(
                metadata["symbols"],
                metadata["months"],
                metadata["symbol_codes"],
                metadata["month_codes"],
                metadata["month_indices"],
                metadata["open_times"],
                metadata["labels"],
                metadata["forward_returns"],
                metadata["trade_returns"],
                metadata["max_future_high_returns"],
                metadata["max_future_low_returns"],
                metadata["quote_volumes"],
                cache_hit["features"],
                {name: index for index, name in enumerate(manifest["feature_columns"])},
                cache_hit["paths"]["features"],
                False,
            )
            log_memory("Loaded compatible binary cache")
            rows = CompactRows(table)
            validate_rows_against_training_manifest(path, rows, training_manifest)
            return rows, list(manifest["feature_columns"]), bool(manifest.get("has_returns"))
        if cache_only:
            raise ValueError(cache_only_missing_message(resolved_cache_dir))
        log_memory("Building aggregate sharded cache from shard caches")
        rows, feature_columns, has_returns = build_sharded_aggregate_cache(
            path,
            feature_storage,
            resolved_cache_dir,
            training_manifest,
            shards,
        )
        validate_rows_against_training_manifest(path, rows, training_manifest)
        return rows, feature_columns, has_returns

    combined_rows = []
    feature_columns = None
    has_returns = False
    for shard in shards:
        shard_rows, shard_feature_columns, shard_has_returns = load_object_rows(
            shard["csv_path"],
            "float32" if feature_storage == "auto" else feature_storage,
        )
        if feature_columns is None:
            feature_columns = list(shard_feature_columns)
        elif list(shard_feature_columns) != feature_columns:
            raise ValueError("shard {} has incompatible feature columns".format(shard["csv_path"]))
        has_returns = has_returns or shard_has_returns
        combined_rows.extend(shard_rows)
    combined_rows.sort(key=lambda row: (row.symbol, row.month_index, row.open_time))
    validate_rows_against_training_manifest(path, combined_rows, training_manifest)
    return combined_rows, feature_columns or [], has_returns


def load_rows(path, feature_storage="float32", memmap_dir=None, cache_dir=None,
              rebuild_cache=False, disable_cache=False,
              training_manifest=None, manifest_path=None, cache_only=False):
    global NORMALIZED_MICROSECOND_OPEN_TIMES
    NORMALIZED_MICROSECOND_OPEN_TIMES = 0
    if training_manifest is None and manifest_path is None:
        training_manifest, manifest_path = load_training_manifest(path)
    elif manifest_path is None:
        manifest_path = training_manifest_path(path)
    if os.path.isdir(path):
        rows, feature_columns, has_returns = load_sharded_rows(
            path,
            feature_storage,
            memmap_dir,
            cache_dir,
            rebuild_cache,
            disable_cache,
            training_manifest,
            manifest_path,
            cache_only,
        )
        CACHE_LOAD_INFO["normalized_microsecond_open_times"] = NORMALIZED_MICROSECOND_OPEN_TIMES
        return rows, feature_columns, has_returns
    if feature_storage in ("memmap32", "memmap64") and not disable_cache:
        persistent_cache_dir = resolve_cache_dir(path, cache_dir)
        rows, feature_columns, has_returns = load_cached_compact_rows(
            path,
            feature_storage,
            persistent_cache_dir,
            rebuild_cache,
            training_manifest,
            manifest_path,
            cache_only,
        )
        validate_rows_against_training_manifest(path, rows, training_manifest)
        CACHE_LOAD_INFO["normalized_microsecond_open_times"] = NORMALIZED_MICROSECOND_OPEN_TIMES
        return rows, feature_columns, has_returns
    CACHE_LOAD_INFO.clear()
    CACHE_LOAD_INFO.update({
        "status": "disabled" if disable_cache else "not_applicable",
        "feature_storage": feature_storage,
    })
    if feature_storage in ("auto", "matrix32", "matrix64", "memmap32", "memmap64"):
        try:
            rows, feature_columns, has_returns = load_compact_rows(path, feature_storage, memmap_dir)
        except ValueError as error:
            if feature_storage != "auto":
                raise
            print("compact storage unavailable ({}); falling back to per-row float32 storage".format(error), file=sys.stderr)
        else:
            validate_rows_against_training_manifest(path, rows, training_manifest)
            CACHE_LOAD_INFO["normalized_microsecond_open_times"] = NORMALIZED_MICROSECOND_OPEN_TIMES
            return rows, feature_columns, has_returns
    rows, feature_columns, has_returns = load_object_rows(path, "float32" if feature_storage == "auto" else feature_storage)
    validate_rows_against_training_manifest(path, rows, training_manifest)
    CACHE_LOAD_INFO["normalized_microsecond_open_times"] = NORMALIZED_MICROSECOND_OPEN_TIMES
    return rows, feature_columns, has_returns


def labels(rows):
    if is_compact_rows(rows):
        return rows.labels_array()
    return [row.label for row in rows]


def compact_positions_array(rows):
    if rows.indices is not None:
        return rows.indices
    return np.arange(len(rows), dtype=np.int32)


def evenly_sample_array(values, target_count):
    count = len(values)
    if target_count <= 0:
        return values[:0]
    if count <= target_count:
        return values
    take_positions = np.linspace(0, count - 1, num=target_count, dtype=np.int64)
    return values[take_positions]


def sample_label_targets(positive_count, negative_count, max_rows, mode, max_positive_fraction):
    if positive_count <= 0 or negative_count <= 0:
        return min(positive_count, max_rows), max(0, min(negative_count, max_rows - min(positive_count, max_rows)))
    if mode == "balanced":
        positive_target = min(positive_count, max(1, max_rows // 2))
        negative_target = max_rows - positive_target
        return positive_target, min(negative_count, negative_target)
    if mode == "chronological":
        total = positive_count + negative_count
        positive_target = int(round(max_rows * positive_count / float(total)))
        positive_target = min(positive_count, max(1, positive_target))
        negative_target = max_rows - positive_target
        return positive_target, min(negative_count, negative_target)

    positive_cap = max(1, int(round(max_rows * max_positive_fraction)))
    positive_target = min(positive_count, positive_cap)
    negative_target = max_rows - positive_target
    if negative_target > negative_count:
        negative_target = negative_count
        positive_target = min(positive_count, max_rows - negative_target)
    return positive_target, negative_target


def sample_rows(rows, max_rows, label_aware=False, sample_mode="stratified", max_positive_fraction=0.33):
    if not max_rows or max_rows <= 0 or len(rows) <= max_rows:
        return rows

    if is_compact_rows(rows):
        base = compact_positions_array(rows)
        if not label_aware or sample_mode == "chronological":
            sampled = evenly_sample_array(base, max_rows).astype(np.int32, copy=False)
            return rows.subset(sampled)

        row_labels = rows.table.labels[base]
        positives = base[row_labels == 1]
        negatives = base[row_labels == 0]
        if len(positives) == 0 or len(negatives) == 0:
            sampled = evenly_sample_array(base, max_rows).astype(np.int32, copy=False)
            return rows.subset(sampled)

        positive_target, negative_target = sample_label_targets(
            len(positives),
            len(negatives),
            max_rows,
            sample_mode,
            max_positive_fraction,
        )

        sampled_positive = evenly_sample_array(positives, positive_target)
        sampled_negative = evenly_sample_array(negatives, negative_target)
        sampled = np.concatenate((sampled_positive, sampled_negative)).astype(np.int32, copy=False)
        sampled.sort()
        return rows.subset(sampled)

    if not label_aware:
        if max_rows <= 1:
            return rows[:1]
        positions = [int(round(i * (len(rows) - 1) / float(max_rows - 1))) for i in range(max_rows)]
        return [rows[position] for position in positions]

    positives = [row for row in rows if row.label == 1]
    negatives = [row for row in rows if row.label == 0]
    if not positives or not negatives:
        return sample_rows(rows, max_rows, label_aware=False, sample_mode=sample_mode, max_positive_fraction=max_positive_fraction)
    positive_target, negative_target = sample_label_targets(
        len(positives),
        len(negatives),
        max_rows,
        sample_mode,
        max_positive_fraction,
    )
    sampled = (
        sample_rows(positives, positive_target, label_aware=False, sample_mode=sample_mode, max_positive_fraction=max_positive_fraction)
        + sample_rows(negatives, negative_target, label_aware=False, sample_mode=sample_mode, max_positive_fraction=max_positive_fraction)
    )
    sampled.sort(key=lambda row: (row.month_index, row.open_time, row.symbol))
    return sampled


def matrix(rows, storage=None):
    if is_compact_rows(rows):
        values = rows.feature_matrix()
        if storage in ("float64", "matrix64", "memmap64"):
            return np.ascontiguousarray(values, dtype=np.float64)
        return np.ascontiguousarray(values, dtype=np.float32)
    if storage in ("float32", "float64") and np is not None:
        dtype = np.float32 if storage == "float32" else np.float64
        if not rows:
            return np.empty((0, 0), dtype=dtype)
        feature_count = len(rows[0].features)
        values = np.empty((len(rows), feature_count), dtype=dtype)
        for index, row in enumerate(rows):
            values[index, :] = row.features
        return values
    return [row.features for row in rows]


def model_matrix(rows, kind, args):
    if is_compact_rows(rows):
        return matrix(rows, args.feature_storage)
    if kind == "lightgbm" and args.feature_storage in ("float32", "float64", "matrix32", "matrix64", "memmap32", "memmap64"):
        return matrix(rows, args.feature_storage)
    return matrix(rows)


def row_feature_array(rows, name):
    if is_compact_rows(rows):
        feature_index = rows.table.feature_lookup.get(name)
        if feature_index is None:
            return None
        return rows.table.feature_column(feature_index, rows.indices)
    values = []
    for row in rows:
        feature_index = getattr(row, "feature_lookup", {}).get(name)
        if feature_index is None:
            return None
        values.append(float(row.features[feature_index]))
    if np is not None:
        return np.asarray(values, dtype=np.float32)
    return values


def feature_value(rows, index, name, default=0.0):
    if is_compact_rows(rows):
        feature_index = rows.table.feature_lookup.get(name)
        if feature_index is None:
            return default
        position = index if rows.indices is None else int(rows.indices[index])
        return rows.table.feature_value_at(position, feature_index)
    row = rows[index]
    feature_index = getattr(row, "feature_lookup", {}).get(name)
    if feature_index is None:
        return default
    return float(row.features[feature_index])


def feature_values_or_default(rows, name, default=0.0):
    values = row_feature_array(rows, name)
    if values is not None:
        return values
    if np is not None and is_compact_rows(rows):
        return np.full(len(rows), default, dtype=np.float32)
    return [default] * len(rows)


def regression_target_name(args):
    return getattr(args, "regression_target", "trade_return")


def regression_target_feature_name(args):
    if regression_target_name(args) == "risk_adjusted_return":
        return "rolling_volatility_60m"
    return ""


def regression_target_feature_values(rows, args):
    feature_name = regression_target_feature_name(args)
    if not feature_name:
        return None
    values = row_feature_array(rows, feature_name)
    if values is None:
        warn_once(
            "missing-regression-target-feature-{}".format(feature_name),
            "Warning: {} is unavailable for regression_target={}; falling back to trade_return.".format(
                feature_name,
                regression_target_name(args),
            ),
        )
    return values


def actual_trade_returns(rows):
    if is_compact_rows(rows):
        values = rows.table.trade_returns if rows.indices is None else rows.table.trade_returns[rows.indices]
        if np is not None:
            return np.asarray(values, dtype=np.float32)
        return [float(value) for value in values]
    values = [row.trade_return for row in rows]
    if np is not None:
        return np.asarray(values, dtype=np.float32)
    return values


def regression_targets_for_rows(rows, args):
    target_name = regression_target_name(args)
    values = actual_trade_returns(rows)
    if np is not None and isinstance(values, np.ndarray):
        targets = values.astype(np.float32, copy=True)
        if target_name in ("net_return", "clipped_net_return"):
            targets -= np.float32(args.fee + args.slippage)
        if target_name in ("clipped_trade_return", "clipped_net_return"):
            targets = np.clip(targets, args.regression_clip_min, args.regression_clip_max).astype(np.float32, copy=False)
        elif target_name == "risk_adjusted_return":
            volatility = regression_target_feature_values(rows, args)
            if volatility is None:
                return targets
            scale = np.maximum(np.asarray(volatility, dtype=np.float32), np.float32(args.risk_adjusted_return_epsilon))
            targets = (targets / scale).astype(np.float32, copy=False)
        return targets
    targets = [float(value) for value in values]
    if target_name in ("net_return", "clipped_net_return"):
        offset = args.fee + args.slippage
        targets = [value - offset for value in targets]
    if target_name in ("clipped_trade_return", "clipped_net_return"):
        targets = [max(args.regression_clip_min, min(args.regression_clip_max, value)) for value in targets]
    elif target_name == "risk_adjusted_return":
        volatility = regression_target_feature_values(rows, args)
        if volatility is None:
            return targets
        targets = [
            float(value) / max(float(volatility[index]), args.risk_adjusted_return_epsilon)
            for index, value in enumerate(targets)
        ]
    return targets


def regression_predictions_to_trade_return(predictions, rows, args):
    target_name = regression_target_name(args)
    if predictions is None:
        return None
    if np is not None and isinstance(predictions, np.ndarray):
        values = np.asarray(predictions, dtype=np.float32)
        if target_name in ("net_return", "clipped_net_return"):
            return values.astype(np.float32, copy=False) + np.float32(args.fee + args.slippage)
        if target_name == "risk_adjusted_return":
            volatility = regression_target_feature_values(rows, args)
            if volatility is None:
                return values
            scale = np.maximum(np.asarray(volatility, dtype=np.float32), np.float32(args.risk_adjusted_return_epsilon))
            return (values.astype(np.float32, copy=False) * scale).astype(np.float32, copy=False)
        return values.astype(np.float32, copy=False)
    converted = [float(value) for value in predictions]
    if target_name in ("net_return", "clipped_net_return"):
        offset = args.fee + args.slippage
        return [value + offset for value in converted]
    if target_name == "risk_adjusted_return":
        volatility = regression_target_feature_values(rows, args)
        if volatility is None:
            return converted
        return [
            value * max(float(volatility[index]), args.risk_adjusted_return_epsilon)
            for index, value in enumerate(converted)
        ]
    return converted


def model_targets(rows, kind, objective_mode, args=None):
    if objective_mode == "classification":
        values = labels(rows)
        if kind == "lightgbm" and np is not None:
            return np.asarray(values, dtype=np.int8)
        return values
    if args is not None:
        values = regression_targets_for_rows(rows, args)
        if kind == "lightgbm" and np is not None:
            return np.asarray(values, dtype=np.float32)
        return values
    if is_compact_rows(rows):
        values = rows.table.trade_returns if rows.indices is None else rows.table.trade_returns[rows.indices]
        if np is not None:
            return np.asarray(values, dtype=np.float32)
        return [float(value) for value in values]
    values = [row.trade_return for row in rows]
    if kind == "lightgbm" and np is not None:
        return np.asarray(values, dtype=np.float32)
    return values


def rows_slice(rows, start, end):
    if is_compact_rows(rows):
        if rows.indices is None:
            return rows.subset(np.arange(start, end, dtype=np.int32))
        return rows.subset(rows.indices[start:end])
    return rows[start:end]


def select_month_range(rows, start, end):
    if is_compact_rows(rows):
        month_indices = rows.table.month_indices
        if rows.indices is None:
            mask = (month_indices >= start) & (month_indices < end)
            return rows.subset(np.nonzero(mask)[0].astype(np.int32, copy=False))
        base = rows.indices
        mask = (month_indices[base] >= start) & (month_indices[base] < end)
        return rows.subset(base[mask])
    return [row for row in rows if start <= row.month_index < end]


def ratio_split_counts(month_count, train_ratio, validation_ratio, test_ratio):
    if month_count < 3:
        return 0, 0, 0
    validation_count = max(1, int(round(month_count * validation_ratio)))
    test_count = max(1, int(round(month_count * test_ratio)))
    if validation_count + test_count >= month_count:
        validation_count = 1
        test_count = 1
    train_count = month_count - validation_count - test_count
    if train_count < 1:
        train_count = 1
        if validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
    return train_count, validation_count, test_count


def walk_forward_split_bounds(fold_start, walk_train_months, walk_validation_months):
    train_start = int(fold_start)
    train_end = train_start + int(walk_train_months)
    validation_start = train_end
    validation_end = validation_start + int(walk_validation_months)
    test_month = validation_end
    return train_start, train_end, validation_start, validation_end, test_month


def select_ratio_split(rows, args):
    if is_compact_rows(rows):
        table = rows.table
        base = rows.indices
        if base is None:
            symbol_values = table.symbol_codes
            month_values = table.month_indices
        else:
            symbol_values = table.symbol_codes[base]
            month_values = table.month_indices[base]
        max_symbol_code = int(np.max(symbol_values)) if len(symbol_values) else -1
        max_month_index = int(np.max(month_values)) if len(month_values) else -1
        split_lookup = np.zeros((max_symbol_code + 1, max_month_index + 1), dtype=np.int8)

        for symbol_code in np.unique(symbol_values):
            months = sorted(int(month) for month in np.unique(month_values[symbol_values == symbol_code]))
            train_count, validation_count, test_count = ratio_split_counts(
                len(months),
                args.train_ratio,
                args.validation_ratio,
                args.test_ratio,
            )
            if not train_count or not validation_count or not test_count:
                continue
            for month in months[:train_count]:
                split_lookup[int(symbol_code), month] = 1
            for month in months[train_count:train_count + validation_count]:
                split_lookup[int(symbol_code), month] = 2
            for month in months[train_count + validation_count:train_count + validation_count + test_count]:
                split_lookup[int(symbol_code), month] = 3

        flags = split_lookup[symbol_values, month_values]
        if rows.indices is None:
            train_indices = np.nonzero(flags == 1)[0].astype(np.int32, copy=False)
            validation_indices = np.nonzero(flags == 2)[0].astype(np.int32, copy=False)
            test_indices = np.nonzero(flags == 3)[0].astype(np.int32, copy=False)
        else:
            train_indices = base[flags == 1]
            validation_indices = base[flags == 2]
            test_indices = base[flags == 3]
        return rows.subset(train_indices), rows.subset(validation_indices), rows.subset(test_indices)

    months_by_symbol = {}
    for row in rows:
        months_by_symbol.setdefault(row.symbol, set()).add(row.month_index)

    split_lookup = {}
    for symbol, month_set in months_by_symbol.items():
        months = sorted(month_set)
        train_count, validation_count, test_count = ratio_split_counts(
            len(months),
            args.train_ratio,
            args.validation_ratio,
            args.test_ratio,
        )
        if not train_count or not validation_count or not test_count:
            continue
        train_months = set(months[:train_count])
        validation_months = set(months[train_count:train_count + validation_count])
        test_months = set(months[train_count + validation_count:train_count + validation_count + test_count])
        for month in train_months:
            split_lookup[(symbol, month)] = "train"
        for month in validation_months:
            split_lookup[(symbol, month)] = "validation"
        for month in test_months:
            split_lookup[(symbol, month)] = "test"

    train_rows = []
    validation_rows = []
    test_rows = []
    for row in rows:
        split = split_lookup.get((row.symbol, row.month_index))
        if split == "train":
            train_rows.append(row)
        elif split == "validation":
            validation_rows.append(row)
        elif split == "test":
            test_rows.append(row)
    return train_rows, validation_rows, test_rows


def auc_score_from_rows(probabilities, rows):
    if is_compact_rows(rows):
        y_true = rows.labels_array()
        if AUC_SAMPLE_ROWS and len(y_true) > AUC_SAMPLE_ROWS:
            positions = np.linspace(0, len(y_true) - 1, num=AUC_SAMPLE_ROWS, dtype=np.int64)
            y_true = y_true[positions]
            probabilities = np.asarray(probabilities)[positions]
        positives = int(np.sum(y_true))
        negatives = len(y_true) - positives
        if positives == 0 or negatives == 0:
            return 0.0
        probability_values = np.asarray(probabilities, dtype=np.float32)
        order = np.argsort(probability_values, kind="mergesort")
        sorted_probabilities = probability_values[order]
        sorted_labels = y_true[order]
        rank_sum = 0.0
        index = 0
        row_count = len(y_true)
        while index < row_count:
            next_index = index + 1
            while next_index < row_count and sorted_probabilities[next_index] == sorted_probabilities[index]:
                next_index += 1
            positive_ties = int(np.sum(sorted_labels[index:next_index]))
            if positive_ties:
                average_rank = (index + 1 + next_index) / 2.0
                rank_sum += positive_ties * average_rank
            index = next_index
        return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)

    pairs = sorted((probability, row.label) for probability, row in zip(probabilities, rows))
    positives = sum(row.label for row in rows)
    negatives = len(rows) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        for tied in range(index, next_index):
            if pairs[tied][1] == 1:
                rank_sum += average_rank
        index = next_index

    return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


ACCEPTANCE_TIER_RULES = {
    "exploration": [
        ("walkforward_mean_portfolio_return", lambda value: value > 0.0, "mean_portfolio_return {:.4f} <= 0"),
        ("walkforward_profitable_fold_rate", lambda value: value >= 0.20, "profitable_fold_rate {:.4f} < 0.2000"),
        ("walkforward_min_portfolio_return", lambda value: value >= -0.08, "worst_fold_return {:.4f} < -0.0800"),
    ],
    "research": [
        ("walkforward_mean_portfolio_return", lambda value: value > 0.0, "mean_portfolio_return {:.4f} <= 0"),
        ("walkforward_profitable_fold_rate", lambda value: value >= 0.35, "profitable_fold_rate {:.4f} < 0.3500"),
        ("walkforward_median_portfolio_return", lambda value: value >= 0.0, "median_portfolio_return {:.4f} < 0"),
        ("active_profitable_fold_rate", lambda value: value >= 0.50, "active_profitable_fold_rate {:.4f} < 0.5000"),
        ("worst_active_fold_return", lambda value: value >= -0.06, "worst_active_fold_return {:.4f} < -0.0600"),
    ],
    "strong": [
        ("walkforward_mean_portfolio_return", lambda value: value > 0.0, "mean_portfolio_return {:.4f} <= 0"),
        ("walkforward_profitable_fold_rate", lambda value: value >= 0.50, "profitable_fold_rate {:.4f} < 0.5000"),
        ("walkforward_median_portfolio_return", lambda value: value > 0.0, "median_portfolio_return {:.4f} <= 0"),
        ("active_profitable_fold_rate", lambda value: value >= 0.60, "active_profitable_fold_rate {:.4f} < 0.6000"),
        ("worst_active_fold_return", lambda value: value >= -0.04, "worst_active_fold_return {:.4f} < -0.0400"),
    ],
}


def fold_status_from_metrics(predicted_trades, portfolio_profit):
    if int(predicted_trades) <= 0:
        return "inactive"
    if float(portfolio_profit) > 0.0:
        return "active_profitable"
    return "active_losing"


def evaluate_acceptance_tier(summary, tier):
    failures = []
    for key, predicate, message in ACCEPTANCE_TIER_RULES.get(tier, []):
        value = float(summary.get(key, 0.0))
        if not predicate(value):
            failures.append(message.format(value))
    return failures


def walkforward_acceptance_summary(records, args):
    fold_records = [row for row in records if str(row.get("split", "")).startswith("walkforward_fold_")]
    overactive_threshold = int(getattr(args, "overactive_trade_threshold", 150))
    if not fold_records:
        acceptance_tier = getattr(args, "acceptance_tier", "none")
        gated = acceptance_tier != "none" or getattr(args, "require_positive_walkforward", False)
        return {
            "walkforward_total_folds": 0,
            "walkforward_profitable_folds": 0,
            "walkforward_profitable_fold_rate": 0.0,
            "walkforward_median_portfolio_return": 0.0,
            "walkforward_mean_portfolio_return": 0.0,
            "walkforward_min_portfolio_return": 0.0,
            "walkforward_max_portfolio_return": 0.0,
            "walkforward_total_portfolio_profit": 0.0,
            "walkforward_total_predicted_trades": 0,
            "walkforward_mean_precision": 0.0,
            "walkforward_median_precision": 0.0,
            "walkforward_max_drawdown_mean": 0.0,
            "walkforward_max_drawdown_worst": 0.0,
            "active_fold_count": 0,
            "inactive_fold_count": 0,
            "active_fold_rate": 0.0,
            "active_profitable_fold_count": 0,
            "active_losing_fold_count": 0,
            "active_profitable_fold_rate": 0.0,
            "profit_per_active_fold": 0.0,
            "median_active_fold_return": 0.0,
            "mean_active_fold_return": 0.0,
            "worst_active_fold_return": 0.0,
            "best_active_fold_return": 0.0,
            "overactive_losing_folds": 0,
        "overactive_losing_fold_rate": 0.0,
        "avg_trades_in_losing_active_folds": 0.0,
        "avg_trades_in_profitable_active_folds": 0.0,
        "meta_filter_enabled_folds": 0,
        "meta_filter_disabled_folds": 0,
        "acceptance_tier": acceptance_tier,
        "accepted": 0 if gated else 1,
            "rejection_reason": "no_walkforward_folds" if gated else "",
            "failed_acceptance_checks": "no_walkforward_folds" if gated else "",
            "strategy_strength": "rejected" if gated else "not_checked",
        }

    profits = [float(row.get("portfolio_profit", 0.0)) for row in fold_records]
    returns = [float(row.get("portfolio_return", 0.0)) for row in fold_records]
    precisions = [float(row.get("precision", 0.0)) for row in fold_records]
    drawdowns = [float(row.get("max_capital_drawdown", 0.0)) for row in fold_records]
    active_records = [row for row in fold_records if int(row.get("predicted_trades", 0)) > 0]
    active_returns = [float(row.get("portfolio_return", 0.0)) for row in active_records]
    active_profits = [float(row.get("portfolio_profit", 0.0)) for row in active_records]
    active_profitable = sum(1 for profit in active_profits if profit > 0.0)
    losing_active_records = [row for row in active_records if float(row.get("portfolio_profit", 0.0)) <= 0.0]
    profitable_active_records = [row for row in active_records if float(row.get("portfolio_profit", 0.0)) > 0.0]
    overactive_losing_folds = sum(
        1
        for row in losing_active_records
        if int(row.get("predicted_trades", 0)) >= overactive_threshold
    )
    profitable_folds = sum(1 for profit in profits if profit > 0.0)
    total_folds = len(fold_records)
    active_fold_count = len(active_records)
    meta_filter_enabled_folds = sum(1 for row in fold_records if int(row.get("meta_filter_enabled", 0)) > 0)

    summary = {
        "walkforward_total_folds": total_folds,
        "walkforward_profitable_folds": profitable_folds,
        "walkforward_profitable_fold_rate": profitable_folds / float(total_folds) if total_folds else 0.0,
        "walkforward_median_portfolio_return": median(returns),
        "walkforward_mean_portfolio_return": sum(returns) / float(total_folds) if total_folds else 0.0,
        "walkforward_min_portfolio_return": min(returns) if returns else 0.0,
        "walkforward_max_portfolio_return": max(returns) if returns else 0.0,
        "walkforward_total_portfolio_profit": sum(profits),
        "walkforward_total_predicted_trades": sum(int(row.get("predicted_trades", 0)) for row in fold_records),
        "walkforward_mean_precision": sum(precisions) / float(total_folds) if total_folds else 0.0,
        "walkforward_median_precision": median(precisions),
        "walkforward_max_drawdown_mean": sum(drawdowns) / float(total_folds) if total_folds else 0.0,
        "walkforward_max_drawdown_worst": max(drawdowns) if drawdowns else 0.0,
        "active_fold_count": active_fold_count,
        "inactive_fold_count": total_folds - active_fold_count,
        "active_fold_rate": active_fold_count / float(total_folds) if total_folds else 0.0,
        "active_profitable_fold_count": active_profitable,
        "active_losing_fold_count": active_fold_count - active_profitable,
        "active_profitable_fold_rate": active_profitable / float(active_fold_count) if active_fold_count else 0.0,
        "profit_per_active_fold": sum(active_profits) / float(active_fold_count) if active_fold_count else 0.0,
        "median_active_fold_return": median(active_returns),
        "mean_active_fold_return": sum(active_returns) / float(active_fold_count) if active_fold_count else 0.0,
        "worst_active_fold_return": min(active_returns) if active_returns else 0.0,
        "best_active_fold_return": max(active_returns) if active_returns else 0.0,
        "overactive_losing_folds": overactive_losing_folds,
        "overactive_losing_fold_rate": overactive_losing_folds / float(active_fold_count) if active_fold_count else 0.0,
        "avg_trades_in_losing_active_folds": (
            sum(int(row.get("predicted_trades", 0)) for row in losing_active_records) / float(len(losing_active_records))
            if losing_active_records else 0.0
        ),
        "avg_trades_in_profitable_active_folds": (
            sum(int(row.get("predicted_trades", 0)) for row in profitable_active_records) / float(len(profitable_active_records))
            if profitable_active_records else 0.0
        ),
        "meta_filter_enabled_folds": meta_filter_enabled_folds,
        "meta_filter_disabled_folds": total_folds - meta_filter_enabled_folds,
        "acceptance_tier": getattr(args, "acceptance_tier", "none"),
        "accepted": 1,
        "rejection_reason": "",
        "failed_acceptance_checks": "",
        "strategy_strength": "not_checked",
    }

    if getattr(args, "acceptance_tier", "none") != "none":
        failures = evaluate_acceptance_tier(summary, args.acceptance_tier)
        if failures:
            summary["accepted"] = 0
            summary["failed_acceptance_checks"] = "; ".join(failures)
            summary["rejection_reason"] = summary["failed_acceptance_checks"]
            summary["strategy_strength"] = "rejected"
        else:
            summary["strategy_strength"] = "{}_pass".format(args.acceptance_tier)
    elif getattr(args, "require_positive_walkforward", False):
        reasons = []
        if summary["walkforward_profitable_fold_rate"] < args.min_profitable_fold_rate:
            reasons.append(
                "profitable_fold_rate {:.4f} < {:.4f}".format(
                    summary["walkforward_profitable_fold_rate"],
                    args.min_profitable_fold_rate,
                )
            )
        if summary["walkforward_median_portfolio_return"] < args.min_median_fold_return:
            reasons.append(
                "median_fold_return {:.4f} < {:.4f}".format(
                    summary["walkforward_median_portfolio_return"],
                    args.min_median_fold_return,
                )
            )
        if summary["walkforward_mean_portfolio_return"] < args.min_mean_fold_return:
            reasons.append(
                "mean_fold_return {:.4f} < {:.4f}".format(
                    summary["walkforward_mean_portfolio_return"],
                    args.min_mean_fold_return,
                )
            )
        if summary["walkforward_max_drawdown_worst"] > args.max_worst_fold_drawdown:
            reasons.append(
                "worst_fold_drawdown {:.4f} > {:.4f}".format(
                    summary["walkforward_max_drawdown_worst"],
                    args.max_worst_fold_drawdown,
                )
            )
        if reasons:
            summary["accepted"] = 0
            summary["rejection_reason"] = "; ".join(reasons)
            summary["failed_acceptance_checks"] = summary["rejection_reason"]
    return summary


def row_month_pairs(rows):
    if not rows:
        return []
    if is_compact_rows(rows):
        table = rows.table
        if rows.indices is None:
            positions = np.arange(len(table.month_indices), dtype=np.int32) if np is not None else range(len(table.month_indices))
        else:
            positions = rows.indices
        pairs = {}
        for position in positions:
            index = int(table.month_indices[int(position)])
            code = int(table.month_codes[int(position)])
            pairs[index] = table.months[code]
        return [pairs[index] for index in sorted(pairs)]
    pairs = {}
    for row in rows:
        pairs[int(row.month_index)] = row.month
    return [pairs[index] for index in sorted(pairs)]


def month_range_bounds(rows):
    months = row_month_pairs(rows)
    if not months:
        return "", ""
    return months[0], months[-1]


def single_month_label(rows):
    months = row_month_pairs(rows)
    if not months:
        return ""
    return months[0]


WALKFORWARD_DIAGNOSTIC_COLUMNS = [
    "fold_index",
    "split",
    "train_start_month",
    "train_end_month",
    "validation_month",
    "test_month",
    "train_rows",
    "validation_rows",
    "test_rows",
    "train_positive_rows",
    "validation_positive_rows",
    "test_positive_rows",
    "validation_positive_rate",
    "test_positive_rate",
    "selected_threshold",
    "selected_score_name",
    "selected_score_threshold",
    "selected_ev_safety_margin",
    "selected_objective_score",
    "calibration",
    "calibration_a",
    "calibration_b",
    "validation_brier_before",
    "validation_brier_after",
    "predicted_trades",
    "true_positive_rows",
    "false_positive_rows",
    "precision",
    "recall",
    "win_rate",
    "average_expected_value",
    "median_expected_value",
    "average_trade_return",
    "median_trade_return",
    "average_profit_after_fee_and_slippage",
    "average_profit_per_trade",
    "portfolio_profit",
    "portfolio_return",
    "max_capital_drawdown",
    "worst_trade",
    "average_position_size",
    "median_position_size",
    "trades_per_day",
    "max_trades_in_day",
    "trades_per_active_day",
    "active_days",
    "profitable_days",
    "losing_days",
    "worst_day_profit",
    "best_day_profit",
    "blocked_trades_total",
    "blocked_by_trade_frequency",
    "blocked_by_drawdown",
    "selected_base_objective_score",
    "selected_penalized_objective_score",
    "selected_validation_trade_count",
    "selected_validation_raw_signal_count",
    "selected_validation_portfolio_profit",
    "selected_validation_portfolio_return",
    "selected_validation_precision",
    "selected_validation_recall",
    "selected_validation_active_days",
    "selected_validation_profit_per_active_day",
    "selected_validation_average_profit_after_fee_and_slippage",
    "selected_validation_total_profit_after_fee_and_slippage",
    "selected_validation_max_drawdown",
    "selected_threshold_tie_rank_reason",
    "threshold_tiebreaker",
    "threshold_tie_epsilon",
    "threshold_target_trades",
    "threshold_target_active_days",
    "inactive_blocker_source",
    "inactive_blocker_metric",
    "inactive_blocker_threshold",
    "inactive_blocker_best_score",
    "inactive_blocker_gap",
    "inactive_closest_symbol",
    "inactive_promising_fold",
    "regression_calibration",
    "regression_calibration_a",
    "regression_calibration_b",
    "regression_calibration_rows",
    "regression_calibration_rmse_before",
    "regression_calibration_rmse_after",
    "regression_calibration_mae_before",
    "regression_calibration_mae_after",
    "regression_target",
    "risk_adjusted_return_feature",
    "hybrid_return_combination",
    "hybrid_min_probability",
    "hybrid_score_mode",
    "hybrid_uncertainty_method",
    "hybrid_uncertainty_penalty",
    "hybrid_uncertainty_global_std",
    "hybrid_uncertainty_rows",
    "conditional_expected_win_return",
    "conditional_expected_loss_return",
    "conditional_payoff_rows",
    "conditional_payoff_positive_rows",
    "conditional_payoff_negative_rows",
    "conditional_payoff_source",
    "ev_payoff_mode",
    "ev_expected_win_return",
    "ev_expected_loss_return",
    "ev_payoff_rows",
    "ev_payoff_positive_rows",
    "ev_payoff_negative_rows",
    "ev_payoff_source",
    "dynamic_hybrid_thresholds",
    "meta_filter",
    "meta_filter_rows",
    "meta_filter_positive_rate",
    "meta_filter_auc_or_accuracy",
    "meta_filter_enabled",
    "meta_filter_disabled_reason",
    "meta_filter_validation_trade_retention",
    "symbol_validation_filter",
    "symbol_filter_stage",
    "symbol_filter_min_candidates",
    "symbol_filter_min_executed",
    "symbol_filter_candidate_weight",
    "symbol_filter_executed_weight",
    "symbol_filter_shrinkage",
    "symbol_filter_enabled",
    "symbol_filter_allowed_symbols",
    "symbol_filter_total_symbols",
    "symbols_blocked_count",
    "symbols_allowed_count",
    "symbol_filter_disabled_reason",
    "symbol_filter_validation_trade_retention",
    "ensemble_windows",
    "ensemble_model_count",
    "ensemble_enabled",
    "normalized_microsecond_open_times",
    "active_fold",
    "profitable_fold",
    "fold_status",
]


def build_walkforward_diagnostic_record(fold_index, split_name, train_rows, validation_rows, test_rows,
                                        selected_threshold, selected_score, metrics, calibration_info,
                                        validation_metrics,
                                        args):
    train_start_month, train_end_month = month_range_bounds(train_rows)
    predicted_trades = int(metrics.get("predicted_trades", 0))
    portfolio_profit = float(metrics.get("portfolio_profit", 0.0))
    calibration_info = calibration_info or {}
    validation_metrics = validation_metrics or {}
    return {
        "fold_index": fold_index,
        "split": split_name,
        "train_start_month": train_start_month,
        "train_end_month": train_end_month,
        "validation_month": single_month_label(validation_rows),
        "test_month": single_month_label(test_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "train_positive_rows": positive_label_count(train_rows),
        "validation_positive_rows": positive_label_count(validation_rows),
        "test_positive_rows": positive_label_count(test_rows),
        "validation_positive_rate": positive_label_count(validation_rows) / float(len(validation_rows)) if validation_rows else 0.0,
        "test_positive_rate": positive_label_count(test_rows) / float(len(test_rows)) if test_rows else 0.0,
        "selected_threshold": validation_metrics.get("selected_threshold", selected_threshold),
        "selected_score_name": validation_metrics.get("selected_score_name", "probability"),
        "selected_score_threshold": validation_metrics.get("selected_score_threshold", selected_threshold),
        "selected_ev_safety_margin": metrics.get("selected_ev_safety_margin", 0.0),
        "selected_objective_score": validation_metrics.get("selected_objective_score", selected_score),
        "calibration": calibration_info.get("calibration", "none"),
        "calibration_a": calibration_info.get("calibration_a", 0.0),
        "calibration_b": calibration_info.get("calibration_b", 0.0),
        "validation_brier_before": calibration_info.get("validation_brier_before", 0.0),
        "validation_brier_after": calibration_info.get("validation_brier_after", 0.0),
        "predicted_trades": predicted_trades,
        "true_positive_rows": metrics.get("true_positive_rows", 0),
        "false_positive_rows": metrics.get("false_positive_rows", 0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
        "average_expected_value": metrics.get("average_expected_value", 0.0),
        "median_expected_value": metrics.get("median_expected_value", 0.0),
        "average_trade_return": metrics.get("average_trade_return", 0.0),
        "median_trade_return": metrics.get("median_trade_return", 0.0),
        "average_profit_after_fee_and_slippage": metrics.get("average_profit_after_fee_and_slippage", 0.0),
        "average_profit_per_trade": metrics.get("average_profit_per_trade", 0.0),
        "portfolio_profit": portfolio_profit,
        "portfolio_return": metrics.get("portfolio_return", 0.0),
        "max_capital_drawdown": metrics.get("max_capital_drawdown", 0.0),
        "worst_trade": metrics.get("worst_trade", 0.0),
        "average_position_size": metrics.get("average_position_size", 0.0),
        "median_position_size": metrics.get("median_position_size", 0.0),
        "trades_per_day": metrics.get("trades_per_day", 0.0),
        "max_trades_in_day": metrics.get("max_trades_in_any_day", 0),
        "trades_per_active_day": metrics.get("trades_per_active_day", 0.0),
        "active_days": metrics.get("active_days", 0),
        "profitable_days": metrics.get("profitable_days", 0),
        "losing_days": metrics.get("losing_days", 0),
        "worst_day_profit": metrics.get("worst_day_profit", 0.0),
        "best_day_profit": metrics.get("best_day_profit", 0.0),
        "blocked_trades_total": metrics.get("blocked_trades_total", 0),
        "blocked_by_trade_frequency": metrics.get("blocked_by_trade_frequency", 0),
        "blocked_by_drawdown": metrics.get("blocked_by_drawdown", 0),
        "selected_base_objective_score": validation_metrics.get("selected_base_objective_score", 0.0),
        "selected_penalized_objective_score": validation_metrics.get("selected_penalized_objective_score", selected_score),
        "selected_validation_trade_count": validation_metrics.get("selected_validation_trade_count", 0),
        "selected_validation_raw_signal_count": validation_metrics.get("selected_validation_raw_signal_count", 0),
        "selected_validation_portfolio_profit": validation_metrics.get("selected_validation_portfolio_profit", 0.0),
        "selected_validation_portfolio_return": validation_metrics.get("selected_validation_portfolio_return", 0.0),
        "selected_validation_precision": validation_metrics.get("selected_validation_precision", 0.0),
        "selected_validation_recall": validation_metrics.get("selected_validation_recall", 0.0),
        "selected_validation_active_days": validation_metrics.get("selected_validation_active_days", 0),
        "selected_validation_profit_per_active_day": validation_metrics.get("selected_validation_profit_per_active_day", 0.0),
        "selected_validation_average_profit_after_fee_and_slippage": validation_metrics.get(
            "selected_validation_average_profit_after_fee_and_slippage",
            0.0,
        ),
        "selected_validation_total_profit_after_fee_and_slippage": validation_metrics.get(
            "selected_validation_total_profit_after_fee_and_slippage",
            0.0,
        ),
        "selected_validation_max_drawdown": validation_metrics.get("selected_validation_max_drawdown", 0.0),
        "selected_threshold_tie_rank_reason": validation_metrics.get("selected_threshold_tie_rank_reason", ""),
        "threshold_tiebreaker": getattr(args, "threshold_tiebreaker", "fewer_trades"),
        "threshold_tie_epsilon": getattr(args, "threshold_tie_epsilon", 1e-9),
        "threshold_target_trades": getattr(args, "threshold_target_trades", 0),
        "threshold_target_active_days": getattr(args, "threshold_target_active_days", 0),
        "inactive_blocker_source": metrics.get("inactive_blocker_source", ""),
        "inactive_blocker_metric": metrics.get("inactive_blocker_metric", ""),
        "inactive_blocker_threshold": metrics.get("inactive_blocker_threshold", 0.0),
        "inactive_blocker_best_score": metrics.get("inactive_blocker_best_score", 0.0),
        "inactive_blocker_gap": metrics.get("inactive_blocker_gap", 0.0),
        "inactive_closest_symbol": metrics.get("inactive_closest_symbol", ""),
        "inactive_promising_fold": metrics.get("inactive_promising_fold", 0),
        "regression_calibration": calibration_info.get("regression_calibration", "none"),
        "regression_calibration_a": calibration_info.get("regression_calibration_a", 1.0),
        "regression_calibration_b": calibration_info.get("regression_calibration_b", 0.0),
        "regression_calibration_rows": calibration_info.get("regression_calibration_rows", 0),
        "regression_calibration_rmse_before": calibration_info.get("regression_calibration_rmse_before", 0.0),
        "regression_calibration_rmse_after": calibration_info.get("regression_calibration_rmse_after", 0.0),
        "regression_calibration_mae_before": calibration_info.get("regression_calibration_mae_before", 0.0),
        "regression_calibration_mae_after": calibration_info.get("regression_calibration_mae_after", 0.0),
        "regression_target": calibration_info.get("regression_target", getattr(args, "regression_target", "trade_return")),
        "risk_adjusted_return_feature": calibration_info.get("risk_adjusted_return_feature", ""),
        "hybrid_return_combination": calibration_info.get("hybrid_return_combination", getattr(args, "hybrid_return_combination", "probability_times_return")),
        "hybrid_min_probability": calibration_info.get("hybrid_min_probability", getattr(args, "hybrid_min_probability", 0.0)),
        "hybrid_score_mode": calibration_info.get("hybrid_score_mode", getattr(args, "hybrid_score_mode", "basic")),
        "hybrid_uncertainty_method": calibration_info.get("hybrid_uncertainty_method", getattr(args, "hybrid_uncertainty_method", "none")),
        "hybrid_uncertainty_penalty": calibration_info.get("hybrid_uncertainty_penalty", getattr(args, "hybrid_uncertainty_penalty", 0.0)),
        "hybrid_uncertainty_global_std": calibration_info.get("hybrid_uncertainty_global_std", 0.0),
        "hybrid_uncertainty_rows": calibration_info.get("hybrid_uncertainty_rows", 0),
        "conditional_expected_win_return": calibration_info.get("conditional_expected_win_return", 0.0),
        "conditional_expected_loss_return": calibration_info.get("conditional_expected_loss_return", 0.0),
        "conditional_payoff_rows": calibration_info.get("conditional_payoff_rows", 0),
        "conditional_payoff_positive_rows": calibration_info.get("conditional_payoff_positive_rows", 0),
        "conditional_payoff_negative_rows": calibration_info.get("conditional_payoff_negative_rows", 0),
        "conditional_payoff_source": calibration_info.get("conditional_payoff_source", "not_used"),
        "ev_payoff_mode": calibration_info.get("ev_payoff_mode", getattr(args, "ev_payoff_mode", "fixed_targets")),
        "ev_expected_win_return": calibration_info.get("ev_expected_win_return", getattr(args, "effective_upside_target", getattr(args, "upside_target", 0.05))),
        "ev_expected_loss_return": calibration_info.get("ev_expected_loss_return", -getattr(args, "effective_downside_stop", getattr(args, "downside_stop", 0.02))),
        "ev_payoff_rows": calibration_info.get("ev_payoff_rows", 0),
        "ev_payoff_positive_rows": calibration_info.get("ev_payoff_positive_rows", 0),
        "ev_payoff_negative_rows": calibration_info.get("ev_payoff_negative_rows", 0),
        "ev_payoff_source": calibration_info.get("ev_payoff_source", "fixed_targets"),
        "dynamic_hybrid_thresholds": calibration_info.get("dynamic_hybrid_thresholds", getattr(args, "dynamic_hybrid_thresholds", "none")),
        "meta_filter": calibration_info.get("meta_filter", getattr(args, "meta_filter", "none")),
        "meta_filter_rows": calibration_info.get("meta_filter_rows", 0),
        "meta_filter_positive_rate": calibration_info.get("meta_filter_positive_rate", 0.0),
        "meta_filter_auc_or_accuracy": calibration_info.get("meta_filter_auc_or_accuracy", 0.0),
        "meta_filter_enabled": calibration_info.get("meta_filter_enabled", 0),
        "meta_filter_disabled_reason": calibration_info.get("meta_filter_disabled_reason", ""),
        "meta_filter_validation_trade_retention": calibration_info.get("meta_filter_validation_trade_retention", 0.0),
        "symbol_validation_filter": calibration_info.get("symbol_validation_filter", getattr(args, "symbol_validation_filter", "none")),
        "symbol_filter_stage": calibration_info.get("symbol_filter_stage", getattr(args, "symbol_filter_stage", "executed")),
        "symbol_filter_min_candidates": calibration_info.get("symbol_filter_min_candidates", getattr(args, "symbol_filter_min_candidates", 0)),
        "symbol_filter_min_executed": calibration_info.get("symbol_filter_min_executed", getattr(args, "symbol_filter_min_executed", 0)),
        "symbol_filter_candidate_weight": calibration_info.get("symbol_filter_candidate_weight", getattr(args, "symbol_filter_candidate_weight", 0.5)),
        "symbol_filter_executed_weight": calibration_info.get("symbol_filter_executed_weight", getattr(args, "symbol_filter_executed_weight", 0.5)),
        "symbol_filter_shrinkage": calibration_info.get("symbol_filter_shrinkage", getattr(args, "symbol_filter_shrinkage", 50.0)),
        "symbol_filter_enabled": calibration_info.get("symbol_filter_enabled", 0),
        "symbol_filter_allowed_symbols": calibration_info.get("symbol_filter_allowed_symbols", 0),
        "symbol_filter_total_symbols": calibration_info.get("symbol_filter_total_symbols", 0),
        "symbols_blocked_count": calibration_info.get("symbols_blocked_count", 0),
        "symbols_allowed_count": calibration_info.get("symbols_allowed_count", 0),
        "symbol_filter_disabled_reason": calibration_info.get("symbol_filter_disabled_reason", ""),
        "symbol_filter_validation_trade_retention": calibration_info.get("symbol_filter_validation_trade_retention", 0.0),
        "ensemble_windows": calibration_info.get("ensemble_windows", ",".join(str(value) for value in getattr(args, "ensemble_window_list", []))),
        "ensemble_model_count": calibration_info.get("ensemble_model_count", 0),
        "ensemble_enabled": calibration_info.get("ensemble_enabled", 0),
        "normalized_microsecond_open_times": metrics.get("normalized_microsecond_open_times", NORMALIZED_MICROSECOND_OPEN_TIMES),
        "active_fold": 1 if predicted_trades > 0 else 0,
        "profitable_fold": 1 if portfolio_profit > 0.0 else 0,
        "fold_status": fold_status_from_metrics(predicted_trades, portfolio_profit),
    }


def write_walkforward_diagnostics(path, records):
    def write_one(output_path):
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=WALKFORWARD_DIAGNOSTIC_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    atomic_write_path(path, write_one)
    log_memory("Walk-forward diagnostics CSV output complete: {}".format(path))


SYMBOL_FILTER_DIAGNOSTIC_COLUMNS = [
    "split",
    "fold_index",
    "symbol",
    "raw_candidate_count",
    "eligible_candidate_count",
    "executed_trade_count",
    "raw_candidate_avg_score",
    "raw_candidate_avg_probability",
    "raw_candidate_avg_predicted_return",
    "eligible_candidate_avg_score",
    "eligible_candidate_avg_probability",
    "eligible_candidate_avg_predicted_return",
    "eligible_candidate_realized_avg_return",
    "eligible_candidate_positive_rate",
    "executed_avg_profit",
    "executed_avg_return",
    "executed_win_rate",
    "candidate_quality",
    "executed_quality",
    "symbol_score",
    "global_quality",
    "symbol_filter_stage",
    "symbol_filter_decision",
    "symbol_filter_reason",
]


def build_symbol_filter_diagnostic_records(symbol_filter_info, split_name, fold_index):
    if not symbol_filter_info:
        return []
    records = []
    for record in symbol_filter_info.get("diagnostics", []):
        item = dict(record)
        item["split"] = split_name
        item["fold_index"] = fold_index
        records.append(item)
    return records


def write_symbol_filter_diagnostics(path, records):
    def write_one(output_path):
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SYMBOL_FILTER_DIAGNOSTIC_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    atomic_write_path(path, write_one)
    log_memory("Symbol filter diagnostics CSV output complete: {}".format(path))


def open_time_minute(open_time):
    value = normalize_open_time_ms(open_time)
    absolute = abs(value)
    if absolute >= 100000000000:
        return value // (60 * 1000)
    return value // 60


def minute_day_id(minute):
    return int(minute) // (24 * 60)


def day_id_to_string(day_id):
    return datetime.datetime.fromtimestamp(int(day_id) * 24 * 60 * 60, datetime.timezone.utc).strftime("%Y-%m-%d")


def minute_day_string(minute):
    return day_id_to_string(minute_day_id(minute))


def row_symbol_name(rows, local_index):
    if is_compact_rows(rows):
        position = local_index if rows.indices is None else int(rows.indices[local_index])
        return rows.table.symbols[int(rows.table.symbol_codes[position])]
    return rows[local_index].symbol


def compact_quote_volume(table, position):
    if table.quote_volumes is not None:
        raw_quote_volume = float(table.quote_volumes[position])
        if raw_quote_volume > 0.0:
            return raw_quote_volume
    feature_index = table.feature_lookup.get("log_quote_volume")
    if feature_index is None:
        raise ValueError("compact rows require log_quote_volume for portfolio sizing")
    # Legacy CSVs do not have raw quote volume. Keep reconstruction slightly
    # conservative so float32 feature storage cannot exceed the volume cap.
    return max(
        0.0,
        math.expm1(table.feature_value_at(position, feature_index))
        * RECONSTRUCTED_QUOTE_VOLUME_DISCOUNT,
    )


def compact_signal_indices(probabilities, threshold, batch_size=1000000):
    chunks = []
    for start in range(0, len(probabilities), batch_size):
        end = min(len(probabilities), start + batch_size)
        selected = np.nonzero(np.asarray(probabilities[start:end]) >= threshold)[0]
        if selected.size:
            chunks.append((selected + start).astype(np.int32, copy=False))
    if not chunks:
        return np.asarray([], dtype=np.int32)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks).astype(np.int32, copy=False)


def compact_open_time_minutes(open_times):
    values = np.asarray(open_times, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64).copy()
    microseconds = np.abs(values) > 10_000_000_000_000
    if np.any(microseconds):
        values[microseconds] = values[microseconds] // 1000
    result = np.empty(len(values), dtype=np.int64)
    milliseconds = np.abs(values) >= 100000000000
    seconds = ~milliseconds
    result[milliseconds] = values[milliseconds] // (60 * 1000)
    result[seconds] = values[seconds] // 60
    return result


def row_probability_threshold(threshold, index):
    if isinstance(threshold, list) or (np is not None and isinstance(threshold, np.ndarray)):
        return float(threshold[index])
    return float(threshold)


def normalize_prediction_bundle(predictions):
    if isinstance(predictions, dict):
        return predictions
    return build_prediction_bundle(probability=predictions, calibrated_probability=predictions)


def portfolio_execution(rows, predictions, threshold, fee, slippage, initial_capital,
                        max_position_fraction, max_volume_fraction, max_trades_per_period,
                        trade_period_minutes, holding_period_minutes,
                        threshold_objective="avg_profit", trade_selection="threshold",
                        top_k_per_minute=3, upside_target=0.05, downside_stop=0.02,
                        ev_safety_margin=0.0, objective_mode="classification",
                        trade_score_name="probability", min_predicted_net_return=0.0,
                        hybrid_min_score=0.0, max_trades_per_day=0,
                        max_trades_per_fold=0, max_losing_trades_per_day=0,
                        max_daily_drawdown=0.0, pause_after_drawdown_minutes=0,
                        capture_blocked_details=False,
                        hybrid_runtime_args=None,
                        symbol_filter_info=None):
    if initial_capital <= 0.0:
        raise ValueError("--initial-capital must be positive")
    bundle = normalize_prediction_bundle(predictions)
    raw_selected = {}

    cash = float(initial_capital)
    invested = 0.0
    peak_equity = float(initial_capital)
    max_drawdown = 0.0
    recent_entry_minutes = deque()
    open_positions = deque()
    executed = {}
    executed_pnls = {}
    executed_minutes = {}
    executed_trade_day_ids = {}
    executed_expected_values = {}
    executed_trade_scores = {}
    executed_selection_ranks = {}
    executed_selected_by_topk = {}
    executed_selected_by_meta_filter = {}
    blocked_flags = {} if capture_blocked_details else None
    blocked_counts = {
        "daily_trade_limit_blocked": 0,
        "fold_trade_limit_blocked": 0,
        "daily_loss_limit_blocked": 0,
        "daily_drawdown_limit_blocked": 0,
        "period_trade_limit_blocked": 0,
    }
    entries_by_day = {}
    realized_pnl_by_day = {}
    realized_losing_trades_by_day = {}
    pause_until_minute_by_day = {}
    fold_trade_count = 0
    use_expected_value_ranking = (
        trade_selection == "topk_ev"
        and bundle.get("probability") is not None
        and objective_mode == "classification"
    )
    hybrid_score_mode = getattr(hybrid_runtime_args, "hybrid_score_mode", "basic") if hybrid_runtime_args is not None else "basic"
    hybrid_uncertainty_penalty = getattr(hybrid_runtime_args, "hybrid_uncertainty_penalty", 0.0) if hybrid_runtime_args is not None else 0.0
    meta_filter_mode = getattr(hybrid_runtime_args, "meta_filter", "none") if hybrid_runtime_args is not None else "none"
    meta_filter_min_probability = getattr(hybrid_runtime_args, "meta_filter_min_probability", 0.0) if hybrid_runtime_args is not None else 0.0
    meta_filter_active = meta_filter_mode != "none" and bundle.get("meta_probability") is not None
    ev_context = resolve_ev_context(bundle, hybrid_runtime_args, upside_target, downside_stop)
    hybrid_context = resolve_hybrid_return_context(bundle, hybrid_runtime_args, upside_target, downside_stop)
    hybrid_probability_gate = hybrid_probability_gate_threshold(bundle, hybrid_runtime_args, upside_target, downside_stop)
    allowed_symbols = None
    if symbol_filter_info and symbol_filter_info.get("enabled"):
        allowed_symbols = set(symbol_filter_info.get("allowed_symbols", []))
    base_hybrid_threshold = float(threshold) if not isinstance(threshold, list) and not (np is not None and isinstance(threshold, np.ndarray)) else float(hybrid_min_score)
    effective_hybrid_thresholds, _ = compute_dynamic_hybrid_thresholds(
        rows,
        hybrid_runtime_args,
        max(base_hybrid_threshold, float(hybrid_min_score)),
    ) if objective_mode == "hybrid" and hybrid_runtime_args is not None else (None, None)

    def record_block(index, reason):
        blocked_counts[reason] += 1
        if capture_blocked_details:
            flags = blocked_flags.setdefault(index, {
                "blocked_by_daily_trade_limit": 0,
                "blocked_by_fold_trade_limit": 0,
                "blocked_by_daily_loss_limit": 0,
                "blocked_by_daily_drawdown_limit": 0,
            })
            if reason == "daily_trade_limit_blocked":
                flags["blocked_by_daily_trade_limit"] = 1
            elif reason == "fold_trade_limit_blocked":
                flags["blocked_by_fold_trade_limit"] = 1
            elif reason == "daily_loss_limit_blocked":
                flags["blocked_by_daily_loss_limit"] = 1
            elif reason == "daily_drawdown_limit_blocked":
                flags["blocked_by_daily_drawdown_limit"] = 1

    def release_positions(until_minute):
        nonlocal cash, invested, peak_equity, max_drawdown
        while open_positions and open_positions[0][0] <= until_minute:
            release_minute, position_size, pnl, entry_day_id = open_positions.popleft()
            invested -= position_size
            cash += position_size + pnl
            day_id = entry_day_id if entry_day_id is not None else minute_day_id(release_minute)
            realized_pnl_by_day[day_id] = realized_pnl_by_day.get(day_id, 0.0) + pnl
            if pnl < 0.0:
                realized_losing_trades_by_day[day_id] = realized_losing_trades_by_day.get(day_id, 0) + 1
            if max_daily_drawdown > 0.0 and realized_pnl_by_day.get(day_id, 0.0) <= -max_daily_drawdown * initial_capital:
                if pause_after_drawdown_minutes > 0:
                    pause_until_minute_by_day[day_id] = max(
                        pause_until_minute_by_day.get(day_id, 0),
                        release_minute + pause_after_drawdown_minutes,
                    )
                else:
                    pause_until_minute_by_day[day_id] = max(
                        pause_until_minute_by_day.get(day_id, 0),
                        (day_id + 1) * 24 * 60,
                    )
            equity = cash + invested
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, peak_equity - equity)

    def execute_candidate(item, selection_rank, selected_by_topk_flag):
        nonlocal cash, invested, fold_trade_count
        local_index, minute, day_id, quote_volume_value, trade_return_value, expected_value, trade_score, calibrated_probability, meta_probability = item
        release_positions(minute)
        if meta_filter_active and meta_probability < meta_filter_min_probability:
            return
        if max_trades_per_fold > 0 and fold_trade_count >= max_trades_per_fold:
            record_block(local_index, "fold_trade_limit_blocked")
            return
        if max_trades_per_day > 0 and entries_by_day.get(day_id, 0) >= max_trades_per_day:
            record_block(local_index, "daily_trade_limit_blocked")
            return
        if max_losing_trades_per_day > 0 and realized_losing_trades_by_day.get(day_id, 0) >= max_losing_trades_per_day:
            record_block(local_index, "daily_loss_limit_blocked")
            return
        pause_until = pause_until_minute_by_day.get(day_id, 0)
        if pause_until and minute < pause_until:
            record_block(local_index, "daily_drawdown_limit_blocked")
            return
        if max_trades_per_period > 0:
            while recent_entry_minutes and recent_entry_minutes[0] <= minute - trade_period_minutes:
                recent_entry_minutes.popleft()
            if len(recent_entry_minutes) >= max_trades_per_period:
                blocked_counts["period_trade_limit_blocked"] += 1
                return
        volume_cap = quote_volume_value * max_volume_fraction
        equity_position_cap = (cash + invested) * max_position_fraction
        position_size = min(equity_position_cap, volume_cap, cash)
        if position_size <= 0.0:
            return
        cash -= position_size
        invested += position_size
        net_return = trade_return_value - fee - slippage
        pnl = position_size * net_return
        open_positions.append((minute + holding_period_minutes, position_size, pnl, day_id))
        if max_trades_per_period > 0:
            recent_entry_minutes.append(minute)
        fold_trade_count += 1
        entries_by_day[day_id] = entries_by_day.get(day_id, 0) + 1
        executed[local_index] = position_size
        executed_pnls[local_index] = pnl
        executed_minutes[local_index] = minute
        executed_trade_day_ids[local_index] = day_id
        executed_expected_values[local_index] = expected_value
        executed_trade_scores[local_index] = trade_score
        executed_selection_ranks[local_index] = selection_rank
        executed_selected_by_topk[local_index] = selected_by_topk_flag
        executed_selected_by_meta_filter[local_index] = 1 if meta_filter_active else 0

    def flush_bucket(bucket):
        if not bucket:
            return
        if trade_selection in ("topk_ev", "topk_score"):
            if use_expected_value_ranking:
                ranked = sorted(bucket, key=lambda item: (-item[5], -item[7]))
            else:
                ranked = sorted(bucket, key=lambda item: (-item[6], -item[7]))
            limit = max(0, int(top_k_per_minute))
            if limit <= 0:
                chosen = ranked
            elif limit == 1:
                chosen = [ranked[0]]
            else:
                chosen = ranked[:limit]
            if use_expected_value_ranking:
                chosen = sorted(chosen, key=lambda item: (-item[7], -item[6]))
            else:
                chosen = sorted(chosen, key=lambda item: (-item[6], -item[7]))
            for rank, item in enumerate(chosen, 1):
                execute_candidate(item, rank, 1)
            return

        if objective_mode == "classification":
            ordered = sorted(bucket, key=lambda item: (-item[7], -item[6]))
        else:
            ordered = sorted(bucket, key=lambda item: (-item[6], -item[7]))
        for item in ordered:
            execute_candidate(item, 1, 0)

    current_minute = None
    current_bucket = []

    def push_candidate(item):
        nonlocal current_minute, current_bucket
        minute = item[1]
        if current_minute is None:
            current_minute = minute
        if minute != current_minute:
            flush_bucket(current_bucket)
            current_bucket = []
            current_minute = minute
        current_bucket.append(item)

    if is_compact_rows(rows):
        if np is not None:
            signal_indices = compact_signal_indices_for_bundle(
                rows,
                bundle,
                threshold,
                objective_mode=objective_mode,
                threshold_objective=threshold_objective,
                trade_score_name=trade_score_name,
                upside_target=upside_target,
                downside_stop=downside_stop,
                fee=fee,
                slippage=slippage,
                ev_safety_margin=ev_safety_margin,
                min_predicted_net_return=min_predicted_net_return,
                hybrid_min_score=hybrid_min_score,
                hybrid_score_mode=hybrid_score_mode,
                hybrid_uncertainty_penalty=hybrid_uncertainty_penalty,
                effective_hybrid_thresholds=effective_hybrid_thresholds,
                ev_context=ev_context,
                hybrid_context=hybrid_context,
            )
            if rows.indices is None:
                absolute_positions = signal_indices
            else:
                absolute_positions = rows.indices[signal_indices]
            if len(signal_indices):
                signal_minutes = compact_open_time_minutes(rows.table.open_times[absolute_positions])
                signal_day_ids = (signal_minutes // (24 * 60)).astype(np.int64, copy=False)
                calibrated_values = maybe_float32_array(bundle.get("calibrated_probability"))
                if calibrated_values is None:
                    calibrated_values = maybe_float32_array(bundle.get("probability"))
                predicted_returns = maybe_float32_array(bundle.get("predicted_trade_return"))
                if predicted_returns is None:
                    predicted_returns = maybe_float32_array(bundle.get("raw_predicted_trade_return"))
                predicted_uncertainty = maybe_float32_array(bundle.get("predicted_return_uncertainty"))
                if calibrated_values is not None:
                    selected_calibrated = calibrated_values[signal_indices].astype(np.float32, copy=False)
                    selected_expected_values = score_batch_ev(
                        selected_calibrated,
                        upside_target,
                        downside_stop,
                        fee,
                        slippage,
                        predicted_trade_returns=predicted_returns[signal_indices].astype(np.float32, copy=False) if predicted_returns is not None else None,
                        objective_mode=objective_mode,
                        ev_context=ev_context,
                    )
                else:
                    selected_calibrated = np.zeros(len(signal_indices), dtype=np.float32)
                    selected_expected_values = np.zeros(len(signal_indices), dtype=np.float32)
                if trade_score_name == "probability":
                    selected_trade_scores = selected_calibrated
                elif trade_score_name == "ev":
                    selected_trade_scores = selected_expected_values
                elif trade_score_name == "predicted_return":
                    if predicted_returns is None:
                        selected_trade_scores = np.zeros(len(signal_indices), dtype=np.float32)
                    else:
                        selected_trade_scores = predicted_returns[signal_indices].astype(np.float32, copy=False) - fee - slippage
                elif trade_score_name == "hybrid":
                    selected_trade_scores = score_batch_hybrid(
                        selected_calibrated,
                        predicted_returns[signal_indices].astype(np.float32, copy=False) if predicted_returns is not None else None,
                        fee,
                        slippage,
                        predicted_uncertainty[signal_indices].astype(np.float32, copy=False) if predicted_uncertainty is not None else None,
                        hybrid_score_mode,
                        hybrid_uncertainty_penalty,
                        hybrid_context=hybrid_context,
                        uncertainty_context=bundle.get("uncertainty_context"),
                        hybrid_runtime_args=hybrid_runtime_args,
                    )
                else:
                    raise ValueError("unknown trade score: {}".format(trade_score_name))

                for offset, local_index in enumerate(signal_indices):
                    local_index = int(local_index)
                    if allowed_symbols is not None and row_symbol_name(rows, local_index) not in allowed_symbols:
                        continue
                    raw_selected[local_index] = 1
                    position = int(absolute_positions[offset])
                    push_candidate((
                        local_index,
                        int(signal_minutes[offset]),
                        int(signal_day_ids[offset]),
                        compact_quote_volume(rows.table, position),
                        float(rows.table.trade_returns[position]),
                        float(selected_expected_values[offset]),
                        float(selected_trade_scores[offset]),
                        float(selected_calibrated[offset]),
                        meta_probability_value(bundle, local_index),
                    ))
        else:
            for local_index in range(len(rows)):
                if objective_mode == "classification":
                    probability = calibrated_probability_value(bundle, local_index)
                    if probability < row_probability_threshold(threshold, local_index):
                        continue
                    if threshold_objective == "ev" and expected_value_for_bundle(bundle, local_index, upside_target, downside_stop, fee, slippage, hybrid_runtime_args) <= ev_safety_margin:
                        continue
                elif objective_mode == "return_regression":
                    if predicted_net_return_value(bundle, local_index, fee, slippage) < max(float(threshold), min_predicted_net_return):
                        continue
                elif hybrid_score_value(bundle, local_index, fee, slippage, hybrid_score_mode, hybrid_uncertainty_penalty, hybrid_runtime_args, upside_target, downside_stop) < (
                        float(effective_hybrid_thresholds[local_index]) if effective_hybrid_thresholds is not None else max(float(threshold), hybrid_min_score)):
                    continue
                elif hybrid_probability_gate > 0.0 and calibrated_probability_value(bundle, local_index) < hybrid_probability_gate:
                    continue
                if allowed_symbols is not None and row_symbol_name(rows, local_index) not in allowed_symbols:
                    continue
                position = local_index if rows.indices is None else int(rows.indices[local_index])
                minute = open_time_minute(rows.table.open_times[position])
                raw_selected[local_index] = 1
                push_candidate((
                    int(local_index),
                    minute,
                    minute_day_id(minute),
                    compact_quote_volume(rows.table, position),
                    float(rows.table.trade_returns[position]),
                    expected_value_for_bundle(bundle, local_index, upside_target, downside_stop, fee, slippage, hybrid_runtime_args),
                    trade_score_value(bundle, local_index, trade_score_name, upside_target, downside_stop, fee, slippage, hybrid_score_mode, hybrid_uncertainty_penalty, hybrid_runtime_args),
                    calibrated_probability_value(bundle, local_index),
                    meta_probability_value(bundle, local_index),
                ))
    else:
        for local_index in range(len(rows)):
            if objective_mode == "classification":
                probability = calibrated_probability_value(bundle, local_index)
                if probability < row_probability_threshold(threshold, local_index):
                    continue
                if threshold_objective == "ev" and expected_value_for_bundle(bundle, local_index, upside_target, downside_stop, fee, slippage, hybrid_runtime_args) <= ev_safety_margin:
                    continue
            elif objective_mode == "return_regression":
                if predicted_net_return_value(bundle, local_index, fee, slippage) < max(float(threshold), min_predicted_net_return):
                    continue
            elif hybrid_score_value(bundle, local_index, fee, slippage, hybrid_score_mode, hybrid_uncertainty_penalty, hybrid_runtime_args, upside_target, downside_stop) < (
                    float(effective_hybrid_thresholds[local_index]) if effective_hybrid_thresholds is not None else max(float(threshold), hybrid_min_score)):
                continue
            elif hybrid_probability_gate > 0.0 and calibrated_probability_value(bundle, local_index) < hybrid_probability_gate:
                continue
            if allowed_symbols is not None and row_symbol_name(rows, local_index) not in allowed_symbols:
                continue
            minute = open_time_minute(rows[local_index].open_time)
            raw_selected[local_index] = 1
            push_candidate((
                local_index,
                minute,
                minute_day_id(minute),
                rows[local_index].quote_volume,
                rows[local_index].trade_return,
                expected_value_for_bundle(bundle, local_index, upside_target, downside_stop, fee, slippage, hybrid_runtime_args),
                trade_score_value(bundle, local_index, trade_score_name, upside_target, downside_stop, fee, slippage, hybrid_score_mode, hybrid_uncertainty_penalty, hybrid_runtime_args),
                calibrated_probability_value(bundle, local_index),
                meta_probability_value(bundle, local_index),
            ))

    flush_bucket(current_bucket)

    release_positions(sys.maxsize)
    portfolio_profit = cash - initial_capital
    position_sizes = list(executed.values())
    active_days = len(entries_by_day)
    realized_day_profits = list(realized_pnl_by_day.values())
    profitable_days = sum(1 for value in realized_day_profits if value > 0.0)
    losing_days = sum(1 for value in realized_day_profits if value < 0.0)
    max_trades_in_any_day = max(entries_by_day.values()) if entries_by_day else 0
    return {
        "executed": executed,
        "executed_pnls": executed_pnls,
        "executed_minutes": executed_minutes,
        "executed_trade_day_ids": executed_trade_day_ids,
        "executed_expected_values": executed_expected_values,
        "executed_trade_scores": executed_trade_scores,
        "executed_selection_ranks": executed_selection_ranks,
        "executed_selected_by_topk": executed_selected_by_topk,
        "executed_selected_by_meta_filter": executed_selected_by_meta_filter,
        "raw_selected": raw_selected,
        "blocked_flags": blocked_flags or {},
        "ending_capital": cash,
        "portfolio_profit": portfolio_profit,
        "portfolio_return": portfolio_profit / initial_capital,
        "max_capital_drawdown": max_drawdown,
        "average_position_size": sum(position_sizes) / len(position_sizes) if position_sizes else 0.0,
        "median_position_size": median(position_sizes),
        "average_profit_per_trade": portfolio_profit / len(position_sizes) if position_sizes else 0.0,
        "worst_trade": min(executed_pnls.values()) if executed_pnls else 0.0,
        "daily_trade_limit_blocked": blocked_counts["daily_trade_limit_blocked"],
        "fold_trade_limit_blocked": blocked_counts["fold_trade_limit_blocked"],
        "daily_loss_limit_blocked": blocked_counts["daily_loss_limit_blocked"],
        "daily_drawdown_limit_blocked": blocked_counts["daily_drawdown_limit_blocked"],
        "blocked_trades_total": sum(blocked_counts.values()),
        "blocked_by_trade_frequency": (
            blocked_counts["daily_trade_limit_blocked"]
            + blocked_counts["fold_trade_limit_blocked"]
            + blocked_counts["daily_loss_limit_blocked"]
            + blocked_counts["period_trade_limit_blocked"]
        ),
        "blocked_by_drawdown": blocked_counts["daily_drawdown_limit_blocked"],
        "max_trades_in_any_day": max_trades_in_any_day,
        "max_trades_in_any_fold": fold_trade_count,
        "trades_per_active_day": float(fold_trade_count) / active_days if active_days else 0.0,
        "active_days": active_days,
        "profitable_days": profitable_days,
        "losing_days": losing_days,
        "worst_day_profit": min(realized_day_profits) if realized_day_profits else 0.0,
        "best_day_profit": max(realized_day_profits) if realized_day_profits else 0.0,
    }


def raw_classification_metrics(rows, predictions, threshold, batch_size=1000000,
                               objective_mode="classification", trade_score_name="probability",
                               upside_target=0.05, downside_stop=0.02, fee=0.0, slippage=0.0,
                               ev_safety_margin=0.0, min_predicted_net_return=0.0,
                               hybrid_min_score=0.0, hybrid_runtime_args=None):
    bundle = normalize_prediction_bundle(predictions)
    hybrid_score_mode = getattr(hybrid_runtime_args, "hybrid_score_mode", "basic") if hybrid_runtime_args is not None else "basic"
    hybrid_uncertainty_penalty = getattr(hybrid_runtime_args, "hybrid_uncertainty_penalty", 0.0) if hybrid_runtime_args is not None else 0.0
    ev_context = resolve_ev_context(bundle, hybrid_runtime_args, upside_target, downside_stop)
    hybrid_context = resolve_hybrid_return_context(bundle, hybrid_runtime_args, upside_target, downside_stop)
    hybrid_probability_gate = hybrid_probability_gate_threshold(bundle, hybrid_runtime_args, upside_target, downside_stop)
    effective_hybrid_thresholds, _ = compute_dynamic_hybrid_thresholds(
        rows,
        hybrid_runtime_args,
        max(float(threshold), float(hybrid_min_score)),
    ) if objective_mode == "hybrid" and hybrid_runtime_args is not None else (None, None)

    def signal(local_index):
        if objective_mode == "classification":
            probability = calibrated_probability_value(bundle, local_index)
            if probability < row_probability_threshold(threshold, local_index):
                return False
            if trade_score_name == "ev" and expected_value_for_bundle(bundle, local_index, upside_target, downside_stop, fee, slippage, hybrid_runtime_args) <= ev_safety_margin:
                return False
            return True
        if objective_mode == "return_regression":
            return predicted_net_return_value(bundle, local_index, fee, slippage) >= max(threshold, min_predicted_net_return)
        if hybrid_score_value(bundle, local_index, fee, slippage, hybrid_score_mode, hybrid_uncertainty_penalty, hybrid_runtime_args, upside_target, downside_stop) < (
            float(effective_hybrid_thresholds[local_index]) if effective_hybrid_thresholds is not None else max(threshold, hybrid_min_score)
        ):
            return False
        if hybrid_probability_gate > 0.0 and calibrated_probability_value(bundle, local_index) < hybrid_probability_gate:
            return False
        return True

    raw_trades = 0
    raw_true_positives = 0
    if is_compact_rows(rows) and np is not None:
        labels_array = rows.labels_array()
        signal_indices = compact_signal_indices_for_bundle(
            rows,
            bundle,
            threshold,
            objective_mode=objective_mode,
            threshold_objective="ev" if trade_score_name == "ev" else "avg_profit",
            trade_score_name=trade_score_name,
            upside_target=upside_target,
            downside_stop=downside_stop,
            fee=fee,
            slippage=slippage,
            ev_safety_margin=ev_safety_margin,
            min_predicted_net_return=min_predicted_net_return,
            hybrid_min_score=hybrid_min_score,
            hybrid_score_mode=hybrid_score_mode,
            hybrid_uncertainty_penalty=hybrid_uncertainty_penalty,
            effective_hybrid_thresholds=effective_hybrid_thresholds,
            batch_size=batch_size,
            classification_ev_from_trade_score=True,
            ev_context=ev_context,
            hybrid_context=hybrid_context,
        )
        raw_trades = int(len(signal_indices))
        if raw_trades:
            raw_true_positives = int(np.sum(labels_array[signal_indices]))
        actual_positive = int(np.sum(labels_array))
    else:
        actual_positive = sum(row.label for row in rows)
        for index, row in enumerate(rows):
            if signal(index):
                raw_trades += 1
                raw_true_positives += row.label
    raw_false_positives = raw_trades - raw_true_positives
    raw_precision = float(raw_true_positives) / raw_trades if raw_trades else 0.0
    raw_recall = float(raw_true_positives) / actual_positive if actual_positive else 0.0
    raw_f1 = 2.0 * raw_precision * raw_recall / (raw_precision + raw_recall) if raw_precision + raw_recall else 0.0
    return {
        "raw_signal_trades": raw_trades,
        "raw_true_positive_rows": raw_true_positives,
        "raw_false_positive_rows": raw_false_positives,
        "raw_precision": raw_precision,
        "raw_recall": raw_recall,
        "raw_f1": raw_f1,
    }


def execution_frequency_metrics(execution):
    minutes = list(execution["executed_minutes"].values())
    if not minutes:
        return {"trades_per_day": 0.0, "trades_per_month": 0.0}
    days = set(minute // (24 * 60) for minute in minutes)
    months = set(datetime.datetime.fromtimestamp(minute * 60, datetime.timezone.utc).strftime("%Y-%m") for minute in minutes)
    return {
        "trades_per_day": len(minutes) / float(len(days)),
        "trades_per_month": len(minutes) / float(len(months)),
    }


def evaluate_compact(rows, predictions, threshold, fee, slippage, compute_auc=True, initial_capital=10000.0,
                     max_position_fraction=0.10, max_volume_fraction=0.01,
                     max_trades_per_period=10, trade_period_minutes=60,
                     holding_period_minutes=5, threshold_objective="avg_profit",
                     trade_selection="threshold", top_k_per_minute=3,
                     upside_target=0.05, downside_stop=0.02, ev_safety_margin=0.0,
                     objective_mode="classification", trade_score_name="probability",
                     min_predicted_net_return=0.0, hybrid_min_score=0.0,
                     max_trades_per_day=0, max_trades_per_fold=0,
                     max_losing_trades_per_day=0, max_daily_drawdown=0.0,
                     pause_after_drawdown_minutes=0, hybrid_runtime_args=None,
                     symbol_filter_info=None):
    bundle = normalize_prediction_bundle(predictions)
    table = rows.table
    actual_positive = int(np.sum(rows.labels_array()))
    predicted_trades = 0
    tp = fp = tn = fn = 0
    returns = []
    trade_returns = []
    expected_values = []
    sum_return = 0.0
    sum_trade_return = 0.0
    sum_expected_value = 0.0
    sum_mfe = 0.0
    sum_mae = 0.0
    total_fee = 0.0
    total_fee_slippage = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    winning_trades = 0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    raw_metrics = raw_classification_metrics(
        rows,
        bundle,
        threshold,
        objective_mode=objective_mode,
        trade_score_name=trade_score_name,
        upside_target=upside_target,
        downside_stop=downside_stop,
        fee=fee,
        slippage=slippage,
        ev_safety_margin=ev_safety_margin,
        min_predicted_net_return=min_predicted_net_return,
        hybrid_min_score=hybrid_min_score,
        hybrid_runtime_args=hybrid_runtime_args,
    )
    execution = portfolio_execution(
        rows,
        bundle,
        threshold,
        fee,
        slippage,
        initial_capital,
        max_position_fraction,
        max_volume_fraction,
        max_trades_per_period,
        trade_period_minutes,
        holding_period_minutes,
        threshold_objective,
        trade_selection,
        top_k_per_minute,
        upside_target,
        downside_stop,
        ev_safety_margin,
        objective_mode,
        trade_score_name,
        min_predicted_net_return,
        hybrid_min_score,
        max_trades_per_day,
        max_trades_per_fold,
        max_losing_trades_per_day,
        max_daily_drawdown,
        pause_after_drawdown_minutes,
        False,
        hybrid_runtime_args,
        symbol_filter_info,
    )

    for local_index, position_size in execution["executed"].items():
        if rows.indices is None:
            position = local_index
        else:
            position = int(rows.indices[local_index])
        label = int(table.labels[position])
        predicted_trades += 1
        if label == 1:
            tp += 1
        else:
            fp += 1
        expected_value = execution["executed_expected_values"].get(local_index, 0.0)

        forward_return = float(table.forward_returns[position])
        trade_return = float(table.trade_returns[position])
        max_future_high_return = float(table.max_future_high_returns[position])
        max_future_low_return = float(table.max_future_low_returns[position])
        after_fee = trade_return - fee
        after_fee_slippage = trade_return - fee - slippage
        total_fee += after_fee
        total_fee_slippage += after_fee_slippage
        sum_return += forward_return
        sum_trade_return += trade_return
        sum_expected_value += expected_value
        sum_mfe += max_future_high_return
        sum_mae += max_future_low_return
        returns.append(forward_return)
        trade_returns.append(trade_return)
        expected_values.append(expected_value)
        if trade_return > 0.0:
            winning_trades += 1
        if after_fee_slippage >= 0.0:
            gross_profit += after_fee_slippage
        else:
            gross_loss += -after_fee_slippage
        equity += after_fee_slippage
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    total = len(rows)
    fn = actual_positive - tp
    tn = total - actual_positive - fp
    precision = float(tp) / predicted_trades if predicted_trades else 0.0
    recall = float(tp) / actual_positive if actual_positive else 0.0
    accuracy = float(tp + tn) / total if total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    trade_count = float(predicted_trades) if predicted_trades else 1.0

    metrics = {
        "rows": total,
        "actual_positive_rows": actual_positive,
        "predicted_trades": predicted_trades,
        "true_positive_rows": tp,
        "false_positive_rows": fp,
        "true_negative_rows": tn,
        "false_negative_rows": fn,
        "auc": auc_score_from_rows(bundle["calibrated_probability"] if bundle.get("calibrated_probability") is not None else bundle.get("probability", []), rows) if compute_auc and bundle.get("probability") is not None else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "win_rate": float(winning_trades) / predicted_trades if predicted_trades else 0.0,
        "average_forward_return": sum_return / trade_count if predicted_trades else 0.0,
        "median_forward_return": median(returns),
        "average_trade_return": sum_trade_return / trade_count if predicted_trades else 0.0,
        "median_trade_return": median(trade_returns),
        "average_expected_value": sum_expected_value / trade_count if predicted_trades else 0.0,
        "median_expected_value": median(expected_values),
        "min_expected_value": min(expected_values) if expected_values else 0.0,
        "selected_ev_safety_margin": ev_safety_margin,
        "average_max_favorable_excursion": sum_mfe / trade_count if predicted_trades else 0.0,
        "average_max_adverse_excursion": sum_mae / trade_count if predicted_trades else 0.0,
        "average_profit_after_fee": total_fee / trade_count if predicted_trades else 0.0,
        "average_profit_after_fee_and_slippage": total_fee_slippage / trade_count if predicted_trades else 0.0,
        "total_profit_after_fee": total_fee,
        "total_profit_after_fee_and_slippage": total_fee_slippage,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "max_drawdown": max_drawdown,
        "initial_capital": initial_capital,
        "ending_capital": execution["ending_capital"],
        "portfolio_profit": execution["portfolio_profit"],
        "portfolio_return": execution["portfolio_return"],
        "average_position_size": execution["average_position_size"],
        "median_position_size": execution["median_position_size"],
        "average_profit_per_trade": execution["average_profit_per_trade"],
        "worst_trade": execution["worst_trade"],
        "max_capital_drawdown": execution["max_capital_drawdown"],
        "selected_threshold": threshold,
        "daily_trade_limit_blocked": execution["daily_trade_limit_blocked"],
        "fold_trade_limit_blocked": execution["fold_trade_limit_blocked"],
        "daily_loss_limit_blocked": execution["daily_loss_limit_blocked"],
        "daily_drawdown_limit_blocked": execution["daily_drawdown_limit_blocked"],
        "blocked_trades_total": execution["blocked_trades_total"],
        "blocked_by_trade_frequency": execution["blocked_by_trade_frequency"],
        "blocked_by_drawdown": execution["blocked_by_drawdown"],
        "max_trades_in_any_day": execution["max_trades_in_any_day"],
        "max_trades_in_any_fold": execution["max_trades_in_any_fold"],
        "trades_per_active_day": execution["trades_per_active_day"],
        "active_days": execution["active_days"],
        "profitable_days": execution["profitable_days"],
        "losing_days": execution["losing_days"],
        "worst_day_profit": execution["worst_day_profit"],
        "best_day_profit": execution["best_day_profit"],
        "normalized_microsecond_open_times": NORMALIZED_MICROSECOND_OPEN_TIMES,
    }
    metrics.update(raw_metrics)
    metrics.update(execution_frequency_metrics(execution))
    return metrics


def evaluate(rows, predictions, threshold, fee, slippage, compute_auc=True, initial_capital=10000.0,
             max_position_fraction=0.10, max_volume_fraction=0.01,
             max_trades_per_period=10, trade_period_minutes=60,
                     holding_period_minutes=5, threshold_objective="avg_profit",
                     trade_selection="threshold", top_k_per_minute=3,
             upside_target=0.05, downside_stop=0.02, ev_safety_margin=0.0,
             objective_mode="classification", trade_score_name="probability",
             min_predicted_net_return=0.0, hybrid_min_score=0.0,
             max_trades_per_day=0, max_trades_per_fold=0,
             max_losing_trades_per_day=0, max_daily_drawdown=0.0,
             pause_after_drawdown_minutes=0, hybrid_runtime_args=None,
             symbol_filter_info=None):
    if is_compact_rows(rows):
        return evaluate_compact(
            rows,
            predictions,
            threshold,
            fee,
            slippage,
            compute_auc,
            initial_capital,
            max_position_fraction,
            max_volume_fraction,
            max_trades_per_period,
            trade_period_minutes,
            holding_period_minutes,
            threshold_objective,
            trade_selection,
            top_k_per_minute,
            upside_target,
            downside_stop,
            ev_safety_margin,
            objective_mode,
            trade_score_name,
            min_predicted_net_return,
            hybrid_min_score,
            max_trades_per_day,
            max_trades_per_fold,
            max_losing_trades_per_day,
            max_daily_drawdown,
            pause_after_drawdown_minutes,
            hybrid_runtime_args,
            symbol_filter_info,
        )

    bundle = normalize_prediction_bundle(predictions)
    actual_positive = sum(row.label for row in rows)
    predicted_trades = 0
    tp = fp = tn = fn = 0
    returns = []
    trade_returns = []
    expected_values = []
    sum_return = 0.0
    sum_trade_return = 0.0
    sum_expected_value = 0.0
    sum_mfe = 0.0
    sum_mae = 0.0
    total_fee = 0.0
    total_fee_slippage = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    winning_trades = 0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    raw_metrics = raw_classification_metrics(
        rows,
        bundle,
        threshold,
        objective_mode=objective_mode,
        trade_score_name=trade_score_name,
        upside_target=upside_target,
        downside_stop=downside_stop,
        fee=fee,
        slippage=slippage,
        ev_safety_margin=ev_safety_margin,
        min_predicted_net_return=min_predicted_net_return,
        hybrid_min_score=hybrid_min_score,
        hybrid_runtime_args=hybrid_runtime_args,
    )
    execution = portfolio_execution(
        rows,
        bundle,
        threshold,
        fee,
        slippage,
        initial_capital,
        max_position_fraction,
        max_volume_fraction,
        max_trades_per_period,
        trade_period_minutes,
        holding_period_minutes,
        threshold_objective,
        trade_selection,
        top_k_per_minute,
        upside_target,
        downside_stop,
        ev_safety_margin,
        objective_mode,
        trade_score_name,
        min_predicted_net_return,
        hybrid_min_score,
        max_trades_per_day,
        max_trades_per_fold,
        max_losing_trades_per_day,
        max_daily_drawdown,
        pause_after_drawdown_minutes,
        False,
        hybrid_runtime_args,
        symbol_filter_info,
    )
    for index, position_size in execution["executed"].items():
        del position_size
        row = rows[index]
        predicted_trades += 1
        if row.label == 1:
            tp += 1
        else:
            fp += 1

        expected_value = execution["executed_expected_values"].get(index, 0.0)
        after_fee = row.trade_return - fee
        after_fee_slippage = row.trade_return - fee - slippage
        total_fee += after_fee
        total_fee_slippage += after_fee_slippage
        sum_return += row.forward_return
        sum_trade_return += row.trade_return
        sum_expected_value += expected_value
        sum_mfe += row.max_future_high_return
        sum_mae += row.max_future_low_return
        returns.append(row.forward_return)
        trade_returns.append(row.trade_return)
        expected_values.append(expected_value)
        if row.trade_return > 0.0:
            winning_trades += 1
        if after_fee_slippage >= 0.0:
            gross_profit += after_fee_slippage
        else:
            gross_loss += -after_fee_slippage
        equity += after_fee_slippage
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    total = len(rows)
    fn = actual_positive - tp
    tn = total - actual_positive - fp
    precision = float(tp) / predicted_trades if predicted_trades else 0.0
    recall = float(tp) / actual_positive if actual_positive else 0.0
    accuracy = float(tp + tn) / total if total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    trade_count = float(predicted_trades) if predicted_trades else 1.0

    metrics = {
        "rows": total,
        "actual_positive_rows": actual_positive,
        "predicted_trades": predicted_trades,
        "true_positive_rows": tp,
        "false_positive_rows": fp,
        "true_negative_rows": tn,
        "false_negative_rows": fn,
        "auc": auc_score_from_rows(bundle["calibrated_probability"] if bundle.get("calibrated_probability") is not None else bundle.get("probability", []), rows) if compute_auc and bundle.get("probability") is not None else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "win_rate": float(winning_trades) / predicted_trades if predicted_trades else 0.0,
        "average_forward_return": sum_return / trade_count if predicted_trades else 0.0,
        "median_forward_return": median(returns),
        "average_trade_return": sum_trade_return / trade_count if predicted_trades else 0.0,
        "median_trade_return": median(trade_returns),
        "average_expected_value": sum_expected_value / trade_count if predicted_trades else 0.0,
        "median_expected_value": median(expected_values),
        "min_expected_value": min(expected_values) if expected_values else 0.0,
        "selected_ev_safety_margin": ev_safety_margin,
        "average_max_favorable_excursion": sum_mfe / trade_count if predicted_trades else 0.0,
        "average_max_adverse_excursion": sum_mae / trade_count if predicted_trades else 0.0,
        "average_profit_after_fee": total_fee / trade_count if predicted_trades else 0.0,
        "average_profit_after_fee_and_slippage": total_fee_slippage / trade_count if predicted_trades else 0.0,
        "total_profit_after_fee": total_fee,
        "total_profit_after_fee_and_slippage": total_fee_slippage,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "max_drawdown": max_drawdown,
        "initial_capital": initial_capital,
        "ending_capital": execution["ending_capital"],
        "portfolio_profit": execution["portfolio_profit"],
        "portfolio_return": execution["portfolio_return"],
        "average_position_size": execution["average_position_size"],
        "median_position_size": execution["median_position_size"],
        "average_profit_per_trade": execution["average_profit_per_trade"],
        "worst_trade": execution["worst_trade"],
        "max_capital_drawdown": execution["max_capital_drawdown"],
        "selected_threshold": threshold,
        "daily_trade_limit_blocked": execution["daily_trade_limit_blocked"],
        "fold_trade_limit_blocked": execution["fold_trade_limit_blocked"],
        "daily_loss_limit_blocked": execution["daily_loss_limit_blocked"],
        "daily_drawdown_limit_blocked": execution["daily_drawdown_limit_blocked"],
        "blocked_trades_total": execution["blocked_trades_total"],
        "blocked_by_trade_frequency": execution["blocked_by_trade_frequency"],
        "blocked_by_drawdown": execution["blocked_by_drawdown"],
        "max_trades_in_any_day": execution["max_trades_in_any_day"],
        "max_trades_in_any_fold": execution["max_trades_in_any_fold"],
        "trades_per_active_day": execution["trades_per_active_day"],
        "active_days": execution["active_days"],
        "profitable_days": execution["profitable_days"],
        "losing_days": execution["losing_days"],
        "worst_day_profit": execution["worst_day_profit"],
        "best_day_profit": execution["best_day_profit"],
        "normalized_microsecond_open_times": NORMALIZED_MICROSECOND_OPEN_TIMES,
    }
    metrics.update(raw_metrics)
    metrics.update(execution_frequency_metrics(execution))
    return metrics


def fit_symbol_validation_filter(rows, predictions, threshold, args, trade_score_name):
    mode = getattr(args, "symbol_validation_filter", "none")
    if mode == "none":
        return {
            "mode": "none",
            "enabled": False,
            "allowed_symbols": [],
            "total_symbols": 0,
            "filtered_symbols": [],
            "symbol_stats": {},
            "diagnostics": [],
        }
    bundle = normalize_prediction_bundle(predictions)
    validation_slippage = args.slippage * args.validation_slippage_multiplier
    execution = portfolio_execution(
        rows,
        bundle,
        threshold,
        args.fee,
        validation_slippage,
        args.initial_capital,
        args.max_position_fraction,
        args.max_volume_fraction,
        args.max_trades_per_period,
        args.trade_period_minutes,
        args.holding_period_minutes,
        args.threshold_objective,
        args.trade_selection,
        args.top_k_per_minute,
        args.upside_target,
        args.downside_stop,
        args.ev_safety_margin,
        args.objective_mode,
        trade_score_name,
        args.min_predicted_net_return,
        args.hybrid_min_score,
        args.max_trades_per_day,
        0,
        args.max_losing_trades_per_day,
        args.max_daily_drawdown,
        args.pause_after_drawdown_minutes,
        False,
        args,
        None,
    )
    stage = getattr(args, "symbol_filter_stage", "executed")
    min_candidates = max(1, int(getattr(args, "symbol_filter_min_candidates", 25)))
    min_executed = max(1, int(getattr(args, "symbol_filter_min_executed", max(1, getattr(args, "min_symbol_validation_trades", 1)))))
    candidate_weight = float(getattr(args, "symbol_filter_candidate_weight", 0.5))
    executed_weight = float(getattr(args, "symbol_filter_executed_weight", 0.5))
    shrinkage = max(0.0, float(getattr(args, "symbol_filter_shrinkage", 50.0)))
    trade_returns = actual_trade_returns(rows)
    labels = rows.labels_array() if is_compact_rows(rows) else [row.label for row in rows]
    symbol_stats = {}

    def stats_for(symbol):
        return symbol_stats.setdefault(symbol, {
            "raw_candidate_count": 0,
            "raw_candidate_score_sum": 0.0,
            "raw_candidate_probability_sum": 0.0,
            "raw_candidate_predicted_return_sum": 0.0,
            "eligible_candidate_count": 0,
            "eligible_candidate_score_sum": 0.0,
            "eligible_candidate_probability_sum": 0.0,
            "eligible_candidate_predicted_return_sum": 0.0,
            "eligible_candidate_realized_return_sum": 0.0,
            "eligible_candidate_positive_sum": 0.0,
            "executed_trade_count": 0,
            "executed_profit": 0.0,
            "executed_net_return_sum": 0.0,
            "executed_win_count": 0,
        })

    global_candidate_sum = 0.0
    global_candidate_count = 0
    global_executed_sum = 0.0
    global_executed_count = 0
    hybrid_score_mode = getattr(args, "hybrid_score_mode", "basic")
    hybrid_uncertainty_penalty = getattr(args, "hybrid_uncertainty_penalty", 0.0)

    for local_index in range(len(rows)):
        symbol = row_symbol_name(rows, int(local_index))
        stats = stats_for(symbol)
        stats["raw_candidate_count"] += 1
        stats["raw_candidate_score_sum"] += float(
            trade_score_value(
                bundle,
                local_index,
                trade_score_name,
                args.upside_target,
                args.downside_stop,
                args.fee,
                validation_slippage,
                hybrid_score_mode,
                hybrid_uncertainty_penalty,
                args,
            )
        )
        stats["raw_candidate_probability_sum"] += float(calibrated_probability_value(bundle, local_index))
        stats["raw_candidate_predicted_return_sum"] += float(predicted_trade_return_value(bundle, local_index))

    for local_index in execution["raw_selected"].keys():
        local_index = int(local_index)
        symbol = row_symbol_name(rows, local_index)
        stats = stats_for(symbol)
        realized_return = float(trade_returns[local_index]) - args.fee - validation_slippage
        stats["eligible_candidate_count"] += 1
        stats["eligible_candidate_score_sum"] += float(
            trade_score_value(
                bundle,
                local_index,
                trade_score_name,
                args.upside_target,
                args.downside_stop,
                args.fee,
                validation_slippage,
                hybrid_score_mode,
                hybrid_uncertainty_penalty,
                args,
            )
        )
        stats["eligible_candidate_probability_sum"] += float(calibrated_probability_value(bundle, local_index))
        stats["eligible_candidate_predicted_return_sum"] += float(predicted_trade_return_value(bundle, local_index))
        stats["eligible_candidate_realized_return_sum"] += realized_return
        stats["eligible_candidate_positive_sum"] += float(labels[local_index])
        global_candidate_sum += realized_return
        global_candidate_count += 1

    for local_index, pnl in execution["executed_pnls"].items():
        local_index = int(local_index)
        symbol = row_symbol_name(rows, local_index)
        stats = stats_for(symbol)
        position_size = float(execution["executed"].get(local_index, 0.0))
        net_return = float(pnl) / position_size if position_size > 0.0 else 0.0
        stats["executed_trade_count"] += 1
        stats["executed_profit"] += float(pnl)
        stats["executed_net_return_sum"] += net_return
        if float(pnl) > 0.0:
            stats["executed_win_count"] += 1
        global_executed_sum += net_return
        global_executed_count += 1

    global_quality = (
        global_candidate_sum / float(global_candidate_count)
        if global_candidate_count > 0 else (
            global_executed_sum / float(global_executed_count)
            if global_executed_count > 0 else 0.0
        )
    )
    allowed_symbols = []
    filtered_symbols = []
    min_trades = max(1, int(getattr(args, "min_symbol_validation_trades", 1)))
    min_avg_profit = float(getattr(args, "min_symbol_validation_average_profit", 0.0))
    min_total_profit = float(getattr(args, "min_symbol_validation_total_profit", 0.0))
    diagnostics = []
    for symbol, stats in sorted(symbol_stats.items()):
        raw_count = int(stats["raw_candidate_count"])
        eligible_count = int(stats["eligible_candidate_count"])
        executed_count = int(stats["executed_trade_count"])
        raw_avg_score = float(stats["raw_candidate_score_sum"]) / float(raw_count) if raw_count else 0.0
        raw_avg_probability = float(stats["raw_candidate_probability_sum"]) / float(raw_count) if raw_count else 0.0
        raw_avg_predicted_return = float(stats["raw_candidate_predicted_return_sum"]) / float(raw_count) if raw_count else 0.0
        eligible_avg_score = float(stats["eligible_candidate_score_sum"]) / float(eligible_count) if eligible_count else 0.0
        eligible_avg_probability = float(stats["eligible_candidate_probability_sum"]) / float(eligible_count) if eligible_count else 0.0
        eligible_avg_predicted_return = float(stats["eligible_candidate_predicted_return_sum"]) / float(eligible_count) if eligible_count else 0.0
        eligible_avg_realized_return = float(stats["eligible_candidate_realized_return_sum"]) / float(eligible_count) if eligible_count else 0.0
        eligible_positive_rate = float(stats["eligible_candidate_positive_sum"]) / float(eligible_count) if eligible_count else 0.0
        executed_avg_profit = float(stats["executed_profit"]) / float(executed_count) if executed_count else 0.0
        executed_avg_return = float(stats["executed_net_return_sum"]) / float(executed_count) if executed_count else 0.0
        executed_win_rate = float(stats["executed_win_count"]) / float(executed_count) if executed_count else 0.0
        stats["average_profit_per_trade"] = executed_avg_profit
        stats["win_rate"] = executed_win_rate
        stats["candidate_quality"] = eligible_avg_realized_return
        stats["executed_quality"] = executed_avg_return

        allowed = True
        reason = "insufficient_evidence"
        symbol_score = 0.0
        decision_metric = 0.0
        eligible_ready = eligible_count >= min_candidates
        executed_ready = executed_count >= max(min_trades, min_executed)

        if mode not in ("positive_avg_profit", "positive_total_profit"):
            raise ValueError("unknown --symbol-validation-filter mode: {}".format(mode))

        if stage == "executed":
            if executed_ready:
                decision_metric = executed_avg_profit if mode == "positive_avg_profit" else float(stats["executed_profit"])
                allowed = decision_metric > (min_avg_profit if mode == "positive_avg_profit" else min_total_profit)
                reason = "executed_metric"
            else:
                allowed = True
                reason = "insufficient_executed_evidence"
        elif stage == "eligible":
            if eligible_ready:
                decision_metric = eligible_avg_realized_return if mode == "positive_avg_profit" else float(stats["eligible_candidate_realized_return_sum"])
                allowed = decision_metric > (min_avg_profit if mode == "positive_avg_profit" else min_total_profit)
                reason = "eligible_metric"
            else:
                allowed = True
                reason = "insufficient_candidate_evidence"
        else:
            candidate_available = eligible_ready
            executed_available = executed_ready
            if candidate_available or executed_available:
                weighted_score = 0.0
                weight_total = 0.0
                if candidate_available:
                    weighted_score += candidate_weight * eligible_avg_realized_return
                    weight_total += candidate_weight
                if executed_available:
                    weighted_score += executed_weight * executed_avg_return
                    weight_total += executed_weight
                if weight_total <= 0.0:
                    blended_raw = eligible_avg_realized_return if candidate_available else executed_avg_return
                else:
                    blended_raw = weighted_score / weight_total
                # Executed trades are a subset of eligible candidates, so adding both
                # double-counts the same evidence and overstates symbol confidence.
                support = max(eligible_count, executed_count)
                effective_weight = float(support) / float(support + shrinkage) if shrinkage > 0.0 else 1.0
                symbol_score = effective_weight * blended_raw + (1.0 - effective_weight) * global_quality
                decision_metric = symbol_score if mode == "positive_avg_profit" else symbol_score * float(max(1, support))
                allowed = decision_metric > (min_avg_profit if mode == "positive_avg_profit" else min_total_profit)
                if candidate_available and executed_available:
                    reason = "candidate_blend"
                elif candidate_available:
                    reason = "candidate_only"
                else:
                    reason = "executed_only"
            else:
                allowed = True
                reason = "insufficient_blended_evidence"

        if allowed:
            allowed_symbols.append(symbol)
        else:
            filtered_symbols.append(symbol)
        diagnostics.append({
            "symbol": symbol,
            "raw_candidate_count": raw_count,
            "eligible_candidate_count": eligible_count,
            "executed_trade_count": executed_count,
            "raw_candidate_avg_score": raw_avg_score,
            "raw_candidate_avg_probability": raw_avg_probability,
            "raw_candidate_avg_predicted_return": raw_avg_predicted_return,
            "eligible_candidate_avg_score": eligible_avg_score,
            "eligible_candidate_avg_probability": eligible_avg_probability,
            "eligible_candidate_avg_predicted_return": eligible_avg_predicted_return,
            "eligible_candidate_realized_avg_return": eligible_avg_realized_return,
            "eligible_candidate_positive_rate": eligible_positive_rate,
            "executed_avg_profit": executed_avg_profit,
            "executed_avg_return": executed_avg_return,
            "executed_win_rate": executed_win_rate,
            "candidate_quality": eligible_avg_realized_return,
            "executed_quality": executed_avg_return,
            "symbol_score": symbol_score if stage == "candidate_blend" else decision_metric,
            "global_quality": global_quality,
            "symbol_filter_stage": stage,
            "symbol_filter_decision": "allowed" if allowed else "blocked",
            "symbol_filter_reason": reason,
        })
    enabled = bool(allowed_symbols) and len(filtered_symbols) > 0
    return {
        "mode": mode,
        "stage": stage,
        "enabled": enabled,
        "allowed_symbols": allowed_symbols if enabled else [],
        "total_symbols": len(symbol_stats),
        "filtered_symbols": filtered_symbols if enabled else [],
        "symbol_stats": symbol_stats,
        "diagnostics": diagnostics,
        "symbols_allowed_count": len(allowed_symbols) if enabled else len(symbol_stats),
        "symbols_blocked_count": len(filtered_symbols) if enabled else 0,
        "symbol_filter_min_candidates": min_candidates,
        "symbol_filter_min_executed": min_executed,
        "symbol_filter_candidate_weight": candidate_weight,
        "symbol_filter_executed_weight": executed_weight,
        "symbol_filter_shrinkage": shrinkage,
    }


def disabled_symbol_filter_info(symbol_filter_info, reason, extra=None):
    if symbol_filter_info is None:
        return None
    result = dict(symbol_filter_info)
    result["enabled"] = False
    result["disabled_reason"] = reason
    if extra:
        result.update(extra)
    return result


def recalibrate_symbol_filter_validation(rows, bundle, threshold, args, selection,
                                         symbol_filter_info, selected_score_name):
    if not symbol_filter_info or not symbol_filter_info.get("enabled"):
        return symbol_filter_info, selection
    baseline_metrics = selection.get("validation_metrics", {})
    baseline_score = float(
        selection.get(
            "objective_score",
            baseline_metrics.get("selected_objective_score", -float("inf")),
        )
    )
    baseline_trade_count = int(baseline_metrics.get("predicted_trades", 0))
    filtered_selection = build_selected_threshold_result(
        threshold,
        evaluate(
            rows,
            bundle,
            threshold,
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
            trade_score_name=score_name_for_args(args),
            min_predicted_net_return=args.min_predicted_net_return,
            hybrid_min_score=args.hybrid_min_score,
            max_trades_per_day=args.max_trades_per_day,
            max_trades_per_fold=0,
            max_losing_trades_per_day=args.max_losing_trades_per_day,
            max_daily_drawdown=args.max_daily_drawdown,
            pause_after_drawdown_minutes=args.pause_after_drawdown_minutes,
            hybrid_runtime_args=args,
            symbol_filter_info=symbol_filter_info,
        ),
        args.threshold_objective,
        0.0 if args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev") and args.profit_safety == "strict" else -float("inf"),
        args.threshold_drawdown_penalty,
        args.threshold_trade_count_penalty,
        args.target_validation_trades,
        selected_score_name,
        args.min_validation_trades,
        args.max_validation_trades,
    )
    trade_count = int(filtered_selection["validation_metrics"].get("predicted_trades", 0))
    precision = float(filtered_selection["validation_metrics"].get("precision", 0.0))
    filter_info = dict(symbol_filter_info)
    filter_info["validation_trade_count"] = trade_count
    filter_info["validation_trade_retention"] = float(trade_count) / float(baseline_trade_count) if baseline_trade_count > 0 else 0.0
    filter_info["validation_portfolio_profit"] = float(filtered_selection["validation_metrics"].get("portfolio_profit", 0.0))
    filter_info["validation_objective_score"] = float(filtered_selection.get("objective_score", baseline_score))
    if trade_count < int(args.min_validation_trades):
        return disabled_symbol_filter_info(filter_info, "validation_under_min_trades"), selection
    if args.max_validation_trades > 0 and trade_count > int(args.max_validation_trades):
        return disabled_symbol_filter_info(filter_info, "validation_over_max_trades"), selection
    if precision + 1e-12 < float(args.min_validation_precision):
        return disabled_symbol_filter_info(filter_info, "validation_under_min_precision"), selection
    if float(filtered_selection.get("objective_score", -float("inf"))) + 1e-12 < baseline_score:
        return disabled_symbol_filter_info(filter_info, "validation_underperformed"), selection
    return filter_info, filtered_selection


def inactive_fold_blocker_check(rows, predictions, threshold, metrics, validation_metrics, args, trade_score_name):
    result = {
        "inactive_blocker_source": "active",
        "inactive_blocker_metric": "",
        "inactive_blocker_threshold": 0.0,
        "inactive_blocker_best_score": 0.0,
        "inactive_blocker_gap": 0.0,
        "inactive_closest_symbol": "",
        "inactive_promising_fold": 0,
    }
    predicted_trades = int(metrics.get("predicted_trades", 0))
    raw_signal_trades = int(metrics.get("raw_signal_trades", 0))
    validation_trade_count = int(validation_metrics.get("selected_validation_trade_count", 0))
    validation_profit = float(validation_metrics.get("selected_validation_portfolio_profit", 0.0))
    if validation_trade_count > 0 and validation_profit > 0.0:
        result["inactive_promising_fold"] = 1
    if predicted_trades > 0:
        return result
    if raw_signal_trades > 0:
        result["inactive_blocker_source"] = "execution_constraints_or_filters"
        result["inactive_blocker_metric"] = "raw_signal_blocked_after_selection"
        return result

    bundle = normalize_prediction_bundle(predictions)
    ev_context = resolve_ev_context(bundle, args, args.upside_target, args.downside_stop)
    hybrid_context = resolve_hybrid_return_context(bundle, args, args.upside_target, args.downside_stop)
    selected_threshold = float(validation_metrics.get("selected_score_threshold", threshold) or threshold)
    no_trade_fallback = selected_threshold >= 1.01 - 1e-12 and validation_trade_count <= 0

    def finalize(source, metric_name, required_threshold, best_score, local_index):
        result["inactive_blocker_source"] = source
        result["inactive_blocker_metric"] = metric_name
        result["inactive_blocker_threshold"] = float(required_threshold)
        result["inactive_blocker_best_score"] = float(best_score)
        result["inactive_blocker_gap"] = float(required_threshold) - float(best_score)
        if local_index is not None and 0 <= int(local_index) < len(rows):
            result["inactive_closest_symbol"] = row_symbol_name(rows, int(local_index))
        return result

    if args.objective_mode == "classification":
        probabilities = maybe_float32_array(bundle.get("calibrated_probability"))
        if probabilities is None:
            probabilities = maybe_float32_array(bundle.get("probability"))
        if probabilities is None or len(probabilities) == 0:
            return finalize("missing_predictions", "probability", selected_threshold, 0.0, None)
        if np is not None and isinstance(probabilities, np.ndarray):
            max_prob_index = int(np.argmax(probabilities))
            max_probability = float(probabilities[max_prob_index])
            if trade_score_name == "ev":
                above_probability = probabilities >= selected_threshold
                if np.any(above_probability):
                    candidate_indices = np.nonzero(above_probability)[0].astype(np.int64, copy=False)
                    ev_scores = score_batch_ev(
                        probabilities[candidate_indices],
                        args.upside_target,
                        args.downside_stop,
                        args.fee,
                        args.slippage * args.validation_slippage_multiplier,
                        predicted_trade_returns=maybe_float32_array(bundle.get("predicted_trade_return"))[candidate_indices] if bundle.get("predicted_trade_return") is not None else None,
                        objective_mode=args.objective_mode,
                        ev_context=ev_context,
                    )
                    best_ev_offset = int(np.argmax(ev_scores))
                    best_ev_index = int(candidate_indices[best_ev_offset])
                    best_ev_score = float(ev_scores[best_ev_offset])
                    if best_ev_score <= args.ev_safety_margin:
                        return finalize(
                            "validation_no_trade_fallback" if no_trade_fallback else "test_threshold",
                            "expected_value",
                            args.ev_safety_margin,
                            best_ev_score,
                            best_ev_index,
                        )
            return finalize(
                "validation_no_trade_fallback" if no_trade_fallback else "test_threshold",
                "probability",
                selected_threshold,
                max_probability,
                max_prob_index,
            )
        max_probability = -float("inf")
        max_prob_index = None
        for index in range(len(probabilities)):
            if float(probabilities[index]) > max_probability:
                max_probability = float(probabilities[index])
                max_prob_index = index
        if trade_score_name == "ev":
            best_ev_score = -float("inf")
            best_ev_index = None
            for index in range(len(probabilities)):
                probability = float(probabilities[index])
                if probability < selected_threshold:
                    continue
                ev_score = expected_value_from_probability(
                    probability,
                    float(ev_context.get("ev_expected_win_return", args.upside_target)),
                    float(-ev_context.get("ev_expected_loss_return", -args.downside_stop)),
                    args.fee,
                    args.slippage * args.validation_slippage_multiplier,
                )
                if ev_score > best_ev_score:
                    best_ev_score = ev_score
                    best_ev_index = index
            if best_ev_index is not None and best_ev_score <= args.ev_safety_margin:
                return finalize(
                    "validation_no_trade_fallback" if no_trade_fallback else "test_threshold",
                    "expected_value",
                    args.ev_safety_margin,
                    best_ev_score,
                    best_ev_index,
                )
        return finalize(
            "validation_no_trade_fallback" if no_trade_fallback else "test_threshold",
            "probability",
            selected_threshold,
            max_probability,
            max_prob_index,
        )

    if args.objective_mode == "return_regression":
        predicted_returns = maybe_float32_array(bundle.get("predicted_trade_return"))
        if predicted_returns is None or len(predicted_returns) == 0:
            return finalize("missing_predictions", "predicted_net_return", max(selected_threshold, args.min_predicted_net_return), 0.0, None)
        if np is not None and isinstance(predicted_returns, np.ndarray):
            net_scores = predicted_returns.astype(np.float32, copy=False) - np.float32(args.fee + args.slippage * args.validation_slippage_multiplier)
            best_index = int(np.argmax(net_scores))
            best_score = float(net_scores[best_index])
        else:
            best_score = -float("inf")
            best_index = None
            for index in range(len(predicted_returns)):
                score = float(predicted_returns[index]) - args.fee - args.slippage * args.validation_slippage_multiplier
                if score > best_score:
                    best_score = score
                    best_index = index
        return finalize(
            "validation_no_trade_fallback" if no_trade_fallback else "test_threshold",
            "predicted_net_return",
            max(selected_threshold, args.min_predicted_net_return),
            best_score,
            best_index,
        )

    probabilities = maybe_float32_array(bundle.get("calibrated_probability"))
    if probabilities is None:
        probabilities = maybe_float32_array(bundle.get("probability"))
    predicted_returns = maybe_float32_array(bundle.get("predicted_trade_return"))
    if predicted_returns is None:
        predicted_returns = maybe_float32_array(bundle.get("raw_predicted_trade_return"))
    predicted_uncertainty = maybe_float32_array(bundle.get("predicted_return_uncertainty"))
    hybrid_requires_predicted_return = hybrid_context.get("hybrid_return_combination", "probability_times_return") != "conditional_payoff"
    if probabilities is None or (predicted_returns is None and hybrid_requires_predicted_return) or len(probabilities) == 0:
        return finalize("missing_predictions", "hybrid_score", max(selected_threshold, args.hybrid_min_score), 0.0, None)
    scores = score_batch_hybrid(
        probabilities,
        predicted_returns,
        args.fee,
        args.slippage * args.validation_slippage_multiplier,
        predicted_uncertainty,
        args.hybrid_score_mode,
        args.hybrid_uncertainty_penalty,
        hybrid_context=hybrid_context,
        uncertainty_context=bundle.get("uncertainty_context"),
        hybrid_runtime_args=args,
    )
    effective_thresholds, _ = compute_dynamic_hybrid_thresholds(
        rows,
        args,
        max(float(selected_threshold), float(args.hybrid_min_score)),
    )
    if np is not None and isinstance(scores, np.ndarray):
        if effective_thresholds is None:
            threshold_values = np.full(len(scores), np.float32(max(float(selected_threshold), float(args.hybrid_min_score))), dtype=np.float32)
        else:
            threshold_values = np.asarray(effective_thresholds, dtype=np.float32)
        hybrid_probability_gate = hybrid_probability_gate_threshold(bundle, args, args.upside_target, args.downside_stop)
        if hybrid_probability_gate > 0.0:
            score_pass_mask = scores.astype(np.float32, copy=False) >= threshold_values
            gate_pass_mask = probabilities.astype(np.float32, copy=False) >= np.float32(hybrid_probability_gate)
            if np.any(score_pass_mask) and not np.any(score_pass_mask & gate_pass_mask):
                candidate_indices = np.nonzero(score_pass_mask)[0].astype(np.int64, copy=False)
                best_prob_offset = int(np.argmax(probabilities[candidate_indices]))
                best_prob_index = int(candidate_indices[best_prob_offset])
                return finalize(
                    "validation_no_trade_fallback" if no_trade_fallback else "test_threshold",
                    "calibrated_probability",
                    hybrid_probability_gate,
                    float(probabilities[best_prob_index]),
                    best_prob_index,
                )
        gaps = threshold_values - scores.astype(np.float32, copy=False)
        best_index = int(np.argmin(gaps))
        best_score = float(scores[best_index])
        required_threshold = float(threshold_values[best_index])
    else:
        threshold_values = effective_thresholds if effective_thresholds is not None else [max(float(selected_threshold), float(args.hybrid_min_score))] * len(scores)
        hybrid_probability_gate = hybrid_probability_gate_threshold(bundle, args, args.upside_target, args.downside_stop)
        if hybrid_probability_gate > 0.0:
            best_prob_index = None
            best_probability = -float("inf")
            any_score_pass = False
            any_gate_pass = False
            for index, score in enumerate(scores):
                if float(score) >= float(threshold_values[index]):
                    any_score_pass = True
                    probability = float(probabilities[index])
                    if probability >= hybrid_probability_gate:
                        any_gate_pass = True
                    if probability > best_probability:
                        best_probability = probability
                        best_prob_index = index
            if any_score_pass and not any_gate_pass:
                return finalize(
                    "validation_no_trade_fallback" if no_trade_fallback else "test_threshold",
                    "calibrated_probability",
                    hybrid_probability_gate,
                    best_probability,
                    best_prob_index,
                )
        best_index = None
        best_gap = float("inf")
        best_score = 0.0
        required_threshold = max(float(selected_threshold), float(args.hybrid_min_score))
        for index, score in enumerate(scores):
            gap = float(threshold_values[index]) - float(score)
            if gap < best_gap:
                best_gap = gap
                best_index = index
                best_score = float(score)
                required_threshold = float(threshold_values[index])
    return finalize(
        "validation_no_trade_fallback" if no_trade_fallback else "test_threshold",
        "hybrid_score",
        required_threshold,
        best_score,
        best_index,
    )


def resolved_profit_balance_target(target_validation_trades=0,
                                   min_validation_trades=0,
                                   max_validation_trades=0,
                                   metrics=None):
    explicit_target = int(target_validation_trades)
    if explicit_target > 0:
        return explicit_target
    minimum = max(1, int(min_validation_trades)) if min_validation_trades > 0 else 1
    maximum = int(max_validation_trades)
    if maximum > 0:
        return max(minimum, int(round((minimum + maximum) / 2.0)))
    if minimum > 1:
        return max(minimum, minimum * 2)
    if metrics is not None:
        return max(1, int(metrics.get("predicted_trades", 0)))
    return 1


def profit_balanced_multiplier(metrics, target_validation_trades=0,
                               min_validation_trades=0, max_validation_trades=0):
    trades = int(metrics.get("predicted_trades", 0))
    if trades <= 0:
        return 0.0
    target = resolved_profit_balance_target(
        target_validation_trades,
        min_validation_trades,
        max_validation_trades,
        metrics,
    )
    coverage = min(1.0, float(trades) / float(max(1, target)))
    return math.sqrt(coverage)


def threshold_score(metrics, objective, zero_trade_profit_score=0.0,
                    target_validation_trades=0, min_validation_trades=0,
                    max_validation_trades=0):
    if metrics["predicted_trades"] == 0:
        return zero_trade_profit_score if objective in ("profit", "profit_balanced", "avg_profit", "ev") else -float("inf")
    if objective == "profit_balanced":
        return float(metrics["portfolio_profit"]) * profit_balanced_multiplier(
            metrics,
            target_validation_trades,
            min_validation_trades,
            max_validation_trades,
        )
    if objective == "avg_profit":
        return metrics["portfolio_profit"] / metrics["predicted_trades"]
    if objective == "ev":
        return float(metrics.get("average_expected_value", 0.0))
    if objective == "precision":
        return metrics["precision"]
    if objective == "recall":
        return metrics["recall"]
    if objective == "f1":
        return metrics["f1"]
    return metrics["portfolio_profit"]


def threshold_penalized_score(metrics, objective, zero_trade_profit_score=0.0,
                              drawdown_penalty=0.0, trade_count_penalty=0.0,
                              target_validation_trades=0,
                              min_validation_trades=0, max_validation_trades=0):
    base_score = threshold_score(
        metrics,
        objective,
        zero_trade_profit_score,
        target_validation_trades,
        min_validation_trades,
        max_validation_trades,
    )
    penalized_score = base_score
    if metrics["predicted_trades"] > 0:
        if drawdown_penalty > 0.0:
            penalized_score -= drawdown_penalty * abs(float(metrics.get("max_capital_drawdown", 0.0)))
        if trade_count_penalty > 0.0 and target_validation_trades > 0:
            excess_trades = max(0, int(metrics.get("predicted_trades", 0)) - int(target_validation_trades))
            penalized_score -= trade_count_penalty * excess_trades
    return base_score, penalized_score


def threshold_rank(metrics, objective, zero_trade_profit_score=0.0, penalized_score=None,
                   target_validation_trades=0, min_validation_trades=0,
                   max_validation_trades=0):
    trades = metrics["predicted_trades"]
    average_profit = metrics["portfolio_profit"] / trades if trades else 0.0
    leading_score = threshold_score(
        metrics,
        objective,
        zero_trade_profit_score,
        target_validation_trades,
        min_validation_trades,
        max_validation_trades,
    ) if penalized_score is None else penalized_score
    return (
        leading_score,
        metrics["portfolio_profit"],
        average_profit,
        metrics["precision"],
        -trades,
    )


def validation_profit_per_active_day(metrics):
    active_days = int(metrics.get("active_days", 0))
    if active_days <= 0:
        return 0.0
    return float(metrics.get("portfolio_profit", 0.0)) / float(active_days)


def resolved_threshold_target_trades(args, min_validation_trades, max_validation_trades):
    explicit_target = int(getattr(args, "threshold_target_trades", 0))
    if explicit_target > 0:
        return explicit_target
    validation_target = int(getattr(args, "target_validation_trades", 0))
    if validation_target > 0:
        return validation_target
    if max_validation_trades > 0:
        return int(round((int(min_validation_trades) + int(max_validation_trades)) / 2.0))
    return int(min_validation_trades)


def active_day_tie_value(active_days, args):
    target_active_days = int(getattr(args, "threshold_target_active_days", 0))
    if target_active_days > 0:
        return -abs(int(active_days) - target_active_days)
    return int(active_days)


def compare_threshold_results(candidate_result, best_result, args, min_validation_trades, max_validation_trades):
    if best_result is None:
        return True, "initial_selection"
    candidate_score = float(candidate_result.get("penalized_objective_score", candidate_result.get("objective_score", -float("inf"))))
    best_score = float(best_result.get("penalized_objective_score", best_result.get("objective_score", -float("inf"))))
    epsilon = float(getattr(args, "threshold_tie_epsilon", 1e-9))
    if candidate_score > best_score + epsilon:
        return True, "higher_objective_score"
    if candidate_score < best_score - epsilon:
        return False, "lower_objective_score"

    tie_breaker = getattr(args, "threshold_tiebreaker", "fewer_trades")
    target_trades = resolved_threshold_target_trades(args, min_validation_trades, max_validation_trades)
    candidate_metrics = candidate_result["validation_metrics"]
    best_metrics = best_result["validation_metrics"]
    candidate_trades = int(candidate_metrics.get("predicted_trades", 0))
    best_trades = int(best_metrics.get("predicted_trades", 0))
    candidate_active_days = int(candidate_metrics.get("active_days", 0))
    best_active_days = int(best_metrics.get("active_days", 0))
    candidate_drawdown = float(candidate_metrics.get("max_capital_drawdown", 0.0))
    best_drawdown = float(best_metrics.get("max_capital_drawdown", 0.0))
    candidate_profit_day = validation_profit_per_active_day(candidate_metrics)
    best_profit_day = validation_profit_per_active_day(best_metrics)
    candidate_gap = abs(candidate_trades - target_trades)
    best_gap = abs(best_trades - target_trades)
    candidate_active_value = active_day_tie_value(candidate_active_days, args)
    best_active_value = active_day_tie_value(best_active_days, args)

    if tie_breaker == "target_trades":
        candidate_tuple = (-candidate_gap, -candidate_drawdown, candidate_active_value, -candidate_trades)
        best_tuple = (-best_gap, -best_drawdown, best_active_value, -best_trades)
        return candidate_tuple > best_tuple, "target_trades"
    if tie_breaker == "active_days":
        candidate_tuple = (candidate_active_value, -candidate_gap, -candidate_drawdown, -candidate_trades)
        best_tuple = (best_active_value, -best_gap, -best_drawdown, -best_trades)
        return candidate_tuple > best_tuple, "active_days"
    if tie_breaker == "balanced":
        candidate_tuple = (candidate_active_value, -candidate_gap, candidate_profit_day, -candidate_drawdown, -candidate_trades)
        best_tuple = (best_active_value, -best_gap, best_profit_day, -best_drawdown, -best_trades)
        return candidate_tuple > best_tuple, "balanced"
    candidate_tuple = threshold_rank(
        candidate_metrics,
        getattr(args, "threshold_objective", "profit_balanced"),
        0.0,
        candidate_score,
        target_trades,
        min_validation_trades,
        max_validation_trades,
    )
    best_tuple = threshold_rank(
        best_metrics,
        getattr(args, "threshold_objective", "profit_balanced"),
        0.0,
        best_score,
        target_trades,
        min_validation_trades,
        max_validation_trades,
    )
    return candidate_tuple > best_tuple, "fewer_trades"


def selected_score_name_for_mode(args):
    if args.objective_mode == "classification":
        return "probability"
    if args.objective_mode == "return_regression":
        return "predicted_return"
    return "hybrid"


def annotate_selected_validation_metrics(metrics, threshold, selected_score_name,
                                         base_score, penalized_score):
    selected_trade_count = int(metrics.get("predicted_trades", 0))
    selected_active_days = int(metrics.get("active_days", 0))
    metrics["selected_threshold"] = threshold
    metrics["selected_score_name"] = selected_score_name
    metrics["selected_score_threshold"] = threshold
    metrics["selected_base_objective_score"] = base_score
    metrics["selected_objective_score"] = penalized_score
    metrics["selected_penalized_objective_score"] = penalized_score
    metrics["selected_validation_trade_count"] = selected_trade_count
    metrics["selected_validation_raw_signal_count"] = int(metrics.get("raw_signal_trades", 0))
    metrics["selected_validation_portfolio_profit"] = float(metrics.get("portfolio_profit", 0.0))
    metrics["selected_validation_portfolio_return"] = float(metrics.get("portfolio_return", 0.0))
    metrics["selected_validation_precision"] = float(metrics.get("precision", 0.0))
    metrics["selected_validation_recall"] = float(metrics.get("recall", 0.0))
    metrics["selected_validation_average_profit_after_fee_and_slippage"] = float(
        metrics.get("average_profit_after_fee_and_slippage", 0.0)
    )
    metrics["selected_validation_total_profit_after_fee_and_slippage"] = float(
        metrics.get("total_profit_after_fee_and_slippage", 0.0)
    )
    metrics["selected_validation_max_drawdown"] = float(metrics.get("max_capital_drawdown", 0.0))
    metrics["selected_validation_active_days"] = selected_active_days
    metrics["selected_validation_profit_per_active_day"] = validation_profit_per_active_day(metrics)
    return metrics


def build_selected_threshold_result(threshold, metrics, objective, zero_trade_profit_score,
                                    drawdown_penalty, trade_count_penalty,
                                    target_validation_trades, selected_score_name,
                                    min_validation_trades=0, max_validation_trades=0):
    base_score, penalized_score = threshold_penalized_score(
        metrics,
        objective,
        zero_trade_profit_score,
        drawdown_penalty,
        trade_count_penalty,
        target_validation_trades,
        min_validation_trades,
        max_validation_trades,
    )
    annotate_selected_validation_metrics(
        metrics,
        threshold,
        selected_score_name,
        base_score,
        penalized_score,
    )
    return {
        "threshold": threshold,
        "selected_threshold": threshold,
        "selected_score_threshold": threshold,
        "selected_score_name": selected_score_name,
        "base_objective_score": base_score,
        "objective_score": penalized_score,
        "penalized_objective_score": penalized_score,
        "validation_trade_count": int(metrics.get("predicted_trades", 0)),
        "validation_raw_signal_count": int(metrics.get("raw_signal_trades", 0)),
        "validation_max_drawdown": float(metrics.get("max_capital_drawdown", 0.0)),
        "validation_metrics": metrics,
        "no_trade_selected": int(metrics.get("predicted_trades", 0)) <= 0,
        "tie_rank_reason": "",
    }


def tune_threshold(rows, predictions, thresholds, objective, fee, slippage,
                   min_validation_trades, max_validation_trades, min_validation_precision,
                   profit_safety, initial_capital,
                   max_position_fraction, max_volume_fraction, max_trades_per_period,
                   trade_period_minutes, holding_period_minutes, trade_selection="threshold",
                   top_k_per_minute=3, upside_target=0.05, downside_stop=0.02,
                   ev_safety_margin=0.0, objective_mode="classification",
                   trade_score_name="probability", min_predicted_net_return=0.0,
                   hybrid_min_score=0.0, max_trades_per_day=0, max_trades_per_fold=0,
                   max_losing_trades_per_day=0, max_daily_drawdown=0.0,
                   pause_after_drawdown_minutes=0, threshold_drawdown_penalty=0.0,
                   threshold_trade_count_penalty=0.0, target_validation_trades=0,
                   hybrid_runtime_args=None):
    thresholds = sorted(set(float(value) for value in thresholds))
    best_result = None
    profit_objective = objective in ("profit", "profit_balanced", "avg_profit", "ev")
    strict_profit = profit_objective and profit_safety == "strict"
    zero_trade_profit_score = 0.0 if strict_profit else -float("inf")
    fallback_result = None
    selected_score_name = "probability" if objective_mode == "classification" else trade_score_name
    if strict_profit:
        best_metrics = evaluate(
            rows,
            predictions,
            1.01,
            fee,
            slippage,
            compute_auc=False,
            initial_capital=initial_capital,
            max_position_fraction=max_position_fraction,
            max_volume_fraction=max_volume_fraction,
            max_trades_per_period=max_trades_per_period,
            trade_period_minutes=trade_period_minutes,
            holding_period_minutes=holding_period_minutes,
            threshold_objective=objective,
            trade_selection=trade_selection,
            top_k_per_minute=top_k_per_minute,
            upside_target=upside_target,
            downside_stop=downside_stop,
            ev_safety_margin=ev_safety_margin,
            objective_mode=objective_mode,
            trade_score_name=trade_score_name,
            min_predicted_net_return=min_predicted_net_return,
            hybrid_min_score=hybrid_min_score,
            max_trades_per_day=max_trades_per_day,
            max_trades_per_fold=max_trades_per_fold,
            max_losing_trades_per_day=max_losing_trades_per_day,
            max_daily_drawdown=max_daily_drawdown,
            pause_after_drawdown_minutes=pause_after_drawdown_minutes,
            hybrid_runtime_args=hybrid_runtime_args,
        )
        best_result = build_selected_threshold_result(
            1.01,
            best_metrics,
            objective,
            zero_trade_profit_score,
            threshold_drawdown_penalty,
            threshold_trade_count_penalty,
            target_validation_trades,
            selected_score_name,
            min_validation_trades,
            max_validation_trades,
        )
        best_result["tie_rank_reason"] = "strict_profit_no_trade_baseline"
        best_result["validation_metrics"]["selected_threshold_tie_rank_reason"] = "strict_profit_no_trade_baseline"
    for threshold in thresholds:
        metrics = evaluate(
            rows,
            predictions,
            threshold,
            fee,
            slippage,
            compute_auc=False,
            initial_capital=initial_capital,
            max_position_fraction=max_position_fraction,
            max_volume_fraction=max_volume_fraction,
            max_trades_per_period=max_trades_per_period,
            trade_period_minutes=trade_period_minutes,
            holding_period_minutes=holding_period_minutes,
            threshold_objective=objective,
            trade_selection=trade_selection,
            top_k_per_minute=top_k_per_minute,
            upside_target=upside_target,
            downside_stop=downside_stop,
            ev_safety_margin=ev_safety_margin,
            objective_mode=objective_mode,
            trade_score_name=trade_score_name,
            min_predicted_net_return=min_predicted_net_return,
            hybrid_min_score=hybrid_min_score,
            max_trades_per_day=max_trades_per_day,
            max_trades_per_fold=max_trades_per_fold,
            max_losing_trades_per_day=max_losing_trades_per_day,
            max_daily_drawdown=max_daily_drawdown,
            pause_after_drawdown_minutes=pause_after_drawdown_minutes,
            hybrid_runtime_args=hybrid_runtime_args,
        )
        result = build_selected_threshold_result(
            threshold,
            metrics,
            objective,
            zero_trade_profit_score,
            threshold_drawdown_penalty,
            threshold_trade_count_penalty,
            target_validation_trades,
            selected_score_name,
            min_validation_trades,
            max_validation_trades,
        )
        too_few = metrics["predicted_trades"] < min_validation_trades
        too_many = max_validation_trades > 0 and metrics["predicted_trades"] > max_validation_trades
        too_imprecise = metrics["predicted_trades"] > 0 and metrics["precision"] < min_validation_precision
        if too_many or too_imprecise:
            continue
        if too_few:
            if metrics["predicted_trades"] > 0:
                better, reason = compare_threshold_results(
                    result,
                    fallback_result,
                    hybrid_runtime_args or argparse.Namespace(
                        threshold_tiebreaker="fewer_trades",
                        threshold_tie_epsilon=1e-9,
                        threshold_target_trades=0,
                        threshold_target_active_days=0,
                        target_validation_trades=target_validation_trades,
                        threshold_objective=objective,
                    ),
                    min_validation_trades,
                    max_validation_trades,
                )
                if better:
                    result["tie_rank_reason"] = reason
                    result["validation_metrics"]["selected_threshold_tie_rank_reason"] = reason
                    fallback_result = result
            continue
        better, reason = compare_threshold_results(
            result,
            best_result,
            hybrid_runtime_args or argparse.Namespace(
                threshold_tiebreaker="fewer_trades",
                threshold_tie_epsilon=1e-9,
                threshold_target_trades=0,
                threshold_target_active_days=0,
                target_validation_trades=target_validation_trades,
                threshold_objective=objective,
            ),
            min_validation_trades,
            max_validation_trades,
        )
        if better:
            result["tie_rank_reason"] = reason
            result["validation_metrics"]["selected_threshold_tie_rank_reason"] = reason
            best_result = result
    if best_result is None:
        if fallback_result is not None:
            best_result = fallback_result
        else:
            best_metrics = evaluate(
                rows,
                predictions,
                1.01,
                fee,
                slippage,
                compute_auc=False,
                initial_capital=initial_capital,
                max_position_fraction=max_position_fraction,
                max_volume_fraction=max_volume_fraction,
                max_trades_per_period=max_trades_per_period,
                trade_period_minutes=trade_period_minutes,
                holding_period_minutes=holding_period_minutes,
                threshold_objective=objective,
                trade_selection=trade_selection,
                top_k_per_minute=top_k_per_minute,
                upside_target=upside_target,
                downside_stop=downside_stop,
                ev_safety_margin=ev_safety_margin,
                objective_mode=objective_mode,
                trade_score_name=trade_score_name,
                min_predicted_net_return=min_predicted_net_return,
                hybrid_min_score=hybrid_min_score,
                max_trades_per_day=max_trades_per_day,
                max_trades_per_fold=max_trades_per_fold,
                max_losing_trades_per_day=max_losing_trades_per_day,
                max_daily_drawdown=max_daily_drawdown,
                pause_after_drawdown_minutes=pause_after_drawdown_minutes,
                hybrid_runtime_args=hybrid_runtime_args,
            )
            best_result = build_selected_threshold_result(
                1.01,
                best_metrics,
                objective,
                zero_trade_profit_score,
                threshold_drawdown_penalty,
                threshold_trade_count_penalty,
                target_validation_trades,
                selected_score_name,
                min_validation_trades,
                max_validation_trades,
            )
            best_result["tie_rank_reason"] = "no_trade_fallback"
            best_result["validation_metrics"]["selected_threshold_tie_rank_reason"] = "no_trade_fallback"
    elif not best_result["validation_metrics"].get("selected_threshold_tie_rank_reason"):
        best_result["validation_metrics"]["selected_threshold_tie_rank_reason"] = best_result.get("tie_rank_reason", "higher_objective_score")
    return best_result


class InternalStumpGBDT(object):
    def __init__(self, n_estimators=12, learning_rate=0.12, max_bins=8, l2=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_bins = max_bins
        self.l2 = l2
        self.task = "classification"
        self.base_score = 0.0
        self.stumps = []
        self.importances = []

    def _thresholds_for_feature(self, values):
        ordered = sorted(values)
        thresholds = []
        for bin_index in range(1, self.max_bins):
            position = int(len(ordered) * bin_index / float(self.max_bins))
            position = min(max(position, 0), len(ordered) - 1)
            value = ordered[position]
            if not thresholds or value != thresholds[-1]:
                thresholds.append(value)
        return thresholds

    def fit(self, x_train, y_train, feature_names, x_validation=None, y_validation=None, args=None):
        del x_validation, y_validation, args
        row_count = len(x_train)
        if row_count == 0:
            raise ValueError("cannot fit internal GBDT with zero rows")
        feature_count = len(feature_names)
        positives = sum(y_train)
        base_probability = min(0.999, max(0.001, positives / float(row_count)))
        self.base_score = math.log(base_probability / (1.0 - base_probability))
        raw_scores = array("d", [self.base_score]) * row_count
        self.importances = [0.0] * feature_count

        thresholds_by_feature = []
        for feature_index in range(feature_count):
            thresholds_by_feature.append(self._thresholds_for_feature(row[feature_index] for row in x_train))

        for _ in range(self.n_estimators):
            probabilities = array("d", (sigmoid(value) for value in raw_scores))
            gradients = array("d", (y_train[i] - probabilities[i] for i in range(row_count)))
            hessians = array("d", (max(1e-6, probabilities[i] * (1.0 - probabilities[i])) for i in range(row_count)))
            total_gradient = sum(gradients)
            total_hessian = sum(hessians)
            base_gain = (total_gradient * total_gradient) / (total_hessian + self.l2)

            best = None
            best_gain = -float("inf")
            for feature_index, thresholds in enumerate(thresholds_by_feature):
                if not thresholds:
                    continue
                bin_count = len(thresholds) + 1
                gradient_bins = [0.0] * bin_count
                hessian_bins = [0.0] * bin_count
                for row_index, row in enumerate(x_train):
                    bin_index = bisect.bisect_right(thresholds, row[feature_index])
                    gradient_bins[bin_index] += gradients[row_index]
                    hessian_bins[bin_index] += hessians[row_index]

                left_gradient = 0.0
                left_hessian = 0.0
                for threshold_index, threshold in enumerate(thresholds):
                    left_gradient += gradient_bins[threshold_index]
                    left_hessian += hessian_bins[threshold_index]
                    right_gradient = total_gradient - left_gradient
                    right_hessian = total_hessian - left_hessian
                    if left_hessian <= 0.0 or right_hessian <= 0.0:
                        continue
                    gain = (
                        (left_gradient * left_gradient) / (left_hessian + self.l2)
                        + (right_gradient * right_gradient) / (right_hessian + self.l2)
                        - base_gain
                    )
                    if gain > best_gain:
                        left_value = max(-5.0, min(5.0, left_gradient / (left_hessian + self.l2)))
                        right_value = max(-5.0, min(5.0, right_gradient / (right_hessian + self.l2)))
                        best_gain = gain
                        best = (feature_index, threshold, left_value, right_value)

            if best is None:
                break

            feature_index, threshold, left_value, right_value = best
            self.stumps.append(best)
            self.importances[feature_index] += max(0.0, best_gain)
            for row_index, row in enumerate(x_train):
                raw_scores[row_index] += self.learning_rate * (
                    left_value if row[feature_index] <= threshold else right_value
                )
        return self

    def predict_proba(self, x_rows):
        probabilities = []
        for row in x_rows:
            raw = self.base_score
            for feature_index, threshold, left_value, right_value in self.stumps:
                raw += self.learning_rate * (left_value if row[feature_index] <= threshold else right_value)
            probabilities.append(sigmoid(raw))
        return probabilities

    def predict_values(self, x_rows):
        return self.predict_proba(x_rows)

    def feature_importance(self, feature_names):
        total = sum(self.importances)
        rows = []
        for name, value in zip(feature_names, self.importances):
            rows.append((name, value, value / total if total else 0.0))
        return rows

    def best_iteration(self):
        return None


class InternalMeanRegressor(object):
    def __init__(self):
        self.task = "regression"
        self.mean_value = 0.0

    def fit(self, x_train, y_train, feature_names, x_validation=None, y_validation=None, args=None):
        del x_train, feature_names, x_validation, y_validation, args
        if len(y_train) == 0:
            self.mean_value = 0.0
        else:
            self.mean_value = sum(float(value) for value in y_train) / float(len(y_train))
        return self

    def predict_values(self, x_rows):
        if np is not None and isinstance(x_rows, np.ndarray):
            return np.full(len(x_rows), self.mean_value, dtype=np.float32)
        return [self.mean_value] * len(x_rows)

    def feature_importance(self, feature_names):
        return [(name, 0.0, 0.0) for name in feature_names]

    def best_iteration(self):
        return None


class ExternalModel(object):
    def __init__(self, model, kind, task):
        self.model = model
        self.kind = kind
        self.task = task

    def fit(self, x_train, y_train, feature_names, x_validation=None, y_validation=None, args=None):
        del feature_names
        fit_kwargs = {}
        training_classes = np.unique(y_train) if np is not None else set(y_train)
        use_validation = x_validation is not None and y_validation is not None and len(y_validation)
        if self.task == "classification" and use_validation and len(training_classes) > 1:
            from lightgbm import early_stopping, log_evaluation
            fit_kwargs["eval_set"] = [(x_validation, y_validation)]
            fit_kwargs["eval_metric"] = args.eval_metric if args is not None else "binary_logloss"
            callbacks = []
            if args is not None and args.early_stopping_rounds > 0:
                callbacks.append(early_stopping(args.early_stopping_rounds, verbose=False))
            if args is not None:
                callbacks.append(log_evaluation(args.log_evaluation_period))
            if callbacks:
                fit_kwargs["callbacks"] = callbacks
        elif self.task == "regression" and use_validation:
            fit_kwargs["eval_set"] = [(x_validation, y_validation)]
            fit_kwargs["eval_metric"] = "l2"
        self.model.fit(x_train, y_train, **fit_kwargs)
        return self

    def predict_proba(self, x_rows):
        best_iteration = getattr(self.model, "best_iteration_", None)
        kwargs = {"num_iteration": best_iteration} if best_iteration else {}
        probabilities = self.model.predict_proba(x_rows, **kwargs)
        if np is not None:
            return np.asarray(probabilities)[:, 1].astype(np.float32, copy=True)
        return [float(row[1]) for row in probabilities]

    def predict_values(self, x_rows):
        if self.task == "classification":
            return self.predict_proba(x_rows)
        best_iteration = getattr(self.model, "best_iteration_", None)
        kwargs = {"num_iteration": best_iteration} if best_iteration else {}
        predictions = self.model.predict(x_rows, **kwargs)
        if np is not None:
            return np.asarray(predictions, dtype=np.float32)
        return [float(value) for value in predictions]

    def feature_importance(self, feature_names):
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        total = float(sum(importances))
        return [
            (name, float(value), float(value) / total if total else 0.0)
            for name, value in zip(feature_names, importances)
        ]

    def best_iteration(self):
        value = getattr(self.model, "best_iteration_", None)
        return int(value) if value else None


def external_available(module_name):
    return importlib.util.find_spec(module_name) is not None


def choose_model_kind(requested):
    if requested == "auto":
        if external_available("lightgbm"):
            return "lightgbm"
        return "internal"
    if requested == "lightgbm":
        if not external_available("lightgbm"):
            raise RuntimeError("LightGBM is not installed")
        return "lightgbm"
    if requested == "internal":
        return "internal"
    raise RuntimeError("Unknown model kind: {}".format(requested))


def class_weight_ratio(y_train, cap):
    positives = int(np.sum(y_train)) if np is not None and isinstance(y_train, np.ndarray) else sum(y_train)
    negatives = len(y_train) - positives
    if positives <= 0:
        return 1.0
    return min(cap, negatives / float(positives))


def positive_label_count(rows):
    if is_compact_rows(rows):
        labels_array = rows.table.labels
        if rows.indices is None:
            return int(np.sum(labels_array))
        total = 0
        batch_size = 5000000
        for start in range(0, len(rows.indices), batch_size):
            total += int(np.sum(labels_array[rows.indices[start:start + batch_size]]))
        return total
    return sum(row.label for row in rows)


def class_weight_ratio_for_rows(rows, cap):
    positives = positive_label_count(rows)
    negatives = len(rows) - positives
    if positives <= 0:
        return 1.0
    return min(cap, negatives / float(positives))


def make_model(kind, params, positive_weight, objective_mode="classification"):
    if kind == "lightgbm":
        if objective_mode == "classification":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                subsample_freq=1,
                colsample_bytree=params["colsample_bytree"],
                min_child_samples=params["min_child_samples"],
                min_split_gain=params["min_split_gain"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                max_bin=params["max_bin"],
                subsample_for_bin=params["subsample_for_bin"],
                histogram_pool_size=params["histogram_pool_size"],
                n_jobs=params["n_jobs"],
                force_col_wise=True,
                objective="binary",
                scale_pos_weight=positive_weight,
                random_state=17,
                verbosity=-1,
            )
            return ExternalModel(model, kind, "classification")
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            min_child_samples=params["min_child_samples"],
            min_split_gain=params["min_split_gain"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            max_bin=params["max_bin"],
            subsample_for_bin=params["subsample_for_bin"],
            histogram_pool_size=params["histogram_pool_size"],
            n_jobs=params["n_jobs"],
            force_col_wise=True,
            objective="regression",
            random_state=17,
            verbosity=-1,
        )
        return ExternalModel(model, kind, "regression")
    if objective_mode == "classification":
        return InternalStumpGBDT(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_bins=params["max_bins"],
            l2=params["l2"],
        )
    return InternalMeanRegressor()


def candidate_params(kind, args):
    if kind == "lightgbm":
        return [
            {"n_estimators": args.n_estimators, "learning_rate": args.learning_rate, "num_leaves": 15,
             "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.85,
             "min_child_samples": 100, "min_split_gain": 0.01, "reg_alpha": 1.0,
             "reg_lambda": 5.0, "max_bin": args.max_bin,
             "subsample_for_bin": args.subsample_for_bin,
             "histogram_pool_size": args.lightgbm_histogram_pool_mb,
             "n_jobs": args.n_jobs},
            {"n_estimators": max(80, args.n_estimators // 2), "learning_rate": args.learning_rate * 1.6,
             "num_leaves": 31, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.85,
             "min_child_samples": 80, "min_split_gain": 0.0, "reg_alpha": 0.25,
             "reg_lambda": 4.0, "max_bin": args.max_bin,
             "subsample_for_bin": args.subsample_for_bin,
             "histogram_pool_size": args.lightgbm_histogram_pool_mb,
             "n_jobs": args.n_jobs},
            {"n_estimators": int(args.n_estimators * 1.5), "learning_rate": args.learning_rate * 0.7,
             "num_leaves": 63, "max_depth": -1, "subsample": 0.85, "colsample_bytree": 0.9,
             "min_child_samples": 60, "min_split_gain": 0.0, "reg_alpha": 0.1,
             "reg_lambda": 3.0, "max_bin": args.max_bin,
             "subsample_for_bin": args.subsample_for_bin,
             "histogram_pool_size": args.lightgbm_histogram_pool_mb,
             "n_jobs": args.n_jobs},
        ]
    return [
        {"n_estimators": max(2, args.internal_estimators // 2), "learning_rate": args.internal_learning_rate,
         "max_bins": args.internal_bins, "l2": args.internal_l2},
        {"n_estimators": args.internal_estimators, "learning_rate": args.internal_learning_rate,
         "max_bins": args.internal_bins, "l2": args.internal_l2},
        {"n_estimators": max(args.internal_estimators * 2, 24), "learning_rate": args.internal_learning_rate * 0.7,
         "max_bins": max(args.internal_bins, 16), "l2": args.internal_l2 * 1.5},
    ]


def cleanup_probabilities(probabilities):
    if np is not None and isinstance(probabilities, np.memmap):
        path = os.path.abspath(str(probabilities.filename))
        close_memmap(probabilities)
        if path in TEMP_PREDICTION_PATHS:
            try:
                os.remove(path)
            except OSError:
                pass
            TEMP_PREDICTION_PATHS.discard(path)


def allocate_prediction_values(row_count, args, prefix):
    if np is None:
        return [0.0] * row_count
    if row_count > int(args.prediction_batch_rows):
        directory = args.memmap_dir or tempfile.gettempdir()
        os.makedirs(directory, exist_ok=True)
        descriptor, path = tempfile.mkstemp(prefix=prefix, suffix=".dat", dir=directory)
        os.close(descriptor)
        path = os.path.abspath(path)
        TEMP_PREDICTION_PATHS.add(path)
        return np.memmap(path, dtype=np.float32, mode="w+", shape=(row_count,))
    return np.empty(row_count, dtype=np.float32)


def copy_prediction_values(values, args, prefix):
    row_count = len(values)
    if np is None or not isinstance(values, np.ndarray):
        return list(values)
    copied = allocate_prediction_values(row_count, args, prefix)
    copied[:] = np.asarray(values, dtype=np.float32)
    if isinstance(copied, np.memmap):
        copied.flush()
    return copied


def cleanup_prediction_bundle(bundle):
    seen = set()
    for key in ("probability", "calibrated_probability", "predicted_trade_return",
                "raw_predicted_trade_return", "predicted_return_uncertainty", "meta_probability"):
        values = bundle.get(key)
        if values is None:
            continue
        identity = id(values)
        if identity in seen:
            continue
        seen.add(identity)
        cleanup_probabilities(values)


def add_prediction_values_in_place(target, values):
    if target is None or values is None:
        return target
    if np is not None and isinstance(target, np.ndarray):
        target[:] = target + np.asarray(values, dtype=np.float32)
        return target
    for index, value in enumerate(values):
        target[index] += float(value)
    return target


def divide_prediction_values_in_place(values, divisor):
    if values is None or divisor == 0:
        return values
    if np is not None and isinstance(values, np.ndarray):
        values[:] = values / np.float32(divisor)
        return values
    for index in range(len(values)):
        values[index] /= float(divisor)
    return values


def average_prediction_bundles(rows, args, bundles):
    if not bundles:
        return build_prediction_bundle()
    row_count = len(rows)
    probability = None
    calibrated_probability = None
    predicted_trade_return = None
    raw_predicted_trade_return = None
    predicted_return_uncertainty = None
    meta_probability = None
    counts = {
        "probability": 0,
        "calibrated_probability": 0,
        "predicted_trade_return": 0,
        "raw_predicted_trade_return": 0,
        "predicted_return_uncertainty": 0,
        "meta_probability": 0,
    }
    for bundle in bundles:
        if bundle.get("probability") is not None:
            if probability is None:
                probability = allocate_prediction_values(row_count, args, "gbdt_ensemble_probability_")
                probability[:] = 0.0
            add_prediction_values_in_place(probability, bundle["probability"])
            counts["probability"] += 1
        if bundle.get("calibrated_probability") is not None:
            if calibrated_probability is None:
                calibrated_probability = allocate_prediction_values(row_count, args, "gbdt_ensemble_calibrated_probability_")
                calibrated_probability[:] = 0.0
            add_prediction_values_in_place(calibrated_probability, bundle["calibrated_probability"])
            counts["calibrated_probability"] += 1
        if bundle.get("predicted_trade_return") is not None:
            if predicted_trade_return is None:
                predicted_trade_return = allocate_prediction_values(row_count, args, "gbdt_ensemble_predicted_trade_return_")
                predicted_trade_return[:] = 0.0
            add_prediction_values_in_place(predicted_trade_return, bundle["predicted_trade_return"])
            counts["predicted_trade_return"] += 1
        if bundle.get("raw_predicted_trade_return") is not None:
            if raw_predicted_trade_return is None:
                raw_predicted_trade_return = allocate_prediction_values(row_count, args, "gbdt_ensemble_raw_predicted_trade_return_")
                raw_predicted_trade_return[:] = 0.0
            add_prediction_values_in_place(raw_predicted_trade_return, bundle["raw_predicted_trade_return"])
            counts["raw_predicted_trade_return"] += 1
        if bundle.get("predicted_return_uncertainty") is not None:
            if predicted_return_uncertainty is None:
                predicted_return_uncertainty = allocate_prediction_values(row_count, args, "gbdt_ensemble_predicted_return_uncertainty_")
                predicted_return_uncertainty[:] = 0.0
            add_prediction_values_in_place(predicted_return_uncertainty, bundle["predicted_return_uncertainty"])
            counts["predicted_return_uncertainty"] += 1
        if bundle.get("meta_probability") is not None:
            if meta_probability is None:
                meta_probability = allocate_prediction_values(row_count, args, "gbdt_ensemble_meta_probability_")
                meta_probability[:] = 0.0
            add_prediction_values_in_place(meta_probability, bundle["meta_probability"])
            counts["meta_probability"] += 1
        cleanup_prediction_bundle(bundle)
    divide_prediction_values_in_place(probability, counts["probability"])
    divide_prediction_values_in_place(calibrated_probability, counts["calibrated_probability"])
    divide_prediction_values_in_place(predicted_trade_return, counts["predicted_trade_return"])
    divide_prediction_values_in_place(raw_predicted_trade_return, counts["raw_predicted_trade_return"])
    divide_prediction_values_in_place(predicted_return_uncertainty, counts["predicted_return_uncertainty"])
    divide_prediction_values_in_place(meta_probability, counts["meta_probability"])
    return build_prediction_bundle(
        probability,
        calibrated_probability,
        predicted_trade_return,
        raw_predicted_trade_return=raw_predicted_trade_return,
        predicted_return_uncertainty=predicted_return_uncertainty,
        meta_probability=meta_probability,
        ev_context=bundles[0].get("ev_context") if bundles else None,
        hybrid_return_context=bundles[0].get("hybrid_return_context") if bundles else None,
        uncertainty_context=bundles[0].get("uncertainty_context") if bundles else None,
    )


def prediction_bundle_for_models(selected, rows, kind, args, stage_prefix):
    if selected.get("ensemble_members"):
        bundles = []
        for member in selected["ensemble_members"]:
            bundles.append(prediction_bundle_for_models(member, rows, kind, args, "{} window {}".format(stage_prefix, member.get("ensemble_window", ""))))
        bundle = average_prediction_bundles(rows, args, bundles)
        if selected.get("ev_payoff_info") is not None:
            bundle["ev_context"] = selected.get("ev_payoff_info")
        if selected.get("hybrid_return_info") is not None:
            bundle["hybrid_return_context"] = selected.get("hybrid_return_info")
        meta_probability = apply_meta_filter(
            rows,
            bundle,
            selected.get("threshold", threshold_for_mode(args)),
            args,
            selected.get("meta_filter_info"),
            selected.get("symbol_filter_info"),
        )
        if meta_probability is not None:
            bundle["meta_probability"] = meta_probability
        return bundle
    probability = None
    calibrated_probability = None
    predicted_trade_return = None
    raw_predicted_trade_return = None
    predicted_return_uncertainty = None
    if selected.get("classification_model") is not None:
        probability = predict_model_values(
            selected["classification_model"],
            rows,
            kind,
            args,
            "{} classification prediction".format(stage_prefix),
        )
        calibration = selected.get("calibration")
        if calibration:
            calibrated_probability = copy_prediction_values(probability, args, "gbdt_calibrated_")
            calibrate_probabilities(calibrated_probability, calibration)
        else:
            calibrated_probability = probability
    if selected.get("regression_model") is not None:
        regression_prediction = predict_model_values(
            selected["regression_model"],
            rows,
            kind,
            args,
            "{} regression prediction".format(stage_prefix),
        )
        raw_predicted_trade_return = regression_predictions_to_trade_return(regression_prediction, rows, args)
        predicted_trade_return = copy_prediction_values(raw_predicted_trade_return, args, "gbdt_regression_calibrated_")
        apply_regression_calibration(predicted_trade_return, selected.get("regression_calibration"))
        predicted_return_uncertainty = apply_uncertainty_model(
            predicted_trade_return,
            selected.get("uncertainty_model"),
        )
        del regression_prediction
    bundle = build_prediction_bundle(
        probability,
        calibrated_probability,
        predicted_trade_return,
        raw_predicted_trade_return=raw_predicted_trade_return,
        predicted_return_uncertainty=predicted_return_uncertainty,
        ev_context=selected.get("ev_payoff_info"),
        hybrid_return_context=selected.get("hybrid_return_info"),
        uncertainty_context=selected.get("uncertainty_model"),
    )
    meta_probability = apply_meta_filter(
        rows,
        bundle,
        selected.get("threshold", threshold_for_mode(args)),
        args,
        selected.get("meta_filter_info"),
        selected.get("symbol_filter_info"),
    )
    if meta_probability is not None:
        bundle["meta_probability"] = meta_probability
    return bundle


def selected_thresholds_for_bundle(bundle, validation_rows, args):
    if args.objective_mode != "classification":
        threshold = threshold_for_mode(args)
        if args.disable_adaptive_thresholds:
            return [threshold], threshold, None
        score_values = score_values_for_bundle(validation_rows, bundle, args)
        thresholds = adaptive_score_thresholds(
            score_values,
            threshold,
            args.adaptive_threshold_sample_rows,
        )
        return thresholds, threshold, score_values
    probabilities = bundle["calibrated_probability"] if bundle.get("calibrated_probability") is not None else bundle.get("probability")
    thresholds = args.thresholds
    if not args.disable_adaptive_thresholds:
        thresholds = adaptive_thresholds(
            probabilities,
            args.thresholds,
            args.min_validation_trades,
            args.adaptive_threshold_sample_rows,
        )
    thresholds = sorted(set(float(threshold) for threshold in thresholds if threshold >= args.min_selected_threshold))
    if not thresholds:
        thresholds = [1.01]
    return thresholds, None, probabilities


def predict_model_values(model, rows, kind, args, stage="prediction"):
    row_count = len(rows)
    batch_size = max(1, int(args.prediction_batch_rows))
    if np is not None:
        use_memmap = row_count > batch_size
        if use_memmap:
            directory = args.memmap_dir or tempfile.gettempdir()
            os.makedirs(directory, exist_ok=True)
            descriptor, path = tempfile.mkstemp(prefix="gbdt_probabilities_", suffix=".dat", dir=directory)
            os.close(descriptor)
            path = os.path.abspath(path)
            TEMP_PREDICTION_PATHS.add(path)
            probabilities = np.memmap(path, dtype=np.float32, mode="w+", shape=(row_count,))
        else:
            probabilities = np.empty(row_count, dtype=np.float32)
        for start in range(0, row_count, batch_size):
            end = min(row_count, start + batch_size)
            chunk_rows = rows_slice(rows, start, end)
            x_chunk = model_matrix(chunk_rows, kind, args)
            probabilities[start:end] = model.predict_values(x_chunk)
            del x_chunk
            del chunk_rows
            if start and start % max(batch_size, 1000000) == 0:
                memory_checkpoint("{} progress: {:,}/{:,} rows".format(stage, start, row_count), args)
        if isinstance(probabilities, np.memmap):
            probabilities.flush()
        gc.collect()
        memory_checkpoint("{} complete: {:,} rows".format(stage, row_count), args)
        return probabilities

    probabilities = []
    for start in range(0, row_count, batch_size):
        end = min(row_count, start + batch_size)
        chunk_rows = rows_slice(rows, start, end)
        x_chunk = model_matrix(chunk_rows, kind, args)
        probabilities.extend(model.predict_values(x_chunk))
        del x_chunk
        del chunk_rows
    gc.collect()
    memory_checkpoint("{} complete: {:,} rows".format(stage, row_count), args)
    return probabilities


def predict_probabilities(model, rows, kind, args, stage="prediction"):
    return predict_model_values(model, rows, kind, args, stage)


def fit_select_model(train_rows, validation_rows, feature_names, args, kind):
    if getattr(args, "ensemble_window_list", None):
        member_args = argparse.Namespace(**vars(args))
        member_args.ensemble_windows = ""
        member_args.ensemble_window_list = []
        member_args.meta_filter = "none"
        ensemble_members = []
        seen_windows = set()
        for window in args.ensemble_window_list:
            window_train_rows = recent_month_window_rows(train_rows, window)
            if not window_train_rows or len(window_train_rows) == len(train_rows) and window in seen_windows:
                continue
            seen_windows.add(window)
            member = fit_select_model(window_train_rows, validation_rows, feature_names, member_args, kind)
            member["ensemble_window"] = window
            ensemble_members.append(member)
        if not ensemble_members:
            raise ValueError("no valid ensemble windows produced training rows")
        bundle = prediction_bundle_for_models(
            {"ensemble_members": ensemble_members, "meta_filter_info": None, "threshold": threshold_for_mode(args)},
            validation_rows,
            kind,
            member_args,
            "ensemble validation",
        )
        bundle["ev_context"] = fit_ev_payoff_context(validation_rows, bundle, args)
        bundle["hybrid_return_context"] = fit_hybrid_return_context(validation_rows, bundle, args)
        thresholds, fixed_threshold, ignored = selected_thresholds_for_bundle(bundle, validation_rows, args)
        del fixed_threshold, ignored
        selection = tune_threshold(
            validation_rows,
            bundle,
            thresholds,
            args.threshold_objective,
            args.fee,
            args.slippage * args.validation_slippage_multiplier,
            args.min_validation_trades,
            args.max_validation_trades,
            args.min_validation_precision,
            args.profit_safety,
            args.initial_capital,
            args.max_position_fraction,
            args.max_volume_fraction,
            args.max_trades_per_period,
            args.trade_period_minutes,
            args.holding_period_minutes,
            args.trade_selection,
            args.top_k_per_minute,
            args.upside_target,
            args.downside_stop,
            args.ev_safety_margin,
            args.objective_mode,
            score_name_for_args(args),
            args.min_predicted_net_return,
            args.hybrid_min_score,
            args.max_trades_per_day,
            0,
            args.max_losing_trades_per_day,
            args.max_daily_drawdown,
            args.pause_after_drawdown_minutes,
            args.threshold_drawdown_penalty,
            args.threshold_trade_count_penalty,
            args.target_validation_trades,
            args,
        )
        meta_filter_info = fit_meta_filter(validation_rows, bundle, selection["threshold"], args)
        meta_filter_info, selection = recalibrate_meta_filter_validation(
            validation_rows,
            bundle,
            selection["threshold"],
            args,
            selection,
            meta_filter_info,
            None,
            selection.get("selected_score_name", selected_score_name_for_mode(args)),
        )
        ev_context = bundle.get("ev_context") or {}
        hybrid_context = bundle.get("hybrid_return_context") or {}
        calibration_info = {
            "calibration": "ensemble",
            "calibration_a": 0.0,
            "calibration_b": 0.0,
            "validation_brier_before": 0.0,
            "validation_brier_after": 0.0,
            "calibration_rows": 0,
            "regression_calibration": "member_average",
            "regression_calibration_a": 1.0,
            "regression_calibration_b": 0.0,
            "regression_calibration_rows": 0,
            "regression_calibration_rmse_before": 0.0,
            "regression_calibration_rmse_after": 0.0,
            "regression_calibration_mae_before": 0.0,
            "regression_calibration_mae_after": 0.0,
            "hybrid_return_combination": hybrid_context.get("hybrid_return_combination", getattr(args, "hybrid_return_combination", "probability_times_return")),
            "hybrid_min_probability": hybrid_context.get("hybrid_min_probability", getattr(args, "hybrid_min_probability", 0.0)),
            "hybrid_score_mode": args.hybrid_score_mode,
            "hybrid_uncertainty_method": args.hybrid_uncertainty_method,
            "hybrid_uncertainty_penalty": args.hybrid_uncertainty_penalty,
            "hybrid_uncertainty_global_std": 0.0,
            "hybrid_uncertainty_rows": 0,
            "conditional_expected_win_return": hybrid_context.get("conditional_expected_win_return", 0.0),
            "conditional_expected_loss_return": hybrid_context.get("conditional_expected_loss_return", 0.0),
            "conditional_payoff_rows": hybrid_context.get("conditional_payoff_rows", 0),
            "conditional_payoff_positive_rows": hybrid_context.get("conditional_payoff_positive_rows", 0),
            "conditional_payoff_negative_rows": hybrid_context.get("conditional_payoff_negative_rows", 0),
            "conditional_payoff_source": hybrid_context.get("conditional_payoff_source", "not_used"),
            "ev_payoff_mode": ev_context.get("ev_payoff_mode", getattr(args, "ev_payoff_mode", "fixed_targets")),
            "ev_expected_win_return": ev_context.get("ev_expected_win_return", getattr(args, "effective_upside_target", getattr(args, "upside_target", 0.05))),
            "ev_expected_loss_return": ev_context.get("ev_expected_loss_return", -getattr(args, "effective_downside_stop", getattr(args, "downside_stop", 0.02))),
            "ev_payoff_rows": ev_context.get("ev_payoff_rows", 0),
            "ev_payoff_positive_rows": ev_context.get("ev_payoff_positive_rows", 0),
            "ev_payoff_negative_rows": ev_context.get("ev_payoff_negative_rows", 0),
            "ev_payoff_source": ev_context.get("ev_payoff_source", "fixed_targets"),
            "meta_filter": meta_filter_info.get("mode", "none") if meta_filter_info else "none",
            "meta_filter_rows": meta_filter_info.get("rows", 0) if meta_filter_info else 0,
            "meta_filter_positive_rate": meta_filter_info.get("positive_rate", 0.0) if meta_filter_info else 0.0,
            "meta_filter_auc_or_accuracy": meta_filter_info.get("auc", meta_filter_info.get("accuracy", 0.0)) if meta_filter_info else 0.0,
            "meta_filter_enabled": 1 if meta_filter_info and meta_filter_info.get("enabled") else 0,
            "ensemble_windows": ",".join(str(value) for value in args.ensemble_window_list),
            "ensemble_model_count": len(ensemble_members),
            "ensemble_enabled": 1,
            "regression_target": args.regression_target,
            "risk_adjusted_return_feature": regression_target_feature_name(args) or "",
            "dynamic_hybrid_thresholds": args.dynamic_hybrid_thresholds,
            "meta_filter_disabled_reason": meta_filter_info.get("disabled_reason", "") if meta_filter_info else "",
            "meta_filter_validation_trade_retention": meta_filter_info.get("validation_trade_retention", 0.0) if meta_filter_info else 0.0,
            "symbol_filter_stage": getattr(args, "symbol_filter_stage", "executed"),
            "symbol_filter_min_candidates": getattr(args, "symbol_filter_min_candidates", 0),
            "symbol_filter_min_executed": getattr(args, "symbol_filter_min_executed", 0),
            "symbol_filter_candidate_weight": getattr(args, "symbol_filter_candidate_weight", 0.5),
            "symbol_filter_executed_weight": getattr(args, "symbol_filter_executed_weight", 0.5),
            "symbol_filter_shrinkage": getattr(args, "symbol_filter_shrinkage", 50.0),
            "symbols_blocked_count": 0,
            "symbols_allowed_count": 0,
        }
        selection["validation_metrics"]["calibration_info"] = dict(calibration_info)
        cleanup_prediction_bundle(bundle)
        return {
            "classification_model": None,
            "regression_model": None,
            "ensemble_members": ensemble_members,
            "ensemble_enabled": True,
            "ensemble_model_count": len(ensemble_members),
            "params": {"ensemble_windows": list(args.ensemble_window_list)},
            "threshold": selection["threshold"],
            "validation_metrics": selection["validation_metrics"],
            "selection": selection,
            "score": selection.get("objective_score", selection["validation_metrics"].get("selected_objective_score", 0.0)),
            "base_score": selection.get("base_objective_score", selection["validation_metrics"].get("selected_base_objective_score", 0.0)),
            "rank": threshold_rank(
                selection["validation_metrics"],
                args.threshold_objective,
                0.0 if args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev") and args.profit_safety == "strict" else -float("inf"),
                selection.get("objective_score", selection["validation_metrics"].get("selected_objective_score", 0.0)),
                args.target_validation_trades,
                args.min_validation_trades,
                args.max_validation_trades,
            ),
            "calibration": None,
            "regression_calibration": None,
            "uncertainty_model": None,
            "trade_score_name": score_name_for_args(args),
            "selected_score_name": selected_score_name_for_mode(args),
            "calibration_info": calibration_info,
            "meta_filter_info": meta_filter_info,
            "ev_payoff_info": dict(bundle.get("ev_context") or {}),
            "hybrid_return_info": dict(bundle.get("hybrid_return_context") or {}),
        }

    sampling_token = start_profile_stage("row_sampling", "objective_mode={}".format(args.objective_mode))
    fit_train_rows = sample_rows(
        train_rows,
        args.max_train_rows,
        label_aware=True,
        sample_mode=args.train_sample_mode,
        max_positive_fraction=args.max_positive_sample_fraction,
    )
    fit_validation_rows = sample_rows(
        validation_rows,
        args.max_validation_rows,
        label_aware=False,
        sample_mode="chronological",
        max_positive_fraction=args.max_positive_sample_fraction,
    )
    if len(fit_train_rows) != len(train_rows) or len(fit_validation_rows) != len(validation_rows):
        sampled_positives = positive_label_count(fit_train_rows)
        print(
            "Fitting on sampled rows: train {}/{} positives={} validation {}/{}".format(
                len(fit_train_rows),
                len(train_rows),
                sampled_positives,
                len(fit_validation_rows),
                len(validation_rows),
            ),
            flush=True,
        )
    finish_profile_stage(sampling_token, rows_processed=len(fit_train_rows) + len(fit_validation_rows))
    memory_checkpoint("Row sampling complete", args)

    x_train = model_matrix(fit_train_rows, kind, args)
    x_validation = model_matrix(fit_validation_rows, kind, args)
    positive_weight = class_weight_ratio_for_rows(train_rows, args.positive_weight_cap)
    score_name = score_name_for_args(args)
    selected_score_name = selected_score_name_for_mode(args)

    y_class_train = None
    y_class_validation = None
    y_reg_train = None
    y_reg_validation = None
    if args.objective_mode in ("classification", "hybrid"):
        y_class_train = model_targets(fit_train_rows, kind, "classification")
        y_class_validation = model_targets(fit_validation_rows, kind, "classification")
    if args.objective_mode in ("return_regression", "hybrid"):
        y_reg_train = model_targets(fit_train_rows, kind, "return_regression", args)
        y_reg_validation = model_targets(fit_validation_rows, kind, "return_regression", args)

    best = None
    for params in candidate_params(kind, args):
        classification_model = None
        regression_model = None
        raw_probabilities = None
        calibrated_probabilities = None
        predicted_trade_return = None
        calibration = None
        regression_calibration = None
        uncertainty_model = None

        if args.objective_mode in ("classification", "hybrid"):
            classification_model = make_model(kind, params, positive_weight, "classification")
            classification_fit_token = start_profile_stage("classification_fit", json.dumps(params, sort_keys=True))
            classification_model.fit(x_train, y_class_train, feature_names, x_validation, y_class_validation, args)
            finish_profile_stage(classification_fit_token, rows_processed=len(fit_train_rows), extra_info=json.dumps(params, sort_keys=True))
            memory_checkpoint(
                "LightGBM classification candidate fit complete" if kind == "lightgbm" else "Internal classification candidate fit complete",
                args,
            )
            sampled_classification_token = start_profile_stage("sampled_validation_classification_prediction", json.dumps(params, sort_keys=True))
            raw_probabilities = classification_model.predict_values(x_validation)
            finish_profile_stage(sampled_classification_token, rows_processed=len(fit_validation_rows), extra_info=json.dumps(params, sort_keys=True))
            memory_checkpoint("Sampled validation classification prediction complete", args)
            calibration = fit_calibration(raw_probabilities, fit_validation_rows, args)
            if calibration:
                calibrated_probabilities = copy_prediction_values(raw_probabilities, args, "gbdt_validation_calibrated_")
                calibrate_probabilities(calibrated_probabilities, calibration)
            else:
                calibrated_probabilities = raw_probabilities

        if args.objective_mode in ("return_regression", "hybrid"):
            regression_model = make_model(kind, params, positive_weight, "return_regression")
            regression_fit_token = start_profile_stage("regression_fit", json.dumps(params, sort_keys=True))
            regression_model.fit(x_train, y_reg_train, feature_names, x_validation, y_reg_validation, args)
            finish_profile_stage(regression_fit_token, rows_processed=len(fit_train_rows), extra_info=json.dumps(params, sort_keys=True))
            memory_checkpoint(
                "LightGBM regression candidate fit complete" if kind == "lightgbm" else "Internal regression candidate fit complete",
                args,
            )
            sampled_regression_token = start_profile_stage("sampled_validation_regression_prediction", json.dumps(params, sort_keys=True))
            regression_predictions = regression_model.predict_values(x_validation)
            finish_profile_stage(sampled_regression_token, rows_processed=len(fit_validation_rows), extra_info=json.dumps(params, sort_keys=True))
            memory_checkpoint("Sampled validation regression prediction complete", args)
            raw_predicted_trade_return = regression_predictions_to_trade_return(regression_predictions, fit_validation_rows, args)
            regression_calibration = fit_regression_calibration(raw_predicted_trade_return, fit_validation_rows, args)
            predicted_trade_return = copy_prediction_values(raw_predicted_trade_return, args, "gbdt_validation_regression_calibrated_")
            apply_regression_calibration(predicted_trade_return, regression_calibration)
            uncertainty_model = fit_uncertainty_model(predicted_trade_return, fit_validation_rows, args)
        else:
            raw_predicted_trade_return = None

        bundle = build_prediction_bundle(
            raw_probabilities,
            calibrated_probabilities,
            predicted_trade_return,
            raw_predicted_trade_return=raw_predicted_trade_return,
            predicted_return_uncertainty=apply_uncertainty_model(predicted_trade_return, uncertainty_model) if predicted_trade_return is not None else None,
            uncertainty_context=uncertainty_model,
        )
        bundle["ev_context"] = fit_ev_payoff_context(fit_validation_rows, bundle, args)
        bundle["hybrid_return_context"] = fit_hybrid_return_context(fit_validation_rows, bundle, args)
        thresholds, fixed_threshold, ignored = selected_thresholds_for_bundle(bundle, fit_validation_rows, args)
        del fixed_threshold, ignored
        selection = tune_threshold(
            fit_validation_rows,
            bundle,
            thresholds,
            args.threshold_objective,
            args.fee,
            args.slippage * args.validation_slippage_multiplier,
            args.min_validation_trades,
            args.max_validation_trades,
            args.min_validation_precision,
            args.profit_safety,
            args.initial_capital,
            args.max_position_fraction,
            args.max_volume_fraction,
            args.max_trades_per_period,
            args.trade_period_minutes,
            args.holding_period_minutes,
            args.trade_selection,
            args.top_k_per_minute,
            args.upside_target,
            args.downside_stop,
            args.ev_safety_margin,
            args.objective_mode,
            score_name,
            args.min_predicted_net_return,
            args.hybrid_min_score,
            args.max_trades_per_day,
            0,
            args.max_losing_trades_per_day,
            args.max_daily_drawdown,
            args.pause_after_drawdown_minutes,
            args.threshold_drawdown_penalty,
            args.threshold_trade_count_penalty,
            args.target_validation_trades,
            args,
        )
        threshold = selection["threshold"]
        metrics = selection["validation_metrics"]
        zero_trade_score = 0.0 if args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev") and args.profit_safety == "strict" else -float("inf")
        score = selection.get(
            "objective_score",
            metrics.get(
                "selected_objective_score",
                threshold_score(
                    metrics,
                    args.threshold_objective,
                    zero_trade_score,
                    args.target_validation_trades,
                    args.min_validation_trades,
                    args.max_validation_trades,
                ),
            ),
        )
        rank = threshold_rank(
            metrics,
            args.threshold_objective,
            zero_trade_score,
            score,
            args.target_validation_trades,
            args.min_validation_trades,
            args.max_validation_trades,
        )
        if best is None or rank > best["rank"]:
            if best is not None:
                old_classification = best.pop("classification_model", None)
                old_regression = best.pop("regression_model", None)
                del old_classification
                del old_regression
            best = {
                "classification_model": classification_model,
                "regression_model": regression_model,
                "params": dict(params),
                "threshold": threshold,
                "validation_metrics": metrics,
                "selection": selection,
                "score": score,
                "base_score": selection.get("base_objective_score", metrics.get("selected_base_objective_score", score)),
                "rank": rank,
                "calibration": calibration,
                "regression_calibration": regression_calibration,
                "uncertainty_model": uncertainty_model,
                "trade_score_name": score_name,
                "selected_score_name": selected_score_name,
                "ev_payoff_info": dict(bundle.get("ev_context") or {}),
                "hybrid_return_info": dict(bundle.get("hybrid_return_context") or {}),
            }
            if classification_model is not None and hasattr(classification_model, "best_iteration") and classification_model.best_iteration():
                best["best_iteration"] = classification_model.best_iteration()
                best["params"]["n_estimators"] = classification_model.best_iteration()
            elif regression_model is not None and hasattr(regression_model, "best_iteration") and regression_model.best_iteration():
                best["best_iteration"] = regression_model.best_iteration()
                best["params"]["n_estimators"] = regression_model.best_iteration()
        else:
            del classification_model
            del regression_model
            calibration = None
        cleanup_prediction_bundle(bundle)
        del bundle
        del calibration
        del regression_calibration
        del uncertainty_model
        gc.collect()
        check_memory_limit("candidate cleanup", args)
    del x_train
    del x_validation
    del y_class_train
    del y_class_validation
    del y_reg_train
    del y_reg_validation
    gc.collect()
    if best is not None and len(fit_validation_rows) != len(validation_rows) and not args.skip_full_validation_retune:
        retune_token = start_profile_stage("full_validation_retune", "rows={}".format(len(validation_rows)))
        bundle = prediction_bundle_for_models(best, validation_rows, kind, args, "full validation retune")
        bundle["ev_context"] = fit_ev_payoff_context(validation_rows, bundle, args)
        bundle["hybrid_return_context"] = fit_hybrid_return_context(validation_rows, bundle, args)
        thresholds, fixed_threshold, ignored = selected_thresholds_for_bundle(bundle, validation_rows, args)
        del fixed_threshold, ignored
        selection = tune_threshold(
            validation_rows,
            bundle,
            thresholds,
            args.threshold_objective,
            args.fee,
            args.slippage * args.validation_slippage_multiplier,
            args.min_validation_trades,
            args.max_validation_trades,
            args.min_validation_precision,
            args.profit_safety,
            args.initial_capital,
            args.max_position_fraction,
            args.max_volume_fraction,
            args.max_trades_per_period,
            args.trade_period_minutes,
            args.holding_period_minutes,
            args.trade_selection,
            args.top_k_per_minute,
            args.upside_target,
            args.downside_stop,
            args.ev_safety_margin,
            args.objective_mode,
            score_name,
            args.min_predicted_net_return,
            args.hybrid_min_score,
            args.max_trades_per_day,
            0,
            args.max_losing_trades_per_day,
            args.max_daily_drawdown,
            args.pause_after_drawdown_minutes,
            args.threshold_drawdown_penalty,
            args.threshold_trade_count_penalty,
            args.target_validation_trades,
            args,
        )
        best["threshold"] = selection["threshold"]
        metrics = selection["validation_metrics"]
        best["validation_metrics"] = metrics
        best["selection"] = selection
        best["score"] = selection.get(
            "objective_score",
            metrics.get(
                "selected_objective_score",
                threshold_score(
                    metrics,
                    args.threshold_objective,
                    0.0 if args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev") and args.profit_safety == "strict" else -float("inf"),
                    args.target_validation_trades,
                    args.min_validation_trades,
                    args.max_validation_trades,
                ),
            ),
        )
        best["base_score"] = selection.get("base_objective_score", metrics.get("selected_base_objective_score", best["score"]))
        best["ev_payoff_info"] = dict(bundle.get("ev_context") or {})
        best["hybrid_return_info"] = dict(bundle.get("hybrid_return_context") or {})
        best["rank"] = threshold_rank(
            metrics,
            args.threshold_objective,
            0.0 if args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev") and args.profit_safety == "strict" else -float("inf"),
            best["score"],
            args.target_validation_trades,
            args.min_validation_trades,
            args.max_validation_trades,
        )
        cleanup_prediction_bundle(bundle)
        del bundle
        gc.collect()
        finish_profile_stage(retune_token, rows_processed=len(validation_rows))
        memory_checkpoint("Full validation threshold retune complete", args)
    if best is not None:
        meta_validation_rows = validation_rows if len(fit_validation_rows) != len(validation_rows) and not args.skip_full_validation_retune else fit_validation_rows
        meta_bundle = prediction_bundle_for_models(best, meta_validation_rows, kind, args, "meta filter validation")
        best["symbol_filter_info"] = fit_symbol_validation_filter(
            meta_validation_rows,
            meta_bundle,
            best["threshold"],
            args,
            best.get("trade_score_name", score_name),
        )
        if best["symbol_filter_info"].get("enabled"):
            print(
                "Validation symbol filter kept {}/{} symbols".format(
                    len(best["symbol_filter_info"].get("allowed_symbols", [])),
                    int(best["symbol_filter_info"].get("total_symbols", 0)),
                ),
                flush=True,
            )
            best["symbol_filter_info"], recalibrated_symbol_selection = recalibrate_symbol_filter_validation(
                meta_validation_rows,
                meta_bundle,
                best["threshold"],
                args,
                best["selection"],
                best["symbol_filter_info"],
                best.get("selected_score_name", selected_score_name),
            )
            if recalibrated_symbol_selection is not best["selection"]:
                best["validation_metrics"] = recalibrated_symbol_selection["validation_metrics"]
                best["selection"] = recalibrated_symbol_selection
                best["score"] = recalibrated_symbol_selection.get("objective_score", best.get("score", 0.0))
                best["base_score"] = recalibrated_symbol_selection.get("base_objective_score", best.get("base_score", best["score"]))
                best["rank"] = threshold_rank(
                    best["validation_metrics"],
                    args.threshold_objective,
                    0.0 if args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev") and args.profit_safety == "strict" else -float("inf"),
                    best["score"],
                    args.target_validation_trades,
                    args.min_validation_trades,
                    args.max_validation_trades,
                )
        best["meta_filter_info"] = fit_meta_filter(
            meta_validation_rows,
            meta_bundle,
            best["threshold"],
            args,
            best["symbol_filter_info"],
        )
        best["meta_filter_info"], recalibrated_selection = recalibrate_meta_filter_validation(
            meta_validation_rows,
            meta_bundle,
            best["threshold"],
            args,
            best["selection"],
            best["meta_filter_info"],
            best["symbol_filter_info"],
            best.get("selected_score_name", selected_score_name_for_mode(args)),
        )
        if recalibrated_selection is not best["selection"]:
            best["validation_metrics"] = recalibrated_selection["validation_metrics"]
            best["selection"] = recalibrated_selection
            best["score"] = recalibrated_selection.get("objective_score", best.get("score", 0.0))
            best["base_score"] = recalibrated_selection.get("base_objective_score", best.get("base_score", best["score"]))
            best["rank"] = threshold_rank(
                best["validation_metrics"],
                args.threshold_objective,
                0.0 if args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev") and args.profit_safety == "strict" else -float("inf"),
                best["score"],
                args.target_validation_trades,
                args.min_validation_trades,
                args.max_validation_trades,
            )
        cleanup_prediction_bundle(meta_bundle)
        del meta_bundle
        calibration = best.get("calibration")
        regression_calibration = best.get("regression_calibration") or {}
        uncertainty_model = best.get("uncertainty_model") or {}
        ev_payoff_info = best.get("ev_payoff_info") or {}
        hybrid_return_info = best.get("hybrid_return_info") or {}
        meta_filter_info = best.get("meta_filter_info") or {}
        symbol_filter_info = best.get("symbol_filter_info") or {}
        best["calibration_info"] = {
            "calibration": calibration.get("mode", "none") if calibration else "none",
            "calibration_a": calibration.get("a", 0.0) if calibration else 0.0,
            "calibration_b": calibration.get("b", 0.0) if calibration else 0.0,
            "validation_brier_before": calibration.get("validation_brier_before", 0.0) if calibration else 0.0,
            "validation_brier_after": calibration.get("validation_brier_after", 0.0) if calibration else 0.0,
            "calibration_rows": calibration.get("rows", 0) if calibration else 0,
            "regression_calibration": regression_calibration.get("mode", "none"),
            "regression_calibration_a": regression_calibration.get("a", 1.0),
            "regression_calibration_b": regression_calibration.get("b", 0.0),
            "regression_calibration_rows": regression_calibration.get("rows", 0),
            "regression_calibration_rmse_before": regression_calibration.get("regression_calibration_rmse_before", 0.0),
            "regression_calibration_rmse_after": regression_calibration.get("regression_calibration_rmse_after", 0.0),
            "regression_calibration_mae_before": regression_calibration.get("regression_calibration_mae_before", 0.0),
            "regression_calibration_mae_after": regression_calibration.get("regression_calibration_mae_after", 0.0),
            "hybrid_return_combination": hybrid_return_info.get("hybrid_return_combination", getattr(args, "hybrid_return_combination", "probability_times_return")),
            "hybrid_min_probability": hybrid_return_info.get("hybrid_min_probability", getattr(args, "hybrid_min_probability", 0.0)),
            "hybrid_score_mode": args.hybrid_score_mode,
            "hybrid_uncertainty_method": uncertainty_model.get("mode", "none"),
            "hybrid_uncertainty_penalty": args.hybrid_uncertainty_penalty,
            "hybrid_uncertainty_global_std": uncertainty_model.get("global_std", 0.0),
            "hybrid_uncertainty_rows": uncertainty_model.get("rows", 0),
            "conditional_expected_win_return": hybrid_return_info.get("conditional_expected_win_return", 0.0),
            "conditional_expected_loss_return": hybrid_return_info.get("conditional_expected_loss_return", 0.0),
            "conditional_payoff_rows": hybrid_return_info.get("conditional_payoff_rows", 0),
            "conditional_payoff_positive_rows": hybrid_return_info.get("conditional_payoff_positive_rows", 0),
            "conditional_payoff_negative_rows": hybrid_return_info.get("conditional_payoff_negative_rows", 0),
            "conditional_payoff_source": hybrid_return_info.get("conditional_payoff_source", "not_used"),
            "ev_payoff_mode": ev_payoff_info.get("ev_payoff_mode", getattr(args, "ev_payoff_mode", "fixed_targets")),
            "ev_expected_win_return": ev_payoff_info.get("ev_expected_win_return", getattr(args, "effective_upside_target", getattr(args, "upside_target", 0.05))),
            "ev_expected_loss_return": ev_payoff_info.get("ev_expected_loss_return", -getattr(args, "effective_downside_stop", getattr(args, "downside_stop", 0.02))),
            "ev_payoff_rows": ev_payoff_info.get("ev_payoff_rows", 0),
            "ev_payoff_positive_rows": ev_payoff_info.get("ev_payoff_positive_rows", 0),
            "ev_payoff_negative_rows": ev_payoff_info.get("ev_payoff_negative_rows", 0),
            "ev_payoff_source": ev_payoff_info.get("ev_payoff_source", "fixed_targets"),
            "meta_filter": meta_filter_info.get("mode", "none"),
            "meta_filter_rows": meta_filter_info.get("rows", 0),
            "meta_filter_positive_rate": meta_filter_info.get("positive_rate", 0.0),
            "meta_filter_auc_or_accuracy": meta_filter_info.get("auc", meta_filter_info.get("accuracy", 0.0)),
            "meta_filter_enabled": 1 if meta_filter_info.get("enabled") else 0,
            "meta_filter_disabled_reason": meta_filter_info.get("disabled_reason", ""),
            "meta_filter_validation_trade_retention": meta_filter_info.get("validation_trade_retention", 0.0),
            "symbol_validation_filter": symbol_filter_info.get("mode", "none"),
            "symbol_filter_stage": symbol_filter_info.get("stage", getattr(args, "symbol_filter_stage", "executed")),
            "symbol_filter_min_candidates": symbol_filter_info.get("symbol_filter_min_candidates", getattr(args, "symbol_filter_min_candidates", 0)),
            "symbol_filter_min_executed": symbol_filter_info.get("symbol_filter_min_executed", getattr(args, "symbol_filter_min_executed", 0)),
            "symbol_filter_candidate_weight": symbol_filter_info.get("symbol_filter_candidate_weight", getattr(args, "symbol_filter_candidate_weight", 0.5)),
            "symbol_filter_executed_weight": symbol_filter_info.get("symbol_filter_executed_weight", getattr(args, "symbol_filter_executed_weight", 0.5)),
            "symbol_filter_shrinkage": symbol_filter_info.get("symbol_filter_shrinkage", getattr(args, "symbol_filter_shrinkage", 50.0)),
            "symbol_filter_enabled": 1 if symbol_filter_info.get("enabled") else 0,
            "symbol_filter_allowed_symbols": len(symbol_filter_info.get("allowed_symbols", [])),
            "symbol_filter_total_symbols": symbol_filter_info.get("total_symbols", 0),
            "symbols_blocked_count": symbol_filter_info.get("symbols_blocked_count", 0),
            "symbols_allowed_count": symbol_filter_info.get("symbols_allowed_count", 0),
            "symbol_filter_disabled_reason": symbol_filter_info.get("disabled_reason", ""),
            "symbol_filter_validation_trade_retention": symbol_filter_info.get("validation_trade_retention", 0.0),
            "ensemble_windows": ",".join(str(value) for value in getattr(args, "ensemble_window_list", [])),
            "ensemble_model_count": len(best.get("ensemble_members", [])),
            "ensemble_enabled": 1 if best.get("ensemble_members") else 0,
            "regression_target": args.regression_target,
            "risk_adjusted_return_feature": regression_target_feature_name(args) or "",
            "dynamic_hybrid_thresholds": args.dynamic_hybrid_thresholds,
        }
        best["validation_metrics"]["calibration_info"] = dict(best["calibration_info"])
    return best


def write_predictions(path, rows, predictions, threshold, model_name,
                      append=False, output_mode="all",
                      initial_capital=10000.0, max_position_fraction=0.10,
                      max_volume_fraction=0.01, max_trades_per_period=10,
                      trade_period_minutes=60, holding_period_minutes=5,
                      fee=0.0, slippage=0.0, threshold_objective="avg_profit",
                      trade_selection="threshold", top_k_per_minute=3,
                      upside_target=0.05, downside_stop=0.02, ev_safety_margin=0.0,
                      objective_mode="classification", trade_score_name="probability",
                      min_predicted_net_return=0.0, hybrid_min_score=0.0,
                      max_trades_per_day=0, max_trades_per_fold=0,
                      max_losing_trades_per_day=0, max_daily_drawdown=0.0,
                      pause_after_drawdown_minutes=0, hybrid_runtime_args=None,
                      symbol_filter_info=None):
    bundle = normalize_prediction_bundle(predictions)
    trade_day_cache = {}
    hybrid_score_mode = getattr(hybrid_runtime_args, "hybrid_score_mode", "basic") if hybrid_runtime_args is not None else "basic"
    hybrid_uncertainty_penalty = getattr(hybrid_runtime_args, "hybrid_uncertainty_penalty", 0.0) if hybrid_runtime_args is not None else 0.0
    meta_filter_min_probability = getattr(hybrid_runtime_args, "meta_filter_min_probability", 0.0) if hybrid_runtime_args is not None else 0.0
    dynamic_hybrid_mode = getattr(hybrid_runtime_args, "dynamic_hybrid_thresholds", "none") if hybrid_runtime_args is not None else "none"
    effective_hybrid_thresholds, regime_buckets = compute_dynamic_hybrid_thresholds(
        rows,
        hybrid_runtime_args,
        max(float(threshold), float(hybrid_min_score)),
    ) if objective_mode == "hybrid" and hybrid_runtime_args is not None else (None, None)
    with open(path, "a" if append else "w", newline="") as handle:
        writer = csv.writer(handle)
        if not append:
            writer.writerow([
                "symbol",
                "month",
                "month_index",
                "open_time",
                "label",
                "probability",
                "calibrated_probability",
                "hybrid_return_combination",
                "hybrid_min_probability",
                "raw_predicted_trade_return",
                "predicted_trade_return",
                "calibrated_predicted_trade_return",
                "predicted_net_return",
                "predicted_return_uncertainty",
                "base_hybrid_score",
                "hybrid_score",
                "hybrid_score_basic",
                "hybrid_score_risk_adjusted",
                "conditional_expected_win_return",
                "conditional_expected_loss_return",
                "conditional_payoff_rows",
                "conditional_payoff_positive_rows",
                "conditional_payoff_negative_rows",
                "conditional_payoff_source",
                "selected_threshold",
                "effective_hybrid_min_score",
                "dynamic_threshold_mode",
                "regime_bucket",
                "raw_signal",
                "predicted",
                "position_size",
                "forward_return",
                "trade_return",
                "max_future_high_return",
                "max_future_low_return",
                "ev_payoff_mode",
                "ev_expected_win_return",
                "ev_expected_loss_return",
                "expected_value_fixed_targets",
                "expected_value_empirical",
                "expected_value_predicted_return",
                "expected_value",
                "trade_score",
                "trade_score_name",
                "regression_target_name",
                "selection_rank",
                "selected_by_topk",
                "meta_probability",
                "meta_filter_min_probability",
                "selected_by_meta_filter",
                "trade_day",
                "blocked_by_daily_trade_limit",
                "blocked_by_fold_trade_limit",
                "blocked_by_daily_loss_limit",
                "blocked_by_daily_drawdown_limit",
                "model_name",
            ])
        if output_mode == "none":
            log_memory("Prediction CSV output skipped (--prediction-output-mode none)")
            return
        execution = portfolio_execution(
            rows,
            bundle,
            threshold,
            fee,
            slippage,
            initial_capital,
            max_position_fraction,
            max_volume_fraction,
            max_trades_per_period,
            trade_period_minutes,
            holding_period_minutes,
            threshold_objective,
            trade_selection,
            top_k_per_minute,
            upside_target,
            downside_stop,
            ev_safety_margin,
            objective_mode,
            trade_score_name,
            min_predicted_net_return,
            hybrid_min_score,
            max_trades_per_day,
            max_trades_per_fold,
            max_losing_trades_per_day,
            max_daily_drawdown,
            pause_after_drawdown_minutes,
            capture_blocked_details=(output_mode == "all"),
            hybrid_runtime_args=hybrid_runtime_args,
            symbol_filter_info=symbol_filter_info,
        )
        executed = execution["executed"]
        raw_selected = execution["raw_selected"]
        executed_trade_day_ids = execution.get("executed_trade_day_ids", {})
        if is_compact_rows(rows):
            table = rows.table
            threshold_is_list = isinstance(threshold, list) or (np is not None and isinstance(threshold, np.ndarray))
            output_indices = sorted(executed) if output_mode == "trades" else range(len(rows))
            for index in output_indices:
                position = index if rows.indices is None else int(rows.indices[index])
                row_threshold = threshold[index] if threshold_is_list else threshold
                probability = raw_probability_value(bundle, index)
                calibrated_probability = calibrated_probability_value(bundle, index)
                predicted_trade_return = predicted_trade_return_value(bundle, index)
                raw_predicted_trade_return = raw_predicted_trade_return_value(bundle, index)
                raw_signal = index in raw_selected
                symbol_code = int(table.symbol_codes[position])
                predicted = index in executed
                block_flags = execution["blocked_flags"].get(index, {
                    "blocked_by_daily_trade_limit": 0,
                    "blocked_by_fold_trade_limit": 0,
                    "blocked_by_daily_loss_limit": 0,
                    "blocked_by_daily_drawdown_limit": 0,
                })
                ev_details = expected_value_details_for_bundle(bundle, index, upside_target, downside_stop, fee, slippage, hybrid_runtime_args)
                hybrid_details = hybrid_score_details_for_bundle(
                    bundle,
                    index,
                    fee,
                    slippage,
                    hybrid_score_mode,
                    hybrid_uncertainty_penalty,
                    hybrid_runtime_args,
                    upside_target,
                    downside_stop,
                )
                expected_value = ev_details["expected_value"]
                predicted_net_return = hybrid_details["calibrated_predicted_trade_return"] - fee - slippage
                hybrid_score_basic = hybrid_details["base_hybrid_score"]
                hybrid_score_risk_adjusted = hybrid_score_value(bundle, index, fee, slippage, "risk_adjusted", hybrid_uncertainty_penalty, hybrid_runtime_args, upside_target, downside_stop)
                hybrid_score = hybrid_details["hybrid_score"]
                effective_threshold = float(effective_hybrid_thresholds[index]) if effective_hybrid_thresholds is not None else (
                    max(float(row_threshold), float(hybrid_min_score)) if objective_mode == "hybrid" else float(row_threshold)
                )
                regime_bucket = regime_buckets[index] if regime_buckets is not None else ""
                writer.writerow([
                    table.symbols[symbol_code],
                    table.months[int(table.month_codes[position])],
                    int(table.month_indices[position]),
                    int(table.open_times[position]),
                    int(table.labels[position]),
                    "{:.12g}".format(probability),
                    "{:.12g}".format(calibrated_probability),
                    hybrid_details["hybrid_return_combination"],
                    "{:.12g}".format(hybrid_details["hybrid_min_probability"]),
                    "{:.12g}".format(raw_predicted_trade_return),
                    "{:.12g}".format(predicted_trade_return),
                    "{:.12g}".format(hybrid_details["calibrated_predicted_trade_return"]),
                    "{:.12g}".format(predicted_net_return),
                    "{:.12g}".format(predicted_return_uncertainty_value(bundle, index)),
                    "{:.12g}".format(hybrid_details["base_hybrid_score"]),
                    "{:.12g}".format(hybrid_score),
                    "{:.12g}".format(hybrid_score_basic),
                    "{:.12g}".format(hybrid_score_risk_adjusted),
                    "{:.12g}".format(hybrid_details["conditional_expected_win_return"]),
                    "{:.12g}".format(hybrid_details["conditional_expected_loss_return"]),
                    int(hybrid_details["conditional_payoff_rows"]),
                    int(hybrid_details["conditional_payoff_positive_rows"]),
                    int(hybrid_details["conditional_payoff_negative_rows"]),
                    hybrid_details["conditional_payoff_source"],
                    "{:.12g}".format(row_threshold),
                    "{:.12g}".format(effective_threshold),
                    dynamic_hybrid_mode,
                    regime_bucket,
                    1 if raw_signal else 0,
                    1 if predicted else 0,
                    "{:.12g}".format(executed.get(index, 0.0)),
                    "{:.12g}".format(float(table.forward_returns[position])),
                    "{:.12g}".format(float(table.trade_returns[position])),
                    "{:.12g}".format(float(table.max_future_high_returns[position])),
                    "{:.12g}".format(float(table.max_future_low_returns[position])),
                    ev_details["ev_payoff_mode"],
                    "{:.12g}".format(ev_details["ev_expected_win_return"]),
                    "{:.12g}".format(ev_details["ev_expected_loss_return"]),
                    "{:.12g}".format(ev_details["expected_value_fixed_targets"]),
                    "{:.12g}".format(ev_details["expected_value_empirical"]),
                    "{:.12g}".format(ev_details["expected_value_predicted_return"]),
                    "{:.12g}".format(expected_value),
                    "{:.12g}".format(execution["executed_trade_scores"].get(index, trade_score_value(bundle, index, trade_score_name, upside_target, downside_stop, fee, slippage, hybrid_score_mode, hybrid_uncertainty_penalty, hybrid_runtime_args))),
                    trade_score_name,
                    getattr(hybrid_runtime_args, "regression_target", "trade_return") if hybrid_runtime_args is not None else "trade_return",
                    execution["executed_selection_ranks"].get(index, 0),
                    execution["executed_selected_by_topk"].get(index, 0),
                    "{:.12g}".format(meta_probability_value(bundle, index)),
                    "{:.12g}".format(meta_filter_min_probability),
                    execution["executed_selected_by_meta_filter"].get(index, 0),
                    trade_day_cache.setdefault(
                        executed_trade_day_ids.get(index, minute_day_id(open_time_minute(table.open_times[position]))),
                        day_id_to_string(executed_trade_day_ids.get(index, minute_day_id(open_time_minute(table.open_times[position])))),
                    ),
                    block_flags["blocked_by_daily_trade_limit"],
                    block_flags["blocked_by_fold_trade_limit"],
                    block_flags["blocked_by_daily_loss_limit"],
                    block_flags["blocked_by_daily_drawdown_limit"],
                    model_name,
                ])
            log_memory("Prediction CSV output complete: {}".format(path))
            return

        threshold_is_list = isinstance(threshold, list) or (np is not None and isinstance(threshold, np.ndarray))
        output_indices = sorted(executed) if output_mode == "trades" else range(len(rows))
        for index in output_indices:
            row = rows[index]
            probability = raw_probability_value(bundle, index)
            calibrated_probability = calibrated_probability_value(bundle, index)
            predicted_trade_return = predicted_trade_return_value(bundle, index)
            raw_predicted_trade_return = raw_predicted_trade_return_value(bundle, index)
            row_threshold = threshold[index] if threshold_is_list else threshold
            raw_signal = index in raw_selected
            predicted = index in executed
            block_flags = execution["blocked_flags"].get(index, {
                "blocked_by_daily_trade_limit": 0,
                "blocked_by_fold_trade_limit": 0,
                "blocked_by_daily_loss_limit": 0,
                "blocked_by_daily_drawdown_limit": 0,
            })
            ev_details = expected_value_details_for_bundle(bundle, index, upside_target, downside_stop, fee, slippage, hybrid_runtime_args)
            hybrid_details = hybrid_score_details_for_bundle(
                bundle,
                index,
                fee,
                slippage,
                hybrid_score_mode,
                hybrid_uncertainty_penalty,
                hybrid_runtime_args,
                upside_target,
                downside_stop,
            )
            expected_value = ev_details["expected_value"]
            predicted_net_return = hybrid_details["calibrated_predicted_trade_return"] - fee - slippage
            hybrid_score_basic = hybrid_details["base_hybrid_score"]
            hybrid_score_risk_adjusted = hybrid_score_value(bundle, index, fee, slippage, "risk_adjusted", hybrid_uncertainty_penalty, hybrid_runtime_args, upside_target, downside_stop)
            hybrid_score = hybrid_details["hybrid_score"]
            effective_threshold = float(effective_hybrid_thresholds[index]) if effective_hybrid_thresholds is not None else (
                max(float(row_threshold), float(hybrid_min_score)) if objective_mode == "hybrid" else float(row_threshold)
            )
            regime_bucket = regime_buckets[index] if regime_buckets is not None else ""
            writer.writerow([
                row.symbol,
                row.month,
                row.month_index,
                row.open_time,
                row.label,
                "{:.12g}".format(probability),
                "{:.12g}".format(calibrated_probability),
                hybrid_details["hybrid_return_combination"],
                "{:.12g}".format(hybrid_details["hybrid_min_probability"]),
                "{:.12g}".format(raw_predicted_trade_return),
                "{:.12g}".format(predicted_trade_return),
                "{:.12g}".format(hybrid_details["calibrated_predicted_trade_return"]),
                "{:.12g}".format(predicted_net_return),
                "{:.12g}".format(predicted_return_uncertainty_value(bundle, index)),
                "{:.12g}".format(hybrid_details["base_hybrid_score"]),
                "{:.12g}".format(hybrid_score),
                "{:.12g}".format(hybrid_score_basic),
                "{:.12g}".format(hybrid_score_risk_adjusted),
                "{:.12g}".format(hybrid_details["conditional_expected_win_return"]),
                "{:.12g}".format(hybrid_details["conditional_expected_loss_return"]),
                int(hybrid_details["conditional_payoff_rows"]),
                int(hybrid_details["conditional_payoff_positive_rows"]),
                int(hybrid_details["conditional_payoff_negative_rows"]),
                hybrid_details["conditional_payoff_source"],
                "{:.12g}".format(row_threshold),
                "{:.12g}".format(effective_threshold),
                dynamic_hybrid_mode,
                regime_bucket,
                1 if raw_signal else 0,
                1 if predicted else 0,
                "{:.12g}".format(executed.get(index, 0.0)),
                "{:.12g}".format(row.forward_return),
                "{:.12g}".format(row.trade_return),
                "{:.12g}".format(row.max_future_high_return),
                "{:.12g}".format(row.max_future_low_return),
                ev_details["ev_payoff_mode"],
                "{:.12g}".format(ev_details["ev_expected_win_return"]),
                "{:.12g}".format(ev_details["ev_expected_loss_return"]),
                "{:.12g}".format(ev_details["expected_value_fixed_targets"]),
                "{:.12g}".format(ev_details["expected_value_empirical"]),
                "{:.12g}".format(ev_details["expected_value_predicted_return"]),
                "{:.12g}".format(expected_value),
                "{:.12g}".format(execution["executed_trade_scores"].get(index, trade_score_value(bundle, index, trade_score_name, upside_target, downside_stop, fee, slippage, hybrid_score_mode, hybrid_uncertainty_penalty, hybrid_runtime_args))),
                trade_score_name,
                getattr(hybrid_runtime_args, "regression_target", "trade_return") if hybrid_runtime_args is not None else "trade_return",
                execution["executed_selection_ranks"].get(index, 0),
                execution["executed_selected_by_topk"].get(index, 0),
                "{:.12g}".format(meta_probability_value(bundle, index)),
                "{:.12g}".format(meta_filter_min_probability),
                execution["executed_selected_by_meta_filter"].get(index, 0),
                trade_day_cache.setdefault(
                    executed_trade_day_ids.get(index, minute_day_id(open_time_minute(row.open_time))),
                    day_id_to_string(executed_trade_day_ids.get(index, minute_day_id(open_time_minute(row.open_time)))),
                ),
                block_flags["blocked_by_daily_trade_limit"],
                block_flags["blocked_by_fold_trade_limit"],
                block_flags["blocked_by_daily_loss_limit"],
                block_flags["blocked_by_daily_drawdown_limit"],
                model_name,
            ])
    log_memory("Prediction CSV output complete: {}".format(path))


METRIC_COLUMNS = [
    "model",
    "split",
    "split_mode",
    "objective_mode",
    "trade_score",
    "trade_score_name",
    "train_ratio",
    "validation_ratio",
    "test_ratio",
    "threshold_objective",
    "selected_threshold",
    "selected_score_name",
    "selected_score_threshold",
    "selected_objective_score",
    "selected_base_objective_score",
    "selected_penalized_objective_score",
    "selected_validation_max_drawdown",
    "selected_validation_trade_count",
    "selected_validation_raw_signal_count",
    "selected_validation_portfolio_profit",
    "selected_validation_portfolio_return",
    "selected_validation_precision",
    "selected_validation_recall",
    "selected_validation_active_days",
    "selected_validation_profit_per_active_day",
    "selected_validation_average_profit_after_fee_and_slippage",
    "selected_validation_total_profit_after_fee_and_slippage",
    "selected_threshold_tie_rank_reason",
    "train_rows",
    "validation_rows",
    "test_rows",
    "auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "predicted_trades",
    "true_positive_rows",
    "false_positive_rows",
    "raw_signal_trades",
    "raw_true_positive_rows",
    "raw_false_positive_rows",
    "raw_precision",
    "raw_recall",
    "raw_f1",
    "win_rate",
    "average_forward_return",
    "median_forward_return",
    "average_trade_return",
    "median_trade_return",
    "average_expected_value",
    "median_expected_value",
    "min_expected_value",
    "selected_ev_safety_margin",
    "average_max_favorable_excursion",
    "average_max_adverse_excursion",
    "average_profit_after_fee",
    "average_profit_after_fee_and_slippage",
    "total_profit_after_fee",
    "total_profit_after_fee_and_slippage",
    "profit_factor",
    "max_drawdown",
    "initial_capital",
    "ending_capital",
    "portfolio_profit",
    "portfolio_return",
    "average_position_size",
    "median_position_size",
    "trades_per_day",
    "trades_per_month",
    "trades_per_active_day",
    "active_days",
    "profitable_days",
    "losing_days",
    "worst_day_profit",
    "best_day_profit",
    "average_profit_per_trade",
    "worst_trade",
    "max_capital_drawdown",
    "daily_trade_limit_blocked",
    "fold_trade_limit_blocked",
    "daily_loss_limit_blocked",
    "daily_drawdown_limit_blocked",
    "blocked_trades_total",
    "blocked_by_trade_frequency",
    "blocked_by_drawdown",
    "max_trades_in_any_day",
    "max_trades_in_any_fold",
    "normalized_microsecond_open_times",
    "max_position_fraction",
    "max_volume_fraction",
    "max_trades_per_period",
    "max_trades_per_day",
    "max_trades_per_fold",
    "max_losing_trades_per_day",
    "max_daily_drawdown",
    "pause_after_drawdown_minutes",
    "trade_period_minutes",
    "holding_period_minutes",
    "min_validation_trades",
    "max_validation_trades",
    "min_validation_precision",
    "min_selected_threshold",
    "min_predicted_net_return",
    "hybrid_min_score",
    "threshold_drawdown_penalty",
    "threshold_trade_count_penalty",
    "target_validation_trades",
    "threshold_tiebreaker",
    "threshold_tie_epsilon",
    "threshold_target_trades",
    "threshold_target_active_days",
    "overactive_trade_threshold",
    "profit_safety",
    "adaptive_thresholds",
    "trade_selection",
    "top_k_per_minute",
    "calibration",
    "calibration_a",
    "calibration_b",
    "validation_brier_before",
    "validation_brier_after",
    "calibration_rows",
    "regression_calibration",
    "regression_calibration_a",
    "regression_calibration_b",
    "regression_calibration_rows",
    "regression_calibration_rmse_before",
    "regression_calibration_rmse_after",
    "regression_calibration_mae_before",
    "regression_calibration_mae_after",
    "regression_target",
    "regression_clip_min",
    "regression_clip_max",
    "risk_adjusted_return_feature",
    "hybrid_return_combination",
    "hybrid_min_probability",
    "hybrid_score_mode",
    "hybrid_uncertainty_method",
    "hybrid_uncertainty_penalty",
    "hybrid_uncertainty_global_std",
    "hybrid_uncertainty_rows",
    "conditional_expected_win_return",
    "conditional_expected_loss_return",
    "conditional_payoff_rows",
    "conditional_payoff_positive_rows",
    "conditional_payoff_negative_rows",
    "conditional_payoff_source",
    "ev_payoff_mode",
    "ev_expected_win_return",
    "ev_expected_loss_return",
    "ev_payoff_rows",
    "ev_payoff_positive_rows",
    "ev_payoff_negative_rows",
    "ev_payoff_source",
    "dynamic_hybrid_thresholds",
    "meta_filter",
    "meta_filter_rows",
    "meta_filter_positive_rate",
    "meta_filter_auc_or_accuracy",
    "meta_filter_enabled",
    "meta_filter_disabled_reason",
    "meta_filter_validation_trade_retention",
    "symbol_validation_filter",
    "symbol_filter_stage",
    "symbol_filter_min_candidates",
    "symbol_filter_min_executed",
    "symbol_filter_candidate_weight",
    "symbol_filter_executed_weight",
    "symbol_filter_shrinkage",
    "symbol_filter_enabled",
    "symbol_filter_allowed_symbols",
    "symbol_filter_total_symbols",
    "symbols_blocked_count",
    "symbols_allowed_count",
    "symbol_filter_disabled_reason",
    "symbol_filter_validation_trade_retention",
    "ensemble_windows",
    "ensemble_model_count",
    "ensemble_enabled",
    "validation_predicted_trades",
    "validation_precision",
    "validation_recall",
    "validation_average_profit_after_fee_and_slippage",
    "validation_total_profit_after_fee_and_slippage",
    "validation_portfolio_profit",
    "validation_portfolio_return",
    "inactive_blocker_source",
    "inactive_blocker_metric",
    "inactive_blocker_threshold",
    "inactive_blocker_best_score",
    "inactive_blocker_gap",
    "inactive_closest_symbol",
    "inactive_promising_fold",
    "walkforward_total_folds",
    "walkforward_profitable_folds",
    "walkforward_profitable_fold_rate",
    "walkforward_median_portfolio_return",
    "walkforward_mean_portfolio_return",
    "walkforward_min_portfolio_return",
    "walkforward_max_portfolio_return",
    "walkforward_total_portfolio_profit",
    "walkforward_total_predicted_trades",
    "walkforward_mean_precision",
    "walkforward_median_precision",
    "walkforward_max_drawdown_mean",
    "walkforward_max_drawdown_worst",
    "selected_objective_finite_folds",
    "selected_objective_nonfinite_folds",
    "selected_no_trade_folds",
    "mean_selected_validation_trade_count",
    "mean_selected_validation_max_drawdown",
    "mean_selected_validation_active_days",
    "mean_selected_validation_profit_per_active_day",
    "mean_selected_base_objective_score",
    "mean_selected_penalized_objective_score",
    "active_fold_count",
    "inactive_fold_count",
    "active_fold_rate",
    "active_profitable_fold_count",
    "active_losing_fold_count",
    "active_profitable_fold_rate",
    "profit_per_active_fold",
    "median_active_fold_return",
    "mean_active_fold_return",
    "worst_active_fold_return",
    "best_active_fold_return",
    "overactive_losing_folds",
    "overactive_losing_fold_rate",
    "avg_trades_in_losing_active_folds",
    "avg_trades_in_profitable_active_folds",
    "acceptance_tier",
    "accepted",
    "failed_acceptance_checks",
    "rejection_reason",
    "strategy_strength",
    "max_rss_gb_observed",
]


def metrics_record(model_name, split, objective, threshold, train_rows, validation_rows, test_rows,
                   metrics, args, validation_metrics=None, selected_objective_score=0.0):
    validation_metrics = validation_metrics if isinstance(validation_metrics, dict) else {}
    selected_score_name = validation_metrics.get("selected_score_name", selected_score_name_for_mode(args))
    resolved_selected_objective_score = validation_metrics.get("selected_objective_score", selected_objective_score)
    record = {
        "model": model_name,
        "split": split,
        "split_mode": args.split_mode,
        "objective_mode": args.objective_mode,
        "trade_score": score_name_for_args(args),
        "trade_score_name": score_name_for_args(args),
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "threshold_objective": objective,
        "selected_threshold": threshold,
        "selected_score_name": selected_score_name,
        "selected_score_threshold": validation_metrics.get("selected_score_threshold", threshold),
        "selected_objective_score": resolved_selected_objective_score,
        "selected_base_objective_score": validation_metrics.get("selected_base_objective_score", 0.0),
        "selected_penalized_objective_score": validation_metrics.get("selected_penalized_objective_score", resolved_selected_objective_score),
        "selected_validation_max_drawdown": validation_metrics.get("selected_validation_max_drawdown", 0.0),
        "selected_validation_trade_count": validation_metrics.get("selected_validation_trade_count", 0),
        "selected_validation_raw_signal_count": validation_metrics.get("selected_validation_raw_signal_count", 0),
        "selected_validation_portfolio_profit": validation_metrics.get("selected_validation_portfolio_profit", 0.0),
        "selected_validation_portfolio_return": validation_metrics.get("selected_validation_portfolio_return", 0.0),
        "selected_validation_precision": validation_metrics.get("selected_validation_precision", 0.0),
        "selected_validation_recall": validation_metrics.get("selected_validation_recall", 0.0),
        "selected_validation_active_days": validation_metrics.get("selected_validation_active_days", 0),
        "selected_validation_profit_per_active_day": validation_metrics.get("selected_validation_profit_per_active_day", 0.0),
        "selected_validation_average_profit_after_fee_and_slippage": validation_metrics.get(
            "selected_validation_average_profit_after_fee_and_slippage",
            0.0,
        ),
        "selected_validation_total_profit_after_fee_and_slippage": validation_metrics.get(
            "selected_validation_total_profit_after_fee_and_slippage",
            0.0,
        ),
        "selected_threshold_tie_rank_reason": validation_metrics.get("selected_threshold_tie_rank_reason", ""),
        "selected_objective_finite_folds": 1 if math.isfinite(float(resolved_selected_objective_score)) else 0,
        "selected_objective_nonfinite_folds": 0 if math.isfinite(float(resolved_selected_objective_score)) else 1,
        "selected_no_trade_folds": 1 if int(validation_metrics.get("selected_validation_trade_count", 0)) <= 0 else 0,
        "mean_selected_validation_trade_count": validation_metrics.get("selected_validation_trade_count", 0),
        "mean_selected_validation_max_drawdown": validation_metrics.get("selected_validation_max_drawdown", 0.0),
        "mean_selected_validation_active_days": validation_metrics.get("selected_validation_active_days", 0),
        "mean_selected_validation_profit_per_active_day": validation_metrics.get("selected_validation_profit_per_active_day", 0.0),
        "mean_selected_base_objective_score": validation_metrics.get("selected_base_objective_score", 0.0),
        "mean_selected_penalized_objective_score": validation_metrics.get("selected_penalized_objective_score", resolved_selected_objective_score),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "initial_capital": args.initial_capital,
        "max_position_fraction": args.max_position_fraction,
        "max_volume_fraction": args.max_volume_fraction,
        "max_trades_per_period": args.max_trades_per_period,
        "max_trades_per_day": args.max_trades_per_day,
        "max_trades_per_fold": args.max_trades_per_fold,
        "max_losing_trades_per_day": args.max_losing_trades_per_day,
        "max_daily_drawdown": args.max_daily_drawdown,
        "pause_after_drawdown_minutes": args.pause_after_drawdown_minutes,
        "trade_period_minutes": args.trade_period_minutes,
        "holding_period_minutes": args.holding_period_minutes,
        "min_validation_trades": args.min_validation_trades,
        "max_validation_trades": args.max_validation_trades,
        "min_validation_precision": args.min_validation_precision,
        "min_selected_threshold": args.min_selected_threshold,
        "min_predicted_net_return": args.min_predicted_net_return,
        "hybrid_min_score": args.hybrid_min_score,
        "hybrid_return_combination": getattr(args, "hybrid_return_combination", "probability_times_return"),
        "hybrid_min_probability": getattr(args, "hybrid_min_probability", 0.0),
        "regression_target": getattr(args, "regression_target", "trade_return"),
        "regression_clip_min": getattr(args, "regression_clip_min", -0.03),
        "regression_clip_max": getattr(args, "regression_clip_max", 0.05),
        "ev_payoff_mode": getattr(args, "ev_payoff_mode", "fixed_targets"),
        "threshold_drawdown_penalty": args.threshold_drawdown_penalty,
        "threshold_trade_count_penalty": args.threshold_trade_count_penalty,
        "target_validation_trades": args.target_validation_trades,
        "threshold_tiebreaker": getattr(args, "threshold_tiebreaker", "fewer_trades"),
        "threshold_tie_epsilon": getattr(args, "threshold_tie_epsilon", 1e-9),
        "threshold_target_trades": getattr(args, "threshold_target_trades", 0),
        "threshold_target_active_days": getattr(args, "threshold_target_active_days", 0),
        "overactive_trade_threshold": args.overactive_trade_threshold,
        "profit_safety": args.profit_safety,
        "adaptive_thresholds": 0 if args.disable_adaptive_thresholds else 1,
        "trade_selection": args.trade_selection,
        "top_k_per_minute": args.top_k_per_minute,
        "selected_ev_safety_margin": args.ev_safety_margin,
        "acceptance_tier": getattr(args, "acceptance_tier", "none"),
        "normalized_microsecond_open_times": NORMALIZED_MICROSECOND_OPEN_TIMES,
        "dynamic_hybrid_thresholds": getattr(args, "dynamic_hybrid_thresholds", "none"),
        "meta_filter": getattr(args, "meta_filter", "none"),
        "ensemble_windows": ",".join(str(value) for value in getattr(args, "ensemble_window_list", [])),
        "ensemble_model_count": len(getattr(args, "ensemble_window_list", [])),
        "ensemble_enabled": 1 if getattr(args, "ensemble_window_list", []) else 0,
    }
    if validation_metrics:
        record.update({
            "validation_predicted_trades": validation_metrics["predicted_trades"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_average_profit_after_fee_and_slippage": validation_metrics["average_profit_after_fee_and_slippage"],
            "validation_total_profit_after_fee_and_slippage": validation_metrics["total_profit_after_fee_and_slippage"],
            "validation_portfolio_profit": validation_metrics["portfolio_profit"],
            "validation_portfolio_return": validation_metrics["portfolio_return"],
        })
    record.update(metrics)
    calibration_info = {}
    if isinstance(validation_metrics, dict):
        calibration_info = validation_metrics.get("calibration_info", {}) or {}
    if not calibration_info and hasattr(args, "_selected_calibration_info"):
        calibration_info = getattr(args, "_selected_calibration_info") or {}
    record.update({
        "calibration": calibration_info.get("calibration", "none"),
        "calibration_a": calibration_info.get("calibration_a", 0.0),
        "calibration_b": calibration_info.get("calibration_b", 0.0),
        "validation_brier_before": calibration_info.get("validation_brier_before", 0.0),
        "validation_brier_after": calibration_info.get("validation_brier_after", 0.0),
        "calibration_rows": calibration_info.get("calibration_rows", 0),
        "regression_calibration": calibration_info.get("regression_calibration", "none"),
        "regression_calibration_a": calibration_info.get("regression_calibration_a", 1.0),
        "regression_calibration_b": calibration_info.get("regression_calibration_b", 0.0),
        "regression_calibration_rows": calibration_info.get("regression_calibration_rows", 0),
        "regression_calibration_rmse_before": calibration_info.get("regression_calibration_rmse_before", 0.0),
        "regression_calibration_rmse_after": calibration_info.get("regression_calibration_rmse_after", 0.0),
        "regression_calibration_mae_before": calibration_info.get("regression_calibration_mae_before", 0.0),
        "regression_calibration_mae_after": calibration_info.get("regression_calibration_mae_after", 0.0),
        "risk_adjusted_return_feature": calibration_info.get("risk_adjusted_return_feature", ""),
        "hybrid_return_combination": calibration_info.get("hybrid_return_combination", getattr(args, "hybrid_return_combination", "probability_times_return")),
        "hybrid_min_probability": calibration_info.get("hybrid_min_probability", getattr(args, "hybrid_min_probability", 0.0)),
        "hybrid_score_mode": calibration_info.get("hybrid_score_mode", getattr(args, "hybrid_score_mode", "basic")),
        "hybrid_uncertainty_method": calibration_info.get("hybrid_uncertainty_method", getattr(args, "hybrid_uncertainty_method", "none")),
        "hybrid_uncertainty_penalty": calibration_info.get("hybrid_uncertainty_penalty", getattr(args, "hybrid_uncertainty_penalty", 0.0)),
        "hybrid_uncertainty_global_std": calibration_info.get("hybrid_uncertainty_global_std", 0.0),
        "hybrid_uncertainty_rows": calibration_info.get("hybrid_uncertainty_rows", 0),
        "conditional_expected_win_return": calibration_info.get("conditional_expected_win_return", 0.0),
        "conditional_expected_loss_return": calibration_info.get("conditional_expected_loss_return", 0.0),
        "conditional_payoff_rows": calibration_info.get("conditional_payoff_rows", 0),
        "conditional_payoff_positive_rows": calibration_info.get("conditional_payoff_positive_rows", 0),
        "conditional_payoff_negative_rows": calibration_info.get("conditional_payoff_negative_rows", 0),
        "conditional_payoff_source": calibration_info.get("conditional_payoff_source", "not_used"),
        "ev_payoff_mode": calibration_info.get("ev_payoff_mode", getattr(args, "ev_payoff_mode", "fixed_targets")),
        "ev_expected_win_return": calibration_info.get("ev_expected_win_return", getattr(args, "effective_upside_target", getattr(args, "upside_target", 0.05))),
        "ev_expected_loss_return": calibration_info.get("ev_expected_loss_return", -getattr(args, "effective_downside_stop", getattr(args, "downside_stop", 0.02))),
        "ev_payoff_rows": calibration_info.get("ev_payoff_rows", 0),
        "ev_payoff_positive_rows": calibration_info.get("ev_payoff_positive_rows", 0),
        "ev_payoff_negative_rows": calibration_info.get("ev_payoff_negative_rows", 0),
        "ev_payoff_source": calibration_info.get("ev_payoff_source", "fixed_targets"),
        "dynamic_hybrid_thresholds": calibration_info.get("dynamic_hybrid_thresholds", getattr(args, "dynamic_hybrid_thresholds", "none")),
        "meta_filter": calibration_info.get("meta_filter", getattr(args, "meta_filter", "none")),
        "meta_filter_rows": calibration_info.get("meta_filter_rows", 0),
        "meta_filter_positive_rate": calibration_info.get("meta_filter_positive_rate", 0.0),
        "meta_filter_auc_or_accuracy": calibration_info.get("meta_filter_auc_or_accuracy", 0.0),
        "meta_filter_enabled": calibration_info.get("meta_filter_enabled", 0),
        "meta_filter_disabled_reason": calibration_info.get("meta_filter_disabled_reason", ""),
        "meta_filter_validation_trade_retention": calibration_info.get("meta_filter_validation_trade_retention", 0.0),
        "symbol_validation_filter": calibration_info.get("symbol_validation_filter", getattr(args, "symbol_validation_filter", "none")),
        "symbol_filter_stage": calibration_info.get("symbol_filter_stage", getattr(args, "symbol_filter_stage", "executed")),
        "symbol_filter_min_candidates": calibration_info.get("symbol_filter_min_candidates", getattr(args, "symbol_filter_min_candidates", 0)),
        "symbol_filter_min_executed": calibration_info.get("symbol_filter_min_executed", getattr(args, "symbol_filter_min_executed", 0)),
        "symbol_filter_candidate_weight": calibration_info.get("symbol_filter_candidate_weight", getattr(args, "symbol_filter_candidate_weight", 0.5)),
        "symbol_filter_executed_weight": calibration_info.get("symbol_filter_executed_weight", getattr(args, "symbol_filter_executed_weight", 0.5)),
        "symbol_filter_shrinkage": calibration_info.get("symbol_filter_shrinkage", getattr(args, "symbol_filter_shrinkage", 50.0)),
        "symbol_filter_enabled": calibration_info.get("symbol_filter_enabled", 0),
        "symbol_filter_allowed_symbols": calibration_info.get("symbol_filter_allowed_symbols", 0),
        "symbol_filter_total_symbols": calibration_info.get("symbol_filter_total_symbols", 0),
        "symbols_blocked_count": calibration_info.get("symbols_blocked_count", 0),
        "symbols_allowed_count": calibration_info.get("symbols_allowed_count", 0),
        "symbol_filter_disabled_reason": calibration_info.get("symbol_filter_disabled_reason", ""),
        "symbol_filter_validation_trade_retention": calibration_info.get("symbol_filter_validation_trade_retention", 0.0),
        "ensemble_windows": calibration_info.get("ensemble_windows", record.get("ensemble_windows", "")),
        "ensemble_model_count": calibration_info.get("ensemble_model_count", record.get("ensemble_model_count", 0)),
        "ensemble_enabled": calibration_info.get("ensemble_enabled", record.get("ensemble_enabled", 0)),
        "accepted": 1,
        "failed_acceptance_checks": "",
        "rejection_reason": "",
        "strategy_strength": "not_checked",
        "max_rss_gb_observed": MAX_RSS_GIB_OBSERVED,
    })
    return record


def write_metrics(path, records):
    def write_one(output_path):
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    atomic_write_path(path, write_one)
    log_memory("Metrics CSV output complete: {}".format(path))


def write_feature_importance(path, model, feature_names, model_name):
    rows = []
    if isinstance(model, dict):
        if model.get("ensemble_members"):
            for member in model.get("ensemble_members", []):
                window_suffix = "window{}".format(member.get("ensemble_window", ""))
                for label, part in (("classification", member.get("classification_model")), ("return_regression", member.get("regression_model"))):
                    if part is None:
                        continue
                    for feature, importance, fraction in part.feature_importance(feature_names):
                        rows.append(("{}_{}_{}".format(model_name, window_suffix, label), feature, importance, fraction))
        for label, part in (("classification", model.get("classification_model")), ("return_regression", model.get("regression_model"))):
            if part is None:
                continue
            for feature, importance, fraction in part.feature_importance(feature_names):
                rows.append(("{}_{}".format(model_name, label), feature, importance, fraction))
    else:
        for feature, importance, fraction in model.feature_importance(feature_names):
            rows.append((model_name, feature, importance, fraction))
    rows.sort(key=lambda item: item[2], reverse=True)
    def write_one(output_path):
        with open(output_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "feature", "importance", "importance_fraction"])
            for output_model_name, feature, importance, fraction in rows:
                writer.writerow([output_model_name, feature, "{:.12g}".format(importance), "{:.12g}".format(fraction)])
    atomic_write_path(path, write_one)


def write_pipeline_profile(path):
    if not PROFILE_ENABLED or not path:
        return
    def write_one(output_path):
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROFILE_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for record in PROFILE_RECORDS:
                writer.writerow(record)
    atomic_write_path(path, write_one)


def select_fixed_split_model(train_rows, validation_rows, feature_names, args, kind):
    selected = fit_select_model(train_rows, validation_rows, feature_names, args, kind)
    selected["fixed_validation_backfill_used"] = 0
    backfill_months = int(getattr(args, "fixed_validation_backfill_months", 0))
    no_trade_selected = float(selected.get("threshold", 0.0)) >= 1.01 - 1e-12
    if backfill_months <= 0 or not no_trade_selected:
        return selected
    backfill = prepare_fixed_split_backfill(train_rows, validation_rows, backfill_months)
    if not backfill:
        return selected
    fallback = fit_select_model(
        backfill["train_rows"],
        backfill["validation_rows"],
        feature_names,
        args,
        kind,
    )
    if float(fallback.get("threshold", 0.0)) >= 1.01 - 1e-12:
        return selected
    fallback["fixed_validation_backfill_used"] = 1
    fallback["fixed_validation_backfill_rows"] = len(backfill["backfill_rows"])
    fallback["fixed_validation_backfill_months"] = backfill["month_count"]
    fallback["fixed_validation_backfill_original_threshold"] = float(selected.get("threshold", 1.01))
    fallback["fixed_validation_backfill_start_month"] = backfill["backfill_start_month"]
    fallback["fixed_validation_backfill_end_month"] = backfill["backfill_end_month"]
    return fallback


def run_fixed_split(rows, feature_names, args, kind, model_name):
    split_token = start_profile_stage("fixed_split_creation", args.split_mode)
    if args.split_mode == "ratio":
        train_rows, validation_rows, test_rows = select_ratio_split(rows, args)
    else:
        train_end = args.train_months
        validation_end = train_end + args.validation_months
        test_end = validation_end + args.test_months
        train_rows = select_month_range(rows, 0, train_end)
        validation_rows = select_month_range(rows, train_end, validation_end)
        test_rows = select_month_range(rows, validation_end, test_end)
    finish_profile_stage(
        split_token,
        rows_processed=len(rows),
        extra_info="train={} validation={} test={}".format(len(train_rows), len(validation_rows), len(test_rows)),
    )
    if not train_rows or not validation_rows or not test_rows:
        raise RuntimeError(
            "not enough rows for fixed split: train={}, validation={}, test={}".format(
                len(train_rows), len(validation_rows), len(test_rows)
            )
        )

    memory_checkpoint("Fixed split created: train={:,} validation={:,} test={:,}".format(
        len(train_rows), len(validation_rows), len(test_rows)), args)
    fit_token = start_profile_stage("fixed_split_fit_select", "objective_mode={}".format(args.objective_mode))
    selected = select_fixed_split_model(train_rows, validation_rows, feature_names, args, kind)
    finish_profile_stage(fit_token, rows_processed=len(train_rows) + len(validation_rows))
    if selected.get("fixed_validation_backfill_used"):
        memory_checkpoint(
            "Fixed split validation backfill activated: {} months / {:,} rows".format(
                int(selected.get("fixed_validation_backfill_months", 0)),
                int(selected.get("fixed_validation_backfill_rows", 0)),
            ),
            args,
        )
    args._selected_calibration_info = selected.get("calibration_info") or {}
    predict_token = start_profile_stage("fixed_test_prediction", model_name)
    bundle = prediction_bundle_for_models(selected, test_rows, kind, args, "fixed test")
    finish_profile_stage(predict_token, rows_processed=len(test_rows))
    test_metrics = evaluate(
        test_rows,
        bundle,
        selected["threshold"],
        args.fee,
        args.slippage * args.test_slippage_multiplier,
        compute_auc=args.objective_mode != "return_regression",
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
        trade_score_name=selected.get("trade_score_name", score_name_for_args(args)),
        min_predicted_net_return=args.min_predicted_net_return,
        hybrid_min_score=args.hybrid_min_score,
        max_trades_per_day=args.max_trades_per_day,
        max_trades_per_fold=args.max_trades_per_fold,
        max_losing_trades_per_day=args.max_losing_trades_per_day,
        max_daily_drawdown=args.max_daily_drawdown,
        pause_after_drawdown_minutes=args.pause_after_drawdown_minutes,
        hybrid_runtime_args=args,
        symbol_filter_info=selected.get("symbol_filter_info"),
    )
    test_metrics["calibration_info"] = selected.get("calibration_info") or {}
    prediction_write_token = start_profile_stage("prediction_output_write", args.predictions_out)
    write_predictions(
        args.predictions_out,
        test_rows,
        bundle,
        selected["threshold"],
        model_name,
        output_mode=args.prediction_output_mode,
        initial_capital=args.initial_capital,
        max_position_fraction=args.max_position_fraction,
        max_volume_fraction=args.max_volume_fraction,
        max_trades_per_period=args.max_trades_per_period,
        trade_period_minutes=args.trade_period_minutes,
        holding_period_minutes=args.holding_period_minutes,
        fee=args.fee,
        slippage=args.slippage * args.test_slippage_multiplier,
        threshold_objective=args.threshold_objective,
        trade_selection=args.trade_selection,
        top_k_per_minute=args.top_k_per_minute,
        upside_target=args.upside_target,
        downside_stop=args.downside_stop,
        ev_safety_margin=args.ev_safety_margin,
        objective_mode=args.objective_mode,
        trade_score_name=selected.get("trade_score_name", score_name_for_args(args)),
        min_predicted_net_return=args.min_predicted_net_return,
        hybrid_min_score=args.hybrid_min_score,
        max_trades_per_day=args.max_trades_per_day,
        max_trades_per_fold=args.max_trades_per_fold,
        max_losing_trades_per_day=args.max_losing_trades_per_day,
        max_daily_drawdown=args.max_daily_drawdown,
        pause_after_drawdown_minutes=args.pause_after_drawdown_minutes,
        hybrid_runtime_args=args,
        symbol_filter_info=selected.get("symbol_filter_info"),
    )
    finish_profile_stage(prediction_write_token, rows_processed=len(test_rows), extra_info=args.predictions_out)
    feature_write_token = start_profile_stage("feature_importance_write", args.feature_importance_out)
    write_feature_importance(args.feature_importance_out, selected, feature_names, model_name)
    finish_profile_stage(feature_write_token, rows_processed=len(feature_names), extra_info=args.feature_importance_out)
    record = metrics_record(
        model_name,
        "fixed",
        args.threshold_objective,
        selected["threshold"],
        train_rows,
        validation_rows,
        test_rows,
        test_metrics,
        args,
        selected["validation_metrics"],
        selected.get("score", 0.0),
    )
    symbol_filter_records = build_symbol_filter_diagnostic_records(
        selected.get("symbol_filter_info"),
        "fixed_split_validation",
        0,
    )
    cleanup_prediction_bundle(bundle)
    del bundle
    gc.collect()
    return record, selected, symbol_filter_records


def aggregate_fold_records(records, model_name, objective):
    if not records:
        return None
    def numeric_values(column, finite_only=False):
        values = []
        for row in records:
            value = row.get(column)
            if not isinstance(value, (int, float)):
                continue
            value = float(value)
            if finite_only and not math.isfinite(value):
                continue
            values.append(value)
        return values

    def mean_or_default(values, default=0.0):
        if not values:
            return default
        return sum(values) / float(len(values))

    selected_objective_values = numeric_values("selected_objective_score", finite_only=True)
    selected_base_values = numeric_values("selected_base_objective_score", finite_only=True)
    selected_penalized_values = numeric_values("selected_penalized_objective_score", finite_only=True)
    selected_validation_trade_counts = numeric_values("selected_validation_trade_count")
    selected_validation_drawdowns = numeric_values("selected_validation_max_drawdown")
    selected_validation_active_days = numeric_values("selected_validation_active_days")
    selected_validation_profit_days = numeric_values("selected_validation_profit_per_active_day")
    selected_finite_folds = len(selected_objective_values)
    selected_nonfinite_folds = len(records) - selected_finite_folds
    selected_no_trade_folds = sum(
        1 for row in records if int(row.get("selected_validation_trade_count", 0)) <= 0
    )
    aggregate = {
        "model": model_name,
        "split": "walkforward_average",
        "split_mode": records[0].get("split_mode", ""),
        "objective_mode": records[0].get("objective_mode", "classification"),
        "trade_score": records[0].get("trade_score", "probability"),
        "trade_score_name": records[0].get("trade_score_name", records[0].get("trade_score", "probability")),
        "regression_target": records[0].get("regression_target", "trade_return"),
        "regression_clip_min": records[0].get("regression_clip_min", 0.0),
        "regression_clip_max": records[0].get("regression_clip_max", 0.0),
        "hybrid_return_combination": records[0].get("hybrid_return_combination", "probability_times_return"),
        "hybrid_min_probability": records[0].get("hybrid_min_probability", 0.0),
        "hybrid_score_mode": records[0].get("hybrid_score_mode", "basic"),
        "hybrid_uncertainty_method": records[0].get("hybrid_uncertainty_method", "none"),
        "hybrid_uncertainty_penalty": records[0].get("hybrid_uncertainty_penalty", 0.0),
        "conditional_expected_win_return": records[0].get("conditional_expected_win_return", 0.0),
        "conditional_expected_loss_return": records[0].get("conditional_expected_loss_return", 0.0),
        "conditional_payoff_rows": mean_or_default(numeric_values("conditional_payoff_rows")),
        "conditional_payoff_positive_rows": mean_or_default(numeric_values("conditional_payoff_positive_rows")),
        "conditional_payoff_negative_rows": mean_or_default(numeric_values("conditional_payoff_negative_rows")),
        "conditional_payoff_source": records[0].get("conditional_payoff_source", "not_used"),
        "dynamic_hybrid_thresholds": records[0].get("dynamic_hybrid_thresholds", "none"),
        "ev_payoff_mode": records[0].get("ev_payoff_mode", "fixed_targets"),
        "ev_payoff_source": records[0].get("ev_payoff_source", "fixed_targets"),
        "meta_filter": records[0].get("meta_filter", "none"),
        "symbol_validation_filter": records[0].get("symbol_validation_filter", "none"),
        "symbol_filter_stage": records[0].get("symbol_filter_stage", "executed"),
        "ensemble_windows": records[0].get("ensemble_windows", ""),
        "ensemble_model_count": records[0].get("ensemble_model_count", 0),
        "ensemble_enabled": records[0].get("ensemble_enabled", 0),
        "train_ratio": records[0].get("train_ratio", 0.0),
        "validation_ratio": records[0].get("validation_ratio", 0.0),
        "test_ratio": records[0].get("test_ratio", 0.0),
        "threshold_objective": objective,
        "threshold_tiebreaker": records[0].get("threshold_tiebreaker", "fewer_trades"),
        "selected_threshold": sum(float(row["selected_threshold"]) for row in records) / len(records),
        "selected_score_name": records[0].get("selected_score_name", "probability"),
        "selected_score_threshold": mean_or_default(numeric_values("selected_score_threshold")),
        "selected_objective_score": mean_or_default(selected_objective_values, -float("inf") if selected_nonfinite_folds == len(records) else 0.0),
        "selected_base_objective_score": mean_or_default(selected_base_values, -float("inf") if selected_nonfinite_folds == len(records) else 0.0),
        "selected_penalized_objective_score": mean_or_default(selected_penalized_values, -float("inf") if selected_nonfinite_folds == len(records) else 0.0),
        "selected_validation_trade_count": mean_or_default(selected_validation_trade_counts),
        "selected_validation_max_drawdown": mean_or_default(selected_validation_drawdowns),
        "selected_threshold_tie_rank_reason": records[0].get("selected_threshold_tie_rank_reason", ""),
        "selected_objective_finite_folds": selected_finite_folds,
        "selected_objective_nonfinite_folds": selected_nonfinite_folds,
        "selected_no_trade_folds": selected_no_trade_folds,
        "mean_selected_validation_trade_count": mean_or_default(selected_validation_trade_counts),
        "mean_selected_validation_max_drawdown": mean_or_default(selected_validation_drawdowns),
        "mean_selected_validation_active_days": mean_or_default(selected_validation_active_days),
        "mean_selected_validation_profit_per_active_day": mean_or_default(selected_validation_profit_days),
        "mean_selected_base_objective_score": mean_or_default(selected_base_values, -float("inf") if selected_nonfinite_folds == len(records) else 0.0),
        "mean_selected_penalized_objective_score": mean_or_default(selected_penalized_values, -float("inf") if selected_nonfinite_folds == len(records) else 0.0),
        "train_rows": sum(int(row["train_rows"]) for row in records),
        "validation_rows": sum(int(row["validation_rows"]) for row in records),
        "test_rows": sum(int(row["test_rows"]) for row in records),
        "min_predicted_net_return": records[0].get("min_predicted_net_return", 0.0),
        "hybrid_min_score": records[0].get("hybrid_min_score", 0.0),
        "profit_safety": records[0].get("profit_safety", ""),
        "adaptive_thresholds": records[0].get("adaptive_thresholds", 0),
        "trade_selection": records[0].get("trade_selection", "threshold"),
        "top_k_per_minute": records[0].get("top_k_per_minute", 0),
        "threshold_tie_epsilon": records[0].get("threshold_tie_epsilon", 1e-9),
        "threshold_target_trades": records[0].get("threshold_target_trades", 0),
        "threshold_target_active_days": records[0].get("threshold_target_active_days", 0),
        "acceptance_tier": records[0].get("acceptance_tier", "none"),
        "meta_filter_rows": mean_or_default(numeric_values("meta_filter_rows")),
        "meta_filter_positive_rate": mean_or_default(numeric_values("meta_filter_positive_rate")),
        "meta_filter_auc_or_accuracy": mean_or_default(numeric_values("meta_filter_auc_or_accuracy")),
        "meta_filter_enabled": int(sum(int(row.get("meta_filter_enabled", 0)) for row in records)),
        "regression_calibration": records[0].get("regression_calibration", "none"),
        "regression_calibration_a": mean_or_default(numeric_values("regression_calibration_a")),
        "regression_calibration_b": mean_or_default(numeric_values("regression_calibration_b")),
        "regression_calibration_rows": mean_or_default(numeric_values("regression_calibration_rows")),
        "regression_calibration_rmse_before": mean_or_default(numeric_values("regression_calibration_rmse_before")),
        "regression_calibration_rmse_after": mean_or_default(numeric_values("regression_calibration_rmse_after")),
        "regression_calibration_mae_before": mean_or_default(numeric_values("regression_calibration_mae_before")),
        "regression_calibration_mae_after": mean_or_default(numeric_values("regression_calibration_mae_after")),
        "risk_adjusted_return_feature": records[0].get("risk_adjusted_return_feature", ""),
        "hybrid_uncertainty_global_std": mean_or_default(numeric_values("hybrid_uncertainty_global_std")),
        "hybrid_uncertainty_rows": mean_or_default(numeric_values("hybrid_uncertainty_rows")),
        "max_rss_gb_observed": MAX_RSS_GIB_OBSERVED,
    }
    for column in METRIC_COLUMNS:
        if column in aggregate or column in (
            "model",
            "split",
            "split_mode",
            "objective_mode",
            "trade_score",
            "trade_score_name",
            "selected_score_name",
            "threshold_objective",
            "profit_safety",
            "trade_selection",
            "acceptance_tier",
            "failed_acceptance_checks",
            "rejection_reason",
            "strategy_strength",
            "selected_threshold",
            "selected_score_threshold",
            "selected_objective_score",
            "selected_base_objective_score",
            "selected_penalized_objective_score",
            "selected_validation_trade_count",
            "selected_validation_max_drawdown",
            "selected_objective_finite_folds",
            "selected_objective_nonfinite_folds",
            "selected_no_trade_folds",
            "mean_selected_validation_trade_count",
            "mean_selected_validation_max_drawdown",
            "mean_selected_base_objective_score",
            "mean_selected_penalized_objective_score",
        ):
            continue
        values = [row.get(column) for row in records if isinstance(row.get(column), (int, float))]
        if values:
            aggregate[column] = sum(values) / len(values)
            continue
        sample_value = next((row.get(column) for row in records if row.get(column) not in (None, "")), None)
        if isinstance(sample_value, str):
            aggregate[column] = sample_value
        else:
            aggregate[column] = 0.0
    aggregate["accepted"] = 1
    aggregate["failed_acceptance_checks"] = ""
    aggregate["rejection_reason"] = ""
    aggregate["strategy_strength"] = "not_checked"
    return aggregate


def max_month_index(rows):
    if not rows:
        return -1
    if is_compact_rows(rows):
        if rows.indices is None:
            return int(np.max(rows.table.month_indices))
        return int(np.max(rows.table.month_indices[rows.indices]))
    return max(row.month_index for row in rows)


def run_walk_forward(rows, feature_names, args, kind, model_name):
    max_month = max_month_index(rows)
    fold_records = []
    diagnostic_records = []
    symbol_filter_records = []
    init_prediction_token = start_profile_stage("prediction_output_write", args.walk_predictions_out)
    write_predictions(
        args.walk_predictions_out,
        [],
        build_prediction_bundle(probability=[], calibrated_probability=[], predicted_trade_return=[]),
        0.0,
        model_name,
        output_mode=args.prediction_output_mode,
        initial_capital=args.initial_capital,
        max_position_fraction=args.max_position_fraction,
        max_volume_fraction=args.max_volume_fraction,
        max_trades_per_period=args.max_trades_per_period,
        trade_period_minutes=args.trade_period_minutes,
        holding_period_minutes=args.holding_period_minutes,
        fee=args.fee,
        slippage=args.slippage,
        max_trades_per_day=args.max_trades_per_day,
        max_trades_per_fold=args.max_trades_per_fold,
        max_losing_trades_per_day=args.max_losing_trades_per_day,
        max_daily_drawdown=args.max_daily_drawdown,
        pause_after_drawdown_minutes=args.pause_after_drawdown_minutes,
        hybrid_runtime_args=args,
    )
    finish_profile_stage(init_prediction_token, rows_processed=0, extra_info=args.walk_predictions_out)
    max_fold_start = max_month - args.walk_train_months - args.walk_validation_months + 1
    for fold_start in range(0, max(0, max_fold_start)):
        train_start, train_end, validation_start, validation_end, test_month = walk_forward_split_bounds(
            fold_start,
            args.walk_train_months,
            args.walk_validation_months,
        )
        if test_month > max_month:
            break
        train_rows = select_month_range(rows, train_start, train_end)
        validation_rows = select_month_range(rows, validation_start, validation_end)
        final_train_rows = select_month_range(rows, train_start, test_month)
        test_rows = select_month_range(rows, test_month, test_month + 1)
        if not train_rows or not validation_rows or not final_train_rows or not test_rows:
            continue

        fold_index = len(fold_records) + 1
        fold_split_token = start_profile_stage("walkforward_fold_{}_split".format(fold_index), "test_month={}".format(test_month))
        memory_checkpoint("Walk-forward fold {} split created: train={:,} validation={:,} test={:,}".format(
            fold_index, len(train_rows), len(validation_rows), len(test_rows)), args)
        finish_profile_stage(
            fold_split_token,
            rows_processed=len(train_rows) + len(validation_rows) + len(test_rows),
            extra_info="train={} validation={} test={}".format(len(train_rows), len(validation_rows), len(test_rows)),
        )
        fold_fit_token = start_profile_stage("walkforward_fold_{}_fit_select".format(fold_index), "objective_mode={}".format(args.objective_mode))
        selected = fit_select_model(train_rows, validation_rows, feature_names, args, kind)
        finish_profile_stage(fold_fit_token, rows_processed=len(train_rows) + len(validation_rows))
        args._selected_calibration_info = selected.get("calibration_info") or {}
        if args.walk_forward_final_model == "selected":
            fit_final_train_rows = None
            final_selected = selected
            memory_checkpoint("Walk-forward using validation-selected model without final refit", args)
        else:
            final_sample_token = start_profile_stage("walkforward_fold_{}_final_train_sampling".format(fold_index))
            fit_final_train_rows = sample_rows(
                final_train_rows,
                args.max_final_train_rows,
                label_aware=True,
                sample_mode=args.train_sample_mode,
                max_positive_fraction=args.max_positive_sample_fraction,
            )
            if len(fit_final_train_rows) != len(final_train_rows):
                print(
                    "Walk-forward final fit sample: train {}/{}".format(
                        len(fit_final_train_rows),
                        len(final_train_rows),
                    )
                )
            finish_profile_stage(final_sample_token, rows_processed=len(fit_final_train_rows))
            selected_classification = selected.pop("classification_model", None)
            selected_regression = selected.pop("regression_model", None)
            del selected_classification
            del selected_regression
            gc.collect()
            positive_weight = class_weight_ratio_for_rows(final_train_rows, args.positive_weight_cap)
            x_final = model_matrix(fit_final_train_rows, kind, args)
            final_selected = {
                "params": dict(selected.get("params", {})),
                "threshold": selected["threshold"],
                "score": selected.get("score", 0.0),
                "trade_score_name": selected.get("trade_score_name", score_name_for_args(args)),
                "selected_score_name": selected.get("selected_score_name", selected_score_name_for_mode(args)),
                "calibration": selected.get("calibration"),
                "regression_calibration": selected.get("regression_calibration"),
                "uncertainty_model": selected.get("uncertainty_model"),
                "meta_filter_info": selected.get("meta_filter_info"),
                "symbol_filter_info": selected.get("symbol_filter_info"),
                "calibration_info": selected.get("calibration_info") or {},
                "validation_metrics": selected.get("validation_metrics", {}),
                "ev_payoff_info": dict(selected.get("ev_payoff_info") or {}),
            }
            if args.objective_mode in ("classification", "hybrid"):
                y_class_final = model_targets(fit_final_train_rows, kind, "classification")
                final_selected["classification_model"] = make_model(kind, selected["params"], positive_weight, "classification")
                fold_class_fit = start_profile_stage("walkforward_fold_{}_classification_fit".format(fold_index))
                final_selected["classification_model"].fit(x_final, y_class_final, feature_names)
                finish_profile_stage(fold_class_fit, rows_processed=len(fit_final_train_rows))
                del y_class_final
                gc.collect()
                memory_checkpoint("Walk-forward final classification fit complete", args)
            else:
                final_selected["classification_model"] = None
            if args.objective_mode in ("return_regression", "hybrid"):
                y_reg_final = model_targets(fit_final_train_rows, kind, "return_regression", args)
                final_selected["regression_model"] = make_model(kind, selected["params"], positive_weight, "return_regression")
                fold_reg_fit = start_profile_stage("walkforward_fold_{}_regression_fit".format(fold_index))
                final_selected["regression_model"].fit(x_final, y_reg_final, feature_names)
                finish_profile_stage(fold_reg_fit, rows_processed=len(fit_final_train_rows))
                del y_reg_final
                gc.collect()
                memory_checkpoint("Walk-forward final regression fit complete", args)
            else:
                final_selected["regression_model"] = None
            del x_final
            gc.collect()
        fold_predict_token = start_profile_stage("walkforward_fold_{}_prediction".format(fold_index), model_name)
        bundle = prediction_bundle_for_models(
            final_selected,
            test_rows,
            kind,
            args,
            "walk-forward fold {}".format(fold_index),
        )
        finish_profile_stage(fold_predict_token, rows_processed=len(test_rows))
        metrics = evaluate(
            test_rows,
            bundle,
            final_selected["threshold"],
            args.fee,
            args.slippage * args.test_slippage_multiplier,
            compute_auc=args.objective_mode != "return_regression",
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
            trade_score_name=final_selected.get("trade_score_name", score_name_for_args(args)),
            min_predicted_net_return=args.min_predicted_net_return,
            hybrid_min_score=args.hybrid_min_score,
            max_trades_per_day=args.max_trades_per_day,
            max_trades_per_fold=args.max_trades_per_fold,
            max_losing_trades_per_day=args.max_losing_trades_per_day,
            max_daily_drawdown=args.max_daily_drawdown,
            pause_after_drawdown_minutes=args.pause_after_drawdown_minutes,
            hybrid_runtime_args=args,
            symbol_filter_info=final_selected.get("symbol_filter_info"),
        )
        metrics.update(
            inactive_fold_blocker_check(
                test_rows,
                bundle,
                final_selected["threshold"],
                metrics,
                selected.get("validation_metrics", {}),
                args,
                final_selected.get("trade_score_name", score_name_for_args(args)),
            )
        )
        metrics["calibration_info"] = final_selected.get("calibration_info") or {}
        fold_output_token = start_profile_stage("walkforward_fold_{}_prediction_output".format(fold_index), args.walk_predictions_out)
        write_predictions(
            args.walk_predictions_out,
            test_rows,
            bundle,
            final_selected["threshold"],
            model_name,
            append=True,
            output_mode=args.prediction_output_mode,
            initial_capital=args.initial_capital,
            max_position_fraction=args.max_position_fraction,
            max_volume_fraction=args.max_volume_fraction,
            max_trades_per_period=args.max_trades_per_period,
            trade_period_minutes=args.trade_period_minutes,
            holding_period_minutes=args.holding_period_minutes,
            fee=args.fee,
            slippage=args.slippage * args.test_slippage_multiplier,
            threshold_objective=args.threshold_objective,
            trade_selection=args.trade_selection,
            top_k_per_minute=args.top_k_per_minute,
            upside_target=args.upside_target,
            downside_stop=args.downside_stop,
            ev_safety_margin=args.ev_safety_margin,
            objective_mode=args.objective_mode,
            trade_score_name=final_selected.get("trade_score_name", score_name_for_args(args)),
            min_predicted_net_return=args.min_predicted_net_return,
            hybrid_min_score=args.hybrid_min_score,
            max_trades_per_day=args.max_trades_per_day,
            max_trades_per_fold=args.max_trades_per_fold,
            max_losing_trades_per_day=args.max_losing_trades_per_day,
            max_daily_drawdown=args.max_daily_drawdown,
            pause_after_drawdown_minutes=args.pause_after_drawdown_minutes,
            hybrid_runtime_args=args,
            symbol_filter_info=final_selected.get("symbol_filter_info"),
        )
        finish_profile_stage(fold_output_token, rows_processed=len(test_rows), extra_info=args.walk_predictions_out)
        split_name = "walkforward_fold_{}_test_month_{}".format(fold_index, test_month + 1)
        symbol_filter_records.extend(
            build_symbol_filter_diagnostic_records(
                final_selected.get("symbol_filter_info"),
                "walkforward_fold_{}_validation".format(fold_index),
                fold_index,
            )
        )
        record = metrics_record(
            model_name,
            split_name,
            args.threshold_objective,
            final_selected["threshold"],
            final_train_rows,
            validation_rows,
            test_rows,
            metrics,
            args,
            selected["validation_metrics"],
            selected.get("score", 0.0),
        )
        fold_records.append(record)
        diagnostic_records.append(
            build_walkforward_diagnostic_record(
                fold_index,
                split_name,
                train_rows,
                validation_rows,
                test_rows,
                final_selected["threshold"],
                selected.get("score", 0.0),
                metrics,
                final_selected.get("calibration_info") or {},
                selected.get("validation_metrics", {}),
                args,
            )
        )
        fold_cleanup_token = start_profile_stage("walkforward_fold_{}_cleanup".format(fold_index))
        del selected
        cleanup_prediction_bundle(bundle)
        del bundle
        final_classification = final_selected.pop("classification_model", None)
        final_regression = final_selected.pop("regression_model", None)
        del final_classification
        del final_regression
        del final_selected
        del fit_final_train_rows
        del train_rows
        del validation_rows
        del final_train_rows
        del test_rows
        gc.collect()
        finish_profile_stage(fold_cleanup_token, extra_info="cleanup complete")
        memory_checkpoint("Walk-forward fold {} cleanup complete".format(fold_index), args)

    if fold_records:
        aggregate = aggregate_fold_records(fold_records, model_name, args.threshold_objective)
        if aggregate:
            aggregate.update(walkforward_acceptance_summary(fold_records, args))
            fold_records.append(aggregate)
    walk_metric_token = start_profile_stage("walkforward_metrics_write", args.walkforward_metrics_out)
    write_metrics(args.walkforward_metrics_out, fold_records)
    finish_profile_stage(walk_metric_token, rows_processed=len(fold_records), extra_info=args.walkforward_metrics_out)
    walk_diag_token = start_profile_stage("walkforward_diagnostics_write", args.walkforward_diagnostics_out)
    write_walkforward_diagnostics(args.walkforward_diagnostics_out, diagnostic_records)
    finish_profile_stage(walk_diag_token, rows_processed=len(diagnostic_records), extra_info=args.walkforward_diagnostics_out)
    return fold_records, symbol_filter_records


def read_logistic_metric(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return row
    return None


def metric_float(record, key):
    if record is None:
        return 0.0
    try:
        return float(record.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def describe_cache_files(paths):
    rows = []
    for name, path in sorted(paths.items()):
        if not isinstance(path, str):
            continue
        exists = os.path.exists(path)
        size_bytes = os.path.getsize(path) if exists and os.path.isfile(path) else 0
        rows.append((name, path, exists, size_bytes))
    return rows


def inspect_cache(args, training_manifest=None, manifest_path=None):
    if args.feature_storage not in ("memmap32", "memmap64"):
        raise ValueError("--inspect-cache requires --feature-storage memmap32 or memmap64")
    if os.path.isdir(args.input):
        shards = discover_sharded_dataset_shards(args.input, training_manifest)
        cache_hit = inspect_sharded_binary_cache(
            args.input,
            args.feature_storage,
            args.cache_dir or None,
            training_manifest,
            shards,
            require_cache_hit=False,
        )
        resolved_cache_dir = resolve_cache_dir(args.input, args.cache_dir or None)
        paths = sharded_dataset_cache_paths(
            args.input,
            resolved_cache_dir,
            np.float32 if args.feature_storage == "memmap32" else np.float64,
        )
        inventory_signature = sharded_inventory_signature(shards)
        source_present = os.path.isdir(args.input)
    else:
        cache_hit = inspect_binary_cache(
            args.input,
            args.feature_storage,
            args.cache_dir or None,
            training_manifest,
            manifest_path,
            require_cache_hit=False,
        )
        resolved_cache_dir = resolve_cache_dir(args.input, args.cache_dir or None)
        paths = cache_paths(
            args.input,
            resolved_cache_dir,
            np.float32 if args.feature_storage == "memmap32" else np.float64,
        )
        inventory_signature = ""
        source_present = os.path.exists(args.input)

    print("Cache inspection", flush=True)
    print("cache_status={}".format("hit" if cache_hit else "missing_or_incompatible"), flush=True)
    print("cache_dir={}".format(resolved_cache_dir), flush=True)
    total_size = 0
    for name, path, exists, size_bytes in describe_cache_files(paths):
        total_size += size_bytes
        print("cache_file {} exists={} size_bytes={} path={}".format(name, int(exists), size_bytes, path), flush=True)
    print("cache_total_size_gb={:.6f}".format(total_size / float(1024 ** 3)), flush=True)
    print("source_present={}".format(int(bool(source_present))), flush=True)
    print("cache_only_runnable={}".format(int(bool(cache_hit))), flush=True)
    if training_manifest:
        print("manifest_label_mode={}".format(training_manifest.get("label_mode", "")), flush=True)
        print("manifest_target_exit_mode={}".format(training_manifest.get("target_exit_mode", "")), flush=True)
        print("manifest_upside_target={}".format(training_manifest.get("upside_target", "")), flush=True)
        print("manifest_downside_stop={}".format(training_manifest.get("downside_stop", "")), flush=True)
        print("manifest_market_regime_features={}".format(int(bool(training_manifest.get("market_regime_features", False)))), flush=True)
        print("manifest_market_breadth_features={}".format(int(bool(training_manifest.get("market_breadth_features", False)))), flush=True)
    if inventory_signature:
        print("inventory_signature={}".format(inventory_signature), flush=True)
    if cache_hit:
        manifest = cache_hit["manifest"]
        print("row_count={}".format(int(manifest.get("row_count", 0))), flush=True)
        print("feature_count={}".format(len(manifest.get("feature_columns", []))), flush=True)
        print("dtype={}".format(cache_hit["dtype"]), flush=True)
        print("normalized_microsecond_open_times={}".format(cache_hit.get("normalized_microsecond_open_times", 0)), flush=True)
        print("metadata_load_seconds={:.6f}".format(cache_hit.get("metadata_load_seconds", 0.0)), flush=True)
        print("memmap_attach_seconds={:.6f}".format(cache_hit.get("memmap_attach_seconds", 0.0)), flush=True)
    return 0


def cache_cleanup(args):
    cache_dir = resolve_cache_dir(args.input, args.cache_dir or None)
    if not os.path.exists(cache_dir):
        print("Cache directory does not exist: {}".format(cache_dir), flush=True)
        return 0
    candidates = []
    for root, _, files in os.walk(cache_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                size_bytes = os.path.getsize(path)
            except OSError:
                size_bytes = 0
            reason = "other"
            if name.endswith(".manifest.json"):
                manifest, _ = load_cache_manifest_file(path)
                if manifest is None:
                    reason = "invalid_manifest"
                elif manifest.get("source_csv_path") != os.path.abspath(args.input) and manifest.get("dataset_path") != os.path.abspath(args.input):
                    reason = "different_input"
                else:
                    reason = "current_input"
            candidates.append({
                "path": path,
                "size_bytes": size_bytes,
                "reason": reason,
            })
    removable = [item for item in candidates if item["reason"] != "current_input"]
    print("Cache cleanup review", flush=True)
    for item in sorted(removable, key=lambda row: row["size_bytes"], reverse=True):
        print("candidate reason={} size_bytes={} path={}".format(item["reason"], item["size_bytes"], item["path"]), flush=True)
    total_bytes = sum(item["size_bytes"] for item in removable)
    print("candidate_total_size_gb={:.6f}".format(total_bytes / float(1024 ** 3)), flush=True)
    if getattr(args, "dry_run", False) or not getattr(args, "confirm_delete", False):
        return 0
    for item in removable:
        try:
            os.remove(item["path"])
        except OSError:
            pass
    return 0


def smoke_test_cache(path, feature_storage, cache_dir, training_manifest=None, manifest_path=None):
    if os.path.isdir(path):
        shards = discover_sharded_dataset_shards(path, training_manifest)
        cache_hit = inspect_sharded_binary_cache(
            path,
            feature_storage,
            cache_dir,
            training_manifest,
            shards,
            require_cache_hit=True,
        )
    else:
        cache_hit = inspect_binary_cache(
            path,
            feature_storage,
            cache_dir,
            training_manifest,
            manifest_path,
            require_cache_hit=True,
        )
    feature_values = cache_hit["features"]
    try:
        manifest = cache_hit["manifest"]
        row_count = int(manifest["row_count"])
        feature_count = len(manifest["feature_columns"])
        if tuple(feature_values.shape) != (row_count, feature_count):
            raise ValueError(
                "cache feature matrix shape mismatch: expected ({}, {}) got {}".format(
                    row_count,
                    feature_count,
                    tuple(feature_values.shape),
                )
            )
        info = {
            "rows": row_count,
            "features": feature_count,
            "dtype": cache_hit["dtype"],
            "cache_status": CACHE_LOAD_INFO.get("status", ""),
            "normalized_microsecond_open_times": cache_hit["normalized_microsecond_open_times"],
            "market_regime_features": bool(training_manifest.get("market_regime_features", False)) if training_manifest else False,
            "market_breadth_features": bool(training_manifest.get("market_breadth_features", False)) if training_manifest else False,
        }
        print("Cache smoke test passed:", flush=True)
        print("rows={}".format(info["rows"]), flush=True)
        print("features={}".format(info["features"]), flush=True)
        print("dtype={}".format(info["dtype"]), flush=True)
        print("cache_status={}".format(CACHE_LOAD_INFO.get("status", "")), flush=True)
        print("market_regime_features={}".format(str(info["market_regime_features"]).lower()), flush=True)
        print("market_breadth_features={}".format(str(info["market_breadth_features"]).lower()), flush=True)
        return info
    finally:
        close_memmap(feature_values)


def json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    return str(value)


def print_comparison(gbdt_record, walk_records, args):
    logistic = read_logistic_metric(args.logistic_metrics_in)
    walkforward_summary = walkforward_acceptance_summary(walk_records, args) if walk_records else None
    print("\nComparison report")
    if logistic:
        print(
            "Logistic: threshold={:.4g} precision={:.4f} recall={:.4f} net_fee_slippage={:.6f}".format(
                metric_float(logistic, "selected_threshold"),
                metric_float(logistic, "test_precision"),
                metric_float(logistic, "test_recall"),
                metric_float(logistic, "total_profit_after_fee_and_slippage"),
            )
        )
    else:
        print("Logistic: metrics file not found at {}".format(args.logistic_metrics_in))
    print(
        "GBDT: model={} threshold={:.4g} precision={:.4f} recall={:.4f} portfolio_profit={:.2f} portfolio_return={:.4%} safety={}".format(
            gbdt_record["model"],
            float(gbdt_record["selected_threshold"]),
            float(gbdt_record["precision"]),
            float(gbdt_record["recall"]),
            float(gbdt_record["portfolio_profit"]),
            float(gbdt_record["portfolio_return"]),
            args.profit_safety,
        )
    )
    print(
        "GBDT validation: trades={} precision={:.4f} recall={:.4f} portfolio_profit={:.2f} portfolio_return={:.4%}".format(
            int(float(gbdt_record.get("validation_predicted_trades", 0))),
            float(gbdt_record.get("validation_precision", 0.0)),
            float(gbdt_record.get("validation_recall", 0.0)),
            float(gbdt_record.get("validation_portfolio_profit", 0.0)),
            float(gbdt_record.get("validation_portfolio_return", 0.0)),
        )
    )
    aggregate = walk_records[-1] if walk_records and walk_records[-1].get("split") == "walkforward_average" else None
    if aggregate:
        print(
            "Walk-forward average: folds={} precision={:.4f} recall={:.4f} portfolio_profit={:.2f} portfolio_return={:.4%}".format(
                len(walk_records) - 1,
                float(aggregate["precision"]),
                float(aggregate["recall"]),
                float(aggregate["portfolio_profit"]),
                float(aggregate["portfolio_return"]),
            )
        )
    if walkforward_summary:
        print(
            "Walk-forward activity: active_folds={} inactive_folds={} active_profitable_rate={:.4f} tier={} strength={}".format(
                int(walkforward_summary.get("active_fold_count", 0)),
                int(walkforward_summary.get("inactive_fold_count", 0)),
                float(walkforward_summary.get("active_profitable_fold_rate", 0.0)),
                walkforward_summary.get("acceptance_tier", "none"),
                walkforward_summary.get("strategy_strength", "not_checked"),
            )
        )
    if walkforward_summary:
        if walkforward_summary["accepted"]:
            print("RUN ACCEPTED: walk-forward stability checks passed")
        else:
            print("RUN REJECTED: {}".format(walkforward_summary["rejection_reason"]))

    if logistic:
        print("Logistic comparison is legacy until the C++ baseline is regenerated with portfolio sizing.")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_run_summaries(args, rows, feature_names, kind, fixed_selected, fixed_record, walk_records):
    aggregate = walk_records[-1] if walk_records and walk_records[-1].get("split") == "walkforward_average" else None
    walkforward_summary = walkforward_acceptance_summary(walk_records, args) if walk_records else walkforward_acceptance_summary([], args)
    run_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    effective_upside_target = getattr(args, "effective_upside_target", getattr(args, "upside_target", 0.05))
    effective_downside_stop = getattr(args, "effective_downside_stop", getattr(args, "downside_stop", 0.02))
    summary = {
        "run_at_utc": run_at,
        "git_commit": git_commit(),
        "args": json_safe(vars(args)),
        "input_csv": os.path.abspath(args.input),
        "input_csv_rows": len(rows),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "model_kind": kind,
        "best_params": fixed_selected.get("params", {}),
        "best_iteration": fixed_selected.get("best_iteration"),
        "cache": CACHE_LOAD_INFO,
        "ev_upside_target_source": getattr(args, "ev_upside_target_source", "default"),
        "ev_downside_stop_source": getattr(args, "ev_downside_stop_source", "default"),
        "manifest_upside_target": getattr(args, "manifest_upside_target", 0.0),
        "manifest_downside_stop": getattr(args, "manifest_downside_stop", 0.0),
        "effective_upside_target": effective_upside_target,
        "effective_downside_stop": effective_downside_stop,
        "market_regime_features": getattr(args, "market_regime_features", False),
        "market_breadth_features": getattr(args, "market_breadth_features", False),
        "normalized_microsecond_open_times": CACHE_LOAD_INFO.get(
            "normalized_microsecond_open_times",
            NORMALIZED_MICROSECOND_OPEN_TIMES,
        ),
        "memory_settings": {
            "feature_storage": args.feature_storage,
            "memmap_dir": args.memmap_dir,
            "cache_dir": args.cache_dir,
            "max_train_rows": args.max_train_rows,
            "max_validation_rows": args.max_validation_rows,
            "max_final_train_rows": args.max_final_train_rows,
            "prediction_batch_rows": args.prediction_batch_rows,
            "max_bin": args.max_bin,
            "subsample_for_bin": args.subsample_for_bin,
            "lightgbm_histogram_pool_mb": args.lightgbm_histogram_pool_mb,
            "n_jobs": args.n_jobs,
            "memory_budget_gb": args.memory_budget_gb,
            "max_rss_gb": args.max_rss_gb,
            "max_rss_gb_observed": MAX_RSS_GIB_OBSERVED,
            "max_rss_stage": MAX_RSS_STAGE,
        },
        "fixed_metrics": fixed_record,
        "walk_forward_aggregate_metrics": aggregate,
        "walk_forward_summary": walkforward_summary,
        "max_rss_gb_observed": MAX_RSS_GIB_OBSERVED,
        "max_rss_stage": MAX_RSS_STAGE,
    }
    def write_one(output_path):
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    atomic_write_path(args.run_summary_out, write_one)

    experiment_fields = [
        "run_at_utc",
        "git_commit",
        "input_csv",
        "input_csv_rows",
        "feature_count",
        "model_kind",
        "objective_mode",
        "trade_score",
        "selected_score_name",
        "selected_score_threshold",
        "selected_threshold",
        "hybrid_return_combination",
        "hybrid_min_probability",
        "conditional_expected_win_return",
        "conditional_expected_loss_return",
        "conditional_payoff_rows",
        "conditional_payoff_positive_rows",
        "conditional_payoff_negative_rows",
        "conditional_payoff_source",
        "ev_payoff_mode",
        "ev_payoff_source",
        "regression_calibration",
        "regression_target",
        "hybrid_score_mode",
        "hybrid_uncertainty_method",
        "dynamic_hybrid_thresholds",
        "meta_filter",
        "symbol_filter_stage",
        "threshold_tiebreaker",
        "ensemble_windows",
        "portfolio_profit",
        "portfolio_return",
        "precision",
        "recall",
        "raw_precision",
        "raw_recall",
        "walk_forward_folds",
        "walk_forward_portfolio_profit",
        "walk_forward_portfolio_return",
        "walkforward_profitable_fold_rate",
        "walkforward_median_portfolio_return",
        "walkforward_mean_portfolio_return",
        "active_fold_rate",
        "active_profitable_fold_rate",
        "median_active_fold_return",
        "worst_active_fold_return",
        "walkforward_total_predicted_trades",
        "overactive_losing_folds",
        "avg_trades_in_losing_active_folds",
        "selected_objective_finite_folds",
        "selected_objective_nonfinite_folds",
        "selected_no_trade_folds",
        "mean_selected_validation_trade_count",
        "mean_selected_validation_max_drawdown",
        "mean_selected_base_objective_score",
        "mean_selected_penalized_objective_score",
        "acceptance_tier",
        "accepted",
        "failed_acceptance_checks",
        "rejection_reason",
        "strategy_strength",
        "ev_upside_target_source",
        "ev_downside_stop_source",
        "manifest_upside_target",
        "manifest_downside_stop",
        "effective_upside_target",
        "effective_downside_stop",
        "market_regime_features",
        "market_breadth_features",
        "normalized_microsecond_open_times",
        "max_rss_gb_observed",
        "max_rss_stage",
        "feature_storage",
        "max_train_rows",
        "prediction_batch_rows",
        "best_params",
    ]
    experiment = {
        "run_at_utc": run_at,
        "git_commit": summary["git_commit"],
        "input_csv": summary["input_csv"],
        "input_csv_rows": len(rows),
        "feature_count": len(feature_names),
        "model_kind": kind,
        "objective_mode": args.objective_mode,
        "trade_score": score_name_for_args(args),
        "selected_score_name": fixed_record.get("selected_score_name", selected_score_name_for_mode(args)),
        "selected_score_threshold": fixed_record.get("selected_score_threshold", fixed_record.get("selected_threshold", 0.0)),
        "selected_threshold": fixed_record.get("selected_threshold", 0.0),
        "hybrid_return_combination": fixed_record.get("hybrid_return_combination", getattr(args, "hybrid_return_combination", "probability_times_return")),
        "hybrid_min_probability": fixed_record.get("hybrid_min_probability", getattr(args, "hybrid_min_probability", 0.0)),
        "conditional_expected_win_return": fixed_record.get("conditional_expected_win_return", 0.0),
        "conditional_expected_loss_return": fixed_record.get("conditional_expected_loss_return", 0.0),
        "conditional_payoff_rows": fixed_record.get("conditional_payoff_rows", 0),
        "conditional_payoff_positive_rows": fixed_record.get("conditional_payoff_positive_rows", 0),
        "conditional_payoff_negative_rows": fixed_record.get("conditional_payoff_negative_rows", 0),
        "conditional_payoff_source": fixed_record.get("conditional_payoff_source", "not_used"),
        "ev_payoff_mode": fixed_record.get("ev_payoff_mode", getattr(args, "ev_payoff_mode", "fixed_targets")),
        "ev_payoff_source": fixed_record.get("ev_payoff_source", "fixed_targets"),
        "regression_calibration": fixed_record.get("regression_calibration", "none"),
        "regression_target": fixed_record.get("regression_target", getattr(args, "regression_target", "trade_return")),
        "hybrid_score_mode": fixed_record.get("hybrid_score_mode", getattr(args, "hybrid_score_mode", "basic")),
        "hybrid_uncertainty_method": fixed_record.get("hybrid_uncertainty_method", getattr(args, "hybrid_uncertainty_method", "none")),
        "dynamic_hybrid_thresholds": fixed_record.get("dynamic_hybrid_thresholds", getattr(args, "dynamic_hybrid_thresholds", "none")),
        "meta_filter": fixed_record.get("meta_filter", getattr(args, "meta_filter", "none")),
        "symbol_filter_stage": fixed_record.get("symbol_filter_stage", getattr(args, "symbol_filter_stage", "executed")),
        "threshold_tiebreaker": fixed_record.get("threshold_tiebreaker", getattr(args, "threshold_tiebreaker", "fewer_trades")),
        "ensemble_windows": fixed_record.get("ensemble_windows", ",".join(str(value) for value in getattr(args, "ensemble_window_list", []))),
        "portfolio_profit": fixed_record.get("portfolio_profit", 0.0),
        "portfolio_return": fixed_record.get("portfolio_return", 0.0),
        "precision": fixed_record.get("precision", 0.0),
        "recall": fixed_record.get("recall", 0.0),
        "raw_precision": fixed_record.get("raw_precision", 0.0),
        "raw_recall": fixed_record.get("raw_recall", 0.0),
        "walk_forward_folds": len(walk_records) - 1 if aggregate else 0,
        "walk_forward_portfolio_profit": aggregate.get("portfolio_profit", 0.0) if aggregate else 0.0,
        "walk_forward_portfolio_return": aggregate.get("portfolio_return", 0.0) if aggregate else 0.0,
        "walkforward_profitable_fold_rate": walkforward_summary.get("walkforward_profitable_fold_rate", 0.0),
        "walkforward_median_portfolio_return": walkforward_summary.get("walkforward_median_portfolio_return", 0.0),
        "walkforward_mean_portfolio_return": walkforward_summary.get("walkforward_mean_portfolio_return", 0.0),
        "active_fold_rate": walkforward_summary.get("active_fold_rate", 0.0),
        "active_profitable_fold_rate": walkforward_summary.get("active_profitable_fold_rate", 0.0),
        "median_active_fold_return": walkforward_summary.get("median_active_fold_return", 0.0),
        "worst_active_fold_return": walkforward_summary.get("worst_active_fold_return", 0.0),
        "walkforward_total_predicted_trades": walkforward_summary.get("walkforward_total_predicted_trades", 0.0),
        "overactive_losing_folds": walkforward_summary.get("overactive_losing_folds", 0),
        "avg_trades_in_losing_active_folds": walkforward_summary.get("avg_trades_in_losing_active_folds", 0.0),
        "selected_objective_finite_folds": aggregate.get("selected_objective_finite_folds", 0) if aggregate else 0,
        "selected_objective_nonfinite_folds": aggregate.get("selected_objective_nonfinite_folds", 0) if aggregate else 0,
        "selected_no_trade_folds": aggregate.get("selected_no_trade_folds", 0) if aggregate else 0,
        "mean_selected_validation_trade_count": aggregate.get("mean_selected_validation_trade_count", 0.0) if aggregate else 0.0,
        "mean_selected_validation_max_drawdown": aggregate.get("mean_selected_validation_max_drawdown", 0.0) if aggregate else 0.0,
        "mean_selected_base_objective_score": aggregate.get("mean_selected_base_objective_score", 0.0) if aggregate else 0.0,
        "mean_selected_penalized_objective_score": aggregate.get("mean_selected_penalized_objective_score", 0.0) if aggregate else 0.0,
        "acceptance_tier": walkforward_summary.get("acceptance_tier", "none"),
        "accepted": walkforward_summary.get("accepted", 1),
        "failed_acceptance_checks": walkforward_summary.get("failed_acceptance_checks", ""),
        "rejection_reason": walkforward_summary.get("rejection_reason", ""),
        "strategy_strength": walkforward_summary.get("strategy_strength", "not_checked"),
        "ev_upside_target_source": getattr(args, "ev_upside_target_source", "default"),
        "ev_downside_stop_source": getattr(args, "ev_downside_stop_source", "default"),
        "manifest_upside_target": getattr(args, "manifest_upside_target", 0.0),
        "manifest_downside_stop": getattr(args, "manifest_downside_stop", 0.0),
        "effective_upside_target": effective_upside_target,
        "effective_downside_stop": effective_downside_stop,
        "market_regime_features": int(bool(getattr(args, "market_regime_features", False))),
        "market_breadth_features": int(bool(getattr(args, "market_breadth_features", False))),
        "normalized_microsecond_open_times": CACHE_LOAD_INFO.get(
            "normalized_microsecond_open_times",
            NORMALIZED_MICROSECOND_OPEN_TIMES,
        ),
        "max_rss_gb_observed": MAX_RSS_GIB_OBSERVED,
        "max_rss_stage": MAX_RSS_STAGE,
        "feature_storage": args.feature_storage,
        "max_train_rows": args.max_train_rows,
        "prediction_batch_rows": args.prediction_batch_rows,
        "best_params": json.dumps(fixed_selected.get("params", {}), sort_keys=True),
    }
    write_header = not os.path.exists(args.experiment_summary_out) or os.path.getsize(args.experiment_summary_out) == 0
    with open(args.experiment_summary_out, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=experiment_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(experiment)
    log_memory("Run summaries written")


def build_parser():
    parser = argparse.ArgumentParser(description="Train/evaluate boosted trees on generated kline samples.")
    parser.add_argument("--input", default="kline_growth_training.csv")
    parser.add_argument("--model", choices=["auto", "lightgbm", "internal"], default="auto")
    parser.add_argument("--split-mode", choices=["fixed", "ratio"], default="fixed")
    parser.add_argument("--objective-mode", choices=["classification", "return_regression", "hybrid"], default="classification")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--validation-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--threshold-grid", default="0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99")
    parser.add_argument("--threshold-objective", choices=["profit", "profit_balanced", "avg_profit", "precision", "recall", "f1", "ev"], default="profit_balanced")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--validation-slippage-multiplier", type=float, default=1.0)
    parser.add_argument("--test-slippage-multiplier", type=float, default=1.0)
    parser.add_argument("--execution-delay-minutes", type=int, default=0)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    parser.add_argument("--max-validation-trades", type=int, default=250)
    parser.add_argument("--min-validation-precision", type=float, default=0.25)
    parser.add_argument("--min-selected-threshold", type=float, default=0.90)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--max-position-fraction", type=float, default=0.10)
    parser.add_argument("--max-volume-fraction", type=float, default=0.01)
    parser.add_argument("--max-trades-per-period", type=int, default=10)
    parser.add_argument("--max-trades-per-day", type=int, default=0)
    parser.add_argument("--max-trades-per-fold", type=int, default=0)
    parser.add_argument("--max-losing-trades-per-day", type=int, default=0)
    parser.add_argument("--max-daily-drawdown", type=float, default=0.0)
    parser.add_argument("--pause-after-drawdown-minutes", type=int, default=0)
    parser.add_argument("--trade-period-minutes", type=int, default=60)
    parser.add_argument("--holding-period-minutes", type=int, default=5)
    parser.add_argument("--trade-selection", choices=["threshold", "topk_ev", "topk_score"], default="threshold")
    parser.add_argument("--trade-score", choices=["auto", "probability", "ev", "predicted_return", "hybrid"], default="auto")
    parser.add_argument("--top-k-per-minute", type=int, default=3)
    parser.add_argument("--profit-safety", choices=["strict", "explore"], default="explore")
    parser.add_argument("--upside-target", type=float, default=0.05)
    parser.add_argument("--downside-stop", type=float, default=0.02)
    parser.add_argument("--ev-safety-margin", type=float, default=0.0)
    parser.add_argument("--ev-payoff-mode", choices=["fixed_targets", "empirical_validation", "predicted_return"], default="fixed_targets")
    parser.add_argument("--ev-payoff-calibration-max-rows", type=int, default=500000)
    parser.add_argument("--ev-payoff-min-positive-rows", type=int, default=25)
    parser.add_argument("--ev-payoff-min-negative-rows", type=int, default=25)
    parser.add_argument("--min-predicted-net-return", type=float, default=0.0)
    parser.add_argument("--hybrid-min-score", type=float, default=0.0)
    parser.add_argument("--hybrid-return-combination", choices=["probability_times_return", "expected_return", "conditional_payoff"], default="probability_times_return")
    parser.add_argument("--hybrid-min-probability", type=float, default=0.0)
    parser.add_argument("--conditional-payoff-min-positive-rows", type=int, default=25)
    parser.add_argument("--conditional-payoff-min-negative-rows", type=int, default=25)
    parser.add_argument("--conditional-payoff-max-rows", type=int, default=500000)
    parser.add_argument("--dynamic-hybrid-thresholds", choices=["none", "btc_regime", "volatility_regime", "btc_volatility_regime"], default="none")
    parser.add_argument("--btc-bullish-threshold", type=float, default=0.01)
    parser.add_argument("--btc-bearish-threshold", type=float, default=-0.01)
    parser.add_argument("--hybrid-min-score-bullish", type=float, default=0.001)
    parser.add_argument("--hybrid-min-score-neutral", type=float, default=0.0015)
    parser.add_argument("--hybrid-min-score-bearish", type=float, default=0.0025)
    parser.add_argument("--volatility-high-threshold", type=float, default=0.02)
    parser.add_argument("--hybrid-min-score-high-vol", type=float, default=0.0025)
    parser.add_argument("--hybrid-min-score-normal-vol", type=float, default=0.001)
    parser.add_argument("--hybrid-score-mode", choices=["basic", "risk_adjusted"], default="basic")
    parser.add_argument("--hybrid-uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--hybrid-uncertainty-penalty-mode", choices=["raw", "relative_return"], default="relative_return")
    parser.add_argument("--hybrid-uncertainty-method", choices=["none", "bucket_residual", "global_residual"], default="none")
    parser.add_argument("--hybrid-uncertainty-buckets", type=int, default=10)
    parser.add_argument("--meta-filter", choices=["none", "logistic", "lightgbm"], default="none")
    parser.add_argument("--meta-filter-min-probability", type=float, default=0.5)
    parser.add_argument("--meta-filter-max-rows", type=int, default=500000)
    parser.add_argument("--symbol-validation-filter", choices=["none", "positive_avg_profit", "positive_total_profit"], default="none")
    parser.add_argument("--symbol-filter-stage", choices=["executed", "eligible", "candidate_blend"], default="executed")
    parser.add_argument("--symbol-filter-min-candidates", type=int, default=25)
    parser.add_argument("--symbol-filter-min-executed", type=int, default=5)
    parser.add_argument("--symbol-filter-candidate-weight", type=float, default=0.5)
    parser.add_argument("--symbol-filter-executed-weight", type=float, default=0.5)
    parser.add_argument("--symbol-filter-shrinkage", type=float, default=50.0)
    parser.add_argument("--min-symbol-validation-trades", type=int, default=3)
    parser.add_argument("--min-symbol-validation-average-profit", type=float, default=0.0)
    parser.add_argument("--min-symbol-validation-total-profit", type=float, default=0.0)
    parser.add_argument("--regression-target", choices=["trade_return", "net_return", "clipped_trade_return", "clipped_net_return", "risk_adjusted_return"], default="trade_return")
    parser.add_argument("--regression-clip-min", type=float, default=-0.03)
    parser.add_argument("--regression-clip-max", type=float, default=0.05)
    parser.add_argument("--risk-adjusted-return-epsilon", type=float, default=1e-6)
    parser.add_argument("--regression-calibration", choices=["none", "linear", "isotonic-lite"], default="none")
    parser.add_argument("--regression-calibration-max-rows", type=int, default=0)
    parser.add_argument("--regression-calibration-buckets", type=int, default=20)
    parser.add_argument("--ensemble-windows", default="")
    parser.add_argument("--threshold-drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--threshold-trade-count-penalty", type=float, default=0.0)
    parser.add_argument("--target-validation-trades", type=int, default=0)
    parser.add_argument("--threshold-tiebreaker", choices=["fewer_trades", "target_trades", "active_days", "balanced"], default="fewer_trades")
    parser.add_argument("--threshold-tie-epsilon", type=float, default=1e-9)
    parser.add_argument("--threshold-target-trades", type=int, default=0)
    parser.add_argument("--threshold-target-active-days", type=int, default=0)
    parser.add_argument("--overactive-trade-threshold", type=int, default=150)
    parser.add_argument("--disable-adaptive-thresholds", action="store_true")
    parser.add_argument("--positive-weight-cap", type=float, default=50.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-bin", type=int, default=63)
    parser.add_argument("--subsample-for-bin", type=int, default=100000)
    parser.add_argument("--lightgbm-histogram-pool-mb", type=float, default=128.0)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--eval-metric", choices=["binary_logloss", "auc"], default="binary_logloss")
    parser.add_argument("--log-evaluation-period", type=int, default=50)
    parser.add_argument("--internal-estimators", type=int, default=24)
    parser.add_argument("--internal-learning-rate", type=float, default=0.08)
    parser.add_argument("--internal-bins", type=int, default=12)
    parser.add_argument("--internal-l2", type=float, default=2.0)
    parser.add_argument("--feature-storage", choices=["auto", "memmap32", "memmap64", "matrix32", "matrix64", "float32", "float64", "list"], default="auto")
    parser.add_argument("--augment-market-breadth-features", action="store_true")
    parser.add_argument("--market-breadth-min-symbols", type=int, default=5)
    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--max-rss-gb", type=float, default=0.0)
    parser.add_argument("--abort-on-memory-limit", action="store_true")
    parser.add_argument("--min-free-disk-gb", type=float, default=0.0)
    parser.add_argument("--abort-on-low-disk", action="store_true")
    parser.add_argument("--max-cache-size-gb", type=float, default=0.0)
    parser.add_argument("--memmap-dir", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--cache-layout", choices=["monolithic", "sharded"], default="monolithic")
    parser.add_argument("--cache-shard-by", choices=["month"], default="month")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--smoke-test-cache", action="store_true")
    parser.add_argument("--inspect-cache", action="store_true")
    parser.add_argument("--cache-cleanup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-delete", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=2000000)
    parser.add_argument("--max-validation-rows", type=int, default=1000000)
    parser.add_argument("--max-final-train-rows", type=int, default=2000000)
    parser.add_argument("--train-sample-mode", choices=["stratified", "balanced", "chronological"], default="stratified")
    parser.add_argument("--max-positive-sample-fraction", type=float, default=0.33)
    parser.add_argument("--prediction-batch-rows", type=int, default=200000)
    parser.add_argument("--adaptive-threshold-sample-rows", type=int, default=1000000)
    parser.add_argument("--auc-sample-rows", type=int, default=1000000)
    parser.add_argument("--calibration", choices=["none", "platt"], default="none")
    parser.add_argument("--calibration-max-rows", type=int, default=0)
    parser.add_argument("--skip-full-validation-retune", action="store_true")
    parser.add_argument("--fixed-validation-backfill-months", type=int, default=2)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--walk-train-months", type=int, default=6)
    parser.add_argument("--walk-validation-months", type=int, default=1)
    parser.add_argument("--walk-forward-final-model", choices=["selected", "refit"], default="selected")
    parser.add_argument("--require-positive-walkforward", action="store_true")
    parser.add_argument("--min-profitable-fold-rate", type=float, default=0.55)
    parser.add_argument("--min-median-fold-return", type=float, default=0.0)
    parser.add_argument("--min-mean-fold-return", type=float, default=0.0)
    parser.add_argument("--max-worst-fold-drawdown", type=float, default=1.0)
    parser.add_argument("--acceptance-tier", choices=["none", "exploration", "research", "strong"], default="none")
    parser.add_argument("--prediction-output-mode", choices=["all", "trades", "none"], default="trades")
    parser.add_argument("--predictions-out", default="kline_growth_predictions_gbdt.csv")
    parser.add_argument("--metrics-out", default="kline_growth_metrics_gbdt.csv")
    parser.add_argument("--walkforward-metrics-out", default="kline_growth_walkforward_metrics.csv")
    parser.add_argument("--walkforward-diagnostics-out", default="kline_growth_walkforward_diagnostics.csv")
    parser.add_argument("--symbol-filter-diagnostics-out", default="kline_growth_symbol_filter_diagnostics.csv")
    parser.add_argument("--walk-predictions-out", default="kline_growth_predictions_gbdt_walkforward.csv")
    parser.add_argument("--feature-importance-out", default="kline_growth_feature_importance.csv")
    parser.add_argument("--logistic-metrics-in", default="kline_growth_metrics_logistic.csv")
    parser.add_argument("--run-summary-out", default="kline_growth_run_summary.json")
    parser.add_argument("--experiment-summary-out", default="kline_growth_experiment_summary.csv")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--compress-outputs", action="store_true")
    parser.add_argument("--output-compression", choices=["gzip", "none"], default="none")
    parser.add_argument("--profile-out", default="kline_growth_pipeline_profile.csv")
    parser.add_argument("--disable-profile", action="store_true")
    parser.add_argument("--cooldown-minutes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--max-trades-per-symbol-month", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main(argv):
    global AUC_SAMPLE_ROWS
    parser = build_parser()
    explicit_flags = parse_explicit_flags(argv)
    args = parser.parse_args(argv)
    profile_reset(not args.disable_profile)
    args._explicit_flags = explicit_flags
    configure_output_paths(args)
    budget_applied = apply_memory_budget_defaults(args, explicit_flags)
    if args.min_validation_trades < 0:
        raise ValueError("--min-validation-trades cannot be negative")
    if args.max_validation_trades < 0:
        raise ValueError("--max-validation-trades cannot be negative")
    if args.max_validation_trades and args.max_validation_trades < args.min_validation_trades:
        raise ValueError("--max-validation-trades must be 0 or at least --min-validation-trades")
    if not 0.0 <= args.min_validation_precision <= 1.0:
        raise ValueError("--min-validation-precision must be between 0 and 1")
    if not 0.0 <= args.min_selected_threshold <= 1.01:
        raise ValueError("--min-selected-threshold must be between 0 and 1.01")
    if args.validation_slippage_multiplier < 0.0:
        raise ValueError("--validation-slippage-multiplier cannot be negative")
    if args.test_slippage_multiplier < 0.0:
        raise ValueError("--test-slippage-multiplier cannot be negative")
    if args.initial_capital <= 0.0:
        raise ValueError("--initial-capital must be positive")
    if not 0.0 < args.max_position_fraction <= 1.0:
        raise ValueError("--max-position-fraction must be between 0 and 1")
    if not 0.0 < args.max_volume_fraction <= 1.0:
        raise ValueError("--max-volume-fraction must be between 0 and 1")
    if args.max_trades_per_period < 0:
        raise ValueError("--max-trades-per-period cannot be negative")
    if args.max_trades_per_day < 0:
        raise ValueError("--max-trades-per-day cannot be negative")
    if args.market_breadth_min_symbols <= 0:
        raise ValueError("--market-breadth-min-symbols must be positive")
    if args.max_trades_per_fold < 0:
        raise ValueError("--max-trades-per-fold cannot be negative")
    if args.max_losing_trades_per_day < 0:
        raise ValueError("--max-losing-trades-per-day cannot be negative")
    if args.max_daily_drawdown < 0.0:
        raise ValueError("--max-daily-drawdown cannot be negative")
    if args.min_free_disk_gb < 0.0:
        raise ValueError("--min-free-disk-gb cannot be negative")
    if args.max_cache_size_gb < 0.0:
        raise ValueError("--max-cache-size-gb cannot be negative")
    if args.pause_after_drawdown_minutes < 0:
        raise ValueError("--pause-after-drawdown-minutes cannot be negative")
    if args.max_trades_per_period and args.trade_period_minutes <= 0:
        raise ValueError("--trade-period-minutes must be positive when --max-trades-per-period is enabled")
    if args.holding_period_minutes <= 0:
        raise ValueError("--holding-period-minutes must be positive")
    if args.top_k_per_minute < 0:
        raise ValueError("--top-k-per-minute cannot be negative")
    if args.upside_target <= 0.0:
        raise ValueError("--upside-target must be positive")
    if args.downside_stop <= 0.0:
        raise ValueError("--downside-stop must be positive")
    if args.ev_payoff_calibration_max_rows < 0:
        raise ValueError("--ev-payoff-calibration-max-rows cannot be negative")
    if args.ev_payoff_min_positive_rows < 0:
        raise ValueError("--ev-payoff-min-positive-rows cannot be negative")
    if args.ev_payoff_min_negative_rows < 0:
        raise ValueError("--ev-payoff-min-negative-rows cannot be negative")
    if not 0.0 <= args.hybrid_min_probability <= 1.0:
        raise ValueError("--hybrid-min-probability must be between 0 and 1")
    if args.conditional_payoff_min_positive_rows < 0:
        raise ValueError("--conditional-payoff-min-positive-rows cannot be negative")
    if args.conditional_payoff_min_negative_rows < 0:
        raise ValueError("--conditional-payoff-min-negative-rows cannot be negative")
    if args.conditional_payoff_max_rows < 0:
        raise ValueError("--conditional-payoff-max-rows cannot be negative")
    if args.threshold_drawdown_penalty < 0.0:
        raise ValueError("--threshold-drawdown-penalty cannot be negative")
    if args.threshold_trade_count_penalty < 0.0:
        raise ValueError("--threshold-trade-count-penalty cannot be negative")
    if args.target_validation_trades < 0:
        raise ValueError("--target-validation-trades cannot be negative")
    if args.threshold_tie_epsilon < 0.0:
        raise ValueError("--threshold-tie-epsilon cannot be negative")
    if args.threshold_target_trades < 0:
        raise ValueError("--threshold-target-trades cannot be negative")
    if args.threshold_target_active_days < 0:
        raise ValueError("--threshold-target-active-days cannot be negative")
    if args.overactive_trade_threshold < 0:
        raise ValueError("--overactive-trade-threshold cannot be negative")
    if args.trade_score == "probability" and args.objective_mode == "return_regression":
        raise ValueError("--trade-score probability is incompatible with --objective-mode return_regression")
    if args.trade_score == "predicted_return" and args.objective_mode == "classification":
        raise ValueError("--trade-score predicted_return requires --objective-mode return_regression or hybrid")
    if args.trade_score == "hybrid" and args.objective_mode != "hybrid":
        raise ValueError("--trade-score hybrid requires --objective-mode hybrid")
    if args.max_bin < 2:
        raise ValueError("--max-bin must be at least 2")
    if args.subsample_for_bin <= 0:
        raise ValueError("--subsample-for-bin must be positive")
    if args.lightgbm_histogram_pool_mb <= 0.0:
        raise ValueError("--lightgbm-histogram-pool-mb must be positive")
    if args.max_train_rows < 0:
        raise ValueError("--max-train-rows cannot be negative")
    if args.max_validation_rows < 0:
        raise ValueError("--max-validation-rows cannot be negative")
    if args.max_final_train_rows < 0:
        raise ValueError("--max-final-train-rows cannot be negative")
    if not 0.0 < args.max_positive_sample_fraction < 1.0:
        raise ValueError("--max-positive-sample-fraction must be between 0 and 1")
    if args.prediction_batch_rows <= 0:
        raise ValueError("--prediction-batch-rows must be positive")
    if args.adaptive_threshold_sample_rows < 2:
        raise ValueError("--adaptive-threshold-sample-rows must be at least 2")
    if args.auc_sample_rows < 0:
        raise ValueError("--auc-sample-rows cannot be negative")
    if args.calibration_max_rows < 0:
        raise ValueError("--calibration-max-rows cannot be negative")
    if args.regression_calibration_max_rows < 0:
        raise ValueError("--regression-calibration-max-rows cannot be negative")
    if args.regression_calibration_buckets < 2:
        raise ValueError("--regression-calibration-buckets must be at least 2")
    if args.hybrid_uncertainty_buckets < 2:
        raise ValueError("--hybrid-uncertainty-buckets must be at least 2")
    if args.fixed_validation_backfill_months < 0:
        raise ValueError("--fixed-validation-backfill-months cannot be negative")
    if args.walk_train_months < 1:
        raise ValueError("--walk-train-months must be at least 1")
    if args.walk_validation_months < 1:
        raise ValueError("--walk-validation-months must be at least 1")
    if args.meta_filter_max_rows < 0:
        raise ValueError("--meta-filter-max-rows cannot be negative")
    if args.min_symbol_validation_trades < 0:
        raise ValueError("--min-symbol-validation-trades cannot be negative")
    if args.symbol_filter_min_candidates < 0:
        raise ValueError("--symbol-filter-min-candidates cannot be negative")
    if args.symbol_filter_min_executed < 0:
        raise ValueError("--symbol-filter-min-executed cannot be negative")
    if args.symbol_filter_candidate_weight < 0.0:
        raise ValueError("--symbol-filter-candidate-weight cannot be negative")
    if args.symbol_filter_executed_weight < 0.0:
        raise ValueError("--symbol-filter-executed-weight cannot be negative")
    if args.symbol_filter_shrinkage < 0.0:
        raise ValueError("--symbol-filter-shrinkage cannot be negative")
    if args.regression_clip_min > args.regression_clip_max:
        raise ValueError("--regression-clip-min must be <= --regression-clip-max")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds cannot be negative")
    if args.log_evaluation_period < 0:
        raise ValueError("--log-evaluation-period cannot be negative")
    if args.cooldown_minutes < 0:
        raise ValueError("--cooldown-minutes cannot be negative")
    if args.max_trades_per_symbol_month < 0:
        raise ValueError("--max-trades-per-symbol-month cannot be negative")
    if args.execution_delay_minutes < 0:
        raise ValueError("--execution-delay-minutes cannot be negative")
    if args.train_ratio <= 0.0 or args.validation_ratio <= 0.0 or args.test_ratio <= 0.0:
        raise ValueError("--train-ratio, --validation-ratio, and --test-ratio must be positive")
    if abs((args.train_ratio + args.validation_ratio + args.test_ratio) - 1.0) > 0.001:
        raise ValueError("--train-ratio + --validation-ratio + --test-ratio must equal 1.0")
    if args.disable_cache and args.rebuild_cache:
        raise ValueError("--disable-cache and --rebuild-cache cannot be used together")
    if args.cache_only and args.disable_cache:
        raise ValueError("--cache-only cannot be combined with --disable-cache")
    if args.cache_only and args.rebuild_cache:
        raise ValueError("--cache-only cannot be combined with --rebuild-cache")
    if (args.cache_only or args.smoke_test_cache) and args.feature_storage not in ("auto", "memmap32", "memmap64"):
        raise ValueError("--cache-only and --smoke-test-cache require memmap-backed feature storage")
    if args.cooldown_minutes:
        print("Warning: --cooldown-minutes is retained as a compatibility alias and is ignored; portfolio entry limits replace cooldown.", file=sys.stderr, flush=True)
    if args.max_trades_per_symbol_month:
        print("Warning: --max-trades-per-symbol-month is retained as a compatibility alias and is ignored; use --max-trades-per-period.", file=sys.stderr, flush=True)
    if args.execution_delay_minutes:
        print("Warning: --execution-delay-minutes requires delayed-entry price data that is not available in this pipeline; continuing with delay disabled.", file=sys.stderr, flush=True)
        args.execution_delay_minutes = 0
    if args.prediction_output_mode == "all":
        print("Warning: --prediction-output-mode all can create very large CSV files.", file=sys.stderr, flush=True)
    if args.acceptance_tier != "none" and args.require_positive_walkforward:
        print("Warning: --acceptance-tier overrides --require-positive-walkforward when both are set.", file=sys.stderr, flush=True)
    if args.compress_outputs and "output_compression" not in explicit_flags and args.output_compression == "none":
        args.output_compression = "gzip"
    if args.cache_layout == "sharded" and not os.path.isdir(args.input):
        print(
            "Warning: --cache-layout sharded currently requires a shard dataset directory input; continuing with monolithic cache behavior.",
            file=sys.stderr,
            flush=True,
        )
    if args.cache_cleanup and not args.dry_run and not args.confirm_delete:
        print("Warning: --cache-cleanup without --confirm-delete performs a dry review only.", file=sys.stderr, flush=True)
    if args.calibration == "platt" and args.calibration_max_rows == 0:
        args.calibration_max_rows = MEMORY_BUDGET_DEFAULTS["calibration_max_rows"] if budget_applied else 500000
    if args.regression_calibration != "none" and args.regression_calibration_max_rows == 0:
        args.regression_calibration_max_rows = MEMORY_BUDGET_DEFAULTS["calibration_max_rows"] if budget_applied else 500000
    args.ensemble_window_list = parse_ensemble_windows(args.ensemble_windows)
    if args.meta_filter != "none" and args.objective_mode != "hybrid":
        raise ValueError("--meta-filter currently requires --objective-mode hybrid")
    if args.dynamic_hybrid_thresholds != "none" and args.objective_mode != "hybrid":
        raise ValueError("--dynamic-hybrid-thresholds currently requires --objective-mode hybrid")
    if args.hybrid_score_mode != "basic" and args.objective_mode != "hybrid":
        raise ValueError("--hybrid-score-mode currently requires --objective-mode hybrid")
    if args.ensemble_window_list and args.walk_forward:
        raise ValueError("--ensemble-windows is currently supported for fixed-split runs only; omit --walk-forward")
    if "input" not in explicit_flags:
        recovered_dataset = discover_recoverable_default_dataset(args.input, args.cache_dir or None)
        if recovered_dataset:
            original_input = os.path.abspath(args.input)
            args.input = recovered_dataset["input_path"]
            if "cache_dir" not in explicit_flags and recovered_dataset.get("cache_dir"):
                args.cache_dir = recovered_dataset["cache_dir"]
            print(
                "Warning: {} is missing its manifest; automatically using {} discovered from {}.".format(
                    original_input,
                    args.input,
                    recovered_dataset["source"],
                ),
                file=sys.stderr,
                flush=True,
            )
            if recovered_dataset.get("cache_dir"):
                print(
                    "Using compatible cache directory: {}".format(args.cache_dir),
                    file=sys.stderr,
                    flush=True,
                )
    training_manifest_for_args, _ = load_training_manifest(args.input)
    apply_manifest_ev_targets(args, training_manifest_for_args, explicit_flags)
    args.thresholds = parse_threshold_grid(args.threshold_grid)
    AUC_SAMPLE_ROWS = args.auc_sample_rows
    kind = choose_model_kind(args.model)
    if args.feature_storage == "auto":
        args.feature_storage = "memmap32" if np is not None else "float32"
    if (args.cache_only or args.smoke_test_cache) and args.feature_storage not in ("memmap32", "memmap64"):
        raise ValueError("--cache-only and --smoke-test-cache require --feature-storage memmap32 or memmap64")
    print_memory_budget_summary(args, budget_applied)
    if args.cache_only:
        print("Cache-only mode enabled: using existing compatible binary cache", flush=True)
    if args.inspect_cache:
        inspect_cache(args, training_manifest_for_args, training_manifest_path(args.input))
        write_pipeline_profile(args.profile_out)
        return 0
    if args.cache_cleanup:
        status = cache_cleanup(args)
        write_pipeline_profile(args.profile_out)
        return status
    rows = None
    feature_names = []
    fixed_selected = {}
    fixed_record = {}
    walk_records = []
    symbol_filter_records = []
    try:
        training_manifest_path_for_args = training_manifest_path(args.input)
        check_free_disk(args.input, args, "startup")
        warn_if_cache_dir_large(resolve_cache_dir(args.input, args.cache_dir or None), args)
        if args.smoke_test_cache:
            smoke_test_cache(
                args.input,
                args.feature_storage,
                args.cache_dir or None,
                training_manifest_for_args,
                training_manifest_path_for_args,
            )
            memory_checkpoint("Cache smoke test complete", args)
            return 0
        check_free_disk(resolve_cache_dir(args.input, args.cache_dir or None), args, "before_cache_load")
        rows, feature_names, has_returns = load_rows(
            args.input,
            args.feature_storage,
            args.memmap_dir or None,
            args.cache_dir or None,
            args.rebuild_cache,
            args.disable_cache,
            training_manifest_for_args,
            training_manifest_path_for_args,
            args.cache_only,
        )
        rows, feature_names = maybe_augment_market_breadth_rows(
            rows,
            feature_names,
            args,
            args.input,
            args.cache_dir or None,
        )
        memory_checkpoint("CSV/cache load complete", args)
        if NORMALIZED_MICROSECOND_OPEN_TIMES:
            log_memory(
                "Normalized {:,} microsecond-like open_time values to milliseconds".format(
                    NORMALIZED_MICROSECOND_OPEN_TIMES
                )
            )
        if not has_returns and args.threshold_objective in ("profit", "profit_balanced", "avg_profit", "ev"):
            print("forward return columns are missing; falling back from {} objective to f1".format(args.threshold_objective), file=sys.stderr, flush=True)
            args.threshold_objective = "f1"
        if not has_returns and args.objective_mode in ("return_regression", "hybrid"):
            raise ValueError("{} requires trade_return columns in the input CSV".format(args.objective_mode))

        model_name = "gbdt_{}".format(kind)
        print("Loaded {} rows with {} features from {} using {} feature storage".format(
            len(rows), len(feature_names), args.input, args.feature_storage), flush=True)
        print("Using {} model path".format(model_name), flush=True)

        fixed_record, fixed_selected, symbol_filter_records = run_fixed_split(rows, feature_names, args, kind, model_name)
        check_free_disk(args.metrics_out, args, "before_metrics_write")
        metrics_write_token = start_profile_stage("metrics_output_write", args.metrics_out)
        write_metrics(args.metrics_out, [fixed_record])
        finish_profile_stage(metrics_write_token, rows_processed=1, extra_info=args.metrics_out)
        gc.collect()
        if args.walk_forward:
            walk_records, walk_symbol_filter_records = run_walk_forward(rows, feature_names, args, kind, model_name)
            symbol_filter_records.extend(walk_symbol_filter_records)
        symbol_diag_token = start_profile_stage("symbol_filter_diagnostics_write", args.symbol_filter_diagnostics_out)
        write_symbol_filter_diagnostics(args.symbol_filter_diagnostics_out, symbol_filter_records)
        finish_profile_stage(symbol_diag_token, rows_processed=len(symbol_filter_records), extra_info=args.symbol_filter_diagnostics_out)
        check_free_disk(args.run_summary_out, args, "before_summary_write")
        summary_write_token = start_profile_stage("summary_output_write", args.run_summary_out)
        write_run_summaries(args, rows, feature_names, kind, fixed_selected, fixed_record, walk_records)
        finish_profile_stage(summary_write_token, rows_processed=1, extra_info=args.run_summary_out)
        print_comparison(fixed_record, walk_records, args)
        del fixed_selected
        gc.collect()
    except MemoryLimitExceeded as error:
        print(
            "gbdt_pipeline stopped at {} after RSS reached {:.2f} GiB (limit {:.2f} GiB)".format(
                error.stage,
                error.rss_gib,
                error.limit_gib,
            ),
            file=sys.stderr,
            flush=True,
        )
        if rows is not None:
            write_run_summaries(args, rows, feature_names, kind, fixed_selected or {}, fixed_record or {}, walk_records)
        return 2
    finally:
        if rows is not None and is_compact_rows(rows):
            rows.cleanup()
        for path in list(TEMP_PREDICTION_PATHS):
            try:
                os.remove(path)
            except OSError:
                pass
            TEMP_PREDICTION_PATHS.discard(path)
        gc.collect()
        write_pipeline_profile(args.profile_out)
        postprocess_output_files(args)
        log_memory("Cleanup complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as error:
        print("gbdt_pipeline failed: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
