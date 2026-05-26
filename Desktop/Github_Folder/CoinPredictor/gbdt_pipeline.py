#!/usr/bin/env python3
"""Train/evaluate a boosted-tree model from kline_growth_training.csv.

The script uses LightGBM when it is installed and otherwise falls back to a
small standard-library boosted-stump model. Splits are chronological by each
symbol's month_index.
"""

import argparse
from array import array
import bisect
import csv
import gc
import importlib.util
import math
import os
import sys


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
])


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


def parse_threshold_grid(text):
    values = [safe_float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("threshold grid cannot be empty")
    return sorted(values)


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


def adaptive_thresholds(probabilities, base_thresholds, min_validation_trades):
    thresholds = set(value for value in base_thresholds if 0.0 <= value <= 1.0)
    ordered = sorted(value for value in probabilities if 0.0 <= value <= 1.0)
    if not ordered:
        return sorted(thresholds)

    quantiles = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
    for quantile in quantiles:
        index = int((len(ordered) - 1) * quantile)
        thresholds.add(ordered[index])

    # Include thresholds that roughly target small, medium, and large numbers of
    # raw validation signals. Cooldown is applied later during evaluation.
    target_counts = [max(1, min_validation_trades), max(1, min_validation_trades * 2), 10, 25, 50, 100, 250, 500, 1000]
    for count in target_counts:
        if count <= len(ordered):
            thresholds.add(ordered[-count])

    thresholds.add(max(0.0, ordered[-1] - 1e-12))
    return sorted(thresholds)


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
        "features",
    )

    def __init__(self, symbol, month, month_index, open_time, label, forward_return, trade_return,
                 max_future_high_return, max_future_low_return, features):
        self.symbol = symbol
        self.month = month
        self.month_index = month_index
        self.open_time = open_time
        self.label = label
        self.forward_return = forward_return
        self.trade_return = trade_return
        self.max_future_high_return = max_future_high_return
        self.max_future_low_return = max_future_low_return
        self.features = features


def make_row(item, feature_columns, month_index_lookup, text_cache, feature_storage):
    symbol = cached_text(text_cache, item.get("symbol", ""))
    month = cached_text(text_cache, item.get("month", ""))
    month_index = int(safe_float(item.get("month_index"), month_index_lookup.get((symbol, month), 0)))
    open_time = int(safe_float(item.get("open_time"), 0.0))
    label = 1 if str(item.get("label", "0")).strip() == "1" else 0
    forward_return = safe_float(item.get("forward_return"), 0.0)
    trade_return = safe_float(item.get("trade_return"), forward_return)
    max_future_high_return = safe_float(item.get("max_future_high_return"), forward_return)
    max_future_low_return = safe_float(item.get("max_future_low_return"), forward_return)
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
        build_features(item, feature_columns, feature_storage),
    )


def load_rows(path, feature_storage="float32"):
    with open(path, newline="") as handle:
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


def labels(rows):
    return [row.label for row in rows]


def matrix(rows):
    return [row.features for row in rows]


def select_month_range(rows, start, end):
    return [row for row in rows if start <= row.month_index < end]


def auc_score_from_rows(probabilities, rows):
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


def trade_limit_key(row):
    return (row.symbol, row.month)


def evaluate(rows, probabilities, threshold, fee, slippage, cooldown_minutes=0,
             max_trades_per_symbol_month=0, compute_auc=True):
    actual_positive = 0
    predicted_trades = 0
    tp = fp = tn = fn = 0
    returns = []
    trade_returns = []
    sum_return = 0.0
    sum_trade_return = 0.0
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
    next_allowed_signal_time = {}
    trades_by_symbol_month = {}
    cooldown_ms = max(0, int(cooldown_minutes)) * 60 * 1000

    for row, probability in zip(rows, probabilities):
        if row.label == 1:
            actual_positive += 1
        raw_signal = probability >= threshold
        predicted = False
        if raw_signal:
            next_allowed = next_allowed_signal_time.get(row.symbol, 0)
            predicted = row.open_time >= next_allowed
            if predicted and max_trades_per_symbol_month > 0:
                limit_key = trade_limit_key(row)
                if trades_by_symbol_month.get(limit_key, 0) >= max_trades_per_symbol_month:
                    predicted = False
            if predicted:
                if max_trades_per_symbol_month > 0:
                    limit_key = trade_limit_key(row)
                    trades_by_symbol_month[limit_key] = trades_by_symbol_month.get(limit_key, 0) + 1
                if cooldown_ms:
                    next_allowed_signal_time[row.symbol] = row.open_time + cooldown_ms
        if not predicted:
            if row.label == 1:
                fn += 1
            else:
                tn += 1
            continue

        predicted_trades += 1
        if row.label == 1:
            tp += 1
        else:
            fp += 1

        after_fee = row.trade_return - fee
        after_fee_slippage = row.trade_return - fee - slippage
        total_fee += after_fee
        total_fee_slippage += after_fee_slippage
        sum_return += row.forward_return
        sum_trade_return += row.trade_return
        sum_mfe += row.max_future_high_return
        sum_mae += row.max_future_low_return
        returns.append(row.forward_return)
        trade_returns.append(row.trade_return)
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
    precision = float(tp) / predicted_trades if predicted_trades else 0.0
    recall = float(tp) / actual_positive if actual_positive else 0.0
    accuracy = float(tp + tn) / total if total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    trade_count = float(predicted_trades) if predicted_trades else 1.0

    return {
        "rows": total,
        "actual_positive_rows": actual_positive,
        "predicted_trades": predicted_trades,
        "true_positive_rows": tp,
        "false_positive_rows": fp,
        "true_negative_rows": tn,
        "false_negative_rows": fn,
        "auc": auc_score_from_rows(probabilities, rows) if compute_auc else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "win_rate": float(winning_trades) / predicted_trades if predicted_trades else 0.0,
        "average_forward_return": sum_return / trade_count if predicted_trades else 0.0,
        "median_forward_return": median(returns),
        "average_trade_return": sum_trade_return / trade_count if predicted_trades else 0.0,
        "median_trade_return": median(trade_returns),
        "average_max_favorable_excursion": sum_mfe / trade_count if predicted_trades else 0.0,
        "average_max_adverse_excursion": sum_mae / trade_count if predicted_trades else 0.0,
        "average_profit_after_fee": total_fee / trade_count if predicted_trades else 0.0,
        "average_profit_after_fee_and_slippage": total_fee_slippage / trade_count if predicted_trades else 0.0,
        "total_profit_after_fee": total_fee,
        "total_profit_after_fee_and_slippage": total_fee_slippage,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "max_drawdown": max_drawdown,
        "selected_threshold": threshold,
    }


def threshold_score(metrics, objective, zero_trade_profit_score=0.0):
    if metrics["predicted_trades"] == 0:
        return zero_trade_profit_score if objective in ("profit", "avg_profit") else -float("inf")
    if objective == "avg_profit":
        return metrics["average_profit_after_fee_and_slippage"]
    if objective == "precision":
        return metrics["precision"]
    if objective == "recall":
        return metrics["recall"]
    if objective == "f1":
        return metrics["f1"]
    return metrics["total_profit_after_fee_and_slippage"]


def tune_threshold(rows, probabilities, thresholds, objective, fee, slippage, cooldown_minutes,
                   min_validation_trades, max_validation_trades, min_validation_precision,
                   max_trades_per_symbol_month, profit_safety):
    best_threshold = thresholds[0]
    best_metrics = None
    best_score = -float("inf")
    profit_objective = objective in ("profit", "avg_profit")
    strict_profit = profit_objective and profit_safety == "strict"
    zero_trade_profit_score = 0.0 if strict_profit else -float("inf")
    fallback_threshold = None
    fallback_metrics = None
    fallback_score = -float("inf")
    if strict_profit:
        best_threshold = 1.01
        best_metrics = evaluate(
            rows,
            probabilities,
            best_threshold,
            fee,
            slippage,
            cooldown_minutes,
            max_trades_per_symbol_month,
            compute_auc=False,
        )
        best_score = 0.0
    for threshold in thresholds:
        metrics = evaluate(
            rows,
            probabilities,
            threshold,
            fee,
            slippage,
            cooldown_minutes,
            max_trades_per_symbol_month,
            compute_auc=False,
        )
        score = threshold_score(metrics, objective, zero_trade_profit_score)
        too_few = metrics["predicted_trades"] < min_validation_trades
        too_many = max_validation_trades > 0 and metrics["predicted_trades"] > max_validation_trades
        too_imprecise = metrics["predicted_trades"] > 0 and metrics["precision"] < min_validation_precision
        if too_many or too_imprecise:
            continue
        if too_few:
            if metrics["predicted_trades"] > 0 and score > fallback_score:
                fallback_threshold = threshold
                fallback_metrics = metrics
                fallback_score = score
            continue
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    if best_metrics is None:
        if fallback_metrics is not None:
            best_threshold = fallback_threshold
            best_metrics = fallback_metrics
        else:
            best_threshold = 1.01
            best_metrics = evaluate(
                rows,
                probabilities,
                best_threshold,
                fee,
                slippage,
                cooldown_minutes,
                max_trades_per_symbol_month,
                compute_auc=False,
            )
    return best_threshold, best_metrics


class InternalStumpGBDT(object):
    def __init__(self, n_estimators=12, learning_rate=0.12, max_bins=8, l2=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_bins = max_bins
        self.l2 = l2
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

    def fit(self, x_train, y_train, feature_names):
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

    def feature_importance(self, feature_names):
        total = sum(self.importances)
        rows = []
        for name, value in zip(feature_names, self.importances):
            rows.append((name, value, value / total if total else 0.0))
        return rows


class ExternalModel(object):
    def __init__(self, model, kind):
        self.model = model
        self.kind = kind

    def fit(self, x_train, y_train, feature_names):
        del feature_names
        self.model.fit(x_train, y_train)
        return self

    def predict_proba(self, x_rows):
        probabilities = self.model.predict_proba(x_rows)
        return [float(row[1]) for row in probabilities]

    def feature_importance(self, feature_names):
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        total = float(sum(importances))
        return [
            (name, float(value), float(value) / total if total else 0.0)
            for name, value in zip(feature_names, importances)
        ]


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
    positives = sum(y_train)
    negatives = len(y_train) - positives
    if positives <= 0:
        return 1.0
    return min(cap, negatives / float(positives))


def make_model(kind, params, positive_weight):
    if kind == "lightgbm":
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
            reg_lambda=params["reg_lambda"],
            objective="binary",
            scale_pos_weight=positive_weight,
            random_state=17,
            verbosity=-1,
        )
        return ExternalModel(model, kind)
    return InternalStumpGBDT(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_bins=params["max_bins"],
        l2=params["l2"],
    )


def candidate_params(kind, args):
    if kind == "lightgbm":
        return [
            {"n_estimators": args.n_estimators, "learning_rate": args.learning_rate, "num_leaves": 31,
             "max_depth": -1, "subsample": 0.9, "colsample_bytree": 0.9,
             "min_child_samples": 50, "reg_lambda": 2.0},
            {"n_estimators": max(80, args.n_estimators // 2), "learning_rate": args.learning_rate * 1.6,
             "num_leaves": 31, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.85,
             "min_child_samples": 80, "reg_lambda": 4.0},
            {"n_estimators": int(args.n_estimators * 1.5), "learning_rate": args.learning_rate * 0.7,
             "num_leaves": 63, "max_depth": -1, "subsample": 0.85, "colsample_bytree": 0.9,
             "min_child_samples": 60, "reg_lambda": 3.0},
        ]
    return [
        {"n_estimators": max(2, args.internal_estimators // 2), "learning_rate": args.internal_learning_rate,
         "max_bins": args.internal_bins, "l2": args.internal_l2},
        {"n_estimators": args.internal_estimators, "learning_rate": args.internal_learning_rate,
         "max_bins": args.internal_bins, "l2": args.internal_l2},
        {"n_estimators": max(args.internal_estimators * 2, 24), "learning_rate": args.internal_learning_rate * 0.7,
         "max_bins": max(args.internal_bins, 16), "l2": args.internal_l2 * 1.5},
    ]


def fit_select_model(train_rows, validation_rows, feature_names, args, kind):
    x_train = matrix(train_rows)
    y_train = labels(train_rows)
    x_validation = matrix(validation_rows)
    positive_weight = class_weight_ratio(y_train, args.positive_weight_cap)

    best = None
    for params in candidate_params(kind, args):
        model = make_model(kind, params, positive_weight)
        model.fit(x_train, y_train, feature_names)
        probabilities = model.predict_proba(x_validation)
        thresholds = args.thresholds
        if not args.disable_adaptive_thresholds:
            thresholds = adaptive_thresholds(probabilities, args.thresholds, args.min_validation_trades)
        thresholds = [threshold for threshold in thresholds if threshold >= args.min_selected_threshold]
        if not thresholds:
            thresholds = [1.01]
        threshold, metrics = tune_threshold(
            validation_rows,
            probabilities,
            thresholds,
            args.threshold_objective,
            args.fee,
            args.slippage,
            args.cooldown_minutes,
            args.min_validation_trades,
            args.max_validation_trades,
            args.min_validation_precision,
            args.max_trades_per_symbol_month,
            args.profit_safety,
        )
        zero_trade_score = 0.0 if args.threshold_objective in ("profit", "avg_profit") and args.profit_safety == "strict" else -float("inf")
        score = threshold_score(metrics, args.threshold_objective, zero_trade_score)
        if best is None or score > best["score"]:
            best = {
                "model": model,
                "params": params,
                "threshold": threshold,
                "validation_metrics": metrics,
                "score": score,
            }
    return best


def write_predictions(path, rows, probabilities, threshold, model_name, cooldown_minutes,
                      max_trades_per_symbol_month=0, append=False):
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
                "selected_threshold",
                "raw_signal",
                "predicted",
                "forward_return",
                "trade_return",
                "max_future_high_return",
                "max_future_low_return",
                "model_name",
            ])
        threshold_is_list = isinstance(threshold, list)
        next_allowed_signal_time = {}
        trades_by_symbol_month = {}
        cooldown_ms = max(0, int(cooldown_minutes)) * 60 * 1000
        for index, (row, probability) in enumerate(zip(rows, probabilities)):
            row_threshold = threshold[index] if threshold_is_list else threshold
            raw_signal = probability >= row_threshold
            predicted = False
            if raw_signal:
                next_allowed = next_allowed_signal_time.get(row.symbol, 0)
                predicted = row.open_time >= next_allowed
                if predicted and max_trades_per_symbol_month > 0:
                    limit_key = trade_limit_key(row)
                    if trades_by_symbol_month.get(limit_key, 0) >= max_trades_per_symbol_month:
                        predicted = False
                if predicted:
                    if max_trades_per_symbol_month > 0:
                        limit_key = trade_limit_key(row)
                        trades_by_symbol_month[limit_key] = trades_by_symbol_month.get(limit_key, 0) + 1
                    if cooldown_ms:
                        next_allowed_signal_time[row.symbol] = row.open_time + cooldown_ms
            writer.writerow([
                row.symbol,
                row.month,
                row.month_index,
                row.open_time,
                row.label,
                "{:.12g}".format(probability),
                "{:.12g}".format(row_threshold),
                1 if raw_signal else 0,
                1 if predicted else 0,
                "{:.12g}".format(row.forward_return),
                "{:.12g}".format(row.trade_return),
                "{:.12g}".format(row.max_future_high_return),
                "{:.12g}".format(row.max_future_low_return),
                model_name,
            ])


METRIC_COLUMNS = [
    "model",
    "split",
    "threshold_objective",
    "selected_threshold",
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
    "win_rate",
    "average_forward_return",
    "median_forward_return",
    "average_trade_return",
    "median_trade_return",
    "average_max_favorable_excursion",
    "average_max_adverse_excursion",
    "average_profit_after_fee",
    "average_profit_after_fee_and_slippage",
    "total_profit_after_fee",
    "total_profit_after_fee_and_slippage",
    "profit_factor",
    "max_drawdown",
    "cooldown_minutes",
    "max_trades_per_symbol_month",
    "min_validation_trades",
    "max_validation_trades",
    "min_validation_precision",
    "min_selected_threshold",
    "profit_safety",
    "adaptive_thresholds",
    "validation_predicted_trades",
    "validation_precision",
    "validation_recall",
    "validation_average_profit_after_fee_and_slippage",
    "validation_total_profit_after_fee_and_slippage",
]


def metrics_record(model_name, split, objective, threshold, train_rows, validation_rows, test_rows,
                   metrics, args, validation_metrics=None):
    record = {
        "model": model_name,
        "split": split,
        "threshold_objective": objective,
        "selected_threshold": threshold,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "cooldown_minutes": args.cooldown_minutes,
        "max_trades_per_symbol_month": args.max_trades_per_symbol_month,
        "min_validation_trades": args.min_validation_trades,
        "max_validation_trades": args.max_validation_trades,
        "min_validation_precision": args.min_validation_precision,
        "min_selected_threshold": args.min_selected_threshold,
        "profit_safety": args.profit_safety,
        "adaptive_thresholds": 0 if args.disable_adaptive_thresholds else 1,
    }
    if validation_metrics:
        record.update({
            "validation_predicted_trades": validation_metrics["predicted_trades"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_average_profit_after_fee_and_slippage": validation_metrics["average_profit_after_fee_and_slippage"],
            "validation_total_profit_after_fee_and_slippage": validation_metrics["total_profit_after_fee_and_slippage"],
        })
    record.update(metrics)
    return record


def write_metrics(path, records):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_feature_importance(path, model, feature_names, model_name):
    rows = model.feature_importance(feature_names)
    rows.sort(key=lambda item: item[1], reverse=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "feature", "importance", "importance_fraction"])
        for feature, importance, fraction in rows:
            writer.writerow([model_name, feature, "{:.12g}".format(importance), "{:.12g}".format(fraction)])


def run_fixed_split(rows, feature_names, args, kind, model_name):
    train_end = args.train_months
    validation_end = train_end + args.validation_months
    test_end = validation_end + args.test_months
    train_rows = select_month_range(rows, 0, train_end)
    validation_rows = select_month_range(rows, train_end, validation_end)
    test_rows = select_month_range(rows, validation_end, test_end)
    if not train_rows or not validation_rows or not test_rows:
        raise RuntimeError(
            "not enough rows for fixed split: train={}, validation={}, test={}".format(
                len(train_rows), len(validation_rows), len(test_rows)
            )
        )

    selected = fit_select_model(train_rows, validation_rows, feature_names, args, kind)
    probabilities = selected["model"].predict_proba(matrix(test_rows))
    test_metrics = evaluate(
        test_rows,
        probabilities,
        selected["threshold"],
        args.fee,
        args.slippage,
        args.cooldown_minutes,
        args.max_trades_per_symbol_month,
    )
    write_predictions(
        args.predictions_out,
        test_rows,
        probabilities,
        selected["threshold"],
        model_name,
        args.cooldown_minutes,
        args.max_trades_per_symbol_month,
    )
    write_feature_importance(args.feature_importance_out, selected["model"], feature_names, model_name)
    return metrics_record(
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
    ), selected


def aggregate_fold_records(records, model_name, objective):
    if not records:
        return None
    aggregate = {
        "model": model_name,
        "split": "walkforward_average",
        "threshold_objective": objective,
        "selected_threshold": sum(float(row["selected_threshold"]) for row in records) / len(records),
        "train_rows": sum(int(row["train_rows"]) for row in records),
        "validation_rows": sum(int(row["validation_rows"]) for row in records),
        "test_rows": sum(int(row["test_rows"]) for row in records),
        "profit_safety": records[0].get("profit_safety", ""),
        "adaptive_thresholds": records[0].get("adaptive_thresholds", 0),
    }
    for column in METRIC_COLUMNS:
        if column in aggregate or column in ("model", "split", "threshold_objective", "profit_safety"):
            continue
        values = [row.get(column) for row in records if isinstance(row.get(column), (int, float))]
        aggregate[column] = sum(values) / len(values) if values else 0.0
    return aggregate


def run_walk_forward(rows, feature_names, args, kind, model_name):
    max_month = max(row.month_index for row in rows) if rows else -1
    fold_records = []
    write_predictions(
        args.walk_predictions_out,
        [],
        [],
        0.0,
        model_name,
        args.cooldown_minutes,
        args.max_trades_per_symbol_month,
    )
    for fold_start in range(0, max_month - args.walk_train_months + 1):
        test_month = fold_start + args.walk_train_months
        if test_month > max_month:
            break
        inner_train_end = fold_start + args.walk_train_months - 1
        validation_start = inner_train_end
        validation_end = validation_start + 1
        train_rows = select_month_range(rows, fold_start, inner_train_end)
        validation_rows = select_month_range(rows, validation_start, validation_end)
        final_train_rows = select_month_range(rows, fold_start, test_month)
        test_rows = select_month_range(rows, test_month, test_month + 1)
        if not train_rows or not validation_rows or not final_train_rows or not test_rows:
            continue

        selected = fit_select_model(train_rows, validation_rows, feature_names, args, kind)
        positive_weight = class_weight_ratio(labels(final_train_rows), args.positive_weight_cap)
        selected_model = selected.pop("model", None)
        del selected_model
        gc.collect()
        final_model = make_model(kind, selected["params"], positive_weight)
        final_model.fit(matrix(final_train_rows), labels(final_train_rows), feature_names)
        probabilities = final_model.predict_proba(matrix(test_rows))
        metrics = evaluate(
            test_rows,
            probabilities,
            selected["threshold"],
            args.fee,
            args.slippage,
            args.cooldown_minutes,
            args.max_trades_per_symbol_month,
        )
        write_predictions(
            args.walk_predictions_out,
            test_rows,
            probabilities,
            selected["threshold"],
            model_name,
            args.cooldown_minutes,
            args.max_trades_per_symbol_month,
            append=True,
        )
        record = metrics_record(
            model_name,
            "walkforward_fold_{}_test_month_{}".format(fold_start + 1, test_month + 1),
            args.threshold_objective,
            selected["threshold"],
            final_train_rows,
            validation_rows,
            test_rows,
            metrics,
            args,
            selected["validation_metrics"],
        )
        fold_records.append(record)
        del selected
        del final_model
        del probabilities
        del train_rows
        del validation_rows
        del final_train_rows
        del test_rows
        gc.collect()

    if fold_records:
        aggregate = aggregate_fold_records(fold_records, model_name, args.threshold_objective)
        if aggregate:
            fold_records.append(aggregate)
    write_metrics(args.walkforward_metrics_out, fold_records)
    return fold_records


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


def print_comparison(gbdt_record, walk_records, args):
    logistic = read_logistic_metric(args.logistic_metrics_in)
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
        "GBDT: model={} threshold={:.4g} precision={:.4f} recall={:.4f} net_fee_slippage={:.6f} safety={}".format(
            gbdt_record["model"],
            float(gbdt_record["selected_threshold"]),
            float(gbdt_record["precision"]),
            float(gbdt_record["recall"]),
            float(gbdt_record["total_profit_after_fee_and_slippage"]),
            args.profit_safety,
        )
    )
    print(
        "GBDT validation: trades={} precision={:.4f} recall={:.4f} net_fee_slippage={:.6f}".format(
            int(float(gbdt_record.get("validation_predicted_trades", 0))),
            float(gbdt_record.get("validation_precision", 0.0)),
            float(gbdt_record.get("validation_recall", 0.0)),
            float(gbdt_record.get("validation_total_profit_after_fee_and_slippage", 0.0)),
        )
    )
    aggregate = walk_records[-1] if walk_records and walk_records[-1].get("split") == "walkforward_average" else None
    if aggregate:
        print(
            "Walk-forward average: folds={} precision={:.4f} recall={:.4f} net_fee_slippage={:.6f}".format(
                len(walk_records) - 1,
                float(aggregate["precision"]),
                float(aggregate["recall"]),
                float(aggregate["total_profit_after_fee_and_slippage"]),
            )
        )

    logistic_profit = metric_float(logistic, "total_profit_after_fee_and_slippage") if logistic else None
    gbdt_profit = float(gbdt_record["total_profit_after_fee_and_slippage"])
    if logistic_profit is not None:
        better = "GBDT" if gbdt_profit > logistic_profit else "Logistic"
        print("{} performed better on profit after fee+slippage for the fixed test split.".format(better))


def build_parser():
    parser = argparse.ArgumentParser(description="Train/evaluate boosted trees on generated kline samples.")
    parser.add_argument("--input", default="kline_growth_training.csv")
    parser.add_argument("--model", choices=["auto", "lightgbm", "internal"], default="auto")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--validation-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--threshold-grid", default="0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99")
    parser.add_argument("--threshold-objective", choices=["profit", "avg_profit", "precision", "recall", "f1"], default="avg_profit")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--cooldown-minutes", type=int, default=10)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    parser.add_argument("--max-validation-trades", type=int, default=250)
    parser.add_argument("--min-validation-precision", type=float, default=0.25)
    parser.add_argument("--min-selected-threshold", type=float, default=0.90)
    parser.add_argument("--max-trades-per-symbol-month", type=int, default=50)
    parser.add_argument("--profit-safety", choices=["strict", "explore"], default="explore")
    parser.add_argument("--disable-adaptive-thresholds", action="store_true")
    parser.add_argument("--positive-weight-cap", type=float, default=50.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--internal-estimators", type=int, default=24)
    parser.add_argument("--internal-learning-rate", type=float, default=0.08)
    parser.add_argument("--internal-bins", type=int, default=12)
    parser.add_argument("--internal-l2", type=float, default=2.0)
    parser.add_argument("--feature-storage", choices=["float32", "float64", "list"], default="float32")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--walk-train-months", type=int, default=6)
    parser.add_argument("--predictions-out", default="kline_growth_predictions_gbdt.csv")
    parser.add_argument("--metrics-out", default="kline_growth_metrics_gbdt.csv")
    parser.add_argument("--walkforward-metrics-out", default="kline_growth_walkforward_metrics.csv")
    parser.add_argument("--walk-predictions-out", default="kline_growth_predictions_gbdt_walkforward.csv")
    parser.add_argument("--feature-importance-out", default="kline_growth_feature_importance.csv")
    parser.add_argument("--logistic-metrics-in", default="kline_growth_metrics_logistic.csv")
    return parser


def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cooldown_minutes < 0:
        raise ValueError("--cooldown-minutes cannot be negative")
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
    if args.max_trades_per_symbol_month < 0:
        raise ValueError("--max-trades-per-symbol-month cannot be negative")
    args.thresholds = parse_threshold_grid(args.threshold_grid)
    kind = choose_model_kind(args.model)
    rows, feature_names, has_returns = load_rows(args.input, args.feature_storage)
    if not has_returns and args.threshold_objective in ("profit", "avg_profit"):
        print("forward return columns are missing; falling back from profit objective to f1", file=sys.stderr)
        args.threshold_objective = "f1"

    model_name = "gbdt_{}".format(kind)
    print("Loaded {} rows with {} features from {} using {} feature storage".format(
        len(rows), len(feature_names), args.input, args.feature_storage))
    print("Using {} model path".format(model_name))

    fixed_record, fixed_selected = run_fixed_split(rows, feature_names, args, kind, model_name)
    write_metrics(args.metrics_out, [fixed_record])
    del fixed_selected
    gc.collect()
    walk_records = []
    if args.walk_forward:
        walk_records = run_walk_forward(rows, feature_names, args, kind, model_name)
    print_comparison(fixed_record, walk_records, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as error:
        print("gbdt_pipeline failed: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
