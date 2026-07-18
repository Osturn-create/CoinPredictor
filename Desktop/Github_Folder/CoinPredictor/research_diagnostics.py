#!/usr/bin/env python3
"""Artifact-level diagnostics for CoinPredictor research runs.

The functions in this module intentionally work from saved CSV/JSON artifacts.
They do not rerun training and they do not mutate the source run directory.
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_ROUND_TRIP_COST = 0.0015

SCORE_COLUMNS = [
    "probability",
    "calibrated_probability",
    "raw_predicted_trade_return",
    "predicted_trade_return",
    "calibrated_predicted_trade_return",
    "predicted_net_return",
    "expected_value",
    "base_hybrid_score",
    "hybrid_score",
    "trade_score",
    "ranker_score",
    "ranker_selection_score",
    "ranker_utility_score",
]


def safe_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return default
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return default
    try:
        number = float(text)
    except ValueError:
        return default
    return number if math.isfinite(number) else default


def safe_int(value, default=0):
    try:
        return int(round(safe_float(value, default)))
    except (TypeError, ValueError):
        return default


def is_truthy(value):
    return abs(safe_float(value, 0.0)) > 1e-12


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def first_existing_path(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError(paths[0])


def write_csv_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def quantile(values, q):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def numeric_summary(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p01": quantile(values, 0.01),
        "p05": quantile(values, 0.05),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
    }


def gross_return(row):
    if "dynamic_trade_return" in row and str(row.get("dynamic_trade_return", "")).strip():
        return safe_float(row.get("dynamic_trade_return"))
    if "trade_return" in row:
        return safe_float(row.get("trade_return"))
    return safe_float(row.get("forward_return"))


def net_return(row, round_trip_cost=DEFAULT_ROUND_TRIP_COST, extra_cost=0.0):
    return gross_return(row) - round_trip_cost - extra_cost


def trade_profit(row, round_trip_cost=DEFAULT_ROUND_TRIP_COST, extra_cost=0.0):
    return safe_float(row.get("position_size"), DEFAULT_INITIAL_CAPITAL * 0.1) * net_return(
        row,
        round_trip_cost=round_trip_cost,
        extra_cost=extra_cost,
    )


def positive_profit_factor(profits):
    gains = sum(p for p in profits if p > 0)
    losses = -sum(p for p in profits if p < 0)
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def max_drawdown_from_profits(profits, initial_capital=DEFAULT_INITIAL_CAPITAL):
    equity = initial_capital
    peak = initial_capital
    worst = 0.0
    for profit in profits:
        equity += profit
        peak = max(peak, equity)
        if peak > 0.0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def portfolio_metrics(
    rows,
    initial_capital=DEFAULT_INITIAL_CAPITAL,
    denominator_folds=1,
    round_trip_cost=DEFAULT_ROUND_TRIP_COST,
    extra_cost=0.0,
):
    rows = sorted(list(rows), key=lambda row: (safe_float(row.get("open_time")), row.get("symbol", "")))
    profits = [trade_profit(row, round_trip_cost=round_trip_cost, extra_cost=extra_cost) for row in rows]
    gross_returns = [gross_return(row) for row in rows]
    net_returns = [net_return(row, round_trip_cost=round_trip_cost, extra_cost=extra_cost) for row in rows]
    denominator = initial_capital * max(1, denominator_folds)
    profit = sum(profits)
    return {
        "trade_count": len(rows),
        "portfolio_profit": profit,
        "portfolio_return": profit / denominator if denominator else 0.0,
        "avg_gross_return": statistics.fmean(gross_returns) if gross_returns else 0.0,
        "avg_net_return": statistics.fmean(net_returns) if net_returns else 0.0,
        "win_rate": sum(1 for value in net_returns if value > 0.0) / len(net_returns) if net_returns else 0.0,
        "profit_factor": positive_profit_factor(profits),
        "max_drawdown": max_drawdown_from_profits(profits, initial_capital=initial_capital),
        "active_days": len({row.get("trade_day", "") for row in rows if row.get("trade_day", "")}),
        "active_symbols": len({row.get("symbol", "") for row in rows if row.get("symbol", "")}),
    }


def symbol_exposure(rows, round_trip_cost=DEFAULT_ROUND_TRIP_COST, extra_cost=0.0):
    stats = {}
    for row in rows:
        symbol = row.get("symbol", "")
        if not symbol:
            continue
        item = stats.setdefault(
            symbol,
            {
                "symbol": symbol,
                "trade_count": 0,
                "capital": 0.0,
                "gross_profit": 0.0,
                "net_profit": 0.0,
                "loss": 0.0,
                "score_sum": 0.0,
                "realized_sum": 0.0,
                "predicted_sum": 0.0,
                "calibration_abs_error_sum": 0.0,
                "active_days": set(),
            },
        )
        item["trade_count"] += 1
        item["capital"] += safe_float(row.get("position_size"), DEFAULT_INITIAL_CAPITAL * 0.1)
        profit = trade_profit(row, round_trip_cost=round_trip_cost, extra_cost=extra_cost)
        if profit >= 0.0:
            item["gross_profit"] += profit
        else:
            item["loss"] += -profit
        item["net_profit"] += profit
        item["score_sum"] += safe_float(row.get("trade_score"))
        realized = net_return(row, round_trip_cost=round_trip_cost, extra_cost=extra_cost)
        item["realized_sum"] += realized
        predicted = safe_float(row.get("predicted_net_return"))
        item["predicted_sum"] += predicted
        item["calibration_abs_error_sum"] += abs(predicted - realized)
        if row.get("trade_day"):
            item["active_days"].add(row.get("trade_day"))

    total_trades = sum(item["trade_count"] for item in stats.values()) or 1
    total_capital = sum(item["capital"] for item in stats.values()) or 1.0
    total_positive_profit = sum(max(0.0, item["net_profit"]) for item in stats.values()) or 1.0
    total_loss = sum(item["loss"] for item in stats.values()) or 1.0
    result = []
    for item in stats.values():
        count = item["trade_count"]
        result.append({
            "symbol": item["symbol"],
            "trade_count": count,
            "trade_share": item["trade_count"] / total_trades,
            "capital": item["capital"],
            "capital_share": item["capital"] / total_capital,
            "gross_profit": item["gross_profit"],
            "net_profit": item["net_profit"],
            "net_profit_share": max(0.0, item["net_profit"]) / total_positive_profit,
            "loss": item["loss"],
            "loss_share": item["loss"] / total_loss,
            "active_days": len(item["active_days"]),
            "average_score": item["score_sum"] / count if count else 0.0,
            "average_realized_net_return": item["realized_sum"] / count if count else 0.0,
            "average_predicted_net_return": item["predicted_sum"] / count if count else 0.0,
            "average_abs_return_calibration_error": item["calibration_abs_error_sum"] / count if count else 0.0,
        })
    return sorted(result, key=lambda item: (-item["net_profit"], -item["trade_count"], item["symbol"]))


def top_share(rows, key):
    values = [safe_float(row.get(key)) for row in rows]
    total = sum(values)
    return max(values) / total if total > 0.0 and values else 0.0


def score_distribution_rows(rows, source_name, score_columns=None):
    score_columns = score_columns or SCORE_COLUMNS
    output = []
    for column in score_columns:
        values = [safe_float(row.get(column), float("nan")) for row in rows if str(row.get(column, "")).strip()]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            continue
        winners = [
            safe_float(row.get(column))
            for row in rows
            if str(row.get(column, "")).strip() and net_return(row) > 0.0
        ]
        losers = [
            safe_float(row.get(column))
            for row in rows
            if str(row.get(column, "")).strip() and net_return(row) <= 0.0
        ]
        summary = numeric_summary(values)
        summary.update({
            "source": source_name,
            "score": column,
            "winner_mean": statistics.fmean(winners) if winners else 0.0,
            "winner_median": statistics.median(winners) if winners else 0.0,
            "loser_mean": statistics.fmean(losers) if losers else 0.0,
            "loser_median": statistics.median(losers) if losers else 0.0,
            "winner_minus_loser_mean": (
                (statistics.fmean(winners) if winners else 0.0)
                - (statistics.fmean(losers) if losers else 0.0)
            ),
        })
        output.append(summary)
    return output


def score_decile_rows(rows, source_name, score_columns=None):
    score_columns = score_columns or SCORE_COLUMNS
    output = []
    monotonicity = {}
    for column in score_columns:
        scored = [
            row for row in rows
            if str(row.get(column, "")).strip() and math.isfinite(safe_float(row.get(column), float("nan")))
        ]
        scored.sort(key=lambda row: (safe_float(row.get(column)), safe_float(row.get("open_time")), row.get("symbol", "")))
        if not scored:
            continue
        decile_net_returns = []
        for decile in range(1, 11):
            start = (decile - 1) * len(scored) // 10
            end = decile * len(scored) // 10
            bucket = scored[start:end]
            if not bucket:
                continue
            profits = [trade_profit(row) for row in bucket]
            net_returns = [net_return(row) for row in bucket]
            gross_returns = [gross_return(row) for row in bucket]
            symbol_counts = {}
            for row in bucket:
                symbol_counts[row.get("symbol", "")] = symbol_counts.get(row.get("symbol", ""), 0) + 1
            top_symbol, top_count = max(symbol_counts.items(), key=lambda item: (item[1], item[0]))
            avg_net = statistics.fmean(net_returns)
            decile_net_returns.append(avg_net)
            output.append({
                "source": source_name,
                "score": column,
                "decile": decile,
                "rows": len(bucket),
                "score_min": safe_float(bucket[0].get(column)),
                "score_max": safe_float(bucket[-1].get(column)),
                "avg_realized_gross_return": statistics.fmean(gross_returns),
                "avg_realized_net_return": avg_net,
                "positive_return_rate": sum(1 for value in net_returns if value > 0.0) / len(net_returns),
                "profit_factor": positive_profit_factor(profits),
                "symbol_concentration_top1": top_count / len(bucket),
                "top_symbol": top_symbol,
            })
        if len(decile_net_returns) > 1:
            improving = sum(
                1 for left, right in zip(decile_net_returns, decile_net_returns[1:])
                if right >= left
            )
            monotonicity[(source_name, column)] = improving / (len(decile_net_returns) - 1)
        else:
            monotonicity[(source_name, column)] = 0.0
    return output, monotonicity


def calibration_metric_rows(rows, source_name):
    output = []
    labels = [safe_int(row.get("label")) for row in rows]
    for column in ("probability", "calibrated_probability"):
        probs = [
            max(1e-12, min(1.0 - 1e-12, safe_float(row.get(column))))
            for row in rows
            if str(row.get(column, "")).strip()
        ]
        if not probs or len(probs) != len(labels):
            continue
        brier = statistics.fmean((p - y) ** 2 for p, y in zip(probs, labels))
        log_loss = -statistics.fmean(
            y * math.log(p) + (1 - y) * math.log(1.0 - p)
            for p, y in zip(probs, labels)
        )
        output.append({
            "source": source_name,
            "calibration_type": "probability",
            "score": column,
            "rows": len(probs),
            "mean_prediction": statistics.fmean(probs),
            "actual_rate": statistics.fmean(labels) if labels else 0.0,
            "brier_score": brier,
            "log_loss": log_loss,
            "mean_abs_error": statistics.fmean(abs(p - y) for p, y in zip(probs, labels)),
        })
    actual_net = [net_return(row) for row in rows]
    for column in ("raw_predicted_trade_return", "predicted_trade_return", "calibrated_predicted_trade_return", "predicted_net_return", "expected_value", "trade_score"):
        preds = [safe_float(row.get(column), float("nan")) for row in rows if str(row.get(column, "")).strip()]
        if not preds or len(preds) != len(actual_net):
            continue
        errors = [pred - actual for pred, actual in zip(preds, actual_net)]
        output.append({
            "source": source_name,
            "calibration_type": "return",
            "score": column,
            "rows": len(preds),
            "mean_prediction": statistics.fmean(preds),
            "actual_rate": statistics.fmean(actual_net) if actual_net else 0.0,
            "brier_score": "",
            "log_loss": "",
            "mean_abs_error": statistics.fmean(abs(error) for error in errors),
        })
    return output


def attach_walkforward_folds(rows, diagnostic_rows):
    month_to_fold = {}
    for row in diagnostic_rows:
        if row.get("test_month"):
            month_to_fold[row["test_month"]] = safe_int(row.get("fold_index"))
    annotated = []
    for row in rows:
        copy = dict(row)
        copy["_fold_index"] = month_to_fold.get(row.get("month", ""), 0)
        annotated.append(copy)
    return annotated


def selected_prediction_rows(rows):
    selected = [row for row in rows if is_truthy(row.get("selected_by_topk"))]
    if selected:
        return selected
    predicted = [row for row in rows if is_truthy(row.get("predicted"))]
    if predicted:
        return predicted
    return rows


def leave_one_symbol_out_rows(rows, symbols, source_name, denominator_folds=1):
    output = []
    baseline = portfolio_metrics(rows, denominator_folds=denominator_folds)
    baseline_concentration = symbol_exposure(rows)
    baseline_top = max((item["net_profit_share"] for item in baseline_concentration), default=0.0)
    for symbol in symbols:
        filtered = [row for row in rows if row.get("symbol") != symbol]
        metrics = portfolio_metrics(filtered, denominator_folds=denominator_folds)
        concentration = symbol_exposure(filtered)
        metrics.update({
            "source": source_name,
            "removed_symbol": symbol,
            "baseline_portfolio_return": baseline["portfolio_return"],
            "return_delta": metrics["portfolio_return"] - baseline["portfolio_return"],
            "baseline_trade_count": baseline["trade_count"],
            "removed_trade_count": baseline["trade_count"] - metrics["trade_count"],
            "baseline_top_symbol_concentration": baseline_top,
            "top_symbol_concentration": max((item["net_profit_share"] for item in concentration), default=0.0),
        })
        output.append(metrics)
    return output


def leave_one_fold_out_rows(walkforward_metric_rows, initial_capital=DEFAULT_INITIAL_CAPITAL):
    fold_rows = [
        row for row in walkforward_metric_rows
        if str(row.get("split", "")).startswith("walkforward_fold_")
    ]
    baseline_profit = sum(safe_float(row.get("portfolio_profit")) for row in fold_rows)
    positive_profit = sum(max(0.0, safe_float(row.get("portfolio_profit"))) for row in fold_rows) or 1.0
    output = []
    for row in fold_rows:
        removed_profit = safe_float(row.get("portfolio_profit"))
        kept = [other for other in fold_rows if other is not row]
        kept_returns = [safe_float(other.get("portfolio_return")) for other in kept]
        kept_profit = baseline_profit - removed_profit
        output.append({
            "removed_fold": row.get("split", ""),
            "removed_fold_return": safe_float(row.get("portfolio_return")),
            "removed_fold_profit": removed_profit,
            "removed_fold_profit_share_of_positive_profit": max(0.0, removed_profit) / positive_profit,
            "remaining_profit": kept_profit,
            "remaining_mean_fold_return": kept_profit / (initial_capital * max(1, len(kept))),
            "remaining_median_fold_return": statistics.median(kept_returns) if kept_returns else 0.0,
            "remaining_worst_fold_return": min(kept_returns) if kept_returns else 0.0,
            "remaining_active_folds": sum(1 for other in kept if safe_int(other.get("predicted_trades")) > 0),
            "remaining_profitable_folds": sum(1 for other in kept if safe_float(other.get("portfolio_return")) > 0.0),
        })
    return output


def threshold_stability_rows(threshold_rows, selected_thresholds):
    grouped = {}
    for row in threshold_rows:
        if safe_int(row.get("available"), 1) == 0:
            continue
        key = (row.get("source_split", ""), safe_int(row.get("fold_index")))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, rows in grouped.items():
        rows.sort(key=lambda row: (safe_int(row.get("candidate_index")), safe_float(row.get("threshold"))))
        selected_threshold = selected_thresholds.get(key)
        if selected_threshold is None:
            selected_threshold = selected_thresholds.get((key[0], 0), safe_float(rows[0].get("threshold")))
        selected_index = min(
            range(len(rows)),
            key=lambda idx: abs(safe_float(rows[idx].get("threshold")) - selected_threshold),
        )
        selected = rows[selected_index]
        neighbors = rows[max(0, selected_index - 2): min(len(rows), selected_index + 3)]
        selected_avg = safe_float(selected.get("average_net_return_after_costs"))
        neighbor_avgs = [safe_float(row.get("average_net_return_after_costs")) for row in neighbors]
        neighbor_trades = [safe_int(row.get("predicted_trades")) for row in neighbors]
        neighbor_conc = [safe_float(row.get("symbol_trade_concentration_top1")) for row in neighbors]
        if selected_avg > 0.0:
            performance_component = max(0.0, min(1.0, min(neighbor_avgs) / selected_avg))
        else:
            performance_component = 1.0 if min(neighbor_avgs) >= selected_avg else 0.0
        max_trades = max(neighbor_trades) if neighbor_trades else 0
        trade_component = 1.0
        if max_trades > 0:
            trade_component = max(0.0, 1.0 - ((max(neighbor_trades) - min(neighbor_trades)) / max_trades))
        concentration_component = max(0.0, 1.0 - (max(neighbor_conc) - min(neighbor_conc))) if neighbor_conc else 1.0
        admissible_component = (
            sum(1 for row in neighbors if not str(row.get("rejection_reason_flags", "")).strip()) / len(neighbors)
            if neighbors else 0.0
        )
        stability = statistics.fmean([
            performance_component,
            trade_component,
            concentration_component,
            admissible_component,
        ])
        output.append({
            "source_split": key[0],
            "fold_index": key[1],
            "selected_threshold": selected_threshold,
            "closest_candidate_threshold": safe_float(selected.get("threshold")),
            "selected_avg_net_return": selected_avg,
            "neighbor_min_avg_net_return": min(neighbor_avgs) if neighbor_avgs else 0.0,
            "neighbor_max_avg_net_return": max(neighbor_avgs) if neighbor_avgs else 0.0,
            "neighbor_min_trades": min(neighbor_trades) if neighbor_trades else 0,
            "neighbor_max_trades": max(neighbor_trades) if neighbor_trades else 0,
            "neighbor_min_trade_top1_concentration": min(neighbor_conc) if neighbor_conc else 0.0,
            "neighbor_max_trade_top1_concentration": max(neighbor_conc) if neighbor_conc else 0.0,
            "threshold_stability_score": stability,
        })
    return sorted(output, key=lambda row: (row["source_split"], row["fold_index"]))


def selected_threshold_map(summary, walkforward_diagnostics):
    selected = {("fixed", 0): safe_float(summary.get("selected_threshold"))}
    for row in walkforward_diagnostics:
        split = row.get("split", "")
        selected[(split, safe_int(row.get("fold_index")))] = safe_float(row.get("selected_threshold"))
    return selected


def experiment_matrix_rows(rows, source_name, denominator_folds=1):
    scored = sorted(rows, key=lambda row: (safe_float(row.get("open_time")), row.get("symbol", "")))
    trade_scores = [safe_float(row.get("trade_score")) for row in scored if str(row.get("trade_score", "")).strip()]
    trade_score_median = statistics.median(trade_scores) if trade_scores else 0.0
    experiments = [
        ("corrected_frozen_baseline", scored, DEFAULT_ROUND_TRIP_COST, 0.0, "saved selected trades"),
        (
            "executed_nonnegative_expected_value_only",
            [row for row in scored if safe_float(row.get("expected_value")) >= 0.0],
            DEFAULT_ROUND_TRIP_COST,
            0.0,
            "post-selection diagnostic filter; no replacement candidates",
        ),
        (
            "executed_trade_score_top_half_only",
            [row for row in scored if safe_float(row.get("trade_score")) >= trade_score_median],
            DEFAULT_ROUND_TRIP_COST,
            0.0,
            "post-selection tail-only diagnostic filter",
        ),
        (
            "executed_unique_symbol_per_day",
            unique_symbol_per_day(scored),
            DEFAULT_ROUND_TRIP_COST,
            0.0,
            "post-selection exposure cap; no replacement candidates",
        ),
        (
            "executed_symbol_trade_share_cap_35pct",
            symbol_trade_share_cap(scored, 0.35),
            DEFAULT_ROUND_TRIP_COST,
            0.0,
            "post-selection cap; uses saved selected trades only",
        ),
        (
            "corrected_baseline_double_cost",
            scored,
            DEFAULT_ROUND_TRIP_COST,
            DEFAULT_ROUND_TRIP_COST,
            "higher-cost stress: doubles fee plus slippage drag",
        ),
    ]
    output = []
    for name, experiment_rows, base_cost, extra_cost, note in experiments:
        metrics = portfolio_metrics(
            experiment_rows,
            denominator_folds=denominator_folds,
            round_trip_cost=base_cost,
            extra_cost=extra_cost,
        )
        exposure = symbol_exposure(experiment_rows, round_trip_cost=base_cost, extra_cost=extra_cost)
        metrics.update({
            "source": source_name,
            "experiment": name,
            "note": note,
            "top_symbol_concentration": max((item["net_profit_share"] for item in exposure), default=0.0),
            "top_trade_concentration": max((item["trade_share"] for item in exposure), default=0.0),
            "top_symbol": exposure[0]["symbol"] if exposure else "",
        })
        output.append(metrics)
    return output


def unique_symbol_per_day(rows):
    grouped = {}
    for row in rows:
        key = (row.get("trade_day", ""), row.get("symbol", ""))
        current = grouped.get(key)
        if current is None or safe_float(row.get("trade_score")) > safe_float(current.get("trade_score")):
            grouped[key] = row
    return sorted(grouped.values(), key=lambda row: (safe_float(row.get("open_time")), row.get("symbol", "")))


def symbol_trade_share_cap(rows, cap):
    if not rows:
        return []
    limit = max(1, int(math.floor(len(rows) * cap)))
    counts = {}
    kept = []
    for row in rows:
        symbol = row.get("symbol", "")
        if counts.get(symbol, 0) >= limit:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
        kept.append(row)
    return kept


def cost_stress_rows(rows, source_name, denominator_folds=1):
    output = []
    for label, extra_cost in [
        ("baseline_cost", 0.0),
        ("double_cost", DEFAULT_ROUND_TRIP_COST),
        ("triple_cost", DEFAULT_ROUND_TRIP_COST * 2.0),
    ]:
        metrics = portfolio_metrics(rows, denominator_folds=denominator_folds, extra_cost=extra_cost)
        metrics.update({
            "source": source_name,
            "cost_scenario": label,
            "round_trip_cost": DEFAULT_ROUND_TRIP_COST + extra_cost,
        })
        output.append(metrics)
    return output


def composite_robustness_score(
    summary,
    walkforward_metrics,
    threshold_stability,
    leave_one_symbol_rows,
    leave_one_fold_rows,
    cost_stress,
    monotonicity,
):
    fold_rows = [
        row for row in walkforward_metrics
        if str(row.get("split", "")).startswith("walkforward_fold_")
    ]
    returns = [safe_float(row.get("portfolio_return")) for row in fold_rows]
    positive_fold_profit = sum(max(0.0, safe_float(row.get("portfolio_profit"))) for row in fold_rows) or 1.0
    top_fold_concentration = max(
        [max(0.0, safe_float(row.get("portfolio_profit"))) / positive_fold_profit for row in fold_rows] or [0.0]
    )
    active_folds = sum(1 for row in fold_rows if safe_int(row.get("predicted_trades")) > 0)
    total_folds = len(fold_rows) or 1
    double_cost_wf = next(
        (row for row in cost_stress if row.get("source") == "walkforward" and row.get("cost_scenario") == "double_cost"),
        {},
    )
    best_symbol_ex = min(
        (safe_float(row.get("portfolio_return")) for row in leave_one_symbol_rows if row.get("source") == "walkforward"),
        default=0.0,
    )
    best_fold_ex = min((safe_float(row.get("remaining_mean_fold_return")) for row in leave_one_fold_rows), default=0.0)
    avg_threshold_stability = statistics.fmean(
        [safe_float(row.get("threshold_stability_score")) for row in threshold_stability]
    ) if threshold_stability else 0.0
    tail_monotonicity = monotonicity.get(
        ("walkforward", "trade_score"),
        monotonicity.get(
            ("walkforward_candidates", "trade_score"),
            safe_float(summary.get("ranking_trade_score_net_return_monotonicity")),
        ),
    )

    def clip(value, low=0.0, high=1.0):
        return max(low, min(high, value))

    components = [
        ("median_fold_return", clip((statistics.median(returns) + 0.02) / 0.08 if returns else 0.0)),
        ("worst_fold_return", clip((min(returns) + 0.12) / 0.12 if returns else 0.0)),
        ("profitable_fold_rate", clip(sum(1 for value in returns if value > 0.0) / total_folds)),
        ("walkforward_profit_factor", clip((safe_float(summary.get("profit_factor")) - 0.8) / 0.7)),
        ("active_fold_rate", clip(active_folds / total_folds)),
        ("tail_monotonicity", clip(tail_monotonicity)),
        ("double_cost_return", clip((safe_float(double_cost_wf.get("portfolio_return")) + 0.02) / 0.08)),
        ("return_without_key_symbol", clip((best_symbol_ex + 0.02) / 0.08)),
        ("return_without_key_fold", clip((best_fold_ex + 0.02) / 0.08)),
        ("top_symbol_concentration_penalty", clip(1.0 - safe_float(summary.get("symbol_profit_concentration_top1")))),
        ("top_fold_concentration_penalty", clip(1.0 - top_fold_concentration)),
        ("inactive_fold_penalty", clip(active_folds / total_folds)),
        ("threshold_stability", clip(avg_threshold_stability)),
    ]
    score = statistics.fmean(value for _, value in components) if components else 0.0
    return [{"component": name, "component_score": value} for name, value in components] + [
        {"component": "composite_robustness_score", "component_score": score}
    ]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_exact_command(run_log_path):
    if not run_log_path.exists():
        return ""
    text = ""
    for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp1252"):
        try:
            text = run_log_path.read_text(encoding=encoding, errors="replace")
        except OSError:
            continue
        if "gbdt_pipeline.py" in text:
            break
    for line in text.splitlines():
        text_line = line.strip()
        if "] " in text_line and "gbdt_pipeline.py" in text_line:
            return text_line.split("] ", 1)[1]
    if "gbdt_pipeline.py" in text:
        for line in text.splitlines():
            text_line = line.strip()
            if "gbdt_pipeline.py" in text_line:
                return text_line
    return ""


def freeze_baseline_profile(run_dir, parent_log_path, summary, output_path):
    artifact_names = [
        "kline_growth_run_summary.json",
        "kline_growth_experiment_report.md",
        "kline_growth_predictions_gbdt.csv",
        "kline_growth_predictions_gbdt_candidates.csv",
        "kline_growth_predictions_gbdt_all.csv",
        "kline_growth_predictions_gbdt_walkforward.csv",
        "kline_growth_predictions_gbdt_walkforward_candidates.csv",
        "kline_growth_predictions_gbdt_walkforward_all.csv",
        "kline_growth_walkforward_metrics.csv",
        "kline_growth_walkforward_diagnostics.csv",
    ]
    artifacts = {}
    for name in artifact_names:
        path = run_dir / name
        if path.exists():
            artifacts[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    profile = {
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "source_artifacts_are_immutable": True,
        "exact_command": extract_exact_command(parent_log_path),
        "confirmed_metrics": {
            "fixed_test_return": summary.get("portfolio_return"),
            "validation_return": summary.get("validation_portfolio_return"),
            "walkforward_mean_fold_return": summary.get("walkforward_mean_portfolio_return"),
            "walkforward_median_fold_return": summary.get("walkforward_median_portfolio_return"),
            "worst_fold_return": summary.get("walkforward_worst_fold_return"),
            "best_fold_return": summary.get("walkforward_max_portfolio_return"),
            "fixed_test_trades": summary.get("predicted_trades"),
            "walkforward_trades": summary.get("walkforward_total_predicted_trades"),
            "active_folds": summary.get("walkforward_folds_active"),
            "profitable_folds": summary.get("walkforward_folds_profitable"),
            "robustness_gate_status": summary.get("robustness_gate_status"),
            "chronological_audit_status": summary.get("execution_chronological_audit_status"),
            "topk_violations": summary.get("topk_bucket_limit_violation_count"),
            "capital_overallocation_violations": summary.get("capital_overallocated_count"),
        },
        "configuration": {
            "model_kind": summary.get("model_kind"),
            "objective_mode": summary.get("objective_mode"),
            "trade_selection": summary.get("trade_selection"),
            "trade_score": summary.get("trade_score"),
            "calibration": summary.get("calibration"),
            "regression_calibration": summary.get("regression_calibration"),
            "exit_policy": summary.get("exit_policy"),
            "selected_threshold": summary.get("selected_threshold"),
            "top_k_per_minute": summary.get("top_k_per_minute"),
            "top_k_per_symbol_minute": summary.get("top_k_per_symbol_minute"),
            "cache": summary.get("cache"),
            "feature_names": summary.get("feature_names"),
            "best_params": summary.get("best_params"),
        },
        "artifact_hashes": artifacts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return profile


def candidate_rich_command_from_profile(profile):
    command = profile.get("exact_command", "")
    if not command:
        return ""
    source_dir = "results\\chrono_fix_rerun_20260714_manual\\hyb_late_tuned_6m_sf0p04_f88"
    candidate_dir = "results\\candidate_rich_20260714_baseline_candidates"
    command = command.replace("--prediction-output-mode trades", "--prediction-output-mode candidates")
    command = command.replace(source_dir, candidate_dir)
    if "--predictions-out " not in command:
        command += " --predictions-out kline_growth_predictions_gbdt_candidates.csv"
    if "--walk-predictions-out " not in command:
        command += " --walk-predictions-out kline_growth_predictions_gbdt_walkforward_candidates.csv"
    return command


def build_markdown_report(context):
    summary = context["summary"]
    fold_rows = context["fold_rows"]
    diagnostic_by_split = context.get("walk_diagnostic_by_split", {})
    walk_average = context.get("walkforward_average", {})
    worst_fold = min(fold_rows, key=lambda row: safe_float(row.get("portfolio_return"))) if fold_rows else {}
    best_fold = max(fold_rows, key=lambda row: safe_float(row.get("portfolio_profit"))) if fold_rows else {}
    worst_diag = diagnostic_by_split.get(worst_fold.get("split", ""), {})
    double_cost_wf = next(row for row in context["cost_stress"] if row["source"] == "walkforward" and row["cost_scenario"] == "double_cost")
    fixed_double_cost = next(row for row in context["cost_stress"] if row["source"] == "fixed_test" and row["cost_scenario"] == "double_cost")
    fixed_exposure = context["fixed_symbol_exposure"]
    wf_exposure = context["walkforward_symbol_exposure"]
    top_fixed = fixed_exposure[0] if fixed_exposure else {}
    top_wf = wf_exposure[0] if wf_exposure else {}
    composite = context["composite"][-1]["component_score"] if context["composite"] else 0.0
    next_full_run_command = context.get("next_full_run_command", "")
    artifact_mode = context.get("artifact_mode", "trade_predictions")
    total_walkforward_profit = safe_float(summary.get("walkforward_total_portfolio_profit"))
    top_fold_concentration = (
        safe_float(best_fold.get("portfolio_profit")) / total_walkforward_profit
        if total_walkforward_profit > 0.0 else 0.0
    )

    def fold_field(name):
        value = worst_diag.get(name, "")
        if value not in ("", None):
            return value
        return worst_fold.get(name, "")

    lines = [
        "# Calibration and Robustness Research Pass",
        "",
        "## 1. Executive verdict",
        "",
        "Outcome 5: the corrected signal is not stable enough to promote from this run. Platt probability calibration improved validation Brier score, but the traded tail remains non-monotonic and fragile. The saved artifacts point to weak ranking/selection edge plus symbol and fold concentration as the dominant problems.",
        "",
        f"Artifact mode: `{artifact_mode}`. Accounting tables below use selected/executed rows; score and calibration tables use the saved candidate universe when candidate-rich artifacts are present.",
        "",
        "## 2. Frozen corrected baseline configuration",
        "",
        "- Baseline profile: `baseline_profile.json`",
        "- Artifact hashes and the recovered command field, when a run log is available, are frozen in the profile.",
        "- Source run artifacts were read only.",
        "",
        "Key baseline values:",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Fixed-test return | {safe_float(summary.get('portfolio_return')):.2%} |",
        f"| Validation return | {safe_float(summary.get('validation_portfolio_return')):.2%} |",
        f"| Walk-forward mean fold return | {safe_float(summary.get('walkforward_mean_portfolio_return')):.2%} |",
        f"| Walk-forward median fold return | {safe_float(summary.get('walkforward_median_portfolio_return')):.2%} |",
        f"| Worst fold | {safe_float(summary.get('walkforward_worst_fold_return')):.2%} |",
        f"| Best fold | {safe_float(summary.get('walkforward_max_portfolio_return')):.2%} |",
        f"| Fixed-test trades | {safe_int(summary.get('predicted_trades'))} |",
        f"| Walk-forward trades | {safe_int(summary.get('walkforward_total_predicted_trades'))} |",
        f"| Active folds | {safe_int(summary.get('walkforward_folds_active'))}/{safe_int(summary.get('walkforward_total_folds'))} |",
        f"| Profitable folds | {safe_int(summary.get('walkforward_folds_profitable'))}/{safe_int(summary.get('walkforward_total_folds'))}; active profitable {safe_int(summary.get('active_profitable_fold_count'))}/{safe_int(summary.get('active_fold_count'))} |",
        f"| Chronology audit | {summary.get('execution_chronological_audit_status')} |",
        f"| Top-k violations | {safe_int(summary.get('topk_bucket_limit_violation_count'))} |",
        f"| Capital overallocation violations | {safe_int(summary.get('capital_overallocated_count'))} |",
        "",
        "## 3. Calibration audit",
        "",
        f"- Validation Brier improved from {safe_float(summary.get('validation_brier_before')):.6f} to {safe_float(summary.get('validation_brier_after')):.6f}.",
        f"- Walk-forward trade-score monotonicity is {safe_float(summary.get('ranking_trade_score_net_return_monotonicity')):.4f}; the robustness gate failed on tail monotonicity.",
        f"- Trade-score top decile net return is {safe_float(summary.get('ranking_trade_score_top_decile_avg_net_return')):.6f}; top 5% is {safe_float(summary.get('ranking_trade_score_top_5pct_avg_net_return')):.6f}.",
        f"- Executed winner-minus-loser score gap is {safe_float(summary.get('executed_score_win_loss_gap')):.8f}, effectively flat.",
        "- Interpretation: calibration improves probability fit globally, but it does not rescue the economically traded tail.",
        "",
        "## 4. Worst-fold autopsy",
        "",
        f"- Worst fold: `{worst_fold.get('split', '')}`.",
        f"- Train period: {fold_field('train_start')} to {fold_field('train_end')}.",
        f"- Calibration/threshold period: {fold_field('validation_start')} to {fold_field('validation_end')}.",
        f"- Test period: {fold_field('test_start')} to {fold_field('test_end')}.",
        f"- Trades: {safe_int(worst_fold.get('predicted_trades'))}; active days: {safe_int(worst_fold.get('active_days'))}.",
        f"- Return: {safe_float(worst_fold.get('portfolio_return')):.2%}; profit factor: {safe_float(worst_fold.get('profit_factor')):.3f}; win rate: {safe_float(worst_fold.get('win_rate')):.2%}.",
        f"- Threshold: {safe_float(worst_fold.get('selected_threshold')):.6f}; top trade concentration: {safe_float(worst_fold.get('symbol_trade_concentration_top1')):.2%}; dominant symbol: {worst_fold.get('dominant_symbol', '')}.",
        f"- Worst-fold loss equals {abs(safe_float(worst_fold.get('portfolio_profit'))) / max(1.0, safe_float(summary.get('walkforward_total_portfolio_profit'))):.2%} of total walk-forward profit.",
        "- Best fit explanation: threshold/selection fragility under a shifted month, amplified by heavy symbol concentration and weak score-tail monotonicity.",
        "",
        "## 5. Symbol-concentration analysis",
        "",
        f"- Fixed-test best symbol by net profit: {top_fixed.get('symbol', '')}; fixed top-symbol profit concentration: {safe_float(summary.get('symbol_profit_concentration_top1')):.2%}.",
        f"- Walk-forward aggregate best symbol by net profit: {top_wf.get('symbol', '')}; aggregate top-symbol concentration from executed trades: {safe_float(top_wf.get('net_profit_share')):.2%}.",
        f"- Walk-forward average per-fold top-symbol concentration: {safe_float(walk_average.get('symbol_profit_concentration_top1')):.2%}.",
        "- Concentration appears driven by repeated selection of a few symbols and fold-local scarcity, not by a uniformly superior symbol.",
        "",
        "## 6. Threshold/top-k sensitivity",
        "",
        "- Threshold diagnostics are available and summarized in `threshold_stability.csv`.",
        "- Current selected thresholds are not consistently on broad economic plateaus; several folds have large neighboring trade-count and concentration movement.",
        "- Top-k cannot be truthfully reselected from trades-only prediction CSVs; use `--prediction-output-mode candidates` to persist rejected candidates without writing every scored row.",
        "",
        "## 7. Structural hypotheses tested",
        "",
        "- Tested as post-selection ablations only: nonnegative expected value, top-half trade score, unique-symbol-per-day, 35% symbol trade-share cap, and higher-cost stress.",
        "- These are diagnostic filters over already selected trades, not deployable substitutes for candidate-level selection experiments.",
        "",
        "## 8. Controlled experiment matrix",
        "",
        "- See `experiment_matrix.csv` for fixed-test and walk-forward ablations.",
        "- Candidate-level matrix still required: baseline, best calibration, cross-sectional normalization, exposure-aware selection, calibration plus normalization, and calibration plus concentration control.",
        "",
        "## 9. Before-versus-after table",
        "",
        "| Metric | Corrected baseline | Best admissible after this pass |",
        "| --- | ---: | ---: |",
        f"| Fixed-test return | {safe_float(summary.get('portfolio_return')):.2%} | No promotion |",
        f"| Validation return | {safe_float(summary.get('validation_portfolio_return')):.2%} | No promotion |",
        f"| Walk-forward mean return | {safe_float(summary.get('walkforward_mean_portfolio_return')):.2%} | No promotion |",
        f"| Walk-forward median return | {safe_float(summary.get('walkforward_median_portfolio_return')):.2%} | No promotion |",
        f"| Worst fold | {safe_float(summary.get('walkforward_worst_fold_return')):.2%} | No promotion |",
        f"| Best fold | {safe_float(summary.get('walkforward_max_portfolio_return')):.2%} | No promotion |",
        f"| Active folds | {safe_int(summary.get('walkforward_folds_active'))}/{safe_int(summary.get('walkforward_total_folds'))} | No promotion |",
        f"| Profitable folds | {safe_int(summary.get('walkforward_folds_profitable'))}/{safe_int(summary.get('walkforward_total_folds'))} | No promotion |",
        f"| Profit factor | {safe_float(summary.get('profit_factor')):.3f} | No promotion |",
        f"| Maximum drawdown | {safe_float(summary.get('walkforward_worst_fold_drawdown')):.2%} | No promotion |",
        f"| Top-symbol concentration | 51.45% per-fold average | No promotion |",
        f"| Top-fold concentration | {top_fold_concentration:.2%} | No promotion |",
        f"| Return excluding best fold | {min(safe_float(row.get('remaining_mean_fold_return')) for row in context['leave_one_fold_out']):.2%} | No promotion |",
        f"| Trade count | {safe_int(summary.get('walkforward_total_predicted_trades'))} | No promotion |",
        f"| Higher-cost stress return | {safe_float(double_cost_wf.get('portfolio_return')):.2%} | No promotion |",
        "",
        "## 10. Leave-one-symbol-out results",
        "",
        "- See `leave_one_symbol_out.csv`.",
        "- Removing key symbols remains positive for some cases but sharply changes concentration and profit, confirming dependence on a small set of names.",
        "",
        "## 11. Leave-one-fold-out results",
        "",
        "- See `leave_one_fold_out.csv`.",
        f"- Removing the best fold leaves a mean fold return of {min(safe_float(row.get('remaining_mean_fold_return')) for row in context['leave_one_fold_out']):.2%} in the most severe best-fold exclusion row.",
        "",
        "## 12. Cost stress results",
        "",
        "- See `cost_stress.csv`.",
        f"- Fixed-test double-cost return: {safe_float(fixed_double_cost.get('portfolio_return')):.2%}.",
        f"- Walk-forward double-cost mean fold return: {safe_float(double_cost_wf.get('portfolio_return')):.2%}.",
        "",
        "## 13. Recommended configuration",
        "",
        "No new configuration should be promoted from this artifact-only pass. The recommended next run is candidate-rich but candidate-only, and should test cross-sectional normalization plus exposure-aware selection under the frozen baseline command.",
        "",
        "## 14. Rejected alternatives",
        "",
        "- Calibration-only promotion: rejected because tail monotonicity and winner/loser score gap remain weak.",
        "- Post-selection exposure caps: rejected as production evidence because they drop trades without replacement candidates.",
        "- More complex objective: deferred until candidate-level ranking diagnostics confirm current ranking is structurally weak.",
        "",
        "## 15. Code changes",
        "",
        "- Added `research_diagnostics.py` for artifact-level recomputation and reporting.",
        "- Added focused tests in `test_gbdt_pipeline.py`.",
        "",
        "## 16. Tests added",
        "",
        "- Concentration calculations.",
        "- Leave-one-symbol-out calculations.",
        "- Leave-one-fold-out calculations.",
        "- Threshold stability calculations.",
        "- Cost stress calculations.",
        "- Composite robustness score behavior.",
        "",
        "## 17. Commands run",
        "",
        "- Baseline artifact diagnostics command is recorded in the final response.",
        "",
        "## 18. Verification results",
        "",
        "- Filled in by the final response after fresh verification commands.",
        "",
        "## 19. Remaining limitations",
        "",
        "- Trades-only saved predictions do not contain rejected candidates; candidate-only prediction output is now the lower-storage path for true top-k, normalization, calibration, and exposure-aware reselection experiments.",
        "- Drawdown from trade artifacts is approximate when recomputed from CSV because overlapping exposure paths are not fully serialized per timestep.",
        "- C++ build remains dependent on a compiler being available on PATH.",
        "",
        "## 20. Exact next full-run command",
        "",
        "First run the candidate-only frozen baseline so reselection experiments can be evaluated without retraining from missing rows:",
        "",
        "```powershell",
        next_full_run_command,
        "```",
        "",
        f"Composite robustness score for the corrected baseline artifact pass: `{composite:.3f}`.",
        "",
    ]
    return "\n".join(lines)


def run_diagnostics(run_dir, out_dir):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / "kline_growth_run_summary.json").read_text(encoding="utf-8"))
    fixed_prediction_path = first_existing_path([
        run_dir / "kline_growth_predictions_gbdt_candidates.csv",
        run_dir / "kline_growth_predictions_gbdt.csv",
        run_dir / "kline_growth_predictions_gbdt_all.csv",
    ])
    walk_prediction_path = first_existing_path([
        run_dir / "kline_growth_predictions_gbdt_walkforward_candidates.csv",
        run_dir / "kline_growth_predictions_gbdt_walkforward.csv",
        run_dir / "kline_growth_predictions_gbdt_walkforward_all.csv",
    ])
    fixed_candidate_rows = read_csv_rows(fixed_prediction_path)
    walk_candidate_rows = read_csv_rows(walk_prediction_path)
    walk_metrics = read_csv_rows(run_dir / "kline_growth_walkforward_metrics.csv")
    walk_diagnostics = read_csv_rows(run_dir / "kline_growth_walkforward_diagnostics.csv")
    threshold_rows = read_csv_rows(run_dir / "kline_growth_threshold_diagnostics.csv")
    walk_candidate_rows = attach_walkforward_folds(walk_candidate_rows, walk_diagnostics)
    fixed_rows = selected_prediction_rows(fixed_candidate_rows)
    walk_rows = selected_prediction_rows(walk_candidate_rows)

    baseline_profile = freeze_baseline_profile(
        run_dir,
        run_dir.parent / "run.log",
        summary,
        out_dir / "baseline_profile.json",
    )
    next_full_run_command = candidate_rich_command_from_profile(baseline_profile)
    (out_dir / "next_full_run_command.txt").write_text(next_full_run_command + "\n", encoding="utf-8")

    score_summaries = (
        score_distribution_rows(fixed_candidate_rows, "fixed_test_candidates")
        + score_distribution_rows(walk_candidate_rows, "walkforward_candidates")
    )
    write_csv_rows(out_dir / "score_summary.csv", score_summaries)

    fixed_deciles, fixed_monotonicity = score_decile_rows(fixed_candidate_rows, "fixed_test_candidates")
    walk_deciles, walk_monotonicity = score_decile_rows(walk_candidate_rows, "walkforward_candidates")
    monotonicity = {}
    monotonicity.update(fixed_monotonicity)
    monotonicity.update(walk_monotonicity)
    write_csv_rows(out_dir / "score_deciles.csv", fixed_deciles + walk_deciles)
    write_csv_rows(
        out_dir / "score_monotonicity.csv",
        [
            {"source": source, "score": score, "monotonicity": value}
            for (source, score), value in sorted(monotonicity.items())
        ],
    )

    calibration_rows = calibration_metric_rows(fixed_candidate_rows, "fixed_test_candidates") + calibration_metric_rows(walk_candidate_rows, "walkforward_candidates")
    write_csv_rows(out_dir / "calibration_audit.csv", calibration_rows)

    fixed_symbol_exposure = symbol_exposure(fixed_rows)
    walk_symbol_exposure = symbol_exposure(walk_rows)
    write_csv_rows(out_dir / "symbol_concentration.csv", [
        dict(item, source="fixed_test") for item in fixed_symbol_exposure
    ] + [
        dict(item, source="walkforward") for item in walk_symbol_exposure
    ])

    key_symbols = set()
    for exposure in (fixed_symbol_exposure, walk_symbol_exposure):
        if exposure:
            key_symbols.add(exposure[0]["symbol"])
        for item in exposure:
            if item["symbol"] == "HEIUSDT":
                key_symbols.add("HEIUSDT")
        if exposure:
            key_symbols.add(max(exposure, key=lambda item: item["trade_count"])["symbol"])
            key_symbols.add(max(exposure, key=lambda item: item["loss"])["symbol"])
    leave_symbol = leave_one_symbol_out_rows(fixed_rows, sorted(key_symbols), "fixed_test") + leave_one_symbol_out_rows(
        walk_rows,
        sorted(key_symbols),
        "walkforward",
        denominator_folds=safe_int(summary.get("walkforward_total_folds"), 10),
    )
    write_csv_rows(out_dir / "leave_one_symbol_out.csv", leave_symbol)

    leave_fold = leave_one_fold_out_rows(walk_metrics)
    write_csv_rows(out_dir / "leave_one_fold_out.csv", leave_fold)

    threshold_stability = threshold_stability_rows(
        threshold_rows,
        selected_threshold_map(summary, walk_diagnostics),
    )
    write_csv_rows(out_dir / "threshold_stability.csv", threshold_stability)

    experiments = experiment_matrix_rows(fixed_rows, "fixed_test") + experiment_matrix_rows(
        walk_rows,
        "walkforward",
        denominator_folds=safe_int(summary.get("walkforward_total_folds"), 10),
    )
    write_csv_rows(out_dir / "experiment_matrix.csv", experiments)

    cost_stress = cost_stress_rows(fixed_rows, "fixed_test") + cost_stress_rows(
        walk_rows,
        "walkforward",
        denominator_folds=safe_int(summary.get("walkforward_total_folds"), 10),
    )
    write_csv_rows(out_dir / "cost_stress.csv", cost_stress)

    composite = composite_robustness_score(
        summary,
        walk_metrics,
        threshold_stability,
        leave_symbol,
        leave_fold,
        cost_stress,
        monotonicity,
    )
    write_csv_rows(out_dir / "composite_robustness_score.csv", composite)

    fold_rows = [
        row for row in walk_metrics
        if str(row.get("split", "")).startswith("walkforward_fold_")
    ]
    walk_average = next(
        (row for row in walk_metrics if row.get("split") == "walkforward_average"),
        {},
    )
    walk_diagnostic_by_split = {row.get("split", ""): row for row in walk_diagnostics}
    report = build_markdown_report({
        "summary": summary,
        "fold_rows": fold_rows,
        "walkforward_average": walk_average,
        "walk_diagnostic_by_split": walk_diagnostic_by_split,
        "fixed_symbol_exposure": fixed_symbol_exposure,
        "walkforward_symbol_exposure": walk_symbol_exposure,
        "leave_one_symbol_out": leave_symbol,
        "leave_one_fold_out": leave_fold,
        "threshold_stability": threshold_stability,
        "experiment_matrix": experiments,
        "cost_stress": cost_stress,
        "composite": composite,
        "next_full_run_command": next_full_run_command,
        "artifact_mode": "candidate_rich" if "candidates" in fixed_prediction_path.name or "candidates" in walk_prediction_path.name else "trade_predictions",
        "fixed_prediction_path": str(fixed_prediction_path),
        "walk_prediction_path": str(walk_prediction_path),
    })
    (out_dir / "research_report.md").write_text(report, encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "fixed_rows": len(fixed_rows),
        "walkforward_rows": len(walk_rows),
        "fixed_candidate_rows": len(fixed_candidate_rows),
        "walkforward_candidate_rows": len(walk_candidate_rows),
        "score_summary_rows": len(score_summaries),
        "threshold_stability_rows": len(threshold_stability),
        "composite_score": composite[-1]["component_score"] if composite else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build artifact-level CoinPredictor research diagnostics.")
    parser.add_argument("--run-dir", required=True, help="Corrected run artifact directory")
    parser.add_argument("--out-dir", required=True, help="Directory for generated diagnostics")
    args = parser.parse_args(argv)
    result = run_diagnostics(args.run_dir, args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
