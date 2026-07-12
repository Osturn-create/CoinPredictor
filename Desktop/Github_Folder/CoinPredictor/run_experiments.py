#!/usr/bin/env python3
import argparse
import csv
import datetime
import itertools
import json
import os
import subprocess
import sys


SAME_SPEC_PROFILE_7_8GB = [
    "--memory-budget-gb", "7.8",
    "--max-rss-gb", "7.8",
    "--model", "lightgbm",
    "--split-mode", "ratio",
    "--walk-forward",
    "--feature-storage", "memmap32",
    "--cache-dir", ".gbdt_cache",
    "--max-train-rows", "1500000",
    "--max-validation-rows", "750000",
    "--max-final-train-rows", "1500000",
    "--prediction-batch-rows", "200000",
    "--prediction-output-mode", "trades",
    "--auc-sample-rows", "1000000",
    "--adaptive-threshold-sample-rows", "1000000",
    "--max-bin", "63",
    "--lightgbm-histogram-pool-mb", "128",
    "--subsample-for-bin", "100000",
    "--n-jobs", "2",
]

COMMON_TRADING_ARGS = [
    "--profit-safety", "explore",
    "--min-validation-trades", "5",
    "--max-validation-trades", "250",
    "--positive-weight-cap", "10",
    "--initial-capital", "10000",
    "--max-position-fraction", "0.10",
    "--max-volume-fraction", "0.01",
    "--max-trades-per-period", "10",
    "--trade-period-minutes", "60",
    "--holding-period-minutes", "5",
    "--fee", "0.001",
    "--slippage", "0.0005",
    "--test-slippage-multiplier", "1.0",
    "--validation-slippage-multiplier", "1.0",
]

RESULT_COLUMNS = [
    "experiment_name",
    "profile",
    "objective_mode",
    "threshold_objective",
    "trade_score",
    "trade_selection",
    "top_percent_per_period",
    "ranker_objective",
    "ranker_min_score",
    "ranker_score_upper_quantile",
    "ranker_score_upper_cap",
    "ranker_score_upper_cap_blocked",
    "ranker_group_minutes",
    "ranker_min_group_size",
    "ranker_threshold_search",
    "ranker_relevance_q1",
    "ranker_relevance_q2",
    "ranker_relevance_q3",
    "ranker_adverse_penalty",
    "walk_train_months",
    "walk_validation_months",
    "walk_test_months",
    "regression_calibration",
    "regression_target",
    "hybrid_return_combination",
    "hybrid_min_probability",
    "hybrid_score_mode",
    "hybrid_uncertainty_method",
    "hybrid_uncertainty_penalty",
    "dynamic_hybrid_thresholds",
    "meta_filter",
    "meta_filter_min_probability",
    "symbol_filter_stage",
    "threshold_tiebreaker",
    "ensemble_windows",
    "ev_safety_margin",
    "min_selected_threshold",
    "min_validation_trades",
    "min_validation_precision",
    "top_k_per_minute",
    "top_k_per_symbol_minute",
    "max_trades_per_period",
    "max_trades_per_symbol_period",
    "symbol_reentry_cooldown_minutes",
    "max_same_symbol_streak",
    "max_symbol_fold_trade_share",
    "max_symbol_fold_trade_share_min_trades",
    "max_validation_trades",
    "threshold_require_positive_top_1pct",
    "threshold_max_raw_signal_share",
    "threshold_min_avg_net_return",
    "threshold_min_top_decile_net_return",
    "threshold_min_score_win_loss_gap",
    "threshold_max_top1_concentration",
    "threshold_max_top3_concentration",
    "threshold_max_trade_top1_concentration",
    "threshold_drawdown_penalty",
    "threshold_trade_count_penalty",
    "threshold_burst_trades_per_day_penalty",
    "threshold_burst_max_trades_in_day_penalty",
    "threshold_floor_snap_penalty_weight",
    "threshold_floor_snap_tolerance",
    "threshold_floor_snap_score_tolerance_ratio",
    "threshold_target_trades_per_day",
    "threshold_target_max_trades_in_day",
    "threshold_short_history_days",
    "threshold_short_history_penalty",
    "target_validation_trades",
    "calibration_window_mode",
    "calibration_recent_ratio",
    "calibration_recent_rows",
    "walk_forward_start_fold",
    "walk_forward_max_folds",
    "min_profitable_fold_rate",
    "min_median_fold_return",
    "robustness_gate_action",
    "min_predicted_net_return",
    "hybrid_min_score",
    "total_profit",
    "portfolio_return",
    "profitable_fold_rate",
    "active_fold_rate",
    "active_profitable_fold_rate",
    "median_active_fold_return",
    "median_return",
    "mean_return",
    "worst_fold_return",
    "worst_active_fold_return",
    "trade_count",
    "total_predicted_trades",
    "overactive_losing_folds",
    "avg_trades_in_losing_active_folds",
    "average_profit_per_trade",
    "accepted",
    "strategy_strength",
    "robustness_gate_status",
    "profitable_but_fragile",
    "robustness_failed_checks",
    "ranking_trade_score_top_1pct_avg_net_return",
    "ranking_trade_score_top_5pct_avg_net_return",
    "ranking_trade_score_top_decile_avg_net_return",
    "ranking_ranker_score_top_1pct_avg_net_return",
    "ranking_ranker_score_top_5pct_avg_net_return",
    "ranking_ranker_score_top_decile_avg_net_return",
    "ranking_trade_score_net_return_monotonicity",
    "ranking_ranker_score_net_return_monotonicity",
    "ranking_trade_score_executed_top_symbol_share",
    "ranking_trade_score_executed_top_month_share",
    "ranking_ranker_score_executed_top_symbol_share",
    "ranking_ranker_score_executed_top_month_share",
    "threshold_diagnostics_primary_rejection",
    "threshold_diagnostics_primary_rejection_count",
    "threshold_diagnostics_best_avg_net_return",
    "threshold_diagnostics_best_avg_net_return_trades",
    "threshold_diagnostics_best_top_decile_net_return",
    "threshold_diagnostics_near_miss_count",
    "threshold_diagnostics_near_miss_ignored_flags",
    "threshold_diagnostics_best_near_miss_source_split",
    "threshold_diagnostics_best_near_miss_fold_index",
    "threshold_diagnostics_best_near_miss_threshold",
    "threshold_diagnostics_best_near_miss_trades",
    "threshold_diagnostics_best_near_miss_avg_net_return",
    "threshold_diagnostics_best_near_miss_top_1pct_net_return",
    "threshold_diagnostics_best_near_miss_top_decile_net_return",
    "threshold_diagnostics_best_near_miss_top1_concentration",
    "threshold_diagnostics_best_near_miss_top3_concentration",
    "threshold_diagnostics_best_near_miss_trade_top1_concentration",
    "threshold_diagnostics_best_near_miss_rejection_flags",
    "max_rss_gb_observed",
    "run_exit_code",
]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run same-spec experiment grids against gbdt_pipeline.py")
    parser.add_argument("--profile", choices=["7.8gb", "7.8gb-overtrade-check", "hybrid-calibration", "hybrid-risk-adjusted", "hybrid-meta-filter", "hybrid-ensemble-small", "hybrid-late-recent", "hybrid-late-recent-tuned", "economic-ranker"], default="7.8gb")
    parser.add_argument("--max-runs", type=int, default=6)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--input", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pipeline-script", default="gbdt_pipeline.py")
    parser.add_argument("--summary-path", default="kline_growth_run_summary.json")
    parser.add_argument("--output", default="kline_growth_experiment_grid_results.csv")
    parser.add_argument("--acceptance-tier", choices=["none", "exploration", "research", "strong"], default="exploration")
    parser.add_argument("--results-root", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    return parser.parse_args(argv)


def same_spec_profile_args(profile):
    if profile not in ("7.8gb", "7.8gb-overtrade-check", "hybrid-calibration", "hybrid-risk-adjusted", "hybrid-meta-filter", "hybrid-ensemble-small", "hybrid-late-recent", "hybrid-late-recent-tuned", "economic-ranker"):
        raise ValueError("unsupported profile: {}".format(profile))
    return list(SAME_SPEC_PROFILE_7_8GB)


def classification_experiments():
    for ev_margin, min_threshold, min_precision, top_k in itertools.product(
        (0.002, 0.001, 0.0005),
        (0.90, 0.80),
        (0.25, 0.20, 0.15),
        (1, 3),
    ):
        yield {
            "objective_mode": "classification",
            "trade_score": "ev",
            "ev_safety_margin": ev_margin,
            "min_selected_threshold": min_threshold,
            "min_validation_precision": min_precision,
            "top_k_per_minute": top_k,
            "max_trades_per_period": 10,
            "max_validation_trades": 250,
            "threshold_drawdown_penalty": 0.0,
            "threshold_trade_count_penalty": 0.0,
            "target_validation_trades": 0,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.0,
        }


def return_regression_experiments():
    for min_net_return, top_k in itertools.product((0.0, 0.001), (1, 3)):
        yield {
            "objective_mode": "return_regression",
            "trade_score": "predicted_return",
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0,
            "min_validation_precision": 0.0,
            "top_k_per_minute": top_k,
            "max_trades_per_period": 10,
            "max_validation_trades": 250,
            "threshold_drawdown_penalty": 0.0,
            "threshold_trade_count_penalty": 0.0,
            "target_validation_trades": 0,
            "min_predicted_net_return": min_net_return,
            "hybrid_min_score": 0.0,
        }


def hybrid_experiments():
    for hybrid_min_score, top_k in itertools.product((0.0, 0.001), (1, 3)):
        yield {
            "objective_mode": "hybrid",
            "trade_score": "hybrid",
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0,
            "min_validation_precision": 0.0,
            "top_k_per_minute": top_k,
            "max_trades_per_period": 10,
            "max_validation_trades": 250,
            "threshold_drawdown_penalty": 0.0,
            "threshold_trade_count_penalty": 0.0,
            "target_validation_trades": 0,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": hybrid_min_score,
        }


def overtrade_check_experiments():
    return [
        {
            "experiment_name": "A_topk3_period10",
            "profile": "7.8gb-overtrade-check",
            "objective_mode": "classification",
            "trade_score": "ev",
            "ev_safety_margin": 0.001,
            "min_selected_threshold": 0.80,
            "min_validation_precision": 0.20,
            "top_k_per_minute": 3,
            "max_trades_per_period": 10,
            "max_validation_trades": 250,
            "threshold_drawdown_penalty": 0.0,
            "threshold_trade_count_penalty": 0.0,
            "target_validation_trades": 0,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.0,
        },
        {
            "experiment_name": "B_topk1_period10",
            "profile": "7.8gb-overtrade-check",
            "objective_mode": "classification",
            "trade_score": "ev",
            "ev_safety_margin": 0.001,
            "min_selected_threshold": 0.80,
            "min_validation_precision": 0.20,
            "top_k_per_minute": 1,
            "max_trades_per_period": 10,
            "max_validation_trades": 250,
            "threshold_drawdown_penalty": 0.0,
            "threshold_trade_count_penalty": 0.0,
            "target_validation_trades": 0,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.0,
        },
        {
            "experiment_name": "C_topk1_period5_val150",
            "profile": "7.8gb-overtrade-check",
            "objective_mode": "classification",
            "trade_score": "ev",
            "ev_safety_margin": 0.001,
            "min_selected_threshold": 0.80,
            "min_validation_precision": 0.20,
            "top_k_per_minute": 1,
            "max_trades_per_period": 5,
            "max_validation_trades": 150,
            "threshold_drawdown_penalty": 0.0,
            "threshold_trade_count_penalty": 0.0,
            "target_validation_trades": 0,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.0,
        },
        {
            "experiment_name": "D_topk1_period5_penalized",
            "profile": "7.8gb-overtrade-check",
            "objective_mode": "classification",
            "trade_score": "ev",
            "ev_safety_margin": 0.001,
            "min_selected_threshold": 0.80,
            "min_validation_precision": 0.20,
            "top_k_per_minute": 1,
            "max_trades_per_period": 5,
            "max_validation_trades": 150,
            "threshold_drawdown_penalty": 0.05,
            "threshold_trade_count_penalty": 0.0005,
            "target_validation_trades": 100,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.0,
        },
    ]


def hybrid_calibration_experiments():
    for regression_calibration, regression_target, hybrid_min_score in itertools.product(
        ("none", "linear"),
        ("trade_return", "net_return", "clipped_net_return"),
        (0.001, 0.0015),
    ):
        yield {
            "profile": "hybrid-calibration",
            "objective_mode": "hybrid",
            "trade_score": "hybrid",
            "regression_calibration": regression_calibration,
            "regression_target": regression_target,
            "hybrid_score_mode": "basic",
            "hybrid_uncertainty_method": "none",
            "hybrid_uncertainty_penalty": 0.0,
            "dynamic_hybrid_thresholds": "none",
            "meta_filter": "none",
            "meta_filter_min_probability": 0.5,
            "ensemble_windows": "",
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0,
            "min_validation_precision": 0.0,
            "top_k_per_minute": 1,
            "max_trades_per_period": 5,
            "max_validation_trades": 150,
            "threshold_drawdown_penalty": 0.0,
            "threshold_trade_count_penalty": 0.0,
            "target_validation_trades": 0,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": hybrid_min_score,
        }


def hybrid_risk_adjusted_experiments():
    for score_mode, uncertainty_method, penalty in itertools.product(
        ("basic", "risk_adjusted"),
        ("global_residual", "bucket_residual"),
        (0.25, 0.5),
    ):
        yield {
            "profile": "hybrid-risk-adjusted",
            "objective_mode": "hybrid",
            "trade_score": "hybrid",
            "regression_calibration": "linear",
            "regression_target": "clipped_net_return",
            "hybrid_score_mode": score_mode,
            "hybrid_uncertainty_method": uncertainty_method,
            "hybrid_uncertainty_penalty": penalty,
            "dynamic_hybrid_thresholds": "btc_volatility_regime",
            "meta_filter": "none",
            "meta_filter_min_probability": 0.5,
            "ensemble_windows": "",
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0,
            "min_validation_precision": 0.0,
            "top_k_per_minute": 1,
            "max_trades_per_period": 5,
            "max_validation_trades": 150,
            "threshold_drawdown_penalty": 0.05,
            "threshold_trade_count_penalty": 0.0005,
            "target_validation_trades": 100,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.001,
        }


def hybrid_meta_filter_experiments():
    for meta_filter, min_probability in itertools.product(
        ("none", "logistic"),
        (0.5, 0.6),
    ):
        yield {
            "profile": "hybrid-meta-filter",
            "objective_mode": "hybrid",
            "trade_score": "hybrid",
            "regression_calibration": "linear",
            "regression_target": "clipped_net_return",
            "hybrid_score_mode": "risk_adjusted",
            "hybrid_uncertainty_method": "bucket_residual",
            "hybrid_uncertainty_penalty": 0.25,
            "dynamic_hybrid_thresholds": "btc_volatility_regime",
            "meta_filter": meta_filter,
            "meta_filter_min_probability": min_probability,
            "ensemble_windows": "",
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0,
            "min_validation_precision": 0.0,
            "top_k_per_minute": 1,
            "max_trades_per_period": 5,
            "max_validation_trades": 150,
            "threshold_drawdown_penalty": 0.05,
            "threshold_trade_count_penalty": 0.0005,
            "target_validation_trades": 100,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.001,
        }


def hybrid_ensemble_small_experiments():
    for ensemble_windows in ("", "6,9"):
        yield {
            "profile": "hybrid-ensemble-small",
            "objective_mode": "hybrid",
            "trade_score": "hybrid",
            "regression_calibration": "linear",
            "regression_target": "clipped_net_return",
            "hybrid_score_mode": "risk_adjusted",
            "hybrid_uncertainty_method": "bucket_residual",
            "hybrid_uncertainty_penalty": 0.25,
            "dynamic_hybrid_thresholds": "btc_volatility_regime",
            "meta_filter": "logistic",
            "meta_filter_min_probability": 0.55,
            "ensemble_windows": ensemble_windows,
            "walk_forward": False,
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0,
            "min_validation_precision": 0.0,
            "top_k_per_minute": 1,
            "max_trades_per_period": 5,
            "max_validation_trades": 150,
            "threshold_drawdown_penalty": 0.05,
            "threshold_trade_count_penalty": 0.0005,
            "target_validation_trades": 100,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.001,
        }


def hybrid_late_recent_experiments():
    for burst_penalty, short_history_penalty, recent_rows in itertools.product(
        (0.03, 0.04),
        (0.12, 0.18),
        (200000, 250000),
    ):
        yield {
            "profile": "hybrid-late-recent",
            "input": "shard_dataset_30_volatile_from_existing",
            "cache_dir": ".gbdt_cache_full30_volatile",
            "objective_mode": "hybrid",
            "trade_score": "hybrid",
            "regression_calibration": "linear",
            "regression_target": "clipped_net_return",
            "hybrid_score_mode": "risk_adjusted",
            "hybrid_uncertainty_method": "bucket_residual",
            "hybrid_uncertainty_penalty": 0.25,
            "dynamic_hybrid_thresholds": "btc_volatility_regime",
            "meta_filter": "logistic",
            "meta_filter_min_probability": 0.55,
            "ensemble_windows": "",
            "walk_forward": True,
            "walk_forward_start_fold": 70,
            "walk_forward_max_folds": 10,
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0,
            "min_validation_precision": 0.0,
            "top_k_per_minute": 1,
            "max_trades_per_period": 5,
            "max_validation_trades": 150,
            "threshold_drawdown_penalty": 0.05,
            "threshold_trade_count_penalty": 0.0005,
            "threshold_burst_trades_per_day_penalty": burst_penalty,
            "threshold_burst_max_trades_in_day_penalty": burst_penalty * 0.5,
            "threshold_target_trades_per_day": 3.5,
            "threshold_target_max_trades_in_day": 6,
            "threshold_short_history_days": 45.0,
            "threshold_short_history_penalty": short_history_penalty,
            "target_validation_trades": 100,
            "calibration_window_mode": "recent",
            "calibration_recent_ratio": 0.0,
            "calibration_recent_rows": recent_rows,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.001,
            "hybrid_min_score_calibration_aware": True,
            "hybrid_min_score_calibration_reference_scale": 0.20,
            "hybrid_min_score_calibration_min_ratio": 0.25,
            "hybrid_min_score_calibration_floor_min": 0.00025,
        }


def hybrid_late_recent_tuned_experiments():
    for walk_train_months, floor_snap_penalty_weight in itertools.product(
        (6, 8),
        (0.04, 0.05),
    ):
        yield {
            "profile": "hybrid-late-recent-tuned",
            "input": "shard_dataset_30_volatile_from_existing",
            "cache_dir": ".gbdt_cache_full30_volatile",
            "objective_mode": "hybrid",
            "trade_score": "hybrid",
            "walk_train_months": walk_train_months,
            "walk_validation_months": 1,
            "walk_test_months": 1,
            "walk_forward": True,
            "walk_forward_start_fold": 88,
            "walk_forward_max_folds": 10,
            "regression_calibration": "linear",
            "regression_target": "clipped_net_return",
            "hybrid_return_combination": "expected_return",
            "hybrid_min_probability": 0.03,
            "hybrid_score_mode": "risk_adjusted",
            "hybrid_uncertainty_method": "bucket_residual",
            "hybrid_uncertainty_penalty": 0.10,
            "dynamic_hybrid_thresholds": "none",
            "meta_filter": "none",
            "meta_filter_min_probability": 0.5,
            "symbol_filter_stage": "candidate_blend",
            "threshold_tiebreaker": "balanced",
            "ensemble_windows": "",
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0005,
            "min_validation_precision": 0.02,
            "top_k_per_minute": 2,
            "top_k_per_symbol_minute": 1,
            "max_trades_per_period": 0,
            "max_validation_trades": 400,
            "threshold_drawdown_penalty": 0.02,
            "threshold_trade_count_penalty": 0.00025,
            "threshold_burst_trades_per_day_penalty": 0.01,
            "threshold_burst_max_trades_in_day_penalty": 0.005,
            "threshold_floor_snap_penalty_weight": floor_snap_penalty_weight,
            "threshold_floor_snap_tolerance": 0.0003,
            "threshold_floor_snap_score_tolerance_ratio": 0.10,
            "threshold_target_trades_per_day": 5.0,
            "threshold_target_max_trades_in_day": 10,
            "threshold_short_history_days": 45.0,
            "threshold_short_history_penalty": 0.06,
            "target_validation_trades": 220,
            "calibration_window_mode": "recent",
            "calibration_recent_ratio": 0.0,
            "calibration_recent_rows": 250000,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.0001,
            "hybrid_min_score_calibration_aware": True,
            "hybrid_min_score_calibration_reference_scale": 0.10,
            "hybrid_min_score_calibration_min_ratio": 0.05,
            "hybrid_min_score_calibration_floor_min": -0.0012,
            "hybrid_min_score_calibration_floor_max": -0.0007,
        }


def economic_ranker_experiments():
    for trade_selection, top_k, top_percent, adverse_penalty, max_trades_per_period, max_validation_trades, target_validation_trades in (
        ("topk_score", 1, 0.0, 0.10, 6, 120, 40),
        ("topk_score", 1, 0.0, 0.20, 6, 120, 40),
        ("top_percent_score", 0, 0.03, 0.10, 6, 120, 40),
    ):
        yield {
            "profile": "economic-ranker",
            "input": "shard_dataset_30_volatile_from_existing",
            "cache_dir": ".gbdt_cache_full30_volatile",
            "max_train_rows": 750000,
            "max_validation_rows": 250000,
            "max_final_train_rows": 750000,
            "prediction_batch_rows": 500000,
            "auc_sample_rows": 250000,
            "adaptive_threshold_sample_rows": 250000,
            "n_estimators": 120,
            "model_candidate_count": 2,
            "skip_full_validation_retune": True,
            "objective_mode": "economic_ranking",
            "threshold_objective": "avg_profit",
            "trade_score": "ranker_score",
            "trade_selection": trade_selection,
            "walk_train_months": 8,
            "walk_validation_months": 2,
            "walk_test_months": 1,
            "walk_forward": True,
            "walk_forward_start_fold": 88,
            "walk_forward_max_folds": 10,
            "ranker_objective": "rank_xendcg",
            "ranker_min_score": -1000000000.0,
            "ranker_score_upper_quantile": 0.90,
            "ranker_threshold_search": True,
            "ranker_group_minutes": 5,
            "ranker_min_group_size": 2,
            "ranker_relevance_q1": 0.50,
            "ranker_relevance_q2": 0.75,
            "ranker_relevance_q3": 0.90,
            "ranker_adverse_penalty": adverse_penalty,
            "ev_safety_margin": 0.0,
            "min_selected_threshold": 0.0,
            "min_validation_trades": 20,
            "min_validation_precision": 0.0,
            "top_k_per_minute": top_k,
            "top_percent_per_period": top_percent,
            "top_k_per_symbol_minute": 1,
            "max_trades_per_period": max_trades_per_period,
            "max_trades_per_symbol_period": 1,
            "symbol_reentry_cooldown_minutes": 240,
            "max_same_symbol_streak": 4,
            "max_symbol_fold_trade_share": 0.30,
            "max_symbol_fold_trade_share_min_trades": 20,
            "max_validation_trades": max_validation_trades,
            "threshold_require_positive_top_1pct": True,
            "threshold_max_raw_signal_share": 0.010,
            "threshold_min_avg_net_return": 0.0004,
            "threshold_min_top_decile_net_return": 0.0,
            "threshold_min_score_win_loss_gap": 0.0,
            "threshold_max_top1_concentration": 0.70,
            "threshold_max_top3_concentration": 0.90,
            "threshold_max_trade_top1_concentration": 0.35,
            "threshold_concentration_cap_mode": "hard",
            "threshold_drawdown_penalty": 0.02,
            "threshold_trade_count_penalty": 0.00025,
            "threshold_burst_trades_per_day_penalty": 0.01,
            "threshold_burst_max_trades_in_day_penalty": 0.005,
            "threshold_target_trades_per_day": 5.0,
            "threshold_target_max_trades_in_day": 10,
            "threshold_short_history_days": 45.0,
            "threshold_short_history_penalty": 0.06,
            "target_validation_trades": target_validation_trades,
            "min_profitable_fold_rate": 0.50,
            "min_median_fold_return": 0.0,
            "robustness_gate_action": "reject",
            "robust_require_positive_top_1pct": True,
            "robust_require_positive_top_5pct": True,
            "robust_require_positive_top_decile": True,
            "robust_min_executed_score_gap": 0.0,
            "min_predicted_net_return": 0.0,
            "hybrid_min_score": 0.0,
        }


def build_experiment_grid_for_profile(profile, full_grid=False, max_runs=6):
    if profile == "7.8gb-overtrade-check":
        experiments = overtrade_check_experiments()
        return experiments[:max(0, max_runs)]
    if profile == "hybrid-calibration":
        return list(hybrid_calibration_experiments())[:max(0, max_runs)]
    if profile == "hybrid-risk-adjusted":
        return list(hybrid_risk_adjusted_experiments())[:max(0, max_runs)]
    if profile == "hybrid-meta-filter":
        return list(hybrid_meta_filter_experiments())[:max(0, max_runs)]
    if profile == "hybrid-ensemble-small":
        return list(hybrid_ensemble_small_experiments())[:max(0, max_runs)]
    if profile == "hybrid-late-recent":
        return list(hybrid_late_recent_experiments())[:max(0, max_runs)]
    if profile == "hybrid-late-recent-tuned":
        return list(hybrid_late_recent_tuned_experiments())[:max(0, max_runs)]
    if profile == "economic-ranker":
        return list(economic_ranker_experiments())[:max(0, max_runs)]
    experiments = list(classification_experiments())
    experiments.extend(return_regression_experiments())
    experiments.extend(hybrid_experiments())
    if not full_grid:
        return experiments[:max(0, max_runs)]
    return experiments


def experiment_name(experiment):
    if experiment.get("profile") == "7.8gb-overtrade-check":
        return experiment["experiment_name"]
    if experiment.get("profile") == "hybrid-calibration":
        return "hyb_cal_{}_{}_score{}".format(
            experiment["regression_calibration"],
            experiment["regression_target"],
            str(experiment["hybrid_min_score"]).replace(".", "p"),
        )
    if experiment.get("profile") == "hybrid-risk-adjusted":
        return "hyb_risk_{}_{}_p{}".format(
            experiment["hybrid_score_mode"],
            experiment["hybrid_uncertainty_method"],
            str(experiment["hybrid_uncertainty_penalty"]).replace(".", "p"),
        )
    if experiment.get("profile") == "hybrid-meta-filter":
        return "hyb_meta_{}_p{}".format(
            experiment["meta_filter"],
            str(experiment["meta_filter_min_probability"]).replace(".", "p"),
        )
    if experiment.get("profile") == "hybrid-ensemble-small":
        label = experiment["ensemble_windows"].replace(",", "_") or "none"
        return "hyb_ens_{}".format(label)
    if experiment.get("profile") == "hybrid-late-recent":
        return "hyb_late_recent_b{}_s{}_rows{}".format(
            str(experiment["threshold_burst_trades_per_day_penalty"]).replace(".", "p"),
            str(experiment["threshold_short_history_penalty"]).replace(".", "p"),
            experiment["calibration_recent_rows"],
        )
    if experiment.get("profile") == "hybrid-late-recent-tuned":
        return "hyb_late_tuned_{}m_sf{}_f{}".format(
            experiment.get("walk_train_months", 6),
            str(experiment.get("threshold_floor_snap_penalty_weight", 0.0)).replace(".", "p"),
            experiment.get("walk_forward_start_fold", 88),
        )
    if experiment.get("profile") == "economic-ranker":
        selection = experiment.get("trade_selection", "topk_score").replace("_", "")
        if experiment.get("trade_selection") == "top_percent_score":
            selection = "toppct{}".format(str(experiment.get("top_percent_per_period", 0.0)).replace(".", "p"))
        else:
            selection = "topk{}".format(experiment.get("top_k_per_minute", 0))
        return "ranker_{}m_{}_adv{}".format(
            experiment.get("walk_train_months", 8),
            selection,
            str(experiment.get("ranker_adverse_penalty", 0.0)).replace(".", "p"),
        )
    if experiment["objective_mode"] == "classification":
        return "cls_ev_m{}_thr{}_prec{}_topk{}".format(
            str(experiment["ev_safety_margin"]).replace(".", "p"),
            str(experiment["min_selected_threshold"]).replace(".", "p"),
            str(experiment["min_validation_precision"]).replace(".", "p"),
            experiment["top_k_per_minute"],
        )
    if experiment["objective_mode"] == "return_regression":
        return "ret_net{}_topk{}".format(
            str(experiment["min_predicted_net_return"]).replace(".", "p"),
            experiment["top_k_per_minute"],
        )
    return "hybrid_score{}_topk{}".format(
        str(experiment["hybrid_min_score"]).replace(".", "p"),
        experiment["top_k_per_minute"],
    )


def set_command_option(command, flag, value):
    for index, item in enumerate(command[:-1]):
        if item == flag:
            command[index + 1] = str(value)
            return
    command.extend([flag, str(value)])


def ensure_command_flag(command, flag):
    if flag not in command:
        command.append(flag)


def build_command(args, experiment):
    pipeline_path = args.pipeline_script
    if not os.path.isabs(pipeline_path):
        pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pipeline_path)
    command = [args.python, pipeline_path]
    command.extend(same_spec_profile_args(args.profile))
    command.extend(COMMON_TRADING_ARGS)
    input_path = experiment.get("input") or args.input
    if input_path:
        command.extend(["--input", input_path])
    cache_dir = experiment.get("cache_dir", "")
    if cache_dir:
        set_command_option(command, "--cache-dir", cache_dir)
    for flag, key in (
        ("--max-train-rows", "max_train_rows"),
        ("--max-validation-rows", "max_validation_rows"),
        ("--max-final-train-rows", "max_final_train_rows"),
        ("--prediction-batch-rows", "prediction_batch_rows"),
        ("--auc-sample-rows", "auc_sample_rows"),
        ("--adaptive-threshold-sample-rows", "adaptive_threshold_sample_rows"),
        ("--n-estimators", "n_estimators"),
        ("--model-candidate-count", "model_candidate_count"),
    ):
        if key in experiment:
            set_command_option(command, flag, experiment[key])
    if experiment.get("skip_full_validation_retune", False):
        ensure_command_flag(command, "--skip-full-validation-retune")
    command.append("--cache-only")
    if not experiment.get("walk_forward", True):
        command = [value for value in command if value != "--walk-forward"]
    command.extend([
        "--objective-mode", experiment["objective_mode"],
        "--trade-selection", experiment.get("trade_selection", "topk_score"),
        "--trade-score", experiment["trade_score"],
        "--walk-train-months", str(experiment.get("walk_train_months", 6)),
        "--walk-validation-months", str(experiment.get("walk_validation_months", 1)),
        "--walk-test-months", str(experiment.get("walk_test_months", 1)),
        "--top-k-per-minute", str(experiment["top_k_per_minute"]),
        "--top-percent-per-period", str(experiment.get("top_percent_per_period", 0.0)),
        "--top-k-per-symbol-minute", str(experiment.get("top_k_per_symbol_minute", 0)),
        "--max-trades-per-period", str(experiment.get("max_trades_per_period", 10)),
        "--max-trades-per-symbol-period", str(experiment.get("max_trades_per_symbol_period", 0)),
        "--symbol-reentry-cooldown-minutes", str(experiment.get("symbol_reentry_cooldown_minutes", 0)),
        "--max-same-symbol-streak", str(experiment.get("max_same_symbol_streak", 0)),
        "--max-symbol-fold-trade-share", str(experiment.get("max_symbol_fold_trade_share", 0.0)),
        "--max-symbol-fold-trade-share-min-trades", str(experiment.get("max_symbol_fold_trade_share_min_trades", 0)),
        "--min-validation-trades", str(experiment.get("min_validation_trades", 5)),
        "--max-validation-trades", str(experiment.get("max_validation_trades", 250)),
        "--threshold-max-raw-signal-share", str(experiment.get("threshold_max_raw_signal_share", 0.0)),
        "--threshold-min-avg-net-return", str(experiment.get("threshold_min_avg_net_return", -999.0)),
        "--threshold-min-top-decile-net-return", str(experiment.get("threshold_min_top_decile_net_return", -999.0)),
        "--threshold-min-score-win-loss-gap", str(experiment.get("threshold_min_score_win_loss_gap", -999.0)),
        "--threshold-max-top1-concentration", str(experiment.get("threshold_max_top1_concentration", 0.0)),
        "--threshold-max-top3-concentration", str(experiment.get("threshold_max_top3_concentration", 0.0)),
        "--threshold-max-trade-top1-concentration", str(experiment.get("threshold_max_trade_top1_concentration", 0.0)),
        "--threshold-concentration-cap-mode", experiment.get("threshold_concentration_cap_mode", "soft"),
        "--threshold-drawdown-penalty", str(experiment.get("threshold_drawdown_penalty", 0.0)),
        "--threshold-trade-count-penalty", str(experiment.get("threshold_trade_count_penalty", 0.0)),
        "--target-validation-trades", str(experiment.get("target_validation_trades", 0)),
        "--acceptance-tier", args.acceptance_tier,
        "--min-profitable-fold-rate", str(experiment.get("min_profitable_fold_rate", 0.0)),
        "--min-median-fold-return", str(experiment.get("min_median_fold_return", -999.0)),
        "--robustness-gate-action", experiment.get("robustness_gate_action", "warn"),
        "--robust-min-executed-score-gap", str(experiment.get("robust_min_executed_score_gap", -999.0)),
        "--run-summary-out", args.summary_path,
        "--walk-forward-start-fold", str(experiment.get("walk_forward_start_fold", 0)),
        "--walk-forward-max-folds", str(experiment.get("walk_forward_max_folds", 0)),
        "--regression-calibration", experiment.get("regression_calibration", "none"),
        "--regression-target", experiment.get("regression_target", "trade_return"),
        "--hybrid-score-mode", experiment.get("hybrid_score_mode", "basic"),
        "--hybrid-uncertainty-method", experiment.get("hybrid_uncertainty_method", "none"),
        "--hybrid-uncertainty-penalty", str(experiment.get("hybrid_uncertainty_penalty", 0.0)),
        "--dynamic-hybrid-thresholds", experiment.get("dynamic_hybrid_thresholds", "none"),
        "--meta-filter", experiment.get("meta_filter", "none"),
        "--meta-filter-min-probability", str(experiment.get("meta_filter_min_probability", 0.5)),
        "--threshold-burst-trades-per-day-penalty", str(experiment.get("threshold_burst_trades_per_day_penalty", 0.0)),
        "--threshold-burst-max-trades-in-day-penalty", str(experiment.get("threshold_burst_max_trades_in_day_penalty", 0.0)),
        "--threshold-floor-snap-penalty-weight", str(experiment.get("threshold_floor_snap_penalty_weight", 0.0)),
        "--threshold-floor-snap-tolerance", str(experiment.get("threshold_floor_snap_tolerance", 0.0)),
        "--threshold-floor-snap-score-tolerance-ratio", str(experiment.get("threshold_floor_snap_score_tolerance_ratio", 0.0)),
        "--threshold-target-trades-per-day", str(experiment.get("threshold_target_trades_per_day", 0.0)),
        "--threshold-target-max-trades-in-day", str(experiment.get("threshold_target_max_trades_in_day", 0)),
        "--threshold-short-history-days", str(experiment.get("threshold_short_history_days", 0.0)),
        "--threshold-short-history-penalty", str(experiment.get("threshold_short_history_penalty", 0.0)),
        "--symbol-filter-stage", experiment.get("symbol_filter_stage", "executed"),
        "--threshold-tiebreaker", experiment.get("threshold_tiebreaker", "fewer_trades"),
    ])
    if experiment.get("threshold_require_positive_top_1pct", False):
        command.append("--threshold-require-positive-top-1pct")
    if experiment.get("robust_require_positive_top_1pct", False):
        command.append("--robust-require-positive-top-1pct")
    if experiment.get("robust_require_positive_top_5pct", False):
        command.append("--robust-require-positive-top-5pct")
    if experiment.get("robust_require_positive_top_decile", False):
        command.append("--robust-require-positive-top-decile")
    if experiment.get("ensemble_windows", ""):
        command.extend(["--ensemble-windows", experiment["ensemble_windows"]])
    if experiment["objective_mode"] == "classification":
        command.extend([
            "--threshold-objective", "ev",
            "--calibration", "platt",
            "--calibration-max-rows", "500000",
            "--calibration-window-mode", experiment.get("calibration_window_mode", "all"),
            "--calibration-recent-ratio", str(experiment.get("calibration_recent_ratio", 0.0)),
            "--calibration-recent-rows", str(experiment.get("calibration_recent_rows", 0)),
            "--ev-safety-margin", str(experiment["ev_safety_margin"]),
            "--min-selected-threshold", str(experiment["min_selected_threshold"]),
            "--min-validation-precision", str(experiment["min_validation_precision"]),
        ])
    elif experiment["objective_mode"] == "return_regression":
        command.extend([
            "--min-predicted-net-return", str(experiment["min_predicted_net_return"]),
        ])
    elif experiment["objective_mode"] == "economic_ranking":
        command.extend([
            "--threshold-objective", experiment.get("threshold_objective", "avg_profit"),
            "--ranker-objective", experiment.get("ranker_objective", "rank_xendcg"),
            "--ranker-min-score", str(experiment.get("ranker_min_score", -1000000000.0)),
            "--ranker-score-upper-quantile", str(experiment.get("ranker_score_upper_quantile", 1.0)),
            "--ranker-group-minutes", str(experiment.get("ranker_group_minutes", 1)),
            "--ranker-min-group-size", str(experiment.get("ranker_min_group_size", 2)),
            "--ranker-relevance-q1", str(experiment.get("ranker_relevance_q1", 0.50)),
            "--ranker-relevance-q2", str(experiment.get("ranker_relevance_q2", 0.75)),
            "--ranker-relevance-q3", str(experiment.get("ranker_relevance_q3", 0.90)),
            "--ranker-adverse-penalty", str(experiment.get("ranker_adverse_penalty", 0.0)),
            "--min-selected-threshold", str(experiment.get("min_selected_threshold", 0.0)),
            "--min-validation-precision", str(experiment.get("min_validation_precision", 0.0)),
        ])
        if experiment.get("ranker_threshold_search", False):
            command.append("--ranker-threshold-search")
    else:
        command.extend([
            "--calibration", "platt",
            "--calibration-max-rows", "500000",
            "--calibration-window-mode", experiment.get("calibration_window_mode", "all"),
            "--calibration-recent-ratio", str(experiment.get("calibration_recent_ratio", 0.0)),
            "--calibration-recent-rows", str(experiment.get("calibration_recent_rows", 0)),
            "--hybrid-return-combination", experiment.get("hybrid_return_combination", "probability_times_return"),
            "--hybrid-min-probability", str(experiment.get("hybrid_min_probability", 0.0)),
            "--hybrid-min-score", str(experiment["hybrid_min_score"]),
            "--min-selected-threshold", str(experiment.get("min_selected_threshold", 0.0)),
            "--min-validation-precision", str(experiment.get("min_validation_precision", 0.0)),
        ])
        if experiment.get("hybrid_min_score_calibration_aware"):
            command.append("--hybrid-min-score-calibration-aware")
        command.extend([
            "--hybrid-min-score-calibration-reference-scale", str(experiment.get("hybrid_min_score_calibration_reference_scale", 0.20)),
            "--hybrid-min-score-calibration-min-ratio", str(experiment.get("hybrid_min_score_calibration_min_ratio", 0.25)),
            "--hybrid-min-score-calibration-floor-min", str(experiment.get("hybrid_min_score_calibration_floor_min", 0.0)),
            "--hybrid-min-score-calibration-floor-max", str(experiment.get("hybrid_min_score_calibration_floor_max", 0.0)),
        ])
    return command


def load_summary(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def float_value(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summary_record(experiment, summary, run_exit_code):
    fixed = summary.get("fixed_metrics") or {}
    walk_aggregate = summary.get("walk_forward_aggregate_metrics") or {}
    walk_summary = summary.get("walk_forward_summary") or {}
    source = walk_aggregate or fixed
    total_trades = int(float_value(walk_summary.get("walkforward_total_predicted_trades", source.get("predicted_trades", 0))))
    return {
        "experiment_name": experiment_name(experiment),
        "profile": experiment.get("profile", ""),
        "objective_mode": experiment["objective_mode"],
        "threshold_objective": experiment.get("threshold_objective", ""),
        "trade_score": experiment["trade_score"],
        "trade_selection": experiment.get("trade_selection", "topk_score"),
        "top_percent_per_period": experiment.get("top_percent_per_period", 0.0),
        "ranker_objective": experiment.get("ranker_objective", "none"),
        "ranker_min_score": experiment.get("ranker_min_score", 0.0),
        "ranker_score_upper_quantile": experiment.get("ranker_score_upper_quantile", 1.0),
        "ranker_score_upper_cap": float_value(source.get("ranker_score_upper_cap", 0.0)),
        "ranker_score_upper_cap_blocked": int(float_value(source.get("ranker_score_upper_cap_blocked", 0))),
        "ranker_group_minutes": experiment.get("ranker_group_minutes", 0),
        "ranker_min_group_size": experiment.get("ranker_min_group_size", 0),
        "ranker_threshold_search": int(bool(experiment.get("ranker_threshold_search", False))),
        "ranker_relevance_q1": experiment.get("ranker_relevance_q1", 0.0),
        "ranker_relevance_q2": experiment.get("ranker_relevance_q2", 0.0),
        "ranker_relevance_q3": experiment.get("ranker_relevance_q3", 0.0),
        "ranker_adverse_penalty": experiment.get("ranker_adverse_penalty", 0.0),
        "walk_train_months": experiment.get("walk_train_months", 6),
        "walk_validation_months": experiment.get("walk_validation_months", 1),
        "walk_test_months": experiment.get("walk_test_months", 1),
        "regression_calibration": experiment.get("regression_calibration", "none"),
        "regression_target": experiment.get("regression_target", "trade_return"),
        "hybrid_return_combination": experiment.get("hybrid_return_combination", "probability_times_return"),
        "hybrid_min_probability": experiment.get("hybrid_min_probability", 0.0),
        "hybrid_score_mode": experiment.get("hybrid_score_mode", "basic"),
        "hybrid_uncertainty_method": experiment.get("hybrid_uncertainty_method", "none"),
        "hybrid_uncertainty_penalty": experiment.get("hybrid_uncertainty_penalty", 0.0),
        "dynamic_hybrid_thresholds": experiment.get("dynamic_hybrid_thresholds", "none"),
        "meta_filter": experiment.get("meta_filter", "none"),
        "meta_filter_min_probability": experiment.get("meta_filter_min_probability", 0.5),
        "symbol_filter_stage": experiment.get("symbol_filter_stage", "executed"),
        "threshold_tiebreaker": experiment.get("threshold_tiebreaker", "fewer_trades"),
        "ensemble_windows": experiment.get("ensemble_windows", ""),
        "ev_safety_margin": experiment.get("ev_safety_margin", 0.0),
        "min_selected_threshold": experiment.get("min_selected_threshold", 0.0),
        "min_validation_trades": experiment.get("min_validation_trades", 5),
        "min_validation_precision": experiment.get("min_validation_precision", 0.0),
        "top_k_per_minute": experiment.get("top_k_per_minute", 0),
        "top_k_per_symbol_minute": experiment.get("top_k_per_symbol_minute", 0),
        "max_trades_per_period": experiment.get("max_trades_per_period", 10),
        "max_trades_per_symbol_period": experiment.get("max_trades_per_symbol_period", 0),
        "symbol_reentry_cooldown_minutes": experiment.get("symbol_reentry_cooldown_minutes", 0),
        "max_same_symbol_streak": experiment.get("max_same_symbol_streak", 0),
        "max_symbol_fold_trade_share": experiment.get("max_symbol_fold_trade_share", 0.0),
        "max_symbol_fold_trade_share_min_trades": experiment.get("max_symbol_fold_trade_share_min_trades", 0),
        "max_validation_trades": experiment.get("max_validation_trades", 250),
        "threshold_require_positive_top_1pct": int(bool(experiment.get("threshold_require_positive_top_1pct", False))),
        "threshold_max_raw_signal_share": experiment.get("threshold_max_raw_signal_share", 0.0),
        "threshold_min_avg_net_return": experiment.get("threshold_min_avg_net_return", -999.0),
        "threshold_min_top_decile_net_return": experiment.get("threshold_min_top_decile_net_return", -999.0),
        "threshold_min_score_win_loss_gap": experiment.get("threshold_min_score_win_loss_gap", -999.0),
        "threshold_max_top1_concentration": experiment.get("threshold_max_top1_concentration", 0.0),
        "threshold_max_top3_concentration": experiment.get("threshold_max_top3_concentration", 0.0),
        "threshold_max_trade_top1_concentration": experiment.get("threshold_max_trade_top1_concentration", 0.0),
        "threshold_drawdown_penalty": experiment.get("threshold_drawdown_penalty", 0.0),
        "threshold_trade_count_penalty": experiment.get("threshold_trade_count_penalty", 0.0),
        "threshold_burst_trades_per_day_penalty": experiment.get("threshold_burst_trades_per_day_penalty", 0.0),
        "threshold_burst_max_trades_in_day_penalty": experiment.get("threshold_burst_max_trades_in_day_penalty", 0.0),
        "threshold_floor_snap_penalty_weight": experiment.get("threshold_floor_snap_penalty_weight", 0.0),
        "threshold_floor_snap_tolerance": experiment.get("threshold_floor_snap_tolerance", 0.0),
        "threshold_floor_snap_score_tolerance_ratio": experiment.get("threshold_floor_snap_score_tolerance_ratio", 0.0),
        "threshold_target_trades_per_day": experiment.get("threshold_target_trades_per_day", 0.0),
        "threshold_target_max_trades_in_day": experiment.get("threshold_target_max_trades_in_day", 0),
        "threshold_short_history_days": experiment.get("threshold_short_history_days", 0.0),
        "threshold_short_history_penalty": experiment.get("threshold_short_history_penalty", 0.0),
        "target_validation_trades": experiment.get("target_validation_trades", 0),
        "calibration_window_mode": experiment.get("calibration_window_mode", "all"),
        "calibration_recent_ratio": experiment.get("calibration_recent_ratio", 0.0),
        "calibration_recent_rows": experiment.get("calibration_recent_rows", 0),
        "walk_forward_start_fold": experiment.get("walk_forward_start_fold", 0),
        "walk_forward_max_folds": experiment.get("walk_forward_max_folds", 0),
        "min_profitable_fold_rate": experiment.get("min_profitable_fold_rate", 0.0),
        "min_median_fold_return": experiment.get("min_median_fold_return", -999.0),
        "robustness_gate_action": experiment.get("robustness_gate_action", "warn"),
        "min_predicted_net_return": experiment.get("min_predicted_net_return", 0.0),
        "hybrid_min_score": experiment.get("hybrid_min_score", 0.0),
        "total_profit": float_value(source.get("portfolio_profit", 0.0)),
        "portfolio_return": float_value(source.get("portfolio_return", 0.0)),
        "profitable_fold_rate": float_value(walk_summary.get("walkforward_profitable_fold_rate", 0.0)),
        "active_fold_rate": float_value(walk_summary.get("active_fold_rate", 0.0)),
        "active_profitable_fold_rate": float_value(walk_summary.get("active_profitable_fold_rate", 0.0)),
        "median_active_fold_return": float_value(walk_summary.get("median_active_fold_return", 0.0)),
        "median_return": float_value(walk_summary.get("walkforward_median_portfolio_return", 0.0)),
        "mean_return": float_value(walk_summary.get("walkforward_mean_portfolio_return", 0.0)),
        "worst_fold_return": float_value(walk_summary.get("walkforward_min_portfolio_return", 0.0)),
        "worst_active_fold_return": float_value(walk_summary.get("worst_active_fold_return", 0.0)),
        "trade_count": total_trades,
        "total_predicted_trades": total_trades,
        "overactive_losing_folds": int(float_value(walk_summary.get("overactive_losing_folds", 0))),
        "avg_trades_in_losing_active_folds": float_value(walk_summary.get("avg_trades_in_losing_active_folds", 0.0)),
        "average_profit_per_trade": float_value(source.get("average_profit_per_trade", 0.0)),
        "accepted": int(float_value(summary.get("accepted", walk_summary.get("accepted", 1)))),
        "strategy_strength": summary.get("strategy_strength", walk_summary.get("strategy_strength", "not_checked")),
        "robustness_gate_status": summary.get("robustness_gate_status", ""),
        "profitable_but_fragile": int(float_value(summary.get("profitable_but_fragile", 0))),
        "robustness_failed_checks": summary.get("robustness_failed_checks", ""),
        "ranking_trade_score_top_1pct_avg_net_return": float_value(summary.get("ranking_trade_score_top_1pct_avg_net_return", 0.0)),
        "ranking_trade_score_top_5pct_avg_net_return": float_value(summary.get("ranking_trade_score_top_5pct_avg_net_return", 0.0)),
        "ranking_trade_score_top_decile_avg_net_return": float_value(summary.get("ranking_trade_score_top_decile_avg_net_return", 0.0)),
        "ranking_ranker_score_top_1pct_avg_net_return": float_value(summary.get("ranking_ranker_score_top_1pct_avg_net_return", 0.0)),
        "ranking_ranker_score_top_5pct_avg_net_return": float_value(summary.get("ranking_ranker_score_top_5pct_avg_net_return", 0.0)),
        "ranking_ranker_score_top_decile_avg_net_return": float_value(summary.get("ranking_ranker_score_top_decile_avg_net_return", 0.0)),
        "ranking_trade_score_net_return_monotonicity": float_value(summary.get("ranking_trade_score_net_return_monotonicity", 0.0)),
        "ranking_ranker_score_net_return_monotonicity": float_value(summary.get("ranking_ranker_score_net_return_monotonicity", 0.0)),
        "ranking_trade_score_executed_top_symbol_share": float_value(summary.get("ranking_trade_score_executed_top_symbol_share", 0.0)),
        "ranking_trade_score_executed_top_month_share": float_value(summary.get("ranking_trade_score_executed_top_month_share", 0.0)),
        "ranking_ranker_score_executed_top_symbol_share": float_value(summary.get("ranking_ranker_score_executed_top_symbol_share", 0.0)),
        "ranking_ranker_score_executed_top_month_share": float_value(summary.get("ranking_ranker_score_executed_top_month_share", 0.0)),
        "threshold_diagnostics_primary_rejection": summary.get("threshold_diagnostics_primary_rejection", ""),
        "threshold_diagnostics_primary_rejection_count": int(float_value(summary.get("threshold_diagnostics_primary_rejection_count", 0))),
        "threshold_diagnostics_best_avg_net_return": float_value(summary.get("threshold_diagnostics_best_avg_net_return", 0.0)),
        "threshold_diagnostics_best_avg_net_return_trades": int(float_value(summary.get("threshold_diagnostics_best_avg_net_return_trades", 0))),
        "threshold_diagnostics_best_top_decile_net_return": float_value(summary.get("threshold_diagnostics_best_top_decile_net_return", 0.0)),
        "threshold_diagnostics_near_miss_count": int(float_value(summary.get("threshold_diagnostics_near_miss_count", 0))),
        "threshold_diagnostics_near_miss_ignored_flags": summary.get("threshold_diagnostics_near_miss_ignored_flags", ""),
        "threshold_diagnostics_best_near_miss_source_split": summary.get("threshold_diagnostics_best_near_miss_source_split", ""),
        "threshold_diagnostics_best_near_miss_fold_index": int(float_value(summary.get("threshold_diagnostics_best_near_miss_fold_index", 0))),
        "threshold_diagnostics_best_near_miss_threshold": float_value(summary.get("threshold_diagnostics_best_near_miss_threshold", 0.0)),
        "threshold_diagnostics_best_near_miss_trades": int(float_value(summary.get("threshold_diagnostics_best_near_miss_trades", 0))),
        "threshold_diagnostics_best_near_miss_avg_net_return": float_value(summary.get("threshold_diagnostics_best_near_miss_avg_net_return", 0.0)),
        "threshold_diagnostics_best_near_miss_top_1pct_net_return": float_value(summary.get("threshold_diagnostics_best_near_miss_top_1pct_net_return", 0.0)),
        "threshold_diagnostics_best_near_miss_top_decile_net_return": float_value(summary.get("threshold_diagnostics_best_near_miss_top_decile_net_return", 0.0)),
        "threshold_diagnostics_best_near_miss_top1_concentration": float_value(summary.get("threshold_diagnostics_best_near_miss_top1_concentration", 0.0)),
        "threshold_diagnostics_best_near_miss_top3_concentration": float_value(summary.get("threshold_diagnostics_best_near_miss_top3_concentration", 0.0)),
        "threshold_diagnostics_best_near_miss_trade_top1_concentration": float_value(summary.get("threshold_diagnostics_best_near_miss_trade_top1_concentration", 0.0)),
        "threshold_diagnostics_best_near_miss_rejection_flags": summary.get("threshold_diagnostics_best_near_miss_rejection_flags", ""),
        "max_rss_gb_observed": float_value(summary.get("memory_settings", {}).get("max_rss_gb_observed", 0.0)),
        "run_exit_code": run_exit_code,
    }


def write_result(path, record):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def run_experiments(args):
    experiments = build_experiment_grid_for_profile(args.profile, args.full_grid, args.max_runs)
    if args.parallel != 1:
        print("Warning: --parallel {} is not enabled yet; running sequentially for RAM safety.".format(args.parallel), flush=True)
    print("Running {} experiments".format(len(experiments)), flush=True)
    pipeline_dir = os.path.dirname(os.path.abspath(
        args.pipeline_script if os.path.isabs(args.pipeline_script) else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.pipeline_script)
    ))
    if not args.results_root:
        args.results_root = os.path.join(
            pipeline_dir,
            "results",
            "experiments_{}".format(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")),
        )
    os.makedirs(args.results_root, exist_ok=True)
    if not os.path.isabs(args.output):
        args.output = os.path.join(args.results_root, os.path.basename(args.output))
    for index, experiment in enumerate(experiments, 1):
        command = build_command(args, experiment)
        run_name = experiment_name(experiment)
        run_dir = os.path.join(args.results_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        summary_path = os.path.join(run_dir, "kline_growth_run_summary.json")
        command.extend([
            "--results-dir", run_dir,
            "--run-summary-out", summary_path,
        ])
        if args.resume and os.path.exists(summary_path):
            try:
                summary = load_summary(summary_path)
            except Exception:
                summary = None
            if summary:
                print("[{}/{}] resume skip {}".format(index, len(experiments), run_name), flush=True)
                write_result(args.output, summary_record(experiment, summary, 0))
                continue
        print("[{}/{}] {}".format(index, len(experiments), " ".join(command)), flush=True)
        completed = subprocess.run(command, cwd=pipeline_dir)
        if not os.path.exists(summary_path):
            raise RuntimeError("expected summary output at {}".format(summary_path))
        summary = load_summary(summary_path)
        write_result(args.output, summary_record(experiment, summary, completed.returncode))
        if completed.returncode != 0:
            print("Experiment {} exited with code {}".format(run_name, completed.returncode), flush=True)


def main(argv):
    args = parse_args(argv)
    run_experiments(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
