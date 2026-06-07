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
    "trade_score",
    "regression_calibration",
    "regression_target",
    "hybrid_score_mode",
    "hybrid_uncertainty_method",
    "hybrid_uncertainty_penalty",
    "dynamic_hybrid_thresholds",
    "meta_filter",
    "meta_filter_min_probability",
    "ensemble_windows",
    "ev_safety_margin",
    "min_selected_threshold",
    "min_validation_precision",
    "top_k_per_minute",
    "max_trades_per_period",
    "max_validation_trades",
    "threshold_drawdown_penalty",
    "threshold_trade_count_penalty",
    "target_validation_trades",
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
    "max_rss_gb_observed",
    "run_exit_code",
]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run same-spec experiment grids against gbdt_pipeline.py")
    parser.add_argument("--profile", choices=["7.8gb", "7.8gb-overtrade-check", "hybrid-calibration", "hybrid-risk-adjusted", "hybrid-meta-filter", "hybrid-ensemble-small"], default="7.8gb")
    parser.add_argument("--max-runs", type=int, default=6)
    parser.add_argument("--full-grid", action="store_true")
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
    if profile not in ("7.8gb", "7.8gb-overtrade-check", "hybrid-calibration", "hybrid-risk-adjusted", "hybrid-meta-filter", "hybrid-ensemble-small"):
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


def build_command(args, experiment):
    pipeline_path = args.pipeline_script
    if not os.path.isabs(pipeline_path):
        pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pipeline_path)
    command = [args.python, pipeline_path]
    command.extend(same_spec_profile_args(args.profile))
    command.extend(COMMON_TRADING_ARGS)
    command.append("--cache-only")
    if not experiment.get("walk_forward", True):
        command = [value for value in command if value != "--walk-forward"]
    command.extend([
        "--objective-mode", experiment["objective_mode"],
        "--trade-selection", "topk_score",
        "--trade-score", experiment["trade_score"],
        "--top-k-per-minute", str(experiment["top_k_per_minute"]),
        "--max-trades-per-period", str(experiment.get("max_trades_per_period", 10)),
        "--max-validation-trades", str(experiment.get("max_validation_trades", 250)),
        "--threshold-drawdown-penalty", str(experiment.get("threshold_drawdown_penalty", 0.0)),
        "--threshold-trade-count-penalty", str(experiment.get("threshold_trade_count_penalty", 0.0)),
        "--target-validation-trades", str(experiment.get("target_validation_trades", 0)),
        "--acceptance-tier", args.acceptance_tier,
        "--run-summary-out", args.summary_path,
        "--regression-calibration", experiment.get("regression_calibration", "none"),
        "--regression-target", experiment.get("regression_target", "trade_return"),
        "--hybrid-score-mode", experiment.get("hybrid_score_mode", "basic"),
        "--hybrid-uncertainty-method", experiment.get("hybrid_uncertainty_method", "none"),
        "--hybrid-uncertainty-penalty", str(experiment.get("hybrid_uncertainty_penalty", 0.0)),
        "--dynamic-hybrid-thresholds", experiment.get("dynamic_hybrid_thresholds", "none"),
        "--meta-filter", experiment.get("meta_filter", "none"),
        "--meta-filter-min-probability", str(experiment.get("meta_filter_min_probability", 0.5)),
    ])
    if experiment.get("ensemble_windows", ""):
        command.extend(["--ensemble-windows", experiment["ensemble_windows"]])
    if experiment["objective_mode"] == "classification":
        command.extend([
            "--threshold-objective", "ev",
            "--calibration", "platt",
            "--calibration-max-rows", "500000",
            "--ev-safety-margin", str(experiment["ev_safety_margin"]),
            "--min-selected-threshold", str(experiment["min_selected_threshold"]),
            "--min-validation-precision", str(experiment["min_validation_precision"]),
        ])
    elif experiment["objective_mode"] == "return_regression":
        command.extend([
            "--min-predicted-net-return", str(experiment["min_predicted_net_return"]),
        ])
    else:
        command.extend([
            "--calibration", "platt",
            "--calibration-max-rows", "500000",
            "--hybrid-min-score", str(experiment["hybrid_min_score"]),
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
        "trade_score": experiment["trade_score"],
        "regression_calibration": experiment.get("regression_calibration", "none"),
        "regression_target": experiment.get("regression_target", "trade_return"),
        "hybrid_score_mode": experiment.get("hybrid_score_mode", "basic"),
        "hybrid_uncertainty_method": experiment.get("hybrid_uncertainty_method", "none"),
        "hybrid_uncertainty_penalty": experiment.get("hybrid_uncertainty_penalty", 0.0),
        "dynamic_hybrid_thresholds": experiment.get("dynamic_hybrid_thresholds", "none"),
        "meta_filter": experiment.get("meta_filter", "none"),
        "meta_filter_min_probability": experiment.get("meta_filter_min_probability", 0.5),
        "ensemble_windows": experiment.get("ensemble_windows", ""),
        "ev_safety_margin": experiment["ev_safety_margin"],
        "min_selected_threshold": experiment["min_selected_threshold"],
        "min_validation_precision": experiment["min_validation_precision"],
        "top_k_per_minute": experiment["top_k_per_minute"],
        "max_trades_per_period": experiment.get("max_trades_per_period", 10),
        "max_validation_trades": experiment.get("max_validation_trades", 250),
        "threshold_drawdown_penalty": experiment.get("threshold_drawdown_penalty", 0.0),
        "threshold_trade_count_penalty": experiment.get("threshold_trade_count_penalty", 0.0),
        "target_validation_trades": experiment.get("target_validation_trades", 0),
        "min_predicted_net_return": experiment["min_predicted_net_return"],
        "hybrid_min_score": experiment["hybrid_min_score"],
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
        "accepted": int(float_value(walk_summary.get("accepted", 1))),
        "strategy_strength": walk_summary.get("strategy_strength", "not_checked"),
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
