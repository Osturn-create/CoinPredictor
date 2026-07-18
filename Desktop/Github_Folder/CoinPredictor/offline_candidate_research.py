#!/usr/bin/env python3
"""Offline candidate reselection research for candidate-rich GBDT artifacts.

This module reads saved candidate-only prediction CSVs, extracts eligible
candidate rows, and tests selection/ranking variants without retraining any
model. Older ``--prediction-output-mode all`` artifacts are still accepted as
a fallback.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics
import time

import candidate_artifacts
import created_file_inventory
import research_diagnostics as rd


CANDIDATE_FIELDS = [
    "_source",
    "_row_index",
    "_fold_index",
    "_fold_split",
    "_candidate_id",
    "_stable_row_id",
    "_artifact_fold_id",
    "_artifact_row_scope",
    "symbol",
    "month",
    "month_index",
    "open_time",
    "decision_bucket_id",
    "execution_bucket_id",
    "period_bucket_id",
    "label",
    "probability",
    "calibrated_probability",
    "hybrid_min_probability",
    "raw_predicted_trade_return",
    "predicted_trade_return",
    "calibrated_predicted_trade_return",
    "predicted_net_return",
    "predicted_return_uncertainty",
    "ranker_score",
    "ranker_selection_score",
    "ranker_utility_score",
    "base_hybrid_score",
    "hybrid_score",
    "expected_value",
    "trade_score",
    "trade_score_name",
    "selected_threshold",
    "effective_hybrid_min_score",
    "raw_signal",
    "predicted",
    "position_size",
    "dynamic_trade_return",
    "fixed_horizon_trade_return",
    "forward_return",
    "trade_return",
    "selection_rank",
    "rank_within_decision_bucket",
    "candidate_count_within_decision_bucket",
    "rank_within_execution_bucket",
    "candidate_count_within_execution_bucket",
    "rank_within_period_bucket",
    "candidate_count_within_period_bucket",
    "configured_top_k",
    "selected_by_topk",
    "selected_by_threshold",
    "selected_by_score_edge",
    "selected_by_symbol_filter",
    "candidate_serialization_stage",
    "rejection_stage",
    "rejection_reason",
    "trade_day",
    "exit_minutes",
    "exit_reason",
]


EXPERIMENT_ORDER = [
    "baseline_executed_trades",
    "baseline_selected_by_topk",
    "score_replay_trade_score",
    "cross_sectional_score_zscore",
    "cross_sectional_score_rank",
    "exposure_aware_trade_score_penalty",
    "exposure_aware_trade_share_cap_35pct",
    "calibration_plus_normalization",
    "calibration_plus_concentration_control",
    "score_edge_disabled",
    "symbol_filter_disabled",
    "score_edge_and_symbol_filter_disabled",
]


BUCKET_AUDIT_FIELDS = [
    "source",
    "bucket_scope",
    "bucket_column",
    "configured_top_k",
    "candidate_rows",
    "selected_rows",
    "executed_rows",
    "bucket_count",
    "bucket_candidate_count_p50",
    "bucket_candidate_count_p90",
    "bucket_candidate_count_max",
    "buckets_at_or_below_top_k",
    "bucket_share_at_or_below_top_k",
    "candidate_share_at_or_below_top_k",
    "selected_rows_in_replaceable_buckets",
    "selected_replaceable_share",
    "profitable_nonselected_candidate_rows",
    "buckets_with_profitable_nonselected_candidates",
    "profitable_nonselected_bucket_share",
    "selected_bucket_candidate_count_p50",
    "selected_bucket_candidate_count_p90",
    "selected_bucket_candidate_count_max",
]


RESEARCH_DECISION_FIELDS = [
    "decision",
    "promoted_experiment",
    "best_walkforward_experiment",
    "best_walkforward_return",
    "baseline_walkforward_return",
    "selection_max_trades",
    "selection_max_active_folds",
    "confirmation_max_trades",
    "confirmation_max_active_folds",
    "exact_execution_selected_replaceable_share",
    "best_double_cost_return",
    "reasons",
    "recommended_next_step",
]


MIN_SELECTION_TRADES_FOR_TUNING = 100
MIN_SELECTION_ACTIVE_FOLDS_FOR_TUNING = 3
MIN_EXACT_REPLACEABLE_SHARE_FOR_RESELECTION = 0.10


CAPABILITY_MATRIX = {
    "trades_only": {
        "executed_trade_diagnostics": "supported",
        "true_topk_reselection": "unsupported",
        "rejected_candidate_analysis": "unsupported",
        "cross_sectional_normalization": "unsupported",
        "exposure_aware_reselection": "unsupported",
        "score_edge_ablation": "unsupported",
        "symbol_filter_ablation": "unsupported",
    },
    "candidate": {
        "executed_trade_diagnostics": "supported",
        "true_topk_reselection": "supported",
        "rejected_candidate_analysis": "supported",
        "cross_sectional_normalization": "supported",
        "exposure_aware_reselection": "supported",
        "score_edge_ablation": "unsupported",
        "symbol_filter_ablation": "unsupported",
    },
    "pre_filter_candidate": {
        "executed_trade_diagnostics": "supported",
        "true_topk_reselection": "supported",
        "rejected_candidate_analysis": "supported",
        "cross_sectional_normalization": "supported",
        "exposure_aware_reselection": "supported",
        "score_edge_ablation": "supported",
        "symbol_filter_ablation": "supported",
    },
    "all_rows": {
        "executed_trade_diagnostics": "supported",
        "true_topk_reselection": "supported",
        "rejected_candidate_analysis": "supported",
        "cross_sectional_normalization": "supported",
        "exposure_aware_reselection": "supported",
        "score_edge_ablation": "supported",
        "symbol_filter_ablation": "supported",
        "full_candidate_generation_recall": "supported",
    },
}


def is_truthy(value) -> bool:
    return abs(rd.safe_float(value, 0.0)) > 1e-12


def safe_int_text(value, default=0) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return default


def artifact_fold_index(value, default=0) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    direct = safe_int_text(text, None)
    if direct is not None:
        return direct
    match = re.search(r"(?:^|[^0-9])fold[_\-\s]*(\d+)$", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|[^0-9])(\d+)$", text)
    if match:
        return int(match.group(1))
    return default


def row_identity(row) -> str:
    candidate_id = str(row.get("_candidate_id") or row.get("candidate_id") or "").strip()
    if candidate_id:
        return "candidate:{}".format(candidate_id)
    stable_row_id = str(row.get("_stable_row_id") or row.get("stable_row_id") or "").strip()
    if stable_row_id:
        return "stable:{}".format(stable_row_id)
    return "{}:{}:{}".format(
        row.get("_source", ""),
        row.get("_fold_index") or row.get("_artifact_fold_id") or row.get("fold_id", ""),
        row.get("_row_index", ""),
    )




def score_value(row, column="trade_score", default=-float("inf")) -> float:
    value = rd.safe_float(row.get(column), default)
    return value if math.isfinite(value) else default


def read_single_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def read_csv_rows(path: Path) -> list[dict]:
    opener = candidate_artifacts.open_candidate_text if str(path).endswith(".gz") else None
    if opener is not None:
        with opener(path, "rt", "csv_gzip") as handle:
            return list(csv.DictReader(handle))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def first_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(paths[0])


def optional_first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def candidate_artifact_snapshot_stale(rows: list[dict], source: str, row_scope: str = "all") -> bool:
    if not rows:
        return False
    first = rows[0]
    existing_scope = str(first.get("_artifact_row_scope") or "all")
    if existing_scope != row_scope:
        return True
    if "_candidate_id" not in first or "_stable_row_id" not in first or "_artifact_fold_id" not in first:
        return True
    required_artifact_fields = (
        "decision_bucket_id",
        "candidate_count_within_decision_bucket",
        "configured_top_k",
    )
    if any(field not in first for field in required_artifact_fields):
        return True
    if source == "walkforward" and not any(rd.safe_int(row.get("_fold_index")) > 0 for row in rows):
        return True
    return False


def write_offline_research_created_files_inventory(out_dir: Path) -> dict:
    files = [
        "candidate_snapshot_fixed.csv",
        "candidate_snapshot_walkforward.csv",
        "candidate_snapshot_summary.csv",
        "offline_experiment_matrix.csv",
        "offline_cost_stress.csv",
        "offline_leave_one_symbol_out.csv",
        "offline_leave_one_fold_out.csv",
        "offline_walkforward_fold_metrics.csv",
        "offline_research_split_matrix.csv",
        "offline_research_decision.csv",
        "offline_selection_overlap.csv",
        "offline_candidate_bucket_audit.csv",
        "offline_selected_trades.csv",
        "position_size_fallbacks.json",
        "research_report.md",
    ]
    entries = [(Path(name).stem, out_dir / name) for name in files]
    summary = created_file_inventory.write_inventory(
        out_dir / "offline_research_created_files.csv",
        entries,
    )
    if summary.get("path"):
        print("Offline research created-files inventory: {}".format(summary["path"]), flush=True)
    return summary


def walkforward_fold_maps(diagnostic_rows: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    month_to_fold = {}
    fold_to_split = {}
    for row in diagnostic_rows:
        fold_index = rd.safe_int(row.get("fold_index"))
        if fold_index <= 0:
            continue
        split = row.get("split") or "walkforward_fold_{}".format(fold_index)
        fold_to_split[fold_index] = split
        if row.get("test_month"):
            month_to_fold[row["test_month"]] = fold_index
    return month_to_fold, fold_to_split


def compact_candidate(row: dict, source: str, row_index: int, month_to_fold: dict[str, int], fold_to_split: dict[int, str]) -> dict:
    fold_index = 0
    fold_split = ""
    if source == "walkforward":
        fold_index = month_to_fold.get(row.get("month", ""), 0)
        fold_split = fold_to_split.get(fold_index, "")
    compact = {field: row.get(field, "") for field in CANDIDATE_FIELDS if not field.startswith("_")}
    compact.update({
        "_source": source,
        "_row_index": str(row_index),
        "_fold_index": str(fold_index),
        "_fold_split": fold_split,
    })
    return compact


def compact_artifact_candidate(row: dict, source: str, manifest: dict) -> dict:
    artifact_fold_id = row.get("fold_id", "")
    fold_index = artifact_fold_index(artifact_fold_id)
    compact = {field: "" for field in CANDIDATE_FIELDS if not field.startswith("_")}
    compact.update({
        "symbol": row.get("symbol", ""),
        "month": row.get("month", ""),
        "month_index": row.get("month_index", ""),
        "open_time": row.get("open_time", ""),
        "decision_bucket_id": row.get("decision_bucket_id", ""),
        "execution_bucket_id": row.get("execution_bucket_id", ""),
        "period_bucket_id": row.get("period_bucket_id", row.get("decision_bucket_id", "")),
        "label": row.get("label", ""),
        "probability": row.get("raw_probability", ""),
        "calibrated_probability": row.get("calibrated_probability", ""),
        "raw_predicted_trade_return": row.get("raw_predicted_return", ""),
        "predicted_trade_return": row.get("calibrated_predicted_return", ""),
        "calibrated_predicted_trade_return": row.get("calibrated_predicted_return", ""),
        "predicted_return_uncertainty": row.get("uncertainty", ""),
        "ranker_score": row.get("ranker_score", ""),
        "ranker_selection_score": row.get("ranker_selection_score", ""),
        "ranker_utility_score": row.get("ranker_utility_score", ""),
        "hybrid_score": row.get("hybrid_score", ""),
        "expected_value": row.get("expected_value", ""),
        "trade_score": row.get("final_preselection_score", ""),
        "trade_score_name": "artifact_final_preselection_score",
        "selected_threshold": row.get("selected_threshold", ""),
        "raw_signal": row.get("raw_signal", "1"),
        "predicted": row.get("predicted", row.get("executed", "")),
        "position_size": row.get("position_size", ""),
        "dynamic_trade_return": row.get("actual_exit_return", ""),
        "fixed_horizon_trade_return": row.get("trade_return", ""),
        "forward_return": row.get("forward_return", ""),
        "trade_return": row.get("trade_return", ""),
        "selection_rank": row.get("rank_within_decision_bucket", ""),
        "rank_within_decision_bucket": row.get("rank_within_decision_bucket", ""),
        "candidate_count_within_decision_bucket": row.get("candidate_count_within_decision_bucket", ""),
        "rank_within_execution_bucket": row.get("rank_within_execution_bucket", ""),
        "candidate_count_within_execution_bucket": row.get("candidate_count_within_execution_bucket", ""),
        "rank_within_period_bucket": row.get("rank_within_period_bucket", ""),
        "candidate_count_within_period_bucket": row.get("candidate_count_within_period_bucket", ""),
        "configured_top_k": row.get("configured_top_k", ""),
        "selected_by_topk": row.get("selected_by_topk", row.get("selected_by_score_before_execution", "")),
        "selected_by_threshold": row.get("selected_by_threshold", "1"),
        "selected_by_score_edge": row.get("selected_by_score_edge", "1"),
        "selected_by_symbol_filter": row.get("selected_by_symbol_filter", "1"),
        "candidate_serialization_stage": row.get("candidate_serialization_stage", manifest.get("candidate_serialization_stage", "post_selection")),
        "rejection_stage": row.get("rejection_stage", ""),
        "rejection_reason": row.get("rejection_reason", ""),
        "trade_day": row.get("trade_day", ""),
        "exit_minutes": row.get("holding_period_minutes", ""),
        "exit_reason": row.get("exit_reason", ""),
    })
    compact.update({
        "_source": source,
        "_row_index": row.get("row_position", ""),
        "_fold_index": str(fold_index),
        "_fold_split": "walkforward_fold_{}".format(fold_index) if fold_index > 0 else "",
        "_candidate_id": row.get("candidate_id", ""),
        "_stable_row_id": row.get("stable_row_id", ""),
        "_artifact_fold_id": artifact_fold_id,
        "_artifact_row_scope": "",
        "_coverage_level": manifest.get("candidate_serialization_stage", "post_selection"),
    })
    return compact


def artifact_row_relevant_for_research(row: dict, manifest: dict) -> bool:
    stage = str(
        row.get("candidate_serialization_stage")
        or manifest.get("candidate_serialization_stage", "")
        or ""
    )
    if stage == "pre_score_edge_pre_symbol_filter" and is_truthy(row.get("selected_by_threshold")):
        return True
    return any(
        is_truthy(row.get(field))
        for field in (
            "raw_signal",
            "predicted",
            "executed",
            "selected_by_topk",
            "selected_by_score_before_execution",
        )
    )


def extract_candidate_artifact(
    artifact_path: Path,
    snapshot_path: Path,
    source: str,
    force: bool = False,
    require_pre_filter: bool = False,
    row_scope: str = "all",
) -> tuple[list[dict], dict]:
    if row_scope not in {"all", "research_relevant"}:
        raise ValueError("unsupported candidate artifact row scope: {}".format(row_scope))
    manifest = candidate_artifacts.load_manifest(artifact_path)
    capabilities = candidate_artifacts.artifact_capabilities(manifest)
    if capabilities["true_topk_reselection"] != "supported":
        raise RuntimeError(
            "candidate artifact cannot support true reselection: {}".format(capabilities.get("reason", "unknown"))
        )
    if require_pre_filter and (
        capabilities.get("score_edge_ablation") != "supported"
        or capabilities.get("symbol_filter_ablation") != "supported"
    ):
        raise RuntimeError(
            "candidate artifact cannot support pre-filter score-edge/symbol-filter ablation: {}".format(
                capabilities.get("reason", "unknown")
            )
        )
    loaded_from_snapshot = snapshot_path.exists() and not force
    rows = []
    if loaded_from_snapshot:
        rows = read_csv_rows(snapshot_path)
        if candidate_artifact_snapshot_stale(rows, source, row_scope):
            loaded_from_snapshot = False
    rows_scanned = 0
    threshold_rows = 0
    score_edge_rows = 0
    symbol_filter_rows = 0
    retained_rows = 0
    start = time.time()
    if not loaded_from_snapshot:
        rows = []
        for rows_scanned, row in enumerate(candidate_artifacts.read_candidate_rows(artifact_path), 1):
            if is_truthy(row.get("selected_by_threshold")):
                threshold_rows += 1
            if is_truthy(row.get("selected_by_score_edge")):
                score_edge_rows += 1
            if is_truthy(row.get("selected_by_symbol_filter")):
                symbol_filter_rows += 1
            if row_scope == "research_relevant" and not artifact_row_relevant_for_research(row, manifest):
                continue
            compact = compact_artifact_candidate(row, source, manifest)
            compact["_artifact_row_scope"] = row_scope
            rows.append(compact)
        retained_rows = len(rows)
        write_csv(snapshot_path, rows, CANDIDATE_FIELDS)
    else:
        rows_scanned = len(rows)
        threshold_rows = sum(1 for row in rows if is_truthy(row.get("selected_by_threshold")))
        score_edge_rows = sum(1 for row in rows if is_truthy(row.get("selected_by_score_edge")))
        symbol_filter_rows = sum(1 for row in rows if is_truthy(row.get("selected_by_symbol_filter")))
        retained_rows = len(rows)
    stats = {
        "source": source,
        "artifact_type": "candidate",
        "coverage_level": capabilities["coverage_level"],
        "artifact_row_scope": row_scope,
        "loaded_from_snapshot": 1 if loaded_from_snapshot else 0,
        "snapshot_path": str(snapshot_path),
        "prediction_path": str(artifact_path),
        "manifest_row_count": manifest.get("row_count", ""),
        "rows_scanned": rows_scanned,
        "candidate_rows": len(rows),
        "artifact_rows_retained": retained_rows,
        "threshold_rows": threshold_rows,
        "score_edge_rows": score_edge_rows,
        "symbol_filter_rows": symbol_filter_rows,
        "raw_signal_rows": sum(1 for row in rows if is_truthy(row.get("raw_signal"))),
        "predicted_rows": sum(1 for row in rows if is_truthy(row.get("predicted"))),
        "selected_rows": sum(1 for row in rows if is_truthy(row.get("selected_by_topk"))),
        "elapsed_seconds": round(time.time() - start, 3),
        "manifest_path": str(artifact_path) + ".manifest.json",
        **capabilities,
    }
    return rows, stats


def extract_candidates(
    prediction_path: Path,
    snapshot_path: Path,
    source: str,
    month_to_fold: dict[str, int],
    fold_to_split: dict[int, str],
    force: bool = False,
) -> tuple[list[dict], dict]:
    if snapshot_path.exists() and not force:
        rows = read_csv_rows(snapshot_path)
        stats = {
            "source": source,
            "artifact_type": "legacy_prediction_csv_snapshot",
            "coverage_level": "legacy_candidate_csv",
            "loaded_from_snapshot": 1,
            "snapshot_path": str(snapshot_path),
            "candidate_rows": len(rows),
            "raw_signal_rows": sum(1 for row in rows if is_truthy(row.get("raw_signal"))),
            "predicted_rows": sum(1 for row in rows if is_truthy(row.get("predicted"))),
            "selected_rows": sum(1 for row in rows if is_truthy(row.get("selected_by_topk"))),
        }
        return rows, stats

    start = time.time()
    rows = []
    scanned = 0
    raw_signal_rows = 0
    predicted_rows = 0
    selected_rows = 0
    nonselected_zero_position_rows = 0
    with prediction_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for scanned, row in enumerate(reader, 1):
            raw = is_truthy(row.get("raw_signal"))
            predicted = is_truthy(row.get("predicted"))
            selected = is_truthy(row.get("selected_by_topk"))
            if raw:
                raw_signal_rows += 1
            if predicted:
                predicted_rows += 1
            if selected:
                selected_rows += 1
            if raw or predicted or selected:
                if not selected and rd.safe_float(row.get("position_size")) <= 0.0:
                    nonselected_zero_position_rows += 1
                rows.append(compact_candidate(row, source, scanned, month_to_fold, fold_to_split))
    write_csv(snapshot_path, rows, CANDIDATE_FIELDS)
    stats = {
        "source": source,
        "artifact_type": "legacy_prediction_csv",
        "coverage_level": "legacy_candidate_csv" if "candidate" in prediction_path.name else "all_rows",
        "loaded_from_snapshot": 0,
        "snapshot_path": str(snapshot_path),
        "prediction_path": str(prediction_path),
        "rows_scanned": scanned,
        "candidate_rows": len(rows),
        "raw_signal_rows": raw_signal_rows,
        "predicted_rows": predicted_rows,
        "selected_rows": selected_rows,
        "nonselected_candidate_zero_position_rows": nonselected_zero_position_rows,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    return rows, stats


def baseline_selected(rows: list[dict]) -> list[dict]:
    selected = [dict(row) for row in rows if is_truthy(row.get("selected_by_topk"))]
    if selected:
        return selected
    return [dict(row) for row in rows if is_truthy(row.get("predicted"))]


def baseline_executed(rows: list[dict]) -> list[dict]:
    executed = [dict(row) for row in rows if is_truthy(row.get("predicted"))]
    if executed:
        return executed
    return baseline_selected(rows)


def has_prefilter_candidate_metadata(rows: list[dict]) -> bool:
    return any(
        str(row.get("candidate_serialization_stage") or "").strip() == "pre_score_edge_pre_symbol_filter"
        for row in rows
    )


def candidate_pool(
    rows: list[dict],
    score_edge_filter: bool = True,
    symbol_filter: bool = True,
) -> list[dict]:
    if has_prefilter_candidate_metadata(rows):
        pool = []
        for row in rows:
            if not is_truthy(row.get("selected_by_threshold")):
                continue
            if score_edge_filter and not is_truthy(row.get("selected_by_score_edge")):
                continue
            if symbol_filter and not is_truthy(row.get("selected_by_symbol_filter")):
                continue
            pool.append(dict(row))
        return pool
    pool = [row for row in rows if is_truthy(row.get("raw_signal"))]
    if pool:
        return [dict(row) for row in pool]
    return [dict(row) for row in rows if is_truthy(row.get("predicted")) or is_truthy(row.get("selected_by_topk"))]


def selected_position_fallbacks(all_rows: list[dict]) -> dict[tuple[str, int | str], float]:
    grouped = {}
    for row in all_rows:
        if not is_truthy(row.get("selected_by_topk")):
            continue
        position = rd.safe_float(row.get("position_size"))
        if position <= 0.0:
            continue
        source = row.get("_source", "")
        fold = rd.safe_int(row.get("_fold_index"))
        grouped.setdefault((source, fold), []).append(position)
        grouped.setdefault((source, "all"), []).append(position)
    fallbacks = {}
    for key, values in grouped.items():
        fallbacks[key] = statistics.median(values)
    return fallbacks


def with_analysis_positions(rows: list[dict], fallbacks: dict[tuple[str, int | str], float]) -> list[dict]:
    materialized = []
    for row in rows:
        copy = dict(row)
        position = rd.safe_float(copy.get("position_size"))
        if position <= 0.0:
            source = copy.get("_source", "")
            fold = rd.safe_int(copy.get("_fold_index"))
            position = fallbacks.get((source, fold), fallbacks.get((source, "all"), rd.DEFAULT_INITIAL_CAPITAL * 0.1))
            copy["position_size"] = "{:.12g}".format(position)
            copy["_position_size_fallback"] = "1"
        else:
            copy["_position_size_fallback"] = "0"
        materialized.append(copy)
    return materialized


def sorted_buckets(rows: list[dict]) -> list[list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get("open_time", ""), []).append(row)
    return [
        grouped[key]
        for key in sorted(grouped, key=lambda value: safe_int_text(value))
    ]


def standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    if stdev <= 1e-12:
        return [0.0 for _ in values]
    return [(value - mean) / stdev for value in values]


def percentile_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    denominator = max(1, len(values) - 1)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and abs(ordered[end][0] - ordered[index][0]) <= 1e-15:
            end += 1
        rank = ((index + end - 1) / 2.0) / denominator
        for _, original_index in ordered[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def score_scale(rows: list[dict]) -> float:
    values = [score_value(row, "trade_score", float("nan")) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 1e-6
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    iqr = rd.quantile(values, 0.75) - rd.quantile(values, 0.25)
    return max(stdev, iqr / 1.349 if iqr > 0.0 else 0.0, abs(statistics.median(values)) * 0.25, 1e-6)


def bucket_scores(bucket: list[dict], mode: str) -> dict[str, float]:
    ids = [row_identity(row) for row in bucket]
    trade_scores = [score_value(row, "trade_score") for row in bucket]
    if mode == "trade_score":
        values = trade_scores
    elif mode == "cross_sectional_zscore":
        values = standardize(trade_scores)
    elif mode == "cross_sectional_rank":
        values = percentile_ranks(trade_scores)
    elif mode == "calibration_plus_normalization":
        columns = [
            "calibrated_probability",
            "calibrated_predicted_trade_return",
            "predicted_net_return",
            "expected_value",
        ]
        weights = [0.35, 0.25, 0.25, 0.15]
        normalized_columns = []
        for column in columns:
            normalized_columns.append(standardize([score_value(row, column, 0.0) for row in bucket]))
        values = []
        for index in range(len(bucket)):
            values.append(sum(weight * normalized[index] for weight, normalized in zip(weights, normalized_columns)))
    else:
        raise ValueError("Unknown score mode: {}".format(mode))
    return dict(zip(ids, values))


def select_topk(
    rows: list[dict],
    score_mode: str,
    top_k: int = 2,
    top_k_per_symbol: int = 1,
    exposure_penalty_strength: float = 0.0,
    max_trade_share: float = 0.0,
    cap_warmup_trades: int = 50,
) -> list[dict]:
    selected = []
    selected_counts = {}
    selected_total = 0
    scale = score_scale(rows)
    for bucket in sorted_buckets(rows):
        scores = bucket_scores(bucket, score_mode)

        def adjusted_score(row: dict) -> float:
            value = scores.get(row_identity(row), -float("inf"))
            if exposure_penalty_strength > 0.0 and selected_total > 0:
                share = selected_counts.get(row.get("symbol", ""), 0) / float(selected_total)
                value -= exposure_penalty_strength * scale * share
            return value

        ranked = sorted(
            bucket,
            key=lambda row: (
                -adjusted_score(row),
                -score_value(row, "trade_score"),
                -score_value(row, "hybrid_score"),
                row.get("symbol", ""),
                safe_int_text(row.get("_row_index")),
            ),
        )
        bucket_symbol_counts = {}
        chosen_in_bucket = 0
        for row in ranked:
            symbol = row.get("symbol", "")
            if top_k_per_symbol > 0 and bucket_symbol_counts.get(symbol, 0) >= top_k_per_symbol:
                continue
            if max_trade_share > 0.0 and selected_total >= cap_warmup_trades:
                projected_share = (selected_counts.get(symbol, 0) + 1) / float(selected_total + 1)
                if projected_share > max_trade_share:
                    continue
            copy = dict(row)
            copy["_offline_score"] = "{:.12g}".format(adjusted_score(row))
            copy["_offline_score_mode"] = score_mode
            selected.append(copy)
            selected_counts[symbol] = selected_counts.get(symbol, 0) + 1
            selected_total += 1
            bucket_symbol_counts[symbol] = bucket_symbol_counts.get(symbol, 0) + 1
            chosen_in_bucket += 1
            if top_k > 0 and chosen_in_bucket >= top_k:
                break
    return selected


def selected_sets(rows: list[dict], fallbacks: dict[tuple[str, int | str], float]) -> dict[str, list[dict]]:
    pool = candidate_pool(rows)
    score_edge_disabled_pool = candidate_pool(rows, score_edge_filter=False)
    symbol_filter_disabled_pool = candidate_pool(rows, symbol_filter=False)
    all_filters_disabled_pool = candidate_pool(rows, score_edge_filter=False, symbol_filter=False)
    experiments = {
        "baseline_executed_trades": baseline_executed(rows),
        "baseline_selected_by_topk": baseline_selected(rows),
        "score_replay_trade_score": select_topk(pool, "trade_score"),
        "cross_sectional_score_zscore": select_topk(pool, "cross_sectional_zscore"),
        "cross_sectional_score_rank": select_topk(pool, "cross_sectional_rank"),
        "exposure_aware_trade_score_penalty": select_topk(
            pool,
            "trade_score",
            exposure_penalty_strength=0.75,
        ),
        "exposure_aware_trade_share_cap_35pct": select_topk(
            pool,
            "trade_score",
            exposure_penalty_strength=0.35,
            max_trade_share=0.35,
        ),
        "calibration_plus_normalization": select_topk(pool, "calibration_plus_normalization"),
        "calibration_plus_concentration_control": select_topk(
            pool,
            "calibration_plus_normalization",
            exposure_penalty_strength=0.50,
            max_trade_share=0.35,
        ),
        "score_edge_disabled": select_topk(score_edge_disabled_pool, "trade_score"),
        "symbol_filter_disabled": select_topk(symbol_filter_disabled_pool, "trade_score"),
        "score_edge_and_symbol_filter_disabled": select_topk(all_filters_disabled_pool, "trade_score"),
    }
    return {name: with_analysis_positions(experiment_rows, fallbacks) for name, experiment_rows in experiments.items()}


def source_denominator(source: str, total_folds: int) -> int:
    return total_folds if source == "walkforward" else 1


def top_symbol_summary(rows: list[dict]) -> dict:
    exposure = rd.symbol_exposure(rows)
    top = exposure[0] if exposure else {}
    return {
        "top_symbol": top.get("symbol", ""),
        "top_symbol_net_profit": top.get("net_profit", 0.0),
        "top_symbol_profit_share": max((item["net_profit_share"] for item in exposure), default=0.0),
        "top_symbol_trade_share": max((item["trade_share"] for item in exposure), default=0.0),
        "symbol_count": len(exposure),
    }


def experiment_metrics(
    source: str,
    experiment: str,
    rows: list[dict],
    total_folds: int,
    saved_reference: dict,
) -> dict:
    metrics = rd.portfolio_metrics(rows, denominator_folds=source_denominator(source, total_folds))
    metrics.update(top_symbol_summary(rows))
    fallback_count = sum(1 for row in rows if is_truthy(row.get("_position_size_fallback")))
    metrics.update({
        "source": source,
        "experiment": experiment,
        "analysis_position_fallback_rows": fallback_count,
        "analysis_position_fallback_share": fallback_count / len(rows) if rows else 0.0,
        "saved_reference_profit": rd.safe_float(saved_reference.get("portfolio_profit")),
        "saved_reference_return": rd.safe_float(saved_reference.get("portfolio_return")),
        "saved_reference_trades": rd.safe_float(saved_reference.get("predicted_trades")),
    })
    metrics["profit_vs_saved_reference"] = metrics["portfolio_profit"] - metrics["saved_reference_profit"]
    metrics["return_vs_saved_reference"] = metrics["portfolio_return"] - metrics["saved_reference_return"]
    metrics["trade_count_vs_saved_reference"] = metrics["trade_count"] - metrics["saved_reference_trades"]
    return metrics


def cost_stress(source: str, experiment: str, rows: list[dict], total_folds: int) -> list[dict]:
    output = []
    for row in rd.cost_stress_rows(
        rows,
        source,
        denominator_folds=source_denominator(source, total_folds),
    ):
        copy = dict(row)
        copy["experiment"] = experiment
        output.append(copy)
    return output


def fold_metric_rows(
    rows: list[dict],
    fold_to_split: dict[int, str],
    total_folds: int,
) -> list[dict]:
    by_fold = {}
    for row in rows:
        fold = rd.safe_int(row.get("_fold_index"))
        if fold > 0:
            by_fold.setdefault(fold, []).append(row)
    output = []
    for fold in range(1, total_folds + 1):
        metrics = rd.portfolio_metrics(by_fold.get(fold, []))
        metrics.update({
            "split": fold_to_split.get(fold, "walkforward_fold_{}".format(fold)),
            "fold_index": fold,
            "predicted_trades": metrics["trade_count"],
        })
        output.append(metrics)
    return output


def selection_overlap(source: str, experiment: str, selected: list[dict], baseline: list[dict]) -> dict:
    selected_ids = {row_identity(row) for row in selected}
    baseline_ids = {row_identity(row) for row in baseline}
    overlap = len(selected_ids & baseline_ids)
    return {
        "source": source,
        "experiment": experiment,
        "selected_trades": len(selected_ids),
        "baseline_trades": len(baseline_ids),
        "overlap_trades": overlap,
        "overlap_share_of_experiment": overlap / len(selected_ids) if selected_ids else 0.0,
        "baseline_retention_share": overlap / len(baseline_ids) if baseline_ids else 0.0,
    }


def execution_bucket_from_open_time(row: dict) -> str:
    open_time = safe_int_text(row.get("open_time"), 0)
    if open_time <= 0:
        return ""
    return str(open_time // 60000)


def configured_top_k_for_rows(rows: list[dict]) -> int:
    values = [
        safe_int_text(row.get("configured_top_k"), 0)
        for row in rows
        if safe_int_text(row.get("configured_top_k"), 0) > 0
    ]
    if values:
        return max(values)
    selected_counts = {}
    for row in baseline_selected(rows):
        key = execution_bucket_from_open_time(row)
        if key:
            selected_counts[key] = selected_counts.get(key, 0) + 1
    return max(selected_counts.values(), default=0)


def bucket_values_equal(rows: list[dict], left_column: str, right_fn) -> bool:
    compared = False
    for row in rows:
        left = str(row.get(left_column, "")).strip()
        right = str(right_fn(row)).strip()
        if not left or not right:
            continue
        compared = True
        if left != right:
            return False
    return compared


def candidate_bucket_audit_row(
    source: str,
    rows: list[dict],
    bucket_scope: str,
    bucket_column: str,
    bucket_key,
) -> dict:
    candidates = candidate_pool(rows)
    selected = baseline_selected(rows)
    selected_ids = {row_identity(row) for row in selected}
    executed_rows = [row for row in rows if is_truthy(row.get("predicted"))]
    top_k = configured_top_k_for_rows(rows)
    buckets = {}
    for row in candidates:
        key = str(bucket_key(row)).strip()
        if key:
            buckets.setdefault(key, []).append(row)
    bucket_sizes = [len(bucket) for bucket in buckets.values()]
    selected_bucket_sizes = []
    buckets_at_or_below_top_k = 0
    candidates_at_or_below_top_k = 0
    selected_rows_in_replaceable_buckets = 0
    profitable_nonselected_candidate_rows = 0
    buckets_with_profitable_nonselected_candidates = 0

    for bucket in buckets.values():
        bucket_size = len(bucket)
        if top_k > 0 and bucket_size <= top_k:
            buckets_at_or_below_top_k += 1
            candidates_at_or_below_top_k += bucket_size
        selected_in_bucket = [
            row
            for row in bucket
            if row_identity(row) in selected_ids or is_truthy(row.get("selected_by_topk"))
        ]
        selected_bucket_sizes.extend([bucket_size] * len(selected_in_bucket))
        if top_k > 0 and bucket_size > top_k:
            selected_rows_in_replaceable_buckets += len(selected_in_bucket)
        profitable_nonselected = [
            row
            for row in bucket
            if row_identity(row) not in selected_ids
            and not is_truthy(row.get("selected_by_topk"))
            and rd.net_return(row) > 0.0
        ]
        profitable_nonselected_candidate_rows += len(profitable_nonselected)
        if profitable_nonselected:
            buckets_with_profitable_nonselected_candidates += 1

    return {
        "source": source,
        "bucket_scope": bucket_scope,
        "bucket_column": bucket_column,
        "configured_top_k": top_k,
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "executed_rows": len(executed_rows),
        "bucket_count": len(buckets),
        "bucket_candidate_count_p50": rd.quantile(bucket_sizes, 0.50),
        "bucket_candidate_count_p90": rd.quantile(bucket_sizes, 0.90),
        "bucket_candidate_count_max": max(bucket_sizes) if bucket_sizes else 0,
        "buckets_at_or_below_top_k": buckets_at_or_below_top_k,
        "bucket_share_at_or_below_top_k": buckets_at_or_below_top_k / len(buckets) if buckets else 0.0,
        "candidate_share_at_or_below_top_k": candidates_at_or_below_top_k / len(candidates) if candidates else 0.0,
        "selected_rows_in_replaceable_buckets": selected_rows_in_replaceable_buckets,
        "selected_replaceable_share": selected_rows_in_replaceable_buckets / len(selected) if selected else 0.0,
        "profitable_nonselected_candidate_rows": profitable_nonselected_candidate_rows,
        "buckets_with_profitable_nonselected_candidates": buckets_with_profitable_nonselected_candidates,
        "profitable_nonselected_bucket_share": (
            buckets_with_profitable_nonselected_candidates / len(buckets) if buckets else 0.0
        ),
        "selected_bucket_candidate_count_p50": rd.quantile(selected_bucket_sizes, 0.50),
        "selected_bucket_candidate_count_p90": rd.quantile(selected_bucket_sizes, 0.90),
        "selected_bucket_candidate_count_max": max(selected_bucket_sizes) if selected_bucket_sizes else 0,
    }


def candidate_bucket_audit_rows(source: str, rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    output = [
        candidate_bucket_audit_row(
            source,
            rows,
            "execution_open_time",
            "open_time_minute",
            execution_bucket_from_open_time,
        )
    ]
    if any(str(row.get("decision_bucket_id", "")).strip() for row in rows):
        output.append(
            candidate_bucket_audit_row(
                source,
                rows,
                "artifact_decision_bucket",
                "decision_bucket_id",
                lambda row: row.get("decision_bucket_id", ""),
            )
        )
    if (
        any(str(row.get("period_bucket_id", "")).strip() for row in rows)
        and not bucket_values_equal(rows, "period_bucket_id", lambda row: row.get("decision_bucket_id", ""))
    ):
        output.append(
            candidate_bucket_audit_row(
                source,
                rows,
                "artifact_period_bucket",
                "period_bucket_id",
                lambda row: row.get("period_bucket_id", ""),
            )
        )
    return output


def leave_one_symbols(source: str, experiment: str, rows: list[dict], total_folds: int) -> list[dict]:
    exposure = rd.symbol_exposure(rows)
    symbols = [row["symbol"] for row in exposure[:8]]
    output = []
    for row in rd.leave_one_symbol_out_rows(
        rows,
        symbols,
        source,
        denominator_folds=source_denominator(source, total_folds),
    ):
        copy = dict(row)
        copy["experiment"] = experiment
        output.append(copy)
    return output


def fold_summary(source: str, experiment: str, fold_rows: list[dict]) -> dict:
    if source != "walkforward":
        return {}
    returns = [rd.safe_float(row.get("portfolio_return")) for row in fold_rows]
    profits = [rd.safe_float(row.get("portfolio_profit")) for row in fold_rows]
    positive_profit = sum(max(0.0, value) for value in profits) or 1.0
    return {
        "walkforward_mean_fold_return": statistics.fmean(returns) if returns else 0.0,
        "walkforward_median_fold_return": statistics.median(returns) if returns else 0.0,
        "walkforward_worst_fold_return": min(returns) if returns else 0.0,
        "walkforward_best_fold_return": max(returns) if returns else 0.0,
        "walkforward_total_profit": sum(profits),
        "walkforward_active_folds": sum(1 for row in fold_rows if rd.safe_int(row.get("predicted_trades")) > 0),
        "walkforward_profitable_folds": sum(1 for row in fold_rows if rd.safe_float(row.get("portfolio_return")) > 0.0),
        "walkforward_top_fold_profit_share": max((max(0.0, value) / positive_profit for value in profits), default=0.0),
    }


def parse_fold_spec(text: str, total_folds: int) -> set[int]:
    folds: set[int] = set()
    text = (text or "").strip()
    if not text:
        return folds
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                raise ValueError("invalid descending fold range: {}".format(part))
            folds.update(range(start, end + 1))
        else:
            folds.add(int(part))
    invalid = sorted(fold for fold in folds if fold < 1 or fold > total_folds)
    if invalid:
        raise ValueError("folds out of range 1..{}: {}".format(total_folds, invalid))
    return folds


def build_research_fold_sets(args: argparse.Namespace, total_folds: int) -> dict[str, set[int]]:
    discovery = parse_fold_spec(args.discovery_folds, total_folds)
    selection = parse_fold_spec(args.selection_folds, total_folds)
    confirmation = parse_fold_spec(args.confirmation_folds, total_folds)
    if confirmation and not selection:
        raise ValueError("--selection-folds is required when --confirmation-folds is provided")
    named = [
        ("discovery", discovery),
        ("selection", selection),
        ("confirmation", confirmation),
    ]
    for left_index, (left_name, left) in enumerate(named):
        for right_name, right in named[left_index + 1:]:
            overlap = sorted(left & right)
            if overlap:
                raise ValueError("{} and {} folds overlap: {}".format(left_name, right_name, overlap))
    if not any((discovery, selection, confirmation)):
        return {"diagnostic_all_folds": set(range(1, total_folds + 1))}
    output = {}
    if discovery:
        output["discovery"] = discovery
    if selection:
        output["selection"] = selection
    if confirmation:
        output["confirmation"] = confirmation
    return output


def research_split_matrix_rows(fold_rows: list[dict], fold_sets: dict[str, set[int]]) -> list[dict]:
    output = []
    by_experiment = {}
    for row in fold_rows:
        by_experiment.setdefault(row.get("experiment", ""), []).append(row)
    for experiment, rows in sorted(by_experiment.items()):
        for phase, folds in fold_sets.items():
            phase_rows = [row for row in rows if rd.safe_int(row.get("fold_index")) in folds]
            returns = [rd.safe_float(row.get("portfolio_return")) for row in phase_rows]
            profits = [rd.safe_float(row.get("portfolio_profit")) for row in phase_rows]
            output.append({
                "experiment": experiment,
                "phase": phase,
                "folds": ",".join(str(fold) for fold in sorted(folds)),
                "fold_count": len(phase_rows),
                "trade_count": sum(rd.safe_int(row.get("predicted_trades")) for row in phase_rows),
                "portfolio_profit": sum(profits),
                "portfolio_return": (sum(profits) / (rd.DEFAULT_INITIAL_CAPITAL * max(1, len(phase_rows)))) if phase_rows else 0.0,
                "mean_fold_return": statistics.fmean(returns) if returns else 0.0,
                "median_fold_return": statistics.median(returns) if returns else 0.0,
                "worst_fold_return": min(returns) if returns else 0.0,
                "best_fold_return": max(returns) if returns else 0.0,
                "active_folds": sum(1 for row in phase_rows if rd.safe_int(row.get("predicted_trades")) > 0),
                "profitable_folds": sum(1 for row in phase_rows if rd.safe_float(row.get("portfolio_return")) > 0.0),
            })
    return output


def research_decision_rows(
    experiment_rows: list[dict],
    split_rows: list[dict],
    bucket_audit_rows: list[dict],
    cost_rows: list[dict],
) -> list[dict]:
    wf_rows = [row for row in experiment_rows if row.get("source") == "walkforward"]
    if not wf_rows:
        return []
    best = max(wf_rows, key=lambda row: rd.safe_float(row.get("portfolio_return")))
    baseline = next((row for row in wf_rows if row.get("experiment") == "baseline_executed_trades"), {})
    selection_rows = [row for row in split_rows if row.get("phase") == "selection"]
    confirmation_rows = [row for row in split_rows if row.get("phase") == "confirmation"]
    selection_max_trades = max((rd.safe_int(row.get("trade_count")) for row in selection_rows), default=0)
    selection_max_active_folds = max((rd.safe_int(row.get("active_folds")) for row in selection_rows), default=0)
    confirmation_max_trades = max((rd.safe_int(row.get("trade_count")) for row in confirmation_rows), default=0)
    confirmation_max_active_folds = max((rd.safe_int(row.get("active_folds")) for row in confirmation_rows), default=0)
    exact_bucket = next(
        (
            row for row in bucket_audit_rows
            if row.get("source") == "walkforward" and row.get("bucket_scope") == "execution_open_time"
        ),
        {},
    )
    exact_replaceable_share = rd.safe_float(exact_bucket.get("selected_replaceable_share"))
    double_cost = next(
        (
            row for row in cost_rows
            if row.get("source") == "walkforward"
            and row.get("experiment") == best.get("experiment")
            and row.get("cost_scenario") == "double_cost"
        ),
        {},
    )
    best_double_cost_return = rd.safe_float(double_cost.get("portfolio_return"))
    reasons = []
    if best.get("experiment") == "baseline_executed_trades":
        reasons.append("baseline_best")
    if selection_rows and (
        selection_max_trades < MIN_SELECTION_TRADES_FOR_TUNING
        or selection_max_active_folds < MIN_SELECTION_ACTIVE_FOLDS_FOR_TUNING
    ):
        reasons.append("selection_underpowered")
    elif not selection_rows:
        reasons.append("no_selection_split")
    if exact_bucket and exact_replaceable_share < MIN_EXACT_REPLACEABLE_SHARE_FOR_RESELECTION:
        reasons.append("candidate_reselection_sparse")
    if double_cost and best_double_cost_return < 0.0:
        reasons.append("cost_stress_failed")

    decision = "do_not_promote_offline_tuning" if reasons else "offline_variant_candidate"
    promoted_experiment = "" if decision == "do_not_promote_offline_tuning" else best.get("experiment", "")
    if "cost_stress_failed" in reasons:
        recommended_next_step = (
            "Run an apples-to-apples production walk-forward with "
            "--threshold-cost-stress-multiplier 2.0 and --threshold-cost-stress-weight 1.0; "
            "do not promote offline reselection variants from this pass."
        )
    elif "selection_underpowered" in reasons or "no_selection_split" in reasons:
        recommended_next_step = (
            "Use the offline pass as diagnostics only, then rerun with a selection phase "
            "containing enough active folds and trades before promoting tuning."
        )
    elif "candidate_reselection_sparse" in reasons:
        recommended_next_step = (
            "Avoid offline reselection tuning; improve production candidate generation or "
            "threshold selection instead."
        )
    elif "baseline_best" in reasons:
        recommended_next_step = "Keep the executed baseline; no offline variant beat it."
    else:
        recommended_next_step = "Promote only after a fresh production walk-forward confirmation."

    return [{
        "decision": decision,
        "promoted_experiment": promoted_experiment,
        "best_walkforward_experiment": best.get("experiment", ""),
        "best_walkforward_return": rd.safe_float(best.get("portfolio_return")),
        "baseline_walkforward_return": rd.safe_float(baseline.get("portfolio_return")),
        "selection_max_trades": selection_max_trades,
        "selection_max_active_folds": selection_max_active_folds,
        "confirmation_max_trades": confirmation_max_trades,
        "confirmation_max_active_folds": confirmation_max_active_folds,
        "exact_execution_selected_replaceable_share": exact_replaceable_share,
        "best_double_cost_return": best_double_cost_return,
        "reasons": ";".join(reasons),
        "recommended_next_step": recommended_next_step,
    }]


def build_report(context: dict) -> str:
    experiment_rows = context["experiment_rows"]
    overlap_rows = context["overlap_rows"]
    cost_rows = context["cost_rows"]
    leave_symbol_rows = context["leave_symbol_rows"]
    leave_fold_rows = context["leave_fold_rows"]
    snapshot_stats = context["snapshot_stats"]
    split_rows = context.get("split_rows", [])
    bucket_audit_rows = context.get("bucket_audit_rows", [])
    decision_rows = context.get("decision_rows")
    if decision_rows is None:
        decision_rows = research_decision_rows(experiment_rows, split_rows, bucket_audit_rows, cost_rows)

    by_source_experiment = {
        (row["source"], row["experiment"]): row
        for row in experiment_rows
    }

    def fmt_pct(value):
        return "{:.2%}".format(rd.safe_float(value))

    def fmt_money(value):
        return "${:,.2f}".format(rd.safe_float(value))

    wf_rows = [row for row in experiment_rows if row["source"] == "walkforward"]
    fixed_rows = [row for row in experiment_rows if row["source"] == "fixed"]
    best_wf = max(wf_rows, key=lambda row: rd.safe_float(row.get("portfolio_return")), default={})
    best_fixed = max(fixed_rows, key=lambda row: rd.safe_float(row.get("portfolio_return")), default={})
    executed_wf = by_source_experiment.get(("walkforward", "baseline_executed_trades"), {})
    baseline_wf = by_source_experiment.get(("walkforward", "baseline_selected_by_topk"), {})
    executed_fixed = by_source_experiment.get(("fixed", "baseline_executed_trades"), {})
    baseline_fixed = by_source_experiment.get(("fixed", "baseline_selected_by_topk"), {})
    double_cost_wf = next(
        (
            row for row in cost_rows
            if row.get("source") == "walkforward"
            and row.get("experiment") == best_wf.get("experiment")
            and row.get("cost_scenario") == "double_cost"
        ),
        {},
    )
    worst_leave_symbol = min(
        (
            row for row in leave_symbol_rows
            if row.get("source") == "walkforward"
            and row.get("experiment") == best_wf.get("experiment")
        ),
        key=lambda row: rd.safe_float(row.get("portfolio_return")),
        default={},
    )
    worst_leave_fold = min(
        (
            row for row in leave_fold_rows
            if row.get("experiment") == best_wf.get("experiment")
        ),
        key=lambda row: rd.safe_float(row.get("remaining_mean_fold_return")),
        default={},
    )

    lines = [
        "# Candidate-Rich Offline Research Pass",
        "",
        "Generated: {}".format(dt.datetime.now(dt.timezone.utc).isoformat()),
        "",
        "## Verdict",
        "",
        "This pass used saved candidate prediction artifacts only. It did not retrain, refit, or change the baseline model.",
        "",
        "- Best walk-forward offline variant by mean fold return: `{}` at {} over {} trades.".format(
            best_wf.get("experiment", ""),
            fmt_pct(best_wf.get("portfolio_return")),
            int(rd.safe_float(best_wf.get("trade_count"))),
        ),
        "- Baseline walk-forward replay: {} over {} trades; saved reference was {}.".format(
            fmt_pct(baseline_wf.get("portfolio_return")),
            int(rd.safe_float(baseline_wf.get("trade_count"))),
            fmt_pct(baseline_wf.get("saved_reference_return")),
        ),
        "- Executed-only walk-forward replay: {} over {} trades.".format(
            fmt_pct(executed_wf.get("portfolio_return")),
            int(rd.safe_float(executed_wf.get("trade_count"))),
        ),
        "- Best fixed-test offline variant: `{}` at {}; baseline fixed replay was {}.".format(
            best_fixed.get("experiment", ""),
            fmt_pct(best_fixed.get("portfolio_return")),
            fmt_pct(baseline_fixed.get("portfolio_return")),
        ),
        "- Double-cost stress for the best walk-forward variant lands at {}.".format(
            fmt_pct(double_cost_wf.get("portfolio_return")),
        ),
        "- Removing its weakest key symbol leaves {} mean return; removing the most important fold leaves {} mean fold return.".format(
            fmt_pct(worst_leave_symbol.get("portfolio_return")),
            fmt_pct(worst_leave_fold.get("remaining_mean_fold_return")),
        ),
        "",
        "## Promotion Decision",
        "",
    ]
    if decision_rows:
        decision = decision_rows[0]
        lines.extend([
            "- Decision: `{}`".format(decision.get("decision", "")),
            "- Promoted offline experiment: `{}`".format(decision.get("promoted_experiment", "") or "none"),
            "- Reasons: `{}`".format(decision.get("reasons", "")),
            "- Selection evidence: `{}` max trades across variants, `{}` max active folds.".format(
                int(rd.safe_float(decision.get("selection_max_trades"))),
                int(rd.safe_float(decision.get("selection_max_active_folds"))),
            ),
            "- Exact execution-bucket selected replaceable share: `{}`.".format(
                fmt_pct(decision.get("exact_execution_selected_replaceable_share")),
            ),
            "- Best-variant double-cost return: `{}`.".format(
                fmt_pct(decision.get("best_double_cost_return")),
            ),
            "- Recommended next step: {}".format(decision.get("recommended_next_step", "")),
            "",
        ])
    else:
        lines.extend([
            "- Decision: `diagnostic_only`",
            "- Recommended next step: no walk-forward candidate evidence was available.",
            "",
        ])
    lines.extend([
        "## Candidate Extraction",
        "",
        "| Source | Coverage | Candidate rows | Raw signals | Predicted | Selected | Scanned rows | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in snapshot_stats:
        lines.append(
            "| {source} | {coverage} | {candidate_rows} | {raw_signal_rows} | {predicted_rows} | {selected_rows} | {rows_scanned} | {note} |".format(
                source=row.get("source", ""),
                coverage=row.get("coverage_level", ""),
                candidate_rows=int(rd.safe_float(row.get("candidate_rows"))),
                raw_signal_rows=int(rd.safe_float(row.get("raw_signal_rows"))),
                predicted_rows=int(rd.safe_float(row.get("predicted_rows"))),
                selected_rows=int(rd.safe_float(row.get("selected_rows"))),
                rows_scanned=int(rd.safe_float(row.get("rows_scanned"))),
                note="snapshot cache" if rd.safe_int(row.get("loaded_from_snapshot")) else "fresh stream",
            )
        )
    for row in snapshot_stats:
        selected_rows = rd.safe_float(row.get("selected_rows"))
        predicted_rows = rd.safe_float(row.get("predicted_rows"))
        if selected_rows > 0.0 and abs(selected_rows - predicted_rows) > 1e-9:
            lines.append(
                "- `{}` selected/ranked rows ({}) differ from executed rows ({}); use `baseline_executed_trades` for saved-run fidelity and `baseline_selected_by_topk` only as a pre-execution ranking diagnostic.".format(
                    row.get("source", ""),
                    int(selected_rows),
                    int(predicted_rows),
                )
            )

    lines.extend([
        "",
        "## Artifact Capability Matrix",
        "",
        "| Artifact | Executed diagnostics | True top-k reselection | Rejected candidates | Cross-sectional normalization | Exposure-aware reselection | Score-edge ablation | Symbol-filter ablation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for artifact_name, capabilities in CAPABILITY_MATRIX.items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                artifact_name,
                capabilities.get("executed_trade_diagnostics", ""),
                capabilities.get("true_topk_reselection", ""),
                capabilities.get("rejected_candidate_analysis", ""),
                capabilities.get("cross_sectional_normalization", ""),
                capabilities.get("exposure_aware_reselection", ""),
                capabilities.get("score_edge_ablation", ""),
                capabilities.get("symbol_filter_ablation", ""),
            )
        )

    lines.extend([
        "",
        "## Candidate Bucket Audit",
        "",
        "| Source | Bucket scope | Buckets | Candidate p50 | Candidate p90 | Max candidates | Buckets <= top-k | Selected replaceable | Profitable alt buckets |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in bucket_audit_rows:
        lines.append(
            "| {source} | `{scope}` | {buckets} | {p50:.1f} | {p90:.1f} | {max_count} | {le_topk} | {replaceable} | {profitable} |".format(
                source=row.get("source", ""),
                scope=row.get("bucket_scope", ""),
                buckets=int(rd.safe_float(row.get("bucket_count"))),
                p50=rd.safe_float(row.get("bucket_candidate_count_p50")),
                p90=rd.safe_float(row.get("bucket_candidate_count_p90")),
                max_count=int(rd.safe_float(row.get("bucket_candidate_count_max"))),
                le_topk=fmt_pct(row.get("bucket_share_at_or_below_top_k")),
                replaceable=fmt_pct(row.get("selected_replaceable_share")),
                profitable=fmt_pct(row.get("profitable_nonselected_bucket_share")),
            )
        )
    if bucket_audit_rows:
        wf_execution = next(
            (
                row for row in bucket_audit_rows
                if row.get("source") == "walkforward" and row.get("bucket_scope") == "execution_open_time"
            ),
            {},
        )
        if wf_execution:
            lines.extend([
                "",
                "- Walk-forward exact execution buckets have p50/p90 candidate counts of {:.1f}/{:.1f}; only {} of selected rows are in buckets with more candidates than top-k.".format(
                    rd.safe_float(wf_execution.get("bucket_candidate_count_p50")),
                    rd.safe_float(wf_execution.get("bucket_candidate_count_p90")),
                    fmt_pct(wf_execution.get("selected_replaceable_share")),
                ),
            ])

    lines.extend([
        "",
        "## Experiment Matrix",
        "",
        "| Source | Experiment | Trades | Return | Profit | Top symbol | Top profit share | Top trade share | Fallback sizing | Overlap with baseline |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    overlap_by_key = {(row["source"], row["experiment"]): row for row in overlap_rows}
    for source in ("fixed", "walkforward"):
        for experiment in EXPERIMENT_ORDER:
            row = by_source_experiment.get((source, experiment))
            if not row:
                continue
            overlap = overlap_by_key.get((source, experiment), {})
            lines.append(
                "| {source} | `{experiment}` | {trades} | {ret} | {profit} | {symbol} | {profit_share} | {trade_share} | {fallback} | {overlap} |".format(
                    source=source,
                    experiment=experiment,
                    trades=int(rd.safe_float(row.get("trade_count"))),
                    ret=fmt_pct(row.get("portfolio_return")),
                    profit=fmt_money(row.get("portfolio_profit")),
                    symbol=row.get("top_symbol", ""),
                    profit_share=fmt_pct(row.get("top_symbol_profit_share")),
                    trade_share=fmt_pct(row.get("top_symbol_trade_share")),
                    fallback=fmt_pct(row.get("analysis_position_fallback_share")),
                    overlap=fmt_pct(overlap.get("overlap_share_of_experiment")),
                )
            )

    lines.extend([
        "",
        "## Readout",
        "",
    ])
    normalized_wf = by_source_experiment.get(("walkforward", "cross_sectional_score_zscore"), {})
    rank_wf = by_source_experiment.get(("walkforward", "cross_sectional_score_rank"), {})
    exposure_wf = by_source_experiment.get(("walkforward", "exposure_aware_trade_score_penalty"), {})
    calibration_wf = by_source_experiment.get(("walkforward", "calibration_plus_normalization"), {})
    concentration_wf = by_source_experiment.get(("walkforward", "calibration_plus_concentration_control"), {})
    lines.extend([
        "- Executed-only versus pre-execution baseline differs by {} on walk-forward and {} on fixed-test. Large gaps mean execution constraints, sizing fallbacks, or non-executed ranked rows make offline reselection exploratory rather than production-faithful.".format(
            fmt_pct(rd.safe_float(baseline_wf.get("portfolio_return")) - rd.safe_float(executed_wf.get("portfolio_return"))),
            fmt_pct(rd.safe_float(baseline_fixed.get("portfolio_return")) - rd.safe_float(executed_fixed.get("portfolio_return"))),
        ),
        "- Cross-sectional z-score/rank normalization is a pure within-bucket monotonic transform for top-k selection. If overlap remains near 100%, it is not a lever by itself; it needs either a changed thresholding rule or a non-monotonic calibrated blend.",
        "- Exposure-aware selection changed walk-forward return by {} versus baseline and changed top-symbol trade share from {} to {}.".format(
            fmt_pct(rd.safe_float(exposure_wf.get("portfolio_return")) - rd.safe_float(baseline_wf.get("portfolio_return"))),
            fmt_pct(baseline_wf.get("top_symbol_trade_share")),
            fmt_pct(exposure_wf.get("top_symbol_trade_share")),
        ),
        "- Calibration plus normalization changed walk-forward return by {} versus baseline.".format(
            fmt_pct(rd.safe_float(calibration_wf.get("portfolio_return")) - rd.safe_float(baseline_wf.get("portfolio_return"))),
        ),
        "- Calibration plus concentration control changed walk-forward return by {} versus baseline and left top-symbol profit share at {}.".format(
            fmt_pct(rd.safe_float(concentration_wf.get("portfolio_return")) - rd.safe_float(baseline_wf.get("portfolio_return"))),
            fmt_pct(concentration_wf.get("top_symbol_profit_share")),
        ),
        "",
        "## Research Split",
        "",
    ])
    if split_rows:
        lines.extend([
            "| Phase | Experiment | Folds | Trades | Return | Median fold | Worst fold | Active folds | Profitable folds |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in split_rows:
            lines.append(
                "| {phase} | `{experiment}` | {folds} | {trades} | {ret} | {median} | {worst} | {active} | {profitable} |".format(
                    phase=row.get("phase", ""),
                    experiment=row.get("experiment", ""),
                    folds=row.get("folds", ""),
                    trades=int(rd.safe_float(row.get("trade_count"))),
                    ret=fmt_pct(row.get("portfolio_return")),
                    median=fmt_pct(row.get("median_fold_return")),
                    worst=fmt_pct(row.get("worst_fold_return")),
                    active=int(rd.safe_float(row.get("active_folds"))),
                    profitable=int(rd.safe_float(row.get("profitable_folds"))),
                )
            )
        lines.append("")
        if any(row.get("phase") == "diagnostic_all_folds" for row in split_rows):
            lines.extend([
                "No discovery/selection/confirmation split was provided, so this pass is diagnostic only and should not be used as a promoted out-of-sample selection.",
                "",
            ])
        else:
            lines.extend([
                "Selection and confirmation phases are reported separately; confirmation rows are not used by this script to pick parameter values.",
                "",
            ])
    else:
        lines.extend([
            "No walk-forward split rows were generated.",
            "",
        ])
    lines.extend([
        "## Cost And Leave-One Checks",
        "",
        "| Experiment | Double-cost WF return | Triple-cost WF return | Worst leave-symbol return | Worst leave-fold mean return |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for experiment in EXPERIMENT_ORDER:
        double_cost = next(
            (
                row for row in cost_rows
                if row.get("source") == "walkforward"
                and row.get("experiment") == experiment
                and row.get("cost_scenario") == "double_cost"
            ),
            {},
        )
        triple_cost = next(
            (
                row for row in cost_rows
                if row.get("source") == "walkforward"
                and row.get("experiment") == experiment
                and row.get("cost_scenario") == "triple_cost"
            ),
            {},
        )
        leave_symbol = min(
            (
                row for row in leave_symbol_rows
                if row.get("source") == "walkforward" and row.get("experiment") == experiment
            ),
            key=lambda row: rd.safe_float(row.get("portfolio_return")),
            default={},
        )
        leave_fold = min(
            (row for row in leave_fold_rows if row.get("experiment") == experiment),
            key=lambda row: rd.safe_float(row.get("remaining_mean_fold_return")),
            default={},
        )
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                experiment,
                fmt_pct(double_cost.get("portfolio_return")),
                fmt_pct(triple_cost.get("portfolio_return")),
                fmt_pct(leave_symbol.get("portfolio_return")),
                fmt_pct(leave_fold.get("remaining_mean_fold_return")),
            )
        )

    lines.extend([
        "",
        "## Files",
        "",
        "- `candidate_snapshot_fixed.csv` and `candidate_snapshot_walkforward.csv`: compact extracted candidate rows.",
        "- `offline_experiment_matrix.csv`: main comparison table.",
        "- `offline_cost_stress.csv`: baseline, double-cost, and triple-cost stress per experiment.",
        "- `offline_leave_one_symbol_out.csv`: leave-one-symbol check for top contributors.",
        "- `offline_leave_one_fold_out.csv`: leave-one-fold check for walk-forward experiments.",
        "- `offline_research_split_matrix.csv`: discovery, selection, and confirmation fold summaries when split folds are provided.",
        "- `offline_research_decision.csv`: machine-readable promotion/diagnostic decision for the offline pass.",
        "- `offline_selection_overlap.csv`: selected-row overlap against the saved baseline.",
        "- `offline_candidate_bucket_audit.csv`: exact execution-bucket and artifact-bucket candidate sufficiency checks.",
        "",
        "Note: replacement candidates that were not executed in the original run can lack a saved position size. Those rows use the median saved selected position size for the same source/fold, and the fallback share is shown in the matrix.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fixed-candidate-artifact", type=Path, default=None)
    parser.add_argument("--walk-candidate-artifact", type=Path, default=None)
    parser.add_argument("--force-rescan", action="store_true")
    parser.add_argument("--require-pre-filter-artifacts", action="store_true")
    parser.add_argument(
        "--candidate-artifact-row-scope",
        choices=("all", "research_relevant"),
        default="all",
        help=(
            "Rows to copy into candidate snapshots. 'all' preserves the full legacy "
            "snapshot; 'research_relevant' streams artifacts and keeps threshold-qualified "
            "or executed/top-k rows needed for offline reselection ablations."
        ),
    )
    parser.add_argument("--discovery-folds", default="", help="Comma/range list of walk-forward folds used for diagnosis only, e.g. 1-3")
    parser.add_argument("--selection-folds", default="", help="Comma/range list of walk-forward folds allowed for choosing a candidate configuration")
    parser.add_argument("--confirmation-folds", default="", help="Comma/range list of walk-forward folds reserved for confirmation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_predictions = None if args.fixed_candidate_artifact else optional_first_existing_path([
        run_dir / "kline_growth_predictions_gbdt_candidates.csv",
        run_dir / "kline_growth_predictions_gbdt_all.csv",
    ])
    walk_predictions = None if args.walk_candidate_artifact else optional_first_existing_path([
        run_dir / "kline_growth_predictions_gbdt_walkforward_candidates.csv",
        run_dir / "kline_growth_predictions_gbdt_walkforward_all.csv",
    ])
    fixed_metrics_path = run_dir / "kline_growth_metrics_gbdt.csv"
    walk_metrics_path = run_dir / "kline_growth_walkforward_metrics.csv"
    walk_diagnostics_path = run_dir / "kline_growth_walkforward_diagnostics.csv"
    fixed_available = bool(args.fixed_candidate_artifact or fixed_predictions)
    walk_available = bool(args.walk_candidate_artifact or walk_predictions)
    if not fixed_available and not walk_available:
        raise FileNotFoundError(
            "no fixed or walk-forward candidate artifacts/prediction CSVs found in {}".format(run_dir)
        )
    required_paths = [
        args.fixed_candidate_artifact,
        args.walk_candidate_artifact,
        fixed_metrics_path if fixed_available else None,
        walk_metrics_path if walk_available else None,
        walk_diagnostics_path if walk_available else None,
    ]
    for path in required_paths:
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(path)

    fixed_reference = read_single_csv(fixed_metrics_path) if fixed_available else {}
    walk_metric_rows = read_csv_rows(walk_metrics_path) if walk_available else []
    walk_reference = next((row for row in walk_metric_rows if row.get("split") == "walkforward_average"), {})
    walk_diagnostics = read_csv_rows(walk_diagnostics_path) if walk_available else []
    month_to_fold, fold_to_split = walkforward_fold_maps(walk_diagnostics)
    total_folds = max(fold_to_split) if fold_to_split else (10 if walk_available else 0)
    fold_sets = build_research_fold_sets(args, total_folds)

    if args.fixed_candidate_artifact:
        fixed_rows, fixed_stats = extract_candidate_artifact(
            args.fixed_candidate_artifact,
            out_dir / "candidate_snapshot_fixed.csv",
            "fixed",
            force=args.force_rescan,
            require_pre_filter=args.require_pre_filter_artifacts,
            row_scope=args.candidate_artifact_row_scope,
        )
    elif fixed_predictions:
        fixed_rows, fixed_stats = extract_candidates(
            fixed_predictions,
            out_dir / "candidate_snapshot_fixed.csv",
            "fixed",
            month_to_fold,
            fold_to_split,
            force=args.force_rescan,
        )
    else:
        fixed_rows, fixed_stats = [], {
            "source": "fixed",
            "artifact_type": "not_provided",
            "coverage_level": "unavailable",
            "candidate_rows": 0,
            "raw_signal_rows": 0,
            "predicted_rows": 0,
            "selected_rows": 0,
        }
    if args.walk_candidate_artifact:
        walk_rows, walk_stats = extract_candidate_artifact(
            args.walk_candidate_artifact,
            out_dir / "candidate_snapshot_walkforward.csv",
            "walkforward",
            force=args.force_rescan,
            require_pre_filter=args.require_pre_filter_artifacts,
            row_scope=args.candidate_artifact_row_scope,
        )
    elif walk_predictions:
        walk_rows, walk_stats = extract_candidates(
            walk_predictions,
            out_dir / "candidate_snapshot_walkforward.csv",
            "walkforward",
            month_to_fold,
            fold_to_split,
            force=args.force_rescan,
        )
    else:
        walk_rows, walk_stats = [], {
            "source": "walkforward",
            "artifact_type": "not_provided",
            "coverage_level": "unavailable",
            "candidate_rows": 0,
            "raw_signal_rows": 0,
            "predicted_rows": 0,
            "selected_rows": 0,
        }
    all_rows = fixed_rows + walk_rows
    fallbacks = selected_position_fallbacks(all_rows)

    source_sets = {
        "fixed": selected_sets(fixed_rows, fallbacks),
        "walkforward": selected_sets(walk_rows, fallbacks),
    }
    references = {
        "fixed": fixed_reference,
        "walkforward": walk_reference,
    }

    experiment_rows = []
    cost_rows = []
    leave_symbol_rows = []
    leave_fold_rows = []
    overlap_rows = []
    fold_rows_out = []
    selected_rows_out = []
    active_sources = ["fixed" if fixed_available else "", "walkforward" if walk_available else ""]
    for source in [item for item in active_sources if item]:
        experiments = source_sets[source]
        baseline = experiments["baseline_selected_by_topk"]
        for experiment in EXPERIMENT_ORDER:
            rows = experiments[experiment]
            metrics = experiment_metrics(source, experiment, rows, total_folds, references[source])
            if source == "walkforward":
                fold_rows = fold_metric_rows(rows, fold_to_split, total_folds)
                metrics.update(fold_summary(source, experiment, fold_rows))
                for fold_row in fold_rows:
                    fold_copy = dict(fold_row)
                    fold_copy["source"] = source
                    fold_copy["experiment"] = experiment
                    fold_rows_out.append(fold_copy)
                for row in rd.leave_one_fold_out_rows(fold_rows):
                    fold_leave = dict(row)
                    fold_leave["source"] = source
                    fold_leave["experiment"] = experiment
                    leave_fold_rows.append(fold_leave)
            experiment_rows.append(metrics)
            cost_rows.extend(cost_stress(source, experiment, rows, total_folds))
            leave_symbol_rows.extend(leave_one_symbols(source, experiment, rows, total_folds))
            overlap_rows.append(selection_overlap(source, experiment, rows, baseline))
            for row in rows:
                selected_copy = {
                    "_source": source,
                    "experiment": experiment,
                    "row_id": row_identity(row),
                    "symbol": row.get("symbol", ""),
                    "month": row.get("month", ""),
                    "fold_index": row.get("_fold_index", ""),
                    "open_time": row.get("open_time", ""),
                    "trade_day": row.get("trade_day", ""),
                    "trade_score": row.get("trade_score", ""),
                    "offline_score": row.get("_offline_score", ""),
                    "position_size": row.get("position_size", ""),
                    "position_size_fallback": row.get("_position_size_fallback", ""),
                    "dynamic_trade_return": row.get("dynamic_trade_return", ""),
                    "net_return": rd.net_return(row),
                }
                selected_rows_out.append(selected_copy)

    snapshot_stats = [fixed_stats, walk_stats]
    split_rows = research_split_matrix_rows(fold_rows_out, fold_sets)
    bucket_audit_rows = []
    if fixed_available:
        bucket_audit_rows.extend(candidate_bucket_audit_rows("fixed", fixed_rows))
    if walk_available:
        bucket_audit_rows.extend(candidate_bucket_audit_rows("walkforward", walk_rows))
    decision_rows = research_decision_rows(experiment_rows, split_rows, bucket_audit_rows, cost_rows)
    write_csv(out_dir / "candidate_snapshot_summary.csv", snapshot_stats)
    write_csv(out_dir / "offline_experiment_matrix.csv", experiment_rows)
    write_csv(out_dir / "offline_cost_stress.csv", cost_rows)
    write_csv(out_dir / "offline_leave_one_symbol_out.csv", leave_symbol_rows)
    write_csv(out_dir / "offline_leave_one_fold_out.csv", leave_fold_rows)
    write_csv(out_dir / "offline_walkforward_fold_metrics.csv", fold_rows_out)
    write_csv(out_dir / "offline_research_split_matrix.csv", split_rows)
    write_csv(out_dir / "offline_research_decision.csv", decision_rows, RESEARCH_DECISION_FIELDS)
    write_csv(out_dir / "offline_selection_overlap.csv", overlap_rows)
    write_csv(out_dir / "offline_candidate_bucket_audit.csv", bucket_audit_rows, BUCKET_AUDIT_FIELDS)
    write_csv(out_dir / "offline_selected_trades.csv", selected_rows_out)
    with (out_dir / "position_size_fallbacks.json").open("w", encoding="utf-8") as handle:
        json.dump({str(key): value for key, value in fallbacks.items()}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report = build_report({
        "experiment_rows": experiment_rows,
        "cost_rows": cost_rows,
        "leave_symbol_rows": leave_symbol_rows,
        "leave_fold_rows": leave_fold_rows,
        "overlap_rows": overlap_rows,
        "snapshot_stats": snapshot_stats,
        "split_rows": split_rows,
        "bucket_audit_rows": bucket_audit_rows,
        "decision_rows": decision_rows,
    })
    (out_dir / "research_report.md").write_text(report, encoding="utf-8")
    print(report)
    write_offline_research_created_files_inventory(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
