"""Versioned artifact contracts shared by CoinPredictor outputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


CANDIDATE_ARTIFACT_SCHEMA_VERSION = 1
PORTFOLIO_LEDGER_SCHEMA_VERSION = 1


CANDIDATE_ARTIFACT_COLUMNS = [
    "candidate_id",
    "source",
    "stable_row_id",
    "symbol",
    "month",
    "month_index",
    "open_time",
    "fold_id",
    "decision_bucket_id",
    "execution_bucket_id",
    "period_bucket_id",
    "row_position",
    "label",
    "forward_return",
    "trade_return",
    "max_future_high_return",
    "max_future_low_return",
    "quote_volume",
    "actual_exit_return",
    "exit_timestamp",
    "holding_period_minutes",
    "raw_probability",
    "calibrated_probability",
    "raw_predicted_return",
    "calibrated_predicted_return",
    "expected_value",
    "hybrid_score",
    "ranker_score",
    "ranker_selection_score",
    "ranker_utility_score",
    "uncertainty",
    "meta_probability",
    "final_preselection_score",
    "selected_threshold",
    "candidate_serialization_stage",
    "selected_by_threshold",
    "selected_by_score_edge",
    "selected_by_symbol_filter",
    "selected_by_topk",
    "raw_signal",
    "predicted",
    "threshold_percentile",
    "candidate_generation_reason",
    "rejection_stage",
    "rejection_reason",
    "selected_by_score_before_execution",
    "executed",
    "position_size",
    "rank_within_decision_bucket",
    "candidate_count_within_decision_bucket",
    "rank_within_execution_bucket",
    "candidate_count_within_execution_bucket",
    "rank_within_period_bucket",
    "candidate_count_within_period_bucket",
    "configured_top_k",
    "symbol_exposure_before_selection",
    "capital_available_before_selection",
    "trade_day",
    "exit_reason",
]


PORTFOLIO_EVENT_COLUMNS = [
    "event_sequence",
    "timestamp",
    "event_type",
    "trade_id",
    "candidate_id",
    "symbol",
    "fold_id",
    "decision_timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "position_quantity",
    "position_notional",
    "entry_price",
    "current_price",
    "realized_pnl",
    "unrealized_pnl",
    "fee",
    "slippage",
    "latency_cost",
    "cash_balance",
    "reserved_capital",
    "total_open_exposure",
    "equity",
    "concurrent_position_count",
]


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_path(path: str | os.PathLike[str], writer: Callable[[str], None]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    try:
        writer(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    def write_one(output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    atomic_write_path(path, write_one)


def manifest_path_for(artifact_path: str | os.PathLike[str]) -> Path:
    return Path(str(artifact_path) + ".manifest.json")
