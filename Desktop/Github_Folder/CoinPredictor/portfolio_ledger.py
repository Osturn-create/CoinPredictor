"""Versioned portfolio event ledger and reconciliation utilities."""

from __future__ import annotations

import csv
import datetime as _dt
import gzip
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from artifact_contracts import (
    PORTFOLIO_EVENT_COLUMNS,
    PORTFOLIO_LEDGER_SCHEMA_VERSION,
    atomic_write_json,
    manifest_path_for,
    sha256_file,
)


class PortfolioLedgerError(RuntimeError):
    pass


def _safe_float(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return "{:.12g}".format(value)
    return value


def open_ledger_text(path: str | os.PathLike[str], mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return open(path, mode, newline="", encoding="utf-8")


class PortfolioLedgerWriter:
    """Stream portfolio events to a temp file and publish atomically."""

    def __init__(self, path: str | os.PathLike[str], metadata: dict | None = None) -> None:
        self.path = Path(path)
        self.metadata = dict(metadata or {})
        self.event_count = 0
        self._tmp_path: Path | None = None
        self._handle = None
        self._writer = None
        self._closed = False

    def __enter__(self) -> "PortfolioLedgerWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        os.close(fd)
        self._tmp_path = Path(tmp)
        if str(self.path).endswith(".gz"):
            self._handle = gzip.open(self._tmp_path, "wt", newline="", encoding="utf-8")
        else:
            self._handle = open(self._tmp_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=PORTFOLIO_EVENT_COLUMNS,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        return self

    def record_event(self, event: dict) -> None:
        if self._writer is None:
            raise PortfolioLedgerError("portfolio ledger writer is not open")
        row = {field: _safe_value(event.get(field, "")) for field in PORTFOLIO_EVENT_COLUMNS}
        row["event_sequence"] = self.event_count
        self._writer.writerow(row)
        self.event_count += 1

    def close(self) -> dict:
        if self._closed:
            return self.manifest()
        if self._handle is None or self._tmp_path is None:
            raise PortfolioLedgerError("portfolio ledger writer was not opened")
        self._handle.close()
        os.replace(self._tmp_path, self.path)
        self._tmp_path = None
        manifest = self.manifest()
        atomic_write_json(manifest_path_for(self.path), manifest)
        self._closed = True
        return manifest

    def manifest(self) -> dict:
        return {
            "artifact_type": "portfolio_event_ledger",
            "schema_version": PORTFOLIO_LEDGER_SCHEMA_VERSION,
            "generation_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "artifact_path": str(self.path),
            "format": "csv_gzip" if str(self.path).endswith(".gz") else "csv",
            "compression": "gzip" if str(self.path).endswith(".gz") else "none",
            "columns": list(PORTFOLIO_EVENT_COLUMNS),
            "event_count": int(self.event_count),
            "file_size": int(self.path.stat().st_size) if self.path.exists() else 0,
            "checksum": sha256_file(self.path) if self.path.exists() else "",
            "complete": True,
            **self.metadata,
        }

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        if exc_type is None:
            self.close()
            return
        if self._tmp_path is not None:
            try:
                os.remove(self._tmp_path)
            except OSError:
                pass


class PortfolioEventCollector:
    """In-memory recorder used by tests and small reconciliations."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def record_event(self, event: dict) -> None:
        row = dict(event)
        row.setdefault("event_sequence", len(self.events))
        self.events.append(row)


class PortfolioEventFanout:
    """Forward each event to multiple recorders."""

    def __init__(self, *recorders) -> None:
        self.recorders = [recorder for recorder in recorders if recorder is not None]

    def record_event(self, event: dict) -> None:
        for recorder in self.recorders:
            recorder.record_event(dict(event))


def load_manifest(path: str | os.PathLike[str]) -> dict:
    import json

    manifest_path = manifest_path_for(path)
    if not manifest_path.exists():
        raise PortfolioLedgerError("portfolio ledger manifest is missing: {}".format(manifest_path))
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", 0)) != PORTFOLIO_LEDGER_SCHEMA_VERSION:
        raise PortfolioLedgerError("unsupported portfolio ledger schema version")
    if not manifest.get("complete"):
        raise PortfolioLedgerError("portfolio ledger is incomplete: {}".format(manifest_path))
    expected = manifest.get("checksum")
    if expected and Path(path).exists() and sha256_file(path) != expected:
        raise PortfolioLedgerError("portfolio ledger checksum mismatch: {}".format(path))
    return manifest


def read_events(path: str | os.PathLike[str]) -> Iterator[dict]:
    load_manifest(path)
    with open_ledger_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def reconcile_events(
    events: Iterable[dict],
    starting_capital: float = 10000.0,
    reported_total_fees: float | None = None,
    reported_total_slippage: float | None = None,
) -> dict:
    ordered = sorted(list(events), key=lambda row: _safe_int(row.get("event_sequence")))
    if not ordered:
        return {
            "portfolio_reconciliation_status": "failed_empty_ledger",
            "portfolio_reconciliation_passed": 0,
            "portfolio_reconciliation_failed": 1,
            "drawdown_precision": "unavailable",
            "mark_to_market_drawdown_available": 0,
        }

    start = _safe_float(ordered[0].get("equity"), starting_capital)
    ending_capital = _safe_float(ordered[-1].get("equity"), start)
    portfolio_profit = ending_capital - start
    total_fees = sum(_safe_float(row.get("fee")) for row in ordered)
    total_slippage = sum(_safe_float(row.get("slippage")) for row in ordered)
    total_latency_cost = sum(_safe_float(row.get("latency_cost")) for row in ordered)
    max_concurrent = max(_safe_int(row.get("concurrent_position_count")) for row in ordered)
    max_open_exposure = max(_safe_float(row.get("total_open_exposure")) for row in ordered)
    max_capital_utilization = 0.0
    peak = start
    max_drawdown = 0.0
    underwater_start = None
    longest_underwater = 0.0
    recovery_duration = 0.0
    positions: dict[str, tuple[str, float]] = {}
    symbol_exposure: dict[str, float] = {}
    max_symbol_exposure = 0.0
    mark_events = 0

    for row in ordered:
        equity = _safe_float(row.get("equity"), start)
        exposure = _safe_float(row.get("total_open_exposure"))
        if equity > 0.0:
            max_capital_utilization = max(max_capital_utilization, exposure / equity)
        timestamp = _safe_float(row.get("timestamp"))
        if equity >= peak:
            if underwater_start is not None:
                recovery_duration = max(recovery_duration, timestamp - underwater_start)
                underwater_start = None
            peak = equity
        elif peak > 0.0:
            if underwater_start is None:
                underwater_start = timestamp
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
            longest_underwater = max(longest_underwater, timestamp - underwater_start)

        event_type = str(row.get("event_type", ""))
        trade_id = str(row.get("trade_id", ""))
        symbol = str(row.get("symbol", ""))
        notional = _safe_float(row.get("position_notional"))
        if event_type == "position_open" and trade_id:
            positions[trade_id] = (symbol, notional)
            symbol_exposure[symbol] = symbol_exposure.get(symbol, 0.0) + notional
        elif event_type == "position_close" and trade_id in positions:
            old_symbol, old_notional = positions.pop(trade_id)
            symbol_exposure[old_symbol] = max(0.0, symbol_exposure.get(old_symbol, 0.0) - old_notional)
        if equity > 0.0:
            max_symbol_exposure = max(
                max_symbol_exposure,
                max((value / equity for value in symbol_exposure.values()), default=0.0),
            )
        if event_type == "mark_to_market":
            mark_events += 1

    fee_ok = reported_total_fees is None or abs(total_fees - float(reported_total_fees)) <= 1e-6
    slippage_ok = reported_total_slippage is None or abs(total_slippage - float(reported_total_slippage)) <= 1e-6
    capital_ok = abs((ending_capital - start) - portfolio_profit) <= 1e-6
    passed = fee_ok and slippage_ok and capital_ok
    return {
        "starting_capital": start,
        "ending_capital": ending_capital,
        "portfolio_profit": portfolio_profit,
        "portfolio_return": portfolio_profit / start if start else 0.0,
        "event_count": len(ordered),
        "portfolio_reconciliation_status": "passed" if passed else "failed",
        "portfolio_reconciliation_passed": 1 if passed else 0,
        "portfolio_reconciliation_failed": 0 if passed else 1,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_latency_cost": total_latency_cost,
        "max_concurrent_positions": max_concurrent,
        "maximum_capital_utilization": max_capital_utilization,
        "maximum_open_exposure": max_open_exposure,
        "maximum_per_symbol_exposure": max_symbol_exposure,
        "exact_realized_drawdown": max_drawdown,
        "exact_mark_to_market_drawdown": max_drawdown if mark_events else 0.0,
        "mark_to_market_drawdown_available": 1 if mark_events else 0,
        "drawdown_precision": "exact_mark_to_market" if mark_events else "exact_realized",
        "longest_underwater_duration_minutes": longest_underwater,
        "time_to_recovery_minutes": recovery_duration,
    }


def reconcile_ledger_file(
    path: str | os.PathLike[str],
    starting_capital: float = 10000.0,
    reported_total_fees: float | None = None,
    reported_total_slippage: float | None = None,
) -> dict:
    return reconcile_events(
        read_events(path),
        starting_capital=starting_capital,
        reported_total_fees=reported_total_fees,
        reported_total_slippage=reported_total_slippage,
    )


def reconcile_events_by_fold(
    events: Iterable[dict],
    starting_capital: float = 10000.0,
) -> dict:
    by_fold: dict[str, list[dict]] = {}
    for event in events:
        fold_id = str(event.get("fold_id", "") or "default")
        by_fold.setdefault(fold_id, []).append(event)
    if not by_fold:
        return {
            "portfolio_reconciliation_status": "failed_empty_ledger",
            "portfolio_reconciliation_passed": 0,
            "portfolio_reconciliation_failed": 1,
            "portfolio_event_count": 0,
            "fold_count": 0,
        }
    reconciliations = {
        fold_id: reconcile_events(fold_events, starting_capital=starting_capital)
        for fold_id, fold_events in sorted(by_fold.items())
    }
    passed = all(row.get("portfolio_reconciliation_passed") for row in reconciliations.values())
    return {
        "portfolio_reconciliation_status": "passed" if passed else "failed",
        "portfolio_reconciliation_passed": 1 if passed else 0,
        "portfolio_reconciliation_failed": 0 if passed else 1,
        "portfolio_event_count": sum(int(row.get("event_count", 0)) for row in reconciliations.values()),
        "fold_count": len(reconciliations),
        "max_concurrent_positions": max(int(row.get("max_concurrent_positions", 0)) for row in reconciliations.values()),
        "maximum_capital_utilization": max(float(row.get("maximum_capital_utilization", 0.0)) for row in reconciliations.values()),
        "maximum_open_exposure": max(float(row.get("maximum_open_exposure", 0.0)) for row in reconciliations.values()),
        "maximum_per_symbol_exposure": max(float(row.get("maximum_per_symbol_exposure", 0.0)) for row in reconciliations.values()),
        "exact_realized_drawdown": max(float(row.get("exact_realized_drawdown", 0.0)) for row in reconciliations.values()),
        "mark_to_market_drawdown_available": 1 if all(row.get("mark_to_market_drawdown_available") for row in reconciliations.values()) else 0,
        "drawdown_precision": "exact_mark_to_market" if all(row.get("mark_to_market_drawdown_available") for row in reconciliations.values()) else "exact_realized",
        "longest_underwater_duration_minutes": max(float(row.get("longest_underwater_duration_minutes", 0.0)) for row in reconciliations.values()),
        "time_to_recovery_minutes": max(float(row.get("time_to_recovery_minutes", 0.0)) for row in reconciliations.values()),
        "fold_reconciliations": reconciliations,
    }


def reconcile_ledger_file_by_fold(
    path: str | os.PathLike[str],
    starting_capital: float = 10000.0,
) -> dict:
    return reconcile_events_by_fold(read_events(path), starting_capital=starting_capital)
