"""Streaming candidate artifact writer and reader."""

from __future__ import annotations

import csv
import datetime as _dt
import gzip
import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from artifact_contracts import (
    CANDIDATE_ARTIFACT_COLUMNS,
    CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    atomic_write_json,
    manifest_path_for,
    sha256_file,
)


SUPPORTED_FORMATS = ("csv", "csv_gzip")


class CandidateArtifactError(RuntimeError):
    pass


def deterministic_candidate_id(*parts: object) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_format(path: str | os.PathLike[str], artifact_format: str = "csv_gzip") -> str:
    value = (artifact_format or "csv_gzip").lower()
    if value == "gzip":
        value = "csv_gzip"
    if value not in SUPPORTED_FORMATS:
        raise CandidateArtifactError(
            "candidate artifact format {!r} is not supported without optional dependencies; "
            "use csv or csv_gzip".format(artifact_format)
        )
    if str(path).endswith(".gz"):
        value = "csv_gzip"
    return value


def open_candidate_text(path: str | os.PathLike[str], mode: str, artifact_format: str):
    if artifact_format == "csv_gzip":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return open(path, mode, newline="", encoding="utf-8")


def _safe_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return "{:.12g}".format(value)
    return value


class CandidateArtifactWriter:
    """Write candidate records atomically with a versioned sidecar manifest."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        metadata: dict | None = None,
        artifact_format: str = "csv_gzip",
        max_rows: int = 0,
    ) -> None:
        self.path = Path(path)
        self.artifact_format = normalize_format(self.path, artifact_format)
        self.metadata = dict(metadata or {})
        self.max_rows = max(0, int(max_rows or 0))
        self.row_count = 0
        self.executed_row_count = 0
        self.source_rows_seen = 0
        self._tmp_path: Path | None = None
        self._handle = None
        self._writer = None
        self._closed = False

    def __enter__(self) -> "CandidateArtifactWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        os.close(fd)
        self._tmp_path = Path(tmp)
        self._handle = open_candidate_text(self._tmp_path, "wt", self.artifact_format)
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=CANDIDATE_ARTIFACT_COLUMNS,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        return self

    def write_record(self, record: dict) -> bool:
        if self._writer is None:
            raise CandidateArtifactError("candidate artifact writer is not open")
        self.source_rows_seen += 1
        if self.max_rows and self.row_count >= self.max_rows:
            return False
        row = {field: _safe_value(record.get(field, "")) for field in CANDIDATE_ARTIFACT_COLUMNS}
        self._writer.writerow(row)
        self.row_count += 1
        if str(row.get("executed", "")).strip() in ("1", "true", "True"):
            self.executed_row_count += 1
        return True

    def write_records(self, records: Iterable[dict]) -> None:
        for record in records:
            self.write_record(record)

    def close(self) -> dict:
        if self._closed:
            return self.manifest()
        if self._handle is None or self._tmp_path is None:
            raise CandidateArtifactError("candidate artifact writer was not opened")
        self._handle.close()
        os.replace(self._tmp_path, self.path)
        self._tmp_path = None
        manifest = self.manifest()
        atomic_write_json(manifest_path_for(self.path), manifest)
        self._closed = True
        return manifest

    def manifest(self) -> dict:
        file_size = self.path.stat().st_size if self.path.exists() else 0
        row_limit_applied = bool(self.max_rows and self.source_rows_seen > self.row_count)
        manifest = {
            "artifact_type": "candidate_predictions",
            "schema_version": CANDIDATE_ARTIFACT_SCHEMA_VERSION,
            "generation_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "artifact_path": str(self.path),
            "format": self.artifact_format,
            "compression": "gzip" if self.artifact_format == "csv_gzip" else "none",
            "columns": list(CANDIDATE_ARTIFACT_COLUMNS),
            "row_count": int(self.row_count),
            "source_rows_seen": int(self.source_rows_seen),
            "executed_row_count": int(self.executed_row_count),
            "file_size": int(file_size),
            "checksum": sha256_file(self.path) if self.path.exists() else "",
            "complete": True,
            "outcomes_available": True,
            "full_candidate_coverage": not row_limit_applied,
            "row_limit": int(self.max_rows),
            "row_limit_applied": row_limit_applied,
            "sampling_method": "first_n_stream_rows" if row_limit_applied else "none",
        }
        manifest.update(self.metadata)
        return manifest

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


def write_candidate_artifact(
    path: str | os.PathLike[str],
    records: Iterable[dict],
    metadata: dict | None = None,
    artifact_format: str = "csv_gzip",
    max_rows: int = 0,
) -> dict:
    writer = CandidateArtifactWriter(path, metadata, artifact_format, max_rows)
    with writer:
        writer.write_records(records)
    return writer.manifest()


def load_manifest(path: str | os.PathLike[str]) -> dict:
    import json

    manifest_path = manifest_path_for(path)
    if not manifest_path.exists():
        raise CandidateArtifactError("candidate artifact manifest is missing: {}".format(manifest_path))
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", 0)) != CANDIDATE_ARTIFACT_SCHEMA_VERSION:
        raise CandidateArtifactError("unsupported candidate artifact schema version")
    if not manifest.get("complete"):
        raise CandidateArtifactError("candidate artifact is incomplete: {}".format(manifest_path))
    expected = manifest.get("checksum")
    if expected and Path(path).exists() and sha256_file(path) != expected:
        raise CandidateArtifactError("candidate artifact checksum mismatch: {}".format(path))
    return manifest


def read_candidate_rows(path: str | os.PathLike[str]) -> Iterator[dict]:
    manifest = load_manifest(path)
    artifact_format = normalize_format(path, manifest.get("format", "csv_gzip"))
    with open_candidate_text(path, "rt", artifact_format) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def artifact_capabilities(manifest: dict | None) -> dict[str, str]:
    if not manifest:
        return {
            "coverage_level": "legacy_trades_only_or_unknown",
            "executed_trade_diagnostics": "supported",
            "true_topk_reselection": "unsupported",
            "rejected_candidate_analysis": "unsupported",
            "cross_sectional_normalization": "unsupported",
            "exposure_aware_reselection": "unsupported",
            "score_edge_ablation": "unsupported",
            "symbol_filter_ablation": "unsupported",
            "candidate_serialization_stage": "unknown",
            "reason": "missing candidate manifest",
        }
    full = bool(manifest.get("full_candidate_coverage", False))
    stage = str(manifest.get("candidate_serialization_stage", "post_selection") or "post_selection")
    pre_filter = stage == "pre_score_edge_pre_symbol_filter"
    coverage_level = "pre_filter_candidate" if pre_filter and full else (
        "pre_filter_candidate_limited" if pre_filter else ("post_selection_candidate" if full else "post_selection_candidate_limited")
    )
    row_limited_reason = "" if full else "candidate artifact was row-limited"
    pre_filter_reason = "" if pre_filter else "candidate artifact was serialized after threshold/score-edge/symbol filters"
    return {
        "coverage_level": coverage_level,
        "executed_trade_diagnostics": "supported",
        "true_topk_reselection": "supported" if full else "unsupported_limited_rows",
        "rejected_candidate_analysis": "supported" if full else "unsupported_limited_rows",
        "cross_sectional_normalization": "supported" if full else "unsupported_limited_rows",
        "exposure_aware_reselection": "supported" if full else "unsupported_limited_rows",
        "score_edge_ablation": "supported" if pre_filter and full else "unsupported",
        "symbol_filter_ablation": "supported" if pre_filter and full else "unsupported",
        "candidate_serialization_stage": stage,
        "reason": row_limited_reason or pre_filter_reason,
    }
