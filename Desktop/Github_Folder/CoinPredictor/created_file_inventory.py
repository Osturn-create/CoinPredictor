"""Write small inventories of files created by a run."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

from artifact_contracts import atomic_write_path


INVENTORY_COLUMNS = ["label", "path", "size_bytes"]


def existing_file_rows(paths: Iterable[tuple[str, str | os.PathLike[str]]]) -> list[dict]:
    rows = []
    seen = set()
    for label, path_value in paths:
        if not path_value:
            continue
        path = Path(path_value).resolve()
        key = str(path)
        if key in seen or not path.exists() or not path.is_file():
            continue
        seen.add(key)
        rows.append({
            "label": label,
            "path": key,
            "size_bytes": int(path.stat().st_size),
        })
    return rows


def write_inventory(
    inventory_path: str | os.PathLike[str],
    paths: Iterable[tuple[str, str | os.PathLike[str]]],
) -> dict:
    rows = existing_file_rows(paths)
    if not rows:
        return {"path": "", "rows": 0, "size_bytes": 0}
    target = Path(inventory_path).resolve()

    def write_one(output_path: str) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    atomic_write_path(target, write_one)
    return {
        "path": str(target),
        "rows": len(rows),
        "size_bytes": int(target.stat().st_size),
    }

