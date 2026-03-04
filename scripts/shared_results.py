"""
shared_results.py  –  common CSV sink for all benchmark scripts
===============================================================
All three benchmark scripts (run_benchmark.py, benchmark_qiskit.py,
verify_correctness.py) import and call `append_rows()` to record their
results in a single file:

    scripts/benchmark_summary.csv

Schema
------
timestamp   – ISO-8601 UTC string, e.g. 2024-03-04T12:00:00Z
source      – which script produced the row
                run_benchmark | benchmark_qiskit | verify_correctness
test_index  – 0-based test vector index used on the CLI  (-1 = N/A)
test_name   – human-readable label, e.g. "STRESS 24: 7q, 5K words, 150 layers"
backend     – cpu_seq | omp | qiskit | julia | gpu
threads     – OMP thread count (1 for cpu_seq/qiskit/julia/gpu)
time_s      – wall-clock seconds  (-1 = N/A / error)
correct     – PASS | FAIL | SKIP | N/A
notes       – free-text (empty string if nothing to add)

Usage
-----
from shared_results import append_rows, Row

rows = [
    Row(test_index=24, test_name="STRESS 23", backend="gpu",
        threads=0, time_s=0.027, source="run_benchmark"),
    ...
]
append_rows(rows)
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT   = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO_ROOT / "scripts" / "benchmark_summary.csv"

COLUMNS = [
    "timestamp", "source", "test_index", "test_name",
    "backend", "threads", "time_s", "correct", "notes",
]


@dataclass
class Row:
    test_name   : str
    backend     : str
    time_s      : float
    source      : str
    test_index  : int            = -1
    threads     : int            = 1
    correct     : str            = "N/A"
    notes       : str            = ""
    timestamp   : str            = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def as_dict(self) -> dict:
        d = asdict(self)
        # Ensure column order
        return {k: d[k] for k in COLUMNS}


def _ascii(s: str) -> str:
    """Replace non-ASCII characters that may appear in test names (e.g. |Z>)."""
    return s.encode("ascii", errors="replace").decode("ascii")


def _ensure_header(path: Path) -> None:
    """Write the CSV header if the file doesn't exist or is empty."""
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)


def append_rows(rows: list[Row], path: Path = SUMMARY_CSV) -> None:
    """Append rows to the shared CSV (creates file + header if needed)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_header(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        for row in rows:
            d = row.as_dict()
            d["test_name"] = _ascii(d["test_name"])
            d["notes"]     = _ascii(d["notes"])
            writer.writerow(d)
    print(f"  [shared_results] {len(rows)} row(s) appended -> {path}")
