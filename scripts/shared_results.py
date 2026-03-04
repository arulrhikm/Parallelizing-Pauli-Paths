"""
shared_results.py  -  shared JSON store for all benchmark scripts
=================================================================
All three benchmark scripts (run_benchmark.py, benchmark_qiskit.py,
verify_correctness.py) import and call `append_rows()` to record their
results in a single file:

    scripts/benchmark_summary.json

De-duplication
--------------
Each entry is keyed by  source :: test_name :: backend :: threads
If a row with the same key already exists, only the timestamp and the
measured fields (time_s, correct, notes) are overwritten - no duplicate
entries are ever created.

Schema (per entry)
------------------
timestamp   - ISO-8601 UTC string, e.g. "2024-03-04T12:00:00Z"
source      - which script produced the row
                run_benchmark | benchmark_qiskit | verify_correctness
test_index  - 0-based test vector index used on the CLI  (-1 = N/A)
test_name   - human-readable label, e.g. "DIVERSE-1: 10q, 30K H+CNOT, 20L"
backend     - cpu_seq | omp | qiskit | julia | gpu
threads     - OMP thread count (1 for cpu_seq/qiskit/julia/gpu, 0 for gpu)
time_s      - wall-clock seconds  (-1 = N/A / error)
correct     - PASS | FAIL | SKIP | N/A
notes       - free-text (empty string if nothing to add)

Usage
-----
from shared_results import append_rows, Row

rows = [
    Row(test_index=39, test_name="DIVERSE-1: 10q, 30K H+CNOT, 20L",
        backend="gpu", threads=0, time_s=0.027, source="run_benchmark"),
    ...
]
append_rows(rows)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
SUMMARY_JSON = REPO_ROOT / "scripts" / "benchmark_summary.json"


@dataclass
class Row:
    test_name  : str
    backend    : str
    time_s     : float
    source     : str
    test_index : int  = -1
    threads    : int  = 1
    nterms     : int  = -1   # output Pauli-word count (-1 = not measured)
    correct    : str  = "N/A"
    notes      : str  = ""
    timestamp  : str  = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def key(self) -> str:
        """Unique identity: (source, test_name, backend, threads)."""
        return f"{self.source}::{self.test_name}::{self.backend}::{self.threads}"

    def as_dict(self) -> dict:
        return asdict(self)


def _load(path: Path) -> dict:
    """Load existing JSON store, or return an empty dict."""
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save(store: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_rows(rows: list[Row], path: Path = SUMMARY_JSON) -> None:
    """Upsert rows into the shared JSON store.

    If an entry with the same (source, test_name, backend, threads) already
    exists it is overwritten; no duplicate entries are created.
    """
    if not rows:
        return

    store = _load(path)
    added = updated = 0

    for row in rows:
        k = row.key()
        if k in store:
            updated += 1
        else:
            added += 1
        store[k] = row.as_dict()

    _save(store, path)
    parts = []
    if added:
        parts.append(f"{added} new")
    if updated:
        parts.append(f"{updated} updated")
    print(f"  [shared_results] {', '.join(parts)} ({len(store)} total) -> {path}")


def load_all(path: Path = SUMMARY_JSON) -> list[dict]:
    """Return all stored entries as a list of dicts (for analysis scripts)."""
    return list(_load(path).values())
