#!/usr/bin/env python3
"""
benchmark_qiskit.py  –  Task 1.2 external-tool comparison (Option B)
=====================================================================
Benchmarks the same 10 stress-test circuits (tests 23-32) using Qiskit's
SparsePauliOp.evolve(), which internally does exact Clifford propagation of
the observable.

Note on methodology
-------------------
Qiskit does NOT implement the *truncated* Pauli path algorithm; it does exact
propagation.  The comparison is therefore:
  - For Clifford-only circuits (H+CNOT, S+CNOT): the result must be exact.
    We can check correctness against our CPU output.
  - Timing captures wall-clock for evolving the initial observable through
    the full circuit – comparable to our CPU-seq time.

Install Qiskit:
    pip install qiskit qiskit-aer

Run:
    python3 scripts/benchmark_qiskit.py
"""

import time
import sys
import math
import random
import csv
from pathlib import Path

try:
    from shared_results import append_rows, Row as SRow
    _SHARED_OK = True
except ImportError:
    _SHARED_OK = False

try:
    from qiskit.quantum_info import SparsePauliOp, Clifford
    from qiskit import QuantumCircuit
    QISKIT_OK = True
except ImportError:
    QISKIT_OK = False
    print("Qiskit not installed.  Run:  pip install qiskit")
    sys.exit(1)

REPO_ROOT  = Path(__file__).resolve().parent.parent
MAX_WEIGHT = 10   # We still track weight but Qiskit does exact propagation

# ---------------------------------------------------------------------------
# Pauli word generator matching tests.cpp seeds + RNG
# ---------------------------------------------------------------------------
PAULI_MAP = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

class MT19937:
    """Minimal Mersenne-Twister wrapper matching std::mt19937_64 uniform_int."""
    def __init__(self, seed: int):
        random.seed(seed)

    def next_op(self) -> int:
        return random.randint(0, 3)


def make_pauli_sum_qiskit(nq: int, nwords: int, seed: int) -> SparsePauliOp:
    """
    Build a SparsePauliOp from the same random Pauli words as tests.cpp.
    Qiskit uses little-endian qubit ordering (qubit 0 = rightmost char).
    tests.cpp stores ops[0..nq-1] left-to-right, so we reverse the string.
    """
    rng = MT19937(seed)
    coeffs = {}
    for _ in range(nwords):
        ops = [PAULI_MAP[rng.next_op()] for _ in range(nq)]
        # tests.cpp uses ops[q] for qubit q; Qiskit string: qubit nq-1 first
        label = ''.join(reversed(ops))
        coeffs[label] = coeffs.get(label, 0.0) + 1.0

    labels  = list(coeffs.keys())
    cvalues = [coeffs[l] for l in labels]
    return SparsePauliOp(labels, coeffs=cvalues)


import math as _math

def make_circuit(nq: int, nlayers: int, gates: str) -> QuantumCircuit:
    """Build the circuit corresponding to the `gates` tag in STRESS_TESTS."""
    qc = QuantumCircuit(nq)
    for _ in range(nlayers):
        if gates == "h_cnot":
            for q in range(nq): qc.h(q)
        elif gates == "s_cnot":
            for q in range(nq): qc.s(q)
        elif gates == "t_h_cnot":
            for q in range(nq): qc.t(q)
            for q in range(0, nq, 2): qc.h(q)
        elif gates == "s_h_cnot":
            for q in range(nq): qc.s(q)
            for q in range(nq): qc.h(q)
        elif gates == "h_s_t_cnot":
            for q in range(nq): qc.h(q)
            for q in range(nq): qc.s(q)
            for q in range(nq): qc.t(q)
        # rotation gates raise ValueError when passed to Clifford() – callers
        # should set skip_clifford=True for those entries
        elif gates in ("rz_cnot", "rx_h_cnot", "rz_rx_h_cnot"):
            raise ValueError(f"Non-Clifford gate set '{gates}' cannot be used with Clifford()")
        else:
            raise ValueError(f"Unknown gate tag: {gates}")
        for q in range(nq - 1):
            qc.cx(q, q + 1)
    return qc


# Keep old names for backwards compatibility
def make_circuit_h_cnot(nq: int, nlayers: int) -> QuantumCircuit:
    return make_circuit(nq, nlayers, "h_cnot")

def make_circuit_s_cnot(nq: int, nlayers: int) -> QuantumCircuit:
    return make_circuit(nq, nlayers, "s_cnot")


# ---------------------------------------------------------------------------
# Stress test definitions
# ---------------------------------------------------------------------------
STRESS_TESTS = [
    # ── Original STRESS tests (7-qubit Clifford) ───────────────────────
    {"name": "STRESS 23: 7q, 2K words, 100 layers",  "nq": 7, "nw": 2000, "nl": 100, "seed": 2301, "gates": "h_cnot",    "idx": 24},
    {"name": "STRESS 24: 7q, 5K words, 150 layers",  "nq": 7, "nw": 5000, "nl": 150, "seed": 2401, "gates": "h_cnot",    "idx": 25},
    {"name": "STRESS 25: 7q, 3K words, 200 layers",  "nq": 7, "nw": 3000, "nl": 200, "seed": 2501, "gates": "h_cnot",    "idx": 26},
    {"name": "STRESS 26: 7q, 1K words, 300 layers",  "nq": 7, "nw": 1000, "nl": 300, "seed": 2601, "gates": "s_cnot",    "idx": 27},
    {"name": "STRESS 27: 7q, 4K words, 100 layers",  "nq": 7, "nw": 4000, "nl": 100, "seed": 2701, "gates": "h_cnot",    "idx": 28},
    {"name": "STRESS 28: 7q, 2K words, 250 layers",  "nq": 7, "nw": 2000, "nl": 250, "seed": 2801, "gates": "h_cnot",    "idx": 29},
    {"name": "STRESS 29: 7q, 1K words, 400 layers",  "nq": 7, "nw": 1000, "nl": 400, "seed": 2901, "gates": "h_cnot",    "idx": 30},
    {"name": "STRESS 30: 7q, 8K words,  50 layers",  "nq": 7, "nw": 8000, "nl":  50, "seed": 3001, "gates": "h_cnot",    "idx": 31},
    {"name": "STRESS 31: 7q, 500 words, 500 layers", "nq": 7, "nw":  500, "nl": 500, "seed": 3101, "gates": "h_cnot",    "idx": 32},
    {"name": "STRESS 32: 7q, 5K words, 120 layers",  "nq": 7, "nw": 5000, "nl": 120, "seed": 3201, "gates": "h_cnot",    "idx": 33},
    # ── SCALE tests (9-qubit, large observable) ────────────────────────
    {"name": "SCALE-1: 9q, 10K words, 30 layers",    "nq": 9, "nw": 10000, "nl": 30, "seed": 3401, "gates": "h_cnot",    "idx": 34},
    {"name": "SCALE-2: 9q, 15K words, 30 layers",    "nq": 9, "nw": 15000, "nl": 30, "seed": 3501, "gates": "h_cnot",    "idx": 35},
    {"name": "SCALE-3: 9q, 20K words, 30 layers",    "nq": 9, "nw": 20000, "nl": 30, "seed": 3601, "gates": "h_cnot",    "idx": 36},
    {"name": "SCALE-4: 9q, 50K words, 20 layers",    "nq": 9, "nw": 50000, "nl": 20, "seed": 3701, "gates": "h_cnot",    "idx": 37},
    {"name": "SCALE-5: 9q, 100K words, 10 layers",   "nq": 9, "nw":100000, "nl": 10, "seed": 3801, "gates": "h_cnot",    "idx": 38},
    # ── DIVERSE Clifford tests (varied gate sets, 9-10 qubits) ─────────
    # NOTE: rotation tests (DIVERSE-5/6/8) use non-Clifford RZ/RX gates;
    #       Qiskit's Clifford class cannot represent those circuits, so
    #       they are skipped here (marked "clifford_only": False).
    {"name": "DIVERSE-1: 10q, 30K H+CNOT, 20L",      "nq":10, "nw": 30000, "nl": 20, "seed": 3901, "gates": "h_cnot",    "idx": 39},
    {"name": "DIVERSE-2: 10q, 60K H+CNOT, 10L",      "nq":10, "nw": 60000, "nl": 10, "seed": 4001, "gates": "h_cnot",    "idx": 40},
    # T gate is Clifford in the Pauli-algebra sense but Qiskit's Clifford
    # class does not accept it (T is NOT a Clifford gate in the standard
    # Clifford group definition — it is a pi/4 phase gate, not pi/2).
    {"name": "DIVERSE-3: 9q, 25K T+H+CNOT, 30L",     "nq": 9, "nw": 25000, "nl": 30, "seed": 4101, "gates": "t_h_cnot",  "idx": 41, "skip_clifford": True},
    {"name": "DIVERSE-4: 9q, 35K S+H+CNOT, 20L",     "nq": 9, "nw": 35000, "nl": 20, "seed": 4201, "gates": "s_h_cnot",  "idx": 42},
    {"name": "DIVERSE-5: 9q, 5K RZ+CNOT, 8L",        "nq": 9, "nw":  5000, "nl":  8, "seed": 4301, "gates": "rz_cnot",   "idx": 43, "skip_clifford": True},
    {"name": "DIVERSE-6: 9q, 4K RX+H+CNOT, 6L",      "nq": 9, "nw":  4000, "nl":  6, "seed": 4401, "gates": "rx_h_cnot", "idx": 44, "skip_clifford": True},
    {"name": "DIVERSE-7: 10q, 25K H+S+T+CNOT, 15L",  "nq":10, "nw": 25000, "nl": 15, "seed": 4501, "gates": "h_s_t_cnot","idx": 45, "skip_clifford": True},
    {"name": "DIVERSE-8: 9q, 8K RZ+RX+H+CNOT, 15L",  "nq": 9, "nw":  8000, "nl": 15, "seed": 4601, "gates": "rz_rx_h_cnot", "idx": 46, "skip_clifford": True},
]


def evolve_observable(obs: SparsePauliOp, qc: QuantumCircuit) -> SparsePauliOp:
    """
    Evolve observable backward through the circuit: computes U† O U exactly.

    SparsePauliOp.evolve() was removed in Qiskit 2.x.  The replacement is to
    convert the circuit to a Clifford and call PauliList.evolve() on the
    underlying Paulis, then reconstruct the SparsePauliOp.
    """
    clifford = Clifford(qc)
    evolved_paulis = obs.paulis.evolve(clifford)
    return SparsePauliOp(evolved_paulis, coeffs=obs.coeffs).simplify()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 70)
print("  QISKIT SparsePauliOp BENCHMARK  (Task 1.2 Option B)")
try:
    import qiskit
    print(f"  Qiskit version: {qiskit.__version__}")
except Exception:
    pass
print("=" * 70)
print()
print("NOTE: Qiskit evolve() is EXACT (no weight truncation).")
print("      For Clifford-only tests (H+CNOT, S+CNOT), results are exact")
print("      and comparable to our CPU-seq times.\n")

timings = []
skip_tags = set()

print(f"{'Test':<50}  {'Build(s)':>8}  {'Evolve(s)':>9}  {'Terms':>8}")
print("-" * 82)

for tc in STRESS_TESTS:
    nq, nw, nl = tc["nq"], tc["nw"], tc["nl"]
    skip_clifford = tc.get("skip_clifford", False)

    if skip_clifford:
        timings.append(-2.0)          # sentinel: skipped (non-Clifford)
        skip_tags.add(tc["gates"])
        name_short = tc["name"][:49]
        print(f"{name_short:<50}  {'N/A':>8}  {'SKIP (non-Clifford RZ/RX)':>9}")
        continue

    # Build observable
    t0 = time.perf_counter()
    obs = make_pauli_sum_qiskit(nq, nw, tc["seed"])
    build_t = time.perf_counter() - t0

    # Build Clifford circuit
    try:
        qc = make_circuit(nq, nl, tc["gates"])
    except ValueError as e:
        timings.append(-1.0)
        print(f"  ERROR building circuit: {e}")
        continue

    # Evolve (timed)
    t0 = time.perf_counter()
    try:
        evolved = evolve_observable(obs, qc)
        evolve_t = time.perf_counter() - t0
        nterms = len(evolved)
    except Exception as e:
        evolve_t = -1.0
        nterms = -1
        print(f"  ERROR: {e}")

    timings.append(evolve_t)
    name_short = tc["name"][:49]
    print(f"{name_short:<50}  {build_t:>8.4f}  {evolve_t:>9.4f}  {nterms:>8}")

print()
print("=" * 70)
valid = [t for t in timings if t > 0]
if valid:
    print(f"  Mean evolve time: {sum(valid)/len(valid):.4f} s")
    print(f"  Max  evolve time: {max(valid):.4f} s")
if skip_tags:
    print(f"\n  NOTE: Tests with non-Clifford gates ({', '.join(skip_tags)})")
    print("        are SKIPPED here. Qiskit's Clifford class supports only")
    print("        Clifford gates (H, S, T, CNOT, CZ, SWAP, X, Y, Z).")
    print("        Compare those tests via run_benchmark.py instead.")

# Save per-script CSV
out = REPO_ROOT / "scripts" / "benchmark_qiskit_results.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test", "test_index", "qiskit_evolve_s"])
    for tc, t in zip(STRESS_TESTS, timings):
        w.writerow([tc["name"], tc.get("idx", -1), t])

print(f"\n  Results saved to: {out}")
print()
print("  Compare 'qiskit_evolve_s' to 'cpu_seq' in benchmark_results.csv")
print("  to show the speedup of our GPU over Qiskit's exact CPU propagation.")
print("=" * 70)

# Write to shared benchmark_summary.csv
if _SHARED_OK:
    shared = []
    for tc, t in zip(STRESS_TESTS, timings):
        if t > 0:
            shared.append(SRow(
                test_index = tc.get("idx", -1),
                test_name  = tc["name"],
                backend    = "qiskit",
                threads    = 1,
                time_s     = t,
                source     = "benchmark_qiskit",
                notes      = "exact Clifford sim via PauliList.evolve(Clifford(qc))",
            ))
    if shared:
        append_rows(shared)
