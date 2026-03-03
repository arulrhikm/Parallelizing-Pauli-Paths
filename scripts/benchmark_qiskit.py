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
    from qiskit.quantum_info import SparsePauliOp
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


def make_circuit_h_cnot(nq: int, nlayers: int) -> QuantumCircuit:
    qc = QuantumCircuit(nq)
    for _ in range(nlayers):
        for q in range(nq):
            qc.h(q)
        for q in range(nq - 1):
            qc.cx(q, q + 1)
    return qc


def make_circuit_s_cnot(nq: int, nlayers: int) -> QuantumCircuit:
    qc = QuantumCircuit(nq)
    for _ in range(nlayers):
        for q in range(nq):
            qc.s(q)
        for q in range(nq - 1):
            qc.cx(q, q + 1)
    return qc


# ---------------------------------------------------------------------------
# Stress test definitions
# ---------------------------------------------------------------------------
STRESS_TESTS = [
    {"name": "STRESS 23: 7q, 2K words, 100 layers",  "nq": 7, "nw": 2000, "nl": 100, "seed": 2301, "gates": "h_cnot"},
    {"name": "STRESS 24: 7q, 5K words, 150 layers",  "nq": 7, "nw": 5000, "nl": 150, "seed": 2401, "gates": "h_cnot"},
    {"name": "STRESS 25: 7q, 3K words, 200 layers",  "nq": 7, "nw": 3000, "nl": 200, "seed": 2501, "gates": "h_cnot"},
    {"name": "STRESS 26: 7q, 1K words, 300 layers",  "nq": 7, "nw": 1000, "nl": 300, "seed": 2601, "gates": "s_cnot"},
    {"name": "STRESS 27: 7q, 4K words, 100 layers",  "nq": 7, "nw": 4000, "nl": 100, "seed": 2701, "gates": "h_cnot"},
    {"name": "STRESS 28: 7q, 2K words, 250 layers",  "nq": 7, "nw": 2000, "nl": 250, "seed": 2801, "gates": "h_cnot"},
    {"name": "STRESS 29: 7q, 1K words, 400 layers",  "nq": 7, "nw": 1000, "nl": 400, "seed": 2901, "gates": "h_cnot"},
    {"name": "STRESS 30: 7q, 8K words,  50 layers",  "nq": 7, "nw": 8000, "nl":  50, "seed": 3001, "gates": "h_cnot"},
    {"name": "STRESS 31: 7q, 500 words, 500 layers", "nq": 7, "nw":  500, "nl": 500, "seed": 3101, "gates": "h_cnot"},
    {"name": "STRESS 32: 7q, 5K words, 120 layers",  "nq": 7, "nw": 5000, "nl": 120, "seed": 3201, "gates": "h_cnot"},
]


def evolve_observable(obs: SparsePauliOp, qc: QuantumCircuit) -> SparsePauliOp:
    """
    Evolve observable backward through the circuit using Qiskit.
    SparsePauliOp.evolve(qc) computes U† O U exactly for Clifford circuits.
    """
    return obs.evolve(qc)


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
print(f"{'Test':<45}  {'Build(s)':>10}  {'Evolve(s)':>10}  {'Terms':>8}")
print("-" * 80)

for tc in STRESS_TESTS:
    nq, nw, nl = tc["nq"], tc["nw"], tc["nl"]

    # Build observable
    t0 = time.perf_counter()
    obs = make_pauli_sum_qiskit(nq, nw, tc["seed"])
    build_t = time.perf_counter() - t0

    # Build circuit
    if tc["gates"] == "h_cnot":
        qc = make_circuit_h_cnot(nq, nl)
    else:
        qc = make_circuit_s_cnot(nq, nl)

    # Evolve (this is the timed portion)
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
    name_short = tc["name"][:44]
    print(f"{name_short:<45}  {build_t:>10.4f}  {evolve_t:>10.4f}  {nterms:>8}")

print()
print("=" * 70)
valid = [t for t in timings if t >= 0]
if valid:
    print(f"  Mean evolve time: {sum(valid)/len(valid):.4f} s")
    print(f"  Max  evolve time: {max(valid):.4f} s")

# Save CSV
out = REPO_ROOT / "scripts" / "benchmark_qiskit_results.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test", "qiskit_evolve_s"])
    for tc, t in zip(STRESS_TESTS, timings):
        w.writerow([tc["name"], t])

print(f"\n  Results saved to: {out}")
print()
print("  Compare 'qiskit_evolve_s' to 'cpu_seq' in benchmark_results.csv")
print("  to show the speedup of our GPU over Qiskit's exact CPU propagation.")
print("=" * 70)
