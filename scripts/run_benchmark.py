#!/usr/bin/env python3
"""
run_benchmark.py  –  Task 1.1 benchmark runner
================================================
Runs stress tests 23-32 (indices 22-31) with:
  • CPU sequential  (pauli_propagation_cpu.exe)
  • OMP 1, 2, 4, 8, 16 threads  (pauli_propagation_omp.exe)
  • GPU  (pauli_propagation_gpu.exe)   [optional]

Outputs timing tables and speedup rows suitable for the paper.

Usage (from repo root on GHC / any Linux machine with the executables built):
    python3 scripts/run_benchmark.py [--no-gpu] [--tests 23-32]

Build the executables first:
    cd src
    make cpu omp        # no CUDA needed
    make gpu            # requires nvcc + compatible GCC
"""

import argparse
import subprocess
import sys
import os
import csv
import re
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent.parent
CPU_EXE    = REPO_ROOT / "pauli_propagation_cpu.exe"
OMP_EXE    = REPO_ROOT / "pauli_propagation_omp.exe"
GPU_EXE    = REPO_ROOT / "pauli_propagation_gpu.exe"

# Command-line index mapping (0-based vector indices):
#   indices 0-21  → basic tests 1-22
#   index  22     → MultiBlock A
#   index  23     → MultiBlock B
#   indices 24-33 → STRESS 23 through STRESS 32
DEFAULT_STRESS_TESTS = list(range(24, 34))   # STRESS 23-32 at vector indices 24-33
OMP_THREAD_COUNTS    = [1, 2, 4, 8, 16]
TIMEOUT_SECONDS      = 300                   # 5 min per test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_test(exe: Path, test_idx: int, mode: str, threads: int = 0,
             timeout: int = TIMEOUT_SECONDS) -> float:
    """
    Run a single test and return wall-clock compute time in seconds.
    Returns -1.0 on failure / timeout.
    """
    if not exe.exists():
        print(f"  [SKIP] Executable not found: {exe}")
        return -1.0

    if mode == "omp":
        cmd = [str(exe), str(test_idx), "omp", "-j", str(threads)]
    elif mode == "gpu":
        cmd = [str(exe), str(test_idx), "gpu"]
    else:
        cmd = [str(exe), str(test_idx), "cpu"]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] test {test_idx} {mode} j={threads} > {timeout}s")
        return -1.0
    except Exception as e:
        print(f"  [ERROR] {e}")
        return -1.0

    # Parse "Propagation completed in X.XXXXXX seconds"
    match = re.search(r"Propagation completed in\s+([\d.eE+\-]+)\s+seconds", output)
    if match:
        return float(match.group(1))

    # Fallback: parse "Time (ms)" table row for this test
    match2 = re.search(r"[\d.]+\s*$", output, re.MULTILINE)
    if match2:
        try:
            return float(match2.group(0)) / 1000.0
        except ValueError:
            pass

    print(f"  [PARSE ERROR] Could not extract time from output:\n{output[-400:]}")
    return -1.0


def fmt(t: float, width: int = 9) -> str:
    if t < 0:
        return f"{'N/A':>{width}}"
    return f"{t:>{width}.3f}"


def speedup(base: float, t: float) -> str:
    if base <= 0 or t <= 0:
        return "   N/A"
    return f"{base / t:>6.1f}x"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pauli propagation three-way benchmark")
    ap.add_argument("--no-gpu",  action="store_true", help="Skip GPU runs")
    ap.add_argument("--no-cpu",  action="store_true", help="Skip sequential CPU runs")
    ap.add_argument("--tests", default="24-33",
                    help="0-based vector indices, e.g. '24-33' (STRESS 23-32) or '24,26,31'")
    ap.add_argument("--threads", default=",".join(map(str, OMP_THREAD_COUNTS)),
                    help="Comma-separated OMP thread counts")
    ap.add_argument("--out", default="benchmark_results.csv",
                    help="Output CSV file (written to scripts/)")
    args = ap.parse_args()

    # Parse test list
    # Indices are 0-based command-line arguments (vector indices into test_cases).
    # Default 24-33 = STRESS tests 23-32 (see tests.cpp).
    if "-" in args.tests:
        lo, hi = args.tests.split("-")
        tests = list(range(int(lo), int(hi) + 1))
    else:
        tests = [int(x) for x in args.tests.split(",")]

    thread_counts = [int(x) for x in args.threads.split(",")]

    print("=" * 70)
    print("  PAULI PROPAGATION THREE-WAY BENCHMARK")
    print(f"  Tests: {tests}")
    print(f"  OMP thread counts: {thread_counts}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check executables
    for label, exe in [("CPU", CPU_EXE), ("OMP", OMP_EXE), ("GPU", GPU_EXE)]:
        status = "✓" if exe.exists() else "✗ MISSING"
        print(f"  {label:4s}: {exe}  {status}")
    print()

    # ------------------------------------------------------------------ Run
    results = {}   # results[test_idx][config_key] = seconds

    for t in tests:
        results[t] = {}

        # CPU sequential
        if not args.no_cpu:
            print(f"[Test {t}] CPU sequential ...")
            results[t]["cpu_seq"] = run_test(CPU_EXE, t, "cpu")
            print(f"         {results[t]['cpu_seq']:.4f} s")

        # OMP
        for j in thread_counts:
            key = f"omp_{j}"
            print(f"[Test {t}] OMP {j:>2d} threads ...")
            results[t][key] = run_test(OMP_EXE, t, "omp", threads=j)
            print(f"         {results[t][key]:.4f} s")

        # GPU
        if not args.no_gpu:
            print(f"[Test {t}] GPU ...")
            results[t]["gpu"] = run_test(GPU_EXE, t, "gpu")
            print(f"         {results[t]['gpu']:.4f} s")

        print()

    # ---------------------------------------------------------- Print table
    print()
    print("=" * 100)
    print("  RESULTS TABLE  (wall-clock compute time, seconds)")
    print("=" * 100)

    # Header
    header_parts = ["Test"]
    if not args.no_cpu:
        header_parts.append("CPU-seq")
    for j in thread_counts:
        header_parts.append(f"OMP-{j}t")
    if not args.no_gpu:
        header_parts.append("GPU")
    if not args.no_cpu:
        for j in thread_counts:
            header_parts.append(f"S(seq/{j}t)")
    if not args.no_gpu and not args.no_cpu:
        header_parts.append("S(seq/GPU)")
    if not args.no_gpu:
        for j in thread_counts:
            header_parts.append(f"S({j}t/GPU)")

    col_w = 10
    print(" | ".join(f"{h:>{col_w}}" for h in header_parts))
    print("-" * (col_w * len(header_parts) + 3 * (len(header_parts) - 1)))

    for t in tests:
        row = [f"{t:>{col_w}}"]
        r = results[t]
        cpu_t = r.get("cpu_seq", -1.0)
        gpu_t = r.get("gpu",     -1.0)

        if not args.no_cpu:
            row.append(fmt(cpu_t, col_w))
        for j in thread_counts:
            row.append(fmt(r.get(f"omp_{j}", -1.0), col_w))
        if not args.no_gpu:
            row.append(fmt(gpu_t, col_w))

        # Speedups
        if not args.no_cpu:
            for j in thread_counts:
                row.append(f"{speedup(cpu_t, r.get(f'omp_{j}', -1.0)):>{col_w}}")
        if not args.no_gpu and not args.no_cpu:
            row.append(f"{speedup(cpu_t, gpu_t):>{col_w}}")
        if not args.no_gpu:
            for j in thread_counts:
                row.append(f"{speedup(r.get(f'omp_{j}', -1.0), gpu_t):>{col_w}}")

        print(" | ".join(row))

    # ---------------------------------------------------------- Save CSV
    out_path = REPO_ROOT / "scripts" / args.out
    configs = []
    if not args.no_cpu:
        configs.append("cpu_seq")
    for j in thread_counts:
        configs.append(f"omp_{j}")
    if not args.no_gpu:
        configs.append("gpu")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + configs)
        for t in tests:
            row = [t] + [results[t].get(c, -1.0) for c in configs]
            writer.writerow(row)

    print()
    print(f"Results saved to: {out_path}")
    print()

    # -------------------------------------------------- Summary speedup table
    print("=" * 70)
    print("  AVERAGE SPEEDUP SUMMARY")
    print("=" * 70)
    valid = [t for t in tests if results[t].get("cpu_seq", -1) > 0]
    if valid:
        print(f"  Baseline: CPU sequential ({len(valid)} tests)")
        for j in thread_counts:
            key = f"omp_{j}"
            sp_vals = [results[t]["cpu_seq"] / results[t][key]
                       for t in valid if results[t].get(key, -1) > 0]
            if sp_vals:
                avg = sum(sp_vals) / len(sp_vals)
                print(f"  OMP {j:>2d} threads vs CPU-seq:  {avg:.1f}x avg speedup")

        if not args.no_gpu:
            for j in thread_counts:
                key = f"omp_{j}"
                gpu_vals = [(results[t].get(key, -1), results[t].get("gpu", -1))
                            for t in valid]
                sp_vals = [o / g for o, g in gpu_vals if o > 0 and g > 0]
                if sp_vals:
                    avg = sum(sp_vals) / len(sp_vals)
                    print(f"  GPU vs OMP {j:>2d} threads:      {avg:.1f}x avg speedup")

            cpu_gpu = [results[t]["cpu_seq"] / results[t]["gpu"]
                       for t in valid if results[t].get("gpu", -1) > 0]
            if cpu_gpu:
                print(f"  GPU vs CPU-seq:             {sum(cpu_gpu)/len(cpu_gpu):.1f}x avg speedup")


if __name__ == "__main__":
    main()
