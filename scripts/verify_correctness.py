#!/usr/bin/env python3
"""
verify_correctness.py
=====================
Builds and runs the correctness verifier (src/verify_correctness.cpp) for
both the OpenMP CPU baseline and (when CUDA is available) the GPU.

Usage – run from the repo root on any machine:
    python3 scripts/verify_correctness.py

On a GHC machine with nvcc, both OMP and GPU are verified.
On a Windows laptop without nvcc, only OMP is verified.

Output:
  Prints PASS/FAIL per test case per implementation.
  Exits with code 0 if everything passed, 1 otherwise.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

REPO   = Path(__file__).resolve().parent.parent
SRC    = REPO / "src"
OMP_EXE = REPO / "verify_correctness_omp.exe"
GPU_EXE = REPO / "verify_correctness_gpu.exe"

# ---------------------------------------------------------------------------
# Compiler detection
# ---------------------------------------------------------------------------
def have(cmd):
    return shutil.which(cmd) is not None

def detect_cxx():
    """Return best available g++ that CUDA supports."""
    for v in ["g++-11", "g++-10", "g++-9", "g++"]:
        if have(v):
            return v
    return None

def detect_nvcc():
    return "nvcc" if have("nvcc") else None

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
BASE_FLAGS = ["-std=c++17", "-O2", "-I" + str(SRC)]

def build_omp(cxx):
    flags = BASE_FLAGS + ["-DCPU_ONLY", "-DOMP_ENABLED", "-fopenmp"]
    sources = ["pauli.cpp", "pauli_omp.cpp", "verify_correctness.cpp"]
    cmd = [cxx] + flags + sources + ["-o", str(OMP_EXE)]
    print(f"Building OMP verifier: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SRC))
    if r.returncode != 0:
        print("BUILD FAILED:\n", r.stderr)
        return False
    print(f"  -> {OMP_EXE}\n")
    return True

def build_gpu(nvcc, cxx):
    # nvcc needs the C++ compiler that produced the .o files to be compatible.
    # Use -Xcompiler to pass -fopenmp to the host compiler.
    flags = BASE_FLAGS + ["-DOMP_ENABLED",
                          "-ccbin", cxx,
                          "-Xcompiler", "-fopenmp"]
    sources = ["pauli.cpp", "pauli_omp.cpp", "pauli_gpu.cu", "verify_correctness.cpp"]
    cmd = [nvcc] + flags + sources + ["-o", str(GPU_EXE)]
    print(f"Building GPU verifier: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SRC))
    if r.returncode != 0:
        print("BUILD FAILED:\n", r.stderr)
        return False
    print(f"  -> {GPU_EXE}\n")
    return True

# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
def run_verifier(exe, label):
    print("=" * 64)
    print(f"  RUNNING: {label}")
    print(f"  Executable: {exe}")
    print("=" * 64)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[:400])
    return r.returncode == 0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cxx  = detect_cxx()
    nvcc = detect_nvcc()

    print("=" * 64)
    print("  PAULI PROPAGATION CORRECTNESS VERIFICATION")
    print(f"  C++ compiler : {cxx or 'NOT FOUND'}")
    print(f"  nvcc         : {nvcc or 'NOT FOUND (GPU skip)'}")
    print("=" * 64 + "\n")

    if not cxx:
        print("ERROR: no g++ compiler found. Install GCC.")
        sys.exit(1)

    results = []

    # ---- OMP ----
    print(">>> Step 1: Build and run OMP verifier\n")
    if build_omp(cxx):
        ok = run_verifier(OMP_EXE, "OMP correctness (1 / 4 / 16 threads vs CPU-seq)")
        results.append(("OMP", ok))
    else:
        results.append(("OMP", False))

    # ---- GPU ----
    if nvcc:
        print("\n>>> Step 2: Build and run GPU verifier\n")
        if build_gpu(nvcc, cxx):
            ok = run_verifier(GPU_EXE, "GPU correctness vs CPU-seq")
            results.append(("GPU", ok))
        else:
            results.append(("GPU", False))
    else:
        print("\n>>> Step 2: SKIPPED (nvcc not found)\n")
        print("    To run GPU verification, install CUDA and re-run on a GHC machine.\n")

    # ---- Summary ----
    print("=" * 64)
    print("  FINAL RESULTS")
    print("=" * 64)
    all_pass = True
    for label, ok in results:
        status = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
        print(f"  {label:6s}: {status}")
        all_pass &= ok
    print()

    if all_pass:
        print("  All verifiers PASSED.")
    else:
        print("  One or more verifiers FAILED – see output above.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
