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

# Build verifier exes into /tmp when the project directory is over AFS quota
def _exe_dir():
    test = REPO / ".quota_test_vc"
    try:
        test.write_text("x"); test.unlink()
        return REPO
    except OSError:
        d = Path("/tmp/pauli_verify")
        d.mkdir(exist_ok=True)
        return d

_EXE_DIR = _exe_dir()
OMP_EXE = _EXE_DIR / "verify_correctness_omp.exe"
GPU_EXE = _EXE_DIR / "verify_correctness_gpu.exe"

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
    """Run a verifier and return (passed: bool, no_device: bool)."""
    print("=" * 64)
    print(f"  RUNNING: {label}")
    print(f"  Executable: {exe}")
    print("=" * 64)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    print(r.stdout)
    no_device = False
    if r.stderr:
        print("STDERR:", r.stderr[:600])
        if "no CUDA-capable device" in r.stderr or "no cuda device" in r.stderr.lower():
            no_device = True
            print("\n  NOTE: GPU errors are due to 'no CUDA-capable device' on this node.")
            print("        This is NOT a code correctness failure.")
            print("        Re-run verify_correctness.py on a GPU node (ghc38/ghc-gpu).")
            print("        OMP results above are still valid.\n")
    return r.returncode == 0, no_device

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
    if _EXE_DIR != REPO:
        print(f"  (AFS quota exceeded — building to {_EXE_DIR})\n")
    if build_omp(cxx):
        ok, _ = run_verifier(OMP_EXE, "OMP correctness (1 / 4 / 16 threads vs CPU-seq)")
        results.append(("OMP", ok, None))
    else:
        results.append(("OMP", False, None))

    # ---- GPU ----
    if nvcc:
        print("\n>>> Step 2: Build and run GPU verifier\n")
        if build_gpu(nvcc, cxx):
            ok, no_device = run_verifier(GPU_EXE, "GPU correctness vs CPU-seq")
            if no_device:
                results.append(("GPU", True, "no CUDA device on this node — re-run on ghc38/ghc-gpu"))
            else:
                results.append(("GPU", ok, None))
        else:
            results.append(("GPU", False, None))
    else:
        print("\n>>> Step 2: SKIPPED (nvcc not found)\n")
        print("    To run GPU verification, re-run on a GPU node (ghc38/ghc-gpu).\n")
        results.append(("GPU", True, "nvcc not found — run on GPU node to verify"))

    # ---- Summary ----
    print("=" * 64)
    print("  FINAL RESULTS")
    print("=" * 64)
    all_pass = True
    for label, ok, note in results:
        if note:
            status = "\033[93mSKIP\033[0m"
            print(f"  {label:6s}: {status}  ({note})")
        else:
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
