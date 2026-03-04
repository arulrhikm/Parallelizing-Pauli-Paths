#!/usr/bin/env python3
"""
build.py  –  Cross-platform build helper (use when `make` is unavailable)
=========================================================================
Replicates the src/Makefile targets for Windows and systems without make.

Usage (from repo root):
    python3 scripts/build.py           # build cpu + omp
    python3 scripts/build.py cpu       # sequential CPU only
    python3 scripts/build.py omp       # OpenMP CPU only
    python3 scripts/build.py gpu       # GPU (requires nvcc, GHC only)
    python3 scripts/build.py all       # cpu + omp + gpu
    python3 scripts/build.py clean     # remove executables + build dirs
    python3 scripts/build.py verify    # build + run correctness verifier
"""

import subprocess, sys, shutil, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"

# ---------------------------------------------------------------------------
# Compiler detection
# ---------------------------------------------------------------------------
def cxx():
    for v in ["g++-11", "g++-10", "g++-9", "g++"]:
        if shutil.which(v): return v
    return None

def nvcc_path():
    return "nvcc" if shutil.which("nvcc") else None

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
BASE  = ["-std=c++17", "-Wall", "-O2"]
ISRC  = [f"-I{SRC}"]

TARGETS = {
    "cpu": {
        "exe":     ROOT / "pauli_propagation_cpu.exe",
        "flags":   BASE + ISRC + ["-DCPU_ONLY"],
        "sources": ["pauli.cpp", "tests.cpp", "main.cpp"],
        "linker":  [],
        "build_dir": SRC / "build_cpu",
    },
    "omp": {
        "exe":     ROOT / "pauli_propagation_omp.exe",
        "flags":   BASE + ISRC + ["-DCPU_ONLY", "-DOMP_ENABLED", "-fopenmp"],
        "sources": ["pauli.cpp", "pauli_omp.cpp", "tests.cpp", "main.cpp"],
        "linker":  ["-fopenmp", "-lgomp"],
        "build_dir": SRC / "build_omp",
    },
}

def build_target(name: str) -> bool:
    if name == "gpu":
        return build_gpu()
    cfg = TARGETS[name]
    compiler = cxx()
    if not compiler:
        print("ERROR: no g++ compiler found."); return False

    cfg["build_dir"].mkdir(exist_ok=True)
    objs = []

    # Compile each source to .o
    for src_name in cfg["sources"]:
        src_file = SRC / src_name
        obj_file = cfg["build_dir"] / src_name.replace(".cpp", ".o")
        cmd = [compiler] + cfg["flags"] + ["-c", str(src_file), "-o", str(obj_file)]
        print(f"  CC  {src_name}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED: {r.stderr}"); return False
        objs.append(str(obj_file))

    # Link
    exe = cfg["exe"]
    cmd = [compiler] + cfg["flags"] + cfg["linker"] + objs + ["-o", str(exe)]
    print(f"  LD  {exe.name}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"LINK FAILED: {r.stderr}"); return False

    print(f"  [OK] Built: {exe}")
    return True


def build_gpu() -> bool:
    nvcc = nvcc_path()
    if not nvcc:
        print("ERROR: nvcc not found. GPU build requires CUDA (run on GHC)."); return False
    compiler = cxx()
    if not compiler:
        print("ERROR: no g++ found."); return False

    exe = ROOT / "pauli_propagation_gpu.exe"
    build_dir = SRC / "build"
    build_dir.mkdir(exist_ok=True)

    flags = ["-std=c++17", "-O2", f"-ccbin {compiler}", f"-I{SRC}"]

    # Compile GPU sources
    gpu_obj = build_dir / "pauli_gpu.o"
    r = subprocess.run([nvcc] + flags + ["-c", str(SRC / "pauli_gpu.cu"), "-o", str(gpu_obj)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CUDA COMPILE FAILED: {r.stderr}"); return False
    print("  CC  pauli_gpu.cu")

    # Compile C++ sources
    cpp_flags = ["-std=c++17", "-O2", f"-I{SRC}"]
    objs = [str(gpu_obj)]
    for src_name in ["pauli.cpp", "tests.cpp", "main.cpp"]:
        obj = build_dir / src_name.replace(".cpp", ".o")
        r = subprocess.run([compiler] + cpp_flags + ["-c", str(SRC / src_name), "-o", str(obj)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED: {r.stderr}"); return False
        print(f"  CC  {src_name}")
        objs.append(str(obj))

    # Link with nvcc
    r = subprocess.run([nvcc] + flags + objs + ["-o", str(exe)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"LINK FAILED: {r.stderr}"); return False
    print(f"  [OK] Built: {exe}")
    return True


def clean():
    for d in ["build_cpu", "build_omp", "build"]:
        p = SRC / d
        if p.exists():
            shutil.rmtree(p); print(f"  rm  {p}")
    for exe_name in ["pauli_propagation_cpu.exe", "pauli_propagation_omp.exe",
                     "pauli_propagation_gpu.exe", "verify_correctness_omp.exe",
                     "verify_correctness_gpu.exe"]:
        p = ROOT / exe_name
        if p.exists():
            p.unlink(); print(f"  rm  {p.name}")
    print("  [OK] Clean complete")


def run_verify():
    import importlib.util, os
    script = ROOT / "scripts" / "verify_correctness.py"
    os.chdir(ROOT)
    r = subprocess.run([sys.executable, str(script)])
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    targets = sys.argv[1:] or ["cpu", "omp"]

    if "clean" in targets:
        clean(); return

    if "verify" in targets:
        targets = [t for t in targets if t != "verify"]
        extra_verify = True
    else:
        extra_verify = False

    if "all" in targets:
        targets = ["cpu", "omp", "gpu"]

    print("=" * 50)
    all_ok = True
    for t in targets:
        if t not in ("cpu", "omp", "gpu"):
            print(f"Unknown target: {t}. Valid: cpu, omp, gpu, all, clean, verify")
            sys.exit(1)
        print(f"Building: {t}")
        ok = build_target(t)
        all_ok &= ok
        print()

    if extra_verify and all_ok:
        all_ok &= run_verify()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
