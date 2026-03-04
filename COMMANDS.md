# Parallel Pauli Paths — Commands Reference

> **One-stop reference for building, verifying, benchmarking, and generating
> paper figures.** Run everything from the **repo root** unless noted.

---

## 0. TL;DR Cheat-Sheet

```bash
# ── Build (GHC) ──────────────────────────────────────────────────────────
cd src && make clean && make all && cd ..       # CPU + OMP + GPU

# ── Correctness (run first, always) ─────────────────────────────────────
python3 scripts/verify_correctness.py          # OMP on any machine; GPU on GHC

# ── Paper benchmark table ────────────────────────────────────────────────
python3 scripts/run_benchmark.py               # CPU-seq vs OMP-1/4/8/16 vs GPU

# ── External-tool comparison ─────────────────────────────────────────────
julia  scripts/benchmark_julia.jl             # PauliPropagation.jl
python3 scripts/benchmark_qiskit.py           # Qiskit SparsePauliOp

# ── Paper figures ────────────────────────────────────────────────────────
python3 scripts/generate_all_figures.py
```

---

## 1. Prerequisites

| What | Where needed | Install |
|------|-------------|---------|
| g++ 11+ (C++17) | CPU, OMP builds | `sudo apt install g++-11` |
| OpenMP (`-fopenmp`) | OMP build | ships with GCC |
| CUDA 11.7 + nvcc | GPU build | GHC machines only |
| Python 3.8+ | all scripts | system Python |
| matplotlib | figure scripts | `pip install matplotlib` |
| Julia 1.9+ | Task 1.2 Option A | `sudo apt install julia` |
| PauliPropagation.jl | Task 1.2 Option A | `julia -e 'using Pkg; Pkg.add("PauliPropagation")'` |
| Qiskit | Task 1.2 Option B | `pip install qiskit` |

---

## 2. Build

All build targets are in **`src/Makefile`**. Run from `src/`.

```bash
cd src

make cpu      # → ../pauli_propagation_cpu.exe  (no CUDA needed)
make omp      # → ../pauli_propagation_omp.exe  (no CUDA needed, needs -fopenmp)
make gpu      # → ../pauli_propagation_gpu.exe  (requires nvcc + CUDA)
make all      # → all three above
make clean    # remove all build artifacts
```

> **Windows (no make):** use the Python build helper instead:
> ```powershell
> python scripts/build.py cpu   # build CPU
> python scripts/build.py omp   # build OMP
> ```

### Manual compilation (fallback)

```bash
# CPU
g++ -std=c++17 -O2 -DCPU_ONLY -I src \
    src/pauli.cpp src/tests.cpp src/main.cpp \
    -o pauli_propagation_cpu.exe

# OMP
g++ -std=c++17 -O2 -DCPU_ONLY -DOMP_ENABLED -fopenmp -I src \
    src/pauli.cpp src/pauli_omp.cpp src/tests.cpp src/main.cpp \
    -o pauli_propagation_omp.exe

# GPU (GHC)
cd src && make gpu && cd ..
```

---

## 3. Test Index Reference

The test suite has **34 tests** (vector indices 0–33):

| CLI index | Display # | Name | Category |
|-----------|-----------|------|----------|
| 0–21 | 1–22 | Hadamard, Bell, GHZ, rotations, … | Basic correctness |
| 22 | 23 | MultiBlock A: No rotations | Integration |
| 23 | 24 | MultiBlock B: with rotations | Integration |
| **24** | **25** | **STRESS 23: 7q, 2K words, 100 layers** | Stress |
| 25 | 26 | STRESS 24: 7q, 5K words, 150 layers | Stress |
| 26 | 27 | STRESS 25: 7q, 3K words, 200 layers | Stress |
| 27 | 28 | STRESS 26: 7q, 1K words, 300 layers | Stress |
| 28 | 29 | STRESS 27: 7q, 4K words, 100 layers | Stress |
| 29 | 30 | STRESS 28: 7q, 2K words, 250 layers | Stress |
| 30 | 31 | STRESS 29: 7q, 1K words, 400 layers | Stress |
| 31 | 32 | STRESS 30: 7q, 8K words,  50 layers | Stress |
| 32 | 33 | STRESS 31: 7q, 500 words, 500 layers | Stress |
| **33** | **34** | **STRESS 32: 7q, 5K words, 120 layers** | Stress |

**Usage:** pass the CLI index directly:

```bash
./pauli_propagation_cpu.exe 24 cpu    # STRESS 23, sequential CPU
./pauli_propagation_omp.exe 24 omp -j 16  # STRESS 23, 16 OMP threads
./pauli_propagation_gpu.exe 24 gpu    # STRESS 23, GPU
./pauli_propagation_cpu.exe all cpu   # run all 34 tests
```

---

## 4. Correctness Verification

### 4a. Full numerical verifier (C++ — most rigorous)

Directly calls `pauli_propagation` (CPU reference), `pauli_propagation_omp`, and
optionally `PauliSimulatorGPU` and compares expectation values for 20 test cases.

```bash
# Auto-builds and runs both OMP and GPU variants:
python3 scripts/verify_correctness.py

# Expected output:
#   OMP checks : 60/60  (100.0%)   ← on any machine
#   GPU checks : 60/60  (100.0%)   ← only on GHC with CUDA
```

Manual build + run:

```bash
# OMP verifier (Windows or Linux):
g++ -std=c++17 -O2 -DCPU_ONLY -DOMP_ENABLED -fopenmp -I src \
    src/pauli.cpp src/pauli_omp.cpp src/verify_correctness.cpp \
    -o verify_correctness_omp.exe
./verify_correctness_omp.exe

# GPU verifier (GHC only):
cd src
nvcc -std=c++17 -O2 -DOMP_ENABLED -Xcompiler -fopenmp -ccbin g++-11 -I. \
    pauli.cpp pauli_omp.cpp pauli_gpu.cu verify_correctness.cpp \
    -o ../verify_correctness_gpu.exe
cd ..
./verify_correctness_gpu.exe
```

### 4b. Smoke test (Python — fast, 1–2 min)

Runs all 34 built-in tests on the OMP executable, verifies no crashes:

```bash
python3 scripts/verify_omp_correctness.py          # full (all 34 tests)
python3 scripts/verify_omp_correctness.py --quick  # basic only (tests 0–21)
```

### 4c. Run full test suite on each executable

```bash
./pauli_propagation_cpu.exe all cpu          # sequential CPU  (34/34 expected)
./pauli_propagation_omp.exe all omp -j 16   # OMP 16-thread   (34/34 expected)
./pauli_propagation_gpu.exe all gpu          # GPU             (34/34 expected)
```

---

## 5. Performance Benchmarks (Paper Table)

### 5a. Primary benchmark: CPU-seq vs OMP vs GPU

Runs stress tests 23–32 (CLI indices 24–33) across all configurations and writes
timing data to `scripts/benchmark_results.csv`.

```bash
python3 scripts/run_benchmark.py                  # full run (may take 30 min)
python3 scripts/run_benchmark.py --no-gpu         # skip GPU (Windows)
python3 scripts/run_benchmark.py --tests 24-28    # subset of tests
```

Output includes:
- Wall-clock times for CPU-seq, OMP-1/2/4/8/16, GPU
- Speedup columns: `S(seq/Nt)`, `S(seq/GPU)`, `S(Nt/GPU)`
- Average speedup summary

### 5b. Single test, single mode

```bash
./pauli_propagation_cpu.exe 24 cpu              # STRESS 23, CPU-seq
./pauli_propagation_omp.exe 24 omp -j 1         # STRESS 23, OMP 1-thread
./pauli_propagation_omp.exe 24 omp -j 4         # STRESS 23, OMP 4-thread
./pauli_propagation_omp.exe 24 omp -j 8         # STRESS 23, OMP 8-thread
./pauli_propagation_omp.exe 24 omp -j 16        # STRESS 23, OMP 16-thread
./pauli_propagation_gpu.exe 24 gpu              # STRESS 23, GPU
```

### 5c. Thread scaling sweep (OMP only)

```bash
for j in 1 2 4 8 16; do
    echo "=== OMP $j threads ==="
    for idx in 24 25 26 27 28 29 30 31 32 33; do
        ./pauli_propagation_omp.exe $idx omp -j $j
    done
done
```

---

## 6. External Tool Comparison (Task 1.2)

### Option A — PauliPropagation.jl (preferred)

```bash
# Install once:
julia -e 'using Pkg; Pkg.add("PauliPropagation")'

# Benchmark (same 10 circuits as STRESS 23–32):
julia scripts/benchmark_julia.jl
```

Output goes to stdout. Compare reported times to the `cpu_seq` column in
`scripts/benchmark_results.csv`.

### Option B — Qiskit SparsePauliOp

```bash
# Install once:
pip install qiskit

# Benchmark:
python3 scripts/benchmark_qiskit.py
# → writes scripts/benchmark_qiskit_results.csv
```

> **Note:** Qiskit's `evolve()` is *exact* (no weight truncation). For
> Clifford-only circuits (STRESS 23–32) results are directly comparable.

---

## 7. Figure Generation

```bash
# All figures at once (recommended):
python3 scripts/generate_all_figures.py

# Individual panels:
python3 scripts/performance_analysis.py       # speedup_analysis.png, parameter_analysis.png
python3 scripts/generate_report_figures.py    # timing_comparison.png, speedup_chart.png
python3 scripts/algorithmic_visualization.py  # clifford_analysis.png, pauli_evolution.png
python3 scripts/correctness_validation.py     # correctness_validation.png
```

Figures are saved to the repo root as `.png` files.

### Copy figures to local machine (from GHC)

```bash
# Run this on YOUR LOCAL machine:
scp arulm@ghc43.ghc.andrew.cmu.edu:~/Parallelizing-Pauli-Paths/*.png ./images/
```

---

## 8. GHC Workflow (End-to-End)

```bash
# 1. SSH to GHC
ssh arulm@ghc43.ghc.andrew.cmu.edu
cd ~/Parallelizing-Pauli-Paths

# 2. Build everything
cd src && make clean && make all && cd ..

# 3. Verify correctness first
python3 scripts/verify_correctness.py

# 4. Run benchmark (saves CSV)
python3 scripts/run_benchmark.py

# 5. External comparison
julia  scripts/benchmark_julia.jl   2>&1 | tee results_julia.txt
python3 scripts/benchmark_qiskit.py 2>&1 | tee results_qiskit.txt

# 6. Generate figures
python3 scripts/generate_all_figures.py

# 7. From LOCAL machine — pull everything
scp arulm@ghc43.ghc.andrew.cmu.edu:~/Parallelizing-Pauli-Paths/*.png ./images/
scp arulm@ghc43.ghc.andrew.cmu.edu:~/Parallelizing-Pauli-Paths/scripts/benchmark_results.csv ./
scp arulm@ghc43.ghc.andrew.cmu.edu:~/Parallelizing-Pauli-Paths/results_*.txt ./
```

---

## 9. Source File Map

```
src/
├── pauli.h / pauli.cpp          # Sequential CPU implementation (std::map)
├── pauli_omp.h / pauli_omp.cpp  # OpenMP CPU baseline (unordered_map + OMP)
├── pauli_gpu.h / pauli_gpu.cu   # CUDA GPU implementation
├── gates.cu_inl                 # Device-side gate conjugation
├── tests.h / tests.cpp          # 34-test suite (basic + stress)
├── main.cpp                     # CLI entry point
├── verify_correctness.cpp       # Numerical correctness checker (OMP + GPU)
├── CycleTimer.h                 # High-res timer
├── exclusiveScan.cu_inl         # CUDA prefix-scan helper
└── Makefile                     # Build targets: cpu, omp, gpu, all, clean

scripts/
├── verify_correctness.py        # ★ Build + run correctness verifier
├── verify_omp_correctness.py    # OMP smoke test (all 34 tests)
├── run_benchmark.py             # ★ CPU-seq / OMP / GPU timing table
├── benchmark_julia.jl           # PauliPropagation.jl comparison
├── benchmark_qiskit.py          # Qiskit SparsePauliOp comparison
├── generate_all_figures.py      # Run all figure scripts at once
├── generate_report_figures.py   # timing + speedup charts
├── performance_analysis.py      # speedup vs parameters
├── algorithmic_visualization.py # Pauli word dynamics
├── correctness_validation.py    # Correctness plots (legacy)
├── quick_test_stress.py         # Quick CPU vs GPU check (legacy)
└── speedup_analysis.py          # Speedup deep-dive (legacy)
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `pauli_propagation_*.exe not found` | `cd src && make <target> && cd ..` |
| `nvcc not found` | Must run on GHC. Check `which nvcc`. |
| `openmp: command not found` | Install: `sudo apt install libomp-dev` |
| OMP binary always uses 1 thread | Pass `-j N` flag: `./pauli_propagation_omp.exe 24 omp -j 16` |
| Wrong test runs | Use CLI index (0-based). STRESS 23 = index **24**. |
| GPU returns `(-1,-1)` | Word count exceeds GPU capacity (`MAX_PAULI_WORDS × MAX_BLOCKS = 204,800`). Use smaller test. |
| `julia: command not found` | Install Julia from [julialang.org](https://julialang.org/downloads/) |
| Qiskit `evolve` very slow | Expected — it's exact (no truncation). Use small test. |
| Plots not showing on GHC | Expected — use `scp` to copy `.png` files locally. |
| Matplotlib not installed | `pip install --user matplotlib` |
| **`Disk quota exceeded` on git pull (GHC)** | See **§10.1** below. |

### 10.1 GHC: Fix "Disk quota exceeded" on git pull

AFS home is over quota, so `git pull` cannot write temp files. **Pull in `/tmp` (no quota), then copy files into your project dir.** Run these commands **on the GHC machine** (e.g. in your SSH session). Replace `YOUR_AFS_PROJECT_DIR` with your actual path (e.g. `$HOME/private/15418/Parallelizing-Pauli-Paths`).

```bash
# 1) Clone repo into /tmp (no AFS quota)
rm -rf /tmp/ppp-pull
git clone --depth 1 https://github.com/arulrhikm/Parallelizing-Pauli-Paths.git /tmp/ppp-pull

# 2) Copy updated files into your AFS project (do NOT copy .git — avoids writing to AFS)
rsync -av --exclude='.git' /tmp/ppp-pull/ YOUR_AFS_PROJECT_DIR/

# 3) Optional: free more space so future git pull works in AFS
cd YOUR_AFS_PROJECT_DIR
rm -rf build build_cpu build_omp src/build src/build_cpu src/build_omp
rm -f *.exe src/*.exe pauli_propagation_*.exe
```

One-liner (paste and replace the path once):

```bash
git clone --depth 1 https://github.com/arulrhikm/Parallelizing-Pauli-Paths.git /tmp/ppp-pull && rsync -av --exclude='.git' /tmp/ppp-pull/ ~/private/15418/Parallelizing-Pauli-Paths/
```

After this, your **source tree** is up to date. To fix `git status` later (after freeing quota): `cd YOUR_AFS_PROJECT_DIR && git fetch origin && git reset --hard origin/main`.
