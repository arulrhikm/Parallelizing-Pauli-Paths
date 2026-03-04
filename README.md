# Parallelizing Pauli Paths (P3)

GPU-accelerated Pauli path propagation for quantum circuit simulation.

**Results:** Up to 626× speedup over single-threaded CPU (average 213×)

## Quick Start

> **Full command reference:** see [`COMMANDS.md`](COMMANDS.md).

```bash
# 1. Build (GHC)
cd src && make clean && make all && cd ..

# 2. Verify correctness
python3 scripts/verify_correctness.py

# 3. Benchmark for paper table
python3 scripts/run_benchmark.py

# 4. External tool comparison
julia   scripts/benchmark_julia.jl        # PauliPropagation.jl
python3 scripts/benchmark_qiskit.py       # Qiskit

# 5. Generate figures
python3 scripts/generate_all_figures.py
```

## Executables

After running `make all` in `src/`, three executables appear in the project root:

| Executable | Description | Needs CUDA? |
|---|---|---|
| `pauli_propagation_cpu.exe` | Sequential CPU (`std::map`) | No |
| `pauli_propagation_omp.exe` | OpenMP multi-thread (`unordered_map`) | No |
| `pauli_propagation_gpu.exe` | CUDA GPU (GHC only) | Yes |

**Usage:**
```bash
./pauli_propagation_cpu.exe <test_idx> cpu
./pauli_propagation_omp.exe <test_idx> omp -j <threads>
./pauli_propagation_gpu.exe <test_idx> gpu
./pauli_propagation_cpu.exe all cpu    # run all 34 tests
```

**Test index reference** (see [`COMMANDS.md §3`](COMMANDS.md#3-test-index-reference)):

| CLI index | Test name |
|---|---|
| 0–21 | Basic correctness tests |
| 22–23 | MultiBlock integration tests |
| **24–33** | **STRESS 23–32 (main benchmark suite)** |

## Project Structure

```
src/
├── pauli.cpp / pauli.h           # Sequential CPU
├── pauli_omp.cpp / pauli_omp.h   # OpenMP CPU baseline
├── pauli_gpu.cu / pauli_gpu.h    # CUDA GPU kernel
├── tests.cpp / tests.h           # 34-test suite
├── main.cpp                      # CLI entry point
├── verify_correctness.cpp        # Numerical OMP+GPU verifier
└── Makefile

scripts/
├── verify_correctness.py   ★ correctness check (OMP + GPU)
├── run_benchmark.py        ★ paper timing table (CPU/OMP/GPU)
├── benchmark_julia.jl        PauliPropagation.jl comparison
├── benchmark_qiskit.py       Qiskit SparsePauliOp comparison
├── build.py                  Cross-platform build helper
└── generate_all_figures.py   All paper figures at once
```

## Troubleshooting

See [`COMMANDS.md §10`](COMMANDS.md#10-troubleshooting) for a full table. Quick reference:

- **Executable not found:** `cd src && make <target> && cd ..`
- **nvcc not found:** must run on GHC (`ghc43.ghc.andrew.cmu.edu`)
- **Wrong test runs:** CLI index is 0-based; STRESS 23 = index **24**
- **Plots not showing on GHC:** copy with `scp *.png ./` from local machine

## CPU-Only Build (No GPU)

```bash
# With make:
cd src && make cpu && cd ..

# Without make (Windows):
python3 scripts/build.py cpu
```

## Requirements

- CUDA toolkit (for GPU execution, tested on RTX 2080)
- C++17 compiler
- Python 3.6+ with matplotlib

## Authors

- Arul Rhik Mazumder (arulm)
- Daniel Ragazzo (dragazzo)

15-418/618 Parallel Computer Architecture and Programming - Carnegie Mellon University
