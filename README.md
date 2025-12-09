# Parallelizing Pauli Paths (P3)

GPU-accelerated Pauli path propagation for quantum circuit simulation.

**Results:** Up to 626× speedup over single-threaded CPU (average 213×)

## Project Structure

```
├── src/                 # C++/CUDA source code
│   ├── pauli.cpp        # CPU implementation
│   ├── pauli_gpu.cu     # GPU CUDA kernel
│   ├── gates.cu_inl     # Device gate operations
│   ├── tests.cpp        # Test suite (34+ tests)
│   └── Makefile         # Build configuration
├── scripts/             # Python visualization & benchmark scripts
├── web/                 # Project website files
│   ├── index.html       # Project homepage
│   ├── milestone.html   # Milestone report
│   ├── proposal.html    # Project proposal
│   ├── report.html      # Final report (HTML)
│   └── styles.css       # Stylesheet
├── images/              # Generated figures
└── final_report.tex     # Final report (LaTeX)
```

## Prerequisites

**For CPU-only version:**
- g++ compiler with C++17 support
- Standard C++ libraries

**For GPU version (requires Linux/SSH to ghc27):**
- CUDA Toolkit (tested with CUDA 11.7)
- nvcc compiler
- NVIDIA GPU with compute capability 6.1 or higher
- g++-11 or compatible C++ compiler

## Quick Start

### 1. Build both executables:
```bash
cd src
make clean && make all
cd ..
```

### 2. Run a test on GPU:
```bash
./pauli_propagation_gpu.exe 25 gpu
```

### 3. Run a test on CPU:
```bash
./pauli_propagation_gpu.exe 25 cpu
```

### 4. Run Python benchmark:
```bash
python3 scripts/quick_test_stress.py
```

### 5. Generate all report figures:
```bash
python3 scripts/generate_all_figures.py
```

## Executables

After running `make all` in `src/`, you get two executables in the project root:

1. **pauli_propagation_gpu.exe**
   - Contains both GPU and CPU code
   - Can run tests in GPU mode or CPU mode
   - Usage: `./pauli_propagation_gpu.exe <test_num> <mode>`
   - Modes: `gpu`, `cpu`

2. **pauli_propagation_cpu.exe**
   - CPU-only build (no CUDA required)
   - Usage: `./pauli_propagation_cpu.exe <test_num> cpu`

## Running Tests

**Single test on GPU:**
```bash
./pauli_propagation_gpu.exe 25 gpu
```

**Single test on CPU:**
```bash
./pauli_propagation_gpu.exe 25 cpu
```

**Test numbers:**
- Tests 1-22: Basic correctness tests
- Tests 25-34: Stress tests (designed for GPU speedup)

## Stress Tests (25-34)

| Test | Words | Layers | Expected Speedup |
|------|-------|--------|------------------|
| 25 | 30K | 500 | 10-50x (HEAVY) |
| 26 | 5K | 150 | 3-8x |
| 27 | 3K | 200 | 3-6x |
| 28 | 1K | 300 | 2-4x |
| 29 | 4K | 100 | 3-7x |
| 30 | 2K | 250 | 2-5x |
| 31 | 1K | 400 | 2-4x |
| 32 | 8K | 50 | 4-10x |
| 33 | 500 | 500 | 2-3x |
| 34 | 5K | 120 | 3-8x |

## Python Scripts

### Quick Benchmarking
- `scripts/quick_test_stress.py` - Runs stress tests 25-34 on CPU and GPU
- `scripts/performance_benchmark.py` - Detailed benchmark with timing
- `scripts/run_all_tests.py` - Simple test runner

### Report Generation

**Generate ALL figures at once:**
```bash
python3 scripts/generate_all_figures.py
```

**Individual sections:**

- **Interactive Demo:**
  ```bash
  python3 scripts/interactive_demo.py
  ```
  - Real-time CPU vs GPU comparison
  - Interactive parameter adjustment

- **Performance Analysis:**
  ```bash
  python3 scripts/performance_analysis.py
  ```
  - Creates: `performance_analysis.png`, `parameter_analysis.png`
  - Shows speedup vs qubits, depth, word count

- **Algorithmic Visualization:**
  ```bash
  python3 scripts/algorithmic_visualization.py
  ```
  - Creates: `clifford_analysis.png`, `memory_analysis.png`, `pauli_evolution.png`
  - Shows Pauli word dynamics, memory patterns

- **Correctness Validation:**
  ```bash
  python3 scripts/correctness_validation.py
  ```
  - Creates: `correctness_validation.png`, `validation_results.json`
  - Shows test suite results, error analysis

- **Summary Figures:**
  ```bash
  python3 scripts/generate_report_figures.py
  ```
  - Creates: `timing_comparison.png`, `speedup_chart.png`, etc.

## Viewing Plots on Remote Server

Since you're running on a remote server (ghc27), plots are saved as PNG files. You cannot view them directly on the server.

**Option 1: Transfer plots to local machine (RECOMMENDED)**
```bash
# From your LOCAL machine, copy plots:
scp arulm@ghc27.ghc.andrew.cmu.edu:~/private/15418/Parallelizing-Pauli-Paths/*.png ./
```

**Option 2: Use X11 forwarding (if available)**
```bash
ssh -X arulm@ghc27.ghc.andrew.cmu.edu
# Then run scripts - plots will open in windows (may be slow)
```

**Installing matplotlib on remote server:**
```bash
pip install --user matplotlib
# Or if you have sudo:
sudo pip install matplotlib
```

## Generated Plot Files

After running the scripts, you'll have these PNG files:
- `performance_analysis.png` - 4-panel speedup analysis
- `parameter_analysis.png` - Parameter sensitivity
- `clifford_analysis.png` - Gate type comparison
- `memory_analysis.png` - Memory patterns
- `pauli_evolution.png` - Word dynamics
- `correctness_validation.png` - Test results
- `timing_comparison.png` - CPU vs GPU timing
- `speedup_chart.png` - Speedup factors
- `performance_scaling.png` - Log-scale performance
- `report_summary.png` - 4-panel summary
- `validation_results.json` - Data file

All plots are saved in the project root directory.

## Makefile Targets

**Build both CPU and GPU:**
```bash
cd src
make all
```

**Build GPU only:**
```bash
make gpu
```

**Build CPU only:**
```bash
make cpu
```

**Clean:**
```bash
make clean
```

## Output Format

When you run a test, you'll see:
```
[GPU] Propagation completed in 0.5 seconds
[GPU] GPU propagation finished. Exiting.
```

or for CPU mode:
```
[CPU] Propagation completed in 5.0 seconds
[CPU] CPU propagation finished. Exiting.
```

## Example Session

```bash
# Build
cd src
make clean && make all
cd ..

# Test GPU
./pauli_propagation_gpu.exe 25 gpu
# Output: [GPU] Propagation completed in 0.5 seconds

# Test CPU
./pauli_propagation_gpu.exe 25 cpu
# Output: [CPU] Propagation completed in 5.0 seconds

# Run full benchmark
python3 scripts/quick_test_stress.py

# Generate all report figures
python3 scripts/generate_all_figures.py

# Copy plots to local machine (from local terminal)
scp arulm@ghc27.ghc.andrew.cmu.edu:~/private/15418/Parallelizing-Pauli-Paths/*.png ./
```

## Troubleshooting

**"pauli_propagation_gpu.exe not found"**
- Make sure you ran: `cd src && make all`
- Executable is in project root, not `src/`

**"nvcc not found"**
- CUDA not installed or not in PATH
- Use a machine with CUDA (e.g., ghc27.ghc.andrew.cmu.edu)

**No GPU speedup:**
- Make sure you're using GPU mode: `./pauli_propagation_gpu.exe 25 gpu`
- Check `nvidia-smi` to verify GPU is available
- Stress tests 25-34 show best speedup (especially test 25)

**Python script errors:**
- Make sure executable exists: `ls pauli_propagation_gpu.exe`
- Install matplotlib for plots: `pip install matplotlib`

## CPU-Only Build (No GPU)

If you want to build without GPU support (e.g., on Windows):

```bash
g++ -DCPU_ONLY -std=c++17 -Wall -O2 src/pauli.cpp src/main.cpp src/tests.cpp -Isrc -o pauli_sim.exe
```

Note: The `-DCPU_ONLY` flag disables GPU code compilation.

## Requirements

- CUDA toolkit (for GPU execution, tested on RTX 2080)
- C++17 compiler
- Python 3.6+ with matplotlib

## Authors

- Arul Rhik Mazumder (arulm)
- Daniel Ragazzo (dragazzo)

15-418/618 Parallel Computer Architecture and Programming - Carnegie Mellon University
