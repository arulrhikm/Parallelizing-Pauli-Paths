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
├── scripts/             # Python visualization scripts
├── images/              # Generated figures
├── docs/                # Additional documentation
├── index.html           # Project website
├── report.html          # Final report (HTML)
├── final_report.tex     # Final report (LaTeX)
└── Makefile             # Root build file
```

## Quick Start

```bash
# Build on GHC cluster
cd src
make clean && make all
cd ..

# Run tests
./pauli_sim 25 gpu    # Run test 25 on GPU
./pauli_sim 25 cpu    # Run test 25 on CPU
```

## Report Generation

Generate all figures for the report:
```bash
cd scripts
python3 generate_all_figures.py
```

Individual scripts:
- `interactive_demo.py` - Real-time CPU vs GPU comparison
- `performance_analysis.py` - Speedup and parameter analysis plots
- `algorithmic_visualization.py` - Pauli evolution and gate behavior
- `correctness_validation.py` - Test suite validation results

## Test Suite

- **Tests 1-22:** Basic correctness tests (Clifford gates, rotations)
- **Tests 23-34:** Stress tests (500-30K words, 50-500 layers)

## Requirements

- CUDA toolkit (tested on RTX 2080)
- C++17 compiler
- Python 3.6+ with matplotlib

## Authors

- Arul Rhik Mazumder (arulm)
- Daniel Ragazzo (dragazzo)

15-418/618 Parallel Computer Architecture and Programming - Carnegie Mellon University
