# Report Figure Generation Guide

## Quick Start

```bash
# Generate all figures for your report
python3 generate_all_figures.py
```

## Report Sections & Scripts

### 1. Primary Demonstration
**Script:** `interactive_demo.py`

```bash
python3 interactive_demo.py
```

**Features:**
- Real-time CPU vs GPU execution with timing comparison
- Interactive parameter adjustment
- Live performance feedback

**Usage:**
- Choose option 1 to compare specific tests
- Choose option 2 to run all stress tests (25-34)

### 2. Performance Analysis
**Script:** `performance_analysis.py`

```bash
python3 performance_analysis.py
```

**Generates:**
- `performance_analysis.png` - Comprehensive speedup plots (4 panels)
- `parameter_analysis.png` - Performance vs word count, circuit depth

**Shows:**
- GPU vs CPU execution times
- Speedup factors by test
- Performance scaling (log scale)
- Speedup distribution
- Parameter sensitivity analysis

### 3. Algorithmic Visualization
**Script:** `algorithmic_visualization.py`

```bash
python3 algorithmic_visualization.py
```

**Generates:**
- `clifford_analysis.png` - Clifford vs non-Clifford gate behavior
- `memory_analysis.png` - Memory usage patterns
- `pauli_evolution.png` - Pauli word population dynamics

**Shows:**
- Comparison of Clifford (constant) vs non-Clifford (exponential) growth
- Memory usage by test and word count
- Evolution of Pauli words during circuit execution

### 4. Correctness Validation
**Script:** `correctness_validation.py`

```bash
python3 correctness_validation.py
```

**Generates:**
- `correctness_validation.png` - Test suite results (4 panels)
- `validation_results.json` - Detailed validation data

**Shows:**
- Pass/fail status for all tests (1-34)
- Speedup by test
- Pass rate by category (basic vs stress tests)
- Error analysis framework

## Additional Scripts

### Quick Benchmarking
```bash
python3 quick_test_stress.py      # Quick stress test check
python3 performance_benchmark.py  # Detailed benchmark
```

### Summary Figures
```bash
python3 generate_report_figures.py  # Creates summary overview plots
```

## Generated Figures Summary

| Figure | Description | Script |
|--------|-------------|--------|
| `performance_analysis.png` | 4-panel speedup analysis | `performance_analysis.py` |
| `parameter_analysis.png` | Parameter sensitivity | `performance_analysis.py` |
| `clifford_analysis.png` | Gate type comparison | `algorithmic_visualization.py` |
| `memory_analysis.png` | Memory patterns | `algorithmic_visualization.py` |
| `pauli_evolution.png` | Word dynamics | `algorithmic_visualization.py` |
| `correctness_validation.png` | Test results | `correctness_validation.py` |
| `timing_comparison.png` | CPU vs GPU timing | `generate_report_figures.py` |
| `speedup_chart.png` | Speedup factors | `generate_report_figures.py` |
| `performance_scaling.png` | Log-scale performance | `generate_report_figures.py` |
| `report_summary.png` | 4-panel summary | `generate_report_figures.py` |

## Test Suite

- **Tests 1-22:** Basic correctness tests
- **Tests 25-34:** Stress tests (GPU speedup demonstration)

## Requirements

- CUDA toolkit (for GPU execution)
- Python 3.6+
- matplotlib: `pip install matplotlib`

## Usage Examples

```bash
# Generate everything
python3 generate_all_figures.py

# Individual sections
python3 performance_analysis.py
python3 algorithmic_visualization.py
python3 correctness_validation.py

# Interactive demo
python3 interactive_demo.py
```

## Report Integration

All generated PNG files are ready for inclusion in your report. The figures cover:

1. **Primary Demonstration:** Use `interactive_demo.py` for live demo
2. **Performance Analysis:** Use `performance_analysis.png` and `parameter_analysis.png`
3. **Algorithmic Visualization:** Use `clifford_analysis.png`, `memory_analysis.png`, `pauli_evolution.png`
4. **Correctness Validation:** Use `correctness_validation.png` and `validation_results.json`
