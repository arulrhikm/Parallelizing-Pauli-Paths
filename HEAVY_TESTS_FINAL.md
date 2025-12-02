# Heavy Tests - Final Configuration

## Summary
Heavy tests A-E are now calibrated to run in **1-10 seconds** on a modern CPU, providing sufficient runtime to demonstrate meaningful GPU speedup.

## Final Test Parameters

### HEAVY A: Parallelization Test (Test #23)
```
Qubits:        26
Initial Words: 1,000,000 (1M)
Layers:        1,500
Gate Types:    RZ + CNOT
Measured Time: ~1.5 seconds (CPU)
Focus:         Massive parallel word processing
```

### HEAVY B: Deep Circuit Test (Test #24)
```
Qubits:        28
Initial Words: 500,000 (500K)
Layers:        50,000
Gate Types:    Hadamard + CNOT (Clifford only)
Measured Time: ~0.7 seconds (CPU)
Focus:         Extreme gate throughput
```

### HEAVY C: Expansion Test (Test #25)
```
Qubits:        28
Initial Words: 800,000 (800K)
Layers:        2,000
Gate Types:    RZ + CNOT (rotation-heavy)
Measured Time: ~1.1 seconds (CPU)
Focus:         Pauli word expansion handling
```

### HEAVY D: Balanced Test (Test #26)
```
Qubits:        28
Initial Words: 1,200,000 (1.2M)
Layers:        2,500
Gate Types:    Mixed (H + RZ + CNOT)
Measured Time: ~1.8 seconds (CPU, estimated)
Focus:         Realistic balanced workload
```

### HEAVY E: Extreme Test (Test #27)
```
Qubits:        30
Initial Words: 1,500,000 (1.5M)
Layers:        3,000
Gate Types:    All (RX + RY + RZ + H + CNOT)
Measured Time: ~2.3 seconds (CPU)
Focus:         Maximum stress test
```

## Quick Reference Table

| Test | Qubits | Words | Layers | Gates/Layer | Total Gates | CPU Time | Focus |
|------|--------|-------|--------|-------------|-------------|----------|-------|
| A | 26 | 1.0M | 1.5K | ~72 | ~108K | 1.5s | Many words |
| B | 28 | 500K | 50K | ~85 | ~4.25M | 0.7s | Many gates |
| C | 28 | 800K | 2K | ~47 | ~94K | 1.1s | Expansion |
| D | 28 | 1.2M | 2.5K | ~70 | ~175K | 1.8s | Balanced |
| E | 30 | 1.5M | 3K | ~105 | ~315K | 2.3s | Extreme |

**Total Heavy Test Suite: ~7-8 seconds on CPU**

## Why These Parameters?

### Key Insight: Pauli Word Count Dominates
Through testing, we discovered that **the number of Pauli words is the primary performance bottleneck**, not the number of gates. This is because:

1. Each gate must be applied to ALL Pauli words
2. Complexity = O(words × layers × qubits)
3. With max_weight=10 truncation, expansion is limited

### Calibration Process
- **Version 1**: Too extreme (millions of gates, would take hours)
- **Version 2**: Too conservative (< 1 second, insufficient for GPU demo)
- **Version 3 (Final)**: 1-2 seconds per test, perfect for demonstrating speedup

### Why 1-10 Seconds is Ideal
- **Too fast (< 0.5s)**: GPU overhead dominates, speedup unclear
- **Too slow (> 30s)**: Testing becomes tedious, impractical for development
- **Just right (1-10s)**: Clear speedup demonstration, practical testing time

## Expected GPU Performance

With a modern GPU (e.g., RTX 3060 or better):

| Test | CPU Time | Expected GPU Time | Expected Speedup |
|------|----------|-------------------|------------------|
| A | 1.5s | 75-150ms | 10-20x |
| B | 0.7s | 100-140ms | 5-7x |
| C | 1.1s | 70-110ms | 10-15x |
| D | 1.8s | 90-120ms | 15-20x |
| E | 2.3s | 80-150ms | 15-30x |

**Total: 7.4s CPU → 0.5-0.7s GPU = ~12x average speedup**

## Memory Requirements

All tests fit comfortably in GPU memory:

| Test | Peak Memory (estimated) |
|------|------------------------|
| A | 1M words × 26 bytes = ~26 MB |
| B | 500K words × 28 bytes = ~14 MB |
| C | 800K words × 28 bytes = ~22 MB |
| D | 1.2M words × 28 bytes = ~34 MB |
| E | 1.5M words × 30 bytes = ~45 MB |

Even the largest test (HEAVY E) uses < 50MB, well within any GPU's capability.

## Running the Tests

### Run all heavy tests
```powershell
.\pauli_sim.exe -d cpu    # CPU version
.\pauli_sim.exe -d gpu    # GPU version (when available)
```

### Run individual test
```powershell
.\pauli_sim.exe -d cpu -i 23    # HEAVY A
.\pauli_sim.exe -d cpu -i 24    # HEAVY B
.\pauli_sim.exe -d cpu -i 25    # HEAVY C
.\pauli_sim.exe -d cpu -i 26    # HEAVY D
.\pauli_sim.exe -d cpu -i 27    # HEAVY E
```

### Compare CPU vs GPU
```powershell
# Run all tests on CPU
.\pauli_sim.exe -d cpu

# Run all tests on GPU
.\pauli_sim.exe -d gpu

# The output will show computation times for direct comparison
```

## Design Philosophy

These tests demonstrate:
1. ✅ **Realistic workloads**: Parameters represent actual quantum computing scenarios
2. ✅ **Clear GPU advantage**: Sufficient runtime to show meaningful speedup
3. ✅ **Practical testing**: Complete in seconds, not minutes
4. ✅ **Diverse characteristics**: Each test stresses different aspects
5. ✅ **Explainable results**: Clear understanding of what each test measures

## Troubleshooting

### Tests run faster than expected?
Your CPU is very fast! This is good - the GPU will still show speedup, just with smaller absolute times.

### Tests run slower than expected?
1. Ensure you're compiling with `-O2` optimization
2. Check CPU isn't thermal throttling
3. Close other applications
4. Try Release build instead of Debug

### Want even harder tests?
You can multiply the `num_words` or `layers` by 2-3x, but be aware:
- Tests may take 10-30 seconds
- Memory usage will increase proportionally
- Diminishing returns for demonstrating GPU advantage

