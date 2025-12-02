# Heavy Tests Summary

## Quick Stats
- **Total Tests**: 27 (Tests 1-22: basic/correctness, Tests 23-27: heavy/performance)
- **Heavy Tests**: 5 (HEAVY A, B, C, D, E)
- **Target Runtime**: 1-10 seconds per heavy test (15-40 seconds total for all heavy tests)
- **Expected GPU Speedup**: 5-50x depending on test characteristics

## Running Heavy Tests

### Run All Tests (including heavy)
```powershell
.\pauli_sim.exe -d cpu    # CPU version
.\pauli_sim.exe -d gpu    # GPU version
```

### Run Single Heavy Test
```powershell
.\pauli_sim.exe -d cpu -i 23    # HEAVY A
.\pauli_sim.exe -d cpu -i 24    # HEAVY B
.\pauli_sim.exe -d cpu -i 25    # HEAVY C
.\pauli_sim.exe -d cpu -i 26    # HEAVY D
.\pauli_sim.exe -d cpu -i 27    # HEAVY E
```

## Test Specifications

### HEAVY A: Parallelization Test (Test #23)
```
Qubits:        24
Initial Words: 200,000
Layers:        300
Gate Types:    RZ + CNOT
Total Gates:   ~21,600
Target Time:   1-3 seconds
GPU Advantage: 10-20x (many words to parallelize)
```

### HEAVY B: Gate Throughput Test (Test #24)
```
Qubits:        28
Initial Words: 1 (dense)
Layers:        15,000
Gate Types:    Hadamard + CNOT
Total Gates:   ~630,000
Target Time:   2-5 seconds
GPU Advantage: 5-10x (limited parallelism, many sequential gates)
```

### HEAVY C: Expansion Test (Test #25)
```
Qubits:        26
Initial Words: 80,000
Layers:        400
Gate Types:    RZ + CNOT (expansion-heavy)
Total Gates:   ~33,600
Target Time:   3-7 seconds
GPU Advantage: 10-15x (expansion + truncation stress)
```

### HEAVY D: Balanced Test (Test #26)
```
Qubits:        26
Initial Words: 120,000
Layers:        500
Gate Types:    Mixed (H + RZ + CNOT)
Total Gates:   ~39,000
Target Time:   4-8 seconds
GPU Advantage: 15-20x (best overall demonstration)
```

### HEAVY E: Extreme Test (Test #27)
```
Qubits:        28
Initial Words: 250,000
Layers:        600
Gate Types:    All (RX + RY + RZ + H + CNOT)
Total Gates:   ~50,400
Target Time:   5-10 seconds
GPU Advantage: 20-50x (maximum stress, GPU shines)
```

## What Each Test Demonstrates

| Test | What It Tests | Why GPU Wins |
|------|---------------|--------------|
| A | Parallel word processing | 200K words processed in parallel |
| B | Sequential gate throughput | Even with 1 word, GPU is faster per gate |
| C | Expansion handling | Parallel expansion + truncation |
| D | Realistic workload | Balanced mix shows overall advantage |
| E | Maximum performance | Extreme parallelism + all gate types |

## Performance Expectations

### CPU Performance (estimated)
- HEAVY A: ~2 seconds
- HEAVY B: ~3 seconds
- HEAVY C: ~5 seconds
- HEAVY D: ~6 seconds
- HEAVY E: ~8 seconds
- **Total: ~24 seconds**

### GPU Performance (estimated with good GPU)
- HEAVY A: ~150ms (13x speedup)
- HEAVY B: ~400ms (7.5x speedup)
- HEAVY C: ~350ms (14x speedup)
- HEAVY D: ~350ms (17x speedup)
- HEAVY E: ~300ms (27x speedup)
- **Total: ~1.5 seconds (16x average speedup)**

## Memory Requirements

| Test | Estimated Peak Memory |
|------|----------------------|
| HEAVY A | ~200K words × 24 bytes = ~4.8 MB |
| HEAVY B | ~1 word (minimal) |
| HEAVY C | ~80K words × 26 bytes = ~2.1 MB |
| HEAVY D | ~120K words × 26 bytes = ~3.1 MB |
| HEAVY E | ~250K words × 28 bytes = ~7.0 MB |

All tests fit comfortably in GPU memory (even on modest GPUs with 2-4GB).

## Troubleshooting

### Tests Too Fast?
If heavy tests complete in < 1 second on CPU, your CPU is very fast. The GPU advantage may be less dramatic but still measurable.

### Tests Too Slow?
If heavy tests take > 15 seconds on CPU:
1. Check if you're running in Debug mode (use Release/O2)
2. Verify CPU isn't thermal throttling
3. Close other applications

### GPU Not Faster?
If GPU isn't showing speedup:
1. Check GPU is actually being used (not falling back to CPU)
2. Verify CUDA is properly installed
3. Check for memory transfer overhead (should be minimal with our design)
4. Try HEAVY E - it should show the clearest advantage

## Design Philosophy

These tests are carefully calibrated to:
1. ✅ Run in reasonable time (1-10s, not minutes)
2. ✅ Demonstrate clear GPU advantage (5-50x speedup)
3. ✅ Test different performance characteristics
4. ✅ Fit in GPU memory comfortably
5. ✅ Be explainable and justifiable
6. ✅ Represent realistic quantum computing workloads

Each parameter (qubits, words, layers) is chosen deliberately to stress specific aspects of the implementation.

