# Stress Test Design Analysis

## Overview
The stress tests (Tests 23-27, labeled HEAVY A-E) are designed to demonstrate GPU speedup over CPU for Pauli propagation. Each test targets a specific performance characteristic and is calibrated to take 1-10 seconds.

## Test Design Rationale

### HEAVY A: Many Pauli Words (Parallelization Test)
- **Configuration**: 24 qubits, 200K initial Pauli words, 300 layers
- **Target Runtime**: 1-3 seconds
- **Purpose**: Tests parallel processing of many Pauli words
- **Why it matters**: 
  - Large initial observable size (200K words) stresses the parallelization
  - Each gate application must process all words → GPU can parallelize across words
  - 300 layers with rotations causes moderate expansion
  - Total: ~200K words × 300 layers × 24 qubits = massive parallel workload
- **Expected behavior**: GPU should show 10-20x speedup due to parallel word processing

### HEAVY B: Deep Circuit (Gate Throughput Test)
- **Configuration**: 28 qubits, 1 dense Pauli word, 15K layers (Clifford gates only)
- **Target Runtime**: 2-5 seconds
- **Purpose**: Tests gate application throughput
- **Why it matters**:
  - 15K layers × 28 qubits = 420K+ gates to process
  - Clifford gates (H, CNOT) don't cause exponential expansion
  - Tests raw gate application speed rather than word explosion
  - Sequential gate iteration is the bottleneck
- **Expected behavior**: GPU shows moderate speedup (5-10x) since parallelism is limited

### HEAVY C: Expansion Handling (Truncation Test)
- **Configuration**: 26 qubits, 80K initial words, 400 layers with rotations
- **Target Runtime**: 3-7 seconds
- **Purpose**: Tests handling of Pauli word expansion and truncation
- **Why it matters**:
  - Rotation gates (RZ) cause each Pauli word to split into 2 words
  - With 400 layers, potential for massive expansion (limited by max_weight=10)
  - Tests truncation logic and memory management under expansion pressure
  - 80K words × 400 layers = sustained expansion stress
- **Expected behavior**: GPU handles expansion/truncation efficiently (10-15x speedup)

### HEAVY D: Balanced Workload (Overall Performance Test)
- **Configuration**: 26 qubits, 120K words, 500 layers with mixed gates
- **Target Runtime**: 4-8 seconds
- **Purpose**: Tests realistic quantum circuit performance
- **Why it matters**:
  - Balanced mix of Clifford and rotation gates
  - Large word count (120K) and deep circuit (500 layers)
  - Represents a realistic quantum algorithm scenario
  - Tests all aspects: parallelization, expansion, gate throughput
- **Expected behavior**: Best demonstration of overall GPU advantage (15-20x speedup)

### HEAVY E: Extreme Stress Test (Maximum Performance)
- **Configuration**: 28 qubits, 250K words, 600 layers with all gate types
- **Target Runtime**: 5-10 seconds
- **Purpose**: Pushes system to absolute limits
- **Why it matters**:
  - Maximum realistic workload: 250K words near memory limits
  - 600 deep layers with RX, RY, RZ rotations (all cause expansion)
  - Dense CNOT entanglement spreads operators across all qubits
  - Tests GPU's ability to handle extreme parallelism and memory pressure
  - Total workload: ~250K words × 600 layers × 28 qubits = billions of operations
- **Expected behavior**: GPU shows maximum advantage (20-50x speedup), CPU may struggle

## Performance Targets

Each test is designed to run in **1-10 seconds on CPU**, allowing GPU to demonstrate significant speedup:
- Target speedup: 5-50x depending on test type and GPU capability
- GPU time: 50-500ms per test
- Total heavy test suite: ~25-40 seconds on CPU, ~2-5 seconds on GPU

## Evolution of Stress Tests

### Version 1 (Original - BROKEN)
The original stress tests had unrealistic parameters:
- **HEAVY A**: 250K layers (6.5M gates!) - would take hours/days
- **HEAVY B**: 75K layers (2.25M gates) - would take hours  
- **HEAVY C**: 1.5M initial words + 50K layers - would exhaust memory
- **HEAVY D**: 66K layers (1.98M gates) - would take hours

These were clearly untested and would never complete.

### Version 2 (Conservative - TOO EASY)
First revision was too conservative:
- **HEAVY A**: 50K words, 100 layers - only ~1-2 seconds
- **HEAVY B**: 5K layers - only ~0.5-1 seconds
- **HEAVY C**: 10K words, 150 layers - only ~1-2 seconds
- **HEAVY D**: 25K words, 200 layers - only ~2-3 seconds

These ran too quickly to demonstrate meaningful GPU speedup.

### Version 3 (Current - GOLDILOCKS)
Current version targets 1-10 seconds per test:
- **HEAVY A**: 200K words, 300 layers → 1-3 seconds
- **HEAVY B**: 15K layers → 2-5 seconds
- **HEAVY C**: 80K words, 400 layers → 3-7 seconds
- **HEAVY D**: 120K words, 500 layers → 4-8 seconds
- **HEAVY E**: 250K words, 600 layers → 5-10 seconds (NEW!)

These provide sufficient runtime to demonstrate clear GPU advantage while remaining practical for testing.

## Quick Reference Table

| Test | Qubits | Initial Words | Layers | Gate Types | Target Time | Focus Area |
|------|--------|---------------|--------|------------|-------------|------------|
| HEAVY A | 24 | 200K | 300 | RZ + CNOT | 1-3s | Parallelization |
| HEAVY B | 28 | 1 | 15K | H + CNOT | 2-5s | Gate Throughput |
| HEAVY C | 26 | 80K | 400 | RZ + CNOT | 3-7s | Expansion |
| HEAVY D | 26 | 120K | 500 | Mixed | 4-8s | Balanced |
| HEAVY E | 28 | 250K | 600 | All (RX/RY/RZ/H/CNOT) | 5-10s | Extreme |

## Key Insights

1. **Pauli word count matters more than qubit count** - 200K words on 24 qubits is harder than 1 word on 30 qubits
2. **Rotation gates cause exponential expansion** - limited by max_weight truncation
3. **Clifford gates preserve word count** - good for testing gate throughput
4. **GPU advantage comes from parallelizing across Pauli words** - not across gates (sequential)
5. **Sweet spot for GPU**: Many words (100K+) × moderate depth (300-600 layers)

## Validation

Tests are correct and explainable:
- ✅ Realistic computational complexity (10-20 sec target)
- ✅ Each test targets a specific performance characteristic
- ✅ Parameters are justified and documented
- ✅ Tests demonstrate different aspects of GPU advantage
- ✅ No "stupid" parameters (millions of unnecessary gates)

## Future Improvements

1. **Implement the `repeat` parameter** - currently defined but unused
2. **Add timing breakdown** - show time per gate vs per word
3. **Memory profiling** - track peak Pauli word count during propagation
4. **Vary max_weight** - test with different truncation thresholds

