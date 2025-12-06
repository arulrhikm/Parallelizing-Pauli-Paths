# Performance Comparison Script

## Usage

The `compare_performance.sh` script automatically runs both CPU and GPU versions of the Pauli simulator and shows the performance comparison.

### Make executable (on Linux/SSH)
```bash
chmod +x compare_performance.sh
```

### Run all tests
```bash
./compare_performance.sh
```

### Run specific test
```bash
./compare_performance.sh 23    # Compare test 23 (HEAVY A)
./compare_performance.sh 27    # Compare test 27 (HEAVY E)
```

## Output

The script displays:
- CPU execution time
- GPU execution time
- Speedup factor (CPU time / GPU time)
- Whether GPU is faster or slower

Full test outputs are saved to:
- `/tmp/cpu_output.txt` - CPU version output
- `/tmp/gpu_output.txt` - GPU version output

## Example Output

```
========================================
  PAULI SIMULATOR PERFORMANCE COMPARISON
========================================

Running test #23

----------------------------------------
Running CPU version...
----------------------------------------

----------------------------------------
Running GPU version...
----------------------------------------

========================================
  RESULTS
========================================

CPU Time:    2.456 seconds
GPU Time:    0.123 seconds

Speedup:     19.97x

✓ GPU is faster!

Full outputs saved to:
  CPU: /tmp/cpu_output.txt
  GPU: /tmp/gpu_output.txt
```

## Notes

- The script must be run from the project root directory
- The `pauli_sim` executable must exist (run `make all` first)
- Heavy tests (23-27) show the most significant GPU speedup
- Small tests may show GPU slower due to overhead
