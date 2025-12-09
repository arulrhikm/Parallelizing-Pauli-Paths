#!/usr/bin/env python3
"""
Quick Stress Test Check - Tests 25-34 CPU vs GPU
"""

import subprocess
import re
import sys
from pathlib import Path

def run_test(test_num, mode):
    """Run a single test and return timing"""
    exe = "./pauli_propagation_gpu.exe"
    cmd = [exe, str(test_num), mode]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
        # Parse: [GPU] Propagation completed in X seconds
        # or: [CPU] Propagation completed in X seconds
        match = re.search(r'Propagation completed in ([\d.]+) seconds', output)
        
        if match:
            return float(match.group(1)), True
        return 0.0, False
        
    except subprocess.TimeoutExpired:
        return 0.0, False
    except Exception as e:
        print(f"Error: {e}")
        return 0.0, False

def main():
    print("=" * 70)
    print("QUICK STRESS TEST CHECK (Tests 25-34)")
    print("=" * 70)
    
    # Check executable
    if not Path("./pauli_propagation_gpu.exe").exists():
        print("Error: pauli_propagation_gpu.exe not found")
        print("Build with: cd src && make all")
        sys.exit(1)
    
    print("\nCommand format: ./pauli_propagation_gpu.exe <test> <mode>")
    print("Modes: gpu, cpu")
    print("\nRunning stress tests...")
    print("-" * 70)
    
    results = []
    
    for test_num in range(25, 35):
        print(f"Test {test_num}: ", end="", flush=True)
        
        # Run CPU mode
        print("CPU...", end="", flush=True)
        cpu_time, cpu_ok = run_test(test_num, "cpu")
        
        # Run GPU mode
        print("GPU...", end="", flush=True)
        gpu_time, gpu_ok = run_test(test_num, "gpu")
        
        if cpu_ok and gpu_ok and gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f" CPU={cpu_time:.2f}s GPU={gpu_time:.2f}s Speedup={speedup:.2f}x")
        else:
            speedup = 0
            print(f" CPU={cpu_time:.2f}s GPU={gpu_time:.2f}s (failed)")
        
        results.append({
            'test': test_num,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'cpu_ok': cpu_ok,
            'gpu_ok': gpu_ok
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<6} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    
    valid_speedups = []
    for r in results:
        cpu_str = f"{r['cpu_time']:.3f}s" if r['cpu_ok'] else "FAIL"
        gpu_str = f"{r['gpu_time']:.3f}s" if r['gpu_ok'] else "FAIL"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] > 0 else "N/A"
        status = "PASS" if r['cpu_ok'] and r['gpu_ok'] else "FAIL"
        print(f"{r['test']:<6} {cpu_str:<12} {gpu_str:<12} {speedup_str:<10} {status}")
        
        if r['speedup'] > 0:
            valid_speedups.append(r['speedup'])
    
    print("-" * 70)
    if valid_speedups:
        print(f"Average Speedup: {sum(valid_speedups)/len(valid_speedups):.2f}x")
        print(f"Max Speedup: {max(valid_speedups):.2f}x")
        print(f"Min Speedup: {min(valid_speedups):.2f}x")
    print("=" * 70)

if __name__ == "__main__":
    main()

