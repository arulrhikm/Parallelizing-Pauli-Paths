#!/usr/bin/env python3
"""
Run All Tests - Simple script to run all stress tests (23-32)
"""

import subprocess
import re
import sys
from pathlib import Path

def run_test(test_num, mode):
    """Run test and return time"""
    # Use the GPU executable for both modes (it supports both cpu and gpu)
    cmd = ["./pauli_propagation_gpu.exe", str(test_num), mode]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        match = re.search(r'Propagation completed in ([\d.]+) seconds', result.stdout)
        if match:
            return float(match.group(1))
        return -1
    except:
        return -1

def main():
    if not Path("./pauli_propagation_gpu.exe").exists():
        print("Error: pauli_propagation_gpu.exe not found")
        print("Build: cd src && make all")
        sys.exit(1)

    print("Running all stress tests (23-32)")
    print("Command: ./pauli_propagation_gpu.exe <test> <mode>")
    print("=" * 60)
    print(f"{'Test':<8}{'CPU':<15}{'GPU':<15}{'Speedup':<12}")
    print("-" * 60)

    total_cpu = 0
    total_gpu = 0
    speedups = []

    for t in range(23, 33):
        cpu = run_test(t, "cpu")
        gpu = run_test(t, "gpu")
        
        if cpu > 0 and gpu > 0:
            speedup = cpu / gpu
            speedups.append(speedup)
            total_cpu += cpu
            total_gpu += gpu
            print(f"{t:<8}{cpu:<15.3f}{gpu:<15.3f}{speedup:<12.2f}x")
        else:
            print(f"{t:<8}{'FAIL':<15}{'FAIL':<15}{'N/A':<12}")

    print("-" * 60)
    if speedups:
        print(f"Total:  {total_cpu:.1f}s (CPU)  {total_gpu:.1f}s (GPU)")
        print(f"Avg Speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"Max Speedup: {max(speedups):.2f}x")

if __name__ == "__main__":
    main()
