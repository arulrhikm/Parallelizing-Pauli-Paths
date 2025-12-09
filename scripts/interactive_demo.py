#!/usr/bin/env python3
"""
Interactive Demo - Real-time CPU vs GPU comparison with parameter adjustment
"""

import subprocess
import re
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

def run_test(test_num, mode):
    """Run a test and return timing and result"""
    exe = "./pauli_propagation_gpu.exe"
    cmd = [exe, str(test_num), mode]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
        # Parse timing
        match = re.search(r'Propagation completed in ([\d.]+) seconds', output)
        if match:
            return float(match.group(1)), True, output
        return 0.0, False, output
    except:
        return 0.0, False, ""

def compare_cpu_gpu(test_num):
    """Compare CPU vs GPU for a test"""
    print(f"\n{'='*60}")
    print(f"Test {test_num}: CPU vs GPU Comparison")
    print('='*60)
    
    print("Running CPU...", end="", flush=True)
    cpu_time, cpu_ok, _ = run_test(test_num, "cpu")
    if cpu_ok:
        print(f" {cpu_time:.3f}s")
    else:
        print(" FAILED")
    
    print("Running GPU...", end="", flush=True)
    gpu_time, gpu_ok, _ = run_test(test_num, "gpu")
    if gpu_ok:
        print(f" {gpu_time:.3f}s")
    else:
        print(" FAILED")
    
    if cpu_ok and gpu_ok and gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"CPU: {cpu_time:.3f}s | GPU: {gpu_time:.3f}s")
        return cpu_time, gpu_time, speedup
    return cpu_time, gpu_time, 0

def interactive_menu():
    """Interactive menu for demo"""
    print("\n" + "="*60)
    print("INTERACTIVE PAULI PROPAGATION DEMO")
    print("="*60)
    print("\nOptions:")
    print("1. Compare CPU vs GPU for a specific test")
    print("2. Run all stress tests (25-34) and show comparison")
    print("3. Performance analysis (qubits, depth, word count)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    return choice

def run_all_stress_tests():
    """Run all stress tests and show comparison"""
    print("\nRunning all stress tests (25-34)...")
    print("This may take a few minutes...\n")
    
    results = []
    for test_num in range(25, 35):
        cpu_time, gpu_time, speedup = compare_cpu_gpu(test_num)
        results.append({
            'test': test_num,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Test':<8}{'CPU (s)':<12}{'GPU (s)':<12}{'Speedup':<12}")
    print("-"*44)
    
    valid_speedups = []
    for r in results:
        cpu_str = f"{r['cpu_time']:.3f}" if r['cpu_time'] > 0 else "FAIL"
        gpu_str = f"{r['gpu_time']:.3f}" if r['gpu_time'] > 0 else "FAIL"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] > 0 else "N/A"
        print(f"{r['test']:<8}{cpu_str:<12}{gpu_str:<12}{speedup_str:<12}")
        if r['speedup'] > 0:
            valid_speedups.append(r['speedup'])
    
    if valid_speedups:
        print("-"*44)
        print(f"Average Speedup: {sum(valid_speedups)/len(valid_speedups):.2f}x")
        print(f"Max Speedup: {max(valid_speedups):.2f}x")
        print(f"Min Speedup: {min(valid_speedups):.2f}x")

def main():
    if not Path("./pauli_propagation_gpu.exe").exists():
        print("Error: pauli_propagation_gpu.exe not found")
        print("Build with: cd src && make all")
        sys.exit(1)
    
    while True:
        choice = interactive_menu()
        
        if choice == '1':
            test_num = input("Enter test number (1-34): ").strip()
            try:
                test_num = int(test_num)
                compare_cpu_gpu(test_num)
            except:
                print("Invalid test number")
        
        elif choice == '2':
            run_all_stress_tests()
        
        elif choice == '3':
            print("\nPerformance analysis available via:")
            print("  python3 performance_analysis.py")
            print("  python3 generate_report_figures.py")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()

