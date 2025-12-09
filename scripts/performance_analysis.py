#!/usr/bin/env python3
"""
Performance Analysis - Comprehensive speedup plots for report
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
    """Run test and return timing"""
    exe = "./pauli_propagation_gpu.exe"
    cmd = [exe, str(test_num), mode]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        match = re.search(r'Propagation completed in ([\d.]+) seconds', result.stdout)
        if match:
            return float(match.group(1)), True
        return 0.0, False
    except:
        return 0.0, False

def collect_stress_test_data():
    """Collect data from stress tests 25-34"""
    print("Collecting stress test data...")
    results = []
    
    for test_num in range(25, 35):
        print(f"  Test {test_num}...", end="", flush=True)
        cpu_time, cpu_ok = run_test(test_num, "cpu")
        gpu_time, gpu_ok = run_test(test_num, "gpu")
        
        speedup = cpu_time / gpu_time if cpu_ok and gpu_ok and gpu_time > 0 else 0
        print(f" CPU={cpu_time:.2f}s GPU={gpu_time:.2f}s Speedup={speedup:.2f}x")
        
        results.append({
            'test': test_num,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
    
    return results

def create_speedup_plots(results):
    """Create comprehensive speedup plots"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping plots")
        return
    
    tests = [r['test'] for r in results]
    cpu_times = [r['cpu_time'] for r in results]
    gpu_times = [r['gpu_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Figure 1: CPU vs GPU timing comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Bar chart comparison
    x = range(len(tests))
    width = 0.35
    ax1.bar([i - width/2 for i in x], cpu_times, width, label='CPU', color='blue', alpha=0.7)
    ax1.bar([i + width/2 for i in x], gpu_times, width, label='GPU', color='red', alpha=0.7)
    ax1.set_xlabel('Test Number')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('CPU vs GPU Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tests)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factors
    valid_tests = [t for t, s in zip(tests, speedups) if s > 0]
    valid_speedups = [s for s in speedups if s > 0]
    if valid_speedups:
        colors = ['green' if s >= 5 else 'orange' if s >= 2 else 'red' for s in valid_speedups]
        ax2.bar(range(len(valid_tests)), valid_speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Test Number')
        ax2.set_ylabel('Speedup Factor (CPU/GPU)')
        ax2.set_title('GPU Speedup Factors')
        ax2.set_xticks(range(len(valid_tests)))
        ax2.set_xticklabels(valid_tests)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log scale performance
    cpu_log = [t if t > 0 else 0.001 for t in cpu_times]
    gpu_log = [t if t > 0 else 0.001 for t in gpu_times]
    ax3.semilogy(tests, cpu_log, 'o-', label='CPU', linewidth=2, markersize=8, color='blue')
    ax3.semilogy(tests, gpu_log, 's-', label='GPU', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Test Number')
    ax3.set_ylabel('Time (seconds, log scale)')
    ax3.set_title('Performance Scaling (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Speedup distribution
    if valid_speedups:
        fast = sum(1 for s in valid_speedups if s >= 5)
        medium = sum(1 for s in valid_speedups if 2 <= s < 5)
        slow = sum(1 for s in valid_speedups if s < 2)
        sizes = [fast, medium, slow]
        labels = [f'>5x ({fast})', f'2-5x ({medium})', f'<2x ({slow})']
        colors = ['green', 'orange', 'red']
        if sum(sizes) > 0:
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
            ax4.set_title('Speedup Distribution')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: performance_analysis.png")

def create_parameter_analysis():
    """Create plots showing performance vs parameters"""
    if not HAS_MATPLOTLIB:
        return
    
    # Test configurations (from tests.cpp)
    test_configs = {
        25: {'words': 30000, 'layers': 500},
        26: {'words': 5000, 'layers': 150},
        27: {'words': 3000, 'layers': 200},
        28: {'words': 1000, 'layers': 300},
        29: {'words': 4000, 'layers': 100},
        30: {'words': 2000, 'layers': 250},
        31: {'words': 1000, 'layers': 400},
        32: {'words': 8000, 'layers': 50},
        33: {'words': 500, 'layers': 500},
        34: {'words': 5000, 'layers': 120}
    }
    
    # Collect data
    print("\nCollecting parameter analysis data...")
    data = []
    for test_num in range(25, 35):
        cpu_time, cpu_ok = run_test(test_num, "cpu")
        gpu_time, gpu_ok = run_test(test_num, "gpu")
        if cpu_ok and gpu_ok:
            config = test_configs[test_num]
            data.append({
                'words': config['words'],
                'layers': config['layers'],
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': cpu_time / gpu_time if gpu_time > 0 else 0
            })
    
    if not data:
        print("No data collected for parameter analysis")
        return
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Speedup vs Word Count
    words = [d['words'] for d in data]
    speedups = [d['speedup'] for d in data]
    ax1.scatter(words, speedups, s=100, alpha=0.6, color='blue')
    ax1.set_xlabel('Initial Pauli Word Count')
    ax1.set_ylabel('Speedup Factor')
    ax1.set_title('GPU Speedup vs Pauli Word Count')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Speedup vs Circuit Depth
    layers = [d['layers'] for d in data]
    ax2.scatter(layers, speedups, s=100, alpha=0.6, color='red')
    ax2.set_xlabel('Circuit Depth (Layers)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('GPU Speedup vs Circuit Depth')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CPU Time vs Word Count
    ax3.scatter(words, [d['cpu_time'] for d in data], s=100, alpha=0.6, color='blue', label='CPU')
    ax3.scatter(words, [d['gpu_time'] for d in data], s=100, alpha=0.6, color='red', label='GPU')
    ax3.set_xlabel('Initial Pauli Word Count')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Execution Time vs Word Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Plot 4: CPU Time vs Circuit Depth
    ax4.scatter(layers, [d['cpu_time'] for d in data], s=100, alpha=0.6, color='blue', label='CPU')
    ax4.scatter(layers, [d['gpu_time'] for d in data], s=100, alpha=0.6, color='red', label='GPU')
    ax4.set_xlabel('Circuit Depth (Layers)')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Execution Time vs Circuit Depth')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: parameter_analysis.png")

def main():
    if not Path("./pauli_propagation_gpu.exe").exists():
        print("Error: pauli_propagation_gpu.exe not found")
        print("Build with: cd src && make all")
        sys.exit(1)
    
    print("="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Collect data
    results = collect_stress_test_data()
    
    # Create plots
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        create_speedup_plots(results)
        create_parameter_analysis()
    
    # Summary
    valid_speedups = [r['speedup'] for r in results if r['speedup'] > 0]
    if valid_speedups:
        print(f"\nSummary:")
        print(f"  Average Speedup: {sum(valid_speedups)/len(valid_speedups):.2f}x")
        print(f"  Maximum Speedup: {max(valid_speedups):.2f}x")
        print(f"  Minimum Speedup: {min(valid_speedups):.2f}x")

if __name__ == "__main__":
    main()

