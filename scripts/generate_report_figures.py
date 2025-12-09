#!/usr/bin/env python3
"""
Generate Report Figures - Creates all plots for the final report
"""

import subprocess
import re
import sys
from pathlib import Path

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

def run_test(test_num, mode):
    """Run a single test and return timing"""
    # Use the GPU executable for both modes (it supports both cpu and gpu)
    exe = "./pauli_propagation_gpu.exe"
    cmd = [exe, str(test_num), mode]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
        match = re.search(r'Propagation completed in ([\d.]+) seconds', output)
        if match:
            return float(match.group(1)), True
        return 0.0, False
        
    except:
        return 0.0, False

def collect_data():
    """Collect benchmark data for all stress tests"""
    print("Collecting benchmark data...")
    print("Command: ./pauli_propagation_gpu.exe <test> <mode>")
    print("-" * 50)
    
    results = []
    
    for test_num in range(25, 35):
        print(f"Test {test_num}: ", end="", flush=True)
        
        cpu_time, cpu_ok = run_test(test_num, "cpu")
        gpu_time, gpu_ok = run_test(test_num, "gpu")
        
        speedup = cpu_time / gpu_time if cpu_ok and gpu_ok and gpu_time > 0 else 0
        
        print(f"CPU={cpu_time:.2f}s GPU={gpu_time:.2f}s Speedup={speedup:.2f}x")
        
        results.append({
            'test': test_num,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'cpu_ok': cpu_ok,
            'gpu_ok': gpu_ok
        })
    
    return results

def create_timing_plot(results):
    """Create CPU vs GPU timing comparison plot"""
    if not HAS_MATPLOTLIB:
        return
    
    tests = [r['test'] for r in results]
    cpu_times = [r['cpu_time'] for r in results]
    gpu_times = [r['gpu_time'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(tests))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], cpu_times, width, label='CPU', color='blue', alpha=0.7)
    bars2 = ax.bar([i + width/2 for i in x], gpu_times, width, label='GPU', color='red', alpha=0.7)
    
    ax.set_xlabel('Test Number')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('CPU vs GPU Propagation Time')
    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timing_comparison.png', dpi=150)
    plt.close()
    print("Created: timing_comparison.png")

def create_speedup_plot(results):
    """Create speedup bar chart"""
    if not HAS_MATPLOTLIB:
        return
    
    tests = [r['test'] for r in results if r['speedup'] > 0]
    speedups = [r['speedup'] for r in results if r['speedup'] > 0]
    
    if not speedups:
        print("No valid speedup data for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if s >= 2 else 'orange' if s >= 1 else 'red' for s in speedups]
    ax.bar(range(len(tests)), speedups, color=colors, alpha=0.7)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='No speedup')
    ax.axhline(y=2, color='green', linestyle=':', linewidth=1, label='2x speedup')
    
    ax.set_xlabel('Test Number')
    ax.set_ylabel('Speedup (CPU/GPU)')
    ax.set_title('GPU Speedup Factor')
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('speedup_chart.png', dpi=150)
    plt.close()
    print("Created: speedup_chart.png")

def create_scaling_plot(results):
    """Create log-scale performance plot"""
    if not HAS_MATPLOTLIB:
        return
    
    tests = [r['test'] for r in results]
    cpu_times = [r['cpu_time'] if r['cpu_time'] > 0 else 0.001 for r in results]
    gpu_times = [r['gpu_time'] if r['gpu_time'] > 0 else 0.001 for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.semilogy(tests, cpu_times, 'o-', label='CPU', linewidth=2, markersize=8, color='blue')
    ax.semilogy(tests, gpu_times, 's-', label='GPU', linewidth=2, markersize=8, color='red')
    
    ax.set_xlabel('Test Number')
    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title('Performance Scaling (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_scaling.png', dpi=150)
    plt.close()
    print("Created: performance_scaling.png")

def create_summary_plot(results):
    """Create comprehensive summary plot with 4 subplots"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    tests = [r['test'] for r in results]
    cpu_times = [r['cpu_time'] for r in results]
    gpu_times = [r['gpu_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Plot 1: Bar comparison
    x = range(len(tests))
    width = 0.35
    ax1.bar([i - width/2 for i in x], cpu_times, width, label='CPU', color='blue', alpha=0.7)
    ax1.bar([i + width/2 for i in x], gpu_times, width, label='GPU', color='red', alpha=0.7)
    ax1.set_xlabel('Test')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('CPU vs GPU Timing')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tests)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup bars
    valid_tests = [t for t, s in zip(tests, speedups) if s > 0]
    valid_speedups = [s for s in speedups if s > 0]
    if valid_speedups:
        colors = ['green' if s >= 2 else 'orange' for s in valid_speedups]
        ax2.bar(range(len(valid_tests)), valid_speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--')
        ax2.set_xlabel('Test')
        ax2.set_ylabel('Speedup')
        ax2.set_title('GPU Speedup Factor')
        ax2.set_xticks(range(len(valid_tests)))
        ax2.set_xticklabels(valid_tests)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log scale
    cpu_log = [t if t > 0 else 0.001 for t in cpu_times]
    gpu_log = [t if t > 0 else 0.001 for t in gpu_times]
    ax3.semilogy(tests, cpu_log, 'o-', label='CPU', color='blue')
    ax3.semilogy(tests, gpu_log, 's-', label='GPU', color='red')
    ax3.set_xlabel('Test')
    ax3.set_ylabel('Time (s, log)')
    ax3.set_title('Log Scale Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pie chart of speedup distribution
    if valid_speedups:
        fast = sum(1 for s in valid_speedups if s >= 5)
        medium = sum(1 for s in valid_speedups if 2 <= s < 5)
        slow = sum(1 for s in valid_speedups if s < 2)
        sizes = [fast, medium, slow]
        labels = [f'>5x ({fast})', f'2-5x ({medium})', f'<2x ({slow})']
        colors = ['green', 'orange', 'red']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        ax4.set_title('Speedup Distribution')
    
    plt.tight_layout()
    plt.savefig('report_summary.png', dpi=150)
    plt.close()
    print("Created: report_summary.png")

def print_summary(results):
    """Print text summary"""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Test':<8}{'CPU (s)':<12}{'GPU (s)':<12}{'Speedup':<10}")
    print("-" * 42)
    
    for r in results:
        cpu_str = f"{r['cpu_time']:.3f}" if r['cpu_ok'] else "FAIL"
        gpu_str = f"{r['gpu_time']:.3f}" if r['gpu_ok'] else "FAIL"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] > 0 else "N/A"
        print(f"{r['test']:<8}{cpu_str:<12}{gpu_str:<12}{speedup_str:<10}")
    
    valid_speedups = [r['speedup'] for r in results if r['speedup'] > 0]
    if valid_speedups:
        print("-" * 42)
        print(f"Average Speedup: {sum(valid_speedups)/len(valid_speedups):.2f}x")
        print(f"Maximum Speedup: {max(valid_speedups):.2f}x")
        print(f"Minimum Speedup: {min(valid_speedups):.2f}x")

def main():
    print("=" * 60)
    print("REPORT FIGURE GENERATOR")
    print("=" * 60)
    
    # Check executable
    if not Path("./pauli_propagation_gpu.exe").exists():
        print("Error: pauli_propagation_gpu.exe not found")
        print("Build with: cd src && make all")
        sys.exit(1)
    
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available - will only print text results")
        print("Install with: pip install matplotlib")
    
    # Collect data
    print("\nPhase 1: Collecting benchmark data")
    print("-" * 60)
    results = collect_data()
    
    # Create plots
    if HAS_MATPLOTLIB:
        print("\nPhase 2: Generating plots")
        print("-" * 60)
        create_timing_plot(results)
        create_speedup_plot(results)
        create_scaling_plot(results)
        create_summary_plot(results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    if HAS_MATPLOTLIB:
        print("Generated files:")
        print("  - timing_comparison.png")
        print("  - speedup_chart.png")
        print("  - performance_scaling.png")
        print("  - report_summary.png")

if __name__ == "__main__":
    main()
