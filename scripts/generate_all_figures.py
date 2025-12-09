#!/usr/bin/env python3
"""
Generate All Report Figures - Master script to create all plots for final report
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Running: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                               capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✓ Completed successfully")
            if result.stdout:
                # Show last few lines
                lines = result.stdout.strip().split('\n')
                for line in lines[-3:]:
                    if line.strip():
                        print(f"  {line}")
        else:
            print("✗ Failed")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("✗ Timeout (10 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("="*70)
    print("PAULI PROPAGATION - REPORT FIGURE GENERATOR")
    print("="*70)
    print("\nThis script will generate all figures for your final report:")
    print("  1. Performance Analysis")
    print("  2. Algorithmic Visualization")
    print("  3. Correctness Validation")
    print("  4. Summary Figures")
    
    if not Path("./pauli_propagation_gpu.exe").exists():
        print("\n✗ Error: pauli_propagation_gpu.exe not found")
        print("Build with: cd src && make all")
        sys.exit(1)
    
    # Check matplotlib
    try:
        import matplotlib
        print("\n✓ Matplotlib available")
    except ImportError:
        print("\n⚠ Warning: matplotlib not installed")
        print("Install with: pip install matplotlib")
        print("Some plots will not be generated")
    
    # Run all scripts
    scripts = [
        ("performance_analysis.py", "Performance Analysis - Speedup plots"),
        ("algorithmic_visualization.py", "Algorithmic Visualization - Pauli dynamics"),
        ("correctness_validation.py", "Correctness Validation - Test suite results"),
        ("generate_report_figures.py", "Summary Figures - Overview plots")
    ]
    
    results = {}
    for script, desc in scripts:
        if Path(script).exists():
            success = run_script(script, desc)
            results[script] = success
        else:
            print(f"\n⚠ {script} not found - skipping")
            results[script] = False
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} scripts")
    
    # List generated files
    print("\nGenerated Figures:")
    figure_files = [
        "performance_analysis.png",
        "parameter_analysis.png",
        "clifford_analysis.png",
        "memory_analysis.png",
        "pauli_evolution.png",
        "correctness_validation.png",
        "timing_comparison.png",
        "speedup_chart.png",
        "performance_scaling.png",
        "report_summary.png"
    ]
    
    found = []
    missing = []
    for fig in figure_files:
        if Path(fig).exists():
            size_kb = Path(fig).stat().st_size / 1024
            print(f"  ✓ {fig} ({size_kb:.1f} KB)")
            found.append(fig)
        else:
            missing.append(fig)
    
    if missing:
        print(f"\n⚠ Missing: {len(missing)} figures")
        print("  (Some may require matplotlib or test execution)")
    
    print(f"\n✓ Total: {len(found)} figures generated")
    print("\nFor interactive demo, run: python3 interactive_demo.py")

if __name__ == "__main__":
    main()

