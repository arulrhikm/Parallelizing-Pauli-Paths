#!/usr/bin/env python3
"""
Generate All Report Figures - Master script to create all plots for final report
"""

import subprocess
import sys
import os
from pathlib import Path

# Always resolve paths relative to this script's directory so the script
# works correctly regardless of where it is invoked from.
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parent

def run_script(script_path: Path, description: str) -> bool:
    """Run a sibling script, working directory set to repo root."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Running: {script_path.name}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=600,
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            print("  [OK] Completed successfully")
            lines = result.stdout.strip().split('\n')
            for line in lines[-3:]:
                if line.strip():
                    print(f"  {line}")
        else:
            print("  [FAIL]")
            if result.stderr:
                print(f"  Error: {result.stderr[:300]}")

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  [FAIL] Timeout (10 minutes)")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
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

    gpu_exe = REPO_ROOT / "pauli_propagation_gpu.exe"
    cpu_exe = REPO_ROOT / "pauli_propagation_cpu.exe"
    if not gpu_exe.exists() and not cpu_exe.exists():
        print("\n  [WARN] No pauli_propagation_*.exe found.")
        print("  Build with: cd src && make all   (or: python3 scripts/build.py cpu)")
        print("  Some scripts may fall back to CPU-only mode.\n")

    # Check matplotlib
    try:
        import matplotlib
        print(f"\n  [OK] matplotlib {matplotlib.__version__} available")
    except ImportError:
        print("\n  [WARN] matplotlib not installed — pip install matplotlib")

    # Scripts to run (all in the same scripts/ directory)
    scripts = [
        ("performance_analysis.py",      "1. Performance Analysis — Speedup plots"),
        ("algorithmic_visualization.py", "2. Algorithmic Visualization — Pauli dynamics"),
        ("correctness_validation.py",    "3. Correctness Validation — Test suite results"),
        ("generate_report_figures.py",   "4. Summary Figures — Overview plots"),
    ]

    results = {}
    for name, desc in scripts:
        path = SCRIPTS_DIR / name
        if path.exists():
            results[name] = run_script(path, desc)
        else:
            print(f"\n  [SKIP] {name} not found")
            results[name] = None   # None = skipped, not failed

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)

    ran      = {k: v for k, v in results.items() if v is not None}
    skipped  = [k for k, v in results.items() if v is None]
    passed   = sum(1 for v in ran.values() if v)

    print(f"\n  Ran:     {len(ran)}/{len(results)} scripts")
    print(f"  Passed:  {passed}/{len(ran)}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

    # List generated figure files (relative to repo root)
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
        "report_summary.png",
    ]

    found, missing = [], []
    for fig in figure_files:
        p = REPO_ROOT / fig
        if p.exists():
            found.append((fig, p.stat().st_size / 1024))
        else:
            missing.append(fig)

    print("\n  Generated figures:")
    for fig, kb in found:
        print(f"    [OK]  {fig}  ({kb:.1f} KB)")
    for fig in missing:
        print(f"    [--]  {fig}  (not generated)")

    print(f"\n  Total: {len(found)}/{len(figure_files)} figures present")
    print(f"\n  To copy figures to local machine (run on local terminal):")
    print(f"    scp arulm@ghc43.ghc.andrew.cmu.edu:~/Parallelizing-Pauli-Paths/*.png ./images/")

if __name__ == "__main__":
    main()

