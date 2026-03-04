#!/usr/bin/env python3
"""
generate_report_figures.py
==========================
Generates paper-quality figures for the Parallel Pauli Paths report.

Deliverable: Table + plots showing
    CPU (1-thread)  vs  CPU (OpenMP, 16-thread)  vs  GPU
across all 10 stress-test configurations (STRESS 23–32, CLI indices 24–33).

Usage (from repo root):
    python3 scripts/generate_report_figures.py

Output files (in repo root):
    timing_comparison.png   — grouped bar chart: CPU-seq / OMP / GPU times
    speedup_chart.png       — speedup bars: OMP/CPU and GPU/CPU
    performance_scaling.png — log-scale line chart
    report_summary.png      — 4-panel summary figure
"""

import subprocess, re, sys, csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print('[WARN] matplotlib not found — only text output will be produced')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO        = SCRIPTS_DIR.parent

CPU_EXE = REPO / "pauli_propagation_cpu.exe"
OMP_EXE = REPO / "pauli_propagation_omp.exe"
GPU_EXE = REPO / "pauli_propagation_gpu.exe"

# Stress tests: CLI indices 24–33 → "STRESS 23" through "STRESS 32"
STRESS_INDICES = list(range(24, 34))
STRESS_LABELS  = [f'S{i-23}' for i in STRESS_INDICES]   # S1 … S10

# OMP thread counts for the comparison
OMP_THREADS = [1, 2, 4, 8, 16]

TIME_RE = re.compile(r'[Pp]ropagation completed in ([\d.]+) seconds')

# Colour scheme (colour-blind friendly)
COL = {
    'CPU-seq': '#2166ac',   # blue
    'OMP-1':   '#abd9e9',
    'OMP-2':   '#74add1',
    'OMP-4':   '#f46d43',
    'OMP-8':   '#d73027',
    'OMP-16':  '#a50026',   # dark red
    'GPU':     '#1a9641',   # green
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_one(exe: Path, idx: int, mode: str, extra=None, timeout=300) -> float:
    """Return wall-clock seconds, or -1.0 on failure."""
    cmd = [str(exe), str(idx), mode] + (extra or [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=str(REPO))
        m = TIME_RE.search(r.stdout)
        return float(m.group(1)) if m else -1.0
    except Exception:
        return -1.0

# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
def collect_data(run_gpu=True, omp_threads=None):
    """
    Returns a list of dicts, one per stress test:
        { 'idx': int, 'label': str,
          'CPU-seq': float, 'OMP-1': float, ..., 'OMP-16': float, 'GPU': float }
    """
    if omp_threads is None:
        omp_threads = OMP_THREADS

    rows = []
    for idx, lbl in zip(STRESS_INDICES, STRESS_LABELS):
        row = {'idx': idx, 'label': lbl}

        # CPU sequential
        if CPU_EXE.exists():
            t = run_one(CPU_EXE, idx, 'cpu')
            row['CPU-seq'] = t
            print(f'  {lbl}  CPU-seq={t:.3f}s' if t > 0 else f'  {lbl}  CPU-seq=FAIL', end='')
        else:
            row['CPU-seq'] = -1.0
            print(f'  {lbl}  CPU-seq=N/A', end='')

        # OMP at each thread count
        if OMP_EXE.exists():
            for nt in omp_threads:
                t = run_one(OMP_EXE, idx, 'omp', ['-j', str(nt)])
                row[f'OMP-{nt}'] = t
                print(f'  OMP-{nt}={t:.3f}s' if t > 0 else f'  OMP-{nt}=FAIL', end='')
        else:
            for nt in omp_threads:
                row[f'OMP-{nt}'] = -1.0

        # GPU
        if run_gpu and GPU_EXE.exists():
            t = run_one(GPU_EXE, idx, 'gpu', timeout=600)
            row['GPU'] = t
            print(f'  GPU={t:.3f}s' if t > 0 else f'  GPU=FAIL', end='')
        else:
            row['GPU'] = -1.0

        print()
        rows.append(row)

    return rows

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
def save_csv(rows, path=None):
    if path is None:
        path = REPO / 'benchmark_results_report.csv'
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'  Saved CSV: {path}')

# ---------------------------------------------------------------------------
# Print text table (the paper deliverable even without matplotlib)
# ---------------------------------------------------------------------------
def print_table(rows):
    """
    Prints the key paper table:
    CPU (1-thread) | OMP-4 | OMP-8 | OMP-16 | GPU | Speedup(OMP-16/CPU) | Speedup(GPU/CPU)
    """
    print()
    print('=' * 95)
    print('BENCHMARK TABLE  —  CPU-seq vs OpenMP vs GPU  (stress tests STRESS 23–32)')
    print('=' * 95)
    hdr = f"{'Test':<6}{'CPU-seq':>10}{'OMP-4':>9}{'OMP-8':>9}{'OMP-16':>9}{'GPU':>10}"
    hdr += f"{'S(OMP-16)':>11}{'S(GPU)':>10}{'S(GPU/OMP)':>12}"
    print(hdr)
    print('-' * 95)

    avg = {'s_omp': [], 's_gpu': [], 's_gpu_omp': []}
    for r in rows:
        cpu = r.get('CPU-seq', -1)
        o4  = r.get('OMP-4',  -1)
        o8  = r.get('OMP-8',  -1)
        o16 = r.get('OMP-16', -1)
        gpu = r.get('GPU',    -1)

        def fmt(t):
            return f'{t:.3f}s' if t > 0 else 'N/A'

        s_omp     = cpu / o16 if cpu > 0 and o16 > 0 else float('nan')
        s_gpu     = cpu / gpu if cpu > 0 and gpu > 0 else float('nan')
        s_gpu_omp = o16 / gpu if o16 > 0 and gpu > 0 else float('nan')

        def fmts(v):
            return f'{v:.1f}x' if v == v else 'N/A'

        print(f"{r['label']:<6}{fmt(cpu):>10}{fmt(o4):>9}{fmt(o8):>9}{fmt(o16):>9}"
              f"{fmt(gpu):>10}{fmts(s_omp):>11}{fmts(s_gpu):>10}{fmts(s_gpu_omp):>12}")

        if s_omp == s_omp:     avg['s_omp'].append(s_omp)
        if s_gpu == s_gpu:     avg['s_gpu'].append(s_gpu)
        if s_gpu_omp == s_gpu_omp: avg['s_gpu_omp'].append(s_gpu_omp)

    print('-' * 95)
    def a(lst): return f'{sum(lst)/len(lst):.1f}x' if lst else 'N/A'
    print(f"{'AVG':<6}{'':>10}{'':>9}{'':>9}{'':>9}{'':>10}"
          f"{a(avg['s_omp']):>11}{a(avg['s_gpu']):>10}{a(avg['s_gpu_omp']):>12}")
    print('=' * 95)
    print('S(OMP-16) = CPU-seq / OMP-16   S(GPU) = CPU-seq / GPU   S(GPU/OMP) = OMP-16 / GPU')
    print()

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_timing_comparison(rows):
    """Grouped bar chart: CPU-seq / OMP-16 / GPU wall-clock times."""
    if not HAS_PLT:
        return
    import numpy as np

    labels  = [r['label'] for r in rows]
    cpu     = [r.get('CPU-seq', 0) for r in rows]
    omp16   = [r.get('OMP-16', 0)  for r in rows]
    gpu     = [r.get('GPU', 0)     for r in rows]

    x = np.arange(len(labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w,   cpu,   w, label='CPU-seq',  color=COL['CPU-seq'], alpha=0.88)
    b2 = ax.bar(x,       omp16, w, label='OMP-16',   color=COL['OMP-16'],  alpha=0.88)
    b3 = ax.bar(x + w,   gpu,   w, label='GPU',      color=COL['GPU'],     alpha=0.88)

    ax.set_xlabel('Stress test (STRESS 23–32)', fontsize=12)
    ax.set_ylabel('Wall-clock time (s)', fontsize=12)
    ax.set_title('CPU-seq  vs  OMP 16-thread  vs  GPU\nPauli Propagation Stress Tests',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # annotate bars with values
    for bar in [b1, b2, b3]:
        for rect in bar:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout()
    out = REPO / 'timing_comparison.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out.name}')


def fig_speedup_chart(rows):
    """Speedup over CPU-seq: OMP-16 and GPU side by side."""
    if not HAS_PLT:
        return
    import numpy as np

    labels = [r['label'] for r in rows]
    s_omp  = [r.get('CPU-seq', -1) / r.get('OMP-16', -1)
              if r.get('CPU-seq', -1) > 0 and r.get('OMP-16', -1) > 0 else 0
              for r in rows]
    s_gpu  = [r.get('CPU-seq', -1) / r.get('GPU', -1)
              if r.get('CPU-seq', -1) > 0 and r.get('GPU', -1) > 0 else 0
              for r in rows]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w/2, s_omp, w, label='OMP-16 / CPU-seq', color=COL['OMP-16'], alpha=0.88)
    b2 = ax.bar(x + w/2, s_gpu, w, label='GPU / CPU-seq',     color=COL['GPU'],    alpha=0.88)

    ax.axhline(1, color='black', ls='--', lw=1, label='1× (no speedup)')
    ax.set_xlabel('Stress test', fontsize=12)
    ax.set_ylabel('Speedup over CPU-seq', fontsize=12)
    ax.set_title('Speedup over Single-Threaded CPU (std::map)\nOMP-16 vs GPU',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar in [b1, b2]:
        for rect in bar:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h:.0f}×', xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out = REPO / 'speedup_chart.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out.name}')


def fig_performance_scaling(rows):
    """Log-scale line chart: CPU-seq / OMP-{1,2,4,8,16} / GPU."""
    if not HAS_PLT:
        return

    labels = [r['label'] for r in rows]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(13, 6))

    series = [('CPU-seq', '-o'), ('OMP-1', '-s'), ('OMP-2', '-^'),
              ('OMP-4', '-D'), ('OMP-8', '-v'), ('OMP-16', '-P'), ('GPU', '-*')]
    for key, fmt in series:
        vals = [r.get(key, -1) for r in rows]
        if all(v < 0 for v in vals):
            continue
        plot_vals = [v if v > 0 else float('nan') for v in vals]
        ax.semilogy(x, plot_vals, fmt, label=key, color=COL.get(key, None),
                    linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Stress test', fontsize=12)
    ax.set_ylabel('Wall-clock time (s, log scale)', fontsize=12)
    ax.set_title('Performance Scaling — All Configurations (Log Scale)', fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    out = REPO / 'performance_scaling.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out.name}')


def fig_report_summary(rows):
    """4-panel figure: timing | speedup | thread scaling | speedup-dist."""
    if not HAS_PLT:
        return
    import numpy as np

    labels = [r['label'] for r in rows]
    n = len(labels)
    x = np.arange(n)
    cpu   = np.array([r.get('CPU-seq', 0) for r in rows])
    omp16 = np.array([r.get('OMP-16',  0) for r in rows])
    gpu   = np.array([r.get('GPU',     0) for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Parallel Pauli Propagation — Performance Summary', fontsize=15, y=1.01)

    # ── 1. Timing bar chart ───────────────────────────────────────────────
    ax = axes[0, 0]
    w = 0.26
    ax.bar(x - w,   cpu,   w, label='CPU-seq',  color=COL['CPU-seq'], alpha=0.85)
    ax.bar(x,       omp16, w, label='OMP-16',   color=COL['OMP-16'],  alpha=0.85)
    ax.bar(x + w,   gpu,   w, label='GPU',      color=COL['GPU'],     alpha=0.85)
    ax.set_title('Wall-Clock Time (Stress Tests)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Time (s)'); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    # ── 2. Speedup over CPU-seq ───────────────────────────────────────────
    ax = axes[0, 1]
    s_omp = np.where((cpu > 0) & (omp16 > 0), cpu / omp16, 0)
    s_gpu = np.where((cpu > 0) & (gpu  > 0), cpu / gpu,  0)
    ax.bar(x - 0.2, s_omp, 0.38, label='OMP-16 / CPU-seq', color=COL['OMP-16'], alpha=0.85)
    ax.bar(x + 0.2, s_gpu, 0.38, label='GPU / CPU-seq',    color=COL['GPU'],    alpha=0.85)
    ax.axhline(1, color='k', ls='--', lw=1)
    ax.set_title('Speedup over CPU-seq')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Speedup'); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    # ── 3. OMP thread-scaling (avg over all stress tests) ────────────────
    ax = axes[1, 0]
    nt_vals, avg_sup = [], []
    for nt in OMP_THREADS:
        key = f'OMP-{nt}'
        sups = [r['CPU-seq'] / r[key] for r in rows
                if r.get('CPU-seq', -1) > 0 and r.get(key, -1) > 0]
        if sups:
            nt_vals.append(nt)
            avg_sup.append(sum(sups) / len(sups))
    if nt_vals:
        ax.plot(nt_vals, avg_sup, 'o-', color=COL['OMP-16'], lw=2, ms=8, label='OMP speedup')
        ax.plot(nt_vals, nt_vals, '--', color='gray', lw=1, label='Linear ideal')
        ax.set_xlabel('OMP thread count'); ax.set_ylabel('Avg speedup over CPU-seq')
        ax.set_title('OMP Thread Scaling (avg over STRESS 23–32)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xticks(nt_vals)
    else:
        ax.text(0.5, 0.5, 'OMP data\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('OMP Thread Scaling')

    # ── 4. GPU/OMP-16 speedup (how much GPU wins vs parallel baseline) ────
    ax = axes[1, 1]
    s_gpu_omp = np.where((omp16 > 0) & (gpu > 0), omp16 / gpu, 0)
    colors = [COL['GPU'] if s > 10 else '#74c476' if s > 3 else '#d9f0a3'
              for s in s_gpu_omp]
    ax.bar(x, s_gpu_omp, color=colors, alpha=0.88)
    ax.axhline(1, color='k', ls='--', lw=1)
    valid = [s for s in s_gpu_omp if s > 0]
    if valid:
        avg_v = sum(valid) / len(valid)
        ax.axhline(avg_v, color='red', ls=':', lw=1.5, label=f'avg {avg_v:.1f}×')
        ax.legend(fontsize=9)
    ax.set_title('GPU / OMP-16 Speedup\n(GPU wins even vs 16-thread CPU)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('GPU speedup over OMP-16')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = REPO / 'report_summary.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out.name}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('=' * 70)
    print('REPORT FIGURE GENERATOR  —  CPU-seq / OMP / GPU comparison')
    print('=' * 70)

    has_gpu = GPU_EXE.exists()
    has_omp = OMP_EXE.exists()
    has_cpu = CPU_EXE.exists()

    if not has_cpu and not has_omp and not has_gpu:
        print('ERROR: no executables found. Build with: python3 scripts/build.py cpu omp')
        sys.exit(1)

    print(f'\n  CPU-seq : {"found" if has_cpu else "NOT found"}')
    print(f'  OMP     : {"found" if has_omp else "NOT found"}')
    print(f'  GPU     : {"found" if has_gpu else "NOT found (GPU skipped)"}')
    print(f'\n  Running stress tests: {STRESS_INDICES}')
    print(f'  OMP thread counts  : {OMP_THREADS}\n')

    # ── Collect data ──────────────────────────────────────────────────────
    rows = collect_data(run_gpu=has_gpu, omp_threads=OMP_THREADS)

    # ── Save CSV ──────────────────────────────────────────────────────────
    save_csv(rows)

    # ── Print paper table ─────────────────────────────────────────────────
    print_table(rows)

    # ── Figures ───────────────────────────────────────────────────────────
    if HAS_PLT:
        print('Generating figures...')
        fig_timing_comparison(rows)
        fig_speedup_chart(rows)
        fig_performance_scaling(rows)
        fig_report_summary(rows)
    else:
        print('[SKIP] Figures skipped — install matplotlib: pip install matplotlib')

    print()
    print('=' * 70)
    print('COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
