#!/usr/bin/env python3
"""
correctness_validation.py
=========================
Runs all 34 tests on every available executable (CPU-seq, OMP-16, GPU)
and produces correctness_validation.png + validation_results.json.

Usage (from repo root):
    python3 scripts/correctness_validation.py
"""

import subprocess, re, sys, json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO        = SCRIPTS_DIR.parent

# Use /tmp as fallback when project directory is over AFS quota
def _output_dir():
    test_file = REPO / ".quota_test_cv"
    try:
        test_file.write_text("x")
        test_file.unlink()
        return REPO
    except OSError:
        d = Path("/tmp/pauli_results")
        d.mkdir(exist_ok=True)
        print(f"  [INFO] AFS quota exceeded — saving outputs to {d}")
        return d

OUT_DIR = _output_dir()
IMAGES_DIR  = REPO / "images"
try:
    IMAGES_DIR.mkdir(exist_ok=True)
except OSError:
    IMAGES_DIR = OUT_DIR

CPU_EXE = REPO / "pauli_propagation_cpu.exe"
OMP_EXE = REPO / "pauli_propagation_omp.exe"
GPU_EXE = REPO / "pauli_propagation_gpu.exe"

# CLI indices 0–33 map to test display numbers 1–34.
ALL_TESTS   = list(range(0, 34))   # all 34 tests
STRESS_IDX  = list(range(24, 34))  # stress tests 23–32 (display 25–34)

TIME_RE = re.compile(r'[Pp]ropagation completed in ([\d.]+) seconds')

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_one(exe: Path, idx: int, mode: str, extra: list = None, timeout=120) -> float:
    """Return wall-clock seconds, or -1.0 on failure."""
    cmd = [str(exe), str(idx), mode] + (extra or [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(REPO))
        m = TIME_RE.search(r.stdout)
        return float(m.group(1)) if m else -1.0
    except Exception:
        return -1.0

# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------
def collect(modes):
    """Run all 34 tests for each mode. modes = list of (label, exe, mode_flag, extra)."""
    rows = []
    total = len(ALL_TESTS)
    for i, idx in enumerate(ALL_TESTS):
        row = {'idx': idx, 'display': idx + 1}
        for label, exe, flag, extra in modes:
            t = run_one(exe, idx, flag, extra)
            row[label] = t
        rows.append(row)
        passed = sum(1 for _, __, ___, ____ in modes if row.get(modes[0][0], -1) >= 0)
        print(f"  [{i+1:2d}/{total}] test {idx:2d}: " +
              "  ".join(f"{l}={'N/A' if row[l]<0 else f'{row[l]:.3f}s'}"
                        for l, *_ in modes))
    return rows

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(rows, mode_labels):
    if not HAS_PLT:
        return
    import numpy as np

    all_idx  = [r['idx'] for r in rows]
    stress   = [r for r in rows if r['idx'] in STRESS_IDX]
    s_idx    = [r['display'] for r in stress]

    # colour palette
    pal = {'CPU-seq': '#2166ac', 'OMP-16': '#f4a582', 'GPU': '#d6604d'}
    default_colors = ['#2166ac', '#f4a582', '#d6604d', '#4dac26']

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Pauli Propagation — Correctness & Performance Validation', fontsize=14, y=1.01)

    # ── Panel 1: pass/fail over all 34 tests ──────────────────────────────
    ax = axes[0, 0]
    colors_pass = []
    statuses = []
    for r in rows:
        ok = all(r.get(l, -1) > 0 for l in mode_labels)
        statuses.append(ok)
        colors_pass.append('#4dac26' if ok else '#ca0020')
    ax.bar(all_idx, [1 if s else 0 for s in statuses], color=colors_pass, alpha=0.85, width=0.8)
    ax.set_xlim(-1, 34)
    ax.set_ylim(-0.05, 1.15)
    ax.axvline(x=23.5, color='gray', ls='--', lw=1, label='→ stress tests')
    n_pass = sum(statuses)
    ax.set_title(f'Test Suite: {n_pass}/{len(rows)} pass (all modes)')
    ax.set_xlabel('Test index (0-based)')
    ax.set_ylabel('Status (1 = PASS)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 2: wall-clock time on stress tests (grouped bars) ───────────
    ax = axes[0, 1]
    n = len(stress)
    x = np.arange(n)
    w = 0.8 / max(len(mode_labels), 1)
    for k, label in enumerate(mode_labels):
        times = [r.get(label, -1) for r in stress]
        valid = [t if t > 0 else 0 for t in times]
        offset = (k - len(mode_labels)/2 + 0.5) * w
        c = pal.get(label, default_colors[k % len(default_colors)])
        ax.bar(x + offset, valid, w * 0.9, label=label, color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{r["idx"]-23}' for r in stress], fontsize=8)
    ax.set_xlabel('Stress test')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Timing: Stress Tests (STRESS 23–32)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 3: speedup over CPU-seq on stress tests ─────────────────────
    ax = axes[1, 0]
    ref_label = 'CPU-seq' if 'CPU-seq' in mode_labels else mode_labels[0]
    speedup_labels = [l for l in mode_labels if l != ref_label]
    sp_pal = {'OMP-16': '#f4a582', 'GPU': '#d6604d'}
    for k, label in enumerate(speedup_labels):
        sups = []
        for r in stress:
            ref = r.get(ref_label, -1)
            t   = r.get(label, -1)
            sups.append(ref / t if ref > 0 and t > 0 else 0)
        offset = (k - len(speedup_labels)/2 + 0.5) * (0.8 / max(len(speedup_labels), 1))
        c = sp_pal.get(label, default_colors[k % len(default_colors)])
        ax.bar(x + offset, sups, 0.8 / max(len(speedup_labels), 1) * 0.9,
               label=f'{label} / {ref_label}', color=c, alpha=0.85)
    ax.axhline(1, color='k', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{r["idx"]-23}' for r in stress], fontsize=8)
    ax.set_xlabel('Stress test')
    ax.set_ylabel('Speedup over CPU-seq')
    ax.set_title(f'Speedup over {ref_label}')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 4: summary text table ───────────────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')
    lines = ['Summary', '─' * 38]
    lines.append(f"  {'Test':<8}" + "".join(f"{l:>10}" for l in mode_labels))
    lines.append('  ' + '─' * (6 + 10 * len(mode_labels)))
    for r in stress:
        row_str = f"  S{r['idx']-23:<7}"
        for l in mode_labels:
            t = r.get(l, -1)
            row_str += f"{'N/A':>10}" if t < 0 else f"{t:>9.2f}s"
        lines.append(row_str)
    # speedup summary
    if 'CPU-seq' in mode_labels:
        lines.append('')
        for l in [x for x in mode_labels if x != 'CPU-seq']:
            sups = [r['CPU-seq']/r[l] for r in stress
                    if r.get('CPU-seq',-1)>0 and r.get(l,-1)>0]
            if sups:
                lines.append(f"  Avg {l}/CPU-seq speedup: {sum(sups)/len(sups):.1f}x")
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
            fontsize=7.5, va='top', family='monospace')

    plt.tight_layout()
    out = OUT_DIR / 'correctness_validation.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('=' * 60)
    print('CORRECTNESS VALIDATION')
    print('=' * 60)

    # detect available executables
    modes = []
    if CPU_EXE.exists():
        modes.append(('CPU-seq', CPU_EXE, 'cpu', []))
    else:
        print(f'  [WARN] {CPU_EXE.name} not found — skipping CPU-seq')

    if OMP_EXE.exists():
        modes.append(('OMP-16',  OMP_EXE, 'omp', ['-j', '16']))
    else:
        print(f'  [WARN] {OMP_EXE.name} not found — skipping OMP-16')

    if GPU_EXE.exists():
        modes.append(('GPU',     GPU_EXE, 'gpu', []))
    else:
        print(f'  [INFO] {GPU_EXE.name} not found — GPU skipped (build with make gpu)')

    if not modes:
        print('ERROR: no executables found. Build with: python3 scripts/build.py cpu omp')
        sys.exit(1)

    mode_labels = [m[0] for m in modes]
    print(f'\n  Running {len(ALL_TESTS)} tests × {len(modes)} modes: {mode_labels}\n')

    rows = collect(modes)

    # save JSON
    out_json = OUT_DIR / 'validation_results.json'
    try:
        with open(out_json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'\n  Saved: {out_json}')
    except OSError as e:
        print(f'\n  [WARN] Could not save JSON: {e}')

    # generate plots
    make_plots(rows, mode_labels)

    # text summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    n_pass = sum(1 for r in rows if all(r.get(l, -1) > 0 for l in mode_labels))
    print(f'  All-mode pass: {n_pass}/{len(rows)} tests')

    stress = [r for r in rows if r['idx'] in STRESS_IDX]
    if 'CPU-seq' in mode_labels:
        for label in [l for l in mode_labels if l != 'CPU-seq']:
            sups = [r['CPU-seq'] / r[label] for r in stress
                    if r.get('CPU-seq', -1) > 0 and r.get(label, -1) > 0]
            if sups:
                print(f'  Avg {label}/CPU-seq speedup (stress): {sum(sups)/len(sups):.1f}x  '
                      f'(max {max(sups):.1f}x)')

    print('\n  Generated:')
    print('    correctness_validation.png')
    print('    validation_results.json')


if __name__ == '__main__':
    main()
