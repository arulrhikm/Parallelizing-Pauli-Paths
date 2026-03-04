#!/usr/bin/env python3
"""
generate_all_figures.py  –  publication-quality plots from benchmark_summary.json
==================================================================================
Reads scripts/benchmark_summary.json (written by run_benchmark.py,
benchmark_qiskit.py, and verify_correctness.py) and produces four figures:

  images/fig1_time_comparison.png    – wall-clock time across all 23 tests
  images/fig2_speedup_scaling.png    – GPU & OMP speedup vs initial word count
  images/fig3_thread_scaling.png     – OMP strong-scaling (speedup vs threads)
  images/fig4_qiskit_vs_gpu.png      – our GPU/CPU vs Qiskit on Clifford tests

Run (from repo root):
    python3 scripts/generate_all_figures.py

The script silently skips any figure for which the required data is absent
(e.g. GPU rows are only present after running on a GHC node).
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import defaultdict

SCRIPTS_DIR  = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPTS_DIR.parent
SUMMARY_JSON = SCRIPTS_DIR / "benchmark_summary.json"
IMAGES_DIR   = REPO_ROOT / "images"

# ── palette (colorblind-friendly) ───────────────────────────────────────────
COLORS = {
    "cpu_seq" : "#d62728",   # red
    "omp_1"   : "#aec7e8",   # light blue
    "omp_2"   : "#6baed6",   # blue
    "omp_4"   : "#2171b5",   # mid blue
    "omp_8"   : "#08519c",   # dark blue
    "omp_16"  : "#08306b",   # navy
    "gpu"     : "#2ca02c",   # green
    "qiskit"  : "#ff7f0e",   # orange
    "julia"   : "#9467bd",   # purple
}

BACKEND_LABELS = {
    "cpu_seq" : "CPU-seq",
    "omp_1"   : "OMP-1t",
    "omp_2"   : "OMP-2t",
    "omp_4"   : "OMP-4t",
    "omp_8"   : "OMP-8t",
    "omp_16"  : "OMP-16t",
    "gpu"     : "GPU",
    "qiskit"  : "Qiskit",
    "julia"   : "Julia",
}

# Ordered test list for consistent x-axis
TEST_ORDER = [
    "STRESS 23: 7q, 2K words, 100 layers",
    "STRESS 24: 7q, 5K words, 150 layers",
    "STRESS 25: 7q, 3K words, 200 layers",
    "STRESS 26: 7q, 1K words, 300 layers",
    "STRESS 27: 7q, 4K words, 100 layers",
    "STRESS 28: 7q, 2K words, 250 layers",
    "STRESS 29: 7q, 1K words, 400 layers",
    "STRESS 30: 7q, 8K words, 50 layers",
    "STRESS 31: 7q, 500 words, 500 layers",
    "STRESS 32: 7q, 5K words, 120 layers",
    "SCALE-1: 9q, 10K words, 30 layers",
    "SCALE-2: 9q, 15K words, 30 layers",
    "SCALE-3: 9q, 20K words, 30 layers",
    "SCALE-4: 9q, 50K words, 20 layers",
    "SCALE-5: 9q, 100K words, 10 layers",
    "DIVERSE-1: 10q, 30K H+CNOT, 20L",
    "DIVERSE-2: 10q, 60K H+CNOT, 10L",
    "DIVERSE-3: 9q, 25K T+H+CNOT, 30L",
    "DIVERSE-4: 9q, 35K S+H+CNOT, 20L",
    "DIVERSE-5: 9q, 5K RZ+CNOT, 8L",
    "DIVERSE-6: 9q, 4K RX+H+CNOT, 6L",
    "DIVERSE-7: 10q, 25K H+S+T+CNOT, 15L",
    "DIVERSE-8: 9q, 8K RZ+RX+H+CNOT, 15L",
]

# Short x-axis labels (≤14 chars)
SHORT_LABELS = {
    "STRESS 23: 7q, 2K words, 100 layers" : "S23 7q 2K",
    "STRESS 24: 7q, 5K words, 150 layers" : "S24 7q 5K",
    "STRESS 25: 7q, 3K words, 200 layers" : "S25 7q 3K",
    "STRESS 26: 7q, 1K words, 300 layers" : "S26 7q 1K",
    "STRESS 27: 7q, 4K words, 100 layers" : "S27 7q 4K",
    "STRESS 28: 7q, 2K words, 250 layers" : "S28 7q 2K",
    "STRESS 29: 7q, 1K words, 400 layers" : "S29 7q 1K",
    "STRESS 30: 7q, 8K words, 50 layers"  : "S30 7q 8K",
    "STRESS 31: 7q, 500 words, 500 layers": "S31 7q 500",
    "STRESS 32: 7q, 5K words, 120 layers" : "S32 7q 5K",
    "SCALE-1: 9q, 10K words, 30 layers"   : "SC1 9q 10K",
    "SCALE-2: 9q, 15K words, 30 layers"   : "SC2 9q 15K",
    "SCALE-3: 9q, 20K words, 30 layers"   : "SC3 9q 20K",
    "SCALE-4: 9q, 50K words, 20 layers"   : "SC4 9q 50K",
    "SCALE-5: 9q, 100K words, 10 layers"  : "SC5 9q 100K",
    "DIVERSE-1: 10q, 30K H+CNOT, 20L"     : "D1 10q H",
    "DIVERSE-2: 10q, 60K H+CNOT, 10L"     : "D2 10q H",
    "DIVERSE-3: 9q, 25K T+H+CNOT, 30L"    : "D3 9q T+H",
    "DIVERSE-4: 9q, 35K S+H+CNOT, 20L"    : "D4 9q S+H",
    "DIVERSE-5: 9q, 5K RZ+CNOT, 8L"       : "D5 9q RZ",
    "DIVERSE-6: 9q, 4K RX+H+CNOT, 6L"     : "D6 9q RX",
    "DIVERSE-7: 10q, 25K H+S+T+CNOT, 15L" : "D7 10q H+S+T",
    "DIVERSE-8: 9q, 8K RZ+RX+H+CNOT, 15L" : "D8 9q RZ+RX",
}

# Approximate initial word count per test (for x-axis in scaling figures)
INIT_WORDS = {
    "STRESS 23: 7q, 2K words, 100 layers" :  2000,
    "STRESS 24: 7q, 5K words, 150 layers" :  5000,
    "STRESS 25: 7q, 3K words, 200 layers" :  3000,
    "STRESS 26: 7q, 1K words, 300 layers" :  1000,
    "STRESS 27: 7q, 4K words, 100 layers" :  4000,
    "STRESS 28: 7q, 2K words, 250 layers" :  2000,
    "STRESS 29: 7q, 1K words, 400 layers" :  1000,
    "STRESS 30: 7q, 8K words, 50 layers"  :  8000,
    "STRESS 31: 7q, 500 words, 500 layers":   500,
    "STRESS 32: 7q, 5K words, 120 layers" :  5000,
    "SCALE-1: 9q, 10K words, 30 layers"   : 10000,
    "SCALE-2: 9q, 15K words, 30 layers"   : 15000,
    "SCALE-3: 9q, 20K words, 30 layers"   : 20000,
    "SCALE-4: 9q, 50K words, 20 layers"   : 50000,
    "SCALE-5: 9q, 100K words, 10 layers"  :100000,
    "DIVERSE-1: 10q, 30K H+CNOT, 20L"     : 30000,
    "DIVERSE-2: 10q, 60K H+CNOT, 10L"     : 60000,
    "DIVERSE-3: 9q, 25K T+H+CNOT, 30L"    : 25000,
    "DIVERSE-4: 9q, 35K S+H+CNOT, 20L"    : 35000,
    "DIVERSE-5: 9q, 5K RZ+CNOT, 8L"       :  5000,
    "DIVERSE-6: 9q, 4K RX+H+CNOT, 6L"     :  4000,
    "DIVERSE-7: 10q, 25K H+S+T+CNOT, 15L" : 25000,
    "DIVERSE-8: 9q, 8K RZ+RX+H+CNOT, 15L" :  8000,
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> dict:
    """
    Returns data[test_name][backend_key] = {"time_s": float, "nterms": int}
    backend_key: "cpu_seq" | "omp_1" | "omp_4" | "omp_8" | "omp_16" |
                 "gpu" | "qiskit" | "julia"
    """
    if not SUMMARY_JSON.exists():
        print(f"  [WARN] {SUMMARY_JSON} not found — run benchmark scripts first.")
        return {}

    raw = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    data: dict = defaultdict(dict)

    for entry in raw.values():
        name    = entry.get("test_name", "")
        backend = entry.get("backend", "")
        threads = int(entry.get("threads", 1))
        t       = float(entry.get("time_s", -1))
        nt      = int(entry.get("nterms", -1))
        correct = entry.get("correct", "N/A")

        if t <= 0:
            continue

        # Build a canonical key
        if backend == "omp":
            key = f"omp_{threads}"
        elif backend in ("cpu_seq", "gpu", "qiskit", "julia"):
            key = backend
        else:
            continue

        # Keep entry if not already present, or if newer (overwrite)
        data[name][key] = {"time_s": t, "nterms": nt, "correct": correct}

    return dict(data)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def savefig(fig, path: Path, dpi: int = 150):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  [OK]  {path.relative_to(REPO_ROOT)}  ({path.stat().st_size/1024:.0f} KB)")


# ── Figure 1: Wall-clock time comparison ─────────────────────────────────────

def fig_time_comparison(data: dict, mpl):
    """
    Grouped bar chart: time (s) per test × backend.
    Shows cpu_seq, omp_8, omp_16, gpu, qiskit on one chart.
    """
    plt = mpl.pyplot
    np  = mpl.numpy

    backends = ["cpu_seq", "omp_8", "omp_16", "gpu", "qiskit"]
    tests    = [t for t in TEST_ORDER if any(t in data and bk in data[t] for bk in backends)]
    if not tests:
        print("  [SKIP] fig1 – no timing data found")
        return

    n_tests    = len(tests)
    n_backends = sum(1 for bk in backends if any(bk in data.get(t, {}) for t in tests))
    bar_w      = 0.8 / max(n_backends, 1)

    fig, ax = plt.subplots(figsize=(max(12, n_tests * 0.6), 5))

    offset_idx = 0
    for bk in backends:
        times = [data.get(t, {}).get(bk, {}).get("time_s", None) for t in tests]
        if all(v is None for v in times):
            continue
        x = np.arange(n_tests)
        heights = [v if v is not None else 0 for v in times]
        bars = ax.bar(x + offset_idx * bar_w, heights,
                      width=bar_w * 0.9,
                      color=COLORS.get(bk, "#888"),
                      label=BACKEND_LABELS.get(bk, bk),
                      zorder=3)
        # Mark missing bars
        for i, v in enumerate(times):
            if v is None:
                ax.text(x[i] + offset_idx * bar_w, 0.001, "—",
                        ha="center", va="bottom", fontsize=6, color="#aaa")
        offset_idx += 1

    ax.set_yscale("log")
    ax.set_xticks(np.arange(n_tests) + (offset_idx - 1) * bar_w / 2)
    ax.set_xticklabels([SHORT_LABELS.get(t, t[:10]) for t in tests],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title("Pauli Propagation: Wall-Clock Time by Backend and Test")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_xlim(-0.5, n_tests)

    savefig(fig, IMAGES_DIR / "fig1_time_comparison.png")
    plt.close(fig)


# ── Figure 2: Speedup scaling ─────────────────────────────────────────────────

def fig_speedup_scaling(data: dict, mpl):
    """
    Line plot: GPU speedup and OMP-8t/16t speedup over CPU-seq
    as a function of initial word count.
    """
    plt = mpl.pyplot
    import math

    tests = [t for t in TEST_ORDER
             if t in data and "cpu_seq" in data[t] and data[t]["cpu_seq"]["time_s"] > 0]
    if not tests:
        print("  [SKIP] fig2 – no cpu_seq data found")
        return

    def speedups_for(backend_key):
        xs, ys, labels = [], [], []
        for t in tests:
            cpu_t = data[t]["cpu_seq"]["time_s"]
            bk_t  = data[t].get(backend_key, {}).get("time_s")
            if bk_t and bk_t > 0:
                xs.append(INIT_WORDS.get(t, 0))
                ys.append(cpu_t / bk_t)
                labels.append(SHORT_LABELS.get(t, t[:8]))
        return xs, ys, labels

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for bk, marker in [("gpu", "o"), ("omp_16", "s"), ("omp_8", "^"), ("qiskit", "D")]:
        xs, ys, lbls = speedups_for(bk)
        if not xs:
            continue
        paired = sorted(zip(xs, ys, lbls))
        xs2, ys2, _ = zip(*paired)
        ax.plot(xs2, ys2, marker=marker, linewidth=2, markersize=6,
                color=COLORS.get(bk, "#888"),
                label=BACKEND_LABELS.get(bk, bk))
        plotted = True

    if not plotted:
        print("  [SKIP] fig2 – no gpu/omp data for speedup chart")
        plt.close(fig)
        return

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Baseline (1×)")
    ax.set_xscale("log")
    ax.set_xlabel("Initial Pauli-word count (log scale)")
    ax.set_ylabel("Speedup over CPU-seq")
    ax.set_title("Speedup vs Problem Size (Pauli-word count)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    savefig(fig, IMAGES_DIR / "fig2_speedup_scaling.png")
    plt.close(fig)


# ── Figure 3: OMP thread scaling ─────────────────────────────────────────────

def fig_thread_scaling(data: dict, mpl):
    """
    Strong-scaling chart: speedup over OMP-1t as a function of thread count,
    one line per test group (averaged within each group).
    """
    plt = mpl.pyplot
    np  = mpl.numpy

    thread_counts = [1, 2, 4, 8, 16]
    keys          = [f"omp_{j}" for j in thread_counts]

    # Group tests for cleaner plot
    groups = {
        "STRESS (7q)"  : [t for t in TEST_ORDER if t.startswith("STRESS")],
        "SCALE (9q)"   : [t for t in TEST_ORDER if t.startswith("SCALE")],
        "DIVERSE Cliff": [t for t in TEST_ORDER
                          if t.startswith("DIVERSE") and
                          not any(x in t for x in ("RZ", "RX", "T+"))],
        "DIVERSE Rot"  : [t for t in TEST_ORDER
                          if t.startswith("DIVERSE") and
                          any(x in t for x in ("RZ", "RX", "T+"))],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    group_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    plotted = False

    for (grp, grp_tests), color in zip(groups.items(), group_colors):
        sp_by_thread = {j: [] for j in thread_counts}
        for t in grp_tests:
            base = data.get(t, {}).get("omp_1", {}).get("time_s")
            if not base or base <= 0:
                continue
            for j, key in zip(thread_counts, keys):
                v = data.get(t, {}).get(key, {}).get("time_s")
                if v and v > 0:
                    sp_by_thread[j].append(base / v)

        xs, ys = [], []
        for j in thread_counts:
            if sp_by_thread[j]:
                xs.append(j)
                ys.append(sum(sp_by_thread[j]) / len(sp_by_thread[j]))

        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=6,
                    color=color, label=grp)
            plotted = True

    if not plotted:
        print("  [SKIP] fig3 – no OMP multi-thread data found")
        plt.close(fig)
        return

    # Ideal linear scaling reference
    ax.plot(thread_counts, thread_counts, "--", color="gray",
            linewidth=1.5, alpha=0.6, label="Ideal linear")

    ax.set_xlabel("OMP thread count")
    ax.set_ylabel("Speedup over OMP-1t")
    ax.set_title("OMP Strong Scaling (speedup vs thread count)")
    ax.set_xticks(thread_counts)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    savefig(fig, IMAGES_DIR / "fig3_thread_scaling.png")
    plt.close(fig)


# ── Figure 4: Qiskit comparison ───────────────────────────────────────────────

def fig_qiskit_vs_ours(data: dict, mpl):
    """
    Side-by-side bars: Qiskit evolve time vs our GPU and CPU-seq
    on the Clifford tests where Qiskit has results.
    """
    plt = mpl.pyplot
    np  = mpl.numpy

    # Only tests where Qiskit ran successfully
    tests = [t for t in TEST_ORDER
             if t in data and "qiskit" in data[t] and data[t]["qiskit"]["time_s"] > 0]
    if not tests:
        print("  [SKIP] fig4 – no Qiskit data found")
        return

    backends = [bk for bk in ("cpu_seq", "omp_8", "gpu", "qiskit")
                if any(bk in data.get(t, {}) for t in tests)]
    if len(backends) < 2:
        print("  [SKIP] fig4 – need at least two backends for comparison")
        return

    n  = len(tests)
    bw = 0.8 / len(backends)
    xs = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 5))
    for i, bk in enumerate(backends):
        heights = [data.get(t, {}).get(bk, {}).get("time_s", 0) or 0 for t in tests]
        ax.bar(xs + i * bw, heights, width=bw * 0.9,
               color=COLORS.get(bk, "#888"),
               label=BACKEND_LABELS.get(bk, bk), zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(xs + (len(backends) - 1) * bw / 2)
    ax.set_xticklabels([SHORT_LABELS.get(t, t[:10]) for t in tests],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title("Our GPU/CPU vs Qiskit SparsePauliOp (Clifford tests)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Annotate speedup (Qiskit / gpu) above each GPU bar
    gpu_idx = backends.index("gpu") if "gpu" in backends else None
    qk_idx  = backends.index("qiskit") if "qiskit" in backends else None
    if gpu_idx is not None and qk_idx is not None:
        for i, t in enumerate(tests):
            qt = data.get(t, {}).get("qiskit", {}).get("time_s", 0) or 0
            gt = data.get(t, {}).get("gpu",    {}).get("time_s", 0) or 0
            if qt > 0 and gt > 0:
                ax.text(xs[i] + gpu_idx * bw, gt * 1.15,
                        f"{qt/gt:.1f}x", ha="center", va="bottom",
                        fontsize=7, color="#555")

    savefig(fig, IMAGES_DIR / "fig4_qiskit_vs_gpu.png")
    plt.close(fig)


# ── Figure 5: Time vs output terms ───────────────────────────────────────────

def fig_time_vs_nterms(data: dict, mpl):
    """
    Scatter + trend: wall-clock time vs number of output Pauli words.
    Only tests/backends where nterms > 0 are plotted.
    """
    plt = mpl.pyplot
    import math

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for bk in ("cpu_seq", "omp_8", "gpu", "qiskit"):
        xs, ys = [], []
        for t in TEST_ORDER:
            entry = data.get(t, {}).get(bk, {})
            nt = entry.get("nterms", -1)
            tv = entry.get("time_s", -1)
            if nt and nt > 0 and tv and tv > 0:
                xs.append(nt)
                ys.append(tv)

        if not xs:
            continue

        ax.scatter(xs, ys, color=COLORS.get(bk, "#888"),
                   label=BACKEND_LABELS.get(bk, bk),
                   s=50, alpha=0.8, zorder=5)

        # Linear regression in log-log space
        if len(xs) >= 3:
            import math
            lx = [math.log10(x) for x in xs]
            ly = [math.log10(y) for y in ys]
            n_ = len(lx); sx = sum(lx); sy = sum(ly)
            sxx = sum(x*x for x in lx); sxy = sum(x*y for x, y in zip(lx, ly))
            slope = (n_*sxy - sx*sy) / (n_*sxx - sx*sx + 1e-15)
            inter = (sy - slope*sx) / n_
            x_fit = sorted(xs)
            y_fit = [10**(slope*math.log10(x)+inter) for x in x_fit]
            ax.plot(x_fit, y_fit, color=COLORS.get(bk, "#888"),
                    linewidth=1.5, linestyle="--", alpha=0.6)
        plotted = True

    if not plotted:
        print("  [SKIP] fig5 – no nterms data found")
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Output Pauli-word count (nterms, log scale)")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title("Time vs Output Pauli-word Count")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    savefig(fig, IMAGES_DIR / "fig5_time_vs_nterms.png")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  PAULI PROPAGATION — FIGURE GENERATOR")
    print(f"  Source: {SUMMARY_JSON}")
    print("=" * 64)

    # Check matplotlib
    try:
        import matplotlib as _mpl
        _mpl.use("Agg")          # non-interactive backend
        import matplotlib.pyplot
        import numpy
        _mpl.pyplot = matplotlib.pyplot
        _mpl.numpy  = numpy
        print(f"  matplotlib {_mpl.__version__}  /  numpy {numpy.__version__}\n")
    except ImportError:
        print("  [FAIL] matplotlib not installed.  pip install matplotlib")
        sys.exit(1)

    data = load_data()
    if not data:
        print("  No data loaded — run benchmark scripts first:\n"
              "    python3 scripts/run_benchmark.py --no-gpu   (on any machine)\n"
              "    python3 scripts/benchmark_qiskit.py\n"
              "    (GPU rows only available after running on a GHC GPU node)\n")
        sys.exit(0)

    # Count available data
    backends_found = set()
    for entries in data.values():
        backends_found.update(entries.keys())
    print(f"  Tests with data : {len(data)}")
    print(f"  Backends found  : {', '.join(sorted(backends_found))}\n")

    # Generate all 5 figures (skip gracefully if data is missing)
    fig_time_comparison(data, _mpl)
    fig_speedup_scaling(data, _mpl)
    fig_thread_scaling(data, _mpl)
    fig_qiskit_vs_ours(data, _mpl)
    fig_time_vs_nterms(data, _mpl)

    print()
    print("=" * 64)
    print("  DONE — figures written to images/")
    print("=" * 64)
    print()
    print("  To copy to local machine from GHC:")
    print("    scp arulm@ghc43.ghc.andrew.cmu.edu:~/Parallelizing-Pauli-Paths/images/*.png ./images/")


if __name__ == "__main__":
    main()
