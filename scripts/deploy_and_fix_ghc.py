#!/usr/bin/env python3
"""
deploy_and_fix_ghc.py
=====================
Uploads all changed local files to GHC via SFTP, then:
  1. Installs qiskit (pip install --user)
  2. Diagnoses and shows the actual error from the failing scripts
  3. Tests correctness_validation.py and generate_report_figures.py

Usage (from repo root on Windows):
    python scripts/deploy_and_fix_ghc.py
"""

import paramiko, sys, os, textwrap
from pathlib import Path

# Force ASCII-safe stdout on Windows (avoids cp1252 encoding errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HOST     = "ghc43.ghc.andrew.cmu.edu"
USER     = "arulm"
PASSWORD = "BenandKen1$"
REMOTE   = "Parallelizing-Pauli-Paths"   # relative to home dir

REPO = Path(__file__).resolve().parent.parent

# Files to upload  (local path → remote path relative to REMOTE/)
UPLOAD = [
    ("scripts/correctness_validation.py",  "scripts/correctness_validation.py"),
    ("scripts/generate_report_figures.py", "scripts/generate_report_figures.py"),
    ("scripts/generate_all_figures.py",    "scripts/generate_all_figures.py"),
    ("scripts/run_benchmark.py",           "scripts/run_benchmark.py"),
    ("scripts/verify_correctness.py",      "scripts/verify_correctness.py"),
    ("scripts/verify_omp_correctness.py",  "scripts/verify_omp_correctness.py"),
    ("scripts/build.py",                   "scripts/build.py"),
    ("scripts/benchmark_julia.jl",         "scripts/benchmark_julia.jl"),
    ("scripts/benchmark_qiskit.py",        "scripts/benchmark_qiskit.py"),
    ("src/pauli_omp.h",                    "src/pauli_omp.h"),
    ("src/pauli_omp.cpp",                  "src/pauli_omp.cpp"),
    ("src/verify_correctness.cpp",         "src/verify_correctness.cpp"),
    ("src/tests.h",                        "src/tests.h"),
    ("src/tests.cpp",                      "src/tests.cpp"),
    ("src/main.cpp",                       "src/main.cpp"),
    ("src/Makefile",                       "src/Makefile"),
    ("COMMANDS.md",                        "COMMANDS.md"),
]

# ─────────────────────────────────────────────────────────────────────────────
def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    return client

def _safe(s):
    return s.encode('ascii', errors='replace').decode('ascii')

def run(client, cmd, label=""):
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode(errors='replace')
    err = stderr.read().decode(errors='replace')
    if label:
        print(f"\n  [{label}]")
        if out: print(_safe(textwrap.indent(out.strip(), "    ")))
        if err: print(_safe(textwrap.indent(err.strip(), "    ERR: ")))
    return out, err

def sftp_makedirs(sftp, remote_dir):
    """Create remote directory and all parents (like mkdir -p)."""
    parts = remote_dir.split('/')
    current = ''
    for part in parts:
        if not part:
            continue
        current = (current + '/' + part) if current else part
        try:
            sftp.stat(current)
        except (FileNotFoundError, IOError):
            try:
                sftp.mkdir(current)
            except Exception:
                pass  # might already exist in a race

def upload_files(client):
    # Discover actual home/project directory
    _, so, _ = client.exec_command("echo $HOME")
    home = so.read().decode().strip()
    # Try both possible remote paths
    for candidate in [f"{home}/{REMOTE}", f"{home}/private/15418/{REMOTE}"]:
        _, so2, _ = client.exec_command(f"test -d {candidate} && echo YES")
        if "YES" in so2.read().decode():
            remote_base = candidate
            break
    else:
        remote_base = f"{home}/{REMOTE}"

    sftp = client.open_sftp()
    print(f"\n  Uploading {len(UPLOAD)} files to {remote_base}/")
    for local_rel, remote_rel in UPLOAD:
        local  = REPO / local_rel
        remote = f"{remote_base}/{remote_rel}"
        if not local.exists():
            print(f"    [SKIP] {local_rel} (not found locally)")
            continue
        remote_dir = remote.rsplit('/', 1)[0]
        sftp_makedirs(sftp, remote_dir)
        sftp.put(str(local), remote)
        print(f"    [OK]  {local_rel}")
    sftp.close()
    return remote_base

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  GHC DEPLOY + FIX")
    print(f"  Host: {HOST}")
    print("=" * 65)

    print(f"\nConnecting to {HOST}...")
    try:
        client = connect()
        print("  Connected.\n")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # ── 1. Upload files ────────────────────────────────────────────────────
    print("Step 1: Upload updated files")
    remote_base = upload_files(client)
    print(f"  Remote project: {remote_base}")

    # ── 2. Install qiskit (done after rb is set, uses correct path) ───────
    # (moved below rb assignment — see Step 2b)

    # ── 3. Check Julia ────────────────────────────────────────────────────
    print("\nStep 3: Check Julia availability")
    out, _ = run(client, "which julia 2>/dev/null || conda run julia --version 2>/dev/null || echo NOJULIA")
    if "NOJULIA" in out or not out.strip():
        # Try conda-based julia
        out2, _ = run(client, "conda install -y -c conda-forge julia 2>&1 | tail -3")
        out3, _ = run(client, "which julia 2>/dev/null || echo NOJULIA")
        if "NOJULIA" in out3:
            print("  [SKIP] Julia not available on GHC — use Option C (run on local machine)")
            print("         See COMMANDS.md §6 for instructions.")
        else:
            print(f"  [OK] Julia at {out3.strip()}")
    else:
        print(f"  [OK] Julia at {out.strip()}")

    rb = remote_base  # shorter alias

    # ── 2b. Install qiskit ────────────────────────────────────────────────
    print("\nStep 2b: Install qiskit")
    # Try venv first (avoids AFS write quota for pip cache)
    out2, _ = run(client,
        "test -f ~/.pauli_venv/bin/pip && echo EXISTS || "
        "python3 -m venv ~/.pauli_venv && echo CREATED")
    out3, _ = run(client,
        "~/.pauli_venv/bin/pip install qiskit 2>&1 | tail -4")
    if "Successfully installed" in out3 or "already satisfied" in out3.lower():
        venv_python = "~/.pauli_venv/bin/python3"
        print(f"  [OK] qiskit ready in venv ({venv_python})")
        # Patch benchmark_qiskit.py shebang to use venv python
        run(client,
            f"sed -i '1s|.*|#!{venv_python}|' {rb}/scripts/benchmark_qiskit.py")
        print(f"  [OK] benchmark_qiskit.py patched to use venv python")
    else:
        print("  [WARN] qiskit install failed (disk quota?)")
        print(_safe("    " + out3.strip()[:200]))

    # ── 3b. Free up AFS disk space ────────────────────────────────────────
    print("\nStep 3b: Free disk space (pycache, old logs, etc.)")
    run(client,
        f"find {rb} -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null; "
        f"find {rb} -name '*.pyc' -delete 2>/dev/null; "
        f"rm -f {rb}/*.log {rb}/ghc_run.log 2>/dev/null; "
        f"echo 'cleaned'")
    out, _ = run(client, f"fs quota {rb} 2>/dev/null || du -sh {rb} 2>/dev/null | head -2")
    print(_safe(f"  Disk: {out.strip()[:200]}"))

    # ── 4. Rebuild executables ────────────────────────────────────────────
    print("\nStep 4: Rebuild executables")
    run(client, f"cd {rb}/src && make clean && make cpu && make omp 2>&1 | tail -10",
        label="make cpu+omp")
    # GPU build (requires nvcc)
    out_gpu, _ = run(client, f"which nvcc 2>/dev/null && echo HAS_NVCC || echo NO_NVCC")
    if "HAS_NVCC" in out_gpu:
        run(client, f"cd {rb}/src && make gpu 2>&1 | tail -8", label="make gpu")
    else:
        print("  [INFO] nvcc not found on this GHC node — GPU exe will use existing binary")
    # verify exes
    out, _ = run(client,
        f"ls -lh {rb}/pauli_propagation_*.exe 2>/dev/null || echo 'no exe found'")
    print(_safe(f"  Exes: {out.strip()[:300]}"))

    # ── 5. Syntax-check updated scripts ───────────────────────────────────
    print("\nStep 5: Syntax-check updated scripts")
    for scr in ["scripts/correctness_validation.py",
                "scripts/generate_report_figures.py",
                "scripts/generate_all_figures.py"]:
        out, err = run(client,
            f"cd {rb} && python3 -m py_compile {scr} && echo 'OK: {scr}'")
        print(f"  {'OK' if 'OK:' in out else 'FAIL'} {scr}")
        if err.strip():
            print(f"    ERR: {err.strip()[:200]}")

    # ── 6. Quick smoke-test correctness_validation ─────────────────────────
    print("\nStep 6: Quick smoke-test correctness_validation.py (timeout 90s)")
    out, err = run(client,
        f"cd {rb} && timeout 90 python3 scripts/correctness_validation.py 2>&1 | head -50",
        label="correctness_validation")

    # ── 7. Quick smoke-test generate_report_figures ────────────────────────
    print("\nStep 7: Quick smoke-test generate_report_figures.py (timeout 90s)")
    out, err = run(client,
        f"cd {rb} && timeout 90 python3 scripts/generate_report_figures.py 2>&1 | head -30",
        label="generate_report_figures")

    # ── 8. Full generate_all_figures run in background ────────────────────
    print("\nStep 8: Full generate_all_figures.py + report figures (background)")
    run(client, f"pkill -f generate_all_figures 2>/dev/null; pkill -f generate_report 2>/dev/null; true")
    # Run both figure generators + report figures, copy PNGs out of /tmp when done
    script = (
        f"cd {rb} && "
        f"python3 scripts/generate_all_figures.py && "
        f"python3 scripts/generate_report_figures.py && "
        f"cp /tmp/pauli_results/*.png {rb}/ 2>/dev/null; "
        f"cp /tmp/pauli_results/*.csv  {rb}/ 2>/dev/null; "
        f"echo 'ALL DONE'"
    )
    run(client, f"nohup bash -c '{script}' > {rb}/ghc_run.log 2>&1 &")
    print(f"  [OK] Running in background.")
    print(f"  Monitor: tail -f {rb}/ghc_run.log")
    print(f"  When done, copy PNGs locally:")
    print(f"    scp arulm@ghc43.ghc.andrew.cmu.edu:{rb}/*.png ./images/")

    client.close()
    print()
    print("=" * 65)
    print("  DONE. Files uploaded and fixes applied.")
    print(f"  Monitor: ssh {USER}@{HOST}")
    print(f"           tail -f ~/{REMOTE}/ghc_run.log")
    print("=" * 65)


if __name__ == "__main__":
    main()
