#!/bin/bash
# Run this script ON THE GHC machine (e.g. after SSH) when "Disk quota exceeded"
# prevents git pull. It clones/pulls in /tmp (no AFS quota) then rsyncs
# tracked files into your AFS project directory.
#
# Usage:
#   cd ~/private/15418/Parallelizing-Pauli-Paths   # your AFS project path
#   bash scripts/pull_via_tmp.sh

set -e
REPO_URL="https://github.com/arulrhikm/Parallelizing-Pauli-Paths.git"
TMP_REPO="/tmp/Parallelizing-Pauli-Paths-pull"
AFS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Project dir (AFS): $AFS_ROOT"
echo "Clone/pull in:     $TMP_REPO"

# Clone or pull in /tmp (no quota)
if [ -d "$TMP_REPO/.git" ]; then
  echo "Updating existing clone in /tmp..."
  (cd "$TMP_REPO" && git fetch origin && git checkout main && git merge --ff-only origin/main)
else
  echo "Cloning into /tmp (shallow to save space)..."
  rm -rf "$TMP_REPO"
  git clone --depth 1 "$REPO_URL" "$TMP_REPO"
  (cd "$TMP_REPO" && git fetch --unshallow 2>/dev/null || true)
fi

# Copy tracked files from /tmp repo into AFS (overwrites with latest).
# We do NOT run git in AFS (e.g. git reset) so we don't hit quota again.
echo "Copying updated files to AFS project dir..."
rsync -av --delete \
  --exclude='.git' \
  --exclude='build' \
  --exclude='build_cpu' \
  --exclude='build_omp' \
  --exclude='*.exe' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='*.pyc' \
  "$TMP_REPO/" "$AFS_ROOT/"

echo "Done. Source tree in AFS is updated from origin/main."
echo "To fix 'git status' later (after freeing quota): cd $AFS_ROOT && git fetch && git reset --hard origin/main"
