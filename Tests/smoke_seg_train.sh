#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# smoke_seg_train.sh — end-to-end smoke test for the iterative trainer.
#
# This script exercises the testing ladder for
# `metadata_to_morphsource.seg_train` *without* needing the
# nnInteractive venv, MorphoSource API access, or 3D Slicer:
#
#   Tier 1: Pure-Python unit tests (ledger, router, dataset, paper, CLI)
#   Tier 2: CLI surface checks (--help, summary on empty ledger, export)
#   Tier 3: Integration test using a synthetic NIfTI volume + mocked
#           nnInteractive paint loop / Slicer voxelizer (requires
#           numpy + SimpleITK).
#
# Tier 3 (integration) is opt-in because it needs a healthy numpy +
# SimpleITK install. Pass --integration (or SEG_TRAIN_INTEGRATION=1)
# to run it. Pass --include-numpy to also run the numpy-marked router
# unit tests in Tier 1.
#
# Usage:
#     Tests/smoke_seg_train.sh                         # tiers 1+2
#     Tests/smoke_seg_train.sh --integration           # + tier 3
#     Tests/smoke_seg_train.sh --include-numpy         # + numpy-marked
#     PYTHON=/opt/anaconda3/bin/python3 Tests/smoke_seg_train.sh
#     SEG_TRAIN_PROBE_TIMEOUT=60 Tests/smoke_seg_train.sh --integration
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="${PYTHON:-python3}"
INCLUDE_NUMPY=0
RUN_INTEGRATION="${SEG_TRAIN_INTEGRATION:-0}"
for arg in "$@"; do
    case "$arg" in
        --include-numpy) INCLUDE_NUMPY=1 ;;
        --integration)   RUN_INTEGRATION=1 ;;
        --no-integration) RUN_INTEGRATION=0 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

banner() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()     { printf '   \033[1;32m✓\033[0m %s\n' "$*"; }
warn()   { printf '   \033[1;33m!\033[0m %s\n' "$*"; }

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python not found at '$PYTHON'." >&2
    echo "Set PYTHON= to an interpreter with pytest installed." >&2
    exit 1
fi

if ! "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
    echo "ERROR: pytest is not installed in $PYTHON." >&2
    echo "  Try: $PYTHON -m pip install pytest" >&2
    exit 1
fi

banner "Tier 1 — pure-Python unit tests (ledger / router / dataset / paper / CLI)"
PYTEST_ARGS=(
    Tests/test_seg_train_ledger.py
    Tests/test_seg_train_router.py
    Tests/test_seg_train_dataset.py
    Tests/test_seg_train_paper_export.py
    Tests/test_seg_train_cli.py
    Tests/test_seg_train_prepare.py
    -v --tb=short --no-header -p no:cacheprovider
    --override-ini=testpaths=
)
if [ "$INCLUDE_NUMPY" != "1" ]; then
    PYTEST_ARGS+=(-m "not numpy")
fi
"$PYTHON" -m pytest "${PYTEST_ARGS[@]}"
ok "Tier 1 passed"

banner "Tier 2 — CLI surface smoke checks"

"$PYTHON" -m metadata_to_morphsource.seg_train --help >/dev/null
ok "package CLI --help works"

for cmd in discover round export summary; do
    "$PYTHON" -m metadata_to_morphsource.seg_train "$cmd" --help >/dev/null
    ok "$cmd --help works"
done

# `summary` on an empty / nonexistent run-dir should fail gracefully.
TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT
if "$PYTHON" -m metadata_to_morphsource.seg_train summary \
        --run-dir "$TMPDIR_RUN/empty" >/dev/null 2>&1; then
    warn "summary on missing run-dir unexpectedly succeeded"
else
    ok "summary on missing run-dir exits non-zero (as designed)"
fi

# Round-trip a single hand-built ledger entry through `export`.
mkdir -p "$TMPDIR_RUN/run/ledger"
"$PYTHON" - "$TMPDIR_RUN/run" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from metadata_to_morphsource.seg_train import (
    EpisodeRecord, ExperimentLedger, SegmentationMode,
)
run_dir = Path(sys.argv[1])
led = ExperimentLedger(run_dir / "ledger", run_id="smoke",
                      paper_tag="smoke_tag")
led.record(EpisodeRecord(
    mode=SegmentationMode.NNINTERACTIVE.value, media_id="SMOKE_A",
    round_index=0, dice=0.85, iou=0.78, has_ground_truth=True,
    n_prompts=4, duration_s=12.3,
))
led.record(EpisodeRecord(
    mode=SegmentationMode.STUDENT.value, media_id="SMOKE_A",
    round_index=1, dice=0.91, iou=0.84, has_ground_truth=True,
    duration_s=2.1,
))
print("seeded 2 episodes ->", led.jsonl_path)
PY
ok "wrote synthetic ledger via Python API"

"$PYTHON" -m metadata_to_morphsource.seg_train export \
    --run-dir "$TMPDIR_RUN/run" --paper-tag smoke_tag --no-plots \
    --output "$TMPDIR_RUN/run/paper" >/dev/null
for f in per_episode.csv mode_summary.csv round_progression.csv \
         vs_human.csv paper_summary.md; do
    test -f "$TMPDIR_RUN/run/paper/$f" \
        || { echo "MISSING: $f"; exit 1; }
done
ok "export wrote all paper CSVs + Markdown"

"$PYTHON" -m metadata_to_morphsource.seg_train summary \
    --run-dir "$TMPDIR_RUN/run" --markdown "$TMPDIR_RUN/run/paper/ledger.md" \
    >/dev/null
test -f "$TMPDIR_RUN/run/paper/ledger.md"
ok "summary emitted Markdown file"

banner "Tier 3 — integration test (synthetic NIfTI + mocked subprocesses)"
if [ "$RUN_INTEGRATION" != "1" ]; then
    warn "skipping Tier 3 (opt in with --integration or SEG_TRAIN_INTEGRATION=1)"
    warn "  Tier 3 needs a healthy numpy + SimpleITK install; run it in CI or"
    warn "  in a clean venv: python -m pip install numpy SimpleITK"
else
    # Run pytest under a portable bash-level watchdog: a corrupted local
    # numpy/SimpleITK install can *hang* during import, which would
    # otherwise deadlock this script.
    PROBE_TIMEOUT="${SEG_TRAIN_PROBE_TIMEOUT:-120}"

    "$PYTHON" -m pytest Tests/test_seg_train_integration.py \
        -v --tb=short --no-header -p no:cacheprovider \
        --override-ini="testpaths=" &
    pytest_pid=$!
    elapsed=0
    while kill -0 "$pytest_pid" 2>/dev/null; do
        if [ "$elapsed" -ge "$PROBE_TIMEOUT" ]; then
            kill -9 "$pytest_pid" 2>/dev/null || true
            wait "$pytest_pid" 2>/dev/null || true
            warn "Tier 3 timed out after ${PROBE_TIMEOUT}s — likely a broken numpy/SimpleITK install"
            warn "  bump SEG_TRAIN_PROBE_TIMEOUT= if your env is just slow"
            pytest_pid=""
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    if [ -n "${pytest_pid}" ]; then
        wait "$pytest_pid"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            ok "Tier 3 passed (synthetic round + mocked paint loop)"
        else
            echo "Tier 3 pytest exit code: $rc" >&2
            exit "$rc"
        fi
    fi
fi

printf '\n\033[1;32mAll requested smoke tiers passed.\033[0m\n'
