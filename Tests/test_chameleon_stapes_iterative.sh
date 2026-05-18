#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_chameleon_stapes_iterative.sh
#
# REAL end-to-end test of `metadata_to_morphsource.seg_train` on a known-
# good MorphoSource open-download pair (Chamaeleo calyptratus, right
# stapes — uf:herp:191369). Unlike the synthetic Tier-3 integration
# test, this test:
#
#   * downloads the real CT (DICOM) and the real GT mesh (PLY),
#   * converts the DICOM series to NIfTI with SimpleITK,
#   * crops the CT around the mesh bbox + a few mm of margin,
#   * voxelises the GT mesh onto the CT grid (Slicer or VTK),
#   * runs the actual nnInteractive paint loop (a real LLM call!),
#   * routes through the trainer's ledger / manifest / metrics / router,
#   * (optionally) trains a 1-epoch student on the resulting pair,
#   * exports the publication-ready CSVs + Markdown summary.
#
# The right stapes is one of the smallest bones in the body, so the
# whole run takes ~5–15 min on the runner depending on Slicer vs VTK
# and OpenAI latency. Costs a small amount of OpenAI quota (typically
# < $0.10 with --max-paint-steps 4–6).
#
# Usage:
#   .github/scripts/install_nninteractive.sh         # one-time
#   .github/scripts/install_seg_train_extras.sh      # one-time
#   Tests/test_chameleon_stapes_iterative.sh         # default run
#
#   # Skip the (slow) student-training step and just exercise round 0:
#   SKIP_TRAINING=1  Tests/test_chameleon_stapes_iterative.sh
#
#   # Bump paint-loop iterations or pick a different output dir:
#   MAX_STEPS=8      Tests/test_chameleon_stapes_iterative.sh
#   RUN_DIR=/tmp/seg_train_chameleon  Tests/test_chameleon_stapes_iterative.sh
#
# Required env vars:
#   MORPHOSOURCE_API_KEY  for the downloads
#   OPENAI_API_KEY        for the paint loop
#
# Optional env vars:
#   NNINTERACTIVE_HOME    venv root (default: ~/.autoresearchclaw/nninteractive)
#   SLICER_BIN            path to Slicer.app/Contents/MacOS/Slicer (macOS)
#   VOXELIZE_BACKEND      slicer | vtk | auto (default: vtk — pure Python)
#   MAX_STEPS             paint-loop iterations (default: 6)
#   CROP_MM               crop margin around mesh in mm (default: 4)
#   STUDENT_EPOCHS        student training epochs (default: 1 for smoke)
#   SKIP_TRAINING         set to 1 to skip the student training step
#   PREP_DIR              compare-script output dir (default: $RUN_DIR/prep)
#   RUN_DIR               trainer run-dir   (default: /tmp/seg_train_stapes_live)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ----- Defaults ------------------------------------------------------
MAX_STEPS="${MAX_STEPS:-6}"
CROP_MM="${CROP_MM:-4}"
VOXELIZE_BACKEND="${VOXELIZE_BACKEND:-vtk}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-1}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
RUN_DIR="${RUN_DIR:-/tmp/seg_train_stapes_live}"
PREP_DIR="${PREP_DIR:-$RUN_DIR/prep}"
SPECIMENS_JSON="$RUN_DIR/specimens.json"
PAPER_TAG="${PAPER_TAG:-chameleon_stapes_live}"
PRESET="${PRESET:-chameleon_stapes}"

NNI_HOME="${NNINTERACTIVE_HOME:-$HOME/.autoresearchclaw/nninteractive}"
NNI_PY="$NNI_HOME/bin/python"

banner() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()     { printf '   \033[1;32m✓\033[0m %s\n' "$*"; }
warn()   { printf '   \033[1;33m!\033[0m %s\n' "$*"; }
die()    { printf '   \033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ----- Sanity checks -------------------------------------------------
banner "Pre-flight checks"

if [ -z "${MORPHOSOURCE_API_KEY:-}" ]; then
    die "MORPHOSOURCE_API_KEY is not set — needed for the CT + GT mesh download."
fi
ok "MORPHOSOURCE_API_KEY present"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    die "OPENAI_API_KEY is not set — needed for the nnInteractive paint loop."
fi
ok "OPENAI_API_KEY present"

if [ ! -x "$NNI_PY" ]; then
    die "nnInteractive venv not found at $NNI_PY. Run: .github/scripts/install_nninteractive.sh"
fi
ok "nnInteractive venv at $NNI_HOME"

if ! "$NNI_PY" -c "import monai" >/dev/null 2>&1; then
    warn "MONAI not installed in venv — student training will be skipped."
    warn "  Run: .github/scripts/install_seg_train_extras.sh"
    SKIP_TRAINING=1
else
    ok "MONAI available in nnInteractive venv"
fi

if [ "$VOXELIZE_BACKEND" = "slicer" ]; then
    SLICER="${SLICER_BIN:-/Applications/Slicer.app/Contents/MacOS/Slicer}"
    if [ ! -x "$SLICER" ]; then
        die "VOXELIZE_BACKEND=slicer but Slicer not found at $SLICER. Install Slicer or set VOXELIZE_BACKEND=vtk."
    fi
    ok "Slicer at $SLICER"
fi

mkdir -p "$RUN_DIR" "$PREP_DIR"
ok "Run dir: $RUN_DIR"

# Apple Silicon (MPS) needs CPU fallback for ops nnU-Net uses that
# aren't yet implemented on MPS (e.g. avg_pool3d.out). Harmless on Linux/CUDA.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

# ----- 1. Prepare the specimen --------------------------------------
banner "Step 1 — Prepare specimen via seg_train prepare ($PRESET)"

# We drive the prepare pipeline through the parent Python (uses the
# nnInteractive venv internally for SimpleITK + VTK) so the resulting
# specimens.json is written by the seg_train package itself.
"$NNI_PY" -m metadata_to_morphsource.seg_train prepare \
    --preset "$PRESET" \
    --prep-dir "$PREP_DIR" \
    --voxelize-backend "$VOXELIZE_BACKEND" \
    --output "$SPECIMENS_JSON"

[ -s "$SPECIMENS_JSON" ] || die "specimens.json was not produced at $SPECIMENS_JSON"
ok "Wrote $SPECIMENS_JSON"

PYBIN="$NNI_PY"
"$PYBIN" - <<PY
import json, os, sys
data = json.loads(open(r"$SPECIMENS_JSON").read())
assert isinstance(data, list) and data, "specimens.json is empty"
sp = data[0]
for k in ("media_id", "volume_path", "gt_label_path", "gt_mesh_path"):
    v = sp.get(k, "")
    if not v or (k.endswith("_path") and not os.path.exists(v)):
        sys.exit(f"specimen field missing or path absent: {k} -> {v!r}")
print(f"  prepared CT      : {sp['volume_path']}")
print(f"  prepared GT label: {sp['gt_label_path']}")
print(f"  raw GT mesh      : {sp['gt_mesh_path']}")
PY
ok "Specimen artefacts exist on disk"

# ----- 2. Round 0: paint-loop + ledger + manifest --------------------
banner "Step 2 — seg_train round 0 (real nnInteractive paint loop)"

ROUND_ARGS=(
    --specimens "$SPECIMENS_JSON"
    --run-dir "$RUN_DIR"
    --paper-tag "$PAPER_TAG"
    --goal "Segment the right stapes"
    --max-paint-steps "$MAX_STEPS"
    --student-epochs "$STUDENT_EPOCHS"
)
if [ "$SKIP_TRAINING" = "1" ]; then
    ROUND_ARGS+=(--skip-training)
    ok "Student training disabled (SKIP_TRAINING=1)"
fi

"$NNI_PY" -m metadata_to_morphsource.seg_train round "${ROUND_ARGS[@]}"

ROUND_REPORT="$RUN_DIR/round_000/round_report.json"
[ -f "$ROUND_REPORT" ] || die "round_000/round_report.json missing"
ok "round_000/round_report.json written"

# ----- 3. Verify ledger ---------------------------------------------
banner "Step 3 — Verify ledger contents (real metrics, real Dice)"

LEDGER_JSONL="$RUN_DIR/ledger/ledger.jsonl"
LEDGER_CSV="$RUN_DIR/ledger/ledger.csv"
[ -f "$LEDGER_JSONL" ] || die "ledger.jsonl missing"
[ -f "$LEDGER_CSV"   ] || die "ledger.csv missing"
ok "ledger.{jsonl,csv} written"

"$PYBIN" - <<PY
import json, sys
rows = [json.loads(line) for line in open(r"$LEDGER_JSONL")]
assert rows, "ledger is empty"
modes = sorted({r["mode"] for r in rows})
print(f"  modes recorded: {modes}")
assert "human_gt" in modes, f"missing human_gt episode (got {modes})"
assert "nninteractive" in modes, f"missing nninteractive episode (got {modes})"

nni_rows = [r for r in rows if r["mode"] == "nninteractive"]
assert nni_rows, "no nnInteractive episode recorded"
top = nni_rows[0]
assert top.get("has_ground_truth"), "nnInteractive episode lacks ground_truth flag"
dice = top.get("dice")
print(f"  nnInteractive Dice vs GT: {dice}")
if dice is None:
    sys.exit("nnInteractive episode has no Dice — metric computation failed")
# stapes is hard; we just want a real positive Dice (sanity, not quality)
assert 0.0 <= dice <= 1.0, f"Dice out of range: {dice}"

duration = top.get("duration_s") or 0
assert duration > 0, "paint-loop duration not recorded"
print(f"  paint-loop duration: {duration:.1f}s")

n_prompts = top.get("n_prompts")
print(f"  n_prompts emitted: {n_prompts}")
assert (n_prompts or 0) > 0, "no prompts recorded — paint loop didn't actually run"
print("  ledger looks healthy ✓")
PY
ok "Ledger sanity checks passed"

# ----- 4. Export paper artefacts ------------------------------------
banner "Step 4 — seg_train export (paper CSVs + summary)"

PAPER_DIR="$RUN_DIR/paper"
"$NNI_PY" -m metadata_to_morphsource.seg_train export \
    --run-dir "$RUN_DIR" --paper-tag "$PAPER_TAG" \
    --output "$PAPER_DIR" --no-plots >/dev/null

for f in per_episode.csv mode_summary.csv round_progression.csv \
         vs_human.csv paper_summary.md; do
    [ -f "$PAPER_DIR/$f" ] || die "missing paper artefact: $f"
done
ok "All paper artefacts written to $PAPER_DIR"

# ----- 5. Optional Round 1 with student inference --------------------
if [ "$SKIP_TRAINING" != "1" ]; then
    STUDENT="$RUN_DIR/round_000/student_weights/student_r000.artifact.json"
    if [ -f "$STUDENT" ]; then
        banner "Step 5 — Round 1 (student inference + router decision)"
        "$NNI_PY" -m metadata_to_morphsource.seg_train round \
            --specimens "$SPECIMENS_JSON" \
            --run-dir   "$RUN_DIR" \
            --paper-tag "$PAPER_TAG" \
            --goal "Segment the right stapes" \
            --max-paint-steps "$MAX_STEPS" \
            --student "$STUDENT" \
            --skip-training
        ROUND1_REPORT="$RUN_DIR/round_001/round_report.json"
        [ -f "$ROUND1_REPORT" ] || die "round_001/round_report.json missing"
        ok "round_001 completed (student + router exercised)"

        # Re-export with both rounds.
        "$NNI_PY" -m metadata_to_morphsource.seg_train export \
            --run-dir "$RUN_DIR" --paper-tag "$PAPER_TAG" \
            --output "$PAPER_DIR" --no-plots >/dev/null
        ok "Re-exported paper artefacts (now covers 2 rounds)"
    else
        warn "Round 0 produced no student artifact at $STUDENT"
        warn "  (training likely skipped or failed; see round_000 logs)"
    fi
fi

banner "Done"
echo "Run dir         : $RUN_DIR"
echo "Specimens JSON  : $SPECIMENS_JSON"
echo "Round 0 report  : $RUN_DIR/round_000/round_report.json"
echo "Ledger          : $LEDGER_JSONL"
echo "Paper summary   : $PAPER_DIR/paper_summary.md"
echo
printf '\033[1;32mLive seg_train smoke run completed successfully.\033[0m\n'
