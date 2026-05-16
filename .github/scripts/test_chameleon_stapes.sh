#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_chameleon_stapes.sh
#
# End-to-end smoke test of the AutoResearchClaw nnInteractive comparison
# harness on a known-good open-download MorphoSource specimen:
#
#   Specimen: uf:herp:191369  (Chamaeleo calyptratus — veiled chameleon)
#   CT:       000408242       Head [CTImageSeries] (1.35 GB)
#   GT mesh:  000790324       Right Stapes [Mesh] (587 KB)
#
# The right stapes is one of the smallest bones in the body, so the test
# crops the CT to just the bbox of the GT mesh + a few mm of margin. This
# keeps the paint loop fast (a few minutes) instead of an hour on the
# whole-head volume.
#
# Outputs land in $OUTPUT_DIR (default: /tmp/nni_compare_chameleon_stapes/),
# including:
#   000408242__vs__000790324/
#     download/                 # raw MorphoSource downloads
#     ct_000408242.nii.gz       # converted from DICOM (if applicable)
#     ct_000408242_cropped.nii.gz
#     gt_voxelized.nii.gz       # GT mesh rasterized onto cropped CT grid
#     nninteractive/            # paint loop labelmap + screenshots + report
#     overlay.png               # 3x3 panel comparison
#     metrics.json              # Dice / IoU / Hausdorff / volume agreement
#     report.md                 # human-readable summary
#
# Usage:
#   .github/scripts/test_chameleon_stapes.sh             # default output dir
#   OUTPUT_DIR=/path/to/out  .github/scripts/test_chameleon_stapes.sh
#   MAX_STEPS=10              .github/scripts/test_chameleon_stapes.sh
#
# Env vars honored:
#   OPENAI_API_KEY          required for the LLM "paint" loop
#   MORPHOSOURCE_API_KEY    required for downloads
#   OUTPUT_DIR              comparison output root
#   MAX_STEPS               LLM iterations (default from preset = 6)
#   VOXELIZE_BACKEND        slicer | vtk | auto  (default: vtk)
#   CROP_MARGIN_MM          mm of context around the stapes (default: 4)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/nni_compare_chameleon_stapes}"
MAX_STEPS="${MAX_STEPS:-6}"
VOXELIZE_BACKEND="${VOXELIZE_BACKEND:-vtk}"
CROP_MARGIN_MM="${CROP_MARGIN_MM:-4}"

log() { printf '[chameleon-stapes-test] %s\n' "$*"; }
die() { printf '[chameleon-stapes-test] ERROR: %s\n' "$*" >&2; exit 1; }

# -- Sanity checks --
if [ -z "${MORPHOSOURCE_API_KEY:-}" ]; then
    die "MORPHOSOURCE_API_KEY is not set. The CT and GT mesh require an API key for download."
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    log "WARNING: OPENAI_API_KEY is not set — the paint loop will fail without it."
fi

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

NNI_HOME="${NNINTERACTIVE_HOME:-$HOME/.autoresearchclaw/nninteractive}"
if [ ! -x "$NNI_HOME/bin/python" ]; then
    die "nnInteractive venv missing at $NNI_HOME. Run: .github/scripts/install_nninteractive.sh"
fi

log "Repo:           $REPO_DIR"
log "Output dir:     $OUTPUT_DIR"
log "Max steps:      $MAX_STEPS"
log "Voxelize:       $VOXELIZE_BACKEND"
log "Crop margin:    ${CROP_MARGIN_MM} mm"
log "nnInteractive:  $NNI_HOME"

# Apple Silicon (MPS) needs CPU fallback for ops nnU-Net uses that aren't
# yet implemented on MPS (e.g. avg_pool3d.out). Harmless on Linux/CUDA.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

log "Launching nninteractive_compare.py with preset=chameleon_stapes…"

python3 .github/scripts/nninteractive_compare.py \
    --preset chameleon_stapes \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --voxelize-backend "$VOXELIZE_BACKEND" \
    --crop-around-mesh-mm "$CROP_MARGIN_MM" \
    | tee "$OUTPUT_DIR/run.log"

PAIR_DIR="$OUTPUT_DIR/000408242__vs__000790324"
if [ -d "$PAIR_DIR" ]; then
    log "Done. Key artifacts:"
    [ -f "$PAIR_DIR/report.md"   ] && log "  $PAIR_DIR/report.md"
    [ -f "$PAIR_DIR/metrics.json" ] && log "  $PAIR_DIR/metrics.json"
    [ -f "$PAIR_DIR/overlay.png" ] && log "  $PAIR_DIR/overlay.png"
fi
