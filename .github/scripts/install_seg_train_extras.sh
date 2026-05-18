#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install_seg_train_extras.sh
#
# Adds the dependencies needed by metadata_to_morphsource.seg_train (the
# iterative student-training pipeline) on top of an already-bootstrapped
# nnInteractive venv:
#
#   - MONAI (for the 3D U-Net student + sliding-window inference)
#   - nibabel + einops + tqdm (MONAI extras)
#   - matplotlib (already pulled by install_nninteractive.sh, re-asserted)
#
# This script is idempotent. Run it once after install_nninteractive.sh
# (or whenever you bump MONAI):
#
#   ./.github/scripts/install_nninteractive.sh
#   ./.github/scripts/install_seg_train_extras.sh
#
# Environment variables:
#   NNINTERACTIVE_HOME  venv root (default: ~/.autoresearchclaw/nninteractive)
# ---------------------------------------------------------------------------

set -euo pipefail

NNI_HOME="${NNINTERACTIVE_HOME:-$HOME/.autoresearchclaw/nninteractive}"

log() { printf '[install_seg_train_extras] %s\n' "$*"; }
die() { printf '[install_seg_train_extras] ERROR: %s\n' "$*" >&2; exit 1; }

[ -x "$NNI_HOME/bin/python" ] || die \
    "nnInteractive venv not found at $NNI_HOME. Run install_nninteractive.sh first."

log "Using venv at $NNI_HOME"
# shellcheck disable=SC1091
source "$NNI_HOME/bin/activate"

python -m pip install --quiet --upgrade pip setuptools wheel

if python -c "import monai; print(monai.__version__)" >/dev/null 2>&1; then
    log "MONAI already installed: $(python -c 'import monai; print(monai.__version__)')"
else
    log "Installing MONAI (with nibabel + einops extras)…"
    pip install "monai[nibabel,einops]>=1.3,<1.6" "tqdm>=4.65"
fi

# matplotlib + pandas are useful for paper_export plots & diagnostics
pip install --quiet "matplotlib>=3.7" "pandas>=1.5" || true

log "Quick sanity check…"
python - <<'PY'
import importlib, json, sys

info = {"python": sys.version.split()[0]}
for mod in ("torch", "monai", "SimpleITK", "matplotlib", "pandas"):
    try:
        m = importlib.import_module(mod)
        info[mod] = getattr(m, "__version__", "installed")
    except Exception as exc:
        info[mod + "_error"] = str(exc)
print(json.dumps(info, indent=2))
PY

log "Done."
