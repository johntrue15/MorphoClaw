#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install_nninteractive.sh
#
# One-time bootstrap of nnInteractive on the AutoResearchClaw runner.
#
# Creates an isolated Python venv at $NNINTERACTIVE_HOME, installs a
# PyTorch build appropriate for the host (Apple-Silicon MPS / CUDA / CPU),
# installs the official `nnInteractive` Python backend (used directly,
# without going through the Slicer plugin) and pre-downloads the
# ~400MB model weights from HuggingFace into $NNINTERACTIVE_MODEL_DIR.
#
# This script is intentionally idempotent: re-running it after a successful
# install only re-checks state and exits ~instantly.
#
# Environment variables (override defaults):
#   NNINTERACTIVE_HOME       venv root (default: ~/.autoresearchclaw/nninteractive)
#   NNINTERACTIVE_MODEL_DIR  model weights cache (default: $NNINTERACTIVE_HOME/models)
#   NNINTERACTIVE_MODEL      HuggingFace model name (default: nnInteractive_v1.0)
#   NNINTERACTIVE_TORCH_INDEX  override pip index URL for torch (e.g. cu126 wheels)
#   NNINTERACTIVE_PY         python interpreter to use (default: python3)
#   NNINTERACTIVE_FORCE      "1" to recreate the venv from scratch
#
# Exits 0 on success, non-zero on hard failure.
# ---------------------------------------------------------------------------

set -euo pipefail

NNI_HOME="${NNINTERACTIVE_HOME:-$HOME/.autoresearchclaw/nninteractive}"
NNI_MODEL_DIR="${NNINTERACTIVE_MODEL_DIR:-$NNI_HOME/models}"
NNI_MODEL_NAME="${NNINTERACTIVE_MODEL:-nnInteractive_v1.0}"
NNI_PY="${NNINTERACTIVE_PY:-python3}"
NNI_FORCE="${NNINTERACTIVE_FORCE:-0}"

# nnInteractive dependency pin: their docs warn PyTorch 2.9.0 has an OOM bug,
# so we cap below it. nnInteractive itself requires torch>=2.6.
TORCH_PIN="torch>=2.6,<2.9"
TORCHVISION_PIN="torchvision"

log() { printf '[install_nninteractive] %s\n' "$*"; }
die() { printf '[install_nninteractive] ERROR: %s\n' "$*" >&2; exit 1; }

log "Home:       $NNI_HOME"
log "Models:     $NNI_MODEL_DIR"
log "Model:      $NNI_MODEL_NAME"
log "Python:     $NNI_PY"

command -v "$NNI_PY" >/dev/null 2>&1 || die "Python not found: $NNI_PY"

if [ "$NNI_FORCE" = "1" ] && [ -d "$NNI_HOME" ]; then
    log "NNINTERACTIVE_FORCE=1 — removing existing venv at $NNI_HOME"
    rm -rf "$NNI_HOME"
fi

# ---------------------------------------------------------------------------
# 1. Create venv
# ---------------------------------------------------------------------------

if [ ! -x "$NNI_HOME/bin/python" ]; then
    log "Creating venv at $NNI_HOME"
    mkdir -p "$(dirname "$NNI_HOME")"
    "$NNI_PY" -m venv "$NNI_HOME"
else
    log "Reusing existing venv"
fi

# shellcheck disable=SC1091
source "$NNI_HOME/bin/activate"

python -m pip install --quiet --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 2. Detect platform and install matching PyTorch
# ---------------------------------------------------------------------------

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
log "Platform: $OS_NAME / $ARCH_NAME"

# Choose a torch wheel index. Caller can force one via NNINTERACTIVE_TORCH_INDEX.
TORCH_INDEX_URL=""
if [ -n "${NNINTERACTIVE_TORCH_INDEX:-}" ]; then
    TORCH_INDEX_URL="$NNINTERACTIVE_TORCH_INDEX"
    log "Using caller-provided torch index: $TORCH_INDEX_URL"
elif [ "$OS_NAME" = "Darwin" ]; then
    # Apple Silicon: MPS support is in default PyPI wheels. CPU fallback
    # works automatically when MPS isn't available.
    TORCH_INDEX_URL=""
    log "macOS — using default PyPI wheels (MPS / CPU fallback)"
elif [ "$OS_NAME" = "Linux" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
        log "Linux + NVIDIA GPU detected — using CUDA 12.6 wheels"
    else
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
        log "Linux without NVIDIA GPU — using CPU-only wheels"
    fi
fi

if python -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    log "PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    if [ -n "$TORCH_INDEX_URL" ]; then
        log "Installing PyTorch from $TORCH_INDEX_URL"
        pip install --index-url "$TORCH_INDEX_URL" "$TORCH_PIN" "$TORCHVISION_PIN"
    else
        log "Installing PyTorch from PyPI"
        pip install "$TORCH_PIN" "$TORCHVISION_PIN"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Install nnInteractive + scientific deps
# ---------------------------------------------------------------------------

if python -c "import nnInteractive" >/dev/null 2>&1; then
    log "nnInteractive already installed"
else
    log "Installing nnInteractive (+ SimpleITK, huggingface_hub, numpy)"
    pip install \
        "nnInteractive>=1.1" \
        "huggingface_hub>=0.24" \
        "SimpleITK>=2.3" \
        "numpy<2.3"
fi

# Optional: install Pillow for screenshot rendering used by nninteractive_segment.py
pip install --quiet "Pillow>=10.0" "matplotlib>=3.7" || true

# ---------------------------------------------------------------------------
# 4. Pre-download model weights
# ---------------------------------------------------------------------------

mkdir -p "$NNI_MODEL_DIR"

if [ -d "$NNI_MODEL_DIR/$NNI_MODEL_NAME" ] && \
   [ "$(ls -A "$NNI_MODEL_DIR/$NNI_MODEL_NAME" 2>/dev/null)" ]; then
    log "Model weights already present at $NNI_MODEL_DIR/$NNI_MODEL_NAME"
else
    log "Downloading $NNI_MODEL_NAME from HuggingFace (~400MB)..."
    python - <<PY
from huggingface_hub import snapshot_download
import os, sys
target = os.environ["NNI_MODEL_DIR"]
name = os.environ["NNI_MODEL_NAME"]
path = snapshot_download(
    repo_id="nnInteractive/nnInteractive",
    allow_patterns=[f"{name}/*"],
    local_dir=target,
)
print(f"Downloaded weights to: {os.path.join(path, name)}")
PY
fi

# ---------------------------------------------------------------------------
# 5. Quick sanity check
# ---------------------------------------------------------------------------

log "Running quick sanity check..."
python - <<'PY'
import importlib, json, os, sys
import torch

info = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "mps_available": getattr(torch.backends, "mps",
                             type("x",(),{"is_available":lambda:False})).is_available(),
}
try:
    import nnInteractive
    info["nnInteractive"] = getattr(nnInteractive, "__version__", "installed")
except Exception as exc:
    info["nnInteractive_error"] = str(exc)
try:
    import SimpleITK as sitk
    info["SimpleITK"] = sitk.Version_VersionString()
except Exception:
    pass
print(json.dumps(info, indent=2))
PY

log "Done. Activate with: source $NNI_HOME/bin/activate"
