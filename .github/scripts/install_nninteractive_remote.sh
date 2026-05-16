#!/usr/bin/env bash
# Bootstrap nnInteractive on a remote Linux host (Jetstream2 / MorphoCloud).
#
# Run this on the REMOTE box (e.g. exouser@149.165.172.55) once. It
# creates a Python 3.10+ venv, installs torch + nnInteractive + the
# WebSocket server's deps (websockets), prefetches the
# nnInteractive_v1.0 weights from HuggingFace, and prints a one-line
# command to launch the server.
#
# Idempotent: re-running is safe and only does work if something is
# missing.
#
# Usage (interactive):
#     bash install_nninteractive_remote.sh
#
# Usage (one-shot from local Mac):
#     scp install_nninteractive_remote.sh exouser@HOST:/tmp/
#     ssh exouser@HOST 'bash /tmp/install_nninteractive_remote.sh'

set -euo pipefail

NNI_HOME="${NNINTERACTIVE_HOME:-$HOME/.autoresearchclaw/nninteractive}"
MODEL_DIR="${NNINTERACTIVE_MODEL_DIR:-$NNI_HOME/models}"
MODEL_NAME="${NNINTERACTIVE_MODEL:-nnInteractive_v1.0}"
HF_REPO="${NNINTERACTIVE_HF_REPO:-nnInteractive/nnInteractive}"
WS_HOST="${NNI_REMOTE_HOST:-127.0.0.1}"
WS_PORT="${NNI_REMOTE_PORT:-8765}"

# Pick the newest python3.10+ on PATH; fail loudly if none found.
pick_python() {
    local best=""
    for cand in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            local ver
            ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")
            local major minor
            major="${ver%.*}"; minor="${ver#*.}"
            if [ "$major" -ge 4 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; }; then
                best="$cand"
                break
            fi
        fi
    done
    [ -n "$best" ] || {
        echo "ERROR: need Python 3.10+ on this host. Install one and re-run." >&2
        exit 1
    }
    echo "$best"
}

log() { printf '[install-remote] %s\n' "$*"; }

PYTHON_BIN="$(pick_python)"
log "Using Python: $($PYTHON_BIN --version 2>&1)"

if [ ! -d "$NNI_HOME" ]; then
    log "Creating venv at $NNI_HOME"
    "$PYTHON_BIN" -m venv "$NNI_HOME"
fi

# shellcheck disable=SC1091
source "$NNI_HOME/bin/activate"

log "Upgrading pip / setuptools / wheel"
pip install --quiet --upgrade pip setuptools wheel

# torch: prefer GPU build if CUDA is present; otherwise CPU wheels.
if command -v nvidia-smi >/dev/null 2>&1; then
    log "Detected NVIDIA driver; installing CUDA-enabled torch"
    pip install --quiet --extra-index-url https://download.pytorch.org/whl/cu124 \
        "torch>=2.1" "torchvision>=0.16" || \
        pip install --quiet "torch>=2.1" "torchvision>=0.16"
else
    log "No NVIDIA driver; installing CPU torch"
    pip install --quiet "torch>=2.1" "torchvision>=0.16"
fi

log "Installing nnInteractive + SimpleITK + WebSocket deps"
pip install --quiet "nnInteractive>=1.1" "SimpleITK>=2.3" "vtk>=9.3" "trimesh>=4.0" "Pillow>=10.0" "matplotlib>=3.7" "websockets>=12" "huggingface_hub>=0.20"

if python -c "import openai" >/dev/null 2>&1; then
    log "openai already installed: $(python -c 'import openai; print(openai.__version__)')"
else
    log "Installing openai (only used if you choose to run the LLM remotely too)"
    pip install --quiet "openai>=1.40"
fi

mkdir -p "$MODEL_DIR"
if [ ! -e "$MODEL_DIR/$MODEL_NAME/plans.json" ] && \
   [ -z "$(ls -A "$MODEL_DIR/$MODEL_NAME"/fold_* 2>/dev/null)" ]; then
    log "Fetching nnInteractive weights ($MODEL_NAME) from $HF_REPO"
    NNI_MODEL_DIR="$MODEL_DIR" NNI_MODEL_NAME="$MODEL_NAME" python - <<'PY'
import os
from huggingface_hub import snapshot_download
target = os.path.join(os.environ["NNI_MODEL_DIR"], os.environ["NNI_MODEL_NAME"])
os.makedirs(target, exist_ok=True)
print(f"Downloading nnInteractive weights to {target}")
snapshot_download(
    repo_id="nnInteractive/nnInteractive",
    local_dir=target,
    local_dir_use_symlinks=False,
    allow_patterns=[f"{os.environ['NNI_MODEL_NAME']}/**"],
)
# Some versions place files inside a model-named subdir; flatten if needed.
sub = os.path.join(target, os.environ["NNI_MODEL_NAME"])
if os.path.isdir(sub):
    for name in os.listdir(sub):
        os.replace(os.path.join(sub, name), os.path.join(target, name))
    os.rmdir(sub)
PY
else
    log "nnInteractive weights already present at $MODEL_DIR/$MODEL_NAME"
fi

log "Sanity check"
python - <<'PY'
import importlib, sys
for mod in ("torch", "SimpleITK", "vtk", "trimesh", "websockets",
            "nnInteractive", "openai"):
    try:
        m = importlib.import_module(mod)
        print(f"  OK  {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  FAIL {mod}: {exc}")
        sys.exit(1)

import torch
print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
PY

log
log "Done. To launch the server (binds to $WS_HOST:$WS_PORT by default):"
log
log "    source $NNI_HOME/bin/activate"
log "    python $(dirname "$(readlink -f "$0")")/nni_ws_server.py --host $WS_HOST --port $WS_PORT"
log
log "From the local box, start an SSH tunnel:"
log
log "    ssh -L $WS_PORT:127.0.0.1:$WS_PORT exouser@<this-host>"
log
log "And export the URL before invoking the comparison:"
log
log "    export NNI_REMOTE_WS=ws://localhost:$WS_PORT"
