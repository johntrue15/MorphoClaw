#!/usr/bin/env bash
# One-shot deployer for the nnInteractive WebSocket server inside a
# MorphoCloud / Jetstream2 instance's Guacamole desktop. Run this from a
# terminal **on the remote box**; it does NOT use SSH.
#
# What it does (idempotent):
#   1. git-clones (or fast-forwards) MorphoClaw into ~/MorphoClaw
#   2. Bootstraps the nnInteractive venv + weights via
#      install_nninteractive_remote.sh
#   3. Stops any previous nni_ws_server.py
#   4. Starts a fresh nni_ws_server.py under nohup, listening on
#      0.0.0.0:$NNI_REMOTE_PORT, with NNI_WS_TOKEN required.
#   5. Prints the wss:// URL the local Mac should connect to and the
#      ports/PIDs/log paths to debug if anything goes wrong.
#
# Required env (set on the calling shell or in the curl one-liner):
#   NNI_WS_TOKEN       Shared-secret token (32+ random chars).
#                      Generate locally with:
#                         python -c 'import secrets; print(secrets.token_urlsafe(32))'
#                      Use the same value on local box's .env.
#
# Optional env:
#   NNI_REMOTE_PORT    Server bind port (default 8765).
#   NNI_REMOTE_HOST    Server bind host (default 0.0.0.0).
#   MORPHOCLAW_REPO    Public repo URL (default johntrue15/MorphoClaw).
#   MORPHOCLAW_BRANCH  Branch (default main).
#
# Typical invocation, pasted directly into the Guacamole terminal:
#
#   curl -fsSL \
#     https://raw.githubusercontent.com/johntrue15/MorphoClaw/main/.github/scripts/start_remote_server.sh \
#     | NNI_WS_TOKEN='paste-your-token-here' bash
#
# The local Mac then sets:
#
#   NNI_REMOTE_WS=wss://http-<your-public-ip-with-dashes>-<port>.proxy-js2-iu.exosphere.app/
#   NNI_WS_TOKEN=<same value as above>
#
# in its .env, and runs any compare normally.

set -euo pipefail

NNI_REMOTE_PORT="${NNI_REMOTE_PORT:-8765}"
NNI_REMOTE_HOST="${NNI_REMOTE_HOST:-0.0.0.0}"
MORPHOCLAW_REPO="${MORPHOCLAW_REPO:-https://github.com/johntrue15/MorphoClaw.git}"
MORPHOCLAW_BRANCH="${MORPHOCLAW_BRANCH:-main}"
REPO_DIR="${MORPHOCLAW_DIR:-$HOME/MorphoClaw}"
NNI_HOME="${NNINTERACTIVE_HOME:-$HOME/.autoresearchclaw/nninteractive}"
LOGFILE="$HOME/nni_ws_server.log"
PIDFILE="$HOME/nni_ws_server.pid"

log() { printf '[remote] %s\n' "$*"; }
die() { printf '[remote] ERROR: %s\n' "$*" >&2; exit 1; }

# ── 0. sanity ────────────────────────────────────────────────────────────
if [ -z "${NNI_WS_TOKEN:-}" ]; then
    die "NNI_WS_TOKEN is required. Generate with:
         python -c 'import secrets; print(secrets.token_urlsafe(32))'
       and re-run as:
         NNI_WS_TOKEN='<token>' bash $(basename "$0")"
fi

for tool in git curl bash; do
    command -v "$tool" >/dev/null 2>&1 \
        || die "missing prerequisite: $tool (install it first)"
done

# ── 1. clone / fast-forward repo ─────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    log "Repo already cloned at $REPO_DIR — fast-forwarding $MORPHOCLAW_BRANCH"
    git -C "$REPO_DIR" fetch --quiet origin "$MORPHOCLAW_BRANCH"
    git -C "$REPO_DIR" checkout --quiet "$MORPHOCLAW_BRANCH"
    git -C "$REPO_DIR" reset --hard --quiet "origin/$MORPHOCLAW_BRANCH"
else
    log "Cloning $MORPHOCLAW_REPO ($MORPHOCLAW_BRANCH) into $REPO_DIR"
    git clone --quiet --branch "$MORPHOCLAW_BRANCH" \
        "$MORPHOCLAW_REPO" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── 2. bootstrap nnInteractive (no-op if already done) ───────────────────
if [ ! -x "$NNI_HOME/bin/python" ]; then
    log "Bootstrapping nnInteractive (one-time, ~5–15 min on a cold box)"
    bash .github/scripts/install_nninteractive_remote.sh
else
    log "nnInteractive venv already present at $NNI_HOME"
fi

# ── 3. stop any previous server ──────────────────────────────────────────
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    log "Stopping previous server (pid=$(cat "$PIDFILE"))"
    kill "$(cat "$PIDFILE")" || true
    sleep 1
fi
pkill -f nni_ws_server.py 2>/dev/null || true
sleep 0.5

# ── 4. launch fresh server ───────────────────────────────────────────────
SERVER="$REPO_DIR/.github/scripts/nni_ws_server.py"
[ -f "$SERVER" ] || die "Server script missing: $SERVER"

log "Starting nni_ws_server.py on ${NNI_REMOTE_HOST}:${NNI_REMOTE_PORT}"
# We export the token via the inherited env so it never appears on the
# command line (and so it isn't visible in `ps`).
export NNI_WS_TOKEN
nohup env \
    NNI_WS_TOKEN="$NNI_WS_TOKEN" \
    NNINTERACTIVE_HOME="$NNI_HOME" \
    "$NNI_HOME/bin/python" -u "$SERVER" \
        --host "$NNI_REMOTE_HOST" \
        --port "$NNI_REMOTE_PORT" \
    > "$LOGFILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"

# Give the server a moment to import everything and start listening.
log "Waiting 8s for the server to come up…"
sleep 8

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Server died early. Last 60 lines of log:"
    tail -60 "$LOGFILE" || true
    die "nni_ws_server.py exited unexpectedly"
fi

if command -v ss >/dev/null 2>&1; then
    LISTEN_OK=$(ss -ltn "( sport = :$NNI_REMOTE_PORT )" 2>/dev/null | tail -n +2)
elif command -v netstat >/dev/null 2>&1; then
    LISTEN_OK=$(netstat -ltn 2>/dev/null | awk -v p=":$NNI_REMOTE_PORT" '$0 ~ p')
else
    LISTEN_OK="(could not verify; ss/netstat not available)"
fi

# Public IP via metadata service, or fall back to whatever we can scrape.
PUBLIC_IP=""
if command -v curl >/dev/null; then
    PUBLIC_IP=$(curl -fsS --max-time 4 https://api.ipify.org 2>/dev/null || true)
fi

cat <<EOF

==========================================================================
nnInteractive WebSocket server is running.

  pid          : $SERVER_PID
  pidfile      : $PIDFILE
  log          : $LOGFILE
  bind         : $NNI_REMOTE_HOST:$NNI_REMOTE_PORT
  listening    : ${LISTEN_OK:-NO LISTENERS — investigate $LOGFILE}
  public IP    : ${PUBLIC_IP:-<unknown — read from JS2 dashboard>}

LOCAL Mac configuration (.env):
  NNI_WS_TOKEN=<the token you used here>
  NNI_REMOTE_WS=wss://http-${PUBLIC_IP//./-}-${NNI_REMOTE_PORT}.proxy-js2-iu.exosphere.app/

Tail the server log:
  tail -f $LOGFILE

Stop the server:
  kill \$(cat $PIDFILE)
==========================================================================
EOF
