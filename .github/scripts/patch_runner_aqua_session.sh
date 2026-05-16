#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# patch_runner_aqua_session.sh
#
# Why this exists
# ---------------
# 3D Slicer on macOS only ships the `cocoa` Qt platform plugin (no
# `offscreen` / `minimal`), so it MUST connect to WindowServer. When the
# GitHub Actions self-hosted runner is loaded as a launchd user agent that
# doesn't declare `LimitLoadToSessionType=Aqua`, launchd loads it in a
# generic per-user bootstrap whose Mach namespace doesn't include
# `com.apple.windowserver.session`. Slicer's startup then aborts in
# `QGuiApplicationPrivate::createPlatformIntegration()`, which is exactly
# what the crash report shows:
#
#   QtCore  QMessageLogger::fatal(...)
#   QtGui   QGuiApplicationPrivate::createPlatformIntegration() + 5707
#   ...
#   abort()
#
# This script patches the runner's LaunchAgent plist to add
# `LimitLoadToSessionType=Aqua`, then unloads/reloads it so the change
# takes effect in the live launchd domain.
#
# It also:
#   - Prints the bootstrap namespace BEFORE / AFTER so the fix is verifiable.
#   - Backs up the plist before modifying.
#   - Is idempotent: re-running after a successful patch is a no-op.
#
# Usage:
#   .github/scripts/patch_runner_aqua_session.sh                 # auto-detect
#   .github/scripts/patch_runner_aqua_session.sh /path/to/agent.plist
# ---------------------------------------------------------------------------

set -euo pipefail

log() { printf '[runner-aqua-patch] %s\n' "$*"; }
die() { printf '[runner-aqua-patch] ERROR: %s\n' "$*" >&2; exit 1; }

PLIST="${1:-}"

if [ -z "$PLIST" ]; then
    # Auto-detect: any actions.runner.* plist in the current user's LaunchAgents.
    matches=( "$HOME/Library/LaunchAgents"/actions.runner.*.plist )
    if [ ! -e "${matches[0]}" ]; then
        die "No actions.runner.*.plist found in $HOME/Library/LaunchAgents/. Pass the plist path explicitly."
    fi
    if [ "${#matches[@]}" -gt 1 ]; then
        log "Multiple runner agents found:"
        for m in "${matches[@]}"; do log "  $m"; done
        die "Pass the desired plist path as the first argument."
    fi
    PLIST="${matches[0]}"
fi

[ -f "$PLIST" ] || die "Plist not found: $PLIST"
LABEL="$(basename "$PLIST" .plist)"

log "Plist:   $PLIST"
log "Label:   $LABEL"

# -- BEFORE state --
log "Current launchctl state for $LABEL:"
launchctl list | awk -v l="$LABEL" '$3==l { print "  ", $0 }' || true

UID_NUM="$(id -u)"
log "Bootstrap snapshot for the runner BEFORE patching:"
launchctl print "gui/$UID_NUM/$LABEL" 2>&1 | sed -n '1,40p' | sed 's/^/  /' || true

# -- Already patched? --
if /usr/libexec/PlistBuddy -c "Print :LimitLoadToSessionType" "$PLIST" 2>/dev/null \
        | grep -qi 'aqua'; then
    log "Plist already declares LimitLoadToSessionType=Aqua — nothing to do."
    log "If Slicer still crashes, run this script anyway with the runner stopped, "
    log "or check System Settings > Privacy & Security > Automation for blocked apps."
    exit 0
fi

# -- Backup --
BACKUP="${PLIST}.bak.$(date +%Y%m%d-%H%M%S)"
cp -p "$PLIST" "$BACKUP"
log "Backup written: $BACKUP"

# -- Patch using PlistBuddy (preserves XML structure) --
if /usr/libexec/PlistBuddy -c "Print :LimitLoadToSessionType" "$PLIST" >/dev/null 2>&1; then
    /usr/libexec/PlistBuddy -c "Set :LimitLoadToSessionType Aqua" "$PLIST"
else
    /usr/libexec/PlistBuddy -c "Add :LimitLoadToSessionType string Aqua" "$PLIST"
fi
log "Added/updated LimitLoadToSessionType=Aqua"

# Sanity check the file is valid plist after editing.
plutil -lint "$PLIST" >/dev/null \
    || die "Plist is invalid after editing — restore from $BACKUP"

# -- Reload the agent --
log "Unloading $LABEL (if loaded)…"
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || \
    launchctl unload "$PLIST" 2>/dev/null || true

log "Loading $LABEL (Aqua session)…"
# bootstrap into the GUI domain (Aqua) explicitly
launchctl bootstrap "gui/$UID_NUM" "$PLIST" 2>/dev/null || \
    launchctl load "$PLIST"

sleep 1

log "AFTER state:"
launchctl list | awk -v l="$LABEL" '$3==l { print "  ", $0 }' || true
launchctl print "gui/$UID_NUM/$LABEL" 2>&1 | sed -n '1,40p' | sed 's/^/  /' || true

log "Done. Slicer launches from this runner should now reach the Aqua bootstrap."
log "Verify with: pgrep -lf Runner.Listener  &&  launchctl print pid/<pid> | grep Aqua"
