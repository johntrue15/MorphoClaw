#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_eval_project358382.sh
#
# Smoke-tests the project-358382 pilot orchestrator without touching the
# remote Slicer / nnInteractive servers and without downloading any
# multi-GB CTs. Three escalating tiers, each opt-in:
#
#   1. import-and-help (default, always runs):
#        - parse the orchestrator file
#        - --help
#        - import the heavy modules (nninteractive_compare,
#          remote_volume_io, run_telemetry, slicer_remote_bright_seed,
#          segmentation_metrics, voxelize_mesh_vtk, crop_around_mesh)
#        - sanity-check the recipe builders compile to valid Python
#
#   2. discovery-only (set RUN_DISCOVERY=1):
#        - hits the MorphoSource API to enumerate the project's
#          (CT, mesh) pairs
#        - validates that the discovery returns at least one pair
#        - requires MORPHOSOURCE_API_KEY in the environment
#
#   3. dry-run (set RUN_DRY=1):
#        - executes the full orchestrator in --dry-run mode (selects 1
#          specimen but skips download / crop / voxelize / push / run /
#          score), proving the parent RunLogger pipeline plumbs together
#        - requires MORPHOSOURCE_API_KEY and SLICER_WEBSERVER_URL (or
#          NNI_REMOTE_URL) just to exercise the validation paths.
#          The actual remote isn't contacted in dry-run.
#
# Usage:
#   Tests/test_eval_project358382.sh
#   RUN_DISCOVERY=1 Tests/test_eval_project358382.sh
#   RUN_DRY=1 RUN_DISCOVERY=1 Tests/test_eval_project358382.sh
#
# Exits 0 only when every requested tier passes.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NNI_HOME="${NNINTERACTIVE_HOME:-$HOME/.autoresearchclaw/nninteractive}"
PY="${NNI_PY_BIN:-$NNI_HOME/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: nnInteractive venv Python not found at $PY"
    echo "       Run .github/scripts/install_nninteractive.sh first, or"
    echo "       set NNI_PY_BIN to a Python with SimpleITK + VTK."
    exit 2
fi
echo "Using $PY ($($PY -c 'import sys; print(sys.version.split()[0])'))"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/runs/_smoke_eval358382}"
mkdir -p "$OUT_ROOT"

# ----------------------------------------------------------------- tier 1
echo
echo "=== Tier 1: parse + import + recipe compile ==="
"$PY" - <<'PY'
import ast, sys, importlib, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
# Handle the case where this is run via heredoc (no __file__).
src_root = pathlib.Path(".github/scripts").resolve()
sys.path.insert(0, str(src_root))

# 1. Parse the orchestrator
orch = src_root / "eval_project358382_pilot.py"
ast.parse(orch.read_text())
print("parse: ok", orch)

# 2. Import sibling modules (heavy ones rely on the venv)
for name in (
    "morphosource_client",
    "morphosource_api_download",
    "find_segmentation_pairs",
    "voxelize_mesh_vtk",
    "crop_around_mesh",
    "segmentation_metrics",
    "remote_volume_io",
    "run_telemetry",
    "slicer_remote_bright_seed",
    "nninteractive_compare",
    "eval_project358382_pilot",
):
    importlib.import_module(name)
    print(f"import: {name}: ok")

# 3. Recipe builders compile to valid Python
import remote_volume_io as r
for label, src in [
    ("LOAD",       r._build_load_source("AAAA", "demo", "a"*64)),
    ("SET_ACTIVE", r.SET_ACTIVE_VOLUME_SRC_TEMPLATE("demo")),
    ("LIST",       r.LIST_VOLUMES_SRC),
]:
    compile(src, f"<{label}>", "exec")
    print(f"recipe-{label}: compiles ({len(src)} bytes)")

# 4. The bright-seed STEP recipe also compiles
import slicer_remote_bright_seed as bs
step_src = bs.STEP_SRC_TEMPLATE.format(click_positive=True, new_segment=True)
compile(step_src, "<STEP>", "exec")
print(f"recipe-STEP: compiles ({len(step_src)} bytes)")

# 5. The pilot orchestrator's argparse builds without exploding
import eval_project358382_pilot as ep
parser = None
try:
    args = ep._parse_args(["--help"])  # exits via SystemExit(0)
except SystemExit as e:
    print(f"--help exit code: {e.code}")
PY

# ----------------------------------------------------------------- tier 2
if [[ "${RUN_DISCOVERY:-0}" == "1" ]]; then
    echo
    echo "=== Tier 2: discovery against MorphoSource ==="
    if [[ -z "${MORPHOSOURCE_API_KEY:-}" ]]; then
        echo "ERROR: MORPHOSOURCE_API_KEY not set; cannot run discovery"
        exit 3
    fi
    "$PY" - <<'PY'
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(".github/scripts").resolve()))
from morphosource_client import MorphoSourceClient
from eval_project358382_pilot import discover_pairs, select_pilot_specimens

pairs = discover_pairs(
    MorphoSourceClient(),
    project_query="Colors of Skull Anatomy",
    project_id="000358382",
    page_size=50, max_pages=1,
)
assert pairs, "discovery returned 0 pairs (expected at least 1)"
print(f"  discovery: {len(pairs)} pairs")
chosen = select_pilot_specimens(pairs, n=3)
for c in chosen:
    print(f"   -> {c.physical_object_id}  taxon={c.taxonomy or '?'}  "
          f"CT={c.ct_media_id}  mesh={c.mesh_media_id}")
PY
else
    echo
    echo "=== Tier 2: skipped (set RUN_DISCOVERY=1 to enable) ==="
fi

# ----------------------------------------------------------------- tier 3
if [[ "${RUN_DRY:-0}" == "1" ]]; then
    echo
    echo "=== Tier 3: orchestrator --dry-run ==="
    if [[ -z "${MORPHOSOURCE_API_KEY:-}" ]]; then
        echo "ERROR: MORPHOSOURCE_API_KEY not set"
        exit 3
    fi
    if [[ -z "${SLICER_WEBSERVER_URL:-}${NNI_REMOTE_URL:-}" ]]; then
        echo "ERROR: SLICER_WEBSERVER_URL / NNI_REMOTE_URL must be set "
        echo "       (the orchestrator validates them even in --dry-run)."
        exit 3
    fi
    DRY_OUT="$OUT_ROOT/dry_$(date +%Y%m%dT%H%M%S)"
    "$PY" .github/scripts/eval_project358382_pilot.py \
        --project-id 000358382 \
        --project-query "Colors of Skull Anatomy" \
        --specimens 1 \
        --budgets 10,25,50,100 \
        --max-pages 1 \
        --out-dir "$DRY_OUT" \
        --dry-run
    echo "  dry-run output -> $DRY_OUT"
    test -f "$DRY_OUT/manifest.json" || { echo "FAIL: manifest.json missing"; exit 4; }
    test -f "$DRY_OUT/inputs.json"   || { echo "FAIL: inputs.json missing"; exit 4; }
    test -f "$DRY_OUT/events.jsonl"  || { echo "FAIL: events.jsonl missing"; exit 4; }
    test -f "$DRY_OUT/report.md"     || { echo "FAIL: report.md missing"; exit 4; }
    test -f "$DRY_OUT/replay.sh"     || { echo "FAIL: replay.sh missing"; exit 4; }
    echo "  artifacts checked: manifest / inputs / events / report / replay"
else
    echo
    echo "=== Tier 3: skipped (set RUN_DRY=1 to enable) ==="
fi

echo
echo "OK"
