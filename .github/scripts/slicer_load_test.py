"""Headless SlicerMorph load probe.

Runs INSIDE 3D Slicer via ``Slicer --no-splash --no-main-window
--python-script slicer_load_test.py``.

Loads the model whose path is passed via the ``SLICER_LOAD_PATH``
environment variable, performs a few SlicerMorph-sanity checks
(slicer.util.loadModel, mesh point count, bounds non-degenerate),
optionally verifies that the SlicerMorph extension is importable,
and writes the result as JSON to ``SLICER_LOAD_OUT`` before exiting.

Exit codes:
    0 - success, JSON written
    2 - SLICER_LOAD_PATH missing or file not found
    3 - load failed
    4 - mesh sanity check failed
    5 - SlicerMorph extension missing (only when required)

The companion pytest test (``Tests/test_slicer_cached_model.py``)
spawns this script and parses the JSON to assert on the result.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

try:
    import slicer  # type: ignore  # provided by 3D Slicer's embedded Python
except Exception as exc:  # pragma: no cover - only meaningful inside Slicer
    sys.stderr.write(f"slicer_load_test: cannot import slicer module: {exc}\n")
    sys.exit(1)


LOAD_PATH = os.environ.get("SLICER_LOAD_PATH", "")
OUT_PATH = os.environ.get("SLICER_LOAD_OUT", "")
REQUIRE_SLICERMORPH = os.environ.get("SLICER_REQUIRE_SLICERMORPH", "0") == "1"
MIN_POINTS = int(os.environ.get("SLICER_MIN_POINTS", "1"))


def _emit(payload: dict, exit_code: int) -> None:
    payload.setdefault("exit_code", exit_code)
    serialised = json.dumps(payload, indent=2)
    print(serialised)
    if OUT_PATH:
        try:
            with open(OUT_PATH, "w", encoding="utf-8") as fh:
                fh.write(serialised)
        except OSError as exc:
            sys.stderr.write(f"slicer_load_test: could not write {OUT_PATH}: {exc}\n")
    slicer.app.exit(exit_code)
    sys.exit(exit_code)


if not LOAD_PATH or not os.path.exists(LOAD_PATH):
    _emit(
        {
            "ok": False,
            "error": f"SLICER_LOAD_PATH not set or missing: {LOAD_PATH!r}",
            "load_path": LOAD_PATH,
        },
        2,
    )

slicermorph_available = False
slicermorph_error = ""
try:
    # SlicerMorph is a meta-extension; one of its sub-modules being importable
    # is a good proxy for the bundle being installed.
    import GPA  # type: ignore  # noqa: F401  -- import for side-effect / availability check

    slicermorph_available = True
except Exception as exc:
    slicermorph_error = f"{type(exc).__name__}: {exc}"

if REQUIRE_SLICERMORPH and not slicermorph_available:
    _emit(
        {
            "ok": False,
            "error": "SlicerMorph extension not available",
            "slicermorph_error": slicermorph_error,
            "load_path": LOAD_PATH,
        },
        5,
    )

try:
    success, model_node = slicer.util.loadModel(LOAD_PATH, returnNode=True)
except Exception as exc:
    _emit(
        {
            "ok": False,
            "error": "loadModel raised",
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
            "load_path": LOAD_PATH,
        },
        3,
    )

if not success or model_node is None:
    _emit(
        {
            "ok": False,
            "error": "loadModel returned no node",
            "load_path": LOAD_PATH,
        },
        3,
    )

mesh = model_node.GetMesh()
if mesh is None:
    _emit(
        {
            "ok": False,
            "error": "Loaded model has no mesh",
            "load_path": LOAD_PATH,
        },
        4,
    )

n_points = mesh.GetNumberOfPoints()
n_cells = mesh.GetNumberOfCells()
bounds = [0.0] * 6
model_node.GetBounds(bounds)
extent = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]

if n_points < MIN_POINTS or all(e == 0 for e in extent):
    _emit(
        {
            "ok": False,
            "error": "Mesh sanity check failed",
            "n_points": n_points,
            "n_cells": n_cells,
            "bounds": bounds,
            "extent": extent,
            "load_path": LOAD_PATH,
        },
        4,
    )

_emit(
    {
        "ok": True,
        "load_path": LOAD_PATH,
        "n_points": n_points,
        "n_cells": n_cells,
        "bounds": bounds,
        "extent": extent,
        "slicermorph_available": slicermorph_available,
        "slicermorph_error": slicermorph_error,
        "node_name": model_node.GetName(),
    },
    0,
)
