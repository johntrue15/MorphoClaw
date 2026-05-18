#!/usr/bin/env python3
"""
Drive a remote 3D Slicer instance via /slicer/exec — full round-trip demo.

Unlike slicer_remote_smoke.py (which only *reads* state), this script
actually *changes* Slicer's MRML scene from this Mac:

  Step 1: snapshot "before" image
  Step 2: load MRHead sample data into Slicer (downloads on the box)
  Step 3: switch the layout to FourUp + reset slice views to fit
  Step 4: snapshot "after" image
  Step 5: ask Slicer for volume metadata (name, dimensions, spacing,
           bounding box) and print it locally

If all five steps succeed, we've proven a complete remote-control loop
that we can later use to drive the SlicerNNInteractive plugin's
add_point_interaction / add_bbox_interaction methods.

The Web Server module's *Slicer API exec* checkbox must be on (see
slicer_remote_smoke.py for setup instructions). This script reads
``$SLICER_WEBSERVER_URL`` (preferred) or ``$NNI_REMOTE_URL`` from the
environment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_OUT_DIR = Path("smoke_out")


# ---------------------------------------------------------------------------
# Tiny HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _read_url() -> str:
    url = (
        os.environ.get("SLICER_WEBSERVER_URL", "").strip()
        or os.environ.get("NNI_REMOTE_URL", "").strip()
    )
    if not url:
        sys.exit(
            "ERROR: SLICER_WEBSERVER_URL (or NNI_REMOTE_URL) not set. "
            "Source your .env first:\n"
            "    set -a && source .env && set +a"
        )
    if url.startswith("ws://"):
        url = "http://" + url[len("ws://"):]
    elif url.startswith("wss://"):
        url = "https://" + url[len("wss://"):]
    return url.rstrip("/")


def post_python(base_url: str, source: str, timeout: float = 120) -> dict:
    """Send Python source to /slicer/exec and return the parsed JSON result."""
    body = source.encode("utf-8")
    req = urllib.request.Request(
        base_url + "/slicer/exec", data=body, method="POST",
        headers={"Content-Type": "text/plain"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        content = e.read()
        status = e.code
    dt_ms = (time.time() - t0) * 1e3
    if status != 200:
        sys.exit(f"  exec FAILED: HTTP {status} ({dt_ms:.0f} ms)\n  body: {content[:300]!r}")
    try:
        result = json.loads(content)
    except Exception as e:
        sys.exit(f"  exec returned non-JSON: {e}\n  body preview: {content[:300]!r}")
    result["_dt_ms"] = dt_ms
    return result


def grab_screenshot(base_url: str, out_path: Path, timeout: float = 30) -> int:
    """GET /slicer/screenshot and write the PNG to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(base_url + "/slicer/screenshot")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        status = resp.status
    dt_ms = (time.time() - t0) * 1e3
    if status != 200:
        sys.exit(f"  screenshot FAILED: HTTP {status}")
    out_path.write_bytes(data)
    print(f"  HTTP 200, {len(data)} bytes ({dt_ms:.0f} ms) -> {out_path}")
    return len(data)


# ---------------------------------------------------------------------------
# Remote Python recipes (each runs server-side via /slicer/exec)
# ---------------------------------------------------------------------------

LOAD_MRHEAD_SRC = textwrap.dedent("""
    import slicer, time
    t0 = time.time()
    # Use SampleData module's API — same code path as the GUI button.
    import SampleData
    try:
        node = SampleData.downloadFromURL(
            nodeNames="MRHead",
            fileNames="MR-head.nrrd",
            uris=("https://github.com/Slicer/SlicerTestingData/releases/"
                  "download/MD5/39b01631b7b38232a220007230624c8e"),
            checksums="MD5:39b01631b7b38232a220007230624c8e",
            loadFiles=True,
        )
        # downloadFromURL returns a list of nodes, but at this point the
        # volume should also be addressable by name.
        vol = slicer.util.getNode("MRHead")
        __execResult["loaded_node"] = vol.GetName()
        __execResult["dimensions"] = list(vol.GetImageData().GetDimensions())
        __execResult["spacing_mm"] = list(vol.GetSpacing())
        bounds = [0.0]*6
        vol.GetRASBounds(bounds)
        __execResult["ras_bounds"] = bounds
        slicer.util.setSliceViewerLayers(background=vol, fit=True)
        __execResult["seconds"] = round(time.time() - t0, 3)
        __execResult["status"] = "ok"
    except Exception as e:
        __execResult["status"] = "error"
        __execResult["error"] = repr(e)
""").strip()


SWITCH_LAYOUT_SRC = textwrap.dedent("""
    import slicer
    layoutManager = slicer.app.layoutManager()
    # 3 = "Four-Up" (red/yellow/green slice + 3D)  — Slicer.qSlicerLayoutManager.SlicerLayoutFourUpView
    layoutManager.setLayout(3)
    # Center each slice view on the loaded volume.
    for sliceViewName in ("Red", "Yellow", "Green"):
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        if sliceWidget:
            sliceWidget.sliceLogic().FitSliceToAll()
    # Reset 3D camera too.
    threeDWidget = layoutManager.threeDWidget(0)
    if threeDWidget:
        threeDWidget.threeDView().resetFocalPoint()
        threeDWidget.threeDView().resetCamera()
    __execResult["layout_id"] = layoutManager.layout
    __execResult["status"] = "ok"
""").strip()


VOLUME_INFO_SRC = textwrap.dedent("""
    import slicer
    info = []
    for vol in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        img = vol.GetImageData()
        info.append({
            "name": vol.GetName(),
            "id": vol.GetID(),
            "dimensions": list(img.GetDimensions()) if img else None,
            "spacing": list(vol.GetSpacing()),
            "origin": list(vol.GetOrigin()),
            "scalar_type": img.GetScalarTypeAsString() if img else None,
            "n_scalars": img.GetPointData().GetScalars().GetNumberOfTuples() if img else None,
        })
    __execResult["volumes"] = info
    __execResult["count"] = len(info)
""").strip()


# Probe SlicerNNInteractive too, so we know if it's installed and which
# server URL it currently points at — useful for the next step.
NNI_PROBE_SRC = textwrap.dedent("""
    import slicer
    out = {"installed": False, "server": None, "version": None}
    try:
        mod = slicer.modules.SlicerNNInteractive
        out["installed"] = True
        try:
            widget = mod.widgetRepresentation()
            self_ = widget.self() if widget else None
            if self_ and hasattr(self_, "server"):
                out["server"] = getattr(self_, "server", None)
        except Exception as e:
            out["widget_error"] = repr(e)
        try:
            settings = slicer.app.userSettings()
            out["saved_server_url"] = settings.value(
                "SlicerNNInteractive/serverURL", ""
            )
        except Exception:
            pass
    except AttributeError:
        out["installed"] = False
    __execResult.update(out)
""").strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--skip-load", action="store_true",
                   help="skip MRHead download (useful if it's already loaded)")
    args = p.parse_args(argv)

    base = _read_url()
    print(f"=== Slicer remote drive demo ===")
    print(f"target: {base}")
    print()

    # ── 1. Before screenshot ────────────────────────────────────────────
    print("[1/5] capturing 'before' screenshot…")
    grab_screenshot(base, args.out_dir / "drive_01_before.png")

    # ── 2. Probe SlicerNNInteractive availability ───────────────────────
    print()
    print("[2/5] probing SlicerNNInteractive plugin (so we know what we have)…")
    nni = post_python(base, NNI_PROBE_SRC, timeout=15)
    print(f"      installed         : {nni.get('installed')}")
    print(f"      saved_server_url  : {nni.get('saved_server_url')!r}")
    print(f"      currently using   : {nni.get('server')!r}")
    if "widget_error" in nni:
        print(f"      widget_error      : {nni['widget_error']}")

    # ── 3. Load MRHead sample data ──────────────────────────────────────
    if not args.skip_load:
        print()
        print("[3/5] loading MRHead sample volume into Slicer…")
        load = post_python(base, LOAD_MRHEAD_SRC, timeout=180)
        if load.get("status") != "ok":
            print(f"      ERROR: {load.get('error')}")
            return 2
        print(f"      loaded            : {load['loaded_node']}")
        print(f"      dimensions        : {load['dimensions']}")
        print(f"      spacing (mm)      : {[round(s, 3) for s in load['spacing_mm']]}")
        print(f"      RAS bounds        : {[round(b, 1) for b in load['ras_bounds']]}")
        print(f"      remote duration   : {load['seconds']} s")
    else:
        print("\n[3/5] skipped MRHead load (per --skip-load)")

    # ── 4. Switch layout + reset views ──────────────────────────────────
    print()
    print("[4/5] switching layout to FourUp + resetting views…")
    layout = post_python(base, SWITCH_LAYOUT_SRC, timeout=15)
    print(f"      layout_id         : {layout.get('layout_id')}  (3 = FourUp)")

    # ── 5. After screenshot + volume info ───────────────────────────────
    print()
    print("[5/5] capturing 'after' screenshot + listing volumes…")
    grab_screenshot(base, args.out_dir / "drive_05_after.png")
    info = post_python(base, VOLUME_INFO_SRC, timeout=15)
    print(f"      volumes in scene  : {info['count']}")
    for v in info["volumes"]:
        print(f"        - {v['name']:20s}  dims={v['dimensions']}  "
              f"spacing={[round(s, 2) for s in v['spacing']]}  "
              f"type={v['scalar_type']}")

    print()
    print("DONE — Slicer state was changed remotely.")
    print(f"  Compare:  {args.out_dir / 'drive_01_before.png'}")
    print(f"            {args.out_dir / 'drive_05_after.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
