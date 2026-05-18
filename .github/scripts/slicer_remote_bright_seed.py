#!/usr/bin/env python3
"""
Bright-spot greedy nnInteractive segmentation, driven from this Mac.

A deterministic counterpart to slicer_remote_loop.py — no LLM, no API
costs. The algorithm is:

  1. Threshold the volume at a chosen percentile (e.g. 99th) and collect
     every voxel above the threshold, sorted by intensity descending.
     This is the candidate seed list.
  2. While we still have budget AND candidates left:
       a. Read the current segmentation mask from the active segmentation
          node.
       b. Skip past any candidate that's already inside the mask.
       c. Click a positive point at the next candidate via
          ``plugin.point_prompt``.
       d. Re-read the mask, compute the voxel delta.
       e. Record the step. If the last few deltas are tiny we stop early
          (segmentation has saturated).

All heavy lifting (volume read, mask read, candidate selection, prompt
call) happens server-side inside Slicer over /slicer/exec; the script
sends a small Python recipe and gets back ~hundreds of bytes of JSON
per step. That's important: a 259x258x421 IMPC volume is ~28 MB, and
streaming it across the proxy every step would be wasteful.

Per-step artifacts go to ``runs/<name>/step_NN/``:
  red.png, yellow.png, green.png, threeD.png, state.json

Env vars
--------
  SLICER_WEBSERVER_URL  e.g. https://http-149-...-2016.proxy-js2-iu.exosphere.app/

Usage
-----
  set -a && source .env && set +a
  python3 .github/scripts/slicer_remote_bright_seed.py \\
      --volume IMPC_sample_data \\
      --reset-first \\
      --intensity-percentile 99 \\
      --max-steps 20 \\
      --out-dir runs/impc_bright_seed
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_telemetry import (  # noqa: E402  (sibling module)
    CAPTURE_REMOTE_ENV_SRC,
    EXPORT_SEGMENTATION_SRC,
    HASH_ACTIVE_VOLUME_SRC,
    RunLogger,
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _read_url() -> str:
    url = (
        os.environ.get("SLICER_WEBSERVER_URL", "").strip()
        or os.environ.get("NNI_REMOTE_URL", "").strip()
    )
    if not url:
        sys.exit("ERROR: set SLICER_WEBSERVER_URL or NNI_REMOTE_URL")
    if url.startswith("ws://"):
        url = "http://" + url[len("ws://"):]
    elif url.startswith("wss://"):
        url = "https://" + url[len("wss://"):]
    return url.rstrip("/")


def http_get(url: str, timeout: float = 20) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"GET {url} -> HTTP {resp.status}")
        return resp.read()


def post_python(base_url: str, source: str, timeout: float = 240) -> dict:
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
    if status != 200:
        raise RuntimeError(f"/slicer/exec -> HTTP {status}: {content[:300]!r}")
    try:
        result = json.loads(content)
    except Exception:
        raise RuntimeError(f"non-JSON exec reply: {content[:300]!r}")
    result["_dt_s"] = round(time.time() - t0, 3)
    return result


# Recipe that returns base64-encoded PNGs of each Slicer widget's *actual*
# rendered output, so segmentation overlays are included. The default
# /slicer/slice endpoint returns the underlying volume only (no
# segmentation overlay), which makes it useless for visual feedback.
GRAB_VIEWS_SRC = """\
import slicer, base64, os, tempfile, traceback
def _grab_widget(w):
    if w is None:
        return None
    pm = w.grab()
    fd, path = tempfile.mkstemp(suffix=".png", prefix="bs_grab_")
    os.close(fd)
    try:
        ok = pm.save(path, "PNG")
        if not ok:
            return None
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("ascii")
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
out = {}
try:
    # Make sure 3D rendering exists for every segmentation that has voxels.
    # Closed-surface reps are needed for the 3D view to actually show
    # something; binary labelmaps alone are 2D-only.
    for sn in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        if "do not touch" in sn.GetName().lower():
            continue
        try:
            sn.CreateClosedSurfaceRepresentation()
        except Exception:
            pass
        d = sn.GetDisplayNode()
        if d:
            d.SetVisibility(True)
            d.SetVisibility2DFill(True)
            d.SetVisibility2DOutline(True)
            d.SetVisibility3D(True)
    lm = slicer.app.layoutManager()
    for color, name in (("Red", "red"), ("Yellow", "yellow"), ("Green", "green")):
        try:
            sw = lm.sliceWidget(color)
            view = sw.sliceView() if sw else None
            if view is not None:
                view.scheduleRender()
                slicer.app.processEvents()
            out[name + "_png_b64"] = _grab_widget(view)
        except Exception as e:
            out[name + "_err"] = repr(e)
    try:
        if lm.threeDViewCount > 0:
            tw = lm.threeDWidget(0)
            tv = tw.threeDView() if tw else None
            if tv is not None:
                tv.resetFocalPoint()
                tv.resetCamera()
                tv.scheduleRender()
                slicer.app.processEvents()
                tv.forceRender()
            out["threeD_png_b64"] = _grab_widget(tv)
    except Exception as e:
        out["threeD_err"] = repr(e)
    out["status"] = "ok"
except Exception as e:
    out["status"] = "exception"
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
__execResult.update(out)
"""


def capture_views(base_url: str, step_dir: Path) -> None:
    """Save red.png / yellow.png / green.png / threeD.png plus a
    ``window.png`` of the full Slicer main window for context."""
    step_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = post_python(base_url, GRAB_VIEWS_SRC, timeout=30)
    except Exception as e:
        (step_dir / "grab.err").write_text(repr(e))
        r = {}
    import base64
    for color in ("red", "yellow", "green", "threeD"):
        b64 = r.get(f"{color}_png_b64")
        if b64:
            try:
                (step_dir / f"{color}.png").write_bytes(base64.b64decode(b64))
            except Exception as e:
                (step_dir / f"{color}.err").write_text(repr(e))
        elif r.get(f"{color}_err"):
            (step_dir / f"{color}.err").write_text(r[f"{color}_err"])
    try:
        (step_dir / "window.png").write_bytes(
            http_get(f"{base_url}/slicer/screenshot", timeout=15)
        )
    except Exception as e:
        (step_dir / "window.err").write_text(repr(e))


# ---------------------------------------------------------------------------
# Server-side recipes (run inside Slicer's Python via /slicer/exec)
# ---------------------------------------------------------------------------

# Set the active volume by name and recenter slice views on it.
SET_ACTIVE_VOLUME_SRC_TEMPLATE = textwrap.dedent("""
    import slicer
    target_name = {target_name!r}
    found = None
    for v in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if v.GetName() == target_name:
            found = v
            break
    if found is None:
        __execResult["status"] = "not_found"
        __execResult["available"] = [
            v.GetName() for v in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        ]
    else:
        sel = slicer.app.applicationLogic().GetSelectionNode()
        sel.SetActiveVolumeID(found.GetID())
        slicer.app.applicationLogic().PropagateVolumeSelection(0)
        slicer.util.setSliceViewerLayers(background=found, fit=True)
        __execResult["status"] = "ok"
        __execResult["volume_id"] = found.GetID()
        img = found.GetImageData()
        __execResult["dimensions_ijk"] = list(img.GetDimensions())
        __execResult["spacing_mm"] = [round(s, 4) for s in found.GetSpacing()]
""").strip()


# Reset segmentation: clear scene + tell the server to forget interactions.
RESET_SEGMENTATION_SRC = textwrap.dedent("""
    import slicer, io, gzip
    import numpy as np
    import requests
    out = {}
    try:
        mod = slicer.modules.slicernninteractive
        plugin = mod.widgetRepresentation().self()
        vol = plugin.get_volume_node()
        if vol is None:
            __execResult["status"] = "no_active_volume"
        else:
            arr = slicer.util.arrayFromVolume(vol)
            empty = np.zeros(arr.shape, dtype=np.uint8)
            buf = io.BytesIO()
            np.save(buf, empty, allow_pickle=False)
            r = requests.post(
                f"{plugin.server}/upload_segment",
                files={"file": ("seg.npy.gz",
                                io.BytesIO(gzip.compress(buf.getvalue())),
                                "application/octet-stream")},
                timeout=120,
            )
            cleared = []
            for sn in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
                if "do not touch" in sn.GetName().lower():
                    continue
                seg = sn.GetSegmentation()
                seg.RemoveAllSegments()
                cleared.append(sn.GetName())
            try:
                plugin.setup_prompts()
            except Exception:
                pass
            out["status"] = "ok"
            out["upload_segment_status"] = r.status_code
            out["cleared_nodes"] = cleared
    except Exception as e:
        out["status"] = "error"
        out["error"] = repr(e)
    __execResult.update(out)
""").strip()


# Build the candidate seed list: voxels above ``intensity_percentile`` of
# the volume's intensity distribution, sorted by intensity descending.
# Stash the result in globals() under ``_BS_STATE`` so subsequent /exec
# calls can read it without re-computing or re-uploading the volume.
INIT_CANDIDATES_SRC_TEMPLATE = textwrap.dedent("""
    import slicer
    import numpy as np
    out = {{}}
    try:
        sel = slicer.app.applicationLogic().GetSelectionNode()
        vol = slicer.mrmlScene.GetNodeByID(sel.GetActiveVolumeID())
        if vol is None:
            __execResult["status"] = "no_active_volume"
        else:
            arr = slicer.util.arrayFromVolume(vol)  # (k, j, i)
            arr_f = arr.astype(np.float32, copy=False)
            t = float(np.percentile(arr_f, {percentile}))
            mask = arr_f >= t
            ks, js, is_ = np.where(mask)
            if len(ks) == 0:
                __execResult["status"] = "no_bright_voxels"
            else:
                intensities = arr_f[ks, js, is_]
                order = np.argsort(-intensities)  # descending
                cap = int({max_candidates})
                if cap > 0 and len(order) > cap:
                    order = order[:cap]
                # Encode candidates compactly: parallel arrays of int.
                cand_kji = np.stack([ks[order], js[order], is_[order]], axis=1).astype(np.int32)
                cand_int = intensities[order].astype(np.float32)
                globals()["_BS_STATE"] = {{
                    "volume_id": vol.GetID(),
                    "volume_name": vol.GetName(),
                    "shape_kji": list(arr.shape),
                    "threshold": t,
                    "percentile": float({percentile}),
                    "next_idx": 0,
                    "history": [],
                    "candidates_kji": cand_kji,
                    "candidates_intensity": cand_int,
                }}
                out["status"] = "ok"
                out["volume_name"] = vol.GetName()
                out["shape_kji"] = list(arr.shape)
                out["threshold"] = t
                out["n_candidates"] = int(len(cand_int))
                out["intensity_min"] = float(cand_int.min())
                out["intensity_max"] = float(cand_int.max())
                out["scalar_type"] = str(arr.dtype)
    except Exception as e:
        out["status"] = "error"
        out["error"] = repr(e)
    __execResult.update(out)
""").strip()


# Step recipe: skim past candidates already inside the mask, click the
# next one, return mask deltas + intensity. ``click_positive`` lets the
# caller occasionally click negative if needed (e.g. cleanup).
#
# Each click goes into its OWN segment (so structures stay separable):
#   - For the first step, we use whatever segment is currently selected
#     (after --reset-first there's an empty default segment, or the
#     plugin auto-creates one when point_prompt fires).
#   - For every subsequent step, we call ``plugin.make_new_segment()``
#     BEFORE the click so nnInteractive paints into a fresh segment
#     instead of refining the previous one.
#
# IMPORTANT: this string is written flush-left (no leading whitespace
# anywhere) so the eventual /slicer/exec call sees valid top-level
# Python. Don't reindent.
STEP_SRC_TEMPLATE = """\
import slicer, time, traceback
import numpy as np
def _bs_read_total_mask(shape):
    total = np.zeros(shape, dtype=bool)
    per_seg = []
    for sn in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        if "do not touch" in sn.GetName().lower():
            continue
        seg = sn.GetSegmentation()
        for ii in range(seg.GetNumberOfSegments()):
            sid = seg.GetNthSegmentID(ii)
            try:
                a = slicer.util.arrayFromSegmentBinaryLabelmap(sn, sid)
            except Exception:
                a = None
            if a is None or a.shape != shape:
                continue
            ab = a > 0
            total |= ab
            per_seg.append({{
                "node": sn.GetName(), "sid": sid,
                "voxels": int(ab.sum()),
            }})
    return total, per_seg
try:
    state = globals().get("_BS_STATE")
    if state is None:
        __execResult["status"] = "not_initialized"
    else:
        shape = tuple(state["shape_kji"])
        cand_kji = state["candidates_kji"]
        cand_int = state["candidates_intensity"]
        idx = int(state["next_idx"])
        mask_before, segs_before = _bs_read_total_mask(shape)
        voxels_before = int(mask_before.sum())
        picked = None
        skipped_inside = 0
        while idx < len(cand_int):
            k = int(cand_kji[idx, 0])
            j = int(cand_kji[idx, 1])
            i = int(cand_kji[idx, 2])
            if mask_before[k, j, i]:
                idx += 1
                skipped_inside += 1
                continue
            picked = (k, j, i, float(cand_int[idx]))
            idx += 1
            break
        if picked is None:
            state["next_idx"] = idx
            __execResult["status"] = "no_more_candidates"
            __execResult["candidates_left"] = 0
            __execResult["voxels"] = voxels_before
            __execResult["skipped_inside"] = skipped_inside
        else:
            k, j, i, intensity = picked
            click_positive = bool({click_positive})
            new_segment = bool({new_segment}) and len(state["history"]) > 0
            mod = slicer.modules.slicernninteractive
            plugin = mod.widgetRepresentation().self()
            new_segment_id = None
            if new_segment:
                plugin.make_new_segment()
                try:
                    new_segment_id = plugin.get_current_segment_id()
                except Exception:
                    new_segment_id = None
            if click_positive:
                plugin.on_prompt_type_positive_clicked()
            else:
                plugin.on_prompt_type_negative_clicked()
            t0 = time.time()
            plugin.point_prompt(xyz=[i, j, k], positive_click=click_positive)
            t1 = time.time()
            mask_after, segs_after = _bs_read_total_mask(shape)
            voxels_after = int(mask_after.sum())
            delta = voxels_after - voxels_before
            new_voxels_at_picked = int(mask_after[k, j, i])
            current_segment_id = None
            try:
                current_segment_id = plugin.get_current_segment_id()
            except Exception:
                pass
            current_segment_voxels = 0
            for s in segs_after:
                if s["sid"] == current_segment_id:
                    current_segment_voxels = s["voxels"]
                    break
            state["next_idx"] = idx
            state["history"].append({{
                "ijk": [i, j, k], "intensity": intensity,
                "voxels_before": voxels_before, "voxels_after": voxels_after,
                "delta": delta, "click_positive": click_positive,
                "skipped_inside": skipped_inside,
                "made_new_segment": new_segment,
                "segment_id": current_segment_id,
                "segment_voxels": current_segment_voxels,
                "n_segments_before": len(segs_before),
                "n_segments_after": len(segs_after),
            }})
            __execResult["status"] = "ok"
            __execResult["picked_ijk"] = [i, j, k]
            __execResult["intensity"] = intensity
            __execResult["voxels_before"] = voxels_before
            __execResult["voxels_after"] = voxels_after
            __execResult["delta"] = delta
            __execResult["new_voxels_at_picked"] = new_voxels_at_picked
            __execResult["click_positive"] = click_positive
            __execResult["skipped_inside"] = skipped_inside
            __execResult["candidates_left"] = int(len(cand_int) - idx)
            __execResult["click_seconds"] = round(t1 - t0, 3)
            __execResult["made_new_segment"] = new_segment
            __execResult["segment_id"] = current_segment_id
            __execResult["segment_voxels"] = current_segment_voxels
            __execResult["n_segments_before"] = len(segs_before)
            __execResult["n_segments_after"] = len(segs_after)
except Exception as e:
    __execResult["status"] = "exception"
    __execResult["error"] = repr(e)
    __execResult["traceback"] = traceback.format_exc()
"""


# One-time configuration to make the segmentation actually visible in
# slice views and the 3D view. Without this, the screenshots show only
# the underlying CT and an empty 3D box.
ENABLE_VISIBILITY_SRC = """\
import slicer, traceback
try:
    nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    for sn in nodes:
        if "do not touch" in sn.GetName().lower():
            continue
        sn.CreateDefaultDisplayNodes()
        sn.CreateClosedSurfaceRepresentation()
        d = sn.GetDisplayNode()
        if d:
            d.SetVisibility(True)
            d.SetVisibility2DFill(True)
            d.SetVisibility2DOutline(True)
            d.SetVisibility3D(True)
            d.SetOpacity(0.6)
            d.SetOpacity2DFill(0.5)
            d.SetOpacity2DOutline(1.0)
    lm = slicer.app.layoutManager()
    for vi in range(lm.threeDViewCount):
        v = lm.threeDWidget(vi).threeDView()
        v.resetFocalPoint()
        v.resetCamera()
    __execResult["nodes"] = [n.GetName() for n in nodes]
    __execResult["status"] = "ok"
except Exception as e:
    __execResult["status"] = "exception"
    __execResult["error"] = repr(e)
    __execResult["traceback"] = traceback.format_exc()
"""


# Recenter slice views on a picked voxel so the next screenshot shows
# the action.
RECENTER_SRC_TEMPLATE = textwrap.dedent("""
    import slicer, vtk
    i, j, k = {i}, {j}, {k}
    sel = slicer.app.applicationLogic().GetSelectionNode()
    vol = slicer.mrmlScene.GetNodeByID(sel.GetActiveVolumeID())
    if vol:
        m = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(m)
        ras4 = [0.0]*4
        m.MultiplyPoint([float(i), float(j), float(k), 1.0], ras4)
        ras = ras4[:3]
        for color in ("Red", "Yellow", "Green"):
            sw = slicer.app.layoutManager().sliceWidget(color)
            if sw:
                sw.sliceLogic().GetSliceNode().JumpSliceByCentering(*ras)
        __execResult["ras"] = [round(r, 2) for r in ras]
""").strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--volume",
                   help="Name of the vtkMRMLScalarVolumeNode to segment "
                        "(default: keep whatever is currently active)")
    p.add_argument("--reset-first", action="store_true",
                   help="reset any existing segmentation on the active volume "
                        "before starting")
    p.add_argument("--intensity-percentile", type=float, default=99.0,
                   help="threshold percentile for bright voxels "
                        "(default 99 — top 1%% intensities)")
    p.add_argument("--max-candidates", type=int, default=200_000,
                   help="hard cap on the number of bright voxels to track "
                        "(default 200k; keeps the state small)")
    p.add_argument("--max-steps", type=int, default=20,
                   help="maximum number of clicks to issue (default 20)")
    p.add_argument("--min-delta", type=int, default=50,
                   help="if the last --patience deltas are all below this "
                        "many voxels, stop early (default 50 voxels)")
    p.add_argument("--patience", type=int, default=3,
                   help="number of consecutive small-delta steps before "
                        "early stopping (default 3)")
    p.add_argument("--max-explosion-frac", type=float, default=0.5,
                   help="if a single click adds more than this fraction "
                        "of the total volume voxels to the mask, treat it "
                        "as runaway and stop (default 0.5)")
    p.add_argument("--no-new-segment-per-click", action="store_true",
                   help="put every click into the SAME segment (default: "
                        "each click after the first creates a fresh segment "
                        "via plugin.make_new_segment(), so structures "
                        "stay separable)")
    p.add_argument("--no-screenshots", action="store_true",
                   help="skip per-step view captures (faster)")
    p.add_argument("--no-export-segmentation", action="store_true",
                   help="skip exporting per-segment + composite NIfTI "
                        "labelmaps as run artifacts (saves bandwidth, but "
                        "results then aren't independently reproducible)")
    p.add_argument("--label", type=str, default=None,
                   help="optional label embedded in the run id "
                        "(e.g. 'mouse_skull')")
    p.add_argument("--out-dir", type=Path,
                   default=Path("runs") / time.strftime("bright_%Y%m%d_%H%M%S"))
    args = p.parse_args(argv)

    base_url = _read_url()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Provenance: open the run logger first, so anything that crashes
    # below this point leaves a partial-but-readable record.
    # ------------------------------------------------------------------
    logger = RunLogger.start(
        root=args.out_dir,
        args={k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()},
        label=args.label,
    )
    logger.log("=== Slicer remote bright-spot greedy segmentation ===")
    logger.log(f"server       : {base_url}")
    logger.log(f"run id       : {logger.run_id}")
    logger.log(f"out          : {args.out_dir}")
    logger.log(f"percentile   : {args.intensity_percentile}")
    logger.log(f"max steps    : {args.max_steps}")
    logger.log(f"min delta    : {args.min_delta}  (patience={args.patience})")
    logger.log("")

    # Local environment (this Mac, git commit, package versions, env vars)
    local_env = logger.record_local_env()
    logger.log(f"git commit   : {local_env.get('git_commit')}  "
               f"dirty={local_env.get('git_dirty')}")
    if local_env.get("git_dirty"):
        logger.log("WARNING: working tree is dirty; the recorded git_commit "
                   "may not match the script that's running.")

    # Remote environment (Slicer + plugin + nnInteractive + torch + device)
    logger.log("-> Capturing remote environment (Slicer/plugin/torch)…")
    try:
        remote_env = post_python(base_url, CAPTURE_REMOTE_ENV_SRC, timeout=60)
        logger.record_remote_env(remote_env)
        logger.log(f"   slicer       : {remote_env.get('slicer_version')}")
        logger.log(f"   torch        : {remote_env.get('torch_version')}  "
                   f"cuda={remote_env.get('torch_cuda_available')}  "
                   f"mps={remote_env.get('torch_mps_available')}")
        logger.log(f"   nnInteractive: {remote_env.get('nninteractive_version')}")
        if "slicernninteractive_git_commit" in remote_env:
            logger.log(f"   plugin commit: {remote_env['slicernninteractive_git_commit']}")
        if "nninteractive_model_total_bytes" in remote_env:
            logger.log(f"   model bytes  : {remote_env['nninteractive_model_total_bytes']:,}")
    except Exception as e:
        logger.log(f"   remote env capture failed: {e!r}")
        logger.event("remote_env_failed", error=repr(e))

    # ------------------------------------------------------------------
    # Volume selection + reset + visibility
    # ------------------------------------------------------------------
    if args.volume:
        logger.log(f"-> Setting active volume to {args.volume!r}")
        r = post_python(base_url,
                        SET_ACTIVE_VOLUME_SRC_TEMPLATE.format(target_name=args.volume),
                        timeout=20)
        if r.get("status") != "ok":
            logger.log(f"   FAILED: {r}")
            logger.event("volume_set_failed", **r)
            logger.finalize(stop_reason={"reason": "volume_not_found", "details": r})
            return 2
        logger.event("volume_set", **r)
        logger.log(f"   id={r['volume_id']}  dims={r.get('dimensions_ijk')}  "
                   f"spacing={r.get('spacing_mm')}")

    # Hash the input volume — the single most important provenance step
    logger.log("-> Hashing active volume…")
    vol_meta = post_python(base_url, HASH_ACTIVE_VOLUME_SRC, timeout=120)
    if vol_meta.get("status") != "ok":
        logger.log(f"   FAILED: {vol_meta}")
        logger.event("volume_hash_failed", **vol_meta)
        logger.finalize(stop_reason={"reason": "volume_hash_failed",
                                     "details": vol_meta})
        return 2
    logger.record_inputs(vol_meta)
    logger.log(f"   sha256(voxels) = {vol_meta['sha256_voxels'][:16]}…  "
               f"shape={vol_meta['shape_kji']}  dtype={vol_meta['dtype']}")

    if args.reset_first:
        logger.log("-> Resetting segmentation (server + scene)…")
        r = post_python(base_url, RESET_SEGMENTATION_SRC, timeout=120)
        logger.log(f"   {r.get('status')}  cleared={r.get('cleared_nodes')}")
        logger.event("reset", **r)

    logger.log("-> Enabling segmentation visibility (2D + 3D)…")
    r = post_python(base_url, ENABLE_VISIBILITY_SRC, timeout=30)
    logger.log(f"   {r.get('status')}  nodes={r.get('nodes')}")
    logger.event("visibility_enabled", **r)

    logger.log("-> Building bright-pixel candidate list…")
    init = post_python(
        base_url,
        INIT_CANDIDATES_SRC_TEMPLATE.format(
            percentile=args.intensity_percentile,
            max_candidates=args.max_candidates,
        ),
        timeout=60,
    )
    if init.get("status") != "ok":
        logger.log(f"   FAILED: {init}")
        logger.event("candidates_failed", **init)
        logger.finalize(stop_reason={"reason": "candidates_failed",
                                     "details": init})
        return 3
    total_voxels = 1
    for d in init.get("shape_kji", [1, 1, 1]):
        total_voxels *= int(d)
    logger.log(f"   volume       : {init['volume_name']}  shape(k,j,i)={init['shape_kji']}")
    logger.log(f"   threshold    : {init['threshold']:.2f}  ({init['scalar_type']})")
    logger.log(f"   candidates   : {init['n_candidates']:,}  "
               f"intensities=[{init['intensity_min']:.1f}, {init['intensity_max']:.1f}]")
    logger.log(f"   total voxels : {total_voxels:,}")
    logger.event("candidates_built", total_voxels=total_voxels, **init)

    # ------------------------------------------------------------------
    # Greedy click loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    consecutive_small = 0
    explosion_threshold = int(args.max_explosion_frac * total_voxels)
    new_segment_per_click = not args.no_new_segment_per_click
    stop_reason = None

    for step in range(args.max_steps):
        step_dir = args.out_dir / f"step_{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        logger.log(f"--- Step {step:02d} -----------------------------------------")
        logger.event("step_begin", step=step)
        t_step0 = time.time()
        r = post_python(
            base_url,
            STEP_SRC_TEMPLATE.format(
                click_positive=True,
                new_segment=new_segment_per_click,
            ),
            timeout=240,
        )
        r["step"] = step
        r["step_wallclock_s"] = round(time.time() - t_step0, 3)

        if r.get("status") == "no_more_candidates":
            logger.log(f"  no more bright candidates outside the mask "
                       f"(voxels={r.get('voxels')})")
            (step_dir / "state.json").write_text(json.dumps(r, indent=2))
            logger.event("step_end", **r)
            stop_reason = {"reason": "no_more_candidates",
                            "step": step,
                            "voxels": r.get("voxels")}
            break

        if r.get("status") != "ok":
            logger.log(f"  STEP FAILED: {r}")
            (step_dir / "state.json").write_text(json.dumps(r, indent=2))
            logger.event("step_failed", **r)
            stop_reason = {"reason": "step_failed", "step": step, "details": r}
            logger.finalize(stop_reason=stop_reason,
                             summary={"steps": len(history), "history": history})
            return 4

        ijk = r["picked_ijk"]
        intensity = r["intensity"]
        before = r["voxels_before"]
        after = r["voxels_after"]
        delta = r["delta"]
        skipped = r["skipped_inside"]
        cand_left = r["candidates_left"]
        click_s = r["click_seconds"]
        seg_id = r.get("segment_id")
        seg_vox = r.get("segment_voxels", 0)
        n_segs = r.get("n_segments_after", "?")
        new_seg = "new-seg" if r.get("made_new_segment") else "same-seg"
        logger.log(f"  picked ijk={ijk}  intensity={intensity:.1f}  {new_seg}  "
                   f"segments={n_segs}  skipped_inside={skipped}  candidates_left={cand_left:,}")
        logger.log(f"  voxels {before:>10,} -> {after:>10,}  delta={delta:+,}  "
                   f"this segment: {seg_id} = {seg_vox:,} vox  ({click_s}s remote, "
                   f"{r['step_wallclock_s']}s round-trip)")

        if not args.no_screenshots:
            try:
                post_python(
                    base_url,
                    RECENTER_SRC_TEMPLATE.format(i=ijk[0], j=ijk[1], k=ijk[2]),
                    timeout=15,
                )
            except Exception as e:
                logger.log(f"  recenter warning: {e}")
                logger.event("recenter_warning", step=step, error=repr(e))
            try:
                capture_views(base_url, step_dir)
            except Exception as e:
                logger.log(f"  screenshot warning: {e}")
                logger.event("screenshot_warning", step=step, error=repr(e))

        (step_dir / "state.json").write_text(json.dumps(r, indent=2))
        logger.event("step_end", **r)
        history.append(r)

        if delta >= explosion_threshold:
            logger.log(f"  RUNAWAY: delta={delta:,} >= {explosion_threshold:,} "
                       f"({100*args.max_explosion_frac:.0f}% of total volume). Stopping.")
            stop_reason = {"reason": "runaway", "step": step,
                           "delta": delta,
                           "explosion_threshold": explosion_threshold}
            break
        if delta <= args.min_delta:
            consecutive_small += 1
            if consecutive_small >= args.patience:
                logger.log(f"  saturated (last {consecutive_small} deltas "
                           f"<= {args.min_delta}). Stopping.")
                stop_reason = {"reason": "saturated", "step": step,
                               "consecutive_small": consecutive_small,
                               "min_delta": args.min_delta,
                               "patience": args.patience}
                break
        else:
            consecutive_small = 0

    if stop_reason is None:
        stop_reason = {"reason": "max_steps", "max_steps": args.max_steps}

    # ------------------------------------------------------------------
    # Export final segmentation as artifacts (NIfTI + checksums) so the
    # paper has hashable, redistributable outputs.
    # ------------------------------------------------------------------
    if not args.no_export_segmentation:
        logger.log("-> Exporting final segmentation as NIfTI artifacts…")
        try:
            export = post_python(base_url, EXPORT_SEGMENTATION_SRC, timeout=300)
            if export.get("status") == "ok":
                comp = export.get("composite") or {}
                if comp.get("data_b64"):
                    data = base64.b64decode(comp["data_b64"])
                    rec = logger.write_artifact(
                        f"artifacts/{comp['filename']}", data,
                        kind="composite_labelmap",
                        extra={"sha256_remote": comp.get("sha256")},
                    )
                    if rec["sha256"] != comp.get("sha256"):
                        logger.log("  WARNING: composite sha256 mismatch "
                                   "(local vs remote)")
                    logger.log(f"  composite     : {rec['path']}  "
                               f"{rec['size_bytes']:,} bytes  "
                               f"sha256={rec['sha256'][:16]}…")
                for seg in export.get("per_segment", []):
                    if not seg.get("data_b64"):
                        continue
                    data = base64.b64decode(seg["data_b64"])
                    rec = logger.write_artifact(
                        f"artifacts/per_segment/{seg['filename']}", data,
                        kind="per_segment_labelmap",
                        extra={"sid": seg["sid"], "name": seg["name"],
                               "color": seg.get("color"),
                               "sha256_remote": seg.get("sha256")},
                    )
                    logger.log(f"  segment {seg['sid']:>14s}: "
                               f"{rec['size_bytes']:>9,} bytes  "
                               f"sha256={rec['sha256'][:12]}…")
            else:
                logger.log(f"  export failed: {export}")
                logger.event("export_failed", **export)
        except Exception as e:
            logger.log(f"  export error: {e}")
            logger.event("export_error", error=repr(e))

    # ------------------------------------------------------------------
    # Summary + replay script
    # ------------------------------------------------------------------
    final_voxels = (history[-1]["voxels_after"] if history else 0)
    summary = {
        "run_id": logger.run_id,
        "volume_name": init.get("volume_name"),
        "volume_sha256_voxels": vol_meta.get("sha256_voxels"),
        "shape_kji": init.get("shape_kji"),
        "spacing_mm": vol_meta.get("spacing_mm"),
        "threshold": init.get("threshold"),
        "n_candidates_initial": init.get("n_candidates"),
        "steps": len(history),
        "history": history,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                  for k, v in vars(args).items()},
        "final_voxel_count": final_voxels,
        "final_voxel_fraction": round(final_voxels / max(total_voxels, 1), 4),
        "stop_reason": stop_reason,
    }
    logger.write_artifact(
        "summary.json", json.dumps(summary, indent=2, default=str).encode(),
        kind="summary",
    )
    # Also keep summary.json at the run root (legacy location).
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # Build the replay script. We strip --out-dir (it's relative anyway
    # and the user will pick a fresh directory) and substitute the
    # captured args verbatim.
    replay_cmd = [
        f'python3 "{Path(__file__).relative_to(Path.cwd().resolve()) if Path(__file__).is_relative_to(Path.cwd().resolve()) else Path(__file__).name}"',
    ]
    arg_pairs = [
        ("--volume", args.volume),
        ("--intensity-percentile", args.intensity_percentile),
        ("--max-candidates", args.max_candidates),
        ("--max-steps", args.max_steps),
        ("--min-delta", args.min_delta),
        ("--patience", args.patience),
        ("--max-explosion-frac", args.max_explosion_frac),
    ]
    for flag, val in arg_pairs:
        if val is not None:
            replay_cmd.append(f'{flag} {val}')
    if args.reset_first:
        replay_cmd.append("--reset-first")
    if args.no_new_segment_per_click:
        replay_cmd.append("--no-new-segment-per-click")
    if args.no_screenshots:
        replay_cmd.append("--no-screenshots")
    replay_cmd.append(f'--out-dir runs/replay_{logger.run_id}')
    if args.label:
        replay_cmd.append(f'--label "{args.label}"')

    logger.write_replay(
        command=replay_cmd,
        env_keys=["SLICER_WEBSERVER_URL", "NNI_REMOTE_URL", "OPENAI_API_KEY"],
    )

    logger.log("")
    logger.log(f"DONE. final voxels = {final_voxels:,} / {total_voxels:,} "
               f"({100*final_voxels/max(total_voxels, 1):.2f}%)")
    logger.log(f"      stop reason = {stop_reason}")
    logger.log(f"      summary    -> {args.out_dir / 'summary.json'}")
    logger.log(f"      events     -> {logger.events_path}")
    logger.log(f"      artifacts  -> {logger.artifacts_dir}")
    logger.log(f"      replay     -> {logger.replay_path}")

    logger.finalize(stop_reason=stop_reason, summary=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
