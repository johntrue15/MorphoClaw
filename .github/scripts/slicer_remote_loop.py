#!/usr/bin/env python3
"""
LLM-driven interactive segmentation loop against a remote 3D Slicer.

Architecture (the "auto-research claw")
---------------------------------------

    ┌────────────────────────┐  HTTPS via Exosphere proxy
    │  Local Mac             │     ┌──────────────────────────────┐
    │  - this script         │ ──▶ │ Jetstream2 / MorphoCloud box │
    │  - OpenAI vision call  │     │  - 3D Slicer (Web Server on  │
    │  - prompt log          │ ◀── │    port 2016, exec enabled)  │
    │                        │     │  - SlicerNNInteractive plug. │
    └────────────────────────┘     │  - nninteractive-slicer-     │
                                   │    server on 127.0.0.1:1527  │
                                   └──────────────────────────────┘

Each loop step:

  1. Capture three slice views (red/yellow/green) and the 3D view through
     ``/slicer/slice?orientation=...`` and ``/slicer/threeD``. These
     already include the live segmentation overlay drawn by Slicer.
  2. Compose them into a single 2×2 grid PNG (saved locally per step).
  3. Send the grid to the OpenAI vision model with a goal prompt + a
     compact history of past actions and the current segmentation
     voxel count.
  4. Parse the model's JSON: ``{"action": "point|bbox|done",
     "i": int, "j": int, "k": int, "positive": bool, "rationale": "..."}``.
  5. Apply the action by POSTing Python to ``/slicer/exec`` that calls
     ``plugin.point_prompt(xyz=[i, j, k], positive_click=...)`` (or the
     bbox / done equivalents). The plugin synchronously roundtrips to
     the local FastAPI server on the box and updates the segmentation
     node, so the next screenshot already shows the new mask.
  6. Optionally re-center each slice view on the picked voxel.
  7. Repeat until the LLM says ``done`` or ``--max-steps`` is reached.

Each step writes:

  ``runs/<run_name>/step_<NN>/grid.png``       composed 4-up grid
  ``runs/<run_name>/step_<NN>/red.png``        raw slice view
  ``runs/<run_name>/step_<NN>/yellow.png``
  ``runs/<run_name>/step_<NN>/green.png``
  ``runs/<run_name>/step_<NN>/threeD.png``
  ``runs/<run_name>/step_<NN>/state.json``     metadata + LLM response

Env vars consumed
-----------------
  ``SLICER_WEBSERVER_URL``  (or NNI_REMOTE_URL)  e.g.
      https://http-149-165-152-225-2016.proxy-js2-iu.exosphere.app/
  ``OPENAI_API_KEY``        OpenAI key (required unless --dry-run)
  ``OPENAI_MODEL``          default ``gpt-4o``
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy imports (so --help works even if PIL/openai aren't installed)
# ---------------------------------------------------------------------------

def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        return Image, ImageDraw, ImageFont
    except ImportError:
        sys.exit("ERROR: pillow is required. Install with: pip install pillow")


def _load_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError:
        sys.exit("ERROR: openai is required. Install with: pip install openai>=1.40")


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
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
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"GET {url} -> HTTP {resp.status}")
        return resp.read()


def post_python(base_url: str, source: str, timeout: float = 180) -> dict:
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
    dt = time.time() - t0
    if status != 200:
        raise RuntimeError(f"/slicer/exec -> HTTP {status}: {content[:200]!r}")
    try:
        result = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"non-JSON exec reply: {e}; preview={content[:200]!r}")
    result["_dt_s"] = round(dt, 3)
    return result


# ---------------------------------------------------------------------------
# Server-side Python recipes
# ---------------------------------------------------------------------------

CAPTURE_METADATA_SRC = textwrap.dedent("""
    import slicer
    out = {}
    try:
        lm = slicer.app.layoutManager()
        for color in ("Red", "Yellow", "Green"):
            sw = lm.sliceWidget(color)
            if sw:
                sn = sw.sliceLogic().GetSliceNode()
                out[f"{color.lower()}_offset_mm"] = round(sn.GetSliceOffset(), 3)
                out[f"{color.lower()}_orientation"] = sn.GetOrientationString()
        sel = slicer.app.applicationLogic().GetSelectionNode()
        vol_id = sel.GetActiveVolumeID() if sel else None
        vol = slicer.mrmlScene.GetNodeByID(vol_id) if vol_id else None
        if vol is None:
            vols = list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
            vol = vols[0] if vols else None
        if vol is not None:
            out["volume_name"] = vol.GetName()
            out["volume_id"] = vol.GetID()
            img = vol.GetImageData()
            out["dimensions_ijk"] = list(img.GetDimensions())
            out["spacing_mm"] = [round(s, 4) for s in vol.GetSpacing()]
            bounds = [0.0]*6
            vol.GetRASBounds(bounds)
            out["ras_bounds"] = [round(b, 2) for b in bounds]
            # Range of intensities (helps the LLM understand contrast)
            scalars = img.GetPointData().GetScalars() if img else None
            if scalars is not None:
                rng = scalars.GetRange()
                out["scalar_range"] = [round(rng[0], 2), round(rng[1], 2)]
                out["scalar_type"] = img.GetScalarTypeAsString()
        # Active segmentation voxel count (excluding the do-not-touch helper)
        total = 0
        seg_nodes_info = []
        for sn in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
            name = sn.GetName()
            if "do not touch" in name.lower():
                continue
            seg = sn.GetSegmentation()
            n_segs = seg.GetNumberOfSegments()
            n_voxels = 0
            for i in range(n_segs):
                sid = seg.GetNthSegmentID(i)
                try:
                    arr = slicer.util.arrayFromSegmentBinaryLabelmap(sn, sid)
                    if arr is not None:
                        n_voxels += int((arr > 0).sum())
                except Exception:
                    pass
            seg_nodes_info.append({"name": name, "n_segments": n_segs,
                                   "voxel_count": n_voxels})
            total += n_voxels
        out["segmentation_voxel_count"] = total
        out["segmentation_nodes"] = seg_nodes_info
        # Plugin status
        try:
            mod = slicer.modules.slicernninteractive
            plugin = mod.widgetRepresentation().self()
            out["plugin_server"] = getattr(plugin, "server", None)
            out["plugin_is_positive"] = getattr(plugin, "is_positive", None)
        except Exception:
            pass
    except Exception as e:
        out["error"] = repr(e)
    __execResult.update(out)
""").strip()


# Reset the in-progress segmentation: clears every segmentation node in
# the scene (except the plugin's "do not touch" scribble helper) AND tells
# the FastAPI server to forget all interactions by uploading an all-zero
# mask. This is the same code path the GUI's "Reset segment" button
# triggers, just packaged for /slicer/exec.
RESET_SEGMENTATION_SRC = textwrap.dedent("""
    import slicer, io, gzip, json
    out = {}
    try:
        import numpy as np
        import requests
        mod = slicer.modules.slicernninteractive
        plugin = mod.widgetRepresentation().self()
        vol = plugin.get_volume_node()
        if vol is None:
            __execResult["status"] = "no_active_volume"
        else:
            arr = slicer.util.arrayFromVolume(vol)  # (k, j, i)
            empty = np.zeros(arr.shape, dtype=np.uint8)
            buf = io.BytesIO()
            np.save(buf, empty, allow_pickle=False)
            compressed = gzip.compress(buf.getvalue())
            r = requests.post(
                f"{plugin.server}/upload_segment",
                files={"file": ("seg.npy.gz", io.BytesIO(compressed),
                                "application/octet-stream")},
                timeout=120,
            )
            out["upload_segment_status"] = r.status_code
            out["upload_segment_body"] = r.text[:200]
            cleared = []
            for sn in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
                if "do not touch" in sn.GetName().lower():
                    continue
                seg = sn.GetSegmentation()
                seg.RemoveAllSegments()
                cleared.append(sn.GetName())
            out["cleared_nodes"] = cleared
            try:
                plugin.setup_prompts()
                out["plugin_setup_prompts"] = "ok"
            except Exception as e:
                out["plugin_setup_prompts"] = repr(e)
            out["status"] = "ok"
    except Exception as e:
        out["status"] = "error"
        out["error"] = repr(e)
    __execResult.update(out)
""").strip()


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
""").strip()


# Set Slicer to FourUp + recenter all slice views on the picked IJK voxel.
# IJK is converted to RAS via the volume's IJKToRAS matrix so the slice
# views actually show the picked location.
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
    else:
        __execResult["error"] = "no active volume"
""").strip()


APPLY_POINT_SRC_TEMPLATE = textwrap.dedent("""
    import slicer, time
    i, j, k = {i}, {j}, {k}
    positive = {positive}
    t0 = time.time()
    mod = slicer.modules.slicernninteractive
    plugin = mod.widgetRepresentation().self()
    # is_positive is a read-only property (mirrors a UI button). Use the
    # plugin's button-click handlers to keep the GUI in sync, then call
    # point_prompt with an explicit positive_click — point_prompt honors
    # the arg directly, the toggle is just to keep self.current_prompt_
    # type_positive consistent for any downstream code that reads it.
    if positive:
        plugin.on_prompt_type_positive_clicked()
    else:
        plugin.on_prompt_type_negative_clicked()
    plugin.point_prompt(xyz=[int(i), int(j), int(k)], positive_click=bool(positive))
    __execResult["applied"] = True
    __execResult["xyz_ijk"] = [int(i), int(j), int(k)]
    __execResult["positive"] = bool(positive)
    __execResult["seconds"] = round(time.time() - t0, 3)
""").strip()


APPLY_BBOX_SRC_TEMPLATE = textwrap.dedent("""
    import slicer, time
    p1 = [{i0}, {j0}, {k0}]
    p2 = [{i1}, {j1}, {k1}]
    positive = {positive}
    t0 = time.time()
    mod = slicer.modules.slicernninteractive
    plugin = mod.widgetRepresentation().self()
    if positive:
        plugin.on_prompt_type_positive_clicked()
    else:
        plugin.on_prompt_type_negative_clicked()
    plugin.bbox_prompt(outer_point_one=p1, outer_point_two=p2,
                      positive_click=bool(positive))
    __execResult["applied"] = True
    __execResult["p1_ijk"] = p1
    __execResult["p2_ijk"] = p2
    __execResult["positive"] = bool(positive)
    __execResult["seconds"] = round(time.time() - t0, 3)
""").strip()


# ---------------------------------------------------------------------------
# View capture + grid composition
# ---------------------------------------------------------------------------

def capture_views(base_url: str, step_dir: Path) -> dict[str, Path]:
    """Fetch all 4 view PNGs to disk and return their paths."""
    step_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, Path] = {}
    for orient, name in [("axial", "red"), ("sagittal", "yellow"),
                         ("coronal", "green")]:
        url = f"{base_url}/slicer/slice?orientation={orient}"
        out = step_dir / f"{name}.png"
        out.write_bytes(http_get(url, timeout=15))
        views[name] = out
    out3d = step_dir / "threeD.png"
    out3d.write_bytes(http_get(base_url + "/slicer/threeD", timeout=15))
    views["threeD"] = out3d
    return views


def compose_grid(views: dict[str, Path], out_path: Path,
                 metadata: dict, step_idx: int) -> Path:
    """Glue the 4 PNGs into a 2x2 grid and add small annotations."""
    Image, ImageDraw, ImageFont = _load_pillow()

    imgs = {name: Image.open(p).convert("RGB") for name, p in views.items()}
    # Normalize sizes — use the largest slice view as the per-cell size,
    # with the 3D view scaled to match.
    cell_w = max(imgs[k].width for k in ("red", "yellow", "green"))
    cell_h = max(imgs[k].height for k in ("red", "yellow", "green"))
    canvas = Image.new("RGB", (cell_w * 2, cell_h * 2), (10, 10, 10))
    layout = [("red", 0, 0), ("yellow", 1, 0),
              ("green", 0, 1), ("threeD", 1, 1)]
    for name, cx, cy in layout:
        im = imgs[name].resize((cell_w, cell_h), Image.BILINEAR)
        canvas.paste(im, (cx * cell_w, cy * cell_h))

    # Annotate each cell with view name + slice offset
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = {
        "red":     f"Red (axial, K offset={metadata.get('red_offset_mm')} mm)",
        "yellow":  f"Yellow (sagittal, I offset={metadata.get('yellow_offset_mm')} mm)",
        "green":   f"Green (coronal, J offset={metadata.get('green_offset_mm')} mm)",
        "threeD":  f"3D view  (step {step_idx})",
    }
    for name, cx, cy in layout:
        x, y = cx * cell_w + 6, cy * cell_h + 6
        # Black backdrop for legibility
        draw.rectangle([x - 2, y - 2, x + 380, y + 16],
                       fill=(0, 0, 0))
        draw.text((x, y), labels[name], fill=(255, 255, 255), font=font)

    canvas.save(out_path, format="PNG", optimize=True)
    return out_path


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You guide an interactive 3D segmentation tool (nnInteractive in 3D Slicer)
    via a remote control loop. After each action you take, the tool re-runs
    inference and you see the updated views.

    You see one composite image with four cells:
        top-left      Red    (axial slice)
        top-right     Yellow (sagittal slice)
        bottom-left   Green  (coronal slice)
        bottom-right  3D view of the current segmentation
    Any existing segmentation overlay is rendered live on the slice views.

    Each step you may emit ONE action. Output strict JSON:

        {"action": "point",
         "i": int, "j": int, "k": int,
         "positive": true|false,
         "rationale": "<=120 chars"}

        {"action": "bbox",
         "i0": int, "j0": int, "k0": int,
         "i1": int, "j1": int, "k1": int,
         "positive": true,
         "rationale": "<=120 chars"}

        {"action": "done",
         "rationale": "<=120 chars"}

    Coordinates are integer IJK voxel indices (0-indexed) inside the volume's
    dimension box (provided in the user message). Use 'positive' true to ADD
    to the segmentation, 'positive' false to remove a region you over-selected.
    Use 'done' when the four views show the target structure cleanly
    segmented and no more refinement is necessary.

    Be deliberate: prefer a single well-placed point or bbox over many
    scattered points. If the target isn't visible at the current slices,
    pick (i, j, k) inside the target so the views auto-recenter on it
    next step.
""").strip()


def call_llm(client, model: str, target: str, grid_path: Path,
             metadata: dict, history: list[dict]) -> dict:
    grid_b64 = base64.b64encode(grid_path.read_bytes()).decode("ascii")

    dims = metadata.get("dimensions_ijk")
    user_text = textwrap.dedent(f"""
        Goal: {target}

        Volume: {metadata.get('volume_name')!r}
        IJK dims (i, j, k):  {dims}
        Spacing mm (i, j, k): {metadata.get('spacing_mm')}
        Scalar range:        {metadata.get('scalar_range')}
        Scalar type:         {metadata.get('scalar_type')}

        Current segmentation voxel count: {metadata.get('segmentation_voxel_count')}
        Slice offsets (mm):
            axial    K plane: {metadata.get('red_offset_mm')}
            sagittal I plane: {metadata.get('yellow_offset_mm')}
            coronal  J plane: {metadata.get('green_offset_mm')}

        Action history (most recent last):
        {json.dumps(history[-6:], indent=2)}

        Decide the next action. Reply with strict JSON only.
    """).strip()

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": [
             {"type": "text", "text": user_text},
             {"type": "image_url",
              "image_url": {"url": f"data:image/png;base64,{grid_b64}",
                            "detail": "high"}},
         ]},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        response_format={"type": "json_object"},
        max_tokens=400,
        temperature=0.2,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception as e:
        return {"action": "error", "raw": raw, "error": repr(e)}


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def apply_action(base_url: str, action: dict, dims_ijk: list[int]) -> dict:
    kind = action.get("action", "")
    if kind == "point":
        i = int(action.get("i", 0))
        j = int(action.get("j", 0))
        k = int(action.get("k", 0))
        i = max(0, min(dims_ijk[0] - 1, i))
        j = max(0, min(dims_ijk[1] - 1, j))
        k = max(0, min(dims_ijk[2] - 1, k))
        positive = bool(action.get("positive", True))
        src = APPLY_POINT_SRC_TEMPLATE.format(
            i=i, j=j, k=k, positive=positive,
        )
        result = post_python(base_url, src, timeout=180)
        result["clamped_ijk"] = [i, j, k]
        # Recenter the views on the picked voxel for the next iteration
        try:
            post_python(base_url,
                        RECENTER_SRC_TEMPLATE.format(i=i, j=j, k=k),
                        timeout=15)
        except Exception as e:
            result["recenter_error"] = repr(e)
        return result
    if kind == "bbox":
        i0 = int(action.get("i0", 0)); j0 = int(action.get("j0", 0))
        k0 = int(action.get("k0", 0)); i1 = int(action.get("i1", 0))
        j1 = int(action.get("j1", 0)); k1 = int(action.get("k1", 0))
        positive = bool(action.get("positive", True))
        src = APPLY_BBOX_SRC_TEMPLATE.format(
            i0=i0, j0=j0, k0=k0, i1=i1, j1=j1, k1=k1, positive=positive,
        )
        result = post_python(base_url, src, timeout=240)
        # Recenter on the bbox center
        ic, jc, kc = (i0 + i1) // 2, (j0 + j1) // 2, (k0 + k1) // 2
        try:
            post_python(base_url,
                        RECENTER_SRC_TEMPLATE.format(i=ic, j=jc, k=kc),
                        timeout=15)
        except Exception:
            pass
        return result
    return {"applied": False, "kind": kind}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target", required=True,
                   help="Free-text description of what to segment "
                        "(e.g. 'the cranial cavity in the IMPC sample')")
    p.add_argument("--volume",
                   help="Name of an MRMLScalarVolumeNode to make active "
                        "before the loop (defaults to whatever is active)")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--reset-first", action="store_true",
                   help="clear any existing segmentation on the active "
                        "volume before the first LLM step (server-side + "
                        "Slicer scene)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("runs") / time.strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    p.add_argument("--dry-run", action="store_true",
                   help="capture views + compose grids but skip the LLM call "
                        "and skip applying any action (useful to just see "
                        "what we would send)")
    args = p.parse_args(argv)

    base_url = _read_url()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Slicer remote LLM segmentation loop ===")
    print(f"target       : {args.target}")
    print(f"server       : {base_url}")
    print(f"out          : {args.out_dir}")
    print(f"model        : {args.model}")
    print()

    # Optional: switch active volume up front
    if args.volume:
        print(f"-> Setting active volume to {args.volume!r}")
        r = post_python(
            base_url,
            SET_ACTIVE_VOLUME_SRC_TEMPLATE.format(target_name=args.volume),
            timeout=20,
        )
        if r.get("status") != "ok":
            print(f"   FAILED: {r}")
            return 2
        print(f"   active volume id: {r.get('volume_id')}")

    if args.reset_first:
        print("-> Resetting any existing segmentation (server + scene)…")
        r = post_python(base_url, RESET_SEGMENTATION_SRC, timeout=120)
        if r.get("status") != "ok":
            print(f"   reset returned: {r}")
        else:
            print(f"   cleared nodes: {r.get('cleared_nodes')}  "
                  f"upload_segment={r.get('upload_segment_status')}")

    client = None
    if not args.dry_run:
        OpenAI = _load_openai()
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "").strip() or None
        )

    history: list[dict] = []

    for step in range(args.max_steps):
        step_dir = args.out_dir / f"step_{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- Step {step:02d} -----------------------------------------")

        meta = post_python(base_url, CAPTURE_METADATA_SRC, timeout=30)
        print(f"  vox_count={meta.get('segmentation_voxel_count')}  "
              f"vol={meta.get('volume_name')}  "
              f"dims={meta.get('dimensions_ijk')}  "
              f"slice_offsets={meta.get('red_offset_mm')}/"
              f"{meta.get('yellow_offset_mm')}/{meta.get('green_offset_mm')} mm")

        views = capture_views(base_url, step_dir)
        grid = compose_grid(views, step_dir / "grid.png", meta, step)
        print(f"  saved views   : {step_dir}")

        if args.dry_run:
            (step_dir / "state.json").write_text(json.dumps(
                {"metadata": meta, "history": history},
                indent=2, default=str))
            print("  (dry-run; skipping LLM)")
            continue

        action = call_llm(client, args.model, args.target, grid, meta, history)
        print(f"  LLM action    : {json.dumps(action)[:240]}")

        # Persist this step's full state for the writeup
        (step_dir / "state.json").write_text(json.dumps({
            "step": step,
            "metadata": meta,
            "llm_action": action,
        }, indent=2, default=str))

        if action.get("action") == "done":
            print("  LLM said done. Stopping.")
            history.append({"step": step, "action": action})
            break
        if action.get("action") == "error":
            print(f"  LLM returned error: {action}")
            history.append({"step": step, "action": action})
            break

        result = apply_action(base_url, action,
                              dims_ijk=meta.get("dimensions_ijk", [10**6]*3))
        print(f"  applied       : voxel_step={result.get('seconds')}s  "
              f"clamped_ijk={result.get('clamped_ijk')}")
        history.append({
            "step": step, "action": action,
            "result_summary": {
                "applied": result.get("applied"),
                "seconds": result.get("seconds"),
                "clamped_ijk": result.get("clamped_ijk"),
                "recenter_ras": None,
            },
        })
        # Tiny pause to let Slicer redraw before the next screenshot
        time.sleep(1.0)

    # Final summary frame
    final_dir = args.out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    meta = post_python(base_url, CAPTURE_METADATA_SRC, timeout=20)
    views = capture_views(base_url, final_dir)
    compose_grid(views, final_dir / "grid.png", meta, len(history))
    (final_dir / "state.json").write_text(json.dumps(
        {"metadata": meta, "history": history}, indent=2, default=str))

    print()
    print(f"DONE. final voxel count = {meta.get('segmentation_voxel_count')}")
    print(f"      see {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
