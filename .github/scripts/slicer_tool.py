#!/usr/bin/env python3
"""
Unified SlicerMorph analysis tool for AutoResearchClaw.

Single entry point: given a MorphoSource media ID, downloads the specimen,
runs Layer 1 (3D Slicer morphometrics) and optionally MONAI AI segmentation,
and returns a structured analysis summary that feeds back into the research
agent's memory.

Usage:
    from slicer_tool import analyze_specimen
    result = analyze_specimen("000769445", topic="chameleon optic nerve")

    # With MONAI whole-body CT segmentation:
    result = analyze_specimen("000769445", monai=True, monai_model="whole-body-3mm")

Or CLI:
    python3 slicer_tool.py 000769445 --topic "chameleon optic nerve"
    python3 slicer_tool.py 000769445 --monai --monai-model "whole-body-3mm"
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from _helpers import load_dotenv as _do_load_dotenv, SLICER_BIN, AUTORESEARCHCLAW_HOME, MORPHOSOURCE_API_BASE

log = logging.getLogger("SlicerTool")

DATA_DIR = AUTORESEARCHCLAW_HOME / "specimens"

_do_load_dotenv()


# ---------------------------------------------------------------------------
# Step 1: Download from MorphoSource
# ---------------------------------------------------------------------------


def _download_specimen(media_id: str) -> dict:
    """Download a specimen from MorphoSource. Returns result dict."""
    from morphosource_api_download import download_media

    out_dir = DATA_DIR / f"media_{media_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading media %s to %s", media_id, out_dir)
    result = download_media(media_id, str(out_dir))

    if result.get("success"):
        log.info("Downloaded: %s (%d bytes, %d mesh files)",
                 result.get("downloaded_file", "?"),
                 result.get("file_size", 0),
                 len(result.get("mesh_files", [])))
    else:
        log.warning("Download failed: %s", result.get("error", "unknown"))

    return result


# ---------------------------------------------------------------------------
# Step 2: Run Slicer analysis (Layer 1 deep morphometrics)
# ---------------------------------------------------------------------------

_SLICER_ANALYSIS_TEMPLATE = '''\
"""Auto-generated Layer 1 analysis script for {media_id}."""
import slicer
import vtk
import numpy as np
import json
import os
from collections import defaultdict

OUTPUT_DIR = "{output_dir}"
PLY_PATH = "{ply_path}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("SlicerTool Layer 1: " + PLY_PATH)
success, node = slicer.util.loadModel(PLY_PATH, returnNode=True)
if not success or not node:
    json.dump({{"error": "Failed to load mesh"}}, open(os.path.join(OUTPUT_DIR, "analysis.json"), "w"))
    slicer.app.exit(1)

mesh = node.GetMesh()
bounds = [0]*6
node.GetBounds(bounds)
center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
npts = mesh.GetNumberOfPoints()
ncells = mesh.GetNumberOfCells()
print(f"  {{npts:,}} vertices, {{ncells:,}} faces")

# Sample points
stride = max(1, npts // 100000)
pts = np.array([mesh.GetPoint(i) for i in range(0, npts, stride)])

# Mass properties
mp = vtk.vtkMassProperties()
mp.SetInputData(mesh)
mp.Update()

# Curvature
cf = vtk.vtkCurvatures()
cf.SetInputData(mesh)
cf.SetCurvatureTypeToMean()
cf.Update()
cs = cf.GetOutput().GetPointData().GetScalars()
curv = np.array([cs.GetValue(i) for i in range(0, min(cs.GetNumberOfTuples(), 100000), stride)])
p5, p95 = np.percentile(curv, [5, 95])
clipped = curv[(curv >= p5) & (curv <= p95)]

# Landmarks
landmarks = []
for name, func in [
    ("snout_tip", lambda p: p[:, 1].argmin()),
    ("occipital", lambda p: p[:, 1].argmax()),
    ("dorsal_peak", lambda p: p[:, 2].argmax()),
    ("ventral", lambda p: p[:, 2].argmin()),
    ("left_max", lambda p: p[:, 0].argmin()),
    ("right_max", lambda p: p[:, 0].argmax()),
]:
    idx = func(pts)
    pt = pts[idx]
    landmarks.append({{"name": name, "position": [round(float(c), 2) for c in pt]}})

lm = {{l["name"]: np.array(l["position"]) for l in landmarks}}
distances = {{}}
for dname, p1, p2 in [("skull_length", "snout_tip", "occipital"),
                       ("skull_height", "ventral", "dorsal_peak"),
                       ("skull_width", "left_max", "right_max")]:
    if p1 in lm and p2 in lm:
        distances[dname] = round(float(np.linalg.norm(lm[p1] - lm[p2])), 2)

# Connectivity
conn = vtk.vtkConnectivityFilter()
conn.SetInputData(mesh)
conn.SetExtractionModeToAllRegions()
conn.ColorRegionsOn()
conn.Update()
n_regions = conn.GetNumberOfExtractedRegions()

# Sphericity
sa = mp.GetSurfaceArea()
vol = mp.GetVolume()
eq_r = (3 * vol / (4 * np.pi)) ** (1/3)
sphericity = (4 * np.pi * eq_r**2) / sa if sa > 0 else 0

# Screenshots
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
tw = slicer.app.layoutManager().threeDWidget(0)
tv = tw.threeDView()
rw = tv.renderWindow()
rw.SetOffScreenRendering(1)
rw.SetSize(1200, 900)
renderer = rw.GetRenderers().GetFirstRenderer()
renderer.SetBackground(0.05, 0.05, 0.1)
camera = renderer.GetActiveCamera()

screenshots = []
for vname, pos, vup in [
    ("anterior", (0, -120, 15), (0, 0, 1)),
    ("lateral", (120, 0, 15), (0, 0, 1)),
    ("dorsal", (0, 0, 120), (0, -1, 0)),
    ("oblique", (80, -80, 60), (0, 0, 1)),
]:
    camera.SetPosition(center[0]+pos[0], center[1]+pos[1], center[2]+pos[2])
    camera.SetFocalPoint(*center)
    camera.SetViewUp(*vup)
    renderer.ResetCameraClippingRange()
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetScale(1)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    fpath = os.path.join(OUTPUT_DIR, f"{{vname}}.png")
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fpath)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    screenshots.append(fpath)

result = {{
    "media_id": "{media_id}",
    "vertices": npts,
    "faces": ncells,
    "bounds": bounds,
    "extent_mm": [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]],
    "surface_area_mm2": round(sa, 2),
    "volume_mm3": round(vol, 2),
    "sphericity": round(sphericity, 4),
    "sa_vol_ratio": round(sa/vol, 4) if vol > 0 else 0,
    "mean_curvature": round(float(np.mean(clipped)), 4),
    "curvature_std": round(float(np.std(clipped)), 4),
    "n_regions": n_regions,
    "landmarks": landmarks,
    "distances": distances,
    "screenshots": screenshots,
}}

with open(os.path.join(OUTPUT_DIR, "analysis.json"), "w") as f:
    json.dump(result, f, indent=2)
print("Analysis complete")
slicer.app.exit(0)
'''


def _run_slicer_analysis(media_id: str, ply_path: str) -> dict:
    """Run advanced SlicerMorph analysis in 3D Slicer headlessly."""
    output_dir = DATA_DIR / f"media_{media_id}" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the standalone advanced analysis script
    from _helpers import SCRIPT_DIR
    script_path = SCRIPT_DIR / "slicer_advanced_analysis.py"

    if not script_path.exists():
        log.error("slicer_advanced_analysis.py not found at %s", script_path)
        return {"error": f"Analysis script not found: {script_path}"}

    # Write config file for the Slicer script (Slicer doesn't pass argv)
    config = {"ply_path": str(ply_path), "output_dir": str(output_dir), "media_id": media_id}
    config_path = SCRIPT_DIR / "_slicer_config.json"
    config_path.write_text(json.dumps(config))

    env = os.environ.copy()
    env["SLICER_PLY_PATH"] = str(ply_path)
    env["SLICER_OUTPUT_DIR"] = str(output_dir)
    env["SLICER_MEDIA_ID"] = media_id

    log.info("Running advanced Slicer analysis on %s", ply_path)
    try:
        result = subprocess.run(
            [SLICER_BIN, "--no-splash", "--python-script", str(script_path)],
            capture_output=True, text=True, timeout=180, env=env,
        )
        log.info("Slicer exit code: %d", result.returncode)
        if result.returncode != 0:
            log.warning("Slicer stderr: %s", result.stderr[-500:] if result.stderr else "")
    except subprocess.TimeoutExpired:
        log.error("Slicer timed out")
        return {"error": "Slicer timed out after 180s"}
    except Exception as exc:
        log.error("Slicer failed: %s", exc)
        return {"error": str(exc)}

    analysis_file = output_dir / "analysis.json"
    if analysis_file.exists():
        data = json.loads(analysis_file.read_text())
        log.info("Analysis complete: %d vertices, %s",
                 data.get("vertices", 0), data.get("measurements", {}))
        return data
    return {"error": "No analysis output produced"}


# ---------------------------------------------------------------------------
# Step 2b: Run MONAI Auto3DSeg (AI segmentation for CT volumes)
# ---------------------------------------------------------------------------


def _run_monai_segmentation(media_id: str, input_path: str, model_id: str = "") -> dict:
    """Run MONAI Auto3DSeg inside 3D Slicer headlessly.

    Requires the MONAIAuto3DSeg extension to be installed in Slicer.
    Install once with: Slicer --python-script slicer_install_monai.py

    Args:
        media_id: MorphoSource media identifier
        input_path: Path to CT volume (NIfTI, NRRD, or DICOM directory)
        model_id: MONAI model identifier (default: auto-detect wholebody)

    Returns:
        Dict with segmentation results or {"error": ...}
    """
    output_dir = DATA_DIR / f"media_{media_id}" / "monai_seg"
    output_dir.mkdir(parents=True, exist_ok=True)

    from _helpers import SCRIPT_DIR
    script_path = SCRIPT_DIR / "slicer_monai_seg.py"

    if not script_path.exists():
        log.error("slicer_monai_seg.py not found at %s", script_path)
        return {"error": f"MONAI script not found: {script_path}"}

    config = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "model_id": model_id,
        "force_cpu": "1",
        "media_id": media_id,
    }
    config_path = SCRIPT_DIR / "_monai_config.json"
    config_path.write_text(json.dumps(config))

    env = os.environ.copy()
    env["MONAI_INPUT_PATH"] = str(input_path)
    env["MONAI_OUTPUT_DIR"] = str(output_dir)
    env["MONAI_MODEL_ID"] = model_id
    env["MONAI_FORCE_CPU"] = "1"
    env["MONAI_MEDIA_ID"] = media_id

    log.info("Running MONAI segmentation on %s (model=%s)", input_path, model_id or "auto")
    try:
        result = subprocess.run(
            [SLICER_BIN, "--no-splash", "--no-main-window", "--python-script", str(script_path)],
            capture_output=True, text=True, timeout=1800, env=env,
        )
        log.info("MONAI Slicer exit code: %d", result.returncode)
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                log.info("  MONAI: %s", line)
        if result.returncode != 0:
            log.warning("MONAI stderr: %s", result.stderr[-500:] if result.stderr else "")
    except subprocess.TimeoutExpired:
        log.error("MONAI segmentation timed out (30 min)")
        return {"error": "MONAI timed out after 1800s"}
    except Exception as exc:
        log.error("MONAI failed: %s", exc)
        return {"error": str(exc)}

    seg_file = output_dir / "monai_seg.json"
    if seg_file.exists():
        data = json.loads(seg_file.read_text())
        n_segs = data.get("n_segments", 0)
        n_organs = len(data.get("organ_volumes", {}))
        log.info("MONAI complete: %d segments, %d organs with volumes, %.1fs",
                 n_segs, n_organs, data.get("segmentation_time_s", 0))
        return data
    return {"error": "No MONAI output produced"}


# ---------------------------------------------------------------------------
# Step 3: Generate summary for AutoResearchClaw memory
# ---------------------------------------------------------------------------


def _build_summary(media_id: str, download_result: dict, analysis: dict,
                    monai_result: dict | None = None) -> str:
    """Build a concise summary string for the research agent's memory."""
    if analysis.get("error"):
        return f"Specimen {media_id}: download={'OK' if download_result.get('success') else 'FAILED'}, analysis FAILED ({analysis['error']})"

    m = analysis.get("measurements", {})
    mp = analysis.get("mass_properties", {})
    pca = analysis.get("pca_shape", {})
    curv = analysis.get("curvature", {}).get("mean", {})

    lines = [
        f"## Specimen Analysis: media {media_id}",
        f"**Mesh:** {analysis.get('vertices', 0):,} vertices, {analysis.get('faces', 0):,} faces",
        f"**Dimensions:** L={m.get('total_length', 'N/A')} W={m.get('total_width', 'N/A')} H={m.get('total_height', 'N/A')} mm",
    ]
    if m.get("bizygomatic_width"):
        lines.append(f"**Bizygomatic width:** {m['bizygomatic_width']} mm")
    if m.get("length_width_ratio"):
        lines.append(f"**L/W ratio:** {m['length_width_ratio']} | **H/L ratio:** {m.get('height_length_ratio', 'N/A')}")
    lines.extend([
        f"**Surface area:** {mp.get('surface_area_mm2', 'N/A')} mm^2 | **Volume:** {mp.get('volume_mm3', 'N/A')} mm^3",
        f"**Sphericity:** {mp.get('sphericity', 'N/A')} | **SA/V:** {mp.get('sa_vol_ratio', 'N/A')}",
        f"**Regions:** {analysis.get('n_regions', 'N/A')} | **Asymmetry:** {analysis.get('bilateral_asymmetry_pct', 'N/A')}%",
        f"**PCA:** PC1={pca.get('pc1_pct', '?')}% elongation={pca.get('elongation', '?')}",
        f"**Curvature:** mean={curv.get('mean', 'N/A')} +/- {curv.get('std', 'N/A')}",
        f"**Landmarks:** {len(analysis.get('landmarks', []))} placed | **Screenshots:** {len(analysis.get('screenshots', []))}",
    ])

    # MONAI AI segmentation results
    if monai_result and not monai_result.get("error"):
        lines.append("")
        lines.append(f"### MONAI AI Segmentation ({monai_result.get('model_title', 'N/A')})")
        lines.append(f"**Segments:** {monai_result.get('n_segments', 0)} anatomical structures identified")
        lines.append(f"**Segmentation time:** {monai_result.get('segmentation_time_s', 0):.1f}s")

        organ_vols = monai_result.get("organ_volumes", {})
        if organ_vols:
            lines.append(f"**Organs with volume data:** {len(organ_vols)}")
            top_organs = sorted(organ_vols.items(),
                                key=lambda x: abs(x[1].get("volume_mm3", 0) or 0),
                                reverse=True)[:10]
            for name, stats in top_organs:
                vol = stats.get("volume_cm3") or stats.get("volume_mm3", 0)
                unit = "cm3" if stats.get("volume_cm3") else "mm3"
                hu = stats.get("mean_hu")
                hu_str = f", mean HU={hu:.0f}" if hu is not None else ""
                lines.append(f"  - {name}: {vol:.1f} {unit}{hu_str}")
    elif monai_result and monai_result.get("error"):
        lines.append(f"\n### MONAI: FAILED ({monai_result['error']})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_specimen(media_id: str, topic: str = "", skip_download: bool = False,
                     monai: bool = False, monai_model: str = "",
                     monai_input: str = "") -> dict:
    """Download and analyze a MorphoSource specimen.

    Returns a dict with: success, media_id, summary, analysis, download_result,
    and optionally monai_result.

    Args:
        media_id: MorphoSource media identifier
        topic: Research topic context
        skip_download: If True, skip download (use cached specimen)
        monai: If True, also run MONAI AI segmentation
        monai_model: MONAI model ID (e.g. "whole-body-3mm-v2.0.0").
                     Empty string auto-detects the best model.
        monai_input: Path to CT volume for MONAI. If empty, searches
                     the downloaded specimen directory for NIfTI/NRRD files.
    """
    t0 = time.time()
    log.info("=" * 50)
    log.info("SlicerTool: analyzing media %s (monai=%s)", media_id, monai)
    log.info("=" * 50)

    # Check if Slicer is available
    if not Path(SLICER_BIN).exists():
        return {
            "success": False, "media_id": media_id,
            "error": "3D Slicer not found at " + SLICER_BIN,
            "summary": f"Specimen {media_id}: Slicer not available on this runner",
        }

    # Check cache — reuse previously downloaded + analyzed specimens
    specimen_dir = DATA_DIR / f"media_{media_id}"
    existing_analysis = specimen_dir / "analysis" / "analysis.json"
    monai_result = None

    if existing_analysis.exists() and not monai:
        log.info("CACHE HIT: reusing analysis for %s", media_id)
        analysis = json.loads(existing_analysis.read_text())

        existing_monai = specimen_dir / "monai_seg" / "monai_seg.json"
        if existing_monai.exists():
            monai_result = json.loads(existing_monai.read_text())

        summary = _build_summary(media_id, {"success": True}, analysis, monai_result)
        return {
            "success": True, "media_id": media_id,
            "summary": summary, "analysis": analysis,
            "monai_result": monai_result,
            "download_result": {"success": True, "cached": True},
            "duration_s": round(time.time() - t0, 1),
        }

    # Step 1: Download (not cached)
    log.info("CACHE MISS: downloading %s", media_id)
    download_result = _download_specimen(media_id)
    if not download_result.get("success"):
        summary = f"Specimen {media_id}: download FAILED — {download_result.get('error', 'unknown')}"
        return {
            "success": False, "media_id": media_id,
            "summary": summary, "download_result": download_result,
            "duration_s": round(time.time() - t0, 1),
        }

    # Find mesh file
    mesh_files = download_result.get("mesh_files", [])
    if not mesh_files:
        summary = f"Specimen {media_id}: downloaded but no mesh files found in archive"
        return {
            "success": False, "media_id": media_id,
            "summary": summary, "download_result": download_result,
            "duration_s": round(time.time() - t0, 1),
        }

    ply_path = mesh_files[0]
    log.info("Using mesh: %s", ply_path)

    # Step 2a: Slicer morphometric analysis
    analysis = _run_slicer_analysis(media_id, ply_path)

    # Step 2b: MONAI AI segmentation (if requested)
    if monai:
        ct_path = monai_input
        if not ct_path:
            ct_path = _find_ct_volume(specimen_dir)
        if ct_path:
            log.info("Running MONAI segmentation on %s", ct_path)
            monai_result = _run_monai_segmentation(media_id, ct_path, monai_model)
        else:
            log.warning("No CT volume found for MONAI segmentation")
            monai_result = {"error": "No CT volume (NIfTI/NRRD/DICOM) found in specimen"}

    # Step 3: Build summary
    summary = _build_summary(media_id, download_result, analysis, monai_result)

    duration = round(time.time() - t0, 1)
    log.info("SlicerTool complete in %.1fs", duration)

    return {
        "success": not bool(analysis.get("error")),
        "media_id": media_id,
        "summary": summary,
        "analysis": analysis,
        "monai_result": monai_result,
        "download_result": download_result,
        "duration_s": duration,
    }


def _find_ct_volume(specimen_dir: Path) -> str | None:
    """Search a specimen directory for CT volume files."""
    volume_exts = {".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd", ".dcm"}
    for root, dirs, files in os.walk(specimen_dir):
        for f in files:
            ext = f.lower()
            if any(ext.endswith(e) for e in volume_exts):
                return os.path.join(root, f)
        # Check for DICOM directories (folders with .dcm files or DICOMDIR)
        for d in dirs:
            sub = os.path.join(root, d)
            dcm_files = [x for x in os.listdir(sub)
                         if x.lower().endswith(".dcm") or x == "DICOMDIR"]
            if dcm_files:
                return sub
    return None


def monai_segment(input_path: str, output_dir: str = "", model_id: str = "",
                  media_id: str = "standalone") -> dict:
    """Standalone MONAI segmentation (no MorphoSource download).

    Convenience function for direct use with local CT volumes.

    Args:
        input_path: Path to CT volume (NIfTI, NRRD) or DICOM directory
        output_dir: Output directory (default: next to input file)
        model_id: MONAI model ID (empty = auto-detect)
        media_id: Identifier label for the output

    Returns:
        Dict with segmentation results
    """
    if not Path(SLICER_BIN).exists():
        return {"error": "3D Slicer not found at " + SLICER_BIN}

    if not output_dir:
        output_dir = str(Path(input_path).parent / "monai_seg")

    return _run_monai_segmentation(media_id, input_path, model_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="SlicerMorph + MONAI analysis tool")
    parser.add_argument("media_id", nargs="?", default=os.environ.get("MEDIA_ID", ""),
                        help="MorphoSource media ID")
    parser.add_argument("--topic", default="", help="Research topic context")
    parser.add_argument("--monai", action="store_true",
                        help="Also run MONAI AI segmentation on CT volume")
    parser.add_argument("--monai-model", default="",
                        help="MONAI model ID (e.g. 'whole-body-3mm-v2.0.0')")
    parser.add_argument("--monai-input", default="",
                        help="Path to CT volume for MONAI (auto-detected if omitted)")
    parser.add_argument("--monai-only", default="",
                        help="Run standalone MONAI segmentation on this file (no MorphoSource)")
    args = parser.parse_args()

    if args.monai_only:
        result = monai_segment(args.monai_only, model_id=args.monai_model)
        print("\n" + "=" * 60)
        if result.get("error"):
            print(f"MONAI FAILED: {result['error']}")
        else:
            print(f"MONAI: {result.get('n_segments', 0)} segments, "
                  f"{len(result.get('organ_volumes', {}))} organs")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0 if not result.get("error") else 1)

    if not args.media_id:
        parser.print_help()
        sys.exit(1)

    result = analyze_specimen(args.media_id, args.topic,
                              monai=args.monai,
                              monai_model=args.monai_model,
                              monai_input=args.monai_input)
    print("\n" + "=" * 60)
    print(result.get("summary", "No summary"))
    print("=" * 60)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("analysis", "monai_result")}, indent=2))
