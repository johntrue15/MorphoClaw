"""
End-to-end test of the nnInteractive paint loop against a MorphoSource
ground-truth segmentation.

Pipeline
--------
1. Resolve a (CT, GT mesh) pair — either from explicit ``--ct-media-id`` /
   ``--gt-media-id`` arguments, or auto-discovered with
   :mod:`find_segmentation_pairs`.
2. Download both via :mod:`morphosource_api_download` (only "open" media
   succeed; restricted media fail fast).
3. Identify the CT volume file (NIfTI/NRRD/DICOM) and the GT mesh file
   (PLY/STL/OBJ) inside the downloaded archives.
4. Voxelize the GT mesh onto the CT's voxel grid via the headless
   ``voxelize_mesh_in_slicer.py`` (3D Slicer subprocess).
5. Run the LLM-driven nnInteractive paint loop on the CT volume in the
   nnInteractive venv (``$NNINTERACTIVE_HOME/bin/python``).
6. Compare the prediction labelmap to the voxelized GT labelmap with
   :mod:`segmentation_metrics` (Dice/IoU/Hausdorff/volume agreement),
   render an overlay panel, and write a Markdown report.

Outputs land in ``$OUTPUT_DIR/<ct_id>__vs__<gt_id>/`` and include:

    download/                     # raw MorphoSource downloads
    gt_voxelized.nii.gz           # GT mesh rasterized onto the CT grid
    nninteractive/                # paint-loop labelmap, screenshots, report
    metrics.json                  # full metrics payload
    overlay.png                   # 3×3 panel: volume / GT / prediction
    report.md                     # human-readable summary

Usage::

    python nninteractive_compare.py \\
        --ct-media-id 000656244 \\
        --gt-media-id 000656245 \\
        --goal "Segment the cranial bone" \\
        --output-dir /tmp/nni_compare

    # Or auto-discover the first viable pair:
    python nninteractive_compare.py --auto-discover \\
        --query "primate skull mesh" --goal "Segment the cranial bone"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _helpers import load_dotenv, AUTORESEARCHCLAW_HOME, SLICER_BIN  # noqa: E402

log = logging.getLogger("nni_compare")
load_dotenv()

NNI_HOME = Path(os.environ.get(
    "NNINTERACTIVE_HOME", str(AUTORESEARCHCLAW_HOME / "nninteractive")
))
NNI_PYTHON = NNI_HOME / "bin" / "python"

VOLUME_EXTS = {".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd"}
MESH_EXTS = {".ply", ".stl", ".obj", ".off", ".gltf", ".glb"}
TIFF_EXTS = {".tif", ".tiff"}
MIN_TIFF_STACK_SIZE = 10  # treat a dir with >= N tiffs as a CT z-stack


@dataclass
class FilePick:
    path: Path
    size: int

    @property
    def display(self) -> str:
        return f"{self.path.name} ({self.size:,} bytes)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_files(root: Path, extensions: set[str]) -> list[FilePick]:
    matches: list[FilePick] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # Multi-extension support (".nii.gz")
        for ext in extensions:
            if p.name.lower().endswith(ext):
                matches.append(FilePick(path=p, size=p.stat().st_size))
                break
    matches.sort(key=lambda fp: fp.size, reverse=True)
    return matches


def _find_ct_volume(directory: Path) -> Optional[FilePick]:
    """Pick a CT input from a downloaded MorphoSource bundle.

    Recognized formats, in priority order:
        1. Pre-converted volumes: NIfTI / NRRD / MHA / NHDR.
        2. DICOM series directory (any subdir with .dcm files).
        3. TIFF z-stack directory (any subdir with >= 10 sequential .tif files).
           Common for paleontology micro-CT exports.
    """
    matches = _walk_files(directory, VOLUME_EXTS)
    if matches:
        return matches[0]

    best_dicom: Optional[FilePick] = None
    best_tiff: Optional[FilePick] = None

    for sub in directory.rglob("*"):
        if not sub.is_dir():
            continue
        try:
            entries = list(sub.iterdir())
        except (OSError, PermissionError):
            continue
        dcm_files = [e for e in entries if e.is_file() and (
            e.suffix.lower() == ".dcm" or e.name.upper() == "DICOMDIR"
        )]
        if dcm_files:
            total = sum(e.stat().st_size for e in dcm_files)
            if best_dicom is None or total > best_dicom.size:
                best_dicom = FilePick(path=sub, size=total)
            continue

        tif_files = [e for e in entries
                     if e.is_file() and e.suffix.lower() in TIFF_EXTS]
        if len(tif_files) >= MIN_TIFF_STACK_SIZE:
            total = sum(e.stat().st_size for e in tif_files)
            if best_tiff is None or total > best_tiff.size:
                best_tiff = FilePick(path=sub, size=total)

    if best_dicom is not None:
        return best_dicom
    return best_tiff


def _ct_input_kind(path: Path) -> str:
    """Return 'volume' (file), 'dicom', or 'tiff' for a CT input path."""
    if path.is_file():
        return "volume"
    try:
        entries = list(path.iterdir())
    except (OSError, PermissionError):
        return "volume"
    if any(e.is_file() and (e.suffix.lower() == ".dcm" or
                            e.name.upper() == "DICOMDIR")
           for e in entries):
        return "dicom"
    if sum(1 for e in entries
           if e.is_file() and e.suffix.lower() in TIFF_EXTS) \
            >= MIN_TIFF_STACK_SIZE:
        return "tiff"
    return "volume"


def _find_mesh(directory: Path) -> Optional[FilePick]:
    matches = _walk_files(directory, MESH_EXTS)
    return matches[0] if matches else None


def _download(media_id: str, dest: Path) -> dict:
    """Fetch a MorphoSource media bundle, skipping the network if already
    cached. A cache hit requires both the original .zip and at least one
    sibling extracted directory (the marker that ``extract_archives`` was
    able to unpack the bundle)."""
    dest.mkdir(parents=True, exist_ok=True)
    cached_zips = list(dest.glob("morphosource_media-id-*.zip"))
    cached_extracted = [p for p in cached_zips
                        if (p.parent / p.stem).is_dir()
                        and any((p.parent / p.stem).iterdir())]
    if cached_extracted:
        log.info("Cache hit for media %s — using existing %s",
                 media_id, dest)
        return {
            "success": True,
            "media_id": media_id,
            "visibility": "open",
            "downloaded_file": str(cached_extracted[0]),
            "file_size": cached_extracted[0].stat().st_size,
            "download_dir": str(dest),
            "from_cache": True,
        }

    from morphosource_api_download import download_media
    log.info("Downloading media %s → %s", media_id, dest)
    return download_media(media_id, str(dest))


def _tiff_stack_to_nifti(tiff_dir: Path, output: Path,
                         media_id: str = "",
                         center_origin: bool = True) -> dict:
    """Convert a TIFF z-stack directory into a single .nii.gz.

    By default uses ``--center-origin`` since MorphoSource TIFF stacks are
    typically generated by tools that center the volume on (0,0,0). If a
    specimen's GT mesh was exported in a different frame, set
    ``center_origin=False`` and pass an explicit origin.
    """
    if not NNI_PYTHON.exists():
        return {"error": f"nnInteractive venv missing at {NNI_PYTHON}"}
    cmd = [
        str(NNI_PYTHON),
        str(SCRIPT_DIR / "tiff_stack_to_nifti.py"),
        "--input-dir", str(tiff_dir),
        "--output", str(output),
    ]
    if center_origin:
        cmd.append("--center-origin")
    if media_id:
        cmd += ["--media-id", media_id]
    log.info("Converting TIFF stack to NIfTI: %s", tiff_dir)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"error": "TIFF->NIfTI conversion timed out"}
    if proc.stdout:
        for line in proc.stdout.strip().split("\n")[-15:]:
            log.info("  tiffstack: %s", line)
    if proc.returncode != 0:
        return {"error": f"TIFF conversion exit {proc.returncode}",
                "stderr_tail": (proc.stderr or "")[-400:]}
    summary_path = output.with_suffix("").with_suffix(".tiffstack.json")
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass
    return {"output_path": str(output)} if output.exists() \
        else {"error": "TIFF conversion produced no output"}


def _voxelize_slicer(reference_volume: Path, mesh: Path, output: Path) -> dict:
    """Invoke Slicer headlessly to voxelize *mesh* onto *reference_volume*'s grid."""
    if not Path(SLICER_BIN).exists():
        return {"error": f"3D Slicer not found at {SLICER_BIN}"}

    config = {
        "reference_volume": str(reference_volume),
        "mesh_path": str(mesh),
        "output_path": str(output),
        "fill_value": 1,
    }
    config_path = SCRIPT_DIR / "_voxelize_config.json"
    config_path.write_text(json.dumps(config))

    env = os.environ.copy()
    env["VOXELIZE_REFERENCE_VOLUME"] = str(reference_volume)
    env["VOXELIZE_MESH_PATH"] = str(mesh)
    env["VOXELIZE_OUTPUT_PATH"] = str(output)
    env["VOXELIZE_FILL_VALUE"] = "1"

    cmd = [
        SLICER_BIN, "--no-splash", "--no-main-window",
        "--python-script", str(SCRIPT_DIR / "voxelize_mesh_in_slicer.py"),
    ]
    log.info("Voxelizing GT mesh in Slicer (this may take 1-5 minutes)…")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=900, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "Slicer voxelization timed out (15 min)"}
    except Exception as exc:
        return {"error": f"Slicer subprocess failed: {exc}"}

    log.info("Slicer voxelize exit code: %d", proc.returncode)
    if proc.stdout:
        for line in proc.stdout.strip().split("\n")[-15:]:
            log.info("  voxelize: %s", line)
    if proc.returncode != 0 and proc.stderr:
        log.warning("voxelize stderr: %s", proc.stderr[-500:])

    if not output.exists():
        return {
            "error": "Voxelization produced no output",
            "stdout_tail": (proc.stdout or "")[-400:],
            "stderr_tail": (proc.stderr or "")[-400:],
        }

    summary_path = output.with_suffix("").with_suffix(".voxelize.json")
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass
    return {"output_path": str(output), "size": output.stat().st_size}


def _voxelize_vtk(reference_volume: Path, mesh: Path, output: Path) -> dict:
    """Voxelize *mesh* onto *reference_volume*'s grid using pure-Python VTK.

    Runs inside the nnInteractive venv (which has SimpleITK + VTK installed
    by ``install_nninteractive.sh``). Avoids 3D Slicer entirely, so it works
    even when the runner is in a non-Aqua bootstrap.
    """
    if not NNI_PYTHON.exists():
        return {
            "error": (
                f"nnInteractive venv not found at {NNI_PYTHON}. "
                "Bootstrap with `.github/scripts/install_nninteractive.sh`."
            ),
        }

    cmd = [
        str(NNI_PYTHON),
        str(SCRIPT_DIR / "voxelize_mesh_vtk.py"),
        "--reference-volume", str(reference_volume),
        "--mesh", str(mesh),
        "--output", str(output),
        "--fill-value", "1",
    ]
    log.info("Voxelizing GT mesh with pure-Python VTK (no Slicer)…")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"error": "VTK voxelization timed out (15 min)"}
    except Exception as exc:
        return {"error": f"VTK voxelize subprocess failed: {exc}"}

    if proc.stdout:
        for line in proc.stdout.strip().split("\n")[-15:]:
            log.info("  voxelize-vtk: %s", line)
    if proc.returncode != 0:
        return {
            "error": f"VTK voxelization returned {proc.returncode}",
            "stderr_tail": (proc.stderr or "")[-400:],
            "stdout_tail": (proc.stdout or "")[-400:],
        }
    if not output.exists():
        return {"error": "VTK voxelization produced no output"}

    summary_path = output.with_suffix("").with_suffix(".voxelize.json")
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass
    return {"output_path": str(output), "size": output.stat().st_size}


def _voxelize(reference_volume: Path, mesh: Path, output: Path,
              backend: str = "auto") -> dict:
    """Dispatch to either the Slicer-based or pure-Python voxelizer.

    backend: "auto" (try Slicer first, fall back to vtk), "slicer", or "vtk".
    """
    backend = (backend or "auto").lower()
    if backend == "slicer":
        return _voxelize_slicer(reference_volume, mesh, output)
    if backend == "vtk":
        return _voxelize_vtk(reference_volume, mesh, output)
    # auto: prefer Slicer if available, otherwise VTK
    if Path(SLICER_BIN).exists():
        log.info("Voxelize backend=auto → trying Slicer first")
        result = _voxelize_slicer(reference_volume, mesh, output)
        if "error" not in result and output.exists():
            return result
        log.warning("Slicer voxelization failed (%s) — falling back to VTK",
                    result.get("error", "unknown"))
    return _voxelize_vtk(reference_volume, mesh, output)


def _dicom_to_nifti(dicom_dir: Path, output: Path) -> dict:
    """Convert a DICOM series directory into a single .nii.gz."""
    if not NNI_PYTHON.exists():
        return {"error": f"nnInteractive venv missing at {NNI_PYTHON}"}
    cmd = [
        str(NNI_PYTHON),
        str(SCRIPT_DIR / "dicom_to_nifti.py"),
        "--input-dir", str(dicom_dir),
        "--output", str(output),
    ]
    log.info("Converting DICOM series to NIfTI: %s", dicom_dir)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"error": "DICOM->NIfTI conversion timed out"}
    if proc.returncode != 0:
        return {"error": f"DICOM conversion exit {proc.returncode}",
                "stderr_tail": (proc.stderr or "")[-400:]}
    if not output.exists():
        return {"error": "DICOM conversion produced no output"}
    return {"output_path": str(output), "size": output.stat().st_size}


def _crop_volume(reference_volume: Path, mesh: Path,
                 output: Path, margin_mm: float) -> dict:
    """Crop a volume to mesh bbox + margin (mm) using SimpleITK."""
    if not NNI_PYTHON.exists():
        return {"error": f"nnInteractive venv missing at {NNI_PYTHON}"}
    cmd = [
        str(NNI_PYTHON),
        str(SCRIPT_DIR / "crop_around_mesh.py"),
        "--reference-volume", str(reference_volume),
        "--mesh", str(mesh),
        "--output", str(output),
        "--margin-mm", str(margin_mm),
    ]
    log.info("Cropping volume around mesh bbox + %.1fmm margin", margin_mm)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"error": "Crop timed out"}
    if proc.returncode != 0:
        return {"error": f"Crop exit {proc.returncode}",
                "stderr_tail": (proc.stderr or "")[-400:]}
    summary_path = output.with_suffix("").with_suffix(".crop.json")
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass
    return {"output_path": str(output)} if output.exists() \
        else {"error": "Crop produced no output"}


def _run_paint_loop(input_volume: Path, goal: str, output_dir: Path,
                     media_id: str, max_steps: int) -> dict:
    """Run nninteractive_loop.py.

    Backend selection:
      * If ``NNI_REMOTE_WS`` is set, the loop talks to a remote
        nnInteractive WebSocket server (see ``nni_ws_server.py``). In
        that case we run the loop with the *current* Python interpreter
        because it doesn't need a local nnInteractive/torch install —
        it only needs websocket-client + SimpleITK + matplotlib + openai.
      * Otherwise, the loop runs in the dedicated nnInteractive venv
        (``$NNINTERACTIVE_HOME/bin/python``) which has the full backend.
    """
    remote_url = os.environ.get("NNI_REMOTE_WS", "").strip()
    if remote_url:
        # Remote backend: use the parent Python (the same one that's
        # invoking this script). All the deps the loop needs in remote
        # mode (websocket-client, SimpleITK, matplotlib, openai) are part
        # of the AutoResearchClaw env.
        loop_python = sys.executable
        log.info("Using remote nnInteractive backend at %s "
                 "(loop python: %s)", remote_url, loop_python)
    else:
        if not NNI_PYTHON.exists():
            return {
                "error": (
                    f"nnInteractive venv not found at {NNI_PYTHON}. "
                    "Bootstrap it once with "
                    "`.github/scripts/install_nninteractive.sh`, or set "
                    "NNI_REMOTE_WS=ws://… to use a remote server."
                ),
            }
        loop_python = str(NNI_PYTHON)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        loop_python,
        str(SCRIPT_DIR / "nninteractive_loop.py"),
        "--input", str(input_volume),
        "--goal", goal,
        "--media-id", media_id,
        "--output-dir", str(output_dir),
        "--max-steps", str(max_steps),
    ]
    env = os.environ.copy()
    env.setdefault("NNINTERACTIVE_HOME", str(NNI_HOME))
    # nnInteractive's nnU-Net backbone hits ops that aren't implemented
    # on Apple Silicon's MPS yet (e.g. aten::avg_pool3d.out). Without this
    # fallback every prompt step errors and the prediction stays empty.
    # CPU is slower but correct; CUDA / Linux ignore this var.
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # On a 16 GB Mac mini, full-thread CPU inference blows past RAM and the
    # OS kills the worker (SIGKILL / exit -9). Even 4 threads triggers OOM
    # on a 119^3 volume. Default to 1 thread; users with more RAM can
    # override any of these env vars before invoking the comparison.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NNINTERACTIVE_TORCH_THREADS"):
        env.setdefault(var, "1")

    log.info("Running nnInteractive paint loop (goal=%s, max_steps=%d)",
             goal, max_steps)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=3600, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "Paint loop timed out (1h)"}
    except Exception as exc:
        return {"error": f"Paint-loop subprocess failed: {exc}"}

    log.info("Paint loop exit code: %d", proc.returncode)
    if proc.stdout:
        for line in proc.stdout.strip().split("\n")[-10:]:
            log.info("  loop: %s", line)
    if proc.returncode != 0 and proc.stderr:
        log.warning("loop stderr: %s", proc.stderr[-500:])

    summary_file = output_dir / f"{media_id}_nni_summary.json"
    if summary_file.exists():
        try:
            return json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            pass
    return {"error": "No nnInteractive summary produced",
            "stdout_tail": (proc.stdout or "")[-400:]}


def _compute_metrics(prediction: Path, ground_truth: Path,
                     volume: Path, overlay_path: Path,
                     metrics_path: Path) -> dict:
    """Run segmentation_metrics.

    Uses the nnInteractive venv when available (it always has SimpleITK),
    otherwise falls back to the current Python — which works in remote
    backend mode where the local box doesn't need nnInteractive/torch but
    does need SimpleITK and matplotlib (already installed in the parent
    AutoResearchClaw env).
    """
    metrics_python = str(NNI_PYTHON) if NNI_PYTHON.exists() else sys.executable
    cmd = [
        metrics_python,
        str(SCRIPT_DIR / "segmentation_metrics.py"),
        "--pred", str(prediction),
        "--gt", str(ground_truth),
        "--volume", str(volume),
        "--output", str(metrics_path),
        "--overlay", str(overlay_path),
    ]
    log.info("Computing comparison metrics…")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=900)
    except Exception as exc:
        return {"error": f"metrics subprocess failed: {exc}"}

    if proc.stdout:
        for line in proc.stdout.strip().split("\n"):
            log.info("  metrics: %s", line)
    if proc.returncode != 0:
        return {"error": f"metrics returned {proc.returncode}",
                "stderr_tail": (proc.stderr or "")[-400:]}
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            pass
    return {"error": "metrics file missing"}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_comparison(ct_media_id: str, gt_media_id: str, goal: str,
                   output_dir: Path, max_steps: int = 12,
                   voxelize_backend: str = "auto",
                   crop_around_mesh_mm: float = 0.0,
                   skip_paint_loop: bool = False) -> dict:
    """End-to-end comparison.

    voxelize_backend  "auto" | "slicer" | "vtk"   (auto = Slicer first, VTK fallback)
    crop_around_mesh_mm  > 0 to crop CT to the GT mesh bbox + margin (mm)
                         before running the paint loop. 0 disables cropping.
    skip_paint_loop     stop after voxelization + alignment report. Useful as
                        a dry run to verify coordinate alignment of the GT mesh
                        against the CT volume *before* spending OpenAI quota
                        on the iterative paint loop.
    """
    t0 = time.time()
    pair_dir = output_dir / f"{ct_media_id}__vs__{gt_media_id}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    download_root = pair_dir / "download"

    # ---- 1. Download CT + GT mesh ----
    ct_dl = _download(ct_media_id, download_root / f"ct_{ct_media_id}")
    if not ct_dl.get("success"):
        return {"success": False, "stage": "download_ct", "result": ct_dl}

    gt_dl = _download(gt_media_id, download_root / f"gt_{gt_media_id}")
    if not gt_dl.get("success"):
        return {"success": False, "stage": "download_gt", "result": gt_dl}

    # ---- 2. Locate input files ----
    ct_dir = Path(ct_dl["download_dir"])
    gt_dir = Path(gt_dl["download_dir"])

    ct_pick = _find_ct_volume(ct_dir)
    if ct_pick is None:
        return {
            "success": False, "stage": "locate_ct",
            "error": f"No CT volume (NIfTI/NRRD/DICOM) under {ct_dir}",
        }
    log.info("Selected CT volume: %s", ct_pick.display)

    mesh_pick = _find_mesh(gt_dir)
    if mesh_pick is None:
        return {
            "success": False, "stage": "locate_mesh",
            "error": f"No mesh (PLY/STL/OBJ) under {gt_dir}",
        }
    log.info("Selected GT mesh: %s", mesh_pick.display)

    # ---- 2b. Normalize the CT input to a single NIfTI on disk ----
    ct_path = ct_pick.path
    kind = _ct_input_kind(ct_path)
    if kind == "dicom":
        log.info("CT is a DICOM series — converting to NIfTI")
        nifti_path = pair_dir / f"ct_{ct_media_id}.nii.gz"
        dicom_result = _dicom_to_nifti(ct_path, nifti_path)
        if "error" in dicom_result:
            return {"success": False, "stage": "dicom_to_nifti",
                    "result": dicom_result}
        ct_path = nifti_path
    elif kind == "tiff":
        log.info("CT is a TIFF z-stack — converting to NIfTI "
                 "(spacing from MorphoSource API)")
        nifti_path = pair_dir / f"ct_{ct_media_id}.nii.gz"
        tiff_result = _tiff_stack_to_nifti(ct_path, nifti_path,
                                           media_id=ct_media_id)
        if "error" in tiff_result:
            return {"success": False, "stage": "tiff_to_nifti",
                    "result": tiff_result}
        ct_path = nifti_path
        log.info("TIFF→NIfTI: size=%s spacing=%s source=%s",
                 tiff_result.get("size"),
                 tiff_result.get("spacing"),
                 tiff_result.get("spacing_source"))

    # ---- 2c. Optional: crop to GT mesh bbox + margin (faster + tractable) ----
    cropped_ct = ct_path
    cropped_summary = None
    if crop_around_mesh_mm and crop_around_mesh_mm > 0:
        cropped_ct = pair_dir / f"ct_{ct_media_id}_cropped.nii.gz"
        cropped_summary = _crop_volume(
            reference_volume=ct_path, mesh=mesh_pick.path,
            output=cropped_ct, margin_mm=crop_around_mesh_mm,
        )
        if "error" in cropped_summary:
            return {"success": False, "stage": "crop", "result": cropped_summary}
        log.info("Cropped CT: %s -> size %s",
                 cropped_ct, cropped_summary.get("crop_size"))

    # ---- 3. Voxelize GT mesh onto (cropped) CT grid ----
    gt_labelmap = pair_dir / "gt_voxelized.nii.gz"
    voxelize_result = _voxelize(cropped_ct, mesh_pick.path, gt_labelmap,
                                backend=voxelize_backend)
    if "error" in voxelize_result:
        return {"success": False, "stage": "voxelize", "result": voxelize_result}

    # ---- 3b. Optional dry-run: stop here, write alignment report ----
    if skip_paint_loop:
        align_report_path = pair_dir / "alignment_report.md"
        align_report_path.write_text(_render_alignment_report(
            ct_media_id, gt_media_id, ct_pick, mesh_pick,
            cropped_summary, voxelize_result, ct_path, cropped_ct,
            gt_labelmap, pair_dir,
        ))
        return {
            "success": True,
            "stage": "voxelize_only",
            "ct_media_id": ct_media_id,
            "gt_media_id": gt_media_id,
            "ct_path_used": str(cropped_ct),
            "mesh_path": str(mesh_pick.path),
            "gt_labelmap": str(gt_labelmap),
            "voxelize_backend": voxelize_result.get("backend",
                                                    voxelize_backend),
            "foreground_voxels": voxelize_result.get("foreground_voxels"),
            "foreground_volume_mm3": voxelize_result.get("foreground_volume_mm3"),
            "crop_summary": cropped_summary,
            "voxelize_summary": voxelize_result,
            "alignment_report": str(align_report_path),
            "duration_s": round(time.time() - t0, 1),
            "skipped_paint_loop": True,
        }

    # ---- 4. Run paint loop on the (cropped) CT volume ----
    nni_dir = pair_dir / "nninteractive"
    nni_result = _run_paint_loop(cropped_ct, goal, nni_dir, ct_media_id,
                                 max_steps)
    if "error" in nni_result:
        return {"success": False, "stage": "paint_loop", "result": nni_result}
    pred_labelmap = Path(nni_result.get("labelmap_path", ""))
    if not pred_labelmap.exists():
        return {"success": False, "stage": "paint_loop",
                "error": f"prediction labelmap missing: {pred_labelmap}"}

    # ---- 5. Compute metrics + render overlay ----
    metrics = _compute_metrics(
        prediction=pred_labelmap,
        ground_truth=gt_labelmap,
        volume=cropped_ct,
        overlay_path=pair_dir / "overlay.png",
        metrics_path=pair_dir / "metrics.json",
    )
    if "error" in metrics:
        return {"success": False, "stage": "metrics", "result": metrics}

    # ---- 6. Markdown report ----
    report_path = pair_dir / "report.md"
    report_path.write_text(_render_report(
        ct_media_id, gt_media_id, goal, ct_pick, mesh_pick, voxelize_result,
        nni_result, metrics, pair_dir, max_steps,
    ))

    return {
        "success": True,
        "ct_media_id": ct_media_id,
        "gt_media_id": gt_media_id,
        "goal": goal,
        "ct_path": str(ct_path),
        "ct_path_used": str(cropped_ct),
        "mesh_path": str(mesh_pick.path),
        "gt_labelmap": str(gt_labelmap),
        "prediction_labelmap": str(pred_labelmap),
        "metrics_path": str(pair_dir / "metrics.json"),
        "overlay_path": str(pair_dir / "overlay.png"),
        "report_path": str(report_path),
        "voxelize_backend": voxelize_result.get("backend",
                                                voxelize_backend),
        "crop_summary": cropped_summary,
        "metrics": metrics,
        "duration_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Presets — quick way to run a known-good test pair
# ---------------------------------------------------------------------------

PRESETS = {
    # Veiled chameleon (Chamaeleo calyptratus), uf:herp:191369
    # — all 7 media items are open-download.
    # The Right Stapes is one of the smallest bones in the body
    # (~3 mm), so this is the fastest possible end-to-end test.
    "chameleon_stapes": {
        "ct_media_id": "000408242",   # Head CT (smallest CT, 1.35 GB)
        "gt_media_id": "000790324",   # Right Stapes (587 KB PLY)
        "goal": "Segment the right stapes (small middle-ear bone). "
                "It is a tiny irregular bone in the inner ear region.",
        "max_steps": 6,
        # The stapes is roughly 3 mm wide; a 1.5 mm margin keeps the cropped
        # CT around 120^3 voxels (~1.7M voxels) which fits in ~16 GB RAM
        # during nnInteractive CPU inference on a Mac mini. Larger margins
        # (e.g. 4.0) blow past 32 GB peak and trigger OS OOM-kills.
        "crop_around_mesh_mm": 1.5,
        "voxelize_backend": "vtk",
    },
}


# ---------------------------------------------------------------------------


def _render_alignment_report(ct_id: str, gt_id: str,
                             ct_pick: FilePick, mesh_pick: FilePick,
                             crop_summary: dict | None,
                             voxelize_result: dict,
                             ct_full: Path, ct_used: Path,
                             gt_labelmap: Path, pair_dir: Path) -> str:
    """Coordinate-alignment dry-run summary (no paint loop run yet)."""
    fg = voxelize_result.get("foreground_voxels", 0)
    fg_mm3 = voxelize_result.get("foreground_volume_mm3", 0.0)
    mesh_world = voxelize_result.get("mesh_world_bounds")
    mesh_idx = voxelize_result.get("mesh_index_bounds")
    ref_dims = voxelize_result.get("reference_dims")
    ref_spacing = voxelize_result.get("reference_spacing_xyz")

    aligned = bool(fg and fg > 0)
    diag = "OK — GT mesh overlaps the CT grid" if aligned else (
        "FAILED — GT mesh produced 0 foreground voxels. Likely a "
        "coordinate-frame mismatch (e.g. LPS vs RAS, mm vs μm, or the "
        "mesh was exported in a transformed space)."
    )

    lines = [
        f"# Alignment dry run — `{ct_id}` vs `{gt_id}`",
        "",
        f"**Status:** {'**ALIGNED**' if aligned else '**NOT ALIGNED**'}  ",
        f"**Diagnosis:** {diag}",
        "",
        "## Inputs",
        "",
        f"- **CT volume:** [`{ct_id}`](https://www.morphosource.org/concern/media/{ct_id})  ",
        f"  source file: `{ct_pick.path.name}` ({ct_pick.size:,} bytes)",
        f"- **GT mesh:**  [`{gt_id}`](https://www.morphosource.org/concern/media/{gt_id})  ",
        f"  source file: `{mesh_pick.path.name}` ({mesh_pick.size:,} bytes)",
        "",
        "## CT preprocessing",
        "",
        f"- Working CT: `{ct_used.name}`",
    ]
    if crop_summary:
        lines.extend([
            f"- Crop margin: {crop_summary.get('margin_mm', 'n/a')} mm",
            f"- Original size: {crop_summary.get('original_size')}",
            f"- Cropped size: {crop_summary.get('crop_size')}",
            f"- Crop bounds (world, mm): {crop_summary.get('world_bounds_xyz')}",
        ])
    else:
        lines.append("- Cropping: disabled (full CT used)")

    lines.extend([
        "",
        "## GT voxelization",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Backend | `{voxelize_result.get('backend','?')}` |",
        f"| Reference dims (x,y,z) | {ref_dims} |",
        f"| Reference spacing (mm) | {ref_spacing} |",
        f"| Mesh #points / #cells | {voxelize_result.get('mesh_n_points')} / {voxelize_result.get('mesh_n_cells')} |",
        f"| Mesh world bounds (mm) | {mesh_world} |",
        f"| Mesh in voxel-index space | {mesh_idx} |",
        f"| GT foreground voxels | **{fg:,}** |",
        f"| GT foreground volume | **{fg_mm3:,.4f} mm³** |",
        "",
        "## Files",
        "",
        f"- [`gt_voxelized.nii.gz`](gt_voxelized.nii.gz) — rasterized GT labelmap",
        f"- [`gt_voxelized.voxelize.json`](gt_voxelized.voxelize.json) — voxelization summary",
    ])
    if crop_summary:
        lines.append(
            f"- [`{Path(crop_summary['output_path']).name}`]"
            f"({Path(crop_summary['output_path']).name}) — cropped CT"
        )
    if not aligned:
        lines.extend([
            "",
            "## Suggested fix when alignment fails",
            "",
            "1. Inspect `gt_voxelized.voxelize.json`: compare `mesh_world_bounds` "
            "to `reference_origin + reference_dims * reference_spacing`.",
            "2. If the mesh bbox is in millimetres but the CT origin/spacing is "
            "in micrometres (or vice versa), the mesh may need a 1000× scale.",
            "3. If only the *sign* of one axis differs, the mesh may be exported "
            "in **LPS** while the CT is in **RAS** (or vice versa); flip "
            "X and Y signs of the mesh.",
            "4. As a last resort, re-export the PLY from MorphoSource using the "
            "CT's coordinate system, or run the legacy Slicer voxelizer "
            "(`--voxelize-backend slicer`) which handles some transforms "
            "automatically.",
        ])
    return "\n".join(lines) + "\n"


def _render_report(ct_id: str, gt_id: str, goal: str,
                   ct_pick: FilePick, mesh_pick: FilePick,
                   voxelize_result: dict, nni_result: dict,
                   metrics: dict, pair_dir: Path,
                   max_steps: int) -> str:
    dice = metrics.get("dice")
    iou = metrics.get("iou")
    voxel_pred = metrics.get("voxel_count_pred", 0)
    voxel_gt = metrics.get("voxel_count_gt", 0)
    n_steps = nni_result.get("n_prompts", nni_result.get("steps", 0))

    lines = [
        f"# nnInteractive vs MorphoSource GT — `{ct_id}` vs `{gt_id}`",
        "",
        f"**Goal:** {goal}  ",
        f"**Max LLM steps:** {max_steps}  ",
        f"**Steps used:** {n_steps}  ",
        "",
        "## Inputs",
        "",
        f"- **CT volume:** [`{ct_id}`](https://www.morphosource.org/concern/media/{ct_id})  ",
        f"  file: `{ct_pick.path.name}` ({ct_pick.size:,} bytes)",
        f"- **GT mesh:**  [`{gt_id}`](https://www.morphosource.org/concern/media/{gt_id})  ",
        f"  file: `{mesh_pick.path.name}` ({mesh_pick.size:,} bytes)",
        "",
        "## Voxelization of GT mesh",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Reference dims | {voxelize_result.get('reference_dims', 'N/A')} |",
        f"| Reference spacing (mm) | {voxelize_result.get('reference_spacing_xyz', 'N/A')} |",
        f"| GT foreground voxels | {voxelize_result.get('foreground_voxels', 'N/A'):,} |",
        f"| GT volume (mm³) | {voxelize_result.get('foreground_volume_mm3', 'N/A')} |",
        "",
        "## Comparison metrics",
        "",
    ]

    # Re-render the metrics table from segmentation_metrics — we can't easily
    # rebuild SegMetrics here (different env), so build it inline.
    metric_rows = [
        ("Dice", f"**{dice:.4f}**" if dice is not None else "N/A"),
        ("IoU (Jaccard)", f"{iou:.4f}" if iou is not None else "N/A"),
        ("Precision", f"{metrics.get('precision', 0):.4f}"),
        ("Recall (sensitivity)", f"{metrics.get('recall', 0):.4f}"),
        ("Volume diff", f"{metrics.get('volume_difference_pct', 0):.2f} %"),
        ("Voxels pred / GT", f"{voxel_pred:,} / {voxel_gt:,}"),
        ("Hausdorff (max, mm)", f"{metrics.get('hausdorff_mm', 'N/A')}"),
        ("Hausdorff (95-pct, mm)", f"{metrics.get('hausdorff_95_mm', 'N/A')}"),
        ("Mean surface distance (mm)",
            f"{metrics.get('average_surface_dist_mm', 'N/A')}"),
        ("Centroid distance (mm)", f"{metrics.get('centroid_distance_mm', 'N/A')}"),
    ]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in metric_rows:
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Visual comparison",
        "",
        "Volume only / GT (blue) / Prediction (orange):",
        "",
        "![overlay](overlay.png)",
        "",
        "## Files",
        "",
        f"- [`gt_voxelized.nii.gz`](gt_voxelized.nii.gz) — GT mesh rasterized onto the CT grid",
        f"- [`nninteractive/{ct_id}_nni_labelmap.nii.gz`](nninteractive/{ct_id}_nni_labelmap.nii.gz) — nnInteractive prediction",
        f"- [`metrics.json`](metrics.json) — full metrics payload",
        f"- [`nninteractive/{ct_id}_nni_report.md`](nninteractive/{ct_id}_nni_report.md) — paint-loop step trace",
    ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare nnInteractive against a MorphoSource GT segmentation"
    )
    p.add_argument("--ct-media-id", default="",
                   help="MorphoSource media ID of the unsegmented CT volume")
    p.add_argument("--gt-media-id", default="",
                   help="MorphoSource media ID of the segmented derivative mesh")
    p.add_argument("--goal", default="",
                   help="Plain-English target for the paint loop, "
                        "e.g. 'Segment the cranial bone'")
    p.add_argument("--output-dir", default="/tmp/nni_compare",
                   help="Where to write outputs (default: /tmp/nni_compare)")
    p.add_argument("--max-steps", type=int, default=12,
                   help="Max LLM iterations for the paint loop")
    p.add_argument("--auto-discover", action="store_true",
                   help="Auto-pick the first viable CT↔mesh pair via "
                        "find_segmentation_pairs (--query controls the search)")
    p.add_argument("--query", default="skull mesh",
                   help="Search query when --auto-discover is set")
    p.add_argument("--require-taxonomy", default="",
                   help="Optional taxonomy filter for --auto-discover")
    p.add_argument("--voxelize-backend", default="auto",
                   choices=["auto", "slicer", "vtk"],
                   help="GT-mesh voxelization backend. 'vtk' is pure-Python "
                        "and works without a display server. 'auto' tries "
                        "Slicer first and falls back to VTK on failure.")
    p.add_argument("--crop-around-mesh-mm", type=float, default=0.0,
                   help="If >0, crop the CT to the GT mesh bbox + margin "
                        "(in mm) before running the paint loop. Useful for "
                        "small parts (e.g. a stapes inside a whole-head CT).")
    p.add_argument("--preset", default="", choices=[""] + list(PRESETS.keys()),
                   help="Pre-canned test pair. Overrides individual fields "
                        "but per-flag arguments still win if explicitly set.")
    p.add_argument("--skip-paint-loop", action="store_true",
                   help="Stop after voxelizing the GT mesh and emit an "
                        "alignment report (alignment_report.md). Useful as a "
                        "dry run before spending OpenAI quota on the loop.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _parse_args()

    ct_id = args.ct_media_id.strip()
    gt_id = args.gt_media_id.strip()
    goal = args.goal.strip()
    max_steps = args.max_steps
    voxelize_backend = args.voxelize_backend
    crop_mm = args.crop_around_mesh_mm

    if args.preset:
        preset = PRESETS[args.preset]
        log.info("Applying preset %r: %s", args.preset, preset)
        ct_id = ct_id or preset["ct_media_id"]
        gt_id = gt_id or preset["gt_media_id"]
        if not goal:
            goal = preset["goal"]
        if max_steps == 12:  # parser default
            max_steps = preset["max_steps"]
        if voxelize_backend == "auto":
            voxelize_backend = preset["voxelize_backend"]
        if crop_mm == 0.0:
            crop_mm = preset["crop_around_mesh_mm"]

    if args.auto_discover and (not ct_id or not gt_id):
        from find_segmentation_pairs import find_pairs
        log.info("Auto-discovery enabled — searching MorphoSource…")
        pairs = find_pairs(query=args.query, max_pairs=1,
                           require_taxonomy=args.require_taxonomy)
        if not pairs:
            log.error("No viable open-download CT↔mesh pair found for query=%r",
                      args.query)
            return 2
        pair = pairs[0]
        ct_id = pair.ct["media_id"]
        gt_id = pair.mesh["media_id"]
        log.info("Auto-picked pair: CT=%s GT=%s (specimen %s — %s)",
                 ct_id, gt_id, pair.physical_object_id, pair.taxonomy)

    if not ct_id or not gt_id:
        log.error("Both --ct-media-id and --gt-media-id are required "
                  "(or use --auto-discover, or --preset).")
        return 2

    if not goal and not args.skip_paint_loop:
        log.error("--goal is required (e.g. \"Segment the cranial bone\"). "
                  "Use --preset for a default goal on a known pair, or "
                  "--skip-paint-loop for an alignment-only dry run.")
        return 2

    log.info(
        "Running comparison: CT=%s GT=%s goal=%r max_steps=%d "
        "backend=%s crop_mm=%.1f skip_paint=%s",
        ct_id, gt_id, goal or "(none — skip-paint-loop)",
        max_steps, voxelize_backend, crop_mm, args.skip_paint_loop,
    )

    result = run_comparison(
        ct_media_id=ct_id,
        gt_media_id=gt_id,
        goal=goal,
        output_dir=Path(args.output_dir),
        max_steps=max_steps,
        voxelize_backend=voxelize_backend,
        crop_around_mesh_mm=crop_mm,
        skip_paint_loop=args.skip_paint_loop,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"},
                     indent=2, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
