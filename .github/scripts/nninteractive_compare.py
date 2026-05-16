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
    """Pick the largest NIfTI/NRRD/MHA file, falling back to a DICOM dir."""
    matches = _walk_files(directory, VOLUME_EXTS)
    if matches:
        return matches[0]
    # DICOM directory fallback
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
            return FilePick(path=sub, size=total)
    return None


def _find_mesh(directory: Path) -> Optional[FilePick]:
    matches = _walk_files(directory, MESH_EXTS)
    return matches[0] if matches else None


def _download(media_id: str, dest: Path) -> dict:
    """Use the existing MorphoSource downloader to fetch a media bundle."""
    from morphosource_api_download import download_media

    dest.mkdir(parents=True, exist_ok=True)
    log.info("Downloading media %s → %s", media_id, dest)
    return download_media(media_id, str(dest))


def _voxelize(reference_volume: Path, mesh: Path, output: Path) -> dict:
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


def _run_paint_loop(input_volume: Path, goal: str, output_dir: Path,
                     media_id: str, max_steps: int) -> dict:
    """Run nninteractive_loop.py inside the dedicated venv."""
    if not NNI_PYTHON.exists():
        return {
            "error": (
                f"nnInteractive venv not found at {NNI_PYTHON}. "
                "Bootstrap it once with `.github/scripts/install_nninteractive.sh`."
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(NNI_PYTHON),
        str(SCRIPT_DIR / "nninteractive_loop.py"),
        "--input", str(input_volume),
        "--goal", goal,
        "--media-id", media_id,
        "--output-dir", str(output_dir),
        "--max-steps", str(max_steps),
    ]
    env = os.environ.copy()
    env.setdefault("NNINTERACTIVE_HOME", str(NNI_HOME))

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
    """Run segmentation_metrics in the nnInteractive venv (it has SimpleITK)."""
    cmd = [
        str(NNI_PYTHON),
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
                   output_dir: Path, max_steps: int = 12) -> dict:
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

    # ---- 3. Voxelize GT mesh onto CT grid ----
    gt_labelmap = pair_dir / "gt_voxelized.nii.gz"
    voxelize_result = _voxelize(ct_pick.path, mesh_pick.path, gt_labelmap)
    if "error" in voxelize_result:
        return {"success": False, "stage": "voxelize", "result": voxelize_result}

    # ---- 4. Run paint loop on the CT volume ----
    nni_dir = pair_dir / "nninteractive"
    nni_result = _run_paint_loop(ct_pick.path, goal, nni_dir, ct_media_id,
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
        volume=ct_pick.path,
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
        "ct_path": str(ct_pick.path),
        "mesh_path": str(mesh_pick.path),
        "gt_labelmap": str(gt_labelmap),
        "prediction_labelmap": str(pred_labelmap),
        "metrics_path": str(pair_dir / "metrics.json"),
        "overlay_path": str(pair_dir / "overlay.png"),
        "report_path": str(report_path),
        "metrics": metrics,
        "duration_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------


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
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _parse_args()

    ct_id = args.ct_media_id.strip()
    gt_id = args.gt_media_id.strip()

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
                  "(or use --auto-discover).")
        return 2

    goal = args.goal.strip()
    if not goal:
        log.error("--goal is required (e.g. \"Segment the cranial bone\").")
        return 2

    result = run_comparison(
        ct_media_id=ct_id,
        gt_media_id=gt_id,
        goal=goal,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"},
                     indent=2, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
