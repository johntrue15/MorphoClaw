"""
Crop a 3D volume to the bounding box of a reference mesh, plus margin.

This is the "tractable input" pre-processing step for the nnInteractive
comparison harness. Working on a 1.3 GB whole-head CT is slow and uses a
lot of memory, but a small anatomical structure (e.g. a single stapes)
only occupies a few mm of the image. We use the ground-truth mesh's
bounding box, expanded by a user-controlled margin, to compute a tight
sub-volume that still gives nnInteractive enough context.

The cropping is done **in physical (world) coordinates** so it is
correct regardless of the volume's direction-cosine matrix.

Outputs:
    <output>                          cropped volume (.nii.gz)
    <output>.crop.json                summary (bbox in world & index coords)

Usage::

    python crop_around_mesh.py \
        --reference-volume /path/to/ct.nii.gz \
        --mesh /path/to/segmentation.ply \
        --output /tmp/ct_cropped.nii.gz \
        --margin-mm 5

If ``--mesh`` is given, the bounding box comes from the mesh.
If ``--mesh-bounds`` is given (xmin xmax ymin ymax zmin zmax in mm),
the crop uses those numbers directly (useful for re-cropping a labelmap
to the same region without re-loading the mesh).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

log = logging.getLogger("crop_around_mesh")


def _import_deps():
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run inside the nnInteractive venv.",
              file=sys.stderr)
        sys.exit(1)
    return np, sitk


def _read_mesh_bounds(mesh_path: Path) -> Tuple[float, float, float, float, float, float]:
    """Return (xmin,xmax,ymin,ymax,zmin,zmax) in world coords."""
    suffix = mesh_path.suffix.lower()
    try:
        import vtk  # type: ignore
    except ImportError:
        vtk = None

    if vtk is not None:
        if suffix == ".ply":
            r = vtk.vtkPLYReader()
        elif suffix == ".stl":
            r = vtk.vtkSTLReader()
        elif suffix == ".obj":
            r = vtk.vtkOBJReader()
        else:
            r = None
        if r is not None:
            r.SetFileName(str(mesh_path))
            r.Update()
            poly = r.GetOutput()
            if poly and poly.GetNumberOfPoints() > 0:
                return tuple(poly.GetBounds())  # type: ignore[return-value]

    # Fallback: trimesh
    try:
        import trimesh  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Need vtk or trimesh to read mesh bounds. "
            "Install one of them in the nnInteractive venv."
        )
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if mesh.vertices.size == 0:
        raise RuntimeError(f"Mesh has no vertices: {mesh_path}")
    mins = mesh.vertices.min(axis=0)
    maxs = mesh.vertices.max(axis=0)
    return (float(mins[0]), float(maxs[0]),
            float(mins[1]), float(maxs[1]),
            float(mins[2]), float(maxs[2]))


def _expand_bounds(bounds, margin_mm: float):
    return (
        bounds[0] - margin_mm, bounds[1] + margin_mm,
        bounds[2] - margin_mm, bounds[3] + margin_mm,
        bounds[4] - margin_mm, bounds[5] + margin_mm,
    )


def crop(reference_volume: Path, output_path: Path,
         margin_mm: float = 5.0,
         mesh_path: Path | None = None,
         explicit_bounds: tuple | None = None,
         summary_path: Path | None = None) -> dict:
    np, sitk = _import_deps()

    log.info("Reading reference volume: %s", reference_volume)
    if reference_volume.is_dir():
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(reference_volume))
        if not files:
            raise RuntimeError(f"No DICOM files in {reference_volume}")
        reader.SetFileNames(files)
        ref = reader.Execute()
    else:
        ref = sitk.ReadImage(str(reference_volume))

    size = ref.GetSize()

    if explicit_bounds is not None:
        bounds = tuple(float(b) for b in explicit_bounds)
        log.info("Using explicit bounds: %s", bounds)
    elif mesh_path is not None:
        bounds = _read_mesh_bounds(mesh_path)
        log.info("Mesh world bounds: %s", bounds)
    else:
        raise ValueError("Provide either --mesh or --mesh-bounds")

    bounds = _expand_bounds(bounds, margin_mm)
    log.info("Bounds + %.1fmm margin: %s", margin_mm, bounds)

    # Sample the 8 corners of the world-space box in image-index space.
    corners = []
    for x in (bounds[0], bounds[1]):
        for y in (bounds[2], bounds[3]):
            for z in (bounds[4], bounds[5]):
                idx = ref.TransformPhysicalPointToContinuousIndex((x, y, z))
                corners.append(idx)
    arr = np.array(corners)
    idx_min = np.floor(arr.min(axis=0)).astype(int)
    idx_max = np.ceil(arr.max(axis=0)).astype(int)

    # Clamp to image extent.
    idx_min = np.maximum(idx_min, 0)
    idx_max = np.minimum(idx_max, np.array(size) - 1)

    if (idx_max < idx_min).any():
        raise RuntimeError(
            f"Mesh bbox does not intersect the reference volume "
            f"(idx_min={idx_min.tolist()}, idx_max={idx_max.tolist()}, "
            f"size={list(size)}). Coordinate frames likely don't match."
        )

    crop_size = (idx_max - idx_min + 1).tolist()
    log.info("Crop index range: min=%s size=%s",
             idx_min.tolist(), crop_size)

    cropped = sitk.RegionOfInterest(
        ref,
        size=[int(s) for s in crop_size],
        index=[int(i) for i in idx_min.tolist()],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(cropped, str(output_path))
    log.info("Wrote cropped volume to %s "
             "(new size=%s, new origin=%s)",
             output_path, cropped.GetSize(), cropped.GetOrigin())

    summary = {
        "reference_volume": str(reference_volume),
        "mesh_path": str(mesh_path) if mesh_path else None,
        "margin_mm": margin_mm,
        "world_bounds_xyz": list(bounds),
        "index_min": idx_min.tolist(),
        "index_max": idx_max.tolist(),
        "crop_size": crop_size,
        "original_size": list(size),
        "output_path": str(output_path),
        "output_origin": list(cropped.GetOrigin()),
        "output_spacing": list(cropped.GetSpacing()),
        "output_direction": list(cropped.GetDirection()),
    }

    if summary_path is None:
        summary_path = output_path.with_suffix("").with_suffix(".crop.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Crop summary: %s", summary_path)
    return summary


def _parse_args():
    p = argparse.ArgumentParser(
        description="Crop a CT volume to a mesh bounding box + margin "
                    "(physical-coordinate aware, direction-cosine safe)."
    )
    p.add_argument("--reference-volume", required=True,
                   help="CT volume (NIfTI/NRRD/MHA) or DICOM series dir")
    p.add_argument("--output", required=True,
                   help="Output cropped volume path (.nii.gz)")
    p.add_argument("--margin-mm", type=float, default=5.0,
                   help="Padding around mesh bbox, in mm (default: 5)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mesh",
                   help="Mesh file whose bbox defines the crop region "
                        "(PLY/STL/OBJ)")
    g.add_argument("--mesh-bounds", nargs=6, type=float,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="Explicit bounds in world coords (mm)")
    p.add_argument("--summary", default="",
                   help="Path for JSON summary (default: <output>.crop.json)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _parse_args()
    t0 = time.time()
    try:
        crop(
            reference_volume=Path(args.reference_volume),
            output_path=Path(args.output),
            margin_mm=args.margin_mm,
            mesh_path=Path(args.mesh) if args.mesh else None,
            explicit_bounds=tuple(args.mesh_bounds) if args.mesh_bounds else None,
            summary_path=Path(args.summary) if args.summary else None,
        )
    except Exception as exc:
        log.error("Crop failed: %s", exc, exc_info=True)
        return 1
    log.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
