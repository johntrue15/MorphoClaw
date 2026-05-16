"""
Pure-Python mesh -> labelmap voxelization (no 3D Slicer).

Drop-in replacement for ``voxelize_mesh_in_slicer.py``: takes a reference
volume (NIfTI/NRRD/MHA/DICOM-series-converted-to-NIfTI) and a surface mesh
(PLY/STL/OBJ), and writes a binary labelmap aligned to the reference voxel
grid.

The algorithm is the canonical one (the same one Slicer uses internally):

    1. Read the reference image with SimpleITK to get
       (origin, spacing, direction, size).
    2. Read the mesh with VTK and transform its vertices from world
       coordinates into reference *voxel-index* coordinates::

           voxel_index = inv(direction) @ (world - origin) / spacing

       That bakes the direction-cosine matrix into the mesh, so we can
       stencil onto a trivial unit-spacing axis-aligned grid.
    3. Use ``vtkPolyDataToImageStencil`` + ``vtkImageStencilToImage`` to
       rasterize the closed surface to a binary mask in voxel space.
    4. Wrap the mask back into a SimpleITK image with the **original**
       origin / spacing / direction so it overlays the CT exactly.
    5. Write a .nii.gz labelmap and a small JSON summary.

Why pure Python rather than Slicer:
    - Slicer's GUI Qt platform plugin aborts when launched from a
      LaunchAgent without an Aqua bootstrap (see RUNNER_SETUP notes).
    - This pipeline only needs voxelization; we don't need a window
      server or Slicer's segment editor for that step.

Usage::

    python voxelize_mesh_vtk.py \
        --reference-volume /path/to/ct.nii.gz \
        --mesh /path/to/segmentation.ply \
        --output /tmp/gt_voxelized.nii.gz \
        --fill-value 1

Exit codes: 0 success, 1 on any error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

log = logging.getLogger("voxelize_vtk")


# ---------------------------------------------------------------------------
# Lazy imports (so --help works without the venv)
# ---------------------------------------------------------------------------


def _import_deps():
    try:
        import numpy as np
        import SimpleITK as sitk
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        print(
            f"Missing dependency: {exc}. "
            "Run inside the nnInteractive venv "
            "($NNINTERACTIVE_HOME/bin/python). "
            "If it's missing vtk/trimesh, re-run install_nninteractive.sh.",
            file=sys.stderr,
        )
        sys.exit(1)
    return np, sitk, vtk, numpy_support


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------


def _read_mesh(vtk, mesh_path: Path):
    """Pick a VTK reader by file extension and return ``vtkPolyData``."""
    suffix = mesh_path.suffix.lower()
    if suffix == ".ply":
        reader = vtk.vtkPLYReader()
    elif suffix == ".stl":
        reader = vtk.vtkSTLReader()
    elif suffix == ".obj":
        reader = vtk.vtkOBJReader()
    elif suffix in (".vtp", ".vtk"):
        reader = vtk.vtkXMLPolyDataReader() if suffix == ".vtp" else vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported mesh format: {suffix} ({mesh_path})")
    reader.SetFileName(str(mesh_path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Mesh has no points: {mesh_path}")
    return poly


# ---------------------------------------------------------------------------
# World <-> voxel index transform (handles direction cosines)
# ---------------------------------------------------------------------------


def _world_to_index_transform(np, vtk, origin: Tuple[float, float, float],
                              spacing: Tuple[float, float, float],
                              direction_3x3) -> "vtk.vtkTransform":
    """Return a vtkTransform that maps world coords -> voxel-index coords.

    ``direction_3x3`` is the SimpleITK 3x3 direction matrix as a 9-tuple
    (row-major) or a 3x3 numpy array.
    """
    D = np.asarray(direction_3x3, dtype=np.float64).reshape(3, 3)
    invD = np.linalg.inv(D)
    inv_spacing = 1.0 / np.asarray(spacing, dtype=np.float64)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = (np.diag(inv_spacing) @ invD)
    M[:3, 3] = -M[:3, :3] @ np.asarray(origin, dtype=np.float64)

    vmat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            vmat.SetElement(r, c, float(M[r, c]))
    t = vtk.vtkTransform()
    t.SetMatrix(vmat)
    return t


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------


def voxelize(reference_volume: Path, mesh_path: Path,
             output_path: Path, fill_value: int = 1,
             summary_path: Path | None = None) -> dict:
    np, sitk, vtk, numpy_support = _import_deps()

    log.info("Reading reference volume: %s", reference_volume)
    if reference_volume.is_dir():
        # DICOM series directory
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(reference_volume))
        if not files:
            raise RuntimeError(f"No DICOM files in {reference_volume}")
        reader.SetFileNames(files)
        ref = reader.Execute()
    else:
        ref = sitk.ReadImage(str(reference_volume))

    size = ref.GetSize()             # (sx, sy, sz)
    origin = ref.GetOrigin()
    spacing = ref.GetSpacing()
    direction = ref.GetDirection()   # 9-tuple

    log.info("Reference: size=%s spacing=%s origin=%s",
             size, spacing, origin)

    log.info("Reading mesh: %s", mesh_path)
    poly = _read_mesh(vtk, mesh_path)
    n_pts = poly.GetNumberOfPoints()
    n_cells = poly.GetNumberOfCells()
    bounds = poly.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax) in WORLD coords
    log.info("Mesh: %d points, %d cells, bounds=%s",
             n_pts, n_cells, bounds)

    # ---- Transform mesh from world -> voxel-index coords ----
    xform = _world_to_index_transform(np, vtk, origin, spacing, direction)
    tfilter = vtk.vtkTransformPolyDataFilter()
    tfilter.SetTransform(xform)
    tfilter.SetInputData(poly)
    tfilter.Update()
    poly_idx = tfilter.GetOutput()

    idx_bounds = poly_idx.GetBounds()
    log.info("Mesh in voxel-index space: bounds=%s", idx_bounds)

    if (idx_bounds[1] < 0 or idx_bounds[3] < 0 or idx_bounds[5] < 0
            or idx_bounds[0] >= size[0]
            or idx_bounds[2] >= size[1]
            or idx_bounds[4] >= size[2]):
        log.warning(
            "Mesh bounds in voxel-index space do not overlap reference "
            "extent (size=%s). Coordinate frames may be incompatible "
            "(e.g. LPS vs RAS, or different physical units).", size,
        )

    # ---- Build a unit-grid vtkImageData with extent = reference size ----
    white = vtk.vtkImageData()
    white.SetSpacing(1.0, 1.0, 1.0)
    white.SetOrigin(0.0, 0.0, 0.0)
    white.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    white.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    white.GetPointData().GetScalars().Fill(int(fill_value))

    # ---- Stencil: rasterize closed surface ----
    stencil_src = vtk.vtkPolyDataToImageStencil()
    stencil_src.SetInputData(poly_idx)
    stencil_src.SetOutputOrigin(0.0, 0.0, 0.0)
    stencil_src.SetOutputSpacing(1.0, 1.0, 1.0)
    stencil_src.SetOutputWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    stencil_src.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(white)
    stencil.SetStencilConnection(stencil_src.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()
    masked = stencil.GetOutput()

    # ---- VTK image -> numpy -> SimpleITK ----
    arr = numpy_support.vtk_to_numpy(masked.GetPointData().GetScalars())
    arr = arr.reshape(size[2], size[1], size[0])  # SITK z,y,x order
    arr = arr.astype(np.uint8)

    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing labelmap to %s", output_path)
    sitk.WriteImage(img, str(output_path))

    fg = int((arr > 0).sum())
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    summary = {
        "reference_volume": str(reference_volume),
        "mesh_path": str(mesh_path),
        "output_path": str(output_path),
        "fill_value": int(fill_value),
        "reference_dims": list(size),
        "reference_spacing_xyz": list(spacing),
        "reference_origin": list(origin),
        "reference_direction": list(direction),
        "mesh_n_points": int(n_pts),
        "mesh_n_cells": int(n_cells),
        "mesh_world_bounds": list(bounds),
        "mesh_index_bounds": list(idx_bounds),
        "foreground_voxels": fg,
        "foreground_volume_mm3": round(fg * voxel_volume, 4),
        "backend": "vtk",
    }

    if summary_path is None:
        summary_path = output_path.with_suffix("").with_suffix(".voxelize.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Foreground voxels: %d (%.2f mm³)",
             fg, summary["foreground_volume_mm3"])
    log.info("Summary: %s", summary_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Voxelize a surface mesh onto a reference volume "
                    "grid (no 3D Slicer required)."
    )
    p.add_argument("--reference-volume", required=True,
                   help="CT volume (NIfTI/NRRD/MHA) or DICOM series directory")
    p.add_argument("--mesh", required=True,
                   help="Surface mesh file (PLY/STL/OBJ)")
    p.add_argument("--output", required=True,
                   help="Output labelmap path (.nii.gz)")
    p.add_argument("--fill-value", type=int, default=1,
                   help="Voxel value inside the mesh (default: 1)")
    p.add_argument("--summary", default="",
                   help="Optional explicit path for the JSON summary "
                        "(default: <output>.voxelize.json)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _parse_args()

    t0 = time.time()
    try:
        voxelize(
            reference_volume=Path(args.reference_volume),
            mesh_path=Path(args.mesh),
            output_path=Path(args.output),
            fill_value=args.fill_value,
            summary_path=Path(args.summary) if args.summary else None,
        )
    except Exception as exc:
        log.error("Voxelization failed: %s", exc, exc_info=True)
        return 1

    log.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
