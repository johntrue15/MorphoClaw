"""
Voxelize a surface mesh onto a reference CT volume's grid (headless 3D Slicer).

MorphoSource "segmented derivative" media are typically PLY/STL/OBJ surface
meshes — a researcher exported the human-curated segmentation as a closed
surface. To compare such a mesh against an nnInteractive labelmap, we need
to rasterize the mesh into the *same* voxel grid as the source CT volume
(same spacing, size, origin, direction).

This script does exactly that, using ``vtkPolyDataToImageStencil``:

    1. Load the reference CT volume (NIfTI/NRRD) — provides the target grid
    2. Load the GT mesh (PLY/STL/OBJ)
    3. Build a polydata-to-image stencil sampled on the reference grid
    4. Rasterize foreground = 1 inside the closed surface
    5. Save the resulting labelmap as NIfTI (aligned to the reference)

Configuration (env vars or _voxelize_config.json sitting next to this script):

    VOXELIZE_REFERENCE_VOLUME   path to CT volume (NIfTI/NRRD)
    VOXELIZE_MESH_PATH          path to GT mesh
    VOXELIZE_OUTPUT_PATH        output NIfTI labelmap path
    VOXELIZE_FILL_VALUE         label value (default: 1)

Invoked headlessly:

    Slicer --no-splash --no-main-window \
        --python-script voxelize_mesh_in_slicer.py
"""
import json
import os
import sys

import slicer
import vtk


def _read_config():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_voxelize_config.json"
    )
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    return {
        "reference_volume": os.environ.get(
            "VOXELIZE_REFERENCE_VOLUME", cfg.get("reference_volume", "")
        ),
        "mesh_path": os.environ.get(
            "VOXELIZE_MESH_PATH", cfg.get("mesh_path", "")
        ),
        "output_path": os.environ.get(
            "VOXELIZE_OUTPUT_PATH", cfg.get("output_path", "")
        ),
        "fill_value": int(os.environ.get(
            "VOXELIZE_FILL_VALUE", cfg.get("fill_value", 1)
        )),
    }


def _bail(msg: str, code: int = 1):
    print(f"[voxelize] ERROR: {msg}", file=sys.stderr)
    slicer.app.exit(code)
    sys.exit(code)


def main():
    cfg = _read_config()

    ref_path = cfg["reference_volume"]
    mesh_path = cfg["mesh_path"]
    out_path = cfg["output_path"]
    fill = cfg["fill_value"]

    if not ref_path or not os.path.exists(ref_path):
        _bail(f"reference_volume missing or not found: {ref_path}")
    if not mesh_path or not os.path.exists(mesh_path):
        _bail(f"mesh_path missing or not found: {mesh_path}")
    if not out_path:
        _bail("output_path is required")

    print(f"[voxelize] reference: {ref_path}")
    print(f"[voxelize] mesh:      {mesh_path}")
    print(f"[voxelize] output:    {out_path}")
    print(f"[voxelize] fill:      {fill}")

    ref_node = slicer.util.loadVolume(ref_path)
    if not ref_node:
        _bail(f"Failed to load reference volume: {ref_path}")
    ref_image = ref_node.GetImageData()
    spacing = ref_node.GetSpacing()
    origin = ref_node.GetOrigin()
    dims = ref_image.GetDimensions()
    print(f"[voxelize]   ref dims    = {dims}")
    print(f"[voxelize]   ref spacing = {spacing}")
    print(f"[voxelize]   ref origin  = {origin}")

    # IJK ↔ RAS so the stencil samples in the right physical space
    ijk_to_ras = vtk.vtkMatrix4x4()
    ref_node.GetIJKToRASMatrix(ijk_to_ras)
    ras_to_ijk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijk_to_ras, ras_to_ijk)

    model_node = slicer.util.loadModel(mesh_path)
    if not model_node:
        _bail(f"Failed to load mesh: {mesh_path}")
    polydata = model_node.GetPolyData()
    print(f"[voxelize]   mesh points = {polydata.GetNumberOfPoints():,}")
    print(f"[voxelize]   mesh cells  = {polydata.GetNumberOfCells():,}")

    # Surface meshes from MorphoSource may not be closed. vtkFillHolesFilter
    # closes small gaps so the stencil works. vtkPolyDataNormals + Triangle
    # ensures consistent orientation.
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(polydata)
    fill_holes.SetHoleSize(1e6)
    fill_holes.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(fill_holes.GetOutputPort())
    triangles.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(triangles.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    closed = normals.GetOutput()
    print(f"[voxelize]   closed mesh: {closed.GetNumberOfPoints():,} pts, "
          f"{closed.GetNumberOfCells():,} cells")

    # Transform mesh from RAS → IJK so the stencil aligns with the reference
    # volume's voxel grid (extent 0..N-1 with unit spacing in voxel space).
    transform = vtk.vtkTransform()
    transform.SetMatrix(ras_to_ijk)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(closed)
    tf.SetTransform(transform)
    tf.Update()

    # Build a blank reference image in IJK voxel space (origin=0, spacing=1)
    blank = vtk.vtkImageData()
    blank.SetDimensions(dims)
    blank.SetSpacing(1.0, 1.0, 1.0)
    blank.SetOrigin(0.0, 0.0, 0.0)
    blank.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    # Fill with zero
    pts = blank.GetPointData().GetScalars()
    n_total = blank.GetNumberOfPoints()
    for i in range(n_total):
        pts.SetTuple1(i, 0)

    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputConnection(tf.GetOutputPort())
    stencil.SetOutputOrigin(blank.GetOrigin())
    stencil.SetOutputSpacing(blank.GetSpacing())
    stencil.SetOutputWholeExtent(blank.GetExtent())
    stencil.Update()

    stenciler = vtk.vtkImageStencil()
    stenciler.SetInputData(blank)
    stenciler.SetStencilConnection(stencil.GetOutputPort())
    stenciler.ReverseStencilOff()
    stenciler.SetBackgroundValue(0)
    stenciler.Update()

    # Make the foreground voxels carry the requested fill value
    cast = vtk.vtkImageCast()
    cast.SetInputConnection(stenciler.GetOutputPort())
    cast.SetOutputScalarTypeToUnsignedChar()
    cast.Update()

    if fill != 1:
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputConnection(cast.GetOutputPort())
        threshold.ThresholdByUpper(1)
        threshold.SetInValue(fill)
        threshold.SetOutValue(0)
        threshold.SetOutputScalarTypeToUnsignedChar()
        threshold.Update()
        result_image = threshold.GetOutput()
    else:
        result_image = cast.GetOutput()

    # Build a labelmap MRML node, copy IJK→RAS from the reference, save.
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode", "VoxelizedGT"
    )
    labelmap_node.SetAndObserveImageData(result_image)
    labelmap_node.SetIJKToRASMatrix(ijk_to_ras)
    labelmap_node.SetSpacing(spacing)
    labelmap_node.SetOrigin(origin)

    # Quick stats
    arr = slicer.util.arrayFromVolume(labelmap_node)
    n_fg = int((arr > 0).sum())
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    print(f"[voxelize]   foreground voxels = {n_fg:,} ({n_fg * voxel_vol:.2f} mm^3)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if not slicer.util.saveNode(labelmap_node, out_path):
        _bail(f"Failed to save labelmap to {out_path}")

    summary = {
        "reference_volume": ref_path,
        "mesh_path": mesh_path,
        "output_path": out_path,
        "fill_value": fill,
        "reference_dims": list(dims),
        "reference_spacing_xyz": [float(s) for s in spacing],
        "reference_origin_xyz": [float(o) for o in origin],
        "foreground_voxels": n_fg,
        "foreground_volume_mm3": round(n_fg * voxel_vol, 3),
    }
    summary_path = os.path.splitext(out_path)[0] + ".voxelize.json"
    if summary_path.endswith(".nii.voxelize.json"):
        summary_path = summary_path.replace(".nii.voxelize.json", ".voxelize.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[voxelize] Done. Summary: {summary_path}")

    slicer.app.exit(0)


main()
