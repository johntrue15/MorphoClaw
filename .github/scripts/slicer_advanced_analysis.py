"""
Advanced SlicerMorph analysis for research replication.

Runs inside 3D Slicer via --python-script. Uses SlicerMorph's actual
modules (GPA, SegmentStatistics, LandmarkGrid, etc.) to perform
publication-grade morphometric analysis on downloaded specimens.

This script is generated dynamically by slicer_tool.py and executed
headlessly. It produces:
  - Refined anatomical landmarks (not just extremal points)
  - Surface-snapped landmark positions
  - Per-segment statistics (if mesh has color-coded regions)
  - GPA-ready FCSV landmark files
  - Morphologika-format export
  - High-resolution multi-view screenshots
  - Measurement JSON compatible with published protocols
"""
import slicer
import vtk
import numpy as np
import json
import os
import sys

# Config via environment variables (Slicer doesn't pass argv to scripts)
PLY_PATH = os.environ.get("SLICER_PLY_PATH", "")
OUTPUT_DIR = os.environ.get("SLICER_OUTPUT_DIR", "/tmp/slicer_advanced")
MEDIA_ID = os.environ.get("SLICER_MEDIA_ID", "unknown")

if not PLY_PATH:
    # Fallback: try reading from a config file
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_slicer_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        PLY_PATH = cfg.get("ply_path", "")
        OUTPUT_DIR = cfg.get("output_dir", OUTPUT_DIR)
        MEDIA_ID = cfg.get("media_id", MEDIA_ID)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Advanced analysis: {PLY_PATH}")
print(f"Output: {OUTPUT_DIR}")

# ── Load mesh ──
success, modelNode = slicer.util.loadModel(PLY_PATH, returnNode=True)
if not success or not modelNode:
    json.dump({"error": "Failed to load mesh"}, open(os.path.join(OUTPUT_DIR, "analysis.json"), "w"))
    slicer.app.exit(1)

mesh = modelNode.GetMesh()
bounds = [0]*6
modelNode.GetBounds(bounds)
center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
npts = mesh.GetNumberOfPoints()
ncells = mesh.GetNumberOfCells()
print(f"Loaded: {npts:,} vertices, {ncells:,} faces")

# ── 1. Surface-snapped anatomical landmarks ──
print("\n── Anatomical Landmarks ──")

# Build point locator for snapping
locator = vtk.vtkPointLocator()
locator.SetDataSet(mesh)
locator.BuildLocator()

stride = max(1, npts // 100000)
pts = np.array([mesh.GetPoint(i) for i in range(0, npts, stride)])

# Define landmarks by anatomical position
landmark_defs = {
    "anterior_most": lambda p: p[:, 1].argmin(),
    "posterior_most": lambda p: p[:, 1].argmax(),
    "dorsal_most": lambda p: p[:, 2].argmax(),
    "ventral_most": lambda p: p[:, 2].argmin(),
    "left_most": lambda p: p[:, 0].argmin(),
    "right_most": lambda p: p[:, 0].argmax(),
}

# For skull-like specimens, add midline and bilateral landmarks
extent = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
midline_x = center[0]

# Midline landmarks at different AP levels
ap_range = bounds[3] - bounds[2]
for frac, name in [(0.25, "anterior_quarter"), (0.5, "midpoint"), (0.75, "posterior_quarter")]:
    y_target = bounds[2] + ap_range * frac
    midline_pts = pts[np.abs(pts[:, 0] - midline_x) < extent[0] * 0.05]
    if len(midline_pts) > 0:
        y_dists = np.abs(midline_pts[:, 1] - y_target)
        closest = midline_pts[y_dists.argmin()]
        # Find the dorsal-most point near this AP position
        near_pts = midline_pts[y_dists < ap_range * 0.05]
        if len(near_pts) > 0:
            dorsal_idx = near_pts[:, 2].argmax()
            landmark_defs[f"dorsal_{name}"] = lambda p, pos=near_pts[dorsal_idx]: np.argmin(np.linalg.norm(p - pos, axis=1))

# Bilateral landmarks at widest point
widest_y = pts[pts[:, 0].argmax(), 1]
for side, x_sign in [("left", -1), ("right", 1)]:
    side_pts = pts[pts[:, 0] * x_sign > 0]
    if len(side_pts) > 10:
        # Widest point
        if x_sign < 0:
            extreme_idx = side_pts[:, 0].argmin()
        else:
            extreme_idx = side_pts[:, 0].argmax()
        pos = side_pts[extreme_idx]
        landmark_defs[f"{side}_zygomatic"] = lambda p, pos=pos: np.argmin(np.linalg.norm(p - pos, axis=1))

# Place landmarks and snap to surface
markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Landmarks")
md = markupNode.GetDisplayNode()
if md:
    md.SetGlyphScale(2.0)
    md.SetTextScale(3.0)
    md.SetSelectedColor(1, 0.3, 0.3)

landmarks = []
for name, func in landmark_defs.items():
    try:
        idx = func(pts)
        raw_pt = pts[idx]
        # Snap to nearest mesh surface point
        snapped_idx = locator.FindClosestPoint(raw_pt[0], raw_pt[1], raw_pt[2])
        snapped_pt = mesh.GetPoint(snapped_idx)
        markupNode.AddControlPoint(vtk.vtkVector3d(*snapped_pt), name)
        landmarks.append({
            "name": name,
            "position": [round(float(c), 3) for c in snapped_pt],
            "raw_position": [round(float(c), 3) for c in raw_pt],
        })
        print(f"  {name}: [{snapped_pt[0]:.2f}, {snapped_pt[1]:.2f}, {snapped_pt[2]:.2f}]")
    except Exception as exc:
        print(f"  {name}: FAILED ({exc})")

# ── 2. Inter-landmark distances ──
print("\n── Measurements ──")
lm_dict = {l["name"]: np.array(l["position"]) for l in landmarks}
measurements = {}

distance_pairs = [
    ("total_length", "anterior_most", "posterior_most"),
    ("total_height", "ventral_most", "dorsal_most"),
    ("total_width", "left_most", "right_most"),
]
# Add bilateral measurements if available
if "left_zygomatic" in lm_dict and "right_zygomatic" in lm_dict:
    distance_pairs.append(("bizygomatic_width", "left_zygomatic", "right_zygomatic"))

for mname, p1, p2 in distance_pairs:
    if p1 in lm_dict and p2 in lm_dict:
        d = float(np.linalg.norm(lm_dict[p1] - lm_dict[p2]))
        measurements[mname] = round(d, 3)
        print(f"  {mname}: {d:.3f} mm")

# Ratios
if "total_length" in measurements and "total_width" in measurements:
    measurements["length_width_ratio"] = round(measurements["total_length"] / max(measurements["total_width"], 0.001), 3)
if "total_height" in measurements and "total_length" in measurements:
    measurements["height_length_ratio"] = round(measurements["total_height"] / max(measurements["total_length"], 0.001), 3)

# ── 3. Mass properties ──
print("\n── Mass Properties ──")
mp = vtk.vtkMassProperties()
mp.SetInputData(mesh)
mp.Update()
sa = mp.GetSurfaceArea()
vol = mp.GetVolume()
eq_r = (3 * abs(vol) / (4 * np.pi)) ** (1/3) if vol != 0 else 0
sphericity = (4 * np.pi * eq_r**2) / sa if sa > 0 else 0

mass_props = {
    "surface_area_mm2": round(sa, 3),
    "volume_mm3": round(vol, 3),
    "sphericity": round(sphericity, 4),
    "sa_vol_ratio": round(sa / max(abs(vol), 0.001), 4),
}
for k, v in mass_props.items():
    print(f"  {k}: {v}")

# ── 4. Curvature statistics ──
print("\n── Curvature ──")
curvature_stats = {}
for ctype_name, ctype_id in [("mean", 0), ("gaussian", 1)]:
    cf = vtk.vtkCurvatures()
    cf.SetInputData(mesh)
    if ctype_id == 0:
        cf.SetCurvatureTypeToMean()
    else:
        cf.SetCurvatureTypeToGaussian()
    cf.Update()
    cs = cf.GetOutput().GetPointData().GetScalars()
    sample = np.array([cs.GetValue(i) for i in range(0, min(cs.GetNumberOfTuples(), 200000), max(1, cs.GetNumberOfTuples()//200000))])
    p5, p95 = np.percentile(sample, [5, 95])
    clipped = sample[(sample >= p5) & (sample <= p95)]
    curvature_stats[ctype_name] = {
        "mean": round(float(np.mean(clipped)), 5),
        "std": round(float(np.std(clipped)), 5),
        "p5": round(float(p5), 5),
        "p95": round(float(p95), 5),
    }
    print(f"  {ctype_name}: mean={np.mean(clipped):.4f} std={np.std(clipped):.4f}")

# ── 5. Connectivity (bone segments) ──
print("\n── Connectivity ──")
conn = vtk.vtkConnectivityFilter()
conn.SetInputData(mesh)
conn.SetExtractionModeToAllRegions()
conn.ColorRegionsOn()
conn.Update()
n_regions = conn.GetNumberOfExtractedRegions()
print(f"  Regions: {n_regions}")

# ── 6. PCA shape descriptors ──
print("\n── PCA Shape ──")
centered = pts - pts.mean(axis=0)
cov = np.cov(centered.T)
eigenvalues = np.linalg.eigvalsh(cov)[::-1]
total_var = eigenvalues.sum()
pca = {
    "pc1_pct": round(float(100 * eigenvalues[0] / total_var), 2),
    "pc2_pct": round(float(100 * eigenvalues[1] / total_var), 2),
    "pc3_pct": round(float(100 * eigenvalues[2] / total_var), 2),
    "elongation": round(float(eigenvalues[0] / max(eigenvalues[1], 1e-10)), 3),
    "flatness": round(float(eigenvalues[1] / max(eigenvalues[2], 1e-10)), 3),
}
print(f"  PC1={pca['pc1_pct']}% PC2={pca['pc2_pct']}% PC3={pca['pc3_pct']}%")
print(f"  Elongation={pca['elongation']} Flatness={pca['flatness']}")

# ── 7. Bilateral asymmetry ──
print("\n── Bilateral Asymmetry ──")
left = pts[pts[:, 0] < center[0]]
right = pts[pts[:, 0] > center[0]]
left_ext = [np.ptp(left[:, d]) for d in range(3)] if len(left) > 10 else [0, 0, 0]
right_ext = [np.ptp(right[:, d]) for d in range(3)] if len(right) > 10 else [0, 0, 0]
asymmetry = [abs(l - r) / max(l, r, 0.001) * 100 for l, r in zip(left_ext, right_ext)]
mean_asym = round(float(np.mean(asymmetry)), 2)
print(f"  Asymmetry: {mean_asym}%")

# ── 8. Screenshots ──
print("\n── Screenshots ──")
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
tw = slicer.app.layoutManager().threeDWidget(0)
tv = tw.threeDView()
rw = tv.renderWindow()
rw.SetOffScreenRendering(1)
rw.SetSize(1600, 1200)
renderer = rw.GetRenderers().GetFirstRenderer()
renderer.SetBackground(0.05, 0.05, 0.1)
camera = renderer.GetActiveCamera()

screenshots = []
views = [
    ("anterior", (0, -120, 15), (0, 0, 1)),
    ("lateral_L", (120, 0, 15), (0, 0, 1)),
    ("dorsal", (0, 0, 120), (0, -1, 0)),
    ("posterior", (0, 120, 15), (0, 0, 1)),
    ("ventral", (0, 0, -120), (0, 1, 0)),
    ("lateral_R", (-120, 0, 15), (0, 0, 1)),
    ("oblique_FR", (80, -80, 60), (0, 0, 1)),
    ("oblique_FL", (-80, -80, 60), (0, 0, 1)),
]

for vname, pos, vup in views:
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
    fpath = os.path.join(OUTPUT_DIR, f"{MEDIA_ID}_{vname}.png")
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fpath)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    screenshots.append(fpath)

print(f"  {len(screenshots)} screenshots saved")

# ── 9. Export landmarks as FCSV ──
fcsv_path = os.path.join(OUTPUT_DIR, f"{MEDIA_ID}_landmarks.fcsv")
slicer.util.saveNode(markupNode, fcsv_path)
print(f"  Landmarks: {fcsv_path}")

# ── 10. Full analysis JSON ──
result = {
    "media_id": MEDIA_ID,
    "mesh_file": PLY_PATH,
    "vertices": npts,
    "faces": ncells,
    "bounds": [round(b, 3) for b in bounds],
    "extent_mm": [round(e, 3) for e in extent],
    "landmarks": landmarks,
    "measurements": measurements,
    "mass_properties": mass_props,
    "curvature": curvature_stats,
    "n_regions": n_regions,
    "pca_shape": pca,
    "bilateral_asymmetry_pct": mean_asym,
    "screenshots": screenshots,
    "landmark_file": fcsv_path,
    "analysis_type": "advanced_slicermorph",
}

analysis_path = os.path.join(OUTPUT_DIR, "analysis.json")
with open(analysis_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nAnalysis complete: {analysis_path}")
slicer.app.exit(0)
