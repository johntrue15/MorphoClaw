"""
Headless MONAI Auto3DSeg segmentation inside 3D Slicer.

Loads a CT volume (NIfTI/NRRD/DICOM), runs a MONAI segmentation model,
and exports results as a labeled NIfTI + JSON summary.

Executed by slicer_tool.py via:
    Slicer --no-splash --no-main-window --python-script slicer_monai_seg.py

Configuration is read from environment variables or _monai_config.json:
    MONAI_INPUT_PATH    - path to the input volume
    MONAI_OUTPUT_DIR    - directory for results
    MONAI_MODEL_ID      - model ID (default: whole-body-3mm)
    MONAI_FORCE_CPU     - "1" to force CPU (default on macOS)
    MONAI_MEDIA_ID      - media ID for labeling results
"""
import slicer
import vtk
import json
import os
import sys
import time
import platform
import numpy as np

# ── Read config ──
INPUT_PATH = os.environ.get("MONAI_INPUT_PATH", "")
OUTPUT_DIR = os.environ.get("MONAI_OUTPUT_DIR", "/tmp/monai_seg")
MODEL_ID = os.environ.get("MONAI_MODEL_ID", "")
FORCE_CPU = os.environ.get("MONAI_FORCE_CPU", "1" if platform.system() == "Darwin" else "0")
MEDIA_ID = os.environ.get("MONAI_MEDIA_ID", "unknown")

if not INPUT_PATH:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_monai_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        INPUT_PATH = cfg.get("input_path", "")
        OUTPUT_DIR = cfg.get("output_dir", OUTPUT_DIR)
        MODEL_ID = cfg.get("model_id", MODEL_ID)
        FORCE_CPU = cfg.get("force_cpu", FORCE_CPU)
        MEDIA_ID = cfg.get("media_id", MEDIA_ID)

if not INPUT_PATH:
    print("[MONAI] ERROR: No input path specified")
    slicer.app.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[MONAI] Input: {INPUT_PATH}")
print(f"[MONAI] Output: {OUTPUT_DIR}")
print(f"[MONAI] Model: {MODEL_ID or '(auto-detect)'}")
print(f"[MONAI] CPU-only: {FORCE_CPU}")

# ── Ensure MONAIAuto3DSeg is available ──
try:
    monai_module = slicer.modules.monaiauto3dseg
    print("[MONAI] MONAIAuto3DSeg extension found")
except AttributeError:
    print("[MONAI] ERROR: MONAIAuto3DSeg extension not installed.")
    print("        Run: Slicer --python-script slicer_install_monai.py")
    result = {"error": "MONAIAuto3DSeg extension not installed", "media_id": MEDIA_ID}
    with open(os.path.join(OUTPUT_DIR, "monai_seg.json"), "w") as f:
        json.dump(result, f, indent=2)
    slicer.app.exit(1)

# ── Load input volume ──
print(f"[MONAI] Loading volume: {INPUT_PATH}")
t0 = time.time()

input_ext = os.path.splitext(INPUT_PATH)[1].lower()
if input_ext in (".gz",):
    input_ext = os.path.splitext(os.path.splitext(INPUT_PATH)[0])[1].lower() + input_ext

if os.path.isdir(INPUT_PATH):
    from DICOMLib import DICOMUtils
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        DICOMUtils.importDicom(INPUT_PATH, db)
        patientUIDs = db.patients()
        if not patientUIDs:
            print("[MONAI] ERROR: No DICOM data found in directory")
            slicer.app.exit(1)
        loadedNodeIDs = DICOMUtils.loadPatientByUID(patientUIDs[0])
        volumeNode = slicer.mrmlScene.GetNodeByID(loadedNodeIDs[0]) if loadedNodeIDs else None
else:
    volumeNode = slicer.util.loadVolume(INPUT_PATH)

if not volumeNode:
    print(f"[MONAI] ERROR: Failed to load volume from {INPUT_PATH}")
    result = {"error": f"Failed to load volume: {INPUT_PATH}", "media_id": MEDIA_ID}
    with open(os.path.join(OUTPUT_DIR, "monai_seg.json"), "w") as f:
        json.dump(result, f, indent=2)
    slicer.app.exit(1)

load_time = time.time() - t0
dims = volumeNode.GetImageData().GetDimensions()
spacing = volumeNode.GetSpacing()
print(f"[MONAI] Volume loaded: {dims[0]}x{dims[1]}x{dims[2]}, "
      f"spacing={spacing[0]:.3f}x{spacing[1]:.3f}x{spacing[2]:.3f} mm, "
      f"{load_time:.1f}s")

# ── Resolve model ID ──
logic = slicer.modules.monaiauto3dseg.widgetRepresentation().self().logic
available_models = logic.models

if not MODEL_ID:
    wholebody_models = [m for m in available_models if "whole" in m["id"].lower() and "body" in m["id"].lower()]
    if wholebody_models:
        MODEL_ID = wholebody_models[0]["id"]
    else:
        ts_models = [m for m in available_models if "totalseg" in m["id"].lower() or "17-seg" in m["id"].lower()]
        if ts_models:
            MODEL_ID = ts_models[0]["id"]
        else:
            MODEL_ID = available_models[0]["id"] if available_models else ""

if not MODEL_ID:
    print("[MONAI] ERROR: No segmentation models available")
    slicer.app.exit(1)

print(f"[MONAI] Using model: {MODEL_ID}")
model_info = None
for m in available_models:
    if m["id"] == MODEL_ID:
        model_info = m
        break

if model_info:
    print(f"[MONAI] Model title: {model_info.get('title', 'N/A')}")
    seg_names = model_info.get("segmentNames", [])
    print(f"[MONAI] Segments: {len(seg_names)} structures")

# ── Create output segmentation node ──
segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
    "vtkMRMLSegmentationNode", f"{MEDIA_ID}_monai_seg"
)

# ── Run segmentation ──
print(f"[MONAI] Starting segmentation (this may take several minutes on CPU)...")
t_seg = time.time()

try:
    logic.useStandardSegmentNames = True
    use_cpu = FORCE_CPU in ("1", "true", "True", "yes")

    segmentationTaskListInfo = logic.process(
        [volumeNode],
        segmentationNode,
        MODEL_ID,
        use_cpu,
        waitForCompletion=True,
    )

    seg_time = time.time() - t_seg
    print(f"[MONAI] Segmentation completed in {seg_time:.1f}s")

except Exception as exc:
    seg_time = time.time() - t_seg
    print(f"[MONAI] ERROR: Segmentation failed after {seg_time:.1f}s: {exc}")
    result = {"error": str(exc), "media_id": MEDIA_ID, "model_id": MODEL_ID}
    with open(os.path.join(OUTPUT_DIR, "monai_seg.json"), "w") as f:
        json.dump(result, f, indent=2)
    slicer.app.exit(1)

# ── Extract segment info ──
print("[MONAI] Extracting segment information...")
segmentation = segmentationNode.GetSegmentation()
n_segments = segmentation.GetNumberOfSegments()

segments_info = []
for i in range(n_segments):
    seg_id = segmentation.GetNthSegmentID(i)
    segment = segmentation.GetSegment(seg_id)
    name = segment.GetName()
    color = segment.GetColor()
    segments_info.append({
        "index": i,
        "id": seg_id,
        "name": name,
        "color": [round(c, 3) for c in color],
    })

print(f"[MONAI] Found {n_segments} segments:")
for seg in segments_info[:20]:
    print(f"  [{seg['index']:3d}] {seg['name']}")
if n_segments > 20:
    print(f"  ... and {n_segments - 20} more")

# ── Compute per-segment volumes using SegmentStatistics ──
print("[MONAI] Computing segment statistics...")
try:
    import SegmentStatistics
    stats_logic = SegmentStatistics.SegmentStatisticsLogic()
    stats_logic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
    stats_logic.getParameterNode().SetParameter("ScalarVolume", volumeNode.GetID())
    stats_logic.computeStatistics()

    stats_table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
    stats_logic.exportToTable(stats_table)

    for seg in segments_info:
        seg_id = seg["id"]
        for stat_name in ["LabelmapSegmentStatisticsPlugin.volume_mm3",
                          "LabelmapSegmentStatisticsPlugin.volume_cm3",
                          "ScalarVolumeSegmentStatisticsPlugin.mean",
                          "ScalarVolumeSegmentStatisticsPlugin.stdev",
                          "ScalarVolumeSegmentStatisticsPlugin.min",
                          "ScalarVolumeSegmentStatisticsPlugin.max"]:
            try:
                value = stats_logic.getStatistics()[seg_id, stat_name]
                short_name = stat_name.split(".")[-1]
                seg[short_name] = round(float(value), 3) if value is not None else None
            except (KeyError, TypeError):
                pass

    print("[MONAI] Statistics computed successfully")
except Exception as exc:
    print(f"[MONAI] WARNING: Could not compute segment statistics: {exc}")

# ── Export segmentation as labelmap NIfTI ──
print("[MONAI] Exporting segmentation...")
labelmap_path = os.path.join(OUTPUT_DIR, f"{MEDIA_ID}_monai_labelmap.nii.gz")
try:
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        segmentationNode, labelmapNode, volumeNode
    )
    slicer.util.saveNode(labelmapNode, labelmap_path)
    print(f"[MONAI] Labelmap saved: {labelmap_path}")
except Exception as exc:
    labelmap_path = None
    print(f"[MONAI] WARNING: Could not export labelmap: {exc}")

# ── Export individual segment STLs for key organs ──
stl_dir = os.path.join(OUTPUT_DIR, "segments_stl")
os.makedirs(stl_dir, exist_ok=True)
exported_stls = []

KEY_ORGANS = {
    "liver", "spleen", "pancreas", "kidney", "heart", "lung",
    "aorta", "vertebra", "femur", "skull", "brain", "stomach",
    "gallbladder", "esophagus", "trachea", "bladder",
}

for seg in segments_info:
    name_lower = seg["name"].lower()
    if any(organ in name_lower for organ in KEY_ORGANS):
        safe_name = seg["name"].replace(" ", "_").replace("/", "_")[:60]
        stl_path = os.path.join(stl_dir, f"{safe_name}.stl")
        try:
            segmentationNode.CreateClosedSurfaceRepresentation()
            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(
                stl_dir, segmentationNode, None, "STL", False, 1.0, False
            )
            exported_stls.append(stl_path)
        except Exception:
            pass
        break  # export all at once, then break

print(f"[MONAI] Exported {len(exported_stls)} segment meshes")

# ── Screenshots ──
print("[MONAI] Capturing screenshots...")
screenshots = []

try:
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

    for axis, name in enumerate(["axial", "sagittal", "coronal"]):
        sliceWidget = slicer.app.layoutManager().sliceWidget(
            ["Red", "Yellow", "Green"][axis]
        )
        if sliceWidget:
            sliceWidget.sliceLogic().FitSliceToAll()
            slicer.app.processEvents()

            screenshot_path = os.path.join(OUTPUT_DIR, f"{MEDIA_ID}_monai_{name}.png")
            slicer.util.forceRenderAllViews()
            pixmap = sliceWidget.grab()
            pixmap.save(screenshot_path)
            screenshots.append(screenshot_path)

    # 3D view
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    segmentationNode.CreateClosedSurfaceRepresentation()

    threeDWidget = slicer.app.layoutManager().threeDWidget(0)
    if threeDWidget:
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()
        threeDView.renderWindow().Render()
        slicer.app.processEvents()

        screenshot_path = os.path.join(OUTPUT_DIR, f"{MEDIA_ID}_monai_3d.png")
        rw = threeDView.renderWindow()
        rw.SetOffScreenRendering(1)
        rw.SetSize(1600, 1200)
        rw.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.SetScale(1)
        w2i.ReadFrontBufferOff()
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(screenshot_path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        screenshots.append(screenshot_path)

    print(f"[MONAI] {len(screenshots)} screenshots saved")
except Exception as exc:
    print(f"[MONAI] WARNING: Screenshot capture failed: {exc}")

# ── Build result JSON ──
total_time = time.time() - t0

organ_volumes = {}
for seg in segments_info:
    vol = seg.get("volume_mm3") or seg.get("volume_cm3")
    if vol and vol > 0:
        organ_volumes[seg["name"]] = {
            "volume_mm3": seg.get("volume_mm3"),
            "volume_cm3": seg.get("volume_cm3"),
            "mean_hu": seg.get("mean"),
            "std_hu": seg.get("stdev"),
            "min_hu": seg.get("min"),
            "max_hu": seg.get("max"),
        }

result = {
    "media_id": MEDIA_ID,
    "model_id": MODEL_ID,
    "model_title": model_info.get("title", "") if model_info else "",
    "input_path": INPUT_PATH,
    "volume_dimensions": list(dims),
    "volume_spacing_mm": [round(s, 4) for s in spacing],
    "n_segments": n_segments,
    "segments": segments_info,
    "organ_volumes": organ_volumes,
    "labelmap_path": labelmap_path,
    "screenshots": screenshots,
    "segmentation_time_s": round(seg_time, 1),
    "total_time_s": round(total_time, 1),
    "platform": platform.system(),
    "cpu_only": FORCE_CPU in ("1", "true", "True", "yes"),
    "analysis_type": "monai_auto3dseg",
}

result_path = os.path.join(OUTPUT_DIR, "monai_seg.json")
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n[MONAI] ═══════════════════════════════════════")
print(f"[MONAI] Segmentation complete!")
print(f"[MONAI] Segments: {n_segments}")
print(f"[MONAI] Organs with volume data: {len(organ_volumes)}")
print(f"[MONAI] Segmentation time: {seg_time:.1f}s")
print(f"[MONAI] Total time: {total_time:.1f}s")
print(f"[MONAI] Results: {result_path}")
print(f"[MONAI] ═══════════════════════════════════════")

slicer.app.exit(0)
