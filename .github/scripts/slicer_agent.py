"""
Slicer Agent — LLM tool-calling loop running INSIDE 3D Slicer.

This script runs as a --python-script inside Slicer. It loads a mesh,
then enters a loop where it:
  1. Takes a screenshot (IMAGE)
  2. Sends it to GPT-4o vision
  3. Gets back a tool call (IMAGE, MOVE, ZOOM, LANDMARK, MEASURE, DONE)
  4. Executes the tool
  5. Repeats

All tools execute natively in the Slicer Python environment with full
access to VTK, the scene, markups, camera, etc.

Config via environment variables:
  SLICER_PLY_PATH, SLICER_OUTPUT_DIR, SLICER_MEDIA_ID,
  SLICER_GOAL, SLICER_MAX_STEPS, OPENAI_API_KEY
"""

import slicer
import vtk
import numpy as np
import json
import os
import sys
import base64
import urllib.request
import urllib.error
import time

# ── Config ──
PLY_PATH = os.environ.get("SLICER_PLY_PATH", "")
OUTPUT_DIR = os.environ.get("SLICER_OUTPUT_DIR", "/tmp/slicer_agent")
MEDIA_ID = os.environ.get("SLICER_MEDIA_ID", "unknown")
GOAL = os.environ.get("SLICER_GOAL", "Examine this specimen and place anatomical landmarks")
MAX_STEPS = int(os.environ.get("SLICER_MAX_STEPS", "8"))
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Fallback: config file
if not PLY_PATH:
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_slicer_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        PLY_PATH = cfg.get("ply_path", "")
        OUTPUT_DIR = cfg.get("output_dir", OUTPUT_DIR)
        MEDIA_ID = cfg.get("media_id", MEDIA_ID)
        GOAL = cfg.get("goal", GOAL)
        MAX_STEPS = cfg.get("max_steps", MAX_STEPS)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"SlicerAgent: {GOAL}")
print(f"Mesh: {PLY_PATH}")
print(f"Max steps: {MAX_STEPS}")

# ── Load mesh ──
success, modelNode = slicer.util.loadModel(PLY_PATH, returnNode=True)
if not success or not modelNode:
    print("FATAL: Failed to load mesh")
    json.dump({"error": "Failed to load mesh"}, open(os.path.join(OUTPUT_DIR, "result.json"), "w"))
    slicer.app.exit(1)

mesh = modelNode.GetMesh()
bounds = [0]*6
modelNode.GetBounds(bounds)
center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
extent = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
max_dim = max(extent)
print(f"Loaded: {mesh.GetNumberOfPoints():,} vertices")

# ── Setup 3D view ──
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
threeDWidget = slicer.app.layoutManager().threeDWidget(0)
threeDView = threeDWidget.threeDView()
renderWindow = threeDView.renderWindow()
renderWindow.SetOffScreenRendering(1)
renderWindow.SetSize(1200, 900)
renderer = renderWindow.GetRenderers().GetFirstRenderer()
renderer.SetBackground(0.05, 0.05, 0.1)
camera = renderer.GetActiveCamera()

# Initial camera position
cam_dist = max_dim * 2.5
camera.SetPosition(center[0] + cam_dist * 0.6, center[1] - cam_dist * 0.6, center[2] + cam_dist * 0.4)
camera.SetFocalPoint(*center)
camera.SetViewUp(0, 0, 1)
renderer.ResetCameraClippingRange()

# ── Landmarks node ──
markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "AgentLandmarks")
md = markupNode.GetDisplayNode()
if md:
    md.SetGlyphScale(max_dim * 0.03)
    md.SetTextScale(max_dim * 0.04)
    md.SetSelectedColor(1, 0.2, 0.2)

# Point locator for surface snapping
locator = vtk.vtkPointLocator()
locator.SetDataSet(mesh)
locator.BuildLocator()

# ── Tool implementations ──

def tool_image(view_name="current"):
    """Take a screenshot from the current or specified view."""
    if view_name != "current":
        views = {
            "anterior":  (0, -1, 0.1),
            "posterior":  (0, 1, 0.1),
            "lateral_L":  (1, 0, 0.1),
            "lateral_R":  (-1, 0, 0.1),
            "dorsal":    (0, 0, 1),
            "ventral":   (0, 0, -1),
            "oblique":   (0.6, -0.6, 0.4),
        }
        if view_name in views:
            d = views[view_name]
            camera.SetPosition(center[0]+d[0]*cam_dist, center[1]+d[1]*cam_dist, center[2]+d[2]*cam_dist)
            camera.SetFocalPoint(*center)
            if view_name == "dorsal":
                camera.SetViewUp(0, -1, 0)
            elif view_name == "ventral":
                camera.SetViewUp(0, 1, 0)
            else:
                camera.SetViewUp(0, 0, 1)
            renderer.ResetCameraClippingRange()

    renderWindow.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renderWindow)
    w2i.SetScale(1)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    fpath = os.path.join(OUTPUT_DIR, f"step_{step_count:02d}_{view_name}.png")
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fpath)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    screenshots.append(fpath)
    return fpath


def tool_move(view_name):
    """Move camera to a named view."""
    tool_image(view_name)
    return f"Moved to {view_name} view"


def tool_zoom(factor):
    """Zoom in (factor > 1) or out (factor < 1)."""
    camera.Zoom(float(factor))
    renderer.ResetCameraClippingRange()
    renderWindow.Render()
    return f"Zoomed by {factor}x"


def tool_landmark(name, x, y, z):
    """Place a landmark at (x,y,z), snapped to mesh surface."""
    point_id = locator.FindClosestPoint(float(x), float(y), float(z))
    snapped = mesh.GetPoint(point_id)
    markupNode.AddControlPoint(vtk.vtkVector3d(*snapped), str(name))
    return f"Placed '{name}' at [{snapped[0]:.2f}, {snapped[1]:.2f}, {snapped[2]:.2f}]"


def tool_remove_landmark(name):
    """Remove a landmark by name."""
    for i in range(markupNode.GetNumberOfControlPoints()):
        if markupNode.GetNthControlPointLabel(i) == name:
            markupNode.RemoveNthControlPoint(i)
            return f"Removed '{name}'"
    return f"Landmark '{name}' not found"


def tool_measure(name1, name2):
    """Measure distance between two landmarks."""
    pts = {}
    for i in range(markupNode.GetNumberOfControlPoints()):
        label = markupNode.GetNthControlPointLabel(i)
        pos = [0, 0, 0]
        markupNode.GetNthControlPointPosition(i, pos)
        pts[label] = np.array(pos)
    if name1 not in pts or name2 not in pts:
        return f"Landmark(s) not found: {name1}, {name2}"
    d = float(np.linalg.norm(pts[name1] - pts[name2]))
    measurements[f"{name1}_to_{name2}"] = round(d, 3)
    return f"Distance {name1} -> {name2} = {d:.3f} mm"


def tool_opacity(value):
    """Set mesh opacity (0.0 to 1.0)."""
    modelNode.GetDisplayNode().SetOpacity(float(value))
    renderWindow.Render()
    return f"Opacity set to {value}"


def tool_color(r, g, b):
    """Set mesh color."""
    modelNode.GetDisplayNode().SetColor(float(r), float(g), float(b))
    renderWindow.Render()
    return f"Color set to ({r}, {g}, {b})"


def get_landmarks_state():
    """Get current landmarks as a list."""
    lms = []
    for i in range(markupNode.GetNumberOfControlPoints()):
        pos = [0, 0, 0]
        markupNode.GetNthControlPointPosition(i, pos)
        lms.append({"name": markupNode.GetNthControlPointLabel(i),
                     "position": [round(c, 3) for c in pos]})
    return lms


# ── OpenAI API call (direct HTTP, no openai package needed) ──

def call_vision_api(image_path, state_text):
    """Send screenshot + state to GPT-4o and get tool call back."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": state_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]},
    ]

    payload = json.dumps({
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 1000,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        print(f"  LLM response: {content[:100]}...")
        return parse_tool_call(content)
    except Exception as exc:
        print(f"  API error: {exc}")
        return {"tool": "DONE", "reason": f"API error: {exc}"}


SYSTEM_PROMPT = """You are an expert morphometrics researcher controlling 3D Slicer.
You see a screenshot of a 3D specimen and must take actions to achieve the goal.

Respond with EXACTLY ONE JSON tool call. No other text.

Tools:
  {"tool":"IMAGE","view":"anterior|posterior|lateral_L|lateral_R|dorsal|ventral|oblique"}
  {"tool":"ZOOM","factor":1.5}
  {"tool":"LANDMARK","name":"nasion","x":0,"y":-10,"z":5}
  {"tool":"REMOVE_LANDMARK","name":"nasion"}
  {"tool":"MEASURE","from":"nasion","to":"bregma"}
  {"tool":"OPACITY","value":0.5}
  {"tool":"DONE","reason":"Goal achieved"}

CRITICAL RULES:
- Do NOT spend more than 2 steps on IMAGE/MOVE. Start placing landmarks quickly.
- Use the bounds and center coordinates to estimate landmark positions.
- Coordinates use LPS convention: Left(+x), Posterior(+y), Superior(+z).
- Landmarks snap to the nearest mesh surface automatically.
- The center of the mesh is at the coordinates shown in the state.
- For a skull: nasion is anterior-superior midline, bregma is dorsal midline,
  prosthion is anterior-inferior midline, inion is posterior midline.
- After placing 2+ landmarks, take an IMAGE to verify, then continue.
- Call DONE only after the goal is fully achieved."""


def parse_tool_call(content):
    """Parse the LLM response into a tool call dict."""
    content = content.strip()
    if "```" in content:
        for part in content.split("```")[1::2]:
            c = part[4:].strip() if part.startswith("json") else part.strip()
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    s, e = content.find("{"), content.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(content[s:e+1])
        except json.JSONDecodeError:
            pass
    return {"tool": "DONE", "reason": "Could not parse response"}


# ── Main loop ──

screenshots = []
measurements = {}
step_count = 0
history = []

print(f"\n{'='*50}")
print(f"Starting agent loop (max {MAX_STEPS} steps)")
print(f"{'='*50}\n")

# Take initial screenshot
step_count = 1
img = tool_image("oblique")
print(f"Step 0: Initial screenshot from oblique view")

for step in range(1, MAX_STEPS + 1):
    step_count = step + 1

    # Build state text
    landmarks = get_landmarks_state()
    state_text = (
        f"GOAL: {GOAL}\n\n"
        f"Step {step}/{MAX_STEPS}\n"
        f"Mesh: {mesh.GetNumberOfPoints():,} vertices\n"
        f"Bounds: {[round(b,1) for b in bounds]}\n"
        f"Center: {[round(c,1) for c in center]}\n"
        f"Extent: {[round(e,1) for e in extent]} mm\n"
        f"Landmarks placed ({len(landmarks)}):\n"
    )
    for lm in landmarks:
        state_text += f"  - {lm['name']}: {lm['position']}\n"
    if measurements:
        state_text += f"Measurements:\n"
        for k, v in measurements.items():
            state_text += f"  - {k}: {v} mm\n"
    state_text += f"\nPrevious actions:\n"
    for h in history[-3:]:
        state_text += f"  - Step {h['step']}: {h['tool']} -> {h['result'][:80]}\n"

    # Call LLM
    last_img = screenshots[-1] if screenshots else img
    print(f"\nStep {step}: Asking LLM...")
    action = call_vision_api(last_img, state_text)
    tool_name = action.get("tool", "DONE")
    print(f"  Tool: {tool_name}")

    # Execute tool
    result = ""
    if tool_name == "IMAGE":
        view = action.get("view", "current")
        fpath = tool_image(view)
        result = f"Screenshot taken: {view}"
    elif tool_name == "MOVE":
        view = action.get("view", "anterior")
        result = tool_move(view)
    elif tool_name == "ZOOM":
        factor = action.get("factor", 1.0)
        result = tool_zoom(factor)
    elif tool_name == "LANDMARK":
        name = action.get("name", f"lm_{step}")
        x, y, z = action.get("x", 0), action.get("y", 0), action.get("z", 0)
        result = tool_landmark(name, x, y, z)
    elif tool_name == "REMOVE_LANDMARK":
        result = tool_remove_landmark(action.get("name", ""))
    elif tool_name == "MEASURE":
        result = tool_measure(action.get("from", ""), action.get("to", ""))
    elif tool_name == "OPACITY":
        result = tool_opacity(action.get("value", 1.0))
    elif tool_name == "COLOR":
        result = tool_color(action.get("r", 0.8), action.get("g", 0.7), action.get("b", 0.6))
    elif tool_name == "DONE":
        print(f"  DONE: {action.get('reason', 'Goal achieved')}")
        history.append({"step": step, "tool": "DONE", "result": action.get("reason", "")})
        break
    else:
        result = f"Unknown tool: {tool_name}"

    print(f"  Result: {result}")
    history.append({"step": step, "tool": tool_name, "result": result, "action": action})

# ── Final output ──
print(f"\n{'='*50}")
print(f"Session complete: {len(history)} steps")
print(f"Landmarks: {len(get_landmarks_state())}")
print(f"Measurements: {len(measurements)}")
print(f"Screenshots: {len(screenshots)}")
print(f"{'='*50}\n")

# Save landmarks as FCSV
if get_landmarks_state():
    fcsv_path = os.path.join(OUTPUT_DIR, f"{MEDIA_ID}_landmarks.fcsv")
    slicer.util.saveNode(markupNode, fcsv_path)
    print(f"Saved landmarks: {fcsv_path}")

# Take final multi-view screenshots
for view in ["anterior", "lateral_L", "dorsal", "oblique"]:
    tool_image(view)

# Save result
result = {
    "media_id": MEDIA_ID,
    "goal": GOAL,
    "steps": len(history),
    "landmarks": get_landmarks_state(),
    "measurements": measurements,
    "screenshots": screenshots,
    "history": history,
    "vertices": mesh.GetNumberOfPoints(),
    "bounds": [round(b, 3) for b in bounds],
}
with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"Result saved: {os.path.join(OUTPUT_DIR, 'result.json')}")
slicer.app.exit(0)
