#!/usr/bin/env python3
"""
Interactive Slicer session with LLM-in-the-loop.

Runs a multi-step loop where:
  1. Slicer takes a screenshot of the current state
  2. LLM sees the screenshot + context and decides the next action
  3. The action is translated into Slicer Python code
  4. Slicer executes the code and takes a new screenshot
  5. Repeat until the LLM says the goal is achieved

The Slicer session persists across steps — each step builds on the last.

Usage:
    python3 slicer_interactive.py \
        --ply /path/to/mesh.ply \
        --goal "Place 12 anatomical landmarks on this primate skull" \
        --max-steps 10 \
        --output-dir /tmp/slicer_interactive
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import load_dotenv, call_llm, SLICER_BIN, AUTORESEARCHCLAW_HOME, get_model_for_tier

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("SlicerInteractive")

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Slicer session manager — persists state across steps
# ---------------------------------------------------------------------------


class SlicerSession:
    """Manages an iterative Slicer session with persistent state.

    Each step writes a Slicer Python script that:
      - Loads the session state from a JSON file
      - Executes the new action
      - Takes a screenshot
      - Saves updated state back to JSON
    """

    def __init__(self, ply_path: str, output_dir: str, media_id: str = "unknown"):
        self.ply_path = ply_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.media_id = media_id
        self.state_file = self.output_dir / "session_state.json"
        self.step_count = 0
        self.history: list[dict] = []

        # Initialize state
        self._write_state({
            "ply_path": ply_path,
            "media_id": media_id,
            "loaded": False,
            "landmarks": [],
            "measurements": {},
            "screenshots": [],
            "notes": [],
        })

    def _write_state(self, state: dict):
        self.state_file.write_text(json.dumps(state, indent=2, default=str))

    def _read_state(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {}

    def run_step(self, action_code: str, step_name: str = "") -> dict:
        """Execute a Slicer step and capture the result."""
        self.step_count += 1
        step_name = step_name or f"step_{self.step_count}"
        screenshot_path = str(self.output_dir / f"{step_name}.png")

        # Build the Slicer script that:
        # 1. Loads the mesh (first time) or restores state
        # 2. Executes the action code
        # 3. Takes a screenshot
        # 4. Saves state
        script = self._build_step_script(action_code, screenshot_path, step_name)
        script_path = self.output_dir / f"{step_name}.py"
        script_path.write_text(script)

        # Run in Slicer
        env = os.environ.copy()
        env["SLICER_SESSION_STATE"] = str(self.state_file)

        log.info("Step %d (%s): executing in Slicer...", self.step_count, step_name)
        try:
            result = subprocess.run(
                [SLICER_BIN, "--no-splash", "--python-script", str(script_path)],
                capture_output=True, text=True, timeout=60, env=env,
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            # Filter noise
            output = "\n".join(
                l for l in output.split("\n")
                if not l.startswith("Error #1") and not l.startswith("libpng")
                and not l.startswith("Switch to module")
            ).strip()
        except subprocess.TimeoutExpired:
            success = False
            output = "Slicer timed out after 120s"
        except Exception as exc:
            success = False
            output = str(exc)

        # Read updated state
        state = self._read_state()

        step_result = {
            "step": self.step_count,
            "name": step_name,
            "success": success,
            "screenshot": screenshot_path if Path(screenshot_path).exists() else None,
            "output": output[-1000:],
            "state": state,
        }
        self.history.append(step_result)

        log.info("Step %d: %s (screenshot=%s)",
                 self.step_count,
                 "SUCCESS" if success else "FAILED",
                 "yes" if step_result["screenshot"] else "no")

        return step_result

    def _build_step_script(self, action_code: str, screenshot_path: str, step_name: str) -> str:
        return f'''
import slicer
import vtk
import numpy as np
import json
import os

STATE_FILE = os.environ.get("SLICER_SESSION_STATE", "{self.state_file}")
SCREENSHOT_PATH = "{screenshot_path}"
STEP_NAME = "{step_name}"

# Load state
with open(STATE_FILE) as f:
    state = json.load(f)

# Load mesh if not already loaded
if not state.get("loaded"):
    print(f"Loading mesh: {{state['ply_path']}}")
    success, modelNode = slicer.util.loadModel(state["ply_path"], returnNode=True)
    if not success:
        state["error"] = "Failed to load mesh"
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        slicer.app.exit(1)

    mesh = modelNode.GetMesh()
    bounds = [0]*6
    modelNode.GetBounds(bounds)
    state["loaded"] = True
    state["vertices"] = mesh.GetNumberOfPoints()
    state["faces"] = mesh.GetNumberOfCells()
    state["bounds"] = bounds
    state["center"] = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
    state["extent"] = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
    print(f"Loaded: {{state['vertices']:,}} vertices")
else:
    # Reload model
    modelNode = slicer.util.getNode("*")
    if modelNode is None:
        success, modelNode = slicer.util.loadModel(state["ply_path"], returnNode=True)
    mesh = modelNode.GetMesh() if modelNode else None

# Get or create landmarks node
markupNode = slicer.mrmlScene.GetFirstNodeByName("InteractiveLandmarks")
if markupNode is None:
    markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "InteractiveLandmarks")
    md = markupNode.GetDisplayNode()
    if md:
        md.SetGlyphScale(2.5)
        md.SetTextScale(3.5)
        md.SetSelectedColor(1, 0.3, 0.3)

# Make these available to the action code
center = state.get("center", [0, 0, 0])
bounds = state.get("bounds", [0]*6)
extent = state.get("extent", [1, 1, 1])

# ── Execute action code ──
print(f"Executing: {{STEP_NAME}}")
try:
{_indent(action_code, 4)}
    print("Action completed successfully")
except Exception as exc:
    print(f"Action error: {{exc}}")
    state.setdefault("notes", []).append(f"Step {{STEP_NAME}} error: {{exc}}")

# ── Update landmark state ──
landmarks = []
for i in range(markupNode.GetNumberOfControlPoints()):
    pos = [0, 0, 0]
    markupNode.GetNthControlPointPosition(i, pos)
    name = markupNode.GetNthControlPointLabel(i)
    landmarks.append({{"name": name, "position": [round(float(c), 3) for c in pos]}})
state["landmarks"] = landmarks

# ── Take screenshot ──
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
tw = slicer.app.layoutManager().threeDWidget(0)
tv = tw.threeDView()
rw = tv.renderWindow()
rw.SetOffScreenRendering(1)
rw.SetSize(1600, 1200)
renderer = rw.GetRenderers().GetFirstRenderer()
renderer.SetBackground(0.05, 0.05, 0.1)
camera = renderer.GetActiveCamera()
camera.SetPosition(center[0]+80, center[1]-80, center[2]+60)
camera.SetFocalPoint(*center)
camera.SetViewUp(0, 0, 1)
renderer.ResetCameraClippingRange()
rw.Render()

w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(rw)
w2i.SetScale(1)
w2i.ReadFrontBufferOff()
w2i.Update()
writer = vtk.vtkPNGWriter()
writer.SetFileName(SCREENSHOT_PATH)
writer.SetInputConnection(w2i.GetOutputPort())
writer.Write()
state.setdefault("screenshots", []).append(SCREENSHOT_PATH)

# ── Save state ──
with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2, default=str)

print(f"State saved: {{len(landmarks)}} landmarks, screenshot: {{SCREENSHOT_PATH}}")
slicer.app.exit(0)
'''


def _indent(code: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in code.split("\n"))


# ---------------------------------------------------------------------------
# LLM vision + decision loop
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert morphometrics researcher controlling 3D Slicer interactively.

You are given:
- A screenshot of the current 3D view in Slicer
- The current state (landmarks placed, measurements, mesh info)
- The research goal

You must decide the NEXT ACTION to take. Output a JSON object:
{
  "done": false,
  "reasoning": "Why I'm taking this action",
  "action_name": "short_name_for_this_step",
  "slicer_code": "Python code to execute inside Slicer (has access to: modelNode, mesh, markupNode, center, bounds, extent, state, vtk, np, slicer)",
  "camera_position": [x_offset, y_offset, z_offset]  // optional: reposition camera
}

When the goal is achieved, set "done": true and include a "summary" field.

Available operations in slicer_code:
- markupNode.AddControlPoint(vtk.vtkVector3d(x, y, z), "landmark_name")  # place landmark
- markupNode.RemoveNthControlPoint(index)  # remove landmark
- modelNode.GetDisplayNode().SetColor(r, g, b)  # change color
- modelNode.GetDisplayNode().SetOpacity(0.5)  # transparency
- vtk.vtkMassProperties(), vtk.vtkCurvatures(), vtk.vtkConnectivityFilter()
- numpy operations on mesh points
- Any VTK filter or Slicer Python API

Rules:
- Each step should do ONE focused thing
- Use the screenshot to verify previous actions worked
- Place landmarks by finding anatomical features on the mesh surface
- Use a vtkPointLocator to snap points to the mesh surface
- Keep code under 30 lines
- Reference specific coordinates from the bounds/center info"""


def _encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _ask_llm_for_action(goal: str, state: dict, screenshot_path: str | None,
                         history: list[dict], step: int) -> dict:
    """Ask the LLM what to do next based on the current state and screenshot."""
    state_summary = {
        "vertices": state.get("vertices"),
        "bounds": state.get("bounds"),
        "center": state.get("center"),
        "extent": state.get("extent"),
        "landmarks_count": len(state.get("landmarks", [])),
        "landmarks": state.get("landmarks", [])[:20],
        "measurements": state.get("measurements", {}),
        "notes": state.get("notes", [])[-5:],
    }

    history_summary = []
    for h in history[-5:]:
        history_summary.append({
            "step": h["step"], "name": h["name"],
            "success": h["success"],
            "output": h["output"][-200:] if h.get("output") else "",
        })

    user_content: list[dict] = [
        {"type": "text", "text": (
            f"GOAL: {goal}\n\n"
            f"Step {step}/{step} (previous steps shown below)\n\n"
            f"Current state:\n{json.dumps(state_summary, indent=2)}\n\n"
            f"History:\n{json.dumps(history_summary, indent=2)}\n\n"
            f"What should I do next? Return a JSON object with done, reasoning, "
            f"action_name, and slicer_code fields."
        )},
    ]

    if screenshot_path and Path(screenshot_path).exists():
        b64 = _encode_image(screenshot_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    # Use vision model (gpt-4o) for image understanding
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"done": True, "reasoning": "No API key", "summary": "Cannot proceed without API key"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=2000,
        )
        content = (response.choices[0].message.content or "").strip()

        # Parse JSON from response
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

        # Try extracting JSON from surrounding text
        s = content.find("{")
        e = content.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(content[s:e + 1])
            except json.JSONDecodeError:
                pass

        # Last resort: build a simple action from the text
        log.warning("Could not parse LLM JSON, attempting text extraction")
        if "done" in content.lower() and ("complete" in content.lower() or "achieved" in content.lower()):
            return {"done": True, "reasoning": "Goal appears complete", "summary": content[:500]}

        return {
            "done": False,
            "reasoning": "LLM response was not valid JSON, using fallback",
            "action_name": f"fallback_step",
            "slicer_code": "# No-op: LLM response could not be parsed\nprint('Fallback step')",
        }
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        return {"done": True, "reasoning": f"LLM error: {exc}", "summary": "Failed"}


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------


def run_interactive_session(
    ply_path: str,
    goal: str,
    output_dir: str,
    media_id: str = "unknown",
    max_steps: int = 10,
) -> dict:
    """Run an iterative LLM-guided Slicer session."""
    session = SlicerSession(ply_path, output_dir, media_id)

    log.info("=" * 60)
    log.info("Interactive Slicer Session")
    log.info("Goal: %s", goal)
    log.info("Mesh: %s", ply_path)
    log.info("Max steps: %d", max_steps)
    log.info("=" * 60)

    # Step 0: Load mesh and take initial screenshot
    init_result = session.run_step(
        "# Initial load — no action needed, just screenshot",
        step_name="00_initial_load",
    )

    if not init_result["success"]:
        return {"success": False, "error": "Failed to load mesh", "history": session.history}

    # Iterative loop
    for step in range(1, max_steps + 1):
        state = session._read_state()
        last_screenshot = session.history[-1].get("screenshot") if session.history else None

        # Ask LLM what to do
        log.info("Step %d: asking LLM for next action...", step)
        action = _ask_llm_for_action(goal, state, last_screenshot, session.history, step)

        if action.get("done"):
            log.info("Goal achieved: %s", action.get("reasoning", ""))
            break

        action_name = action.get("action_name", f"step_{step}")
        slicer_code = action.get("slicer_code", "# no-op")
        log.info("Action: %s — %s", action_name, action.get("reasoning", "")[:80])

        # Execute in Slicer
        step_result = session.run_step(slicer_code, step_name=f"{step:02d}_{action_name}")

        if not step_result["success"]:
            log.warning("Step %d failed, continuing...", step)

    # Final state
    final_state = session._read_state()

    # Save FCSV if landmarks were placed
    if final_state.get("landmarks"):
        fcsv_content = _build_fcsv(final_state["landmarks"])
        fcsv_path = session.output_dir / f"{media_id}_interactive_landmarks.fcsv"
        fcsv_path.write_text(fcsv_content)
        log.info("Saved %d landmarks to %s", len(final_state["landmarks"]), fcsv_path)

    # Save session log
    session_log = {
        "goal": goal,
        "ply_path": ply_path,
        "media_id": media_id,
        "steps_completed": session.step_count,
        "max_steps": max_steps,
        "final_landmarks": final_state.get("landmarks", []),
        "final_measurements": final_state.get("measurements", {}),
        "history": [
            {k: v for k, v in h.items() if k != "state"}
            for h in session.history
        ],
    }
    log_path = session.output_dir / "interactive_session.json"
    log_path.write_text(json.dumps(session_log, indent=2, default=str))

    log.info("Session complete: %d steps, %d landmarks",
             session.step_count, len(final_state.get("landmarks", [])))

    return {
        "success": True,
        "steps": session.step_count,
        "landmarks": final_state.get("landmarks", []),
        "measurements": final_state.get("measurements", {}),
        "screenshots": [h.get("screenshot") for h in session.history if h.get("screenshot")],
        "session_log": str(log_path),
    }


def _build_fcsv(landmarks: list[dict]) -> str:
    """Build a Slicer FCSV file from landmark dicts."""
    lines = [
        "# Markups fiducial file version = 4.11",
        "# CoordinateSystem = LPS",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID",
    ]
    for i, lm in enumerate(landmarks):
        pos = lm["position"]
        name = lm["name"]
        lines.append(f"{i},{pos[0]},{pos[1]},{pos[2]},0,0,0,1,1,1,0,{name},,")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive LLM-guided Slicer session")
    parser.add_argument("--ply", required=True, help="Path to PLY/STL/OBJ mesh file")
    parser.add_argument("--goal", required=True, help="What to achieve (e.g. 'Place 12 cranial landmarks')")
    parser.add_argument("--media-id", default="unknown", help="MorphoSource media ID")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum interaction steps")
    parser.add_argument("--output-dir", default="/tmp/slicer_interactive", help="Output directory")
    args = parser.parse_args()

    result = run_interactive_session(
        ply_path=args.ply,
        goal=args.goal,
        output_dir=args.output_dir,
        media_id=args.media_id,
        max_steps=args.max_steps,
    )

    print(json.dumps({k: v for k, v in result.items() if k != "screenshots"}, indent=2, default=str))


if __name__ == "__main__":
    main()
