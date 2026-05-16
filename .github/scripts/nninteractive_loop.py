"""
LLM-in-the-loop iterative segmentation with nnInteractive.

This is AutoResearchClaw's "paint" loop — the agent's eyes-and-hands wrapper
around `nnInteractive_segment.Segmenter`. Each step:

    1. Render orthogonal previews of the current mask over the volume
    2. Send the screenshots + state to the LLM (vision model)
    3. Parse a tool call: ADD_POINT / ADD_BBOX / RESET / DONE
    4. Apply it via nnInteractive — segmentation refines incrementally
    5. Repeat until DONE or `max_steps` reached

Outputs a final NIfTI labelmap, the prompt history, and a JSON summary
suitable for posting to a GitHub issue or feeding back into research_agent.

Usage:

    python3 nninteractive_loop.py \
        --input /path/to/volume.nii.gz \
        --goal  "Segment the cranial cavity" \
        --media-id 000769445 \
        --output-dir /tmp/nni_loop \
        --max-steps 12
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow this file to be run with the nnInteractive venv's Python (where
# `_helpers` is not normally on sys.path because it lives in the parent
# autoresearchclaw env). We tolerate either layout.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from nninteractive_segment import (  # noqa: E402
    NNInteractiveUnavailable, Segmenter, SegmenterConfig,
)

try:
    from _helpers import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    # _helpers is part of the larger AutoResearchClaw env; running standalone
    # in the nnInteractive venv just means we read OPENAI_API_KEY from the
    # ambient env directly.
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("nni_loop")


SYSTEM_PROMPT = """You are a senior morphometrics researcher controlling
nnInteractive, a 3D promptable segmentation model, to extract a structure
from a 3D medical/CT volume.

You are shown three orthogonal screenshots (axial / coronal / sagittal)
of the volume with the *current* segmentation mask overlaid in red. Each
screenshot has voxel coordinates printed alongside it.

You must decide ONE next action. Respond with EXACTLY one JSON object,
no other text:

{"tool":"ADD_POINT","x":INT,"y":INT,"z":INT,"positive":TRUE_OR_FALSE,"label":"...","reason":"..."}
{"tool":"ADD_BBOX","x":[X1,X2],"y":[Y1,Y2],"z":[Z1,Z2],"positive":TRUE_OR_FALSE,"label":"...","reason":"..."}
{"tool":"RESET","reason":"start over because the mask is wrong"}
{"tool":"DONE","reason":"mask now matches the goal","summary":"<2-3 sentence summary>"}

Rules:
- Coordinates are voxel indices in (x, y, z) order matching the printed
  axes. The volume's shape is given to you as image_shape_xyz.
- Bounding boxes must be 2D for nnInteractive: one of x/y/z must span a
  single voxel, e.g. z:[42,43].
- A positive point adds tissue similar to that voxel; negative removes.
- After 1–2 confirmation steps, prefer DONE. Avoid loops.
- If the mask drifts badly, prefer RESET over many corrective negatives.
- The goal you must achieve is shown in the user message."""


@dataclass
class StepRecord:
    step: int
    tool: str
    args: dict
    result: str
    voxel_count: int
    screenshots: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vision LLM call (uses the openai package if available, else raw HTTP).
# We keep the dependency surface minimal because this script is run inside
# the nnInteractive venv, which only has torch + nnInteractive by default.
# ---------------------------------------------------------------------------


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _call_vision_llm(api_key: str, model: str, system: str,
                     state_text: str, image_paths: list[str],
                     max_tokens: int = 800) -> Optional[str]:
    """Vision model call. Returns the LLM text or None on failure."""
    user_content: list[dict] = [{"type": "text", "text": state_text}]
    for p in image_paths:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_b64(p)}"},
        })

    # Prefer the openai SDK if available
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except ImportError:
        pass
    except Exception as exc:
        log.error("openai SDK call failed: %s", exc)
        return None

    # Fallback: raw urllib HTTP
    import urllib.request
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        log.error("Raw HTTP LLM call failed: %s", exc)
        return None


def _parse_action(text: str) -> dict:
    """Extract a single JSON action from the LLM's response."""
    if not text:
        return {"tool": "DONE", "reason": "empty LLM response"}
    if "```" in text:
        for chunk in text.split("```")[1::2]:
            body = chunk[4:].strip() if chunk.startswith("json") else chunk.strip()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if 0 <= s < e:
        try:
            return json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            pass
    log.warning("Could not parse LLM response as JSON; treating as DONE")
    return {"tool": "DONE", "reason": "unparseable LLM response", "raw": text[:300]}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_loop(input_path: str, goal: str, output_dir: str,
             media_id: str = "unknown", max_steps: int = 12,
             vision_model: str = "") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "success": False,
            "error": "OPENAI_API_KEY not set; the nnInteractive paint loop "
                     "needs a vision model to choose prompts.",
        }
    vision_model = vision_model or os.environ.get(
        "NNINTERACTIVE_VISION_MODEL", "gpt-4o"
    )

    cfg = SegmenterConfig(input_path=input_path, output_dir=str(output),
                          media_id=media_id)
    try:
        seg = Segmenter(cfg)
    except NNInteractiveUnavailable as exc:
        log.error("%s", exc)
        return {"success": False, "error": str(exc)}

    log.info("=" * 60)
    log.info("nnInteractive paint loop")
    log.info("Goal:     %s", goal)
    log.info("Volume:   %s", input_path)
    log.info("Shape:    %s (z, y, x)", seg.image_shape_zyx)
    log.info("Spacing:  %s mm (x, y, z)", seg.sitk_image.GetSpacing())
    log.info("Max steps:%d", max_steps)
    log.info("Vision model: %s", vision_model)
    log.info("=" * 60)

    # Initial preview (no prompts yet → empty mask)
    initial = seg.save_orthogonal_previews(name_prefix=f"{media_id}_step00")

    history: list[StepRecord] = []
    z, y, x = seg.image_shape_zyx
    image_shape_xyz = [x, y, z]

    for step in range(1, max_steps + 1):
        last_screens = (
            history[-1].screenshots if history else initial
        )
        state_text = _build_state_text(
            goal=goal,
            step=step,
            max_steps=max_steps,
            image_shape_xyz=image_shape_xyz,
            spacing_xyz=list(seg.sitk_image.GetSpacing()),
            voxel_count=seg.voxel_count(),
            volume_mm3=seg.volume_mm3(),
            history=history,
        )

        log.info("Step %d/%d — asking LLM (%d voxels currently in mask)",
                 step, max_steps, seg.voxel_count())
        text = _call_vision_llm(api_key, vision_model, SYSTEM_PROMPT,
                                state_text, last_screens, max_tokens=800)
        action = _parse_action(text or "")
        tool = action.get("tool", "DONE").upper()
        log.info("  Tool: %s — %s", tool, action.get("reason", "")[:120])

        # Execute the tool
        if tool == "DONE":
            history.append(StepRecord(
                step=step, tool="DONE", args=action,
                result=action.get("reason", "done"),
                voxel_count=seg.voxel_count(),
            ))
            break

        if tool == "RESET":
            seg.reset_segment()
            history.append(StepRecord(
                step=step, tool="RESET", args=action,
                result="segment reset",
                voxel_count=seg.voxel_count(),
            ))
        elif tool == "ADD_POINT":
            try:
                seg.add_point(
                    int(action["x"]), int(action["y"]), int(action["z"]),
                    positive=bool(action.get("positive", True)),
                    label=str(action.get("label", "")),
                )
                result = (
                    f"point ({action['x']},{action['y']},{action['z']}) "
                    f"{'pos' if action.get('positive', True) else 'neg'}"
                )
            except Exception as exc:
                log.error("ADD_POINT failed: %s", exc)
                result = f"error: {exc}"
            history.append(StepRecord(
                step=step, tool="ADD_POINT", args=action,
                result=result,
                voxel_count=seg.voxel_count(),
            ))
        elif tool == "ADD_BBOX":
            try:
                seg.add_bbox(
                    action["x"], action["y"], action["z"],
                    positive=bool(action.get("positive", True)),
                    label=str(action.get("label", "")),
                )
                result = f"bbox x={action['x']} y={action['y']} z={action['z']}"
            except Exception as exc:
                log.error("ADD_BBOX failed: %s", exc)
                result = f"error: {exc}"
            history.append(StepRecord(
                step=step, tool="ADD_BBOX", args=action,
                result=result,
                voxel_count=seg.voxel_count(),
            ))
        else:
            log.warning("Unknown tool '%s' — treating as DONE", tool)
            history.append(StepRecord(
                step=step, tool="DONE", args=action,
                result=f"unknown tool '{tool}'",
                voxel_count=seg.voxel_count(),
            ))
            break

        # Render the new state for the next iteration
        screens = seg.save_orthogonal_previews(name_prefix=f"{media_id}_step{step:02d}")
        history[-1].screenshots = screens

    # Finalise — dump labelmap, summary, and a markdown report.
    labelmap_path = seg.save_labelmap()
    summary_path = seg.export_summary({
        "goal": goal,
        "vision_model": vision_model,
        "labelmap_path": labelmap_path,
        "history": [_record_to_dict(r) for r in history],
    })
    report_path = _write_report(output, media_id, goal, history,
                                seg, labelmap_path)

    log.info("Done. %d steps, final mask: %d voxels (%.2f mm^3)",
             len(history), seg.voxel_count(), seg.volume_mm3())

    return {
        "success": True,
        "media_id": media_id,
        "goal": goal,
        "steps": len(history),
        "voxel_count": seg.voxel_count(),
        "volume_mm3": round(seg.volume_mm3(), 3),
        "labelmap_path": labelmap_path,
        "summary_path": summary_path,
        "report_path": report_path,
        "history": [_record_to_dict(r) for r in history],
    }


def _build_state_text(*, goal: str, step: int, max_steps: int,
                      image_shape_xyz: list, spacing_xyz: list,
                      voxel_count: int, volume_mm3: float,
                      history: list[StepRecord]) -> str:
    lines = [
        f"GOAL: {goal}",
        "",
        f"Step {step}/{max_steps}",
        f"image_shape_xyz: {image_shape_xyz}",
        f"voxel_spacing_mm_xyz: {[round(s, 4) for s in spacing_xyz]}",
        f"current_mask_voxels: {voxel_count}",
        f"current_mask_volume_mm3: {round(volume_mm3, 2)}",
        "",
        "Previous actions (most recent last):",
    ]
    for r in history[-6:]:
        args_short = {k: v for k, v in r.args.items()
                      if k not in ("reason", "summary", "raw")}
        lines.append(
            f"  step {r.step}: {r.tool} {json.dumps(args_short, default=str)} "
            f"-> {r.result} (mask={r.voxel_count} voxels)"
        )
    if not history:
        lines.append("  (none)")
    lines.append("")
    lines.append("Choose ONE next action as a JSON object. No prose.")
    return "\n".join(lines)


def _record_to_dict(r: StepRecord) -> dict:
    return {
        "step": r.step,
        "tool": r.tool,
        "args": r.args,
        "result": r.result,
        "voxel_count": r.voxel_count,
        "screenshots": r.screenshots,
    }


def _write_report(output: Path, media_id: str, goal: str,
                  history: list[StepRecord], seg: Segmenter,
                  labelmap_path: str) -> str:
    lines = [
        f"# nnInteractive paint loop — {media_id}",
        "",
        f"**Goal:** {goal}",
        f"**Steps:** {len(history)}",
        f"**Final voxel count:** {seg.voxel_count():,}",
        f"**Final volume:** {seg.volume_mm3():.2f} mm³",
        f"**Labelmap:** `{labelmap_path}`",
        f"**Device:** `{seg.device}`",
        "",
        "## Prompt history",
        "",
        "| Step | Tool | Args | Mask voxels | Result |",
        "|-----:|------|------|-----------:|--------|",
    ]
    for r in history:
        args_str = json.dumps({k: v for k, v in r.args.items()
                               if k not in ("reason", "summary", "raw")},
                              default=str)
        lines.append(
            f"| {r.step} | `{r.tool}` | `{args_str}` | "
            f"{r.voxel_count:,} | {r.result} |"
        )
    lines.append("")
    lines.append("## Final preview")
    lines.append("")
    for view in ("axial", "coronal", "sagittal"):
        lines.append(f"![{view}]({media_id}_nni_{view}.png)")
    out_path = output / f"{media_id}_nni_report.md"
    out_path.write_text("\n".join(lines))
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Iterative LLM-driven nnInteractive segmentation")
    p.add_argument("--input", required=True, help="Path to volume (NIfTI/NRRD)")
    p.add_argument("--goal", required=True, help="What to segment")
    p.add_argument("--media-id", default="unknown")
    p.add_argument("--output-dir", default="/tmp/nni_loop")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--vision-model", default="",
                   help="OpenAI vision model (default: $NNINTERACTIVE_VISION_MODEL or gpt-4o)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()
    result = run_loop(
        input_path=args.input,
        goal=args.goal,
        output_dir=args.output_dir,
        media_id=args.media_id,
        max_steps=args.max_steps,
        vision_model=args.vision_model,
    )
    result["duration_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in result.items() if k != "history"},
                     indent=2, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
