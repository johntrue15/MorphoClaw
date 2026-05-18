"""Generate (CT, label) training pairs from nnInteractive and curated GT.

Two production paths supply pseudo-labels into the iterative dataset:

1. **MorphoSource human-curated meshes** — voxelised onto the source CT's
   grid via the existing ``voxelize_mesh_in_slicer.py`` Slicer pipeline.
   These are the gold-standard *human* references; the trainer always
   keeps them in the held-out validation pool.

2. **nnInteractive paint loop** — the LLM-driven loop already in
   ``.github/scripts/nninteractive_loop.py`` is invoked headlessly. The
   resulting NIfTI labelmap becomes the pseudo-label for the same CT.

We don't re-implement either pipeline; we shell out so the heavy
torch/Slicer dependencies stay isolated to their existing venvs. The
caller passes paths and we return a ``PseudoLabelResult`` describing
what landed where.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


log = logging.getLogger("seg_train.pseudo_label")


# Paths to the existing AutoResearchClaw scripts (relative to repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"

NNI_HOME_ENV = "NNINTERACTIVE_HOME"
DEFAULT_NNI_HOME = str(Path.home() / ".autoresearchclaw" / "nninteractive")
DEFAULT_SLICER_BIN = "/Applications/Slicer.app/Contents/MacOS/Slicer"


def _nni_python() -> Path:
    home = Path(os.environ.get(NNI_HOME_ENV, DEFAULT_NNI_HOME))
    return home / "bin" / "python"


def _slicer_bin() -> Path:
    return Path(os.environ.get("SLICER_BIN", DEFAULT_SLICER_BIN))


# ---------------------------------------------------------------------------


@dataclass
class PseudoLabelResult:
    success: bool
    label_path: str = ""
    probability_path: str = ""
    summary_path: str = ""
    n_prompts: int = 0
    prompt_kinds: list = field(default_factory=list)
    duration_s: float = 0.0
    method: str = ""  # "nninteractive" | "voxelize_mesh"
    model_version: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# nnInteractive paint-loop wrapper
# ---------------------------------------------------------------------------


def run_paint_loop(
    *,
    volume_path: str,
    goal: str,
    output_dir: str,
    media_id: str,
    max_steps: int = 12,
    vision_model: str = "",
    timeout_s: int = 3600,
) -> PseudoLabelResult:
    """Invoke ``nninteractive_loop.py`` in the dedicated venv.

    Returns a :class:`PseudoLabelResult` whose ``label_path`` points at
    the produced ``*_nni_labelmap.nii.gz`` (suitable for adding to the
    iterative dataset manifest).
    """
    nni_py = _nni_python()
    if not nni_py.exists():
        return PseudoLabelResult(
            success=False,
            method="nninteractive",
            error=f"nnInteractive venv not found at {nni_py}. Bootstrap it via "
                  ".github/scripts/install_nninteractive.sh",
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(nni_py),
        str(SCRIPTS_DIR / "nninteractive_loop.py"),
        "--input", str(volume_path),
        "--goal", goal,
        "--media-id", media_id,
        "--output-dir", str(out),
        "--max-steps", str(max_steps),
    ]
    if vision_model:
        cmd.extend(["--vision-model", vision_model])

    env = os.environ.copy()
    env.setdefault(NNI_HOME_ENV, str(_nni_python().parent.parent))

    log.info("Running nnInteractive paint loop (goal=%s) on %s",
             goal, volume_path)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return PseudoLabelResult(
            success=False, method="nninteractive",
            error=f"paint loop timed out after {timeout_s}s",
            duration_s=time.time() - t0,
        )
    except Exception as exc:
        return PseudoLabelResult(
            success=False, method="nninteractive",
            error=f"subprocess failed: {exc}",
            duration_s=time.time() - t0,
        )

    duration = time.time() - t0
    if proc.returncode != 0:
        return PseudoLabelResult(
            success=False, method="nninteractive",
            error=f"paint loop exit {proc.returncode}: "
                  f"{(proc.stderr or proc.stdout or '')[-400:]}",
            duration_s=duration,
        )

    summary_path = out / f"{media_id}_nni_summary.json"
    if not summary_path.exists():
        return PseudoLabelResult(
            success=False, method="nninteractive",
            error=f"summary not produced at {summary_path}",
            duration_s=duration,
        )
    summary = json.loads(summary_path.read_text())

    label_path = summary.get("labelmap_path", "")
    prompts = summary.get("prompts", [])
    return PseudoLabelResult(
        success=bool(label_path) and Path(label_path).exists(),
        label_path=label_path,
        summary_path=str(summary_path),
        n_prompts=int(summary.get("n_prompts", len(prompts))),
        prompt_kinds=[p.get("kind", "?") for p in prompts],
        duration_s=round(duration, 2),
        method="nninteractive",
        model_version=os.environ.get(
            "NNINTERACTIVE_MODEL", "nnInteractive_v1.0"
        ),
        extra={"summary": summary},
    )


# ---------------------------------------------------------------------------
# Voxelize MorphoSource GT mesh onto the CT grid
# ---------------------------------------------------------------------------


def voxelize_curated_mesh(
    *,
    reference_volume: str,
    mesh_path: str,
    output_path: str,
    fill_value: int = 1,
    timeout_s: int = 900,
) -> PseudoLabelResult:
    """Run ``voxelize_mesh_in_slicer.py`` to rasterise *mesh_path* onto the
    voxel grid of *reference_volume*.
    """
    slicer = _slicer_bin()
    if not slicer.exists():
        return PseudoLabelResult(
            success=False, method="voxelize_mesh",
            error=f"3D Slicer not found at {slicer}",
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["VOXELIZE_REFERENCE_VOLUME"] = str(reference_volume)
    env["VOXELIZE_MESH_PATH"] = str(mesh_path)
    env["VOXELIZE_OUTPUT_PATH"] = str(output_path)
    env["VOXELIZE_FILL_VALUE"] = str(fill_value)

    cmd = [
        str(slicer), "--no-splash", "--no-main-window",
        "--python-script", str(SCRIPTS_DIR / "voxelize_mesh_in_slicer.py"),
    ]
    log.info("Voxelising curated mesh %s onto %s",
             mesh_path, reference_volume)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return PseudoLabelResult(
            success=False, method="voxelize_mesh",
            error=f"voxelizer timed out after {timeout_s}s",
            duration_s=time.time() - t0,
        )
    except Exception as exc:
        return PseudoLabelResult(
            success=False, method="voxelize_mesh",
            error=f"voxelizer subprocess failed: {exc}",
            duration_s=time.time() - t0,
        )

    duration = time.time() - t0
    if proc.returncode != 0 or not Path(output_path).exists():
        return PseudoLabelResult(
            success=False, method="voxelize_mesh",
            error=f"voxelize exit {proc.returncode}: "
                  f"{(proc.stderr or proc.stdout or '')[-400:]}",
            duration_s=duration,
        )

    summary_path = Path(output_path).with_suffix("").with_suffix(
        ".voxelize.json"
    )
    extra: dict = {}
    if summary_path.exists():
        try:
            extra = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass

    return PseudoLabelResult(
        success=True,
        label_path=str(output_path),
        summary_path=str(summary_path) if summary_path.exists() else "",
        duration_s=round(duration, 2),
        method="voxelize_mesh",
        model_version="curated_mesh_v1",
        extra=extra,
    )
