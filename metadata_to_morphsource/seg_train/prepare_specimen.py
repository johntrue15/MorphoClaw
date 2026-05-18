"""Download + voxelise a real MorphoSource (CT, GT mesh) pair into a
trainer-ready specimen list.

This module is the bridge between the existing
``.github/scripts/nninteractive_compare.py`` prep pipeline (which
already downloads, converts DICOM/TIFF stacks to NIfTI, crops around a
mesh, and voxelises the GT) and
:class:`~metadata_to_morphsource.seg_train.iterative_trainer.IterativeTrainer`.

We **shell out** to ``nninteractive_compare.py --skip-paint-loop`` so
all the heavy SimpleITK/VTK/Slicer dependencies stay isolated to the
existing nnInteractive venv. After it finishes we walk the resulting
``<output_dir>/<ct_id>__vs__<gt_id>/`` directory, locate the four
artefacts the trainer needs (CT NIfTI, raw mesh, voxelised GT label,
voxelisation summary), and emit a single ``specimens.json`` entry
identical in schema to what
:func:`metadata_to_morphsource.seg_train.cli._cmd_discover` writes — so
the resulting JSON drops straight into ``seg_train round --specimens``.

Presets
-------
``chameleon_stapes`` — the canonical fast end-to-end pair
(Chamaeleo calyptratus, right stapes). About 5–15 minutes on the
runner depending on Slicer/VTK availability.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


log = logging.getLogger("seg_train.prepare")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
COMPARE_SCRIPT = SCRIPTS_DIR / "nninteractive_compare.py"


# Presets mirror those in nninteractive_compare.py so users can ask for
# them by name without remembering MorphoSource media IDs. Keep this
# list in sync with PRESETS in that script.
PRESETS: dict[str, dict] = {
    "chameleon_stapes": {
        "ct_media_id": "000408242",
        "gt_media_id": "000790324",
        "physical_object_id": "uf:herp:191369",
        "taxonomy": "Chamaeleo calyptratus",
        "goal": (
            "Segment the right stapes (small middle-ear bone). It is a "
            "tiny irregular bone in the inner ear region."
        ),
        "morphosource_query": "chameleon stapes",
        "max_steps": 6,
        "crop_around_mesh_mm": 4.0,
        "voxelize_backend": "vtk",
    },
}


# ---------------------------------------------------------------------------


@dataclass
class PreparedSpecimen:
    """One prepared (CT, GT) pair, ready for ``seg_train round``."""

    ct_media_id: str
    gt_media_id: str
    physical_object_id: str
    taxonomy: str
    morphosource_query: str
    goal: str
    volume_path: str        # NIfTI on the CT grid (cropped, if applicable)
    gt_mesh_path: str       # raw mesh (.ply / .stl / ...)
    gt_label_path: str      # GT voxelised onto ``volume_path`` grid
    pair_dir: str           # nninteractive_compare.py output root
    voxelize_summary: dict = field(default_factory=dict)
    duration_s: float = 0.0

    def to_specimen_dict(self) -> dict:
        """Return a dict matching ``SpecimenInput``'s constructor."""
        return {
            "media_id": self.ct_media_id,
            "physical_object_id": self.physical_object_id,
            "taxonomy": self.taxonomy,
            "morphosource_query": self.morphosource_query,
            "volume_path": self.volume_path,
            "gt_mesh_path": self.gt_mesh_path,
            # gt_label_path lets the trainer skip re-voxelising; the
            # ``gt_mesh_path`` is still recorded for provenance.
            "gt_label_path": self.gt_label_path,
            "gt_media_id": self.gt_media_id,
        }


# ---------------------------------------------------------------------------


def _resolve_python(prefer_nni: bool = True) -> str:
    """Pick the best Python to drive ``nninteractive_compare.py``.

    The compare script needs SimpleITK + VTK (or Slicer). Those live
    in the nnInteractive venv, so we prefer that interpreter and fall
    back to the current one (works on CI runners that pre-install the
    deps system-wide).
    """
    if prefer_nni:
        nni_home = os.environ.get(
            "NNINTERACTIVE_HOME",
            str(Path.home() / ".autoresearchclaw" / "nninteractive"),
        )
        candidate = Path(nni_home) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _find_paint_loop_inputs(pair_dir: Path) -> tuple[Path, Path, Path]:
    """Locate the prepared CT NIfTI, the GT voxelmap, and the raw mesh
    inside ``pair_dir`` (the layout written by ``run_comparison``).
    """
    candidates = list(pair_dir.glob("*_cropped.nii.gz"))
    if candidates:
        ct_volume = candidates[0]
    else:
        ct_volume = next(iter(pair_dir.glob("ct_*.nii.gz")), Path())
    if not ct_volume or not ct_volume.exists():
        raise FileNotFoundError(
            f"Could not find a prepared CT NIfTI under {pair_dir}. "
            f"Did `nninteractive_compare.py --skip-paint-loop` succeed?"
        )

    gt_label = pair_dir / "gt_voxelized.nii.gz"
    if not gt_label.exists():
        raise FileNotFoundError(
            f"Voxelised GT not found at {gt_label}. The voxeliser must "
            f"have failed — see `alignment_report.md` in {pair_dir}."
        )

    download_dir = pair_dir / "download"
    mesh: Optional[Path] = None
    mesh_exts = (".ply", ".stl", ".obj", ".off", ".gltf", ".glb")
    if download_dir.exists():
        best_size = -1
        for p in download_dir.rglob("*"):
            if p.is_file() and p.name.lower().endswith(mesh_exts):
                size = p.stat().st_size
                if size > best_size:
                    mesh, best_size = p, size
    if mesh is None:
        # Compare-script keeps a copy of the mesh near the labelmap
        # in some runs; fall back to a wider walk.
        for p in pair_dir.rglob("*"):
            if p.is_file() and p.name.lower().endswith(mesh_exts):
                mesh = p
                break
    if mesh is None:
        raise FileNotFoundError(
            f"Could not locate the raw mesh under {pair_dir}/download. "
            f"The MorphoSource download may have failed silently."
        )
    return ct_volume, gt_label, mesh


def prepare_specimen(
    *,
    ct_media_id: str,
    gt_media_id: str,
    output_dir: os.PathLike | str,
    goal: str = "",
    physical_object_id: str = "",
    taxonomy: str = "",
    morphosource_query: str = "",
    crop_around_mesh_mm: float = 0.0,
    voxelize_backend: str = "auto",
    use_nninteractive_python: bool = True,
    timeout_s: int = 1800,
) -> PreparedSpecimen:
    """Run the full download → DICOM conv → crop → voxelise pipeline
    and return a :class:`PreparedSpecimen`.

    The prep step itself runs no LLM and does not invoke nnInteractive
    — it simply prepares the CT volume + voxelised GT so the
    iterative trainer has a concrete (volume, label) pair to work
    with on a real, live specimen.
    """
    if not COMPARE_SCRIPT.exists():
        raise FileNotFoundError(
            f"nninteractive_compare.py missing at {COMPARE_SCRIPT}; "
            f"the prep helper relies on it for download/DICOM/voxelize."
        )

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        _resolve_python(prefer_nni=use_nninteractive_python),
        str(COMPARE_SCRIPT),
        "--ct-media-id", ct_media_id,
        "--gt-media-id", gt_media_id,
        "--output-dir", str(out),
        "--voxelize-backend", voxelize_backend,
        "--skip-paint-loop",
    ]
    if crop_around_mesh_mm and crop_around_mesh_mm > 0:
        cmd.extend(["--crop-around-mesh-mm", str(crop_around_mesh_mm)])

    log.info("Preparing specimen via: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s,
        env=os.environ.copy(),
    )
    duration = time.time() - t0
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(
            f"Specimen prep failed (exit {proc.returncode}): {msg}"
        )

    pair_dir = out / f"{ct_media_id}__vs__{gt_media_id}"
    if not pair_dir.exists():
        raise FileNotFoundError(
            f"Expected pair output dir {pair_dir} was not created — "
            f"compare-script logs:\n{(proc.stdout or '')[-1500:]}"
        )

    ct_volume, gt_label, mesh = _find_paint_loop_inputs(pair_dir)

    summary_path = pair_dir / "gt_voxelized.voxelize.json"
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            pass

    return PreparedSpecimen(
        ct_media_id=ct_media_id,
        gt_media_id=gt_media_id,
        physical_object_id=physical_object_id,
        taxonomy=taxonomy,
        morphosource_query=morphosource_query,
        goal=goal,
        volume_path=str(ct_volume),
        gt_mesh_path=str(mesh),
        gt_label_path=str(gt_label),
        pair_dir=str(pair_dir),
        voxelize_summary=summary,
        duration_s=round(duration, 2),
    )


def prepare_preset(
    name: str,
    output_dir: os.PathLike | str,
    *,
    voxelize_backend: Optional[str] = None,
    use_nninteractive_python: bool = True,
    timeout_s: int = 1800,
) -> PreparedSpecimen:
    """Resolve a named preset and prepare it via :func:`prepare_specimen`."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {sorted(PRESETS)}"
        )
    p = PRESETS[name]
    return prepare_specimen(
        ct_media_id=p["ct_media_id"],
        gt_media_id=p["gt_media_id"],
        output_dir=output_dir,
        goal=p.get("goal", ""),
        physical_object_id=p.get("physical_object_id", ""),
        taxonomy=p.get("taxonomy", ""),
        morphosource_query=p.get("morphosource_query", ""),
        crop_around_mesh_mm=float(p.get("crop_around_mesh_mm", 0.0)),
        voxelize_backend=voxelize_backend or p.get("voxelize_backend", "auto"),
        use_nninteractive_python=use_nninteractive_python,
        timeout_s=timeout_s,
    )


def write_specimens_json(
    prepared: list[PreparedSpecimen], output_path: os.PathLike | str,
) -> Path:
    """Serialise prepared specimens to a JSON list compatible with
    ``seg_train round --specimens``.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [p.to_specimen_dict() for p in prepared]
    out.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %d prepared specimens to %s", len(payload), out)
    return out
