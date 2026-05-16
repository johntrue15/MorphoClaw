"""
Headless interface to the nnInteractive Python backend.

This module wraps `nnInteractiveInferenceSession` so that the rest of
AutoResearchClaw can:

    1. Load a 3D volume (NIfTI / NRRD / SimpleITK-readable)
    2. Apply a series of point / bbox / scribble / lasso prompts
    3. Retrieve the resulting binary segmentation mask
    4. Persist it as a NIfTI labelmap + 3 orthogonal preview PNGs

The primary consumer is `nninteractive_loop.py`, which runs an LLM-in-the-loop
"paint" session — but `Segmenter` is also useful as a standalone tool for
scripted batch segmentation (e.g. seeded with a fixed prompt list).

Design notes
------------
- The nnInteractive package requires its own venv with a compatible torch
  build, so this script is normally invoked via the Python interpreter at
  ``$NNINTERACTIVE_HOME/bin/python`` (set up by ``install_nninteractive.sh``).
  When imported from the regular AutoResearchClaw env it raises
  ``NNInteractiveUnavailable`` with installation hints.
- Device selection prefers CUDA, then MPS (Apple Silicon), then CPU. Override
  with ``NNINTERACTIVE_DEVICE``.
- Model weights are loaded from ``$NNINTERACTIVE_MODEL_DIR/<MODEL_NAME>``,
  prefetched by ``install_nninteractive.sh``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("nninteractive")

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------

NNI_HOME = Path(
    os.environ.get(
        "NNINTERACTIVE_HOME",
        str(Path.home() / ".autoresearchclaw" / "nninteractive"),
    )
)
NNI_MODEL_DIR = Path(
    os.environ.get("NNINTERACTIVE_MODEL_DIR", str(NNI_HOME / "models"))
)
NNI_MODEL_NAME = os.environ.get("NNINTERACTIVE_MODEL", "nnInteractive_v1.0")


class NNInteractiveUnavailable(RuntimeError):
    """Raised when the nnInteractive backend cannot be imported."""


# ---------------------------------------------------------------------------
# Lazy heavy imports
# ---------------------------------------------------------------------------


def _require_backend():
    """Import torch + nnInteractive + SimpleITK or raise a friendly error."""
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        import SimpleITK as sitk  # noqa: F401
        from nnInteractive.inference.inference_session import (  # noqa: F401
            nnInteractiveInferenceSession,
        )
    except ImportError as exc:
        raise NNInteractiveUnavailable(
            "nnInteractive backend not importable. "
            "Run `.github/scripts/install_nninteractive.sh` and re-run this "
            f"script with the venv's Python: {NNI_HOME}/bin/python "
            f"(import error: {exc})"
        ) from exc


def _select_device():
    import torch

    forced = os.environ.get("NNINTERACTIVE_DEVICE", "").strip().lower()
    if forced:
        log.info("Using forced device: %s", forced)
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        log.info("CUDA unavailable — using Apple Silicon MPS device")
        return torch.device("mps")
    log.info("CUDA/MPS unavailable — using CPU (slow)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Prompt data classes
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    """A single nnInteractive prompt."""

    kind: str  # "point" | "bbox" | "reset"
    coords: tuple | list | None = None
    include: bool = True  # True = positive, False = negative
    label: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "coords": list(self.coords) if self.coords is not None else None,
            "include": self.include,
            "label": self.label,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Segmenter — wraps nnInteractiveInferenceSession with persistence helpers
# ---------------------------------------------------------------------------


@dataclass
class SegmenterConfig:
    input_path: str
    output_dir: str
    media_id: str = "unknown"
    model_dir: str = ""
    use_torch_compile: bool = False
    do_autozoom: bool = True
    use_pinned_memory: bool = True
    verbose: bool = False


class Segmenter:
    """Headless wrapper around `nnInteractiveInferenceSession`."""

    def __init__(self, config: SegmenterConfig):
        _require_backend()
        import numpy as np
        import SimpleITK as sitk
        import torch
        from nnInteractive.inference.inference_session import (
            nnInteractiveInferenceSession,
        )

        self._np = np
        self._sitk = sitk
        self._torch = torch

        self.config = config
        self.input_path = Path(config.input_path)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = Path(config.model_dir or NNI_MODEL_DIR)
        candidate = model_dir / NNI_MODEL_NAME
        # Allow either ".../models" or ".../models/nnInteractive_v1.0" to be passed.
        if (candidate / "plans.json").exists() or any(candidate.glob("fold_*")):
            self.model_path = candidate
        elif (model_dir / "plans.json").exists() or any(model_dir.glob("fold_*")):
            self.model_path = model_dir
        else:
            self.model_path = candidate

        if not self.model_path.exists():
            raise NNInteractiveUnavailable(
                f"Model weights not found at {self.model_path}. "
                "Run install_nninteractive.sh to prefetch them."
            )

        self.device = _select_device()
        log.info(
            "Initialising nnInteractive session (device=%s, model=%s)",
            self.device, self.model_path,
        )

        self._session = nnInteractiveInferenceSession(
            device=self.device,
            use_torch_compile=config.use_torch_compile,
            verbose=config.verbose,
            torch_n_threads=os.cpu_count() or 4,
            do_autozoom=config.do_autozoom,
            use_pinned_memory=config.use_pinned_memory and self.device.type == "cuda",
        )
        self._session.initialize_from_trained_model_folder(str(self.model_path))

        log.info("Loading volume: %s", self.input_path)
        self.sitk_image = sitk.ReadImage(str(self.input_path))
        arr = sitk.GetArrayFromImage(self.sitk_image)  # z, y, x
        if arr.ndim != 3:
            raise ValueError(
                f"Expected a 3D volume; got shape {arr.shape} from {self.input_path}"
            )
        self.image_shape_zyx = tuple(arr.shape)
        self._session.set_image(arr[None])  # (1, z, y, x)
        self.target = torch.zeros(self.image_shape_zyx, dtype=torch.uint8)
        self._session.set_target_buffer(self.target)

        self.prompts_log: list[Prompt] = []

    # ------------------------------------------------------------------
    # Prompt API — coords are in (x, y, z) world-of-array order matching
    # the nnInteractive README. Internally the session uses (x, y, z) too.
    # ------------------------------------------------------------------

    def add_point(self, x: int, y: int, z: int, *, positive: bool = True,
                  label: str = "") -> None:
        self._session.add_point_interaction(
            (int(x), int(y), int(z)), include_interaction=bool(positive)
        )
        self.prompts_log.append(
            Prompt(kind="point", coords=(int(x), int(y), int(z)),
                   include=bool(positive), label=label)
        )

    def add_bbox(self, x_range: Iterable[int], y_range: Iterable[int],
                 z_range: Iterable[int], *, positive: bool = True,
                 label: str = "") -> None:
        bbox = [
            [int(x_range[0]), int(x_range[1])],
            [int(y_range[0]), int(y_range[1])],
            [int(z_range[0]), int(z_range[1])],
        ]
        self._session.add_bbox_interaction(bbox, include_interaction=bool(positive))
        self.prompts_log.append(
            Prompt(kind="bbox", coords=bbox, include=bool(positive), label=label)
        )

    def reset_segment(self) -> None:
        self._session.reset_interactions()
        self.target.zero_()
        self._session.set_target_buffer(self.target)
        self.prompts_log.append(Prompt(kind="reset"))

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    @property
    def mask_array(self):
        return self.target.detach().cpu().numpy()

    def voxel_count(self) -> int:
        return int((self.target > 0).sum().item())

    def volume_mm3(self) -> float:
        spacing = self.sitk_image.GetSpacing()  # (sx, sy, sz)
        voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
        return self.voxel_count() * voxel_vol

    def bounding_box(self) -> Optional[dict]:
        np = self._np
        mask = self.mask_array
        if mask.sum() == 0:
            return None
        zs, ys, xs = np.where(mask > 0)
        return {
            "x": [int(xs.min()), int(xs.max())],
            "y": [int(ys.min()), int(ys.max())],
            "z": [int(zs.min()), int(zs.max())],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_labelmap(self, name: str = "") -> str:
        sitk = self._sitk
        np = self._np

        out_name = name or f"{self.config.media_id}_nni_labelmap.nii.gz"
        out_path = self.output_dir / out_name
        mask = self.mask_array.astype(np.uint8)

        seg_image = sitk.GetImageFromArray(mask)
        seg_image.CopyInformation(self.sitk_image)
        sitk.WriteImage(seg_image, str(out_path), useCompression=True)

        log.info("Wrote labelmap: %s (%d voxels, %.2f mm^3)",
                 out_path, self.voxel_count(), self.volume_mm3())
        return str(out_path)

    def save_orthogonal_previews(self, name_prefix: str = "") -> list[str]:
        """Render axial / coronal / sagittal mid-slice previews with mask overlay."""
        np = self._np
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed; skipping preview screenshots")
            return []

        sitk = self._sitk
        arr = sitk.GetArrayFromImage(self.sitk_image)  # z, y, x
        mask = self.mask_array
        prefix = name_prefix or f"{self.config.media_id}_nni"

        # Choose slice that maximises mask coverage (or middle if empty)
        def _best_slice(axis: int) -> int:
            if mask.sum() == 0:
                return arr.shape[axis] // 2
            sums = mask.sum(axis=tuple(i for i in range(3) if i != axis))
            return int(np.argmax(sums))

        previews: list[str] = []
        views = [
            ("axial",    0, lambda v, s: (v[s, :, :], mask[s, :, :])),
            ("coronal",  1, lambda v, s: (v[:, s, :], mask[:, s, :])),
            ("sagittal", 2, lambda v, s: (v[:, :, s], mask[:, :, s])),
        ]
        for view_name, axis, slicer in views:
            s = _best_slice(axis)
            img_slice, mask_slice = slicer(arr, s)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_slice, cmap="gray", origin="lower")
            if mask_slice.sum() > 0:
                overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax.imshow(overlay, cmap="autumn", alpha=0.45, origin="lower")
            ax.set_title(f"{view_name} (slice {s}) — {self.config.media_id}")
            ax.set_axis_off()
            out_path = self.output_dir / f"{prefix}_{view_name}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            previews.append(str(out_path))
        return previews

    def export_summary(self, extra: Optional[dict] = None) -> str:
        summary = {
            "media_id": self.config.media_id,
            "input_path": str(self.input_path),
            "model_path": str(self.model_path),
            "device": str(self.device),
            "image_shape_zyx": list(self.image_shape_zyx),
            "voxel_spacing_xyz": list(self.sitk_image.GetSpacing()),
            "n_prompts": len(self.prompts_log),
            "prompts": [p.to_dict() for p in self.prompts_log],
            "voxel_count": self.voxel_count(),
            "volume_mm3": round(self.volume_mm3(), 3),
            "bounding_box_voxels": self.bounding_box(),
        }
        if extra:
            summary.update(extra)
        out_path = self.output_dir / f"{self.config.media_id}_nni_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        return str(out_path)


# ---------------------------------------------------------------------------
# Scripted batch entry-point: read a JSON prompt list, run, save outputs
# ---------------------------------------------------------------------------


def run_scripted(prompt_file: str, input_path: str, output_dir: str,
                 media_id: str = "unknown") -> dict:
    """Apply a JSON list of prompts and save results.

    Prompt file format:
    [
      {"kind": "point", "x": 120, "y": 80, "z": 45, "positive": true, "label": "heart"},
      {"kind": "bbox",  "x": [50, 100], "y": [30, 90], "z": [20, 21], "positive": true},
      {"kind": "reset"}
    ]
    """
    cfg = SegmenterConfig(input_path=input_path, output_dir=output_dir,
                          media_id=media_id)
    seg = Segmenter(cfg)

    prompts = json.loads(Path(prompt_file).read_text())
    for i, p in enumerate(prompts):
        kind = p.get("kind", "point")
        if kind == "point":
            seg.add_point(p["x"], p["y"], p["z"],
                          positive=p.get("positive", True),
                          label=p.get("label", f"p{i}"))
        elif kind == "bbox":
            seg.add_bbox(p["x"], p["y"], p["z"],
                         positive=p.get("positive", True),
                         label=p.get("label", f"bb{i}"))
        elif kind == "reset":
            seg.reset_segment()
        else:
            log.warning("Unknown prompt kind: %s (skipped)", kind)

    labelmap = seg.save_labelmap()
    previews = seg.save_orthogonal_previews()
    summary_path = seg.export_summary({
        "labelmap_path": labelmap,
        "preview_screenshots": previews,
    })
    return {
        "labelmap_path": labelmap,
        "preview_screenshots": previews,
        "summary_path": summary_path,
        "voxel_count": seg.voxel_count(),
        "volume_mm3": round(seg.volume_mm3(), 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Run nnInteractive on a CT/MRI volume")
    p.add_argument("--input", required=True, help="Path to NIfTI/NRRD volume")
    p.add_argument("--prompts", required=True, help="JSON file with prompt list")
    p.add_argument("--output-dir", default="/tmp/nninteractive",
                   help="Where to write labelmap and previews")
    p.add_argument("--media-id", default="standalone")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    try:
        result = run_scripted(args.prompts, args.input, args.output_dir,
                              args.media_id)
    except NNInteractiveUnavailable as exc:
        log.error(str(exc))
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
