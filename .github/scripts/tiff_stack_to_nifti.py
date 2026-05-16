"""
Convert a TIFF z-stack (e.g. micro-CT exported from VGStudio Max) into a
single NIfTI (.nii.gz) suitable for nnInteractive / SimpleITK.

MorphoSource's micro-CT downloads are commonly a numerically-sorted set of
2D TIFF slices plus VGStudio sidecar files (.vgi/.vgl/.pca). Voxel spacing
isn't carried by the TIFF tags, so we rely on the MorphoSource API
metadata (``x_pixel_spacing``, ``y_pixel_spacing``, ``slice_thickness``)
or explicit ``--spacing-xyz`` from the caller.

Usage::

    python tiff_stack_to_nifti.py \
        --input-dir /path/to/Z_stack/ \
        --output /tmp/ct.nii.gz \
        --media-id 000408242

If ``--media-id`` is given, voxel spacing is fetched from the MorphoSource
API and applied. Otherwise pass ``--spacing-xyz X Y Z`` (mm) explicitly.
Without either, spacing defaults to 1.0 mm and a warning is printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("tiff_stack_to_nifti")


def _import_deps():
    try:
        import SimpleITK as sitk
        import numpy as np
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run inside the nnInteractive venv.",
              file=sys.stderr)
        sys.exit(1)
    return sitk, np


def _natural_sort_key(p: Path):
    parts = re.split(r"(\d+)", p.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def _list_tiffs(input_dir: Path, recursive: bool = True) -> list[Path]:
    pat = "**/*" if recursive else "*"
    matches: list[Path] = []
    for p in input_dir.glob(pat):
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            matches.append(p)
    matches.sort(key=_natural_sort_key)
    return matches


def _spacing_from_morphosource(media_id: str) -> Optional[tuple]:
    """Fetch voxel spacing (sx, sy, sz) in mm from the MorphoSource API."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from morphosource_client import MorphoSourceClient
        from _helpers import safe_first
    except Exception as exc:
        log.warning("Cannot import morphosource_client: %s", exc)
        return None

    try:
        client = MorphoSourceClient()
        rec = client.get_media(media_id)
    except Exception as exc:
        log.warning("Failed to query MorphoSource for %s: %s", media_id, exc)
        return None

    if rec.error:
        log.warning("MorphoSource API error: %s", rec.error)
        return None

    inner = rec.data.get("response", rec.data) if rec.data else {}
    if isinstance(inner, dict) and "media" in inner:
        inner = inner["media"]
    if isinstance(inner, dict) and "media" in inner:
        inner = inner["media"]
    if not isinstance(inner, dict):
        return None

    sx_raw = safe_first(inner.get("x_pixel_spacing"))
    sy_raw = safe_first(inner.get("y_pixel_spacing"))
    sz_raw = safe_first(inner.get("slice_thickness"))
    try:
        sx = float(sx_raw) if sx_raw else None
        sy = float(sy_raw) if sy_raw else sx  # XY usually isotropic
        sz = float(sz_raw) if sz_raw else sx
        if sx is None:
            return None
        return (sx, sy, sz)
    except (TypeError, ValueError):
        return None


def convert(input_dir: Path, output_path: Path,
            spacing_xyz: Optional[tuple] = None,
            origin_xyz: Optional[tuple] = None,
            center_origin: bool = False,
            media_id: str = "",
            summary_path: Path | None = None,
            recursive: bool = True) -> dict:
    sitk, _ = _import_deps()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    files = _list_tiffs(input_dir, recursive=recursive)
    if not files:
        raise RuntimeError(f"No TIFF files under {input_dir}")
    log.info("Found %d TIFF slices in %s", len(files), input_dir)
    log.info("First slice: %s", files[0].name)
    log.info("Last  slice: %s", files[-1].name)

    spacing_source = "explicit"
    if spacing_xyz is None:
        if media_id:
            api = _spacing_from_morphosource(media_id)
            if api:
                spacing_xyz = api
                spacing_source = "morphosource_api"
        if spacing_xyz is None:
            log.warning(
                "No spacing provided and MorphoSource lookup failed — "
                "defaulting to (1.0, 1.0, 1.0) mm. The mesh-CT alignment "
                "WILL be wrong unless the GT mesh was exported in voxel "
                "coordinates."
            )
            spacing_xyz = (1.0, 1.0, 1.0)
            spacing_source = "default_1mm"

    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]),
                  float(spacing_xyz[2]))
    log.info("Voxel spacing (mm): (%g, %g, %g)  [source: %s]",
             sx, sy, sz, spacing_source)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in files])
    img = reader.Execute()

    img.SetSpacing((sx, sy, sz))

    size = img.GetSize()
    if origin_xyz is not None:
        origin_final = tuple(float(v) for v in origin_xyz)
        origin_source = "explicit"
    elif center_origin:
        origin_final = (
            -0.5 * (size[0] - 1) * sx,
            -0.5 * (size[1] - 1) * sy,
            -0.5 * (size[2] - 1) * sz,
        )
        origin_source = "centered_on_volume"
    else:
        origin_final = (0.0, 0.0, 0.0)
        origin_source = "default_zero"

    img.SetOrigin(origin_final)
    log.info("Volume size = %s, origin = %s [source: %s]",
             size, origin_final, origin_source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
    log.info("Wrote NIfTI: %s (%d bytes)",
             output_path, output_path.stat().st_size)

    summary = {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "media_id": media_id,
        "n_slices": len(files),
        "first_slice": files[0].name,
        "last_slice": files[-1].name,
        "size": list(img.GetSize()),
        "spacing": list(img.GetSpacing()),
        "origin": list(img.GetOrigin()),
        "direction": list(img.GetDirection()),
        "spacing_source": spacing_source,
        "origin_source": origin_source,
    }
    if summary_path is None:
        summary_path = output_path.with_suffix("").with_suffix(".tiffstack.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _parse_args():
    p = argparse.ArgumentParser(
        description="Convert a TIFF z-stack to a single NIfTI volume."
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory containing the .tif slices "
                        "(searched recursively unless --no-recursive)")
    p.add_argument("--output", required=True,
                   help="Output NIfTI path (.nii.gz)")
    p.add_argument("--media-id", default="",
                   help="MorphoSource media ID to look up voxel spacing")
    p.add_argument("--spacing-xyz", nargs=3, type=float,
                   metavar=("SX", "SY", "SZ"),
                   help="Explicit voxel spacing in mm (overrides API)")
    p.add_argument("--origin-xyz", nargs=3, type=float,
                   default=None,
                   metavar=("OX", "OY", "OZ"),
                   help="Explicit volume origin in mm (overrides "
                        "--center-origin). Default: (0,0,0) unless "
                        "--center-origin is set.")
    p.add_argument("--center-origin", action="store_true",
                   help="Set the origin so the volume is centered on "
                        "(0,0,0) in world coordinates. Common for "
                        "MorphoSource TIFF stacks where the segmentation "
                        "tool worked with a centered volume.")
    p.add_argument("--no-recursive", action="store_true",
                   help="Do not recurse into subdirectories")
    p.add_argument("--summary", default="")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _parse_args()
    t0 = time.time()
    try:
        convert(
            input_dir=Path(args.input_dir),
            output_path=Path(args.output),
            spacing_xyz=tuple(args.spacing_xyz) if args.spacing_xyz else None,
            origin_xyz=tuple(args.origin_xyz) if args.origin_xyz else None,
            center_origin=args.center_origin,
            media_id=args.media_id,
            summary_path=Path(args.summary) if args.summary else None,
            recursive=not args.no_recursive,
        )
    except Exception as exc:
        log.error("TIFF stack conversion failed: %s", exc, exc_info=True)
        return 1
    log.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
