"""
Convert a DICOM series directory to a single NIfTI (.nii.gz).

MorphoSource's CT distributions are typically DICOM series. nnInteractive's
loop expects a single NIfTI / NRRD on disk, so this helper centralizes the
conversion. Uses SimpleITK's GDCM-backed reader so it handles the usual
multi-frame / multi-series quirks.

Usage::

    python dicom_to_nifti.py \
        --input-dir /tmp/ct_dicom_series/ \
        --output /tmp/ct.nii.gz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("dicom_to_nifti")


def _import_deps():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run inside the nnInteractive venv.",
              file=sys.stderr)
        sys.exit(1)
    return sitk


def convert(input_dir: Path, output_path: Path,
            series_id: str = "",
            summary_path: Path | None = None) -> dict:
    sitk = _import_deps()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(input_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {input_dir}")
    log.info("Found %d series in %s", len(series_ids), input_dir)

    chosen_id = series_id
    if not chosen_id:
        # Pick the largest series (most slices = the volume of interest).
        best = (None, -1)
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(input_dir), sid)
            if len(files) > best[1]:
                best = (sid, len(files))
        chosen_id = best[0]
        log.info("Auto-selecting largest series: %s (%d files)",
                 chosen_id, best[1])

    files = reader.GetGDCMSeriesFileNames(str(input_dir), chosen_id)
    if not files:
        raise RuntimeError(f"Series {chosen_id} has no files")

    reader.SetFileNames(files)
    img = reader.Execute()
    log.info("Loaded volume: size=%s spacing=%s",
             img.GetSize(), img.GetSpacing())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
    log.info("Wrote %s", output_path)

    summary = {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "series_id": chosen_id,
        "n_slices": len(files),
        "size": list(img.GetSize()),
        "spacing": list(img.GetSpacing()),
        "origin": list(img.GetOrigin()),
        "direction": list(img.GetDirection()),
    }
    if summary_path is None:
        summary_path = output_path.with_suffix("").with_suffix(".dicom.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _parse_args():
    p = argparse.ArgumentParser(
        description="Convert a DICOM series directory to a single .nii.gz"
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory containing the DICOM series")
    p.add_argument("--output", required=True,
                   help="Output NIfTI path (.nii.gz)")
    p.add_argument("--series-id", default="",
                   help="Specific GDCM series ID (default: largest series)")
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
            series_id=args.series_id,
            summary_path=Path(args.summary) if args.summary else None,
        )
    except Exception as exc:
        log.error("DICOM conversion failed: %s", exc, exc_info=True)
        return 1
    log.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
