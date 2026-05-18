"""
Pilot evaluation: project 000358382 vs bright-seed nnInteractive.

For ``--specimens`` chameleon (or whichever project) specimens, score the
bright-seed greedy nnInteractive segmenter against ``.ply`` ground truth
at click budgets ``--budgets`` and produce a reproducible, paper-grade
results bundle.

Bright-seed is deterministic, so we run *once per specimen* at the max
budget with stop-rules disabled, then **compose unions of segments
1..K** post-hoc to get the metrics at each budget. ~4x cheaper than four
independent runs, semantically identical.

Pipeline (per specimen)::

    download CT + .ply   (morphosource_api_download.download_media)
    locate volume + mesh inside the archives  (nninteractive_compare helpers)
    convert TIFF/DICOM   -> .nii.gz  (only if needed)
    crop CT around mesh bbox + margin  (in-process via crop_around_mesh.crop)
    voxelize .ply        -> gt_voxelized.nii.gz  (voxelize_mesh_vtk.voxelize)
    push CT to remote Slicer  (remote_volume_io.push_volume)
    run bright-seed at budget=max-budget, --no-stop-rules
       (slicer_remote_bright_seed.main programmatic invocation)
    pull per-segment NIfTI artifacts (already exported by bright-seed)
    compose unions per budget  (post-hoc slicing)
    score each composite vs GT  (segmentation_metrics.compare_labelmaps)
    write per-specimen report.md

Then aggregate -> ``results.csv``, ``dice_vs_budget.png``, ``report.md``.

Should be invoked from the nnInteractive venv Python (the one with
SimpleITK / VTK / matplotlib / numpy installed). Run::

    set -a && source .env && set +a
    "$HOME/.autoresearchclaw/nninteractive/bin/python" \\
        .github/scripts/eval_project358382_pilot.py \\
        --project-query "Colors of Skull Anatomy" \\
        --project-id 000358382 \\
        --specimens 3 \\
        --budgets 10,25,50,100 \\
        --out-dir runs/pilot_chameleon

The script is idempotent: a partially-completed run can be resumed by
pointing at the same ``--out-dir``; cached downloads, crops, voxelized
GTs, and per-segment NIfTIs are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Set MPLCONFIGDIR before matplotlib first-import to silence its warning.
# Falls back to /tmp/<user>/matplotlib if the home cache dir isn't writable
# (e.g. when running inside a sandbox that restricts $HOME writes).
def _ensure_mplconfigdir() -> None:
    candidates = [
        os.environ.get("MPLCONFIGDIR"),
        str(Path.home() / ".cache" / "autoresearchclaw" / "matplotlib"),
        f"/tmp/{os.environ.get('USER', 'mpl')}_matplotlib",
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            Path(cand).mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = cand
            return
        except Exception:
            continue
_ensure_mplconfigdir()

from _helpers import load_dotenv, safe_first  # noqa: E402
from morphosource_client import MorphoSourceClient  # noqa: E402
from run_telemetry import RunLogger, capture_local_env  # noqa: E402

log = logging.getLogger("pilot_eval")
load_dotenv()


# ---------------------------------------------------------------------------
# Discovery: list specimens with both an open CT and an open mesh
# ---------------------------------------------------------------------------

VOLUMETRIC_TOKENS = (
    "ct image series",
    "volumetric image series",
    "ct dicom",
    "micro ct",
    "microct",
    "volumetric",
    "image series",
)
MESH_TOKENS = (
    "mesh",
    "surface model",
    "3d surface",
    "model",
)


def _safe_first(value: Any) -> str:
    return safe_first(value)


def _media_type(item: dict) -> str:
    return _safe_first(
        item.get("media_type") or item.get("media_type_ssi", "")
    ).lower()


def _visibility(item: dict) -> str:
    return _safe_first(
        item.get("visibility") or item.get("visibility_ssi", "")
    ).lower()


def _physical_object_id(item: dict) -> str:
    for key in (
        "physical_object_id",
        "physical_object_id_ssi",
        "physical_object_id_ssim",
        "physical_object",
    ):
        value = item.get(key)
        if value:
            return _safe_first(value)
    return ""


def _media_id(item: dict) -> str:
    for key in ("id", "media_id", "media_id_ssi"):
        v = item.get(key)
        if v:
            return _safe_first(v)
    return ""


def _media_parent_id(item: dict) -> str:
    for key in ("media_parent_id", "media_parent_id_ssi", "media_parent"):
        v = item.get(key)
        if v:
            return _safe_first(v)
    return ""


def _matches_any(text: str, tokens: Iterable[str]) -> bool:
    text = (text or "").lower()
    return any(tok in text for tok in tokens)


def _is_open(item: dict) -> bool:
    return _visibility(item).startswith("open")


def _file_size(item: dict) -> Optional[int]:
    fs = item.get("file_size") or item.get("file_size_all")
    if isinstance(fs, list) and fs:
        try:
            return int(fs[0])
        except Exception:
            return None
    if isinstance(fs, (int, str)):
        try:
            return int(fs)
        except Exception:
            return None
    return None


def _genus(taxon: str) -> str:
    """Return the genus token of a taxonomy string ('Squalus acanthias' -> 'Squalus').

    Falls back to the entire string when there's no whitespace.
    """
    return (taxon or "").strip().split()[0] if taxon else ""


@dataclass
class SpecimenPair:
    physical_object_id: str
    physical_object_title: str
    taxonomy: str
    ct_media_id: str
    ct_title: str
    ct_media_type: str
    ct_file_size: Optional[int]
    mesh_media_id: str
    mesh_title: str
    mesh_media_type: str
    mesh_file_size: Optional[int]
    project_query: str

    @property
    def slug(self) -> str:
        genus = _genus(self.taxonomy) or "specimen"
        safe = "".join(c if c.isalnum() else "_" for c in genus)
        return f"{self.physical_object_id}__{safe}__{self.ct_media_id}__{self.mesh_media_id}"

    def to_dict(self) -> dict:
        return asdict(self)


def discover_pairs(client: MorphoSourceClient,
                    project_query: str,
                    project_id: str = "",
                    page_size: int = 50,
                    max_pages: int = 5,
                    cross_project_lookup: bool = True) -> list[SpecimenPair]:
    """Return all (CT, mesh) pairs found inside (or linked to) the project.

    Strategy:

    1. Search ``q=<project_query>`` and pull every page (up to
       ``max_pages`` * ``page_size`` records).
    2. Group the records by ``physical_object_id``.
    3. For each specimen, find at least one open CT and one open mesh.
       Both must be open-download.
    4. When the project query alone doesn't yield a CT for a specimen
       that has a mesh (common for derivative-mesh projects like
       "Colors of Skull Anatomy"), fall back to
       ``find_segmentation_pairs._list_media_for_object`` to look up
       sibling media on the physical object — this can pick up the
       parent CT even though it lives in a different project.
       Set ``cross_project_lookup=False`` to disable.
    """
    log.info("Discovering pairs in project %r (q=%r)", project_id, project_query)
    all_items: list[dict] = []
    for page in range(1, max_pages + 1):
        sr = client.search_media(q=project_query, per_page=page_size, page=page)
        if not sr.items:
            break
        all_items.extend(sr.items)
        if sr.total_count is not None and len(all_items) >= sr.total_count:
            break
    log.info("Pulled %d media records for project query", len(all_items))

    by_specimen: dict[str, list[dict]] = {}
    for item in all_items:
        spec = _physical_object_id(item)
        if not spec:
            continue
        by_specimen.setdefault(spec, []).append(item)

    # Cache /api/media/{id} replies so we don't hit MorphoSource twice
    # for the same parent CT.
    parent_cache: dict[str, dict] = {}

    def _resolve_parent_ct(parent_id: str) -> Optional[dict]:
        """Fetch a parent CT record by media id; require open + volumetric."""
        if not parent_id:
            return None
        if parent_id in parent_cache:
            return parent_cache[parent_id]
        try:
            rec = client.get_media(parent_id)
        except Exception as exc:
            log.warning("get_media(%s) failed: %s", parent_id, exc)
            parent_cache[parent_id] = {}
            return None
        data = rec.data or {}
        m = data.get("response", data)
        if isinstance(m, dict):
            m = m.get("media", m)
        if not isinstance(m, dict):
            parent_cache[parent_id] = {}
            return None
        parent_cache[parent_id] = m
        if not _is_open(m):
            log.info("parent %s exists but visibility=%s — skipping",
                     parent_id, _visibility(m))
            return None
        if not _matches_any(_media_type(m), VOLUMETRIC_TOKENS):
            log.info("parent %s exists but media_type=%s — skipping",
                     parent_id, _media_type(m))
            return None
        return m

    pairs: list[SpecimenPair] = []
    for spec, records in by_specimen.items():
        cts = [r for r in records
               if _is_open(r) and _matches_any(_media_type(r), VOLUMETRIC_TOKENS)]
        meshes = [r for r in records
                  if _is_open(r) and _matches_any(_media_type(r), MESH_TOKENS)]
        if not meshes:
            continue
        # When no CT lives inside the project query, follow each mesh's
        # ``media_parent_id`` (the parent CT it was derived from). For
        # derivative-mesh projects like 000358382 "Colors of Skull
        # Anatomy", this is the only way to discover the source CT.
        if not cts and cross_project_lookup:
            for mesh_record in meshes:
                pid = _media_parent_id(mesh_record)
                if not pid:
                    continue
                parent = _resolve_parent_ct(pid)
                if parent:
                    log.info("specimen %s: linked to open parent CT %s via "
                             "media_parent_id of mesh %s",
                             spec, pid, _media_id(mesh_record))
                    cts.append(parent)
        if not cts:
            continue
        # Pick the largest CT (proxy for highest fidelity) and largest mesh.
        ct = max(cts, key=lambda r: _file_size(r) or 0)
        mesh = max(meshes, key=lambda r: _file_size(r) or 0)
        taxon = _safe_first(
            ct.get("physical_object_taxonomy_name")
            or mesh.get("physical_object_taxonomy_name")
            or ct.get("physical_object_taxonomy_name_ssim", "")
        )
        title = (
            _safe_first(ct.get("physical_object_title"))
            or _safe_first(mesh.get("physical_object_title"))
            or ""
        )
        pairs.append(SpecimenPair(
            physical_object_id=spec,
            physical_object_title=title,
            taxonomy=taxon,
            ct_media_id=_media_id(ct),
            ct_title=_safe_first(ct.get("title")),
            ct_media_type=_media_type(ct),
            ct_file_size=_file_size(ct),
            mesh_media_id=_media_id(mesh),
            mesh_title=_safe_first(mesh.get("title")),
            mesh_media_type=_media_type(mesh),
            mesh_file_size=_file_size(mesh),
            project_query=project_query,
        ))
    log.info("Discovered %d specimens with both open CT and open mesh", len(pairs))
    return pairs


def select_pilot_specimens(pairs: list[SpecimenPair],
                           n: int = 3,
                           max_ct_bytes: Optional[int] = 5 * (1 << 30),
                           ) -> list[SpecimenPair]:
    """Pick *n* specimens, taxa-diverse (one per genus), within size cap."""
    seen_genera: set[str] = set()
    chosen: list[SpecimenPair] = []
    sortable = sorted(pairs, key=lambda p: (p.ct_file_size or 0))
    for p in sortable:
        if max_ct_bytes is not None and p.ct_file_size and p.ct_file_size > max_ct_bytes:
            continue
        g = _genus(p.taxonomy) or p.physical_object_id
        if g in seen_genera:
            continue
        chosen.append(p)
        seen_genera.add(g)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        # Fall back: relax taxa diversity to fill the quota.
        for p in sortable:
            if p in chosen:
                continue
            if max_ct_bytes is not None and p.ct_file_size and p.ct_file_size > max_ct_bytes:
                continue
            chosen.append(p)
            if len(chosen) >= n:
                break
    return chosen[:n]


# ---------------------------------------------------------------------------
# Per-specimen preparation: download, locate, convert, crop, voxelize
# ---------------------------------------------------------------------------

# Lazy imports for heavy modules so --help works on system Python.
_HEAVY_IMPORTS_DONE = False


def _import_heavy():
    global _HEAVY_IMPORTS_DONE
    if _HEAVY_IMPORTS_DONE:
        return
    import SimpleITK  # noqa: F401
    import numpy  # noqa: F401
    import vtk  # noqa: F401
    _HEAVY_IMPORTS_DONE = True


def download_pair(pair: SpecimenPair, out_dir: Path) -> dict:
    """Download CT + mesh archives. Returns a dict of local download dirs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from morphosource_api_download import download_media

    ct_dir = out_dir / "ct_download"
    mesh_dir = out_dir / "mesh_download"
    ct_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    def _need_download(target: Path) -> bool:
        zips = list(target.glob("morphosource_media-id-*.zip"))
        for z in zips:
            extracted = target / z.stem
            if extracted.is_dir() and any(extracted.iterdir()):
                return False
        return True

    ct_result: dict
    if _need_download(ct_dir):
        log.info("Downloading CT %s -> %s", pair.ct_media_id, ct_dir)
        ct_result = download_media(pair.ct_media_id, str(ct_dir))
    else:
        log.info("Cached CT %s in %s", pair.ct_media_id, ct_dir)
        ct_result = {"success": True, "media_id": pair.ct_media_id,
                      "from_cache": True, "download_dir": str(ct_dir)}

    mesh_result: dict
    if _need_download(mesh_dir):
        log.info("Downloading mesh %s -> %s", pair.mesh_media_id, mesh_dir)
        mesh_result = download_media(pair.mesh_media_id, str(mesh_dir))
    else:
        log.info("Cached mesh %s in %s", pair.mesh_media_id, mesh_dir)
        mesh_result = {"success": True, "media_id": pair.mesh_media_id,
                        "from_cache": True, "download_dir": str(mesh_dir)}

    return {"ct": ct_result, "mesh": mesh_result,
            "ct_dir": str(ct_dir), "mesh_dir": str(mesh_dir)}


def prepare_ct(ct_dir: Path, out_path: Path,
                media_id: str = "") -> dict:
    """Locate CT inside *ct_dir* and write a normalised .nii.gz to *out_path*.

    Reuses the file-finder + converters from nninteractive_compare to
    avoid duplicating logic.
    """
    from nninteractive_compare import (
        _find_ct_volume,
        _ct_input_kind,
        _dicom_to_nifti,
        _tiff_stack_to_nifti,
    )
    pick = _find_ct_volume(ct_dir)
    if pick is None:
        return {"error": f"No CT volume / DICOM dir / TIFF stack found in {ct_dir}"}
    src = pick.path
    kind = _ct_input_kind(src)
    log.info("CT input: kind=%s path=%s size=%s", kind, src, pick.size)
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("Cached prepared CT at %s", out_path)
        return {"output_path": str(out_path), "kind": kind,
                "source_path": str(src), "from_cache": True}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "volume":
        # Already a NIfTI/NRRD/MHA — re-write as compressed NIfTI on the
        # output path so the pipeline is uniform.
        _import_heavy()
        import SimpleITK as sitk
        img = sitk.ReadImage(str(src))
        sitk.WriteImage(img, str(out_path))
        return {"output_path": str(out_path), "kind": kind,
                "source_path": str(src)}
    if kind == "dicom":
        return _dicom_to_nifti(src, out_path)
    if kind == "tiff":
        return _tiff_stack_to_nifti(src, out_path, media_id=media_id,
                                     center_origin=True)
    return {"error": f"Unrecognised CT kind: {kind}"}


def find_mesh(mesh_dir: Path) -> Optional[Path]:
    from nninteractive_compare import _find_mesh
    pick = _find_mesh(mesh_dir)
    return pick.path if pick else None


def crop_and_voxelize(ct_path: Path, mesh_path: Path, out_dir: Path,
                       margin_mm: float = 5.0,
                       max_voxel_axis: Optional[int] = 384) -> dict:
    """Crop *ct_path* around *mesh_path* bbox and voxelize the mesh.

    If ``max_voxel_axis`` is set and any axis of the cropped CT exceeds
    it, resample the cropped CT (linear) and the voxelized GT (nearest
    neighbour) to a coarser grid that keeps every axis at or below the
    cap. Both outputs share the same grid so metrics remain valid.
    Returns dict of paths and summaries.
    """
    from nninteractive_compare import (
        crop_volume_around_mesh,
        voxelize_mesh_to_labelmap,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped = out_dir / "ct_cropped.nii.gz"
    voxelized = out_dir / "gt_voxelized.nii.gz"

    if cropped.exists() and cropped.stat().st_size > 0:
        log.info("Cached cropped CT at %s", cropped)
        crop_summary = {"output_path": str(cropped), "from_cache": True}
    else:
        crop_summary = crop_volume_around_mesh(
            reference_volume=ct_path, mesh=mesh_path,
            output=cropped, margin_mm=margin_mm,
        )
        if "error" in crop_summary:
            return {"error": "crop_failed", "details": crop_summary}

    if voxelized.exists() and voxelized.stat().st_size > 0:
        log.info("Cached voxelized GT at %s", voxelized)
        vox_summary = {"output_path": str(voxelized), "from_cache": True}
    else:
        vox_summary = voxelize_mesh_to_labelmap(
            reference_volume=cropped, mesh=mesh_path,
            output=voxelized, fill_value=1, backend="vtk",
        )
        if "error" in vox_summary:
            return {"error": "voxelize_failed", "details": vox_summary}

    resample_summary = None
    if max_voxel_axis and max_voxel_axis > 0:
        resample_summary = _resample_cap_axis(cropped, voxelized, max_voxel_axis)

    return {"crop": crop_summary, "voxelize": vox_summary,
            "resample": resample_summary,
            "cropped_path": str(cropped),
            "voxelized_gt_path": str(voxelized)}


def _resample_cap_axis(ct_path: Path, gt_path: Path,
                        max_axis: int) -> Optional[dict]:
    """Resample *ct_path* and *gt_path* in place to keep every axis <= max_axis.

    The CT is resampled with linear interpolation; the GT labelmap with
    nearest-neighbour to keep its binary value space exact. Both end up
    on the same grid (origin/direction preserved, spacing scaled).
    """
    _import_heavy()
    import SimpleITK as sitk
    import numpy as np
    img = sitk.ReadImage(str(ct_path))
    size = img.GetSize()
    if max(size) <= max_axis:
        log.info("Cropped CT %s: max axis %d <= cap %d (no resample)",
                 ct_path.name, max(size), max_axis)
        return {"resampled": False, "size_before": list(size)}

    scale = max_axis / float(max(size))
    new_size = [max(1, int(round(s * scale))) for s in size]
    spacing = img.GetSpacing()
    new_spacing = [
        spacing[i] * (size[i] / float(new_size[i])) for i in range(3)
    ]
    log.info("Resampling CT %s: %s spacing=%s -> %s spacing=%s "
             "(cap=%d, scale=%.3f)",
             ct_path.name, list(size), [round(s, 5) for s in spacing],
             new_size, [round(s, 5) for s in new_spacing],
             max_axis, scale)

    def _resample(src: "sitk.Image", interp: int) -> "sitk.Image":
        ref = sitk.Image(new_size, src.GetPixelID())
        ref.SetSpacing(new_spacing)
        ref.SetOrigin(src.GetOrigin())
        ref.SetDirection(src.GetDirection())
        return sitk.Resample(src, ref, sitk.Transform(), interp,
                             0, src.GetPixelID())

    ct_new = _resample(img, sitk.sitkLinear)
    sitk.WriteImage(ct_new, str(ct_path))
    log.info("  wrote %s (%d bytes)", ct_path.name, ct_path.stat().st_size)

    gt = sitk.ReadImage(str(gt_path))
    gt_new = _resample(gt, sitk.sitkNearestNeighbor)
    sitk.WriteImage(gt_new, str(gt_path))
    log.info("  wrote %s (%d bytes)", gt_path.name, gt_path.stat().st_size)

    return {
        "resampled": True,
        "size_before": list(size),
        "size_after": list(new_size),
        "spacing_before": [float(s) for s in spacing],
        "spacing_after": [float(s) for s in new_spacing],
        "max_axis_cap": max_axis,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Bright-seed invocation
# ---------------------------------------------------------------------------

def run_bright_seed(volume_name: str, out_dir: Path, label: str,
                     max_steps: int,
                     intensity_percentile: float = 99.0,
                     no_screenshots: bool = False) -> int:
    """Run slicer_remote_bright_seed.main() in-process.

    Always passes ``--no-stop-rules`` so we can post-hoc slice unions at
    every budget without re-running. Returns the integer exit code that
    bright_seed.main returns.
    """
    import slicer_remote_bright_seed as bs
    argv = [
        "--volume", volume_name,
        "--reset-first",
        "--intensity-percentile", str(intensity_percentile),
        "--max-steps", str(max_steps),
        "--no-stop-rules",
        "--label", label,
        "--out-dir", str(out_dir),
    ]
    if no_screenshots:
        argv.append("--no-screenshots")
    log.info("bright_seed argv: %s", " ".join(argv))
    return bs.main(argv)


def _per_segment_paths_in_click_order(bright_seed_dir: Path) -> list[Path]:
    """Return the per-segment NIfTIs ordered by click index.

    Reads ``summary.json`` (written by bright_seed) and follows the
    ``history[i].segment_id`` for each step. Falls back to a numeric
    sort by filename suffix if summary.json is missing or empty.
    """
    summary_path = bright_seed_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            ordered_sids = []
            for step in summary.get("history", []):
                sid = step.get("segment_id")
                if sid and sid not in ordered_sids:
                    ordered_sids.append(sid)
            paths = []
            seg_dir = bright_seed_dir / "artifacts" / "per_segment"
            for sid in ordered_sids:
                safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in sid)
                cand = seg_dir / f"{safe}.nii.gz"
                if cand.exists():
                    paths.append(cand)
            if paths:
                return paths
        except Exception as exc:
            log.warning("Could not read click order from %s: %s",
                         summary_path, exc)
    # Fallback: sort by trailing integer when present.
    seg_dir = bright_seed_dir / "artifacts" / "per_segment"
    if not seg_dir.exists():
        return []
    files = list(seg_dir.glob("*.nii.gz"))

    def _key(p: Path) -> tuple:
        stem = p.stem.replace(".nii", "")
        bits = stem.split("_")
        try:
            return (0, int(bits[-1]))
        except Exception:
            return (1, stem)

    return sorted(files, key=_key)


# ---------------------------------------------------------------------------
# Post-hoc union composer + per-budget metrics
# ---------------------------------------------------------------------------

def compose_union(per_segment_paths_sorted: list[Path], K: int,
                   out_path: Path) -> dict:
    """Union the first *K* per-segment binary labelmaps into *out_path*.

    All inputs must share grid (origin/spacing/direction/size). The
    function copies the geometric metadata from the *last* read input.
    """
    _import_heavy()
    import SimpleITK as sitk
    import numpy as np

    if not per_segment_paths_sorted:
        return {"error": "no_per_segment_inputs", "K_requested": K, "K_used": 0}

    K_used = min(K, len(per_segment_paths_sorted))
    arr = None
    last_img = None
    contributing = []
    for p in per_segment_paths_sorted[:K_used]:
        img = sitk.ReadImage(str(p))
        a = sitk.GetArrayFromImage(img) > 0
        arr = a if arr is None else (arr | a)
        last_img = img
        contributing.append(str(p))

    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    if last_img is not None:
        out.CopyInformation(last_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_path))
    return {
        "output_path": str(out_path),
        "K_requested": K,
        "K_used": K_used,
        "contributing_paths": contributing,
        "voxels_set": int(arr.sum()),
        "shape_kji": list(arr.shape),
    }


def score_against_gt(pred_path: Path, gt_path: Path) -> dict:
    """Compute Dice/IoU/Hausdorff via segmentation_metrics.compare_labelmaps.

    Falls back to a stripped-down dict if metrics import fails.
    """
    try:
        from segmentation_metrics import compare_labelmaps
        m = compare_labelmaps(str(pred_path), str(gt_path),
                                compute_surface_distances=True)
        return asdict(m) if hasattr(m, "__dataclass_fields__") else dict(m.__dict__)
    except Exception as exc:
        log.warning("compare_labelmaps failed: %s", exc, exc_info=True)
        return {"error": repr(exc), "traceback": traceback.format_exc()}


def post_hoc_metrics_for_specimen(specimen_dir: Path,
                                   gt_path: Path,
                                   per_segment_paths_sorted: list[Path],
                                   budgets: list[int],
                                   actual_clicks: int,
                                   logger: Optional[RunLogger] = None,
                                   ) -> list[dict]:
    """For each *budget* in *budgets*, compose union-of-K and score.

    Returns a list of dicts, one per budget, each including:
        budget, K_used, composite_path, sha256_composite, metrics, ...
    """
    composites_dir = specimen_dir / "composites"
    metrics_dir = specimen_dir / "metrics"
    composites_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for K in budgets:
        comp_path = composites_dir / f"composite_at_{K:03d}.nii.gz"
        comp_summary = compose_union(per_segment_paths_sorted, K, comp_path)
        if "error" in comp_summary:
            log.warning("compose_union(K=%d) error: %s", K, comp_summary)
            rows.append({"budget": K, "error": comp_summary["error"]})
            if logger:
                logger.event("composite_failed", budget=K, **comp_summary)
            continue
        # Hash the composite before metrics so it's recorded even if
        # comparison errors out.
        from run_telemetry import sha256_file
        sha = sha256_file(comp_path)
        comp_summary["sha256"] = sha
        if logger:
            logger.event("composite_ok", budget=K, **comp_summary)

        metrics = score_against_gt(comp_path, gt_path)
        out = {
            "budget": K,
            "K_used": comp_summary["K_used"],
            "actual_clicks": actual_clicks,
            "composite_path": str(comp_path),
            "composite_sha256": sha,
            "composite_voxels_set": comp_summary["voxels_set"],
            "metrics": metrics,
        }
        (metrics_dir / f"metrics_at_{K:03d}.json").write_text(
            json.dumps(out, indent=2, default=str)
        )
        rows.append(out)
        if logger:
            logger.event("metrics_ok", budget=K,
                          dice=metrics.get("dice"),
                          iou=metrics.get("iou"),
                          K_used=comp_summary["K_used"])
    return rows


# ---------------------------------------------------------------------------
# Per-specimen orchestrator
# ---------------------------------------------------------------------------

@dataclass
class SpecimenResult:
    pair: SpecimenPair
    specimen_dir: Path
    bright_seed_dir: Path
    gt_path: Optional[Path] = None
    cropped_ct_path: Optional[Path] = None
    bright_seed_summary: Optional[dict] = None
    actual_clicks: int = 0
    stop_reason: Optional[dict] = None
    metric_rows: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pair": self.pair.to_dict(),
            "specimen_dir": str(self.specimen_dir),
            "bright_seed_dir": str(self.bright_seed_dir),
            "gt_path": str(self.gt_path) if self.gt_path else None,
            "cropped_ct_path": str(self.cropped_ct_path) if self.cropped_ct_path else None,
            "actual_clicks": self.actual_clicks,
            "stop_reason": self.stop_reason,
            "metric_rows": self.metric_rows,
            "bright_seed_summary": self.bright_seed_summary,
            "error": self.error,
            "duration_s": self.duration_s,
        }


def run_specimen(pair: SpecimenPair, specimen_dir: Path,
                 budgets: list[int],
                 max_steps: int,
                 intensity_percentile: float,
                 margin_mm: float,
                 base_url: str,
                 parent_logger: RunLogger,
                 max_voxel_axis: Optional[int] = 384,
                 no_screenshots: bool = False,
                 dry_run: bool = False) -> SpecimenResult:
    """Run the full per-specimen pipeline.

    Errors at any stage are captured and the partial result is returned;
    the parent run continues with the next specimen.
    """
    t0 = time.time()
    specimen_dir.mkdir(parents=True, exist_ok=True)
    bright_seed_dir = specimen_dir / "bright_seed"
    result = SpecimenResult(pair=pair, specimen_dir=specimen_dir,
                              bright_seed_dir=bright_seed_dir)

    # Per-specimen inputs.json (records the discovery info verbatim)
    (specimen_dir / "manifest.json").write_text(
        json.dumps({
            "pair": pair.to_dict(),
            "budgets": budgets,
            "max_steps": max_steps,
            "intensity_percentile": intensity_percentile,
            "margin_mm": margin_mm,
            "base_url": base_url,
        }, indent=2, default=str)
    )

    parent_logger.log("")
    parent_logger.log(f"=== Specimen {pair.physical_object_id}  "
                       f"taxon={pair.taxonomy or '?'}  "
                       f"CT={pair.ct_media_id}  mesh={pair.mesh_media_id} ===")
    parent_logger.event("specimen_begin", **pair.to_dict(),
                          specimen_dir=str(specimen_dir))

    if dry_run:
        parent_logger.log("  [dry-run] skipping download/crop/voxelize/run")
        result.duration_s = round(time.time() - t0, 2)
        parent_logger.event("specimen_end_dry", **result.to_dict())
        return result

    # 1. Download
    try:
        dl = download_pair(pair, specimen_dir / "downloads")
    except Exception as exc:
        result.error = f"download_failed: {exc!r}"
        parent_logger.log(f"  download FAILED: {exc!r}")
        parent_logger.event("specimen_failed", reason="download", error=repr(exc),
                              traceback=traceback.format_exc())
        result.duration_s = round(time.time() - t0, 2)
        return result
    if not dl["ct"].get("success") or not dl["mesh"].get("success"):
        result.error = f"download_unsuccessful: ct={dl['ct'].get('error')} mesh={dl['mesh'].get('error')}"
        parent_logger.log(f"  download UNSUCCESSFUL: {result.error}")
        parent_logger.event("specimen_failed", reason="download", details=dl)
        result.duration_s = round(time.time() - t0, 2)
        return result

    # 2. Locate + prepare CT
    ct_dir = Path(dl["ct_dir"])
    mesh_dir = Path(dl["mesh_dir"])
    ct_nifti = specimen_dir / "ct_volume.nii.gz"
    try:
        prep = prepare_ct(ct_dir, ct_nifti, media_id=pair.ct_media_id)
    except Exception as exc:
        result.error = f"prepare_ct_failed: {exc!r}"
        parent_logger.log(f"  prepare_ct FAILED: {exc!r}")
        parent_logger.event("specimen_failed", reason="prepare_ct",
                              error=repr(exc), traceback=traceback.format_exc())
        result.duration_s = round(time.time() - t0, 2)
        return result
    if "error" in prep:
        result.error = f"prepare_ct_error: {prep['error']}"
        parent_logger.log(f"  prepare_ct returned: {prep}")
        parent_logger.event("specimen_failed", reason="prepare_ct", details=prep)
        result.duration_s = round(time.time() - t0, 2)
        return result
    parent_logger.log(f"  CT prepared: {ct_nifti}  (kind={prep.get('kind')})")

    mesh_path = find_mesh(mesh_dir)
    if mesh_path is None:
        result.error = "no_mesh_in_archive"
        parent_logger.log("  no mesh found in mesh download archive")
        parent_logger.event("specimen_failed", reason="no_mesh", mesh_dir=str(mesh_dir))
        result.duration_s = round(time.time() - t0, 2)
        return result
    parent_logger.log(f"  mesh located: {mesh_path}")

    # 3. Crop + voxelize
    try:
        cv = crop_and_voxelize(ct_nifti, mesh_path,
                                 specimen_dir / "preprocessing",
                                 margin_mm=margin_mm,
                                 max_voxel_axis=max_voxel_axis)
    except Exception as exc:
        result.error = f"crop_voxelize_failed: {exc!r}"
        parent_logger.log(f"  crop+voxelize FAILED: {exc!r}")
        parent_logger.event("specimen_failed", reason="crop_voxelize",
                              error=repr(exc), traceback=traceback.format_exc())
        result.duration_s = round(time.time() - t0, 2)
        return result
    if "error" in cv:
        result.error = f"crop_voxelize_error: {cv['error']}"
        parent_logger.log(f"  crop+voxelize returned: {cv}")
        parent_logger.event("specimen_failed", reason="crop_voxelize", details=cv)
        result.duration_s = round(time.time() - t0, 2)
        return result

    cropped_ct = Path(cv["cropped_path"])
    gt_voxelized = Path(cv["voxelized_gt_path"])
    result.cropped_ct_path = cropped_ct
    result.gt_path = gt_voxelized
    parent_logger.log(f"  cropped CT  : {cropped_ct}  "
                       f"({cropped_ct.stat().st_size:,} bytes)")
    parent_logger.log(f"  voxelized GT: {gt_voxelized}  "
                       f"({gt_voxelized.stat().st_size:,} bytes)")

    # 4. Push CT to remote Slicer (chunked /slicer/exec upload, then load)
    try:
        from remote_volume_io import push_volume
        push_name = f"pilot_{pair.physical_object_id}_ct"
        last_pct = [-1]

        def _push_progress(sent: int, total: int) -> None:
            pct = int(100 * sent / max(1, total))
            if pct >= last_pct[0] + 10 or sent == total:
                parent_logger.log(
                    f"  upload {pct:>3d}%  ({sent:,}/{total:,} bytes)"
                )
                last_pct[0] = pct

        push = push_volume(base_url, cropped_ct, name=push_name,
                            chunk_bytes=6 * 1024 * 1024,
                            per_chunk_timeout=60.0,
                            load_timeout=240.0,
                            progress=_push_progress)
    except Exception as exc:
        result.error = f"push_volume_failed: {exc!r}"
        parent_logger.log(f"  push_volume FAILED: {exc!r}")
        parent_logger.event("specimen_failed", reason="push_volume",
                              error=repr(exc), traceback=traceback.format_exc())
        result.duration_s = round(time.time() - t0, 2)
        return result
    if push.get("status") != "ok":
        result.error = f"push_volume_status: {push.get('status')}"
        parent_logger.log(f"  push_volume returned non-ok: {push}")
        parent_logger.event("specimen_failed", reason="push_volume", details=push)
        result.duration_s = round(time.time() - t0, 2)
        return result
    parent_logger.log(f"  pushed to remote Slicer: {push.get('volume_name')}  "
                       f"shape={push.get('shape_kji')}  "
                       f"local_sha256={push.get('local_sha256','')[:16]}…")
    parent_logger.event("volume_pushed", **{
        k: v for k, v in push.items() if k != "_dt_s"
    }, dt_s=push.get("_dt_s"))

    # 5. Run bright-seed (programmatic)
    try:
        rc = run_bright_seed(
            volume_name=push_name,
            out_dir=bright_seed_dir,
            label=pair.slug,
            max_steps=max_steps,
            intensity_percentile=intensity_percentile,
            no_screenshots=no_screenshots,
        )
    except Exception as exc:
        result.error = f"bright_seed_failed: {exc!r}"
        parent_logger.log(f"  bright_seed FAILED: {exc!r}")
        parent_logger.event("specimen_failed", reason="bright_seed",
                              error=repr(exc), traceback=traceback.format_exc())
        result.duration_s = round(time.time() - t0, 2)
        return result
    parent_logger.log(f"  bright_seed exit code: {rc}")
    parent_logger.event("bright_seed_done", exit_code=rc,
                         out_dir=str(bright_seed_dir))

    # bright_seed writes summary.json + per_segment NIfTIs into bright_seed_dir.
    summary_path = bright_seed_dir / "summary.json"
    if summary_path.exists():
        try:
            result.bright_seed_summary = json.loads(summary_path.read_text())
            result.actual_clicks = int(result.bright_seed_summary.get("steps", 0))
            result.stop_reason = result.bright_seed_summary.get("stop_reason")
        except Exception as exc:
            parent_logger.log(f"  WARNING: could not parse summary.json: {exc!r}")

    per_segment = _per_segment_paths_in_click_order(bright_seed_dir)
    parent_logger.log(f"  per-segment artifacts: {len(per_segment)} files")
    if not per_segment:
        result.error = "no_per_segment_artifacts"
        parent_logger.event("specimen_failed", reason="no_per_segment_artifacts",
                              bright_seed_dir=str(bright_seed_dir))
        result.duration_s = round(time.time() - t0, 2)
        return result

    # 6. Post-hoc unions + metrics
    try:
        rows = post_hoc_metrics_for_specimen(
            specimen_dir=specimen_dir,
            gt_path=gt_voxelized,
            per_segment_paths_sorted=per_segment,
            budgets=budgets,
            actual_clicks=result.actual_clicks,
            logger=parent_logger,
        )
    except Exception as exc:
        result.error = f"post_hoc_metrics_failed: {exc!r}"
        parent_logger.log(f"  post_hoc_metrics FAILED: {exc!r}")
        parent_logger.event("specimen_failed", reason="post_hoc_metrics",
                              error=repr(exc), traceback=traceback.format_exc())
        result.duration_s = round(time.time() - t0, 2)
        return result

    result.metric_rows = rows
    for row in rows:
        m = row.get("metrics", {})
        if "error" in m:
            parent_logger.log(f"  budget={row['budget']:>3d}  metrics ERROR: {m['error']}")
        else:
            parent_logger.log(
                f"  budget={row['budget']:>3d}  K_used={row['K_used']:>3d}  "
                f"dice={m.get('dice', 0):.4f}  iou={m.get('iou', 0):.4f}  "
                f"voxels_pred={row.get('composite_voxels_set'):>10,}"
            )

    # Per-specimen report.md
    _write_specimen_report(result)
    result.duration_s = round(time.time() - t0, 2)
    parent_logger.log(f"  specimen done in {result.duration_s:.1f}s")
    parent_logger.event("specimen_end", **result.to_dict())
    return result


def _write_specimen_report(result: SpecimenResult) -> None:
    p = result.pair
    md = []
    md.append(f"# Specimen {p.physical_object_id}")
    md.append("")
    md.append(f"- **Title**: {p.physical_object_title}")
    md.append(f"- **Taxonomy**: {p.taxonomy or '(unknown)'}")
    md.append(f"- **CT media**: {p.ct_media_id} — {p.ct_title}")
    md.append(f"- **Mesh media**: {p.mesh_media_id} — {p.mesh_title}")
    md.append(f"- **Project query**: {p.project_query}")
    md.append("")
    md.append("## Bright-seed run")
    if result.bright_seed_summary:
        bs = result.bright_seed_summary
        md.append(f"- run_id: `{bs.get('run_id')}`")
        md.append(f"- volume_sha256_voxels: `{bs.get('volume_sha256_voxels','')[:32]}`")
        md.append(f"- threshold: {bs.get('threshold')}")
        md.append(f"- shape_kji: {bs.get('shape_kji')}")
        md.append(f"- spacing_mm: {bs.get('spacing_mm')}")
        md.append(f"- actual clicks: {result.actual_clicks}")
        md.append(f"- stop reason: `{result.stop_reason}`")
    md.append("")
    md.append("## Metrics by budget")
    md.append("")
    md.append("| budget | K_used | Dice | IoU | precision | recall | "
                "Hausdorff_mm | mean_surf_dist_mm | voxels_pred | voxels_gt |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result.metric_rows:
        m = row.get("metrics") or {}
        if "error" in m:
            md.append(f"| {row['budget']} | {row.get('K_used','-')} | "
                        "ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR |")
            continue
        md.append("| {b} | {ku} | {d:.4f} | {iou:.4f} | {pr:.4f} | "
                   "{rc:.4f} | {h} | {ms} | {vp} | {vg} |".format(
                       b=row["budget"], ku=row["K_used"],
                       d=m.get("dice", 0), iou=m.get("iou", 0),
                       pr=m.get("precision", 0), rc=m.get("recall", 0),
                       h=("{:.3f}".format(m["hausdorff_mm"])
                            if isinstance(m.get("hausdorff_mm"), (int, float)) else "-"),
                       ms=("{:.4f}".format(m["average_surface_dist_mm"])
                             if isinstance(m.get("average_surface_dist_mm"), (int, float)) else "-"),
                       vp=row.get("composite_voxels_set", 0),
                       vg=m.get("voxel_count_gt", 0),
                   ))
    md.append("")
    md.append("## Composites")
    for row in result.metric_rows:
        if row.get("composite_path"):
            md.append(f"- budget {row['budget']:>3d}: `{row['composite_path']}`  "
                        f"sha256=`{row.get('composite_sha256','')[:16]}…`")
    (result.specimen_dir / "report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Aggregation: results.csv + plot + top-level report
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "specimen_id", "taxon", "genus",
    "ct_media_id", "mesh_media_id",
    "budget", "K_used", "actual_clicks", "stop_reason",
    "dice", "iou", "precision", "recall",
    "false_positive_rate", "false_negative_rate",
    "hausdorff_mm", "hausdorff_95_mm",
    "average_surface_dist_mm", "centroid_distance_mm",
    "voxels_pred", "voxels_gt",
    "volume_mm3_pred", "volume_mm3_gt",
    "spacing_mm", "shape_kji",
    "composite_path", "composite_sha256",
    "click_seconds_p50", "click_seconds_p95", "total_wall_s",
]


def _click_seconds_percentiles(bright_seed_summary: Optional[dict]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not bright_seed_summary:
        return None, None, None
    history = bright_seed_summary.get("history") or []
    secs = [h.get("step_wallclock_s") for h in history
            if isinstance(h.get("step_wallclock_s"), (int, float))]
    if not secs:
        return None, None, None
    secs_sorted = sorted(secs)
    n = len(secs_sorted)

    def _pct(p: float) -> float:
        idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return float(secs_sorted[idx])

    return _pct(50), _pct(95), float(sum(secs))


def aggregate_results(results: list[SpecimenResult],
                       out_dir: Path) -> Path:
    """Write results.csv with one row per (specimen, budget)."""
    out_csv = out_dir / "results.csv"
    rows: list[dict] = []
    for r in results:
        click_p50, click_p95, total_wall = _click_seconds_percentiles(
            r.bright_seed_summary
        )
        for mr in r.metric_rows:
            m = mr.get("metrics") or {}
            stop = (r.stop_reason or {}).get("reason") if r.stop_reason else None
            row = {
                "specimen_id": r.pair.physical_object_id,
                "taxon": r.pair.taxonomy,
                "genus": _genus(r.pair.taxonomy),
                "ct_media_id": r.pair.ct_media_id,
                "mesh_media_id": r.pair.mesh_media_id,
                "budget": mr.get("budget"),
                "K_used": mr.get("K_used"),
                "actual_clicks": r.actual_clicks,
                "stop_reason": stop,
                "dice": m.get("dice"),
                "iou": m.get("iou"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "false_positive_rate": m.get("false_positive_rate"),
                "false_negative_rate": m.get("false_negative_rate"),
                "hausdorff_mm": m.get("hausdorff_mm"),
                "hausdorff_95_mm": m.get("hausdorff_95_mm"),
                "average_surface_dist_mm": m.get("average_surface_dist_mm"),
                "centroid_distance_mm": m.get("centroid_distance_mm"),
                "voxels_pred": mr.get("composite_voxels_set") or m.get("voxel_count_pred"),
                "voxels_gt": m.get("voxel_count_gt"),
                "volume_mm3_pred": m.get("volume_mm3_pred"),
                "volume_mm3_gt": m.get("volume_mm3_gt"),
                "spacing_mm": m.get("spacing_xyz_mm"),
                "shape_kji": m.get("image_shape_zyx"),
                "composite_path": mr.get("composite_path"),
                "composite_sha256": mr.get("composite_sha256"),
                "click_seconds_p50": click_p50,
                "click_seconds_p95": click_p95,
                "total_wall_s": total_wall,
            }
            rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    log.info("Wrote %d rows -> %s", len(rows), out_csv)
    return out_csv


def plot_dice_vs_budget(results: list[SpecimenResult],
                          out_path: Path) -> Optional[Path]:
    """One panel: x=budget, y=Dice, one line per specimen."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("matplotlib unavailable; skipping plot: %s", exc)
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for r in results:
        xs, ys = [], []
        for mr in r.metric_rows:
            m = mr.get("metrics") or {}
            if "dice" in m and isinstance(m["dice"], (int, float)):
                xs.append(mr["budget"])
                ys.append(m["dice"])
        if xs:
            label = (
                f"{_genus(r.pair.taxonomy) or r.pair.physical_object_id}"
                f" ({r.pair.physical_object_id})"
            )
            ax.plot(xs, ys, marker="o", label=label)
            plotted += 1
    if not plotted:
        log.warning("No Dice data to plot")
        plt.close(fig)
        return None
    ax.set_xlabel("Click budget (K)")
    ax.set_ylabel("Dice (composite of segments 1..K vs GT)")
    ax.set_title("Bright-seed nnInteractive: Dice vs click budget")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("Saved Dice-vs-budget plot -> %s", out_path)
    return out_path


def write_top_report(results: list[SpecimenResult],
                       budgets: list[int],
                       out_dir: Path,
                       parent_run_id: str,
                       project_query: str,
                       project_id: str) -> Path:
    md = []
    md.append(f"# Project {project_id} pilot evaluation — bright-seed nnInteractive")
    md.append("")
    md.append(f"- **Run ID**: `{parent_run_id}`")
    md.append(f"- **Project query**: `{project_query}`")
    md.append(f"- **Specimens**: {len(results)}")
    md.append(f"- **Budgets**: {', '.join(str(b) for b in budgets)}")
    md.append("")
    md.append("## Per-specimen Dice")
    md.append("")
    header_budgets = " | ".join(f"K={b}" for b in budgets)
    md.append(f"| Specimen | Taxon | Actual clicks | Stop reason | {header_budgets} |")
    sep = "|---|---|---:|---|" + "---:|" * len(budgets)
    md.append(sep)
    for r in results:
        cells = []
        cells.append(f"`{r.pair.physical_object_id}`")
        cells.append(r.pair.taxonomy or "?")
        cells.append(str(r.actual_clicks))
        cells.append((r.stop_reason or {}).get("reason") if r.stop_reason else "?")
        budget_to_dice: dict[int, Optional[float]] = {b: None for b in budgets}
        for mr in r.metric_rows:
            m = mr.get("metrics") or {}
            if isinstance(m.get("dice"), (int, float)):
                budget_to_dice[mr["budget"]] = float(m["dice"])
        for b in budgets:
            v = budget_to_dice.get(b)
            cells.append("-" if v is None else f"{v:.4f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Artifacts")
    md.append("- [`results.csv`](results.csv) — full per-(specimen, budget) metrics")
    md.append("- [`dice_vs_budget.png`](dice_vs_budget.png) — Dice-vs-budget curve")
    md.append("- `replay.sh` — re-execute this experiment")
    md.append("- `specimens/<id>/` — per-specimen run, including:")
    md.append("    - `bright_seed/` — the deterministic click loop run "
                "(events.jsonl, log.txt, summary.json, artifacts/per_segment/Segment_*.nii.gz)")
    md.append("    - `composites/composite_at_*.nii.gz` — post-hoc unions per budget")
    md.append("    - `metrics/metrics_at_*.json` — full SegMetrics dict per budget")
    md.append("    - `report.md` — per-specimen report")
    md.append("")
    md.append("## Reproducibility")
    md.append("Each per-specimen `bright_seed/manifest.json` records the exact "
                "args; each `bright_seed/inputs.json` records `sha256_voxels` of "
                "the cropped CT. The composites and per-segment NIfTIs are hashed "
                "in `bright_seed/artifacts/index.json` and again in "
                "`composites/composite_at_*.nii.gz`'s sha256 column of `results.csv`.")
    out_path = out_dir / "report.md"
    out_path.write_text("\n".join(md))
    log.info("Wrote top-level report -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_budgets(s: str) -> list[int]:
    out = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    if not out:
        raise argparse.ArgumentTypeError("must provide at least one budget")
    if any(b <= 0 for b in out):
        raise argparse.ArgumentTypeError("budgets must be positive integers")
    return out


def _read_base_url() -> str:
    url = (
        os.environ.get("SLICER_WEBSERVER_URL", "").strip()
        or os.environ.get("NNI_REMOTE_URL", "").strip()
    )
    if not url:
        raise SystemExit("ERROR: SLICER_WEBSERVER_URL (or NNI_REMOTE_URL) "
                          "must be set in .env or the environment.")
    if url.startswith("ws://"):
        url = "http://" + url[len("ws://"):]
    elif url.startswith("wss://"):
        url = "https://" + url[len("wss://"):]
    return url.rstrip("/")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-id", default="000358382",
                   help="MorphoSource project ID (informational; used in run "
                        "metadata only). Default: 000358382")
    p.add_argument("--project-query", default="Colors of Skull Anatomy",
                   help="Search query that returns this project's media. "
                        "Default: %(default)r")
    p.add_argument("--specimens", type=int, default=3,
                   help="Target number of *successful* specimens to score "
                        "(default: 3). The orchestrator keeps trying "
                        "candidates from discovery until this many succeed "
                        "or --max-attempts is reached.")
    p.add_argument("--max-attempts", type=int, default=0,
                   help="Hard cap on specimen attempts; 0 = 2x --specimens. "
                        "Failures (frame-mismatched meshes, push timeouts, "
                        "...) are recorded in events.jsonl and the run "
                        "continues with the next candidate.")
    p.add_argument("--specimens-manifest", type=Path, default=None,
                   help="Optional JSON file pinning specific (CT, mesh) IDs. "
                        "Format: a JSON list of objects matching SpecimenPair "
                        "(at minimum: physical_object_id, ct_media_id, "
                        "mesh_media_id, taxonomy).")
    p.add_argument("--budgets", type=_parse_budgets, default=[10, 25, 50, 100],
                   help="Comma-separated click budgets (default: 10,25,50,100)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max bright-seed clicks per specimen "
                        "(default: max(budgets))")
    p.add_argument("--intensity-percentile", type=float, default=99.0,
                   help="Bright-seed intensity threshold percentile (default 99)")
    p.add_argument("--margin-mm", type=float, default=5.0,
                   help="Crop margin around mesh bbox in mm (default 5)")
    p.add_argument("--max-voxel-axis", type=int, default=384,
                   help="Resample the cropped CT (linear) + voxelized GT "
                        "(nearest neighbour) so no axis exceeds this many "
                        "voxels. Keeps the base64-over-/slicer/exec push "
                        "below the proxy's HTTP body cap. Set 0 to disable. "
                        "Default: 384.")
    p.add_argument("--max-ct-gb", type=float, default=5.0,
                   help="Skip specimens whose CT archive is larger than this "
                        "(default 5 GB; set 0 to disable)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("runs") / time.strftime("pilot_%Y%m%d_%H%M%S"))
    p.add_argument("--label", default="project358382_pilot",
                   help="Label embedded in the parent run id (default: "
                        "project358382_pilot)")
    p.add_argument("--no-screenshots", action="store_true",
                   help="Pass --no-screenshots to bright_seed (faster)")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover + select + log everything, but skip the "
                        "download / crop / voxelize / push / run / score "
                        "stages. Useful for validating the discovery output.")
    p.add_argument("--page-size", type=int, default=50,
                   help="MorphoSource search per_page (default 50)")
    p.add_argument("--max-pages", type=int, default=5,
                   help="Max pages to pull during discovery (default 5)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    base_url = _read_base_url()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parent_logger = RunLogger.start(
        root=args.out_dir,
        args={k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()},
        label=args.label,
    )
    parent_logger.log(f"=== Project {args.project_id} pilot evaluation ===")
    parent_logger.log(f"run id        : {parent_logger.run_id}")
    parent_logger.log(f"server        : {base_url}")
    parent_logger.log(f"out           : {args.out_dir}")
    parent_logger.log(f"specimens     : {args.specimens}")
    parent_logger.log(f"budgets       : {args.budgets}")
    parent_logger.log(f"intensity_pct : {args.intensity_percentile}")
    parent_logger.log(f"margin_mm     : {args.margin_mm}")

    local_env = parent_logger.record_local_env()
    parent_logger.log(f"git commit    : {local_env.get('git_commit')}  "
                       f"dirty={local_env.get('git_dirty')}")

    # ---- Specimen selection ----
    if args.specimens_manifest:
        try:
            manifest_data = json.loads(args.specimens_manifest.read_text())
        except Exception as exc:
            parent_logger.log(f"FAILED to read manifest {args.specimens_manifest}: {exc!r}")
            parent_logger.finalize(stop_reason={"reason": "bad_manifest",
                                                  "error": repr(exc)})
            return 2
        chosen: list[SpecimenPair] = []
        for entry in manifest_data:
            try:
                chosen.append(SpecimenPair(**{
                    "physical_object_id": entry["physical_object_id"],
                    "physical_object_title": entry.get("physical_object_title", ""),
                    "taxonomy": entry.get("taxonomy", ""),
                    "ct_media_id": entry["ct_media_id"],
                    "ct_title": entry.get("ct_title", ""),
                    "ct_media_type": entry.get("ct_media_type", ""),
                    "ct_file_size": entry.get("ct_file_size"),
                    "mesh_media_id": entry["mesh_media_id"],
                    "mesh_title": entry.get("mesh_title", ""),
                    "mesh_media_type": entry.get("mesh_media_type", ""),
                    "mesh_file_size": entry.get("mesh_file_size"),
                    "project_query": entry.get("project_query", args.project_query),
                }))
            except KeyError as exc:
                parent_logger.log(f"manifest entry missing key {exc}: {entry!r}")
                parent_logger.finalize(stop_reason={"reason": "bad_manifest_entry",
                                                      "missing_key": str(exc)})
                return 2
        if args.specimens > 0:
            chosen = chosen[:args.specimens]
        parent_logger.log(f"manifest pinned {len(chosen)} specimens")
    else:
        client = MorphoSourceClient()
        all_pairs = discover_pairs(
            client=client,
            project_query=args.project_query,
            project_id=args.project_id,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )
        max_bytes = (
            None if args.max_ct_gb <= 0
            else int(args.max_ct_gb * (1 << 30))
        )
        # Pull a few extra candidates so the loop has fallbacks when the
        # first picks fail (e.g. coordinate-frame mismatches).
        attempt_cap = args.max_attempts or max(args.specimens * 2,
                                                  args.specimens + 2)
        chosen = select_pilot_specimens(
            all_pairs, n=attempt_cap, max_ct_bytes=max_bytes,
        )
        parent_logger.log(f"discovery returned {len(all_pairs)} specimens; "
                           f"queued {len(chosen)} candidates "
                           f"(targeting {args.specimens} successes)")

    # Persist the chosen specimen list as part of inputs.json
    inputs_payload = {
        "project_id": args.project_id,
        "project_query": args.project_query,
        "budgets": args.budgets,
        "max_steps": args.max_steps or max(args.budgets),
        "intensity_percentile": args.intensity_percentile,
        "margin_mm": args.margin_mm,
        "max_ct_gb": args.max_ct_gb,
        "specimens": [c.to_dict() for c in chosen],
        "base_url": base_url,
    }
    parent_logger.record_inputs(inputs_payload)

    if not chosen:
        parent_logger.log("FAILED: no specimens selected")
        parent_logger.finalize(stop_reason={"reason": "no_specimens"})
        return 3

    parent_logger.log("Selected specimens:")
    for i, p in enumerate(chosen, 1):
        size_gb = (p.ct_file_size or 0) / (1 << 30) if p.ct_file_size else 0.0
        parent_logger.log(f"  {i}. {p.physical_object_id}  "
                           f"taxon={p.taxonomy or '?'}  "
                           f"CT={p.ct_media_id} ({size_gb:.2f} GB)  "
                           f"mesh={p.mesh_media_id}")

    # ---- Per-specimen pipeline ----
    max_steps = args.max_steps or max(args.budgets)
    results: list[SpecimenResult] = []
    n_succeeded = 0
    specimens_root = args.out_dir / "specimens"
    specimens_root.mkdir(parents=True, exist_ok=True)
    target = args.specimens if not args.specimens_manifest else len(chosen)
    for i, pair in enumerate(chosen, 1):
        if not args.dry_run and n_succeeded >= target:
            parent_logger.log(f"hit target {target} successes after "
                               f"{i-1} attempts; stopping early")
            parent_logger.event("target_reached",
                                  successes=n_succeeded,
                                  attempts=i - 1, target=target)
            break
        spec_dir = specimens_root / pair.slug
        try:
            r = run_specimen(
                pair=pair, specimen_dir=spec_dir,
                budgets=args.budgets,
                max_steps=max_steps,
                intensity_percentile=args.intensity_percentile,
                margin_mm=args.margin_mm,
                base_url=base_url,
                parent_logger=parent_logger,
                max_voxel_axis=args.max_voxel_axis,
                no_screenshots=args.no_screenshots,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            parent_logger.log(f"  catastrophic failure on specimen "
                               f"{pair.physical_object_id}: {exc!r}")
            parent_logger.event("specimen_catastrophic", **pair.to_dict(),
                                  error=repr(exc),
                                  traceback=traceback.format_exc())
            r = SpecimenResult(
                pair=pair, specimen_dir=spec_dir,
                bright_seed_dir=spec_dir / "bright_seed",
                error=f"catastrophic: {exc!r}",
            )
        results.append(r)
        if not r.error and r.metric_rows:
            n_succeeded += 1

    # ---- Aggregate ----
    if not args.dry_run:
        try:
            aggregate_results(results, args.out_dir)
        except Exception as exc:
            parent_logger.log(f"aggregate_results failed: {exc!r}")
            parent_logger.event("aggregate_failed", error=repr(exc),
                                  traceback=traceback.format_exc())
        try:
            plot_dice_vs_budget(results, args.out_dir / "dice_vs_budget.png")
        except Exception as exc:
            parent_logger.log(f"plot_dice_vs_budget failed: {exc!r}")
            parent_logger.event("plot_failed", error=repr(exc),
                                  traceback=traceback.format_exc())

    write_top_report(
        results=results, budgets=args.budgets,
        out_dir=args.out_dir,
        parent_run_id=parent_logger.run_id,
        project_query=args.project_query,
        project_id=args.project_id,
    )

    # ---- Replay script (top-level pilot) ----
    # Use an absolute path to the orchestrator + the venv interpreter
    # that was used for this run, so the replay script works regardless
    # of how deep ``out_dir`` is nested under the repo root.
    repo_root = SCRIPT_DIR.parents[1]
    orchestrator_path = SCRIPT_DIR / "eval_project358382_pilot.py"
    try:
        orch_str = str(orchestrator_path.relative_to(repo_root))
    except ValueError:
        orch_str = str(orchestrator_path)
    replay_cmd = [
        f'"{sys.executable}" "{orch_str}"',
        f'--project-id {args.project_id}',
        f'--project-query "{args.project_query}"',
        f'--specimens {args.specimens}',
        f'--budgets {",".join(str(b) for b in args.budgets)}',
        f'--max-steps {max_steps}',
        f'--intensity-percentile {args.intensity_percentile}',
        f'--margin-mm {args.margin_mm}',
        f'--max-voxel-axis {args.max_voxel_axis}',
        f'--max-ct-gb {args.max_ct_gb}',
        f'--out-dir runs/replay_{parent_logger.run_id}',
        f'--label "{args.label}"',
    ]
    if args.no_screenshots:
        replay_cmd.append("--no-screenshots")
    parent_logger.write_replay(
        command=replay_cmd,
        env_keys=[
            "SLICER_WEBSERVER_URL", "NNI_REMOTE_URL",
            "MORPHOSOURCE_API_KEY", "OPENAI_API_KEY",
        ],
    )

    # ---- Console summary ----
    parent_logger.log("")
    parent_logger.log("Per-specimen summary:")
    for r in results:
        if r.error:
            parent_logger.log(f"  {r.pair.physical_object_id}: ERROR={r.error}")
            continue
        bests = []
        for mr in r.metric_rows:
            m = mr.get("metrics") or {}
            if isinstance(m.get("dice"), (int, float)):
                bests.append((mr["budget"], m["dice"]))
        bests_str = "  ".join(f"K={b}:dice={d:.3f}" for b, d in bests)
        parent_logger.log(f"  {r.pair.physical_object_id}  "
                           f"({r.pair.taxonomy or '?'})  {bests_str}")

    n_failed = sum(1 for r in results if r.error)
    parent_logger.log("")
    parent_logger.log(f"DONE. {len(results) - n_failed}/{len(results)} specimens "
                       f"succeeded.")
    parent_logger.log(f"  results.csv -> {args.out_dir / 'results.csv'}")
    parent_logger.log(f"  plot        -> {args.out_dir / 'dice_vs_budget.png'}")
    parent_logger.log(f"  report      -> {args.out_dir / 'report.md'}")
    parent_logger.log(f"  replay      -> {parent_logger.replay_path}")

    parent_logger.finalize(stop_reason={
        "reason": "done" if n_failed == 0 else "partial",
        "n_specimens": len(results),
        "n_failed": n_failed,
    })
    return 0 if n_failed == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
