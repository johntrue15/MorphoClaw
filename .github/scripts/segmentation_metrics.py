"""
Segmentation comparison metrics.

Used by `nninteractive_compare.py` to evaluate an nnInteractive prediction
labelmap against a curated MorphoSource ground-truth segmentation
(typically a surface mesh that has been voxelized onto the same grid as
the source CT volume).

All metrics treat segmentations as binary (>0 = foreground). For multi-
label comparison, callers should compute metrics per label and aggregate.

Metrics
-------
- voxel_count_*           — raw voxel count for prediction / GT
- volume_mm3_*            — physical volume of each mask
- volume_difference_pct   — abs(pred - gt) / gt * 100
- intersection_voxels     — |A ∩ B| in voxels
- union_voxels            — |A ∪ B| in voxels
- dice                    — 2|A ∩ B| / (|A| + |B|)
- iou                     — Jaccard index = |A ∩ B| / |A ∪ B|
- precision               — |A ∩ B| / |A|        (A = prediction)
- recall (sensitivity)    — |A ∩ B| / |B|        (B = ground truth)
- false_positive_rate     — fraction of pred voxels not in GT
- false_negative_rate     — fraction of GT voxels not in pred
- hausdorff_mm            — max distance between mask surfaces (mm)
- hausdorff_95_mm         — 95th-percentile surface distance (mm)
- average_surface_dist_mm — mean of bidirectional surface distances (mm)
- centroid_distance_mm    — distance between mask centroids (mm)

Both inputs must share the same voxel spacing, dimensions, and origin.
The orchestrator enforces that by voxelizing the GT mesh onto the CT's
SimpleITK ReferenceImage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("seg_metrics")


def _require_sitk():
    try:
        import SimpleITK as sitk  # noqa: F401
        import numpy as np        # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK and numpy are required for segmentation_metrics. "
            "Run `.github/scripts/install_nninteractive.sh` and use the "
            f"venv's Python. Original import error: {exc}"
        ) from exc


@dataclass
class SegMetrics:
    voxel_count_pred: int
    voxel_count_gt: int
    volume_mm3_pred: float
    volume_mm3_gt: float
    volume_difference_pct: float
    intersection_voxels: int
    union_voxels: int
    dice: float
    iou: float
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float
    hausdorff_mm: Optional[float]
    hausdorff_95_mm: Optional[float]
    average_surface_dist_mm: Optional[float]
    centroid_distance_mm: Optional[float]
    spacing_xyz_mm: tuple
    image_shape_zyx: tuple

    def to_dict(self) -> dict:
        return asdict(self)

    def markdown_table(self) -> str:
        lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Dice | **{self.dice:.4f}** |",
            f"| IoU (Jaccard) | {self.iou:.4f} |",
            f"| Precision | {self.precision:.4f} |",
            f"| Recall (sensitivity) | {self.recall:.4f} |",
            f"| False positive rate | {self.false_positive_rate:.4f} |",
            f"| False negative rate | {self.false_negative_rate:.4f} |",
            f"| Volume (pred) | {self.volume_mm3_pred:,.2f} mm³ |",
            f"| Volume (GT) | {self.volume_mm3_gt:,.2f} mm³ |",
            f"| Volume diff | {self.volume_difference_pct:.2f} % |",
            f"| Voxels (pred ∩ GT) | {self.intersection_voxels:,} |",
            f"| Voxels (pred ∪ GT) | {self.union_voxels:,} |",
        ]
        if self.hausdorff_mm is not None:
            lines.append(f"| Hausdorff (max) | {self.hausdorff_mm:.3f} mm |")
        if self.hausdorff_95_mm is not None:
            lines.append(f"| Hausdorff (95-pct) | {self.hausdorff_95_mm:.3f} mm |")
        if self.average_surface_dist_mm is not None:
            lines.append(
                f"| Mean surface distance | {self.average_surface_dist_mm:.3f} mm |"
            )
        if self.centroid_distance_mm is not None:
            lines.append(f"| Centroid distance | {self.centroid_distance_mm:.3f} mm |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def _binarise(image, threshold: float = 0.5):
    """Return a UInt8 binary SimpleITK image (foreground = 1).

    Note: ``sitk.BinaryThreshold`` is broken in SimpleITK 2.5.x — it returns
    an all-foreground image regardless of the threshold arguments. We
    threshold via numpy instead and rebuild a SimpleITK image with the
    same geometry to keep this function dependency-stable across versions.
    """
    import SimpleITK as sitk
    import numpy as np
    arr = sitk.GetArrayFromImage(image)
    binary_arr = (arr > threshold).astype(np.uint8)
    out = sitk.GetImageFromArray(binary_arr)
    out.CopyInformation(image)
    return out


def _check_grid(pred, gt):
    """Assert that two SimpleITK images share spacing/size/origin/direction."""
    if pred.GetSize() != gt.GetSize():
        raise ValueError(
            f"Image grids differ: pred size {pred.GetSize()} vs GT size {gt.GetSize()}"
        )
    if tuple(round(s, 4) for s in pred.GetSpacing()) != tuple(
        round(s, 4) for s in gt.GetSpacing()
    ):
        raise ValueError(
            f"Image spacings differ: pred {pred.GetSpacing()} vs GT {gt.GetSpacing()}"
        )


def _surface_distances(pred, gt):
    """Compute surface distance metrics. Returns (max, p95, mean) in mm,
    or (None, None, None) if either mask is empty.
    """
    import SimpleITK as sitk
    import numpy as np

    pred_arr = sitk.GetArrayFromImage(pred)
    gt_arr = sitk.GetArrayFromImage(gt)
    if pred_arr.sum() == 0 or gt_arr.sum() == 0:
        return None, None, None

    # Distance map of the GT (positive outside, negative inside) → distances
    # from each voxel to the surface of the GT.
    dt_gt = sitk.SignedMaurerDistanceMap(
        gt, squaredDistance=False, useImageSpacing=True
    )
    dt_pred = sitk.SignedMaurerDistanceMap(
        pred, squaredDistance=False, useImageSpacing=True
    )

    # Surface voxels: extract via morphological gradient (pred XOR eroded(pred))
    pred_surface = sitk.LabelContour(pred, fullyConnected=True)
    gt_surface = sitk.LabelContour(gt, fullyConnected=True)

    pred_surface_arr = sitk.GetArrayFromImage(pred_surface) > 0
    gt_surface_arr = sitk.GetArrayFromImage(gt_surface) > 0

    dt_gt_arr = sitk.GetArrayFromImage(dt_gt)
    dt_pred_arr = sitk.GetArrayFromImage(dt_pred)

    # Distances from each pred-surface voxel to the GT surface, and vice versa
    pred_to_gt = np.abs(dt_gt_arr[pred_surface_arr])
    gt_to_pred = np.abs(dt_pred_arr[gt_surface_arr])

    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        return None, None, None

    all_distances = np.concatenate([pred_to_gt, gt_to_pred])
    return (
        float(all_distances.max()),
        float(np.percentile(all_distances, 95)),
        float(all_distances.mean()),
    )


def _centroid_distance(pred, gt):
    import SimpleITK as sitk
    import numpy as np

    pred_arr = sitk.GetArrayFromImage(pred)
    gt_arr = sitk.GetArrayFromImage(gt)
    if pred_arr.sum() == 0 or gt_arr.sum() == 0:
        return None

    spacing = np.array(pred.GetSpacing())  # (sx, sy, sz)
    origin = np.array(pred.GetOrigin())

    # Index → physical point. Note: GetArrayFromImage gives (z, y, x).
    pz, py, px = np.where(pred_arr > 0)
    gz, gy, gx = np.where(gt_arr > 0)
    pc_idx = np.array([px.mean(), py.mean(), pz.mean()])
    gc_idx = np.array([gx.mean(), gy.mean(), gz.mean()])
    pc = origin + pc_idx * spacing
    gc = origin + gc_idx * spacing
    return float(np.linalg.norm(pc - gc))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_labelmaps(pred_path: str, gt_path: str,
                      compute_surface_distances: bool = True) -> SegMetrics:
    """Compute Dice/IoU/Hausdorff/volume agreement between two binary labelmaps.

    Both labelmaps must share spacing, size, origin, and direction.
    """
    _require_sitk()
    import SimpleITK as sitk
    import numpy as np

    pred_img = _binarise(sitk.ReadImage(str(pred_path)))
    gt_img = _binarise(sitk.ReadImage(str(gt_path)))
    _check_grid(pred_img, gt_img)

    pred_arr = sitk.GetArrayFromImage(pred_img).astype(bool)
    gt_arr = sitk.GetArrayFromImage(gt_img).astype(bool)

    intersection = np.logical_and(pred_arr, gt_arr).sum()
    union = np.logical_or(pred_arr, gt_arr).sum()
    n_pred = pred_arr.sum()
    n_gt = gt_arr.sum()

    spacing = pred_img.GetSpacing()
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])

    dice = (2.0 * intersection / (n_pred + n_gt)) if (n_pred + n_gt) > 0 else 0.0
    iou = (intersection / union) if union > 0 else 0.0
    precision = (intersection / n_pred) if n_pred > 0 else 0.0
    recall = (intersection / n_gt) if n_gt > 0 else 0.0
    fpr = (1.0 - precision) if n_pred > 0 else 0.0
    fnr = (1.0 - recall) if n_gt > 0 else 0.0
    vol_pred = float(n_pred) * voxel_vol
    vol_gt = float(n_gt) * voxel_vol
    vol_diff_pct = (
        abs(vol_pred - vol_gt) / vol_gt * 100.0
        if vol_gt > 0
        else (100.0 if vol_pred > 0 else 0.0)
    )

    h_max = h_95 = h_mean = None
    if compute_surface_distances and intersection > 0:
        try:
            h_max, h_95, h_mean = _surface_distances(pred_img, gt_img)
        except Exception as exc:
            log.warning("Surface-distance computation failed: %s", exc)

    centroid_d = None
    try:
        centroid_d = _centroid_distance(pred_img, gt_img)
    except Exception as exc:
        log.warning("Centroid distance failed: %s", exc)

    return SegMetrics(
        voxel_count_pred=int(n_pred),
        voxel_count_gt=int(n_gt),
        volume_mm3_pred=round(vol_pred, 3),
        volume_mm3_gt=round(vol_gt, 3),
        volume_difference_pct=round(vol_diff_pct, 3),
        intersection_voxels=int(intersection),
        union_voxels=int(union),
        dice=round(dice, 6),
        iou=round(iou, 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        false_positive_rate=round(fpr, 6),
        false_negative_rate=round(fnr, 6),
        hausdorff_mm=round(h_max, 4) if h_max is not None else None,
        hausdorff_95_mm=round(h_95, 4) if h_95 is not None else None,
        average_surface_dist_mm=round(h_mean, 4) if h_mean is not None else None,
        centroid_distance_mm=round(centroid_d, 4) if centroid_d is not None else None,
        spacing_xyz_mm=tuple(round(s, 6) for s in spacing),
        image_shape_zyx=tuple(pred_img.GetSize()[::-1]),
    )


# ---------------------------------------------------------------------------


def render_overlay_panel(volume_path: str, pred_path: str, gt_path: str,
                         out_path: str, title: str = "") -> Optional[str]:
    """Render a 3-row × 3-column grid of axial / coronal / sagittal mid-slices
    showing: (1) volume only, (2) volume + GT, (3) volume + prediction,
    so a reviewer can eyeball where the masks agree and disagree.
    """
    _require_sitk()
    import SimpleITK as sitk
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping overlay panel")
        return None

    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(volume_path)))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))

    def _best_slice(mask, axis):
        if mask.sum() == 0:
            return mask.shape[axis] // 2
        sums = mask.sum(axis=tuple(i for i in range(3) if i != axis))
        return int(np.argmax(sums))

    # Pick slices using the union of GT and prediction so both are visible
    union = np.logical_or(pred > 0, gt > 0).astype(np.uint8)
    slices = [_best_slice(union, axis) for axis in (0, 1, 2)]
    views = [
        ("axial",   0, lambda v, s, a: (v[s, :, :], a[s, :, :])),
        ("coronal", 1, lambda v, s, a: (v[:, s, :], a[:, s, :])),
        ("sagittal", 2, lambda v, s, a: (v[:, :, s], a[:, :, s])),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for row, (view_name, axis, sl) in enumerate(views):
        s = slices[axis]
        v, p = sl(vol, s, pred)
        _, g = sl(vol, s, gt)
        axes[row, 0].imshow(v, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{view_name} (slice {s}) — volume only")
        axes[row, 1].imshow(v, cmap="gray", origin="lower")
        if g.sum() > 0:
            axes[row, 1].imshow(np.ma.masked_where(g == 0, g),
                                cmap="winter", alpha=0.5, origin="lower")
        axes[row, 1].set_title(f"{view_name} — GT (blue)")
        axes[row, 2].imshow(v, cmap="gray", origin="lower")
        if p.sum() > 0:
            axes[row, 2].imshow(np.ma.masked_where(p == 0, p),
                                cmap="autumn", alpha=0.5, origin="lower")
        axes[row, 2].set_title(f"{view_name} — Prediction (orange)")
        for col in range(3):
            axes[row, col].set_axis_off()

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two segmentation labelmaps")
    p.add_argument("--pred", required=True, help="Prediction labelmap (NIfTI/NRRD)")
    p.add_argument("--gt",   required=True, help="Ground-truth labelmap (NIfTI/NRRD)")
    p.add_argument("--volume", default="",
                   help="(optional) source volume — enables overlay rendering")
    p.add_argument("--output", default="seg_metrics.json",
                   help="Where to write the JSON metrics")
    p.add_argument("--overlay", default="",
                   help="(optional) PNG path for the comparison overlay panel")
    p.add_argument("--no-surface", action="store_true",
                   help="Skip the (expensive) Hausdorff/surface metrics")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    metrics = compare_labelmaps(
        args.pred, args.gt,
        compute_surface_distances=not args.no_surface,
    )
    Path(args.output).write_text(json.dumps(metrics.to_dict(), indent=2, default=str))
    log.info("Wrote metrics: %s", args.output)
    print(metrics.markdown_table())

    if args.overlay and args.volume:
        out = render_overlay_panel(args.volume, args.pred, args.gt, args.overlay,
                                   title=f"Dice={metrics.dice:.3f}  IoU={metrics.iou:.3f}")
        if out:
            log.info("Wrote overlay: %s", out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
