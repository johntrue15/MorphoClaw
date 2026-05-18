"""Command-line entry point for the iterative segmentation trainer.

Three subcommands wire the package's pieces into reproducible runs:

- ``discover``  : pull MorphoSource (CT, mesh) pairs, resolve volume
                   paths via the existing downloader, and write a
                   specimen list (``specimens.json``) the trainer can
                   replay later.
- ``round``     : run one iterative round on a specimen list. Writes a
                   ``round_NNN/round_report.json`` and updates the
                   ledger + manifest. Optionally trains a fresh student.
- ``export``    : emit the paper artefacts (CSVs / plots / Markdown
                   summary) from the current ledger.

Typical workflow on the runner:

.. code-block:: bash

    # 1. Bootstrap nnInteractive once (existing script):
    .github/scripts/install_nninteractive.sh
    .github/scripts/install_seg_train_extras.sh   # MONAI on the same venv

    # 2. Discover 25 candidate specimens with curated GT meshes:
    python -m metadata_to_morphsource.seg_train discover \\
        --query "primate skull mesh" --max-pairs 25 \\
        --output runs/skull_v1/specimens.json

    # 3. Run round 0 (no student yet — pure nnInteractive seed):
    python -m metadata_to_morphsource.seg_train round \\
        --specimens runs/skull_v1/specimens.json \\
        --run-dir   runs/skull_v1 \\
        --paper-tag chameleon_skull_v1 \\
        --goal "Segment the cranial bone"

    # 4. Subsequent rounds use the student trained in the previous
    #    round and let the router decide when to invoke nnInteractive:
    python -m metadata_to_morphsource.seg_train round \\
        --specimens runs/skull_v1/specimens.json \\
        --run-dir   runs/skull_v1 \\
        --student   runs/skull_v1/round_000/student_weights/student_r000.artifact.json

    # 5. Export paper data at any point:
    python -m metadata_to_morphsource.seg_train export \\
        --run-dir   runs/skull_v1 \\
        --output    runs/skull_v1/paper
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .confidence_router import ConfidenceRouter, RouterPolicy
from .dataset import DatasetManifest
from .experiment_ledger import ExperimentLedger
from .iterative_trainer import (
    IterativeTrainer, RoundConfig, SpecimenInput,
)
from .paper_export import export_paper_artifacts
from .prepare_specimen import (
    PRESETS as PREPARE_PRESETS,
    prepare_preset,
    prepare_specimen,
    write_specimens_json,
)


log = logging.getLogger("seg_train.cli")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


# ---------------------------------------------------------------------------
# Subcommand: discover
# ---------------------------------------------------------------------------


def _cmd_discover(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        from find_segmentation_pairs import find_pairs  # type: ignore
        from morphosource_api_download import download_media  # type: ignore
    except Exception as exc:
        log.error("MorphoSource helpers unavailable: %s", exc)
        return 2

    pairs = find_pairs(
        query=args.query,
        max_pairs=args.max_pairs,
        max_candidates=args.max_candidates,
        require_taxonomy=args.require_taxonomy,
    )
    if not pairs:
        log.error("No (CT, mesh) pairs found for query=%r", args.query)
        return 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    download_root = Path(args.download_dir or
                         (out.parent / "downloads")).resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    specimens: list[dict] = []
    for i, pair in enumerate(pairs):
        ct_id = pair.ct["media_id"]
        gt_id = pair.mesh["media_id"]
        ct_dir = download_root / f"ct_{ct_id}"
        gt_dir = download_root / f"gt_{gt_id}"
        log.info("[%d/%d] Downloading CT %s + mesh %s",
                 i + 1, len(pairs), ct_id, gt_id)

        ct_dl = download_media(ct_id, str(ct_dir))
        gt_dl = download_media(gt_id, str(gt_dir))
        if not (ct_dl.get("success") and gt_dl.get("success")):
            log.warning("Skipping pair %s/%s: CT=%s GT=%s",
                        ct_id, gt_id,
                        ct_dl.get("error", "ok"),
                        gt_dl.get("error", "ok"))
            continue

        # Find largest volume + mesh files inside the downloads.
        try:
            from .._compat_filepicker import (  # type: ignore
                find_volume_file, find_mesh_file,
            )
        except Exception:
            volume_file = _largest_with_extensions(
                ct_dir,
                {".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd"},
            )
            mesh_file = _largest_with_extensions(
                gt_dir, {".ply", ".stl", ".obj", ".off"},
            )
        else:
            volume_file = find_volume_file(ct_dir)
            mesh_file = find_mesh_file(gt_dir)

        if not volume_file or not mesh_file:
            log.warning(
                "Could not locate volume/mesh under %s and %s",
                ct_dir, gt_dir,
            )
            continue

        specimens.append({
            "media_id": ct_id,
            "physical_object_id": pair.physical_object_id,
            "taxonomy": pair.taxonomy,
            "morphosource_query": args.query,
            "volume_path": str(volume_file),
            "gt_mesh_path": str(mesh_file),
            "gt_media_id": gt_id,
        })

    out.write_text(json.dumps(specimens, indent=2))
    log.info("Wrote %d specimens to %s", len(specimens), out)
    return 0 if specimens else 2


def _largest_with_extensions(directory: Path, extensions: set[str]) -> Optional[Path]:
    best: Optional[Path] = None
    best_size = -1
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        for ext in extensions:
            if name.endswith(ext):
                size = p.stat().st_size
                if size > best_size:
                    best, best_size = p, size
                break
    return best


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------


def _cmd_prepare(args: argparse.Namespace) -> int:
    """Download + voxelise one or more real (CT, GT) pairs and emit a
    trainer-ready ``specimens.json``. The actual paint loop is *not*
    executed here — that happens in ``seg_train round``.
    """
    prepared: list = []
    pairs: list[dict] = []

    if args.preset:
        if args.preset not in PREPARE_PRESETS:
            log.error("Unknown preset %r. Available: %s",
                      args.preset, sorted(PREPARE_PRESETS))
            return 2
        try:
            ps = prepare_preset(
                args.preset,
                output_dir=args.prep_dir,
                voxelize_backend=args.voxelize_backend or None,
                use_nninteractive_python=not args.system_python,
                timeout_s=args.prep_timeout,
            )
        except Exception as exc:
            log.exception("Prep failed for preset %s: %s", args.preset, exc)
            return 1
        prepared.append(ps)
    else:
        if not args.ct_media_id or not args.gt_media_id:
            log.error("Either --preset or both --ct-media-id and "
                      "--gt-media-id are required.")
            return 2
        pairs.append({
            "ct_media_id": args.ct_media_id,
            "gt_media_id": args.gt_media_id,
            "physical_object_id": args.physical_object_id,
            "taxonomy": args.taxonomy,
            "morphosource_query": args.morphosource_query,
            "goal": args.goal or "",
        })
        for pair in pairs:
            try:
                ps = prepare_specimen(
                    ct_media_id=pair["ct_media_id"],
                    gt_media_id=pair["gt_media_id"],
                    output_dir=args.prep_dir,
                    goal=pair["goal"],
                    physical_object_id=pair["physical_object_id"],
                    taxonomy=pair["taxonomy"],
                    morphosource_query=pair["morphosource_query"],
                    crop_around_mesh_mm=args.crop_mm,
                    voxelize_backend=args.voxelize_backend or "auto",
                    use_nninteractive_python=not args.system_python,
                    timeout_s=args.prep_timeout,
                )
            except Exception as exc:
                log.exception(
                    "Prep failed for %s vs %s: %s",
                    pair["ct_media_id"], pair["gt_media_id"], exc,
                )
                return 1
            prepared.append(ps)

    out = Path(args.output)
    write_specimens_json(prepared, out)

    digest = [{
        "ct_media_id": p.ct_media_id,
        "gt_media_id": p.gt_media_id,
        "volume_path": p.volume_path,
        "gt_label_path": p.gt_label_path,
        "gt_mesh_path": p.gt_mesh_path,
        "duration_s": p.duration_s,
        "fg_voxels": p.voxelize_summary.get("foreground_voxels"),
        "fg_mm3": p.voxelize_summary.get("foreground_volume_mm3"),
    } for p in prepared]
    print(json.dumps({"specimens_json": str(out),
                      "n_prepared": len(prepared),
                      "details": digest}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: round
# ---------------------------------------------------------------------------


def _cmd_round(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = run_dir / "ledger"
    manifest_path = run_dir / "manifest.json"

    ledger = ExperimentLedger(ledger_dir, run_id=args.run_id or "",
                              paper_tag=args.paper_tag or "")
    if manifest_path.exists():
        manifest = DatasetManifest.load(str(manifest_path))
    else:
        manifest = DatasetManifest(root=str(run_dir / "manifest_root"),
                                   seed=args.seed)
        manifest.save(str(manifest_path))

    policy = RouterPolicy(
        min_confidence=args.router_min_confidence,
        max_entropy=args.router_max_entropy,
        require_uncertainty=args.router_require_uncertainty,
        name=args.router_name,
    )

    cfg = RoundConfig(
        goal=args.goal,
        output_root=str(run_dir),
        paper_tag=args.paper_tag or "",
        max_paint_steps=args.max_paint_steps,
        train_after_round=not args.skip_training,
        student_epochs=args.student_epochs,
        n_dropout_samples=args.n_dropout_samples,
        graduation_dice_threshold=args.graduation_dice_threshold,
        student_min_reliability=args.student_min_reliability,
        seed=args.seed,
    )

    trainer = IterativeTrainer(
        ledger=ledger,
        manifest=manifest,
        router=ConfidenceRouter(policy),
        round_config=cfg,
    )

    specimens_data = json.loads(Path(args.specimens).read_text())
    specimens = [SpecimenInput(**d) for d in specimens_data]
    log.info("Loaded %d specimens from %s", len(specimens), args.specimens)

    report = trainer.run_round(
        specimens=specimens,
        student_artifact_path=args.student or "",
        skip_training=args.skip_training,
    )

    # Persist manifest one more time.
    manifest.save(str(manifest_path))

    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------


def _cmd_export(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    ledger_dir = run_dir / "ledger"
    if not (ledger_dir / "ledger.jsonl").exists():
        log.error("No ledger found at %s; run a round first", ledger_dir)
        return 2

    ledger = ExperimentLedger(ledger_dir)
    out = Path(args.output or (run_dir / "paper")).resolve()
    info = export_paper_artifacts(
        ledger, output_dir=out,
        paper_tag=args.paper_tag or None,
        include_plots=not args.no_plots,
    )
    print(json.dumps(info, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: summary
# ---------------------------------------------------------------------------


def _cmd_summary(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    ledger_dir = run_dir / "ledger"
    if not ledger_dir.exists():
        log.error("No ledger directory at %s", ledger_dir)
        return 2
    ledger = ExperimentLedger(ledger_dir)
    summary = ledger.summary()
    print(json.dumps(summary, indent=2, default=str))
    if args.markdown:
        md_path = ledger.export_summary_markdown(args.markdown)
        log.info("Wrote markdown summary to %s", md_path)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="metadata_to_morphsource.seg_train",
        description="Iteratively train a segmentation student model "
                    "via nnInteractive on MorphoSource specimens.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # discover --------------------------------------------------------
    d = sub.add_parser("discover",
                       help="Find MorphoSource (CT, mesh) pairs and download them")
    d.add_argument("--query", default="skull mesh")
    d.add_argument("--max-pairs", type=int, default=10)
    d.add_argument("--max-candidates", type=int, default=80)
    d.add_argument("--require-taxonomy", default="")
    d.add_argument("--download-dir", default="",
                   help="Where to cache the MorphoSource downloads")
    d.add_argument("--output", required=True,
                   help="JSON specimen list to write")
    d.set_defaults(func=_cmd_discover)

    # prepare ---------------------------------------------------------
    pp = sub.add_parser(
        "prepare",
        help="Download a real MorphoSource (CT, GT mesh) pair, convert "
             "DICOM/TIFF stacks to NIfTI, optionally crop, voxelise the "
             "GT mesh, and write specimens.json for `seg_train round`.",
    )
    pp.add_argument("--preset", default="",
                    choices=[""] + sorted(PREPARE_PRESETS.keys()),
                    help="Pre-canned (CT, GT) pair (overrides individual IDs).")
    pp.add_argument("--ct-media-id", default="",
                    help="MorphoSource media ID of the CT volume.")
    pp.add_argument("--gt-media-id", default="",
                    help="MorphoSource media ID of the GT mesh.")
    pp.add_argument("--physical-object-id", default="",
                    help="Specimen ID (e.g. 'uf:herp:191369').")
    pp.add_argument("--taxonomy", default="")
    pp.add_argument("--morphosource-query", default="")
    pp.add_argument("--goal", default="",
                    help="Plain-English target. Optional here — required "
                         "later by `seg_train round`. Stored alongside the "
                         "preset's default for convenience.")
    pp.add_argument("--prep-dir", default="prepared",
                    help="Where the download + voxelisation artefacts land "
                         "(passed straight to nninteractive_compare.py).")
    pp.add_argument("--output", required=True,
                    help="Path of the specimens.json to emit.")
    pp.add_argument("--voxelize-backend", default="",
                    choices=["", "auto", "slicer", "vtk"],
                    help="GT-mesh voxeliser. Empty = use the preset default.")
    pp.add_argument("--crop-mm", type=float, default=0.0,
                    help="Margin (mm) for cropping the CT around the mesh "
                         "bbox; 0 = no crop.")
    pp.add_argument("--prep-timeout", type=int, default=1800,
                    help="Subprocess timeout (s) for the prep pipeline.")
    pp.add_argument("--system-python", action="store_true",
                    help="Use the parent Python instead of the nnInteractive "
                         "venv for the prep stage. Only useful when "
                         "SimpleITK + VTK are installed system-wide.")
    pp.set_defaults(func=_cmd_prepare)

    # round -----------------------------------------------------------
    r = sub.add_parser("round", help="Run one iterative round")
    r.add_argument("--specimens", required=True,
                   help="Path to specimens.json from `discover`")
    r.add_argument("--run-dir", required=True,
                   help="Directory under which round_NNN/ subfolders live")
    r.add_argument("--goal", required=True,
                   help="Plain-English target, e.g. \"Segment the cranial bone\"")
    r.add_argument("--paper-tag", default="")
    r.add_argument("--run-id", default="")
    r.add_argument("--student", default="",
                   help="Optional path to a *.artifact.json from a "
                        "previous round; absent => Round 0 nnInteractive only")
    r.add_argument("--max-paint-steps", type=int, default=12)
    r.add_argument("--student-epochs", type=int, default=30)
    r.add_argument("--n-dropout-samples", type=int, default=4)
    r.add_argument("--graduation-dice-threshold", type=float, default=0.85)
    r.add_argument("--student-min-reliability", type=float, default=0.7)
    r.add_argument("--router-min-confidence", type=float, default=0.85)
    r.add_argument("--router-max-entropy", type=float, default=0.35)
    r.add_argument("--router-require-uncertainty", action="store_true")
    r.add_argument("--router-name", default="default_v1")
    r.add_argument("--skip-training", action="store_true")
    r.add_argument("--seed", type=int, default=1234)
    r.set_defaults(func=_cmd_round)

    # export ----------------------------------------------------------
    e = sub.add_parser("export", help="Generate paper CSVs/plots from the ledger")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--output", default="")
    e.add_argument("--paper-tag", default="")
    e.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib plot generation (CSVs + Markdown only)")
    e.set_defaults(func=_cmd_export)

    # summary ---------------------------------------------------------
    s = sub.add_parser("summary",
                       help="Print quick aggregate stats of the ledger")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--markdown", default="",
                   help="Optional path to write a Markdown summary")
    s.set_defaults(func=_cmd_summary)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
