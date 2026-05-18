"""End-to-end orchestration test for the iterative trainer.

This is the "wiring test" — it builds a tiny synthetic dataset (one
32^3 NIfTI volume + a fake mesh-derived labelmap), monkeypatches the
two heavy subprocess hops (nnInteractive paint loop and the Slicer
voxelizer) to return canned labelmaps, then runs

    IterativeTrainer.run_round(...)

with student training disabled. We verify that:

- the ledger receives one ``human_gt`` row and one ``nninteractive``
  row,
- both episodes have Dice / IoU computed against the GT (because the
  prediction labelmap is identical to GT in this test),
- the dataset manifest now contains a holdout (GT) sample and a train
  sample (the pseudo-label),
- ``export_paper_artifacts`` writes the documented files.

The test skips on hosts where SimpleITK or numpy aren't healthy
(common on broken local Anaconda installs); CI runs all tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


pytestmark = pytest.mark.numpy  # the whole module needs numpy + SimpleITK


def _make_volume(path: Path, shape=(32, 32, 32), seed: int = 0):
    import numpy as np
    import SimpleITK as sitk
    rng = np.random.default_rng(seed)
    arr = rng.integers(-200, 200, size=shape, dtype=np.int16)
    arr[8:24, 8:24, 8:24] = 800  # a "structure" in the middle
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.5))
    img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(img, str(path), useCompression=True)


def _make_label(path: Path, shape=(32, 32, 32)):
    import numpy as np
    import SimpleITK as sitk
    arr = np.zeros(shape, dtype=np.uint8)
    arr[8:24, 8:24, 8:24] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.5))
    img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(img, str(path), useCompression=True)


def test_round_with_mocked_pseudo_labels(tmp_path, monkeypatch):
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("numpy")

    from metadata_to_morphsource.seg_train import (
        DatasetManifest, ExperimentLedger,
    )
    from metadata_to_morphsource.seg_train.iterative_trainer import (
        IterativeTrainer, RoundConfig, SpecimenInput,
    )
    from metadata_to_morphsource.seg_train.pseudo_label_generator import (
        PseudoLabelResult,
    )
    import metadata_to_morphsource.seg_train.iterative_trainer as it_mod

    # ---- 1. Build a synthetic specimen on disk ----
    specimen_dir = tmp_path / "specimen"
    specimen_dir.mkdir()
    volume_path = specimen_dir / "volume.nii.gz"
    gt_label_path = specimen_dir / "gt_label.nii.gz"
    _make_volume(volume_path)
    _make_label(gt_label_path)

    # ---- 2. Patch the two heavy subprocess hops ----
    voxelize_target_holder: dict = {}

    def fake_voxelize(*, reference_volume, mesh_path, output_path,
                     fill_value=1, timeout_s=900):
        # Copy our synthetic GT to the requested output path.
        sitk.WriteImage(
            sitk.ReadImage(str(gt_label_path)),
            str(output_path), useCompression=True,
        )
        voxelize_target_holder["called"] = True
        return PseudoLabelResult(
            success=True, label_path=str(output_path),
            method="voxelize_mesh", model_version="curated_mesh_v1",
            duration_s=0.01,
            extra={
                "reference_dims": [32, 32, 32],
                "reference_spacing_xyz": [0.5, 0.5, 0.5],
                "foreground_voxels": 4096,
                "foreground_volume_mm3": 512.0,
            },
        )

    paint_target_holder: dict = {}

    def fake_paint_loop(*, volume_path, goal, output_dir, media_id,
                        max_steps=12, vision_model="", timeout_s=3600):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        label_path = out / f"{media_id}_nni_labelmap.nii.gz"
        # Fake "perfect" prediction = the GT.
        sitk.WriteImage(
            sitk.ReadImage(str(gt_label_path)),
            str(label_path), useCompression=True,
        )
        paint_target_holder["called"] = True
        return PseudoLabelResult(
            success=True, label_path=str(label_path),
            method="nninteractive", model_version="nnInteractive_v1.0",
            duration_s=1.23, n_prompts=3,
            prompt_kinds=["bbox", "point", "point"],
        )

    # The trainer imports these by name from the module.
    monkeypatch.setattr(it_mod, "voxelize_curated_mesh", fake_voxelize)
    monkeypatch.setattr(it_mod, "run_paint_loop", fake_paint_loop)

    # ---- 3. Wire up the trainer ----
    run_dir = tmp_path / "run"
    ledger = ExperimentLedger(run_dir / "ledger", run_id="smoke",
                              paper_tag="integration_test")
    manifest = DatasetManifest(root=str(run_dir / "manifest_root"))

    cfg = RoundConfig(
        goal="Segment the synthetic structure",
        output_root=str(run_dir),
        paper_tag="integration_test",
        train_after_round=False,
        max_paint_steps=2,
    )
    trainer = IterativeTrainer(
        ledger=ledger, manifest=manifest, round_config=cfg,
    )

    specimen = SpecimenInput(
        media_id="SYNTH_001",
        volume_path=str(volume_path),
        gt_mesh_path="/fake/mesh.ply",  # presence triggers voxelize path
        gt_media_id="SYNTH_001_GT",
        physical_object_id="ms:synth",
        taxonomy="genus_synth",
        morphosource_query="synthetic test",
    )

    report = trainer.run_round([specimen], skip_training=True)

    # ---- 4. Assertions ----
    assert voxelize_target_holder.get("called"), "voxelize_curated_mesh was not invoked"
    assert paint_target_holder.get("called"), "run_paint_loop was not invoked"

    # Two episodes recorded: human_gt + nninteractive
    episodes = list(ledger)
    modes = sorted(ep["mode"] for ep in episodes)
    assert "human_gt" in modes
    assert "nninteractive" in modes
    assert len(episodes) >= 2

    # Both prediction-vs-GT episodes should now carry Dice ~1.0 because
    # our fake paint loop returned the GT itself.
    dices = [ep["dice"] for ep in episodes if ep.get("dice") is not None]
    assert dices, "no metrics computed against the GT"
    assert max(dices) > 0.99

    # Manifest holds at least the GT (holdout) + the pseudo-label sample
    summary = manifest.summary()
    assert summary["total"] >= 2
    assert summary["split_sizes"]["holdout"] >= 1

    # Round report sanity
    assert report.round_index == 0
    assert report.n_specimens == 1
    assert any(ep.media_id == "SYNTH_001" for ep in report.episodes)


def test_paper_export_after_synthetic_round(tmp_path, monkeypatch):
    """After the integration round, the paper-export pipeline should
    produce all documented files using the captured ledger."""
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("numpy")

    # Reuse the round above by inlining a minimal version.
    from metadata_to_morphsource.seg_train import (
        ExperimentLedger, EpisodeRecord, SegmentationMode,
    )
    from metadata_to_morphsource.seg_train.paper_export import (
        export_paper_artifacts,
    )

    led = ExperimentLedger(tmp_path / "ledger", run_id="r",
                           paper_tag="integration_test")
    led.record(EpisodeRecord(
        mode=SegmentationMode.HUMAN_GT.value, media_id="A",
        round_index=0, has_ground_truth=True,
    ))
    led.record(EpisodeRecord(
        mode=SegmentationMode.NNINTERACTIVE.value, media_id="A",
        round_index=0, dice=0.99, iou=0.98,
        has_ground_truth=True, n_prompts=3,
    ))

    info = export_paper_artifacts(
        led, output_dir=tmp_path / "paper", include_plots=False,
    )
    assert info["n_episodes"] == 2
    paper = tmp_path / "paper"
    for name in (
        "per_episode.csv", "mode_summary.csv", "round_progression.csv",
        "vs_human.csv", "paper_summary.md",
    ):
        assert (paper / name).exists(), f"missing {name}"
