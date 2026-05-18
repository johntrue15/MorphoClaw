"""Unit tests for the ``seg_train prepare`` helper.

These are pure-Python tests — they don't touch the network, don't run
Slicer, and don't invoke the LLM. They verify:

- the ``chameleon_stapes`` preset has the canonical media IDs and goal,
- :func:`_find_paint_loop_inputs` correctly walks a fake ``pair_dir``
  layout that mirrors what ``nninteractive_compare.py --skip-paint-loop``
  produces,
- :func:`write_specimens_json` round-trips through ``SpecimenInput`` so
  ``seg_train round --specimens`` can consume the prepared list,
- ``prepare_specimen`` raises a ``RuntimeError`` (rather than crashing)
  when the underlying compare-script subprocess fails.

The full live-data flow is exercised by
``Tests/test_chameleon_stapes_iterative.sh``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


def test_chameleon_stapes_preset_canonical_ids():
    from metadata_to_morphsource.seg_train.prepare_specimen import PRESETS
    assert "chameleon_stapes" in PRESETS
    p = PRESETS["chameleon_stapes"]
    assert p["ct_media_id"] == "000408242"
    assert p["gt_media_id"] == "000790324"
    assert p["physical_object_id"] == "uf:herp:191369"
    assert "stapes" in p["goal"].lower()
    assert p["voxelize_backend"] in {"vtk", "slicer", "auto"}


def _seed_fake_pair_dir(root: Path, ct_id: str, gt_id: str) -> Path:
    """Mirror the layout written by nninteractive_compare.py
    --skip-paint-loop so _find_paint_loop_inputs has something to walk.
    """
    pair = root / f"{ct_id}__vs__{gt_id}"
    pair.mkdir(parents=True)
    (pair / f"ct_{ct_id}_cropped.nii.gz").write_bytes(b"FAKE_NIFTI")
    (pair / "gt_voxelized.nii.gz").write_bytes(b"FAKE_LABEL")
    (pair / "gt_voxelized.voxelize.json").write_text(json.dumps({
        "foreground_voxels": 1234,
        "foreground_volume_mm3": 78.9,
        "reference_dims": [64, 64, 64],
        "reference_spacing_xyz": [0.05, 0.05, 0.05],
    }))
    download = pair / "download" / f"gt_{gt_id}"
    download.mkdir(parents=True)
    mesh = download / f"{gt_id}.ply"
    mesh.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    return pair


def test_find_paint_loop_inputs_locates_artifacts(tmp_path):
    from metadata_to_morphsource.seg_train.prepare_specimen import (
        _find_paint_loop_inputs,
    )
    pair = _seed_fake_pair_dir(tmp_path, "000408242", "000790324")
    ct, gt_label, mesh = _find_paint_loop_inputs(pair)
    assert ct.name.endswith("_cropped.nii.gz")
    assert gt_label.name == "gt_voxelized.nii.gz"
    assert mesh.suffix.lower() == ".ply"
    for p in (ct, gt_label, mesh):
        assert p.exists(), f"{p} should exist"


def test_find_paint_loop_inputs_falls_back_when_no_crop(tmp_path):
    from metadata_to_morphsource.seg_train.prepare_specimen import (
        _find_paint_loop_inputs,
    )
    pair = tmp_path / "x__vs__y"
    pair.mkdir()
    (pair / "ct_x.nii.gz").write_bytes(b"NIFTI")
    (pair / "gt_voxelized.nii.gz").write_bytes(b"LABEL")
    download = pair / "download"
    download.mkdir()
    (download / "y.stl").write_bytes(b"STL")

    ct, gt, mesh = _find_paint_loop_inputs(pair)
    assert ct.name == "ct_x.nii.gz"
    assert mesh.suffix == ".stl"
    assert gt.name == "gt_voxelized.nii.gz"


def test_find_paint_loop_inputs_raises_when_label_missing(tmp_path):
    from metadata_to_morphsource.seg_train.prepare_specimen import (
        _find_paint_loop_inputs,
    )
    pair = tmp_path / "x__vs__y"
    pair.mkdir()
    (pair / "ct_x_cropped.nii.gz").write_bytes(b"NIFTI")
    with pytest.raises(FileNotFoundError):
        _find_paint_loop_inputs(pair)


def test_write_specimens_json_round_trips_through_specimen_input(tmp_path):
    from metadata_to_morphsource.seg_train.prepare_specimen import (
        PreparedSpecimen, write_specimens_json,
    )
    from metadata_to_morphsource.seg_train.iterative_trainer import (
        SpecimenInput,
    )
    ps = PreparedSpecimen(
        ct_media_id="000408242", gt_media_id="000790324",
        physical_object_id="uf:herp:191369",
        taxonomy="Chamaeleo calyptratus",
        morphosource_query="chameleon stapes",
        goal="Segment the right stapes",
        volume_path=str(tmp_path / "vol.nii.gz"),
        gt_mesh_path=str(tmp_path / "mesh.ply"),
        gt_label_path=str(tmp_path / "label.nii.gz"),
        pair_dir=str(tmp_path),
    )
    out = write_specimens_json([ps], tmp_path / "specimens.json")
    assert out.exists()

    data = json.loads(out.read_text())
    assert isinstance(data, list) and len(data) == 1
    sp = SpecimenInput(**data[0])
    assert sp.media_id == "000408242"
    assert sp.gt_media_id == "000790324"
    assert sp.physical_object_id == "uf:herp:191369"
    assert sp.gt_mesh_path.endswith("mesh.ply")
    assert sp.gt_label_path.endswith("label.nii.gz")
    assert sp.has_gt is True


def test_prepare_specimen_surfaces_subprocess_failure(tmp_path, monkeypatch):
    """If nninteractive_compare.py exits non-zero, prepare_specimen
    should raise RuntimeError with the captured stderr — never silently
    return an empty PreparedSpecimen.
    """
    import metadata_to_morphsource.seg_train.prepare_specimen as ps_mod

    class _FakeCompletedProc:
        def __init__(self, rc, out, err):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    monkeypatch.setattr(ps_mod.subprocess, "run",
                        lambda *a, **kw: _FakeCompletedProc(
                            2, "", "morphosource: download forbidden"
                        ))
    # Make COMPARE_SCRIPT pretend to exist so we hit the subprocess hop.
    monkeypatch.setattr(ps_mod, "COMPARE_SCRIPT", tmp_path / "fake.py")
    (tmp_path / "fake.py").write_text("# placeholder")

    with pytest.raises(RuntimeError) as exc:
        ps_mod.prepare_specimen(
            ct_media_id="000408242",
            gt_media_id="000790324",
            output_dir=tmp_path,
            voxelize_backend="vtk",
            use_nninteractive_python=False,
        )
    assert "exit 2" in str(exc.value)
    assert "morphosource" in str(exc.value)


def test_prepare_preset_unknown_name_raises(tmp_path):
    from metadata_to_morphsource.seg_train.prepare_specimen import (
        prepare_preset,
    )
    with pytest.raises(KeyError):
        prepare_preset("not_a_preset", output_dir=tmp_path)
