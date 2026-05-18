"""Tests for the iterative-training dataset manifest.

These cover deterministic split assignment, "human GT always to
holdout" rule, reliability-weighted querying, and round-trip JSON
serialisation.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from metadata_to_morphsource.seg_train import (  # noqa: E402
    DatasetManifest, SegmentationSample, SplitName,
)
from metadata_to_morphsource.seg_train.dataset import (  # noqa: E402
    DEFAULT_RELIABILITY, reliability_for,
)


def _make_sample(media_id: str, mode: str = "nninteractive",
                 phys_obj: str = "", *, exists: bool = True,
                 tmp_path=None) -> SegmentationSample:
    if exists and tmp_path is not None:
        vol = tmp_path / f"{media_id}_vol.nii.gz"
        lab = tmp_path / f"{media_id}_lab.nii.gz"
        vol.write_bytes(b"fake")
        lab.write_bytes(b"fake")
        volume = str(vol)
        label = str(lab)
    else:
        volume = f"/nonexistent/{media_id}_vol.nii.gz"
        label = f"/nonexistent/{media_id}_lab.nii.gz"

    return SegmentationSample(
        sample_id=f"{media_id}__{mode}",
        media_id=media_id,
        physical_object_id=phys_obj or f"obj_{media_id}",
        volume_path=volume,
        label_path=label,
        mode=mode,
    )


def test_reliability_table():
    assert reliability_for("human_gt") == 1.0
    assert reliability_for("nninteractive") < 1.0
    assert reliability_for("student") < reliability_for("nninteractive")
    assert reliability_for("unknown_mode") == 0.5


def test_reliability_overrides():
    overrides = {"student": 0.99}
    assert reliability_for("student", overrides) == 0.99


def test_human_gt_always_holdout(tmp_path):
    m = DatasetManifest(root=str(tmp_path), seed=7)
    s = m.add_sample(_make_sample("AAA", mode="human_gt", tmp_path=tmp_path))
    assert s.sample_id in m.splits[SplitName.HOLDOUT.value]
    assert s.reliability == 1.0


def test_split_assignment_keyed_on_specimen(tmp_path):
    m = DatasetManifest(root=str(tmp_path), seed=42)
    # Two media of the same physical specimen → same split.
    a = m.add_sample(_make_sample("A", mode="nninteractive",
                                  phys_obj="ms:1", tmp_path=tmp_path))
    b = m.add_sample(_make_sample("B", mode="nninteractive",
                                  phys_obj="ms:1", tmp_path=tmp_path))
    split_a = next(s for s, ids in m.splits.items() if a.sample_id in ids)
    split_b = next(s for s, ids in m.splits.items() if b.sample_id in ids)
    assert split_a == split_b


def test_split_distribution_balanced(tmp_path):
    m = DatasetManifest(root=str(tmp_path), seed=1234,
                        train_ratio=0.7, val_ratio=0.2,
                        holdout_ratio=0.1)
    for i in range(60):
        m.add_sample(_make_sample(
            f"M{i:03d}", mode="nninteractive",
            phys_obj=f"ms:{i:03d}", tmp_path=tmp_path,
        ))
    sizes = m.summary()["split_sizes"]
    # With 60 samples and 70/20/10, expect approximately 42/12/6 — allow ±15
    # leeway because the SHA-based hash isn't perfectly uniform on small N.
    assert 30 <= sizes["train"] <= 55
    assert 5 <= sizes["val"] <= 25
    assert sizes["train"] + sizes["val"] + sizes["holdout"] == 60


def test_min_reliability_filter(tmp_path):
    m = DatasetManifest(root=str(tmp_path), seed=99)
    m.add_sample(_make_sample("A", mode="student", tmp_path=tmp_path))
    m.add_sample(_make_sample("B", mode="nninteractive",
                              tmp_path=tmp_path))
    train = m.split_samples(SplitName.TRAIN.value, min_reliability=0.7)
    assert all(s.reliability >= 0.7 for s in train)


def test_existing_files_only(tmp_path):
    m = DatasetManifest(root=str(tmp_path), seed=99)
    real = m.add_sample(_make_sample("A", mode="nninteractive",
                                     tmp_path=tmp_path))
    ghost = m.add_sample(_make_sample(
        "B", mode="nninteractive", phys_obj="ms:other",
        exists=False,
    ))
    train = m.split_samples(SplitName.TRAIN.value)
    train_ids = {s.sample_id for s in train}
    if real.sample_id in m.splits[SplitName.TRAIN.value]:
        assert real.sample_id in train_ids
    assert ghost.sample_id not in train_ids


def test_round_trip_serialisation(tmp_path):
    m = DatasetManifest(root=str(tmp_path), seed=42)
    for i in range(5):
        m.add_sample(_make_sample(
            f"M{i}", mode="nninteractive", phys_obj=f"ms:{i}",
            tmp_path=tmp_path,
        ))
    out = m.save(str(tmp_path / "manifest.json"))
    loaded = DatasetManifest.load(out)
    assert len(loaded.samples) == 5
    assert loaded.seed == 42
    assert sum(len(v) for v in loaded.splits.values()) == 5


def test_upsert_replaces_in_place(tmp_path):
    m = DatasetManifest(root=str(tmp_path), seed=7)
    s = m.add_sample(_make_sample("M0", mode="nninteractive",
                                  tmp_path=tmp_path))
    s.notes = "updated"
    m.add_sample(s)
    assert len(m.samples) == 1
    assert m.samples[0].notes == "updated"
