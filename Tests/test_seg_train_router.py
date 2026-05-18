"""Tests for the confidence-based segmentation router.

The router is a small, deterministic policy gate. Tests cover:

- accept-when-confident,
- correct-on-low-confidence,
- correct-on-high-entropy,
- reject-on-empty / runaway prediction,
- ``binary_entropy`` numerical sanity (with and without numpy),
- ``summarise_probability_volume`` end-to-end shape.

The numpy-dependent tests are guarded with ``@pytest.mark.numpy`` so
hosts with a corrupted numpy install can simply
``pytest -m "not numpy"`` to skip them. CI runs without that marker
filter so the full suite still gets exercised.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from metadata_to_morphsource.seg_train import (  # noqa: E402
    ConfidenceRouter, RouterDecision, RouterPolicy,
)
from metadata_to_morphsource.seg_train.confidence_router import (  # noqa: E402
    StudentPrediction, binary_entropy, summarise_probability_volume,
)


def _pred(**kwargs) -> StudentPrediction:
    base = dict(voxel_count=1_000, total_voxels=10_000_000,
                mean_confidence=0.9, mean_entropy=0.1)
    base.update(kwargs)
    return StudentPrediction(**base)


def test_accept_when_confident():
    router = ConfidenceRouter(RouterPolicy())
    res = router.decide(_pred())
    assert res.decision == RouterDecision.ACCEPT_STUDENT
    assert res.routes_to_nninteractive is False


def test_correct_on_low_confidence():
    router = ConfidenceRouter(RouterPolicy(min_confidence=0.85))
    res = router.decide(_pred(mean_confidence=0.5))
    assert res.decision == RouterDecision.CORRECT_WITH_NNINTERACTIVE
    assert res.routes_to_nninteractive is True
    assert "0.500" in res.reason


def test_correct_on_high_entropy():
    router = ConfidenceRouter(RouterPolicy(max_entropy=0.3))
    res = router.decide(_pred(mean_entropy=0.55))
    assert res.decision == RouterDecision.CORRECT_WITH_NNINTERACTIVE
    assert "entropy" in res.reason


def test_reject_on_empty():
    router = ConfidenceRouter(RouterPolicy())
    res = router.decide(_pred(voxel_count=0))
    assert res.decision == RouterDecision.REJECT
    assert res.diagnostics["foreground_fraction"] == 0.0


def test_reject_on_runaway():
    router = ConfidenceRouter(RouterPolicy(max_foreground_fraction=0.5))
    res = router.decide(_pred(voxel_count=8_000_000,
                              total_voxels=10_000_000))
    assert res.decision == RouterDecision.REJECT
    assert res.routes_to_nninteractive is True


def test_require_uncertainty_falls_back():
    router = ConfidenceRouter(RouterPolicy(require_uncertainty=True))
    res = router.decide(_pred(mean_confidence=None, mean_entropy=None))
    assert res.decision == RouterDecision.CORRECT_WITH_NNINTERACTIVE


def test_no_uncertainty_permitted():
    router = ConfidenceRouter(RouterPolicy(require_uncertainty=False))
    res = router.decide(_pred(mean_confidence=None, mean_entropy=None))
    assert res.decision == RouterDecision.ACCEPT_STUDENT


# ---------------------------------------------------------------------------


@pytest.mark.numpy
def test_binary_entropy_known_values():
    pytest.importorskip("numpy")
    # H(0.5) = ln 2 ≈ 0.693
    assert binary_entropy([0.5, 0.5]) == pytest.approx(0.693, abs=1e-2)
    # H at 0 / 1 → 0
    assert binary_entropy([0.0, 1.0]) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.numpy
def test_summarise_probability_volume():
    np = pytest.importorskip("numpy")
    arr = np.zeros((4, 4, 4), dtype=float)
    arr[1:3, 1:3, 1:3] = 0.95
    summary = summarise_probability_volume(arr, threshold=0.5)
    assert summary.voxel_count == 8
    assert summary.total_voxels == 64
    assert summary.foreground_fraction == pytest.approx(8 / 64)
    assert summary.mean_confidence == pytest.approx(0.95, abs=1e-3)
    assert summary.bounding_box_xyz == [[1, 2], [1, 2], [1, 2]]
