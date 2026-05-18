"""Tests for the publication-export stage.

We populate a synthetic ledger and verify:

- ``aggregate_by_mode`` and ``aggregate_by_round`` produce the expected
  rows (and tolerate missing GT rows).
- ``vs_human_table`` collapses rows back per-specimen.
- ``export_paper_artifacts`` writes the documented files (CSVs +
  Markdown summary).

Plot generation is best-effort (matplotlib may be missing in the test
env) so we don't assert PNG existence — only that the call doesn't
crash and the documented summary file exists.
"""

import csv
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from metadata_to_morphsource.seg_train import (  # noqa: E402
    EpisodeRecord, ExperimentLedger, SegmentationMode,
)
from metadata_to_morphsource.seg_train.paper_export import (  # noqa: E402
    aggregate_by_mode, aggregate_by_round, vs_human_table,
    export_paper_artifacts,
)




def _populate_ledger(tmp_path):
    led = ExperimentLedger(tmp_path / "ledger", run_id="rr",
                           paper_tag="t1")
    # Round 0: nnInteractive only.
    for media, dice, dur in [("A", 0.7, 90.0), ("B", 0.85, 60.0)]:
        led.record(EpisodeRecord(
            mode=SegmentationMode.NNINTERACTIVE.value,
            media_id=media, taxonomy="genus_x",
            round_index=0, dice=dice, has_ground_truth=True,
            n_prompts=4, duration_s=dur,
            routed_to_nninteractive=True,
        ))
    # Round 1: student attempts and corrections.
    led.record(EpisodeRecord(
        mode=SegmentationMode.STUDENT.value,
        media_id="A", taxonomy="genus_x",
        round_index=1, dice=0.78, has_ground_truth=True,
        duration_s=4.0,
    ))
    led.record(EpisodeRecord(
        mode=SegmentationMode.STUDENT_CORRECTED.value,
        media_id="B", taxonomy="genus_x",
        round_index=1, dice=0.89, has_ground_truth=True,
        n_prompts=2, duration_s=45.0,
        routed_to_nninteractive=True,
    ))
    return led


def test_aggregate_by_mode(tmp_path):
    led = _populate_ledger(tmp_path)
    rows = aggregate_by_mode(list(led))
    by_mode = {r["mode"]: r for r in rows}
    assert "nninteractive" in by_mode
    assert "student" in by_mode
    assert by_mode["nninteractive"]["n"] == 2
    assert by_mode["student"]["n"] == 1
    assert by_mode["nninteractive"]["mean_dice"] == pytest.approx(
        (0.7 + 0.85) / 2, rel=1e-3
    )


def test_aggregate_by_round(tmp_path):
    led = _populate_ledger(tmp_path)
    rows = aggregate_by_round(list(led))
    rounds = {r["round_index"]: r for r in rows}
    assert rounds[0]["n_episodes"] == 2
    assert rounds[1]["n_episodes"] == 2
    assert rounds[1]["n_student"] == 1
    assert rounds[1]["n_corrected"] == 1
    assert rounds[1]["mean_student_dice"] == pytest.approx(0.78)


def test_vs_human_table(tmp_path):
    led = _populate_ledger(tmp_path)
    rows = vs_human_table(list(led))
    by_specimen = {r["media_id"]: r for r in rows}
    assert "A" in by_specimen and "B" in by_specimen
    assert by_specimen["A"].get("nninteractive_dice") == pytest.approx(0.7)
    assert by_specimen["A"].get("student_dice") == pytest.approx(0.78)


def test_export_paper_artifacts_writes_files(tmp_path):
    """Exercise the end-to-end export path with plots disabled — keeps
    the test pure-Python so a corrupted matplotlib/numpy doesn't break
    CI on developer laptops."""
    led = _populate_ledger(tmp_path)
    info = export_paper_artifacts(
        led, output_dir=tmp_path / "paper", include_plots=False,
    )
    assert info["n_episodes"] == 4
    assert info["plots"] == {}

    out = tmp_path / "paper"
    for name in ("per_episode.csv", "mode_summary.csv",
                 "round_progression.csv", "vs_human.csv",
                 "paper_summary.md"):
        assert (out / name).exists(), f"missing {name}"

    # Per-episode CSV should have exactly the 4 rows we recorded.
    with (out / "per_episode.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4

    # Markdown summary mentions the paper tag.
    md = (out / "paper_summary.md").read_text()
    assert "Iterative Segmentation Experiment" in md
    assert "nninteractive" in md
