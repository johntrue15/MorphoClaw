"""Tests for the iterative-training experiment ledger.

The ledger is the data backbone of the planned publication, so we want
strict guarantees:

- Schema-stable CSV output (same columns every time).
- Append-only JSONL with idempotent CSV header.
- ``EpisodeRecord.finalize`` derives ``volume_difference_pct`` correctly.
- ``ExperimentLedger.summary`` aggregates by mode and round.
- ``export_summary_markdown`` renders even on an empty-ish ledger.
"""

import csv
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from metadata_to_morphsource.seg_train import (  # noqa: E402
    EpisodeRecord, ExperimentLedger, SegmentationMode,
)
from metadata_to_morphsource.seg_train.experiment_ledger import (  # noqa: E402
    CSV_COLUMNS, SCHEMA_VERSION,
)


def test_schema_version_constant():
    """Schema version must look like a semver string."""
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_csv_columns_match_record_fields():
    """Every CSV column must correspond to a serialisable record field."""
    record = EpisodeRecord().to_csv_row()
    missing = [c for c in CSV_COLUMNS if c not in record]
    assert not missing, f"CSV columns missing from record: {missing}"


def test_record_finalize_computes_volume_difference():
    rec = EpisodeRecord(
        volume_pred_mm3=120.0,
        volume_gt_mm3=100.0,
    )
    rec.finalize()
    assert rec.volume_difference_pct == pytest.approx(20.0, rel=1e-3)


def test_record_finalize_handles_zero_gt():
    rec = EpisodeRecord(volume_pred_mm3=10.0, volume_gt_mm3=0.0)
    rec.finalize()
    # Division by zero is avoided; field stays None.
    assert rec.volume_difference_pct is None


def test_ledger_records_jsonl_and_csv(tmp_path):
    ledger = ExperimentLedger(tmp_path, run_id="abc123",
                              paper_tag="skull_v1")
    ledger.record(EpisodeRecord(
        media_id="000656244",
        physical_object_id="ms:1",
        mode=SegmentationMode.NNINTERACTIVE.value,
        round_index=0,
        dice=0.8,
        has_ground_truth=True,
    ))
    ledger.record(EpisodeRecord(
        media_id="000656244",
        physical_object_id="ms:1",
        mode=SegmentationMode.STUDENT.value,
        round_index=1,
        dice=0.9,
        has_ground_truth=True,
    ))

    jsonl_lines = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert len(jsonl_lines) == 2
    assert jsonl_lines[0]["media_id"] == "000656244"
    # run_id and paper_tag are inherited if not set on the record.
    assert all(line["run_id"] == "abc123" for line in jsonl_lines)
    assert all(line["paper_tag"] == "skull_v1" for line in jsonl_lines)

    with (tmp_path / "ledger.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["mode"] == "nninteractive"
    assert float(rows[1]["dice"]) == pytest.approx(0.9)


def test_ledger_summary_aggregates_by_mode(tmp_path):
    ledger = ExperimentLedger(tmp_path, run_id="r")
    for dice in (0.7, 0.8, 0.9):
        ledger.record(EpisodeRecord(
            mode=SegmentationMode.NNINTERACTIVE.value,
            dice=dice,
            round_index=0,
        ))
    ledger.record(EpisodeRecord(
        mode=SegmentationMode.STUDENT.value, dice=0.5, round_index=1,
    ))

    summary = ledger.summary()
    assert summary["n_episodes"] == 4
    assert summary["modes"]["nninteractive"] == 3
    assert summary["modes"]["student"] == 1
    assert summary["mean_dice_by_mode"]["nninteractive"] == pytest.approx(0.8)
    assert summary["mean_dice_by_mode"]["student"] == pytest.approx(0.5)
    assert summary["rounds"][0] == 3
    assert summary["rounds"][1] == 1


def test_ledger_filtering_iterators(tmp_path):
    ledger = ExperimentLedger(tmp_path, run_id="r")
    ledger.record(EpisodeRecord(
        mode="nninteractive", round_index=0, paper_tag="A",
    ))
    ledger.record(EpisodeRecord(
        mode="student", round_index=1, paper_tag="A",
    ))
    ledger.record(EpisodeRecord(
        mode="student", round_index=1, paper_tag="B",
    ))

    a_eps = ledger.episodes(paper_tag="A")
    assert len(a_eps) == 2
    assert all(ep["paper_tag"] == "A" for ep in a_eps)

    student_a = ledger.episodes(paper_tag="A", mode="student")
    assert len(student_a) == 1


def test_export_summary_markdown(tmp_path):
    ledger = ExperimentLedger(tmp_path, run_id="r", paper_tag="paper-1")
    ledger.record(EpisodeRecord(
        mode=SegmentationMode.NNINTERACTIVE.value,
        round_index=0, dice=0.7,
    ))
    md_path = ledger.export_summary_markdown(tmp_path / "summary.md")
    text = open(md_path).read()
    assert "paper-1" in text
    assert "`nninteractive`" in text
    assert "0.7000" in text
