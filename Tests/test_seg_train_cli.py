"""Smoke tests for the seg_train CLI subcommand surface.

We don't exercise the full discover/round/export pipelines (they need
nnInteractive + Slicer + the network) — only that the parser is wired
correctly and the export / summary subcommands work against a
pre-populated ledger.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from metadata_to_morphsource.seg_train import (  # noqa: E402
    EpisodeRecord, ExperimentLedger, SegmentationMode,
)
from metadata_to_morphsource.seg_train.cli import build_parser, main  # noqa: E402


def test_parser_has_subcommands():
    parser = build_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "command" in actions
    sub = actions["command"]
    names = set()
    for action in sub.choices.values():  # type: ignore[attr-defined]
        names.add(action.prog.split()[-1])
    assert {"discover", "round", "export", "summary"}.issubset(names)


def test_export_subcommand_round_trip(tmp_path):
    """Exercise `seg_train export` with --no-plots so the test is pure
    Python (no matplotlib in the path)."""
    run_dir = tmp_path / "run"
    led = ExperimentLedger(run_dir / "ledger", run_id="r",
                           paper_tag="paper-1")
    led.record(EpisodeRecord(
        mode=SegmentationMode.NNINTERACTIVE.value,
        media_id="A", round_index=0, dice=0.8,
        has_ground_truth=True,
    ))

    rc = main(["export", "--run-dir", str(run_dir),
               "--output", str(run_dir / "paper"),
               "--no-plots"])
    assert rc == 0
    assert (run_dir / "paper" / "paper_summary.md").exists()


def test_summary_subcommand(tmp_path, capsys):
    run_dir = tmp_path / "run"
    led = ExperimentLedger(run_dir / "ledger", run_id="r")
    led.record(EpisodeRecord(
        mode=SegmentationMode.STUDENT.value, dice=0.6,
        round_index=2,
    ))
    rc = main(["summary", "--run-dir", str(run_dir),
               "--markdown", str(run_dir / "summary.md")])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["n_episodes"] == 1
    assert (run_dir / "summary.md").exists()


def test_round_requires_specimens_argument():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["round", "--run-dir", "x"])
