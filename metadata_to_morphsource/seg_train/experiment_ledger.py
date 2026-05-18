"""Append-only segmentation experiment ledger.

Every segmentation event in the iterative training pipeline (whether
performed by nnInteractive, the student model, or a hybrid pass) is
recorded as one :class:`EpisodeRecord` and appended to two files:

- ``ledger.jsonl``  — line-delimited JSON, the canonical record.
- ``ledger.csv``    — flat CSV with the most-common columns, convenient
                       for quick inspection in pandas / a spreadsheet.

The ledger is the data backbone of the publication. Each row answers
the questions a reviewer will ask:

- Who produced this segmentation? (mode, model_version, prompts_used)
- How long did it take, and on what device?
- How does it compare to the curated GT, if any?
- How does it compare to the *previous* student version?
- What is the provenance of the input data (media_id, query, taxonomy)?

The schema is intentionally flat so it can be loaded into pandas with
``pd.read_csv("ledger.csv")``. Nested values (prompts, metric details)
also live in the JSONL so they can be reconstructed when needed.

Concurrency
-----------
Writes to ``ledger.jsonl`` are guarded by ``fcntl.flock`` so multiple
worker processes (or workflow steps) can append safely on POSIX. The
CSV write rebuilds the row from the JSONL append.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import json
import os
import platform
import socket
import sys
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


SCHEMA_VERSION = "1.0.0"


class SegmentationMode(str, Enum):
    """Where this segmentation came from."""

    HUMAN_GT = "human_gt"  # curated MorphoSource mesh, voxelised
    NNINTERACTIVE = "nninteractive"  # nnInteractive paint loop alone
    MONAI_AUTO = "monai_auto"  # MONAI Auto3DSeg (pretrained)
    STUDENT = "student"  # our trained student, fully autonomous
    STUDENT_CORRECTED = "student_corrected"  # student + nnInteractive correction
    ENSEMBLE = "ensemble"  # multiple models combined


@dataclass
class EpisodeRecord:
    """One segmentation event.

    All fields default to safe blanks so callers can populate
    progressively as the pipeline runs (e.g. fill in metrics later).
    Use :meth:`finalize` before appending to compute derived fields.
    """

    # Identity ------------------------------------------------------------
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp_utc: str = field(default_factory=lambda: _dt.datetime.now(
        _dt.timezone.utc).isoformat(timespec="seconds"))
    schema_version: str = SCHEMA_VERSION

    # Run / experiment grouping ------------------------------------------
    run_id: str = ""
    round_index: int = 0
    paper_tag: str = ""  # free-form: which experiment this contributes to

    # Specimen ------------------------------------------------------------
    media_id: str = ""
    physical_object_id: str = ""
    taxonomy: str = ""
    morphosource_query: str = ""
    volume_path: str = ""
    volume_shape_zyx: tuple = ()  # type: ignore[assignment]
    volume_spacing_xyz_mm: tuple = ()  # type: ignore[assignment]

    # Segmentation production --------------------------------------------
    mode: str = SegmentationMode.NNINTERACTIVE.value
    goal: str = ""
    prediction_path: str = ""
    n_prompts: int = 0
    prompt_kinds: list = field(default_factory=list)
    operator: str = ""  # "llm:gpt-4o", "human:initials", "student:v3", ...
    model_version: str = ""
    device: str = ""
    duration_s: float = 0.0

    # Quantitative comparison vs ground truth (optional) -----------------
    has_ground_truth: bool = False
    ground_truth_path: str = ""
    ground_truth_source: str = ""  # "morphosource_mesh", "expert_annotation", ...
    dice: Optional[float] = None
    iou: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    hausdorff_mm: Optional[float] = None
    hausdorff_95_mm: Optional[float] = None
    average_surface_dist_mm: Optional[float] = None
    centroid_distance_mm: Optional[float] = None
    volume_pred_mm3: Optional[float] = None
    volume_gt_mm3: Optional[float] = None
    volume_difference_pct: Optional[float] = None

    # Self-supervised quality signals -----------------------------------
    confidence_score: Optional[float] = None  # mean foreground softmax
    entropy_score: Optional[float] = None     # mean per-voxel entropy
    voxel_count: int = 0

    # Comparison vs previous student version (no GT needed) -------------
    previous_model_version: str = ""
    dice_vs_previous: Optional[float] = None

    # Routing diagnostics ------------------------------------------------
    router_policy: str = ""
    router_threshold: Optional[float] = None
    routed_to_nninteractive: bool = False

    # Reproducibility ----------------------------------------------------
    host: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=lambda:
                          f"{platform.system()} {platform.release()} "
                          f"{platform.machine()}")
    python_version: str = field(default_factory=lambda:
                                sys.version.split()[0])
    git_sha: str = field(default_factory=lambda: os.environ.get(
        "GITHUB_SHA", "") or os.environ.get("GIT_SHA", ""))

    # Free-form payload (kept in JSONL but flattened away from CSV) -----
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def finalize(self) -> "EpisodeRecord":
        """Compute derived fields and clamp tuples for serialisation."""
        if self.volume_pred_mm3 is not None and self.volume_gt_mm3 not in (
            None, 0.0
        ) and self.volume_difference_pct is None:
            self.volume_difference_pct = round(
                abs(self.volume_pred_mm3 - self.volume_gt_mm3)
                / self.volume_gt_mm3 * 100.0, 4
            )
        self.volume_shape_zyx = tuple(self.volume_shape_zyx) if self.volume_shape_zyx else ()
        self.volume_spacing_xyz_mm = (
            tuple(round(float(s), 6) for s in self.volume_spacing_xyz_mm)
            if self.volume_spacing_xyz_mm else ()
        )
        self.prompt_kinds = list(self.prompt_kinds)
        return self

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # tuples -> lists for JSON
        d["volume_shape_zyx"] = list(self.volume_shape_zyx)
        d["volume_spacing_xyz_mm"] = list(self.volume_spacing_xyz_mm)
        return d

    def to_csv_row(self) -> dict:
        """Flat row used by ``ledger.csv`` (drops nested fields)."""
        d = self.to_dict()
        d["volume_shape_zyx"] = ",".join(str(x) for x in self.volume_shape_zyx)
        d["volume_spacing_xyz_mm"] = ",".join(
            f"{x:.6g}" for x in self.volume_spacing_xyz_mm
        )
        d["prompt_kinds"] = ",".join(self.prompt_kinds)
        d.pop("extra", None)
        return d


# ---------------------------------------------------------------------------
# Ledger writer
# ---------------------------------------------------------------------------


CSV_COLUMNS: tuple = (
    "episode_id",
    "timestamp_utc",
    "schema_version",
    "run_id",
    "round_index",
    "paper_tag",
    "media_id",
    "physical_object_id",
    "taxonomy",
    "morphosource_query",
    "volume_path",
    "volume_shape_zyx",
    "volume_spacing_xyz_mm",
    "mode",
    "goal",
    "prediction_path",
    "n_prompts",
    "prompt_kinds",
    "operator",
    "model_version",
    "device",
    "duration_s",
    "has_ground_truth",
    "ground_truth_path",
    "ground_truth_source",
    "dice",
    "iou",
    "precision",
    "recall",
    "hausdorff_mm",
    "hausdorff_95_mm",
    "average_surface_dist_mm",
    "centroid_distance_mm",
    "volume_pred_mm3",
    "volume_gt_mm3",
    "volume_difference_pct",
    "confidence_score",
    "entropy_score",
    "voxel_count",
    "previous_model_version",
    "dice_vs_previous",
    "router_policy",
    "router_threshold",
    "routed_to_nninteractive",
    "host",
    "platform",
    "python_version",
    "git_sha",
)


class ExperimentLedger:
    """Append-only segmentation event log.

    Parameters
    ----------
    root:
        Directory that holds ``ledger.jsonl`` and ``ledger.csv``. Created
        if it does not exist.
    run_id:
        Identifier shared by all episodes from a single training run. If
        not supplied, a fresh UUID is generated.
    paper_tag:
        Optional free-form tag that groups runs that contribute to the
        same paper experiment (e.g. ``"chameleon_skull_v1"``).
    """

    JSONL_NAME = "ledger.jsonl"
    CSV_NAME = "ledger.csv"

    def __init__(
        self,
        root: os.PathLike | str,
        *,
        run_id: str = "",
        paper_tag: str = "",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / self.JSONL_NAME
        self.csv_path = self.root / self.CSV_NAME
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.paper_tag = paper_tag
        self._lock = threading.Lock()
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    # ------------------------------------------------------------------

    def record(self, episode: EpisodeRecord) -> EpisodeRecord:
        """Finalise *episode*, write to disk, and return it."""
        if not episode.run_id:
            episode.run_id = self.run_id
        if not episode.paper_tag:
            episode.paper_tag = self.paper_tag
        episode.finalize()
        with self._lock:
            self._append_jsonl(episode)
            self._append_csv(episode)
        return episode

    # ------------------------------------------------------------------

    def _append_jsonl(self, episode: EpisodeRecord) -> None:
        text = json.dumps(episode.to_dict(), default=str) + "\n"
        # POSIX advisory lock: tolerate non-fcntl platforms by best-effort
        # write — an append on a single file is atomic for short writes
        # on most Unixes anyway.
        try:
            import fcntl  # type: ignore
            with self.jsonl_path.open("a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(text)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except ImportError:
            with self.jsonl_path.open("a") as f:
                f.write(text)

    def _append_csv(self, episode: EpisodeRecord) -> None:
        row = episode.to_csv_row()
        # Ensure schema-stable column set
        row = {k: row.get(k, "") for k in CSV_COLUMNS}
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        if not self.jsonl_path.exists():
            return iter(())
        with self.jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def episodes(
        self,
        *,
        mode: Optional[str] = None,
        run_id: Optional[str] = None,
        paper_tag: Optional[str] = None,
        round_index: Optional[int] = None,
    ) -> list[dict]:
        """Filtered view of recorded episodes."""

        out: list[dict] = []
        for ep in self:
            if mode is not None and ep.get("mode") != mode:
                continue
            if run_id is not None and ep.get("run_id") != run_id:
                continue
            if paper_tag is not None and ep.get("paper_tag") != paper_tag:
                continue
            if round_index is not None and ep.get("round_index") != round_index:
                continue
            out.append(ep)
        return out

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Aggregate stats useful for at-a-glance reporting."""
        episodes = list(self)
        if not episodes:
            return {"n_episodes": 0}
        modes: dict[str, int] = {}
        dice_by_mode: dict[str, list[float]] = {}
        rounds: dict[int, int] = {}
        for ep in episodes:
            mode = ep.get("mode", "unknown")
            modes[mode] = modes.get(mode, 0) + 1
            if ep.get("dice") is not None:
                dice_by_mode.setdefault(mode, []).append(float(ep["dice"]))
            r = int(ep.get("round_index", 0))
            rounds[r] = rounds.get(r, 0) + 1

        def _mean(xs: Iterable[float]) -> Optional[float]:
            xs = list(xs)
            return sum(xs) / len(xs) if xs else None

        return {
            "n_episodes": len(episodes),
            "modes": modes,
            "rounds": rounds,
            "mean_dice_by_mode": {
                k: round(_mean(v), 4) if _mean(v) is not None else None
                for k, v in dice_by_mode.items()
            },
            "ledger_path": str(self.jsonl_path),
        }

    # ------------------------------------------------------------------

    def export_summary_markdown(self, path: os.PathLike | str) -> str:
        """Write a small Markdown summary of the ledger to *path*."""
        s = self.summary()
        lines = [
            "# Segmentation experiment ledger",
            "",
            f"- run_id: `{self.run_id}`",
            f"- paper_tag: `{self.paper_tag or '(none)'}`",
            f"- total episodes: {s['n_episodes']}",
            "",
            "## Mode breakdown",
            "",
            "| Mode | Count | Mean Dice |",
            "|------|------:|---------:|",
        ]
        for mode, count in sorted(s.get("modes", {}).items()):
            mean_dice = s.get("mean_dice_by_mode", {}).get(mode)
            mean_str = f"{mean_dice:.4f}" if mean_dice is not None else "—"
            lines.append(f"| `{mode}` | {count} | {mean_str} |")

        lines += ["", "## Rounds", "",
                  "| Round | Episodes |",
                  "|------:|--------:|"]
        for r, count in sorted(s.get("rounds", {}).items()):
            lines.append(f"| {r} | {count} |")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return str(out)
