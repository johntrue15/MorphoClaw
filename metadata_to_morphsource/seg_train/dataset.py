"""Dataset manifest and split bookkeeping for the iterative trainer.

The iterative loop produces *(volume, label)* pairs from several sources
with different reliability:

================================  ========================
Source                            Reliability weight
================================  ========================
Curated MorphoSource GT mesh       1.00 (gold standard)
nnInteractive (LLM-driven)         0.75 (review / refine)
nnInteractive + human confirm      0.95
Student model + nnInteractive fix  0.85
Student model alone                0.60 (only used for SSL)
================================  ========================

We therefore need a small JSON-serialisable manifest that:

- Tracks every sample with provenance (mode, round produced, human?).
- Splits samples deterministically into ``train`` / ``val`` /
  ``holdout`` *by physical specimen*, never by media id, so the same
  scan does not leak across splits.
- Lets the trainer query "give me all train samples whose reliability
  is at least *r*".
- Snapshots itself to ``manifest.json`` so a later round can re-load the
  same split and add new pseudo-labelled samples on top.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


log = logging.getLogger("seg_train.dataset")


# ---------------------------------------------------------------------------
# Reliability table
# ---------------------------------------------------------------------------


# Default reliability weights per mode. Overridable via the manifest's
# ``reliability_overrides`` mapping.
DEFAULT_RELIABILITY: dict[str, float] = {
    "human_gt": 1.0,
    "expert_annotation": 1.0,
    "morphosource_mesh": 0.98,
    "nninteractive_human_confirmed": 0.95,
    "student_corrected": 0.85,
    "nninteractive": 0.75,
    "monai_auto": 0.7,
    "ensemble": 0.8,
    "student": 0.6,
}


def reliability_for(mode: str, overrides: Optional[dict[str, float]] = None) -> float:
    if overrides and mode in overrides:
        return float(overrides[mode])
    return float(DEFAULT_RELIABILITY.get(mode, 0.5))


# ---------------------------------------------------------------------------
# Sample / split data classes
# ---------------------------------------------------------------------------


class SplitName(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    HOLDOUT = "holdout"


@dataclass
class SegmentationSample:
    """A (volume, label) pair on disk."""

    sample_id: str
    media_id: str
    physical_object_id: str
    volume_path: str
    label_path: str
    mode: str
    round_index: int = 0
    reliability: float = 0.5
    has_human_review: bool = False
    taxonomy: str = ""
    source_run_id: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentationSample":
        known = {f.name for f in dataclasses.fields(cls)}
        cleaned = {k: v for k, v in d.items() if k in known}
        return cls(**cleaned)

    def exists(self) -> bool:
        return Path(self.volume_path).exists() and Path(self.label_path).exists()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _hash_to_unit_interval(key: str) -> float:
    """Deterministic [0, 1) hash used for split assignment."""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return n / 2**64


@dataclass
class DatasetManifest:
    """Snapshot of the iterative training dataset on disk.

    Persisted as ``manifest.json`` inside the dataset root. The schema is
    intentionally explicit (no implicit defaults beyond reliability) so a
    reviewer can audit how samples flowed into each split.
    """

    root: str = ""
    samples: list[SegmentationSample] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=dict)
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    holdout_ratio: float = 0.1
    seed: int = 1234
    reliability_overrides: dict[str, float] = field(default_factory=dict)

    # ---------------- Load / save ----------------

    def save(self, path: Optional[str] = None) -> str:
        out = Path(path or (Path(self.root) / "manifest.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": self.root,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "holdout_ratio": self.holdout_ratio,
            "seed": self.seed,
            "reliability_overrides": self.reliability_overrides,
            "splits": self.splits,
            "samples": [s.to_dict() for s in self.samples],
        }
        out.write_text(json.dumps(payload, indent=2, default=str))
        return str(out)

    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        data = json.loads(Path(path).read_text())
        m = cls(
            root=data.get("root", str(Path(path).parent)),
            train_ratio=float(data.get("train_ratio", 0.7)),
            val_ratio=float(data.get("val_ratio", 0.2)),
            holdout_ratio=float(data.get("holdout_ratio", 0.1)),
            seed=int(data.get("seed", 1234)),
            reliability_overrides=dict(data.get("reliability_overrides", {})),
        )
        m.samples = [SegmentationSample.from_dict(s) for s in data.get("samples", [])]
        m.splits = {k: list(v) for k, v in data.get("splits", {}).items()}
        return m

    # ---------------- Mutation -------------------

    def add_sample(self, sample: SegmentationSample) -> SegmentationSample:
        """Add (or upsert by ``sample_id``) and re-assign its split."""
        if not sample.sample_id:
            sample.sample_id = sample.media_id or hashlib.md5(
                f"{sample.volume_path}|{sample.label_path}".encode("utf-8")
            ).hexdigest()[:12]
        if sample.reliability == 0.5 and sample.mode:
            sample.reliability = reliability_for(
                sample.mode, self.reliability_overrides
            )
        existing_idx = next(
            (i for i, s in enumerate(self.samples)
             if s.sample_id == sample.sample_id), None
        )
        if existing_idx is not None:
            self.samples[existing_idx] = sample
        else:
            self.samples.append(sample)
        self._assign_split(sample)
        return sample

    def _assign_split(self, sample: SegmentationSample) -> str:
        # Group split assignment by physical_object_id so two media of
        # the same specimen never straddle splits. Fall back to media_id.
        bucket_key = (
            sample.physical_object_id
            or sample.media_id
            or sample.sample_id
        )
        # Holdout takes priority for "human_gt" / curated samples so that
        # validation is always done against the gold-standard pool.
        u = _hash_to_unit_interval(f"{self.seed}|{bucket_key}")
        if sample.has_human_review or sample.mode in (
            "human_gt", "expert_annotation"
        ):
            split = SplitName.HOLDOUT.value
        else:
            if u < self.train_ratio:
                split = SplitName.TRAIN.value
            elif u < self.train_ratio + self.val_ratio:
                split = SplitName.VAL.value
            else:
                split = SplitName.HOLDOUT.value
        # Remove from any existing split, add to the new one.
        for s in self.splits.values():
            if sample.sample_id in s:
                s.remove(sample.sample_id)
        self.splits.setdefault(split, []).append(sample.sample_id)
        return split

    # ---------------- Query ----------------------

    def get(self, sample_id: str) -> Optional[SegmentationSample]:
        for s in self.samples:
            if s.sample_id == sample_id:
                return s
        return None

    def split_samples(
        self, split: str, *, min_reliability: float = 0.0
    ) -> list[SegmentationSample]:
        ids = set(self.splits.get(split, []))
        return [
            s for s in self.samples
            if s.sample_id in ids and s.reliability >= min_reliability
            and s.exists()
        ]

    def by_mode(self, mode: str) -> list[SegmentationSample]:
        return [s for s in self.samples if s.mode == mode]

    def __iter__(self) -> Iterator[SegmentationSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    # ---------------- Reporting ------------------

    def summary(self) -> dict:
        counts: dict[str, dict] = {s.value: {} for s in SplitName}
        for split in SplitName:
            for s in self.split_samples(split.value):
                counts[split.value][s.mode] = counts[split.value].get(
                    s.mode, 0
                ) + 1
        return {
            "total": len(self.samples),
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "holdout": self.holdout_ratio,
            },
            "split_sizes": {
                split.value: len(self.splits.get(split.value, []))
                for split in SplitName
            },
            "split_modes": counts,
        }

    def shuffled_train_iter(
        self,
        *,
        min_reliability: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> Iterable[SegmentationSample]:
        rng = rng or random.Random(self.seed)
        items = self.split_samples(
            SplitName.TRAIN.value, min_reliability=min_reliability
        )
        rng.shuffle(items)
        return items
