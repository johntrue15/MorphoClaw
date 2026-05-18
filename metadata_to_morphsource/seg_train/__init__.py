"""Iterative self-training segmentation package.

This sub-package extends AutoResearchClaw with an iterative segmentation
training loop:

    Round 0  : nnInteractive paint loop produces pseudo-labels for
               MorphoSource specimens with curated GT meshes.
    Round 1+ : a custom student model (MONAI 3D U-Net) is trained on
               the accumulating (CT, pseudo-label) pairs. New specimens
               are routed by ``confidence_router`` either through the
               student alone (if confident) or through nnInteractive
               (for low-confidence corrections).
    Graduation: when validation Dice on a held-out human-curated set
               clears the configured threshold, the student runs
               autonomously while every prediction continues to be
               logged for the publication ledger.

All segmentation episodes are recorded by ``ExperimentLedger`` — the
JSONL + CSV log that backs the paper's "human vs autonomous" comparison
tables.

Modules
-------
- :mod:`experiment_ledger`     event recording, schema, summary stats
- :mod:`dataset`               manifest + train/val/holdout splits
- :mod:`student_model`         MONAI 3D U-Net (lazy torch import)
- :mod:`pseudo_label_generator` nnInteractive + voxelize mesh wrapper
- :mod:`confidence_router`     entropy-based dispatch policy
- :mod:`iterative_trainer`     round orchestrator
- :mod:`paper_export`          plots / tables for publication
- :mod:`cli`                   CLI: ``python -m metadata_to_morphsource.seg_train ...``
"""

from .experiment_ledger import (  # noqa: F401
    EpisodeRecord,
    ExperimentLedger,
    SegmentationMode,
)
from .confidence_router import (  # noqa: F401
    ConfidenceRouter,
    RouterDecision,
    RouterPolicy,
)
from .dataset import (  # noqa: F401
    DatasetManifest,
    SegmentationSample,
    SplitName,
)

__all__ = [
    "EpisodeRecord",
    "ExperimentLedger",
    "SegmentationMode",
    "ConfidenceRouter",
    "RouterDecision",
    "RouterPolicy",
    "DatasetManifest",
    "SegmentationSample",
    "SplitName",
]
