"""Decide whether the student model can stand alone on a new specimen.

The router consumes a `StudentPrediction` (foreground probability map +
optional uncertainty volume) and emits one of three decisions:

- ``ACCEPT_STUDENT``           — student is confident; ship its mask.
- ``CORRECT_WITH_NNINTERACTIVE`` — student is unsure on a non-trivial
                                   region; hand over to the LLM-driven
                                   nnInteractive paint loop, seeding it
                                   with the student's high-confidence
                                   bbox.
- ``REJECT``                    — student produced essentially nothing
                                   (or everything) — likely OOD; punt
                                   to nnInteractive from scratch.

The policy parameters are deliberately exposed: the publication will
report decision curves at multiple thresholds.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


log = logging.getLogger("seg_train.router")


class RouterDecision(str, enum.Enum):
    ACCEPT_STUDENT = "accept_student"
    CORRECT_WITH_NNINTERACTIVE = "correct_with_nninteractive"
    REJECT = "reject"


@dataclass
class RouterPolicy:
    """Tunable thresholds for :class:`ConfidenceRouter`.

    Attributes
    ----------
    min_confidence:
        Mean foreground probability below which the student is judged
        "low confidence" overall.
    max_entropy:
        Mean per-voxel binary entropy above which the student is judged
        "uncertain" (only meaningful if probability volume is supplied).
    min_foreground_fraction:
        Volumes where the predicted foreground is < this fraction of the
        total voxels are routed to nnInteractive (likely empty mask).
    max_foreground_fraction:
        And the symmetric upper bound (likely run-away segmentation).
    require_uncertainty:
        If True, the router refuses to accept the student unless an
        uncertainty estimate was supplied.
    """

    min_confidence: float = 0.85
    max_entropy: float = 0.35
    min_foreground_fraction: float = 1e-5
    max_foreground_fraction: float = 0.6
    require_uncertainty: bool = False
    name: str = "default_v1"


@dataclass
class StudentPrediction:
    """Lightweight description of a student prediction.

    The router does not load the volume; it operates on the summary
    statistics so it can run in either the parent env or the
    nnInteractive venv without a torch dependency.
    """

    label_path: str = ""
    voxel_count: int = 0
    total_voxels: int = 0
    mean_confidence: Optional[float] = None
    mean_entropy: Optional[float] = None
    bounding_box_xyz: Optional[list] = None  # [[x0,x1],[y0,y1],[z0,z1]]
    notes: str = ""

    @property
    def foreground_fraction(self) -> float:
        if self.total_voxels <= 0:
            return 0.0
        return float(self.voxel_count) / float(self.total_voxels)


@dataclass
class RouteResult:
    decision: RouterDecision
    reason: str
    diagnostics: dict = field(default_factory=dict)
    policy_name: str = ""
    threshold: Optional[float] = None  # main reason threshold (for ledger)

    @property
    def routes_to_nninteractive(self) -> bool:
        return self.decision in (
            RouterDecision.CORRECT_WITH_NNINTERACTIVE,
            RouterDecision.REJECT,
        )


# ---------------------------------------------------------------------------


class ConfidenceRouter:
    """Decide between accepting the student and invoking nnInteractive.

    The router is intentionally simple — its purpose is not to model
    student error, but to provide a deterministic, *auditable* gate
    whose threshold curve can be reported in the paper.
    """

    def __init__(self, policy: Optional[RouterPolicy] = None) -> None:
        self.policy = policy or RouterPolicy()

    # ------------------------------------------------------------------

    def decide(self, prediction: StudentPrediction) -> RouteResult:
        p = self.policy
        diags: dict = {
            "voxel_count": prediction.voxel_count,
            "foreground_fraction": prediction.foreground_fraction,
            "mean_confidence": prediction.mean_confidence,
            "mean_entropy": prediction.mean_entropy,
        }

        # Empty / runaway predictions are rejected outright.
        if prediction.foreground_fraction <= p.min_foreground_fraction:
            return RouteResult(
                decision=RouterDecision.REJECT,
                reason=(
                    f"foreground fraction {prediction.foreground_fraction:.2e} "
                    f"<= min {p.min_foreground_fraction:.2e}"
                ),
                diagnostics=diags,
                policy_name=p.name,
                threshold=p.min_foreground_fraction,
            )
        if prediction.foreground_fraction >= p.max_foreground_fraction:
            return RouteResult(
                decision=RouterDecision.REJECT,
                reason=(
                    f"foreground fraction {prediction.foreground_fraction:.3f} "
                    f">= max {p.max_foreground_fraction:.3f}"
                ),
                diagnostics=diags,
                policy_name=p.name,
                threshold=p.max_foreground_fraction,
            )

        # Confidence checks. If neither metric was supplied and the
        # policy demands uncertainty, fall back to nnInteractive.
        if (prediction.mean_confidence is None
                and prediction.mean_entropy is None):
            if p.require_uncertainty:
                return RouteResult(
                    decision=RouterDecision.CORRECT_WITH_NNINTERACTIVE,
                    reason="policy requires uncertainty, none supplied",
                    diagnostics=diags,
                    policy_name=p.name,
                )
            # No uncertainty available and policy allows it — accept.
            return RouteResult(
                decision=RouterDecision.ACCEPT_STUDENT,
                reason="no uncertainty available; policy permits",
                diagnostics=diags,
                policy_name=p.name,
            )

        if (prediction.mean_confidence is not None
                and prediction.mean_confidence < p.min_confidence):
            return RouteResult(
                decision=RouterDecision.CORRECT_WITH_NNINTERACTIVE,
                reason=(
                    f"mean confidence {prediction.mean_confidence:.3f} "
                    f"< min {p.min_confidence:.3f}"
                ),
                diagnostics=diags,
                policy_name=p.name,
                threshold=p.min_confidence,
            )

        if (prediction.mean_entropy is not None
                and prediction.mean_entropy > p.max_entropy):
            return RouteResult(
                decision=RouterDecision.CORRECT_WITH_NNINTERACTIVE,
                reason=(
                    f"mean entropy {prediction.mean_entropy:.3f} "
                    f"> max {p.max_entropy:.3f}"
                ),
                diagnostics=diags,
                policy_name=p.name,
                threshold=p.max_entropy,
            )

        return RouteResult(
            decision=RouterDecision.ACCEPT_STUDENT,
            reason="confidence above threshold",
            diagnostics=diags,
            policy_name=p.name,
            threshold=p.min_confidence,
        )


# ---------------------------------------------------------------------------
# Pure helpers for callers that *do* have probability arrays
# ---------------------------------------------------------------------------


def binary_entropy(p) -> float:
    """Mean per-voxel binary entropy of a probability array.

    Accepts either a numpy array or a python iterable. Probabilities
    outside (0, 1) contribute 0 entropy. Returns 0 for empty inputs.
    """
    try:
        import numpy as np
    except ImportError:
        # No numpy in env: best-effort scalar reduction.
        vals = list(p)
        if not vals:
            return 0.0
        eps = 1e-9
        s = 0.0
        for q in vals:
            if 0.0 < q < 1.0:
                s += -(q * math.log(q + eps)
                       + (1 - q) * math.log(1 - q + eps))
        return s / len(vals)
    arr = np.asarray(p, dtype=float)
    if arr.size == 0:
        return 0.0
    eps = 1e-9
    safe = np.clip(arr, eps, 1 - eps)
    h = -(safe * np.log(safe) + (1 - safe) * np.log(1 - safe))
    return float(h.mean())


def summarise_probability_volume(prob_array, threshold: float = 0.5) -> StudentPrediction:
    """Compute a :class:`StudentPrediction` from a softmax/sigmoid array.

    Expects a 3D numpy array with values in [0, 1]. Foreground voxels
    are determined by ``prob_array > threshold``.
    """
    import numpy as np

    arr = np.asarray(prob_array)
    fg_mask = arr > threshold
    n_fg = int(fg_mask.sum())
    total = int(arr.size)
    mean_conf = (
        float(arr[fg_mask].mean()) if n_fg > 0 else float(arr.max())
    )
    entropy = binary_entropy(arr)

    bbox = None
    if n_fg > 0:
        zs, ys, xs = np.where(fg_mask)
        bbox = [
            [int(xs.min()), int(xs.max())],
            [int(ys.min()), int(ys.max())],
            [int(zs.min()), int(zs.max())],
        ]
    return StudentPrediction(
        voxel_count=n_fg,
        total_voxels=total,
        mean_confidence=round(mean_conf, 6),
        mean_entropy=round(entropy, 6),
        bounding_box_xyz=bbox,
    )
