#!/usr/bin/env python3
"""
MorphoDepot Integrity Graph — Plato's-Cave-style validation DAG.

This module implements the data structures, scoring, trust propagation,
and release-decision logic used by the post-AutoResearchClaw verification
system.  It deliberately mirrors the architecture described in
"Plato's Cave" (Verma et al.):

    * Take an unstructured research artefact
    * Decompose it into a directed acyclic graph (DAG)
    * Score every node on multiple dimensions
    * Propagate trust from parents to children (low-trust parents
      attenuate child confidence)
    * Expose the result to a human reviewer

It is *intentionally* dependency-free (only Python stdlib) so the same
module can be reused by GitHub Actions, the local dashboard, and the
Tests/ suite without bringing in `requests`, `openai`, or `flask`.

Glossary
--------
Role
    The kind of artefact a node represents.  We use a small ontology
    tailored to MorphoDepot rather than the paper's research-paper
    ontology.  See :data:`ROLES`.

Metric
    A 0.0-1.0 score on one of six dimensions (credibility, relevance,
    evidence_strength, method_rigor, reproducibility, authority_support).

Effective score
    A node's metrics blended with its parents' overall trust.  This is
    the "trust gate": a perfectly-segmented mask whose underlying
    specimen ID is unverified will *not* be released as authoritative.

Release decision
    Three weighted blends on the propagated metrics:
    scientific_validity, ai_training_validity, commercial_release_validity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("IntegrityGraph")


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

# The MorphoDepot role ontology.  These are the only valid `role` strings
# for a node; the verifier rejects unknown roles to keep the DAG schema
# stable across runs (mirrors the Plato's Cave "ontology compliance"
# pre-flight check).
ROLES: Tuple[str, ...] = (
    "ResearchTopic",  # The research goal as written by the user
    "SourceRecord",  # A MorphoSource media or physical-object record
    "Specimen",  # The physical object behind a media record
    "MediaList",  # A MorphoSource media list (batch seed)
    "Query",  # A query the agent issued against MorphoSource
    "SearchResult",  # The aggregated result of a query
    "Discovery",  # An LLM-extracted discovery / claim
    "Citation",  # A DOI / paper extracted from cite_as / description
    "AISegmentation",  # A 3D-Slicer / MONAI / nnInteractive output
    "Measurement",  # A quantitative measurement (volume, distance, etc.)
    "ExpertReview",  # Human accept / edit / reject decision
    "ReleaseDecision",  # The aggregated, propagated verdict node
)

METRIC_KEYS: Tuple[str, ...] = (
    "credibility",
    "relevance",
    "evidence_strength",
    "method_rigor",
    "reproducibility",
    "authority_support",
)

DEFAULT_METRICS: Dict[str, float] = dict.fromkeys(METRIC_KEYS, 0.5)

# Release-decision blends.  Each weight maps a metric key to a positive
# float; the score for a release dimension is the weighted mean across
# only the metrics with non-zero weight.  This makes it cheap to add new
# release dimensions (e.g. "publication_validity") later.
RELEASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "scientific_validity": {
        "credibility": 1.0,
        "evidence_strength": 1.0,
        "method_rigor": 1.0,
        "relevance": 0.5,
    },
    "ai_training_validity": {
        "evidence_strength": 1.0,
        "method_rigor": 1.0,
        "reproducibility": 1.5,
        "credibility": 0.5,
    },
    "commercial_release_validity": {
        "credibility": 1.0,
        "evidence_strength": 1.0,
        "authority_support": 2.0,
        "method_rigor": 0.5,
    },
}

# Status thresholds applied to the *minimum* of the three release scores.
# Tunable; see calibration TODO in README.
STATUS_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.85, "verified"),
    (0.65, "conditionally_verified"),
    (0.45, "needs_review"),
    (0.0, "rejected"),
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into ``[lo, hi]`` for safe metric arithmetic."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IntegrityNode:
    """One artefact in the integrity DAG.

    ``metrics`` are the *raw* per-node scores produced by a verifier.  The
    *effective* score (after trust propagation) is computed by the graph
    and stored in :attr:`propagated_metrics` after :meth:`IntegrityGraph.propagate`
    has been called.
    """

    id: int
    role: str
    text: str
    parents: List[int] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_METRICS))
    evidence: Dict[str, Any] = field(default_factory=dict)
    verifier: str = ""
    propagated_metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Unknown role {self.role!r}; valid roles: {', '.join(ROLES)}")
        # Ensure all six metric keys are present and clamped.
        normalised: Dict[str, float] = {}
        for key in METRIC_KEYS:
            normalised[key] = _clamp(float(self.metrics.get(key, DEFAULT_METRICS[key])))
        self.metrics = normalised

    @property
    def overall(self) -> float:
        """Mean of the six raw metrics; the node's stand-alone trust."""
        return sum(self.metrics.values()) / len(self.metrics)

    @property
    def effective_overall(self) -> float:
        """Mean of the propagated metrics, falling back to raw if needed."""
        if self.propagated_metrics:
            return sum(self.propagated_metrics.values()) / len(self.propagated_metrics)
        return self.overall

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class IntegrityGraph:
    """Acyclic, validated graph of integrity nodes.

    Construction order matters: every parent referenced by a child must
    already exist when the child is added.  This guarantees acyclicity
    by construction, mirroring the paper's pre-flight "ontology compliance,
    referential integrity, acyclicity" check.
    """

    def __init__(self, morphodepot_id: str = "", source: Optional[Dict[str, Any]] = None):
        self.morphodepot_id = morphodepot_id
        self.source: Dict[str, Any] = source or {}
        self._nodes: Dict[int, IntegrityNode] = {}
        self._next_id: int = 0
        self._propagated: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_node(
        self,
        role: str,
        text: str,
        parents: Optional[Iterable[int]] = None,
        metrics: Optional[Dict[str, float]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        verifier: str = "",
        notes: Optional[List[str]] = None,
    ) -> int:
        """Add a node and return its integer id.

        Raises :class:`KeyError` if any parent id is unknown, which
        prevents cycles (you cannot reference a not-yet-created node).
        """
        parent_ids = list(parents or [])
        for pid in parent_ids:
            if pid not in self._nodes:
                raise KeyError(f"parent id {pid} does not exist")

        node = IntegrityNode(
            id=self._next_id,
            role=role,
            text=text,
            parents=parent_ids,
            metrics=dict(metrics or DEFAULT_METRICS),
            evidence=evidence or {},
            verifier=verifier,
            notes=list(notes or []),
        )
        self._nodes[node.id] = node
        self._next_id += 1
        self._propagated = False
        return node.id

    def get_node(self, node_id: int) -> IntegrityNode:
        return self._nodes[node_id]

    @property
    def nodes(self) -> List[IntegrityNode]:
        return list(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of integrity errors; an empty list means valid.

        Checks: ontology compliance, referential integrity, acyclicity,
        and metric range.  We never raise here so callers can attach the
        list to their report and let a human triage it.
        """
        errors: List[str] = []
        seen_ids: set[int] = set()
        for node in self._nodes.values():
            if node.role not in ROLES:
                errors.append(f"node {node.id}: unknown role {node.role!r}")
            for pid in node.parents:
                if pid not in self._nodes:
                    errors.append(f"node {node.id}: dangling parent {pid}")
                if pid >= node.id:
                    errors.append(f"node {node.id}: parent {pid} is not earlier (cycle risk)")
            for key, val in node.metrics.items():
                if not (0.0 <= val <= 1.0):
                    errors.append(f"node {node.id}: metric {key}={val} outside [0,1]")
            seen_ids.add(node.id)
        return errors

    # ------------------------------------------------------------------
    # Trust propagation
    # ------------------------------------------------------------------

    def propagate(self, parent_weight: float = 0.6) -> None:
        """Propagate trust from parents to children.

        ``parent_weight`` controls how much a parent's overall score
        attenuates a child's per-metric score.  ``0`` disables propagation
        entirely; ``1`` makes a child entirely subordinate to its
        weakest parent.  The default ``0.6`` follows the Plato's Cave
        guidance that gates should "reduce, not replace" child trust.

        The blend formula per metric is::

            propagated = raw_metric * ((1 - w) + w * parent_overall)

        where ``parent_overall`` is the *minimum* of all parent overall
        scores (worst-link semantics; matches morphology lineage logic
        where a single bad parent is disqualifying).
        """
        w = _clamp(parent_weight, 0.0, 1.0)
        # Process nodes in id order; because ids are monotonically assigned
        # and parents must be earlier ids, this is a valid topological order.
        for node in sorted(self._nodes.values(), key=lambda n: n.id):
            if not node.parents:
                node.propagated_metrics = dict(node.metrics)
                continue
            parent_overalls = [
                self._nodes[pid].propagated_metrics.get(
                    "_overall",
                    self._nodes[pid].overall,
                )
                for pid in node.parents
            ]
            gate = min(parent_overalls) if parent_overalls else 1.0
            multiplier = (1.0 - w) + w * gate
            propagated: Dict[str, float] = {}
            for key, val in node.metrics.items():
                propagated[key] = _clamp(val * multiplier)
            propagated["_overall"] = sum(propagated[k] for k in METRIC_KEYS) / len(METRIC_KEYS)
            node.propagated_metrics = propagated
        self._propagated = True

    # ------------------------------------------------------------------
    # Release decision
    # ------------------------------------------------------------------

    def release_decision(self) -> Dict[str, Any]:
        """Aggregate propagated metrics into a release decision dict.

        Always calls :meth:`propagate` first if it hasn't been called.
        Returns a dict matching the user-spec JSON schema, with the
        addition of ``per_node`` so the GitHub comment can show *why*
        a particular release dimension is low.
        """
        if not self._propagated:
            self.propagate()

        # Aggregate across non-Topic nodes (the topic is the "hypothesis";
        # it has no metrics worth blending into the release decision).
        per_metric_sums: Dict[str, float] = dict.fromkeys(METRIC_KEYS, 0.0)
        per_metric_counts: Dict[str, int] = dict.fromkeys(METRIC_KEYS, 0)
        for node in self._nodes.values():
            if node.role == "ResearchTopic":
                continue
            for key in METRIC_KEYS:
                per_metric_sums[key] += node.propagated_metrics.get(key, node.metrics[key])
                per_metric_counts[key] += 1

        per_metric_means: Dict[str, float] = {}
        for key in METRIC_KEYS:
            count = per_metric_counts[key] or 1
            per_metric_means[key] = round(per_metric_sums[key] / count, 4)

        scores: Dict[str, float] = {}
        for dim, weights in RELEASE_WEIGHTS.items():
            num = sum(weights[k] * per_metric_means[k] for k in weights)
            den = sum(weights.values()) or 1.0
            scores[dim] = round(num / den, 4)

        worst = min(scores.values()) if scores else 0.0
        status = "rejected"
        for threshold, label in STATUS_THRESHOLDS:
            if worst >= threshold:
                status = label
                break

        return {
            **scores,
            "status": status,
            "weakest_dimension": min(scores, key=scores.get) if scores else "",
            "per_metric_means": per_metric_means,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        if not self._propagated:
            self.propagate()
        return {
            "morphodepot_id": self.morphodepot_id,
            "source": self.source,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "release_decision": self.release_decision(),
            "validation_errors": self.validate(),
        }

    def export_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        log.info("Wrote integrity graph to %s (%d nodes)", out, len(self._nodes))
        return out

    # ------------------------------------------------------------------
    # Markdown reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _badge_emoji(score: float) -> str:
        if score >= 0.85:
            return "🟢"
        if score >= 0.65:
            return "🟡"
        if score >= 0.45:
            return "🟠"
        return "🔴"

    def to_markdown(self, max_nodes: int = 25) -> str:
        """Render a Markdown integrity report suitable for a GitHub comment.

        The report contains: badge header, release-decision table,
        per-metric breakdown, weakest-link explanation, and a node table
        truncated to *max_nodes* rows.
        """
        decision = self.release_decision()
        status = decision["status"]
        status_emoji = {
            "verified": "🟢",
            "conditionally_verified": "🟡",
            "needs_review": "🟠",
            "rejected": "🔴",
        }.get(status, "⚪")

        lines: List[str] = []
        lines.append("## MorphoDepot Integrity Report")
        lines.append("")
        lines.append(f"**Status:** {status_emoji} `{status}`  ")
        if self.morphodepot_id:
            lines.append(f"**MorphoDepot ID:** `{self.morphodepot_id}`  ")
        if self.source:
            src = self.source
            mid = src.get("media_id") or src.get("issue_number")
            if mid:
                lines.append(f"**Source:** {src.get('repository', 'GitHub')} `{mid}`  ")
        lines.append(f"**Nodes:** {len(self._nodes)}  ")
        lines.append(f"**Weakest dimension:** `{decision['weakest_dimension']}`  ")
        lines.append("")

        # Release-decision table
        lines.append("### Release decision")
        lines.append("")
        lines.append("| Dimension | Score | Badge |")
        lines.append("|-----------|------:|:-----:|")
        for dim in ("scientific_validity", "ai_training_validity", "commercial_release_validity"):
            s = decision[dim]
            lines.append(f"| `{dim}` | {s:.2f} | {self._badge_emoji(s)} |")
        lines.append("")

        # Per-metric averages
        lines.append("### Per-metric (propagated, mean across nodes)")
        lines.append("")
        lines.append("| Metric | Mean |")
        lines.append("|--------|-----:|")
        for k, v in decision["per_metric_means"].items():
            lines.append(f"| `{k}` | {v:.2f} |")
        lines.append("")

        # Node table (truncated)
        lines.append(f"### Nodes ({min(len(self._nodes), max_nodes)} of {len(self._nodes)})")
        lines.append("")
        lines.append("| # | Role | Text | Parents | Overall | Verifier |")
        lines.append("|---:|------|------|---------|--------:|----------|")
        for node in self._nodes.values():
            if node.id >= max_nodes:
                lines.append(f"| ... | ... | _+{len(self._nodes) - max_nodes} more nodes_ | | | |")
                break
            text = node.text.replace("|", "\\|")
            if len(text) > 80:
                text = text[:77] + "..."
            parents = ", ".join(str(p) for p in node.parents) or "-"
            lines.append(
                f"| {node.id} | `{node.role}` | {text} | {parents} | "
                f"{node.effective_overall:.2f} | {node.verifier or '-'} |"
            )
        lines.append("")

        # Validation errors
        errors = self.validate()
        if errors:
            lines.append("### Validation errors")
            lines.append("")
            for err in errors[:20]:
                lines.append(f"- {err}")
            lines.append("")

        lines.append("---")
        lines.append(
            "_Generated by `verify_research_run.py` — Plato's-Cave-style "
            "integrity DAG.  See `docs/INTEGRITY_VERIFIER.md` for methodology._"
        )
        return "\n".join(lines)


__all__ = [
    "DEFAULT_METRICS",
    "METRIC_KEYS",
    "RELEASE_WEIGHTS",
    "ROLES",
    "STATUS_THRESHOLDS",
    "IntegrityGraph",
    "IntegrityNode",
]
