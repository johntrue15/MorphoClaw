#!/usr/bin/env python3
"""
Verifier agents for the MorphoDepot integrity DAG.

Each agent inspects an upstream artefact (a MorphoSource record, a
search result, a discovery sentence, an issue comment, etc.) and
returns a dict-like ``VerifierResult`` carrying:

    * Six per-metric scores in ``[0.0, 1.0]``
    * Free-form ``evidence`` (what the agent looked at)
    * Free-form ``notes`` (human-readable findings)

The agents are intentionally cheap: only the AI-QC verifier issues an
LLM call, and even that is optional.  The remaining verifiers are pure
field inspection plus, where available, MorphoSource API lookups.

Each verifier mirrors one of the five MVP roles described in the
"Plato's Cave for MorphoSource" pitch:

    Agent             | What it checks
    ------------------|--------------------------------------------------
    MetadataVerifier  | media id, specimen id, taxonomy, modality, DOI
    FileVerifier      | hash, voxel spacing, dimensions, format validity
    LineageVerifier   | parent media, raw vs derived, processing chain
    AIQCVerifier      | claim plausibility, anatomical sanity (LLM)
    ExpertVerifier    | reviewer identity, accept/edit/reject from issue

The verifiers do *not* know about the DAG; they return scores and the
caller (verify_research_run.py) attaches them to nodes.  This makes
them trivially unit-testable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from _helpers import safe_first

log = logging.getLogger("IntegrityVerifiers")

# Lazy imports so this module stays usable in lightweight contexts
# (e.g. unit tests with no `requests`, no `openai`).
_call_llm = None


def _get_call_llm():
    """Return the shared call_llm helper, or None if openai isn't available."""
    global _call_llm
    if _call_llm is None:
        try:
            from _helpers import call_llm

            _call_llm = call_llm
        except ImportError:

            def _noop_call_llm(*_a, **_kw):  # type: ignore[no-redef]
                return None

            _call_llm = _noop_call_llm
    return _call_llm


# ---------------------------------------------------------------------------
# Dataclass returned by every verifier
# ---------------------------------------------------------------------------


@dataclass
class VerifierResult:
    """Output of a single verifier run for a single artefact."""

    verifier: str
    metrics: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Metadata verifier
# ---------------------------------------------------------------------------


_REQUIRED_METADATA_FIELDS = (
    "id",
    "title",
    "media_type",
    "physical_object_id",
    "physical_object_taxonomy_name",
    "physical_object_organization",
    "visibility",
)


class MetadataVerifier:
    """Check that a MorphoSource record is internally complete and resolvable.

    Strategy:
      * Count how many "required" identifying fields are populated.
      * Treat the record as "resolved" if and only if it carries a real
        ``id`` field (i.e. the upstream cache populated it from API or
        disk).  This deliberately avoids a *second* network call per
        media id — the verifier no longer fetches; it scores what the
        cache already knows.
      * Penalise missing taxonomy / organisation / DOI heavily because
        they are the levers that downstream commercial-release decisions
        depend on.

    The ``allow_network`` flag is kept for backward compatibility but is
    a no-op: this verifier is now strictly offline and relies on the
    shared :class:`integrity_cache.RecordCache` to perform any HTTP
    work upstream.
    """

    name = "metadata"

    def __init__(self, allow_network: bool = True):
        self.allow_network = allow_network  # kept for API compatibility

    def verify(self, record: Dict[str, Any]) -> VerifierResult:
        result = VerifierResult(verifier=self.name)
        if not isinstance(record, dict):
            result.metrics = {
                "credibility": 0.0,
                "relevance": 0.0,
                "evidence_strength": 0.0,
                "method_rigor": 0.0,
                "reproducibility": 0.0,
                "authority_support": 0.0,
            }
            result.notes.append("record is not a dict")
            return result

        present = [f for f in _REQUIRED_METADATA_FIELDS if safe_first(record.get(f))]
        coverage = len(present) / len(_REQUIRED_METADATA_FIELDS)
        result.evidence["fields_present"] = present
        result.evidence["coverage"] = round(coverage, 3)

        media_id = safe_first(record.get("id"))
        org = safe_first(record.get("physical_object_organization"))
        taxon = safe_first(record.get("physical_object_taxonomy_name"))
        doi_text = " ".join(safe_first(record.get(k)) for k in ("doi", "cite_as", "description"))
        has_doi = bool(re.search(r"10\.\d{4,}/", doi_text))

        # The cache marked this record as "resolved" if there's at least
        # one substantive field beyond the bare media id.  We no longer
        # make a second API call here.
        substantive_fields = [f for f in present if f not in ("id",)]
        resolved = bool(media_id) and len(substantive_fields) >= 2
        result.evidence["resolved_via_cache"] = resolved

        # Score the six dimensions
        result.metrics = {
            "credibility": 0.5
            + 0.3 * (1.0 if media_id else 0.0)
            + 0.2 * (1.0 if resolved else 0.0),
            "relevance": coverage,
            "evidence_strength": coverage,
            "method_rigor": 0.5 + 0.5 * (1.0 if media_id and taxon else 0.0),
            "reproducibility": 0.7 if media_id else 0.3,
            "authority_support": (0.5 if org else 0.2) + (0.4 if has_doi else 0.0),
        }

        if not media_id:
            result.notes.append("missing media id")
        if not taxon:
            result.notes.append("missing taxonomy_name")
        if not org:
            result.notes.append("missing organisation")
        if not has_doi:
            result.notes.append("no DOI in cite_as / description")
        if media_id and not resolved:
            result.notes.append("record is bare media id only (cache miss / network skipped)")

        return result


# ---------------------------------------------------------------------------
# 2. File verifier
# ---------------------------------------------------------------------------


class FileVerifier:
    """Check voxel spacing, dimensions, and modality fields.

    This verifier does *not* download files; it scores how confidently
    a downstream pipeline could process the record based on the metadata
    that MorphoSource exposes.  The real heavy file inspection happens
    inside the SlicerTool path (slicer_layer2.py) which already populates
    the ``analysis`` dict; we re-use that dict when present.
    """

    name = "file"

    _NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")

    def verify(
        self,
        record: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None,
    ) -> VerifierResult:
        result = VerifierResult(verifier=self.name)
        if not isinstance(record, dict):
            result.metrics = dict.fromkeys(
                (
                    "credibility",
                    "relevance",
                    "evidence_strength",
                    "method_rigor",
                    "reproducibility",
                    "authority_support",
                ),
                0.0,
            )
            result.notes.append("record is not a dict")
            return result

        modality = safe_first(record.get("modality"))
        media_type = safe_first(record.get("media_type"))
        device = safe_first(record.get("device"))
        x_spacing = safe_first(record.get("x_spacing")) or safe_first(record.get("voxel_size_x"))
        y_spacing = safe_first(record.get("y_spacing")) or safe_first(record.get("voxel_size_y"))
        z_spacing = safe_first(record.get("z_spacing")) or safe_first(record.get("voxel_size_z"))
        slices = safe_first(record.get("slice_count")) or safe_first(record.get("slices"))

        spacings = [s for s in (x_spacing, y_spacing, z_spacing) if s]
        spacing_numeric = []
        for s in spacings:
            m = self._NUMERIC_RE.search(s)
            if m:
                try:
                    spacing_numeric.append(float(m.group()))
                except ValueError:
                    continue
        spacing_consistent = (
            len(spacing_numeric) >= 2
            and (max(spacing_numeric) / max(min(spacing_numeric), 1e-9)) < 100
        )

        slice_count = 0
        if slices:
            m = self._NUMERIC_RE.search(slices)
            if m:
                try:
                    slice_count = int(float(m.group()))
                except ValueError:
                    slice_count = 0

        result.evidence.update(
            {
                "modality": modality,
                "media_type": media_type,
                "device": device,
                "spacing_values": spacing_numeric,
                "spacing_consistent": spacing_consistent,
                "slice_count": slice_count,
            }
        )

        if analysis:
            result.evidence["analysis_summary"] = {
                k: analysis.get(k)
                for k in ("vertices", "volume_mm3", "surface_area_mm2", "distances")
                if k in analysis
            }

        # Score
        spacing_score = 0.0
        if spacing_numeric:
            spacing_score = 0.6 + (0.4 if spacing_consistent else 0.0)
        elif analysis:
            # If a downstream pipeline already computed measurements we
            # treat that as "spacing was good enough".
            spacing_score = 0.7

        modality_score = 0.7 if modality else 0.3
        media_type_score = 0.8 if media_type else 0.4

        result.metrics = {
            "credibility": modality_score,
            "relevance": media_type_score,
            "evidence_strength": (spacing_score + media_type_score) / 2.0,
            "method_rigor": 0.7 if device else 0.4,
            "reproducibility": spacing_score,
            "authority_support": 0.5,
        }

        if not modality:
            result.notes.append("missing modality")
        if not spacings:
            result.notes.append("no voxel-spacing fields populated")
        if slice_count == 0 and "Volumetric" in media_type:
            result.notes.append("volumetric media has no slice_count")

        return result


# ---------------------------------------------------------------------------
# 3. Lineage verifier
# ---------------------------------------------------------------------------


class LineageVerifier:
    """Check raw / derived status and parent-media chain.

    Plato's Cave principle: a derived artefact inherits trust from its
    parent.  An orphan derived mesh with no parent is suspicious; a
    raw CT stack with parent metadata is well-grounded.

    Parameters
    ----------
    cache
        Optional :class:`integrity_cache.RecordCache` used for parent
        media lookups.  When provided, the verifier delegates the
        parent fetch to the cache (which de-duplicates across all
        verifiers, throttles politely, and short-circuits behind the
        cache's circuit breaker).  When ``None``, the verifier never
        touches the network — it only checks the locally available
        ``media_parent_id`` field.
    allow_network
        Backward-compatibility flag; ignored when ``cache`` is None.
    """

    name = "lineage"

    _RAW_INDICATORS = ("Volumetric Image Series", "CT Image Series", "Image Series")
    _DERIVED_INDICATORS = ("Mesh", "Surface", "Reconstruction", "Segmentation")

    def __init__(self, cache: Optional[Any] = None, allow_network: bool = True):
        self.cache = cache
        self.allow_network = allow_network

    def verify(self, record: Dict[str, Any]) -> VerifierResult:
        result = VerifierResult(verifier=self.name)
        if not isinstance(record, dict):
            result.metrics = dict.fromkeys(
                (
                    "credibility",
                    "relevance",
                    "evidence_strength",
                    "method_rigor",
                    "reproducibility",
                    "authority_support",
                ),
                0.0,
            )
            return result

        media_type = safe_first(record.get("media_type"))
        parent_id = safe_first(record.get("media_parent_id"))
        derived = any(token in media_type for token in self._DERIVED_INDICATORS)
        raw = any(token in media_type for token in self._RAW_INDICATORS)

        parent_resolved = False
        if derived and parent_id and self.cache is not None and self.allow_network:
            try:
                parent_record = self.cache.get(parent_id)
                parent_resolved = bool(parent_record)
                result.evidence["parent_resolved"] = parent_resolved
            except Exception as exc:
                result.notes.append(f"parent cache lookup raised: {exc}")

        result.evidence.update(
            {
                "media_type": media_type,
                "parent_id": parent_id,
                "is_derived": derived,
                "is_raw": raw,
            }
        )

        # Score
        if (raw and not derived) or (derived and parent_id and parent_resolved):
            credibility = 0.85
            evidence = 0.8
        elif derived and parent_id:
            credibility = 0.7
            evidence = 0.65
        elif derived and not parent_id:
            credibility = 0.35
            evidence = 0.4
            result.notes.append("derived media has no parent id (orphan)")
        else:
            credibility = 0.5
            evidence = 0.5

        result.metrics = {
            "credibility": credibility,
            "relevance": 0.7 if media_type else 0.3,
            "evidence_strength": evidence,
            "method_rigor": 0.75 if (raw or parent_id) else 0.45,
            "reproducibility": 0.8 if parent_resolved else (0.6 if parent_id else 0.4),
            "authority_support": 0.6,
        }

        return result


# ---------------------------------------------------------------------------
# 4. AI QC verifier
# ---------------------------------------------------------------------------


_AI_QC_SYSTEM_PROMPT = (
    "You are an expert morphology curator reviewing an AI-generated "
    "discovery / claim about a MorphoSource specimen.  Score the claim on "
    "six dimensions, each from 0.0 (definitely false / unsupported) to 1.0 "
    "(definitely true / well supported), and return STRICT JSON of the form:\n"
    "{\n"
    '  "credibility": 0.0,\n'
    '  "relevance": 0.0,\n'
    '  "evidence_strength": 0.0,\n'
    '  "method_rigor": 0.0,\n'
    '  "reproducibility": 0.0,\n'
    '  "authority_support": 0.0,\n'
    '  "notes": "1-2 short sentences"\n'
    "}\n"
    "Do NOT hallucinate references.  If you cannot judge a dimension, "
    "use 0.5.  Be conservative."
)


class AIQCVerifier:
    """Use an LLM to score the plausibility of an AI-generated claim.

    The verifier is *opt-in*: if no OPENAI_API_KEY is available, or if
    ``use_llm=False`` is passed, it falls back to a heuristic scorer that
    rewards specific terms (genus, anatomy, modality) and penalises vague
    boilerplate.  This keeps the unit tests fully offline.
    """

    name = "ai_qc"

    # Heuristic vocabulary used in the offline fallback path.
    _CONCRETE_TERMS = (
        "specimen",
        "media",
        "morphosource",
        "ct",
        "mesh",
        "scan",
        "voxel",
        "modality",
        "open",
        "doi",
        "taxonomy",
    )
    _VAGUE_TERMS = (
        "interesting",
        "promising",
        "could be",
        "might be",
        "potentially",
        "various",
        "several",
    )

    def __init__(self, use_llm: bool = True, tier: str = "fast"):
        self.use_llm = use_llm and bool(os.environ.get("OPENAI_API_KEY"))
        self.tier = tier

    def _heuristic(self, claim: str) -> Dict[str, float]:
        text = (claim or "").lower()
        concrete_hits = sum(1 for t in self._CONCRETE_TERMS if t in text)
        vague_hits = sum(1 for t in self._VAGUE_TERMS if t in text)
        length_bonus = min(len(text) / 400.0, 1.0)
        signal = max(
            0.0, min(1.0, 0.4 + 0.08 * concrete_hits - 0.08 * vague_hits + 0.2 * length_bonus)
        )
        return {
            "credibility": signal,
            "relevance": signal,
            "evidence_strength": signal * 0.9,
            "method_rigor": signal * 0.85,
            "reproducibility": signal * 0.7,
            "authority_support": 0.4,
        }

    def verify(
        self,
        claim: str,
        topic: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> VerifierResult:
        result = VerifierResult(verifier=self.name)
        result.evidence["claim"] = claim
        if topic:
            result.evidence["topic"] = topic

        if not self.use_llm:
            metrics = self._heuristic(claim)
            result.metrics = metrics
            result.notes.append("heuristic scorer (no OPENAI_API_KEY)")
            return result

        call_llm = _get_call_llm()
        user_msg = f"Research topic: {topic or '(unknown)'}\n\n" f"Claim under review:\n{claim}\n\n"
        if context:
            user_msg += (
                f"Context (verbatim from MorphoSource search):\n{json.dumps(context)[:1500]}\n"
            )

        content = call_llm(
            [
                {"role": "system", "content": _AI_QC_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=400,
            json_mode=True,
            label="IntegrityAIQC",
            tier=self.tier,
        )

        if not content:
            metrics = self._heuristic(claim)
            result.metrics = metrics
            result.notes.append("LLM unavailable; fell back to heuristic")
            return result

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            result.metrics = self._heuristic(claim)
            result.notes.append("LLM returned non-JSON; fell back to heuristic")
            return result

        metrics = {}
        for key in (
            "credibility",
            "relevance",
            "evidence_strength",
            "method_rigor",
            "reproducibility",
            "authority_support",
        ):
            try:
                metrics[key] = float(parsed.get(key, 0.5))
            except (TypeError, ValueError):
                metrics[key] = 0.5
        result.metrics = metrics
        if isinstance(parsed.get("notes"), str):
            result.notes.append(parsed["notes"])
        return result


# ---------------------------------------------------------------------------
# 5. Expert verifier
# ---------------------------------------------------------------------------


# Reactions awarded by GitHub users on issue comments.  We use the
# emoji-vs-thumbs-down ratio as a lightweight proxy for "expert review"
# until a dedicated reviewer field is added.
_POSITIVE_REACTIONS = {"+1", "heart", "hooray", "rocket"}
_NEGATIVE_REACTIONS = {"-1", "confused"}

_REVIEW_KEYWORDS_ACCEPT = (
    "approved",
    "looks good",
    "lgtm",
    "verified",
    "accept",
    "correct",
    "confirmed",
)
_REVIEW_KEYWORDS_REJECT = (
    "incorrect",
    "wrong",
    "rejected",
    "not valid",
    "bad data",
    "duplicate",
    "needs fix",
)


class ExpertVerifier:
    """Translate a GitHub issue's reactions / comments into a trust score.

    Inputs come from the GitHub REST API: the issue itself plus its
    comments and (optionally) reactions.  The verifier ignores the
    research agent's own comments and looks only at human authors.
    """

    name = "expert"

    def __init__(self, bot_logins: Optional[List[str]] = None):
        # github-actions[bot] is what `actions/github-script` posts as.
        self.bot_logins = set(
            bot_logins
            or [
                "github-actions[bot]",
                "actions-user",
                "AutoResearchClaw",
            ]
        )

    @staticmethod
    def _is_bot(login: str) -> bool:
        if not login:
            return True
        login = login.lower()
        return login.endswith("[bot]") or login.startswith("bot-")

    def verify(
        self,
        issue: Dict[str, Any],
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> VerifierResult:
        result = VerifierResult(verifier=self.name)
        comments = comments or []

        human_comments: List[Dict[str, Any]] = []
        for c in comments:
            user = (c.get("user") or {}).get("login") or ""
            if user in self.bot_logins or self._is_bot(user):
                continue
            human_comments.append(c)

        accept_hits = 0
        reject_hits = 0
        text_blob = " ".join(c.get("body", "") for c in human_comments).lower()
        for kw in _REVIEW_KEYWORDS_ACCEPT:
            if kw in text_blob:
                accept_hits += 1
        for kw in _REVIEW_KEYWORDS_REJECT:
            if kw in text_blob:
                reject_hits += 1

        reactions = issue.get("reactions") or {}
        positive = sum(int(reactions.get(k, 0)) for k in _POSITIVE_REACTIONS)
        negative = sum(int(reactions.get(k, 0)) for k in _NEGATIVE_REACTIONS)

        labels = [
            (lab.get("name") if isinstance(lab, dict) else str(lab))
            for lab in issue.get("labels", [])
        ]
        label_accept = any(lab in {"verified", "approved", "expert-reviewed"} for lab in labels)
        label_reject = any(lab in {"rejected", "needs-fix", "invalid"} for lab in labels)

        result.evidence.update(
            {
                "human_comment_count": len(human_comments),
                "accept_keywords": accept_hits,
                "reject_keywords": reject_hits,
                "positive_reactions": positive,
                "negative_reactions": negative,
                "labels": labels,
            }
        )

        # Score
        if label_reject:
            base = 0.1
            result.notes.append("issue carries a 'rejected' / 'invalid' / 'needs-fix' label")
        elif label_accept:
            base = 0.95
            result.notes.append("issue carries a 'verified' / 'approved' label")
        elif human_comments:
            net = (accept_hits - reject_hits) + (positive - negative)
            base = 0.5 + 0.1 * net
            base = max(0.1, min(0.95, base))
        else:
            base = 0.4
            result.notes.append("no human reviewer activity detected")

        result.metrics = {
            "credibility": base,
            "relevance": 0.7 if human_comments else 0.5,
            "evidence_strength": 0.6 if human_comments else 0.4,
            "method_rigor": 0.6,
            "reproducibility": 0.6,
            "authority_support": base,
        }
        return result


__all__ = [
    "AIQCVerifier",
    "ExpertVerifier",
    "FileVerifier",
    "LineageVerifier",
    "MetadataVerifier",
    "VerifierResult",
]
