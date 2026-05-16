"""
Tests for the five integrity verifier agents.

These run completely offline:

  * MorphoSource API calls are stubbed via ``allow_network=False``
  * The AI-QC LLM call is disabled via ``use_llm=False``
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))

from integrity_verifiers import (  # noqa: E402
    AIQCVerifier,
    ExpertVerifier,
    FileVerifier,
    LineageVerifier,
    MetadataVerifier,
    VerifierResult,
)


# ---------------------------------------------------------------------------
# Sample MorphoSource record (mirrors the sample in knowledge_graph.py)
# ---------------------------------------------------------------------------


SAMPLE_MEDIA_RECORD = {
    "id": ["000769445"],
    "title": ["Skull"],
    "media_type": ["Mesh"],
    "modality": ["MicroNanoXRayComputedTomography"],
    "visibility": ["open"],
    "media_parent_id": ["000408242"],
    "physical_object_id": ["000408235"],
    "physical_object_title": ["uf:herp:191369"],
    "physical_object_organization": ["FLMNH Division of Herpetology"],
    "physical_object_taxonomy_name": ["Chamaeleo calyptratus"],
    "device": ["Nikon XT H 225 ST"],
    "x_spacing": ["0.05 mm"],
    "y_spacing": ["0.05 mm"],
    "z_spacing": ["0.05 mm"],
    "slice_count": ["1024"],
    "doi": ["10.17602/M2/M769445"],
    "cite_as": [
        "When re-using this media, please cite Almecija et al., 2024 SciData "
        "(doi: https://doi.org/10.1038/s41597-024-04261-5)."
    ],
}

EMPTY_RECORD: dict = {}


class TestMetadataVerifier:
    def test_full_record_scores_high(self):
        v = MetadataVerifier(allow_network=False)
        result = v.verify(SAMPLE_MEDIA_RECORD)
        assert isinstance(result, VerifierResult)
        assert result.metrics["credibility"] >= 0.7
        assert result.metrics["evidence_strength"] >= 0.7
        assert result.metrics["authority_support"] >= 0.7
        # Verifier should never make a second API call: a fully-populated
        # record is treated as "resolved via cache".
        assert result.evidence.get("resolved_via_cache") is True

    def test_empty_record_scores_low(self):
        v = MetadataVerifier(allow_network=False)
        result = v.verify(EMPTY_RECORD)
        assert result.metrics["evidence_strength"] == 0.0
        assert "missing media id" in " ".join(result.notes)

    def test_bare_id_record_is_marked_unresolved(self):
        v = MetadataVerifier(allow_network=False)
        result = v.verify({"id": ["000123456"]})
        assert result.evidence.get("resolved_via_cache") is False
        assert any("bare media id" in n for n in result.notes)

    def test_non_dict_input_fails_safely(self):
        v = MetadataVerifier(allow_network=False)
        result = v.verify("not a dict")  # type: ignore[arg-type]
        assert all(score == 0.0 for score in result.metrics.values())


class TestFileVerifier:
    def test_full_record_scores_high(self):
        v = FileVerifier()
        result = v.verify(SAMPLE_MEDIA_RECORD)
        assert result.metrics["reproducibility"] >= 0.6
        assert result.evidence["spacing_consistent"] is True
        assert result.evidence["spacing_values"] == [0.05, 0.05, 0.05]

    def test_missing_modality_lowers_credibility(self):
        record = dict(SAMPLE_MEDIA_RECORD)
        record.pop("modality")
        v = FileVerifier()
        result = v.verify(record)
        assert result.metrics["credibility"] < 0.5

    def test_with_analysis_dict(self):
        v = FileVerifier()
        result = v.verify(EMPTY_RECORD, analysis={
            "vertices": 12345,
            "volume_mm3": 6789.0,
            "surface_area_mm2": 123.4,
        })
        assert result.evidence["analysis_summary"]["vertices"] == 12345


class TestLineageVerifier:
    def test_raw_specimen_is_credible(self):
        record = dict(SAMPLE_MEDIA_RECORD,
                      media_type=["Volumetric Image Series"],
                      media_parent_id=[""])
        v = LineageVerifier(cache=None, allow_network=False)
        result = v.verify(record)
        assert result.metrics["credibility"] >= 0.8

    def test_orphan_derived_is_penalised(self):
        record = dict(SAMPLE_MEDIA_RECORD,
                      media_type=["Mesh"],
                      media_parent_id=[""])
        v = LineageVerifier(cache=None, allow_network=False)
        result = v.verify(record)
        assert result.metrics["credibility"] <= 0.45
        assert "orphan" in " ".join(result.notes).lower()

    def test_derived_with_parent_id_is_okay(self):
        v = LineageVerifier(cache=None, allow_network=False)
        result = v.verify(SAMPLE_MEDIA_RECORD)
        assert result.metrics["credibility"] >= 0.6

    def test_parent_lookup_uses_cache(self):
        """Parent fetch must go through the shared cache, not raw HTTP."""
        from integrity_cache import RecordCache

        # Fetcher that records every call, returning a fake parent record
        calls = []

        def fetcher(mid):
            calls.append(mid)
            return {"id": [mid], "title": ["parent"]}

        cache = RecordCache(cache_dir=None, fetcher=fetcher,
                             min_delay_s=0.0, circuit_breaker_threshold=0)
        v = LineageVerifier(cache=cache, allow_network=True)
        result = v.verify(SAMPLE_MEDIA_RECORD)
        # One call to the fetcher, for the parent media id.
        assert calls == ["000408242"]
        assert result.evidence["parent_resolved"] is True


class TestAIQCVerifier:
    def test_heuristic_path_runs_without_openai(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        v = AIQCVerifier(use_llm=True)  # gets demoted to heuristic
        result = v.verify("MorphoSource media 000769445 has open-access CT mesh data",
                          topic="lizard cranial morphology")
        assert "heuristic" in " ".join(result.notes).lower()
        for k in ("credibility", "relevance", "evidence_strength"):
            assert 0.0 <= result.metrics[k] <= 1.0

    def test_concrete_claim_outscores_vague_claim(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        v = AIQCVerifier(use_llm=False)
        concrete = v.verify(
            "Open-access CT mesh of Chamaeleo calyptratus on MorphoSource "
            "media 000769445 with voxel spacing 0.05 mm",
            topic="reptile skull",
        )
        vague = v.verify(
            "There are several interesting and potentially promising specimens.",
            topic="reptile skull",
        )
        assert concrete.metrics["credibility"] > vague.metrics["credibility"]


class TestExpertVerifier:
    def test_no_human_reviewer(self):
        v = ExpertVerifier()
        issue = {"number": 1, "labels": [], "reactions": {}}
        result = v.verify(issue, comments=[])
        assert result.metrics["credibility"] <= 0.5
        assert "no human reviewer" in " ".join(result.notes).lower()

    def test_accept_keyword_raises_score(self):
        v = ExpertVerifier()
        issue = {"number": 2, "labels": [], "reactions": {}}
        comments = [
            {"user": {"login": "reviewer1"}, "body": "Looks good. LGTM."},
        ]
        result = v.verify(issue, comments=comments)
        assert result.metrics["credibility"] > 0.5

    def test_verified_label_overrides_comments(self):
        v = ExpertVerifier()
        issue = {"number": 3,
                 "labels": [{"name": "verified"}, {"name": "research-agent"}],
                 "reactions": {}}
        result = v.verify(issue, comments=[])
        assert result.metrics["credibility"] >= 0.9

    def test_rejected_label_floors_score(self):
        v = ExpertVerifier()
        issue = {"number": 4,
                 "labels": [{"name": "rejected"}],
                 "reactions": {}}
        result = v.verify(issue, comments=[])
        assert result.metrics["credibility"] <= 0.2

    def test_bot_comments_are_ignored(self):
        v = ExpertVerifier()
        issue = {"number": 5, "labels": [], "reactions": {}}
        comments = [
            {"user": {"login": "github-actions[bot]"},
             "body": "Looks good. Verified. Accepted."},
        ]
        result = v.verify(issue, comments=comments)
        # bot comments stripped -> no human activity detected
        assert "no human reviewer" in " ".join(result.notes).lower()
