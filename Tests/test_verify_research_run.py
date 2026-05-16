"""
Tests for the verify_research_run main entry point.

Focuses on the pure-Python helpers (extract_media_ids,
extract_discoveries, extract_queries, extract_topic, build_graph in
``--no-network`` mode) so the suite stays offline.
"""

import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))

import verify_research_run as vrr  # noqa: E402


SAMPLE_ISSUE_BODY = """### AutoResearchClaw Research Program

**Research Topic:**
Identify open-access CT scans of Chamaeleo calyptratus.

**Research Depth:** 5 cycles
**Seed Media ID:** [`000769445`](https://www.morphosource.org/concern/media/000769445)

---

### Key Discoveries
- MorphoSource media 000769445 has open-access mesh data
- Specimen uf:herp:191369 from FLMNH is well annotated
- Several CT scans available with voxel spacing 0.05 mm
- A potentially promising lead: CT mesh of Chamaeleo

### Queries
- "Chamaeleo calyptratus" -> 12 results
- "FLMNH herpetology" -> 3 results
- "lizard skull mesh" -> 0 results

---
_AutoResearchClaw issue 1/3_
"""


class TestExtractors:
    def test_extract_media_ids_url(self):
        ids = vrr.extract_media_ids(
            "see https://www.morphosource.org/concern/media/000769445 for details"
        )
        assert "000769445" in ids

    def test_extract_media_ids_token(self):
        ids = vrr.extract_media_ids("media id 000408242 referenced")
        assert "000408242" in ids

    def test_extract_discoveries_section(self):
        discoveries = vrr.extract_discoveries(SAMPLE_ISSUE_BODY)
        assert any("MorphoSource media 000769445" in d for d in discoveries)
        assert any("uf:herp:191369" in d for d in discoveries)

    def test_extract_queries(self):
        queries = vrr.extract_queries(SAMPLE_ISSUE_BODY)
        text_to_hits = dict(queries)
        assert text_to_hits.get("Chamaeleo calyptratus") == 12
        assert text_to_hits.get("lizard skull mesh") == 0

    def test_extract_topic(self):
        issue = {"title": "Research [5x/3 issues]: Chamaeleo CT scans",
                 "body": SAMPLE_ISSUE_BODY}
        topic = vrr.extract_topic(issue)
        assert "Chamaeleo calyptratus" in topic


class TestBuildGraphOffline:
    def test_build_graph_no_network(self):
        issue = {
            "number": 999,
            "title": "Research [5x/3 issues]: Chamaeleo CT scans",
            "body": SAMPLE_ISSUE_BODY,
            "labels": [{"name": "research-agent"}],
            "reactions": {"+1": 1},
        }
        graph = vrr.build_graph(
            issue, comments=[], child_issues=[],
            use_llm=False, allow_network=False,
        )
        # Topic + at least one source + at least one discovery + expert
        assert len(graph) >= 4
        roles = [n.role for n in graph.nodes]
        assert "ResearchTopic" in roles
        assert "Discovery" in roles
        assert "ExpertReview" in roles

    def test_decision_for_offline_run_is_finite(self):
        issue = {
            "number": 999,
            "title": "Research [5x/3 issues]: Chamaeleo CT scans",
            "body": SAMPLE_ISSUE_BODY,
            "labels": [],
            "reactions": {},
        }
        graph = vrr.build_graph(
            issue, comments=[], child_issues=[],
            use_llm=False, allow_network=False,
        )
        decision = graph.release_decision()
        for k in ("scientific_validity", "ai_training_validity",
                  "commercial_release_validity"):
            assert 0.0 <= decision[k] <= 1.0


class TestSynthesizeFromReport:
    def test_offline_report_round_trip(self, tmp_path):
        report = {
            "topic": "Chamaeleo CT",
            "final_memory": {
                "all_discoveries": [
                    "MorphoSource media 000769445 has open-access mesh",
                    "Found Chamaeleo calyptratus specimen uf:herp:191369",
                ],
                "queries_tried": [
                    {"query": "Chamaeleo calyptratus", "count": 12},
                ],
            },
        }
        path = tmp_path / "research_report.json"
        path.write_text(json.dumps(report), encoding="utf-8")

        issue, comments = vrr._synthesize_issue_from_report(path)
        assert issue["number"] == 0
        assert "Chamaeleo CT" in issue["body"]
        assert comments == []

        graph = vrr.build_graph(
            issue, comments=comments, child_issues=[],
            use_llm=False, allow_network=False,
        )
        assert len(graph) >= 3


class TestExitCodes:
    @pytest.mark.parametrize("status,expected", [
        ("verified", 0),
        ("conditionally_verified", 0),
        ("needs_review", 1),
        ("rejected", 2),
        ("anything_else", 2),
    ])
    def test_exit_code_for(self, status, expected):
        assert vrr._exit_code_for(status) == expected


class TestClientNameShadowingRegression:
    """Regression test for the variable-name collision between
    ``GitHubClient`` and the polite MorphoSource client.

    Before the fix, ``main()`` reused ``client`` for both objects, so
    by the time ``post_or_print()`` was called with the GitHub client,
    it was actually holding a ``MorphoSourceClient`` (no ``.enabled``).
    """

    def test_github_client_has_enabled(self):
        gh = vrr.GitHubClient(repo="owner/repo", token="t")
        assert hasattr(gh, "enabled")
        assert gh.enabled is True

    def test_polite_client_does_not_collide(self):
        ms = vrr.build_polite_client()
        # Polite client is either None (no morphosource_client import)
        # or a real MorphoSourceClient — never something that should
        # pose as a GitHubClient.
        assert not isinstance(ms, vrr.GitHubClient)

    def test_post_or_print_dry_run_does_not_touch_client(self, capsys):
        # Pass a deliberately-broken client object; dry-run must not
        # touch it.  This is what saved us in production after the fix.
        class BrokenClient:
            pass

        vrr.post_or_print(BrokenClient(), 123, "## report", dry_run=True)
        captured = capsys.readouterr()
        assert "## report" in captured.out
