"""
Tests for the Plato's-Cave-style integrity graph.

These cover the data structures, validation, trust propagation, the
release-decision blends, and the Markdown rendering used in the
GitHub-issue verification comment.
"""

import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))

from integrity_graph import (  # noqa: E402
    IntegrityGraph,
    IntegrityNode,
    METRIC_KEYS,
    RELEASE_WEIGHTS,
    ROLES,
)


class TestIntegrityNode:
    def test_unknown_role_raises(self):
        with pytest.raises(ValueError):
            IntegrityNode(id=0, role="NotARealRole", text="x")

    def test_metrics_are_clamped(self):
        node = IntegrityNode(id=0, role="ResearchTopic", text="x",
                             metrics={"credibility": 1.5, "relevance": -0.3})
        assert node.metrics["credibility"] == 1.0
        assert node.metrics["relevance"] == 0.0
        for k in METRIC_KEYS:
            assert k in node.metrics

    def test_overall_is_mean(self):
        node = IntegrityNode(id=0, role="ResearchTopic", text="x",
                             metrics={k: 0.5 for k in METRIC_KEYS})
        assert node.overall == pytest.approx(0.5)


class TestIntegrityGraph:
    def _build_simple(self):
        g = IntegrityGraph(morphodepot_id="MD-T")
        topic = g.add_node("ResearchTopic", "test topic")
        source = g.add_node("SourceRecord", "media 123",
                            parents=[topic],
                            metrics={k: 0.9 for k in METRIC_KEYS})
        discovery = g.add_node("Discovery", "claim",
                               parents=[source],
                               metrics={k: 0.9 for k in METRIC_KEYS})
        return g, topic, source, discovery

    def test_add_node_assigns_increasing_ids(self):
        g, topic, source, discovery = self._build_simple()
        assert topic == 0 and source == 1 and discovery == 2
        assert len(g) == 3

    def test_dangling_parent_raises(self):
        g = IntegrityGraph()
        with pytest.raises(KeyError):
            g.add_node("SourceRecord", "x", parents=[42])

    def test_validate_clean_graph(self):
        g, *_ = self._build_simple()
        assert g.validate() == []

    def test_propagation_attenuates_low_parent(self):
        g = IntegrityGraph()
        topic = g.add_node("ResearchTopic", "t",
                           metrics={k: 0.2 for k in METRIC_KEYS})
        child = g.add_node("Discovery", "c",
                           parents=[topic],
                           metrics={k: 1.0 for k in METRIC_KEYS})
        g.propagate(parent_weight=0.6)
        topic_node = g.get_node(topic)
        child_node = g.get_node(child)
        # Parent has no parents itself, propagated == raw
        assert topic_node.propagated_metrics["credibility"] == pytest.approx(0.2)
        # Child raw=1.0 but parent gate=0.2, so propagated < 1.0
        assert child_node.propagated_metrics["credibility"] < 1.0
        assert child_node.propagated_metrics["credibility"] < child_node.metrics["credibility"]

    def test_propagation_zero_weight_is_identity(self):
        g, _, _, did = self._build_simple()
        g.propagate(parent_weight=0.0)
        node = g.get_node(did)
        for k in METRIC_KEYS:
            assert node.propagated_metrics[k] == pytest.approx(node.metrics[k])

    def test_release_decision_keys(self):
        g, *_ = self._build_simple()
        decision = g.release_decision()
        for k in RELEASE_WEIGHTS:
            assert k in decision
            assert 0.0 <= decision[k] <= 1.0
        assert decision["status"] in (
            "verified", "conditionally_verified", "needs_review", "rejected",
        )
        assert "weakest_dimension" in decision
        assert decision["per_metric_means"]

    def test_release_decision_high_quality_graph_is_verified(self):
        g = IntegrityGraph()
        topic = g.add_node("ResearchTopic", "t",
                           metrics={k: 0.95 for k in METRIC_KEYS})
        for _ in range(3):
            g.add_node("SourceRecord", "media",
                       parents=[topic],
                       metrics={k: 0.95 for k in METRIC_KEYS})
        decision = g.release_decision()
        assert decision["status"] == "verified"

    def test_release_decision_bad_lineage_is_rejected(self):
        g = IntegrityGraph()
        topic = g.add_node("ResearchTopic", "t",
                           metrics={k: 0.05 for k in METRIC_KEYS})
        g.add_node("SourceRecord", "media",
                   parents=[topic],
                   metrics={k: 0.10 for k in METRIC_KEYS})
        decision = g.release_decision()
        assert decision["status"] in ("rejected", "needs_review")

    def test_export_json_roundtrip(self, tmp_path):
        g, *_ = self._build_simple()
        path = g.export_json(tmp_path / "graph.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["morphodepot_id"] == "MD-T"
        assert len(data["nodes"]) == 3
        assert "release_decision" in data
        assert "validation_errors" in data

    def test_to_markdown_contains_status(self):
        g, *_ = self._build_simple()
        md = g.to_markdown()
        assert "## MorphoDepot Integrity Report" in md
        assert "scientific_validity" in md
        assert "ai_training_validity" in md
        assert "commercial_release_validity" in md

    def test_all_roles_constructible(self):
        g = IntegrityGraph()
        topic = g.add_node("ResearchTopic", "t")
        for role in ROLES:
            if role == "ResearchTopic":
                continue
            g.add_node(role, f"node of {role}", parents=[topic])
        assert len(g) == len(ROLES)
        assert g.validate() == []
