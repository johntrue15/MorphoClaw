---
title: AutoResearchClaw
description: Autonomous MorphoSource research agent with a live knowledge graph, iterative segmentation, and integrity verification.
hide:
  - navigation
---

<div class="arc-hero" markdown>

# AutoResearchClaw

<p class="arc-hero-subtitle">
  An autonomous MorphoSource research agent. It searches for 3D specimen data,
  runs headless 3D Slicer + SlicerMorph analyses, builds a live knowledge graph,
  and produces structured research reports as GitHub Issues &mdash; with an
  integrity verifier scoring every output.
</p>

<div class="arc-hero-actions" markdown>

[:material-rocket-launch: Quick Start](quick-start.md){ .arc-btn .arc-btn-primary }
[:material-graph-outline: Explore the Knowledge Graph](knowledge-graph.md){ .arc-btn .arc-btn-secondary }
[:material-message-question: Submit a Query](query.md){ .arc-btn .arc-btn-secondary }

</div>

</div>

<div class="arc-badges" markdown>

[![License](https://img.shields.io/github/license/johntrue15/Metadata-to-Morphsource-compare?style=flat-square)](https://github.com/johntrue15/Metadata-to-Morphsource-compare/blob/main/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/johntrue15/Metadata-to-Morphsource-compare/tests.yml?branch=main&label=tests&style=flat-square)](https://github.com/johntrue15/Metadata-to-Morphsource-compare/actions/workflows/tests.yml)
[![Code Quality](https://img.shields.io/github/actions/workflow/status/johntrue15/Metadata-to-Morphsource-compare/code-quality.yml?branch=main&label=code%20quality&style=flat-square)](https://github.com/johntrue15/Metadata-to-Morphsource-compare/actions/workflows/code-quality.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/johntrue15/Metadata-to-Morphsource-compare/docs.yml?branch=main&label=docs&style=flat-square)](https://github.com/johntrue15/Metadata-to-Morphsource-compare/actions/workflows/docs.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?style=flat-square)](https://www.python.org/)

</div>

## Knowledge Graph at a glance

{% set s = kg_stats() %}

<div class="arc-stats" markdown>

<div class="arc-stat">
  <div class="arc-stat-value">{{ s.media }}</div>
  <div class="arc-stat-label">Media</div>
</div>
<div class="arc-stat">
  <div class="arc-stat-value">{{ s.specimens }}</div>
  <div class="arc-stat-label">Specimens</div>
</div>
<div class="arc-stat">
  <div class="arc-stat-value">{{ s.taxa }}</div>
  <div class="arc-stat-label">Taxa</div>
</div>
<div class="arc-stat">
  <div class="arc-stat-value">{{ s.papers }}</div>
  <div class="arc-stat-label">Papers</div>
</div>
<div class="arc-stat">
  <div class="arc-stat-value">{{ s.institutions }}</div>
  <div class="arc-stat-label">Institutions</div>
</div>
<div class="arc-stat">
  <div class="arc-stat-value">{{ s.total_edges }}</div>
  <div class="arc-stat-label">Connections</div>
</div>

</div>

{% if kg_has_data() %}
:material-update: Last updated **{{ s.generated_at or "unknown" }}** across **{{ kg_run_count() }}** archived run(s).
[Open the live graph &rarr;](knowledge-graph.md)
{% else %}
!!! info "No knowledge graph data published yet"
    The graph is empty because no AutoResearchClaw run has published a snapshot
    to `docs/data/knowledge_graph.json` yet. Trigger the
    [**AutoResearchClaw Agent** workflow](https://github.com/johntrue15/Metadata-to-Morphsource-compare/actions/workflows/autoresearchclaw.yml)
    once and the graph will populate automatically on the next push.
{% endif %}

## How it works

```mermaid
flowchart LR
    Topic["Research topic +<br/>optional seed media ID"] --> Agent

    subgraph Agent["AutoResearchClaw Agent (research_agent.py)"]
        direction TB
        Inner["Inner loop (research_depth):<br/>decompose &rarr; search &rarr;<br/>evaluate &rarr; build memory"]
        Outer["Outer loop (github_issues):<br/>aggregate &rarr; create issue"]
        Inner --> Outer
    end

    Agent --> Tools["Tool calls:<br/>MorphoSource API, downloads,<br/>3D Slicer / SlicerMorph,<br/>literature, knowledge graph,<br/>UBERON ontology"]
    Tools --> Outputs

    subgraph Outputs["Outputs"]
        direction TB
        Issues["GitHub Issues"]
        Dash["Local Dashboard"]
        KG["Knowledge Graph<br/>(JSON + interactive)"]
    end

    Outputs --> Verifier["MorphoDepot Integrity Verifier<br/>(Plato's Cave DAG)"]
```

## What's inside

<div class="arc-grid" markdown>

<div class="arc-card" markdown>
### :material-loop: Two-loop research engine
The inner loop runs fast cycles &mdash; decompose, search MorphoSource, evaluate,
remember. The outer loop posts cumulative findings as GitHub issues. Memory
carries between cycles. <a href="architecture/">Read more &rarr;</a>
</div>

<div class="arc-card" markdown>
### :material-rotate-3d-variant: 3D Slicer + SlicerMorph
Headless morphometric analysis pipeline: dimensions, surface area, volume,
curvature, PCA, landmarks, plus publication-grade screenshots from cached
specimens. <a href="reference/cli/">CLI &rarr;</a>
</div>

<div class="arc-card" markdown>
### :material-brain: nnInteractive paint loop
LLM-driven iterative segmentation that converges on the right mask in a
handful of clicks. Comparison harness validates against MorphoSource
ground-truth meshes. <a href="ITERATIVE_SEGMENTATION/">Methodology &rarr;</a>
</div>

<div class="arc-card" markdown>
### :material-graph: Live knowledge graph
Builds connections between media, specimens, papers, institutions, and
taxa. Auto-publishes a JSON snapshot after every run so this site stays
fresh. <a href="knowledge-graph/">Explore &rarr;</a>
</div>

<div class="arc-card" markdown>
### :material-shield-check: Integrity verifier
A Plato's-Cave trust DAG over each run's issues. Five MVP agents produce
six per-metric scores and a release-readiness verdict. <a href="INTEGRITY_VERIFIER/">Read more &rarr;</a>
</div>

<div class="arc-card" markdown>
### :material-school: Iterative self-training
Bootstraps a custom 3D student model from nnInteractive outputs, graduates
it to autonomous operation when Dice clears threshold, and logs every
event for paper export. <a href="ITERATIVE_SEGMENTATION/">Pipeline &rarr;</a>
</div>

</div>

## Get started in two minutes

1. Read the [**Quick Start**](quick-start.md) to submit your first query.
2. Skim the [**Architecture**](architecture.md) to understand the two-loop engine.
3. Watch the [**Knowledge Graph**](knowledge-graph.md) grow after each run.

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) &mdash; the inspiration
- [MorphoSource](https://www.morphosource.org/) &mdash; 3D specimen data repository
- [3D Slicer](https://www.slicer.org/) &mdash; open-source medical image computing
- [SlicerMorph](https://slicermorph.github.io/) &mdash; 3D morphometrics for Slicer
