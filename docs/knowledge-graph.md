---
title: Live knowledge graph
description: Interactive, auto-updated graph of MorphoSource media, specimens, papers, institutions, and taxa discovered by AutoResearchClaw.
hide:
  - toc
---

# Live knowledge graph

The graph below visualises every MorphoSource record, specimen, paper, taxon,
and institution that AutoResearchClaw has touched. It refreshes automatically
on every push to `main` after the
[AutoResearchClaw Agent workflow](https://github.com/johntrue15/MorphoClaw/actions/workflows/autoresearchclaw.yml)
publishes a new snapshot.

<div id="kg-app" class="kg-app">
  <aside class="kg-sidebar" id="kgSidebar">
    <div class="kg-section">
      <div class="kg-section-title">Snapshot</div>
      <select id="kgRunPicker" class="kg-select" aria-label="Select snapshot"></select>
      <div class="kg-meta" id="kgMeta">Loading&hellip;</div>
    </div>

    <div class="kg-section">
      <div class="kg-section-title">Search</div>
      <input
        type="search"
        id="kgSearch"
        class="kg-input"
        placeholder="Search labels, ids, taxa&hellip;"
        autocomplete="off"
      />
    </div>

    <div class="kg-section">
      <div class="kg-section-title">Node types</div>
      <div id="kgTypeFilters" class="kg-checkbox-group" role="group" aria-label="Toggle node types"></div>
    </div>

    <div class="kg-section">
      <div class="kg-section-title">Relations</div>
      <div id="kgRelationFilters" class="kg-checkbox-group" role="group" aria-label="Toggle relation types"></div>
    </div>

    <div class="kg-section">
      <div class="kg-section-title">Min degree
        <span class="kg-pill" id="kgDegreeValue">0</span>
      </div>
      <input
        type="range"
        id="kgDegree"
        class="kg-range"
        min="0"
        max="10"
        step="1"
        value="0"
        aria-label="Hide nodes with fewer than N connections"
      />
    </div>

    <div class="kg-section">
      <button id="kgFitBtn" class="kg-btn">Fit to screen</button>
      <button id="kgResetBtn" class="kg-btn kg-btn-secondary">Reset filters</button>
    </div>
  </aside>

  <div class="kg-stage">
    <div id="kgGraph" class="kg-graph" role="img" aria-label="Knowledge graph network visualization"></div>

    <div id="kgEmpty" class="kg-empty" hidden>
      <h2>No runs published yet</h2>
      <p>
        Kick off the
        <a href="https://github.com/johntrue15/MorphoClaw/actions/workflows/autoresearchclaw.yml" target="_blank" rel="noopener">AutoResearchClaw Agent workflow</a>
        and the graph will populate automatically on the next push to <code>main</code>.
      </p>
    </div>

    <div id="kgLegend" class="kg-legend" aria-label="Node type legend"></div>

    <aside id="kgDetails" class="kg-details" hidden>
      <button class="kg-close" id="kgDetailsClose" aria-label="Close details">&times;</button>
      <h3 id="kgDetailsTitle"></h3>
      <div class="kg-details-type" id="kgDetailsType"></div>
      <dl id="kgDetailsProps" class="kg-details-props"></dl>
      <div class="kg-details-actions" id="kgDetailsActions"></div>
    </aside>
  </div>
</div>

<!--
  vis-network is loaded by docs/assets/js/knowledge-graph.js itself.
  The script computes its own URL (via document.currentScript.src) and
  loads the locally-vendored copy at
  docs/assets/vendor/vis-network/vis-network.min.js, falling back to a
  public CDN only if the vendored copy is unreachable. This means the
  viewer works regardless of the docs site's base URL (gh-pages subpath,
  preview deployments, offline mirrors, etc.).
-->

!!! tip "How the graph stays live"
    After every AutoResearchClaw run, the workflow runs `docs/scripts/merge_graphs.py`,
    which copies the latest `~/.autoresearchclaw/graphs/*.json` into `docs/data/runs/`
    and updates the cumulative `docs/data/knowledge_graph.json`. A commit to `main`
    triggers the docs workflow and the new graph appears here automatically.

## Schema

Nodes use the same colours and shapes as the
[`KnowledgeGraph.export_html`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/scripts/knowledge_graph.py)
helper, so the view here matches the per-run interactive HTML attached to each workflow artifact.

| Node type | Meaning | Connects via |
|-----------|---------|--------------|
| Media | A MorphoSource media record (CT series, mesh, image, etc.) | `BELONGS_TO`, `DERIVED_FROM`, `CITED_IN` |
| Specimen | A physical specimen object | `HELD_BY`, `IS_TAXON` |
| Paper | DOI / citation extracted from records | `CITED_IN` |
| Institution | Organisation that holds the specimen | `HELD_BY` |
| Taxon | Linnaean taxon (incl. GBIF hierarchy) | `IS_TAXON`, `HAS_RANK` |
| MediaList | A MorphoSource curated list seed | `CONTAINS` |

For the JSON shape itself, see [`docs/data/knowledge_graph.json`](https://github.com/johntrue15/MorphoClaw/blob/main/docs/data/knowledge_graph.json) &mdash;
it is Neo4j / Cytoscape compatible.
