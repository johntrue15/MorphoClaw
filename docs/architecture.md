---
title: Architecture
description: How AutoResearchClaw's two-loop research engine, tool calls, and integrity verifier fit together.
---

# Architecture

AutoResearchClaw is built around a **two-loop research engine** that drives a
small set of tool calls. Every loop iteration writes to a shared memory and
to the knowledge graph; every research issue feeds an independent integrity
verifier.

## The two-loop engine

```mermaid
flowchart TB
    Start([Research topic + optional seed]) --> Inner

    subgraph Inner["Inner loop &mdash; research_depth cycles"]
        direction TB
        D[Decompose topic] --> S[Search MorphoSource]
        S --> E[Evaluate results]
        E --> M[Update memory<br/>queries, discoveries, dead ends]
        M --> N{Next direction?}
        N -->|yes| D
    end

    Inner --> Outer

    subgraph Outer["Outer loop &mdash; github_issues reports"]
        direction TB
        A[Aggregate findings] --> R[Synthesize narrative]
        R --> I[Open GitHub issue]
    end

    Outer --> End([Tracking issue + reports])
```

- **Inner loop** runs `research_depth` fast cycles. Each cycle: decompose the
  topic into smaller MorphoSource queries, search the API, evaluate the
  results, and grow memory (`queries tried`, `discoveries`, `dead ends`,
  `next directions`).
- **Outer loop** posts cumulative findings as `github_issues` GitHub issue
  reports at regular intervals. Memory carries between cycles, so later
  reports build on what earlier ones found.

The reference implementation lives in
[`.github/scripts/research_agent.py`](https://github.com/johntrue15/Metadata-to-Morphsource-compare/blob/main/.github/scripts/research_agent.py).

## Tool calls

```mermaid
flowchart LR
    Agent["research_agent"] --> Tools

    subgraph Tools["Tool calls"]
        direction TB
        T1["morphosource_api.search"]
        T2["morphosource_api_download"]
        T3["slicer_tool (3D Slicer / SlicerMorph)"]
        T4["literature_search<br/>(PubMed + Google Scholar)"]
        T5["knowledge_graph"]
        T6["ontology_search (UBERON)"]
        T7["nnInteractive paint loop"]
    end
```

Every tool is a plain Python module under
[`.github/scripts/`](https://github.com/johntrue15/Metadata-to-Morphsource-compare/tree/main/.github/scripts).
They are imported directly by the agent and also exposed to the LLM as
OpenAI function-calling schemas (see `chat_handler.TOOLS`).

## The knowledge graph

The graph is built incrementally as records come back from MorphoSource. Each
record fans out into the canonical entities and relations:

```mermaid
flowchart LR
    Media -->|BELONGS_TO| Specimen
    Media -->|DERIVED_FROM| Media2[Media (parent)]
    Specimen -->|HELD_BY| Institution
    Specimen -->|IS_TAXON| Taxon
    Taxon -->|HAS_RANK| RankedTaxon[Taxon (GBIF rank)]
    Media -->|CITED_IN| Paper
    MediaList -->|CONTAINS| Media
```

See [`.github/scripts/knowledge_graph.py`](https://github.com/johntrue15/Metadata-to-Morphsource-compare/blob/main/.github/scripts/knowledge_graph.py)
for the implementation. The same module renders the per-run interactive HTML
attached to each workflow artifact. The
[**Live knowledge graph**](knowledge-graph.md) page on this site loads the
cumulative JSON published after every run.

## Iterative segmentation

```mermaid
flowchart LR
    Pairs["Open (CT, mesh) pairs<br/>from MorphoSource"] --> R0
    R0["Round 0: nnInteractive paint loop<br/>on every pair"] --> Seed["Seed dataset"]
    Seed --> Train0["Train Student v0"]
    Train0 --> RN

    subgraph RN["Round n &geq; 1"]
        direction TB
        Inf["Student inference"]
        Route["ConfidenceRouter:<br/>accept / correct / reject"]
        Refit["Retrain"]
        Inf --> Route --> Refit
    end

    RN -->|Dice &geq; threshold (x2)| Grad["Graduate:<br/>student runs autonomously"]
    Grad --> Ledger["ExperimentLedger<br/>(paper export)"]
```

Full methodology in [**Iterative Segmentation**](ITERATIVE_SEGMENTATION.md).

## Integrity verifier (Plato's Cave)

After every research run, the verifier builds a trust DAG over the GitHub
issues produced by the run. Five MVP agents score each issue across six
per-metric dimensions, scores propagate parent → child via a *trust gate*,
and the verifier emits three release scores:
`scientific_validity`, `ai_training_validity`, `commercial_release_validity`.

```mermaid
flowchart LR
    Issues["GitHub issues<br/>from a research run"] --> Verifiers

    subgraph Verifiers["MVP verifier agents"]
        direction TB
        V1[Metadata]
        V2[File]
        V3[Lineage]
        V4[AI QC]
        V5[Expert]
    end

    Verifiers --> Scores["Per-metric scores (x6)"]
    Scores --> DAG["Trust-DAG propagation"]
    DAG --> Release{Release verdict}
    Release --> Label["issue label:<br/>integrity-verified |<br/>integrity-conditionally_verified |<br/>integrity-needs_review |<br/>integrity-rejected"]
```

Read the full methodology, ontology, and calibration plan in
[**Integrity Verifier**](INTEGRITY_VERIFIER.md).

## Self-hosted execution

The heavy lifting (3D Slicer, nnInteractive, specimen cache) runs on a
self-hosted Mac mini runner. See [**Workflows**](reference/workflows.md) for
the matrix of jobs and which ones target `self-hosted` vs `ubuntu-latest`.
