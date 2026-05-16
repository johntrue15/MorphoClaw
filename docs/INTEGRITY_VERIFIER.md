# MorphoDepot Integrity Verifier — Plato's Cave for MorphoSource

The integrity verifier runs **after** an AutoResearchClaw research program
to score the data the agent wrote into GitHub Issues, before a human or a
downstream consumer treats those issues as authoritative.

The design follows the Plato's-Cave methodology (Verma et al.) but
replaces the paper's research-paper ontology with a MorphoDepot
ontology built around MorphoSource records, AI segmentations, and
expert review.

## Why

A run of `research_agent.py` produces:

- 1 parent tracking issue
- N child issues with discoveries, queries, and media references
- A `research_report.json` artefact
- An optional 3D-Slicer specimen analysis dump

None of those outputs come with provenance or trust metadata.  The
verifier closes that gap by:

1. **Decomposing** the run into a directed acyclic graph (DAG)
2. **Scoring** every node on six dimensions
3. **Propagating** trust from parents to children (the Plato's-Cave
   "trust gate")
4. **Deciding** scientific / AI-training / commercial release fitness
5. **Surfacing** the verdict to the maintainer as a labelled GitHub
   comment plus a JSON artefact

Crucially, the verifier never claims to *replace* expert review.  It
produces evidence that helps a curator focus their attention.

## Architecture

```
                 ┌────────────────────────────────────────┐
                 │  AutoResearchClaw run (issue + comments)│
                 └─────────────────┬──────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │  verify_research_run.py                          │
        │   1. Fetch parent + child issues                 │
        │   2. Extract media IDs / discoveries / queries   │
        │   3. Build IntegrityGraph                        │
        │      ├─ MetadataVerifier   (MorphoSource API)    │
        │      ├─ FileVerifier       (voxel/spacing/dims)  │
        │      ├─ LineageVerifier    (parent media chain)  │
        │      ├─ AIQCVerifier       (LLM plausibility)    │
        │      └─ ExpertVerifier     (issue reactions)     │
        │   4. propagate(parent_weight=0.6)                │
        │   5. release_decision()                          │
        │   6. Post Markdown report + label issue          │
        └──────────────────────────────────────────────────┘
```

Files:

| File | Purpose |
|------|---------|
| `.github/scripts/integrity_graph.py` | DAG data structures, trust propagation, release decision, Markdown rendering |
| `.github/scripts/integrity_verifiers.py` | Five MVP verifier agents |
| `.github/scripts/integrity_cache.py` | Polite, persistent MorphoSource record cache (memory + disk + circuit breaker) |
| `.github/scripts/verify_research_run.py` | CLI entry point + GitHub orchestration |
| `.github/workflows/verify_research_run.yml` | Workflow (`workflow_dispatch`, `workflow_run`, `issue_comment`) |
| `Tests/test_integrity_*.py` | Offline pytest coverage (66 tests) |

## DAG ontology

The MorphoDepot ontology used by the DAG is intentionally small.  Only
roles in this list are accepted:

| Role             | Meaning                                                                |
|------------------|------------------------------------------------------------------------|
| `ResearchTopic`  | The user's research goal (the "hypothesis")                            |
| `SourceRecord`   | A MorphoSource media / physical-object record                          |
| `Specimen`       | The physical object behind a media record                              |
| `MediaList`      | A MorphoSource media list (batch seed)                                 |
| `Query`          | A query the agent issued against MorphoSource                          |
| `SearchResult`   | The aggregated result of a query                                       |
| `Discovery`      | An LLM-extracted discovery / claim                                     |
| `Citation`       | A DOI / paper extracted from `cite_as` / `description`                 |
| `AISegmentation` | A 3D-Slicer / MONAI / nnInteractive output (reserved for layer 2/3)    |
| `Measurement`    | A quantitative measurement (reserved for SlicerMorph integration)      |
| `ExpertReview`   | Human accept / edit / reject decision                                  |
| `ReleaseDecision`| (Reserved)                                                              |

Adding a role is a one-line change in `ROLES` plus a verifier.

## Six metric dimensions

Every node carries six per-metric scores in `[0.0, 1.0]`:

| Metric              | What it measures (MorphoDepot-specific gloss)                    |
|---------------------|------------------------------------------------------------------|
| `credibility`       | Does the artefact come from an authoritative source?             |
| `relevance`         | Does it match the requested topic / specimen / anatomy / task?   |
| `evidence_strength` | Concrete data: voxel size, slice count, hits, file integrity     |
| `method_rigor`      | Was the method (search, AI, segmentation) recorded?              |
| `reproducibility`   | Can the result be regenerated (hash, model version, prompt log)? |
| `authority_support` | Reviewed by the right expert / cited / institutional permission  |

The defaults in `DEFAULT_METRICS` are 0.5 — the verifier is conservative
when it cannot judge.

## Trust propagation

Trust propagation implements the paper's "trust gate" pattern.  After
all nodes are added, `IntegrityGraph.propagate(parent_weight=w)` walks
the DAG in topological order and for every non-root node computes:

```
gate           = min(parent.overall for parent in node.parents)
multiplier     = (1 - w) + w * gate
propagated[k]  = clamp(node.metrics[k] * multiplier)
```

`w = 0` disables propagation; `w = 1` makes a child entirely
subordinate to its weakest parent.  The default `w = 0.6` mirrors the
Plato's-Cave guidance that gates should *reduce*, not *replace*, child
trust.  Worst-link semantics (`min` over parents) matches morphology
lineage logic where a single bad parent is disqualifying.

## Release decision

The release decision blends the propagated metrics, averaged over
all non-Topic nodes, into three weighted scores:

| Dimension                       | Weights (metric → weight)                                                   |
|---------------------------------|-----------------------------------------------------------------------------|
| `scientific_validity`           | credibility:1, evidence_strength:1, method_rigor:1, relevance:0.5           |
| `ai_training_validity`          | evidence_strength:1, method_rigor:1, reproducibility:1.5, credibility:0.5   |
| `commercial_release_validity`   | credibility:1, evidence_strength:1, authority_support:2, method_rigor:0.5   |

Status thresholds (applied to the *minimum* of the three scores):

| Threshold | Status                     | Emoji |
|----------:|----------------------------|:-----:|
|     ≥0.85 | `verified`                 | 🟢    |
|     ≥0.65 | `conditionally_verified`   | 🟡    |
|     ≥0.45 | `needs_review`             | 🟠    |
|      <0.45| `rejected`                 | 🔴    |

Tunable in `integrity_graph.RELEASE_WEIGHTS` and `STATUS_THRESHOLDS`.

## Triggering

Three ways to trigger the verifier:

### 1. Automatically after every AutoResearchClaw run (default)

The AutoResearchClaw workflow now ends with a `trigger-integrity-verifier`
job that dispatches `verify_research_run.yml` for the tracking issue.
Toggle off via the workflow's `run_integrity_verifier` input.

### 2. Manual `workflow_dispatch`

```
Actions → Verify Research Run → Run workflow
  issue_number = 223
  use_llm      = true
  dry_run      = false
```

### 3. `/verify` comment on a tracking issue

A maintainer can post `/verify` on any issue.  Optional arguments:

```
/verify 224,225           # explicit child issues
/verify --no-llm          # skip the LLM AI-QC call
/verify --dry-run         # print the report without commenting
```

## Local usage

```bash
# Verify a live issue (needs $GITHUB_TOKEN)
python .github/scripts/verify_research_run.py \
    --issue 223 \
    --repo  my-org/my-repo

# Re-score a saved research_report.json (no GitHub, no network)
python .github/scripts/verify_research_run.py \
    --report .github/scripts/research_report.json \
    --no-llm --no-network --dry-run
```

Outputs land in `~/.autoresearchclaw/integrity/` (timestamped pairs of
`*.json` and `*.md`).  Override the home directory with
`AUTORESEARCHCLAW_HOME=/path/to/dir`.

## Being a good citizen of the MorphoSource API

The verifier hits the public MorphoSource API for every distinct media
ID it sees.  An early version fanned three independent calls per media
ID and used the global 30 s timeout × 3 retries — a single ``/verify``
run could keep a slow MorphoSource server pinned for half an hour.
The current implementation is deliberately conservative:

| Knob | Default | Override (CLI / env) | Notes |
|------|---------|----------------------|-------|
| Per-call timeout | 10 s | `--api-timeout` / `VERIFIER_API_TIMEOUT` | down from the 30 s used by `research_agent` |
| Retries per call | 1 | `--api-retries` / `VERIFIER_API_RETRIES` | one retry, not three |
| Min delay between calls | 0.5 s | `--api-min-delay` / `VERIFIER_API_MIN_DELAY` | enforced by `RecordCache` |
| Circuit-breaker threshold | 3 consecutive failures | `--circuit-breaker` / `VERIFIER_CIRCUIT_BREAKER` | once open, no more network calls this run |
| Disk-cache TTL | 7 days | `--cache-ttl-days` / `VERIFIER_CACHE_TTL_DAYS` | `0` disables expiry |
| Disk cache | enabled | `--no-cache` | persistent under `~/.autoresearchclaw/integrity/cache/` |
| Max distinct media IDs | 10 | `--max-media` | hard cap to keep total traffic bounded |
| Network entirely off | — | `--no-network` | falls back to bare-id scoring |

How the cache eliminates duplicate fetches:

1. `MetadataVerifier` no longer calls the API at all — it scores the
   record the cache already has (looking for ≥2 substantive fields to
   call it "resolved").
2. `LineageVerifier` routes its parent-media lookup through the same
   `RecordCache`, so the parent fetch is reused if any other path
   already requested it.
3. Negative results are memoized for the lifetime of the run — if
   `media/004088332` times out once, the verifier never re-asks for
   it during the same `/verify`.
4. After the configured number of consecutive failures the cache
   opens its circuit breaker, refuses further outbound calls, and
   surfaces the fact in both the JSON artefact and the GitHub
   comment so a maintainer can re-run later.

The Markdown report appended to the GitHub issue includes a small
**Cache & API politeness** block summarising memory hits, disk hits,
network calls, and breaker state.  A breaker-tripped run is also
called out with a ⚠️ banner so the reader knows scores were partly
based on cache-only data.

## Exit codes

The CLI returns non-zero only when the integrity status is bad enough
that a downstream automation should *refuse* to release the data:

| Status                     | Exit code |
|----------------------------|----------:|
| `verified`                 |        0  |
| `conditionally_verified`   |        0  |
| `needs_review`             |        1  |
| `rejected`                 |        2  |

The GitHub Actions job intentionally only fails on exit > 1 so that
`needs_review` runs still post a comment and surface labels.

## Calibration TODO

Per the Plato's-Cave paper's own caveat, the per-metric scoring weights
are not yet calibrated against expert-reviewed morphology datasets.
The recommended path:

1. Collect a corpus of `verify_research_run` outputs with expert
   accept/reject labels (the `ExpertVerifier` already records the
   reviewer's vote).
2. Fit logistic-regression weights for each release dimension against
   the expert ground truth.
3. Land the new `RELEASE_WEIGHTS` in `integrity_graph.py` with a
   changelog entry.

Until then, treat the scores as an *attention-prioritisation* tool, not
as an autonomous release gate.

## Related work

- Verma et al., *Plato's Cave: A Trust-Gated DAG for Research Verification*
- MorphoSource [API documentation](https://www.morphosource.org/api)
- `knowledge_graph.py` — sister module that builds the *factual* graph
  of MorphoSource records (this verifier builds the *trust* graph)
