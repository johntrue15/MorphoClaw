# AutoResearchClaw

An autonomous MorphoSource research agent inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Runs on a self-hosted Mac mini, searches MorphoSource for 3D specimen data, downloads and analyzes specimens with 3D Slicer + SlicerMorph, and produces structured research reports as GitHub Issues.

## Architecture

```
Research Topic + Seed Media ID/List
        |
        v
  ┌─────────────────────────────────┐
  │  AutoResearchClaw Agent         │
  │  (research_agent.py)            │
  │                                 │
  │  Inner loop (research_depth):   │
  │    decompose → search →         │
  │    evaluate → build memory      │
  │                                 │
  │  Outer loop (github_issues):    │
  │    aggregate → create issue     │
  │                                 │
  │  Tool calls:                    │
  │    - MorphoSource API search    │
  │    - Specimen download          │
  │    - 3D Slicer analysis         │
  │    - Literature search          │
  │    - Knowledge graph            │
  │    - UBERON ontology lookup     │
  └─────────────────────────────────┘
        |
        v
  GitHub Issues + Dashboard + Knowledge Graph
```

## Quick Start

1. Go to **Actions** > **AutoResearchClaw Agent** > **Run workflow**
2. Enter your research topic
3. Set **research depth** (internal cycles, default 10) and **GitHub issues** (reports to create, default 3)
4. Optionally provide a **seed media ID** or **media list ID** from MorphoSource
5. Results post as GitHub Issues; detailed logs available as artifacts

## Workflow Dispatch Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `research_topic` | (required) | Research goal or question |
| `research_depth` | 10 | Number of internal research cycles |
| `github_issues` | 3 | Number of GitHub issues to create with findings |
| `media_id` | | MorphoSource media ID to seed research (e.g. `000769445`) |
| `media_list_id` | | MorphoSource media list ID for batch seeding (e.g. `000656244`) |
| `openai_model` | gpt-5.4 | OpenAI model to use |
| `run_integrity_verifier` | true | Chain the Plato's-Cave integrity verifier after the run |

## Required Secrets

Configure in **Settings > Secrets and variables > Actions**:

- `OPENAI_API_KEY` -- OpenAI API key for query decomposition, evaluation, and synthesis
- `MORPHOSOURCE_API_KEY` -- MorphoSource API key for specimen downloads

## Features

### Two-Loop Research Engine
- **Inner loop** runs `research_depth` fast cycles: decompose topic, search MorphoSource API, evaluate results, build memory
- **Outer loop** creates `github_issues` reports at regular intervals aggregating findings
- Memory accumulates across cycles: queries tried, discoveries, dead ends, next directions

### 3D Slicer + SlicerMorph Integration
- Downloads open-access mesh specimens from MorphoSource
- Runs headless 3D Slicer morphometric analysis (landmarks, curvature, volume, connectivity)
- Produces publication-quality screenshots from multiple viewpoints
- Specimen cache at `~/.autoresearchclaw/specimens/` avoids re-downloading

### 3-Layer Analysis Pipeline
- **Layer 1**: Automated morphometrics (dimensions, surface area, volume, curvature, PCA, landmarks)
- **Layer 2**: Literature-guided analysis (PubMed + Google Scholar search, LLM-designed measurement protocols)
- **Layer 3**: Multi-specimen comparison (Procrustes alignment, PCA, distance matrices)

### Knowledge Graph
- Builds connections between media, specimens, papers, institutions, and taxa
- Exports as JSON (Neo4j/Cytoscape compatible) and Mermaid diagrams
- Tracks parent-child media relationships, DOI citations, taxonomic hierarchy

### UBERON Ontology Search (DRAGON-AI)
- Expands anatomy terms using the UBERON cross-species anatomy ontology
- Maps terms like "skull" to synonyms: cranium, neurocranium, calvaria, braincase
- Improves MorphoSource search coverage

### Local Dashboard
- Flask app at `http://localhost:5001` on the Mac mini
- Live-updating event log with cycle-by-cycle progress
- Score trends, query history, knowledge graph stats

### MorphoDepot Integrity Verifier (Plato's Cave)
- Runs automatically after every AutoResearchClaw run (toggleable)
- Builds a Plato's-Cave-style trust DAG over the run's GitHub issues
- Five MVP verifier agents: metadata, file, lineage, AI QC, expert
- Six per-metric scores propagate parent → child via a "trust gate"
- Produces a MorphoDepot Integrity Report comment with three release
  scores: `scientific_validity`, `ai_training_validity`,
  `commercial_release_validity`
- Status badge labels the issue (`integrity-verified`,
  `integrity-conditionally_verified`, `integrity-needs_review`,
  `integrity-rejected`)
- Re-run any time by commenting `/verify` on a tracking issue
- See [`docs/INTEGRITY_VERIFIER.md`](docs/INTEGRITY_VERIFIER.md) for
  the full methodology, ontology, and calibration TODO

## Self-Hosted Runner

The research pipeline runs on a self-hosted Mac mini runner with:
- 3D Slicer 5.10.0 + SlicerMorph
- Anaconda Python 3.12
- Persistent specimen cache

Runner setup:
```bash
cd ~/actions-runner-morphosource
./config.sh --url https://github.com/USER/REPO --token TOKEN
./run.sh
```

## nnInteractive — Iterative LLM-Driven Segmentation

AutoResearchClaw can drive [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive)
to *iteratively segment* a 3D volume from MorphoSource. The workflow:

1. The agent downloads a CT/MRI volume for the active media_id.
2. `nninteractive_loop.py` opens an nnInteractive session and renders three
   orthogonal previews of the current (empty) mask.
3. A vision-capable LLM looks at the previews and emits ONE JSON action:
   `ADD_POINT` (positive/negative), `ADD_BBOX`, `RESET`, or `DONE`.
4. The action is applied; nnInteractive refines the mask; new previews
   are rendered.
5. Repeat until `DONE` or `nninteractive_max_steps`.

Outputs land in `~/.autoresearchclaw/specimens/media_<id>/nninteractive/`
as a NIfTI labelmap (`*_nni_labelmap.nii.gz`), a JSON summary, a Markdown
report, and the per-step PNG previews.

### One-time bootstrap on the runner

Run the **Bootstrap nnInteractive** workflow once (or whenever you want to
upgrade the backend):

> Actions → **Bootstrap nnInteractive** → Run workflow

Under the hood it calls `.github/scripts/install_nninteractive.sh`, which:
- creates a venv at `~/.autoresearchclaw/nninteractive`
- installs PyTorch (CUDA wheel on Linux+NVIDIA, default PyPI wheel on
  Apple Silicon for MPS/CPU support)
- installs `nnInteractive>=1.1`, `SimpleITK`, `huggingface_hub`
- pre-downloads the ~400 MB `nnInteractive_v1.0` weights from HuggingFace

Optionally it also installs the **SlicerNNInteractive** 3D Slicer extension
so a researcher can later open the labelmap, click around, and refine
prompts manually.

### Enable it on a research run

When dispatching the **AutoResearchClaw Agent** workflow, toggle:

| Input | Notes |
|-------|-------|
| `enable_nninteractive` | Turn on the paint loop for analyzed specimens |
| `nninteractive_goal` | Plain-English target, e.g. `"Segment the cranial cavity"` |
| `nninteractive_max_steps` | LLM iteration cap (default 12) |

The pipeline runs an idempotent bootstrap step before the agent starts,
then `research_agent._run_specimen_analysis` automatically passes
`nninteractive=True` into `slicer_tool.analyze_specimen`. Resulting
labelmaps and PNGs are uploaded as the `nninteractive-segmentations` artifact.

### GPU caveat

`nnInteractive`'s upstream docs recommend Linux/Windows + an NVIDIA GPU
(≥10 GB VRAM). On the Apple Silicon Mac mini runner the bootstrap script
falls back to **MPS** (and finally **CPU**) — segmentation works but is
slower per prompt. If you have a GPU box on the network you can point
`NNINTERACTIVE_DEVICE` and the venv at it, or run the
`nninteractive-slicer-server` there and use the SlicerNNInteractive plugin
manually for visual review.

### Standalone CLI

```bash
# Activate the dedicated venv
source ~/.autoresearchclaw/nninteractive/bin/activate

# One-shot iterative paint on a local volume
python .github/scripts/nninteractive_loop.py \
    --input /path/to/volume.nii.gz \
    --goal  "Segment the heart" \
    --media-id 000656244 \
    --output-dir /tmp/nni_loop \
    --max-steps 10

# Or via slicer_tool (downloads from MorphoSource first)
python .github/scripts/slicer_tool.py 000769445 \
    --nninteractive --nni-goal "Segment the cranial cavity"
```

### Comparison Harness — nnInteractive vs Curated MorphoSource GT

Validate the paint loop end-to-end by comparing it against a human-curated
segmented derivative of the same MorphoSource specimen:

> Actions → **nnInteractive Comparison Test** → Run workflow

You can either supply both media IDs explicitly or let the harness auto-pick
the first viable open-download CT↔mesh pair:

| Input | Notes |
|-------|-------|
| `ct_media_id` | MorphoSource ID of the unsegmented CT volume (open download) |
| `gt_media_id` | MorphoSource ID of the segmented mesh derivative (open, same specimen) |
| `auto_discover` | If true and the IDs are blank, run `find_segmentation_pairs.py` to pick one |
| `discover_query` | Search query for auto-discover (default: `"skull mesh"`) |
| `require_taxonomy` | Optional taxonomy filter for auto-discover, e.g. `"Chamaeleo"` |
| `goal` | Plain-English target, e.g. `"Segment the cranial bone"` |
| `max_steps` | LLM iteration cap (default 12) |

Pipeline:

1. `find_segmentation_pairs.py` (only when auto-discovering) finds open
   CT ↔ mesh pairs that share the same `physical_object_id`.
2. `morphosource_api_download.download_media` fetches both media bundles.
3. `voxelize_mesh_in_slicer.py` runs headless 3D Slicer to rasterize the GT
   mesh onto the **CT's exact voxel grid** via `vtkPolyDataToImageStencil`,
   producing `gt_voxelized.nii.gz`.
4. `nninteractive_loop.py` (in the dedicated venv) runs the LLM "paint" loop
   on the CT and writes a prediction labelmap.
5. `segmentation_metrics.py` computes Dice / IoU / precision / recall /
   volume agreement / Hausdorff (max + 95-pct) / mean surface distance /
   centroid distance, and renders a 3×3 axial/coronal/sagittal overlay
   panel showing the volume, GT (blue), and prediction (orange).
6. A Markdown report (`report.md`) is written and posted as a GitHub issue
   with the `nninteractive`, `segmentation-test` labels.

All artifacts are uploaded as the `nninteractive-comparison` artifact so
you can download the labelmaps for inspection in 3D Slicer.

#### Standalone

```bash
# Discover a candidate pair without running the full comparison
python .github/scripts/find_segmentation_pairs.py \
    --query "skull mesh" --max-pairs 5 \
    --output candidate_pairs.json

# Run the full comparison with explicit IDs
python .github/scripts/nninteractive_compare.py \
    --ct-media-id 000656244 \
    --gt-media-id 000656245 \
    --goal "Segment the cranial bone" \
    --output-dir /tmp/nni_compare
```

> **Caveat:** The "ground truth" here is a human-curated MorphoSource
> derivative — not a perfect oracle. Surface meshes may not be watertight
> (the voxelizer applies `vtkFillHolesFilter`) and may exclude internal
> structures. Sub-perfect Dice doesn't necessarily mean nnInteractive is
> wrong; the overlay panel is the source of truth for visual judgment.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) install lint/test tooling – same versions CI uses
pip install -r requirements-dev.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run locally
cd .github/scripts
python research_agent.py "Your research topic" \
  --research-depth 10 \
  --github-issues 1 \
  --media-list 000656244

# Start dashboard
python dashboard.py
# Open http://localhost:5001
```

### Linting locally (matches CI)

The `code-quality.yml` workflow runs the same tools listed in
`requirements-dev.txt`.  To reproduce its checks locally:

```bash
# Format + import sort
black --line-length=100 .
ruff check --fix .

# Strict subset (always-blocking) – mirrors the CI gate
ruff check --select=E9,F63,F7,F82,F821,F823,B006,B008,B018 .

# Security scan (HIGH severity is the CI gate)
bandit -r . -c pyproject.toml \
  --exclude ./Tests,./tests,./.venv,./venv,./data \
  --severity-level high --confidence-level medium

# Workflow + YAML lint
yamllint .github/
```

Diff-only enforcement (what the PR gate uses) can be reproduced with
`ruff check $(git diff --name-only origin/main...HEAD -- '*.py')` and
the equivalent `black --check` invocation.  Pre-existing technical
debt is intentionally tolerated by the full-repo run; new code must
be clean.

## Project Structure

```
.github/
  scripts/
    _helpers.py              # Shared utilities (dotenv, LLM, constants)
    research_agent.py        # Main autonomous research agent
    slicer_tool.py           # 3D Slicer download + analysis + nnInteractive entry-point
    morphosource_api_download.py  # MorphoSource file downloader
    knowledge_graph.py       # Media-specimen-paper graph builder
    ontology_search.py       # UBERON ontology term expansion
    citation_extractor.py    # DOI/paper extraction from records
    literature_search.py     # PubMed + Google Scholar search
    slicer_layer2.py         # Literature-guided deep analysis
    slicer_layer3.py         # Multi-specimen comparison
    dashboard.py             # Local Flask monitoring dashboard
    program.md               # Research strategy (Karpathy-style)
    query_formatter.py       # Natural language to API URL converter
    morphosource_api.py      # MorphoSource API search handler
    integrity_graph.py       # Plato's-Cave DAG, scoring, trust propagation
    integrity_verifiers.py   # Five MVP verifier agents
    verify_research_run.py   # Post-run integrity verification entry point
    install_nninteractive.sh        # Bootstrap nnInteractive venv + weights on the runner
    slicer_install_nninteractive.py # Installs SlicerNNInteractive extension into 3D Slicer
    nninteractive_segment.py        # Headless nnInteractiveInferenceSession wrapper
    nninteractive_loop.py           # LLM-in-the-loop "paint" segmentation driver
    nninteractive_compare.py        # Compare paint-loop output to a MorphoSource GT
    voxelize_mesh_in_slicer.py      # Slicer: rasterize a mesh onto a CT's voxel grid
    segmentation_metrics.py         # Dice/IoU/Hausdorff + overlay panel rendering
    find_segmentation_pairs.py      # Discover open-download CT ↔ mesh pairs on MorphoSource
  workflows/
    autoresearchclaw.yml         # Main research workflow (nnInteractive toggle inside)
    bootstrap_nninteractive.yml  # One-time setup of nnInteractive on the runner
    nninteractive_compare.yml    # Test the paint loop vs a MorphoSource GT segmentation
    verify_research_run.yml      # Integrity verifier (chained after run + on /verify)
    tests.yml                    # CI tests
    code-quality.yml             # Linting + agent tool-call smoke tests (cloud)
    slicer-integration.yml       # SlicerMorph cached-model regression (self-hosted)
    parse_morphosource.yml       # CSV comparison workflow
```

### Agent tool-call + SlicerMorph regression tests

Two test suites guard the agent's external surface:

- `Tests/test_tool_calls.py` — imports every Python "tool" the agent
  invokes, asserts its signature against the contract the agent
  relies on, validates the OpenAI function-tool schemas in
  `chat_handler.TOOLS`, and exercises each tool's offline / no-API-key
  path so a network outage produces a structured error dict, never an
  exception. Runs on every PR via the `tool-tests` job in
  `code-quality.yml`.
- `Tests/test_slicer_cached_model.py` — three tiers:
  1. always-on validation of the deterministic ASCII PLY fixture at
     `Tests/fixtures/tetrahedron.ply`;
  2. discovery test for the real MorphoSource cache under
     `data/morphosource-download-*/**/*.ply` (skipped on cloud
     runners where the 112 MB binary is not present);
  3. headless 3D Slicer integration that spawns
     `Slicer --python-script .github/scripts/slicer_load_test.py`
     against the cached model and asserts on vertex count + bounds
     (skipped unless `SLICER_BIN` exists; promoted to a hard
     requirement on the self-hosted runner via
     `SLICER_REQUIRE_INTEGRATION=1`).

Tier 1 + 2 run in cloud CI; Tier 3 runs on the self-hosted Mac mini
via `slicer-integration.yml` (triggered on `slicer_*` script changes
or on demand).

## API Client Migration (morphosource_client)

All MorphoSource HTTP calls now go through a unified client module
(`.github/scripts/morphosource_client.py`). This replaces ad-hoc
`requests.get` calls scattered across `morphosource_api.py`,
`chat_handler.py`, and `research_agent.py`.

### What changed

| Before | After |
|--------|-------|
| `_extract_result_count` returned `len(media_list)` — the current page size | `_extract_counts` returns `(total_count, returned_count)` from `pages.total_count` |
| `chat_handler.search_morphosource` did raw `requests.get` | Delegates to `MorphoSourceClient.search_media` |
| `research_agent.fetch_seed_media` / `_fetch_media_list` had inline HTTP | Delegates to `MorphoSourceClient.get_media` / `search_media` |
| No retry logic on transient 429/5xx errors | Exponential-backoff retry with configurable `max_retries` |
| No standard response contract | Every search returns `SearchResponse(returned_count, total_count, items, ...)` |

### Breaking changes

- `summary["count"]` in `morphosource_api.search_morphosource` now reports the
  **repository-wide total** (from `pages.total_count`), not the number of items
  on the current page. Downstream code that treated `count` as a page-size
  estimate will now see the correct total.
- `summary` also includes `returned_count` and `total_count` for callers that
  need both values.

### Running integration tests

```bash
MORPHOSOURCE_LIVE_TESTS=1 pytest Tests/test_morphosource_client.py -k Live -v
```

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- Autonomous AI research experiments
- [MorphoSource](https://www.morphosource.org/) -- 3D data repository for biological specimens
- [3D Slicer](https://www.slicer.org/) -- Open-source medical image computing
- [SlicerMorph](https://slicermorph.github.io/) -- 3D morphometrics for Slicer

## License

MIT License -- see [LICENSE](LICENSE).
