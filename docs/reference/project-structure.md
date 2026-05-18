---
title: Project structure
description: Where to find every script, package, workflow, and test in the AutoResearchClaw repository.
---

# Project structure

```text
.github/
  scripts/
    _helpers.py                    # Shared utilities (dotenv, LLM, constants)
    research_agent.py              # Main autonomous research agent
    slicer_tool.py                 # 3D Slicer download + analysis + nnInteractive entry
    morphosource_api.py            # MorphoSource search handler
    morphosource_api_download.py   # MorphoSource file downloader
    morphosource_client.py         # Unified HTTP client (retries, contracts)
    knowledge_graph.py             # Media-specimen-paper graph builder + exporters
    ontology_search.py             # UBERON ontology term expansion
    citation_extractor.py          # DOI / paper extraction
    literature_search.py           # PubMed + Google Scholar search
    slicer_layer2.py               # Literature-guided analysis
    slicer_layer3.py               # Multi-specimen comparison
    dashboard.py                   # Local Flask monitoring dashboard
    query_formatter.py             # Natural language → API URL
    integrity_graph.py             # Plato's-Cave DAG and scoring
    integrity_verifiers.py         # Five MVP verifier agents
    verify_research_run.py         # Post-run integrity verification entry
    install_nninteractive.sh       # Bootstrap nnInteractive venv + weights
    slicer_install_nninteractive.py
    nninteractive_segment.py       # Headless nnInteractiveInferenceSession wrapper
    nninteractive_loop.py          # LLM-in-the-loop "paint" segmentation driver
    nninteractive_compare.py       # Paint-loop output vs MorphoSource GT
    voxelize_mesh_in_slicer.py     # GT mesh → labelmap via headless Slicer (legacy)
    voxelize_mesh_vtk.py           # GT mesh → labelmap via pure-Python VTK
    crop_around_mesh.py            # SimpleITK CT crop to mesh bbox + margin
    dicom_to_nifti.py              # SimpleITK DICOM-series → .nii.gz
    test_chameleon_stapes.sh       # One-shot smoke test
    patch_runner_aqua_session.sh   # Add LimitLoadToSessionType=Aqua to the runner agent
    segmentation_metrics.py        # Dice/IoU/Hausdorff + overlay panel
    find_segmentation_pairs.py     # Discover open-download CT ↔ mesh pairs
    install_seg_train_extras.sh    # Add MONAI to the nnInteractive venv
    chat_handler.py                # OpenAI function-calling driver
    chatgpt_processor.py           # Issue-driven query processor
    grade_response.py              # Multi-response grading
    program.md                     # Karpathy-style research strategy
  workflows/
    autoresearchclaw.yml           # Main research workflow
    bootstrap_nninteractive.yml    # Set up nnInteractive on the runner
    nninteractive_compare.yml      # Validate paint loop vs MorphoSource GT
    iterative_segmentation_training.yml  # Round-based student training
    seg_train_live_chameleon.yml   # Tier-4 live test
    seg_train_tests.yml            # Tiers 1+2+3 CI tests
    verify_research_run.yml        # Integrity verifier
    tests.yml                      # pytest suite
    code-quality.yml               # Linting + tool-call smoke tests
    slicer-integration.yml         # SlicerMorph regression
    parse_morphosource.yml         # Legacy CSV comparison
    on-request-opened.yml          # Query-request handler
    on-admin-mention.yml           # Admin-mention handler
    issue-commands.yml             # Slash-command router (e.g. /verify)
    request-initial-comments.yml
    request-labeler.yml
    request-notify-admin.yml
    labels.yml
    docs.yml                       # Builds and deploys this docs site

metadata_to_morphsource/
  seg_train/
    experiment_ledger.py           # Append-only JSONL+CSV log
    dataset.py                     # Manifest + train/val/holdout splits
    student_model.py               # MONAI 3D U-Net trainer + inference
    pseudo_label_generator.py      # Wraps the nnInteractive paint loop
    confidence_router.py           # Student vs nnInteractive dispatch
    iterative_trainer.py           # Round orchestrator + graduation rule
    paper_export.py                # CSVs / plots / Markdown for publication
    cli.py                         # `python -m metadata_to_morphsource.seg_train ...`

docs/
  index.md                         # Landing page
  quick-start.md                   # Two-minute onboarding
  query.md                         # In-site query submission form
  knowledge-graph.md               # Live KG viewer
  architecture.md                  # Two-loop engine + Mermaid diagrams
  ITERATIVE_SEGMENTATION.md        # Methodology for the student loop
  INTEGRITY_VERIFIER.md            # Plato's-Cave verifier
  ISSUE_AUTOMATION.md              # Issue automation
  QUERY_SYSTEM_GUIDE.md            # Query pipeline overview
  QUERY_SUBMISSION_GUIDE.md        # Submission protocol
  MULTI_RESPONSE_GRADING.md        # Multi-response grading
  reference/
    cli.md
    workflows.md
    secrets.md
    project-structure.md           # (this file)
  assets/
    css/extra.css
    css/knowledge-graph.css
    js/knowledge-graph.js
    img/logo.svg
    img/favicon.svg
  data/
    knowledge_graph.json           # Cumulative snapshot (auto-managed)
    runs/_manifest.json            # Per-run history (auto-managed)
    runs/<timestamp>.json          # Per-run snapshots
  scripts/
    merge_graphs.py                # Snapshot publisher
  main.py                          # mkdocs-macros hook (kg_stats() etc.)
  overrides/main.html              # Material theme override

Tests/
  test_tool_calls.py               # Tool-import + schema + offline-path tests
  test_slicer_cached_model.py      # 3D Slicer regression (3 tiers)
  test_seg_train_*.py              # seg-train unit / integration / paper export tests
  smoke_seg_train.sh               # Integration smoke test
  test_chameleon_stapes_iterative.sh
```

For the canonical, code-truth listing, browse the
[repository tree on GitHub](https://github.com/johntrue15/MorphoClaw).
