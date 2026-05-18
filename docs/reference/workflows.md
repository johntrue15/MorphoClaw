---
title: Workflows
description: Every GitHub Actions workflow shipped with AutoResearchClaw, what it does, where it runs, and what artifacts it produces.
---

# Workflows

AutoResearchClaw ships with a fleet of GitHub Actions workflows. They split
into four families: the main research pipeline, CI / quality, segmentation
training and validation, and issue automation.

## Research pipeline

| Workflow | Runs on | What it does |
|----------|---------|--------------|
| [`autoresearchclaw.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/autoresearchclaw.yml) | `self-hosted` (Mac mini) | Main agent run. Two-loop research engine, optional nnInteractive paint loop, optional integrity verifier. Publishes the knowledge-graph snapshot to `docs/data/`. |
| [`verify_research_run.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/verify_research_run.yml) | `ubuntu-latest` | Integrity verifier. Triggered after `autoresearchclaw.yml` and on demand via `/verify` issue comments. |
| [`parse_morphosource.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/parse_morphosource.yml) | `ubuntu-latest` | Legacy CSV comparison workflow. |

### `autoresearchclaw.yml` inputs

| Input | Default | Description |
|-------|---------|-------------|
| `research_topic` | (required) | Research goal or question |
| `research_depth` | `10` | Internal research cycles |
| `github_issues` | `3` | GitHub issue reports to create |
| `media_id` | &mdash; | Seed MorphoSource media ID |
| `media_list_id` | &mdash; | Batch seed via a MorphoSource media list |
| `continue_from` | &mdash; | Issue number to resume from |
| `openai_model` | `gpt-5.4` | OpenAI model |
| `enable_nninteractive` | `false` | Enable the LLM paint loop |
| `nninteractive_goal` | &mdash; | Target structure, e.g. `"Segment the cranial cavity"` |
| `nninteractive_max_steps` | `12` | Max LLM iterations |
| `run_integrity_verifier` | `true` | Chain the verifier after the run |

### Artifacts produced by `autoresearchclaw.yml`

- `research-report` &mdash; `research_report.json`, `research_report.md`, `research_agent.log`.
- `knowledge-graph` &mdash; the raw JSON / HTML from `~/.autoresearchclaw/graphs/`.
- `nninteractive-segmentations` &mdash; NIfTI labelmaps + previews when `enable_nninteractive` is on.

After the artifact upload step, a **Publish knowledge graph snapshot** step
runs `docs/scripts/merge_graphs.py` and commits the fresh snapshot to
`main`. That commit triggers the docs build, refreshing
[the live graph page](../knowledge-graph.md).

## Segmentation: training and validation

| Workflow | Runs on | What it does |
|----------|---------|--------------|
| [`bootstrap_nninteractive.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/bootstrap_nninteractive.yml) | `self-hosted` | One-time install of `nnInteractive` + weights into `~/.autoresearchclaw/nninteractive`. |
| [`nninteractive_compare.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/nninteractive_compare.yml) | `self-hosted` | End-to-end comparison of the paint loop vs a curated MorphoSource GT mesh. Computes Dice/IoU/Hausdorff. |
| [`iterative_segmentation_training.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/iterative_segmentation_training.yml) | `self-hosted` | Round-based student training (Round 0 nnInteractive → Round N student). Emits ledger + paper exports. |
| [`seg_train_live_chameleon.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/seg_train_live_chameleon.yml) | `self-hosted` | Tier-4 live test on the chameleon stapes pair. |
| [`seg_train_tests.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/seg_train_tests.yml) | `ubuntu-latest` | Tiers 1+2+3 unit / CLI / integration tests for the seg-train package. |
| [`slicer-integration.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/slicer-integration.yml) | `self-hosted` | SlicerMorph regression against the cached model. |
| [`verify-comparison.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/verify-comparison.yml) | `ubuntu-latest` | Verifies comparison-request issues against MorphoSource. |

## Issue automation

| Workflow | Runs on | What it does |
|----------|---------|--------------|
| [`on-request-opened.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/on-request-opened.yml) | `ubuntu-latest` | Reacts to new `query-request` issues: formats the query, runs the MorphoSource search, posts results. |
| [`on-admin-mention.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/on-admin-mention.yml) | `ubuntu-latest` | Handles admin-mention commands in issue comments. |
| [`issue-commands.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/issue-commands.yml) | `ubuntu-latest` | Processes slash-commands like `/verify`. |
| [`request-initial-comments.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/request-initial-comments.yml) | `ubuntu-latest` | Posts the initial onboarding comment on new request issues. |
| [`request-labeler.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/request-labeler.yml) | `ubuntu-latest` | Auto-labels request issues. |
| [`request-notify-admin.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/request-notify-admin.yml) | `ubuntu-latest` | Pings the admin on new requests. |
| [`labels.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/labels.yml) | `ubuntu-latest` | Syncs the label set defined in `.github/labels.yml`. |

## CI / quality

| Workflow | Runs on | What it does |
|----------|---------|--------------|
| [`tests.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/tests.yml) | `ubuntu-latest` | pytest suite. |
| [`code-quality.yml`](https://github.com/johntrue15/MorphoClaw/blob/main/.github/workflows/code-quality.yml) | `ubuntu-latest` | `black`, `ruff`, `bandit`, `yamllint`, and the agent tool-call smoke tests (`Tests/test_tool_calls.py`). |
| `docs.yml` | `ubuntu-latest` | Builds this MkDocs site and deploys to Pages. |

## Triggering a workflow

Most workflows are `workflow_dispatch` enabled, so you can run them from
**Actions → *workflow name* → Run workflow**. The ones triggered by issue
events (`on-request-opened`, `issue-commands`, …) fire automatically on the
relevant GitHub events.

See [**Quick Start**](../quick-start.md) for the recommended starting point.
