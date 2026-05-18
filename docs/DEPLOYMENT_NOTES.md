---
title: Deployment notes
description: One-time setup for the MkDocs docs site and live knowledge graph publishing.
---

# Deployment notes

This page captures the **one-time** maintainer steps needed to bring the
new MkDocs documentation site &mdash; including the live knowledge graph
&mdash; online. The actual content changes are committed; the steps below
only have to happen once, on this repository's GitHub settings.

!!! warning "Required first step: switch the Pages source"
    Until you flip **Settings → Pages → Source** from *Deploy from a branch*
    to **GitHub Actions**, GitHub's default Jekyll builder
    (`actions/jekyll-build-pages`) will continue to run on every push.

    A transitional [`_config.yml`](https://github.com/johntrue15/Metadata-to-Morphsource-compare/blob/main/_config.yml)
    is committed at the repo root that tells Jekyll to **skip** everything in
    `docs/`, `mkdocs.yml`, `requirements*.txt`, `metadata_to_morphsource/`,
    and friends, so the legacy Jekyll build no longer chokes on MkDocs Jinja
    macros (`{% set %}`, `{{ kg_stats() }}`). Without it, you would see:

    ```
    Liquid Exception: Liquid syntax error (line 34): Unknown tag 'set' in docs/index.md
    ```

    A `.nojekyll` file at the repo root is also committed so the output (once
    Pages serves it) is not post-processed by Jekyll. But neither file stops
    the Jekyll *build* action from being auto-triggered &mdash; only flipping
    the Pages source to **GitHub Actions** does that. Both shims become no-ops
    after the switch.

## 1. Switch the GitHub Pages source (REQUIRED)

The repository was previously served from **Deploy from a branch → `main` /docs**
(the legacy `docs/index.html` query form). The new docs site is built by
the [Docs workflow](https://github.com/johntrue15/Metadata-to-Morphsource-compare/blob/main/.github/workflows/docs.yml)
and deployed via `actions/deploy-pages`, so the source must change to
**GitHub Actions**.

1. Open **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Save.

The next push to `main` (or a manual run of **Actions → Docs → Run workflow**)
will build the MkDocs site and publish it at
<https://johntrue15.github.io/Metadata-to-Morphsource-compare/>.

> The legacy URL <https://johntrue15.github.io/Metadata-to-Morphsource-compare/query.html>
> still works &mdash; the original HTML form is preserved verbatim and copied
> through by MkDocs as a static asset.

## 2. (Already configured) Workflow permissions for the snapshot commit

The
[`autoresearchclaw.yml`](https://github.com/johntrue15/Metadata-to-Morphsource-compare/blob/main/.github/workflows/autoresearchclaw.yml)
workflow now grants `contents: write` to the `research-pipeline` job so the
**Commit refreshed knowledge graph snapshot** step can push `docs/data/**`
back to `main` via `stefanzweifel/git-auto-commit-action@v5`. No additional
secret or PAT is needed &mdash; the default `secrets.GITHUB_TOKEN` is used.

If **Settings → Actions → General → Workflow permissions** is currently set
to *Read repository contents and packages permissions*, change it to
**Read and write permissions** (this is required for the auto-commit step
to push). Alternatively, leave it on read-only and grant write only to
this workflow via the per-job `permissions:` block (already done).

## 3. (Optional) Manual first deploy

You can confirm everything works before the first agent run:

1. **Actions → Docs → Run workflow** &mdash; this builds and deploys the
   site with the empty-state knowledge graph.
2. Open <https://johntrue15.github.io/Metadata-to-Morphsource-compare/>
   and the [knowledge-graph page](https://johntrue15.github.io/Metadata-to-Morphsource-compare/knowledge-graph/).
3. The graph viewer should show the *No runs published yet* empty state
   with a link to the AutoResearchClaw workflow.

Then trigger a real research run via **Actions → AutoResearchClaw Agent →
Run workflow**. After it completes, the **Publish knowledge graph
snapshot** step writes `docs/data/knowledge_graph.json` and commits to
`main`, which automatically re-builds and re-deploys the docs site with
real data.

## What changed in the repo

| Area | Change |
|------|--------|
| `mkdocs.yml`, `requirements-docs.txt`, `main.py` | New MkDocs Material site config + macros module |
| `.github/workflows/docs.yml` | New: builds + deploys the docs site |
| `docs/index.md`, `quick-start.md`, `query.md`, `knowledge-graph.md`, `architecture.md`, `reference/**` | New polished docs pages |
| `docs/assets/{js,css,img}/` | Knowledge-graph viewer + theme assets |
| `docs/data/knowledge_graph.json`, `docs/data/runs/_manifest.json` | Placeholder snapshots (auto-overwritten by the runner) |
| `docs/scripts/merge_graphs.py` | Snapshot publisher run on the self-hosted runner |
| `.github/workflows/autoresearchclaw.yml` | New trailing steps: run `merge_graphs.py` and `git-auto-commit-action` |
| `docs/index.html` | Moved verbatim to `docs/query.html` for backward compatibility |
| `README.md` | Trimmed to a short overview with prominent docs-site links |
| `.gitignore` | Adds `site/` (MkDocs build output) |
| `.nojekyll`, `_config.yml` (root) | Transitional shims to keep the legacy Jekyll Pages builder from failing while the maintainer switches the Pages source. No-ops after the switch. |

No application code (`research_agent.py`, `metadata_to_morphsource/**`,
`knowledge_graph.py`, the integrity verifier, the seg-train trainer, the
nnInteractive scripts) was modified. Only the deployment / docs surface
is new.
