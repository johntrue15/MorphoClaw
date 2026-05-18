---
title: Secrets & Env
description: Required and optional environment variables and GitHub Actions secrets used by AutoResearchClaw.
---

# Secrets & Env

Configure secrets in **Settings → Secrets and variables → Actions** (and
mirror them into `.env` for local runs &mdash; see `.env.example`).

## Required

| Name | Where used | Description |
|------|------------|-------------|
| `OPENAI_API_KEY` | `research_agent`, `chat_handler`, `query_formatter`, integrity verifier | OpenAI API key for decomposition, evaluation, synthesis, and verifier scoring. |
| `MORPHOSOURCE_API_KEY` | `morphosource_api`, `morphosource_api_download` | MorphoSource API key for searching and downloading specimen media. |

## Optional

| Name | Default | Description |
|------|---------|-------------|
| `OPENAI_MODEL` | `gpt-5.4` | LLM model id passed to OpenAI. |
| `OPENAI_BASE_URL` | (OpenAI) | Override for proxy / Azure deployments. |
| `NNINTERACTIVE_HOME` | `~/.autoresearchclaw/nninteractive` | Path to the dedicated nnInteractive venv. |
| `NNINTERACTIVE_DEVICE` | auto (CUDA → MPS → CPU) | Force `cuda:0` / `mps` / `cpu`. |
| `AUTORESEARCHCLAW_HOME` | `~/.autoresearchclaw` | Cache directory for specimens, graphs, runs. |
| `SLICER_BIN` | (auto-detected) | Override path to the `Slicer` executable. |
| `SLICER_REQUIRE_INTEGRATION` | `0` | Promote Tier-3 Slicer tests from "skip if missing" to "hard fail". |
| `MORPHOSOURCE_LIVE_TESTS` | `0` | Enable network-reaching client tests. |
| `DEBUG` | `false` | Verbose logging. |

## GitHub Actions auth

The workflows that post issues / comments / artifacts use the default
`secrets.GITHUB_TOKEN` provided by Actions. No additional configuration is
required. The `Publish knowledge graph snapshot` step on
`autoresearchclaw.yml` uses the same token via
[`stefanzweifel/git-auto-commit-action`](https://github.com/stefanzweifel/git-auto-commit-action)
to commit `docs/data/**` back to `main`.

## Local `.env` template

The repo ships an [`.env.example`](https://github.com/johntrue15/MorphoClaw/blob/main/.env.example).
Copy it and fill in your values:

```bash
cp .env.example .env
```

Never commit a real `.env` &mdash; it is in `.gitignore`.

## Verifying secrets in CI

The `code-quality.yml` workflow includes a smoke test
([Tests/test_tool_calls.py](https://github.com/johntrue15/MorphoClaw/blob/main/Tests/test_tool_calls.py))
that imports every tool and asserts each gracefully returns a structured
error dict when its API key is missing &mdash; so a missing secret does not
crash the agent, it just disables that tool.
