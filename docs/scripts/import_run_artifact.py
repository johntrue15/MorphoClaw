#!/usr/bin/env python3
"""Import a ``knowledge-graph`` artifact from a past AutoResearchClaw run.

This is the manual counterpart to the automatic publish step in
``.github/workflows/autoresearchclaw.yml``. Use it when:

* a research run produced a knowledge-graph artifact **before** the publish
  step was wired into the workflow, or
* you want to re-import / replay a specific older run's snapshot.

Two input modes are supported:

1. **--zip PATH** &mdash; point at an artifact zip you downloaded manually
   from ``https://github.com/<owner>/<repo>/actions/runs/<run_id>``.
2. **--run-id ID** &mdash; pass a workflow run ID; the script fetches the
   ``knowledge-graph`` artifact via the GitHub REST API. Needs a
   ``GITHUB_TOKEN`` / ``GH_TOKEN`` env var with ``actions:read`` scope.

After extraction the script calls :mod:`docs.scripts.merge_graphs` so the
imported snapshot is merged into ``docs/data/knowledge_graph.json`` and
recorded in ``docs/data/runs/_manifest.json`` exactly as if the runner had
published it itself.

Examples
--------

::

    # Manually downloaded zip:
    python docs/scripts/import_run_artifact.py \\
        --zip ~/Downloads/knowledge-graph.zip \\
        --run-id 25757036789 \\
        --run-url https://github.com/johntrue15/MorphoClaw/actions/runs/25757036789

    # API mode (needs GITHUB_TOKEN exported):
    GITHUB_TOKEN=ghp_xxx python docs/scripts/import_run_artifact.py \\
        --run-id 25757036789 \\
        --repo johntrue15/MorphoClaw

After it finishes, ``git status`` should show changes under ``docs/data/`` ready
to commit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("import_run_artifact")

DEFAULT_ARTIFACT_NAME = "knowledge-graph"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _gh_token() -> Optional[str]:
    return (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("ARC_GITHUB_TOKEN")
    )


def _gh_request(url: str, *, token: Optional[str], accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Accept", accept)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "import_run_artifact/1.0 (autoresearchclaw)")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310 - GitHub API
            return resp.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"GitHub API {e.code} for {url}: {body[:400]}") from e


def _resolve_artifact_url(repo: str, run_id: int, name: str, token: Optional[str]) -> str:
    """Return the ``archive_download_url`` for the artifact matching ``name``."""

    if not token:
        raise RuntimeError(
            "Downloading artifacts via the GitHub API requires a token. "
            "Export GITHUB_TOKEN (or GH_TOKEN) with the actions:read scope and retry."
        )
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
    payload = json.loads(_gh_request(url, token=token).decode("utf-8"))
    artifacts = payload.get("artifacts") or []
    matches = [a for a in artifacts if a.get("name") == name]
    if not matches:
        names = ", ".join(sorted({a.get("name", "?") for a in artifacts}))
        raise RuntimeError(
            f"No artifact named {name!r} on run {run_id} (saw: {names or 'none'})"
        )
    return matches[0]["archive_download_url"]


def _download_artifact(repo: str, run_id: int, name: str, dest: Path) -> None:
    token = _gh_token()
    download_url = _resolve_artifact_url(repo, run_id, name, token)
    log.info("Resolved artifact %r on run %s to %s", name, run_id, download_url)

    blob = _gh_request(download_url, token=token, accept="application/octet-stream")
    dest.write_bytes(blob)
    log.info("Wrote %d bytes to %s", len(blob), dest)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_zip(zip_path: Path, dest_dir: Path) -> int:
    """Extract ``*.json`` and ``*.html`` files from the artifact zip."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            base = Path(member.filename).name
            if not base or member.is_dir():
                continue
            if not (base.endswith(".json") or base.endswith(".html")):
                continue
            out = dest_dir / base
            with zf.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
            log.debug("Extracted %s -> %s", member.filename, out)
    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", dest="zip_path", help="Path to a manually downloaded artifact zip")
    src.add_argument("--run-id", type=int, help="Workflow run ID to fetch the artifact for")
    ap.add_argument(
        "--repo",
        default="johntrue15/MorphoClaw",
        help="owner/repo (only used for API mode). Default: johntrue15/MorphoClaw",
    )
    ap.add_argument(
        "--artifact-name",
        default=DEFAULT_ARTIFACT_NAME,
        help=f"Artifact name to import (default: {DEFAULT_ARTIFACT_NAME})",
    )
    ap.add_argument(
        "--run-url",
        default="",
        help="Optional GitHub run URL to stamp into the manifest (links to the run)",
    )
    ap.add_argument(
        "--topic",
        default="(historical import)",
        help="Optional human-readable topic for the manifest entry",
    )
    ap.add_argument(
        "--out",
        default="docs/data",
        help="docs/data directory (default: docs/data)",
    )
    ap.add_argument(
        "--keep-temp",
        action="store_true",
        help="Don't delete the temporary extraction directory (useful for debugging)",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[import_artifact] %(message)s",
    )

    workdir = Path(tempfile.mkdtemp(prefix="arc_kg_artifact_"))
    log.info("Working in %s", workdir)
    try:
        zip_path = workdir / "artifact.zip"
        if args.zip_path:
            src_zip = Path(args.zip_path).expanduser()
            if not src_zip.exists():
                log.error("--zip path does not exist: %s", src_zip)
                return 2
            shutil.copyfile(src_zip, zip_path)
            log.info("Copied local zip %s -> %s", src_zip, zip_path)
        else:
            _download_artifact(args.repo, args.run_id, args.artifact_name, zip_path)

        extract_dir = workdir / "extracted"
        n = _extract_zip(zip_path, extract_dir)
        if n == 0:
            log.error(
                "No .json or .html files found inside %s; refusing to publish.", zip_path
            )
            return 3
        log.info("Extracted %d files into %s", n, extract_dir)

        json_files = list(extract_dir.glob("*.json"))
        if not json_files:
            log.error("Artifact contained no JSON graph files.")
            return 4
        log.info("JSON snapshots in artifact: %s", [p.name for p in json_files])

        # Hand off to merge_graphs.py so we go through exactly the same code
        # path the autoresearchclaw.yml workflow uses on the runner.
        script = Path(__file__).resolve().parent / "merge_graphs.py"
        cmd = [
            sys.executable,
            str(script),
            "--in",
            str(extract_dir),
            "--out",
            str(Path(args.out).resolve()),
        ]
        if args.run_id:
            cmd += ["--run-id", str(args.run_id)]
        if args.run_url:
            cmd += ["--run-url", args.run_url]
        if args.topic:
            cmd += ["--topic", args.topic]
        if args.verbose:
            cmd += ["--verbose"]
        log.info("Running: %s", " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            log.error("merge_graphs.py exited with %s", rc)
            return rc

        log.info("Done. git status should now show changes under %s/", args.out)
        return 0
    finally:
        if args.keep_temp:
            log.info("Kept temp dir for inspection: %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
