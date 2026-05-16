#!/usr/bin/env python3
r"""
Verify an AutoResearchClaw run by building a Plato's-Cave-style
integrity DAG over its GitHub issue(s).

Pipeline
--------
1.  Resolve a parent tracking issue + any sibling "child" issues that
    AutoResearchClaw produced (matched by the ``research-agent`` label
    and a shared run id, OR an explicit list passed on the CLI).
2.  Pull the issue bodies, comments, and reactions via the GitHub REST
    API (re-using the same pattern as ``research_agent.GitHubIssueReporter``).
3.  Extract the discoveries (LLM-claims) and any media / specimen IDs
    that were written into the body.
4.  Build an integrity DAG:
        ResearchTopic -> SourceRecord(s) -> Discovery(ies) -> ExpertReview
                      \-> Query -> SearchResult ----------/
5.  Run the five MVP verifier agents (metadata, file, lineage, AI QC,
    expert) and attach their scores to the corresponding nodes.
6.  Propagate trust, compute release decision, write a JSON artefact,
    and post a Markdown verification comment back on the parent issue.

Usage
-----
::

    python verify_research_run.py --issue 223
    python verify_research_run.py --issue 223 --child-issues 224,225,226
    python verify_research_run.py --report research_report.json
    python verify_research_run.py --issue 223 --no-llm   # offline AI-QC
    python verify_research_run.py --issue 223 --dry-run  # do not post

Exit codes
----------
0   integrity status is ``verified`` or ``conditionally_verified``
1   integrity status is ``needs_review``
2   integrity status is ``rejected`` (or hard error)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _helpers import (
    AUTORESEARCHCLAW_HOME,
    load_dotenv,
    safe_first,
)
from integrity_cache import RecordCache
from integrity_graph import IntegrityGraph
from integrity_verifiers import (
    AIQCVerifier,
    ExpertVerifier,
    FileVerifier,
    LineageVerifier,
    MetadataVerifier,
)

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    else logging.INFO,
    format=LOG_FORMAT,
    stream=sys.stdout,
)
log = logging.getLogger("VerifyRun")

REPORTS_DIR = AUTORESEARCHCLAW_HOME / "integrity"


def _ensure_reports_dir() -> Path:
    """Create the integrity-reports directory on demand.

    Done lazily so importing this module (e.g. during ``pytest``) does
    not require write access to ``~/.autoresearchclaw``.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


# ---------------------------------------------------------------------------
# Lightweight GitHub REST client (re-uses research_agent's pattern)
# ---------------------------------------------------------------------------


class GitHubClient:
    """Minimal stdlib-only GitHub REST client with retries.

    Mirrors the retry logic in :class:`research_agent.GitHubIssueReporter`
    so that this script behaves identically when running on the same
    self-hosted runner.
    """

    _MAX_RETRIES = 4
    _RETRYABLE_STATUSES = {500, 502, 503, 504}

    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token

    @property
    def enabled(self) -> bool:
        return bool(self.repo and self.token)

    def _api(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        accept: str = "application/vnd.github.v3+json",
    ) -> Tuple[int, Any]:
        url = f"https://api.github.com/repos/{self.repo}/{path.lstrip('/')}"
        for attempt in range(1, self._MAX_RETRIES + 1):
            data = json.dumps(payload).encode("utf-8") if payload else None
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Authorization", f"token {self.token}")
            req.add_header("Accept", accept)
            if data:
                req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode("utf-8")
                    parsed = json.loads(body) if body else None
                    return resp.getcode(), parsed
            except urllib.error.HTTPError as exc:
                if exc.code in self._RETRYABLE_STATUSES and attempt < self._MAX_RETRIES:
                    delay = min(5 * (2 ** (attempt - 1)), 60)
                    log.warning("GH %s %s -> %d, retrying in %ds",
                                method, path, exc.code, delay)
                    time.sleep(delay)
                    continue
                log.warning("GH %s %s -> %d", method, path, exc.code)
                return exc.code, None
            except Exception as exc:
                if attempt < self._MAX_RETRIES:
                    delay = min(5 * (2 ** (attempt - 1)), 60)
                    log.warning("GH %s %s failed: %s, retrying in %ds",
                                method, path, exc, delay)
                    time.sleep(delay)
                    continue
                log.warning("GH %s %s failed: %s", method, path, exc)
                return 0, None
        return 0, None

    def get_issue(self, number: int) -> Optional[Dict[str, Any]]:
        # Use squirrel-girl preview to get reactions counts.
        status, data = self._api(
            "GET", f"issues/{number}",
            accept="application/vnd.github.squirrel-girl-preview+json",
        )
        return data if status == 200 else None

    def list_comments(self, number: int) -> List[Dict[str, Any]]:
        status, data = self._api("GET", f"issues/{number}/comments")
        return data if status == 200 and isinstance(data, list) else []

    def list_issues_by_label(self, label: str, since: str = "", state: str = "all"
                              ) -> List[Dict[str, Any]]:
        path = f"issues?labels={label}&state={state}&per_page=100"
        if since:
            path += f"&since={since}"
        status, data = self._api("GET", path)
        return data if status == 200 and isinstance(data, list) else []

    def post_comment(self, number: int, body: str) -> bool:
        status, _ = self._api("POST", f"issues/{number}/comments", {"body": body})
        return status == 201

    def update_labels(self, number: int, labels: List[str]) -> bool:
        status, _ = self._api("PUT", f"issues/{number}/labels", {"labels": labels})
        return status == 200


# ---------------------------------------------------------------------------
# Extraction from issue text
# ---------------------------------------------------------------------------


_MEDIA_ID_RE = re.compile(r"\b(?:media[ _/-]?(?:id)?[ /:]?)?0{0,3}(\d{6,9})\b", re.IGNORECASE)
_MORPHOSOURCE_URL_RE = re.compile(
    r"morphosource\.org/(?:concern/)?(?:media|physical_objects|media-lists)/(\d{6,9})",
    re.IGNORECASE,
)
_DISCOVERY_BULLET_RE = re.compile(r"^[-*]\s+(.+?)$", re.MULTILINE)
_QUERY_LINE_RE = re.compile(
    r'"([^"\n]{2,80})"\s*(?:->|\u2192|--)\s*(\d+)\s*results?', re.IGNORECASE
)


def extract_media_ids(text: str) -> List[str]:
    """Extract MorphoSource-style 9-digit (zero-padded) media IDs.

    Combines two strategies:
      * URL-based extraction from morphosource.org links (very precise)
      * Token-based extraction for plain ``000769445``-style IDs (broader)
    """
    found: List[str] = []
    seen: set[str] = set()
    for m in _MORPHOSOURCE_URL_RE.finditer(text or ""):
        mid = m.group(1).lstrip("0")
        if mid and mid not in seen:
            seen.add(mid)
            found.append(mid.zfill(9))
    for m in _MEDIA_ID_RE.finditer(text or ""):
        mid = m.group(1).lstrip("0")
        if mid and mid not in seen and 6 <= len(mid) <= 9:
            seen.add(mid)
            found.append(mid.zfill(9))
    return found


def extract_discoveries(body: str) -> List[str]:
    """Pull bullet-pointed discoveries from a ``### Key Discoveries`` section."""
    if not body:
        return []
    m = re.search(
        r"###\s*(?:Key\s+)?Discoveries\s*\n+(.*?)(?:\n###|\Z)",
        body, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        # Fall back to top-level bullets in the whole body
        bullets = _DISCOVERY_BULLET_RE.findall(body)
    else:
        bullets = _DISCOVERY_BULLET_RE.findall(m.group(1))
    out = []
    for b in bullets:
        b = b.strip()
        if 6 <= len(b) <= 600 and not b.startswith("**"):
            out.append(b)
    return out[:20]


def extract_queries(body: str) -> List[Tuple[str, int]]:
    """Pull ``"query" -> N results`` pairs from the issue body."""
    pairs: List[Tuple[str, int]] = []
    for m in _QUERY_LINE_RE.finditer(body or ""):
        try:
            pairs.append((m.group(1), int(m.group(2))))
        except (TypeError, ValueError):
            continue
    return pairs[:30]


def extract_topic(issue: Dict[str, Any]) -> str:
    """Best-effort topic extraction from the issue body."""
    body = issue.get("body") or ""
    m = re.search(r"\*\*Research Topic:\*\*\s*\n+(.+?)(?:\n\*\*|\n###|\n---|\Z)",
                  body, re.DOTALL)
    if m:
        return m.group(1).strip()
    title = issue.get("title") or ""
    title = re.sub(r"^Research\s*\[[^\]]+\]:\s*", "", title)
    return title.strip()


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    issue: Dict[str, Any],
    comments: List[Dict[str, Any]],
    child_issues: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
    use_llm: bool = True,
    allow_network: bool = True,
    cache: Optional[RecordCache] = None,
    max_media: int = 10,
) -> IntegrityGraph:
    """Construct the integrity DAG for a research run.

    Parameters
    ----------
    cache
        A pre-built :class:`integrity_cache.RecordCache`.  When ``None``
        a network-disabled cache is created locally so this function
        remains drop-in usable from unit tests.
    max_media
        Hard cap on how many distinct MorphoSource media IDs the
        verifier will inspect.  Defaults to ``10`` (down from the old
        20) to keep total API traffic bounded even on issues that
        mention dozens of records.
    """
    topic = extract_topic(issue)
    issue_number = issue.get("number")

    if cache is None:
        cache = RecordCache(cache_dir=None, fetcher=None)  # offline default

    graph = IntegrityGraph(
        morphodepot_id=f"GH-{issue_number}",
        source={"repository": "GitHub", "issue_number": issue_number, "topic": topic},
    )

    # 1. Topic node (the "hypothesis")
    topic_id = graph.add_node(
        role="ResearchTopic",
        text=topic[:300] or "(no topic)",
        metrics={k: 0.85 for k in ("credibility", "relevance", "evidence_strength",
                                    "method_rigor", "reproducibility", "authority_support")},
        verifier="(seed)",
    )

    # 2. SourceRecord nodes for every media id mentioned across the
    # parent + child issues.  All API calls go through the shared
    # cache so the three verifier paths (metadata / file / lineage)
    # share a single fetch per media id.
    metadata_v = MetadataVerifier(allow_network=allow_network)
    file_v = FileVerifier()
    lineage_v = LineageVerifier(cache=cache if allow_network else None,
                                  allow_network=allow_network)

    full_text = (issue.get("body") or "") + "\n" + "\n".join(
        c.get("body", "") for c in comments
    )
    for child, ccomments in child_issues:
        full_text += "\n" + (child.get("body") or "")
        for cc in ccomments:
            full_text += "\n" + cc.get("body", "")

    media_ids = extract_media_ids(full_text)
    log.info("Found %d unique media IDs across run (capped to %d)",
             len(media_ids), max_media)

    source_node_ids: Dict[str, int] = {}
    record_cache: Dict[str, Dict[str, Any]] = {}
    for mid in media_ids[:max_media]:
        record = cache.get(mid) if allow_network else {"id": [mid]}
        if not record:
            record = {"id": [mid]}
        record_cache[mid] = record
        meta_result = metadata_v.verify(record)
        snid = graph.add_node(
            role="SourceRecord",
            text=f"MorphoSource media {mid}: "
                 f"{safe_first(record.get('title')) or '(no title)'}"[:200],
            parents=[topic_id],
            metrics=meta_result.metrics,
            evidence={"media_id": mid, **meta_result.evidence},
            verifier=meta_result.verifier,
            notes=meta_result.notes,
        )
        source_node_ids[mid] = snid

        # File and lineage verifiers attach as siblings to the SourceRecord.
        # Modelling them as children-of-source preserves the trust gate.
        file_result = file_v.verify(record)
        graph.add_node(
            role="Specimen" if file_result.evidence.get("media_type", "")
                                .lower().startswith("specimen")
                            else "SourceRecord",
            text=f"File metadata for media {mid}",
            parents=[snid],
            metrics=file_result.metrics,
            evidence=file_result.evidence,
            verifier=file_result.verifier,
            notes=file_result.notes,
        )
        lineage_result = lineage_v.verify(record)
        graph.add_node(
            role="SourceRecord",
            text=f"Lineage for media {mid}",
            parents=[snid],
            metrics=lineage_result.metrics,
            evidence=lineage_result.evidence,
            verifier=lineage_result.verifier,
            notes=lineage_result.notes,
        )

    # 3. Query and SearchResult nodes
    queries = extract_queries(issue.get("body") or "")
    for child, _cc in child_issues:
        queries.extend(extract_queries(child.get("body") or ""))

    for query_text, hits in queries[:20]:
        # A high hit count is *evidence* of relevance; zero hits means
        # the agent failed (low relevance, low evidence strength).
        hits_norm = min(hits / 50.0, 1.0)
        qid = graph.add_node(
            role="Query",
            text=query_text[:200],
            parents=[topic_id],
            metrics={
                "credibility":      0.7,
                "relevance":        0.5 + 0.4 * hits_norm,
                "evidence_strength": hits_norm,
                "method_rigor":     0.7,
                "reproducibility":  0.85,
                "authority_support": 0.4,
            },
            evidence={"query": query_text, "hits": hits},
            verifier="(parsed-from-issue)",
        )
        graph.add_node(
            role="SearchResult",
            text=f"{hits} results for {query_text!r}",
            parents=[qid],
            metrics={
                "credibility":      0.85 if hits > 0 else 0.4,
                "relevance":        hits_norm,
                "evidence_strength": hits_norm,
                "method_rigor":     0.7,
                "reproducibility":  0.85,
                "authority_support": 0.4,
            },
            evidence={"hits": hits},
            verifier="(parsed-from-issue)",
        )

    # 4. Discovery nodes scored by AI-QC verifier
    aiqc = AIQCVerifier(use_llm=use_llm)
    discoveries = extract_discoveries(issue.get("body") or "")
    for child, _cc in child_issues:
        discoveries.extend(extract_discoveries(child.get("body") or ""))

    discovery_node_ids: List[int] = []
    for disc in discoveries[:20]:
        # If the discovery references one of our media ids, link it.
        parents = [topic_id]
        for mid in extract_media_ids(disc):
            if mid in source_node_ids:
                parents.append(source_node_ids[mid])

        qc_result = aiqc.verify(claim=disc, topic=topic)
        did = graph.add_node(
            role="Discovery",
            text=disc[:300],
            parents=parents,
            metrics=qc_result.metrics,
            evidence=qc_result.evidence,
            verifier=qc_result.verifier,
            notes=qc_result.notes,
        )
        discovery_node_ids.append(did)

    # 5. Expert review derived from the parent issue's comments + reactions.
    expert_v = ExpertVerifier()
    expert_result = expert_v.verify(issue, comments)
    parents_for_expert = discovery_node_ids or list(source_node_ids.values()) or [topic_id]
    graph.add_node(
        role="ExpertReview",
        text=f"Expert review of issue #{issue_number} "
             f"({expert_result.evidence.get('human_comment_count', 0)} human comments, "
             f"{expert_result.evidence.get('positive_reactions', 0)} positive reactions)",
        parents=parents_for_expert,
        metrics=expert_result.metrics,
        evidence=expert_result.evidence,
        verifier=expert_result.verifier,
        notes=expert_result.notes,
    )

    # 6. Citations harvested from the search-result records
    for mid, record in record_cache.items():
        for citation_text, doi in _extract_dois(record):
            if mid not in source_node_ids:
                continue
            graph.add_node(
                role="Citation",
                text=f"DOI {doi} (from media {mid})",
                parents=[source_node_ids[mid]],
                metrics={
                    "credibility":      0.9,
                    "relevance":        0.7,
                    "evidence_strength": 0.7,
                    "method_rigor":     0.7,
                    "reproducibility":  0.85,
                    "authority_support": 0.95,
                },
                evidence={"doi": doi, "source": citation_text[:200]},
                verifier="citation-extract",
            )

    return graph


def _extract_dois(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return a list of ``(source_field_text, doi)`` extracted from a record."""
    out: List[Tuple[str, str]] = []
    if not isinstance(record, dict):
        return out
    pattern = re.compile(r'10\.\d{4,}/[^\s,;"\'\]>)]+')
    for field_name in ("doi", "cite_as", "description", "short_description"):
        text = safe_first(record.get(field_name))
        if not text:
            continue
        for m in pattern.finditer(text):
            doi = m.group().rstrip(".")
            out.append((text, doi))
    return out


def build_polite_client(timeout_s: float = 10.0, max_retries: int = 1):
    """Construct a low-impact MorphoSourceClient for verification use.

    The default client used by ``research_agent`` and friends has a
    30-second timeout and 3 retries — appropriate for the data-gathering
    path.  The verifier should *never* burn a minute waiting for one
    media id, so we hand the cache a much politer client (10 s timeout,
    a single retry) and let the cache's circuit breaker handle the
    rest.

    Returns ``None`` if the underlying ``morphosource_client`` module
    cannot be imported, which puts the cache in offline mode.
    """
    try:
        from morphosource_client import MorphoSourceClient
    except ImportError:
        return None
    return MorphoSourceClient(timeout=timeout_s, max_retries=max_retries,
                               backoff_factor=1.0)


def make_morphosource_fetcher(client) -> "Optional[Any]":
    """Wrap *client* as a ``fetcher(media_id) -> dict | None`` callable.

    Returns ``None`` when *client* is ``None``.
    """
    if client is None:
        return None

    def fetcher(media_id: str):
        try:
            record = client.get_media(media_id)
        except Exception as exc:
            log.warning("MorphoSource fetch raised for %s: %s", media_id, exc)
            return None
        if record.error or not record.data:
            return None
        data = record.data
        if isinstance(data.get("response"), dict):
            data = data["response"]
        return data

    return fetcher


# ---------------------------------------------------------------------------
# Sibling / child issue resolution
# ---------------------------------------------------------------------------


_CHILD_NUMBER_RE = re.compile(r"#(\d+)")


def discover_child_issues(
    client: GitHubClient,
    parent: Dict[str, Any],
    parent_comments: List[Dict[str, Any]],
    explicit: Optional[List[int]] = None,
) -> List[int]:
    """Return the list of child-issue numbers belonging to the same run.

    Resolution strategy (first match wins):
      1. ``--child-issues`` CLI argument
      2. ``#NNN`` references in parent issue body / comments
      3. Issues opened after the parent that share the
         ``research-agent`` label
    """
    if explicit:
        return [n for n in explicit if n != parent.get("number")]

    text_blob = (parent.get("body") or "") + "\n" + "\n".join(
        c.get("body", "") for c in parent_comments
    )
    referenced = []
    for m in _CHILD_NUMBER_RE.finditer(text_blob):
        n = int(m.group(1))
        if n != parent.get("number"):
            referenced.append(n)
    referenced = sorted(set(referenced))
    if referenced:
        return referenced[:10]

    # Fallback: look at recent research-agent-labelled issues
    if not client.enabled:
        return []
    since = parent.get("created_at") or ""
    candidates = client.list_issues_by_label("research-agent", since=since)
    return [
        c["number"] for c in candidates
        if c.get("number") != parent.get("number")
        and not c.get("pull_request")
    ][:10]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_artifacts(graph: IntegrityGraph, issue_number: Optional[int]) -> Dict[str, Path]:
    """Write JSON and Markdown reports under ``~/.autoresearchclaw/integrity/``."""
    reports_dir = _ensure_reports_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"issue_{issue_number}" if issue_number else "report"
    base = reports_dir / f"{ts}_{suffix}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    graph.export_json(json_path)
    md_path.write_text(graph.to_markdown(), encoding="utf-8")
    log.info("Wrote integrity artifacts: %s, %s", json_path, md_path)
    return {"json": json_path, "markdown": md_path}


def post_or_print(client: Optional[GitHubClient], issue_number: Optional[int],
                   markdown: str, dry_run: bool) -> None:
    if dry_run or client is None or not client.enabled or not issue_number:
        print(markdown)
        return
    if client.post_comment(issue_number, markdown):
        log.info("Posted integrity report comment on issue #%d", issue_number)
    else:
        log.warning("Failed to post integrity report; printing instead")
        print(markdown)


def label_issue(client: Optional[GitHubClient], issue: Dict[str, Any],
                 status: str) -> None:
    """Update the parent issue's labels with an ``integrity-<status>`` tag."""
    if client is None or not client.enabled:
        return
    existing = [
        (lab.get("name") if isinstance(lab, dict) else str(lab))
        for lab in issue.get("labels", [])
        if (lab.get("name") if isinstance(lab, dict) else str(lab))
        and not (lab.get("name") if isinstance(lab, dict) else str(lab)).startswith("integrity-")
    ]
    new_labels = sorted(set(existing + [f"integrity-{status}"]))
    client.update_labels(issue["number"], new_labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            log.warning("Ignoring invalid issue number: %s", part)
    return out


def _exit_code_for(status: str) -> int:
    if status in ("verified", "conditionally_verified"):
        return 0
    if status == "needs_review":
        return 1
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=int, default=None,
                        help="Parent tracking issue number")
    parser.add_argument("--child-issues", type=str, default="",
                        help="Comma-separated list of child issue numbers")
    parser.add_argument("--report", type=str, default=None,
                        help="Path to a research_report.json (offline mode)")
    parser.add_argument("--repo", type=str,
                        default=os.environ.get("GITHUB_REPOSITORY", ""),
                        help="owner/repo (defaults to $GITHUB_REPOSITORY)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable the AI-QC LLM call (use heuristic only)")
    parser.add_argument("--no-network", action="store_true",
                        help="Skip MorphoSource and CrossRef API calls")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the integrity report instead of posting it")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Override output JSON path")
    parser.add_argument("--max-media", type=int, default=10,
                        help="Maximum distinct media IDs to inspect (default: 10)")
    parser.add_argument("--api-timeout", type=float,
                        default=float(os.environ.get("VERIFIER_API_TIMEOUT", "10")),
                        help="Per-call MorphoSource timeout in seconds (default: 10)")
    parser.add_argument("--api-retries", type=int,
                        default=int(os.environ.get("VERIFIER_API_RETRIES", "1")),
                        help="Per-call MorphoSource retry count (default: 1)")
    parser.add_argument("--api-min-delay", type=float,
                        default=float(os.environ.get("VERIFIER_API_MIN_DELAY", "0.5")),
                        help="Minimum seconds between MorphoSource calls (default: 0.5)")
    parser.add_argument("--circuit-breaker", type=int,
                        default=int(os.environ.get("VERIFIER_CIRCUIT_BREAKER", "3")),
                        help="Open the circuit after N consecutive failures (default: 3)")
    parser.add_argument("--cache-ttl-days", type=int,
                        default=int(os.environ.get("VERIFIER_CACHE_TTL_DAYS", "7")),
                        help="Disk-cache TTL in days (default: 7; 0 disables TTL)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the on-disk record cache")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN", "")
    client: Optional[GitHubClient] = None
    if args.repo and token and args.issue:
        client = GitHubClient(args.repo, token)
        if not client.enabled:
            client = None

    issue: Optional[Dict[str, Any]] = None
    comments: List[Dict[str, Any]] = []
    child_pairs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []

    if args.report:
        issue, comments = _synthesize_issue_from_report(Path(args.report))
    elif client and args.issue:
        issue = client.get_issue(args.issue)
        if issue is None:
            log.error("Could not fetch issue #%d from %s", args.issue, args.repo)
            return 2
        comments = client.list_comments(args.issue)
        explicit = _parse_int_list(args.child_issues)
        child_numbers = discover_child_issues(client, issue, comments, explicit=explicit)
        log.info("Discovered %d child issue(s): %s", len(child_numbers), child_numbers)
        for n in child_numbers:
            child = client.get_issue(n)
            if child is None:
                continue
            child_pairs.append((child, client.list_comments(n)))
    else:
        log.error("Provide either --issue (with $GITHUB_TOKEN) or --report")
        parser.print_help()
        return 2

    if issue is None:
        log.error("No issue resolved; aborting")
        return 2

    # Build a verification-specific MorphoSource client (short timeout,
    # low retries) and a polite, persistent record cache.  This is what
    # keeps a single ``/verify`` run from turning into a 30-minute
    # connection storm against the public MorphoSource API.  We use a
    # distinct variable name (``ms_client``) so it never shadows the
    # GitHubClient bound above, which would break post_or_print().
    cache: Optional[RecordCache]
    if args.no_network:
        cache = RecordCache(cache_dir=None, fetcher=None,
                             min_delay_s=0.0, circuit_breaker_threshold=0)
    else:
        ms_client = build_polite_client(
            timeout_s=args.api_timeout,
            max_retries=args.api_retries,
        )
        cache_dir = None if args.no_cache else _ensure_reports_dir() / "cache"
        cache = RecordCache(
            cache_dir=cache_dir,
            ttl_days=args.cache_ttl_days,
            fetcher=make_morphosource_fetcher(ms_client),
            min_delay_s=args.api_min_delay,
            circuit_breaker_threshold=args.circuit_breaker,
        )
        log.info("Verifier client: timeout=%.1fs retries=%d min-delay=%.2fs "
                 "breaker=%d cache-dir=%s",
                 args.api_timeout, args.api_retries, args.api_min_delay,
                 args.circuit_breaker, cache_dir or "(disabled)")

    graph = build_graph(
        issue, comments, child_pairs,
        use_llm=not args.no_llm,
        allow_network=not args.no_network,
        cache=cache,
        max_media=args.max_media,
    )
    decision = graph.release_decision()
    log.info("Integrity status: %s | scientific=%.2f | training=%.2f | commercial=%.2f",
             decision["status"],
             decision["scientific_validity"],
             decision["ai_training_validity"],
             decision["commercial_release_validity"])
    if cache:
        log.info("Cache: %s", cache.stats.summary())
        if cache.stats.circuit_open:
            log.warning(
                "MorphoSource circuit breaker opened mid-run after %d "
                "failure(s); %d media id(s) were scored with cache-only "
                "data.  Consider re-running ``/verify`` later.",
                cache.stats.network_failures,
                cache.stats.skipped_after_breaker,
            )

    artefacts = write_artifacts(graph, args.issue)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(graph.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        log.info("Also wrote %s", args.out_json)

    markdown = graph.to_markdown()
    if cache and (cache.stats.network_calls or cache.stats.disk_hits
                   or cache.stats.skipped_after_breaker):
        breaker_note = ""
        if cache.stats.circuit_open:
            breaker_note = (
                f"\n> ⚠️ MorphoSource circuit breaker tripped after "
                f"{cache.stats.network_failures} failure(s); "
                f"{cache.stats.skipped_after_breaker} media id(s) were "
                f"scored from cache only.  Re-run `/verify` later for "
                f"a fresh check.\n"
            )
        markdown += (
            f"\n\n### Cache & API politeness\n\n"
            f"- Memory hits: `{cache.stats.memory_hits}`\n"
            f"- Disk hits: `{cache.stats.disk_hits}`\n"
            f"- Network calls: `{cache.stats.network_calls}` "
            f"(`{cache.stats.network_failures}` failed)\n"
            f"- Min delay between calls: `{cache.min_delay_s:.2f}s`\n"
            f"- Circuit breaker: `{'open' if cache.stats.circuit_open else 'closed'}`\n"
            f"{breaker_note}"
        )
    post_or_print(client, args.issue, markdown, args.dry_run)
    if not args.dry_run and client and issue:
        label_issue(client, issue, decision["status"])

    # GitHub Actions output
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"integrity_status={decision['status']}\n")
            f.write(f"scientific_validity={decision['scientific_validity']}\n")
            f.write(f"ai_training_validity={decision['ai_training_validity']}\n")
            f.write(f"commercial_release_validity={decision['commercial_release_validity']}\n")
            f.write(f"integrity_report={artefacts['markdown']}\n")
            if cache:
                f.write(f"cache_network_calls={cache.stats.network_calls}\n")
                f.write(f"cache_network_failures={cache.stats.network_failures}\n")
                f.write(f"cache_circuit_open={'true' if cache.stats.circuit_open else 'false'}\n")

    return _exit_code_for(decision["status"])


def _synthesize_issue_from_report(report_path: Path
                                   ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build a fake issue + comment list from a local research_report.json.

    Used when running the verifier offline (no GitHub token).  This lets
    the same code path serve unit tests, local CI, and post-mortem
    analysis of older runs.
    """
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    topic = data.get("topic") or "(unknown topic)"
    memory = data.get("final_memory") or {}

    discoveries = memory.get("all_discoveries", [])
    queries = memory.get("queries_tried", [])
    body_lines = [
        f"**Research Topic:**\n{topic}",
        "",
        "### Key Discoveries",
    ]
    for d in discoveries[:20]:
        body_lines.append(f"- {d}")
    body_lines += ["", "### Queries"]
    for q in queries[:30]:
        if isinstance(q, dict):
            body_lines.append(f'- "{q.get("query", "")}" -> {q.get("count", 0)} results')

    issue = {
        "number": 0,
        "title": f"Offline report: {topic}",
        "body": "\n".join(body_lines),
        "labels": [],
        "reactions": {},
    }
    return issue, []


if __name__ == "__main__":
    sys.exit(main())
