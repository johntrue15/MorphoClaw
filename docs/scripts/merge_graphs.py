#!/usr/bin/env python3
"""Publish the latest AutoResearchClaw knowledge-graph JSON into ``docs/data/``.

This is invoked by the ``Publish knowledge graph snapshot`` step in
``.github/workflows/autoresearchclaw.yml`` after every run on the self-hosted
runner. It is also safe to call locally::

    python docs/scripts/merge_graphs.py \\
        --in  ~/.autoresearchclaw/graphs \\
        --out docs/data \\
        --run-id "$GITHUB_RUN_ID" \\
        --run-url "https://github.com/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"

What it does:

* Reads every ``*.json`` file in ``--in`` and picks the most recent one
  (by mtime). Skips files that don't have the expected node/edge shape.
* Copies that snapshot into ``--out/runs/<ISO-timestamp>.json``.
* Updates ``--out/knowledge_graph.json`` to be the **union** of every
  snapshot in ``--out/runs/`` (de-duplicated by node id and by
  ``(source, target, relation)``).
* Refreshes ``--out/runs/_manifest.json`` so the docs page can offer a
  run-picker.

Exits 0 with a clear message on the no-op path (no graphs to publish) so
the calling workflow can keep ``if: always()`` semantics without failing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("merge_graphs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        log.warning("Skipping malformed JSON %s: %s", path, exc)
        return None


def _looks_like_graph(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("nodes"), list)
        and isinstance(payload.get("edges"), list)
    )


@dataclass
class SnapshotInfo:
    path: Path
    mtime: float
    payload: Dict[str, Any]


def _discover_snapshots(input_dir: Path) -> List[SnapshotInfo]:
    if not input_dir.exists():
        return []
    out: List[SnapshotInfo] = []
    for f in sorted(input_dir.glob("*.json")):
        payload = _safe_load_json(f)
        if not _looks_like_graph(payload):
            continue
        out.append(SnapshotInfo(path=f, mtime=f.stat().st_mtime, payload=payload))
    out.sort(key=lambda s: s.mtime)
    return out


def _stable_iso_for_mtime(mtime: float) -> str:
    return _dt.datetime.fromtimestamp(mtime, _dt.timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ"
    )


def _stats_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    stats = dict(payload.get("stats") or {})
    if "total_nodes" not in stats:
        stats["total_nodes"] = len(payload.get("nodes") or [])
    if "total_edges" not in stats:
        stats["total_edges"] = len(payload.get("edges") or [])
    return stats


def _edge_key(edge: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(edge.get("source", "")),
        str(edge.get("target", "")),
        str(edge.get("relation", "")),
    )


def _merge_payloads(snapshots: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Union nodes by id (later snapshots override properties) and edges by
    ``(source, target, relation)``."""

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for payload in snapshots:
        for n in payload.get("nodes") or []:
            nid = str(n.get("id") or "").strip()
            if not nid:
                continue
            if nid in nodes:
                # Shallow-merge so the latest snapshot wins on per-prop overlap
                # but earlier properties are preserved when later ones drop them.
                merged = dict(nodes[nid])
                merged.update(n)
                nodes[nid] = merged
            else:
                nodes[nid] = dict(n)
        for e in payload.get("edges") or []:
            edges[_edge_key(e)] = dict(e)

    merged_nodes = list(nodes.values())
    merged_edges = list(edges.values())
    type_counts: Dict[str, int] = {}
    for n in merged_nodes:
        t = n.get("type") or "Unknown"
        type_counts[t] = type_counts.get(t, 0) + 1

    stats = {
        "media": type_counts.get("Media", 0),
        "specimens": type_counts.get("Specimen", 0),
        "papers": type_counts.get("Paper", 0),
        "institutions": type_counts.get("Institution", 0),
        "taxa": type_counts.get("Taxon", 0),
        "media_lists": type_counts.get("MediaList", 0),
        "total_nodes": len(merged_nodes),
        "total_edges": len(merged_edges),
    }
    return {"nodes": merged_nodes, "edges": merged_edges, "stats": stats}


def _hash_payload(payload: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {"nodes": payload.get("nodes"), "edges": payload.get("edges")},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in",
        dest="input_dir",
        default="~/.autoresearchclaw/graphs",
        help="Directory holding the per-run *.json files produced by knowledge_graph.py",
    )
    ap.add_argument(
        "--out",
        dest="output_dir",
        default="docs/data",
        help="Destination directory inside the docs site (default: docs/data)",
    )
    ap.add_argument(
        "--run-id",
        default="",
        help="Workflow run id (used in manifest); defaults to env GITHUB_RUN_ID",
    )
    ap.add_argument(
        "--run-url",
        default="",
        help="Workflow run URL (used in manifest); defaults to constructed from env",
    )
    ap.add_argument(
        "--topic",
        default="",
        help="Optional human-readable research topic for the manifest",
    )
    ap.add_argument(
        "--max-history",
        type=int,
        default=50,
        help="Keep at most this many per-run snapshots (oldest pruned). Default: 50",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    import os

    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[merge_graphs] %(message)s",
    )

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    snapshots = _discover_snapshots(input_dir)
    if not snapshots:
        log.info("No graph snapshots found in %s; nothing to publish.", input_dir)
        return 0

    latest = snapshots[-1]
    iso = _stable_iso_for_mtime(latest.mtime)
    target_name = f"{iso}.json"
    target_path = runs_dir / target_name

    # Avoid re-committing identical content.
    new_hash = _hash_payload(latest.payload)
    existing = _safe_load_json(target_path) if target_path.exists() else None
    if existing and _hash_payload(existing) == new_hash:
        log.info("Snapshot %s already up to date (hash %s).", target_name, new_hash)
    else:
        # Stamp the payload with provenance so the docs page can show it.
        run_id = args.run_id or os.environ.get("GITHUB_RUN_ID", "")
        run_url = args.run_url or (
            f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}"
            if run_id and os.environ.get("GITHUB_REPOSITORY")
            else ""
        )
        payload = dict(latest.payload)
        payload["generated_at"] = _now_iso()
        if "stats" not in payload or not isinstance(payload["stats"], dict):
            payload["stats"] = _stats_from(payload)
        payload["source"] = {
            "file": latest.path.name,
            "run_id": run_id,
            "run_url": run_url,
            "topic": args.topic,
            "hash": new_hash,
        }
        target_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info(
            "Wrote per-run snapshot %s (%d nodes, %d edges, hash %s).",
            target_path,
            len(payload.get("nodes") or []),
            len(payload.get("edges") or []),
            new_hash,
        )

    # Rebuild the cumulative graph from every per-run file under runs/.
    per_run_files = sorted(runs_dir.glob("*.json"))
    # Optional pruning of oldest snapshots when we blow past the cap.
    if args.max_history > 0 and len(per_run_files) > args.max_history:
        for old in per_run_files[: len(per_run_files) - args.max_history]:
            log.info("Pruning old snapshot %s", old.name)
            old.unlink()
        per_run_files = sorted(runs_dir.glob("*.json"))

    cumulative_payloads: List[Dict[str, Any]] = []
    manifest_runs: List[Dict[str, Any]] = []
    for f in per_run_files:
        payload = _safe_load_json(f)
        if not _looks_like_graph(payload):
            continue
        cumulative_payloads.append(payload)
        src = payload.get("source") or {}
        manifest_runs.append(
            {
                "file": f.name,
                "generated_at": payload.get("generated_at"),
                "stats": payload.get("stats") or _stats_from(payload),
                "run_id": src.get("run_id", ""),
                "run_url": src.get("run_url", ""),
                "topic": src.get("topic", ""),
                "hash": src.get("hash"),
            }
        )

    merged = _merge_payloads(cumulative_payloads)
    merged["generated_at"] = _now_iso()
    merged["source"] = {
        "kind": "cumulative",
        "snapshot_count": len(cumulative_payloads),
    }
    (output_dir / "knowledge_graph.json").write_text(
        json.dumps(merged, indent=2, default=str), encoding="utf-8"
    )
    log.info(
        "Wrote cumulative graph (%d nodes, %d edges across %d runs).",
        merged["stats"]["total_nodes"],
        merged["stats"]["total_edges"],
        len(cumulative_payloads),
    )

    manifest = {
        "runs": manifest_runs,
        "generated_at": _now_iso(),
        "count": len(manifest_runs),
    }
    (runs_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log.info("Wrote run manifest (%d entries).", len(manifest_runs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
