"""mkdocs-macros plugin hook: expose knowledge-graph stats to templates.

Usage in any markdown page:

    {{ kg_stats().total_nodes }} nodes across {{ kg_stats().media }} media

The data is read from ``docs/data/knowledge_graph.json`` at build time so
the home page and any other page can show fresh counts without hitting a
network endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _load_kg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"runs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}


def define_env(env) -> None:  # noqa: D401 -- mkdocs-macros entry point
    """Register macros / variables used in docs/*.md."""

    docs_dir = Path(env.project_dir) / "docs"
    kg_path = docs_dir / "data" / "knowledge_graph.json"
    manifest_path = docs_dir / "data" / "runs" / "_manifest.json"

    @env.macro
    def kg_stats() -> Dict[str, Any]:
        data = _load_kg(kg_path)
        stats = dict(data.get("stats") or {})
        stats.setdefault("media", 0)
        stats.setdefault("specimens", 0)
        stats.setdefault("papers", 0)
        stats.setdefault("institutions", 0)
        stats.setdefault("taxa", 0)
        stats.setdefault("total_nodes", 0)
        stats.setdefault("total_edges", 0)
        stats["generated_at"] = data.get("generated_at")
        return stats

    @env.macro
    def kg_run_count() -> int:
        return len(_load_manifest(manifest_path).get("runs", []))

    @env.macro
    def kg_has_data() -> bool:
        return kg_stats().get("total_nodes", 0) > 0
