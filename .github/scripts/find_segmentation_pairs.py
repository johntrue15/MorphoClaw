"""
Find candidate (CT volume, segmented mesh) pairs on MorphoSource for testing.

We need a pair of media records that:

    1. Belong to the same physical object (specimen).
    2. One is a 3D volumetric scan (CT / micro-CT / MRI).
    3. The other is a derived surface segmentation (Mesh / Surface model).
    4. Both have `visibility = open` (open download).

MorphoSource doesn't expose a dedicated "is segmentation derivative of"
relation in every record, so we use the physical-object linkage as a
robust proxy: any open mesh whose specimen also has an open CT volume
is a viable comparison pair.

Usage::

    python find_segmentation_pairs.py \\
        --query "skull mesh" \\
        --max-pairs 5 \\
        --output candidate_pairs.json

The script can also be imported::

    from find_segmentation_pairs import find_pairs
    pairs = find_pairs(query="primate skull", max_pairs=5)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _helpers import safe_first, load_dotenv  # noqa: E402
from morphosource_client import MorphoSourceClient  # noqa: E402

log = logging.getLogger("find_pairs")
load_dotenv()


# Heuristic media-type vocabulary used by MorphoSource. Values are lowercased
# substrings; an item matches if any substring is contained in its media_type.
VOLUMETRIC_TYPES = (
    "ct image series",
    "volumetric image series",
    "ct dicom",
    "micro ct",
    "microct",
    "volumetric",
    "image series",
)
MESH_TYPES = (
    "mesh",
    "surface model",
    "3d surface",
    "model",
)


def _media_type(item: dict) -> str:
    return safe_first(item.get("media_type") or item.get("media_type_ssi", "")).lower()


def _visibility(item: dict) -> str:
    return safe_first(
        item.get("visibility") or item.get("visibility_ssi", "")
    ).lower()


def _physical_object_id(item: dict) -> str:
    """MorphoSource reports the parent physical object under various keys."""
    for key in (
        "physical_object_id",
        "physical_object_id_ssi",
        "physical_object_id_ssim",
        "physical_object_id_tesim",
        "physical_object",
    ):
        value = item.get(key)
        if value:
            return safe_first(value)
    return ""


def _media_id(item: dict) -> str:
    for key in ("id", "media_id", "media_id_ssi"):
        v = item.get(key)
        if v:
            return safe_first(v)
    return ""


def _title(item: dict) -> str:
    return safe_first(item.get("title") or item.get("title_tesim", ""))


def _is_open(item: dict) -> bool:
    v = _visibility(item)
    return v.startswith("open")


def _matches_any(text: str, vocabulary: Iterable[str]) -> bool:
    text = (text or "").lower()
    return any(token in text for token in vocabulary)


@dataclass
class CandidatePair:
    physical_object_id: str
    physical_object_title: str
    taxonomy: str
    ct: dict = field(default_factory=dict)
    mesh: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def is_complete(self) -> bool:
        return bool(self.ct.get("media_id") and self.mesh.get("media_id"))


# ---------------------------------------------------------------------------


def _summarise_item(item: dict) -> dict:
    return {
        "media_id": _media_id(item),
        "title": _title(item),
        "media_type": _media_type(item),
        "visibility": _visibility(item),
        "taxonomy": safe_first(
            item.get("physical_object_taxonomy_name")
            or item.get("physical_object_taxonomy_name_ssim", "")
        ),
        "physical_object_id": _physical_object_id(item),
    }


def _list_media_for_object(client: MorphoSourceClient, object_id: str) -> List[dict]:
    """Return all media records linked to a physical object."""
    if not object_id:
        return []
    object_id = object_id.strip()
    # Try the physical-object endpoint's nested media list first.
    try:
        rec = client.get_physical_object(object_id)
        if rec.data:
            response = rec.data.get("response", rec.data)
            inner = response.get("physical_object", response)
            if isinstance(inner, dict):
                inner_media = inner.get("media") or []
                if isinstance(inner_media, list) and inner_media:
                    return list(inner_media)
    except Exception as exc:
        log.debug("get_physical_object(%s) failed: %s", object_id, exc)

    # Fallback: search /media filtered by physical_object_id.
    for filter_key in (
        "physical_object_id_ssi",
        "physical_object_id",
        "f[physical_object_id_ssim][]",
    ):
        try:
            sr = client.search_media(per_page=50, **{filter_key: object_id})
        except TypeError:
            continue  # Not a recognised kw arg
        if sr.items:
            return sr.items
    return []


# ---------------------------------------------------------------------------


def find_pairs(query: str = "skull", max_pairs: int = 5,
               max_candidates: int = 50,
               require_taxonomy: str = "") -> List[CandidatePair]:
    """Search MorphoSource and return up to *max_pairs* CT↔mesh pairs."""
    client = MorphoSourceClient()
    log.info("Searching media for query=%r (open download, mesh-like)", query)

    pairs: List[CandidatePair] = []
    seen_objects = set()
    page = 1
    examined = 0

    while len(pairs) < max_pairs and examined < max_candidates:
        sr = client.search_media(q=query, per_page=25, page=page)
        if not sr.items:
            break

        for item in sr.items:
            examined += 1
            if examined > max_candidates:
                break

            mtype = _media_type(item)
            if not _matches_any(mtype, MESH_TYPES):
                continue
            if not _is_open(item):
                continue

            object_id = _physical_object_id(item)
            if not object_id or object_id in seen_objects:
                continue
            seen_objects.add(object_id)

            taxon = safe_first(
                item.get("physical_object_taxonomy_name")
                or item.get("physical_object_taxonomy_name_ssim", "")
            )
            if require_taxonomy and require_taxonomy.lower() not in taxon.lower():
                continue

            siblings = _list_media_for_object(client, object_id)
            if not siblings:
                continue

            # Keep only OPEN siblings, then look for a CT/volumetric one.
            ct_record: Optional[dict] = None
            for sib in siblings:
                if _media_id(sib) == _media_id(item):
                    continue  # Skip the mesh itself
                if not _is_open(sib):
                    continue
                if _matches_any(_media_type(sib), VOLUMETRIC_TYPES):
                    ct_record = sib
                    break

            if ct_record is None:
                continue

            specimen_title = (
                safe_first(item.get("physical_object_title"))
                or safe_first(item.get("physical_object_title_tesim", ""))
                or "(unknown specimen)"
            )

            pair = CandidatePair(
                physical_object_id=object_id,
                physical_object_title=specimen_title,
                taxonomy=taxon,
                ct=_summarise_item(ct_record),
                mesh=_summarise_item(item),
            )
            pairs.append(pair)
            log.info(
                "PAIR #%d: object %s (%s) — CT %s + mesh %s",
                len(pairs), object_id, taxon[:40] or "no taxonomy",
                pair.ct["media_id"], pair.mesh["media_id"],
            )
            if len(pairs) >= max_pairs:
                break

        if sr.total_count is not None and examined >= sr.total_count:
            break
        page += 1

    log.info("Examined %d mesh candidates → found %d complete pairs",
             examined, len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Find open-download CT ↔ mesh pairs on MorphoSource"
    )
    p.add_argument("--query", default="skull mesh",
                   help="MorphoSource search query (default: 'skull mesh')")
    p.add_argument("--max-pairs", type=int, default=5,
                   help="Maximum number of pairs to return (default: 5)")
    p.add_argument("--max-candidates", type=int, default=50,
                   help="Cap on mesh candidates examined (default: 50)")
    p.add_argument("--require-taxonomy", default="",
                   help="Only keep pairs whose taxonomy contains this string")
    p.add_argument("--output", default="",
                   help="Write JSON list of pairs to this file (default: stdout)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _parse_args()

    pairs = find_pairs(
        query=args.query,
        max_pairs=args.max_pairs,
        max_candidates=args.max_candidates,
        require_taxonomy=args.require_taxonomy,
    )

    payload = [p.to_dict() for p in pairs]
    text = json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(text)
        log.info("Wrote %d pairs to %s", len(pairs), args.output)
    print(text)
    return 0 if pairs else 2


if __name__ == "__main__":
    sys.exit(main())
