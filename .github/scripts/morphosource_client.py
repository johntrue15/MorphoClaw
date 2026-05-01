#!/usr/bin/env python3
"""Unified MorphoSource API client.

Provides typed search/fetch methods for media, physical objects,
organizations, and projects.  Every search returns a standardized
:class:`SearchResponse` so downstream code never has to guess whether
a count came from pagination metadata or from ``len(items)``.

Usage::

    from morphosource_client import MorphoSourceClient

    client = MorphoSourceClient()
    resp = client.search_media(q="Serpentes", per_page=25)
    print(resp.total_count, resp.returned_count, resp.items[:3])
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore

log = logging.getLogger("MorphoSourceClient")

_DEFAULT_BASE = "https://www.morphosource.org/api"
_DEFAULT_TIMEOUT = 30
_DEFAULT_RETRIES = 3
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Standardized response contracts
# ---------------------------------------------------------------------------

@dataclass
class SearchResponse:
    """Normalized search result returned by every ``search_*`` method.

    ``returned_count`` is always ``len(items)`` — the items in *this* page.
    ``total_count`` comes from API pagination metadata and represents the
    repository-wide total matching the query.  When the API does not report
    a total, ``total_count`` is ``None``.
    """

    query: str
    endpoint: str
    filters: Dict[str, Any] = field(default_factory=dict)
    page: Optional[int] = None
    per_page: Optional[int] = None
    returned_count: int = 0
    total_count: Optional[int] = None
    items: List[Dict[str, Any]] = field(default_factory=list)
    fetched_at: str = ""
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)
    status_code: Optional[int] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat()
        self.returned_count = len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_response", None)
        return d


@dataclass
class MediaRecord:
    """Single media item returned by ``get_media``."""

    media_id: str
    data: Dict[str, Any]
    status_code: Optional[int] = None
    error: Optional[str] = None
    fetched_at: str = ""

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PhysicalObjectRecord:
    """Single physical object returned by ``get_physical_object``."""

    object_id: str
    data: Dict[str, Any]
    status_code: Optional[int] = None
    error: Optional[str] = None
    fetched_at: str = ""

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MorphoSourceClient:
    """HTTP client for the MorphoSource REST API.

    Parameters
    ----------
    base_url : str
        API base URL. Override via ``MORPHOSOURCE_API_BASE`` env var.
    api_key : str | None
        Bearer token. Override via ``MORPHOSOURCE_API_KEY`` env var.
    timeout : float
        Per-request timeout in seconds. Override via ``API_TIMEOUT`` env var.
    max_retries : int
        How many times to retry on transient errors (429, 5xx).
    backoff_factor : float
        Exponential backoff multiplier (seconds).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = _DEFAULT_RETRIES,
        backoff_factor: float = 1.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MORPHOSOURCE_API_BASE", _DEFAULT_BASE)
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("MORPHOSOURCE_API_KEY", "")
        self.timeout = timeout or float(
            os.environ.get("API_TIMEOUT", str(_DEFAULT_TIMEOUT))
        )
        self.max_retries = max(1, max_retries)
        self.backoff_factor = backoff_factor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        """Execute an HTTP request with retry logic.

        Returns ``(status_code, json_body_or_None, error_string_or_None)``.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = self._headers()

        if _requests is None:
            return self._request_stdlib(method, url, params, headers)

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = _requests.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    delay = self.backoff_factor * (2 ** (attempt - 1))
                    log.warning(
                        "MorphoSource %s %s -> %d (attempt %d/%d, retry in %.1fs)",
                        method, path, resp.status_code, attempt, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    continue

                data: Optional[Dict[str, Any]] = None
                error: Optional[str] = None
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        error = f"Invalid JSON response from {url}"
                else:
                    error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                return resp.status_code, data, error

            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (2 ** (attempt - 1))
                    log.warning(
                        "MorphoSource %s %s failed: %s (attempt %d/%d, retry in %.1fs)",
                        method, path, exc, attempt, self.max_retries, delay,
                    )
                    time.sleep(delay)

        return 0, None, last_error or "Request failed after retries"

    def _request_stdlib(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        """Fallback using :mod:`urllib` when ``requests`` is unavailable."""
        import urllib.request
        import urllib.error

        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                return resp.getcode(), json.loads(body.decode("utf-8")), None
        except urllib.error.HTTPError as exc:
            return exc.code, None, f"HTTP {exc.code}"
        except Exception as exc:
            return 0, None, str(exc)

    # ------------------------------------------------------------------
    # Pagination / count extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pagination(data: Dict[str, Any]) -> tuple[Optional[int], List[Dict[str, Any]], str]:
        """Extract (total_count, items_list, items_key) from an API payload.

        MorphoSource wraps results under an optional ``response`` key
        and stores the total in ``pages.total_count``.
        """
        payload = data
        if isinstance(data.get("response"), dict):
            payload = data["response"]

        items: List[Dict[str, Any]] = []
        items_key = ""
        for key in ("media", "physical_objects", "assets", "organizations", "projects"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                items_key = key
                break

        total_count: Optional[int] = None
        pages = payload.get("pages")
        if isinstance(pages, dict):
            tc = pages.get("total_count")
            if isinstance(tc, int):
                total_count = tc

        return total_count, items, items_key

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def _search(
        self,
        endpoint: str,
        q: str = "",
        page: int = 1,
        per_page: int = 25,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        params: Dict[str, Any] = {"locale": "en", "per_page": str(per_page), "page": str(page)}
        if q:
            params["q"] = q
        if extra_params:
            params.update(extra_params)

        status, data, error = self._request("GET", endpoint, params=params)

        if error or data is None:
            return SearchResponse(
                query=q, endpoint=endpoint, filters=extra_params or {},
                page=page, per_page=per_page,
                status_code=status, error=error,
            )

        total_count, items, _ = self._extract_pagination(data)

        return SearchResponse(
            query=q,
            endpoint=endpoint,
            filters=extra_params or {},
            page=page,
            per_page=per_page,
            returned_count=len(items),
            total_count=total_count,
            items=items,
            status_code=status,
            raw_response=data,
        )

    def search_media(
        self,
        q: str = "",
        page: int = 1,
        per_page: int = 25,
        **extra,
    ) -> SearchResponse:
        """Search the ``/media`` endpoint."""
        extra.setdefault("search_field", "all_fields")
        return self._search("media", q=q, page=page, per_page=per_page, extra_params=extra or None)

    def search_physical_objects(
        self,
        q: str = "",
        page: int = 1,
        per_page: int = 25,
        **extra,
    ) -> SearchResponse:
        """Search the ``/physical-objects`` endpoint."""
        return self._search("physical-objects", q=q, page=page, per_page=per_page, extra_params=extra or None)

    def search_organizations(
        self,
        q: str = "",
        page: int = 1,
        per_page: int = 25,
        **extra,
    ) -> SearchResponse:
        """Search the ``/organizations`` endpoint (institutions / collections)."""
        return self._search("organizations", q=q, page=page, per_page=per_page, extra_params=extra or None)

    def search_projects(
        self,
        q: str = "",
        page: int = 1,
        per_page: int = 25,
        **extra,
    ) -> SearchResponse:
        """Search the ``/projects`` endpoint (teams / projects)."""
        return self._search("projects", q=q, page=page, per_page=per_page, extra_params=extra or None)

    # ------------------------------------------------------------------
    # Public single-record methods
    # ------------------------------------------------------------------

    def get_media(self, media_id: str) -> MediaRecord:
        """Fetch a single media record by ID."""
        media_id = str(media_id).strip().lstrip("0").zfill(9)
        status, data, error = self._request("GET", f"media/{media_id}")
        return MediaRecord(
            media_id=media_id,
            data=data or {},
            status_code=status,
            error=error,
        )

    def get_physical_object(self, object_id: str) -> PhysicalObjectRecord:
        """Fetch a single physical object by ID."""
        object_id = str(object_id).strip().lstrip("0").zfill(9)
        status, data, error = self._request("GET", f"physical-objects/{object_id}")
        return PhysicalObjectRecord(
            object_id=object_id,
            data=data or {},
            status_code=status,
            error=error,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def search_by_endpoint(
        self,
        endpoint: str,
        params: Dict[str, Any],
    ) -> SearchResponse:
        """Generic search using a raw endpoint name and parameter dict.

        Used by legacy callers that already have pre-built parameter dicts
        (e.g. ``morphosource_api.search_morphosource``).
        """
        q = params.pop("q", params.pop("search_field_value", ""))
        page = int(params.pop("page", 1))
        per_page = int(params.pop("per_page", 25))
        params.pop("locale", None)
        return self._search(
            endpoint, q=q, page=page, per_page=per_page,
            extra_params=params if params else None,
        )


# Module-level singleton (lazily created on first import)
_default_client: Optional[MorphoSourceClient] = None


def get_client() -> MorphoSourceClient:
    """Return the module-level singleton :class:`MorphoSourceClient`."""
    global _default_client
    if _default_client is None:
        _default_client = MorphoSourceClient()
    return _default_client


__all__ = [
    "MorphoSourceClient",
    "SearchResponse",
    "MediaRecord",
    "PhysicalObjectRecord",
    "get_client",
]
