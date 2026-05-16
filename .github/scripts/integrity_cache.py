#!/usr/bin/env python3
"""
Polite, persistent MorphoSource record cache for the integrity verifier.

The verifier used to fan out 3 independent ``GET /api/media/<id>``
calls per media id (one in ``_fetch_media_record``, one in
``MetadataVerifier`` for re-resolution, one in ``LineageVerifier`` for
the parent lookup).  When the MorphoSource server got slow, the
default 30-second timeout × 3 retries × 3 verifier paths × N media ids
turned a single ``/verify`` run into a 30-minute connection storm.

This module replaces those ad-hoc calls with a single :class:`RecordCache`
that:

* memoizes records in-process so the three verifier paths share a
  single fetch
* persists records to disk under
  ``~/.autoresearchclaw/integrity/cache/media/`` with a configurable
  TTL (default 7 days), so re-running ``/verify`` on the same issue is
  free
* enforces a configurable minimum delay between *outgoing* HTTP calls
  (default 0.5 s) so we never burst on the public API
* opens a circuit breaker after N consecutive failures (default 3) and
  refuses further network calls until the run finishes
* is happy to run with ``allow_network=False`` for unit tests and
  fully-offline replays of a research report

Nothing in this module imports the ``MorphoSourceClient`` directly so
the verifier can pass in a *short-timeout, low-retry* client built
specifically for integrity verification (see
``verify_research_run.build_polite_client``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("IntegrityCache")

# A "fetcher" is anything callable as ``fetcher(media_id) -> dict | None``.
# The cache deliberately doesn't import MorphoSourceClient so it can be
# used with a stub in tests.
Fetcher = Callable[[str], Optional[Dict[str, Any]]]


_VALID_MEDIA_ID_RE = re.compile(r"^\d{4,12}$")


def _normalise_media_id(media_id: str) -> str:
    """Return a 9-digit zero-padded MorphoSource media id, or ``""``."""
    if media_id is None:
        return ""
    s = str(media_id).strip().lstrip("0")
    if not s:
        return "000000000"
    if not s.isdigit():
        return ""
    return s.zfill(9)


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Counters surfaced in the verifier's Markdown summary."""

    memory_hits: int = 0
    disk_hits: int = 0
    network_calls: int = 0
    network_failures: int = 0
    circuit_open: bool = False
    skipped_after_breaker: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_hits": self.memory_hits,
            "disk_hits": self.disk_hits,
            "network_calls": self.network_calls,
            "network_failures": self.network_failures,
            "circuit_open": self.circuit_open,
            "skipped_after_breaker": self.skipped_after_breaker,
        }

    def summary(self) -> str:
        return (
            f"{self.memory_hits} mem-hit / {self.disk_hits} disk-hit / "
            f"{self.network_calls} net "
            f"({self.network_failures} fail; "
            f"breaker={'open' if self.circuit_open else 'closed'}; "
            f"skipped={self.skipped_after_breaker})"
        )


# ---------------------------------------------------------------------------
# RecordCache
# ---------------------------------------------------------------------------


class RecordCache:
    """Three-tier (memory + disk + network) record cache for media records.

    Parameters
    ----------
    cache_dir
        Disk location for cached payloads.  ``None`` disables disk cache.
    ttl_days
        How long a cached record is considered fresh.  ``0`` disables TTL
        (records are always considered fresh).
    fetcher
        Callable used to populate the cache when it misses.  Receives a
        zero-padded media id and returns either a record dict or
        ``None`` on failure.  ``None`` puts the cache in offline mode.
    min_delay_s
        Minimum wall-clock delay between successive *outbound* HTTP calls
        (the politeness throttle).  Defaults to ``0.5`` s.
    circuit_breaker_threshold
        Open the circuit after this many *consecutive* failures.  Once
        open, no further network calls are made for the lifetime of the
        cache (a re-run reopens it).  ``0`` disables the breaker.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_days: int = 7,
        fetcher: Optional[Fetcher] = None,
        min_delay_s: float = 0.5,
        circuit_breaker_threshold: int = 3,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = max(0, int(ttl_days * 86400))
        self.fetcher = fetcher
        self.min_delay_s = max(0.0, float(min_delay_s))
        self.circuit_breaker_threshold = max(0, int(circuit_breaker_threshold))

        self._memory: Dict[str, Dict[str, Any]] = {}
        self._negative: set[str] = set()  # ids that returned no data this run
        self._consecutive_failures: int = 0
        self._last_call_at: float = 0.0
        self._lock = threading.Lock()
        self.stats = CacheStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def circuit_open(self) -> bool:
        return (
            self.circuit_breaker_threshold > 0
            and self._consecutive_failures >= self.circuit_breaker_threshold
        )

    def get(self, media_id: str) -> Dict[str, Any]:
        """Return the cached record for *media_id*, fetching if needed.

        Always returns a dict.  Returns ``{}`` (and remembers the
        negative result) when the record cannot be resolved.
        """
        mid = _normalise_media_id(media_id)
        if not mid:
            return {}

        # 1. Memory tier
        if mid in self._memory:
            self.stats.memory_hits += 1
            return self._memory[mid]
        if mid in self._negative:
            self.stats.memory_hits += 1
            return {}

        # 2. Disk tier
        disk = self._read_disk(mid)
        if disk is not None:
            self.stats.disk_hits += 1
            self._memory[mid] = disk
            return disk

        # 3. Network tier (with circuit breaker + politeness throttle)
        if self.fetcher is None:
            self._negative.add(mid)
            return {}
        if self.circuit_open:
            self.stats.skipped_after_breaker += 1
            self.stats.circuit_open = True
            self._negative.add(mid)
            return {}

        record = self._fetch_with_throttle(mid)
        if record:
            self._memory[mid] = record
            self._write_disk(mid, record)
            return record
        self._negative.add(mid)
        return {}

    def prime(self, media_id: str, record: Dict[str, Any]) -> None:
        """Insert a record into the cache without making any network call.

        Useful when the verifier already has a record (e.g. from a search
        result) and wants the verifiers to share it.
        """
        mid = _normalise_media_id(media_id)
        if not mid or not isinstance(record, dict) or not record:
            return
        self._memory[mid] = record
        self._write_disk(mid, record)

    def has(self, media_id: str) -> bool:
        """Return ``True`` if *media_id* would resolve without network."""
        mid = _normalise_media_id(media_id)
        if not mid:
            return False
        if mid in self._memory:
            return True
        return self._read_disk(mid) is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_with_throttle(self, mid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            elapsed = time.monotonic() - self._last_call_at
            if elapsed < self.min_delay_s:
                time.sleep(self.min_delay_s - elapsed)
            self._last_call_at = time.monotonic()
            self.stats.network_calls += 1
        try:
            record = self.fetcher(mid) if self.fetcher else None
        except Exception as exc:
            log.warning("Cache fetcher raised for %s: %s", mid, exc)
            record = None

        if not record:
            self._consecutive_failures += 1
            self.stats.network_failures += 1
            if self.circuit_open:
                self.stats.circuit_open = True
                log.warning(
                    "Integrity cache circuit breaker OPEN after %d "
                    "consecutive failures; subsequent media lookups will "
                    "be skipped for this run.",
                    self._consecutive_failures,
                )
            return None

        # Success resets the breaker
        self._consecutive_failures = 0
        return record

    def _path_for(self, mid: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        # Shard by the first 3 digits to keep directories small.
        return self.cache_dir / "media" / mid[:3] / f"{mid}.json"

    def _read_disk(self, mid: str) -> Optional[Dict[str, Any]]:
        path = self._path_for(mid)
        if path is None or not path.is_file():
            return None
        try:
            if self.ttl_seconds > 0:
                age = time.time() - path.stat().st_mtime
                if age > self.ttl_seconds:
                    return None
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            log.debug("Could not read cache %s: %s", path, exc)
        return None

    def _write_disk(self, mid: str, record: Dict[str, Any]) -> None:
        path = self._path_for(mid)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, default=str), encoding="utf-8")
        except OSError as exc:
            log.debug("Could not write cache %s: %s", path, exc)


__all__ = [
    "CacheStats",
    "Fetcher",
    "RecordCache",
]
