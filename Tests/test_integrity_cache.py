"""
Tests for the polite, persistent MorphoSource record cache used by
the integrity verifier.

These cover the memory + disk + network tiers, the politeness throttle,
the circuit breaker, the disk-TTL, and the offline path.  Every test
runs without touching the real MorphoSource API.
"""

import os
import sys
import time
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))

from integrity_cache import (  # noqa: E402
    CacheStats,
    RecordCache,
    _normalise_media_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountingFetcher:
    """Stub fetcher that records every call and can be programmed to fail."""

    def __init__(self, response, *, fail_after=None):
        self.response = response
        self.fail_after = fail_after
        self.calls = []

    def __call__(self, media_id):
        self.calls.append(media_id)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            return None
        if isinstance(self.response, Exception):
            raise self.response
        if callable(self.response):
            return self.response(media_id)
        return self.response


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalise:
    @pytest.mark.parametrize("raw,expected", [
        ("769445", "000769445"),
        ("000769445", "000769445"),
        ("  408242  ", "000408242"),
        (769445, "000769445"),
        ("", "000000000"),
        (None, ""),
        ("not-a-number", ""),
    ])
    def test_normalise(self, raw, expected):
        assert _normalise_media_id(raw) == expected


# ---------------------------------------------------------------------------
# Memory tier
# ---------------------------------------------------------------------------


class TestMemoryTier:
    def test_get_returns_empty_when_no_fetcher(self):
        cache = RecordCache(cache_dir=None, fetcher=None)
        assert cache.get("123456") == {}
        assert cache.stats.network_calls == 0

    def test_first_miss_calls_fetcher_then_memo(self, tmp_path):
        record = {"id": ["000123456"], "title": ["x"]}
        fetcher = CountingFetcher(record)
        cache = RecordCache(cache_dir=tmp_path, fetcher=fetcher,
                             min_delay_s=0.0, ttl_days=0,
                             circuit_breaker_threshold=0)
        first = cache.get("123456")
        second = cache.get("123456")
        third = cache.get("000123456")  # different padding -> same id
        assert first == record
        assert second is first
        assert third is first
        assert len(fetcher.calls) == 1
        assert cache.stats.memory_hits == 2
        assert cache.stats.network_calls == 1

    def test_negative_results_are_memoized(self):
        fetcher = CountingFetcher(None)  # always fails
        cache = RecordCache(cache_dir=None, fetcher=fetcher,
                             min_delay_s=0.0, circuit_breaker_threshold=0)
        assert cache.get("123") == {}
        assert cache.get("123") == {}
        assert cache.get("000000123") == {}
        assert len(fetcher.calls) == 1

    def test_prime_skips_network(self, tmp_path):
        record = {"id": ["000999999"], "title": ["primed"]}
        fetcher = CountingFetcher({"id": ["x"]})
        cache = RecordCache(cache_dir=tmp_path, fetcher=fetcher,
                             min_delay_s=0.0)
        cache.prime("999999", record)
        assert cache.get("999999") == record
        assert len(fetcher.calls) == 0


# ---------------------------------------------------------------------------
# Disk tier + TTL
# ---------------------------------------------------------------------------


class TestDiskTier:
    def test_disk_hit_skips_fetcher(self, tmp_path):
        record = {"id": ["000123456"], "title": ["disk"]}
        # Pre-populate disk cache via prime() through a separate cache
        a = RecordCache(cache_dir=tmp_path, fetcher=CountingFetcher(record),
                         min_delay_s=0.0)
        a.prime("123456", record)
        # New cache instance reading the same dir
        fetcher = CountingFetcher({"id": ["should-not-call"]})
        b = RecordCache(cache_dir=tmp_path, fetcher=fetcher, min_delay_s=0.0)
        assert b.get("123456") == record
        assert b.stats.disk_hits == 1
        assert len(fetcher.calls) == 0

    def test_ttl_expiry_triggers_refetch(self, tmp_path):
        cache_dir = tmp_path / "cache"
        # Write a record to disk and back-date it
        record = {"id": ["000123456"]}
        seed = RecordCache(cache_dir=cache_dir, fetcher=None, min_delay_s=0.0)
        seed.prime("123456", record)
        path = seed._path_for("000123456")
        old = time.time() - 10 * 86400  # 10 days ago
        os.utime(path, (old, old))

        # New cache with 7-day TTL should treat the disk record as stale
        fresh_record = {"id": ["000123456"], "title": ["fresh"]}
        fetcher = CountingFetcher(fresh_record)
        b = RecordCache(cache_dir=cache_dir, fetcher=fetcher,
                         ttl_days=7, min_delay_s=0.0)
        assert b.get("123456") == fresh_record
        assert len(fetcher.calls) == 1

    def test_ttl_zero_disables_expiry(self, tmp_path):
        cache_dir = tmp_path / "cache"
        record = {"id": ["000123456"]}
        seed = RecordCache(cache_dir=cache_dir, fetcher=None, min_delay_s=0.0)
        seed.prime("123456", record)
        path = seed._path_for("000123456")
        os.utime(path, (1.0, 1.0))  # epoch ~1970

        fetcher = CountingFetcher({"id": ["should-not-call"]})
        b = RecordCache(cache_dir=cache_dir, fetcher=fetcher,
                         ttl_days=0, min_delay_s=0.0)
        assert b.get("123456") == record
        assert len(fetcher.calls) == 0

    def test_no_cache_dir_does_not_persist(self, tmp_path):
        record = {"id": ["000123456"]}
        fetcher = CountingFetcher(record)
        a = RecordCache(cache_dir=None, fetcher=fetcher, min_delay_s=0.0)
        a.get("123456")
        # New cache instance -> would re-fetch since nothing on disk
        fetcher2 = CountingFetcher(record)
        b = RecordCache(cache_dir=None, fetcher=fetcher2, min_delay_s=0.0)
        b.get("123456")
        assert len(fetcher2.calls) == 1


# ---------------------------------------------------------------------------
# Politeness throttle
# ---------------------------------------------------------------------------


class TestThrottle:
    def test_min_delay_is_enforced(self):
        # Two distinct ids so the cache must hit the fetcher twice
        records = {"000000111": {"id": ["111"]}, "000000222": {"id": ["222"]}}
        fetcher = CountingFetcher(lambda mid: records.get(mid))
        cache = RecordCache(cache_dir=None, fetcher=fetcher, min_delay_s=0.05,
                             circuit_breaker_threshold=0)

        t0 = time.monotonic()
        cache.get("111")
        cache.get("222")
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.05
        assert len(fetcher.calls) == 2

    def test_zero_delay_skips_sleep(self):
        records = {"000000111": {"id": ["111"]}, "000000222": {"id": ["222"]}}
        fetcher = CountingFetcher(lambda mid: records.get(mid))
        cache = RecordCache(cache_dir=None, fetcher=fetcher, min_delay_s=0.0)
        t0 = time.monotonic()
        cache.get("111")
        cache.get("222")
        assert (time.monotonic() - t0) < 0.5  # generous; just no sleep added


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_breaker_opens_after_threshold(self):
        fetcher = CountingFetcher(None)  # always fails
        cache = RecordCache(cache_dir=None, fetcher=fetcher,
                             min_delay_s=0.0, circuit_breaker_threshold=3)

        # First three failures -> three real network calls
        for mid in ("100", "200", "300"):
            assert cache.get(mid) == {}
        assert len(fetcher.calls) == 3
        assert cache.circuit_open is True
        assert cache.stats.circuit_open is True

        # Subsequent ids skip the network entirely
        for mid in ("400", "500", "600"):
            assert cache.get(mid) == {}
        assert len(fetcher.calls) == 3  # unchanged
        assert cache.stats.skipped_after_breaker == 3

    def test_success_resets_breaker(self):
        # First call fails, second succeeds, then more failures shouldn't
        # immediately open the breaker because the counter reset.
        responses = iter([None, {"id": ["222"]}, None, None])

        def fetcher(_mid):
            return next(responses)

        cache = RecordCache(cache_dir=None, fetcher=fetcher,
                             min_delay_s=0.0, circuit_breaker_threshold=3)
        cache.get("111")
        assert cache._consecutive_failures == 1
        cache.get("222")
        assert cache._consecutive_failures == 0
        cache.get("333")
        cache.get("444")
        assert cache.circuit_open is False  # only 2 failures since reset

    def test_threshold_zero_disables_breaker(self):
        fetcher = CountingFetcher(None)
        cache = RecordCache(cache_dir=None, fetcher=fetcher,
                             min_delay_s=0.0, circuit_breaker_threshold=0)
        for i in range(5):
            cache.get(f"00000010{i}")
        assert cache.circuit_open is False
        assert cache.stats.skipped_after_breaker == 0
        assert len(fetcher.calls) == 5


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_dict_round_trip(self):
        s = CacheStats(memory_hits=1, disk_hits=2, network_calls=3,
                       network_failures=1, circuit_open=True,
                       skipped_after_breaker=4)
        d = s.to_dict()
        assert d["network_calls"] == 3
        assert d["circuit_open"] is True
        text = s.summary()
        assert "1 mem-hit" in text and "2 disk-hit" in text
        assert "open" in text
