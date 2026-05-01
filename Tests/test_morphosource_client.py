"""
Tests for the morphosource_client module.

Covers:
  - SearchResponse contract enforcement
  - Pagination metadata extraction (total_count vs returned_count)
  - Retry logic on transient HTTP errors
  - Timeout propagation
  - All typed search and single-record methods
  - The key invariant: returned_count != total_count is handled correctly
  - Integration-test mode behind MORPHOSOURCE_LIVE_TESTS env flag
"""
import os
import sys
import json
import types
import time
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.github', 'scripts'))

# Provide lightweight stubs when packages are unavailable
if 'openai' not in sys.modules:
    openai_stub = types.ModuleType('openai')

    class _StubOpenAI:
        def __init__(self, *_, **__):
            pass

    openai_stub.OpenAI = _StubOpenAI
    sys.modules['openai'] = openai_stub

if 'requests' not in sys.modules:
    requests_stub = types.ModuleType('requests')

    class _StubRequest:
        def __init__(self, method, url, params=None):
            self.method = method
            self.url = url
            self.params = params or {}

        def prepare(self):
            from urllib.parse import urlencode
            query = urlencode(self.params, doseq=True)
            prepared = types.SimpleNamespace()
            prepared.url = f"{self.url}?{query}" if query else self.url
            return prepared

    def _stub_get(*_, **__):
        raise NotImplementedError("requests.get should be patched in tests")

    requests_stub.Request = _StubRequest
    requests_stub.get = _stub_get
    requests_stub.Session = MagicMock
    requests_stub.RequestException = Exception
    sys.modules['requests'] = requests_stub

from morphosource_client import (
    MorphoSourceClient,
    SearchResponse,
    MediaRecord,
    PhysicalObjectRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.content = json.dumps(json_data or {}).encode()
    return resp


def _make_paginated_media(items, total_count):
    """Build a MorphoSource-style JSON payload with pagination."""
    return {
        "response": {
            "media": items,
            "pages": {
                "total_count": total_count,
                "current_page": 1,
                "per_page": len(items),
            }
        }
    }


# ---------------------------------------------------------------------------
# SearchResponse dataclass tests
# ---------------------------------------------------------------------------


class TestSearchResponse:
    def test_returned_count_is_len_items(self):
        resp = SearchResponse(
            query="test",
            endpoint="media",
            items=[{"id": "1"}, {"id": "2"}],
        )
        assert resp.returned_count == 2

    def test_total_count_independent_of_items(self):
        resp = SearchResponse(
            query="test",
            endpoint="media",
            items=[{"id": "1"}],
            total_count=500,
        )
        assert resp.returned_count == 1
        assert resp.total_count == 500

    def test_total_count_none_when_not_provided(self):
        resp = SearchResponse(query="q", endpoint="e", items=[])
        assert resp.total_count is None

    def test_fetched_at_is_populated(self):
        resp = SearchResponse(query="q", endpoint="e")
        assert resp.fetched_at != ""

    def test_to_dict_excludes_raw_response(self):
        resp = SearchResponse(
            query="q", endpoint="e",
            raw_response={"big": "payload"},
        )
        d = resp.to_dict()
        assert "raw_response" not in d
        assert d["query"] == "q"


# ---------------------------------------------------------------------------
# Pagination extraction
# ---------------------------------------------------------------------------


class TestExtractPagination:
    def test_top_level_media(self):
        data = {"media": [{"id": "1"}, {"id": "2"}], "pages": {"total_count": 100}}
        total, items, key = MorphoSourceClient._extract_pagination(data)
        assert total == 100
        assert len(items) == 2
        assert key == "media"

    def test_nested_response_key(self):
        data = _make_paginated_media([{"id": "1"}], 42)
        total, items, key = MorphoSourceClient._extract_pagination(data)
        assert total == 42
        assert len(items) == 1
        assert key == "media"

    def test_physical_objects(self):
        data = {
            "response": {
                "physical_objects": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "pages": {"total_count": 999}
            }
        }
        total, items, key = MorphoSourceClient._extract_pagination(data)
        assert total == 999
        assert len(items) == 3
        assert key == "physical_objects"

    def test_no_pages_key(self):
        data = {"media": [{"id": "1"}]}
        total, items, key = MorphoSourceClient._extract_pagination(data)
        assert total is None
        assert len(items) == 1

    def test_empty_response(self):
        total, items, key = MorphoSourceClient._extract_pagination({})
        assert total is None
        assert items == []


# ---------------------------------------------------------------------------
# Core invariant: returned_count != total_count
# ---------------------------------------------------------------------------


class TestReturnedCountVsTotalCount:
    """Prove that the client correctly distinguishes page size from
    repository-wide totals — the key fix this refactor delivers."""

    @patch('morphosource_client._requests')
    def test_page_size_differs_from_total(self, mock_requests):
        page_items = [{"id": str(i)} for i in range(10)]
        json_data = _make_paginated_media(page_items, total_count=1234)
        mock_requests.request.return_value = _mock_response(200, json_data)

        client = MorphoSourceClient(base_url="https://example.com/api", api_key="k")
        resp = client.search_media(q="Serpentes", per_page=10)

        assert resp.returned_count == 10
        assert resp.total_count == 1234
        assert resp.returned_count != resp.total_count

    @patch('morphosource_client._requests')
    def test_single_page_total_equals_returned(self, mock_requests):
        page_items = [{"id": "1"}, {"id": "2"}]
        json_data = _make_paginated_media(page_items, total_count=2)
        mock_requests.request.return_value = _mock_response(200, json_data)

        client = MorphoSourceClient(base_url="https://example.com/api", api_key="k")
        resp = client.search_media(q="Anolis")

        assert resp.returned_count == 2
        assert resp.total_count == 2

    @patch('morphosource_client._requests')
    def test_zero_results(self, mock_requests):
        json_data = _make_paginated_media([], total_count=0)
        mock_requests.request.return_value = _mock_response(200, json_data)

        client = MorphoSourceClient(base_url="https://example.com/api", api_key="k")
        resp = client.search_media(q="nonexistent")

        assert resp.returned_count == 0
        assert resp.total_count == 0


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @patch('morphosource_client._requests')
    @patch('morphosource_client.time.sleep')
    def test_retries_on_500(self, mock_sleep, mock_requests):
        fail_resp = _mock_response(500, text="Server Error")
        ok_resp = _mock_response(200, _make_paginated_media([{"id": "1"}], 1))
        mock_requests.request.side_effect = [fail_resp, ok_resp]

        client = MorphoSourceClient(
            base_url="https://example.com/api",
            max_retries=3,
            backoff_factor=0.01,
        )
        resp = client.search_media(q="test")

        assert resp.returned_count == 1
        assert mock_requests.request.call_count == 2
        mock_sleep.assert_called_once()

    @patch('morphosource_client._requests')
    @patch('morphosource_client.time.sleep')
    def test_retries_on_429(self, mock_sleep, mock_requests):
        rate_limit = _mock_response(429, text="Rate Limited")
        ok_resp = _mock_response(200, _make_paginated_media([], 0))
        mock_requests.request.side_effect = [rate_limit, ok_resp]

        client = MorphoSourceClient(
            base_url="https://example.com/api",
            max_retries=2,
            backoff_factor=0.01,
        )
        resp = client.search_media(q="test")

        assert resp.error is None
        assert mock_requests.request.call_count == 2

    @patch('morphosource_client._requests')
    @patch('morphosource_client.time.sleep')
    def test_retries_on_network_exception(self, mock_sleep, mock_requests):
        mock_requests.request.side_effect = [
            ConnectionError("refused"),
            _mock_response(200, _make_paginated_media([{"id": "1"}], 5)),
        ]

        client = MorphoSourceClient(
            base_url="https://example.com/api",
            max_retries=2,
            backoff_factor=0.01,
        )
        resp = client.search_media(q="test")

        assert resp.returned_count == 1
        assert resp.total_count == 5

    @patch('morphosource_client._requests')
    @patch('morphosource_client.time.sleep')
    def test_exhausts_retries(self, mock_sleep, mock_requests):
        mock_requests.request.side_effect = ConnectionError("down")

        client = MorphoSourceClient(
            base_url="https://example.com/api",
            max_retries=2,
            backoff_factor=0.01,
        )
        resp = client.search_media(q="test")

        assert resp.error is not None
        assert resp.returned_count == 0
        assert mock_requests.request.call_count == 2


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    @patch('morphosource_client._requests')
    def test_timeout_propagated(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            200, _make_paginated_media([], 0)
        )

        client = MorphoSourceClient(
            base_url="https://example.com/api",
            timeout=42.0,
        )
        client.search_media(q="test")

        call_kwargs = mock_requests.request.call_args
        assert call_kwargs.kwargs.get("timeout") == 42.0 or call_kwargs[1].get("timeout") == 42.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @patch('morphosource_client._requests')
    def test_non_200_returns_error(self, mock_requests):
        mock_requests.request.return_value = _mock_response(404, text="Not Found")

        client = MorphoSourceClient(base_url="https://example.com/api")
        resp = client.search_media(q="gone")

        assert resp.error is not None
        assert "404" in resp.error
        assert resp.returned_count == 0

    @patch('morphosource_client._requests')
    def test_invalid_json_returns_error(self, mock_requests):
        bad_resp = MagicMock()
        bad_resp.status_code = 200
        bad_resp.json.side_effect = ValueError("bad json")
        bad_resp.content = b"not json"
        mock_requests.request.return_value = bad_resp

        client = MorphoSourceClient(base_url="https://example.com/api")
        resp = client.search_media(q="test")

        assert resp.error is not None
        assert "JSON" in resp.error


# ---------------------------------------------------------------------------
# Typed search methods
# ---------------------------------------------------------------------------


class TestSearchMethods:
    @patch('morphosource_client._requests')
    def test_search_physical_objects(self, mock_requests):
        data = {
            "response": {
                "physical_objects": [{"id": "A"}],
                "pages": {"total_count": 50}
            }
        }
        mock_requests.request.return_value = _mock_response(200, data)

        client = MorphoSourceClient(base_url="https://example.com/api")
        resp = client.search_physical_objects(q="Anolis", per_page=5)

        assert resp.endpoint == "physical-objects"
        assert resp.returned_count == 1
        assert resp.total_count == 50

    @patch('morphosource_client._requests')
    def test_search_organizations(self, mock_requests):
        data = {"organizations": [{"id": "org1"}], "pages": {"total_count": 3}}
        mock_requests.request.return_value = _mock_response(200, data)

        client = MorphoSourceClient(base_url="https://example.com/api")
        resp = client.search_organizations(q="museum")

        assert resp.endpoint == "organizations"
        assert resp.returned_count == 1
        assert resp.total_count == 3

    @patch('morphosource_client._requests')
    def test_search_projects(self, mock_requests):
        data = {"projects": [{"id": "p1"}, {"id": "p2"}], "pages": {"total_count": 2}}
        mock_requests.request.return_value = _mock_response(200, data)

        client = MorphoSourceClient(base_url="https://example.com/api")
        resp = client.search_projects(q="team")

        assert resp.endpoint == "projects"
        assert resp.returned_count == 2


# ---------------------------------------------------------------------------
# Single-record methods
# ---------------------------------------------------------------------------


class TestSingleRecordMethods:
    @patch('morphosource_client._requests')
    def test_get_media(self, mock_requests):
        data = {"response": {"id": "000123456", "title": "Skull Mesh"}}
        mock_requests.request.return_value = _mock_response(200, data)

        client = MorphoSourceClient(base_url="https://example.com/api")
        record = client.get_media("123456")

        assert record.media_id == "000123456"
        assert record.data["response"]["title"] == "Skull Mesh"
        assert record.error is None

    @patch('morphosource_client._requests')
    def test_get_physical_object(self, mock_requests):
        data = {"response": {"id": "000789", "title": "Specimen"}}
        mock_requests.request.return_value = _mock_response(200, data)

        client = MorphoSourceClient(base_url="https://example.com/api")
        record = client.get_physical_object("789")

        assert record.object_id == "000000789"
        assert record.error is None

    @patch('morphosource_client._requests')
    def test_get_media_error(self, mock_requests):
        mock_requests.request.return_value = _mock_response(404, text="Not Found")

        client = MorphoSourceClient(base_url="https://example.com/api")
        record = client.get_media("999")

        assert record.error is not None
        assert "404" in record.error


# ---------------------------------------------------------------------------
# search_by_endpoint (legacy compatibility)
# ---------------------------------------------------------------------------


class TestSearchByEndpoint:
    @patch('morphosource_client._requests')
    def test_generic_search(self, mock_requests):
        data = _make_paginated_media([{"id": "1"}], 10)
        mock_requests.request.return_value = _mock_response(200, data)

        client = MorphoSourceClient(base_url="https://example.com/api")
        resp = client.search_by_endpoint("media", {"q": "skull", "per_page": "5", "page": "2"})

        assert resp.endpoint == "media"
        assert resp.returned_count == 1
        assert resp.total_count == 10


# ---------------------------------------------------------------------------
# _extract_counts (morphosource_api backward compat)
# ---------------------------------------------------------------------------


class TestExtractCounts:
    """Verify the refactored _extract_counts function in morphosource_api."""

    def test_total_preferred_over_len(self):
        from morphosource_api import _extract_counts
        data = _make_paginated_media(
            [{"id": str(i)} for i in range(10)],
            total_count=500,
        )
        total, returned = _extract_counts(data)
        assert total == 500
        assert returned == 10

    def test_fallback_to_len_when_no_pages(self):
        from morphosource_api import _extract_counts
        data = {"media": [{"id": "1"}, {"id": "2"}]}
        total, returned = _extract_counts(data)
        assert total == 2
        assert returned == 2

    def test_empty_returns_zero(self):
        from morphosource_api import _extract_counts
        total, returned = _extract_counts({})
        assert total == 0
        assert returned == 0

    def test_extract_result_count_uses_total(self):
        from morphosource_api import _extract_result_count
        data = _make_paginated_media(
            [{"id": "1"}],
            total_count=999,
        )
        assert _extract_result_count(data) == 999


# ---------------------------------------------------------------------------
# Integration test mode (behind env flag)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MORPHOSOURCE_LIVE_TESTS"),
    reason="Set MORPHOSOURCE_LIVE_TESTS=1 to run live API tests",
)
class TestLiveIntegration:
    """Live integration tests against the real MorphoSource API.

    Enable with: MORPHOSOURCE_LIVE_TESTS=1 pytest Tests/test_morphosource_client.py -k Live
    """

    def test_live_search_media(self):
        client = MorphoSourceClient()
        resp = client.search_media(q="skull", per_page=5)
        assert resp.error is None
        assert resp.returned_count <= 5
        if resp.total_count is not None:
            assert resp.total_count >= resp.returned_count

    def test_live_search_physical_objects(self):
        client = MorphoSourceClient()
        resp = client.search_physical_objects(q="Anolis", per_page=3)
        assert resp.error is None
        assert resp.returned_count <= 3

    def test_live_get_media(self):
        client = MorphoSourceClient()
        record = client.get_media("000407755")
        assert record.error is None
        assert record.data
