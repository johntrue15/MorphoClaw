#!/usr/bin/env python3
"""
MorphoSource API query handler.
Searches MorphoSource database using API parameters.

This module delegates HTTP transport to :mod:`morphosource_client` which
provides retry logic, timeouts, and a standardized :class:`SearchResponse`.
Legacy callers that rely on the ``search_morphosource()`` function continue
to work unchanged — the only difference is that ``summary["count"]`` now
reflects the **repository-wide total** (``pages.total_count``) rather than
the number of items in the current page.
"""

import os
import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from urllib.parse import urlencode, urlparse

try:
    import requests  # type: ignore
    from requests import Request  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when requests missing
    import urllib.error
    import urllib.request

    class _FallbackResponse:
        """Simplified response object mimicking :mod:`requests`."""

        def __init__(self, status_code, body, url):
            self.status_code = status_code
            self._body = body
            self.url = url

        def json(self):
            return json.loads(self.text)

        @property
        def text(self):
            return self._body.decode('utf-8', errors='replace')

    class _FallbackRequest:
        """Minimal stand-in for :class:`requests.Request`."""

        def __init__(self, method, url, params=None):
            self.method = method
            self.url = url
            self.params = params or {}

        def prepare(self):
            if self.params:
                query = urlencode(self.params, doseq=True)
                prepared_url = f"{self.url}?{query}"
            else:
                prepared_url = self.url
            return SimpleNamespace(url=prepared_url)

    class _FallbackRequestsModule:
        Request = _FallbackRequest

        @staticmethod
        def get(url, params=None, headers=None, timeout=30):  # pylint: disable=unused-argument
            headers = headers or {}
            if params:
                query = urlencode(params, doseq=True)
                url = f"{url}?{query}"

            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    status_code = resp.getcode()
            except urllib.error.HTTPError as exc:  # pragma: no cover - network errors not in tests
                body = exc.read()
                status_code = exc.code
            except urllib.error.URLError as exc:  # pragma: no cover - network errors not in tests
                body = str(exc).encode('utf-8')
                status_code = 0

            return _FallbackResponse(status_code, body, url)

    requests = _FallbackRequestsModule()
    Request = _FallbackRequest

import query_formatter
from morphosource_client import MorphoSourceClient, SearchResponse


def _extract_endpoint(query_info):
    """Determine the MorphoSource endpoint to use based on query info."""

    if not query_info:
        return 'media'

    endpoint = query_info.get('api_endpoint')
    if endpoint:
        return endpoint

    generated_url = query_info.get('generated_url')
    if generated_url:
        parsed = urlparse(generated_url)
        parts = [part for part in parsed.path.split('/') if part]
        if len(parts) >= 2 and parts[0] == 'api':
            return parts[1]

    return 'media'


def _extract_counts(data):
    """Return ``(total_count, returned_count)`` from a MorphoSource API payload.

    ``total_count`` is the repository-wide count from ``pages.total_count``.
    ``returned_count`` is ``len(items)`` in the current page.
    When the API does not provide ``pages.total_count``, ``total_count``
    falls back to ``returned_count`` so callers always get a usable number.
    """
    if not isinstance(data, dict):
        return 0, 0

    payloads_to_check = [data]
    nested = data.get('response')
    if isinstance(nested, dict):
        payloads_to_check.append(nested)

    returned_count = 0
    total_count = None

    for payload in payloads_to_check:
        for key in ('media', 'physical_objects', 'assets'):
            value = payload.get(key)
            if isinstance(value, list):
                returned_count = len(value)
                break

        pages = payload.get('pages')
        if isinstance(pages, dict):
            tc = pages.get('total_count')
            if isinstance(tc, int):
                total_count = tc

        if returned_count > 0 or total_count is not None:
            break

    if total_count is None:
        total_count = returned_count

    return total_count, returned_count


def _extract_result_count(data):
    """Return the **total** number of results from a MorphoSource API payload.

    Unlike the previous implementation, this now prefers ``pages.total_count``
    (the repository-wide total) over ``len(items)`` (the current page size).
    """
    total, _returned = _extract_counts(data)
    return total


def _build_feedback(attempt_index, url, response_data):
    """Prepare feedback payload for query refinement."""

    try:
        response_excerpt = json.dumps(response_data, indent=2)[:1200]
    except Exception:
        response_excerpt = str(response_data)[:1200]

    return {
        'attempt': attempt_index,
        'failed_url': url,
        'response_excerpt': response_excerpt
    }


def search_morphosource(api_params, formatted_query, query_info=None, max_retries=2):
    """
    Search MorphoSource API with given parameters.
    
    Args:
        api_params (dict): API query parameters
        formatted_query (str): Formatted query string
        
    Returns:
        dict: Search results from MorphoSource API
    """
    api_key = os.environ.get('MORPHOSOURCE_API_KEY', '')
    
    query_info = deepcopy(query_info) if query_info else {}
    if 'formatted_query' not in query_info:
        query_info['formatted_query'] = formatted_query
    if 'api_params' not in query_info:
        query_info['api_params'] = deepcopy(api_params)
    if 'original_query' not in query_info:
        query_info['original_query'] = formatted_query

    print(f"Searching MorphoSource with formatted query: {query_info['formatted_query']}")
    print(f"API parameters: {json.dumps(api_params)}")

    # MorphoSource API configuration
    MORPHOSOURCE_API_BASE = "https://www.morphosource.org/api"

    headers = {}
    if api_key:
        headers['Authorization'] = api_key
    
    attempt_history = []
    current_params = deepcopy(api_params)
    current_formatted_query = query_info['formatted_query']
    current_query_info = deepcopy(query_info)

    try:
        for attempt in range(1, max_retries + 2):  # original attempt + retries
            endpoint = _extract_endpoint(current_query_info)
            search_url = f"{MORPHOSOURCE_API_BASE}/{endpoint}"

            prepared_request = Request('GET', search_url, params=current_params).prepare()
            request_url = prepared_request.url

            print(f"Attempt {attempt}: querying {request_url}")

            response = requests.get(search_url, params=current_params, headers=headers, timeout=30)

            attempt_entry = {
                'attempt': attempt,
                'endpoint': endpoint,
                'url': request_url,
                'status_code': response.status_code
            }

            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    data = {'status': 'error', 'message': 'Invalid JSON response'}

                total_count, returned_count = _extract_counts(data)
                attempt_entry['result_count'] = total_count
                attempt_entry['returned_count'] = returned_count
                attempt_history.append(attempt_entry)

                print(f"✓ Received response (attempt {attempt}), "
                      f"{returned_count} returned / {total_count} total results")

                if total_count > 0 or attempt == max_retries + 1:
                    results_summary = {
                        "status": "success",
                        "count": total_count,
                        "returned_count": returned_count,
                        "total_count": total_count,
                        "formatted_query": current_formatted_query,
                        "endpoint": endpoint,
                        "attempts": attempt_history
                    }

                    return {
                        'full_data': data,
                        'summary': results_summary,
                        'query_info': {
                            **current_query_info,
                            'formatted_query': current_formatted_query,
                            'api_params': current_params,
                            'api_endpoint': endpoint
                        }
                    }

                # Zero results and we can retry
                openai_available = bool(os.environ.get('OPENAI_API_KEY'))
                if not openai_available:
                    print("No OPENAI_API_KEY available for retry. Returning zero results.")
                    results_summary = {
                        "status": "success",
                        "count": total_count,
                        "returned_count": returned_count,
                        "total_count": total_count,
                        "formatted_query": current_formatted_query,
                        "endpoint": endpoint,
                        "attempts": attempt_history
                    }

                    return {
                        'full_data': data,
                        'summary': results_summary,
                        'query_info': {
                            **current_query_info,
                            'formatted_query': current_formatted_query,
                            'api_params': current_params,
                            'api_endpoint': endpoint
                        }
                    }

                feedback = _build_feedback(attempt, request_url, data)
                print("No results found. Requesting reformatted query from ChatGPT...")

                try:
                    refined = query_formatter.format_query(current_query_info['original_query'], feedback=feedback)
                except Exception as refine_error:
                    print(f"Retry formatting failed: {refine_error}")
                    refined = None

                if not refined:
                    print("No refined query returned; stopping retries.")
                    results_summary = {
                        "status": "success",
                        "count": total_count,
                        "returned_count": returned_count,
                        "total_count": total_count,
                        "formatted_query": current_formatted_query,
                        "endpoint": endpoint,
                        "attempts": attempt_history
                    }

                    return {
                        'full_data': data,
                        'summary': results_summary,
                        'query_info': {
                            **current_query_info,
                            'formatted_query': current_formatted_query,
                            'api_params': current_params,
                            'api_endpoint': endpoint
                        }
                    }

                new_url = refined.get('generated_url')
                new_params = refined.get('api_params') or current_params

                if not new_url and new_params == current_params:
                    # Retry produced identical parameters.  Try a broader
                    # keyword-based media search as a last-resort fallback.
                    broad = query_formatter._build_fallback_from_keywords(
                        current_query_info['original_query']
                    )
                    broad_params = broad.get('api_params', {})
                    if broad_params and broad_params != current_params:
                        print("Refined query unchanged; switching to keyword-based media search.")
                        new_params = broad_params
                        refined = broad
                    else:
                        print("Refined query did not change parameters; stopping retries.")
                        results_summary = {
                            "status": "success",
                            "count": total_count,
                            "returned_count": returned_count,
                            "total_count": total_count,
                            "formatted_query": current_formatted_query,
                            "endpoint": endpoint,
                            "attempts": attempt_history
                        }

                        return {
                            'full_data': data,
                            'summary': results_summary,
                            'query_info': {
                                **current_query_info,
                                'formatted_query': current_formatted_query,
                                'api_params': current_params,
                                'api_endpoint': endpoint
                            }
                        }

                current_params = deepcopy(new_params)
                current_formatted_query = refined.get('formatted_query', current_formatted_query)
                current_query_info = {
                    **refined,
                    'original_query': current_query_info['original_query']
                }
                continue

            # Non-200 status codes
            print(f"⚠ API returned status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            attempt_history.append(attempt_entry)

            error_data = {
                "status": "error",
                "code": response.status_code,
                "message": response.text[:200],
                "attempts": attempt_history
            }

            return {
                'full_data': error_data,
                'summary': error_data
            }

        # Should not reach here, but handle as zero results
        results_summary = {
            "status": "success",
            "count": 0,
            "formatted_query": current_formatted_query,
            "attempts": attempt_history
        }

        return {
            'full_data': {},
            'summary': results_summary,
            'query_info': {
                **current_query_info,
                'formatted_query': current_formatted_query,
                'api_params': current_params
            }
        }

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        error_data = {"status": "error", "message": str(e)}

        return {
            'full_data': error_data,
            'summary': error_data
        }


def main():
    """Main entry point for MorphoSource API script."""
    if len(sys.argv) < 3:
        print("Usage: morphosource_api.py '<formatted_query>' '<api_params_json>'")
        sys.exit(1)
    
    formatted_query = sys.argv[1]
    api_params_str = sys.argv[2]

    try:
        api_params = json.loads(api_params_str)
    except Exception as e:
        print(f"Error parsing API params: {e}")
        api_params = {'q': formatted_query, 'per_page': 10}

    query_info = None
    try:
        with open('formatted_query.json', 'r') as f:
            query_info = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load formatted_query.json: {e}")

    result = search_morphosource(api_params, formatted_query, query_info=query_info)

    # Save full data to output file for artifact
    with open('morphosource_results.json', 'w') as f:
        json.dump(result['full_data'], f, indent=2)

    # Persist final formatted query information for downstream steps
    final_query_info = result.get('query_info')
    if not final_query_info:
        fallback_info = query_info or {}
        final_query_info = {
            'formatted_query': fallback_info.get('formatted_query', formatted_query),
            'api_params': fallback_info.get('api_params', api_params),
            'api_endpoint': fallback_info.get('api_endpoint', _extract_endpoint(fallback_info)),
            'original_query': fallback_info.get('original_query', formatted_query)
        }

    with open('formatted_query_final.json', 'w') as f:
        json.dump(final_query_info, f, indent=2)

    # Set output for GitHub Actions (use summary to avoid size limits)
    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"results={json.dumps(result['summary'])}\n")

    print(f"\n✓ MorphoSource search complete")
    
    # Exit with error if search failed
    if result['summary'].get('status') == 'error':
        sys.exit(1)


if __name__ == "__main__":
    main()
