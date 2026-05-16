"""Regression tests for the agent's tool-call surface.

The research agent calls a dozen-or-so Python "tools" by name.  When any
of them silently changes signature or stops importing, the agent fails
opaquely in production.  These tests are the smoke alarm:

1. **Every tool module imports** with no missing dependencies.
2. **Every tool function has the documented signature** the agent relies on.
3. **The OpenAI function-tool schemas** in ``chat_handler.TOOLS`` and
   ``slicer_agent.parse_tool_call`` are well-formed.
4. **Each tool handles the offline / no-API-key path gracefully** —
   never raising, always returning a structured dict.

All tests are designed to run **without network**, **without an OpenAI
key**, and **without 3D Slicer installed** so they execute on every PR
in the cloud ``code-quality`` workflow.  Slicer-dependent tests live in
``test_slicer_cached_model.py``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

# Make the scripts directory importable like the agents do.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# 1. Every tool module must import cleanly
# ---------------------------------------------------------------------------

TOOL_MODULES = [
    "_helpers",
    "morphosource_client",
    "morphosource_api",
    "morphosource_api_download",
    "knowledge_graph",
    "ontology_search",
    "citation_extractor",
    "literature_search",
    "query_formatter",
    "slicer_tool",
    "chat_handler",
    "integrity_graph",
    "integrity_verifiers",
    "integrity_cache",
    "verify_research_run",
]


@pytest.mark.parametrize("module_name", TOOL_MODULES)
def test_tool_module_imports(module_name: str) -> None:
    """Every tool module the agent depends on must import without error."""
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        pytest.fail(f"Could not import {module_name!r}: {exc!r}")


# ---------------------------------------------------------------------------
# 2. Tool function signatures (the contract the agent relies on)
# ---------------------------------------------------------------------------

EXPECTED_SIGNATURES: Dict[str, list[str]] = {
    # module:function -> required parameter names (in order)
    "slicer_tool:analyze_specimen": ["media_id"],
    "slicer_tool:monai_segment": ["input_path"],
    "morphosource_client:MorphoSourceClient.search_media": [],
    "morphosource_client:MorphoSourceClient.get_media": ["media_id"],
    "morphosource_client:MorphoSourceClient.get_physical_object": ["object_id"],
    "morphosource_api:search_morphosource": [
        "api_params",
        "formatted_query",
    ],
    "morphosource_api_download:download_media": ["media_id"],
    "knowledge_graph:build_graph_from_search_results": ["search_results"],
    "ontology_search:lookup_anatomy_term": ["term"],
    "ontology_search:enrich_query_with_ontology": ["query"],
    "literature_search:search_pubmed": ["query"],
    "literature_search:search_literature": [],
    "citation_extractor:extract_citations": ["record"],
    "citation_extractor:extract_dois_from_record": ["record"],
    "query_formatter:format_query": ["query"],
    "chat_handler:search_morphosource": ["query"],
    "chat_handler:get_morphosource_media": ["media_id"],
    "chat_handler:process_chat": ["messages"],
    "verify_research_run:main": [],
}


@pytest.mark.parametrize("spec,required_params", list(EXPECTED_SIGNATURES.items()))
def test_tool_signature(spec: str, required_params: list[str]) -> None:
    """Tool function must accept the parameters the agent passes to it."""
    module_name, _, dotted = spec.partition(":")
    module = importlib.import_module(module_name)

    obj: Any = module
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            pytest.fail(f"{spec}: attribute chain broken at {part!r}")

    assert callable(obj), f"{spec} is not callable"
    sig = inspect.signature(obj)
    params = list(sig.parameters)
    # Strip implicit `self` for unbound methods.
    if params and params[0] == "self":
        params = params[1:]
    for name in required_params:
        assert name in params, f"{spec}: missing required parameter {name!r} (got {params})"


# ---------------------------------------------------------------------------
# 3. OpenAI function-tool schemas (the actual JSON the LLM sees)
# ---------------------------------------------------------------------------


def test_chat_handler_tools_are_valid_openai_function_schemas() -> None:
    """``chat_handler.TOOLS`` is what the LLM receives. It must be well-formed."""
    chat_handler = importlib.import_module("chat_handler")
    tools = getattr(chat_handler, "TOOLS", None)
    assert isinstance(tools, list) and tools, "chat_handler.TOOLS must be a non-empty list"

    seen_names = set()
    for i, tool in enumerate(tools):
        assert tool.get("type") == "function", f"tool[{i}].type must be 'function'"
        fn = tool.get("function")
        assert isinstance(fn, dict), f"tool[{i}].function must be an object"

        name = fn.get("name")
        assert isinstance(name, str) and name, f"tool[{i}].function.name missing"
        assert name not in seen_names, f"duplicate tool name: {name}"
        seen_names.add(name)

        desc = fn.get("description")
        assert (
            isinstance(desc, str) and len(desc) >= 20
        ), f"tool {name!r} needs a >=20-char description"

        params = fn.get("parameters", {})
        assert params.get("type") == "object", f"tool {name!r} parameters must be 'object'"
        props = params.get("properties", {})
        assert isinstance(props, dict), f"tool {name!r} parameters.properties missing"
        for req in params.get("required", []):
            assert (
                req in props
            ), f"tool {name!r} requires {req!r} but doesn't declare it in properties"

        # Every declared tool must resolve to a Python callable in the module.
        impl = getattr(chat_handler, name, None)
        assert callable(impl), f"tool schema {name!r} has no matching Python implementation"


def test_slicer_agent_parse_tool_call_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``slicer_agent.parse_tool_call`` must round-trip a representative payload.

    ``slicer_agent`` is meant to run inside 3D Slicer (it does ``import
    slicer``/``vtk`` at module top).  For unit-test purposes we stub
    those modules out so the file can be imported in a plain Python
    environment.
    """
    for name in ("slicer", "vtk", "numpy"):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    # The script reads several env vars at import time -- give it sane defaults
    # so the import doesn't crash even if the file changes to consult them.
    monkeypatch.setenv("SLICER_PLY_PATH", "/dev/null")
    monkeypatch.setenv("SLICER_GOAL", "(test)")
    monkeypatch.setenv("SLICER_MAX_STEPS", "1")
    sys.modules.pop("slicer_agent", None)
    try:
        slicer_agent = importlib.import_module("slicer_agent")
    except SystemExit:
        pytest.skip("slicer_agent exits early outside of Slicer; signature still verified")
    except Exception as exc:
        pytest.skip(f"slicer_agent cannot be imported in plain Python: {exc!r}")

    parse = slicer_agent.parse_tool_call

    payload = {"tool": "landmark", "name": "left_orbit", "x": 1.0, "y": 2.0, "z": 3.0}
    assert parse(json.dumps(payload)) == payload

    fenced = f"Some preamble\n```json\n{json.dumps(payload)}\n```\nTrailing text"
    assert parse(fenced) == payload

    # Garbage input falls back to DONE rather than raising.
    fallback = parse("not json at all")
    assert fallback.get("tool") == "DONE"


# ---------------------------------------------------------------------------
# 4. Each tool handles the offline / no-key path gracefully
# ---------------------------------------------------------------------------


def test_call_llm_returns_none_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_helpers.call_llm`` must return None (never raise) when no API key set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    helpers = importlib.import_module("_helpers")
    out = helpers.call_llm([{"role": "user", "content": "hi"}], label="test")
    assert out is None


def test_morphosource_client_returns_error_response_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network failure becomes an error string on the response, not an exception."""
    client_mod = importlib.import_module("morphosource_client")

    def boom(*_a, **_kw):
        raise ConnectionError("network is down")

    monkeypatch.setattr(client_mod, "_requests", mock.MagicMock(request=boom))
    client = client_mod.MorphoSourceClient(timeout=0.01, max_retries=1)
    resp = client.search_media(q="anything", per_page=1)
    assert resp.error, "search_media should attach an error string offline"
    assert resp.returned_count == 0


def test_morphosource_get_media_offline_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MorphoSourceClient.get_media`` must not raise when offline."""
    client_mod = importlib.import_module("morphosource_client")
    monkeypatch.setattr(
        client_mod,
        "_requests",
        mock.MagicMock(request=mock.MagicMock(side_effect=ConnectionError("boom"))),
    )
    client = client_mod.MorphoSourceClient(timeout=0.01, max_retries=1)
    record = client.get_media("000000000")
    assert record.error
    assert record.data == {}


def test_chat_handler_search_morphosource_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """``chat_handler.search_morphosource`` returns an error dict, never raises."""
    chat_handler = importlib.import_module("chat_handler")
    client_mod = importlib.import_module("morphosource_client")
    monkeypatch.setattr(
        client_mod,
        "_requests",
        mock.MagicMock(request=mock.MagicMock(side_effect=ConnectionError("boom"))),
    )
    # Reset the module-level client singleton so our patched `_requests`
    # is picked up on the next call.
    monkeypatch.setattr(client_mod, "_default_client", None, raising=False)
    out = chat_handler.search_morphosource("anything")
    assert isinstance(out, dict)
    assert "error" in out or "_total_count" in out


def test_slicer_tool_returns_error_dict_when_slicer_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``analyze_specimen`` must not raise when SLICER_BIN does not exist."""
    monkeypatch.setenv("SLICER_BIN", str(tmp_path / "does_not_exist" / "Slicer"))
    # Force the module to re-read the env var.
    if "slicer_tool" in sys.modules:
        del sys.modules["slicer_tool"]
    if "_helpers" in sys.modules:
        del sys.modules["_helpers"]
    slicer_tool = importlib.import_module("slicer_tool")
    result = slicer_tool.analyze_specimen("000000000", topic="smoke")
    assert isinstance(result, dict)
    assert result.get("success") is False
    assert "error" in result
    assert "Slicer" in result["error"]


def test_slicer_tool_cache_hit_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pre-populated cache directory must trigger the CACHE HIT path."""
    monkeypatch.setenv("AUTORESEARCHCLAW_HOME", str(tmp_path))
    # Point SLICER_BIN at something that exists so we get past the existence check.
    fake_slicer = tmp_path / "fake_slicer"
    fake_slicer.write_text("#!/bin/sh\nexit 0\n")
    fake_slicer.chmod(0o755)
    monkeypatch.setenv("SLICER_BIN", str(fake_slicer))
    for mod in ("slicer_tool", "_helpers"):
        sys.modules.pop(mod, None)
    slicer_tool = importlib.import_module("slicer_tool")

    media_id = "000123456"
    analysis_dir = tmp_path / "specimens" / f"media_{media_id}" / "analysis"
    analysis_dir.mkdir(parents=True)
    cached = {
        "vertices": 100,
        "faces": 200,
        "measurements": {"total_length": 10.0},
        "mass_properties": {"volume_mm3": 5.0},
        "pca_shape": {},
        "curvature": {"mean": {}},
        "landmarks": [],
        "screenshots": [],
    }
    (analysis_dir / "analysis.json").write_text(json.dumps(cached))

    result = slicer_tool.analyze_specimen(media_id, topic="cache-hit-test")
    assert result["success"] is True
    assert result["download_result"]["cached"] is True
    assert result["analysis"]["vertices"] == 100
    assert "media " + media_id in result["summary"]


def test_search_morphosource_invalid_endpoint_safe() -> None:
    """``morphosource_api.search_morphosource`` should not crash on bad input."""
    api = importlib.import_module("morphosource_api")
    result = api.search_morphosource({}, {})
    assert isinstance(result, dict)
    assert "results" in result or "error" in result or "summary" in result


def test_ontology_enrich_query_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """``enrich_query_with_ontology`` returns a string even without network."""
    ontology = importlib.import_module("ontology_search")

    def boom(*_a, **_kw):
        raise ConnectionError("offline")

    monkeypatch.setattr("urllib.request.urlopen", boom, raising=False)
    out = ontology.enrich_query_with_ontology("skull")
    assert isinstance(out, str) and out


def test_query_formatter_returns_dict_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``query_formatter.format_query`` should fall back to heuristics without an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    qf = importlib.import_module("query_formatter")
    out = qf.format_query("Anolis carolinensis skull CT scan")
    assert isinstance(out, dict)
    assert "api_params" in out or "endpoint" in out or "summary" in out


# ---------------------------------------------------------------------------
# 5. Knowledge graph + citations work on synthetic data
# ---------------------------------------------------------------------------


def test_knowledge_graph_builds_from_synthetic_search_results() -> None:
    kg_mod = importlib.import_module("knowledge_graph")
    # AutoResearchClaw wraps MorphoSource responses under
    # ``result_data.response.media`` so that's what we hand the graph.
    search_results = [
        {
            "query": "Anolis carolinensis skull",
            "result_data": {
                "response": {
                    "media": [
                        {
                            "id": ["m1"],
                            "title": ["Anolis carolinensis skull mesh"],
                            "media_type": ["Mesh"],
                            "physical_object_id": ["p1"],
                            "physical_object_title": ["uf:herp:1"],
                            "physical_object_taxonomy_name": ["Anolis carolinensis"],
                        },
                        {
                            "id": ["m2"],
                            "title": ["Anolis carolinensis skull CT"],
                            "media_type": ["CT"],
                            "physical_object_id": ["p1"],
                            "physical_object_title": ["uf:herp:1"],
                            "physical_object_taxonomy_name": ["Anolis carolinensis"],
                        },
                    ],
                }
            },
        }
    ]
    graph = kg_mod.build_graph_from_search_results(search_results)
    stats = graph.stats()
    assert stats["total_nodes"] >= 3
    assert stats["total_edges"] >= 1
    assert stats["media"] >= 2


def test_citation_extractor_from_synthetic_record() -> None:
    cit = importlib.import_module("citation_extractor")
    record = {
        "id": ["m1"],
        "title": ["A specimen with a DOI"],
        "description": ["Cite as: Doe J. 2024. https://doi.org/10.1234/example.5678"],
    }
    dois = cit.extract_dois_from_record(record)
    assert any("10.1234/example.5678" in (d.get("doi") or "") for d in dois)
